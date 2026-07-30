import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ContextFiLMConvNeXt(nn.Module):
    def __init__(self, num_labels: int, num_contexts: int, initial_biases: list = None, dropout: float = 0.0, context_dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.context_dropout = context_dropout
        
        # We use convnextv2_tiny as our backbone
        self.backbone = timm.create_model(
            "convnextv2_tiny.fcmae", 
            pretrained=True, 
            num_classes=0,
            drop_path_rate=dropout
        )
        
        # Context MLP
        self.context_mlp = nn.Sequential(
            nn.Linear(num_contexts, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )
        
        # ConvNeXt-Tiny dims
        dims = [96, 192, 384, 768]
        
        # We need FiLM parameters (gamma, beta) for four injection points:
        # 1. After Stem (96 channels)
        # 2. After Stage 0 (96 channels)
        # 3. After Stage 1 (192 channels)
        # 4. After Stage 2 (384 channels)
        # Total parameters = 2 * (96 + 96 + 192 + 384) = 1536
        
        # We initialize FiLM generator to 0 so the initial state is identity modulation
        self.film_gen = nn.Linear(256, 1536)
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)
        
        # Disease-specific Attention (N heads, one for each disease)
        self.disease_attn = nn.Conv2d(dims[3], num_labels, kernel_size=1)
        nn.init.zeros_(self.disease_attn.weight)
        if self.disease_attn.bias is not None:
            nn.init.zeros_(self.disease_attn.bias)
        
        # Final classifiers per disease
        self.head_norm = nn.LayerNorm(dims[3])
        self.head_drop = nn.Dropout(dropout)
        self.classifiers = nn.ModuleList([
            nn.Linear(dims[3], 1) for _ in range(num_labels)
        ])
        
        if initial_biases is not None:
            for i in range(num_labels):
                self.classifiers[i].bias.data.fill_(initial_biases[i])
                
    def forward(self, img, context):
        # 0. Context Dropout (only during training)
        if self.training and self.context_dropout > 0:
            # Drop entire context vectors for some samples in the batch
            # Actually, doing it randomly per-sample is good, or we can just randomly zero out the whole batch
            mask = (torch.rand(context.shape[0], 1, device=context.device) > self.context_dropout).float()
            context = context * mask

        # 1. Generate FiLM params
        ctx_emb = self.context_mlp(context)
        film_params = self.film_gen(ctx_emb) # [B, 1536]
        
        split_sizes = [96, 96, 96, 96, 192, 192, 384, 384]
        g0, b0, g1, b1, g2, b2, g3, b3 = torch.split(film_params, split_sizes, dim=1)
        
        g0 = g0.view(-1, g0.size(1), 1, 1)
        b0 = b0.view(-1, b0.size(1), 1, 1)
        g1 = g1.view(-1, g1.size(1), 1, 1)
        b1 = b1.view(-1, b1.size(1), 1, 1)
        g2 = g2.view(-1, g2.size(1), 1, 1)
        b2 = b2.view(-1, b2.size(1), 1, 1)
        g3 = g3.view(-1, g3.size(1), 1, 1)
        b3 = b3.view(-1, b3.size(1), 1, 1)
        
        # 2. Forward pass through backbone with manual FiLM injection
        # Intercept stem
        x = self.backbone.stem(img)
        x = x * (1 + g0) + b0
        
        # Intercept stages
        x = self.backbone.stages[0](x)
        x = x * (1 + g1) + b1
        
        x = self.backbone.stages[1](x)
        x = x * (1 + g2) + b2
        
        x = self.backbone.stages[2](x)
        x = x * (1 + g3) + b3
        
        x = self.backbone.stages[3](x) # [B, 768, H/32, W/32]
        
        # 3. Disease-specific Spatial Attention
        B, C, H, W = x.shape
        attn_logits = self.disease_attn(x) # [B, num_labels, H, W]
        attn_weights = F.softmax(attn_logits.view(B, self.num_labels, -1), dim=-1) # [B, num_labels, H*W]
        
        feat_flat = x.view(B, C, -1).transpose(1, 2) # [B, H*W, 768]
        
        # Batch matrix multiply: [B, num_labels, H*W] @ [B, H*W, 768] -> [B, num_labels, 768]
        attended = torch.bmm(attn_weights, feat_flat)
        attended = self.head_norm(attended)
        attended = self.head_drop(attended)
        
        # 4. Final logit generation
        logits = []
        for i, clf in enumerate(self.classifiers):
            # clf takes [B, 768] -> [B, 1]
            logits.append(clf(attended[:, i, :]))
            
        return torch.cat(logits, dim=1) # [B, num_labels]
