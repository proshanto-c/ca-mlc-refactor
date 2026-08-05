import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import GATConv, HeteroConv
except ImportError as exc:
    raise SystemExit("Missing dependency: torch-geometric. Install it before using this model.") from exc


@dataclass
class DemographicSpec:
    name: str
    kind: str  # numeric or categorical
    mean: float = 0.0
    std: float = 1.0
    vocab: Optional[Dict[str, int]] = None


class PatientGraphModel(nn.Module):
    """Heterogeneous Graph Neural Network for Patient-Level or Image-Level Predictions."""
    def __init__(
        self,
        num_labels: int,
        demographic_specs: Sequence[DemographicSpec],
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_images_per_patient: int = 32,
        heads: int = 2,
        prediction_level: str = "patient",
        initial_biases: Optional[Sequence[float]] = None,
        use_context: bool = True,
        use_image_features: bool = True,
    ) -> None:
        super().__init__()
        
        if prediction_level not in ["patient", "image"]:
            raise ValueError("prediction_level must be either 'patient' or 'image'")
            
        self.prediction_level = prediction_level
        self.use_context = use_context
        self.use_image_features = use_image_features
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.demographic_specs = list(demographic_specs)
        self.dropout = dropout
        self.max_images_per_patient = max_images_per_patient

        # --- NEW: Load Pre-trained Vision Backbone ---
        try:
            import timm
        except ImportError as exc:
            raise SystemExit("Missing dependency: timm. Install it via 'pip install timm'") from exc

        # Load RETFound ViT-Large using timm.
        # dynamic_img_size=True allows the model to accept images of size 256x256 instead of default 224x224.
        self.cnn = timm.create_model("hf_hub:bitfount/RETFound_MAE", pretrained=True, dynamic_img_size=True)
        
        # Freeze the CNN (Highly recommended so you don't run out of GPU memory)
        for param in self.cnn.parameters():
            param.requires_grad = False

        # --- Node Encoders ---
        cnn_out_dim = 1024 # ViT-Large feature dimension always has 1024 channels
        
        # 2. UPDATE patch_encoder to accept cnn_out_dim instead of patch_dim
        self.patch_encoder = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # --- Demographic Encoders ---
        self.demo_field_embed = nn.Embedding(len(self.demographic_specs), hidden_dim)
        self.demo_kind_embed = nn.Embedding(2, hidden_dim)
        self.demo_numeric = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.demo_cat_tables = nn.ModuleList()
        self.cat_field_to_table: Dict[int, int] = {}
        cat_count = 0
        for field_idx, spec in enumerate(self.demographic_specs):
            if spec.kind == "categorical":
                size = len(spec.vocab or {"__UNK__": 0})
                self.demo_cat_tables.append(nn.Embedding(size, hidden_dim))
                self.cat_field_to_table[field_idx] = cat_count
                cat_count += 1

        self.image_idx_embed = nn.Embedding(max_images_per_patient + 1, hidden_dim)
        self.eye_idx_embed = nn.Embedding(4, hidden_dim)
        self.patient_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.label_token = nn.Embedding(num_labels, hidden_dim)

        self.type_bias = nn.ParameterDict(
            {
                "patch": nn.Parameter(torch.zeros(hidden_dim)),
                "image": nn.Parameter(torch.zeros(hidden_dim)),
                "demographic": nn.Parameter(torch.zeros(hidden_dim)),
                "patient": nn.Parameter(torch.zeros(hidden_dim)),
                "label": nn.Parameter(torch.zeros(hidden_dim)),
            }
        )

        # --- Heterogeneous Convolutions ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            
            # Base structural pathways
            conv_dict = {
                ("label", "correlates", "label"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=True),
            }
            if self.use_image_features:
                conv_dict.update({
                    ("patch", "adjacent", "patch"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=True),
                    ("patch", "to", "image"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                    ("image", "to", "patch"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                })
            
            # Context Pathways (removed for image-only baselines)
            if self.use_context:
                conv_dict.update({
                    ("image", "to", "patient"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                    ("patient", "to", "image"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                    ("demographic", "to", "patient"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                    ("patient", "to", "demographic"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                })
            
            # --- DYNAMIC LABEL PATHWAYS ---
            if self.prediction_level == "patient":
                conv_dict[("patient", "to", "label")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False)
                conv_dict[("label", "to", "patient")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False)
            elif self.prediction_level == "image":
                conv_dict[("image", "to", "label")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False)
                conv_dict[("label", "to", "image")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False)

            conv = HeteroConv(conv_dict, aggr="mean")
            self.convs.append(conv)
            
            self.norms.append(
                nn.ModuleDict(
                    {
                        "patch": nn.LayerNorm(hidden_dim),
                        "image": nn.LayerNorm(hidden_dim),
                        "demographic": nn.LayerNorm(hidden_dim),
                        "patient": nn.LayerNorm(hidden_dim),
                        "label": nn.LayerNorm(hidden_dim),
                    }
                )
            )

        self.label_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        self.image_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels, bias=False),
        )

        if initial_biases is not None:
            bias_tensor = torch.tensor(initial_biases, dtype=torch.float32)
        else:
            bias_tensor = torch.zeros(num_labels, dtype=torch.float32)

        self.output_bias = nn.Parameter(bias_tensor)

    def encode_demographics(self, data: HeteroData) -> torch.Tensor:
        field_idx = data["demographic"].field_idx.long()
        kind = data["demographic"].kind.long()
        num_value = data["demographic"].num_value.float()
        cat_value = data["demographic"].cat_value.long()
        missing = data["demographic"].missing.float()

        out = torch.zeros((field_idx.size(0), self.hidden_dim), device=field_idx.device)

        for i, spec in enumerate(self.demographic_specs):
            mask = field_idx == i
            if not mask.any():
                continue

            base = self.demo_field_embed(field_idx[mask]) + self.demo_kind_embed(kind[mask])
            if spec.kind == "numeric":
                emb = self.demo_numeric(torch.cat([num_value[mask], missing[mask]], dim=-1))
            else:
                emb = self.demo_cat_tables[self.cat_field_to_table[i]](cat_value[mask])
            out[mask] = base + emb

        return out + self.type_bias["demographic"]

    def encode_images(self, data: HeteroData) -> torch.Tensor:
        image_idx = data["image"].image_idx.long().clamp(min=0, max=self.max_images_per_patient)
        eye_idx = data["image"].eye_idx.long().clamp(min=0, max=3)
        return self.image_idx_embed(image_idx) + self.eye_idx_embed(eye_idx) + self.type_bias["image"]

    def forward(self, data: HeteroData) -> torch.Tensor:
        
        # --- GNN MESSAGE PASSING ---
        # 1. Standard GNN Dictionary Setup
        x_dict = {
            "image": self.encode_images(data),
            "demographic": self.encode_demographics(data),
            "patient": self.patient_token.expand(data["patient"].num_nodes, -1) + self.type_bias["patient"],
            "label": self.label_token(data["label"].label_idx.long()) + self.type_bias["label"],
        }
        
        # --- RETFOUND FEATURE EXTRACTION ---
        if self.use_image_features:
            # 2. Get the batched raw images from PyG: [Total_Images_in_Batch, 3, 256, 256]
            imgs = data["image"].raw_images 
            
            # 3. Extract Deep Spatial Features using the frozen RETFound model
            with torch.no_grad(): # Keep gradient graph small to save massive amounts of VRAM
                # ViT forward_features outputs shape [B, num_patches + 1, C] where +1 is the class token
                features = self.cnn.forward_features(imgs)
                
            # 4. Reshape the sequence into our patch nodes
            if features.dim() == 3:
                expected_patches = (imgs.shape[2] // 16) * (imgs.shape[3] // 16)
                if features.shape[1] > expected_patches:
                    patch_features_seq = features[:, 1:, :] 
                else:
                    patch_features_seq = features
                    
                b, num_patches, c = patch_features_seq.shape
                patch_features = patch_features_seq.reshape(-1, c)
            else:
                b, c, h, w = features.shape
                patch_features = features.permute(0, 2, 3, 1).reshape(-1, c)
                
            x_dict["patch"] = self.patch_encoder(patch_features) + self.type_bias["patch"]

        for conv, norm in zip(self.convs, self.norms):
            out_dict = conv(x_dict, data.edge_index_dict)
            next_x = {}
            for node_type, x in x_dict.items():
                if node_type in out_dict:
                    h = F.relu(out_dict[node_type])
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    next_x[node_type] = norm[node_type](x + h)
                else:
                    next_x[node_type] = x
            x_dict = next_x

        # --- DYNAMIC OUTPUT ROUTING ---
        if self.prediction_level == "patient":
            # Read from the label nodes (which aggregate the whole patient's state)
            logits = self.label_head(x_dict["label"]).squeeze(-1)
            num_targets = data["patient"].num_nodes
            logits = logits.view(num_targets, self.num_labels)
            return logits + self.output_bias
            
        elif self.prediction_level == "image":
            # Read directly from the enriched image nodes
            # Output inherently matches shape [num_images, num_labels]
            logits = self.image_head(x_dict["image"])
            return logits + self.output_bias