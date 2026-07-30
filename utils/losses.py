import torch
import torch.nn as nn

class AsymmetricLossOptimized(nn.Module):
    """
    Optimized implementation of Asymmetric Loss for multi-label classification.
    Ref: https://arxiv.org/abs/2009.14119
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        
        # We also accept pos_weight to maintain interface compatibility with BCEWithLogitsLoss,
        # but in ASL it's less necessary. We can apply it if needed.
        self.pos_weight = None

    def forward(self, x, y):
        # x is logits, y is targets
        
        # Calculate probabilities
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos
        
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
            
        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        
        # Asymmetric Focusing
        loss = los_pos
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
                
            loss *= one_sided_w
            los_neg *= one_sided_w
            
        loss = -loss - los_neg
        
        # Apply positive weight if configured by the orchestrator
        if self.pos_weight is not None:
            # We only scale the positive components of the loss
            loss = torch.where(y == 1, loss * self.pos_weight, loss)
            
        return loss.mean()
