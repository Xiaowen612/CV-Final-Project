"""Independent implementation of SVD residual adaptation (Effort-style)."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F


class SVDResidualLinear(nn.Module):
    """Freeze top singular subspace; optimize all three tail factors.

    U0,V0 store the initial residual bases, whose complements are the frozen
    principal subspaces. This avoids storing two additional full square bases.
    """
    def __init__(self, linear, rank=1, initialize=True):
        super().__init__()
        w = linear.weight.detach().float()
        if w.shape[0] != w.shape[1] or not 0 < rank < min(w.shape):
            raise ValueError('Effort attention projections must be square, with 0 < tail rank < width.')
        n = w.shape[0]
        if initialize:
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            ur, sr, vr = u[:, -rank:], s[-rank:], vh[-rank:]
            main = w - (ur * sr) @ vr
        else:
            ur, sr, vr = torch.zeros(n, rank), torch.zeros(rank), torch.zeros(rank, n)
            main = torch.zeros_like(w)
        self.register_buffer('main', main.clone())
        self.register_buffer('u0', ur.clone())
        self.register_buffer('v0', vr.clone())
        self.register_buffer('original_energy', w.square().sum())
        self.register_buffer('main_energy', main.square().sum())
        self.register_buffer('bias', linear.bias.detach().clone() if linear.bias is not None else None)
        self.u, self.s, self.v = nn.Parameter(ur.clone()), nn.Parameter(sr.clone()), nn.Parameter(vr.clone())

    def forward(self, x):
        return F.linear(x, self.main, self.bias) + F.linear(F.linear(x, self.v) * self.s, self.u)

    def constraints(self):
        # Equivalent to Frobenius norms of full concatenated-basis orthogonality
        # errors, using only rank-sized Gram matrices and complement projections.
        u, v, s = self.u.float(), self.v.float(), self.s.float()
        identity = torch.eye(len(s), device=s.device)
        u_perp = u - self.u0 @ (self.u0.T @ u)
        v_perp = v - (v @ self.v0.T) @ self.v0
        u_error = 2 * u_perp.square().sum() + (u.T @ u - identity).square().sum()
        v_error = 2 * v_perp.square().sum() + (v @ v.T - identity).square().sum()
        orth = 0.5 * torch.sqrt(u_error + 1e-12) + 0.5 * torch.sqrt(v_error + 1e-12)
        cross = ((u.T @ self.main) * v).sum(dim=1)
        tail_energy = ((u.T @ u) * (v @ v.T) * s[:, None] * s[None, :]).sum()
        energy = (self.main_energy + 2 * (s * cross).sum() + tail_energy - self.original_energy).abs()
        return orth, energy


class Detector(nn.Module):
    def __init__(self, cfg, method, pretrained=True, model_spec=None):
        super().__init__()
        self.method = method
        self.is_clip = method.startswith('clip') or method.startswith('effort')
        self.model_spec = model_spec or {}
        if self.is_clip:
            from transformers import CLIPVisionModel, CLIPVisionConfig
            if pretrained:
                self.backbone = CLIPVisionModel.from_pretrained(cfg['clip_id'], revision=cfg.get('clip_revision', 'main'))
            else:
                self.backbone = CLIPVisionModel(CLIPVisionConfig(**self.model_spec))
            self.model_spec = self.backbone.config.to_dict()
            self.head = nn.Linear(self.backbone.config.hidden_size, 1)
            if method != 'clip_full':
                self.backbone.requires_grad_(False)
            if method.startswith('effort'):
                for layer in self.backbone.vision_model.encoder.layers:
                    for name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                        old = getattr(layer.self_attn, name)
                        setattr(layer.self_attn, name, SVDResidualLinear(old, cfg['residual_rank'], pretrained))
            if cfg.get('gradient_checkpointing') and method != 'clip_linear':
                self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        elif method == 'tiny_smoke':
            self.backbone = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.head = nn.Linear(8, 1)
        else:
            import timm
            names = {'resnet50': 'resnet50', 'cnnspot': 'resnet50',
                     'deit_small': 'deit_small_patch16_224', 'swin_tiny': 'swin_tiny_patch4_window7_224'}
            if method not in names:
                raise ValueError(f'Unknown method {method}')
            self.backbone = timm.create_model(names[method], pretrained=pretrained, num_classes=0)
            self.head = nn.Linear(self.backbone.num_features, 1)

    def train(self, mode=True):
        super().train(mode)
        if self.method == 'clip_linear':
            self.backbone.eval()
        return self

    def forward(self, x):
        with torch.no_grad() if self.method == 'clip_linear' else nullcontext():
            feature = self.backbone(pixel_values=x).pooler_output if self.is_clip else self.backbone(x)
        return self.head(feature).squeeze(-1)

    def regularization(self, cfg):
        orth_weight, energy_weight = cfg['lambda_orth'], cfg['lambda_energy']
        if self.method in {'effort_no_orth', 'effort_no_constraints'}:
            orth_weight = 0
        if self.method in {'effort_no_energy', 'effort_no_constraints'}:
            energy_weight = 0
        modules = [m for m in self.modules() if isinstance(m, SVDResidualLinear)]
        if not modules or not (orth_weight or energy_weight):
            return self.head.weight.sum() * 0
        # Constraints are kept in float32 even when image forward uses AMP.
        with torch.autocast(device_type=self.head.weight.device.type, enabled=False):
            penalties = [m.constraints() for m in modules]
            return sum(orth_weight * a + energy_weight * b for a, b in penalties) / len(modules)
