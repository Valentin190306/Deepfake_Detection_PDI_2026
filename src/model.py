"""ViT propio (entrenado desde cero) para clasificación binaria de imágenes.

Entrada : (B, C, S, S) con C = 5 (RGB normalizado + mapa + confianza de TruFor)
          y (B,) con el score global de TruFor.
Salida  : un logit por imagen (manipulada vs auténtica).

Dos decisiones pensadas para el régimen "desde cero con ~2400 imágenes":
  - el score global de TruFor entra como un token extra (prior), así el ViT
    puede apoyarse en el detector ya entrenado;
  - con `residual`, el logit final es `vit_logit + alpha * logit(score)`, con
    alpha=1 inicial. En la época 0 el modelo equivale a TruFor y solo aprende
    una corrección: no puede arrancar peor que el baseline.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth por muestra (regularización fuerte para datasets chicos)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep) / keep
        return x * mask


class LayerScale(nn.Module):
    def __init__(self, dim, init=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward(self, x):
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim, heads, drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.drop = drop
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 ls_init=1e-4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, attn_drop)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hidden, dim), nn.Dropout(drop))
        self.ls1, self.ls2 = LayerScale(dim, ls_init), LayerScale(dim, ls_init)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.n1(x))))
        return x + self.drop_path(self.ls2(self.mlp(self.n2(x))))


def _logit(p, eps=1e-4):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


class TruForViTClassifier(nn.Module):
    def __init__(self, in_chans=5, img_size=384, patch=16, dim=384, depth=8, heads=6,
                 mlp_ratio=4.0, drop=0.15, attn_drop=0.0, drop_path=0.1,
                 use_score=True, residual=True):
        super().__init__()
        assert img_size % patch == 0, "img_size debe ser múltiplo de patch"
        self.grid = img_size // patch
        self.patch_embed = nn.Conv2d(in_chans, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.grid ** 2, dim))

        dpr = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.Sequential(*[
            Block(dim, heads, mlp_ratio, drop, attn_drop, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

        self.use_score = use_score
        self.score_proj = nn.Linear(2, dim) if use_score else None
        self.residual = residual
        self.alpha = nn.Parameter(torch.ones(1)) if residual else None

        self.apply(self._init)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x, score=None, return_parts=False):
        B = x.shape[0]
        t = self.patch_embed(x).flatten(2).transpose(1, 2)          # (B, N, dim)
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1]

        prior = None
        if score is not None:
            prior = _logit(score)
            if self.use_score:
                feats = torch.stack([score, prior], -1)
                cls = cls + self.score_proj(feats).unsqueeze(1)

        t = torch.cat([cls, t + self.pos_embed[:, 1:]], 1)
        t = self.norm(self.blocks(t))
        vit_logit = self.head(t[:, 0]).squeeze(-1)

        logit = vit_logit
        if self.residual and prior is not None:
            logit = logit + self.alpha * prior
        if return_parts:
            # `vit_logit` es la rama propia, sin el prior: sirve para la loss
            # auxiliar, que evita que el modelo se limite a calibrar TruFor.
            return logit, vit_logit
        return logit


def build_model(**cfg) -> TruForViTClassifier:
    """Construye el modelo desde un dict de hiperparámetros (config.json)."""
    keys = ("in_chans", "img_size", "patch", "dim", "depth", "heads", "mlp_ratio",
            "drop", "attn_drop", "drop_path", "use_score", "residual")
    return TruForViTClassifier(**{k: cfg[k] for k in keys if k in cfg})


def n_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
