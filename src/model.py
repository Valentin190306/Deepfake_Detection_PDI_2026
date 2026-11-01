"""ViT propio (entrenado desde cero, sin pesos preentrenados) para segmentacion por pixel.

Entrada : (B, C, H, W) con C = 3 (RGB normalizado) + 2 por variante (mapa y confianza de TruFor).
Salida  : logits (B, 1, H, W).

Piezas: patch embedding -> + posicion aprendible -> N bloques Transformer (pre-LN) -> decoder
convolucional que recupera la resolucion original. Se concatena la entrada original a la
ultima etapa (skip) para que el modelo pueda apoyarse directamente en el mapa de TruFor.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads, drop):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.drop = heads, drop
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class ViTEncoder(nn.Module):
    def __init__(self, in_chans, img_size, patch, dim, depth, heads, drop):
        super().__init__()
        assert img_size % patch == 0
        self.grid = img_size // patch
        self.patch_embed = nn.Conv2d(in_chans, dim, patch, patch)     # parches + proyeccion lineal
        self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2, dim))   # sin token [CLS]: no hace falta
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.Sequential(*[Block(dim, heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        t = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos   # (B, N, dim)
        t = self.norm(self.blocks(self.drop(t)))
        return t.transpose(1, 2).reshape(x.size(0), -1, self.grid, self.grid)


class Up(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                         nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ViTRefiner(nn.Module):
    def __init__(self, in_chans, img_size=512, patch=16, dim=256, depth=8, heads=8,
                 drop=0.1, residual=False):
        super().__init__()
        assert patch & (patch - 1) == 0, "patch debe ser potencia de 2"
        self.enc = ViTEncoder(in_chans, img_size, patch, dim, depth, heads, drop)
        self.proj = nn.Conv2d(dim, 256, 1)
        ups, c = [], 256
        for _ in range(int(math.log2(patch))):
            ups.append(Up(c, max(c // 2, 32)))
            c = max(c // 2, 32)
        self.up = nn.Sequential(*ups)
        self.head = nn.Sequential(nn.Conv2d(c + in_chans, 64, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 1, 1))
        # Modo residual: el modelo parte del baseline (logit del mapa de TruFor) y aprende una correccion
        self.residual = residual
        if residual:
            self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        f = self.up(self.proj(self.enc(x)))
        out = self.head(torch.cat([f, x], 1))
        if self.residual:
            m = x[:, 3:4].clamp(1e-4, 1 - 1e-4)            # canal 3 = mapa TruFor de la 1ra variante
            out = out + self.alpha * torch.log(m / (1 - m))
        return out
