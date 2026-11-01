"""Dataset: imagen original + mapas cacheados de TruFor (por variante) -> mascara real."""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def list_images(root):
    return sorted(p for p in (Path(root) / "images").iterdir() if p.suffix.lower() in EXT)


def split(paths, val_frac=0.2, seed=0, group=lambda p: p.stem.split("__")[0]):
    """Divide por GRUPO (no por archivo) para que una imagen y sus derivados
    (par real/falso, recortes) no queden en train y val a la vez.
    Por defecto el grupo es el stem hasta '__' (prepare_imd2020.py nombra <clave>__auth a la autentica
    de <clave>). Ajusta `group` si tu dataset usa otra convencion."""
    groups = sorted({group(p) for p in paths})
    random.Random(seed).shuffle(groups)
    val_g = set(groups[: int(len(groups) * val_frac)])
    train = [p for p in paths if group(p) not in val_g]
    val = [p for p in paths if group(p) in val_g]
    return train, val


def find_mask(root, stem):
    for name in (f"{stem}.png", f"{stem}_mask.png", f"{stem}.tif", f"{stem}.jpg"):
        p = Path(root) / "masks" / name
        if p.exists():
            return p
    return None  # imagen autentica: sin mascara => todo ceros


class ForensicsDataset(Dataset):
    """Canales de entrada: RGB(3) + por cada variante [mapa, confianza] => 3 + 2*V."""

    def __init__(self, root, cache, paths, variants=("raw",), size=512, train=False):
        self.root, self.cache = Path(root), Path(cache)
        self.paths, self.variants = paths, list(variants)
        self.size, self.train = size, train

    def __len__(self):
        return len(self.paths)

    def _resize(self, t, mode):
        kw = {"align_corners": False} if mode == "bilinear" else {}
        return F.interpolate(t[None], size=(self.size, self.size), mode=mode, **kw)[0]

    def __getitem__(self, i):
        p = self.paths[i]
        rgb = np.array(Image.open(p).convert("RGB"))
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        H, W = rgb.shape[1:]

        chans = []
        for v in self.variants:
            d = np.load(self.cache / v / f"{p.stem}.npz")
            m = torch.from_numpy(d["map"]).float()
            c = torch.from_numpy(d["conf"]).float() if "conf" in d.files else torch.zeros_like(m)
            chans += [m[None], c[None]]
        maps = torch.cat(chans, 0)
        if tuple(maps.shape[1:]) != (H, W):
            raise ValueError(f"{p.name}: el mapa de TruFor {tuple(maps.shape[1:])} no coincide con la imagen {(H, W)}")

        mp = find_mask(self.root, p.stem)
        if mp is None:
            y = torch.zeros(1, H, W)
        else:
            m = np.array(Image.open(mp).convert("L"))
            if m.shape != (H, W):   # ej: rotacion EXIF u otro tamano => mascara desalineada
                raise ValueError(f"{p.name}: la mascara {m.shape} no coincide con la imagen {(H, W)}")
            y = (torch.from_numpy(m) > 127).float()[None]

        # El resize va DESPUES de TruFor (que corrio a resolucion nativa)
        x = torch.cat([(self._resize(rgb, "bilinear") - MEAN) / STD,
                       self._resize(maps, "bilinear")], 0)
        y = self._resize(y, "nearest")

        if self.train:  # grupo dihedral: se aplica igual a imagen, mapas y mascara
            if random.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            k = random.randint(0, 3)
            if k:
                x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
        return x, y
