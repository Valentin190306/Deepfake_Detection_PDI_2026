"""Preprocesamientos aplicados ANTES de que la imagen entre a TruFor.

Una VARIANTE es una lista ordenada de pasos. Cada paso recibe y devuelve
uint8 RGB HxWx3. TruFor trabaja a resolución nativa, así que estos pasos son
la única forma de simular condiciones reales (recompresión, borroneo, etc.)
sin desincronizar los mapas: si se aplicaran DESPUÉS de TruFor, el mapa ya no
correspondería a la imagen que ve el clasificador.

Para agregar una variante nueva:
  1) escribir la función del paso (uint8 RGB -> uint8 RGB)
  2) registrarla en STEPS
  3) declarar la variante en VARIANTS
No hay que tocar extract_maps.py ni dataset.py.

`--max-side` de extract_maps.py y detect.py es aparte: es un límite técnico
de memoria para la GPU, no una variante científica. Se aplica ANTES de la
variante, así los filtros trabajan a la resolución que ve TruFor.
"""
import io
from functools import partial

import cv2
import numpy as np
from PIL import Image, ImageFilter


def identity(img: np.ndarray) -> np.ndarray:
    return img


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """Recompresión JPEG en memoria (mismo tamaño)."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.array(im.convert("RGB"))


def gaussian(img: np.ndarray, radius: float) -> np.ndarray:
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius)))


def max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    """Baja el lado mayor a max_side conservando aspecto. Nunca agranda."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.array(Image.fromarray(img).resize(size, Image.LANCZOS))


def log_residual(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gauss (sigma) -> laplaciano 3x3, devuelto como residuo centrado en gris.

    Es el filtro que ganó el barrido de scripts/preproc_test.py (rama testing):
    el Gauss chico saca el ruido más fino y el laplaciano deja sólo las altas
    frecuencias. El residuo se escala por el percentil 99 de |laplaciano| (el
    mismo para los tres canales) para usar todo el rango uint8. Replica
    exactamente aquel barrido: el umbral calibrado en detect.py depende de eso.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    lap = cv2.Laplacian(blurred.astype(np.float32), cv2.CV_32F, ksize=1)
    scale = max(float(np.percentile(np.abs(lap), 99)), 1e-6)
    return np.clip(np.rint(128 + 127 * lap / scale), 0, 255).astype(np.uint8)


STEPS = {
    "identity": identity,
    "jpeg70": partial(jpeg, quality=70),
    "jpeg90": partial(jpeg, quality=90),
    "blur1": partial(gaussian, radius=1.0),
    "resize1280": partial(max_side, max_side=1280),
    "resize1024": partial(max_side, max_side=1024),
    "log05": partial(log_residual, sigma=0.5),
}

VARIANTS = {
    "raw": ["identity"],
    "jpeg70": ["jpeg70"],
    "jpeg90": ["jpeg90"],
    "blur1": ["blur1"],
    "resize1280": ["resize1280"],
    "resize1024+jpeg70": ["resize1024", "jpeg70"],
    "log05": ["log05"],
}


def check_variant(variant: str) -> str:
    if variant not in VARIANTS:
        raise SystemExit(
            f"Variante desconocida: {variant} (definidas: {', '.join(VARIANTS)})")
    return variant


def is_untouched(variant: str) -> bool:
    return all(s == "identity" for s in VARIANTS[variant])


def apply_variant(img: np.ndarray, variant: str) -> np.ndarray:
    """Aplica la variante y valida el contrato: uint8 RGB HxWx3."""
    check_variant(variant)
    for step in VARIANTS[variant]:
        img = STEPS[step](img)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"la variante '{variant}' devolvió {img.dtype} {img.shape}; se esperaba uint8 HxWx3")
    return np.ascontiguousarray(img)
