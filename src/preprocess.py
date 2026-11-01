"""Registro de preprocesamientos.

Una VARIANTE = una lista de pasos aplicados antes de correr TruFor.
Hoy solo existe 'raw' (sin tocar la imagen). Para agregar una nueva:
  1) escribir la funcion del paso (uint8 RGB HxWx3 -> uint8 RGB HxWx3)
  2) registrarla en STEPS
  3) declarar la variante en VARIANTS
No hay que tocar dataset.py ni train.py.
"""
import numpy as np


def identity(img: np.ndarray) -> np.ndarray:
    return img


STEPS = {
    "identity": identity,
    # Ejemplo futuro (no activo):
    # "whites-15": lambda im: reduce_whites(im, 0.15),
}

VARIANTS = {
    "raw": ["identity"],
    # "whites-15": ["whites-15"],
}


def is_untouched(variant: str) -> bool:
    return all(s == "identity" for s in VARIANTS[variant])


def apply_variant(img: np.ndarray, variant: str) -> np.ndarray:
    for step in VARIANTS[variant]:
        img = STEPS[step](img)
    return img
