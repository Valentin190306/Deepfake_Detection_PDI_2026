"""Normalización de tamaño de los mapas de TruFor ANTES de que los use el ViT.

Por qué existe este paso
------------------------
TruFor devuelve los mapas a resolución nativa (hasta ~3000x4400 en IMD2020).
Guardarlos así ocuparía decenas de GB y el ViT igual necesita una entrada de
tamaño fijo, así que la normalización se hace UNA sola vez, inmediatamente
después de TruFor, y se cachea el resultado ya listo para entrenar.

Política (ver README, "Normalización de tamaño")
-----------------------------------------------
El presupuesto de píxeles es SxS: cualquier método descarta la misma cantidad
de información, lo que cambia es *cómo*. Por eso:

  - resize "squash" a SxS (sin padding): no deforma nada importante para una
    clasificación y no gasta píxeles en relleno. Se descartó el letterbox
    (conservar el aspecto y rellenar): el ancho del relleno le muestra al ViT
    la proporción de la foto, que no dice nada de la edición pero puede
    correlacionar con la clase en el dataset (el modelo aprendería el atajo);
    además, en una foto 4:3 se pierde un 25% de los píxeles en relleno, justo
    lo que necesitan las regiones editadas chicas.
  - heatmap: `blend` = 0.5*area + 0.5*max. `area` conserva la forma pero diluye
    regiones chicas; `max` conserva el pico pero engorda el área. El promedio
    de ambos conserva las dos cosas.
  - confianza: `area` (interesa la distribución, no los picos).
  - RGB: bilinear con antialias.

Todo se guarda en uint8 (1/255 de precisión es de sobra para probabilidades).
"""
import numpy as np
import torch
import torch.nn.functional as F

MAP_MODES = ("area", "max", "blend")


def _pool(x: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    """(C,H,W) float -> (C,size,size). Funciona tanto para bajar como para subir."""
    if x.shape[-2:] == (size, size):
        return x
    if mode == "area":
        return F.adaptive_avg_pool2d(x, (size, size))
    if mode == "max":
        return F.adaptive_max_pool2d(x, (size, size))
    if mode == "blend":
        area = F.adaptive_avg_pool2d(x, (size, size))
        peak = F.adaptive_max_pool2d(x, (size, size))
        return 0.5 * area + 0.5 * peak
    raise ValueError(f"modo desconocido: {mode} (válidos: {MAP_MODES})")


def to_tensor(a: np.ndarray) -> torch.Tensor:
    """(H,W) o (C,H,W) numpy -> float32 tensor, con NaN/inf saneados a 0."""
    t = torch.as_tensor(np.asarray(a), dtype=torch.float32)
    if t.ndim == 2:
        t = t[None]
    return torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def normalize_maps(heat: np.ndarray, conf: np.ndarray | None, size: int,
                   heat_mode: str = "blend", conf_mode: str = "area") -> np.ndarray:
    """Mapas de TruFor (HxW, resolución nativa) -> (2, size, size) float32.

    Si `conf` es None (el modelo no la devolvió) se rellena con 0.5: "no sé",
    que es más honesto que 0 ("nada de confianza") o 1 ("total confianza").
    """
    h = _pool(to_tensor(heat), size, heat_mode)
    if conf is None:
        c = torch.full_like(h, 0.5)
    else:
        c = _pool(to_tensor(conf), size, conf_mode)
    return torch.cat([h, c], 0).numpy().astype(np.float32)


def resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    """uint8 HxWx3 -> uint8 size x size x 3, bilinear con antialias."""
    if rgb.shape[:2] == (size, size):
        return np.ascontiguousarray(rgb)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
    t = F.interpolate(t[None], size=(size, size), mode="bilinear",
                      align_corners=False, antialias=True)[0]
    return t.clamp(0, 255).round().byte().permute(1, 2, 0).numpy()


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def from_uint8(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) / 255.0
