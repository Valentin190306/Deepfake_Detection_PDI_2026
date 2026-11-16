"""Backbones preentrenados (timm) para el clasificador sobre mapas de TruFor.

Por qué un backbone preentrenado
--------------------------------
El ViT desde cero colapsó a calibrar el score de TruFor: con ~1000 imágenes
únicas no alcanza para aprender detección desde cero. Un backbone preentrenado
arranca sabiendo "cómo es una foto", y sólo tiene que aprender la parte nueva:
qué pinta tiene una edición.

Los cinco canales
-----------------
El backbone espera 3 canales (RGB) y nosotros tenemos 5 (RGB + mapa + confianza).
El stem se "infla": se copian los pesos RGB y los dos canales nuevos arrancan en
cero. Al inicio el modelo es exactamente el preentrenado (los mapas no aportan
nada) y los canales nuevos aprenden desde la primera época, porque el gradiente
es x·δ y el mapa no es cero.

El stem queda entrenable en TODOS los modos, incluso con el backbone congelado:
si se congela, los mapas no pueden entrar al modelo y el experimento mide otra
cosa.

Modos de entrenamiento
----------------------
  linear   backbone congelado; se entrena el stem, la norma final y la cabeza.
           Mide cuánto valen los features preentrenados (transfer learning).
  partial  además se descongelan las últimas etapas/bloques. Con datasets chicos
           suele ser el mejor punto: lo genérico queda congelado y lo específico
           se adapta.
  full     todo entrenable, con lr más bajo para el backbone y decaimiento por
           capa (las primeras capas se mueven menos).
"""
import timm
import torch
import torch.nn as nn

# nombre corto -> modelo de timm. Los cuatro están disponibles en timm 0.9 y 1.x.
# Requiere timm 1.0.x: los pesos se bajan de HuggingFace y con timm 0.9.16 +
# huggingface_hub 2.x la descarga falla (ver requirements.txt).
BACKBONES = {
    "swin_t": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "vit_s16": "vit_small_patch16_384.augreg_in21k_ft_in1k",
    "vit_b16": "vit_base_patch16_384.augreg_in21k_ft_in1k",
    "convnext_t": "convnext_tiny.in12k_ft_in1k",
}


def find_stem(model) -> tuple[str, nn.Conv2d]:
    """Primer conv que mira la imagen completa (patch embed / stem)."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.in_channels == 3 and min(mod.kernel_size) > 1:
            return name, mod
    raise RuntimeError("no encontré el stem (conv de 3 canales) del backbone")


def inflate_stem(model, in_chans: int) -> str | None:
    """Reemplaza el conv de entrada por uno de `in_chans` canales.

    - in_chans > 3: copia los pesos RGB y deja los canales nuevos en cero.
    - in_chans < 3: cada canal nuevo es el promedio de los kernels RGB (mantiene
      la escala; sólo se usa en la ablación de mapas sin RGB).
    Devuelve el nombre del módulo inflado, o None si no hacía falta.
    """
    if in_chans == 3:
        return None
    name, old = find_stem(model)
    new = nn.Conv2d(in_chans, old.out_channels, old.kernel_size, old.stride,
                    old.padding, old.dilation, old.groups, old.bias is not None)
    with torch.no_grad():
        if in_chans > 3:
            new.weight.zero_()
            new.weight[:, :3].copy_(old.weight)
        else:
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, new)
    return name


def create_backbone(name: str, in_chans: int = 5, img_size: int = 384,
                    pretrained: bool = True, drop: float = 0.0,
                    drop_path: float = 0.1):
    """Crea el backbone sin cabeza (`num_classes=0`) y lo adapta a `in_chans`.

    Con `num_classes=0` el `forward` devuelve el vector de features ya pooled
    (768 en Swin-T, 384 en ViT-S), listo para una capa lineal.

    `img_size` no lo aceptan todos los modelos en timm 0.9.x (ConvNeXt, por
    ejemplo, es convolucional puro y no lo necesita), así que se pasa sólo si el
    constructor lo admite.
    """
    if name not in BACKBONES:
        raise SystemExit(f"backbone desconocido: {name} (válidos: {', '.join(BACKBONES)})")
    kw = dict(pretrained=pretrained, num_classes=0, drop_rate=drop,
              drop_path_rate=drop_path)
    try:
        model = _create(BACKBONES[name], img_size, **kw)
    except (RuntimeError, ImportError) as e:
        # timm 0.9.x + huggingface_hub 1/2: los pesos están en HF y no se pueden
        # bajar. El requirements.txt fija timm 1.0.x justamente por esto.
        msg = str(e).lower()
        if pretrained and ("hugging" in msg or "hf_hub" in msg or "hub model" in msg):
            raise SystemExit(
                f"No pude bajar los pesos de {BACKBONES[name]}: {e}\n"
                f"Es un choque de versiones. Actualizá timm (pip install -U 'timm>=1.0') "
                f"o usá --no-pretrained para entrenar sin pesos de ImageNet.") from e
        raise
    stem = inflate_stem(model, in_chans)
    return model, stem


def _create(model_name: str, img_size: int, **kw):
    """`timm.create_model` tolerando modelos que no aceptan `img_size`."""
    try:
        return timm.create_model(model_name, img_size=img_size, **kw)
    except TypeError as e:
        if "img_size" not in str(e):
            raise
        # modelo sin `img_size` en el constructor: usa su default y acepta
        # cualquier entrada en el forward (los convolucionales son adaptativos)
        return timm.create_model(model_name, **kw)


def stage_units(model):
    """Lista de etapas (Swin/ConvNeXt) o bloques (ViT), de entrada a salida."""
    for attr in ("layers", "stages", "blocks"):
        mod = getattr(model, attr, None)
        if isinstance(mod, (nn.ModuleList, nn.Sequential)):
            return attr, list(mod)
    return None, []


def layer_scale(name: str, n_units: int, decay: float) -> float | None:
    """Factor de lr para un parámetro según su profundidad.

    Las primeras capas se mueven menos que las últimas (`decay ** (n - i)`), que
    es lo que evita destruir features genéricos al hacer fine-tuning completo.
    Devuelve None si el parámetro no pertenece a ninguna etapa (norm final,
    stem), y en ese caso el llamador decide el lr.
    """
    if decay <= 0:
        return 1.0
    parts = name.split(".")
    for key in ("layers", "stages", "blocks"):
        if key in parts:
            i = parts.index(key)
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                return decay ** (n_units - int(parts[i + 1]))
    return None


def default_unfreeze(attr: str | None, n_units: int) -> int:
    """Cuántas unidades descongelar en `partial` si no se especifica.

    En un ViT (`blocks`) los últimos 4; en un modelo jerárquico (`layers` /
    `stages`) la última etapa, que es la más específica y la más chica.
    """
    return min(4, n_units) if attr == "blocks" else 1


def configure_trainable(model, ft: str, unfreeze_blocks: int = 0, stem: str | None = None):
    """Congela/descongela el backbone según el modo. Devuelve cuántas unidades."""
    if ft not in ("linear", "partial", "full"):
        raise SystemExit(f"--ft desconocido: {ft} (linear, partial, full)")
    for p in model.parameters():
        p.requires_grad = False

    # el stem siempre: es la única puerta de entrada de los mapas
    if stem:
        for p in model.get_submodule(stem).parameters():
            p.requires_grad = True
    # la norma final va con la cabeza: es la que deja los features "listos"
    if hasattr(model, "norm"):
        for p in model.norm.parameters():
            p.requires_grad = True

    if ft == "full":
        for p in model.parameters():
            p.requires_grad = True
        return len(stage_units(model)[1])

    if ft == "linear":
        return 0

    attr, units = stage_units(model)
    n = unfreeze_blocks or default_unfreeze(attr, len(units))
    n = max(0, min(n, len(units)))
    for u in units[len(units) - n:]:
        for p in u.parameters():
            p.requires_grad = True
    return n


def _fmt(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def describe(model, stem: str | None) -> str:
    """Resumen legible para el log de entrenamiento."""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    attr, units = stage_units(model)
    return (f"{_fmt(total)} params ({_fmt(train)} entrenables) | "
            f"{len(units)} {attr or 'unidades'} | stem={'inflado' if stem else 'RGB'}")
