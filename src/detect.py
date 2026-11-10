"""
detect.py - Detección de manipulación de imágenes con TruFor (modelo preentrenado).

NO entrena nada. Baja/usa un checkpoint ya entrenado y clasifica imágenes.

Para cada imagen devuelve:
  - score   : probabilidad global de que la imagen esté manipulada (0 a 1)
  - veredicto: MANIPULADA / AUTENTICA según --threshold
  - mapa    : heatmap de DÓNDE estaría la manipulación (localización)
  - conf    : mapa de confianza del propio modelo sobre su predicción

Uso:

    # Una imagen
    python src/detect.py third_party/trufor/images/tampered1.png

    # Una carpeta entera + visualizaciones PNG
    python src/detect.py mis_imagenes/ --out runs/deteccion --viz

    # Resultados a JSON para procesar después
    python src/detect.py mis_imagenes/ --json runs/resultados.json

    # Sin filtro ni límite de tamaño (TruFor tal cual, como en el paper)
    python src/detect.py mis_imagenes/ --variant raw --max-side 0

Por defecto la imagen se achica (conservando el aspecto) hasta que el lado
mayor mide 1000 px, se le aplica el filtro ganador del barrido de
preprocesamientos (variante "log05": Gauss sigma=0.5 -> laplaciano) y el
veredicto usa el umbral calibrado para esa variante (ver THRESHOLDS).

El modelo es TruFor (Guillaro et al., CVPR 2023), GRIP-UNINA. Licencia de uso
sin fines de lucro: hay que citar a los autores si se publican resultados.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import VARIANTS, apply_variant  # noqa: E402
from preprocess import max_side as limit_side  # noqa: E402

# El código y los pesos de TruFor viven en third_party/ (no son nuestros).
REPO = Path(__file__).resolve().parent.parent
TRUFOR_SRC = REPO / "third_party" / "trufor" / "src"

# Extensiones que PIL abre sin drama. Evita intentar cargar .txt, .npz, etc.
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Lado máximo con el que entra en una GPU de 8 GB (y con el que se calibraron
# los umbrales de abajo).
DEFAULT_MAX_SIDE = 1000
DEFAULT_VARIANT = "log05"

# Umbral calibrado por variante: el que maximizó la balanced accuracy en el
# barrido de scripts/preproc_test.py (rama testing), sobre 21 imágenes propias
# (10 auténticas, 11 manipuladas) limitadas a 1000 px. Con validación cruzada
# de 5 folds, log05 dio balanced accuracy 0.764 contra 0.718 sin filtro.
# Con tan pocas imágenes la diferencia es de ~1 imagen: si cambia el filtro,
# el límite de tamaño o el tipo de imágenes, hay que recalibrar.
# Las variantes que no están acá usan 0.5, el umbral original de TruFor.
THRESHOLDS = {
    "log05": 0.1315,
    "raw": 0.3525,
}


def find_weights():
    """Busca trufor.pth.tar donde sea que lo haya dejado el unzip."""
    hits = sorted((REPO / "third_party" / "trufor").rglob("trufor.pth.tar"))
    if not hits:
        sys.exit(
            "No encontré trufor.pth.tar.\n"
            "Bajalo con:\n"
            "  curl -L -o /tmp/w.zip https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip\n"
            "  unzip /tmp/w.zip -d third_party/trufor/"
        )
    return hits[0]


def load_model(device):
    """Construye la red TruFor y le carga los pesos entrenados."""
    if str(TRUFOR_SRC) not in sys.path:
        sys.path.insert(0, str(TRUFOR_SRC))

    from config import _C as config
    from models.cmx.builder_np_conf import myEncoderDecoder

    weights = find_weights()
    config.defrost()
    config.merge_from_file(str(TRUFOR_SRC / "trufor.yaml"))
    # PRETRAINED queda vacío a propósito: el checkpoint ya trae TODO
    # (backbone SegFormer + Noiseprint++ + decoder), no hay que bajar extras.
    config.MODEL.PRETRAINED = ""
    config.freeze()

    model = myEncoderDecoder(cfg=config)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, weights


def preprocess(path, variant=DEFAULT_VARIANT, max_side=DEFAULT_MAX_SIDE):
    """
    RGB uint8 -> (límite de tamaño) -> variante -> float en [0, 1).

    TruFor acepta cualquier tamaño, pero las fotos grandes no entran en la GPU:
    si max_side > 0 se baja el lado mayor a ese valor conservando el aspecto
    (nunca se agranda ni se deforma). Achicar borra parte de las huellas de
    bajo nivel que busca TruFor, así que con max_side=0 no se toca.
    """
    arr = np.array(Image.open(path).convert("RGB"))
    if max_side:
        arr = limit_side(arr, max_side)
    t = torch.tensor(apply_variant(arr, variant).transpose(2, 0, 1), dtype=torch.float) / 256.0
    # (1,3,H,W) para la red, y la imagen sin filtrar (mismo tamaño que los mapas) para visualizar
    return t.unsqueeze(0), arr


@torch.no_grad()
def analyze(model, rgb, device):
    """Una pasada del modelo. Devuelve score global, mapa de manipulación y confianza."""
    pred, conf, det, _npp = model(rgb.to(device))

    # det -> un único logit de imagen completa: ESTE es el clasificador binario
    score = torch.sigmoid(det).item() if det is not None else float("nan")
    # pred -> 2 canales por píxel; el canal 1 es "manipulado"
    heatmap = torch.softmax(pred.squeeze(0), dim=0)[1].cpu().numpy()
    conf_map = torch.sigmoid(conf.squeeze(0))[0].cpu().numpy() if conf is not None else None
    return score, heatmap, conf_map


def save_viz(out_png, original, heatmap, conf_map, score, verdict):
    """Guarda una figura: imagen original | heatmap | overlay (+ confianza)."""
    import matplotlib
    matplotlib.use("Agg")  # sin ventana, solo archivo
    import matplotlib.pyplot as plt

    n = 3 if conf_map is None else 4
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4))

    axes[0].imshow(original)
    axes[0].set_title("original")

    axes[1].imshow(heatmap, cmap="RdBu_r", vmin=0, vmax=1)
    axes[1].set_title("mapa de manipulación")

    axes[2].imshow(original)
    axes[2].imshow(heatmap, cmap="RdBu_r", vmin=0, vmax=1, alpha=0.5)
    axes[2].set_title("overlay")

    if conf_map is not None:
        axes[3].imshow(conf_map, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("confianza del modelo")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{Path(out_png).stem}  —  score {score:.3f}  →  {verdict}", fontsize=13)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def collect_images(target):
    """Acepta un archivo, una carpeta (recursiva) o un glob."""
    p = Path(target)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
    hits = sorted(Path().glob(target))
    return [f for f in hits if f.is_file() and f.suffix.lower() in IMG_EXT]


def main():
    ap = argparse.ArgumentParser(description="Detecta manipulación en imágenes con TruFor")
    ap.add_argument("input", help="imagen, carpeta o glob")
    ap.add_argument("--out", default="runs/deteccion", help="carpeta de salida para las visualizaciones")
    ap.add_argument("--viz", action="store_true", help="guardar PNG con heatmap por imagen")
    ap.add_argument("--json", default=None, help="guardar todos los scores en este .json")
    ap.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(VARIANTS),
                    help=f"filtro antes de TruFor (default: {DEFAULT_VARIANT}; raw = sin filtro)")
    ap.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                    help="lado máximo en px; las imágenes más grandes se achican conservando el "
                         f"aspecto (0 = no achicar, default: {DEFAULT_MAX_SIDE})")
    ap.add_argument("--threshold", type=float, default=None,
                    help="score por encima del cual se declara MANIPULADA "
                         "(default: el calibrado para la variante, ver THRESHOLDS; si no hay, 0.5)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()
    if args.threshold is None:
        args.threshold = THRESHOLDS.get(args.variant, 0.5)
        if args.max_side != DEFAULT_MAX_SIDE:
            print(f"AVISO: el umbral de '{args.variant}' se calibró con --max-side {DEFAULT_MAX_SIDE}; "
                  f"con {args.max_side} puede no ser el mejor.")

    images = collect_images(args.input)
    if not images:
        sys.exit(f"No encontré imágenes en: {args.input}")

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    model, weights = load_model(device)
    print(f"Modelo: TruFor | pesos: {weights.name} | dispositivo: {device}")
    print(f"{len(images)} imagen(es) | variante: {args.variant} | "
          f"lado máximo: {args.max_side or 'sin límite'} | umbral: {args.threshold}\n")

    results = []
    for path in images:
        rgb, original = preprocess(path, args.variant, args.max_side)
        try:
            score, heatmap, conf_map = analyze(model, rgb, device)
        except torch.cuda.OutOfMemoryError:
            # Si aun con el límite de tamaño no entra (o se usó --max-side 0),
            # reintentamos en CPU: lento pero sin achicar más la imagen.
            print(f"  (sin memoria GPU en {path.name}, reintento en CPU)")
            torch.cuda.empty_cache()
            model.to("cpu")
            score, heatmap, conf_map = analyze(model, rgb, torch.device("cpu"))
            model.to(device)

        verdict = "MANIPULADA" if score >= args.threshold else "AUTENTICA"
        print(f"  {score:.4f}  {verdict:<10}  {path}")

        results.append({
            "file": str(path),
            "score": score,
            "verdict": verdict,
            "variant": args.variant,
            "threshold": args.threshold,
            "size": list(original.shape[:2]),   # tamaño con el que entró a TruFor
            # Cuánto del área se marca como manipulada: útil para descartar
            # activaciones dispersas que son ruido y no una región editada.
            "area_sospechosa": float((heatmap >= 0.5).mean()),
        })

        if args.viz:
            save_viz(Path(args.out) / f"{path.stem}_trufor.png",
                     original, heatmap, conf_map, score, verdict)

    if args.viz:
        print(f"\nVisualizaciones en {args.out}/")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"Resultados en {args.json}")


if __name__ == "__main__":
    main()
