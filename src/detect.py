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

El preprocesamiento de la entrada (resize, suavizado, brillo, blancos, CLAHE,
realce) se configura en la lista PREPROC, más abajo. Para comparar varias
configuraciones sobre un directorio, ver run_experiments().

El modelo es TruFor (Guillaro et al., CVPR 2023), GRIP-UNINA. Licencia de uso
sin fines de lucro: hay que citar a los autores si se publican resultados.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# El código y los pesos de TruFor viven en third_party/ (no son nuestros).
REPO = Path(__file__).resolve().parent.parent
TRUFOR_SRC = REPO / "third_party" / "trufor" / "src"

# Extensiones que PIL abre sin drama. Evita intentar cargar .txt, .npz, etc.
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Preprocesamiento de la entrada. Los pasos activos se aplican en este orden,
# antes de pasarle la imagen a TruFor: para prender/apagar uno, cambiar el
# True/False; para cambiar el orden, mover la línea.
#
# Ojo: TruFor se apoya en el ruido de bajo nivel (Noiseprint++). Lo que suaviza
# o redimensiona borra parte de esa señal; lo que realza altas frecuencias o el
# contraste local la amplifica (y también los artefactos de JPEG). Comparar
# siempre contra la corrida sin el paso.
PREPROC = [
    # (paso,      activo, parámetros)
    ("resize",    False,  {"size": (600, 800)}),  # (ancho, alto)
    ("suavizado", True, {"metodo": "bilateral", "d": 5, "sigma_color": 25, "sigma_space": 5}),
    ("brillo",    True, {"factor": 1.0, "offset": 0, "gamma": 1.0}),
    ("blancos",   False, {"umbral": 200, "fuerza": 0.5}),
    ("clahe",     False, {"clip": 2.0, "grilla": 8}),
    ("realce",    False, {"sigma": 1.0, "cantidad": 0.5}),
]


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


# Pasos de preprocesamiento: todos reciben y devuelven RGB uint8 (H,W,3).

def _to_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def resize(img, size=(600, 800)):
    """
    Redimensiona a (ancho, alto) con Lanczos, sin respetar la relación de aspecto.
    TruFor acepta cualquier tamaño: esto solo ahorra memoria, a costa de
    interpolar justamente las huellas de bajo nivel que busca.
    """
    return np.array(Image.fromarray(img).resize(size, Image.Resampling.LANCZOS))


def denoise(img, metodo="bilateral", sigma=1.0, ksize=3, d=5, sigma_color=25, sigma_space=5, h=5):
    """
    Suavizado de ruido. Cada método usa sus propios parámetros:
      gaussiano (sigma)                  : borra el ruido parejo, bordes incluidos.
      mediana   (ksize, impar)           : bueno contra ruido impulsivo (sal y pimienta).
      bilateral (d, sigma_color, sigma_space): suaviza zonas planas y respeta los bordes.
      nlmeans   (h = intensidad)         : non-local means, el más fuerte y el más lento.
    """
    if metodo == "gaussiano":
        return cv2.GaussianBlur(img, (0, 0), sigma)
    if metodo == "mediana":
        return cv2.medianBlur(img, ksize)
    if metodo == "bilateral":
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    if metodo == "nlmeans":
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    raise ValueError(f"suavizado: método '{metodo}' desconocido (gaussiano, mediana, bilateral, nlmeans)")


def brightness(img, factor=1.0, offset=0, gamma=1.0):
    """
    Ajuste de brillo: primero gamma (no lineal), después factor * x + offset.
    gamma < 1 levanta las sombras, donde el ruido del sensor queda más a la vista.
    """
    x = (img.astype(np.float32) / 255.0) ** gamma
    return _to_uint8(x * 255.0 * factor + offset)


def reduce_whites(img, umbral=200, fuerza=0.5):
    """
    Reducción de blancos (como el control "Blancos" de Lightroom): comprime la
    luminancia por encima de `umbral`. fuerza=0 no cambia nada, fuerza=1 aplana
    todo lo que pasa el umbral. R, G y B se escalan por el mismo factor para no
    correr el color. Las zonas quemadas (255) siguen sin ruido: esto no recupera
    información, solo baja su peso.
    """
    x = img.astype(np.float32)
    y = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)  # luminancia
    y_new = np.where(y > umbral, umbral + (y - umbral) * (1.0 - fuerza), y)
    gain = y_new / np.maximum(y, 1e-6)
    return _to_uint8(x * gain[..., None])


def clahe(img, clip=2.0, grilla=8):
    """
    Ecualización adaptativa de histograma (CLAHE) sobre la L de Lab, así no toca
    el color. Sube el contraste local: en zonas planas u oscuras el ruido queda
    más marcado, y con él la diferencia entre la región pegada y el resto.
    clip limita cuánto se amplifica; grilla es la cantidad de bloques por lado.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grilla, grilla)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def sharpen(img, sigma=1.0, cantidad=0.5):
    """
    Realce de altas frecuencias (unsharp mask): x + cantidad * (x - blur(x)).
    Es lo opuesto al suavizado: amplifica el residuo de ruido que mira
    Noiseprint++. También amplifica los bloques de JPEG, así que en imágenes
    muy comprimidas puede generar falsos positivos.
    """
    x = img.astype(np.float32)
    return _to_uint8(x + cantidad * (x - cv2.GaussianBlur(x, (0, 0), sigma)))


PASOS = {
    "resize": resize,
    "suavizado": denoise,
    "brillo": brightness,
    "blancos": reduce_whites,
    "clahe": clahe,
    "realce": sharpen,
}


def active_steps(pasos):
    """Devuelve [(nombre, parámetros)] de los pasos activos, validando los nombres."""
    activos = [(nombre, params) for nombre, activo, params in pasos if activo]
    for nombre, _ in activos:
        if nombre not in PASOS:
            sys.exit(f"Paso de preprocesamiento desconocido: '{nombre}'. Opciones: {', '.join(PASOS)}")
    return activos


def describe_steps(pasos):
    """Texto legible de los pasos activos, ej.: 'clahe(clip=2.0, grilla=8) -> realce(...)'."""
    return " -> ".join(
        f"{nombre}({', '.join(f'{k}={v}' for k, v in params.items())})" for nombre, params in active_steps(pasos)
    ) or "ninguno"


def preprocess(path, pasos=PREPROC):
    """
    Abre la imagen como RGB uint8, le aplica los pasos activos de `pasos`
    (ver PREPROC) y la pasa a float en [0, 1) como espera TruFor.
    """
    arr = np.array(Image.open(path).convert("RGB"))
    for nombre, params in active_steps(pasos):
        arr = PASOS[nombre](arr, **params)

    t = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float) / 256.0
    return t.unsqueeze(0), arr  # (1,3,H,W) para la red, y la entrada ya preprocesada para visualizar


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


def run_model(model, rgb, device, name=""):
    """analyze() con reintento en CPU si la imagen no entra en la GPU."""
    try:
        return analyze(model, rgb, device)
    except torch.cuda.OutOfMemoryError:
        # Imágenes grandes no entran en 8 GB. Reintentamos en CPU (lento pero
        # exacto) antes que redimensionar, que arruinaría la detección.
        print(f"  (sin memoria GPU en {name}, reintento en CPU)")
        torch.cuda.empty_cache()
        model.to("cpu")
        try:
            return analyze(model, rgb, torch.device("cpu"))
        finally:
            model.to(device)


def save_viz(out_png, original, heatmap, conf_map, score, verdict):
    """Guarda una figura: entrada al modelo | heatmap | overlay (+ confianza)."""
    import matplotlib
    matplotlib.use("Agg")  # sin ventana, solo archivo
    import matplotlib.pyplot as plt

    n = 3 if conf_map is None else 4
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4))

    axes[0].imshow(original)
    axes[0].set_title("entrada al modelo")

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


def run_experiments(directorio, configs, out=None, threshold=0.5, device="auto"):
    """
    Corre todas las imágenes de `directorio` (recursivo) con cada configuración
    de `configs` = {nombre: lista de pasos con el mismo formato que PREPROC}.
    Imprime una tabla de scores (imagen x configuración) y, si se pasa `out`,
    la guarda en CSV con una fila por imagen y configuración. Devuelve esas filas.

        run_experiments("pruebas/", {
            "sin_preproc": [],
            "clahe":       [("clahe", True, {"clip": 2.0, "grilla": 8})],
            "bilateral+realce": [
                ("suavizado", True, {"metodo": "bilateral", "d": 5, "sigma_color": 25, "sigma_space": 5}),
                ("realce",    True, {"sigma": 1.0, "cantidad": 0.5}),
            ],
        }, out="runs/experimento.csv")
    """
    images = collect_images(directorio)
    if not images:
        sys.exit(f"No encontré imágenes en: {directorio}")
    descripciones = {nombre: describe_steps(pasos) for nombre, pasos in configs.items()}  # valida antes de cargar el modelo

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    model, _ = load_model(device)
    print(f"{len(images)} imagen(es) x {len(configs)} configuración(es) | dispositivo: {device} | umbral: {threshold}")
    for nombre, desc in descripciones.items():
        print(f"  {nombre}: {desc}")
    print(f"\n{'imagen':<30}" + "".join(f"{nombre[:14]:>15}" for nombre in configs))

    rows = []
    for path in images:
        scores = []
        for nombre, pasos in configs.items():
            rgb, _ = preprocess(path, pasos)
            score, heatmap, _ = run_model(model, rgb, device, path.name)
            scores.append(score)
            rows.append({
                "config": nombre,
                "file": str(path),
                "score": score,
                "verdict": "MANIPULADA" if score >= threshold else "AUTENTICA",
                "area_sospechosa": float((heatmap >= 0.5).mean()),
                "pasos": descripciones[nombre],
            })
        print(f"{path.name[:29]:<30}" + "".join(f"{s:>15.4f}" for s in scores), flush=True)

    # Qué fracción de las imágenes declara MANIPULADA cada configuración
    print(f"{'% MANIPULADA':<30}" + "".join(
        f"{100 * np.mean([r['verdict'] == 'MANIPULADA' for r in rows if r['config'] == nombre]):>14.1f}%"
        for nombre in configs
    ))

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResultados en {out}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Detecta manipulación en imágenes con TruFor")
    ap.add_argument("input", help="imagen, carpeta o glob")
    ap.add_argument("--out", default="runs/deteccion", help="carpeta de salida para las visualizaciones")
    ap.add_argument("--viz", action="store_true", help="guardar PNG con heatmap por imagen")
    ap.add_argument("--json", default=None, help="guardar todos los scores en este .json")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="score por encima del cual se declara MANIPULADA")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    images = collect_images(args.input)
    if not images:
        sys.exit(f"No encontré imágenes en: {args.input}")
    preproc = describe_steps(PREPROC)  # valida la config antes de cargar el modelo

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    model, weights = load_model(device)
    print(f"Modelo: TruFor | pesos: {weights.name} | dispositivo: {device}")
    print(f"Preprocesamiento: {preproc}")
    print(f"{len(images)} imagen(es) | umbral: {args.threshold}\n")

    results = []
    for path in images:
        rgb, original = preprocess(path)
        score, heatmap, conf_map = run_model(model, rgb, device, path.name)

        verdict = "MANIPULADA" if score >= args.threshold else "AUTENTICA"
        print(f"  {score:.4f}  {verdict:<10}  {path}")

        results.append({
            "file": str(path),
            "score": score,
            "verdict": verdict,
            "size": list(original.shape[:2]),
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
