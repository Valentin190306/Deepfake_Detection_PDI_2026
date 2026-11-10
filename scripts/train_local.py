"""Pipeline completo de entrenamiento en local (GPU propia) sobre InpaintCOCO.

Es la versión .py de notebooks/train_colab.ipynb, con el dataset
phiyodr/inpaintCOCO (Hugging Face) en lugar de IMD2020. Pasos, en orden:

  prepare   baja InpaintCOCO y arma <data>/{images,masks,manifest.csv}
  extract   TruFor una vez por imagen -> cache de mapas (src/extract_maps.py)
  pack      cache -> memmap .npy (src/pack_cache.py)
  smoke     prueba rápida de 3 épocas y 20 pasos (sólo con --smoke)
  train     entrenamiento completo del ViT (src/train.py)
  curves    curvas de log.csv -> curvas.png
  diagnose  ¿el ViT aporta algo propio o sólo copia el score de TruFor?
  eval      métricas, ROC/PR y galería sobre la validación (src/eval.py)

Todo es reanudable: prepare reutiliza el manifest si ya existe, extract saltea
las imágenes ya cacheadas y train retoma desde last.pt. Si se corta, se vuelve
a lanzar el mismo comando.

    python scripts/train_local.py                        # todos los pasos
    python scripts/train_local.py --smoke                # con la prueba rápida antes de entrenar
    python scripts/train_local.py --steps train eval     # sólo algunos pasos
    python scripts/train_local.py --variant log05        # con el filtro ganador antes de TruFor
    python scripts/train_local.py --steps train --fresh --seed 1   # flags extra -> train.py
    python scripts/train_local.py --predict mis_fotos/   # probar el modelo ya entrenado

Los argumentos que este script no conoce se le pasan tal cual a train.py
(ej. --fresh, --seed, --resplit, --aux-weight, --no-score).

Sobre InpaintCOCO
-----------------
1260 pares en un único split ("test"): una foto de COCO 2017 val y su versión
con un cambio puntual hecho por inpainting con difusión, más la máscara de la
zona editada. Las imágenes vienen a 512x512. Cada par es una escena: el grupo
es el id de COCO, así una original y su editada nunca quedan una en train y la
otra en validación. El conjunto queda 1:1: si una foto de COCO tiene varias
editadas, la original se repite una vez por cada una (igual que
`--pairs balanced` con IMD2020; train.py y eval.py deduplican la validación).

A diferencia de IMD2020, InpaintCOCO no está entre los datos con los que se
entrenó TruFor, así que las métricas no están infladas por ese lado.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DATASET = "phiyodr/inpaintCOCO"
STEPS = ("prepare", "extract", "pack", "smoke", "train", "curves", "diagnose", "eval")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def run(*args):
    """Corre un script del repo con el mismo intérprete; corta si falla."""
    cmd = [sys.executable, *map(str, args)]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def binary_mask(mask, original, edited):
    """Máscara binaria uint8 (255 = zona editada).

    La tarjeta del dataset no dice qué color marca la zona, así que se elige la
    polaridad en la que la diferencia entre la original y la editada es mayor
    adentro de la máscara que afuera. Devuelve (máscara, si hubo que invertirla).
    """
    m = np.array(mask.convert("L").resize(edited.size, Image.NEAREST)) > 127
    flipped = False
    if original.size == edited.size and m.any() and (~m).any():
        diff = np.abs(np.asarray(original, np.int16) - np.asarray(edited, np.int16)).mean(-1)
        flipped = diff[~m].mean() > diff[m].mean()
        if flipped:
            m = ~m
    return (m * 255).astype(np.uint8), flipped


def prepare(data_dir, splits_path, overwrite):
    manifest = data_dir / "manifest.csv"
    if manifest.exists() and not overwrite:
        print(f"[prepare] ya existe {manifest}; se reutiliza (--overwrite-data para rehacerlo)")
        return
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Falta la librería 'datasets': uv pip install -r requirements.txt")

    print(f"[prepare] bajando {DATASET} (~1 GB la primera vez, después queda en el cache de HF)")
    ds = load_dataset(DATASET)["test"]
    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "masks").mkdir(parents=True, exist_ok=True)

    rows, written, inverted = [], set(), 0
    for i, ex in enumerate(ds):
        coco_id = int(ex["coco_details"]["id"])
        group = f"coco_{coco_id}"
        auth_ref = f"images/ic_coco_{coco_id:012d}.png"
        fake_ref = f"images/ic_inp_{i:04d}.png"
        mask_ref = f"masks/ic_inp_{i:04d}.png"

        original = ex["coco_image"].convert("RGB")
        edited = ex["inpaint_image"].convert("RGB")
        mask, flipped = binary_mask(ex["mask"], original, edited)
        inverted += flipped

        # PNG: sin pérdida, así no se agrega una compresión que el dataset no trae.
        if auth_ref not in written:
            original.save(data_dir / auth_ref)
            written.add(auth_ref)
        edited.save(data_dir / fake_ref)
        Image.fromarray(mask).save(data_dir / mask_ref)

        common = {"group": group, "concept": ex["concept"]}
        rows.append({"file": fake_ref, "mask": mask_ref, "label": 1, "source": "tampered",
                     "width": edited.size[0], "height": edited.size[1], **common})
        rows.append({"file": auth_ref, "mask": "", "label": 0, "source": "authentic",
                     "width": original.size[0], "height": original.size[1], **common})
        if (i + 1) % 200 == 0 or i + 1 == len(ds):
            print(f"  {i + 1}/{len(ds)}")

    fields = ["file", "mask", "label", "group", "source", "width", "height", "concept"]
    tmp = manifest.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(manifest)   # atómico: un manifest a medias no queda como válido

    # Un manifest nuevo puede traer otros grupos: el split viejo ya no sirve.
    if splits_path.exists():
        splits_path.unlink()
        print(f"[prepare] se borró {splits_path} (se regenera al entrenar)")
    print(f"[prepare] {manifest}: {len(rows)} filas | {len(ds)} editadas | "
          f"{len(written)} originales únicas | {len({r['group'] for r in rows})} escenas")
    if inverted:
        print(f"[prepare] {inverted} máscaras venían invertidas y se corrigieron")


def plot_curves(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (run_dir / "log.csv").open(newline="") as f:
        log = list(csv.DictReader(f))
    col = lambda name: [float(r[name]) for r in log]
    epoch = col("epoch")

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for c in ("acc", "bal_acc", "f1"):
        ax[0].plot(epoch, col(c), label=c)
    ax[0].set_title("clasificación")
    for c in ("auc", "ap"):
        ax[1].plot(epoch, col(c), label=c)
    ax[1].set_title("ranking")
    ax[2].plot(epoch, col("train_loss"), label="train")
    ax[2].plot(epoch, col("val_loss"), label="val")
    ax[2].set_title("loss")
    for a in ax:
        a.legend()
        a.grid(alpha=.3)
    fig.tight_layout()
    out = run_dir / "curvas.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[curves] {out}")


def diagnose(manifest, splits_path, packed, ckpt_path):
    """Compara el ranking de la rama propia del ViT (sin el prior) con el del
    score de TruFor. AUC del ViT solo ~0.5 y spearman ~-1 = sólo calibra."""
    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from dataset import MapDataset, apply_splits, dedupe_rows, read_manifest
    from model import build_model

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = build_model(**cfg).eval()
    model.load_state_dict(ck["model"])
    _, val = apply_splits(read_manifest(manifest), json.loads(splits_path.read_text()))
    ds = MapDataset(packed=packed, rows=dedupe_rows(val),
                    channels="maps+rgb" if cfg["in_chans"] == 5 else "maps")

    alpha = model.alpha.item() if model.residual else 0.0
    vit, prior, ys = [], [], []
    with torch.no_grad():
        for x, y, s in DataLoader(ds, batch_size=32):
            sc = s.clamp(1e-4, 1 - 1e-4)
            p = torch.log(sc / (1 - sc))
            vit.append((model(x, s) - alpha * p).numpy())
            prior.append(p.numpy())
            ys.append(y.numpy())
    vit, prior, ys = map(np.concatenate, (vit, prior, ys))
    print(f"[diagnose] alpha {alpha:.3f} | std(vit) {vit.std():.3f} vs std(prior) {prior.std():.3f}")
    print(f"[diagnose] AUC vit solo {roc_auc_score(ys, vit):.3f} | AUC prior (TruFor) "
          f"{roc_auc_score(ys, prior):.3f} | AUC final {roc_auc_score(ys, vit + alpha * prior):.3f}")
    print(f"[diagnose] spearman(vit, prior) {spearmanr(vit, prior).statistic:.3f}")


def predict(folder, cache_dir, ckpt_path, out_dir):
    """TruFor + ViT sobre cada imagen de `folder`, con la misma configuración
    con la que se generó el cache (así la entrada es idéntica a la del
    entrenamiento). Imprime una tabla y guarda una figura por imagen."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from dataset import CHANNELS, MEAN, STD
    from detect import analyze, load_model
    from maps import normalize_maps, resize_rgb, to_uint8
    from model import build_model
    from preprocess import apply_variant
    from preprocess import max_side as limit_side

    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)
    if not paths:
        raise SystemExit(f"No hay imágenes en {folder}")
    if not ckpt_path.exists():
        raise SystemExit(f"No existe {ckpt_path}: entrená primero")

    meta = json.loads((cache_dir / "_meta.json").read_text())
    size = int(meta["map_size"])
    max_side_used = int(meta.get("max_side") or 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(**ck["cfg"]).to(device).eval()
    model.load_state_dict(ck["model"])
    c0, c1 = CHANNELS["maps+rgb" if ck["cfg"]["in_chans"] == 5 else "maps"]
    thr = float(ck.get("thr", 0.5))
    trufor, _ = load_model(device)
    print(f"[predict] época {ck.get('epoch')} | variante {meta['variant']} | {size}px | umbral {thr:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for path in paths:
        arr = np.array(Image.open(path).convert("RGB"))
        if max_side_used:
            arr = limit_side(arr, max_side_used)
        arr = apply_variant(arr, meta["variant"])
        t = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float) / 256.0
        score, heat, conf = analyze(trufor, t.unsqueeze(0), device)

        m = normalize_maps(heat, conf, size, meta["heat_mode"], meta["conf_mode"])
        # Se pasa por uint8 igual que el cache: la entrada queda byte a byte igual.
        u8 = np.concatenate([resize_rgb(arr, size).transpose(2, 0, 1),
                             to_uint8(m[0])[None], to_uint8(m[1])[None]], 0)
        x = torch.from_numpy(u8).float() / 255.0
        x = torch.cat([(x[:3] - MEAN) / STD, x[3:]], 0)[c0:c1].unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(model(x.to(device), torch.tensor([score], device=device))).item()
        verdict = "MANIPULADA" if prob >= thr else "auténtica"
        results.append((path.name, score, prob, verdict))

        rgb = u8[:3].transpose(1, 2, 0)
        fig, ax = plt.subplots(1, 4, figsize=(17, 4.5))
        ax[0].imshow(rgb)
        ax[0].set_title(path.name[:40], fontsize=9)
        ax[1].imshow(m[0], cmap="RdBu_r", vmin=0, vmax=1)
        ax[1].set_title("mapa de manipulación")
        ax[2].imshow(m[1], cmap="viridis", vmin=0, vmax=1)
        ax[2].set_title("confianza")
        ax[3].imshow(rgb)
        ax[3].imshow(m[0], cmap="RdBu_r", vmin=0, vmax=1, alpha=0.45)
        ax[3].set_title(f"overlay | ViT {prob:.3f} ({verdict})")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{path.stem}.png", dpi=100)
        plt.close(fig)

    print(f"\n{'archivo':<30}{'TruFor':>8}{'ViT':>8}   veredicto")
    for name, score, prob, verdict in sorted(results, key=lambda r: -r[2]):
        print(f"{name[:30]:<30}{score:>8.3f}{prob:>8.3f}   {verdict}")
    print(f"\n[predict] figuras en {out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Entrenamiento local TruFor + ViT sobre InpaintCOCO (versión .py del notebook).",
        epilog="Los argumentos desconocidos se le pasan a src/train.py.")
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS),
                    help="pasos a correr, en el orden del pipeline (default: todos)")
    ap.add_argument("--smoke", action="store_true", help="correr la prueba rápida antes de entrenar")
    ap.add_argument("--predict", type=Path, default=None,
                    help="carpeta con imágenes propias: corre TruFor + ViT sobre ellas y termina")
    ap.add_argument("--data", type=Path, default=Path("data/inpaintcoco"))
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--packed-dir", type=Path, default=Path("packed"))
    ap.add_argument("--runs", type=Path, default=Path("runs/inpaintcoco"))
    ap.add_argument("--overwrite-data", action="store_true", help="rehacer el manifest aunque exista")
    ap.add_argument("--variant", default="raw", help="variante de src/preprocess.py (ej. log05)")
    ap.add_argument("--map-size", type=int, default=384, help="lado S de entrada del ViT")
    ap.add_argument("--max-side", type=int, default=1000,
                    help="lado máximo antes de TruFor (memoria de la GPU; InpaintCOCO viene a 512)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=500, help="remuestreos para el IC 95%% de eval")
    a, train_extra = ap.parse_known_args()

    from extract_maps import cache_key
    resolve = lambda p: p if p.is_absolute() else REPO / p
    data, runs = resolve(a.data), resolve(a.runs)
    manifest = data / "manifest.csv"
    splits = runs / "splits.json"
    key = cache_key(a.variant, a.map_size, a.max_side)
    cache_dir = resolve(a.cache) / key
    packed = resolve(a.packed_dir) / f"inpaintcoco__{a.variant}__s{a.map_size}"
    vit_dir = runs / f"vit__{a.variant}"
    if train_extra:
        print(f"flags extra para train.py: {' '.join(train_extra)}")

    if a.predict is not None:
        predict(a.predict, cache_dir, vit_dir / "best.pt", runs / "prediccion")
        return

    import torch
    if not torch.cuda.is_available():
        print("AVISO: no hay GPU visible para torch; todo va a correr en CPU (muy lento).")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if not any((REPO / "third_party" / "trufor").rglob("trufor.pth.tar")):
        subprocess.run(["bash", "scripts/get_trufor.sh"], cwd=REPO, check=True)

    steps = [s for s in STEPS if s in a.steps and (s != "smoke" or a.smoke)]
    common = ["--manifest", manifest, "--packed", packed, "--splits", splits]
    for step in steps:
        print(f"\n===== {step} =====", flush=True)
        if step == "prepare":
            prepare(data, splits, a.overwrite_data)
        elif step == "extract":
            run("src/extract_maps.py", "--manifest", manifest, "--cache", resolve(a.cache),
                "--variants", a.variant, "--map-size", a.map_size, "--max-side", a.max_side)
        elif step == "pack":
            packed.parent.mkdir(parents=True, exist_ok=True)
            run("src/pack_cache.py", "--cache", cache_dir, "--out", packed, "--overwrite")
        elif step == "smoke":
            run("src/train.py", *common, "--out", runs / "smoke", "--size", a.map_size,
                "--epochs", 3, "--limit-steps", 20, "--patience", 0, "--fresh")
        elif step == "train":
            run("src/train.py", *common, "--out", vit_dir, "--size", a.map_size,
                "--epochs", a.epochs, "--bs", a.bs, "--workers", a.workers, *train_extra)
        elif step == "curves":
            plot_curves(vit_dir)
        elif step == "diagnose":
            diagnose(manifest, splits, packed, vit_dir / "best.pt")
        elif step == "eval":
            run("src/eval.py", *common, "--ckpt", vit_dir / "best.pt", "--out", runs / f"eval__{a.variant}",
                "--split", "val", "--gallery", 6, "--bootstrap", a.bootstrap, "--workers", a.workers)

    print(f"\nListo. Modelo en {vit_dir}; para probarlo con tus fotos:\n"
          f"  python scripts/train_local.py --variant {a.variant} --predict <carpeta>")


if __name__ == "__main__":
    main()
