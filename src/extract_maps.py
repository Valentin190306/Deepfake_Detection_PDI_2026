"""Paso 1: calcula y cachea la salida de TruFor (mapa, confianza, score) por imagen y variante.

Reutiliza load_model/analyze de detect.py: hay UN solo lugar que sabe como llamar a TruFor.
Es REANUDABLE (saltea lo ya cacheado) y escribe de forma atomica, asi que si Colab
se desconecta a mitad de camino no quedan archivos corruptos.

  python src/extract_maps.py --images data/images --cache cache --variants raw

Las variantes se aplican EN MEMORIA (ver preprocess.py): nunca se re-guarda la imagen,
por lo que no hay recompresion extra. Salida: cache/<variante>/<stem>.npz
con claves: map (float16), conf (float16), score (float32).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect import IMG_EXT, analyze, load_model  # noqa: E402
from preprocess import VARIANTS, apply_variant  # noqa: E402


def to_tensor(arr):
    # Misma conversion que detect.preprocess: SIN resize, uint8 -> [0, 1)
    t = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float) / 256.0
    return t.unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--variants", nargs="+", default=["raw"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()

    images = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        sys.exit(f"No hay imagenes en {a.images}")
    if len({p.stem for p in images}) != len(images):
        sys.exit("Hay imagenes con el mismo nombre y distinta extension: el cache usa el stem como clave.")
    for v in a.variants:
        if v not in VARIANTS:
            sys.exit(f"Variante desconocida: {v} (definidas: {list(VARIANTS)})")

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    model, weights = load_model(device)
    print(f"TruFor | pesos: {weights.name} | dispositivo: {device} | {len(images)} imagenes")

    failed = []
    for v in a.variants:
        out = Path(a.cache) / v
        out.mkdir(parents=True, exist_ok=True)
        todo = [p for p in images if not (out / f"{p.stem}.npz").exists()]
        print(f"[{v}] pendientes: {len(todo)} de {len(images)}")

        for i, p in enumerate(todo, 1):
            try:
                arr = apply_variant(np.array(Image.open(p).convert("RGB")), v)
                if arr.dtype != np.uint8 or arr.ndim != 3:
                    raise ValueError("el preprocesamiento debe devolver uint8 HxWx3")
                rgb = to_tensor(arr)
                try:
                    score, heat, conf = analyze(model, rgb, device)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    model.to("cpu")
                    try:
                        score, heat, conf = analyze(model, rgb, torch.device("cpu"))
                    finally:
                        model.to(device)

                data = {"map": heat.astype(np.float16), "score": np.float32(score)}
                if conf is not None:
                    data["conf"] = conf.astype(np.float16)
                tmp = out / f"{p.stem}.tmp.npz"
                np.savez_compressed(tmp, **data)
                tmp.replace(out / f"{p.stem}.npz")   # atomico
            except Exception as e:  # una imagen rota no debe frenar el lote
                failed.append((v, p.name, repr(e)))
                print(f"  ERROR {p.name}: {e!r}")
            if i % 25 == 0 or i == len(todo):
                print(f"  [{v}] {i}/{len(todo)}")

    if failed:
        print(f"\n{len(failed)} imagen(es) fallaron:")
        for v, n, e in failed:
            print(f"  [{v}] {n}: {e}")


if __name__ == "__main__":
    main()
