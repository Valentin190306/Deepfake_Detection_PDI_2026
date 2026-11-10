"""Paso 2: consolidar el cache de .npz en un único .npy memmap para entrenar rápido.

Leer 2400 archivos .npz sueltos desde Google Drive (FUSE) en cada época es
lento y frágil. Este script los junta una vez en un array contiguo local:

    packed.npy   uint8 (N, 5, S, S)   canales: R, G, B, mapa, confianza
    packed.json  stems, scores, tamaño, variante

El .npy se abre con mmap_mode="r", así que no hace falta tener 2 GB de RAM.
Hay que re-empaquetar si se agregan imágenes o se cambia el cache.

    python src/pack_cache.py --cache cache/raw__s384__ms1280 --out packed/raw__s384
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="carpeta con los .npz de una clave de cache")
    ap.add_argument("--out", required=True, help="ruta base del paquete (sin extensión)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    cache = Path(a.cache)
    meta_path = cache / "_meta.json"
    if not meta_path.exists():
        sys.exit(f"No existe {meta_path}: ¿corriste extract_maps.py?")
    meta = json.loads(meta_path.read_text())

    files = sorted(p for p in cache.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    if not files:
        sys.exit(f"No hay .npz en {cache}")

    out = Path(a.out)
    npy, js = out.with_suffix(".npy"), out.with_suffix(".json")
    if npy.exists() and not a.overwrite:
        sys.exit(f"Ya existe {npy} (usá --overwrite para regenerarlo)")

    first = np.load(files[0])
    s = int(first["map"].shape[0])
    for f in files[:5]:
        d = np.load(f)
        if int(d["map"].shape[0]) != s:
            sys.exit(f"{f.name}: tamaño {d['map'].shape} != {s}; el cache está mezclado")

    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8,
                                    shape=(len(files), 5, s, s))
    stems, scores, sizes = [], [], []
    for i, f in enumerate(files):
        d = np.load(f)
        arr[i, 0:3] = np.transpose(d["rgb"], (2, 0, 1))
        arr[i, 3] = d["map"]
        arr[i, 4] = d["conf"]
        stems.append(f.stem)
        scores.append(float(d["score"]))
        sizes.append([int(x) for x in d["orig_hw"]])
        if (i + 1) % 250 == 0 or i + 1 == len(files):
            print(f"  {i + 1}/{len(files)}")
    arr.flush()
    del arr

    js.write_text(json.dumps({
        "stems": stems, "scores": scores, "orig_hw": sizes,
        "size": s, "channels": ["R", "G", "B", "map", "conf"], **meta,
    }))
    print(f"\n{npy} ({npy.stat().st_size / 1e9:.2f} GB) + {js.name} | {len(files)} imágenes")


if __name__ == "__main__":
    main()
