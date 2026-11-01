"""Prepara IMD2020 (version de Kaggle: saikotahmed/imd2020) en el formato del proyecto:

    <out>/images/<prefijo><clave>.<ext>      imagenes ORIGINALES (se copian los bytes tal cual)
    <out>/masks/<prefijo><clave>.png         mascara binaria (255 = manipulado)

Carpetas esperadas dentro del dataset descargado (en cualquier nivel):
    img/ (manipuladas) + mask/ (mascaras)  y, opcionalmente, Authentic/ (autenticas, sin mascara).
Las autenticas se guardan como <prefijo><clave>__auth: si su nombre coincide con una manipulada,
comparten grupo y dataset.split() las mantiene del mismo lado (train o val).

No conozco los nombres de archivo exactos del dataset: el script imprime ejemplos y diagnostica
(sin pares, tamanos que no coinciden, mascaras posiblemente invertidas) en vez de adivinar en silencio.
"""
import argparse
import re
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SUFFIX = re.compile(r"[_\-.]?(mask|gt|edit|edited|orig|original|real|auth|authentic|tamp|tampered|fake|forged)$", re.I)


def key(stem):
    """Nombre canonico: saca sufijos tipo _mask / _orig / _edit para poder emparejar archivos."""
    prev = None
    while prev != stem:
        prev, stem = stem, SUFFIX.sub("", stem)
    return stem.lower()


def files(d):
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in EXT) if d else []


def find_dir(root, names):
    cands = [d for d in [root, *root.rglob("*")] if d.is_dir() and d.name.lower() in names]
    return max(cands, key=lambda d: len(files(d)), default=None)


def index(paths):
    idx, dup = {}, 0
    for p in paths:
        k = key(p.stem)
        dup += k in idx
        idx[k] = p
    return idx, dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="carpeta del dataset descargado")
    ap.add_argument("--out", default="data")
    ap.add_argument("--prefix", default="imd_")
    ap.add_argument("--invert", action="store_true", help="si las mascaras vienen con 255 = autentico")
    a = ap.parse_args()

    root, out = Path(a.src), Path(a.out)
    d_img = find_dir(root, {"img", "images", "image"})
    d_mask = find_dir(root, {"mask", "masks"})
    d_auth = find_dir(root, {"authentic", "real", "authentic_images"})
    if not (d_img and d_mask):
        sys.exit(f"No encontre carpetas img/ y mask/ dentro de {root}. Contenido:\n"
                 + "\n".join(str(p) for p in sorted(root.glob('*'))[:30]))
    print(f"img: {d_img}\nmask: {d_mask}\nauthentic: {d_auth}")

    imgs, d1 = index(files(d_img))
    masks, d2 = index(files(d_mask))
    auths, d3 = index(files(d_auth))
    print(f"archivos -> img: {len(imgs)} | mask: {len(masks)} | authentic: {len(auths)}")
    print("ejemplos img:", [p.name for p in list(imgs.values())[:3]])
    print("ejemplos mask:", [p.name for p in list(masks.values())[:3]])
    print("ejemplos authentic:", [p.name for p in list(auths.values())[:3]])
    if d1 + d2 + d3:
        print(f"AVISO: {d1 + d2 + d3} nombres canonicos repetidos (se conserva uno por clave).")

    common = sorted(set(imgs) & set(masks))
    print(f"pares img-mask: {len(common)} de {len(imgs)} imagenes")
    if len(common) < 0.5 * len(imgs):
        sys.exit("Menos de la mitad de las imagenes tiene mascara: el emparejamiento por nombre no funciona. "
                 "Revisa los ejemplos de arriba y ajusta SUFFIX en este script.")

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    fracs, bad_size, n_ok = [], [], 0
    for k in common:
        ip, mp = imgs[k], masks[k]
        with Image.open(ip) as i, Image.open(mp) as m:
            if i.size != m.size:
                bad_size.append((ip.name, i.size, m.size))
                continue
            arr = np.array(m.convert("L")) > 127
        if a.invert:
            arr = ~arr
        fracs.append(float(arr.mean()))
        stem = f"{a.prefix}{k}"
        dst_i = out / "images" / f"{stem}{ip.suffix.lower()}"
        if not dst_i.exists():
            shutil.copy2(ip, dst_i)                       # bytes originales, sin recomprimir
        Image.fromarray((arr * 255).astype(np.uint8)).save(out / "masks" / f"{stem}.png")
        n_ok += 1

    n_auth = 0
    for k, p in auths.items():
        dst = out / "images" / f"{a.prefix}{k}__auth{p.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(p, dst)
        n_auth += 1

    med = statistics.median(fracs) if fracs else float("nan")
    print(f"\nListo: {n_ok} manipuladas con mascara + {n_auth} autenticas en {out}/")
    print(f"area manipulada (mediana): {med:.1%}")
    if med > 0.5:
        print("AVISO: la mediana es > 50%: las mascaras parecen INVERTIDAS. Repeti con --invert.")
    if bad_size:
        print(f"{len(bad_size)} pares omitidos por tamano distinto imagen/mascara, ej.: {bad_size[:3]}")
    matched = sum(k in imgs for k in auths)
    print(f"autenticas con nombre equivalente a una manipulada (comparten grupo): {matched} de {n_auth}")


if __name__ == "__main__":
    main()
