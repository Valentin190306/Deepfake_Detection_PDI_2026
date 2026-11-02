"""Prepara IMD2020 (versión de Kaggle: saikotahmed/imd2020) en el formato del proyecto.

Salida
------
    <out>/images/<prefijo><stem><ext>          manipuladas (bytes originales)
    <out>/images/<prefijo><stem>__auth<ext>    auténticas
    <out>/masks/<prefijo><stem>.png            máscara binaria (255 = manipulado)
    <out>/manifest.csv                         file, mask, label, group, source, width, height

Layout esperado (se busca en cualquier nivel de --src):
    img/ (manipuladas) + mask/ (máscaras) + Authentic/ (auténticas, sin máscara)

Dos claves distintas
--------------------
Emparejar archivos y agrupar escenas son problemas distintos y mezclarlos
rompe cualquiera de los dos:

  pair_key   empareja imagen <-> máscara. Sólo saca sufijos de máscara, así
             cada archivo conserva su identidad:
                 <escena>_fake_01.jpg  <->  <escena>_fake_01_mask.png
  group_key  identifica la ESCENA. Saca el tipo de versión (_orig, _fake, ...)
             y el índice de variante (_01, _02, ...):
                 <escena>_orig.jpg, <escena>_fake.jpg, <escena>_fake_01.jpg
                 -> todas al grupo <escena>

Grupos y fuga de información
----------------------------
IMD2020 empareja cada imagen manipulada con su original. Si el original cae en
train y la manipulada en validación, la métrica es mentira. Por eso el split se
hace por `group`, no por archivo:

  1. Si el dataset conserva una carpeta por escena, el grupo es esa carpeta.
  2. Si no, el grupo sale de `group_key()` (el nombre sin sufijos ni índice).
  3. `--dedupe-hash` es la red de seguridad para datasets cuyos nombres no
     codifican la escena: calcula un dHash por imagen y fusiona los grupos con
     imágenes casi idénticas. En `auto` sólo se activa si el nombre no agrupó
     nada (grupos == archivos); si el nombre ya agrupa, el hash sobre-fusiona.

El script es conservador: diagnostica e informa (pares sin máscara, tamaños que
no coinciden, máscaras posiblemente invertidas, grupos fusionados) en vez de
adivinar en silencio.
"""
import argparse
import csv
import re
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Sufijos de máscara: <escena>_fake_01_mask.png -> <escena>_fake_01
MASK_SUFFIX = re.compile(r"[_\-.]?(mask|masks|gt|groundtruth|label)$", re.I)
# Sufijos de versión: <escena>_orig / <escena>_fake / <escena>_edited -> <escena>
GROUP_SUFFIX = re.compile(
    r"[_\-.]?(mask|masks|gt|groundtruth|label|orig|original|real|auth|authentic|"
    r"fake|fakes|edited|edit|manip|manipulated|tampered|forged|inpaint|inpainting|"
    r"synth|synthetic)$", re.I)
# Índice de variante: <escena>_fake_01 -> <escena>_fake
VARIANT_INDEX = re.compile(r"[_\-.]\d+$")


def _strip(stem: str, pattern: re.Pattern) -> str:
    prev = None
    while prev != stem:
        prev, stem = stem, pattern.sub("", stem)
    return stem


def pair_key(stem: str) -> str:
    """Clave única por archivo, para emparejar imagen y máscara."""
    return _strip(stem, MASK_SUFFIX).lower()


def group_key(stem: str) -> str:
    """Escena a la que pertenece el archivo (base del split, sin fuga de datos)."""
    prev = None
    while prev != stem:
        prev = stem
        stem = _strip(stem, VARIANT_INDEX)
        stem = _strip(stem, GROUP_SUFFIX)
    return stem.lower()


def safe(stem: str) -> str:
    """Nombre apto para archivo (los grupos por carpeta pueden traer separadores)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "img"


def files(d: Path | None) -> list[Path]:
    if d is None:
        return []
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in EXT)


def find_dir(root: Path, names: set[str]) -> Path | None:
    cands = [d for d in [root, *root.rglob("*")] if d.is_dir() and d.name.lower() in names]
    return max(cands, key=lambda d: len(files(d)), default=None)


def index(paths: list[Path]) -> tuple[dict[str, Path], int]:
    idx, dup = {}, 0
    for p in paths:
        k = pair_key(p.stem)
        dup += k in idx
        idx.setdefault(k, p)
    return idx, dup


def dhash(path: Path, size: int = 8) -> int:
    """dHash de 64 bits: robusto a recompresión y a ediciones locales chicas."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def merge_by_hash(paths: list[Path], groups: list[str], threshold: int, verbose=True):
    """Fusiona grupos cuyas imágenes son casi idénticas (distancia de Hamming <= threshold)."""
    if verbose:
        print(f"  calculando dHash de {len(paths)} imágenes...")
    hashes = np.array([dhash(p) for p in paths], dtype=np.uint64)
    bits = np.unpackbits(hashes.view(np.uint8).reshape(len(paths), 8), axis=1).astype(np.int16)
    # Hamming(a,b) = #bits donde difieren
    d = bits @ (1 - bits).T + (1 - bits) @ bits.T
    np.fill_diagonal(d, 64)

    uf = UnionFind(len(paths))
    ii, jj = np.where(np.triu(d <= threshold, 1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        uf.union(i, j)

    # un grupo base puede tener varios archivos: se fusiona con el representante
    rep = {}
    for i, g in enumerate(groups):
        r = uf.find(i)
        rep.setdefault(r, g)
    merged = [rep[uf.find(i)] for i in range(len(paths))]

    nn = d.min(1)
    if verbose:
        hist = Counter(int(x // 4 * 4) for x in nn)
        print("  distancia al vecino más cercano (bits): "
              + ", ".join(f"{k}-{k + 3}:{v}" for k, v in sorted(hist.items())))
        sizes = Counter(merged)
        print(f"  grupos: {len(set(groups))} -> {len(set(merged))} "
              f"(mayor: {max(sizes.values())} archivos)")
        if max(sizes.values()) > 20:
            print("  AVISO: un grupo enorme sugiere umbral demasiado alto; bajá --dedupe-threshold.")
    return merged


def main():
    ap = argparse.ArgumentParser(description="Prepara IMD2020 para el clasificador")
    ap.add_argument("--src", required=True, help="carpeta del dataset descargado")
    ap.add_argument("--out", default="data")
    ap.add_argument("--prefix", default="imd_")
    ap.add_argument("--invert", action="store_true", help="si las máscaras vienen con 255 = auténtico")
    ap.add_argument("--dedupe-hash", choices=["auto", "on", "off"], default="auto",
                    help="fusionar grupos con imágenes casi idénticas (auto: solo si el nombre no agrupa)")
    ap.add_argument("--dedupe-threshold", type=int, default=8, help="distancia Hamming máxima (0-64)")
    ap.add_argument("--no-copy", action="store_true",
                    help="no copiar imágenes; el manifest apunta a los archivos originales")
    ap.add_argument("--dry-run", action="store_true", help="solo diagnosticar, no escribir nada")
    a = ap.parse_args()

    root, out = Path(a.src), Path(a.out)
    d_img = find_dir(root, {"img", "images", "image", "tampered"})
    d_mask = find_dir(root, {"mask", "masks"})
    d_auth = find_dir(root, {"authentic", "real", "authentic_images", "pristine"})
    if not (d_img and d_mask):
        sys.exit(f"No encontré carpetas img/ y mask/ dentro de {root}. Contenido:\n"
                 + "\n".join(str(p) for p in sorted(root.glob('*'))[:30]))
    print(f"img:        {d_img}\nmask:       {d_mask}\nauthentic:  {d_auth}")

    imgs, d1 = index(files(d_img))
    masks, d2 = index(files(d_mask))
    auths, d3 = index(files(d_auth))
    print(f"archivos -> img: {len(imgs)} | mask: {len(masks)} | authentic: {len(auths)}")
    print("ejemplos img:", [p.name for p in list(imgs.values())[:3]])
    print("ejemplos mask:", [p.name for p in list(masks.values())[:3]])
    print("ejemplos authentic:", [p.name for p in list(auths.values())[:3]])
    if d1 + d2 + d3:
        print(f"AVISO: {d1 + d2 + d3} nombres canónicos repetidos (se conserva uno por clave).")

    common = sorted(set(imgs) & set(masks))
    print(f"pares img-mask: {len(common)} de {len(imgs)} imágenes")
    if len(common) < 0.5 * len(imgs):
        sys.exit("Menos de la mitad de las imágenes tiene máscara: el emparejamiento por nombre "
                 "no funciona. Revisá los ejemplos de arriba y ajustá MASK_SUFFIX.")

    # ---- grupo base -------------------------------------------------------
    def folder_group(p: Path, base: Path) -> str:
        """Carpeta de escena si existe; si el archivo está suelto, su nombre canónico."""
        rel = str(p.parent.relative_to(base))
        return group_key(p.stem) if rel == "." else rel

    flat = all(p.parent == d_img for p in imgs.values())
    if flat:
        groups = {k: group_key(imgs[k].stem) for k in common}
        groups.update({f"auth:{k}": group_key(auths[k].stem) for k in auths})
    else:  # una carpeta por escena (layout oficial de IMD2020)
        groups = {k: folder_group(imgs[k], d_img) for k in common}
        groups.update({f"auth:{k}": folder_group(auths[k], d_auth) for k in auths})
    print(f"layout {'plano' if flat else 'por carpetas'} | grupos base: {len(set(groups.values()))}")

    tamp_groups = {groups[k] for k in common}
    auth_groups = [groups[f"auth:{k}"] for k in auths]
    matched = sum(g in tamp_groups for g in auth_groups)
    print(f"auténticas emparejadas por nombre con una manipulada: {matched} de {len(auths)}")
    sizes = Counter(groups.values())
    if sizes:
        print(f"tamaño de grupo: mayor {max(sizes.values())}, "
              f"con más de 1 archivo: {sum(1 for v in sizes.values() if v > 1)}")

    all_paths = [imgs[k] for k in common] + [auths[k] for k in auths]
    all_keys = list(common) + [f"auth:{k}" for k in auths]
    all_groups = [groups[k] for k in all_keys]

    # `auto`: el hash es la red de seguridad para cuando el nombre no vincula
    # las auténticas con sus manipuladas. Si el nombre ya las agrupa, fusionar
    # por similitud sobre-fusiona escenas distintas.
    no_name_grouping = len(set(all_groups)) == len(all_groups)
    no_auth_links = bool(auths) and matched == 0
    use_hash = a.dedupe_hash == "on" or (
        a.dedupe_hash == "auto" and (no_name_grouping or no_auth_links))
    if a.dedupe_hash == "auto":
        print("dedupe por hash: " + (
            "activado (el nombre no vincula auténticas con manipuladas)" if use_hash
            else "innecesario (el nombre ya agrupa las escenas)"))
    if use_hash and all_paths:
        print("Fusionando grupos por similitud visual (dHash):")
        all_groups = merge_by_hash(all_paths, all_groups, a.dedupe_threshold)

    # ---- copiar y escribir manifest --------------------------------------
    if not a.dry_run:
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "masks").mkdir(parents=True, exist_ok=True)

    rows, used = [], set()
    fracs, bad_size = [], []

    def dst_name(stem: str, ext: str, auth: bool) -> str:
        # El stem tiene que ser único: también se usa como nombre de máscara.
        base = f"{a.prefix}{stem}{'__auth' if auth else ''}"
        name, n = base, 1
        while name.lower() in used:
            n += 1
            name = f"{base}_{n}"
        used.add(name.lower())
        return f"{name}{ext}"

    for idx, k in enumerate(all_keys):
        auth = k.startswith("auth:")
        src = all_paths[idx]
        with Image.open(src) as im:
            w, h = im.size

        # Manipuladas: pair key (única por archivo). Auténticas: la escena, así
        # <escena>_orig.jpg queda como imd_<escena>__auth.jpg.
        stem = all_groups[idx] if auth else k.removeprefix("auth:")
        name = dst_name(safe(stem), src.suffix.lower(), auth)
        if not a.no_copy:
            dst = out / "images" / name
            if not dst.exists() and not a.dry_run:
                shutil.copy2(src, dst)          # bytes originales, sin recomprimir
            img_ref = f"images/{name}"
        else:
            img_ref = str(src.resolve())

        mask_ref = ""
        if not auth:
            mp = masks[k]
            with Image.open(mp) as m:
                if m.size != (w, h):
                    bad_size.append((src.name, (w, h), m.size))
                    continue
                arr = np.array(m.convert("L")) > 127
            if a.invert:
                arr = ~arr
            fracs.append(float(arr.mean()))
            if not a.no_copy:
                mask_name = f"{Path(name).stem}.png"
                if not a.dry_run:
                    Image.fromarray((arr * 255).astype(np.uint8)).save(out / "masks" / mask_name)
                mask_ref = f"masks/{mask_name}"
            else:
                mask_ref = str(mp.resolve())

        rows.append({
            "file": img_ref, "mask": mask_ref, "label": 0 if auth else 1,
            "group": all_groups[idx], "source": "authentic" if auth else "tampered",
            "width": w, "height": h,
        })

    med = statistics.median(fracs) if fracs else float("nan")
    n_auth = sum(r["label"] == 0 for r in rows)
    n_tamp = sum(r["label"] == 1 for r in rows)
    print(f"\nManipuladas: {n_tamp} | auténticas: {n_auth} | "
          f"grupos: {len(set(r['group'] for r in rows))}")
    print(f"área manipulada (mediana): {med:.1%}")
    if med > 0.5:
        print("AVISO: la mediana es > 50%: las máscaras parecen INVERTIDAS. Repetí con --invert.")
    if bad_size:
        print(f"{len(bad_size)} pares omitidos por tamaño distinto imagen/máscara, ej.: {bad_size[:3]}")
    if n_auth and n_auth / max(n_tamp, 1) < 0.4:
        print("AVISO: hay muchas menos auténticas que manipuladas (desbalance). "
              "train.py lo compensa con pos_weight, pero conviene reportar balanced accuracy y AUC.")
    if flat and not use_hash and matched < len(auths):
        print(f"AVISO: {len(auths) - matched} auténticas no comparten grupo con ninguna "
              "manipulada. Si alguna es el original de una manipulada con otro nombre, hay "
              "fuga entre train y validación: probá --dedupe-hash on.")

    if not rows:
        sys.exit("No quedó ninguna imagen válida: revisá los avisos de arriba.")

    if a.dry_run:
        print("\n--dry-run: no se escribió nada.")
        return

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nListo: {out}/manifest.csv ({len(rows)} filas)")
    print("Siguiente paso: python src/extract_maps.py --manifest "
          f"{out}/manifest.csv --cache cache --variants raw")


if __name__ == "__main__":
    main()
