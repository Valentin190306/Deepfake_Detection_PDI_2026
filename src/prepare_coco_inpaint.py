"""Prepara phiyodr/InpaintCOCO (HuggingFace) en el formato del proyecto.

Qué trae el dataset
-------------------
1260 filas, cada una con:

    coco_image      la foto ORIGINAL de COCO, 512x512   -> auténtica
    inpaint_image   la misma foto con un objeto borrado  -> manipulada
    mask            la zona que se rellenó
    coco_details    metadatos, incluido `coco_url`

O sea que la clase auténtica viene incluida y pareada: no hay que bajar COCO
aparte. La manipulación es *generada por un modelo* (inpainting), no edición
manual como IMD2020; es otra familia, y eso es justamente lo que interesa para
un test que no sea el set de entrenamiento de TruFor.

Salida
------
    <out>/images/<id>_auth.png      auténticas (una por escena)
    <out>/images/<id>_fake.png      manipuladas
    <out>/masks/<id>.png            máscara binaria (255 = manipulado)
    <out>/manifest.csv              file, mask, label, group, source, width, height, dataset

Se guarda en PNG (sin pérdida) para no re-comprimir lo que ya viene comprimido:
una segunda pasada de JPEG metería artefactos que no estaban en el original.

Si la misma foto de COCO aparece en varias filas (varios inpainting de la misma
escena), la auténtica se escribe una sola vez y se repite una fila por variante,
igual que `prepare_imd2020.py --pairs balanced`: cada escena pesa lo mismo en
las dos clases y el conjunto queda 1:1.

Uso:
    python src/prepare_coco_inpaint.py --out data/coco_inpaint
    python src/prepare_coco_inpaint.py --out /tmp/prueba --limit 20   # smoke test
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

COLUMNS = ["file", "mask", "label", "group", "source", "width", "height", "dataset"]


def scene_id(row: dict, index: int) -> str:
    """Id estable de la escena: el nombre de la foto de COCO si está disponible.

    Usar el id de COCO (y no el índice de la fila) hace que dos inpainting de la
    misma foto caigan en el mismo grupo y no se separen entre train y val.
    """
    details = row.get("coco_details") or {}
    if isinstance(details, dict):
        url = details.get("coco_url")
        if isinstance(url, str) and url.strip():
            stem = Path(url.strip().split("?")[0]).stem
            if stem:
                return stem
        cid = details.get("id")                 # id numérico de COCO
        if cid is not None:
            return f"{int(cid):012d}"
    return f"{index:04d}"


def _atomic_save(img: Image.Image, path: Path):
    """Guarda en un .tmp y renombra: si Colab se corta, no deja una PNG truncada."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, format="PNG", optimize=True)
    tmp.replace(path)


def _save_png(img: Image.Image, path: Path):
    _atomic_save(img.convert("RGB"), path)


def prepare(rows, out: Path, overwrite: bool = False) -> list[dict]:
    """Escribe imágenes y devuelve las filas del manifest.

    `rows` es cualquier iterable de dicts con `coco_image`, `inpaint_image` y
    `mask` como imágenes PIL. Separado de `main` para poder testearlo sin red.
    """
    out = Path(out)
    img_dir, mask_dir = out / "images", out / "masks"
    manifest: list[dict] = []
    auth_saved: dict[str, str] = {}
    skipped = 0

    for i, row in enumerate(rows):
        auth, fake, mask = row["coco_image"], row["inpaint_image"], row["mask"]
        if not (auth.size == fake.size == mask.size):
            print(f"AVISO fila {i}: tamaños distintos "
                  f"{auth.size}, {fake.size}, {mask.size}; se omite.")
            skipped += 1
            continue
        sid = scene_id(row, i)
        w, h = auth.size

        # auténtica: se escribe una sola vez por escena, pero se emite una fila
        # por cada variante manipulada (igual que `--pairs balanced` en IMD2020):
        # así cada escena pesa lo mismo en las dos clases y el conjunto queda 1:1.
        if sid not in auth_saved:
            rel = f"images/{sid}_auth.png"
            dst = out / rel
            if overwrite or not dst.exists():
                _save_png(auth, dst)
            auth_saved[sid] = rel
        manifest.append({"file": auth_saved[sid], "mask": "", "label": 0, "group": sid,
                         "source": "inpaintcoco", "width": w, "height": h,
                         "dataset": "coco"})

        rel_fake = f"images/{sid}_fake_{i:04d}.png"
        rel_mask = f"masks/{sid}_{i:04d}.png"
        dst_fake, dst_mask = out / rel_fake, out / rel_mask
        if overwrite or not dst_fake.exists():
            _save_png(fake, dst_fake)
        if overwrite or not dst_mask.exists():
            bin_mask = (np.array(mask.convert("L")) > 127).astype(np.uint8) * 255
            _atomic_save(Image.fromarray(bin_mask), dst_mask)

        manifest.append({"file": rel_fake, "mask": rel_mask, "label": 1, "group": sid,
                         "source": "inpaintcoco", "width": w, "height": h,
                         "dataset": "coco"})

    if skipped:
        print(f"{skipped} filas omitidas por tamaño inconsistente")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Prepara phiyodr/InpaintCOCO")
    ap.add_argument("--out", default="data/coco_inpaint")
    ap.add_argument("--dataset", default="phiyodr/InpaintCOCO",
                    help="id en HuggingFace (ojo con las mayúsculas)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="procesar sólo las primeras N (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Falta `datasets`: pip install datasets  (o usá el entorno de Colab)")

    print(f"cargando {a.dataset} [{a.split}]...")
    ds = load_dataset(a.dataset, split=a.split)
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
    print(f"{len(ds)} filas")

    out = Path(a.out)
    rows = prepare(ds, out, overwrite=a.overwrite)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    n_auth = sum(1 for r in rows if r["label"] == 0)
    n_fake = len(rows) - n_auth
    scenes = len({r["group"] for r in rows})
    print(f"\nListo: {out}/manifest.csv")
    print(f"  {scenes} escenas | {n_auth} auténticas | {n_fake} manipuladas | "
          f"{n_auth + n_fake} filas")
    print("Siguiente paso:")
    print(f"  python src/extract_maps.py --manifest {out}/manifest.csv "
          f"--cache cache/coco --variants raw --map-size 384")
    print(f"  python src/pack_cache.py --cache cache/coco/raw__s384 --out packed/coco__s384")


if __name__ == "__main__":
    main()
