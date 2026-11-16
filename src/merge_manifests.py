"""Une varios manifests en uno solo para entrenar con todos los datasets juntos.

Problema que resuelve
---------------------
IMD2020 y COCOInpaint tienen caches empaquetados distintos y pueden compartir
nombres de archivo (COCO numera sus fotos). Fusionar los .npy obligaría a
renombrar todo, así que en vez de eso cada fila lleva su propia base en la
columna `packed` y `MapDataset` resuelve a qué array pertenece.

Además:
  - agrega la columna `dataset`
  - prefija `group` con el nombre del dataset, para que dos escenas de conjuntos
    distintos no puedan caer en el mismo grupo por casualidad
  - convierte `file` y `mask` a rutas ABSOLUTAS: los manifests originales son
    relativos a su propia carpeta y desde `data/` no resolverían

El manifest resultante es específico de esta máquina (por las rutas absolutas),
pero es un artefacto derivado: se regenera en segundos.

Uso:
    python src/merge_manifests.py --out data/manifest_all.csv \
        --input imd  data/manifest.csv             packed/imd__s384 \
        --input coco data/coco_inpaint/manifest.csv packed/coco__s384
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

BASE_COLUMNS = ["file", "mask", "label", "group", "dataset", "source", "width",
                "height", "packed"]


def load(path: Path, name: str, packed: str) -> list[dict]:
    """Lee un manifest y lo deja listo para mezclar."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"No existe {path}")
    # El manifest no sirve de nada si el paquete al que apunta no existe: el
    # error aparecería recién al entrenar, después de mezclar todo.
    npy, js = Path(packed).with_suffix(".npy"), Path(packed).with_suffix(".json")
    if not npy.exists():
        raise SystemExit(f"{name}: no existe {npy}. Corré src/pack_cache.py antes de mezclar.")
    if not js.exists():
        raise SystemExit(f"{name}: no existe {js} (lo escribe src/pack_cache.py).")
    root = path.resolve().parent
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r = dict(r)
            r["dataset"] = name
            r["group"] = f"{name}:{r['group']}"
            r["packed"] = packed
            for key in ("file", "mask"):
                value = (r.get(key) or "").strip()
                if value:
                    p = Path(value)
                    r[key] = str(p if p.is_absolute() else (root / p).resolve())
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description="Une manifests de varios datasets")
    ap.add_argument("--out", default="data/manifest_all.csv")
    ap.add_argument("--input", nargs=3, action="append", required=True,
                    metavar=("NOMBRE", "MANIFEST", "PACKED"),
                    help="repetible: nombre del dataset, su manifest y su cache empaquetado")
    a = ap.parse_args()

    rows, columns = [], list(BASE_COLUMNS)
    for name, manifest, packed in a.input:
        part = load(Path(manifest), name, packed)
        for r in part:
            for k in r:
                if k not in columns:
                    columns.append(k)
        rows.extend(part)
        labels = Counter(r["label"] for r in part)
        scenes = len({r["group"] for r in part})
        print(f"{name:6s} {len(part):5d} filas | {scenes:4d} escenas | "
              f"{labels.get('0', 0)} auténticas | {labels.get('1', 0)} manipuladas")

    if not rows:
        raise SystemExit("No hay filas: revisá los manifests de entrada.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, restval="")
        w.writeheader()
        w.writerows(rows)

    labels = Counter(r["label"] for r in rows)
    print(f"\n{out}: {len(rows)} filas | {len({r['group'] for r in rows})} escenas | "
          f"{labels.get('0', 0)} auténticas | {labels.get('1', 0)} manipuladas")
    print("Siguiente paso:")
    print(f"  python src/train.py --manifest {out} --backbone swin_t --ft partial \\")
    print(f"      --out runs/swin_partial --size 384")


if __name__ == "__main__":
    main()
