"""Dataset de clasificación: mapas de TruFor normalizados (memmap) -> etiqueta de imagen.

El .npy empaquetado tiene 5 canales por imagen: R, G, B, mapa, confianza.
El split es POR GRUPO (escena), nunca por archivo: en IMD2020 cada manipulada
tiene un original casi idéntico, y si caen en lados distintos la validación
mide memorización en vez de detección.
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Normalización ImageNet para el canal RGB: los backbones preentrenados la
# esperan. Los mapas ya vienen en 0-1 y no se tocan.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

CHANNELS = {
    "maps+rgb": (0, 5),   # R, G, B, mapa, confianza
    "maps": (3, 5),       # mapa, confianza
}


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        if r.get("width") and r.get("height"):
            r["width"], r["height"] = int(r["width"]), int(r["height"])
    return rows


def dedupe_rows(rows: list[dict]) -> list[dict]:
    """Una fila por (dataset, archivo, etiqueta).

    `prepare_imd2020.py --pairs balanced` repite la auténtica una vez por cada
    variante manipulada, para que las dos clases pesen igual. Eso está bien para
    entrenar (cada escena aporta lo mismo) pero no para medir: la misma foto
    contaría varias veces en la validación. Acá se colapsan las copias.

    `dataset` entra en la clave porque dos datasets pueden compartir el nombre de
    archivo (COCO numera sus fotos, IMD2020 usa hashes).
    """
    seen, out = set(), []
    for r in rows:
        key = (r.get("dataset", ""), r["file"], r["label"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def load_packed(base: str | Path):
    """Carga packed.npy (+ .json) y devuelve (memmap, meta, stem -> fila)."""
    base = Path(base)
    npy = base.with_suffix(".npy")
    js = base.with_suffix(".json")
    if not npy.exists():
        raise SystemExit(f"No existe {npy}. Corré src/pack_cache.py primero.")
    meta = json.loads(js.read_text())
    arr = np.load(npy, mmap_mode="r")
    if arr.shape[0] != len(meta["stems"]):
        raise SystemExit(f"{npy}: {arr.shape[0]} filas != {len(meta['stems'])} stems")
    return arr, meta, {s: i for i, s in enumerate(meta["stems"])}


def make_splits(rows: list[dict], val_frac: float = 0.2, seed: int = 0) -> dict:
    """Split por grupo, estratificado por (dataset, etiqueta). Reproducible.

    Estratificar por dataset además de por etiqueta evita que, al mezclar dos
    conjuntos, uno de los dos quede con una sola clase (o sin escenas) en la
    validación.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)

    # Un grupo mixto (original + manipulada fusionados por hash) cuenta como
    # positivo para estratificar: siempre caen juntos del mismo lado.
    by_stratum = defaultdict(list)
    for g, rs in groups.items():
        label = 1 if any(r["label"] == 1 for r in rs) else 0
        by_stratum[(rs[0].get("dataset", ""), label)].append(g)

    rng = random.Random(seed)
    val_groups = set()
    for stratum in sorted(by_stratum):
        gs = by_stratum[stratum][:]
        rng.shuffle(gs)
        total = sum(len(groups[g]) for g in gs)
        acc, chosen = 0, 0
        for g in gs:
            if acc >= total * val_frac or chosen >= len(gs) - 1:
                break                      # siempre queda >=1 grupo en train
            val_groups.add(g)
            acc += len(groups[g])
            chosen += 1

    split = {g: ("val" if g in val_groups else "train") for g in groups}
    counts = {"train": 0, "val": 0, "train_pos": 0, "val_pos": 0}
    per_ds = defaultdict(lambda: {"train": 0, "val": 0, "train_pos": 0, "val_pos": 0})
    for r in rows:
        s = split[r["group"]]
        counts[s] += 1
        counts[f"{s}_pos"] += r["label"]
        d = r.get("dataset", "")
        per_ds[d][s] += 1
        per_ds[d][f"{s}_pos"] += r["label"]
    return {"seed": seed, "val_frac": val_frac, "groups": split, "counts": counts,
            "by_dataset": {k: v for k, v in per_ds.items()}}


def apply_splits(rows: list[dict], splits: dict):
    """Devuelve (train, val) usando los grupos guardados. Falla si falta un grupo."""
    missing = {r["group"] for r in rows} - set(splits["groups"])
    if missing:
        raise SystemExit(
            f"El split no conoce {len(missing)} grupos (ej: {sorted(missing)[:3]}). "
            "Regenerá el split con make_splits para no filtrar datos entre train y val.")
    train = [r for r in rows if splits["groups"][r["group"]] == "train"]
    val = [r for r in rows if splits["groups"][r["group"]] == "val"]
    return train, val


def pos_weight(rows: list[dict]) -> float:
    """Peso para la clase positiva en BCEWithLogits (negativos / positivos)."""
    pos = sum(r["label"] for r in rows)
    neg = len(rows) - pos
    return max(neg, 1) / max(pos, 1)


class MapDataset(Dataset):
    """Lee de uno o varios memmaps empaquetados. Devuelve (x, y, score).

    Cada fila puede traer su propia base en la columna `packed` (el manifest
    merged de IMD2020 + COCOInpaint la escribe). Si no la trae, se usa el
    `--packed` de la línea de comandos para todas: es el caso de un dataset
    solo.

    Dos bases distintas no se pueden fusionar en un solo array sin renombrar
    archivos, y dos datasets pueden compartir el nombre de archivo (COCO numera
    sus fotos). Por eso se indexa por (base, stem) y no sólo por stem.
    """

    def __init__(self, packed: str | None = None, rows: list[dict] | None = None,
                 channels: str = "maps+rgb", train: bool = False):
        if channels not in CHANNELS:
            raise SystemExit(f"channels desconocido: {channels} (válidos: {list(CHANNELS)})")
        self.c0, self.c1 = CHANNELS[channels]
        self.train = train
        self.metas: dict[str, dict] = {}
        self.sets: dict[str, tuple] = {}
        self.rows: list[dict] = []
        self.items: list[tuple[str, int]] = []

        missing = 0
        for r in rows or []:
            base = r.get("packed") or packed
            if not base:
                raise SystemExit(
                    f"La fila {r['file']} no trae `packed` y no pasaste --packed. "
                    "Generá el manifest merged con src/merge_manifests.py o pasá --packed.")
            if base not in self.metas:
                arr, meta, index = load_packed(base)
                self.metas[base] = meta
                self.sets[base] = (arr, index)
            index = self.sets[base][1]
            stem = Path(r["file"]).stem
            if stem not in index:
                missing += 1
                continue
            self.rows.append(r)
            self.items.append((base, index[stem]))

        if missing:
            names = ", ".join(Path(b).name for b in self.metas)
            print(f"AVISO: {missing} filas del manifest no están en el cache ({names}); "
                  "se omiten.")
        self.scores = np.asarray(
            [self.metas[b]["scores"][i] for b, i in self.items], dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def raw(self, k: int) -> np.ndarray:
        """Array uint8 (5, S, S) de la fila k, para visualizaciones."""
        base, i = self.items[k]
        return np.array(self.sets[base][0][i])

    @property
    def sizes(self) -> dict[str, int]:
        return {b: int(m["size"]) for b, m in self.metas.items()}

    def summary(self) -> str:
        parts = []
        for b, m in self.metas.items():
            parts.append(f"{Path(b).name}:{m.get('variant', '?')}/{m['size']}px")
        return " | ".join(parts)

    def __getitem__(self, k):
        base, i = self.items[k]
        x = torch.from_numpy(np.array(self.sets[base][0][i])).float() / 255.0  # (5, S, S)
        if self.train:                                                # dihedral: igual en los 5 canales
            if random.random() < 0.5:
                x = x.flip(-1)
            if random.random() < 0.5:
                x = x.flip(-2)
            if random.random() < 0.5:
                x = torch.rot90(x, random.randint(1, 3), (-2, -1))

        rgb = (x[:3] - MEAN) / STD
        x = torch.cat([rgb, x[3:]], 0)[self.c0:self.c1]
        return x, torch.tensor(float(self.rows[k]["label"])), torch.tensor(self.scores[k])
