"""Paso 1: correr TruFor una vez por imagen y cachear lo que consume el ViT.

Por cada fila del manifest y cada variante:

    imagen -> --max-side -> preprocess.apply_variant() -> TruFor -> maps.normalize_maps()

y se guarda un .npz con los mapas YA normalizados al tamaño del ViT. Es
obligatorio normalizar acá: a resolución nativa los mapas de IMD2020 ocuparían
decenas de GB y el ViT de todos modos necesita una entrada fija.

Salida: <cache>/<clave>/<stem>.npz con
    map      uint8 (S,S)      mapa de manipulación, 0-255
    conf     uint8 (S,S)      mapa de confianza, 0-255
    score    float32 ()       score global de TruFor
    rgb      uint8 (S,S,3)    thumbnail de la imagen, para no releer el original
    orig_hw  int32 (2)        alto, ancho originales (metadata)

La clave del cache incluye variante, tamaño y max-side, así que cambiar
cualquiera de los tres no reutiliza mapas viejos por error. Ojo: antes el
max-side se aplicaba DESPUÉS de la variante; para raw da lo mismo, pero los
caches viejos de jpeg/blur se hicieron con el otro orden.

Es REANUDABLE (saltea lo ya cacheado) y escribe de forma atómica: si Colab se
desconecta a mitad de camino no quedan archivos corruptos.

    python src/extract_maps.py --manifest data/manifest.csv --cache cache \
        --variants raw --map-size 384 --max-side 1280
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect import analyze, load_model  # noqa: E402
from maps import normalize_maps, resize_rgb, to_uint8  # noqa: E402
from preprocess import VARIANTS, apply_variant  # noqa: E402
from preprocess import max_side as resize_max_side  # noqa: E402

def cache_key(variant: str, size: int, max_side: int) -> str:
    key = f"{variant}__s{size}"
    return f"{key}__ms{max_side}" if max_side else key


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    # Misma conversión que detect.preprocess: SIN resize, uint8 -> [0, 1)
    t = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float) / 256.0
    return t.unsqueeze(0)


def read_manifest(path: Path) -> list[dict]:
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--variants", nargs="+", default=["raw"])
    ap.add_argument("--map-size", type=int, default=384,
                    help="lado S al que se normalizan los mapas (entrada del ViT)")
    ap.add_argument("--max-side", type=int, default=1000,
                    help="si >0, baja el lado mayor a este valor conservando el aspecto, antes "
                         "de la variante y de TruFor (memoria de la GPU; default: 1000)")
    ap.add_argument("--heat-mode", default="blend", choices=["area", "max", "blend"])
    ap.add_argument("--conf-mode", default="area", choices=["area", "max", "blend"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=0, help="procesar solo las primeras N (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="recalcular aunque exista el cache")
    a = ap.parse_args()

    manifest = Path(a.manifest)
    if not manifest.exists():
        sys.exit(f"No existe {manifest}. Corré primero src/prepare_imd2020.py")
    rows = read_manifest(manifest)
    # `--pairs balanced` repite la auténtica una vez por variante manipulada: se
    # procesa una vez por archivo, si no TruFor corre de más sobre la misma foto.
    seen, unique = set(), []
    for r in rows:
        stem = Path(r["file"]).stem
        if stem in seen:
            continue
        seen.add(stem)
        unique.append(r)
    if len(unique) != len(rows):
        print(f"manifest: {len(rows)} filas -> {len(unique)} imágenes únicas")
    rows = unique
    if a.limit:
        rows = rows[:a.limit]
    for v in a.variants:
        if v not in VARIANTS:
            sys.exit(f"Variante desconocida: {v} (definidas: {', '.join(VARIANTS)})")

    root = manifest.parent
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    model, weights = load_model(device)
    print(f"TruFor | pesos: {weights.name} | dispositivo: {device} | {len(rows)} imágenes")

    failed = []
    for v in a.variants:
        key = cache_key(v, a.map_size, a.max_side)
        out = Path(a.cache) / key
        out.mkdir(parents=True, exist_ok=True)
        (out / "_meta.json").write_text(json.dumps({
            "variant": v, "map_size": a.map_size, "max_side": a.max_side,
            "heat_mode": a.heat_mode, "conf_mode": a.conf_mode,
        }, indent=2))

        todo = [r for r in rows
                if a.overwrite or not (out / f"{Path(r['file']).stem}.npz").exists()]
        print(f"\n[{key}] pendientes: {len(todo)} de {len(rows)}")

        t0 = time.monotonic()
        for i, r in enumerate(todo, 1):
            stem = Path(r["file"]).stem
            src = Path(r["file"]) if Path(r["file"]).is_absolute() else root / r["file"]
            try:
                arr = np.array(Image.open(src).convert("RGB"))
                if a.max_side:
                    arr = resize_max_side(arr, a.max_side)
                arr = apply_variant(arr, v)
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

                m = normalize_maps(heat, conf, a.map_size, a.heat_mode, a.conf_mode)
                h, w = arr.shape[:2]
                tmp = out / f"{stem}.tmp.npz"
                np.savez_compressed(
                    tmp,
                    map=to_uint8(m[0]),
                    conf=to_uint8(m[1]),
                    score=np.float32(score),
                    rgb=resize_rgb(arr, a.map_size),
                    orig_hw=np.array([h, w], dtype=np.int32),
                )
                tmp.replace(out / f"{stem}.npz")   # atómico
            except Exception as e:  # una imagen rota no debe frenar el lote
                failed.append((key, r["file"], repr(e)))
                print(f"  ERROR {r['file']}: {e!r}")

            if i % 25 == 0 or i == len(todo):
                el = time.monotonic() - t0
                eta = el / i * (len(todo) - i)
                print(f"  [{key}] {i}/{len(todo)} | {el / i:.2f} s/img | ETA {eta / 60:.0f} min")

    if failed:
        print(f"\n{len(failed)} imagen(es) fallaron:")
        for k, n, e in failed[:20]:
            print(f"  [{k}] {n}: {e}")
        sys.exit(1)
    print("\nCache completo. Siguiente paso: python src/pack_cache.py --cache "
          f"{Path(a.cache) / cache_key(a.variants[0], a.map_size, a.max_side)} --out packed")


if __name__ == "__main__":
    main()
