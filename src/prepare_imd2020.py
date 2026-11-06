"""Prepara IMD2020 (versión de Kaggle: saikotahmed/imd2020) en el formato del proyecto.

Salida
------
    <out>/images/<prefijo><pair_key><ext>          manipuladas (bytes originales)
    <out>/images/<prefijo><pair_key>__auth<ext>    auténticas
    <out>/masks/<prefijo><pair_key>.png            máscara binaria (255 = manipulado)
    <out>/manifest.csv                             file, mask, label, group, source, width, height

Layout esperado (se busca en cualquier nivel de --src):
    img/ (manipuladas) + mask/ (máscaras) + Authentic/ (auténticas, sin máscara)

Dos claves distintas
--------------------
Emparejar archivos y agrupar escenas son problemas distintos y mezclarlos
rompe cualquiera de los dos:

  pair_key   empareja imagen <-> máscara. Sólo saca sufijos de máscara, así
             cada archivo conserva su identidad:
                 <escena>_fake_01.jpg  <->  <escena>_fake_01_mask.png
             Es también el nombre de archivo en la salida, para que la política
             de agrupamiento se pueda cambiar sin recopiar nada.
  group_key  identifica la ESCENA. Saca el tipo de versión (_orig, _fake, ...)
             y el índice de variante (_01, _02, ...):
                 <escena>_orig.jpg, <escena>_fake.jpg, <escena>_fake_01.jpg
                 -> todas al grupo <escena>

Grupos y fuga de información
----------------------------
IMD2020 empareja cada imagen manipulada con su original. Si el original cae en
train y la manipulada en validación, la métrica es mentira. Por eso el split se
hace por `group`, no por archivo:

  1. `--group-by content` (default en layout plano) compara las IMÁGENES, no los
     nombres: dos archivos son la misma escena si sus miniaturas correlacionan
     por encima de --ncc-threshold. Es la única opción fiable cuando el release
     renumeró los archivos, y el repack de Kaggle lo hizo: sólo 38 de 414
     auténticas comparten id con una manipulada y de esas 38 la mayoría ni
     siquiera son la misma foto (NCC mediana 0.71, 29% por debajo de 0.5).
  2. `--group-by name` usa el nombre: si el dataset conserva una carpeta por
     escena el grupo es esa carpeta; si no, `group_key()` saca el tipo de
     versión (_orig, _fake, ...) y el índice de variante (_01, _02, ...).
  3. En modo `name`, `--link-authentic` (tamaño + dHash mutuo) y
     `--dedupe-hash` (fusión de grupos casi idénticos) son la red de seguridad.

Qué imágenes entran (`--pairs`)
-------------------------------
El agrupamiento resuelve la fuga, pero no la confusión escena <-> clase: si las
manipuladas salen de escenas que nunca aparecen como auténticas, el modelo puede
aprender "esta escena es trucada" en vez de "esta zona está editada".

  off       todas las imágenes (default, compatible con cualquier dataset).
  unique    sólo escenas con las DOS clases: la única diferencia entre clases es
            la edición. Una fila por archivo.
  balanced  igual que `unique`, pero emite una fila auténtica por cada variante
            manipulada de la escena. Cada escena pesa lo mismo en las dos clases
            y el conjunto queda 1:1 (pos_weight = 1), a costa de repetir la misma
            auténtica. En IMD2020: 1606 -> 264 escenas y 2424 -> 1552 filas
            (1042 imágenes únicas), descartando el 61% de las manipuladas.

Las copias apuntan al MISMO archivo (no se duplican bytes): el cache guarda un
.npz por imagen única.

Cómo se compara el contenido
----------------------------
  * dHash de 64 bits como FILTRO de candidatos: barato y tolera la recompresión.
  * correlación normalizada (NCC) a --work-size en gris como CONFIRMACIÓN,
    calculada IGNORANDO la zona editada (la máscara del manipulada, dilatada).
    Comparar la imagen entera castiga los pares con ediciones grandes: pares
    original/manipulada con NCC global 0.80-0.85 dan 0.995 fuera de la máscara.
    Con la zona editada ignorada, la misma foto da ~0.99 y dos fotos distintas
    0.02 (control sobre IMD2020: p99.9 = 0.79, 0.06% por encima de 0.90).
  * complete-linkage: dos grupos se unen sólo si TODOS sus pares superan el
    umbral. Single-linkage encadena (A~B y B~C arrastran C aunque A y C no se
    parezcan) y termina fusionando escenas distintas.
  * sólo se comparan píxeles de imágenes del mismo tamaño: alinear resoluciones
    distintas exige reescalar y eso inventa parecidos.

El script es conservador: diagnostica e informa (pares sin máscara, tamaños que
no coinciden, máscaras posiblemente invertidas, grupos y auténticas sin pareja)
en vez de adivinar en silencio.
"""
import argparse
import csv
import re
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

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


def image_size(path: Path) -> tuple[int, int]:
    """(ancho, alto) leyendo sólo el header, sin decodificar la imagen."""
    with Image.open(path) as im:
        return im.size


def dhash(path: Path, size: int = 8) -> int:
    """dHash de 64 bits: robusto a recompresión y a ediciones locales chicas."""
    with Image.open(path) as im:
        # draft hace que libjpeg decodifique a escala 1/8 en vez de tamaño completo:
        # para 2400 imágenes grandes es la diferencia entre segundos y minutos.
        im.draft("L", (size + 1, size))
        a = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def _bits(hashes: np.ndarray) -> np.ndarray:
    """(n,) uint64 -> (n, 64) int16 con los bits de cada hash."""
    return np.unpackbits(hashes.view(np.uint8).reshape(len(hashes), 8), axis=1).astype(np.int16)


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matriz (len(a), len(b)) de distancias de Hamming entre dos lotes de bits."""
    return a @ (1 - b).T + (1 - a) @ b.T


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


def link_authentic(auth: list[dict], tamp: list[dict], threshold: int = 8,
                   verbose: bool = True) -> dict[int, str]:
    """Enlaza cada auténtica con el grupo de su original manipulado.

    Se apoya en dos invariantes: una imagen y su versión manipulada tienen las
    MISMAS dimensiones y un dHash casi idéntico (la edición es local). Se exige
    además que el par sea el mejor match MUTUO, lo que evita cadenas del tipo
    A~T1, T1~T2, ... que terminan fusionando escenas distintas.

    `auth` y `tamp` son listas de dicts con claves path, size y group.
    Devuelve {índice dentro de `auth`: grupo de la manipulada enlazada}.
    """
    tamp_by_size = defaultdict(list)
    for j, t in enumerate(tamp):
        tamp_by_size[t["size"]].append(j)
    auth_by_size = defaultdict(list)
    for i, a in enumerate(auth):
        auth_by_size[a["size"]].append(i)

    cache = {}

    def h(p):
        if p not in cache:
            cache[p] = dhash(p)
        return cache[p]

    links, dists = {}, []
    for size, tids in tamp_by_size.items():
        aids = auth_by_size.get(size)
        if not aids:
            continue
        ha = np.array([h(auth[i]["path"]) for i in aids], dtype=np.uint64)
        ht = np.array([h(tamp[j]["path"]) for j in tids], dtype=np.uint64)
        d = _hamming(_bits(ha), _bits(ht))
        best_t = d.argmin(1)                 # para cada auténtica, su mejor manipulada
        best_a = d.argmin(0)                 # para cada manipulada, su mejor auténtica
        for r, i in enumerate(aids):
            j = int(best_t[r])
            if int(best_a[j]) != r or int(d[r, j]) > threshold:
                continue                     # no es el mejor mutuo: no se enlaza
            links[i] = tamp[tids[j]]["group"]
            dists.append(int(d[r, j]))

    if verbose:
        print(f"  tamaño+dHash: {len(links)} enlaces aceptados (umbral {threshold})")
        if dists:
            hist = Counter(x // 4 * 4 for x in dists)
            print("  distancia de los enlaces: "
                  + ", ".join(f"{k}-{k + 3}:{v}" for k, v in sorted(hist.items())))
    return links


def merge_by_hash(paths: list[Path], groups: list[str], threshold: int, verbose=True):
    """Fusiona grupos cuyas imágenes son casi idénticas (distancia de Hamming <= threshold)."""
    if verbose:
        print(f"  calculando dHash de {len(paths)} imágenes...")
    hashes = np.array([dhash(p) for p in paths], dtype=np.uint64)
    bits = _bits(hashes)
    d = _hamming(bits, bits)
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


def work_gray(path: Path, size: int) -> np.ndarray:
    """Gris a resolución de trabajo: chico para caber en RAM, grande para ver el detalle."""
    with Image.open(path) as im:
        im.draft("L", (size * 2, size * 2))     # JPEG: decodifica a 1/8
        return np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def work_mask(path: Path | None, size: int, dilate: int = 9) -> np.ndarray:
    """Máscara booleana a resolución de trabajo, dilatada (vacía si no hay máscara).

    Se dilata porque la comparación es a baja resolución: sin dilatación un borde
    de la edición quedaría fuera de la zona ignorada y ensuciaría la correlación.
    """
    if path is None:
        return np.zeros((size, size), dtype=bool)
    with Image.open(path) as m:
        a = m.convert("L").resize((size, size), Image.NEAREST)
        if dilate > 1:
            a = a.filter(ImageFilter.MaxFilter(dilate))
    return np.asarray(a) > 127


def masked_ncc(G: np.ndarray, M: np.ndarray, i: int, j: int,
               min_keep: float = 0.25) -> float:
    """NCC de dos imágenes ignorando la zona editada de CUALQUIERA de las dos.

    Comparar la imagen entera castiga a los pares con ediciones grandes: medido
    sobre IMD2020, pares original/manipulada con NCC global 0.80-0.85 tienen NCC
    fuera de la máscara 0.995. Ignorar lo editado separa de verdad "misma foto"
    (mediana 0.99) de "foto distinta" (media 0.02).
    """
    keep = ~(M[i] | M[j])
    if keep.mean() < min_keep:
        return float("nan")                     # casi todo editado: no se puede decidir
    a = G[i][keep].astype(np.float32)
    b = G[j][keep].astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return float("nan")                     # región uniforme: NCC indefinida
    return float((a * b).sum() / (na * nb))


def candidate_pairs(hashes: np.ndarray, threshold: int, chunk: int = 512):
    """Pares (i, j) con i < j y distancia de Hamming <= threshold.

    Se procesa por bloques de filas para no materializar la matriz n x n: con
    10 000 imágenes serían 100 M de enteros.
    """
    bits = _bits(hashes)
    oi, oj = [], []
    for s in range(0, len(bits), chunk):
        d = _hamming(bits[s:s + chunk], bits)
        ii, jj = np.where(d <= threshold)
        ii = ii + s
        keep = jj > ii
        oi.append(ii[keep])
        oj.append(jj[keep])
    if not oi:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(oi), np.concatenate(oj)


def cluster_complete(n: int, edges: dict[tuple[int, int], float], threshold: float) -> list[int]:
    """Complete-linkage: une dos grupos sólo si TODOS sus pares superan el umbral.

    Single-linkage encadena (A~B, B~C arrastra C aunque A y C no se parezcan) y
    termina armando un grupo con fotos distintas. Acá se aceptan primero las
    aristas más fuertes, así los grupos crecen de adentro hacia afuera.
    """
    par = list(range(n))
    members = {i: [i] for i in range(n)}

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for (i, j), w in sorted(edges.items(), key=lambda kv: -kv[1]):
        if w < threshold:
            break
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        ok = True
        for x in members[ri]:
            for y in members[rj]:
                if x != y and edges.get((min(x, y), max(x, y)), 0.0) < threshold:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            par[rj] = ri
            members[ri].extend(members.pop(rj))
    return [find(i) for i in range(n)]


def group_by_content(paths: list[Path], masks: list[Path | None], sizes: list[tuple[int, int]],
                     ncc_threshold: float = 0.90, dhash_threshold: int = 16,
                     work_size: int = 128, verbose: bool = True) -> tuple[list[str], dict]:
    """Agrupa imágenes que son la MISMA foto, sin mirar el nombre del archivo.

    dHash filtra candidatos (barato, invariante a la recompresión) y la
    correlación fuera de la máscara confirma (caro pero selectivo). Sólo se
    comparan píxeles de imágenes del mismo tamaño: alinear resoluciones distintas
    exige reescalar y eso inventa parecidos.

    Devuelve una etiqueta de grupo por imagen y un dict con las estadísticas.
    """
    t0 = time.monotonic()
    hashes = np.array([dhash(p) for p in paths], dtype=np.uint64)
    ii, jj = candidate_pairs(hashes, dhash_threshold)
    wh = np.array([w * 100003 + h for w, h in sizes])
    same = wh[ii] == wh[jj]
    ii, jj = ii[same], jj[same]

    G = np.stack([work_gray(p, work_size) for p in paths])
    M = np.stack([work_mask(m, work_size, max(3, (work_size // 14) | 1)) for m in masks])
    vals = np.array([masked_ncc(G, M, int(a), int(b)) for a, b in zip(ii, jj)])
    known = ~np.isnan(vals)
    edges = {(int(a), int(b)): float(v) for a, b, v in zip(ii[known], jj[known], vals[known])}
    roots = cluster_complete(len(paths), edges, ncc_threshold)

    members = defaultdict(list)
    for i, r in enumerate(roots):
        members[r].append(i)

    labels, used = [""] * len(paths), set()
    for idxs in members.values():
        rep = min(idxs)                       # representante estable: el primero
        base = paths[rep].stem.lower()
        name, k = base, 1
        while name in used:                   # dos grupos no pueden llamarse igual
            k += 1
            name = f"{base}_{k}"
        used.add(name)
        for i in idxs:
            labels[i] = name

    multi = [v for v in members.values() if len(v) > 1]
    internal = np.array([edges[(min(a, b), max(a, b))]
                         for v in multi for x, a in enumerate(v) for b in v[x + 1:]])
    info = {
        "groups": len(members), "largest": max(len(v) for v in members.values()),
        "multi": len(multi), "candidates": len(ii), "pairs": len(edges),
        "undecided": int((~known).sum()),
    }
    if verbose:
        print(f"  {len(paths)} imágenes -> {info['groups']} grupos "
              f"(mayor: {info['largest']} archivos, con más de uno: {info['multi']})")
        print(f"  candidatos dHash<={dhash_threshold}: {info['candidates']} pares | "
              f"comparables: {info['pairs']} | sin zona limpia: {info['undecided']} | "
              f"confirmados NCC>={ncc_threshold:.2f}: "
              f"{sum(1 for w in edges.values() if w >= ncc_threshold)}")
        if internal.size:
            print(f"  pares dentro de un grupo: {internal.size} | "
                  f"NCC mediana {np.median(internal):.3f} | "
                  f">=0.99: {(internal >= 0.99).mean():.1%} | mínimo {internal.min():.3f}")
        if info["largest"] > 30:
            print("  AVISO: un grupo muy grande sugiere umbral bajo; subí --ncc-threshold.")
        print(f"  ({time.monotonic() - t0:.1f}s)")
    return labels, info


def main():
    ap = argparse.ArgumentParser(description="Prepara IMD2020 para el clasificador")
    ap.add_argument("--src", required=True, help="carpeta del dataset descargado")
    ap.add_argument("--out", default="data")
    ap.add_argument("--prefix", default="imd_")
    ap.add_argument("--invert", action="store_true", help="si las máscaras vienen con 255 = auténtico")
    ap.add_argument("--group-by", choices=["auto", "name", "content"], default="auto",
                    help="cómo definir la escena: 'content' compara las imágenes (fiable aunque "
                         "los nombres sean identificadores sin sentido), 'name' usa el nombre del "
                         "archivo, 'auto' = content en layout plano y name si hay carpeta por escena")
    ap.add_argument("--ncc-threshold", type=float, default=0.90,
                    help="correlación mínima fuera de la zona editada para considerar dos "
                         "imágenes la misma foto (0-1; 0.90 deja ~0.06%% de falsos positivos)")
    ap.add_argument("--dhash-candidates", type=int, default=16,
                    help="distancia Hamming máxima para que un par sea candidato (0-64)")
    ap.add_argument("--work-size", type=int, default=128,
                    help="resolución de trabajo para correlacionar (más alto = más preciso "
                         "y más lento)")
    ap.add_argument("--link-authentic", choices=["on", "off"], default="on",
                    help="sólo con --group-by name: enlazar cada auténtica con su original "
                         "manipulado (tamaño + dHash)")
    ap.add_argument("--link-threshold", type=int, default=8,
                    help="sólo con --group-by name: distancia Hamming máxima para enlazar (0-64)")
    ap.add_argument("--dedupe-hash", choices=["auto", "on", "off"], default="auto",
                    help="sólo con --group-by name: fusionar grupos con imágenes casi idénticas "
                         "(auto: solo si nada agrupó)")
    ap.add_argument("--dedupe-threshold", type=int, default=8,
                    help="sólo con --group-by name: distancia Hamming máxima (0-64)")
    ap.add_argument("--pairs", choices=["off", "unique", "balanced"], default="off",
                    help="qué imágenes entran al manifest: 'off' = todas, 'unique' = sólo "
                         "escenas con auténtica y manipulada, 'balanced' = además repite la "
                         "auténtica una vez por variante para quedar 1:1")
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

    # ---- grupo base por nombre -------------------------------------------
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
    print(f"layout {'plano' if flat else 'por carpetas'} | "
          f"grupos por nombre: {len(set(groups.values()))}")

    all_keys = list(common) + [f"auth:{k}" for k in auths]
    all_paths = [imgs[k] for k in common] + [auths[k] for k in auths]
    dims = {k: image_size(p) for k, p in zip(all_keys, all_paths)}

    tamp_groups = {groups[k] for k in common}
    auth_keys = [f"auth:{k}" for k in auths]
    matched_name = sum(groups[k] in tamp_groups for k in auth_keys)
    print(f"auténticas emparejadas por nombre con una manipulada: {matched_name} de {len(auths)}")
    tamp_sizes = {dims[k] for k in common}
    same_size = sum(dims[k] in tamp_sizes for k in auth_keys)
    print(f"auténticas con alguna manipulada del mismo tamaño: {same_size} de {len(auths)}")

    # ---- agrupamiento -----------------------------------------------------
    # El nombre sólo es fiable si el dataset conserva una carpeta por escena. En
    # un layout plano lo más probable es que sea un identificador sin relación
    # con la escena (es el caso del repack de IMD2020 en Kaggle).
    content = a.group_by == "content" or (a.group_by == "auto" and flat)
    if content and len(all_paths) > 1:
        print("\nAgrupando por CONTENIDO (se ignora el nombre del archivo):")
        mask_paths = [None if k.startswith("auth:") else masks[k] for k in all_keys]
        labels, info = group_by_content(all_paths, mask_paths,
                                        [dims[k] for k in all_keys],
                                        a.ncc_threshold, a.dhash_candidates, a.work_size)
        if info["groups"] == len(all_paths) and len(set(groups.values())) < len(groups):
            print("  el contenido no encontró fotos repetidas: se agrupa por nombre.")
            content = False
        else:
            for k, lab in zip(all_keys, labels):
                groups[k] = lab
    elif content:
        content = False

    if content:
        print("  enlace por tamaño+dHash y dedupe por hash: innecesarios "
              "(el contenido ya agrupa las escenas).")
    else:
        # ---- enlace auténtica -> escena manipulada ------------------------
        # Necesario cuando el release renumeró una de las carpetas: el nombre ya
        # no vincula el original con su manipulación. Es seguro por construcción,
        # así que si las auténticas fueran un pool independiente simplemente
        # enlaza 0.
        auth_list = list(auths)
        pending = [i for i, k in enumerate(auth_keys) if groups[k] not in tamp_groups]
        if a.link_authentic == "on" and pending and common:
            print(f"Enlazando {len(pending)} auténticas sin par por nombre (tamaño + dHash):")
            tamp_items = [{"path": imgs[k], "size": dims[k], "group": groups[k]} for k in common]
            auth_items = [{"path": auths[auth_list[i]], "size": dims[f"auth:{auth_list[i]}"],
                           "group": groups[f"auth:{auth_list[i]}"]} for i in pending]
            links = link_authentic(auth_items, tamp_items, a.link_threshold)
            for r, g in links.items():
                groups[f"auth:{auth_list[pending[r]]}"] = g

        all_groups = [groups[k] for k in all_keys]
        # `auto`: el hash global es la red de seguridad para cuando ni el nombre
        # ni el enlace lograron agrupar. Si ya agruparon, fusionar por similitud
        # sobre-fusiona escenas distintas (aparece un grupo enorme).
        no_name_grouping = len(set(all_groups)) == len(all_groups)
        no_auth_links = bool(auths) and matched_name == 0
        use_hash = a.dedupe_hash == "on" or (
            a.dedupe_hash == "auto" and (no_name_grouping or no_auth_links))
        if a.dedupe_hash == "auto":
            print("dedupe por hash: " + (
                "activado (ni el nombre ni el enlace agruparon)" if use_hash
                else "innecesario (el nombre o el enlace ya agrupan las escenas)"))
        if use_hash and all_paths:
            print("Fusionando grupos por similitud visual (dHash):")
            for k, g in zip(all_keys, merge_by_hash(all_paths, all_groups, a.dedupe_threshold)):
                groups[k] = g

    all_groups = [groups[k] for k in all_keys]
    tamp_groups = {groups[k] for k in common}
    matched = sum(groups[k] in tamp_groups for k in auth_keys)
    how = "contenido" if content else "dHash"
    extra = f" (+{matched - matched_name} por {how})" if matched > matched_name else ""
    print(f"auténticas con escena compartida con una manipulada: {matched} de {len(auths)}{extra}")
    sizes = Counter(all_groups)
    print(f"tamaño de grupo: mayor {max(sizes.values())}, "
          f"con más de 1 archivo: {sum(1 for v in sizes.values() if v > 1)}")

    # ---- selección de pares ----------------------------------------------
    # `off` deja todo. `unique` se queda sólo con las escenas que tienen las dos
    # clases, para que la única diferencia entre clases sea la edición y no la
    # escena. `balanced` además emite una auténtica por cada variante manipulada:
    # cada escena pesa lo mismo en ambas clases y el conjunto queda 1:1.
    emit = []                          # (key, key de la manipulada que lo duplica o None)
    if a.pairs == "off":
        emit = [(k, None) for k in all_keys]
    else:
        by_group = defaultdict(lambda: {"a": [], "t": []})
        for k, g in zip(all_keys, all_groups):
            by_group[g]["a" if k.startswith("auth:") else "t"].append(k)
        mixed = {g: v for g, v in by_group.items() if v["a"] and v["t"]}
        only_t = {g: v for g, v in by_group.items() if v["t"] and not v["a"]}
        only_a = {g: v for g, v in by_group.items() if v["a"] and not v["t"]}
        if not mixed:
            sys.exit(f"--pairs {a.pairs}: ningún grupo tiene auténtica y manipulada a la "
                     "vez; no queda nada que entrenar. Revisá el agrupamiento.")

        for g, v in mixed.items():
            for i, k in enumerate(v["t"]):
                emit.append((k, None))
                if a.pairs == "balanced":
                    # round-robin: si la escena tiene 2 auténticas se usan las dos
                    emit.append((v["a"][i % len(v["a"])], k))
            if a.pairs == "unique":
                emit += [(k, None) for k in v["a"]]

        n_t = sum(1 for k, _ in emit if not k.startswith("auth:"))
        n_a = sum(1 for k, _ in emit if k.startswith("auth:"))
        n_dup = sum(1 for _, owner in emit if owner is not None)
        print(f"\nPareado (--pairs {a.pairs}): {len(mixed)} grupos mixtos de {len(by_group)}")
        print(f"  descartados: {len(only_t)} grupos sólo manipulada "
              f"({sum(len(v['t']) for v in only_t.values())} img) | "
              f"{len(only_a)} sólo auténtica ({sum(len(v['a']) for v in only_a.values())} img)")
        print(f"  filas: {len(emit)} ({n_t} manipuladas + {n_a} auténticas) | "
              f"imágenes únicas: {len({k for k, _ in emit})}")
        if n_dup:
            uniq_a = len({k for k, _ in emit if k.startswith("auth:")})
            print(f"  auténticas: {uniq_a} únicas -> {n_a} filas "
                  f"({n_a / max(uniq_a, 1):.1f}x)")

    # ---- copiar y escribir manifest --------------------------------------
    if not a.dry_run:
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "masks").mkdir(parents=True, exist_ok=True)

    used = set()
    fracs, bad_size = [], []
    tpl = {}                           # key -> fila, una por archivo único
    path_of = dict(zip(all_keys, all_paths))
    group_of = dict(zip(all_keys, all_groups))

    def dst_name(stem: str, ext: str, auth: bool) -> str:
        # El stem tiene que ser único: también se usa como nombre de máscara.
        base = f"{a.prefix}{stem}{'__auth' if auth else ''}"
        name, n = base, 1
        while name.lower() in used:
            n += 1
            name = f"{base}_{n}"
        used.add(name.lower())
        return f"{name}{ext}"

    # Una pasada por archivo único: copiar y validar dos veces la misma auténtica
    # sería desperdiciar tiempo y ensuciar el conteo del área manipulada.
    for k in dict.fromkeys(k for k, _ in emit):
        auth = k.startswith("auth:")
        src = path_of[k]
        w, h = dims[k]

        # El nombre de archivo sale del pair key (identidad del archivo), no del
        # grupo: así cambiar la política de agrupamiento no recopia nada.
        name = dst_name(safe(k.removeprefix("auth:")), src.suffix.lower(), auth)
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

        tpl[k] = {
            "file": img_ref, "mask": mask_ref, "label": 0 if auth else 1,
            "group": group_of[k], "source": "authentic" if auth else "tampered",
            "width": w, "height": h,
        }

    # Expandir los pares. Si una manipulada no pasó la validación, su auténtica
    # duplicada también se cae; si no, el 1:1 se rompe.
    rows = []
    for k, owner in emit:
        if owner is not None and owner not in tpl:
            continue
        if k in tpl:
            rows.append(dict(tpl[k]))

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
    if a.pairs != "off":
        print(f"NOTA: --pairs {a.pairs} descarta las escenas de una sola clase "
              f"({sum(len(v['t']) for v in only_t.values())} manipuladas y "
              f"{sum(len(v['a']) for v in only_a.values())} auténticas); ver el resumen de arriba.")
    elif auths and matched < len(auths):
        if content:
            print(f"NOTA: {len(auths) - matched} auténticas no comparten grupo con ninguna "
                  "manipulada: su versión editada no está en el release, o la edición es tan "
                  "grande que no se reconoce. No hay nada que enlazar.")
        else:
            print(f"AVISO: {len(auths) - matched} auténticas quedaron sin escena compartida con "
                  "una manipulada. Si alguna es el original de una manipulada, puede haber fuga "
                  "entre train y validación: probá --group-by content.")

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
