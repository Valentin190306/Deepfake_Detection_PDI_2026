"""Barrido de preprocesamientos de TruFor sobre imágenes etiquetadas por carpeta.

Estructura esperada (desde la raíz del repositorio)::

    pruebas/
      originales/   # imágenes auténticas
      modificadas/  # imágenes manipuladas

El programa compara todas las configuraciones permitidas por --max-steps y
guarda métricas agregadas y predicciones por imagen en runs/.

Las métricas "tuned" eligen el umbral con las mismas imágenes con las que se
miden, así que son optimistas. Las "cv" lo eligen por validación cruzada: cada
imagen se clasifica con un umbral que se buscó sin mirarla. Al final, una
validación cruzada anidada estima cuánto rinde el procedimiento completo
(elegir la mejor configuración y su umbral) sobre imágenes que no participaron
de la elección.
"""
import argparse
import csv
import itertools
import math
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import detect


# Pasa bajo, primera derivada y segunda derivada. Las combinaciones respetan
# este orden, así que incluyen gauss -> laplaciano (LoG), sobel -> laplaciano
# (2da derivada sobre la 1ra) y las tres juntas. El generador considera también
# la opción de no aplicar cada filtro.
# Cada configuración extra es una oportunidad más de ganar por casualidad:
# conviene que la lista sea corta.
FILTERS = [
    ("suavizado", [
        {"metodo": "gaussiano", "sigma": 0.5},
        {"metodo": "gaussiano", "sigma": 1.0},
        {"metodo": "gaussiano", "sigma": 2.0},
    ]),
    ("sobel", [
        {"ksize": 3},
    ]),
    ("laplaciano", [
        {"modo": "residuo"},
        {"modo": "realce", "cantidad": 0.5},
        {"modo": "realce", "cantidad": 1.0},
    ]),
]


def collect_labeled_images(root):
    """Devuelve (ruta, etiqueta) con etiqueta 0=original y 1=modificada."""
    folders = (("originales", 0), ("modificadas", 1))
    samples = []
    for folder_name, label in folders:
        folder = root / folder_name
        if not folder.is_dir():
            raise SystemExit(f"No existe la carpeta requerida: {folder}")
        images = detect.collect_images(str(folder))
        if not images:
            raise SystemExit(f"No encontré imágenes en: {folder}")
        samples.extend((path, label) for path in images)

    if len({label for _, label in samples}) < 2:
        raise SystemExit("Se necesitan imágenes en originales/ y modificadas/.")
    return sorted(samples, key=lambda sample: str(sample[0]))


def fit_size(path, max_size):
    """Devuelve (ancho, alto) para que la imagen entre en max_size x max_size
    manteniendo la relación de aspecto, o None si ya entra (o max_size es 0)."""
    with Image.open(path) as img:
        width, height = img.size
    if not max_size or max(width, height) <= max_size:
        return None
    scale = max_size / max(width, height)
    return (max(1, round(width * scale)), max(1, round(height * scale)))


def with_size_limit(steps, size):
    """Antepone un resize que respeta la relación de aspecto si hace falta."""
    if size is None:
        return steps
    return [("resize", True, {"size": size}), *steps]


def generate_configs(max_steps):
    """Enumera combinaciones de filtros (en el orden declarado arriba)."""
    configs = []
    choices = [[None, *variants] for _, variants in FILTERS]
    for selection in itertools.product(*choices):
        active = [(FILTERS[i][0], params) for i, params in enumerate(selection) if params]
        if len(active) <= max_steps:
            steps = [(name, True, params) for name, params in active]
            configs.append((steps, detect.describe_steps(steps)))
    return configs


def classification_metrics(labels, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels)
    balanced_accuracy = (recall + specificity) / 2
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def confusion_metrics(labels, scores, threshold):
    return classification_metrics(labels, [int(score >= threshold) for score in scores])


def selection_key(threshold, metrics):
    """Criterio para elegir umbral (y configuración): balanced accuracy, después
    F1, después el umbral más cercano a 0.5."""
    return (metrics["balanced_accuracy"], metrics["f1"], -abs(threshold - 0.5))


def best_threshold(labels, scores):
    """Busca el umbral con mejor balanced accuracy; desempata por F1 y 0.5."""
    unique_scores = sorted(set(scores))
    candidates = [0.0]
    candidates.extend((a + b) / 2 for a, b in zip(unique_scores, unique_scores[1:]))
    candidates.append(math.nextafter(unique_scores[-1], math.inf))

    evaluated = [
        (threshold, confusion_metrics(labels, scores, threshold))
        for threshold in candidates
    ]
    return max(evaluated, key=lambda item: selection_key(*item))


def stratified_folds(labels, n_folds, seed):
    """Asigna cada imagen a un fold (0..n_folds-1) repartiendo cada clase parejo."""
    rng = random.Random(seed)
    folds = [0] * len(labels)
    position = 0
    for label in (0, 1):
        indices = [i for i, y in enumerate(labels) if y == label]
        rng.shuffle(indices)
        for index in indices:
            folds[index] = position % n_folds
            position += 1
    return folds


def cross_validate(labels, scores_by_config, folds, config_ids):
    """Para cada fold elige, solo con las imágenes de los otros folds, la mejor
    configuración de config_ids y su umbral, y con eso clasifica las del fold.
    Devuelve las métricas de esas predicciones y la configuración y el umbral
    elegidos en cada fold."""
    predictions = [0] * len(labels)
    choices = []
    for fold in sorted(set(folds)):
        train = [i for i, f in enumerate(folds) if f != fold]
        train_labels = [labels[i] for i in train]
        candidates = []
        for config_id in config_ids:
            scores = scores_by_config[config_id]
            threshold, metrics = best_threshold(train_labels, [scores[i] for i in train])
            candidates.append((selection_key(threshold, metrics), config_id, threshold))
        _, config_id, threshold = max(candidates, key=lambda candidate: candidate[0])
        choices.append((config_id, threshold))
        for i, f in enumerate(folds):
            if f == fold:
                predictions[i] = int(scores_by_config[config_id][i] >= threshold)
    return classification_metrics(labels, predictions), choices


def run(root, output_dir, max_steps, device_arg, max_size, n_folds, seed):
    samples = collect_labeled_images(root)
    configs = generate_configs(max_steps)
    labels = [label for _, label in samples]
    smallest_class = min(labels.count(0), labels.count(1))
    if not 2 <= n_folds <= smallest_class:
        raise SystemExit(f"--folds tiene que estar entre 2 y {smallest_class} (imágenes de la clase más chica).")
    folds = stratified_folds(labels, n_folds, seed)
    fit_sizes = {path: fit_size(path, max_size) for path, _ in samples}
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device_arg == "auto" else device_arg
    )

    model, weights = detect.load_model(device)
    print(
        f"{len(samples)} imágenes ({sum(labels)} modificadas, "
        f"{len(labels) - sum(labels)} originales) x {len(configs)} configuraciones"
    )
    print(f"Modelo: {weights.name} | dispositivo: {device} | máximo de filtros: {max_steps}")
    reduced = sum(size is not None for size in fit_sizes.values())
    if reduced:
        print(f"{reduced} imágenes superan {max_size}x{max_size} px y se reducen manteniendo la relación de aspecto.")
    print(f"Ranking por balanced_accuracy con umbral elegido por validación cruzada ({n_folds} folds).\n")

    summary_rows = []
    detail_rows = []
    scores_by_config = {}
    descriptions = {}
    for config_number, (steps, description) in enumerate(configs, start=1):
        scores = []
        print(f"[{config_number}/{len(configs)}] {description}", flush=True)
        for (path, label), fold in zip(samples, folds):
            rgb, _ = detect.preprocess(path, pasos=with_size_limit(steps, fit_sizes[path]))
            score, _heatmap, _conf_map = detect.run_model(model, rgb, device, path.name)
            scores.append(score)
            detail_rows.append({
                "config_id": config_number,
                "config": description,
                "file": str(path),
                "fold": fold + 1,
                "label": "MANIPULADA" if label else "AUTENTICA",
                "score": score,
                "verdict_0_5": "MANIPULADA" if score >= 0.5 else "AUTENTICA",
            })
        scores_by_config[config_number] = scores
        descriptions[config_number] = description

        fixed_metrics = confusion_metrics(labels, scores, 0.5)
        threshold, tuned_metrics = best_threshold(labels, scores)
        cv_metrics, _ = cross_validate(labels, scores_by_config, folds, [config_number])
        summary_rows.append({
            "config_id": config_number,
            "config": description,
            "threshold_best": threshold,
            **{f"fixed_{key}": value for key, value in fixed_metrics.items()},
            **{f"tuned_{key}": value for key, value in tuned_metrics.items()},
            **{f"cv_{key}": value for key, value in cv_metrics.items()},
        })

    summary_rows.sort(
        key=lambda row: (
            row["cv_balanced_accuracy"],
            row["cv_f1"],
            row["tuned_balanced_accuracy"],
        ),
        reverse=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "preproc_test_resumen.csv"
    details_path = output_dir / "preproc_test_detalle.csv"
    write_csv(summary_path, summary_rows)
    write_csv(details_path, detail_rows)

    print("\nMejores configuraciones (ordenadas por balanced accuracy con validación cruzada):")
    for row in summary_rows[:10]:
        print(
            f"  cv={row['cv_balanced_accuracy']:.3f}  tuned={row['tuned_balanced_accuracy']:.3f}  "
            f"umbral={row['threshold_best']:.4f}  [{row['config_id']}] {row['config']}"
        )

    baseline_id = next(config_id for config_id, (steps, _) in enumerate(configs, start=1) if not steps)
    baseline = next(row for row in summary_rows if row["config_id"] == baseline_id)
    nested_metrics, choices = cross_validate(labels, scores_by_config, folds, list(scores_by_config))
    print(f"\nValidación cruzada anidada: en cada fold se elige configuración y umbral sin mirar sus imágenes.")
    print(f"  elegir la mejor configuración + umbral -> balanced accuracy {nested_metrics['balanced_accuracy']:.3f}")
    print(f"  sin preprocesamiento (solo umbral)     -> balanced accuracy {baseline['cv_balanced_accuracy']:.3f}")
    print("  Elegida en cada fold (si cambia mucho entre folds, la ganadora es ruido):")
    for fold, (config_id, threshold) in enumerate(choices, start=1):
        print(f"    fold {fold}: umbral={threshold:.4f}  [{config_id}] {descriptions[config_id]}")
    print(f"\nResumen: {summary_path}\nPredicción por imagen: {details_path}")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compara por fuerza bruta filtros de entrada para TruFor."
    )
    parser.add_argument("--root", type=Path, default=Path("pruebas"),
                        help="carpeta que contiene originales/ y modificadas/ (default: pruebas)")
    parser.add_argument("--out", type=Path, default=Path("runs"),
                        help="carpeta donde guardar los CSV (default: runs)")
    parser.add_argument("--max-steps", type=int, choices=range(0, len(FILTERS) + 1), default=len(FILTERS),
                        help=f"máximo de filtros combinados por configuración (default: {len(FILTERS)})")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-size", type=int, default=1000,
                        help="lado máximo en px; las imágenes más grandes se reducen manteniendo "
                             "la relación de aspecto (0 = no reducir, default: 1000)")
    parser.add_argument("--folds", type=int, default=5,
                        help="cantidad de folds de la validación cruzada (default: 5)")
    parser.add_argument("--seed", type=int, default=0,
                        help="semilla para repartir las imágenes en folds (default: 0)")
    args = parser.parse_args()
    run(args.root, args.out, args.max_steps, args.device, args.max_size, args.folds, args.seed)


if __name__ == "__main__":
    main()
