"""Barrido de preprocesamientos de TruFor sobre imágenes etiquetadas por carpeta.

Estructura esperada (desde la raíz del repositorio)::

    pruebas/
      originales/   # imágenes auténticas
      modificadas/  # imágenes manipuladas

El programa compara todas las configuraciones permitidas por --max-steps y
guarda métricas agregadas y predicciones por imagen en runs/.
"""
import argparse
import csv
import itertools
import math
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import detect


# Cada valor de estos conjuntos representa una variante de un filtro.
# El generador considera también la opción de no aplicar cada filtro.
FILTERS = [
    ("suavizado", [
        {"metodo": "gaussiano", "sigma": 0.5},
        {"metodo": "gaussiano", "sigma": 1.0},
        {"metodo": "mediana", "ksize": 3},
        {"metodo": "bilateral", "d": 5, "sigma_color": 25, "sigma_space": 5},
        {"metodo": "bilateral", "d": 9, "sigma_color": 50, "sigma_space": 9},
        {"metodo": "nlmeans", "h": 3},
        {"metodo": "nlmeans", "h": 7},
    ]),
    ("brillo", [
        {"factor": 0.8, "offset": 0, "gamma": 1.0},
        {"factor": 1.2, "offset": 0, "gamma": 1.0},
        {"factor": 1.0, "offset": 0, "gamma": 0.8},
        {"factor": 1.0, "offset": 0, "gamma": 1.2},
    ]),
    ("blancos", [
        {"umbral": 180, "fuerza": 0.5},
        {"umbral": 200, "fuerza": 0.5},
        {"umbral": 220, "fuerza": 0.5},
    ]),
    ("clahe", [
        {"clip": 1.5, "grilla": 8},
        {"clip": 2.0, "grilla": 8},
        {"clip": 3.0, "grilla": 8},
        {"clip": 2.0, "grilla": 16},
    ]),
    ("realce", [
        {"sigma": 0.5, "cantidad": 0.5},
        {"sigma": 1.0, "cantidad": 0.5},
        {"sigma": 1.0, "cantidad": 1.0},
        {"sigma": 2.0, "cantidad": 1.0},
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


def confusion_metrics(labels, scores, threshold):
    predictions = [int(score >= threshold) for score in scores]
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
    return max(
        evaluated,
        key=lambda item: (
            item[1]["balanced_accuracy"],
            item[1]["f1"],
            -abs(item[0] - 0.5),
        ),
    )


def run(root, output_dir, max_steps, device_arg, max_size):
    samples = collect_labeled_images(root)
    configs = generate_configs(max_steps)
    labels = [label for _, label in samples]
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
    print("La métrica principal es balanced_accuracy; se informa también el resultado con umbral 0.5.\n")

    summary_rows = []
    detail_rows = []
    for config_number, (steps, description) in enumerate(configs, start=1):
        scores = []
        print(f"[{config_number}/{len(configs)}] {description}", flush=True)
        for path, label in samples:
            rgb, _ = detect.preprocess(path, pasos=with_size_limit(steps, fit_sizes[path]))
            score, _heatmap, _conf_map = detect.run_model(model, rgb, device, path.name)
            scores.append(score)
            detail_rows.append({
                "config_id": config_number,
                "config": description,
                "file": str(path),
                "label": "MANIPULADA" if label else "AUTENTICA",
                "score": score,
                "verdict_0_5": "MANIPULADA" if score >= 0.5 else "AUTENTICA",
            })

        fixed_metrics = confusion_metrics(labels, scores, 0.5)
        threshold, tuned_metrics = best_threshold(labels, scores)
        summary_rows.append({
            "config_id": config_number,
            "config": description,
            "threshold_best": threshold,
            **{f"fixed_{key}": value for key, value in fixed_metrics.items()},
            **{f"tuned_{key}": value for key, value in tuned_metrics.items()},
        })

    summary_rows.sort(
        key=lambda row: (
            row["tuned_balanced_accuracy"],
            row["tuned_f1"],
            row["fixed_balanced_accuracy"],
        ),
        reverse=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "preproc_test_resumen.csv"
    details_path = output_dir / "preproc_test_detalle.csv"
    write_csv(summary_path, summary_rows)
    write_csv(details_path, detail_rows)

    print("\nMejores configuraciones (ordenadas por balanced accuracy con umbral ajustado):")
    for row in summary_rows[:10]:
        print(
            f"  {row['tuned_balanced_accuracy']:.3f}  "
            f"F1={row['tuned_f1']:.3f}  umbral={row['threshold_best']:.4f}  "
            f"{row['config'] or 'sin_preproc'}"
        )
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
    parser.add_argument("--max-steps", type=int, choices=range(0, len(FILTERS) + 1), default=2,
                        help="máximo de filtros combinados por configuración (default: 2)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-size", type=int, default=1000,
                        help="lado máximo en px; las imágenes más grandes se reducen manteniendo "
                             "la relación de aspecto (0 = no reducir, default: 1000)")
    args = parser.parse_args()
    run(args.root, args.out, args.max_steps, args.device, args.max_size)


if __name__ == "__main__":
    main()
