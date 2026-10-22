"""
train.py - Entrenamiento mínimo para clasificar patches "authentic" vs "manipulated".

Estructura de datos esperada (patches ya extraídos, ideal en PNG sin pérdida):

    data/
    ├── train/
    │   ├── authentic/      *.png
    │   └── manipulated/    *.png
    ├── val/
    │   ├── authentic/
    │   └── manipulated/
    └── test/               (solo se usa al final, con --test)
        ├── authentic/
        └── manipulated/

Uso típico:

    # 1) Prueba de cordura: sobreajustar 32 imágenes (debe llegar a loss ~ 0)
    python train.py --data data --overfit 32 --epochs 30 --batch-size 8

    # 2) Baseline real con ResNet18
    python train.py --data data --model resnet18 --epochs 10 --out runs/resnet18

    # 3) Fine-tuning de ViT (patches de 224x224)
    python train.py --data data --model vit_small_patch16_224 --lr 5e-5 --out runs/vit_s

    # 4) Retomar si se cortó la sesión (Colab) desde runs/<nombre>/last.pt
    python train.py --data data --model resnet18 --out runs/resnet18 --resume

    # 5) Evaluación FINAL en test con el mejor checkpoint (hacerlo una sola vez,
    #    cuando ya cerraste todas las decisiones de modelo e hiperparámetros)
    python train.py --data data --model resnet18 --out runs/resnet18 --test

Dependencias: torch, torchvision, timm, scikit-learn
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def set_seed(seed):
    """Fija las semillas para que un experimento sea repetible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform(augment, mean, std):
    """
    Transformaciones de las imágenes. OJO: no hay Resize ni blur ni JPEG.
    Cualquiera de esas operaciones destruye las huellas de la manipulación.
    Solo usamos flips (no alteran el ruido local) y normalización.
    """
    ops = []
    if augment:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    ops += [
        transforms.ToTensor(),  # PIL -> tensor, escala a [0, 1]
        # Cada modelo preentrenado espera su propia normalización (mean/std);
        # la obtenemos de timm según el modelo elegido.
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(ops)


def save_checkpoint(path, model, optimizer, scaler, epoch, best_score, args):
    """
    Guarda el estado completo del entrenamiento. Se escribe primero a un
    archivo temporal y luego se renombra, así, si la sesión se corta justo
    mientras guarda, no queda un checkpoint corrupto.
    """
    tmp = Path(str(path) + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),          # los pesos
            "optimizer": optimizer.state_dict(),  # estado interno de AdamW
            "scaler": scaler.state_dict(),        # estado de precisión mixta
            "epoch": epoch,                       # última época completada
            "best_score": float(best_score),      # mejor métrica de validación
            "args": vars(args),                   # configuración usada
        },
        tmp,
    )
    tmp.replace(path)


# --------------------------------------------------------------------------
# Evaluación
# --------------------------------------------------------------------------
@torch.no_grad()  # no calculamos gradientes: más rápido y menos memoria
def evaluate(model, loader, criterion, device):
    model.eval()  # desactiva dropout y usa estadísticas fijas en batchnorm
    total_loss, n = 0.0, 0
    all_probs, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        n += len(y)
        # Probabilidad de la clase 1 ("manipulated")
        all_probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        all_labels.append(y.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs >= 0.5).astype(int)

    # Matriz de confusión, con "manipulated" como clase positiva
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # El AUC necesita ejemplos de ambas clases
    auc = roc_auc_score(labels, probs) if len(set(labels.tolist())) == 2 else float("nan")

    return {
        "loss": total_loss / n,
        "acc": float((preds == labels).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": float(auc),
    }


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="carpeta con train/ y val/")
    p.add_argument("--model", default="resnet18", help="nombre de timm: resnet18, vit_small_patch16_224, ...")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate (bajo para fine-tuning)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=2, help="procesos que cargan datos (Colab tiene pocos núcleos)")
    p.add_argument("--out", default="runs/exp", help="carpeta de salida (checkpoints e historial)")
    p.add_argument("--resume", action="store_true", help="retomar desde last.pt si existe")
    p.add_argument("--test", action="store_true", help="evaluar best.pt en data/test y salir (no entrena)")
    p.add_argument("--overfit", type=int, default=0, help="si N>0: entrenar y evaluar solo con N imágenes (prueba de cordura)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ---- 0. Modelo (se crea primero para conocer su normalización) --------
    # pretrained=True: arranca con pesos preentrenados (fine-tuning) y reemplaza
    # la última capa por una de 2 salidas (auténtico / manipulado).
    model = timm.create_model(args.model, pretrained=True, num_classes=2).to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean, std = cfg["mean"], cfg["std"]
    print(f"Modelo: {args.model} | normalización mean={mean} std={std}")

    # ---- 1. Datos ---------------------------------------------------------
    # ImageFolder asigna las etiquetas según el nombre de las subcarpetas,
    # en orden alfabético: authentic=0, manipulated=1.
    in_overfit = args.overfit > 0
    train_ds = datasets.ImageFolder(Path(args.data) / "train", build_transform(not in_overfit, mean, std))
    val_ds = datasets.ImageFolder(Path(args.data) / "val", build_transform(False, mean, std))
    expected = {"authentic": 0, "manipulated": 1}
    assert train_ds.class_to_idx == expected, (
        f"Las carpetas deben llamarse authentic y manipulated. Encontré: {train_ds.class_to_idx}"
    )

    # Pesos por clase: si hay muchos más patches auténticos que manipulados,
    # el error en la clase minoritaria cuesta más. Evita el "todo auténtico".
    counts = np.bincount(train_ds.targets, minlength=2)
    class_weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32).to(device)
    print(f"Patches por clase en train: {dict(zip(train_ds.classes, counts.tolist()))}")

    if in_overfit:
        # Muestreo aleatorio (ImageFolder ordena por clase, no sirve tomar los primeros N).
        # Entrenamos y evaluamos sobre las MISMAS imágenes y sin augmentation:
        # si el modelo no logra memorizarlas, hay un bug.
        idx = random.sample(range(len(train_ds)), min(args.overfit, len(train_ds)))
        train_ds = Subset(train_ds, idx)
        val_ds = train_ds
        print(f"MODO OVERFIT: usando {len(idx)} imágenes para train y val")

    pin = device.type == "cuda"
    # DataLoader: agrupa muestras en lotes y (en train) las mezcla cada época.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)

    # ---- 2. Modelo, pérdida, optimizador ---------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Modo test: evaluación final y salir ------------------------------
    if args.test:
        test_ds = datasets.ImageFolder(Path(args.data) / "test", build_transform(False, mean, std))
        assert test_ds.class_to_idx == expected, f"Carpetas inesperadas en test: {test_ds.class_to_idx}"
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=pin)
        # Siempre el MEJOR checkpoint (elegido por validación), no el último.
        ckpt = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        m = evaluate(model, test_loader, criterion, device)
        print("RESULTADOS EN TEST: " + " | ".join(f"{k} {v:.4f}" for k, v in m.items()))
        (out_dir / "test_results.json").write_text(json.dumps(m, indent=2))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Precisión mixta: acelera y ahorra memoria, solo en GPU.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- 3. Retomar si corresponde ---------------------------------------
    start_epoch, best_score = 0, -1.0
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        best_score = ckpt["best_score"]
        print(f"Retomando desde la época {start_epoch} (mejor score hasta ahora: {best_score:.4f})")

    # ---- 4. Ciclo de entrenamiento ---------------------------------------
    history = out_dir / "history.csv"
    if not history.exists():
        history.write_text("epoch,train_loss,val_loss,val_acc,val_precision,val_recall,val_f1,val_auc\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()  # modo entrenamiento (activa dropout, batchnorm actualiza estadísticas)
        running_loss, n = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()                       # borra los gradientes del lote anterior
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)                       # forward: predicciones
                loss = criterion(logits, y)             # qué tan mal predijo
            scaler.scale(loss).backward()               # backward: calcula gradientes
            scaler.step(optimizer)                      # actualiza los pesos
            scaler.update()

            running_loss += loss.item() * len(y)
            n += len(y)

        train_loss = running_loss / n
        m = evaluate(model, val_loader, criterion, device)

        # Métrica para decidir cuál es el "mejor" modelo: AUC (no depende del umbral).
        score = m["auc"] if not np.isnan(m["auc"]) else m["f1"]
        is_best = score > best_score
        if is_best:
            best_score = score

        # ---- 5. Checkpoints ----
        save_checkpoint(last_path, model, optimizer, scaler, epoch + 1, best_score, args)
        if is_best:
            save_checkpoint(best_path, model, optimizer, scaler, epoch + 1, best_score, args)

        with open(history, "a") as f:
            f.write(f"{epoch + 1},{train_loss:.5f},{m['loss']:.5f},{m['acc']:.4f},"
                    f"{m['precision']:.4f},{m['recall']:.4f},{m['f1']:.4f},{m['auc']:.4f}\n")

        print(f"[{epoch + 1}/{args.epochs}] {time.time() - t0:.0f}s | "
              f"train_loss {train_loss:.4f} | val_loss {m['loss']:.4f} | "
              f"acc {m['acc']:.3f} | P {m['precision']:.3f} R {m['recall']:.3f} "
              f"F1 {m['f1']:.3f} | AUC {m['auc']:.3f}" + ("  <- mejor" if is_best else ""))

    print(f"Listo. Mejor score de validación: {best_score:.4f}. Checkpoints en {out_dir}/")


if __name__ == "__main__":
    main()