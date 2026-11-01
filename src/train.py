"""Paso 3: entrenar el ViT clasificador sobre los mapas cacheados de TruFor.

Reanudable: guarda last.pt cada época y, si existe, retoma desde ahí. best.pt
son los pesos de la mejor época según AUC de validación. Todo se apoya en el
split por grupo guardado en --splits.

Ejemplo para la T4 de Colab:

    python src/train.py --manifest data/manifest.csv --packed packed/raw__s384 \
        --out runs/vit --epochs 150 --bs 16 --size 384
"""
import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import MapDataset, apply_splits, load_packed, make_splits, pos_weight, read_manifest
from model import build_model, n_params


# --------------------------------------------------------------------------- métricas
def metrics(y_true: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, f1_score, roc_auc_score)
    pred = (p >= thr).astype(int)
    out = {
        "acc": accuracy_score(y_true, pred),
        "bal_acc": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": float("nan"),
        "ap": float("nan"),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = roc_auc_score(y_true, p)
        out["ap"] = average_precision_score(y_true, p)
    return out


def best_threshold(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import f1_score
    thrs = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_true, (p >= t).astype(int), zero_division=0) for t in thrs]
    i = int(np.argmax(f1s))
    return float(thrs[i]), float(f1s[i])


@torch.no_grad()
def predict(model, loader, device, limit: int = 0) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for i, (x, y, score) in enumerate(loader):
        logits = model(x.to(device), score.to(device))
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.numpy().astype(int))
        if limit and i + 1 >= limit:
            break
    return np.concatenate(ys), np.concatenate(ps)


# --------------------------------------------------------------------------- EMA
class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.detach().clone() for k, v in sd.items()}


# --------------------------------------------------------------------------- utilidades
def save_atomic(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def param_groups(model, wd: float):
    """Weight decay solo en matrices; nada en normas, biases, tokens ni alpha."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or name.endswith("alpha") or "cls_token" in name
         or "pos_embed" in name else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


def main():
    ap = argparse.ArgumentParser(description="Entrena el ViT clasificador (TruFor maps)")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--packed", default="packed/raw__s384", help="base del .npy empaquetado")
    ap.add_argument("--splits", default="runs/splits.json")
    ap.add_argument("--out", default="runs/vit")
    ap.add_argument("--channels", default="maps+rgb", choices=["maps+rgb", "maps"])
    # arquitectura (tiene que coincidir con el tamaño empaquetado)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--attn-drop", type=float, default=0.0)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--no-score", action="store_true", help="no usar el score global de TruFor")
    ap.add_argument("--no-residual", action="store_true", help="no partir del baseline de TruFor")
    # optimización
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--min-lr", type=float, default=0.01, help="fracción del lr al final del coseno")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--pos-weight", default="auto", help="'auto', 'none' o un número")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999, help="0 para desactivar")
    ap.add_argument("--patience", type=int, default=20)
    # ejecución
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resplit", action="store_true", help="regenerar el split aunque exista")
    ap.add_argument("--fresh", action="store_true", help="ignorar last.pt y empezar de cero")
    ap.add_argument("--limit-steps", type=int, default=0, help="cortar cada época a N pasos (smoke test)")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not a.no_amp

    # ---- datos ------------------------------------------------------------
    rows = read_manifest(Path(a.manifest))
    splits_path = Path(a.splits)
    if splits_path.exists() and not a.resplit:
        splits = json.loads(splits_path.read_text())
    else:
        splits = make_splits(rows, seed=a.seed)
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits_path.write_text(json.dumps(splits, indent=2))
    train_rows, val_rows = apply_splits(rows, splits)

    _, meta, index = load_packed(a.packed)
    if int(meta["size"]) != a.size:
        sys.exit(f"El cache está empaquetado a {meta['size']}px pero --size es {a.size}. "
                 f"Usá --size {meta['size']} o regenerá el cache.")
    in_chans = 5 if a.channels == "maps+rgb" else 2

    ds_kw = dict(packed=a.packed, channels=a.channels)
    train_ds = MapDataset(rows=train_rows, train=True, **ds_kw)
    val_ds = MapDataset(rows=val_rows, train=False, **ds_kw)
    dl = lambda ds, shuffle: DataLoader(
        ds, batch_size=a.bs, shuffle=shuffle, num_workers=a.workers,
        pin_memory=device.type == "cuda", drop_last=shuffle and len(ds) > a.bs,
        persistent_workers=a.workers > 0)
    train_dl, val_dl = dl(train_ds, True), dl(val_ds, False)
    print(f"train={len(train_ds)} val={len(val_ds)} | {meta.get('variant', '?')} "
          f"| {meta['size']}px | canales={a.channels} | device={device} amp={amp}")

    # baseline: qué AUC tiene el score de TruFor sobre esta misma validación
    val_y = np.array([r["label"] for r in val_ds.rows])
    val_s = np.array([meta["scores"][index[Path(r["file"]).stem]] for r in val_ds.rows])
    base = metrics(val_y, val_s)
    print(f"baseline TruFor (val): AUC {base['auc']:.3f} | AP {base['ap']:.3f} | "
          f"F1@0.5 {base['f1']:.3f}")

    # ---- modelo -----------------------------------------------------------
    cfg = dict(in_chans=in_chans, img_size=a.size, patch=a.patch, dim=a.dim, depth=a.depth,
               heads=a.heads, drop=a.drop, attn_drop=a.attn_drop, drop_path=a.drop_path,
               use_score=not a.no_score, residual=not a.no_residual)
    model = build_model(**cfg).to(device)
    print(f"ViT desde cero: {n_params(model):.1f}M parámetros "
          f"({a.depth} bloques, dim {a.dim}, patch {a.patch})")

    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.999))

    def lr_lambda(ep):
        if ep < a.warmup:
            return (ep + 1) / max(1, a.warmup)
        prog = (ep - a.warmup) / max(1, a.epochs - a.warmup)
        return a.min_lr + (1 - a.min_lr) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    if a.pos_weight == "auto":
        pw = pos_weight(train_rows)
    elif a.pos_weight == "none":
        pw = 1.0
    else:
        pw = float(a.pos_weight)
    pw_t = torch.tensor(pw, device=device)
    print(f"pos_weight={pw:.2f}")

    ema = EMA(model, a.ema) if a.ema > 0 else None
    eval_model = copy.deepcopy(model).to(device) if ema else model

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg, **vars(a), "pos_weight_used": pw}, indent=2))
    log_path = out / "log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,lr,train_loss,val_loss,acc,bal_acc,f1,auc,ap,thr,f1_best,secs\n")

    start, best, bad = 0, -1.0, 0
    last = out / "last.pt"
    if last.exists() and not a.fresh:
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        if ema and ck.get("ema") is not None:
            ema.load_state_dict(ck["ema"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck.get("bad", 0)
        print(f"Reanudando desde la época {start + 1} (mejor AUC {best:.4f})")

    # ---- loop -------------------------------------------------------------
    t0 = time.monotonic()
    for ep in range(start, a.epochs):
        model.train()
        total, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y, score) in enumerate(train_dl):
            x, y, score = x.to(device), y.to(device), score.to(device)
            y_s = y * (1 - a.label_smoothing) + a.label_smoothing / 2
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = F.binary_cross_entropy_with_logits(model(x, score), y_s, pos_weight=pw_t)
                loss = loss / a.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % a.grad_accum == 0 or i + 1 == len(train_dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)
            total += loss.item() * a.grad_accum
            n += 1
            if a.limit_steps and i + 1 >= a.limit_steps:
                break
        sched.step()

        if ema:
            eval_model.load_state_dict(ema.state_dict())
        y_true, p = predict(eval_model, val_dl, device, a.limit_steps)
        m = metrics(y_true, p)
        thr, f1b = best_threshold(y_true, p)
        val_loss = float(F.binary_cross_entropy(
            torch.tensor(p).clamp(1e-6, 1 - 1e-6), torch.tensor(y_true, dtype=torch.float32)).item())
        secs = time.monotonic() - t0
        t0 = time.monotonic()

        print(f"ep {ep + 1:03d}/{a.epochs} | lr {sched.get_last_lr()[0]:.1e} | "
              f"loss {total / max(n, 1):.3f} | val {val_loss:.3f} | "
              f"acc {m['acc']:.3f} bal {m['bal_acc']:.3f} F1 {m['f1']:.3f} "
              f"AUC {m['auc']:.3f} AP {m['ap']:.3f} | F1* {f1b:.3f}@{thr:.2f} | {secs:.0f}s")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, f"{sched.get_last_lr()[0]:.6g}", f"{total / max(n, 1):.4f}",
                                    f"{val_loss:.4f}", f"{m['acc']:.4f}", f"{m['bal_acc']:.4f}",
                                    f"{m['f1']:.4f}", f"{m['auc']:.4f}", f"{m['ap']:.4f}",
                                    f"{thr:.2f}", f"{f1b:.4f}", f"{secs:.0f}"])

        improved = m["auc"] > best
        if improved:
            best = m["auc"]
            bad = 0
            save_atomic({"model": eval_model.state_dict(), "cfg": cfg, "epoch": ep,
                         "metrics": m, "thr": thr}, out / "best.pt")
        else:
            bad += 1
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "ema": ema.state_dict() if ema else None,
                     "epoch": ep, "best": best, "bad": bad, "cfg": cfg, "args": vars(a)},
                    out / "last.pt")

        if a.patience and bad >= a.patience:
            print(f"Early stopping: {a.patience} épocas sin mejorar AUC.")
            break

    print(f"\nListo. Mejor AUC de validación: {best:.4f} | checkpoints en {out}/")


if __name__ == "__main__":
    main()
