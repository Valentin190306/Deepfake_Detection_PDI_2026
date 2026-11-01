"""Paso 2: entrenar un ViT PROPIO (desde cero) y compararlo contra TruFor puro (baseline).

Reanudable: guarda last.pt en cada epoca y, si existe, retoma desde ahi.
best.pt = pesos de la mejor epoca (F1 de validacion). config.json = hiperparametros del modelo.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ForensicsDataset, list_images, split
from model import ViTRefiner


def f1_iou(pred, gt, eps=1e-6):
    tp = (pred * gt).sum((1, 2, 3))
    fp = (pred * (1 - gt)).sum((1, 2, 3))
    fn = ((1 - pred) * gt).sum((1, 2, 3))
    return 2 * tp / (2 * tp + fp + fn + eps), tp / (tp + fp + fn + eps)


def loss_fn(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def evaluate(model, dl, device):
    """F1/IoU por imagen sobre imagenes con mascara no vacia. ViT vs TruFor (umbral 0.5)."""
    model.eval()
    acc = {"vit": [], "trufor": []}
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        keep = y.sum((1, 2, 3)) > 0
        if keep.sum() == 0:
            continue
        preds = {"vit": (torch.sigmoid(model(x)) > 0.5).float(),
                 "trufor": (x[:, 3:4] > 0.5).float()}   # canal 3 = mapa TruFor de la 1ra variante
        for k, pr in preds.items():
            f1, iou = f1_iou(pr[keep], y[keep])
            acc[k].append(torch.stack([f1, iou], 1).cpu())
    return {k: (torch.cat(v).mean(0).tolist() if v else [float('nan')] * 2) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--variants", nargs="+", default=["raw"])
    ap.add_argument("--out", default="runs/vit", help="carpeta de checkpoints (en Colab: dentro de Drive)")
    ap.add_argument("--test-prefix", default="",
                    help="imagenes cuyo nombre empieza asi quedan FUERA de train/val y se evaluan aparte (ej: own_)")
    # Arquitectura
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--residual", action="store_true", help="partir del baseline TruFor y aprender una correccion")
    # Optimizacion (desde cero: mas epocas, warmup y clipping)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = list_images(a.root)
    test_p = [p for p in paths if a.test_prefix and p.stem.startswith(a.test_prefix)]
    if a.test_prefix and not test_p:
        print(f"AVISO: ninguna imagen empieza con '{a.test_prefix}'; no hay test externo.")
    held = set(test_p)
    train_p, val_p = split([p for p in paths if p not in held])
    print(f"train={len(train_p)} val={len(val_p)} test_externo={len(test_p)} dispositivo={device}")

    mk = lambda ps, tr: DataLoader(
        ForensicsDataset(a.root, a.cache, ps, a.variants, a.size, tr),
        batch_size=a.bs, shuffle=tr, num_workers=2, drop_last=tr)
    train_dl, val_dl = mk(train_p, True), mk(val_p, False)
    test_dl = mk(test_p, False) if test_p else None

    model = ViTRefiner(3 + 2 * len(a.variants), a.size, a.patch, a.dim, a.depth, a.heads,
                       residual=a.residual).to(device)
    print(f"Parametros: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M (desde cero)")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    # warmup lineal + coseno, por epoca
    lam = lambda e: min(1.0, (e + 1) / max(1, a.warmup)) * 0.5 * (1 + math.cos(math.pi * e / a.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    start, best = 0, 0.0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"Reanudando desde la epoca {start + 1}")

    for ep in range(start, a.epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()

        r = evaluate(model, val_dl, device)
        print(f"ep {ep + 1:03d} | loss {loss.item():.3f} | "
              f"ViT F1 {r['vit'][0]:.3f} IoU {r['vit'][1]:.3f} | "
              f"TruFor F1 {r['trufor'][0]:.3f} IoU {r['trufor'][1]:.3f}")
        if test_dl is not None:
            rt = evaluate(model, test_dl, device)
            print(f"        test externo | ViT F1 {rt['vit'][0]:.3f} IoU {rt['vit'][1]:.3f} | "
                  f"TruFor F1 {rt['trufor'][0]:.3f} IoU {rt['trufor'][1]:.3f}")
        if r["vit"][0] > best:
            best = r["vit"][0]
            torch.save(model.state_dict(), out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best}, out / "last.tmp")
        (out / "last.tmp").replace(out / "last.pt")   # atomico


if __name__ == "__main__":
    main()
