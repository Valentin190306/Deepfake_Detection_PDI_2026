"""Evalúa un checkpoint del ViT: métricas, matriz de confusión, curvas y galería.

Sirve tanto para la validación de IMD2020 como para un test externo: si el
manifest es de otro dataset, usá --split all (no hay split guardado para él).

    python src/eval.py --manifest data/manifest.csv --packed packed/raw__s384 \
        --ckpt runs/vit/best.pt --out runs/eval --split val

    # test externo (otro manifest + su propio cache empaquetado)
    python src/eval.py --manifest data_own/manifest.csv --packed packed/own \
        --ckpt runs/vit/best.pt --out runs/eval_own --split all
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MapDataset, apply_splits, dedupe_rows, load_packed, read_manifest
from model import build_model
from train import metrics


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y, score in loader:
        ps.append(torch.sigmoid(model(x.to(device), score.to(device))).float().cpu().numpy())
        ys.append(y.numpy().astype(int))
    return np.concatenate(ys), np.concatenate(ps)


def plots(y, p, thr, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                                 roc_curve)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
    ax[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[0].set_xticks([0, 1], ["pred 0", "pred 1"])
    ax[0].set_yticks([0, 1], ["real 0", "real 1"])
    ax[0].set_title(f"Matriz de confusión (thr {thr:.2f})")

    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, p)
        ax[1].plot(fpr, tpr, label=f"AUC {auc(fpr, tpr):.3f}")
        ax[1].plot([0, 1], [0, 1], "--", color="gray")
        ax[1].set_xlabel("falsos positivos")
        ax[1].set_ylabel("verdaderos positivos")
        ax[1].set_title("ROC")
        ax[1].legend()

        pre, rec, _ = precision_recall_curve(y, p)
        ax[2].plot(rec, pre)
        ax[2].set_xlabel("recall")
        ax[2].set_ylabel("precision")
        ax[2].set_title("Precision-Recall")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "metricas.png", dpi=120)
    plt.close(fig)


def gallery(ds, y, p, thr, out: Path, n: int):
    """Original | mapa | confianza | overlay. Errores más confiados primero."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(-np.abs(p - thr))            # los más "seguros" primero
    errs = [i for i in order if (p[i] >= thr) != y[i]]
    oks = [i for i in order if (p[i] >= thr) == y[i]]
    pick = (errs + oks)[:n]
    if not pick:
        return

    fig, axes = plt.subplots(len(pick), 4, figsize=(13, 3.1 * len(pick)), squeeze=False)
    for r, i in enumerate(pick):
        arr = np.array(ds.arr[ds.idx[i]])           # (5, S, S) uint8
        rgb = arr[:3].transpose(1, 2, 0).astype(np.float32) / 255.0
        heat = arr[3].astype(np.float32) / 255.0
        conf = arr[4].astype(np.float32) / 255.0
        overlay = rgb * (0.4 + 0.6 * heat[..., None])
        err = (p[i] >= thr) != y[i]
        panels = [(rgb, "original"), (heat, "mapa TruFor"), (conf, "confianza"), (overlay, "overlay")]
        for c, (im, title) in enumerate(panels):
            axes[r, c].imshow(im, cmap=None if c in (0, 3) else "inferno", vmin=0, vmax=1)
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
        axes[r, 0].set_title(f"{ds.rows[i]['file'].split('/')[-1]} | real {y[i]} | p {p[i]:.3f}",
                             fontsize=9, color="red" if err else "black")
    fig.tight_layout()
    fig.savefig(out / "galeria.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--packed", default="packed/raw__s384")
    ap.add_argument("--ckpt", default="runs/vit/best.pt")
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--splits", default="runs/splits.json")
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=None, help="por defecto, el guardado en el checkpoint")
    ap.add_argument("--gallery", type=int, default=6, help="0 para no generar galería")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_manifest(Path(a.manifest))
    if a.split == "all":
        sel = rows
    else:
        splits = json.loads(Path(a.splits).read_text())
        train_rows, val_rows = apply_splits(rows, splits)
        sel = val_rows if a.split == "val" else train_rows

    # El manifest pareado repite la auténtica por cada variante manipulada: para
    # medir se cuenta una vez por foto, si no la misma imagen pesa varias veces.
    n_sel = len(sel)
    sel = dedupe_rows(sel)
    if len(sel) != n_sel:
        print(f"{a.split}: {n_sel} filas -> {len(sel)} únicas (manifest pareado)")

    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = build_model(**cfg).to(device)
    model.load_state_dict(ck["model"])

    ds = MapDataset(packed=a.packed, rows=sel,
                    channels="maps+rgb" if cfg["in_chans"] == 5 else "maps")
    loader = DataLoader(ds, batch_size=a.bs, shuffle=False,
                        num_workers=a.workers, pin_memory=device.type == "cuda")
    y, p = predict(model, loader, device)

    thr = a.threshold if a.threshold is not None else float(ck.get("thr", 0.5))
    m = metrics(y, p, thr)
    m.update({"thr": thr, "n": int(len(y)), "pos": int(y.sum()),
              "split": a.split, "ckpt": str(a.ckpt), "packed": str(a.packed)})

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))

    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "prob", "pred", "error"])
        for r, pi in zip(ds.rows, p):
            w.writerow([r["file"], r["label"], f"{pi:.4f}", int(pi >= thr),
                        int((pi >= thr) != r["label"])])

    print(f"{a.split}: n={m['n']} pos={m['pos']} | thr {thr:.2f}")
    print(f"acc {m['acc']:.3f} | bal_acc {m['bal_acc']:.3f} | F1 {m['f1']:.3f} | "
          f"AUC {m['auc']:.3f} | AP {m['ap']:.3f}")
    print(f"errores: {int(sum(1 for r, pi in zip(ds.rows, p) if (pi >= thr) != r['label']))}")
    print(f"-> {out}/metrics.json, predictions.csv")

    try:
        plots(y, p, thr, out)
        print(f"-> {out}/metricas.png")
    except ImportError:
        print("(matplotlib no está instalado: se omiten los gráficos)")
    if a.gallery:
        try:
            gallery(ds, y, p, thr, out, a.gallery)
            print(f"-> {out}/galeria.png")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
