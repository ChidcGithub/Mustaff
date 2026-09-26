"""
训练 selection 模型（给每个音符选列），teacher forcing

用法：
  python -m training.train_selection --epochs 15 --device cuda
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import SelectionDataset, load_split
from .models import SelectionNet, count_params, HISTORY

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
METRICS_PATH = os.path.join(OUT_DIR, "selection_metrics.json")


def collate_selection(batch, keys: int = 4):
    """变长序列 padding：cols 用 -100（CE 忽略），prev_cols 用 <BOS>=keys"""
    maxn = max(item[1].shape[0] for item in batch)
    W, F = batch[0][0].shape[1], batch[0][0].shape[2]
    B = len(batch)
    ctx = torch.zeros(B, maxn, W, F)
    prev = torch.full((B, maxn, HISTORY), keys, dtype=torch.long)
    cols = torch.full((B, maxn), -100, dtype=torch.long)
    stars = torch.zeros(B)
    for i, (c, p, t, s) in enumerate(batch):
        n = p.shape[0]
        ctx[i, :n] = c
        prev[i, :n] = p
        cols[i, :n] = t
        stars[i] = s
    return ctx, prev, cols, stars


@torch.no_grad()
def evaluate(model, ds, device):
    model.eval()
    correct = 0
    total = 0
    # 列频率先验 baseline
    prior = torch.zeros(model.keys)
    for ctx, prev_cols, cols, stars in ds:
        prior += torch.bincount(cols, minlength=model.keys)
    prior_pred = int(prior.argmax())

    for ctx, prev_cols, cols, stars in ds:
        ctx = ctx[None].to(device)
        prev_cols = prev_cols[None].to(device)
        stars = stars[None].to(device)
        logits = model(ctx, prev_cols, stars)[0]
        pred = logits.argmax(dim=-1).cpu()
        correct += int((pred == cols).sum())
        total += len(cols)
    acc = correct / max(1, total)
    prior_acc = float(prior[prior_pred] / prior.sum())
    return acc, prior_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    split = load_split()
    train_ds = SelectionDataset(split["train"])
    val_ds = SelectionDataset(split["val"])
    print(f"训练 charts: {len(train_ds)}  val charts: {len(val_ds)}  device: {args.device}")

    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0,
                        collate_fn=collate_selection)

    model = SelectionNet(keys=4).to(args.device)
    print(f"参数量: {count_params(model)/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for ctx, prev_cols, cols, stars in loader:
            ctx = ctx.to(args.device)
            prev_cols = prev_cols.to(args.device)
            cols = cols.to(args.device)
            stars = stars.to(args.device)
            logits = model(ctx, prev_cols, stars)  # [B, N, keys]
            loss = criterion(logits.reshape(-1, model.keys), cols.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        acc, prior_acc = evaluate(model, val_ds, args.device)
        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}] loss={np.mean(losses):.4f}  val acc={acc:.4f} (先验 {prior_acc:.4f})  {dt:.0f}s")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "acc": acc, "prior_acc": prior_acc})

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "selection_best.pt"))

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_acc": best_acc, "history": history}, f, indent=1)
    print(f"[Done] 最佳 val acc={best_acc:.4f}，模型已存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
