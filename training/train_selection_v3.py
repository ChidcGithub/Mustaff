"""
训练 selection v3：事件序列 → 256 组合分类（4列×{无,音符,长音头,长音尾}）

评估：teacher-forced combo 精确准确率 vs 众数先验
用法：python -m training.train_selection_v3 --epochs 15 --device cuda
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import load_split
from .dataset_v3 import SelectionDatasetV3, collate_sel_v3
from .models_v3 import SelectionLSTM, count_params

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
BEST_PATH = os.path.join(OUT_DIR, "selection_v3_best.pt")
METRICS_PATH = os.path.join(OUT_DIR, "selection_v3_metrics.json")


@torch.no_grad()
def evaluate(model, val_ds, device, batch=32):
    model.eval()
    loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0,
                        collate_fn=collate_sel_v3)
    correct, total = 0, 0
    prior_correct = 0
    prior_id = 0  # 众数组合（几乎所有时刻四列皆空在事件序列中不存在，众数通常是单音符列）
    counts = np.zeros(256, dtype=np.int64)
    for _, _, _, t in loader:
        m = t >= 0
        counts += np.bincount(t[m].numpy(), minlength=256)
    prior_id = int(counts.argmax())
    for audio, prev, cond, target in loader:
        audio = audio.to(device)
        prev = prev.to(device)
        cond = cond.to(device)
        logits, _ = model(audio, prev, cond)
        pred = logits.argmax(dim=-1).cpu()
        m = target >= 0
        correct += int((pred[m] == target[m]).sum())
        prior_correct += int((target[m] == prior_id).sum())
        total += int(m.sum())
    return correct / max(1, total), prior_correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    split = load_split()
    train_ds = SelectionDatasetV3(split["train"], mirror=True)
    val_ds = SelectionDatasetV3(split["val"], mirror=False)
    print(f"训练 charts: {len(train_ds)}  val charts: {len(val_ds)}  device: {args.device}", flush=True)

    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0,
                        collate_fn=collate_sel_v3)

    model = SelectionLSTM().to(args.device)
    print(f"参数量: {count_params(model)/1e6:.2f}M", flush=True)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for audio, prev, cond, target in loader:
            audio = audio.to(args.device)
            prev = prev.to(args.device)
            cond = cond.to(args.device)
            target = target.to(args.device)
            logits, _ = model(audio, prev, cond)
            loss = criterion(logits.reshape(-1, model.n_combo), target.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        acc, prior = evaluate(model, val_ds, args.device)
        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}] loss={np.mean(losses):.4f}  "
              f"val combo_acc={acc:.4f} (先验 {prior:.4f})  {dt:.0f}s", flush=True)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "acc": acc})
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_acc": best_acc, "history": history}, f, indent=1)
    print(f"[Done] 最佳 val combo_acc={best_acc:.4f}，模型已存到 {BEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
