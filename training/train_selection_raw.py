"""
训练 selection v4-raw：事件序列 → 256 组合分类（4列×{无,音符,长音头,长音尾}）

与 train_selection_v3 完全同构（scheduled sampling 25%、EMA、镜像增广、combo acc 评估），
仅音频上下文从 82 维特征窗口换成 placement-raw 前端的拍级嵌入缓存。

首次运行会自动用 placement_raw_best.pt 构建嵌入缓存（raw_emb/<set>.npy，~5GB）。
用法：python -m training.train_selection_raw --epochs 15 --device cuda
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import load_split
from .dataset_raw import (
    SelectionDatasetRaw, build_embedding_cache, collate_sel_v3,
)
from .models_raw import raw_selection_model, count_params

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
BEST_PATH = os.path.join(OUT_DIR, "selection_raw_best.pt")
METRICS_PATH = os.path.join(OUT_DIR, "selection_raw_metrics.json")


@torch.no_grad()
def evaluate(model, val_ds, device, batch=32):
    """teacher-forced combo 精确准确率 vs 众数先验（与 v3 同口径）"""
    model.eval()
    loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0,
                        collate_fn=collate_sel_v3)
    correct, total = 0, 0
    prior_correct = 0
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
    ap.add_argument("--skip-emb-cache", action="store_true", help="跳过嵌入缓存检查/构建")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    split = load_split()

    if not args.skip_emb_cache:
        print("[emb] 检查/构建拍级嵌入缓存（已存在的 set 跳过）...", flush=True)
        t0 = time.time()
        n = build_embedding_cache(split["train"] + split["val"] + split["test"], args.device)
        print(f"[emb] 本次新缓存 {n} 个 set，用时 {time.time()-t0:.0f}s", flush=True)

    train_ds = SelectionDatasetRaw(split["train"], mirror=True)
    val_ds = SelectionDatasetRaw(split["val"], mirror=False)
    print(f"训练 charts: {len(train_ds)}  val charts: {len(val_ds)}  device: {args.device}",
          flush=True)

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=4 if use_cuda else 0, pin_memory=use_cuda,
                        persistent_workers=use_cuda, prefetch_factor=4 if use_cuda else None,
                        collate_fn=collate_sel_v3)

    model = raw_selection_model().to(args.device)
    print(f"参数量: {count_params(model)/1e6:.2f}M", flush=True)

    def _forward(audio, prev, cond):
        if use_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(audio, prev, cond)
        return model(audio, prev, cond)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def ema_update(decay: float = 0.999):
        for ep, p in zip(ema_model.parameters(), model.parameters()):
            ep.mul_(decay).add_(p.detach(), alpha=1 - decay)
        for eb, b in zip(ema_model.buffers(), model.buffers()):
            eb.copy_(b)

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        ss_p = 0.25 * min(1.0, (epoch - 1) / max(1.0, args.epochs / 2))
        for audio, prev, cond, target in loader:
            audio = audio.to(args.device, non_blocking=True)
            prev = prev.to(args.device, non_blocking=True)
            cond = cond.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            prev_in = prev
            if ss_p > 0:
                with torch.no_grad():
                    own = _forward(audio, prev, cond)[0].argmax(dim=-1)
                prev_in = prev.clone()
                B, N, H = prev.shape
                n_range = torch.arange(N, device=args.device)
                for h in range(H):
                    back = H - h
                    src = (n_range - back).clamp(min=0)
                    valid = ((n_range - back) >= 0)[None, :] & (target >= 0)
                    repl = torch.where((torch.rand(B, N, device=args.device) < ss_p) & valid,
                                       own[:, src], prev_in[:, :, h])
                    prev_in[:, :, h] = repl
            logits, _ = _forward(audio, prev_in, cond)
            loss = criterion(logits.reshape(-1, model.n_combo).float(), target.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update()
            losses.append(loss.item())
        sched.step()

        acc, prior = evaluate(ema_model, val_ds, args.device)
        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}] loss={np.mean(losses):.4f}  ss_p={ss_p:.2f}  "
              f"val combo_acc={acc:.4f} (先验 {prior:.4f})  {dt:.0f}s", flush=True)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "acc": acc, "ss_p": ss_p})
        if acc > best_acc:
            best_acc = acc
            torch.save(ema_model.state_dict(), BEST_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_acc": best_acc, "history": history}, f, indent=1)
    print(f"[Done] 最佳 val combo_acc={best_acc:.4f}，模型已存到 {BEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
