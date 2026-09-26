"""
训练 placement 模型（何时放音符），并与启发式 baseline 对比 F1

用法：
  python -m training.train_placement --epochs 15 --device cuda
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import PlacementDataset, load_split, NPZ_DIR, DATA_DIR
from .models import PlacementNet, count_params

OUT_DIR = os.path.join(ROOT := os.path.dirname(os.path.abspath(__file__)), "output")
METRICS_PATH = os.path.join(OUT_DIR, "placement_metrics.json")

# 帧长（与 build_dataset 一致）
FRAME_MS = 512 / 22050 * 1000  # ≈23.2ms
EVAL_TOL_FRAMES = 2  # ±46ms 容差


def peak_pick(probs: np.ndarray, threshold: float = 0.5, min_dist: int = 2) -> np.ndarray:
    """概率序列 → 峰值帧号"""
    peaks = []
    for i in range(1, len(probs) - 1):
        if probs[i] >= threshold and probs[i] >= probs[i - 1] and probs[i] >= probs[i + 1]:
            peaks.append(i)
    # 最小间距去重（保留概率高的）
    out = []
    for p in sorted(peaks, key=lambda i: -probs[i]):
        if all(abs(p - q) > min_dist for q in out):
            out.append(p)
    return np.array(sorted(out), dtype=np.int64)


def onset_baseline(feats: np.ndarray) -> np.ndarray:
    """启发式 baseline：对 onset 强度通道做峰值检测（等价于现有管线的 onset 检测）"""
    onset = feats[:, -1].astype(np.float32)
    # 与 librosa onset_detect(delta≈0.07) 近似：局部最大 + 阈值
    thr = max(0.05, float(onset.mean()) + 0.5 * float(onset.std()))
    return peak_pick(onset / (onset.max() + 1e-8), threshold=thr, min_dist=2)


def f1_score(pred: np.ndarray, truth_frames: np.ndarray, tol: int = EVAL_TOL_FRAMES):
    """音符级 F1：预测帧与真实帧在 ±tol 内匹配"""
    if len(pred) == 0 and len(truth_frames) == 0:
        return 1.0, 1.0, 1.0
    if len(pred) == 0:
        return 0.0, 0.0, 1.0
    if len(truth_frames) == 0:
        return 0.0, 1.0, 0.0
    used = np.zeros(len(truth_frames), dtype=bool)
    tp = 0
    for p in pred:
        d = np.abs(truth_frames - p)
        j = int(np.argmin(d))
        if d[j] <= tol and not used[j]:
            used[j] = True
            tp += 1
    prec = tp / len(pred)
    rec = tp / len(truth_frames)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return f1, prec, rec


@torch.no_grad()
def evaluate(model, files, device, keys_desc="model"):
    """在 val 集上逐 chart 评估 F1（对 baseline 同样适用时由调用方处理）"""
    model.eval()
    f1s, f1s_base = [], []
    for fn in files:
        path = os.path.join(NPZ_DIR, fn)
        with np.load(path) as z:
            feats_full = z["features"].astype(np.float32)
            for k in z.files:
                if not k.startswith("labels_"):
                    continue
                bid = k[len("labels_"):]
                labels = z[k]
                stars = float(z[f"stars_{bid}"])
                truth_frames = np.nonzero(labels)[0]

                # 整首歌前向（分块 2048 帧拼接）
                T = feats_full.shape[0]
                probs = np.zeros(T, dtype=np.float32)
                step = 2048
                for s in range(0, T, step):
                    chunk = feats_full[s: s + step]
                    pad = step - chunk.shape[0]
                    if pad > 0:
                        chunk = np.pad(chunk, ((0, pad), (0, 0)))
                    x = torch.from_numpy(chunk)[None].to(device)
                    st = torch.tensor([stars / 10.0], dtype=torch.float32, device=device)
                    logits = model(x, st)
                    p = torch.sigmoid(logits)[0].cpu().numpy()
                    probs[s: s + step] = p[: T - s] if pad == 0 else p[: T - s]

                pred = peak_pick(probs)
                f1, _, _ = f1_score(pred, truth_frames)
                f1s.append(f1)

                pred_base = onset_baseline(feats_full)
                f1b, _, _ = f1_score(pred_base, truth_frames)
                f1s_base.append(f1b)

    return float(np.mean(f1s)), float(np.mean(f1s_base))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--samples-per-epoch", type=int, default=4000)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    split = load_split()
    train_ds = PlacementDataset(split["train"], samples_per_epoch=args.samples_per_epoch)
    print(f"训练集 charts: {len(train_ds.items)}  val sets: {len(split['val'])}  device: {args.device}")

    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    model = PlacementNet().to(args.device)
    print(f"参数量: {count_params(model)/1e6:.2f}M")

    # 正样本占比约 6-14%，pos_weight 补偿类别不平衡
    pos_weight = torch.tensor(8.0, device=args.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1 = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for feats, labels, stars in loader:
            feats = feats.to(args.device)
            labels = labels.to(args.device)
            stars = stars.to(args.device)
            logits = model(feats, stars)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        f1, f1_base = evaluate(model, split["val"], args.device)
        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}] loss={np.mean(losses):.4f}  val F1={f1:.4f} (baseline {f1_base:.4f})  {dt:.0f}s")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "f1": f1, "f1_base": f1_base})

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "placement_best.pt"))

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_f1": best_f1, "history": history}, f, indent=1)
    print(f"[Done] 最佳 val F1={best_f1:.4f}，模型已存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
