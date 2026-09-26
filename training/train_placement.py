"""
训练 placement 模型 v2：多任务（音符头 / 长音头 / 长音时长回归）

评估指标：
- note F1：音符头 ±2 帧匹配（与启发式 baseline 对比）
- hold F1：在帧匹配的基础上，还要求预测为长音且时长误差 ≤ max(4帧, 25%)

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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import PlacementDataset, load_split, NPZ_DIR, DATA_DIR
from .features import CH_ONSET
from .models import PlacementNet, count_params

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
METRICS_PATH = os.path.join(OUT_DIR, "placement_metrics.json")

FRAME_MS = 512 / 22050 * 1000  # ≈23.2ms
EVAL_TOL_FRAMES = 2  # ±46ms 容差
MIN_HOLD_FRAMES = 4  # 短于此的长音无意义（≈93ms）


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
    onset = feats[:, CH_ONSET].astype(np.float32)
    thr = max(0.05, float(onset.mean()) + 0.5 * float(onset.std()))
    return peak_pick(onset / (onset.max() + 1e-8), threshold=thr, min_dist=2)


def match_frames(pred: np.ndarray, truth: np.ndarray, tol: int = EVAL_TOL_FRAMES):
    """预测帧与真实帧贪心匹配，返回 (tp, matched_truth_indices_set)"""
    used = np.zeros(len(truth), dtype=bool)
    matched_truth = set()
    tp = 0
    for p in pred:
        if len(truth) == 0:
            break
        d = np.abs(truth - p)
        j = int(np.argmin(d))
        if d[j] <= tol and not used[j]:
            used[j] = True
            matched_truth.add(j)
            tp += 1
    return tp, matched_truth


def f1_from(tp: int, n_pred: int, n_truth: int):
    prec = tp / max(1, n_pred)
    rec = tp / max(1, n_truth)
    return 2 * prec * rec / (prec + rec + 1e-8), prec, rec


def multitask_loss(note_logits, hold_logits, len_pred, head, hold_head, dur, device):
    """note BCE + hold BCE（仅 head 帧）+ 时长 MSE（仅 hold head 帧）"""
    pos_note = torch.tensor(8.0, device=device)
    loss_note = F.binary_cross_entropy_with_logits(note_logits, head, pos_weight=pos_note)

    head_mask = head > 0.5
    if head_mask.any():
        # 长音头占音符头 ~8%，pos_weight 提高长音召回（用户要求模型能输出长音）
        pos_hold = torch.tensor(4.0, device=device)
        loss_hold = F.binary_cross_entropy_with_logits(
            hold_logits[head_mask], hold_head[head_mask], pos_weight=pos_hold
        )
    else:
        loss_hold = torch.tensor(0.0, device=device)

    hold_mask = hold_head > 0.5
    if hold_mask.any():
        loss_len = F.mse_loss(len_pred[hold_mask], dur[hold_mask])
    else:
        loss_len = torch.tensor(0.0, device=device)

    return loss_note + 0.5 * loss_hold + 0.2 * loss_len, loss_note.item(), loss_hold.item(), loss_len.item()


@torch.no_grad()
def evaluate(model, files, device, hold_threshold: float = 0.5):
    """val 集逐 chart 评估：note F1 / hold F1 / baseline F1"""
    model.eval()
    f1s, f1s_base, f1s_hold = [], [], []
    for fn in files:
        path = os.path.join(NPZ_DIR, fn)
        with np.load(path) as z:
            feats_full = z["features"].astype(np.float32)
            for k in z.files:
                if not k.startswith("labels_"):
                    continue
                bid = k[len("labels_"):]
                truth_frames = z[f"frames_{bid}"]
                truth_holds = z[f"holds_{bid}"]
                truth_is_hold = truth_holds > truth_frames
                stars = float(z[f"stars_{bid}"])

                T = feats_full.shape[0]
                probs = np.zeros(T, dtype=np.float32)
                holdps = np.zeros(T, dtype=np.float32)
                lens = np.zeros(T, dtype=np.float32)
                step = 2048
                for s in range(0, T, step):
                    chunk = feats_full[s: s + step]
                    pad = step - chunk.shape[0]
                    if pad > 0:
                        chunk = np.pad(chunk, ((0, pad), (0, 0)))
                    x = torch.from_numpy(chunk)[None].to(device)
                    st = torch.tensor([stars / 10.0], dtype=torch.float32, device=device)
                    nl, hl, lp = model(x, st)
                    probs[s: s + step] = torch.sigmoid(nl)[0].cpu().numpy()[: T - s]
                    holdps[s: s + step] = torch.sigmoid(hl)[0].cpu().numpy()[: T - s]
                    lens[s: s + step] = lp[0].cpu().numpy()[: T - s]

                pred = peak_pick(probs)
                tp, _ = match_frames(pred, truth_frames)
                f1, _, _ = f1_from(tp, len(pred), len(truth_frames))
                f1s.append(f1)

                pred_base = onset_baseline(feats_full)
                tpb, _ = match_frames(pred_base, truth_frames)
                f1b, _, _ = f1_from(tpb, len(pred_base), len(truth_frames))
                f1s_base.append(f1b)

                # hold F1：帧匹配 + 类型正确 + 时长相近
                pred_hold_frames = [p for p in pred if holdps[p] > hold_threshold]
                tp_h = 0
                used_t = np.zeros(len(truth_frames), dtype=bool)
                for p in pred_hold_frames:
                    if len(truth_frames) == 0:
                        break
                    d = np.abs(truth_frames - p)
                    j = int(np.argmin(d))
                    if d[j] > EVAL_TOL_FRAMES or used_t[j] or not truth_is_hold[j]:
                        continue
                    pred_len = np.expm1(lens[p])
                    true_len = truth_holds[j] - truth_frames[j]
                    if abs(pred_len - true_len) <= max(4, 0.25 * true_len):
                        used_t[j] = True
                        tp_h += 1
                n_truth_hold = int(truth_is_hold.sum())
                f1h, _, _ = f1_from(tp_h, len(pred_hold_frames), n_truth_hold)
                f1s_hold.append(f1h)

    return float(np.mean(f1s)), float(np.mean(f1s_base)), float(np.mean(f1s_hold))


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
    val_files = split["val"][:80]  # val 子采样，控制每轮评估耗时
    train_ds = PlacementDataset(split["train"], samples_per_epoch=args.samples_per_epoch)
    print(f"训练集 charts: {len(train_ds.items)}  val sets: {len(val_files)}  device: {args.device}", flush=True)

    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    model = PlacementNet().to(args.device)
    print(f"参数量: {count_params(model)/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_score = 0.0
    best_f1 = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses, ln_, lh_, ll_ = [], [], [], []
        for feats, head, hold_head, dur, stars in loader:
            feats = feats.to(args.device)
            head = head.to(args.device)
            hold_head = hold_head.to(args.device)
            dur = dur.to(args.device)
            stars = stars.to(args.device)
            note_logits, hold_logits, len_pred = model(feats, stars)
            loss, l_n, l_h, l_l = multitask_loss(
                note_logits, hold_logits, len_pred, head, hold_head, dur, args.device
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            ln_.append(l_n)
            lh_.append(l_h)
            ll_.append(l_l)
        sched.step()

        f1, f1_base, f1_hold = evaluate(model, val_files, args.device)
        score = f1 + f1_hold  # 存档指标：音符质量 + 长音质量（v2 目标是长音）
        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}] loss={np.mean(losses):.4f} "
              f"(note {np.mean(ln_):.3f} hold {np.mean(lh_):.3f} len {np.mean(ll_):.3f})  "
              f"val noteF1={f1:.4f} holdF1={f1_hold:.4f} (baseline {f1_base:.4f})  {dt:.0f}s", flush=True)
        history.append({
            "epoch": epoch, "loss": float(np.mean(losses)),
            "f1": f1, "f1_hold": f1_hold, "f1_base": f1_base,
        })

        if score > best_score:
            best_score = score
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "placement_best.pt"))

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_f1": best_f1, "history": history}, f, indent=1)
    print(f"[Done] 最佳 val noteF1={best_f1:.4f}，模型已存到 {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
