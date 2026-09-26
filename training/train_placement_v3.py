"""
训练 placement v3：拍级 Transformer，多标签预测每拍 48 时隙的「事件」

评估：slot F1（线性时隙 ±2 容差内贪心匹配）
用法：python -m training.train_placement_v3 --epochs 20 --device cuda
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import load_split, NPZ_DIR
from .dataset_v3 import PlacementDatasetV3, collate_place_v3, chart_events, events_to_slots
from .features import FRAME_MS
from .models_v3 import PlacementTransformer, SLOTS_PER_BEAT, beats_to_units, count_params
from .dataset_v3 import normalize_feats

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
BEST_PATH = os.path.join(OUT_DIR, "placement_v3_best.pt")
METRICS_PATH = os.path.join(OUT_DIR, "placement_v3_metrics.json")

SLOT_TOL = 2  # 线性时隙容差（±2 slot ≈ ±18~25ms，网格内已足够精确）


def slot_f1(pred_L: np.ndarray, truth_L: np.ndarray, tol: int = SLOT_TOL):
    if len(truth_L) == 0 or len(pred_L) == 0:
        return 0.0
    used = np.zeros(len(truth_L), dtype=bool)
    tp = 0
    for p in pred_L:
        d = np.abs(truth_L - p)
        j = int(np.argmin(d))
        if d[j] <= tol and not used[j]:
            used[j] = True
            tp += 1
    prec = tp / len(pred_L)
    rec = tp / len(truth_L)
    return 2 * prec * rec / (prec + rec + 1e-8)


@torch.no_grad()
def evaluate(model, files, device, threshold=0.5, max_sets=40):
    model.eval()
    f1s, ratios = [], []
    for fn in files[:max_sets]:
        path = os.path.join(NPZ_DIR, fn)
        with np.load(path) as z:
            feats_full = normalize_feats(z["features"])
            beat_times = z["beat_times"].astype(np.float64)
            fd_idx = int(z["fd_idx"])
            bpm = float(json.loads(bytes(z["meta"]).decode("utf-8")).get("bpm", 0.0))
            for k in z.files:
                if not k.startswith("labels_"):
                    continue
                bid = k[len("labels_"):]
                stars = float(z[f"stars_{bid}"])
                units, bar_phase = beats_to_units(feats_full, beat_times, fd_idx)
                if units.shape[0] < 4:
                    continue
                x = torch.from_numpy(units)[None].to(device)
                bp = torch.from_numpy(bar_phase)[None].to(device)
                cond = torch.tensor([[np.clip(bpm / 300.0, 0, 2), stars / 10.0]],
                                    dtype=torch.float32, device=device)
                probs = torch.sigmoid(model(x, bp, cond))[0].cpu().numpy()  # [B,48]
                pb, ps = np.nonzero(probs > threshold)
                pred_L = pb * SLOTS_PER_BEAT + ps

                ev_f, _ = chart_events(z[f"frames_{bid}"].astype(np.int64),
                                       z[f"cols_{bid}"].astype(np.int64),
                                       z[f"holds_{bid}"].astype(np.int64))
                tb, ts = events_to_slots(ev_f, beat_times, units.shape[0])
                truth_L = tb[tb >= 0] * SLOTS_PER_BEAT + ts[tb >= 0]
                f1s.append(slot_f1(pred_L, truth_L))
                ratios.append(len(pred_L) / max(1, len(truth_L)))
    return float(np.mean(f1s)), float(np.mean(ratios))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--samples-per-epoch", type=int, default=8000)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    split = load_split()
    train_ds = PlacementDatasetV3(split["train"], samples_per_epoch=args.samples_per_epoch)
    print(f"训练集 charts: {len(train_ds.items)}  device: {args.device}", flush=True)

    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0,
                        collate_fn=collate_place_v3, drop_last=True)

    model = PlacementTransformer().to(args.device)
    print(f"参数量: {count_params(model)/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pos_w = torch.tensor(6.0, device=args.device)

    best_f1 = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for units, bar, label, cond, mask in loader:
            units = units.to(args.device)
            bar = bar.to(args.device)
            label = label.to(args.device)
            cond = cond.to(args.device)
            mask = mask.to(args.device)
            logits = model(units, bar, cond, pad_mask=mask)
            loss = F.binary_cross_entropy_with_logits(
                logits[~mask], label[~mask], pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        f1, ratio = evaluate(model, split["val"], args.device)
        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}] loss={np.mean(losses):.4f}  "
              f"val slotF1={f1:.4f}  数量比={ratio:.2f}  {dt:.0f}s", flush=True)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "slot_f1": f1, "count_ratio": ratio})
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), BEST_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_slot_f1": best_f1, "history": history}, f, indent=1)
    print(f"[Done] 最佳 val slotF1={best_f1:.4f}，模型已存到 {BEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
