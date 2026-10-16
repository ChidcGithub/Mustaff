"""
训练 placement v5-mert：冻结 MERT 表征（每拍 8×768）+ 拍级 Transformer

超参与评估口径与 v3.1 完全一致（slot F1 ±2、pos_w 4.0、diag_w 0.3、EMA 0.999、
bf16 AMP、warmup 1 epoch → cosine、25 epochs），仅音频输入换源。
用法：python -m training.train_placement_mert --epochs 25 --device cuda
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import load_split, NPZ_DIR
from .dataset_mert import MERT_DIR, PlacementDatasetMERT, collate_place_v3
from .dataset_v3 import chart_events, events_to_slots
from .models_mert import MertProjection, D_MERT, count_params
from .models_v3 import SLOTS_PER_BEAT
from .train_placement_v3 import slot_f1

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
BEST_PATH = os.path.join(OUT_DIR, "placement_mert_best.pt")
METRICS_PATH = os.path.join(OUT_DIR, "placement_mert_metrics.json")


@torch.no_grad()
def evaluate(model, files, device, threshold=0.5, max_sets=40):
    """与 train_placement_v3.evaluate 同口径：EMA 权重 + 固定阈值 0.5 的 val slot F1"""
    model.eval()
    f1s, ratios = [], []
    for fn in files[:max_sets]:
        path = os.path.join(NPZ_DIR, fn)
        mert_path = os.path.join(MERT_DIR, fn[:-4] + ".npy")
        if not os.path.exists(mert_path):
            continue
        with np.load(path) as z:
            beat_times = z["beat_times"].astype(np.float64)
            fd_idx = int(z["fd_idx"])
            bpm = float(json.loads(bytes(z["meta"]).decode("utf-8")).get("bpm", 0.0))
            for k in z.files:
                if not k.startswith("labels_"):
                    continue
                bid = k[len("labels_"):]
                stars = float(z[f"stars_{bid}"])
                units = np.load(mert_path, mmap_mode="r").astype(np.float32)
                n_units = units.shape[0]
                if n_units < 4:
                    continue
                bar_phase = (np.mod(np.arange(n_units) - fd_idx, 4) / 4.0).astype(np.float32)
                x = torch.from_numpy(units)[None].to(device)
                bp = torch.from_numpy(bar_phase)[None].to(device)
                cond = torch.tensor([[np.clip(bpm / 300.0, 0, 2), stars / 10.0]],
                                    dtype=torch.float32, device=device)
                if device.startswith("cuda"):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(x, bp, cond)[0]
                else:
                    out = model(x, bp, cond)[0]
                probs = torch.sigmoid(out.float())[0].cpu().numpy()

                pb, ps = np.nonzero(probs > threshold)
                pred_L = pb * SLOTS_PER_BEAT + ps
                ev_f, _ = chart_events(z[f"frames_{bid}"].astype(np.int64),
                                       z[f"cols_{bid}"].astype(np.int64),
                                       z[f"holds_{bid}"].astype(np.int64))
                tb, ts = events_to_slots(ev_f, beat_times, n_units)
                truth_L = tb[tb >= 0] * SLOTS_PER_BEAT + ts[tb >= 0]
                f1s.append(slot_f1(pred_L, truth_L))
                ratios.append(len(pred_L) / max(1, len(truth_L)))
    return float(np.mean(f1s)), float(np.mean(ratios))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--samples-per-epoch", type=int, default=8000)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    split = load_split()
    train_ds = PlacementDatasetMERT(split["train"], samples_per_epoch=args.samples_per_epoch)
    print(f"训练集 charts: {len(train_ds.items)}  device: {args.device}", flush=True)

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=4 if use_cuda else 0, pin_memory=use_cuda,
                        persistent_workers=use_cuda, prefetch_factor=4 if use_cuda else None,
                        collate_fn=collate_place_v3, drop_last=True)

    model = MertProjection().to(args.device)
    print(f"参数量: {count_params(model)/1e6:.2f}M（另有冻结 MERT 95M 只在特征构建时用）",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=1),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - 1))],
        milestones=[1])
    pos_w = torch.tensor(4.0, device=args.device)
    diag_w = 0.3

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def ema_update(decay: float = 0.999):
        for ep, p in zip(ema_model.parameters(), model.parameters()):
            ep.mul_(decay).add_(p.detach(), alpha=1 - decay)
        for eb, b in zip(ema_model.buffers(), model.buffers()):
            eb.copy_(b)

    def _forward(units, bar, cond, mask):
        if use_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(units, bar, cond, pad_mask=mask)
        return model(units, bar, cond, pad_mask=mask)

    best_f1 = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for units, bar, label, cond, mask in loader:
            units = units.to(args.device, non_blocking=True)
            bar = bar.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)
            cond = cond.to(args.device, non_blocking=True)
            mask = mask.to(args.device, non_blocking=True)
            logits, diag = _forward(units, bar, cond, mask)
            loss = F.binary_cross_entropy_with_logits(
                logits[~mask].float(), label[~mask], pos_weight=pos_w)
            loss = loss + diag_w * F.mse_loss(diag.float(), cond)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update()
            losses.append(loss.item())
        sched.step()

        f1, ratio = evaluate(ema_model, split["val"], args.device)
        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}] loss={np.mean(losses):.4f}  "
              f"val slotF1={f1:.4f}  数量比={ratio:.2f}  {dt:.0f}s", flush=True)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "slot_f1": f1, "count_ratio": ratio})
        if f1 > best_f1:
            best_f1 = f1
            torch.save(ema_model.state_dict(), BEST_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_slot_f1": best_f1, "history": history}, f, indent=1)
    print(f"[Done] 最佳 val slotF1={best_f1:.4f}，模型已存到 {BEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
