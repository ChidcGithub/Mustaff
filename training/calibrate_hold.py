"""长音阈值标定：在 val 子集上扫 hold_threshold，选 holdF1 最优值"""
import numpy as np
import torch

from training.dataset import load_split
from training.models import PlacementNet
from training.train_placement import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PlacementNet()
model.load_state_dict(torch.load("training/output/placement_best.pt",
                                 map_location=device, weights_only=True))
model.to(device).eval()

files = load_split()["val"][:25]
print(f"标定集: {len(files)} 个 set")
best = (0.5, 0.0)
for thr in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
    f1, _, f1_hold = evaluate(model, files, device, hold_threshold=thr)
    print(f"hold_threshold={thr:.2f}  noteF1={f1:.4f}  holdF1={f1_hold:.4f}", flush=True)
    if f1_hold > best[1]:
        best = (thr, f1_hold)
print(f"[Done] 最优 hold_threshold={best[0]}（holdF1={best[1]:.4f}）")
