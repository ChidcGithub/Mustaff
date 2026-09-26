"""
用训练好的模型生成谱面（推理）

generate_notes_ml(y, sr, keys, stars) → notes 列表（与 BeatMapper 输出同构）

流程：
1. 提取音频特征（与 build_dataset 一致）
2. PlacementNet 逐帧打分 → 峰值提取得到音符帧
3. SelectionNet 自回归选列（nucleus 采样）
4. BeatMapper._resolve_overlaps 做合法性收尾
"""

import os
from typing import List, Dict, Any, Optional

import numpy as np
import torch

from .build_dataset import SR, HOP, extract_features
from .models import PlacementNet, SelectionNet
from .train_placement import peak_pick

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(ROOT, "output")

_CTX_W = 32
_FRAME_MS = HOP / SR * 1000.0

_model_cache: Dict[str, Any] = {}


def _load_models(model_dir: str, keys: int, device: str):
    if "placement" not in _model_cache:
        pm = PlacementNet()
        pm.load_state_dict(torch.load(os.path.join(model_dir, "placement_best.pt"), map_location=device))
        pm.to(device).eval()
        _model_cache["placement"] = pm
    key = f"selection_{keys}"
    if key not in _model_cache:
        sm = SelectionNet(keys=keys)
        sm.load_state_dict(torch.load(os.path.join(model_dir, "selection_best.pt"), map_location=device))
        sm.to(device).eval()
        _model_cache[key] = sm
    return _model_cache["placement"], _model_cache[key]


def models_available(model_dir: str = DEFAULT_MODEL_DIR) -> bool:
    return (
        os.path.exists(os.path.join(model_dir, "placement_best.pt"))
        and os.path.exists(os.path.join(model_dir, "selection_best.pt"))
    )


@torch.no_grad()
def generate_notes_ml(
    y: np.ndarray,
    sr: int = SR,
    keys: int = 4,
    stars: float = 5.0,
    model_dir: str = DEFAULT_MODEL_DIR,
    threshold: float = 0.5,
    temperature: float = 1.0,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """从音频波形生成 4K 音符（hit 为主；长音留待后续版本）"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    placement, selection = _load_models(model_dir, keys, device)

    feats = extract_features(y.astype(np.float32)).astype(np.float32)
    T = feats.shape[0]
    stars_t = torch.tensor([stars / 10.0], dtype=torch.float32, device=device)

    # 1) placement：分块前向拼成整首歌概率
    probs = np.zeros(T, dtype=np.float32)
    step = 2048
    for s in range(0, T, step):
        chunk = feats[s: s + step]
        pad = step - chunk.shape[0]
        if pad > 0:
            chunk = np.pad(chunk, ((0, pad), (0, 0)))
        x = torch.from_numpy(chunk)[None].to(device)
        logits = placement(x, stars_t)
        p = torch.sigmoid(logits)[0].cpu().numpy()
        probs[s: s + step] = p[: T - s]

    frames = peak_pick(probs, threshold=threshold, min_dist=2)
    if len(frames) == 0:
        return []

    # 2) selection：自回归选列
    half = _CTX_W // 2

    def _ctx(frame: int) -> np.ndarray:
        lo = max(0, frame - half)
        hi = min(T, frame + half)
        w = feats[lo:hi]
        if w.shape[0] < _CTX_W:
            pad_before = max(0, half - frame)
            w = np.pad(w, ((pad_before, _CTX_W - w.shape[0] - pad_before), (0, 0)))
        return w

    notes: List[Dict[str, Any]] = []
    prev_col = keys  # <BOS>
    for f in frames:
        ctx = torch.from_numpy(_ctx(int(f)))[None, None].to(device)  # [1,1,W,65]
        prev = torch.tensor([[prev_col]], dtype=torch.long, device=device)
        logits = selection(ctx, prev, stars_t)[0, 0]  # [keys]
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-4)
        col = int(torch.distributions.Categorical(logits=logits).sample().item())
        prev_col = col
        notes.append({
            "time": int(round(int(f) * _FRAME_MS)),
            "column": col,
            "type": "hit",
            "end_time": None,
            "speed": 10.0,
        })

    # 3) 合法性收尾（复用 BeatMapper 的冲突消解）
    from mustaff.mapper import BeatMapper
    notes = BeatMapper(keys=keys)._resolve_overlaps(notes)
    return notes
