"""
PyTorch 数据集 v5-mert（配合 models_mert）

MERT 表征已在 build_dataset_mert 预计算为 [n_units, 8, 768] fp16（冻结编码器，
曲集内所有 chart 共享同一份按拍单元）——placement 与 selection 直接读同一份，
标签/增广/条件管线与 v3 逐行同构（collate 直接复用 dataset_v3 的通用版本）。
"""

import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import DATA_DIR, NPZ_DIR
from .dataset_v3 import chart_events, events_to_slots, PERM_TABLE  # noqa: F401
from .dataset_raw import _load_chart_meta
from .models_v3 import SLOTS_PER_BEAT, BOS, SEL_HISTORY
from .models_mert import D_MERT
from .features import FRAME_MS  # noqa: F401（条件向量里 ioi 用）

MERT_DIR = os.path.join(DATA_DIR, "mert_beats")
PLACE_WINDOW = 256
SEL_SEQ = 128


class PlacementDatasetMERT(Dataset):
    """返回 (units [W, 8, 768] float32, bar_phase [W], label [W, 48], cond [2])"""

    def __init__(self, files: List[str], window: int = PLACE_WINDOW,
                 samples_per_epoch: int = 8000):
        self.window = window
        self.samples_per_epoch = samples_per_epoch
        self.items: List[Tuple[str, str, str]] = []  # (npz_path, mert_path, bid)
        for fn in files:
            path = os.path.join(NPZ_DIR, fn)
            mert_path = os.path.join(MERT_DIR, fn[:-4] + ".npy")
            if not (os.path.exists(mert_path) and os.path.exists(path)):
                continue
            try:
                units = np.load(mert_path, mmap_mode="r")
            except Exception:
                continue
            with np.load(path) as z:
                if "beat_times" not in z.files or len(z["beat_times"]) < 8:
                    continue
                n_units = len(z["beat_times"]) - 1
                if units.shape != (n_units, 8, D_MERT):
                    continue
                for k in z.files:
                    if k.startswith("labels_"):
                        self.items.append((path, mert_path, k[len("labels_"):]))
        if not self.items:
            raise RuntimeError("PlacementDatasetMERT: 无样本（先跑 training.build_dataset_mert）")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, _idx):
        path, mert_path, bid = self.items[np.random.randint(len(self.items))]
        frames, cols, holds, stars, beat_times, fd_idx, bpm = _load_chart_meta(path, bid)
        units_full = np.load(mert_path, mmap_mode="r")
        n_units = units_full.shape[0]

        ev_f, _ = chart_events(frames, cols, holds)
        ev_b, ev_s = events_to_slots(ev_f, beat_times, n_units)
        label = np.zeros((n_units, SLOTS_PER_BEAT), dtype=np.float32)
        ok = ev_b >= 0
        label[ev_b[ok], ev_s[ok]] = 1.0

        start = 0
        if n_units > self.window:
            for _ in range(8):
                s = np.random.randint(0, n_units - self.window + 1)
                if label[s: s + self.window].sum() >= 8:
                    start = s
                    break
            else:
                start = np.random.randint(0, n_units - self.window + 1)

        units = np.asarray(units_full[start: start + self.window], dtype=np.float32)
        bar_phase = (np.mod(np.arange(n_units) - fd_idx, 4)
                     / 4.0).astype(np.float32)[start: start + self.window]
        label = label[start: start + self.window]
        cond = np.array([np.clip(bpm / 300.0, 0, 2), stars / 10.0], dtype=np.float32)
        return (
            torch.from_numpy(units),
            torch.from_numpy(bar_phase),
            torch.from_numpy(label),
            torch.from_numpy(cond),
        )


class SelectionDatasetMERT(Dataset):
    """返回 (audio_ctx [N, 5*8*768], prev [N,16], cond [N,9], target [N])"""

    def __init__(self, files: List[str], keys: int = 4, seq_n: int = SEL_SEQ,
                 ctx_beats: int = 2, mirror: bool = True):
        self.keys = keys
        self.seq_n = seq_n
        self.ctx_beats = ctx_beats
        self.mirror = mirror
        self.items: List[Tuple[str, str, str]] = []  # (npz_path, mert_path, bid)
        for fn in files:
            path = os.path.join(NPZ_DIR, fn)
            mert_path = os.path.join(MERT_DIR, fn[:-4] + ".npy")
            if not (os.path.exists(mert_path) and os.path.exists(path)):
                continue
            with np.load(path) as z:
                if "beat_times" not in z.files or len(z["beat_times"]) < 8:
                    continue
                for k in z.files:
                    if not k.startswith("labels_"):
                        continue
                    bid = k[len("labels_"):]
                    if int(z[f"frames_{bid}"].shape[0]) >= 8:
                        self.items.append((path, mert_path, bid))
        if not self.items:
            raise RuntimeError("SelectionDatasetMERT: 无样本（先跑 training.build_dataset_mert）")

    def __len__(self):
        return len(self.items)

    def _audio_ctx(self, units: np.ndarray, beat_idx: int) -> np.ndarray:
        c = self.ctx_beats
        B = units.shape[0]
        lo, hi = beat_idx - c, beat_idx + c + 1
        w = np.zeros((2 * c + 1, units.shape[1], units.shape[2]), dtype=np.float32)
        src_lo, src_hi = max(0, lo), min(B, hi)
        w[src_lo - lo: src_hi - lo] = units[src_lo:src_hi]
        return w.reshape(-1)

    def __getitem__(self, idx):
        path, mert_path, bid = self.items[idx]
        frames, cols, holds, stars, beat_times, fd_idx, bpm = _load_chart_meta(path, bid)
        units = np.load(mert_path, mmap_mode="r").astype(np.float32)
        n_units = units.shape[0]

        ev_f, combos = chart_events(frames, cols, holds)
        ev_b, ev_s = events_to_slots(ev_f, beat_times, n_units)
        ok = ev_b >= 0
        ev_b, ev_s, combos, ev_f = ev_b[ok], ev_s[ok], combos[ok], ev_f[ok]
        if len(combos) < 8:
            return self[(idx + 1) % len(self)]

        if self.mirror and np.random.rand() < 0.75:
            pi = np.random.randint(PERM_TABLE.shape[0])
            combos = PERM_TABLE[pi][torch.from_numpy(combos)].numpy()

        n = len(combos)
        if n > self.seq_n:
            start = np.random.randint(0, n - self.seq_n + 1)
            sl = slice(start, start + self.seq_n)
            ev_b, ev_s, combos, ev_f = ev_b[sl], ev_s[sl], combos[sl], ev_f[sl]
        N = len(combos)

        audio_ctx = np.stack([self._audio_ctx(units, int(b)) for b in ev_b])
        prev = np.full((N, SEL_HISTORY), BOS, dtype=np.int64)
        for h in range(1, SEL_HISTORY + 1):
            prev[h:, -h] = combos[:-h]

        slot_rad = ev_s / SLOTS_PER_BEAT * 2 * np.pi
        bar_idx = np.mod(ev_b - fd_idx, 4)
        from .features import FRAME_MS
        ioi = np.diff(ev_f.astype(np.float64), prepend=ev_f[0]) * FRAME_MS
        cond = np.stack([
            np.full(N, np.clip(bpm / 300.0, 0, 2), dtype=np.float32),
            np.full(N, stars / 10.0, dtype=np.float32),
            np.sin(slot_rad), np.cos(slot_rad),
            (bar_idx == 0).astype(np.float32), (bar_idx == 1).astype(np.float32),
            (bar_idx == 2).astype(np.float32), (bar_idx == 3).astype(np.float32),
            (np.log1p(ioi) / 8.0).astype(np.float32),
        ], axis=1)

        return (
            torch.from_numpy(audio_ctx.astype(np.float32)),
            torch.from_numpy(prev),
            torch.from_numpy(cond.astype(np.float32)),
            torch.from_numpy(combos),
        )


# collate 与 v3 完全同构（维度从 batch 推断），直接复用
from .dataset_v3 import collate_place_v3, collate_sel_v3  # noqa: E402,F401
