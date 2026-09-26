"""
PyTorch 数据集

PlacementDataset：随机裁剪窗口 [1024 帧 ≈ 23.7s]，返回 (特征, 标签, 星级)
SelectionDataset：按 chart 返回音符序列样本（上下文窗口 + 列序列）
"""

import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
NPZ_DIR = os.path.join(DATA_DIR, "npz")
SPLIT_PATH = os.path.join(DATA_DIR, "split.json")

WINDOW = 1024      # placement 训练窗口（帧）
CTX_W = 32         # selection 音频上下文窗口（帧，居中）
SEQ_N = 128        # selection 每段序列的音符数


def load_split() -> dict:
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def list_charts(npz_path: str) -> List[str]:
    """列出一个 npz 里所有 chart 的 beatmap id"""
    with np.load(npz_path) as z:
        return [k[len("labels_"):] for k in z.files if k.startswith("labels_")]


class PlacementDataset(Dataset):
    """每次返回一个随机窗口：(feats [1024,65], labels [1024], stars)"""

    def __init__(self, files: List[str], window: int = WINDOW, samples_per_epoch: int = 4000):
        self.window = window
        self.samples_per_epoch = samples_per_epoch
        self.items: List[Tuple[str, str]] = []  # (npz_path, bid)
        self._lengths = {}
        for fn in files:
            path = os.path.join(NPZ_DIR, fn)
            for bid in list_charts(path):
                with np.load(path) as z:
                    T = int(z["features"].shape[0])
                if T >= window // 2:
                    self.items.append((path, bid))
                    self._lengths[(path, bid)] = T
        if not self.items:
            raise RuntimeError("PlacementDataset: 没有可用样本，请先运行 build_dataset")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, _idx):
        path, bid = self.items[np.random.randint(len(self.items))]
        T = self._lengths[(path, bid)]
        with np.load(path) as z:
            feats = z["features"].astype(np.float32)
            labels = z[f"labels_{bid}"].astype(np.float32)
            stars = float(z[f"stars_{bid}"])
        if T <= self.window:
            start = 0
            pad = self.window - T
            feats = np.pad(feats, ((0, pad), (0, 0)))
            labels = np.pad(labels, (0, pad))
        else:
            start = np.random.randint(0, T - self.window + 1)
            feats = feats[start: start + self.window]
            labels = labels[start: start + self.window]
        return (
            torch.from_numpy(feats),
            torch.from_numpy(labels),
            torch.tensor(stars / 10.0, dtype=torch.float32),
        )


class SelectionDataset(Dataset):
    """每次返回一段音符序列：(ctx [N,32,65], prev_cols [N], cols [N], stars)"""

    def __init__(self, files: List[str], keys: int = 4, ctx_w: int = CTX_W, seq_n: int = SEQ_N):
        self.keys = keys
        self.ctx_w = ctx_w
        self.seq_n = seq_n
        self.samples = []
        for fn in files:
            path = os.path.join(NPZ_DIR, fn)
            with np.load(path) as z:
                feats = z["features"].astype(np.float32)
                for bid in list_charts(path):
                    frames = z[f"frames_{bid}"]
                    cols = z[f"cols_{bid}"]
                    stars = float(z[f"stars_{bid}"])
                    if len(frames) < 8:
                        continue
                    self.samples.append((feats, frames, cols, stars))
        if not self.samples:
            raise RuntimeError("SelectionDataset: 没有可用样本，请先运行 build_dataset")

    def __len__(self):
        return len(self.samples)

    def _context(self, feats: np.ndarray, frame: int) -> np.ndarray:
        half = self.ctx_w // 2
        T = feats.shape[0]
        lo = max(0, frame - half)
        hi = min(T, frame + half)
        w = feats[lo:hi]
        if w.shape[0] < self.ctx_w:
            pad_before = max(0, half - frame)
            pad_after = self.ctx_w - w.shape[0] - pad_before
            w = np.pad(w, ((pad_before, pad_after), (0, 0)))
        return w

    def __getitem__(self, idx):
        feats, frames, cols, stars = self.samples[idx]
        n = len(frames)
        if n > self.seq_n:
            start = np.random.randint(0, n - self.seq_n + 1)
            frames = frames[start: start + self.seq_n]
            cols = cols[start: start + self.seq_n]
        N = len(frames)
        ctx = np.stack([self._context(feats, int(f)) for f in frames])  # [N, W, 65]
        prev_cols = np.concatenate([[self.keys], cols[:-1]])            # BOS = keys
        return (
            torch.from_numpy(ctx),
            torch.from_numpy(prev_cols.astype(np.int64)),
            torch.from_numpy(cols.astype(np.int64)),
            torch.tensor(stars / 10.0, dtype=torch.float32),
        )
