"""
PyTorch 数据集（v2：含长音标签）

npz 中每个 chart 存了 frames（音符头帧）/ cols / holds（长音结束帧，hit 为 -1），
本模块在线推导训练标签：

PlacementDataset：
  返回 (feats, head, hold_head, dur_log1p, stars)
  - head:      [W] 该帧是否为音符头
  - hold_head: [W] 该帧是否为长音头
  - dur_log1p: [W] 长音时长 log1p(帧)，非长音头处为 0

SelectionDataset：
  返回 (ctx [N,32,82], prev_cols [N,HISTORY], cols [N], stars)
"""

import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .models import HISTORY

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


def build_frame_labels(frames: np.ndarray, holds: np.ndarray, start: int, length: int):
    """窗口 [start, start+length) 内推导 head / hold_head / dur_log1p"""
    head = np.zeros(length, dtype=np.float32)
    hold_head = np.zeros(length, dtype=np.float32)
    dur = np.zeros(length, dtype=np.float32)
    mask = (frames >= start) & (frames < start + length)
    for f, h in zip(frames[mask] - start, holds[mask]):
        head[f] = 1.0
        if h > f + start:  # 有效长音（结束帧 > 头帧）
            hold_head[f] = 1.0
            dur[f] = np.log1p(h - (f + start))
    return head, hold_head, dur


class PlacementDataset(Dataset):
    """每次返回一个随机窗口：(feats, head, hold_head, dur_log1p, stars)"""

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
            frames = z[f"frames_{bid}"]
            holds = z[f"holds_{bid}"]
            stars = float(z[f"stars_{bid}"])
        if T <= self.window:
            start = 0
            pad = self.window - T
            feats = np.pad(feats, ((0, pad), (0, 0)))
        else:
            start = np.random.randint(0, T - self.window + 1)
            feats = feats[start: start + self.window]
        head, hold_head, dur = build_frame_labels(frames, holds, start, self.window)
        return (
            torch.from_numpy(feats),
            torch.from_numpy(head),
            torch.from_numpy(hold_head),
            torch.from_numpy(dur),
            torch.tensor(stars / 10.0, dtype=torch.float32),
        )


class SelectionDataset(Dataset):
    """每次返回一段音符序列：(ctx [N,32,82], prev_cols [N,HISTORY], cols [N], stars)

    只存 (path, bid) 索引，__getitem__ 时才读 npz——避免全量特征常驻内存（大数据集会爆）。
    """

    def __init__(self, files: List[str], keys: int = 4, ctx_w: int = CTX_W, seq_n: int = SEQ_N):
        self.keys = keys
        self.ctx_w = ctx_w
        self.seq_n = seq_n
        self.items: List[Tuple[str, str]] = []
        for fn in files:
            path = os.path.join(NPZ_DIR, fn)
            with np.load(path) as z:
                for k in z.files:
                    if not k.startswith("labels_"):
                        continue
                    bid = k[len("labels_"):]
                    if int(z[f"frames_{bid}"].shape[0]) >= 8:
                        self.items.append((path, bid))
        if not self.items:
            raise RuntimeError("SelectionDataset: 没有可用样本，请先运行 build_dataset")

    def __len__(self):
        return len(self.items)

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
        path, bid = self.items[idx]
        with np.load(path) as z:
            feats = z["features"].astype(np.float32)
            frames = z[f"frames_{bid}"]
            cols = z[f"cols_{bid}"]
            stars = float(z[f"stars_{bid}"])
        n = len(frames)
        if n > self.seq_n:
            start = np.random.randint(0, n - self.seq_n + 1)
            frames = frames[start: start + self.seq_n]
            cols = cols[start: start + self.seq_n]
        N = len(frames)
        ctx = np.stack([self._context(feats, int(f)) for f in frames])  # [N, W, 82]
        # 前 HISTORY 列，不足处填 <BOS>=keys
        prev = np.full((N, HISTORY), self.keys, dtype=np.int64)
        for h in range(1, HISTORY + 1):
            prev[h:, -h] = cols[:-h]
        return (
            torch.from_numpy(ctx),
            torch.from_numpy(prev),
            torch.from_numpy(cols.astype(np.int64)),
            torch.tensor(stars / 10.0, dtype=torch.float32),
        )
