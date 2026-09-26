"""
PyTorch 数据集 v3：节拍网格标签（配合 models_v3）

PlacementDatasetV3：
  返回 (units [B,8,82], bar_phase [B], label [B,48], cond [2], pad 由 collate 处理)
  label[b, s] = 第 b 拍第 s 时隙是否有事件（音符头或长音尾）

SelectionDatasetV3：
  返回 (audio_ctx [N, 5*8*82], prev [N,16], cond [N,9], target [N])
  target = 256 组合 id（4 列 base-4 编码：0无 1音符 2长音头 3长音尾）
  支持列镜像/任意排列增广（24 种排列保持手法合法性）

两个数据集都是懒加载（只存 (path, bid) 索引），加载时对 mel/chroma 通道做逐歌归一化。
"""

import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import NPZ_DIR, load_split  # noqa: F401  (复用路径与划分)
from .features import FRAME_MS, BEATS_PER_BAR
from .models_v3 import (
    SLOTS_PER_BEAT, BEAT_FEAT_STEPS, N_COMBO, BOS, SEL_HISTORY, beats_to_units,
)

NORM_CH = 76            # 仅前 76 通道（mel+chroma）做逐歌归一化
PLACE_WINDOW = 256      # placement 训练窗口（拍）
SEL_SEQ = 128           # selection 每段序列的事件数


def chart_events(frames: np.ndarray, cols: np.ndarray, holds: np.ndarray):
    """chart 原始数组 → (event_frames [E], combo_ids [E])

    事件 = 音符头 ∪ 长音尾；每个事件编码 4 列状态（base-4: 0无/1音符/2长音头/3长音尾）
    同列同时刻头尾相撞时优先记头。
    """
    times = sorted(set(frames.tolist()) | set(h for h in holds.tolist() if h > 0))
    head_at = {}
    for f, c, h in zip(frames.tolist(), cols.tolist(), holds.tolist()):
        head_at.setdefault(f, {})[c] = 2 if h > f else 1
    tail_at = {}
    for f, c, h in zip(frames.tolist(), cols.tolist(), holds.tolist()):
        if h > f:
            tail_at.setdefault(h, {})[c] = 3
    combos = []
    for t in times:
        states = [0, 0, 0, 0]
        for c, s in tail_at.get(t, {}).items():
            states[c] = s
        for c, s in head_at.get(t, {}).items():
            states[c] = s  # 头优先
        combo = states[0] + states[1] * 4 + states[2] * 16 + states[3] * 64
        combos.append(combo)
    return np.array(times, dtype=np.int64), np.array(combos, dtype=np.int64)


def events_to_slots(event_frames: np.ndarray, beat_times: np.ndarray, n_units: int):
    """事件帧号 → (beat_idx, slot) 数组；落在拍网格外的返回处为 -1"""
    if n_units < 1 or len(beat_times) < 2:
        return np.full(len(event_frames), -1), np.zeros(len(event_frames), dtype=np.int64)
    t_ms = event_frames.astype(np.float64) * FRAME_MS
    b = np.searchsorted(beat_times, t_ms, side="right") - 1
    b = np.clip(b, 0, len(beat_times) - 2)
    t0 = beat_times[b]
    t1 = beat_times[b + 1]
    slot = np.round((t_ms - t0) / np.maximum(t1 - t0, 1e-6) * SLOTS_PER_BEAT).astype(np.int64)
    slot = np.clip(slot, 0, SLOTS_PER_BEAT - 1)
    valid = (b < n_units) & (t_ms >= beat_times[0]) & (t_ms <= beat_times[-1])
    return np.where(valid, b, -1), slot


def normalize_feats(feats: np.ndarray) -> np.ndarray:
    """mel/chroma 通道逐歌 0 均值 1 方差（ITGPT 式，缩小歌曲间音色差）"""
    out = feats.astype(np.float32).copy()
    mu = out[:, :NORM_CH].mean(axis=0, keepdims=True)
    sd = out[:, :NORM_CH].std(axis=0, keepdims=True) + 1e-5
    out[:, :NORM_CH] = (out[:, :NORM_CH] - mu) / sd
    return out


# 24 种列排列 × 256 组合的查表（镜像/排列增广用）
def _build_perm_table() -> torch.Tensor:
    import itertools
    perms = list(itertools.permutations(range(4)))
    table = torch.zeros(len(perms), N_COMBO, dtype=torch.long)
    for pi, p in enumerate(perms):
        for combo in range(N_COMBO):
            digits = [(combo >> (2 * c)) & 3 for c in range(4)]
            new = [0, 0, 0, 0]
            for c in range(4):
                new[p[c]] = digits[c]
            table[pi, combo] = sum(new[c] << (2 * c) for c in range(4))
    return table


PERM_TABLE = _build_perm_table()


def _load_chart(path: str, bid: str):
    with np.load(path) as z:
        feats = normalize_feats(z["features"])
        frames = z[f"frames_{bid}"].astype(np.int64)
        cols = z[f"cols_{bid}"].astype(np.int64)
        holds = z[f"holds_{bid}"].astype(np.int64)
        stars = float(z[f"stars_{bid}"])
        beat_times = z["beat_times"].astype(np.float64) if "beat_times" in z.files else np.empty(0)
        fd_idx = int(z["fd_idx"]) if "fd_idx" in z.files else 0
        meta = __import__("json").loads(bytes(z["meta"]).decode("utf-8"))
    return feats, frames, cols, holds, stars, beat_times, fd_idx, float(meta.get("bpm", 0.0))


class PlacementDatasetV3(Dataset):
    def __init__(self, files: List[str], window: int = PLACE_WINDOW, samples_per_epoch: int = 6000):
        self.window = window
        self.samples_per_epoch = samples_per_epoch
        self.items: List[Tuple[str, str]] = []
        for fn in files:
            path = os.path.join(NPZ_DIR, fn)
            with np.load(path) as z:
                if "beat_times" not in z.files or len(z["beat_times"]) < 8:
                    continue
                for k in z.files:
                    if k.startswith("labels_"):
                        self.items.append((path, k[len("labels_"):]))
        if not self.items:
            raise RuntimeError("PlacementDatasetV3: 无样本（缺 beat_times？先跑 --backfill-beats）")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, _idx):
        path, bid = self.items[np.random.randint(len(self.items))]
        feats, frames, cols, holds, stars, beat_times, fd_idx, bpm = _load_chart(path, bid)
        units, bar_phase = beats_to_units(feats, beat_times, fd_idx)  # [B,8,82]
        n_units = units.shape[0]
        ev_f, _ = chart_events(frames, cols, holds)
        ev_b, ev_s = events_to_slots(ev_f, beat_times, n_units)

        label = np.zeros((n_units, SLOTS_PER_BEAT), dtype=np.float32)
        ok = ev_b >= 0
        label[ev_b[ok], ev_s[ok]] = 1.0

        if n_units > self.window:
            # 偏向含事件的窗口（纯长空白段学不到东西）
            for _ in range(8):
                start = np.random.randint(0, n_units - self.window + 1)
                if label[start: start + self.window].sum() >= 8:
                    break
            units = units[start: start + self.window]
            bar_phase = bar_phase[start: start + self.window]
            label = label[start: start + self.window]

        cond = np.array([np.clip(bpm / 300.0, 0, 2), stars / 10.0], dtype=np.float32)
        return (
            torch.from_numpy(units),
            torch.from_numpy(bar_phase),
            torch.from_numpy(label),
            torch.from_numpy(cond),
        )


def collate_place_v3(batch):
    """变长拍序列 padding + pad_mask（True=pad）"""
    B = len(batch)
    T = max(x[0].shape[0] for x in batch)
    S, F = batch[0][0].shape[1], batch[0][0].shape[2]
    units = torch.zeros(B, T, S, F)
    bar = torch.zeros(B, T)
    label = torch.zeros(B, T, SLOTS_PER_BEAT)
    mask = torch.ones(B, T, dtype=torch.bool)
    cond = torch.stack([x[3] for x in batch])
    for i, (u, bp, lb, _) in enumerate(batch):
        t = u.shape[0]
        units[i, :t] = u
        bar[i, :t] = bp
        label[i, :t] = lb
        mask[i, :t] = False
    return units, bar, label, cond, mask


class SelectionDatasetV3(Dataset):
    def __init__(self, files: List[str], keys: int = 4, seq_n: int = SEL_SEQ,
                 ctx_beats: int = 2, mirror: bool = True):
        self.keys = keys
        self.seq_n = seq_n
        self.ctx_beats = ctx_beats
        self.mirror = mirror
        self.items: List[Tuple[str, str]] = []
        for fn in files:
            path = os.path.join(NPZ_DIR, fn)
            with np.load(path) as z:
                if "beat_times" not in z.files or len(z["beat_times"]) < 8:
                    continue
                for k in z.files:
                    if not k.startswith("labels_"):
                        continue
                    bid = k[len("labels_"):]
                    if int(z[f"frames_{bid}"].shape[0]) >= 8:
                        self.items.append((path, bid))
        if not self.items:
            raise RuntimeError("SelectionDatasetV3: 无样本（缺 beat_times？先跑 --backfill-beats）")

    def __len__(self):
        return len(self.items)

    def _audio_ctx(self, units: np.ndarray, beat_idx: int) -> np.ndarray:
        """事件所在拍 ±ctx_beats 的音频窗口 → [(2c+1)*8*82]"""
        c = self.ctx_beats
        B = units.shape[0]
        lo, hi = beat_idx - c, beat_idx + c + 1
        w = np.zeros((2 * c + 1, units.shape[1], units.shape[2]), dtype=np.float32)
        src_lo, src_hi = max(0, lo), min(B, hi)
        w[src_lo - lo: src_hi - lo] = units[src_lo:src_hi]
        return w.reshape(-1)

    def __getitem__(self, idx):
        path, bid = self.items[idx]
        feats, frames, cols, holds, stars, beat_times, fd_idx, bpm = _load_chart(path, bid)
        units, _ = beats_to_units(feats, beat_times, fd_idx)
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
        bar_idx = np.mod(ev_b - fd_idx, BEATS_PER_BAR)
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


def collate_sel_v3(batch):
    """变长事件序列 padding，target 用 -100（CE ignore）"""
    B = len(batch)
    N = max(x[1].shape[0] for x in batch)
    A = batch[0][0].shape[1]
    audio = torch.zeros(B, N, A)
    prev = torch.full((B, N, SEL_HISTORY), BOS, dtype=torch.long)
    cond = torch.zeros(B, N, 9)
    target = torch.full((B, N), -100, dtype=torch.long)
    for i, (a, p, c, t) in enumerate(batch):
        n = p.shape[0]
        audio[i, :n] = a
        prev[i, :n] = p
        cond[i, :n] = c
        target[i, :n] = t
    return audio, prev, cond, target
