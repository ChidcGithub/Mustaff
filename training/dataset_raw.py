"""
PyTorch 数据集 v4-raw（配合 models_raw）

PlacementDatasetRaw：
  返回 (beats [B, RAW_LEN] float32（已除 RAW_NORM_SCALE）, bar_phase [B],
        label [B, 48], cond [2])
  label[b, s] = 第 b 拍第 s 时隙是否有事件（音符头或长音尾）——与 v3 同一标签管线

SelectionDatasetRaw：
  返回 (audio_ctx [N, 5*8*128], prev [N,16], cond [N,9], target [N])
  音频上下文来自 placement-raw 前端输出的拍级嵌入缓存（raw_emb/<set>.npy，
  由 build_embedding_cache 一次性离线计算），标签/增广管线与 v3 逐行同构。

波形矩阵读的是未压缩 .npy 的 mmap 随机窗口——零拷贝，页缓存自动接管热区。
"""

import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import DATA_DIR, NPZ_DIR
from .dataset_v3 import chart_events, events_to_slots, PERM_TABLE
from .models_v3 import SLOTS_PER_BEAT, N_COMBO, BOS, SEL_HISTORY
from .models_raw import RAW_LEN, FEAT_DIM_RAW
from .raw_audio import RAW_NORM_SCALE

RAW_DIR = os.path.join(DATA_DIR, "raw_beats")
EMB_DIR = os.path.join(DATA_DIR, "raw_emb")

PLACE_WINDOW = 256      # placement 训练窗口（拍），与 v3 的 PLACE_WINDOW 同量级
SEL_SEQ = 128           # selection 每段序列的事件数


def _load_chart_meta(path: str, bid: str):
    """只读标签与元数据（不解压 82 维 features，raw 路径用不到）"""
    with np.load(path) as z:
        frames = z[f"frames_{bid}"].astype(np.int64)
        cols = z[f"cols_{bid}"].astype(np.int64)
        holds = z[f"holds_{bid}"].astype(np.int64)
        stars = float(z[f"stars_{bid}"])
        beat_times = z["beat_times"].astype(np.float64) if "beat_times" in z.files else np.empty(0)
        fd_idx = int(z["fd_idx"]) if "fd_idx" in z.files else 0
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
    return frames, cols, holds, stars, beat_times, fd_idx, float(meta.get("bpm", 0.0))


class PlacementDatasetRaw(Dataset):
    def __init__(self, files: List[str], window: int = PLACE_WINDOW,
                 samples_per_epoch: int = 6000, augment: bool = True):
        self.window = window
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.items: List[Tuple[str, str, str]] = []  # (npz_path, raw_path, bid)
        for fn in files:
            path = os.path.join(NPZ_DIR, fn)
            raw_path = os.path.join(RAW_DIR, fn[:-4] + ".npy")
            if not os.path.exists(raw_path):
                continue
            try:
                raw = np.load(raw_path, mmap_mode="r")
            except Exception:
                continue
            with np.load(path) as z:
                if "beat_times" not in z.files or len(z["beat_times"]) < 8:
                    continue
                n_units = len(z["beat_times"]) - 1
                for k in z.files:
                    if k.startswith("labels_") and raw.shape == (n_units, RAW_LEN):
                        self.items.append((path, raw_path, k[len("labels_"):]))
        if not self.items:
            raise RuntimeError("PlacementDatasetRaw: 无样本（先跑 training.build_dataset_raw）")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, _idx):
        path, raw_path, bid = self.items[np.random.randint(len(self.items))]
        frames, cols, holds, stars, beat_times, fd_idx, bpm = _load_chart_meta(path, bid)
        raw = np.load(raw_path, mmap_mode="r")
        n_units = raw.shape[0]

        ev_f, _ = chart_events(frames, cols, holds)
        ev_b, ev_s = events_to_slots(ev_f, beat_times, n_units)
        label = np.zeros((n_units, SLOTS_PER_BEAT), dtype=np.float32)
        ok = ev_b >= 0
        label[ev_b[ok], ev_s[ok]] = 1.0

        start = 0
        if n_units > self.window:
            # 偏向含事件的窗口（与 v3 同策略）
            for _ in range(8):
                s = np.random.randint(0, n_units - self.window + 1)
                if label[s: s + self.window].sum() >= 8:
                    start = s
                    break
            else:
                start = np.random.randint(0, n_units - self.window + 1)

        beats = np.asarray(raw[start: start + self.window], dtype=np.float32) / RAW_NORM_SCALE
        # 训练增广（标签保持）：极性翻转 + 增益抖动——抑制对绝对波形相位的记忆
        if self.augment:
            if np.random.rand() < 0.5:
                beats = -beats
            beats = beats * np.float32(np.random.uniform(0.75, 1.25))
        bar_phase = (np.mod(np.arange(n_units) - fd_idx, 4)
                     / 4.0).astype(np.float32)[start: start + self.window]
        label = label[start: start + self.window]
        cond = np.array([np.clip(bpm / 300.0, 0, 2), stars / 10.0], dtype=np.float32)
        return (
            torch.from_numpy(beats),
            torch.from_numpy(bar_phase),
            torch.from_numpy(label),
            torch.from_numpy(cond),
        )


def collate_place_raw(batch):
    """变长拍序列 padding + pad_mask（True=pad）——波形矩阵版"""
    B = len(batch)
    T = max(x[0].shape[0] for x in batch)
    L = batch[0][0].shape[1]
    beats = torch.zeros(B, T, L)
    bar = torch.zeros(B, T)
    label = torch.zeros(B, T, SLOTS_PER_BEAT)
    mask = torch.ones(B, T, dtype=torch.bool)
    cond = torch.stack([x[3] for x in batch])
    for i, (u, bp, lb, _) in enumerate(batch):
        t = u.shape[0]
        beats[i, :t] = u
        bar[i, :t] = bp
        label[i, :t] = lb
        mask[i, :t] = False
    return beats, bar, label, cond, mask


def build_embedding_cache(files: List[str], device: str, emb_dir: str = EMB_DIR,
                          chunk: int = 512, workers_log_every: int = 100) -> int:
    """用训好的 placement-raw 前端把全部曲集波形 → 拍级嵌入缓存 [n_units, 8, 128] fp16

    selection 训练/评估的数据源。已存在且形状正确的 set 跳过，可重复调用。
    返回本次新构建的 set 数。
    """
    import torch as _t
    from .models_raw import PlacementRaw

    os.makedirs(emb_dir, exist_ok=True)
    ckpt_mtime = os.path.getmtime(ckpt_path)
    marker = os.path.join(emb_dir, ".ckpt_mtime")
    if os.path.exists(marker):
        try:
            with open(marker) as mf:
                if abs(float(mf.read().strip()) - ckpt_mtime) > 1e-6:
                    stale = [f for f in os.listdir(emb_dir) if f.endswith(".npy")]
                    for f in stale:
                        os.remove(os.path.join(emb_dir, f))
                    print(f"[emb] placement ckpt 已更新，清空旧缓存 {len(stale)} 个重建", flush=True)
        except Exception:
            pass
    model = PlacementRaw()
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "output", "placement_raw_best.pt")
    sd = _t.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    frontend = model.frontend

    n_built = 0
    with _t.no_grad():
        for i, fn in enumerate(files):
            path = os.path.join(NPZ_DIR, fn)
            if not os.path.exists(path):
                continue
            out_path = os.path.join(emb_dir, fn[:-4] + ".npy")
            with np.load(path) as z:
                if "beat_times" not in z.files:
                    continue
                n_units = len(z["beat_times"]) - 1
            if n_units < 4:
                continue
            if os.path.exists(out_path):
                try:
                    if np.load(out_path, mmap_mode="r").shape == (n_units, 8, FEAT_DIM_RAW):
                        continue
                except Exception:
                    pass
                os.remove(out_path)
            raw_path = os.path.join(RAW_DIR, fn[:-4] + ".npy")
            if not os.path.exists(raw_path):
                continue
            raw = np.load(raw_path, mmap_mode="r")
            embs = np.zeros((n_units, 8, FEAT_DIM_RAW), dtype=np.float16)
            for s in range(0, n_units, chunk):
                seg = np.asarray(raw[s: s + chunk], dtype=np.float32) / RAW_NORM_SCALE
                x = _t.from_numpy(seg)[None].to(device)
                e = frontend(x)[0].cpu().numpy().astype(np.float16)
                embs[s: s + len(seg)] = e
            np.save(out_path, embs)
            n_built += 1
            if n_built % workers_log_every == 0:
                print(f"  [emb] 已缓存 {n_built} 个 set", flush=True)
    with open(marker, "w") as mf:
        mf.write(str(ckpt_mtime))
    return n_built


class SelectionDatasetRaw(Dataset):
    def __init__(self, files: List[str], keys: int = 4, seq_n: int = SEL_SEQ,
                 ctx_beats: int = 2, mirror: bool = True):
        self.keys = keys
        self.seq_n = seq_n
        self.ctx_beats = ctx_beats
        self.mirror = mirror
        self.items: List[Tuple[str, str]] = []  # (npz_path, emb_path)
        for fn in files:
            path = os.path.join(NPZ_DIR, fn)
            emb_path = os.path.join(EMB_DIR, fn[:-4] + ".npy")
            if not os.path.exists(emb_path):
                continue
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
            raise RuntimeError("SelectionDatasetRaw: 无样本（先跑 build_embedding_cache）")

    def __len__(self):
        return len(self.items)

    def _audio_ctx(self, emb: np.ndarray, beat_idx: int) -> np.ndarray:
        """事件所在拍 ±ctx_beats 的嵌入窗口 → [(2c+1)*8*D]"""
        c = self.ctx_beats
        B = emb.shape[0]
        lo, hi = beat_idx - c, beat_idx + c + 1
        w = np.zeros((2 * c + 1, emb.shape[1], emb.shape[2]), dtype=np.float32)
        src_lo, src_hi = max(0, lo), min(B, hi)
        w[src_lo - lo: src_hi - lo] = emb[src_lo:src_hi]
        return w.reshape(-1)

    def __getitem__(self, idx):
        path, bid = self.items[idx]
        frames, cols, holds, stars, beat_times, fd_idx, bpm = _load_chart_meta(path, bid)
        emb = np.load(os.path.join(EMB_DIR, self._emb_name(path)),
                      mmap_mode="r").astype(np.float32)
        n_units = emb.shape[0]

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

        audio_ctx = np.stack([self._audio_ctx(emb, int(b)) for b in ev_b])
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

    @staticmethod
    def _emb_name(npz_path: str) -> str:
        return os.path.basename(npz_path)[:-4] + ".npy"


# selection 的 collate 与 v3 完全同构（维度从 batch 推断），直接复用 collate_sel_v3
from .dataset_v3 import collate_sel_v3  # noqa: E402,F401
