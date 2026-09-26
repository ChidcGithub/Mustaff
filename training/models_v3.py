"""
模型定义 v3：节拍网格化（ITGPT 路线移植到 4K mania）

与 v2 的根本区别：
- 时间轴从 23ms 帧换成「拍」：每拍 48 时隙（SLOTS_PER_BEAT），
  1/64 以内的细分全部落在合法格子内，标签不再带 timing 噪声
- placement 输出「事件」（音符头 + 长音尾统一）时隙多标签；
  长音时长不再回归——由 selection 在尾事件上放 release 自然涌现
- selection 输出 256 组合（4 列 × {0无,1音符,2长音头,3长音尾}），
  和弦/双押/长音原生支持

PlacementTransformer：拍级音频特征 → Transformer 全局上下文 → 每拍 48 时隙事件 logits
SelectionLSTM：事件序列（音频窗口 ⊕ 前 16 事件组合 embedding ⊕ 拍相位 ⊕ 条件）→ 256 类
"""

import torch
import torch.nn as nn

from .features import FEATURE_DIM

SLOTS_PER_BEAT = 48     # 每拍时隙数（整除 2/3/4/6/8/12/16 → 覆盖到 1/64 音符）
BEAT_FEAT_STEPS = 8     # 每拍重采样的音频特征步数
N_COMBO = 256           # 4 列 × 4 状态
BOS = N_COMBO           # selection 起始符
SEL_HISTORY = 16        # selection 参考的历史事件数


def beats_to_units(feats, beat_times_ms, fd_idx: int = 0, steps: int = BEAT_FEAT_STEPS):
    """帧特征 [T, F] 按拍网格重采样 → (units [B, steps, F], bar_phase [B])

    最后一拍之后/第一拍之前的区域直接丢弃（边缘几秒无拍覆盖）。
    """
    import numpy as np
    from .features import FRAME_MS, BEATS_PER_BAR
    n_beats = len(beat_times_ms) - 1
    if n_beats < 2:
        return np.zeros((0, steps, feats.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.float32)
    times = np.arange(feats.shape[0]) * FRAME_MS  # 帧时刻 ms
    out = np.zeros((n_beats, steps, feats.shape[1]), dtype=np.float32)
    for b in range(n_beats):
        t0, t1 = beat_times_ms[b], beat_times_ms[b + 1]
        ts = t0 + (t1 - t0) * (np.arange(steps) + 0.5) / steps
        idx = np.clip(np.searchsorted(times, ts), 0, feats.shape[0] - 1)
        out[b] = feats[idx]
    bar_pos = np.mod(np.arange(n_beats) - fd_idx, BEATS_PER_BAR).astype(np.float32) / BEATS_PER_BAR
    return out, bar_pos


class BeatFiLM(nn.Module):
    """用 (bpm, stars) 生成 scale/shift 调制拍级特征"""

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(2, dim * 2)

    def forward(self, x, cond):
        gb = self.fc(cond)[:, :, None]  # [B, 2d, 1]
        d = x.shape[1]
        return x * (1 + gb[:, :d]) + gb[:, d:]


class PlacementTransformer(nn.Module):
    """拍级放置：输入 [B, T_beats, steps, F] + (bpm, stars) → 事件 logits [B, T_beats, 48]"""

    def __init__(self, n_feats: int = FEATURE_DIM, steps: int = BEAT_FEAT_STEPS,
                 d_model: int = 192, n_layers: int = 4, n_head: int = 8,
                 max_beats: int = 4096, slots: int = SLOTS_PER_BEAT):
        super().__init__()
        self.slots = slots
        self.steps = steps
        self.inp = nn.Linear(steps * n_feats, d_model)
        self.pos_emb = nn.Embedding(max_beats, d_model)
        self.bar_emb = nn.Embedding(4, d_model)
        self.film = BeatFiLM(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=d_model * 4, dropout=0.1,
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, slots)

    def forward(self, units, bar_phase, cond, pad_mask=None):
        """units [B,T,S,F]，bar_phase [B,T]（0/0.25/0.5/0.75），cond [B,2]，pad_mask [B,T] True=pad"""
        B, T = units.shape[:2]
        h = self.inp(units.reshape(B, T, -1))
        h = h + self.pos_emb.weight[:T][None]
        h = h + self.bar_emb((bar_phase * 4).long().clamp(0, 3))
        h = self.film(h.transpose(1, 2), cond).transpose(1, 2)
        h = self.enc(h, src_key_padding_mask=pad_mask)
        return self.head(h)  # [B, T, 48]


class SelectionLSTM(nn.Module):
    """事件级组合选择：每步 → 256 类（4列×{无,音符,长音头,长音尾}）"""

    def __init__(self, keys: int = 4, n_feats: int = FEATURE_DIM, steps: int = BEAT_FEAT_STEPS,
                 ctx_beats: int = 2, d_audio: int = 256, d_hist: int = 32,
                 hidden: int = 512, n_combo: int = N_COMBO):
        super().__init__()
        self.n_combo = n_combo
        self.ctx_beats = ctx_beats
        audio_in = (2 * ctx_beats + 1) * steps * n_feats
        self.audio_proj = nn.Sequential(nn.Linear(audio_in, d_audio), nn.GELU(), nn.Linear(d_audio, d_audio))
        self.combo_emb = nn.Embedding(n_combo + 1, d_hist, padding_idx=n_combo)  # n_combo=BOS
        self.cond_fc = nn.Linear(2 + 2 + 4 + 1, 32)  # (bpm,stars)+slot sincos+bar onehot+log_ioi
        in_dim = d_audio + d_hist * SEL_HISTORY + 32
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, n_combo)

    def forward(self, audio_ctx, prev_combos, cond_feats, hidden=None):
        """
        audio_ctx:   [B, N, audio_in]   每事件周围 ±ctx_beats 拍的音频特征（已 flatten）
        prev_combos: [B, N, SEL_HISTORY] 前 16 个事件的组合 id（不足处为 BOS）
        cond_feats:  [B, N, 9]          (bpm,stars,slot_sin,slot_cos,bar0..3,log_ioi)
        hidden:      可选 LSTM 隐状态（推理时逐步携带）
        返回 (logits [B, N, 256], hidden)
        """
        a = self.audio_proj(audio_ctx)
        e = self.combo_emb(prev_combos).reshape(prev_combos.shape[0], prev_combos.shape[1], -1)
        c = self.cond_fc(cond_feats)
        step = torch.cat([a, e, c], dim=-1)
        h, hidden = self.lstm(step, hidden)
        return self.head(h), hidden


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
