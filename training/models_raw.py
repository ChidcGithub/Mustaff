"""
模型定义 v4-raw：原始波形前端 + 复用 v3 节拍网格结构

与 v3 的唯一区别在输入侧：不再是 82 维手工特征按拍重采样，而是
「每拍原始波形（RAW_LEN 点）→ 1D 卷积前端 → [steps, D] 拍级嵌入」，
之后进入与 v3 完全相同的 PlacementTransformer（FiLM/诊断头/EMA）。

Selection 不需要新类：SelectionLSTM 已参数化 n_feats/steps，
raw 路径实例化 SelectionLSTM(n_feats=FEAT_DIM_RAW) 即可（audio_proj 输入维度随之改变）。

前端设计（SampleCNN 式分形下采样，~0.2M 参数）：
  Conv(k31,s8) → Conv(k15,s8) → Conv(k9,s4) → Conv(k7,s2)
  4096 → 512 → 64 → 16 → 8，每层 GroupNorm + GELU
  总感受野 ≈2.2k 采样点，覆盖拍窗口过半；GroupNorm 提供逐拍尺度不变性，
  与构建侧 p95 曲集归一叠加，训练/推理口径一致。
"""

import torch
import torch.nn as nn

from .models_v3 import (
    BEAT_FEAT_STEPS, PlacementTransformer, count_params,
)

RAW_LEN = 4096      # 与 raw_audio.RAW_LEN 一致（避免循环导入此处重复声明常量）
FEAT_DIM_RAW = 128  # 前端输出每步嵌入维度


class RawFrontend(nn.Module):
    """每拍原始波形 [B, T, L] → 拍级嵌入 [B, T, steps, d]"""

    def __init__(self, raw_len: int = RAW_LEN, steps: int = BEAT_FEAT_STEPS,
                 d: int = FEAT_DIM_RAW, dropout: float = 0.1):
        super().__init__()
        self.raw_len = raw_len
        self.steps = steps
        self.d = d
        # (in_ch, out_ch, kernel, stride)：4096 → 512 → 64 → 16 → 8
        blocks = [(1, 32, 31, 8), (32, 64, 15, 8), (64, 96, 9, 4), (96, 128, 7, 2)]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        ch_in = 1
        for ch_out, k, s in [(b[1], b[2], b[3]) for b in blocks]:
            pad = (k - 1) // 2  # 同宽填充，保证下采样序列长度恰为整除
            self.convs.append(nn.Conv1d(ch_in, ch_out, k, stride=s, padding=pad))
            self.norms.append(nn.GroupNorm(min(8, ch_out), ch_out))
            ch_in = ch_out
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)  # raw 波形信息密度高，小数据集需要强正则
        if raw_len % (8 * 8 * 4 * 2) != 0:
            raise ValueError(f"raw_len={raw_len} 不能被前端总步长 512 整除")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, L] float（已除 RAW_NORM_SCALE）→ [B, T, steps, d]"""
        B, T, L = x.shape
        s = x.reshape(B * T, 1, L)
        for conv, norm in zip(self.convs, self.norms):
            s = self.drop(self.act(norm(conv(s))))
        return s.reshape(B, T, self.steps, self.d)


class PlacementRaw(nn.Module):
    """RawFrontend + v3 PlacementTransformer（结构/诊断头/FiLM 原样复用）

    checkpoint 直接保存整个模块的 state_dict（键前缀 frontend. / transformer.），
    raw_available() 用键前缀区分 v3/v2/raw 三代权重。
    """

    def __init__(self, d_model: int = 192, n_layers: int = 4, n_head: int = 8,
                 raw_len: int = RAW_LEN, steps: int = BEAT_FEAT_STEPS,
                 d: int = FEAT_DIM_RAW):
        super().__init__()
        self.frontend = RawFrontend(raw_len=raw_len, steps=steps, d=d)
        self.transformer = PlacementTransformer(
            n_feats=d, steps=steps, d_model=d_model, n_layers=n_layers, n_head=n_head)

    def forward(self, beats, bar_phase, cond, pad_mask=None):
        """beats [B, T, L]（已归一化浮点）→ (event_logits [B, T, 48], diag_pred [B, 2])"""
        h = self.frontend(beats)
        return self.transformer(h, bar_phase, cond, pad_mask=pad_mask)


def raw_selection_model() -> "SelectionLSTM":  # noqa: F821（运行时导入避免循环）
    from .models_v3 import SelectionLSTM
    return SelectionLSTM(n_feats=FEAT_DIM_RAW)


__all__ = ["RAW_LEN", "FEAT_DIM_RAW", "RawFrontend", "PlacementRaw",
           "raw_selection_model", "count_params"]
