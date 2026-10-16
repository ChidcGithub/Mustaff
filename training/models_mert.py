"""
模型定义 v5-mert：冻结 MERT 预训练表征 + 复用 v3 节拍网格结构

与 v4-raw 的区别：音频前端从「从零训练的卷积前端」换成「冻结的 MERT-v1-95M
（13 层 hidden states 均值池化，预计算为每拍 8×768 嵌入）」——不参与训练，
无梯度、无显存负担；placement/selection 与 v3 完全同构（n_feats=768）。

数据侧已在 build_dataset_mert 把 MERT 表征预计算为 [n_units, 8, 768] fp16，
因此训练/推理侧不再需要 torch transformers——只有特征构建脚本依赖它。
"""

import torch
import torch.nn as nn

from .models_v3 import BEAT_FEAT_STEPS, PlacementTransformer, count_params

D_MERT = 768       # 与 build_dataset_mert.D_MERT 一致
MERT_STEPS = 8     # 每拍重采样步数（与 BEAT_FEAT_STEPS 一致）


class MertProjection(nn.Module):
    """MERT 每拍单元 [B, T, 8, 768] → 投影 [B, T, 8, d]，再进标准 PlacementTransformer

    输入已 fp16 存储 → float32；768 维对 Transformer 偏大，先投影降维。
    """

    def __init__(self, d_in: int = D_MERT, steps: int = MERT_STEPS,
                 d_model: int = 192, n_layers: int = 4, n_head: int = 8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_model),
            nn.GELU(),
        )
        self.transformer = PlacementTransformer(
            n_feats=d_model, steps=steps, d_model=d_model,
            n_layers=n_layers, n_head=n_head)
        self.d_model = d_model

    def forward(self, units, bar_phase, cond, pad_mask=None):
        """units [B, T, 8, 768] → (event_logits [B, T, 48], diag_pred [B, 2])"""
        h = self.proj(units)
        return self.transformer(h, bar_phase, cond, pad_mask=pad_mask)


def mert_selection_model() -> "SelectionLSTM":  # noqa: F821（运行时导入避免循环）
    from .models_v3 import SelectionLSTM
    return SelectionLSTM(n_feats=D_MERT, steps=MERT_STEPS)


__all__ = ["D_MERT", "MERT_STEPS", "MertProjection", "mert_selection_model",
           "count_params"]
