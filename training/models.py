"""
模型定义

PlacementNet：每帧是否放音符（难度条件化，FiLM 调制）
  输入 [B, 65, T]（log-mel 64 + onset 1）→ 每帧 logit
  参照 DDC/TaikoNation 的 CNN+RNN placement 思路，规模按 4060 8G 可控设计

SelectionNet：给每个音符选列（自回归，teacher forcing 训练）
  每步输入 = 音频上下文（CNN 编码）+ 前一个音符列 embedding + 星级
  LSTM → 4 类 logit
"""

import torch
import torch.nn as nn

N_FEATS = 65  # 64 log-mel + 1 onset strength


class FiLM(nn.Module):
    """用难度星级生成 scale/shift 调制特征通道"""

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(1, channels * 2)

    def forward(self, x: torch.Tensor, stars: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T], stars: [B]（已归一化到 ~[0,1]）
        gamma_beta = self.fc(stars.unsqueeze(-1))[:, :, None]  # [B, 2C, 1]
        C = x.shape[1]
        gamma = gamma_beta[:, :C]
        beta = gamma_beta[:, C:]
        return x * (1 + gamma) + beta


class PlacementNet(nn.Module):
    def __init__(self, n_feats: int = N_FEATS, channels: int = 128, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feats, channels, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.GELU(),
        )
        self.film = FiLM(channels)
        self.lstm = nn.LSTM(
            channels, hidden, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.1,
        )
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor, stars: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 65]（帧优先），stars: [B] → logits [B, T]"""
        h = x.transpose(1, 2)          # [B, 65, T]
        h = self.conv(h)               # [B, C, T]
        h = self.film(h, stars)
        h = h.transpose(1, 2)          # [B, T, C]
        h, _ = self.lstm(h)            # [B, T, 2H]
        return self.head(h).squeeze(-1)  # [B, T]


class ContextCNN(nn.Module):
    """把音符周围的音频窗口 [B, W, 65] 编码成固定向量"""

    def __init__(self, n_feats: int = N_FEATS, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feats, 64, 5, padding=2), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        h = self.net(w.transpose(1, 2)).squeeze(-1)
        return self.fc(h)  # [B, out_dim]


class SelectionNet(nn.Module):
    """自回归列选择：step 输入 = 上下文 ⊕ 上一列 ⊕ 星级 → 当前列分布"""

    def __init__(self, keys: int = 4, ctx_dim: int = 128, hidden: int = 192):
        super().__init__()
        self.keys = keys
        self.ctx_encoder = ContextCNN(out_dim=ctx_dim)
        self.col_emb = nn.Embedding(keys + 1, 32, padding_idx=keys)  # keys 作为 <BOS>
        self.stars_fc = nn.Linear(1, 16)
        self.lstm = nn.LSTM(ctx_dim + 32 + 16, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, keys)

    def forward(self, ctx_windows: torch.Tensor, prev_cols: torch.Tensor, stars: torch.Tensor) -> torch.Tensor:
        """
        ctx_windows: [B, N, W, 65]  每个音符的音频窗口
        prev_cols:   [B, N]         每个音符的前一列（首音符为 keys=<BOS>）
        stars:       [B]
        返回 logits: [B, N, keys]
        """
        B, N, W, F = ctx_windows.shape
        ctx = self.ctx_encoder(ctx_windows.reshape(B * N, W, F)).reshape(B, N, -1)
        emb = self.col_emb(prev_cols)                    # [B, N, 32]
        s = self.stars_fc(stars[:, None])[:, None, :].expand(B, N, -1)  # [B, N, 16]
        step = torch.cat([ctx, emb, s], dim=-1)
        h, _ = self.lstm(step)
        return self.head(h)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
