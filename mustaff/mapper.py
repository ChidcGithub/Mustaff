"""
谱面映射模块

将音频分析结果（onset、音高、能量）映射为音游音符。
- 横向：onset 时间点 → 音符出现时间
- 纵向：音高/频率 → 轨道列（Column），采用直方图均衡化避免堆积
- 强度：音量/RMS → 音符类型或难度修饰
"""

from typing import List, Dict, Any, Optional
import numpy as np


class BeatMapper:
    """将音频特征映射为音游音符"""

    def __init__(
        self,
        keys: int = 4,
        min_pitch: Optional[float] = None,
        max_pitch: Optional[float] = None,
        use_energy_for_ln: bool = True,
        ln_threshold_ratio: float = 0.7,
        density_filter_ms: float = 30.0,
    ):
        """
        Args:
            keys: 轨道数（4K/6K/7K）
            min_pitch: 最小音高（Hz），自动检测时设为 None
            max_pitch: 最大音高（Hz），自动检测时设为 None
            use_energy_for_ln: 是否根据能量自动生成长按音符（Hold/LN）
            ln_threshold_ratio: 能量超过该比例时生成长按音符
            density_filter_ms: 密度过滤阈值（毫秒），小于该间隔的音符会被过滤
        """
        self.keys = keys
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.use_energy_for_ln = use_energy_for_ln
        self.ln_threshold_ratio = ln_threshold_ratio
        self.density_filter_ms = density_filter_ms

    def map_notes(
        self,
        features: List[Dict[str, Any]],
        auto_range: bool = True,
    ) -> List[Dict[str, Any]]:
        """将特征列表映射为音符列表

        采用"音高排序 + 循环轮询"的直方图均衡化策略，
        确保各列音符数量均匀分布，避免堆积在某一列。

        Args:
            features: AudioAnalyzer.get_note_features() 的输出
            auto_range: 是否根据输入特征自动计算音高范围

        Returns:
            音符列表，每个音符包含：
                - time: 时间（毫秒）
                - column: 轨道列（0-based）
                - type: 'hit' 或 'hold'
                - end_time: hold 的结束时间（毫秒），hit 时为 None
        """
        if not features:
            return []

        # 分离有音高和无音高的音符
        pitched = []
        unpitched = []
        for i, feat in enumerate(features):
            if feat["pitch_hz"] is not None:
                pitched.append((i, feat))
            else:
                unpitched.append((i, feat))

        # ---------- 1. 有音高音符：按音高排序后循环轮询分配列 ----------
        # 这样每列得到的音符数量完全相等（或接近），实现直方图均衡化
        pitched.sort(key=lambda x: x[1]["pitch_hz"])
        pitched_columns = {}
        for rank, (orig_idx, feat) in enumerate(pitched):
            # 循环分配：第0个→列0，第1个→列1，...，第keys个→列0
            col = rank % self.keys
            pitched_columns[orig_idx] = col

        # ---------- 2. 无音高音符：按时间顺序轮询分配到各列 ----------
        unpitched.sort(key=lambda x: x[1]["time_ms"])
        unpitched_columns = {}
        for rank, (orig_idx, feat) in enumerate(unpitched):
            col = rank % self.keys
            unpitched_columns[orig_idx] = col

        # ---------- 3. 合并并按原始顺序处理，进行密度过滤和类型判断 ----------
        rms_values = [f["rms"] for f in features]
        rms_max = max(rms_values) if rms_values else 1.0
        ln_threshold = rms_max * self.ln_threshold_ratio

        notes = []
        last_time_per_column: Dict[int, int] = {}

        for i, feat in enumerate(features):
            time_ms = feat["time_ms"]
            rms = feat["rms"]

            # 获取初始列
            if i in pitched_columns:
                column = pitched_columns[i]
            else:
                column = unpitched_columns[i]

            # 密度过滤：同一列太密时尝试相邻列，都不行才丢弃
            original_col = column
            placed = False
            for offset in [0, -1, 1, -2, 2]:
                test_col = original_col + offset
                if test_col < 0 or test_col >= self.keys:
                    continue
                last_time = last_time_per_column.get(test_col, -999999)
                if time_ms - last_time >= self.density_filter_ms:
                    column = test_col
                    placed = True
                    break

            if not placed:
                continue  # 丢弃过密音符

            # 判断音符类型
            if self.use_energy_for_ln and rms >= ln_threshold:
                duration = int(200 + (rms / rms_max) * 800)
                note_type = "hold"
                end_time = time_ms + duration
            else:
                note_type = "hit"
                end_time = None

            last_time_per_column[column] = time_ms
            notes.append({
                "time": time_ms,
                "column": column,
                "type": note_type,
                "end_time": end_time,
            })

        notes.sort(key=lambda x: x["time"])
        return notes
