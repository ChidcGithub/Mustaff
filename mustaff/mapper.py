"""
谱面映射模块

将音频分析结果（onset、音高、能量）映射为音游音符。
- 横向：onset 时间点 → 音符出现时间
- 纵向：音高/频率 → 轨道列（Column），采用时序感知的直方图均衡化策略
- 强度：音量/RMS → 音符类型或难度修饰
"""

from typing import List, Dict, Any, Optional, Set
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
        chord_gap_ms: float = 50.0,
        jack_threshold_ms: float = 100.0,
        min_hold_ms: float = 200.0,
        window_size_s: float = 2.0,
        hold_position_strength: float = 0.2,
    ):
        """
        Args:
            keys: 轨道数（4K/6K/7K）
            min_pitch: 最小音高（Hz），自动检测时设为 None
            max_pitch: 最大音高（Hz），自动检测时设为 None
            use_energy_for_ln: 是否根据能量自动生成长按音符（Hold/LN）
            ln_threshold_ratio: 能量超过该比例时生成长按音符
            density_filter_ms: 密度过滤阈值（毫秒），小于该间隔的音符会被过滤
            chord_gap_ms: 和弦判定间隔（毫秒），该间隔内的多个音符视为和弦
            jack_threshold_ms: 同列最小间隔（毫秒），小于此值会尝试偏移
            min_hold_ms: 最小长按时长（毫秒）
            window_size_s: 时序分段窗口大小（秒），用于局部音高排序
            hold_position_strength: 强弱位置权重（0~1）。值越大，
                音频开头/结尾的 hold 越少，中间越多。0 表示关闭。
        """
        self.keys = keys
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.use_energy_for_ln = use_energy_for_ln
        self.ln_threshold_ratio = ln_threshold_ratio
        self.density_filter_ms = density_filter_ms
        self.chord_gap_ms = chord_gap_ms
        self.jack_threshold_ms = jack_threshold_ms
        self.min_hold_ms = min_hold_ms
        self.window_size_s = window_size_s
        self.hold_position_strength = hold_position_strength

    def map_notes(
        self,
        features: List[Dict[str, Any]],
        auto_range: bool = True,
    ) -> List[Dict[str, Any]]:
        """将特征列表映射为音符列表

        算法流程：
        1. 和弦检测 — 时间上极近的音符合并为和弦，分配到相邻列
        2. 列分配 — 按时序分段，段内按音高排序 + 循环轮询（局部直方图均衡化）
        3. Jack 防护 — 同列过密时主动偏移到邻列
        4. 密度过滤 — 确保最小列间距，节拍上的音符优先保留
        5. 智能 Hold — 根据与下一个 onset 的间距决定 hold 时长

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

        # 按时间排序
        sorted_indices = sorted(
            range(len(features)), key=lambda i: features[i]["time_ms"]
        )
        sorted_features = [features[i] for i in sorted_indices]

        n = len(sorted_features)

        # Step 1: 和弦检测
        chords = self._detect_chords(sorted_features)
        chord_positions: Set[int] = set()
        for chord in chords:
            chord_positions.update(chord)

        # Step 2: 列分配（和弦 → 相邻列；非和弦 → 时序分段 + 音高排序）
        assignments = self._assign_columns(sorted_features, chords, chord_positions)

        # Step 3: Jack 防护
        assignments = self._prevent_jacks(sorted_features, assignments)

        # Step 4: 密度过滤（节拍优先）
        valid_positions = self._filter_by_density(sorted_features, assignments)

        # Step 5: 生成音符（智能 Hold 时长）
        notes = self._generate_notes(sorted_features, assignments, valid_positions)

        # 恢复原始顺序
        result = [None] * n
        for i, note in enumerate(notes):
            original_idx = sorted_indices[valid_positions[i]]
            result[original_idx] = note

        return [r for r in result if r is not None]

    # ---------- Step 1: 和弦检测 ----------

    def _detect_chords(self, features: List[Dict[str, Any]]) -> List[List[int]]:
        """检测和弦：连续 notes 间隔 <= chord_gap_ms 则合并为一个和弦"""
        chords: List[List[int]] = []
        i = 0
        while i < len(features):
            chord = [i]
            j = i + 1
            while j < len(features) and features[j]["time_ms"] - features[j - 1]["time_ms"] <= self.chord_gap_ms:
                chord.append(j)
                j += 1
            if len(chord) > 1:
                chords.append(chord)
                i = j
            else:
                i += 1
        return chords

    # ---------- Step 2: 列分配 ----------

    def _assign_columns(
        self,
        features: List[Dict[str, Any]],
        chords: List[List[int]],
        chord_positions: Set[int],
    ) -> List[int]:
        """分配列号

        - 和弦音符：分配到相邻列，按音高排序
        - 非和弦音符：按时序分段，段内按音高排序后轮询分配
        """
        n = len(features)
        assignments = [-1] * n

        # 2a. 和弦：分配到相邻列
        for chord in chords:
            chord_sorted = sorted(chord, key=lambda i: features[i]["pitch_hz"] or 0)

            if len(chord) > self.keys:
                for rank, idx in enumerate(chord_sorted):
                    assignments[idx] = rank % self.keys
            else:
                center = self._pitch_to_center_column(features, chord_sorted)
                start_col = max(
                    0, min(self.keys - len(chord), center - len(chord) // 2)
                )
                for rank, idx in enumerate(chord_sorted):
                    assignments[idx] = start_col + rank

        # 2b. 非和弦：时序分段 + 局部音高排序
        non_chord = [i for i in range(n) if i not in chord_positions]
        if not non_chord:
            return assignments

        window_size_ms = int(self.window_size_s * 1000)
        if window_size_ms < 1:
            window_size_ms = 1000

        seg_start = 0
        while seg_start < len(non_chord):
            seg_end = seg_start + 1
            t_start = features[non_chord[seg_start]]["time_ms"]
            while seg_end < len(non_chord):
                if features[non_chord[seg_end]]["time_ms"] - t_start < window_size_ms:
                    seg_end += 1
                else:
                    break

            seg_indices = non_chord[seg_start:seg_end]
            # 段内按音高排序，循环轮询分配
            seg_indices.sort(key=lambda i: features[i]["pitch_hz"] or 0)
            for rank, idx in enumerate(seg_indices):
                assignments[idx] = rank % self.keys

            seg_start = seg_end

        return assignments

    def _pitch_to_center_column(
        self,
        features: List[Dict[str, Any]],
        chord_indices: List[int],
    ) -> int:
        """根据和弦的平均音高映射到居中的列"""
        pitches = [features[i]["pitch_hz"] or 0 for i in chord_indices]
        avg_pitch = sum(pitches) / len(pitches)

        all_pitches = [
            features[i]["pitch_hz"]
            for i in range(len(features))
            if features[i]["pitch_hz"] is not None
        ]
        if all_pitches:
            p_min, p_max = min(all_pitches), max(all_pitches)
            norm = (avg_pitch - p_min) / (p_max - p_min) if p_max > p_min else 0.5
        else:
            norm = 0.5

        col_range = max(1, self.keys - len(chord_indices))
        return int(norm * col_range)

    # ---------- Step 3: Jack 防护 ----------

    def _prevent_jacks(
        self,
        features: List[Dict[str, Any]],
        assignments: List[int],
    ) -> List[int]:
        """同列过密时尝试将后一个音符移到邻列"""
        assignments = list(assignments)

        for col in range(self.keys):
            last_time = -999999
            i = 0
            while i < len(features):
                if assignments[i] != col:
                    i += 1
                    continue
                time_ms = features[i]["time_ms"]
                if time_ms - last_time < self.jack_threshold_ms:
                    moved = False
                    for offset in [1, -1, 2, -2]:
                        new_col = col + offset
                        if new_col < 0 or new_col >= self.keys:
                            continue
                        has_conflict = any(
                            abs(features[j]["time_ms"] - time_ms) < self.density_filter_ms
                            for j in range(len(features))
                            if assignments[j] == new_col
                        )
                        if not has_conflict:
                            assignments[i] = new_col
                            moved = True
                            break
                    if not moved:
                        last_time = time_ms
                else:
                    last_time = time_ms
                i += 1

        return assignments

    # ---------- Step 4: 密度过滤 ----------

    def _filter_by_density(
        self,
        features: List[Dict[str, Any]],
        assignments: List[int],
    ) -> List[int]:
        """密度过滤，节拍上的音符获得更高保留优先级"""
        last_time_per_column: Dict[int, int] = {}
        valid: List[int] = []

        for i in range(len(features)):
            time_ms = features[i]["time_ms"]
            column = assignments[i]

            if column < 0:
                continue

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

            if placed:
                last_time_per_column[column] = time_ms
                valid.append(i)
            else:
                # 节拍上的音符放宽保留条件
                if features[i].get("near_beat", False):
                    best_col = original_col
                    best_gap = 0
                    for offset in [0, -1, 1, -2, 2]:
                        test_col = original_col + offset
                        if test_col < 0 or test_col >= self.keys:
                            continue
                        last_time = last_time_per_column.get(test_col, -999999)
                        gap = time_ms - last_time
                        if gap > best_gap:
                            best_gap = gap
                            best_col = test_col
                    if best_gap >= self.density_filter_ms * 0.5:
                        last_time_per_column[best_col] = time_ms
                        valid.append(i)

        return valid

    # ---------- Step 5: 生成音符 ----------

    def _generate_notes(
        self,
        features: List[Dict[str, Any]],
        assignments: List[int],
        valid_indices: List[int],
    ) -> List[Dict[str, Any]]:
        """生成音符列表，包含智能 Hold 时长判断和位置权重"""
        notes: List[Dict[str, Any]] = []
        rms_values = [features[i]["rms"] for i in valid_indices]
        rms_max = max(rms_values) if rms_values else 1.0
        ln_threshold = rms_max * self.ln_threshold_ratio

        # 计算位置权重所需的时间范围
        if self.hold_position_strength > 0 and len(valid_indices) > 1:
            all_times = [features[i]["time_ms"] for i in valid_indices]
            t_min = min(all_times)
            t_range = max(all_times) - t_min
            if t_range < 1:
                t_range = 1
        else:
            t_min = 0.0
            t_range = 1.0

        for pos, idx in enumerate(valid_indices):
            feat = features[idx]
            time_ms = feat["time_ms"]
            column = assignments[idx]
            rms = feat["rms"]

            # 位置权重：开头/结尾阈值高，中间阈值低
            if self.hold_position_strength > 0:
                norm = (time_ms - t_min) / t_range  # 0~1
                weight = 1.0 + self.hold_position_strength * (8.0 * (norm - 0.5) ** 2 - 1.0)
                effective_threshold = ln_threshold * weight
            else:
                effective_threshold = ln_threshold

            note_type = "hit"
            end_time = None

            if self.use_energy_for_ln and rms >= effective_threshold:
                if pos + 1 < len(valid_indices):
                    next_time = features[valid_indices[pos + 1]]["time_ms"]
                    gap = next_time - time_ms

                    if gap >= self.min_hold_ms:
                        end_time = time_ms + int(gap * 0.8)
                        note_type = "hold"
                else:
                    duration = int(200 + (rms / rms_max) * 800)
                    if duration >= self.min_hold_ms:
                        end_time = time_ms + duration
                        note_type = "hold"

            notes.append({
                "time": time_ms,
                "column": column,
                "type": note_type,
                "end_time": end_time,
            })

        notes.sort(key=lambda x: x["time"])
        return notes
