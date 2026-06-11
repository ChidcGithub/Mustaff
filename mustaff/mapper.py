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
        snap_to_beat: bool = False,
        snap_resolution: int = 8,
        complexity: float = 1.0,
        ln_tendency: float = 0.5,
        speed_variation: float = 0.5,
        speed_smoothing: int = 5,
        contrast: float = 1.0,
    ):
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
        self.snap_to_beat = snap_to_beat
        self.snap_resolution = snap_resolution
        self.complexity = complexity
        self.ln_tendency = ln_tendency
        self.speed_variation = max(0.0, min(1.0, speed_variation))
        self.speed_smoothing = max(1, speed_smoothing)
        self.contrast = max(0.1, min(3.5, contrast))

    def map_notes(
        self,
        features: List[Dict[str, Any]],
        auto_range: bool = True,
        beat_subdivisions: Optional[np.ndarray] = None,
        rms_full: Optional[np.ndarray] = None,
        pitches_full: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        if not features:
            return []

        sorted_indices = sorted(
            range(len(features)), key=lambda i: features[i]["time_ms"]
        )
        sorted_features = [features[i] for i in sorted_indices]

        n = len(sorted_features)

        chords = self._detect_chords(sorted_features)
        chord_positions: Set[int] = set()
        for chord in chords:
            chord_positions.update(chord)

        assignments = self._assign_columns(sorted_features, chords, chord_positions)

        assignments = self._prevent_jacks(sorted_features, assignments)

        valid_positions = self._filter_by_density(sorted_features, assignments)

        notes = self._generate_notes(
            sorted_features, assignments, valid_positions,
            rms_full=rms_full, pitches_full=pitches_full,
        )

        if self.snap_to_beat and beat_subdivisions is not None and len(beat_subdivisions) > 0:
            notes = self._snap_notes(notes, beat_subdivisions)

        self._calculate_speeds(notes, sorted_features, valid_positions)

        result = [None] * n
        for i, note in enumerate(notes):
            original_idx = sorted_indices[valid_positions[i]]
            result[original_idx] = note

        return [r for r in result if r is not None]

    def _snap_notes(
        self,
        notes: List[Dict[str, Any]],
        beat_subdivisions: np.ndarray,
    ) -> List[Dict[str, Any]]:
        if not notes:
            return []

        snapped: List[Dict[str, Any]] = []
        for note in notes:
            t = note["time"] / 1000.0
            idx = np.argmin(np.abs(beat_subdivisions - t))
            snapped_time = int(round(beat_subdivisions[idx] * 1000))

            snapped_note = dict(note)
            snapped_note["time"] = snapped_time
            if snapped_note.get("end_time") is not None:
                end_t = snapped_note["end_time"] / 1000.0
                idx_end = np.argmin(np.abs(beat_subdivisions - end_t))
                snapped_note["end_time"] = int(round(beat_subdivisions[idx_end] * 1000))

            snapped.append(snapped_note)

        merged: List[Dict[str, Any]] = []
        time_col_map: Dict[tuple, Dict] = {}
        for note in snapped:
            key = (note["time"], note["column"])
            if key in time_col_map:
                existing = time_col_map[key]
                if note["type"] == "hold" and existing["type"] == "hit":
                    existing["type"] = "hold"
                    existing["end_time"] = note["end_time"]
            else:
                time_col_map[key] = dict(note)

        merged = sorted(time_col_map.values(), key=lambda x: x["time"])
        return merged

    def _detect_chords(self, features: List[Dict[str, Any]]) -> List[List[int]]:
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

    def _assign_columns(
        self,
        features: List[Dict[str, Any]],
        chords: List[List[int]],
        chord_positions: Set[int],
    ) -> List[int]:
        n = len(features)
        assignments = [-1] * n

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
            seg_indices.sort(key=lambda i: features[i]["pitch_hz"] or 0)

            used_columns = set()
            band_positions: Dict[int, List[int]] = {}
            for idx in seg_indices:
                band = features[idx].get("band", 0)
                band_positions.setdefault(band, []).append(idx)

            sorted_bands = sorted(band_positions.items())
            for band, band_indices in sorted_bands:
                for idx in band_indices:
                    for col in range(self.keys):
                        if col not in used_columns:
                            assignments[idx] = col
                            used_columns.add(col)
                            break

            unassigned = [i for i in seg_indices if assignments[i] < 0]
            for rank, idx in enumerate(unassigned):
                assignments[idx] = rank % self.keys

            seg_start = seg_end

        return assignments

    def _pitch_to_center_column(
        self,
        features: List[Dict[str, Any]],
        chord_indices: List[int],
    ) -> int:
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

    def _prevent_jacks(
        self,
        features: List[Dict[str, Any]],
        assignments: List[int],
    ) -> List[int]:
        adjusted_threshold = self.jack_threshold_ms / max(0.1, self.complexity)
        assignments = list(assignments)

        col_indices: Dict[int, List[int]] = {c: [] for c in range(self.keys)}
        for i, col in enumerate(assignments):
            if 0 <= col < self.keys:
                col_indices[col].append(i)

        for col in range(self.keys):
            last_time = -999999
            i = 0
            while i < len(features):
                if assignments[i] != col:
                    i += 1
                    continue
                time_ms = features[i]["time_ms"]
                if time_ms - last_time < adjusted_threshold:
                    moved = False
                    for offset in [1, -1, 2, -2]:
                        new_col = col + offset
                        if new_col < 0 or new_col >= self.keys:
                            continue
                        has_conflict = any(
                            abs(features[j]["time_ms"] - time_ms) < self.density_filter_ms
                            for j in col_indices[new_col]
                        )
                        if not has_conflict:
                            col_indices[col].remove(i)
                            col_indices[new_col].append(i)
                            assignments[i] = new_col
                            moved = True
                            break
                    if not moved:
                        last_time = time_ms
                else:
                    last_time = time_ms
                i += 1

        return assignments

    def _filter_by_density(
        self,
        features: List[Dict[str, Any]],
        assignments: List[int],
    ) -> List[int]:
        adjusted_density = self.density_filter_ms / max(0.1, self.complexity)
        adjusted_density_beat = adjusted_density * 0.5

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
                if time_ms - last_time >= adjusted_density:
                    column = test_col
                    placed = True
                    break

            if placed:
                last_time_per_column[column] = time_ms
                valid.append(i)
            else:
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
                    if best_gap >= adjusted_density_beat:
                        last_time_per_column[best_col] = time_ms
                        valid.append(i)

        return valid

    def _calc_sustain_score(
        self, rms_slice: np.ndarray, pitches_slice: Optional[np.ndarray],
        rms_global_max: float,
    ) -> float:
        if len(rms_slice) < 2:
            return 0.0
        avg_rms = float(np.mean(rms_slice))
        min_rms = float(np.min(rms_slice))
        rms_base = max(rms_global_max, 1e-6)
        avg_ratio = avg_rms / rms_base
        min_ratio = min_rms / rms_base
        energy_score = avg_ratio * 0.6 + min_ratio * 0.4

        # 对比度调整能量分数
        energy_score = np.clip((energy_score - 0.5) * self.contrast + 0.5, 0.0, 1.0)

        stability = 0.5
        if pitches_slice is not None and len(pitches_slice) > 0:
            valid_p = pitches_slice[~np.isnan(pitches_slice)]
            if len(valid_p) > 1:
                p_std = float(np.std(valid_p))
                stability = 1.0 - min(p_std / 12.0, 1.0)
        return energy_score * 0.7 + stability * 0.3

    def _calculate_speeds(
        self,
        notes: List[Dict[str, Any]],
        features: List[Dict[str, Any]],
        valid_positions: List[int],
    ) -> None:
        """根据能量和密度动态计算每个音符的下落速度（原地修改 notes）"""
        if not notes or self.speed_variation == 0.0:
            return

        n = len(notes)
        variation = self.speed_variation

        # 因子 1: 能量 (RMS)
        rms_values = np.array([features[valid_positions[i]]["rms"] for i in range(min(n, len(valid_positions)))])
        if len(rms_values) < n:
            rms_values = np.pad(rms_values, (0, n - len(rms_values)))
        rms_min = float(np.min(rms_values))
        rms_range = float(np.max(rms_values)) - rms_min
        if rms_range < 1e-6:
            energy_norm = np.zeros(n)
        else:
            energy_norm = (rms_values - rms_min) / rms_range

        # 对比度：居中缩放能量分布
        energy_norm = np.clip((energy_norm - 0.5) * self.contrast + 0.5, 0.0, 1.0)

        energy_factors = 0.7 + 0.6 * energy_norm  # [0.7, 1.3]

        # 因子 2: 局部密度 (±500ms 内的音符数)
        times = np.array([note["time"] for note in notes], dtype=float)
        density_window_ms = 500.0
        left = np.searchsorted(times, times - density_window_ms)
        right = np.searchsorted(times, times + density_window_ms, side='right')
        counts = right - left - 1
        density_factors = 1.0 - 0.2 * np.clip(counts / 10.0, 0.0, 1.0)

        # 组合
        raw_speeds = 10.0 * energy_factors * density_factors

        # 缩放变化幅度
        raw_speeds = 10.0 + (raw_speeds - 10.0) * variation

        # 因子 3: 滑动平均平滑
        window = self.speed_smoothing
        if window > 1 and n >= window:
            kernel = np.ones(window) / window
            padded = np.pad(raw_speeds, (window // 2, window // 2), mode="edge")
            smoothed = np.convolve(padded, kernel, mode="valid")
            raw_speeds[:len(smoothed)] = smoothed[:n]

        # 写入 note["speed"]，保留两位小数
        for i, note in enumerate(notes):
            note["speed"] = round(float(raw_speeds[i]), 2)

    def _generate_notes(
        self,
        features: List[Dict[str, Any]],
        assignments: List[int],
        valid_indices: List[int],
        rms_full: Optional[np.ndarray] = None,
        pitches_full: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        notes: List[Dict[str, Any]] = []
        rms_values = [features[i]["rms"] for i in valid_indices]
        rms_max = max(rms_values) if rms_values else 1.0
        ln_threshold = rms_max * self.ln_threshold_ratio
        rms_global_max = float(np.max(rms_full)) if rms_full is not None and len(rms_full) > 0 else rms_max

        if self.hold_position_strength > 0 and len(valid_indices) > 1:
            all_times = [features[i]["time_ms"] for i in valid_indices]
            t_min = min(all_times)
            t_range = max(all_times) - t_min
            if t_range < 1:
                t_range = 1
        else:
            t_min = 0.0
            t_range = 1.0

        complexity = max(0.5, min(2.0, self.complexity))
        hold_chance = 1.0 - (complexity - 0.5) * 0.3
        hold_chance = max(0.3, min(1.0, hold_chance))
        sustain_threshold = max(0.05, 1.0 - self.ln_tendency * 0.7)

        for pos, idx in enumerate(valid_indices):
            feat = features[idx]
            time_ms = feat["time_ms"]
            column = assignments[idx]
            rms = feat["rms"]

            # 对比度调整 RMS
            rms_norm = rms / rms_max if rms_max > 0 else 0.0
            rms_contrasted = np.clip((rms_norm - 0.5) * self.contrast + 0.5, 0.0, 1.0)
            rms = rms_contrasted * rms_max

            if self.hold_position_strength > 0:
                norm = (time_ms - t_min) / t_range
                weight = 1.0 + self.hold_position_strength * (8.0 * (norm - 0.5) ** 2 - 1.0)
                effective_threshold = ln_threshold * weight
            else:
                effective_threshold = ln_threshold

            note_type = "hit"
            end_time = None

            if self.use_energy_for_ln:
                if rms_full is not None and pos + 1 < len(valid_indices):
                    next_feat = features[valid_indices[pos + 1]]
                    gap = next_feat["time_ms"] - time_ms
                    if gap >= self.min_hold_ms:
                        f_start = int(feat["frame"])
                        f_end = int(next_feat["frame"])
                        if f_end > f_start:
                            rms_slice = rms_full[f_start:f_end]
                            p_slice = pitches_full[f_start:f_end] if pitches_full is not None else None
                            score = self._calc_sustain_score(rms_slice, p_slice, rms_global_max)
                            if score >= sustain_threshold:
                                best_gap = gap
                                for k in range(2, len(valid_indices) - pos):
                                    far_feat = features[valid_indices[pos + k]]
                                    far_gap = far_feat["time_ms"] - time_ms
                                    if far_gap < best_gap + self.min_hold_ms:
                                        continue
                                    far_end = int(far_feat["frame"])
                                    if far_end <= f_start:
                                        continue
                                    rms_slice_far = rms_full[f_start:far_end]
                                    p_slice_far = pitches_full[f_start:far_end] if pitches_full is not None else None
                                    score_far = self._calc_sustain_score(rms_slice_far, p_slice_far, rms_global_max)
                                    if score_far >= sustain_threshold:
                                        best_gap = far_gap
                                    else:
                                        break
                                end_time = time_ms + int(best_gap * 0.8)
                                note_type = "hold"
                elif pos + 1 < len(valid_indices):
                    next_time = features[valid_indices[pos + 1]]["time_ms"]
                    gap = next_time - time_ms
                    if rms >= effective_threshold and gap >= self.min_hold_ms:
                        if np.random.random() < hold_chance:
                            end_time = time_ms + int(gap * 0.8)
                            note_type = "hold"
                else:
                    if rms >= effective_threshold:
                        duration = int(200 + (rms / rms_max) * 800)
                        if duration >= self.min_hold_ms:
                            if np.random.random() < hold_chance:
                                end_time = time_ms + duration
                                note_type = "hold"

            notes.append({
                "time": time_ms,
                "column": column,
                "type": note_type,
                "end_time": end_time,
                "speed": 10.0,
            })

        notes.sort(key=lambda x: x["time"])
        return notes
