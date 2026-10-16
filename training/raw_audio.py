"""
raw v4 音频 → 每拍波形矩阵的唯一实现（build_dataset_raw 与推理 generate 共用，保证同口径）

表示：每拍（beat_times[b] → beat_times[b+1]）的音频段做抗混叠低通后，
时间重采样到 RAW_LEN 点，振幅按曲集 p95(|y|) 归一到 RAW_NORM_SCALE，存 int16。
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

SR_RAW = 22050
RAW_LEN = 4096            # 每拍波形点数（4096@190bpm ≈ 13kHz 有效率）
RAW_NORM_SCALE = 12000.0  # p95 归一后的 int16 参考幅度


def warp_beats_raw(y: np.ndarray, beat_times_ms: np.ndarray,
                   sr: int = SR_RAW, raw_len: int = RAW_LEN,
                   anti_alias: bool = True) -> np.ndarray:
    """波形 → [n_units, raw_len] int16（n_units = len(beat_times) - 1）

    y 须已按目标口径截断（构建侧截到 meta.duration；推理侧全长）。
    抗混叠：按全曲最长拍对应的最小重采样率定截止（0.45 × sr × raw_len / max_beat_samples），
    整曲一次零相位 sosfiltfilt，之后逐拍线性插值不再产生混叠。
    """
    y = np.asarray(y, dtype=np.float32)
    beat_times_ms = np.asarray(beat_times_ms, dtype=np.float64)
    n_units = len(beat_times_ms) - 1
    if n_units < 1 or len(y) < sr // 2:
        return np.zeros((max(n_units, 0), raw_len), dtype=np.int16)

    if anti_alias:
        beat_samples = np.diff(beat_times_ms) * sr / 1000.0
        # 抗异常网格：librosa 丢拍/倍拍会把最长拍拉大一倍 → 截止砍半。
        # 用 2.5×中位数封顶（训练/推理同规则，正常歌曲 median≈max 无影响）
        median_bs = float(np.median(beat_samples))
        max_bs = min(float(beat_samples.max()), 2.5 * median_bs)
        cutoff = 0.45 * sr * raw_len / max(max_bs, 1.0)
        cutoff = float(np.clip(cutoff, 200.0, sr * 0.475))
        try:
            sos = butter(4, cutoff, btype="low", fs=float(sr), output="sos")
            y = sosfiltfilt(sos, y).astype(np.float32)
        except Exception:
            pass  # 极短信号 filtfilt 失败时退化为未滤波（p95 归一提供基本保护）

    p95 = float(np.percentile(np.abs(y), 95))
    if p95 < 1e-6:
        return np.zeros((n_units, raw_len), dtype=np.int16)
    y = y / p95 * RAW_NORM_SCALE

    rows = np.zeros((n_units, raw_len), dtype=np.int16)
    dst = (np.arange(raw_len) + 0.5) / raw_len
    for b in range(n_units):
        i0 = max(0, int(beat_times_ms[b] * sr / 1000.0))
        i1 = min(len(y), max(int(beat_times_ms[b + 1] * sr / 1000.0), i0 + 2))
        if i1 <= i0:
            continue  # 拍点越过音频末尾（timing 外推超界）→ 该拍保持全零
        seg = y[i0:i1]
        rows[b] = np.clip(
            np.interp(dst * len(seg) - 0.5, np.arange(len(seg), dtype=np.float64), seg),
            -32767, 32767).astype(np.int16)
    return rows
