"""
音频特征提取（v3：注入音乐结构先验）

特征维度（82）：
  0-63   log-mel(64)        音色/频谱
  64-75  chroma(12)         旋律音级（列分配与音高相关，每帧按 max 归一）
  76     percussive onset   打击乐起振强度（鼓点 → 密度依据）
  77     broadband onset    全频带起振强度（旋律/软音头的起振）
  78     log1p(rms)         能量（段落/强度）
  79     beat_phase         拍内位置 [0,1)
  80     bar_phase          小节内位置 [0,1)（4/4 拍，重拍锚定）
  81     bpm/300            速度常数通道

节拍相位训练时来自 .osu timing points（ground truth，offset 即重拍），
推理时用 librosa beat_track + 重拍锚点启发式估计（estimate_beats）。
"""

import numpy as np
import librosa

SR = 22050
HOP = 512           # ≈23.2ms @22050
N_MELS = 64
N_CHROMA = 12
FEATURE_DIM = 82    # 64 mel + 12 chroma + 2 onset + 1 rms + 2 phase + 1 bpm

# 通道下标（供评估/可视化定位）
CH_ONSET_PERC = 76
CH_ONSET = 77       # 全频带 onset（评估 baseline 用，等价于启发式管线）
CH_BEAT_PH = 79
CH_BAR_PH = 80
CH_BPM = 81

FRAME_MS = HOP / SR * 1000.0
BEATS_PER_BAR = 4


def beats_from_timing_points(points, duration_ms: float, beats_per_bar: int = BEATS_PER_BAR):
    """uninherited timing points [(offset_ms, beat_len_ms)] → (全部拍时刻, 首个重拍下标)

    按段生成拍网格；第一个 timing point 之前的拍按其 beat_len 向前外推。
    osu 约定 uninherited point 的 offset 即小节重拍。
    """
    if not points:
        return np.empty(0, dtype=np.float64), 0
    points = sorted(points)
    first_off, first_bl = points[0]
    beats = []
    k = 1
    while first_off - k * first_bl >= -first_bl:
        beats.append(first_off - k * first_bl)
        k += 1
    beats.reverse()
    fd_idx = len(beats)
    for i, (off, bl) in enumerate(points):
        end = points[i + 1][0] if i + 1 < len(points) else duration_ms
        t = off
        while t < end:
            beats.append(t)
            t += bl
    return np.array(beats, dtype=np.float64), fd_idx


def dominant_bpm(points, duration_ms: float) -> float:
    """覆盖时长最长的 timing point 的 BPM"""
    if not points:
        return 0.0
    points = sorted(points)
    best_span, best_bl = -1.0, points[0][1]
    for i, (off, bl) in enumerate(points):
        end = points[i + 1][0] if i + 1 < len(points) else duration_ms
        if end - off > best_span:
            best_span, best_bl = end - off, bl
    return float(60000.0 / best_bl) if best_bl > 0 else 0.0


def beat_phase_features(times_ms: np.ndarray, beat_times: np.ndarray, fd_idx: int,
                        beats_per_bar: int = BEATS_PER_BAR):
    """每帧的 (拍内相位, 小节内相位)，范围 [0,1)"""
    T = len(times_ms)
    if len(beat_times) < 2:
        return np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    idx = np.searchsorted(beat_times, times_ms, side="right") - 1
    idx = np.clip(idx, 0, len(beat_times) - 2)
    b0 = beat_times[idx]
    b1 = beat_times[idx + 1]
    beat_ph = np.clip((times_ms - b0) / np.maximum(b1 - b0, 1e-6), 0.0, 1.0)
    bar_ph = (np.mod(idx - fd_idx, beats_per_bar) + beat_ph) / beats_per_bar
    return beat_ph.astype(np.float32), bar_ph.astype(np.float32)


def extract_features(y: np.ndarray, beat_times_ms=None, first_downbeat: int = 0,
                     bpm: float = 0.0) -> np.ndarray:
    """波形 → [T, 82] float16（见模块 docstring 的通道定义）"""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))  # [F, T]
    mel = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max).T  # [T, 64]

    # HPSS 频谱软掩码分离（帧对齐，比时域版快）
    S_harm, S_perc = librosa.decompose.hpss(S)

    mel_perc = librosa.feature.melspectrogram(S=S_perc, sr=SR, n_mels=N_MELS)
    onset_perc = librosa.onset.onset_strength(
        S=librosa.power_to_db(mel_perc, ref=np.max), sr=SR, hop_length=HOP)
    onset_perc = onset_perc / (onset_perc.max() + 1e-8)

    # 全频带 onset（旋律/软音头起振，v1 同款）
    onset_bb = librosa.onset.onset_strength(S=log_mel.T, sr=SR, hop_length=HOP)
    onset_bb = onset_bb / (onset_bb.max() + 1e-8)

    chroma = librosa.feature.chroma_stft(S=S_harm ** 2, sr=SR, n_chroma=N_CHROMA).T  # [T,12]

    rms = librosa.feature.rms(S=S)[0]
    rms = np.log1p(rms * 100.0)

    T = log_mel.shape[0]
    times_ms = np.arange(T) * FRAME_MS
    beat_ph, bar_ph = beat_phase_features(
        times_ms, np.asarray(beat_times_ms, dtype=np.float64)
        if beat_times_ms is not None else np.empty(0),
        first_downbeat)
    bpm_ch = np.full(T, np.clip(bpm / 300.0, 0.0, 2.0), dtype=np.float32)

    feats = np.concatenate([
        log_mel[:T], chroma[:T],
        onset_perc[:T, None], onset_bb[:T, None], rms[:T, None],
        beat_ph[:, None], bar_ph[:, None], bpm_ch[:, None],
    ], axis=1)
    return feats.astype(np.float16)


def estimate_beats(y: np.ndarray, sr: int = SR, beats_per_bar: int = BEATS_PER_BAR):
    """推理侧节拍估计：librosa beat_track + 重拍锚点启发式

    重拍启发式：四种相位轮转中，拍点处平均 onset 最强的那一路视为重拍
    （4/4 音乐底鼓通常落在重拍）。返回 (beat_times_ms, first_downbeat_idx, bpm)。
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP)
    bpm = float(np.atleast_1d(tempo)[0])
    if len(beat_frames) < 2 * beats_per_bar:
        return np.empty(0, dtype=np.float64), 0, bpm
    beat_frames = np.clip(beat_frames, 0, len(onset_env) - 1)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP) * 1000.0
    strengths = [
        float(onset_env[beat_frames[r::beats_per_bar]].mean())
        if len(beat_frames[r::beats_per_bar]) else 0.0
        for r in range(beats_per_bar)
    ]
    return beat_times, int(np.argmax(strengths)), bpm
