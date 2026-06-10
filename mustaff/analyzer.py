"""
音频分析模块

使用 librosa 提取音频的：
- Onset（音符起始点，横向时间特征）
- 音高 / 频率（纵向特征）
- RMS 能量 / 音量
- 节拍 (Beat)
"""

from typing import List, Tuple, Optional
import numpy as np
import librosa


class AudioAnalyzer:
    """音频分析器，提取音游谱面所需的各种特征"""

    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        fmin: float = librosa.note_to_hz("C2"),
        fmax: float = librosa.note_to_hz("C7"),
    ):
        """
        Args:
            sr: 采样率
            hop_length: 帧移长度
            fmin: 最小检测频率
            fmax: 最大检测频率
        """
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

        self.y: Optional[np.ndarray] = None
        self.duration: float = 0.0

        # 分析结果
        self.onset_frames: Optional[np.ndarray] = None
        self.onset_times: Optional[np.ndarray] = None
        self.beat_times: Optional[np.ndarray] = None
        self.beat_frames: Optional[np.ndarray] = None
        self.pitches: Optional[np.ndarray] = None
        self.pitch_confidences: Optional[np.ndarray] = None
        self.rms: Optional[np.ndarray] = None
        self.tempo: float = 120.0

    def load(self, filepath: str) -> "AudioAnalyzer":
        """加载音频文件

        Args:
            filepath: 音频文件路径

        Returns:
            self，方便链式调用
        """
        self.y, sr_loaded = librosa.load(filepath, sr=self.sr, mono=True)
        if sr_loaded != self.sr:
            self.sr = sr_loaded
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        return self

    def load_array(self, y: np.ndarray, sr: int) -> "AudioAnalyzer":
        """直接从 numpy 数组加载音频

        Args:
            y: 音频波形数组（单声道）
            sr: 采样率
        """
        self.y = y.astype(np.float32)
        self.sr = sr
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        return self

    def analyze(self) -> "AudioAnalyzer":
        """执行全部分析流程

        Returns:
            self，方便链式调用
        """
        if self.y is None:
            raise RuntimeError("请先调用 load() 或 load_array() 加载音频")

        self._analyze_onset()
        self._analyze_beat()
        self._analyze_pitch()
        self._analyze_rms()
        return self

    def _analyze_onset(self) -> None:
        """提取 onset（音符起始点）"""
        # 使用 librosa 的 onset 检测
        self.onset_frames = librosa.onset.onset_detect(
            y=self.y,
            sr=self.sr,
            hop_length=self.hop_length,
            backtrack=False,
        )
        self.onset_times = librosa.frames_to_time(
            self.onset_frames, sr=self.sr, hop_length=self.hop_length
        )

    def _analyze_beat(self) -> None:
        """提取节拍"""
        tempo, self.beat_frames = librosa.beat.beat_track(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )
        self.tempo = float(tempo) if tempo and tempo > 0 else 120.0
        self.beat_times = librosa.frames_to_time(
            self.beat_frames, sr=self.sr, hop_length=self.hop_length
        )

    def _analyze_pitch(self) -> None:
        """提取音高（使用 pYIN 算法）"""
        pitches, voiced_flag, voiced_probs = librosa.pyin(
            self.y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=self.hop_length,
        )
        self.pitches = pitches
        self.pitch_confidences = voiced_probs

    def _analyze_rms(self) -> None:
        """提取 RMS 能量"""
        self.rms = librosa.feature.rms(
            y=self.y, hop_length=self.hop_length
        )[0]

    def get_note_features(self) -> List[dict]:
        """获取每个 onset 点的综合特征

        Returns:
            每个 onset 对应的 dict 列表，包含：
                - time_ms: 时间（毫秒）
                - frame: 帧索引
                - pitch_hz: 音高（Hz），若未检测到则为 None
                - pitch_midi: 音高（MIDI 音符编号），若未检测到则为 None
                - confidence: 音高置信度
                - rms: RMS 能量值
                - near_beat: 是否靠近节拍点
        """
        if self.onset_frames is None:
            raise RuntimeError("请先调用 analyze()")

        results = []
        beat_set = set(self.beat_frames.tolist()) if self.beat_frames is not None else set()

        for frame in self.onset_frames:
            time_s = librosa.frames_to_time(frame, sr=self.sr, hop_length=self.hop_length)
            time_ms = int(round(time_s * 1000))

            # 获取该帧的音高
            pitch_hz = None
            pitch_midi = None
            confidence = 0.0
            if self.pitches is not None and 0 <= frame < len(self.pitches):
                p = self.pitches[frame]
                if not np.isnan(p) and p > 0:
                    pitch_hz = float(p)
                    pitch_midi = float(librosa.hz_to_midi(p))
                    confidence = float(self.pitch_confidences[frame]) if self.pitch_confidences is not None else 0.0

            # 获取 RMS
            rms_val = 0.0
            if self.rms is not None and 0 <= frame < len(self.rms):
                rms_val = float(self.rms[frame])

            near_beat = frame in beat_set

            results.append({
                "time_ms": time_ms,
                "frame": int(frame),
                "pitch_hz": pitch_hz,
                "pitch_midi": pitch_midi,
                "confidence": confidence,
                "rms": rms_val,
                "near_beat": near_beat,
            })

        return results

    def get_pitch_range(self) -> Tuple[Optional[float], Optional[float]]:
        """获取检测到的音高范围（Hz）

        Returns:
            (min_pitch, max_pitch)，若无可用的则为 (None, None)
        """
        if self.pitches is None:
            return None, None
        valid = self.pitches[~np.isnan(self.pitches)]
        if len(valid) == 0:
            return None, None
        return float(valid.min()), float(valid.max())

    def get_rms_range(self) -> Tuple[float, float]:
        """获取 RMS 能量范围"""
        if self.rms is None:
            return 0.0, 0.0
        return float(self.rms.min()), float(self.rms.max())
