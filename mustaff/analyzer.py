from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, List, Tuple, Optional
import numpy as np
import librosa


def _scalar(x):
    return x.item() if hasattr(x, 'item') else x


def _run_pyin(y: np.ndarray, sr: int, fmin: float, fmax: float, hop_length: int):
    """Module-level function for ProcessPoolExecutor — runs librosa.pyin in a subprocess."""
    return librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)


ProgressCallback = Callable[[int, str], None]


class AudioAnalyzer:
    """音频分析器，提取音游谱面所需的各种特征"""

    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        fmin: float = librosa.note_to_hz("C2"),
        fmax: float = librosa.note_to_hz("C7"),
        onset_sensitivity: float = 0.5,
        backtrack: bool = False,
        multi_band: bool = False,
        n_fft: int = 2048,
        min_bpm: float = 50.0,
        max_bpm: float = 200.0,
        multi_process_pitch: bool = False,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.onset_sensitivity = onset_sensitivity
        self.backtrack = backtrack
        self.multi_band = multi_band
        self.n_fft = n_fft
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.multi_process_pitch = multi_process_pitch

        self.y: Optional[np.ndarray] = None
        self.duration: float = 0.0

        self.onset_frames: Optional[np.ndarray] = None
        self.onset_times: Optional[np.ndarray] = None
        self.onset_strengths: Optional[np.ndarray] = None
        self.onset_bands: Optional[np.ndarray] = None
        self.beat_times: Optional[np.ndarray] = None
        self.beat_frames: Optional[np.ndarray] = None
        self.pitches: Optional[np.ndarray] = None
        self.pitch_confidences: Optional[np.ndarray] = None
        self.rms: Optional[np.ndarray] = None
        self.tempo: float = 120.0

    def load(self, filepath: str) -> "AudioAnalyzer":
        self.y, sr_loaded = librosa.load(filepath, sr=self.sr, mono=True)
        if sr_loaded != self.sr:
            self.sr = sr_loaded
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        return self

    def load_array(self, y: np.ndarray, sr: int) -> "AudioAnalyzer":
        self.y = y.astype(np.float32)
        self.sr = sr
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        return self

    def analyze(
        self,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> "AudioAnalyzer":
        if self.y is None:
            raise RuntimeError("请先调用 load() 或 load_array() 加载音频")

        if progress_callback:
            progress_callback(5, "分析中... (并行提取特征)")

        n_workers = max(1, (os.cpu_count() or 4) - 1)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(self._analyze_onset): "onset",
                pool.submit(self._analyze_beat): "beat",
                pool.submit(self._analyze_rms): "rms",
            }

            if self.multi_process_pitch:
                pitch_pool = ProcessPoolExecutor(max_workers=1)
                pitch_future = pitch_pool.submit(
                    _run_pyin, self.y, self.sr, self.fmin, self.fmax, self.hop_length,
                )
            else:
                futures[pool.submit(self._analyze_pitch)] = "pitch"

            for f in as_completed(futures):
                f.result()

            if self.multi_process_pitch:
                pitches, _, voiced_probs = pitch_future.result()
                self.pitches = pitches
                self.pitch_confidences = voiced_probs
                pitch_pool.shutdown(wait=False)
            else:
                pass

        if progress_callback:
            progress_callback(85, "分析完成")

        return self

    def _onset_delta(self) -> float:
        return 0.005 + (1.0 - self.onset_sensitivity) * 0.13

    def _analyze_onset(self) -> None:
        if self.multi_band:
            self._analyze_onset_multiband()
            return

        onset_env = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft,
        )
        self.onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=self.hop_length,
            backtrack=self.backtrack,
            delta=self._onset_delta(),
        )
        self.onset_times = librosa.frames_to_time(
            self.onset_frames, sr=self.sr, hop_length=self.hop_length
        )
        self.onset_strengths = onset_env[self.onset_frames] if len(self.onset_frames) > 0 else np.array([])
        self.onset_bands = np.zeros(len(self.onset_frames), dtype=int)

    def _analyze_onset_multiband(self) -> None:
        bands = [
            ("low", 0, 250),
            ("mid", 250, 2000),
            ("high", 2000, 8000),
        ]

        S = np.abs(librosa.stft(self.y, n_fft=self.n_fft, hop_length=self.hop_length))

        def _detect_band(band_idx: int, lo: float, hi: float):
            lo_bin = max(0, int(lo * self.n_fft // self.sr))
            hi_bin = min(self.n_fft // 2, int(hi * self.n_fft // self.sr))
            n_bins = hi_bin - lo_bin
            if n_bins < 2:
                return band_idx, np.array([], dtype=int)

            band_S = S[lo_bin:hi_bin, :]
            if band_S.size == 0:
                return band_idx, np.array([], dtype=int)

            onset_env = librosa.onset.onset_strength(
                S=band_S, sr=self.sr, hop_length=self.hop_length,
                n_fft=self.n_fft,
            )
            frames = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=self.sr, hop_length=self.hop_length,
                backtrack=self.backtrack,
                delta=self._onset_delta(),
            )
            return band_idx, frames

        n_workers = min(max(1, (os.cpu_count() or 4) - 1), len(bands))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_detect_band, i, lo, hi) for i, (_, lo, hi) in enumerate(bands)]
            all_onset_frames = []
            all_bands = []
            for f in as_completed(futures):
                band_idx, frames = f.result()
                all_onset_frames.extend(frames.tolist())
                all_bands.extend([band_idx] * len(frames))

        if not all_onset_frames:
            self.onset_frames = np.array([], dtype=int)
            self.onset_times = np.array([])
            self.onset_strengths = np.array([])
            self.onset_bands = np.array([], dtype=int)
            return

        frames_arr = np.array(all_onset_frames, dtype=int)
        bands_arr = np.array(all_bands, dtype=int)

        order = np.argsort(frames_arr)
        frames_arr = frames_arr[order]
        bands_arr = bands_arr[order]

        dup_mask = np.ones(len(frames_arr), dtype=bool)
        for i in range(1, len(frames_arr)):
            if frames_arr[i] == frames_arr[i - 1]:
                dup_mask[i] = False
        self.onset_frames = frames_arr[dup_mask]
        self.onset_bands = bands_arr[dup_mask]
        self.onset_times = librosa.frames_to_time(
            self.onset_frames, sr=self.sr, hop_length=self.hop_length
        )
        self.onset_strengths = np.ones(len(self.onset_frames))

    def _analyze_beat(self) -> None:
        tempo, self.beat_frames = librosa.beat.beat_track(
            y=self.y, sr=self.sr, hop_length=self.hop_length,
        )
        t = _scalar(tempo)
        self.tempo = t if t > 0 else 120.0
        self.beat_times = librosa.frames_to_time(
            self.beat_frames, sr=self.sr, hop_length=self.hop_length
        )

    def _analyze_pitch(self) -> None:
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
        self.rms = librosa.feature.rms(
            y=self.y, hop_length=self.hop_length
        )[0]

    def get_beat_subdivisions(self, resolution: int = 8) -> np.ndarray:
        if self.beat_times is None or len(self.beat_times) < 2:
            if self.duration > 0:
                beat_interval = 60.0 / self.tempo
                grid = np.arange(0, self.duration, beat_interval / resolution)
                return grid
            return np.array([])

        beat_interval = np.median(np.diff(self.beat_times))
        if beat_interval <= 0:
            beat_interval = 60.0 / self.tempo

        sub_interval = beat_interval / resolution
        first_beat = self.beat_times[0]
        last_beat = self.beat_times[-1]
        grid = np.arange(first_beat, last_beat + sub_interval, sub_interval)
        return grid

    def get_note_features(self) -> List[dict]:
        if self.onset_frames is None:
            raise RuntimeError("请先调用 analyze()")

        results = []
        beat_set = set(self.beat_frames.tolist()) if self.beat_frames is not None else set()

        for i, frame in enumerate(self.onset_frames):
            time_s = librosa.frames_to_time(frame, sr=self.sr, hop_length=self.hop_length)
            time_ms = int(round(_scalar(time_s) * 1000))

            pitch_hz = None
            pitch_midi = None
            confidence = 0.0
            if self.pitches is not None and 0 <= frame < len(self.pitches):
                p = self.pitches[frame]
                if not np.isnan(p) and p > 0:
                    pitch_hz = _scalar(p)
                    pitch_midi = _scalar(librosa.hz_to_midi(p))
                    confidence = _scalar(self.pitch_confidences[frame]) if self.pitch_confidences is not None else 0.0

            rms_val = 0.0
            if self.rms is not None and 0 <= frame < len(self.rms):
                rms_val = _scalar(self.rms[frame])

            near_beat = frame in beat_set
            onset_strength = _scalar(self.onset_strengths[i]) if self.onset_strengths is not None and i < len(self.onset_strengths) else 0.0
            band = _scalar(self.onset_bands[i]) if self.onset_bands is not None and i < len(self.onset_bands) else 0

            results.append({
                "time_ms": time_ms,
                "frame": int(frame),
                "pitch_hz": pitch_hz,
                "pitch_midi": pitch_midi,
                "confidence": confidence,
                "rms": rms_val,
                "near_beat": near_beat,
                "onset_strength": onset_strength,
                "band": band,
            })

        return results

    def get_pitch_range(self) -> Tuple[Optional[float], Optional[float]]:
        if self.pitches is None:
            return None, None
        valid = self.pitches[~np.isnan(self.pitches)]
        if len(valid) == 0:
            return None, None
        return float(valid.min()), float(valid.max())

    def get_rms_range(self) -> Tuple[float, float]:
        if self.rms is None:
            return 0.0, 0.0
        return float(self.rms.min()), float(self.rms.max())
