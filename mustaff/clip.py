"""
片段截取：导出时把音频和谱面同步切片

- slice_notes: 截取 [start_ms, end_ms) 内的音符，时间平移到片段起点为 0，
  长音尾部超出片段终点时截断
- slice_audio_file: 解码（含 miniaudio 兜底）→ 切片 → 重编码写出
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 可直接重编码写出的扩展名（soundfile/libsndfile 支持）
_WRITABLE_EXT = {".mp3", ".ogg", ".wav", ".flac"}


def slice_notes(
    notes: List[Dict[str, Any]],
    start_ms: float,
    end_ms: float,
) -> List[Dict[str, Any]]:
    """截取 [start_ms, end_ms) 内的音符，时间以片段起点为 0"""
    sliced = []
    for n in notes:
        t = n["time"]
        if not (start_ms <= t < end_ms):
            continue
        m = dict(n)
        m["time"] = int(t - start_ms)
        if m.get("end_time") is not None:
            # 长音尾超出片段则截断
            m["end_time"] = int(min(m["end_time"], end_ms) - start_ms)
        sliced.append(m)
    return sliced


def slice_audio_file(
    audio_path: str,
    start_ms: float,
    end_ms: float,
    out_dir: str,
    out_base: Optional[str] = None,
) -> Tuple[str, float]:
    """把音频的 [start_ms, end_ms) 切片写出，返回 (输出路径, 实际片段时长 ms)

    输出扩展名优先保持原格式（mp3/ogg/wav/flac），不支持则退回 ogg/wav。
    """
    import soundfile as sf
    from .analyzer import AudioAnalyzer

    analyzer = AudioAnalyzer()
    analyzer.load(audio_path)
    y, sr = analyzer.y, analyzer.sr

    total_ms = analyzer.duration * 1000.0
    start_ms = max(0.0, start_ms)
    end_ms = min(end_ms, total_ms)
    if end_ms <= start_ms:
        raise ValueError(f"片段区间无效: {start_ms:.0f}ms ~ {end_ms:.0f}ms（音频总长 {total_ms:.0f}ms）")

    s0 = int(start_ms / 1000.0 * sr)
    s1 = int(end_ms / 1000.0 * sr)
    seg = y[s0:s1]
    actual_ms = len(seg) / sr * 1000.0

    base = out_base or os.path.splitext(os.path.basename(audio_path))[0]
    tag = f".slice_{int(start_ms)}-{int(end_ms)}"
    ext = os.path.splitext(audio_path)[1].lower()
    candidates = ([ext] if ext in _WRITABLE_EXT else []) + [".ogg", ".wav"]

    os.makedirs(out_dir, exist_ok=True)
    last_err: Optional[Exception] = None
    for e in candidates:
        out_path = os.path.join(out_dir, f"{base}{tag}{e}")
        try:
            if e == ".ogg":
                sf.write(out_path, seg, sr, format="OGG", subtype="VORBIS")
            elif e == ".mp3":
                sf.write(out_path, seg, sr, format="MP3", subtype="MPEG_LAYER_III")
            else:
                sf.write(out_path, seg, sr)
            return out_path, actual_ms
        except Exception as err:
            last_err = err
            if os.path.exists(out_path):
                os.remove(out_path)
    raise RuntimeError(f"音频切片写出失败: {last_err}")
