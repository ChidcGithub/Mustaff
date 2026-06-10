"""
Mustaff - 自动从音频生成音游曲谱

通过分析音频的横向（时间/节拍）和纵向（频率/音调）特征，
自动生成音游（如 osu!mania）曲谱。
"""

__version__ = "0.1.0"

from .analyzer import AudioAnalyzer
from .mapper import BeatMapper

__all__ = ["AudioAnalyzer", "BeatMapper"]
