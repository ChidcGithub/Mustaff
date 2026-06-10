from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseExporter(ABC):
    """谱面导出器基类"""

    def __init__(self, notes: List[Dict[str, Any]], bpm: float, offset: float = 0.0):
        """
        Args:
            notes: 音符列表，每个音符为 dict，包含：
                - time: 时间戳（毫秒）
                - column: 轨道列索引（从 0 开始）
                - type: 音符类型（'hit', 'hold'）
                - end_time: 若为 hold，表示结束时间（毫秒）
            bpm: 谱面主 BPM
            offset: 第一拍偏移（毫秒）
        """
        self.notes = notes
        self.bpm = bpm
        self.offset = offset

    @abstractmethod
    def export(self, filepath: str, **kwargs) -> None:
        """导出谱面到文件"""
        pass

    @abstractmethod
    def to_string(self, **kwargs) -> str:
        """返回谱面内容的字符串表示"""
        pass
