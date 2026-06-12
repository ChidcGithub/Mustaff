"""
自定义 JSON 谱面导出器

输出通用的 JSON 格式，便于第三方工具渲染或导入。
"""

import json
import os
from typing import Dict, Any, List
from .base import BaseExporter


class JsonExporter(BaseExporter):
    """导出自定义 JSON 格式的谱面文件"""

    def __init__(
        self,
        notes: List[Dict[str, Any]],
        bpm: float,
        offset: float = 0.0,
        keys: int = 4,
        title: str = "Untitled",
        artist: str = "Unknown Artist",
        version: str = "Auto-generated",
    ):
        """
        Args:
            notes: 音符列表
            bpm: 主 BPM
            offset: 偏移（毫秒）
            keys: 轨道数
            title: 歌曲标题
            artist: 艺术家
            version: 难度名
        """
        super().__init__(notes, bpm, offset)
        self.keys = keys
        self.title = title
        self.artist = artist
        self.version = version

    def to_dict(self) -> Dict[str, Any]:
        """返回谱面的字典表示"""
        return {
            "version": "1.0",
            "metadata": {
                "title": self.title,
                "artist": self.artist,
                "creator": "Mustaff",
                "version": self.version,
                "keys": self.keys,
            },
            "timing": {
                "bpm": round(self.bpm, 2),
                "offset": round(self.offset, 2),
            },
            "notes": [
                {
                    "time": n["time"],
                    "column": n.get("column"),
                    "columns": n.get("columns"),
                    "type": n["type"],
                    "end_time": n.get("end_time"),
                    "speed": n.get("speed", 10.0),
                }
                for n in self.notes
            ],
        }

    def to_string(self, indent: int = 2, **kwargs) -> str:
        """生成 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def export(self, filepath: str, indent: int = 2, **kwargs) -> None:
        """导出到 JSON 文件

        Args:
            filepath: 输出文件路径
            indent: JSON 缩进
        """
        content = self.to_string(indent=indent, **kwargs)
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
