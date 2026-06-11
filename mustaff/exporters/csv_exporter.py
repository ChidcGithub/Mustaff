"""
CSV 谱面导出器

输出自定义 CSV 格式的谱面文件。
格式：
  T行（元数据）: T,索引,英文名,音乐ID,谱面时长(秒)
  N行（音符）:   N,类型(0=短音/1=长音),时长,速度,到达判定线时间,轨道(0-3)
"""

import os
from typing import Dict, Any, List
from .base import BaseExporter


class CsvExporter(BaseExporter):
    """导出自定义 CSV 格式的谱面文件"""

    def __init__(
        self,
        notes: List[Dict[str, Any]],
        bpm: float,
        offset: float = 0.0,
        keys: int = 4,
        title: str = "Untitled",
        artist: str = "Unknown Artist",
        version: str = "Auto-generated",
        time_unit: str = "seconds",
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
            time_unit: 输出时间单位，"seconds" 或 "milliseconds"
        """
        super().__init__(notes, bpm, offset)
        self.keys = keys
        self.title = title
        self.artist = artist
        self.version = version
        self.time_unit = time_unit

    def to_string(self, **kwargs) -> str:
        """生成 CSV 字符串"""
        lines = []

        # 计算谱面时长（秒）
        if self.notes:
            max_time_ms = max(n["time"] for n in self.notes)
            duration_sec = max_time_ms / 1000.0
        else:
            duration_sec = 0.0

        # T 行: T,索引,英文名,音乐ID,谱面时长(秒)
        lines.append(f"T,1,{self.title},-1,{duration_sec:.3f}")

        # N 行: N,类型,时长,速度,到达判定线时间,轨道
        for n in self.notes:
            note_type_val = 1 if n["type"] == "hold" else 0
            speed = n.get("speed", 10.0)

            if self.time_unit == "seconds":
                time_val = n["time"] / 1000.0
                if n["type"] == "hold" and n.get("end_time"):
                    hold_duration = (n["end_time"] - n["time"]) / 1000.0
                else:
                    hold_duration = 1.0
            else:
                time_val = float(n["time"])
                if n["type"] == "hold" and n.get("end_time"):
                    hold_duration = float(n["end_time"] - n["time"])
                else:
                    hold_duration = 1.0

            lines.append(
                f"N,{note_type_val},{hold_duration:.3f},{speed:.1f},{time_val:.3f},{n['column']}"
            )

        return "\n".join(lines) + "\n"

    def export(self, filepath: str, **kwargs) -> None:
        """导出到 CSV 文件"""
        content = self.to_string(**kwargs)
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
