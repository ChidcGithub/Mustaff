"""
Malody 谱面导出器（.mc）

Malody 谱面为 JSON 格式：
  meta: 元数据（歌曲信息、列数、作者等）
  time: BPM 列表，节拍位置用 [bar, num, den] 表示（bar + num/den 个小节）
  note: 音符列表，长音带 endbeat 字段
"""

import json
import math
import os
import time
from typing import Dict, Any, List
from .base import BaseExporter

# beat 分数分母：可整除 2/3/4/6/8 等常见切分，兼顾精度与可读性
_BEAT_DEN = 192


class MalodyExporter(BaseExporter):
    """导出 Malody 格式的 .mc 谱面文件"""

    def __init__(
        self,
        notes: List[Dict[str, Any]],
        bpm: float,
        offset: float = 0.0,
        keys: int = 4,
        title: str = "Untitled",
        artist: str = "Unknown Artist",
        creator: str = "Mustaff",
        version: str = "Auto-generated",
        audio_filename: str = "audio.mp3",
    ):
        super().__init__(notes, bpm, offset)
        self.keys = keys
        self.title = title
        self.artist = artist
        self.creator = creator
        self.version = version
        self.audio_filename = audio_filename

    def _ms_to_beat(self, ms: float) -> List[int]:
        """毫秒时间 → [bar, num, den]（Malody 节拍以歌曲起点为 0）"""
        beats = ms / 60000.0 * self.bpm
        bar = math.floor(beats / 4.0)
        num = int(round((beats - bar * 4.0) / 4.0 * _BEAT_DEN))
        if num >= _BEAT_DEN:
            bar += 1
            num -= _BEAT_DEN
        return [int(bar), num, _BEAT_DEN]

    def to_dict(self) -> Dict[str, Any]:
        note_list = []
        for n in self.notes:
            note_type = n["type"]
            if note_type == "double_hit":
                for col in n.get("columns", [n.get("column", 0)]):
                    note_list.append({
                        "beat": self._ms_to_beat(n["time"]),
                        "column": col % self.keys,
                    })
            elif note_type == "hold":
                col = n["column"]
                if col < 0 or col >= self.keys:
                    col = col % self.keys
                end_time = n["end_time"] if n.get("end_time") else n["time"] + 200
                note_list.append({
                    "beat": self._ms_to_beat(n["time"]),
                    "endbeat": self._ms_to_beat(end_time),
                    "column": col,
                })
            else:
                col = n["column"]
                if col < 0 or col >= self.keys:
                    col = col % self.keys
                note_list.append({
                    "beat": self._ms_to_beat(n["time"]),
                    "column": col,
                })

        note_list.sort(key=lambda e: (e["beat"][0], e["beat"][1] / e["beat"][2]))

        return {
            "meta": {
                "creator": self.creator,
                "background": "",
                "version": self.version,
                "id": 0,
                "mode": 0,
                "time": int(time.time()),
                "song": {
                    "title": self.title,
                    "artist": self.artist,
                    "file": self.audio_filename,
                    "bpm": round(self.bpm, 2),
                },
                "mode_ext": {"column": self.keys, "bar": 0},
            },
            "time": [{"beat": [0, 0, 1], "bpm": round(self.bpm, 2)}],
            "note": note_list,
        }

    def to_string(self, indent: int = 2, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def export(self, filepath: str, indent: int = 2, **kwargs) -> None:
        content = self.to_string(indent=indent, **kwargs)
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
