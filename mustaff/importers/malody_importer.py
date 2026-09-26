"""
Malody 谱面导入器（.mc）

读取 Malody JSON 格式谱面，节拍位置 [bar, num, den] 按 BPM 换算回毫秒。
"""

import json
from typing import Dict, Any, List


class MalodyImporter:
    """导入 Malody 格式的 .mc 谱面文件"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.notes: List[Dict[str, Any]] = []
        self.keys: int = 4
        self.bpm: float = 120.0
        self.offset: float = 0.0
        self.title: str = "Untitled"
        self.artist: str = "Unknown Artist"
        self.version: str = "Auto-generated"
        self._parse()

    def _beat_to_ms(self, beat: List) -> int:
        bar, num, den = int(beat[0]), int(beat[1]), int(beat[2]) or 1
        beats = (bar + num / den) * 4.0
        return int(round(beats * 60000.0 / self.bpm))

    def _parse(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("meta", {})
        song = meta.get("song", {})
        self.title = song.get("title", "Untitled")
        self.artist = song.get("artist", "Unknown Artist")
        self.version = meta.get("version", "Auto-generated")
        self.keys = int(meta.get("mode_ext", {}).get("column", 4))

        time_list = data.get("time", [])
        if time_list:
            try:
                self.bpm = float(time_list[0].get("bpm", 120.0))
            except (TypeError, ValueError):
                self.bpm = 120.0

        for n in data.get("note", []):
            if "beat" not in n:
                continue
            start_ms = self._beat_to_ms(n["beat"])
            col = int(n.get("column", 0))
            if "endbeat" in n:
                end_ms = self._beat_to_ms(n["endbeat"])
                self.notes.append({
                    "time": start_ms,
                    "column": col,
                    "type": "hold",
                    "end_time": end_ms,
                    "speed": 10.0,
                })
            else:
                self.notes.append({
                    "time": start_ms,
                    "column": col,
                    "type": "hit",
                    "end_time": None,
                    "speed": 10.0,
                })

        self.notes.sort(key=lambda n: n["time"])

    def get_info(self) -> Dict[str, Any]:
        return {
            "notes": self.notes,
            "keys": self.keys,
            "bpm": self.bpm,
            "offset": self.offset,
            "title": self.title,
            "artist": self.artist,
            "version": self.version,
        }
