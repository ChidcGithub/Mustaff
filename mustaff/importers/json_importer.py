"""
JSON 谱面导入器

读取由 JsonExporter 生成的 JSON 格式谱面文件。
"""

import json
import os
from typing import Dict, Any, List, Optional


class JsonImporter:
    """导入自定义 JSON 格式的谱面文件"""

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

    def _parse(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        timing = data.get("timing", {})

        self.keys = int(meta.get("keys", 4))
        self.bpm = float(timing.get("bpm", 120.0))
        self.offset = float(timing.get("offset", 0.0))
        self.title = meta.get("title", "Untitled")
        self.artist = meta.get("artist", "Unknown Artist")
        self.version = meta.get("version", "Auto-generated")

        self.notes = []
        for n in data.get("notes", []):
            note = {
                "time": int(n["time"]),
                "column": int(n["column"]),
                "type": n.get("type", "hit"),
                "end_time": int(n["end_time"]) if n.get("end_time") is not None else None,
            }
            self.notes.append(note)

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
