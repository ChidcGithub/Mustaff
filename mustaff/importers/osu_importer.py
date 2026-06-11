"""
osu!mania 谱面导入器

读取 osu file format v14 格式的 .osu 文件。
格式参考：https://osu.ppy.sh/wiki/zh/Client/File_formats/osu_%28file_format%29
"""

import os
import re
from typing import Dict, Any, List


class OsuImporter:
    """导入 osu!mania 格式的 .osu 谱面文件"""

    # osu!mania 列的 X 坐标映射（与 exporter 一致）
    COLUMN_X = {
        1: [256],
        2: [128, 384],
        3: [86, 256, 426],
        4: [64, 192, 320, 448],
        5: [51, 153, 256, 358, 460],
        6: [42, 128, 213, 298, 384, 470],
        7: [36, 109, 182, 256, 329, 402, 475],
        8: [32, 96, 160, 224, 288, 352, 416, 480],
        9: [28, 85, 142, 199, 256, 313, 370, 427, 484],
    }

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.notes: List[Dict[str, Any]] = []
        self.keys: int = 4
        self.bpm: float = 120.0
        self.offset: float = 0.0
        self.title: str = "Untitled"
        self.artist: str = "Unknown Artist"
        self.version: str = "Auto-generated"
        self.audio_filename: str = ""
        self._parse()

    def _find_column(self, x: int, keys: int) -> int:
        """根据 X 坐标找到最近的列"""
        positions = self.COLUMN_X.get(keys, self.COLUMN_X[4])
        min_dist = abs(x - positions[0])
        best = 0
        for i, pos in enumerate(positions):
            dist = abs(x - pos)
            if dist < min_dist:
                min_dist = dist
                best = i
        return best

    def _parse(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        section = ""
        keys_set = False
        x_positions: List[int] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue

            if section == "General":
                if line.startswith("AudioFilename:"):
                    self.audio_filename = line.split(":", 1)[1].strip()

            elif section == "Metadata":
                if line.startswith("Title:"):
                    self.title = line.split(":", 1)[1].strip()
                elif line.startswith("Artist:"):
                    self.artist = line.split(":", 1)[1].strip()
                elif line.startswith("Version:"):
                    self.version = line.split(":", 1)[1].strip()

            elif section == "Difficulty":
                if line.startswith("CircleSize:"):
                    self.keys = int(float(line.split(":", 1)[1].strip()))
                    keys_set = True

            elif section == "TimingPoints":
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        beat_length = float(parts[1])
                        if beat_length > 0:
                            self.bpm = round(60000.0 / beat_length, 2)
                        self.offset = float(parts[0])
                    except (ValueError, ZeroDivisionError):
                        pass

            elif section == "HitObjects":
                parts = line.split(",")
                if len(parts) >= 6:
                    try:
                        x = int(parts[0])
                        time_ms = int(parts[2])
                        note_type = int(parts[3])

                        if not keys_set:
                            self.keys = 4

                        col = self._find_column(x, self.keys)

                        if note_type & 128:
                            # Hold note: type 128, endTime in part[5]
                            end_str = parts[5].split(":")[0]
                            end_time = int(end_str)
                            self.notes.append({
                                "time": time_ms,
                                "column": col,
                                "type": "hold",
                                "end_time": end_time,
                            })
                        else:
                            # Hit note: type 1
                            self.notes.append({
                                "time": time_ms,
                                "column": col,
                                "type": "hit",
                                "end_time": None,
                            })
                    except (ValueError, IndexError):
                        continue

        # 如果没找到 CircleSize，根据实际列数推断
        if not keys_set and self.notes:
            max_col = max(n["column"] for n in self.notes)
            self.keys = max_col + 1

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
