"""
CSV 谱面导入器

读取自定义 CSV 格式的谱面文件。
格式：
  T行（元数据）: T,索引,英文名,音乐ID,谱面时长(秒)
  N行（音符）:   N,类型(0=短音/1=长音),时长,速度,到达判定线时间,轨道(0-3)
"""

from typing import Dict, Any, List


class CsvImporter:
    """导入自定义 CSV 格式的谱面文件"""

    def __init__(self, filepath: str, time_unit: str = "seconds"):
        """
        Args:
            filepath: CSV 文件路径
            time_unit: 时间单位，"seconds" 或 "milliseconds"
        """
        self.filepath = filepath
        self.time_unit = time_unit
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
            lines = f.readlines()

        max_col = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue

            row_type = parts[0].upper()

            if row_type == "T":
                # T,索引,英文名,音乐ID,谱面时长(秒)
                if len(parts) >= 3 and parts[2]:
                    self.title = parts[2]
                if len(parts) >= 5 and parts[4]:
                    try:
                        duration_sec = float(parts[4])
                        self._duration_sec = duration_sec
                    except ValueError:
                        pass

            elif row_type == "N":
                # N,类型,时长,速度,到达判定线时间,轨道
                if len(parts) < 6:
                    continue
                try:
                    note_type_val = int(parts[1])
                    hold_duration = float(parts[2])
                    speed = float(parts[3])
                    time_val = float(parts[4])
                    column = int(parts[5])
                except (ValueError, IndexError):
                    continue

                # 转换时间单位为毫秒
                if self.time_unit == "seconds":
                    time_ms = time_val * 1000
                    hold_duration_ms = hold_duration * 1000
                else:
                    time_ms = time_val
                    hold_duration_ms = hold_duration

                time_ms = int(time_ms)

                if note_type_val == 1 and hold_duration_ms > 0:
                    end_time = int(time_ms + hold_duration_ms)
                    self.notes.append({
                        "time": time_ms,
                        "column": column,
                        "type": "hold",
                        "end_time": end_time,
                        "speed": speed,
                    })
                else:
                    self.notes.append({
                        "time": time_ms,
                        "column": column,
                        "type": "hit",
                        "end_time": None,
                        "speed": speed,
                    })

                if column > max_col:
                    max_col = column

        self.notes.sort(key=lambda n: n["time"])

        # 从最大轨道数推断 keys
        self.keys = max_col + 1 if max_col >= 0 else 4

        # 如果检测到 duration，补 2s 作为末尾
        if hasattr(self, "_duration_sec"):
            pass  # duration 已记录，GUI 会使用

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
