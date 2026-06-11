"""
osu!mania 谱面导出器

生成符合 osu! 文件格式规范的 .osu 文件。
格式参考：https://osu.ppy.sh/wiki/zh/Client/File_formats/osu_%28file_format%29
"""

import os
from typing import Dict, Any, List
from .base import BaseExporter


class OsuManiaExporter(BaseExporter):
    """导出 osu!mania 格式的 .osu 谱面文件"""

    # osu!mania 列的 X 坐标映射（512 像素宽，均分）
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
        """
        Args:
            notes: 音符列表
            bpm: 主 BPM
            offset: 偏移（毫秒）
            keys: 轨道数
            title: 歌曲标题
            artist: 艺术家
            creator: 谱面作者
            version: 难度名
            audio_filename: 音频文件名
        """
        super().__init__(notes, bpm, offset)
        self.keys = keys
        self.title = title
        self.artist = artist
        self.creator = creator
        self.version = version
        self.audio_filename = audio_filename

        if keys not in self.COLUMN_X:
            raise ValueError(f"不支持的轨道数: {keys}，osu!mania 支持 1-9K")

    def to_string(self, **kwargs) -> str:
        """生成 .osu 文件内容字符串"""
        lines = []
        lines.append("osu file format v14")
        lines.append("")

        # [General]
        lines.append("[General]")
        lines.append(f"AudioFilename: {self.audio_filename}")
        lines.append("Mode: 3")  # 3 = osu!mania
        lines.append("")

        # [Metadata]
        lines.append("[Metadata]")
        lines.append(f"Title:{self.title}")
        lines.append(f"TitleUnicode:{self.title}")
        lines.append(f"Artist:{self.artist}")
        lines.append(f"ArtistUnicode:{self.artist}")
        lines.append(f"Creator:{self.creator}")
        lines.append(f"Version:{self.version}")
        lines.append("BeatmapID:0")
        lines.append("BeatmapSetID:-1")
        lines.append("")

        # [Difficulty]
        lines.append("[Difficulty]")
        lines.append(f"HPDrainRate:5")
        lines.append(f"CircleSize:{self.keys}")
        lines.append(f"OverallDifficulty:5")
        lines.append(f"ApproachRate:5")
        lines.append(f"SliderMultiplier:1.4")
        lines.append(f"SliderTickRate:1")
        lines.append("")

        # [TimingPoints]
        lines.append("[TimingPoints]")
        # 主时间点
        beat_length = 60000.0 / self.bpm
        # 格式：offset, beatLength, meter, sampleSet, sampleIndex, volume, uninherited, effects
        lines.append(f"{round(self.offset)},{beat_length:.6f},4,0,0,100,1,0")
        lines.append("")

        # [HitObjects]
        lines.append("[HitObjects]")
        x_positions = self.COLUMN_X[self.keys]

        for note in self.notes:
            col = note["column"]
            if col < 0 or col >= self.keys:
                col = col % self.keys
            x = x_positions[col]
            y = 192  # osu!mania 的默认 Y 坐标
            time = int(note["time"])
            note_type = note["type"]

            if note_type == "hold":
                # 长按音符：x,y,time,type,hitSound,endTime:hitSample
                end_time = int(note["end_time"]) if note["end_time"] else time + 200
                # type 128 = hold note (1<<7)
                lines.append(f"{x},{y},{time},128,0,{end_time}:0:0:0:0:")
            else:
                # 普通音符：x,y,time,type,hitSound,hitSample
                # type 1 = hit circle
                lines.append(f"{x},{y},{time},1,0,0:0:0:0:")

        return "\n".join(lines)

    def export(self, filepath: str, **kwargs) -> None:
        """导出到文件

        Args:
            filepath: 输出文件路径，建议以 .osu 结尾
        """
        content = self.to_string(**kwargs)
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
