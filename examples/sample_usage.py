"""
Mustaff 使用示例

展示如何通过 Python API 使用 Mustaff 分析音频并生成谱面。
"""

from mustaff.analyzer import AudioAnalyzer
from mustaff.mapper import BeatMapper
from mustaff.exporters.osu_mania import OsuManiaExporter
from mustaff.exporters.json_exporter import JsonExporter


def main():
    audio_file = "your_song.mp3"  # 替换为你的音频文件

    # 1. 分析音频
    print("正在分析音频...")
    analyzer = AudioAnalyzer(sr=22050)
    analyzer.load(audio_file)
    analyzer.analyze()

    print(f"BPM: {analyzer.tempo:.1f}")
    print(f"Onset 数量: {len(analyzer.onset_times)}")
    print(f"音频时长: {analyzer.duration:.2f}s")

    min_p, max_p = analyzer.get_pitch_range()
    if min_p and max_p:
        print(f"音高范围: {min_p:.1f}Hz ~ {max_p:.1f}Hz")

    # 2. 提取特征并映射为音符
    features = analyzer.get_note_features()
    mapper = BeatMapper(keys=4, density_filter_ms=30.0, ln_threshold_ratio=0.7)
    notes = mapper.map_notes(features)
    print(f"生成音符数: {len(notes)}")

    # 3. 导出谱面
    # osu!mania 格式
    osu_exporter = OsuManiaExporter(
        notes=notes,
        bpm=analyzer.tempo,
        keys=4,
        title="My Song",
        artist="Unknown",
        version="Normal",
        audio_filename=audio_file,
    )
    osu_exporter.export("output.osu")
    print("已导出: output.osu")

    # JSON 格式
    json_exporter = JsonExporter(
        notes=notes,
        bpm=analyzer.tempo,
        keys=4,
        title="My Song",
        artist="Unknown",
        version="Normal",
    )
    json_exporter.export("output.json")
    print("已导出: output.json")


if __name__ == "__main__":
    main()
