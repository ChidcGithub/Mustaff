"""
基础测试
"""

import os
import pytest
import numpy as np

from mustaff.analyzer import AudioAnalyzer
from mustaff.mapper import BeatMapper
from mustaff.presets import DIFFICULTY_PRESETS, PRESET_ORDER
from mustaff.utils import split_artist_title
from mustaff.exporters.osu_mania import OsuManiaExporter
from mustaff.exporters.json_exporter import JsonExporter
from mustaff.exporters.csv_exporter import CsvExporter
from mustaff.exporters.malody import MalodyExporter
from mustaff.importers.csv_importer import CsvImporter
from mustaff.importers.malody_importer import MalodyImporter
from mustaff.clip import slice_notes, slice_audio_file


def create_dummy_audio(duration=2.0, sr=22050):
    """生成一个带有一些脉冲的测试音频"""
    t = np.linspace(0, duration, int(sr * duration))
    # 简单的 440Hz 正弦波 + 脉冲
    y = np.sin(2 * np.pi * 440 * t) * 0.3
    # 添加一些 onset 脉冲
    for onset_time in [0.5, 1.0, 1.5]:
        idx = int(onset_time * sr)
        if idx < len(y):
            y[idx:idx+1024] += np.hanning(min(1024, len(y) - idx)) * 0.7
    return y.astype(np.float32)


def test_analyzer():
    y = create_dummy_audio()
    analyzer = AudioAnalyzer(sr=22050)
    analyzer.load_array(y, sr=22050)
    analyzer.analyze()

    assert analyzer.duration > 0
    assert analyzer.onset_times is not None
    assert len(analyzer.onset_times) > 0
    assert analyzer.tempo > 0 or analyzer.tempo == 120.0  # 若检测失败则回退到默认值


def _make_click_track(bpm, duration=8.0, sr=22050):
    """生成指定 BPM 的脉冲音轨"""
    y = np.zeros(int(sr * duration), dtype=np.float32)
    interval = 60.0 / bpm
    t = 0.3
    while t < duration - 0.05:
        idx = int(t * sr)
        n = min(1024, len(y) - idx)
        y[idx:idx + n] += np.hanning(n) * 0.8
        t += interval
    return y


def test_beat_tempo_respects_max_bpm():
    sr = 22050
    y = _make_click_track(bpm=240.0, sr=sr)
    analyzer = AudioAnalyzer(sr=sr, min_bpm=50.0, max_bpm=100.0)
    analyzer.load_array(y, sr=sr)
    analyzer._analyze_beat()

    # 检测到的 BPM 超出上限时，应八度校正回 [min_bpm, max_bpm]
    assert analyzer.min_bpm <= analyzer.tempo <= analyzer.max_bpm

    # 节拍帧间隔应与校正后的 BPM 保持一致
    if len(analyzer.beat_times) > 2:
        interval_s = float(np.median(np.diff(analyzer.beat_times)))
        assert abs(interval_s - 60.0 / analyzer.tempo) < 0.08


def test_beat_tempo_respects_min_bpm():
    sr = 22050
    y = _make_click_track(bpm=60.0, duration=12.0, sr=sr)
    analyzer = AudioAnalyzer(sr=sr, min_bpm=100.0, max_bpm=200.0)
    analyzer.load_array(y, sr=sr)
    analyzer._analyze_beat()

    # 60 BPM 低于下限时，应八度倍增至范围内（60→120）
    assert analyzer.min_bpm <= analyzer.tempo <= analyzer.max_bpm

    if len(analyzer.beat_times) > 2:
        interval_s = float(np.median(np.diff(analyzer.beat_times)))
        assert abs(interval_s - 60.0 / analyzer.tempo) < 0.08


def test_mapper():
    features = [
        {"time_ms": 500, "pitch_hz": 200, "rms": 0.5, "confidence": 0.9},
        {"time_ms": 1000, "pitch_hz": 400, "rms": 0.8, "confidence": 0.9},
        {"time_ms": 1500, "pitch_hz": 800, "rms": 0.3, "confidence": 0.9},
    ]
    mapper = BeatMapper(keys=4)
    notes = mapper.map_notes(features, auto_range=True)

    assert len(notes) == 3
    assert all("time" in n and "column" in n and "type" in n for n in notes)
    # 音高越高列越大（或保持单调关系，视映射而定）
    assert notes[0]["column"] <= notes[2]["column"]


def test_osu_exporter():
    notes = [
        {"time": 500, "column": 0, "type": "hit", "end_time": None},
        {"time": 1000, "column": 1, "type": "hold", "end_time": 1200},
    ]
    exporter = OsuManiaExporter(notes=notes, bpm=120.0, keys=4)
    content = exporter.to_string()

    assert "osu file format v14" in content
    assert "Mode: 3" in content
    assert "CircleSize:4" in content
    assert "500,1,0,0:0:0:0:" in content  # hit note
    assert "1000,128,0,1200:0:0:0:0:" in content  # hold note


def test_json_exporter():
    notes = [
        {"time": 500, "column": 0, "type": "hit", "end_time": None},
    ]
    exporter = JsonExporter(notes=notes, bpm=120.0, keys=4)
    data = exporter.to_dict()

    assert data["metadata"]["keys"] == 4
    assert data["timing"]["bpm"] == 120.0
    assert len(data["notes"]) == 1
    assert data["notes"][0]["type"] == "hit"


def test_export_file(tmp_path):
    notes = [
        {"time": 500, "column": 0, "type": "hit", "end_time": None},
    ]
    exporter = OsuManiaExporter(notes=notes, bpm=120.0, keys=4)
    path = tmp_path / "test.osu"
    exporter.export(str(path))
    assert path.exists()
    assert "osu file format v14" in path.read_text(encoding="utf-8")


def test_csv_exporter():
    notes = [
        {"time": 500, "column": 0, "type": "hit", "end_time": None, "speed": 10.0},
        {"time": 1000, "column": 1, "type": "hold", "end_time": 1500, "speed": 12.0},
    ]
    exporter = CsvExporter(notes=notes, bpm=120.0, keys=4, title="TestSong", time_unit="seconds")
    content = exporter.to_string()

    lines = content.strip().split("\n")
    assert lines[0].startswith("T,")
    assert "TestSong" in lines[0]
    assert lines[1].startswith("N,0,")
    assert lines[2].startswith("N,1,")

    # 秒模式：500ms → 0.500, hold duration = 500ms → 0.500
    assert "0.500" in lines[1]
    assert "0.500" in lines[2]


def test_csv_exporter_ms():
    notes = [
        {"time": 500, "column": 0, "type": "hit", "end_time": None, "speed": 10.0},
    ]
    exporter = CsvExporter(notes=notes, bpm=120.0, keys=4, time_unit="milliseconds")
    content = exporter.to_string()

    lines = content.strip().split("\n")
    # 毫秒模式：直接输出 500
    assert "500.000" in lines[1]


def test_csv_importer_seconds(tmp_path):
    csv_content = "T,1,MySong,-1,10.000\nN,0,1.000,10.0,0.500,0\nN,1,0.500,12.0,1.000,1\n"
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    importer = CsvImporter(str(csv_file), time_unit="seconds")
    info = importer.get_info()

    assert len(info["notes"]) == 2
    assert info["notes"][0]["time"] == 500
    assert info["notes"][0]["column"] == 0
    assert info["notes"][0]["type"] == "hit"
    assert info["notes"][0]["speed"] == 10.0
    assert info["notes"][1]["time"] == 1000
    assert info["notes"][1]["type"] == "hold"
    assert info["notes"][1]["end_time"] == 1500
    assert info["notes"][1]["speed"] == 12.0


def test_csv_importer_milliseconds(tmp_path):
    csv_content = "T,1,MySong,-1,10.000\nN,0,1,10.0,500,0\n"
    csv_file = tmp_path / "test_ms.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    importer = CsvImporter(str(csv_file), time_unit="milliseconds")
    info = importer.get_info()

    assert info["notes"][0]["time"] == 500


def test_csv_roundtrip(tmp_path):
    notes = [
        {"time": 500, "column": 0, "type": "hit", "end_time": None, "speed": 10.0},
        {"time": 1000, "column": 2, "type": "hold", "end_time": 1800, "speed": 15.0},
    ]
    exporter = CsvExporter(notes=notes, bpm=120.0, keys=4, title="Roundtrip", time_unit="seconds")
    csv_path = tmp_path / "roundtrip.csv"
    exporter.export(str(csv_path))

    importer = CsvImporter(str(csv_path), time_unit="seconds")
    info = importer.get_info()

    assert len(info["notes"]) == 2
    assert info["notes"][0]["time"] == notes[0]["time"]
    assert info["notes"][0]["column"] == notes[0]["column"]
    assert info["notes"][0]["type"] == notes[0]["type"]
    assert info["notes"][1]["time"] == notes[1]["time"]
    assert info["notes"][1]["end_time"] == notes[1]["end_time"]
    assert info["notes"][1]["speed"] == notes[1]["speed"]


def test_csv_importer_skips_comments(tmp_path):
    csv_content = (
        "T,1,MySong,-1,5.000\n"
        "N,0,1.000,10.0,0.500,0\n"
        "这是注释行\n"
        "\n"
        "N,0,1.000,10.0,1.000,1\n"
    )
    csv_file = tmp_path / "comments.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    importer = CsvImporter(str(csv_file), time_unit="seconds")
    assert len(importer.notes) == 2


def test_pitch_column_mapping_monotonic():
    """音高直方图均衡映射：音高越高列越靠右"""
    features = [
        {"time_ms": 500 + i * 400, "frame": 0, "pitch_hz": p, "pitch_midi": None,
         "confidence": 0.9, "rms": 0.5, "near_beat": False, "onset_strength": 1.0, "band": 0}
        for i, p in enumerate([200.0, 400.0, 800.0, 1600.0])
    ]
    mapper = BeatMapper(keys=4, use_energy_for_ln=False)
    notes = mapper.map_notes(features)

    assert len(notes) == 4
    cols = [n["column"] for n in notes]
    assert cols == sorted(cols)  # 单调不减
    assert cols[0] == 0 and cols[-1] == 3  # 覆盖两端


def test_difficulty_presets_density():
    """难度预设：单列高密度下 Easy 应比 Expert 过滤掉更多音符"""
    features = [
        {"time_ms": 100 + i * 60, "frame": 0, "pitch_hz": 440.0, "pitch_midi": None,
         "confidence": 0.9, "rms": 0.5, "near_beat": False, "onset_strength": 1.0, "band": 0}
        for i in range(40)
    ]
    counts = {}
    for name in PRESET_ORDER:
        mapper = BeatMapper(keys=1, **DIFFICULTY_PRESETS[name])
        counts[name] = len(mapper.map_notes(features))

    assert counts["Expert"] == 40  # 20ms/1.8 ≈ 11ms 密度，全保留
    assert counts["Easy"] < counts["Expert"]
    assert counts["Easy"] <= counts["Normal"] <= counts["Hard"]


def test_split_artist_title():
    title, artist = split_artist_title("C:/music/Alan Walker - Faded.mp3")
    assert title == "Faded"
    assert artist == "Alan Walker"

    title, artist = split_artist_title("plain_name.flac")
    assert title == "plain_name"
    assert artist is None


def test_get_offset_prefers_beat_near_first_onset():
    analyzer = AudioAnalyzer()
    analyzer.beat_times = np.array([0.0, 0.5, 1.0, 1.5])
    analyzer.onset_times = np.array([0.42, 1.05])
    # 最接近 0.42s 的节拍是 0.5s
    assert analyzer.get_offset_ms() == 500.0

    # 无节拍时退化为首个 onset
    analyzer2 = AudioAnalyzer()
    analyzer2.onset_times = np.array([0.42])
    assert analyzer2.get_offset_ms() == 420.0

    # 都没有时为 0
    analyzer3 = AudioAnalyzer()
    assert analyzer3.get_offset_ms() == 0.0


def test_malody_exporter():
    notes = [
        {"time": 1000, "column": 2, "type": "hit", "end_time": None, "speed": 10.0},
        {"time": 2000, "column": 1, "type": "hold", "end_time": 2500, "speed": 10.0},
    ]
    exporter = MalodyExporter(notes=notes, bpm=120.0, keys=4, title="Song", artist="Artist")
    data = exporter.to_dict()

    assert data["meta"]["song"]["title"] == "Song"
    assert data["meta"]["mode_ext"]["column"] == 4
    assert data["time"][0]["bpm"] == 120.0
    assert len(data["note"]) == 2
    # 1000ms @120BPM = 2 拍 = 半个小节 → [0, 96, 192]
    assert data["note"][0]["beat"] == [0, 96, 192]
    assert data["note"][1]["endbeat"] == [1, 48, 192]  # 2500ms = 5 拍 = 第 1 小节第 1 拍


def test_malody_roundtrip(tmp_path):
    notes = [
        {"time": 1000, "column": 2, "type": "hit", "end_time": None, "speed": 10.0},
        {"time": 2000, "column": 1, "type": "hold", "end_time": 2500, "speed": 10.0},
        {"time": 3000, "columns": [0, 3], "type": "double_hit", "end_time": None, "speed": 10.0},
    ]
    exporter = MalodyExporter(notes=notes, bpm=120.0, keys=4, title="RT", artist="A")
    path = tmp_path / "test.mc"
    exporter.export(str(path))

    importer = MalodyImporter(str(path))
    info = importer.get_info()

    assert info["keys"] == 4
    assert info["bpm"] == 120.0
    assert info["title"] == "RT"
    # double_hit 展开为两个 hit，共 4 个音符
    assert len(info["notes"]) == 4

    by_time = {(n["time"], n["column"]): n for n in info["notes"]}
    assert (1000, 2) in by_time
    assert (2000, 1) in by_time
    assert by_time[(2000, 1)]["type"] == "hold"
    assert by_time[(2000, 1)]["end_time"] == 2500
    assert (3000, 0) in by_time and (3000, 3) in by_time


def test_slice_notes():
    notes = [
        {"time": 500, "column": 0, "type": "hit", "end_time": None},
        {"time": 1000, "column": 1, "type": "hit", "end_time": None},
        {"time": 1500, "column": 2, "type": "hold", "end_time": 2500},  # 尾部超出片段
        {"time": 3000, "column": 3, "type": "hit", "end_time": None},   # 区间外
    ]
    sliced = slice_notes(notes, 900, 2000)
    assert len(sliced) == 2
    assert sliced[0]["time"] == 100
    assert sliced[1]["time"] == 600
    assert sliced[1]["end_time"] == 1100  # 2500 截断到 2000 再平移

    # 区间外为空
    assert slice_notes(notes, 5000, 6000) == []


def test_slice_audio(tmp_path):
    import soundfile as sf
    sr = 22050
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, int(sr * 2.0))).astype(np.float32)
    src = tmp_path / "song.wav"
    sf.write(str(src), y, sr)

    out_path, actual_ms = slice_audio_file(str(src), 500.0, 1500.0, str(tmp_path))
    assert os.path.exists(out_path)
    assert abs(actual_ms - 1000.0) < 20

    y2, sr2 = sf.read(out_path)
    assert sr2 == sr
    assert abs(len(y2) / sr2 - 1.0) < 0.02

    # 无效区间报错
    with pytest.raises(ValueError):
        slice_audio_file(str(src), 1500.0, 500.0, str(tmp_path))
