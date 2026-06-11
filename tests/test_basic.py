"""
基础测试
"""

import os
import pytest
import numpy as np

from mustaff.analyzer import AudioAnalyzer
from mustaff.mapper import BeatMapper
from mustaff.exporters.osu_mania import OsuManiaExporter
from mustaff.exporters.json_exporter import JsonExporter
from mustaff.exporters.csv_exporter import CsvExporter
from mustaff.importers.csv_importer import CsvImporter


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
