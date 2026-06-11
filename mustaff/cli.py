from __future__ import annotations

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import click
from .analyzer import AudioAnalyzer
from .mapper import BeatMapper
from .exporters.osu_mania import OsuManiaExporter
from .exporters.json_exporter import JsonExporter
from .exporters.csv_exporter import CsvExporter
from .preview import generate_preview


class ProgressSpinner:
    """简易进度动画"""

    def __init__(self):
        self._chars = "|/-\\"
        self._idx = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_msg = ""

    def start(self, msg: str = ""):
        self._current_msg = msg
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while self._running:
            ch = self._chars[self._idx % len(self._chars)]
            click.echo(f"\r{ch} {self._current_msg}", nl=False)
            self._idx += 1
            time.sleep(0.08)

    def update(self, msg: str):
        self._current_msg = msg

    def stop(self, msg: str = ""):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        spaces = " " * (len(self._current_msg) + 2)
        click.echo(f"\r{spaces}\r{msg}", nl=False)


@click.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", "-o", default=".", help="输出目录")
@click.option("--format", "-f", type=click.Choice(["osu", "json", "csv", "all"], case_sensitive=False), default="all", help="输出格式")
@click.option("--csv-time-unit", type=click.Choice(["seconds", "milliseconds"], case_sensitive=False), default="seconds", help="CSV 导出时间单位", show_default=True)
@click.option("--keys", "-k", type=click.IntRange(1, 9), default=4, help="轨道数 (1-9)")
@click.option("--title", "-t", default=None, help="歌曲标题（默认使用文件名）")
@click.option("--artist", "-a", default="Unknown Artist", help="艺术家")
@click.option("--difficulty", "-d", default="Auto-generated", help="难度名")
@click.option("--sr", default=22050, help="分析采样率")
@click.option("--density", default=30.0, help="密度过滤阈值（毫秒）")
@click.option("--ln-threshold", default=0.7, help="长按音符能量阈值（0.0-1.0）")
@click.option("--preview", "-p", is_flag=True, default=False, help="同时生成谱面预览图 (PNG)")
@click.option("--onset-sensitivity", default=0.5, help="Onset 检测灵敏度 (0.1-1.0)", show_default=True)
@click.option("--backtrack/--no-backtrack", default=False, help="启用 Onset 回溯定位", show_default=True)
@click.option("--multi-band/--no-multi-band", default=False, help="多频段 Onset 检测", show_default=True)
@click.option("--snap-to-beat", is_flag=True, default=False, help="吸附到节拍网格")
@click.option("--snap-resolution", type=click.Choice(["4", "8", "16", "32"], case_sensitive=False), default="8", help="节拍吸附精度", show_default=True)
@click.option("--min-bpm", default=50.0, help="最小 BPM", show_default=True)
@click.option("--max-bpm", default=200.0, help="最大 BPM", show_default=True)
@click.option("--complexity", default=1.0, help="谱面复杂度 (0.5-2.0)", show_default=True)
@click.option("--ln-tendency", default=0.5, help="长音倾向 (0=尽量少, 1=尽量多)", show_default=True)
@click.option("--contrast", default=1.0, help="能量对比度 (0.1-3.5)", show_default=True)
def main(
    input_file: str,
    output_dir: str,
    format: str,
    keys: int,
    title: str,
    artist: str,
    difficulty: str,
    sr: int,
    density: float,
    ln_threshold: float,
    preview: bool,
    onset_sensitivity: float,
    backtrack: bool,
    multi_band: bool,
    snap_to_beat: bool,
    snap_resolution: str,
    min_bpm: float,
    max_bpm: float,
    complexity: float,
    ln_tendency: float,
    csv_time_unit: str,
    contrast: float,
):
    """Mustaff - 从音频自动生成音游曲谱

    INPUT_FILE: 输入音频文件路径（mp3/wav/flac 等）
    """
    if title is None:
        title = os.path.splitext(os.path.basename(input_file))[0]

    spinner = ProgressSpinner()
    spinner.start("加载音频...")

    t0 = time.time()
    analyzer = AudioAnalyzer(
        sr=sr,
        onset_sensitivity=onset_sensitivity,
        backtrack=backtrack,
        multi_band=multi_band,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )
    analyzer.load(input_file)

    def on_progress(pct: int, msg: str):
        spinner.update(msg)

    spinner.update("分析中...")
    analyzer.analyze(progress_callback=on_progress)

    t_analyze = time.time() - t0
    spinner.stop(f"  BPM: {analyzer.tempo:.1f}  Onset: {len(analyzer.onset_times)}  ({t_analyze:.1f}s)")

    features = analyzer.get_note_features()

    beat_subdivisions = None
    if snap_to_beat:
        beat_subdivisions = analyzer.get_beat_subdivisions(resolution=int(snap_resolution))
        click.echo(f"  节拍网格: {len(beat_subdivisions)} 点 (1/{snap_resolution} 拍)")

    mapper = BeatMapper(
        keys=keys,
        density_filter_ms=density,
        ln_threshold_ratio=ln_threshold,
        snap_to_beat=snap_to_beat,
        snap_resolution=int(snap_resolution),
        complexity=complexity,
        ln_tendency=ln_tendency,
        contrast=contrast,
    )
    notes = mapper.map_notes(
        features, beat_subdivisions=beat_subdivisions,
        rms_full=analyzer.rms, pitches_full=analyzer.pitches,
    )

    click.echo(f"  音符数: {len(notes)}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    export_tasks = []
    if format in ("osu", "all"):
        osu_path = os.path.join(output_dir, f"{base_name}.osu")
        export_tasks.append(("osu", osu_path, lambda p=osu_path: _export_osu(
            notes=notes, bpm=analyzer.tempo, keys=keys,
            title=title, artist=artist, version=difficulty,
            audio_filename=os.path.basename(input_file), path=p,
        )))

    if format in ("json", "all"):
        json_path = os.path.join(output_dir, f"{base_name}.json")
        export_tasks.append(("json", json_path, lambda p=json_path: _export_json(
            notes=notes, bpm=analyzer.tempo, keys=keys,
            title=title, artist=artist, version=difficulty, path=p,
        )))

    if format in ("csv", "all"):
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        export_tasks.append(("csv", csv_path, lambda p=csv_path: _export_csv(
            notes=notes, bpm=analyzer.tempo, keys=keys,
            title=title, artist=artist, version=difficulty,
            time_unit=csv_time_unit, path=p,
        )))

    if len(export_tasks) > 1:
        spinner.update("并行导出...")
        try:
            with ThreadPoolExecutor(max_workers=len(export_tasks)) as ex:
                fut_map = {ex.submit(task): name for name, _, task in export_tasks}
                for fut in as_completed(fut_map):
                    try:
                        fut.result()
                        click.echo(f"[OK] 已导出 {fut_map[fut]} 谱面")
                    except Exception as e:
                        click.echo(f"[Error] 导出 {fut_map[fut]} 失败: {e}", err=True)
        except Exception as e:
            click.echo(f"[Error] 并行导出失败: {e}", err=True)
    else:
        for name, path, task in export_tasks:
            try:
                task()
                click.echo(f"[OK] 已导出 {name} 谱面: {path}")
            except Exception as e:
                click.echo(f"[Error] 导出 {name} 失败: {e}", err=True)

    if preview:
        spinner.update("生成预览图...")
        preview_path = os.path.join(output_dir, f"{base_name}.png")
        generate_preview(
            notes=notes, keys=keys,
            duration_ms=int(analyzer.duration * 1000),
            title=f"{title} [{keys}K]",
            save_path=preview_path,
        )
        click.echo(f"[OK] 已导出预览图: {preview_path}")

    click.echo(f"\n[Done] 总耗时: {time.time() - t0:.1f}s")


def _export_osu(notes, bpm, keys, title, artist, version, audio_filename, path):
    OsuManiaExporter(
        notes=notes, bpm=bpm, offset=0.0, keys=keys,
        title=title, artist=artist, version=version,
        audio_filename=audio_filename,
    ).export(path)


def _export_json(notes, bpm, keys, title, artist, version, path):
    JsonExporter(
        notes=notes, bpm=bpm, offset=0.0, keys=keys,
        title=title, artist=artist, version=version,
    ).export(path)


def _export_csv(notes, bpm, keys, title, artist, version, time_unit, path):
    CsvExporter(
        notes=notes, bpm=bpm, offset=0.0, keys=keys,
        title=title, artist=artist, version=version,
        time_unit=time_unit,
    ).export(path)


if __name__ == "__main__":
    main()
