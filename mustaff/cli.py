from __future__ import annotations

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import click
from .analyzer import AudioAnalyzer
from .mapper import BeatMapper
from .presets import DIFFICULTY_PRESETS, PRESET_ORDER
from .utils import split_artist_title
from .exporters.osu_mania import OsuManiaExporter
from .exporters.json_exporter import JsonExporter
from .exporters.csv_exporter import CsvExporter
from .exporters.malody import MalodyExporter
from .preview import generate_preview
from .clip import slice_notes, slice_audio_file


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
@click.option("--format", "-f", type=click.Choice(["osu", "json", "csv", "mc", "all"], case_sensitive=False), default="all", help="输出格式")
@click.option("--csv-time-unit", type=click.Choice(["seconds", "milliseconds"], case_sensitive=False), default="seconds", help="CSV 导出时间单位", show_default=True)
@click.option("--keys", "-k", type=click.IntRange(1, 9), default=4, help="轨道数 (1-9)")
@click.option("--title", "-t", default=None, help="歌曲标题（默认从文件名解析）")
@click.option("--artist", "-a", default="Unknown Artist", help="艺术家（默认从文件名解析）")
@click.option("--difficulty", "-d", default="Auto-generated", help="难度名")
@click.option("--preset", type=click.Choice(["custom", "Easy", "Normal", "Hard", "Expert", "all"], case_sensitive=False), default="custom", help="难度预设（覆盖手动映射参数；all 批量生成全部难度）", show_default=True)
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
@click.option("--multi-process-pitch/--no-multi-process-pitch", default=False, help="音高检测使用多进程并行", show_default=True)
@click.option("--double-hit/--no-double-hit", default=False, help="合并同时下落的音符为双音", show_default=True)
@click.option("--double-hit-threshold", default=0.5, help="双音 RMS 能量阈值 (0.0-1.0)", show_default=True)
@click.option("--ml/--no-ml", default=False, help="使用训练好的 ML 模型生成（需先运行 training 流程）", show_default=True)
@click.option("--ml-stars", default=5.0, help="ML 生成的目标难度星级 (1-10)", show_default=True)
@click.option("--clip-start", default=None, type=float, help="截取片段起点（秒），与 --clip-end 搭配")
@click.option("--clip-end", default=None, type=float, help="截取片段终点（秒），与 --clip-start 搭配")
def main(
    input_file: str,
    output_dir: str,
    format: str,
    keys: int,
    title: str,
    artist: str,
    difficulty: str,
    preset: str,
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
    multi_process_pitch: bool,
    double_hit: bool,
    double_hit_threshold: float,
    ml: bool,
    ml_stars: float,
    clip_start: Optional[float],
    clip_end: Optional[float],
):
    """Mustaff - 从音频自动生成音游曲谱

    INPUT_FILE: 输入音频文件路径（mp3/wav/flac 等）
    """
    # 从文件名解析元数据："Artist - Title.mp3"
    parsed_title, parsed_artist = split_artist_title(input_file)
    if title is None:
        title = parsed_title
    if artist == "Unknown Artist" and parsed_artist:
        artist = parsed_artist

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
        multi_process_pitch=multi_process_pitch,
    )
    analyzer.load(input_file)

    def on_progress(pct: int, msg: str):
        spinner.update(msg)

    spinner.update("分析中...")
    analyzer.analyze(progress_callback=on_progress)

    t_analyze = time.time() - t0
    spinner.stop(f"  BPM: {analyzer.tempo:.1f}  Onset: {len(analyzer.onset_times)}  ({t_analyze:.1f}s)")

    features = analyzer.get_note_features()
    offset = analyzer.get_offset_ms()

    # ML 模式：用训练模型直接生成，跳过启发式映射与预设
    ml_notes = None
    if ml:
        import sys
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from training.generate import generate_notes_ml, models_available
        except ImportError as e:
            raise click.ClickException(f"ML 模块不可用（需要 torch 和 training/ 目录）: {e}")
        if not models_available():
            raise click.ClickException("未找到训练好的模型 training/output/*.pt，请先运行训练流程")
        spinner.update("ML 生成中...")
        ml_notes = generate_notes_ml(analyzer.y, sr=analyzer.sr, keys=keys, stars=ml_stars)
        click.echo(f"  音符数 [ML {ml_stars:g}★]: {len(ml_notes)}")

    # 难度任务列表: (难度名, 输出文件名后缀, BeatMapper 参数；None 表示用 ml_notes)
    preset_key = preset if preset in DIFFICULTY_PRESETS else preset.lower()
    if ml:
        jobs = [(f"ML-{ml_stars:g}stars", "", None)]
    elif preset_key == "all":
        jobs = [(name, f" [{name}]", dict(DIFFICULTY_PRESETS[name])) for name in PRESET_ORDER]
    elif preset_key in DIFFICULTY_PRESETS:
        jobs = [(preset_key, f" [{preset_key}]", dict(DIFFICULTY_PRESETS[preset_key]))]
    else:
        jobs = [(difficulty, "", {
            "density_filter_ms": density,
            "ln_threshold_ratio": ln_threshold,
            "snap_to_beat": snap_to_beat,
            "snap_resolution": int(snap_resolution),
            "complexity": complexity,
            "ln_tendency": ln_tendency,
            "contrast": contrast,
            "enable_double_hit": double_hit,
            "double_hit_threshold": double_hit_threshold,
        })]

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    audio_filename = os.path.basename(input_file)

    # 截取片段：音频切片只切一次，各难度谱面用同一区间过滤
    do_clip = clip_start is not None or clip_end is not None
    clip_start_ms = 0.0
    clip_end_ms = analyzer.duration * 1000.0
    if do_clip:
        clip_start_ms = (clip_start or 0.0) * 1000.0
        if clip_end is not None:
            clip_end_ms = clip_end * 1000.0
        clip_end_ms = min(clip_end_ms, analyzer.duration * 1000.0)
        sliced_path, clip_len_ms = slice_audio_file(
            input_file, clip_start_ms, clip_end_ms, output_dir
        )
        audio_filename = os.path.basename(sliced_path)
        base_name = os.path.splitext(audio_filename)[0]
        offset = offset - clip_start_ms
        click.echo(f"  截取片段: {clip_start_ms/1000:.1f}s ~ {clip_end_ms/1000:.1f}s → {audio_filename}")

    export_tasks = []
    first_notes = None
    for version_label, suffix, mapper_kwargs in jobs:
        if mapper_kwargs is None:
            # ML 模式：音符已生成
            job_notes = ml_notes
            if first_notes is None:
                first_notes = job_notes
            click.echo(f"  音符数 [{version_label}]: {len(job_notes)}")
        else:
            subdivisions = None
            if mapper_kwargs.get("snap_to_beat"):
                subdivisions = analyzer.get_beat_subdivisions(
                    resolution=mapper_kwargs.get("snap_resolution", 8)
                )

            mapper = BeatMapper(keys=keys, **mapper_kwargs)
            job_notes = mapper.map_notes(
                features, beat_subdivisions=subdivisions,
                rms_full=analyzer.rms, pitches_full=analyzer.pitches,
            )
            if first_notes is None:
                first_notes = job_notes
            click.echo(f"  音符数 [{version_label}]: {len(job_notes)}")

        if do_clip:
            job_notes = slice_notes(job_notes, clip_start_ms, clip_end_ms)
            click.echo(f"  截取后 [{version_label}]: {len(job_notes)}")

        out_base = base_name + suffix
        common = dict(
            notes=job_notes, bpm=analyzer.tempo, offset=offset, keys=keys,
            title=title, artist=artist, version=version_label,
        )
        if format in ("osu", "all"):
            path = os.path.join(output_dir, f"{out_base}.osu")
            export_tasks.append(("osu", path, lambda p=path, kw=common: OsuManiaExporter(
                audio_filename=audio_filename, **kw).export(p)))
        if format in ("json", "all"):
            path = os.path.join(output_dir, f"{out_base}.json")
            export_tasks.append(("json", path, lambda p=path, kw=common: JsonExporter(**kw).export(p)))
        if format in ("csv", "all"):
            path = os.path.join(output_dir, f"{out_base}.csv")
            export_tasks.append(("csv", path, lambda p=path, kw=common: CsvExporter(
                time_unit=csv_time_unit, **kw).export(p)))
        if format in ("mc", "all"):
            path = os.path.join(output_dir, f"{out_base}.mc")
            export_tasks.append(("mc", path, lambda p=path, kw=common: MalodyExporter(
                audio_filename=audio_filename, **kw).export(p)))

    if len(export_tasks) > 1:
        spinner.update("并行导出...")
        with ThreadPoolExecutor(max_workers=min(len(export_tasks), 8)) as ex:
            fut_map = {ex.submit(task): (name, path) for name, path, task in export_tasks}
            for fut in as_completed(fut_map):
                name, path = fut_map[fut]
                try:
                    fut.result()
                    click.echo(f"[OK] 已导出 {name} 谱面: {path}")
                except Exception as e:
                    click.echo(f"[Error] 导出 {name} 失败: {e}", err=True)
    else:
        for name, path, task in export_tasks:
            try:
                task()
                click.echo(f"[OK] 已导出 {name} 谱面: {path}")
            except Exception as e:
                click.echo(f"[Error] 导出 {name} 失败: {e}", err=True)

    if preview and first_notes is not None:
        spinner.update("生成预览图...")
        preview_path = os.path.join(output_dir, f"{base_name}.png")
        duration_ms = int(clip_end_ms - clip_start_ms) if do_clip else int(analyzer.duration * 1000)
        generate_preview(
            notes=first_notes, keys=keys,
            duration_ms=duration_ms,
            title=f"{title} [{keys}K]",
            save_path=preview_path,
        )
        click.echo(f"[OK] 已导出预览图: {preview_path}")

    click.echo(f"\n[Done] 总耗时: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
