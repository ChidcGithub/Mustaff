"""
命令行工具

Usage:
    mustaff-cli input.mp3 --output-dir ./maps --format osu --keys 4
"""

import os
import click
from .analyzer import AudioAnalyzer
from .mapper import BeatMapper
from .exporters.osu_mania import OsuManiaExporter
from .exporters.json_exporter import JsonExporter
from .preview import generate_preview


@click.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output-dir", "-o",
    default=".",
    help="输出目录",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["osu", "json", "both"], case_sensitive=False),
    default="both",
    help="输出格式",
)
@click.option(
    "--keys", "-k",
    type=click.IntRange(1, 9),
    default=4,
    help="轨道数 (1-9)",
)
@click.option(
    "--title", "-t",
    default=None,
    help="歌曲标题（默认使用文件名）",
)
@click.option(
    "--artist", "-a",
    default="Unknown Artist",
    help="艺术家",
)
@click.option(
    "--difficulty", "-d",
    default="Auto-generated",
    help="难度名",
)
@click.option(
    "--sr",
    default=22050,
    help="分析采样率",
)
@click.option(
    "--density",
    default=30.0,
    help="密度过滤阈值（毫秒）",
)
@click.option(
    "--ln-threshold",
    default=0.7,
    help="长按音符能量阈值（0.0-1.0）",
)
@click.option(
    "--preview", "-p",
    is_flag=True,
    default=False,
    help="同时生成谱面预览图 (PNG)",
)
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
):
    """Mustaff - 从音频自动生成音游曲谱

    INPUT_FILE: 输入音频文件路径（mp3/wav/flac 等）
    """
    if title is None:
        title = os.path.splitext(os.path.basename(input_file))[0]

    click.echo(f"[Mustaff] 正在分析音频: {input_file}")

    # 分析音频
    analyzer = AudioAnalyzer(sr=sr)
    analyzer.load(input_file)
    analyzer.analyze()

    click.echo(f"   检测到 BPM: {analyzer.tempo:.1f}")
    click.echo(f"   检测到 Onset 数: {len(analyzer.onset_times)}")
    click.echo(f"   音频时长: {analyzer.duration:.2f}s")

    # 获取特征并映射
    features = analyzer.get_note_features()
    mapper = BeatMapper(
        keys=keys,
        density_filter_ms=density,
        ln_threshold_ratio=ln_threshold,
    )
    notes = mapper.map_notes(features)

    click.echo(f"   生成音符数: {len(notes)}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    exported = []

    if format in ("osu", "both"):
        osu_path = os.path.join(output_dir, f"{base_name}.osu")
        exporter = OsuManiaExporter(
            notes=notes,
            bpm=analyzer.tempo,
            offset=0.0,
            keys=keys,
            title=title,
            artist=artist,
            version=difficulty,
            audio_filename=os.path.basename(input_file),
        )
        exporter.export(osu_path)
        exported.append(osu_path)
        click.echo(f"[OK] 已导出 osu!mania 谱面: {osu_path}")

    if format in ("json", "both"):
        json_path = os.path.join(output_dir, f"{base_name}.json")
        exporter = JsonExporter(
            notes=notes,
            bpm=analyzer.tempo,
            offset=0.0,
            keys=keys,
            title=title,
            artist=artist,
            version=difficulty,
        )
        exporter.export(json_path)
        exported.append(json_path)
        click.echo(f"[OK] 已导出 JSON 谱面: {json_path}")

    # 生成预览图
    if preview:
        preview_path = os.path.join(output_dir, f"{base_name}.png")
        generate_preview(
            notes=notes,
            keys=keys,
            duration_ms=int(analyzer.duration * 1000),
            title=f"{title} [{keys}K]",
            save_path=preview_path,
        )
        click.echo(f"[OK] 已导出预览图: {preview_path}")

    click.echo("\n[Done] 完成！")


if __name__ == "__main__":
    main()
