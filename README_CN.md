# Mustaff

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()
[![Code Style](https://img.shields.io/badge/code%20style-black-black)]()

[English Documentation](README.md)

**Mustaff** 是一款基于 Python 的自动音游谱面生成工具，通过分析音频的时间维度（节拍/音符起始点）与频谱维度（音高/音量）特征，自动生成可游玩的音游谱面。支持 osu!mania 与自定义 JSON 等多种输出格式，并提供命令行工具、图形界面以及交互式下落式预览引擎。

---

## 功能特性

- **音频分析** — 基于 `librosa` 实现 onset 检测、节拍跟踪、音高提取（pYIN 算法）与 RMS 能量分析
- **谱面映射** — 音高到轨道的映射采用直方图均衡化策略，确保音符在轨道间均匀分布；能量驱动自动生成长按音符（Hold/LN）
- **多格式导出** — 支持 osu!mania（`.osu`，1K–9K）与自定义 JSON 谱面格式
- **交互式预览** — 下落式音符实时预览，集成音频播放、自动打击光效、可拖拽时间轴与自适应画布渲染
- **多入口使用** — 提供 Python API、命令行工具（CLI）以及基于 tkinter 的图形界面（GUI），支持后台线程生成与进度显示

---

## 安装

### 环境要求

- Python 3.9 或更高版本
- 音频播放支持（pygame mixer）

### 下载预编译可执行文件

**Windows**、**macOS** 和 **Linux** 的预编译可执行文件可通过 [GitHub Releases](https://github.com/ChidcGithub/Mustaff/releases) 获取。

| 平台 | 命令行工具 | 图形界面 |
|------|-----------|---------|
| Windows | `mustaff-cli.exe` | `mustaff-gui.exe` |
| macOS | `mustaff-cli` | `mustaff-gui` |
| Linux | `mustaff-cli` | `mustaff-gui` |

无需安装 Python，下载对应平台的压缩包，解压后即可直接运行。

### 从源码安装

```bash
git clone https://github.com/ChidcGithub/Mustaff.git
cd Mustaff
pip install -e .
```

### 仅安装依赖

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 命令行工具

```bash
# 生成 4K osu!mania 谱面并附带预览图
mustaff-cli "song.mp3" --keys 4 --format both --preview -o ./output

# 生成 7K 谱面并指定难度名
mustaff-cli "song.flac" --keys 7 --format osu --difficulty "Hard" -o ./maps

# 查看全部参数
mustaff-cli --help
```

### Python API

```python
from mustaff.analyzer import AudioAnalyzer
from mustaff.mapper import BeatMapper
from mustaff.exporters.osu_mania import OsuManiaExporter

# 分析音频
analyzer = AudioAnalyzer()
analyzer.load("song.mp3").analyze()

# 映射为音符
features = analyzer.get_note_features()
mapper = BeatMapper(keys=4)
notes = mapper.map_notes(features)

# 导出谱面
exporter = OsuManiaExporter(
    notes=notes,
    bpm=analyzer.tempo,
    keys=4,
    title="歌曲标题",
    artist="艺术家",
    version="Normal",
)
exporter.export("song.osu")
```

### 图形界面

```bash
python -m mustaff.gui.app
```

启动图形界面后，选择音频文件、调整参数并点击**生成谱面**。生成完成后点击**播放预览**即可进入交互式下落式预览模式。

---

## 项目结构

```
Mustaff/
├── mustaff/
│   ├── __init__.py
│   ├── analyzer.py              # 音频分析核心（librosa）
│   ├── mapper.py                # 谱面映射逻辑
│   ├── cli.py                   # 命令行接口
│   ├── preview.py               # 静态预览图生成
│   ├── gui/
│   │   ├── app.py               # 主 GUI 程序
│   │   └── preview_player.py    # 交互式音游预览引擎
│   └── exporters/
│       ├── base.py
│       ├── osu_mania.py         # osu!mania .osu 导出器
│       └── json_exporter.py     # 自定义 JSON 导出器
├── tests/
├── examples/
├── requirements.txt
├── pyproject.toml
└── README_CN.md
```

---

## 技术细节

| 模块 | 技术方案 |
|------|----------|
| Onset 检测 | `librosa.onset.onset_detect` |
| 节拍跟踪 | `librosa.beat.beat_track` |
| 音高提取 | `librosa.pyin`（pYIN 算法） |
| 能量分析 | `librosa.feature.rms` |
| 列映射 | 直方图均衡化（基于分位数的均匀分布） |
| 音频播放 | `pygame.mixer.music` |
| GUI 渲染 | `tkinter.Canvas` 动态自适应 |
| 静态预览 | `matplotlib`（暗色主题） |

---

## GitHub Actions 自动构建

本项目使用 [GitHub Actions](https://github.com/ChidcGithub/Mustaff/actions) 在每次推送到 main 分支时自动构建并发布跨平台可执行文件。

| 工作流 | 状态 |
|--------|------|
| 构建与发布 | [![Build](https://github.com/ChidcGithub/Mustaff/actions/workflows/build.yml/badge.svg)](https://github.com/ChidcGithub/Mustaff/actions) |

要触发新的版本发布，推送一个标签即可：

```bash
git tag v0.1.0
git push origin v0.1.0
```

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。
