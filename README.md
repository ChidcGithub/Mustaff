# Mustaff

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()
[![Code Style](https://img.shields.io/badge/code%20style-black-black)]()
[![Build](https://github.com/ChidcGithub/Mustaff/actions/workflows/build.yml/badge.svg)](https://github.com/ChidcGithub/Mustaff/actions)

[中文文档](README_CN.md)

**Mustaff** is a Python-based tool that automatically generates rhythm game beatmaps by analyzing the temporal (beat/onset) and spectral (pitch/volume) features of audio files. It supports multiple output formats including osu!mania and custom JSON, and provides a command-line interface, a graphical user interface, and an interactive rhythm game preview engine.

---

## Features

- **Audio Analysis** — Powered by `librosa` for onset detection, beat tracking, pitch extraction (pYIN), and RMS energy analysis
- **Beat Mapping** — Pitch-to-column mapping with histogram equalization for balanced distribution across lanes; energy-based hold note (LN) generation; auto-resolve hold/short-note column conflicts
- **Multi-Format Export** — osu!mania (`.osu`, 1K–9K) and custom JSON beatmap formats
- **Interactive Preview** — Real-time falling-note preview with audio playback, auto-hit effects, seekable timeline, and dynamic canvas rendering
- **Double Hit (Chord)** — Optionally merge simultaneous notes into chord hits with configurable RMS energy threshold; resolves column conflicts by relocating to free lanes
- **Multiple Interfaces** — Python API, CLI tool, and tkinter-based GUI with threaded generation and progress reporting

---

## Installation

### Requirements

- Python 3.9 or higher
- Audio playback support (pygame mixer)

### Download Pre-built Binaries

Pre-built executables for **Windows**, **macOS**, and **Linux** are available via [GitHub Releases](https://github.com/ChidcGithub/Mustaff/releases).

| Platform | CLI | GUI |
|----------|-----|-----|
| Windows | `mustaff-cli.exe` | `mustaff-gui.exe` |
| macOS | `mustaff-cli` | `mustaff-gui` |
| Linux | `mustaff-cli` | `mustaff-gui` |

No Python installation is required. Download the archive for your platform, extract, and run directly.

### Install from Source

```bash
git clone https://github.com/ChidcGithub/Mustaff.git
cd Mustaff
pip install -e .
```

### Install Dependencies Only

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Command Line

```bash
# Generate a 4K osu!mania beatmap with preview image
mustaff-cli "song.mp3" --keys 4 --format both --preview -o ./output

# 7K beatmap with custom difficulty
mustaff-cli "song.flac" --keys 7 --format osu --difficulty "Hard" -o ./maps

# View all options
mustaff-cli --help
```

### Python API

```python
from mustaff.analyzer import AudioAnalyzer
from mustaff.mapper import BeatMapper
from mustaff.exporters.osu_mania import OsuManiaExporter

# Analyze audio
analyzer = AudioAnalyzer()
analyzer.load("song.mp3").analyze()

# Map to notes
features = analyzer.get_note_features()
mapper = BeatMapper(keys=4)
notes = mapper.map_notes(features)

# Export
exporter = OsuManiaExporter(
    notes=notes,
    bpm=analyzer.tempo,
    keys=4,
    title="Song Title",
    artist="Artist",
    version="Normal",
)
exporter.export("song.osu")
```

### GUI

```bash
python -m mustaff.gui.app
```

Launch the graphical interface, select an audio file, adjust parameters, and click **Generate**. Use **Play Preview** to enter the interactive falling-note preview mode.

---

## Project Structure

```
Mustaff/
├── mustaff/
│   ├── __init__.py
│   ├── analyzer.py              # Audio analysis core (librosa)
│   ├── mapper.py                # Beat mapping logic
│   ├── cli.py                   # Command-line interface
│   ├── preview.py               # Static preview image generation
│   ├── gui/
│   │   ├── app.py               # Main GUI application
│   │   └── preview_player.py    # Interactive rhythm game preview engine
│   └── exporters/
│       ├── base.py
│       ├── osu_mania.py         # osu!mania .osu exporter
│       └── json_exporter.py     # Custom JSON exporter
├── tests/
├── examples/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Technical Details

| Module | Technology |
|--------|------------|
| Onset Detection | `librosa.onset.onset_detect` |
| Beat Tracking | `librosa.beat.beat_track` |
| Pitch Extraction | `librosa.pyin` (pYIN algorithm) |
| Energy Analysis | RMS via `librosa.feature.rms` |
| Column Mapping | Histogram equalization (quantile-based distribution) |
| Audio Playback | `pygame.mixer.music` |
| GUI Rendering | `tkinter.Canvas` with dynamic resize |
| Static Preview | `matplotlib` (dark theme) |

---

## GitHub Actions CI

This project uses [GitHub Actions](https://github.com/ChidcGithub/Mustaff/actions) to automatically build and release cross-platform executables on every push to the main branch.

| Workflow | Status |
|----------|--------|
| Build & Release | [![Build](https://github.com/ChidcGithub/Mustaff/actions/workflows/build.yml/badge.svg)](https://github.com/ChidcGithub/Mustaff/actions) |

To trigger a new release, push a tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

---

## License

This project is licensed under the [MIT License](LICENSE).
