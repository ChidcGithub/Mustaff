"""
难度预设：将 BeatMapper 的映射参数打包为不同难度等级

所有键均为 BeatMapper.__init__ 的参数名，可直接 **kwargs 传入。
分析器参数（如 onset_sensitivity）不属于预设，保持全局一致，
这样多难度批量生成时只需分析一次音频。
"""

from typing import Any, Dict

DIFFICULTY_PRESETS: Dict[str, Dict[str, Any]] = {
    "Easy": {
        "complexity": 0.6,
        "density_filter_ms": 80.0,
        "ln_tendency": 0.3,
        "ln_threshold_ratio": 0.8,
        "enable_double_hit": False,
        "snap_to_beat": True,
        "snap_resolution": 4,
    },
    "Normal": {
        "complexity": 1.0,
        "density_filter_ms": 50.0,
        "ln_tendency": 0.5,
        "ln_threshold_ratio": 0.7,
        "enable_double_hit": False,
        "snap_to_beat": True,
        "snap_resolution": 8,
    },
    "Hard": {
        "complexity": 1.4,
        "density_filter_ms": 30.0,
        "ln_tendency": 0.6,
        "ln_threshold_ratio": 0.7,
        "enable_double_hit": True,
        "double_hit_threshold": 0.5,
        "snap_to_beat": False,
        "snap_resolution": 8,
    },
    "Expert": {
        "complexity": 1.8,
        "density_filter_ms": 20.0,
        "ln_tendency": 0.7,
        "ln_threshold_ratio": 0.6,
        "enable_double_hit": True,
        "double_hit_threshold": 0.4,
        "snap_to_beat": False,
        "snap_resolution": 8,
    },
}

# 批量生成时的固定顺序
PRESET_ORDER = ["Easy", "Normal", "Hard", "Expert"]
