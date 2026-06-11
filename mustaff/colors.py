"""
公共配色常量和工具函数
"""

from typing import List
import colorsys


LANE_PALETTES = {
    4: ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"],
    6: ["#FF6B6B", "#FFD93D", "#4ECDC4", "#45B7D1", "#A78BFA", "#FFA07A"],
    7: ["#FF6B6B", "#FFD93D", "#4ECDC4", "#45B7D1", "#6C5CE7", "#A78BFA", "#FFA07A"],
}


def lane_colors(keys: int) -> List[str]:
    """获取指定轨道数的配色方案"""
    if keys in LANE_PALETTES:
        return LANE_PALETTES[keys]
    colors = []
    for i in range(keys):
        hue = (i / keys) * 0.85
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors
