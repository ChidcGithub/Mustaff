"""
谱面预览图生成模块

使用 matplotlib 生成音游谱面的可视化预览图。
- 横向：时间轴
- 纵向：轨道列
- 不同颜色区分音符类型
"""

from typing import List, Dict, Any, Optional
import numpy as np


def generate_preview(
    notes: List[Dict[str, Any]],
    keys: int,
    duration_ms: Optional[float] = None,
    title: str = "Beatmap Preview",
    figsize: tuple = (14, 5),
    dpi: int = 150,
    save_path: Optional[str] = None,
) -> Any:
    """生成谱面预览图

    Args:
        notes: 音符列表
        keys: 轨道数
        duration_ms: 音频总时长（毫秒），用于设置横轴范围
        title: 图表标题
        figsize: 图像尺寸
        dpi: 图像DPI
        save_path: 保存路径，若为None则返回Figure对象

    Returns:
        matplotlib Figure 对象（若save_path为None）
    """
    import matplotlib
    matplotlib.use("Agg")  # 无头模式
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 颜色配置
    hit_color = "#4FC3F7"      # 浅蓝
    hold_color = "#FF8A65"     # 橙红
    lane_colors = ["#1a1a2e", "#16213e"]  # 轨道背景交替色

    # 绘制轨道背景
    for i in range(keys):
        ax.axhspan(i - 0.5, i + 0.5, color=lane_colors[i % 2], alpha=0.3, zorder=0)

    # 轨道分隔线
    for i in range(keys + 1):
        ax.axhline(i - 0.5, color="#444", linewidth=0.5, zorder=1)

    hit_times = []
    hit_cols = []
    hold_segments = []

    for note in notes:
        col = note["column"]
        t = note["time"] / 1000.0  # 转为秒
        note_type = note.get("type", "hit")

        if col < 0 or col >= keys:
            col = col % keys

        if note_type == "hold":
            end_t = (note.get("end_time") or note["time"] + 200) / 1000.0
            hold_segments.append((t, end_t, col))
        else:
            hit_times.append(t)
            hit_cols.append(col)

    # 绘制 hit 音符（小圆点）
    if hit_times:
        ax.scatter(
            hit_times,
            hit_cols,
            c=hit_color,
            s=20,
            alpha=0.9,
            zorder=3,
            edgecolors="white",
            linewidths=0.3,
        )

    # 绘制 hold 音符（横向条）
    for start_t, end_t, col in hold_segments:
        ax.barh(
            col,
            width=end_t - start_t,
            left=start_t,
            height=0.6,
            color=hold_color,
            alpha=0.85,
            zorder=2,
            edgecolor="white",
            linewidth=0.3,
        )

    # 设置纵轴
    ax.set_yticks(range(keys))
    ax.set_yticklabels([f"Key {i+1}" for i in range(keys)])
    ax.set_ylim(-0.6, keys - 0.4)
    ax.invert_yaxis()  # 第1轨在上方

    # 设置横轴
    if duration_ms:
        ax.set_xlim(0, duration_ms / 1000.0)
    elif notes:
        max_time = max(n["time"] for n in notes) / 1000.0
        ax.set_xlim(0, max_time * 1.05)
    ax.set_xlabel("Time (s)", fontsize=10)

    # 样式
    ax.set_facecolor("#0f0f1a")
    fig.patch.set_facecolor("#0f0f1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 标题
    hit_count = len(hit_times)
    hold_count = len(hold_segments)
    info_text = f"Keys: {keys}  |  Notes: {hit_count} hits + {hold_count} holds = {len(notes)} total"
    ax.set_title(f"{title}\n{info_text}", fontsize=12, color="white", pad=10)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=hit_color, edgecolor="white", label="Hit"),
        Patch(facecolor=hold_color, edgecolor="white", label="Hold (LN)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        facecolor="#1a1a2e",
        edgecolor="#555",
        labelcolor="white",
        fontsize=8,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return None
    else:
        return fig
