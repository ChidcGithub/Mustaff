from typing import List, Dict, Any, Optional
import numpy as np


LANE_PALETTES = {
    4: ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"],
    6: ["#FF6B6B", "#FFD93D", "#4ECDC4", "#45B7D1", "#A78BFA", "#FFA07A"],
    7: ["#FF6B6B", "#FFD93D", "#4ECDC4", "#45B7D1", "#6C5CE7", "#A78BFA", "#FFA07A"],
}


def _lane_colors(keys: int) -> list:
    if keys in LANE_PALETTES:
        return LANE_PALETTES[keys]
    import colorsys
    colors = []
    for i in range(keys):
        hue = (i / keys) * 0.85
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def generate_preview(
    notes: List[Dict[str, Any]],
    keys: int,
    duration_ms: Optional[float] = None,
    title: str = "Beatmap Preview",
    figsize: tuple = (14, 5),
    dpi: int = 150,
    save_path: Optional[str] = None,
) -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    lane_colors_bg = ["#1a1a2e", "#16213e"]
    lane_colors_notes = _lane_colors(keys)

    for i in range(keys):
        ax.axhspan(i - 0.5, i + 0.5, color=lane_colors_bg[i % 2], alpha=0.3, zorder=0)
    for i in range(keys + 1):
        ax.axhline(i - 0.5, color="#444", linewidth=0.5, zorder=1)

    max_time_s = 0.0
    if duration_ms:
        max_time_s = duration_ms / 1000.0
    elif notes:
        max_time_s = max(n["time"] for n in notes) / 1000.0 * 1.05

    if max_time_s > 0:
        if max_time_s <= 30:
            grid_spacing = 1
        elif max_time_s <= 120:
            grid_spacing = 5
        else:
            grid_spacing = 10

        for t in np.arange(0, max_time_s + grid_spacing, grid_spacing):
            ax.axvline(t, color="#555", linewidth=0.3, linestyle="--", alpha=0.4, zorder=0)

    hit_by_col: Dict[int, tuple] = {}
    hold_segments = []

    for note in notes:
        col = note["column"]
        t = note["time"] / 1000.0
        if col < 0 or col >= keys:
            col = col % keys
        note_type = note.get("type", "hit")
        if note_type == "hold":
            end_t = (note.get("end_time") or note["time"] + 200) / 1000.0
            hold_segments.append((t, end_t, col))
        else:
            hit_by_col.setdefault(col, ([], []))
            hit_by_col[col][0].append(t)
            hit_by_col[col][1].append(col)

    for col, (times, cols) in hit_by_col.items():
        color = lane_colors_notes[col % len(lane_colors_notes)]
        ax.scatter(
            times, cols,
            c=color, s=35, alpha=0.9, zorder=3,
            edgecolors="white", linewidths=0.5,
        )

    for start_t, end_t, col in hold_segments:
        color = lane_colors_notes[col % len(lane_colors_notes)]
        ax.barh(
            col, width=end_t - start_t, left=start_t,
            height=0.65, color=color, alpha=0.7, zorder=2,
            edgecolor=color, linewidth=0.5,
        )

    ax.set_yticks(range(keys))
    ax.set_yticklabels([f"Key {i+1}" for i in range(keys)])
    ax.set_ylim(-0.6, keys - 0.4)
    ax.invert_yaxis()

    if max_time_s > 0:
        ax.set_xlim(0, max_time_s)

    ax.set_xlabel("Time (s)", fontsize=10, color="white")

    ax.set_facecolor("#0f0f1a")
    fig.patch.set_facecolor("#0f0f1a")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    hit_count = sum(len(t) for t, _ in hit_by_col.values())
    hold_count = len(hold_segments)
    info_text = f"Keys: {keys}  |  Notes: {hit_count} hits + {hold_count} holds = {len(notes)} total"
    ax.set_title(f"{title}\n{info_text}", fontsize=12, color="white", pad=10)

    legend_elements = []
    for i in range(min(keys, len(lane_colors_notes))):
        legend_elements.append(
            Patch(facecolor=lane_colors_notes[i], edgecolor="white", label=f"Lane {i+1}")
        )
    legend_elements.append(
        Patch(facecolor=lane_colors_notes[0], edgecolor="white", alpha=0.5, label="Hold (LN)")
    )
    legend = ax.legend(
        handles=legend_elements,
        loc="upper right",
        facecolor="#1a1a2e",
        edgecolor="#555",
        labelcolor="white",
        fontsize=7,
        ncol=2,
    )
    try:
        legend.legend_handles[-1].set_alpha(0.5)
    except (AttributeError, IndexError):
        try:
            legend.legendHandles[-1].set_alpha(0.5)
        except (AttributeError, IndexError):
            pass

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return None
    else:
        return fig
