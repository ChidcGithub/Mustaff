"""
Mustaff 交互式音游预览引擎

三种模式：
- auto: 自动打击预览（生成后默认可看）
- play: 可玩模式，键盘输入 + Perfect/Great/Good/Bad/Miss 判定 + 连击计分
- edit: 谱面编辑，双击添加/切换长音，拖拽移动，Del 删除，Ctrl+Z/Ctrl+Y 撤销重做

基础功能：
- 播放音频（pygame.mixer.music）
- 下落式音符渲染（tkinter Canvas）
- 时间轴拖拽控制
"""

import bisect
import copy
import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any, Optional, Set
import os
from ..colors import lane_colors as get_lane_colors, lighten_color

# pygame 用于音频播放
try:
    import pygame
except ImportError:
    pygame = None


# 判定窗口: (名称, 毫秒容差, 颜色, 准确率权重)
JUDGE_WINDOWS = [
    ("PERFECT", 30.0, "#00c853", 1.0),
    ("GREAT", 60.0, "#0078d4", 0.9),
    ("GOOD", 90.0, "#ff9800", 0.6),
    ("BAD", 120.0, "#e53935", 0.3),
]
MISS_WINDOW_MS = 120.0
HOLD_TAIL_WINDOW_MS = 150.0

# 各轨道数的键盘布局（tkinter keysym）
KEY_LAYOUTS = {
    1: ["space"],
    2: ["f", "j"],
    3: ["f", "space", "j"],
    4: ["d", "f", "j", "k"],
    5: ["d", "f", "space", "j", "k"],
    6: ["s", "d", "f", "j", "k", "l"],
    7: ["s", "d", "f", "space", "j", "k", "l"],
    8: ["a", "s", "d", "f", "j", "k", "l", "semicolon"],
    9: ["a", "s", "d", "f", "space", "j", "k", "l", "semicolon"],
}
KEY_LABELS = {"space": "SP", "semicolon": ";"}


class AudioPlayer:
    """基于 pygame.mixer.music 的音频播放器"""

    def __init__(self):
        self._loaded = False
        self._filepath: Optional[str] = None
        self._duration_ms: float = 0.0
        self._paused = False
        self._start_offset_ms: float = 0.0
        self._pause_pos_ms: float = 0.0

        if pygame and not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

    def load(self, filepath: str, duration_ms: float = 0.0) -> bool:
        """加载音频文件

        Args:
            filepath: 音频文件路径
            duration_ms: 音频总时长（毫秒）

        Returns:
            是否成功加载
        """
        if not pygame:
            return False
        if not os.path.exists(filepath):
            return False
        try:
            pygame.mixer.music.load(filepath)
            self._filepath = filepath
            self._duration_ms = duration_ms
            self._loaded = True
            self._paused = False
            self._start_offset_ms = 0.0
            self._pause_pos_ms = 0.0
            return True
        except Exception:
            return False

    def play(self, start_ms: float = 0.0) -> None:
        """从指定位置开始播放"""
        if not self._loaded or not pygame:
            return
        start_sec = max(0.0, start_ms / 1000.0)
        self._start_offset_ms = start_ms
        self._paused = False
        try:
            pygame.mixer.music.play(start=start_sec)
        except Exception:
            pass

    def pause(self) -> None:
        """暂停播放"""
        if not self._loaded or not pygame:
            return
        if self.is_playing():
            self._pause_pos_ms = self.get_position_ms()
            pygame.mixer.music.pause()
            self._paused = True

    def unpause(self) -> None:
        """继续播放"""
        if not self._loaded or not pygame:
            return
        if self._paused:
            pygame.mixer.music.unpause()
            self._paused = False

    def toggle_pause(self) -> None:
        """切换暂停状态"""
        if self._paused:
            self.unpause()
        else:
            self.pause()

    def stop(self) -> None:
        """停止播放"""
        if not self._loaded or not pygame:
            return
        pygame.mixer.music.stop()
        self._paused = False
        self._start_offset_ms = 0.0
        self._pause_pos_ms = 0.0

    def seek(self, ms: float) -> None:
        """跳转到指定位置（毫秒）"""
        if not self._loaded or not pygame:
            return
        was_playing = self.is_playing()
        self._paused = False
        self._start_offset_ms = ms
        self._pause_pos_ms = ms
        sec = max(0.0, ms / 1000.0)
        try:
            if was_playing:
                pygame.mixer.music.play(start=sec)
            else:
                pygame.mixer.music.play(start=sec)
                pygame.mixer.music.pause()
                self._paused = True
        except Exception:
            pass

    def get_position_ms(self) -> float:
        """获取当前播放位置（毫秒）"""
        if not self._loaded or not pygame:
            return 0.0
        if self._paused:
            return self._pause_pos_ms
        pos = pygame.mixer.music.get_pos()
        if pos < 0:
            pos = 0
        return self._start_offset_ms + pos

    def is_playing(self) -> bool:
        """是否正在播放"""
        if not self._loaded or not pygame:
            return False
        return pygame.mixer.music.get_busy() or self._paused

    def is_paused(self) -> bool:
        """是否处于暂停状态"""
        return self._paused

    def get_duration_ms(self) -> float:
        return self._duration_ms


class PreviewCanvas(tk.Canvas):
    """音游预览画布：下落式音符渲染 + 自动/可玩/编辑三种模式"""

    # 配色 (Light)
    BG_COLOR = "#f0f0f0"
    LANE_COLORS = ["#e0e0e0", "#d4d4d4"]
    LANE_LINE_COLOR = "#cccccc"
    JUDGE_LINE_COLOR = "#0078d4"
    HIT_EFFECT_COLOR = "#000000"
    TIMELINE_BG = "#e0e0e0"
    TIMELINE_FG = "#0078d4"
    TEXT_COLOR = "#000000"
    SELECT_COLOR = "#e53935"

    def __init__(
        self,
        parent,
        notes: List[Dict[str, Any]],
        keys: int = 4,
        duration_ms: float = 0.0,
        audio_player: Optional[AudioPlayer] = None,
        scale: float = 1.0,
        mode: str = "auto",
        on_notes_changed=None,
        snap_grid_ms: Optional[List[int]] = None,
        **kwargs
    ):
        self.keys = keys
        self.notes = notes
        self.duration_ms = duration_ms
        self.player = audio_player
        self._scale = scale
        self.mode = mode if mode in ("auto", "play", "edit") else "auto"
        self._on_notes_changed = on_notes_changed
        self._snap_grid_ms = sorted(int(t) for t in snap_grid_ms) if snap_grid_ms else None

        self._rebuild_note_index()

        self.lane_width = int(70 * scale)
        self.fall_speed = int(350 * scale)
        self.hit_disappear_ms = 120

        self._running = False
        self._hit_effects: List[Dict[str, Any]] = []
        self._hit_set: Set[Any] = set()
        self._note_items: List[int] = []
        self._effect_items: List[int] = []

        # 时间轴拖拽
        self._dragging_timeline = False
        self._last_render_ms: float = 0.0

        # 可玩模式状态
        self._targets: List[Dict[str, Any]] = []
        self._target_times: List[float] = []
        self._miss_scan_pos = 0
        self._pressed_cols: Set[int] = set()
        self._holding: Dict[int, int] = {}  # col -> note 下标（notes 列表内）
        self._combo = 0
        self._max_combo = 0
        self._judge_counts: Dict[str, int] = {}
        self._judge_weight_sum = 0.0
        self._judge_total = 0

        # 编辑模式状态
        self._selected: Optional[Dict[str, Any]] = None
        self._drag_info: Optional[Dict[str, Any]] = None
        self._undo_stack: List[List[Dict[str, Any]]] = []
        self._redo_stack: List[List[Dict[str, Any]]] = []

        # 布局参数（初始默认值，会在首次 _on_resize 时更新）
        s = self._scale
        self.canvas_width = max(int(400*s), keys * self.lane_width + int(40*s))
        self.canvas_height = int(520*s)
        self.judge_line_y = int(400*s)
        self.lane_offset_x = int(20*s)
        self._timeline_y = int(490*s)
        self._timeline_half = int(8*s)
        self._play_btn = None
        self._play_text = None
        self._timeline_fg = None
        self._time_text = None
        self._combo_text = None
        self._stats_text = None
        self._hint_text = None

        super().__init__(
            parent,
            bg=self.BG_COLOR,
            highlightthickness=0,
            **kwargs
        )

        # 绑定尺寸变化事件
        self.bind("<Configure>", self._on_resize)

        # 延迟初始化：等 Canvas 获得实际尺寸后再构建静态元素
        self.after(100, self._deferred_init)

    # ------------------ 内部工具 ------------------

    def _rebuild_note_index(self):
        """重建时间排序索引（notes 变化后调用）"""
        self._sorted_times = sorted(n["time"] for n in self.notes)
        self._note_indices_by_time = sorted(range(len(self.notes)), key=lambda i: self.notes[i]["time"])
        self._max_hold_duration_ms = max(
            (n.get("end_time", n["time"]) - n["time"] for n in self.notes if n.get("type") == "hold" and n.get("end_time")),
            default=0.0
        )
        self._min_note_speed = min(
            (n.get("speed", 10.0) for n in self.notes), default=10.0
        ) if self.notes else 10.0

    def _note_columns(self, note: Dict[str, Any]) -> List[int]:
        if note.get("type") == "double_hit":
            return [c % self.keys for c in note.get("columns", [])]
        col = note.get("column", 0)
        if col < 0 or col >= self.keys:
            col = col % self.keys
        return [col]

    def _time_to_y(self, t: float, speed: float = 10.0) -> float:
        dt = (t - self._last_render_ms) / 1000.0
        effective_speed = self.fall_speed * (speed / 10.0)
        return self.judge_line_y - dt * effective_speed

    def _y_to_time(self, y: float) -> float:
        """以 speed=10 基准把 y 坐标反算为时间（毫秒）"""
        return self._last_render_ms + (self.judge_line_y - y) / self.fall_speed * 1000.0

    def _snap_ms(self, t: float) -> float:
        """吸附到节拍网格（若有），否则对齐到 5ms"""
        if self._snap_grid_ms:
            i = bisect.bisect_left(self._snap_grid_ms, t)
            candidates = []
            if i < len(self._snap_grid_ms):
                candidates.append(self._snap_grid_ms[i])
            if i > 0:
                candidates.append(self._snap_grid_ms[i - 1])
            if candidates:
                return float(min(candidates, key=lambda g: abs(g - t)))
        return round(t / 5.0) * 5.0

    # ------------------ 初始化 ------------------

    def _deferred_init(self):
        """延迟初始化，确保 Canvas 已有实际尺寸"""
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 100 or h < 100:
            self.after(100, self._deferred_init)
            return
        self._on_resize()
        self._bind_events()

    def _on_resize(self, event=None):
        new_w = self.winfo_width()
        new_h = self.winfo_height()
        s = self._scale

        if new_w < int(100*s) or new_h < int(100*s):
            return
        if new_w == getattr(self, '_last_width', 0) and new_h == getattr(self, '_last_height', 0):
            return
        self._last_width = new_w
        self._last_height = new_h

        self.canvas_width = new_w
        self.canvas_height = new_h

        self.judge_line_y = self.canvas_height - int(110*s)
        if self.judge_line_y < int(100*s):
            self.judge_line_y = int(100*s)
        self.view_window_ms = (self.judge_line_y / self.fall_speed) * 1000

        available_width = self.canvas_width - int(40*s)
        self.lane_width = max(int(40*s), min(int(90*s), available_width // self.keys))
        total_lanes_width = self.keys * self.lane_width
        self.lane_offset_x = (self.canvas_width - total_lanes_width) // 2

        # 清除所有旧元素并重建
        self.delete("static")
        self.delete("timeline")
        self.delete("btn")
        self._build_static_elements()

        # 如果正在运行，立即重绘一帧以同步
        self._render_frame(self.player.get_position_ms() if self.player else 0.0)

    def _lane_label(self, i: int) -> str:
        if self.mode == "play":
            layout = KEY_LAYOUTS.get(self.keys, KEY_LAYOUTS[4])
            if i < len(layout):
                return KEY_LABELS.get(layout[i], layout[i]).upper()
        return f"D{i+1}"

    def _mode_hint(self) -> str:
        if self.mode == "play":
            layout = KEY_LAYOUTS.get(self.keys, KEY_LAYOUTS[4])
            keys_hint = " ".join(KEY_LABELS.get(k, k).upper() for k in layout)
            return f"可玩模式 | 按键: {keys_hint}"
        if self.mode == "edit":
            return "编辑模式 | 双击空白添加 / 双击音符切换长音 / 拖拽移动 / Del 删除 / Ctrl+Z 撤销"
        return "自动播放"

    def _build_static_elements(self):
        ox = self.lane_offset_x
        ch = self.canvas_height
        s = self._scale

        for i in range(self.keys):
            x0 = ox + i * self.lane_width
            x1 = x0 + self.lane_width
            color = self.LANE_COLORS[i % 2]
            self.create_rectangle(
                x0, 0, x1, ch,
                fill=color, outline="", tags="static"
            )

        for i in range(self.keys + 1):
            x = ox + i * self.lane_width
            self.create_line(
                x, 0, x, ch,
                fill=self.LANE_LINE_COLOR, width=max(1, int(1*s)), tags="static"
            )

        for i in range(self.keys):
            x = ox + i * self.lane_width + self.lane_width // 2
            self.create_text(
                x, self.judge_line_y + int(15*s),
                text=self._lane_label(i),
                fill=self.TEXT_COLOR,
                font=("Consolas", 10),
                tags="static"
            )

        self.create_line(
            ox, self.judge_line_y,
            ox + self.keys * self.lane_width, self.judge_line_y,
            fill=self.JUDGE_LINE_COLOR, width=max(1, int(2*s)), tags="static"
        )

        self._timeline_y = ch - int(20*s)
        self._timeline_half = int(8*s)
        self.create_rectangle(
            ox, self._timeline_y - self._timeline_half,
            ox + self.keys * self.lane_width, self._timeline_y + self._timeline_half,
            fill=self.TIMELINE_BG, outline="#444", width=1, tags="static"
        )
        self._timeline_fg = self.create_rectangle(
            ox, self._timeline_y - self._timeline_half, ox, self._timeline_y + self._timeline_half,
            fill=self.TIMELINE_FG, outline="", tags="timeline"
        )

        self._time_text = self.create_text(
            self.canvas_width // 2, int(20*s),
            text="00:00.000 / 00:00.000",
            fill=self.TEXT_COLOR,
            font=("Consolas", 12),
            tags="static"
        )

        # 可玩模式 HUD：连击（中上）与统计（左上）
        self._combo_text = self.create_text(
            self.canvas_width // 2, int(70*s),
            text="",
            fill=self.TEXT_COLOR,
            font=("Consolas", max(10, int(24*s)), "bold"),
            tags="static"
        )
        self._stats_text = self.create_text(
            int(8*s), int(44*s),
            anchor="w",
            text="",
            fill=self.TEXT_COLOR,
            font=("Consolas", max(7, int(9*s))),
            tags="static"
        )
        self._hint_text = self.create_text(
            self.canvas_width - int(8*s), int(44*s),
            anchor="e",
            text=self._mode_hint(),
            fill="#666666",
            font=("Consolas", max(7, int(8*s))),
            tags="static"
        )

        btn_y = self.judge_line_y + int(45*s)
        if btn_y + int(25*s) > self._timeline_y - int(10*s):
            btn_y = self._timeline_y - int(35*s)
        self._play_btn = self.create_rectangle(
            self.canvas_width // 2 - int(30*s), btn_y,
            self.canvas_width // 2 + int(30*s), btn_y + int(25*s),
            fill=self.JUDGE_LINE_COLOR, outline="#005a9e", width=1, tags="btn"
        )
        self._play_text = self.create_text(
            self.canvas_width // 2, btn_y + int(12*s),
            text="▶ 播放",
            fill=self.TEXT_COLOR,
            font=("Consolas", 11),
            tags="btn"
        )

    def _bind_events(self):
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Double-Button-1>", self._on_double_click)
        self.bind("<Button-3>", self._on_right_click)
        self.bind("<KeyPress>", self._on_key_press)
        self.bind("<KeyRelease>", self._on_key_release)

    # ------------------ 模式管理 ------------------

    def set_mode(self, mode: str):
        """切换 auto / play / edit 模式"""
        if mode not in ("auto", "play", "edit") or mode == self.mode:
            return
        self.mode = mode
        self._hit_set.clear()
        self._hit_effects.clear()
        self._selected = None
        self._drag_info = None
        if mode == "play":
            self._reset_play_state()
        if self._hint_text is not None:
            self.itemconfig(self._hint_text, text=self._mode_hint())
            # 轨道标签随模式变化（可玩模式显示键位）
            self._last_width = 0  # 强制走一次 _on_resize 重建静态元素
            self._on_resize()

    # ------------------ 可玩模式 ------------------

    def _reset_play_state(self):
        """重置判定状态并重建判定目标"""
        self._targets = []
        for idx, n in enumerate(self.notes):
            note_type = n.get("type", "hit")
            if note_type == "double_hit":
                for c in self._note_columns(n):
                    self._targets.append({"time": n["time"], "col": c, "kind": "hit", "note": idx, "judged": None})
            elif note_type == "hold" and n.get("end_time"):
                c = self._note_columns(n)[0]
                self._targets.append({"time": n["time"], "col": c, "kind": "head", "note": idx, "judged": None})
                self._targets.append({"time": n["end_time"], "col": c, "kind": "tail", "note": idx, "judged": None})
            else:
                c = self._note_columns(n)[0]
                self._targets.append({"time": n["time"], "col": c, "kind": "hit", "note": idx, "judged": None})
        self._targets.sort(key=lambda t: t["time"])
        self._target_times = [t["time"] for t in self._targets]
        self._miss_scan_pos = 0
        self._pressed_cols.clear()
        self._holding.clear()
        self._combo = 0
        self._max_combo = 0
        self._judge_counts = {}
        self._judge_weight_sum = 0.0
        self._judge_total = 0

    def _judge_name(self, abs_dt: float) -> str:
        for name, window, _color, _weight in JUDGE_WINDOWS:
            if abs_dt <= window:
                return name
        return "BAD"

    def _judge_weight(self, name: str) -> float:
        for n, _w, _c, weight in JUDGE_WINDOWS:
            if n == name:
                return weight
        return 0.0

    def _judge_color(self, name: str) -> str:
        for n, _w, color, _wt in JUDGE_WINDOWS:
            if n == name:
                return color
        return "#e53935"

    def _register_judgment(self, name: str, col: int, popup: bool = True):
        self._judge_counts[name] = self._judge_counts.get(name, 0) + 1
        self._judge_weight_sum += self._judge_weight(name)
        self._judge_total += 1
        if name in ("MISS",):
            self._combo = 0
        else:
            self._combo += 1
            self._max_combo = max(self._max_combo, self._combo)
        if popup:
            s = self._scale
            cx = self.lane_offset_x + col * self.lane_width + self.lane_width // 2
            self._hit_effects.append({
                "kind": "text",
                "x": cx, "y": self.judge_line_y - int(40*s),
                "text": name, "color": self._judge_color(name),
                "alpha": 1.0, "alpha_speed": 0.03, "dy": -0.6 * s,
            })

    def _accuracy(self) -> float:
        if self._judge_total == 0:
            return 100.0
        return 100.0 * self._judge_weight_sum / self._judge_total

    def _spawn_lane_flash(self, col: int):
        self._hit_effects.append({
            "kind": "lane", "col": col,
            "alpha": 0.35, "alpha_speed": 0.06,
        })

    def _play_key_press(self, col: int):
        self._spawn_lane_flash(col)
        if not self.player or not self.player.is_playing() or self.player.is_paused():
            return
        now = self.player.get_position_ms()
        best = None
        best_abs = 1e18
        for tg in self._targets:
            if tg["col"] != col or tg["judged"] is not None or tg["kind"] == "tail":
                continue
            d = abs(now - tg["time"])
            if d < best_abs:
                best, best_abs = tg, d
        if best is None or best_abs > MISS_WINDOW_MS:
            return
        result = self._judge_name(best_abs)
        best["judged"] = result
        self._register_judgment(result, col)
        if best["kind"] == "head":
            self._holding[col] = best["note"]

    def _play_key_release(self, col: int):
        if col not in self._holding:
            return
        note_idx = self._holding.pop(col)
        if not self.player or not self.player.is_playing() or self.player.is_paused():
            return
        tail = next(
            (tg for tg in self._targets
             if tg["note"] == note_idx and tg["kind"] == "tail" and tg["col"] == col and tg["judged"] is None),
            None,
        )
        if tail is None:
            return
        now = self.player.get_position_ms()
        d = abs(now - tail["time"])
        if d <= HOLD_TAIL_WINDOW_MS:
            result = self._judge_name(d)
        else:
            result = "MISS"  # 过早松开
        tail["judged"] = result
        self._register_judgment(result, col)

    def _sweep_misses(self, current_ms: float):
        """把已超过判定窗口仍未击中的目标标记为 MISS"""
        if not self._targets:
            return
        limit = current_ms - MISS_WINDOW_MS
        while self._miss_scan_pos < len(self._targets):
            tg = self._targets[self._miss_scan_pos]
            if tg["time"] > limit:
                break
            self._miss_scan_pos += 1
            if tg["judged"] is not None:
                continue
            tg["judged"] = "MISS"
            self._register_judgment("MISS", tg["col"])
            if tg["kind"] == "head":
                self._holding.pop(tg["col"], None)
                # 长音头 MISS 则尾巴一起 MISS
                for tg2 in self._targets:
                    if tg2["note"] == tg["note"] and tg2["kind"] == "tail" and tg2["judged"] is None:
                        tg2["judged"] = "MISS"
                        self._register_judgment("MISS", tg2["col"], popup=False)
                        break

    def _update_play_hud(self):
        if self._combo_text is None:
            return
        combo_text = str(self._combo) if self._combo >= 2 else ""
        self.itemconfig(self._combo_text, text=combo_text)
        c = self._judge_counts
        stats = (
            f"ACC {self._accuracy():.2f}%  "
            f"P:{c.get('PERFECT', 0)} G:{c.get('GREAT', 0)} "
            f"g:{c.get('GOOD', 0)} B:{c.get('BAD', 0)} M:{c.get('MISS', 0)}"
        )
        self.itemconfig(self._stats_text, text=stats)

    # ------------------ 编辑模式 ------------------

    def _push_undo(self):
        self._undo_stack.append(copy.deepcopy(self.notes))
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(copy.deepcopy(self.notes))
        self.notes = self._undo_stack.pop()
        self._selected = None
        self._commit_edit()

    def _redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(copy.deepcopy(self.notes))
        self.notes = self._redo_stack.pop()
        self._selected = None
        self._commit_edit()

    def _commit_edit(self):
        """编辑生效：重建索引并通知外部"""
        self._rebuild_note_index()
        if self._on_notes_changed:
            self._on_notes_changed(self.notes)
        self._render_frame(self._last_render_ms)

    def _find_note_at(self, x: int, y: int):
        """命中检测，返回 (note, part)；part 为 'move' 或 'tail'"""
        s = self._scale
        col = int((x - self.lane_offset_x) // self.lane_width)
        if col < 0 or col >= self.keys:
            return None
        best = None
        best_d = int(12*s)
        for n in self.notes:
            cols = self._note_columns(n)
            if col not in cols:
                continue
            speed = n.get("speed", 10.0)
            y_head = self._time_to_y(n["time"], speed)
            if n.get("type") == "hold" and n.get("end_time"):
                y_tail = self._time_to_y(n["end_time"], speed)
                y0, y1 = min(y_head, y_tail), max(y_head, y_tail)
                if abs(y - y_tail) <= int(8*s):
                    d = abs(y - y_tail)
                    if d < best_d:
                        best, best_d = (n, "tail"), d
                elif y0 - int(6*s) <= y <= y1 + int(6*s):
                    d = abs(y - y_head)
                    if d < best_d:
                        best, best_d = (n, "move"), d
            else:
                d = abs(y - y_head)
                if d < best_d:
                    best, best_d = (n, "move"), d
        return best

    def _edit_click(self, event):
        hit = self._find_note_at(event.x, event.y)
        if hit:
            note, part = hit
            self._selected = note
            cols = self._note_columns(note)
            self._drag_info = {
                "note": note,
                "part": part,
                "start_y": event.y,
                "orig_time": note["time"],
                "orig_end": note.get("end_time"),
                "orig_cols": cols,
                "moved": False,
            }
        else:
            self._selected = None
            self._drag_info = None
        self._render_frame(self._last_render_ms)

    def _edit_drag(self, event):
        info = self._drag_info
        if info is None:
            return
        note = info["note"]
        if not info["moved"]:
            self._push_undo()
            info["moved"] = True

        if info["part"] == "tail":
            t = self._y_to_time(event.y)
            t = max(t, note["time"] + 50)
            note["end_time"] = int(self._snap_ms(t))
        else:
            # 水平：换列；垂直：改时间（保持长音时长）
            new_col = int((event.x - self.lane_offset_x) // self.lane_width)
            new_col = max(0, min(self.keys - 1, new_col))
            dt = (info["start_y"] - event.y) / self.fall_speed * 1000.0
            new_time = max(0, int(self._snap_ms(info["orig_time"] + dt)))
            dt_snapped = new_time - info["orig_time"]
            note["time"] = new_time
            if info["orig_end"] is not None:
                note["end_time"] = int(info["orig_end"] + dt_snapped)

            orig_cols = info["orig_cols"]
            if note.get("type") == "double_hit":
                delta = new_col - min(orig_cols)
                delta = max(-min(orig_cols), min(self.keys - 1 - max(orig_cols), delta))
                note["columns"] = sorted(c + delta for c in orig_cols)
            else:
                note["column"] = new_col

        self._render_frame(self._last_render_ms)

    def _edit_double_click(self, event):
        hit = self._find_note_at(event.x, event.y)
        self._push_undo()
        if hit:
            note, _part = hit
            if note.get("type") == "hold":
                note["type"] = "hit"
                note["end_time"] = None
            elif note.get("type") == "hit":
                note["type"] = "hold"
                note["end_time"] = note["time"] + 300
        else:
            col = int((event.x - self.lane_offset_x) // self.lane_width)
            if 0 <= col < self.keys and event.y < self.judge_line_y:
                t = int(max(0, self._snap_ms(self._y_to_time(event.y))))
                self.notes.append({
                    "time": t,
                    "column": col,
                    "type": "hit",
                    "end_time": None,
                    "speed": 10.0,
                })
        self._commit_edit()

    def _edit_delete(self, note: Optional[Dict[str, Any]] = None):
        target = note or self._selected
        if target is None or target not in self.notes:
            return
        self._push_undo()
        self.notes.remove(target)
        if self._selected is target:
            self._selected = None
        self._commit_edit()

    # ------------------ 事件处理 ------------------

    def _on_click(self, event):
        self.focus_set()

        # 检查是否点击播放按钮
        if self._play_btn is None:
            return
        btn_coords = self.coords(self._play_btn)
        if btn_coords and len(btn_coords) == 4:
            if btn_coords[0] <= event.x <= btn_coords[2] and btn_coords[1] <= event.y <= btn_coords[3]:
                self._toggle_play()
                return

        s = self._scale
        ox = self.lane_offset_x
        timeline_x0 = ox
        timeline_x1 = ox + self.keys * self.lane_width
        if (timeline_x0 <= event.x <= timeline_x1 and
                self._timeline_y - int(12*s) <= event.y <= self._timeline_y + int(12*s)):
            self._dragging_timeline = True
            self._seek_to_x(event.x)
            return

        if self.mode == "edit":
            self._edit_click(event)

    def _on_drag(self, event):
        if self._dragging_timeline:
            self._seek_to_x(event.x)
            return
        if self.mode == "edit" and self._drag_info is not None:
            self._edit_drag(event)

    def _on_release(self, _event):
        self._dragging_timeline = False
        if self.mode == "edit" and self._drag_info is not None:
            moved = self._drag_info.get("moved", False)
            self._drag_info = None
            if moved:
                self._commit_edit()

    def _on_double_click(self, event):
        if self.mode == "edit":
            self._edit_double_click(event)

    def _on_right_click(self, event):
        if self.mode != "edit":
            return
        hit = self._find_note_at(event.x, event.y)
        if hit:
            self._edit_delete(note=hit[0])

    def _on_key_press(self, event):
        if self.mode == "edit":
            keysym = event.keysym
            ctrl = bool(event.state & 0x0004)
            if keysym == "Delete":
                self._edit_delete()
            elif ctrl and keysym.lower() == "z":
                if event.state & 0x0001:
                    self._redo()
                else:
                    self._undo()
            elif ctrl and keysym.lower() == "y":
                self._redo()
            return

        if self.mode != "play":
            return
        key = event.keysym.lower()
        layout = KEY_LAYOUTS.get(self.keys, KEY_LAYOUTS[4])
        if key not in layout:
            return
        col = layout.index(key)
        if col in self._pressed_cols:
            return  # 忽略键盘连发
        self._pressed_cols.add(col)
        self._play_key_press(col)

    def _on_key_release(self, event):
        if self.mode != "play":
            return
        key = event.keysym.lower()
        layout = KEY_LAYOUTS.get(self.keys, KEY_LAYOUTS[4])
        if key not in layout:
            return
        col = layout.index(key)
        self._pressed_cols.discard(col)
        self._play_key_release(col)

    def _seek_to_x(self, x: int):
        ox = self.lane_offset_x
        timeline_x0 = ox
        timeline_x1 = ox + self.keys * self.lane_width
        ratio = max(0.0, min(1.0, (x - timeline_x0) / (timeline_x1 - timeline_x0)))
        ms = ratio * self.duration_ms
        if self.player:
            was_playing = self.player.is_playing() and not self.player.is_paused()
            self.player.seek(ms)
            if not was_playing:
                self.player.pause()
        if self.mode == "play":
            self._reset_play_state()
            # 跳过已过去的音符（不计判定），当前判定窗口内的仍可击打
            if self.player:
                now = self.player.get_position_ms()
                self._miss_scan_pos = bisect.bisect_right(self._target_times, now + MISS_WINDOW_MS)
        self._render_frame(ms)

    def _toggle_play(self):
        if not self.player:
            return
        if self.player.is_playing() and not self.player.is_paused():
            self.player.pause()
            self.itemconfig(self._play_text, text="▶ 播放")
        else:
            if self.player.is_paused():
                self.player.unpause()
            else:
                pos = self.player.get_position_ms()
                if pos >= self.duration_ms - 100:
                    self.player.seek(0)
                    if self.mode == "play":
                        self._reset_play_state()
                else:
                    self.player.play(pos)
            self.itemconfig(self._play_text, text="⏸ 暂停")
            if not self._running:
                self._start_loop()

    # ------------------ 渲染循环 ------------------

    def _start_loop(self):
        self._running = True
        self._render_loop()

    def _stop_loop(self):
        self._running = False

    def _render_loop(self):
        if not self._running:
            return
        if self.player:
            current_ms = self.player.get_position_ms()
            if current_ms >= self.duration_ms and not self.player.is_paused():
                self.player.stop()
                self.itemconfig(self._play_text, text="▶ 播放")
                self._running = False
        else:
            current_ms = 0.0

        self._render_frame(current_ms)

        if self._running:
            self.after(16, self._render_loop)  # ~60fps

    def _render_frame(self, current_ms: float):
        """渲染一帧"""
        self._last_render_ms = current_ms

        for item in self._note_items:
            self.delete(item)
        self._note_items.clear()

        for item in self._effect_items:
            self.delete(item)
        self._effect_items.clear()

        self.delete("mask")

        self._update_time_text(current_ms)
        self._update_timeline(current_ms)

        playing = self.player is not None and not self.player.is_paused() and self.player.is_playing()

        if self.mode == "play" and playing:
            self._sweep_misses(current_ms)
        if self.mode == "play":
            self._update_play_hud()

        view_top_ms = current_ms - int(200 * self._scale)
        # 用最小速度扩展可视窗口，确保低速音符从屏幕顶部进入
        min_speed = max(self._min_note_speed, 1.0)
        view_bottom_ms = current_ms + self.view_window_ms * (10.0 / min_speed)

        lane_colors = get_lane_colors(self.keys)

        # 扩展起始索引以包含 body 仍在可见区域的长 hold
        adjusted_top = view_top_ms - self._max_hold_duration_ms
        start_idx = bisect.bisect_left(self._sorted_times, adjusted_top)
        end_idx = bisect.bisect_right(self._sorted_times, view_bottom_ms)

        for pos in range(start_idx, end_idx):
            idx = self._note_indices_by_time[pos]
            note = self.notes[idx]
            note_time = note["time"]
            note_type = note.get("type", "hit")
            end_time = note.get("end_time")
            note_speed = note.get("speed", 10.0)

            columns = self._note_columns(note)

            s = self._scale

            if note_type == "hold" and end_time:
                visible = not (end_time < view_top_ms or note_time > view_bottom_ms)
            else:
                visible = not (note_time < view_top_ms or note_time > view_bottom_ms)
            if not visible:
                continue

            is_selected = self.mode == "edit" and note is self._selected

            if note_type == "double_hit":
                col_min = min(columns)
                col_max = max(columns)
                note_color = lane_colors[col_min % len(lane_colors)]
                x0 = self.lane_offset_x + col_min * self.lane_width + int(4*s)
                x1 = self.lane_offset_x + (col_max + 1) * self.lane_width - int(4*s)
                y = self._time_to_y(note_time, note_speed)
                if int(-20*s) <= y <= self.canvas_height + int(20*s):
                    outline_color = self.SELECT_COLOR if is_selected else lighten_color(note_color, 0.3)
                    item = self.create_rectangle(
                        x0, y - int(6*s), x1, y + int(6*s),
                        fill=note_color, outline=outline_color, width=max(1, int(2 if is_selected else 1))
                    )
                    self._note_items.append(item)
            else:
                for col in columns:
                    note_color = lane_colors[col % len(lane_colors)]
                    x0 = self.lane_offset_x + col * self.lane_width + int(4*s)
                    x1 = self.lane_offset_x + (col + 1) * self.lane_width - int(4*s)

                    if note_type == "hold" and end_time:
                        y_head = self._time_to_y(note_time, note_speed)
                        y_tail = self._time_to_y(end_time, note_speed)
                        y_head = max(int(-20*s), min(self.canvas_height + int(20*s), y_head))
                        y_tail = max(int(-20*s), min(self.canvas_height + int(20*s), y_tail))

                        body_outline = self.SELECT_COLOR if is_selected else note_color
                        item = self.create_rectangle(
                            x0 + int(4*s), y_head, x1 - int(4*s), y_tail,
                            fill=note_color, outline=body_outline, width=1, stipple="gray25"
                        )
                        self._note_items.append(item)
                        head_outline = self.SELECT_COLOR if is_selected else lighten_color(note_color, 0.3)
                        item = self.create_rectangle(
                            x0, y_head - int(4*s), x1, y_head + int(4*s),
                            fill=note_color, outline=head_outline, width=max(1, int(2 if is_selected else 1))
                        )
                        self._note_items.append(item)
                        if is_selected:
                            # 长音尾部手柄，提示可拖拽
                            item = self.create_rectangle(
                                x0, y_tail - int(3*s), x1, y_tail + int(3*s),
                                fill="#ffffff", outline=self.SELECT_COLOR, width=max(1, int(1*s))
                            )
                            self._note_items.append(item)
                    else:
                        y = self._time_to_y(note_time, note_speed)
                        if int(-20*s) <= y <= self.canvas_height + int(20*s):
                            outline_color = self.SELECT_COLOR if is_selected else lighten_color(note_color, 0.3)
                            item = self.create_rectangle(
                                x0 + int(8*s), y - int(6*s), x1 - int(8*s), y + int(6*s),
                                fill=note_color, outline=outline_color, width=max(1, int(2 if is_selected else 1))
                            )
                            self._note_items.append(item)

            # 自动打击特效仅在 auto 模式
            if self.mode != "auto" or not playing:
                continue

            if idx not in self._hit_set:
                hit_window = max(50, min(120, int(80 * (10.0 / max(note_speed, 1.0)))))
                if current_ms >= note_time and current_ms <= note_time + hit_window:
                    self._hit_set.add(idx)
                    for col in columns:
                        self._spawn_hit_effect(col)

            if note_type == "hold" and end_time:
                hold_end_key = ("hold_end", idx)
                if hold_end_key not in self._hit_set:
                    hit_window = max(50, min(120, int(80 * (10.0 / max(note_speed, 1.0)))))
                    if current_ms >= end_time and current_ms <= end_time + hit_window:
                        self._hit_set.add(hold_end_key)
                        self._spawn_hit_effect(col, is_hold_end=True)

        # 判定线下方遮罩
        s = self._scale
        mask_bottom = self.canvas_height
        if self.judge_line_y < mask_bottom:
            self.create_rectangle(
                self.lane_offset_x, self.judge_line_y,
                self.lane_offset_x + self.keys * self.lane_width, mask_bottom,
                fill=self.BG_COLOR, outline="", tags="mask"
            )
            self.tag_raise("timeline")
            self.tag_raise("btn")

        self._update_hit_effects()

    def _update_time_text(self, current_ms: float):
        def fmt(ms: float) -> str:
            s = int(ms // 1000)
            m = s // 60
            s = s % 60
            ms_part = int(ms % 1000)
            return f"{m:02d}:{s:02d}.{ms_part:03d}"
        self.itemconfig(self._time_text, text=f"{fmt(current_ms)} / {fmt(self.duration_ms)}")

    def _update_timeline(self, current_ms: float):
        ox = self.lane_offset_x
        timeline_x0 = ox
        timeline_x1 = ox + self.keys * self.lane_width
        ratio = min(1.0, current_ms / self.duration_ms) if self.duration_ms > 0 else 0.0
        x = timeline_x0 + ratio * (timeline_x1 - timeline_x0)
        half = self._timeline_half
        self.coords(self._timeline_fg, timeline_x0, self._timeline_y - half, x, self._timeline_y + half)

    def _spawn_hit_effect(self, col: int, is_hold_end: bool = False):
        cx = self.lane_offset_x + col * self.lane_width + self.lane_width // 2
        cy = self.judge_line_y
        lane_colors = get_lane_colors(self.keys)
        base_color = lane_colors[col % len(lane_colors)]
        color = base_color if not is_hold_end else "#ffffff"
        s = self._scale

        # 外圈扩散环
        self._hit_effects.append({
            "kind": "ring",
            "cx": cx, "cy": cy,
            "size": self.lane_width // 4,
            "alpha": 0.8,
            "alpha_speed": 0.06,
            "expand_speed": int(4*s),
            "color": color,
        })
        # 内圈闪光
        self._hit_effects.append({
            "kind": "dot",
            "cx": cx, "cy": cy,
            "size": int(6*s),
            "alpha": 1.0,
            "alpha_speed": 0.15,
            "expand_speed": int(1*s),
            "color": "#ffffff",
        })

    def _update_hit_effects(self):
        new_effects = []
        s = self._scale
        for eff in self._hit_effects:
            eff["alpha"] -= eff["alpha_speed"]
            if eff["alpha"] <= 0:
                continue
            new_effects.append(eff)
            kind = eff.get("kind", "ring")

            if kind == "ring":
                eff["size"] += eff["expand_speed"]
                width = max(1, int(3 * eff["alpha"] * s))
                item = self.create_oval(
                    eff["cx"] - eff["size"], eff["cy"] - eff["size"],
                    eff["cx"] + eff["size"], eff["cy"] + eff["size"],
                    outline=eff["color"], width=width,
                )
                self._effect_items.append(item)
            elif kind == "dot":
                eff["size"] += eff["expand_speed"]
                r = int(eff["size"] * eff["alpha"])
                if r > 0:
                    item = self.create_oval(
                        eff["cx"] - r, eff["cy"] - r,
                        eff["cx"] + r, eff["cy"] + r,
                        fill=eff["color"], outline="",
                    )
                    self._effect_items.append(item)
            elif kind == "lane":
                col = eff["col"]
                x0 = self.lane_offset_x + col * self.lane_width
                x1 = x0 + self.lane_width
                # 用 stipple 模拟半透明
                item = self.create_rectangle(
                    x0, 0, x1, self.judge_line_y,
                    fill="#ffffff", outline="", stipple="gray75",
                )
                self._effect_items.append(item)
            elif kind == "text":
                eff["y"] += eff.get("dy", 0.0)
                item = self.create_text(
                    eff["x"], eff["y"],
                    text=eff["text"], fill=eff["color"],
                    font=("Consolas", max(8, int(13*s)), "bold"),
                )
                self._effect_items.append(item)

        self._hit_effects = new_effects

    # ------------------ 公共接口 ------------------

    def set_notes(self, notes: List[Dict[str, Any]], duration_ms: float):
        """设置新的音符数据"""
        self.notes = notes
        self.duration_ms = duration_ms
        self._rebuild_note_index()
        self._hit_set.clear()
        self._hit_effects.clear()
        self._selected = None
        self._drag_info = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        if self.mode == "play":
            self._reset_play_state()

    def reset(self):
        """重置预览状态"""
        self._stop_loop()
        self._hit_set.clear()
        self._hit_effects.clear()
        self._selected = None
        self._drag_info = None
        if self.mode == "play":
            self._reset_play_state()
        if self.player:
            self.player.stop()
        self._render_frame(0.0)
        self.itemconfig(self._play_text, text="▶ 播放")
