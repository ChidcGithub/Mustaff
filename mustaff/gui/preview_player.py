"""
Mustaff 交互式音游预览引擎

功能：
- 播放音频（pygame.mixer.music）
- 下落式音符渲染（tkinter Canvas）
- 自动打击效果
- 时间轴拖拽控制
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any, Optional, Set
import os
import bisect
from ..colors import lane_colors as get_lane_colors, lighten_color

# pygame 用于音频播放
try:
    import pygame
except ImportError:
    pygame = None


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

    def get_duration_ms(self) -> float:
        return self._duration_ms


class PreviewCanvas(tk.Canvas):
    """音游预览画布：下落式音符渲染 + 自动打击 + 时间轴"""

    # 配色 (Light)
    BG_COLOR = "#f0f0f0"
    LANE_COLORS = ["#e0e0e0", "#d4d4d4"]
    LANE_LINE_COLOR = "#cccccc"
    JUDGE_LINE_COLOR = "#0078d4"
    HIT_EFFECT_COLOR = "#000000"
    TIMELINE_BG = "#e0e0e0"
    TIMELINE_FG = "#0078d4"
    TEXT_COLOR = "#000000"

    def __init__(
        self,
        parent,
        notes: List[Dict[str, Any]],
        keys: int = 4,
        duration_ms: float = 0.0,
        audio_player: Optional[AudioPlayer] = None,
        scale: float = 1.0,
        **kwargs
    ):
        self.keys = keys
        self.notes = notes
        self.duration_ms = duration_ms
        self.player = audio_player
        self._scale = scale

        self._sorted_times = sorted(n["time"] for n in notes)
        self._note_indices_by_time = sorted(range(len(notes)), key=lambda i: notes[i]["time"])
        self._max_hold_duration_ms = max(
            (n.get("end_time", n["time"]) - n["time"] for n in notes if n.get("type") == "hold" and n.get("end_time")),
            default=0.0
        )
        self._min_note_speed = min(
            (n.get("speed", 10.0) for n in notes), default=10.0
        ) if notes else 10.0

        self.lane_width = int(70 * scale)
        self.fall_speed = int(350 * scale)
        self.hit_disappear_ms = 120

        self._running = False
        self._hit_effects: List[Dict[str, Any]] = []
        self._hit_set: Set[int] = set()
        self._note_items: List[int] = []
        self._effect_items: List[int] = []

        # 时间轴拖拽
        self._dragging_timeline = False

        # 布局参数（初始默认值，会在首次 _on_resize 时更新）
        s = self._scale
        self.canvas_width = max(int(400*s), keys * self.lane_width + int(40*s))
        self.canvas_height = int(520*s)
        self.judge_line_y = int(400*s)
        self.lane_offset_x = int(20*s)
        self._timeline_y = int(490*s)
        self._play_btn = None
        self._play_text = None
        self._timeline_fg = None
        self._time_text = None

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

    def _deferred_init(self):
        """延迟初始化，确保 Canvas 已有实际尺寸"""
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 100 or h < 100:
            # 再等一帧
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

    # ------------------ 初始化 ------------------

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
                text=f"D{i+1}",
                fill=self.TEXT_COLOR,
                font=("Consolas", max(8, int(10*s))),
                tags="static"
            )

        self.create_line(
            ox, self.judge_line_y,
            ox + self.keys * self.lane_width, self.judge_line_y,
            fill=self.JUDGE_LINE_COLOR, width=max(1, int(2*s)), tags="static"
        )

        self._timeline_y = ch - int(20*s)
        self.create_rectangle(
            ox, self._timeline_y - int(8*s),
            ox + self.keys * self.lane_width, self._timeline_y + int(8*s),
            fill=self.TIMELINE_BG, outline="#444", width=1, tags="static"
        )
        self._timeline_fg = self.create_rectangle(
            ox, self._timeline_y - int(8*s), ox, self._timeline_y + int(8*s),
            fill=self.TIMELINE_FG, outline="", tags="timeline"
        )

        self._time_text = self.create_text(
            self.canvas_width // 2, int(20*s),
            text="00:00.000 / 00:00.000",
            fill=self.TEXT_COLOR,
            font=("Consolas", max(8, int(12*s))),
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
            font=("Consolas", max(8, int(11*s))),
            tags="btn"
        )

    def _bind_events(self):
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    # ------------------ 事件处理 ------------------

    def _on_click(self, event):
        # 检查是否点击播放按钮
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

    def _on_drag(self, event):
        if self._dragging_timeline:
            self._seek_to_x(event.x)

    def _on_release(self, _event):
        self._dragging_timeline = False

    def _seek_to_x(self, x: int):
        ox = self.lane_offset_x
        timeline_x0 = ox
        timeline_x1 = ox + self.keys * self.lane_width
        ratio = max(0.0, min(1.0, (x - timeline_x0) / (timeline_x1 - timeline_x0)))
        ms = ratio * self.duration_ms
        if self.player:
            was_playing = self.player.is_playing() and not self.player._paused
            self.player.seek(ms)
            if not was_playing:
                self.player.pause()
        self._render_frame(ms)

    def _toggle_play(self):
        if not self.player:
            return
        if self.player.is_playing() and not self.player._paused:
            self.player.pause()
            self.itemconfig(self._play_text, text="▶ 播放")
        else:
            if self.player._paused:
                self.player.unpause()
            else:
                pos = self.player.get_position_ms()
                if pos >= self.duration_ms - 100:
                    self.player.seek(0)
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
            if current_ms >= self.duration_ms and not self.player._paused:
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
        for item in self._note_items:
            self.delete(item)
        self._note_items.clear()

        for item in self._effect_items:
            self.delete(item)
        self._effect_items.clear()

        self.delete("mask")

        self._update_time_text(current_ms)
        self._update_timeline(current_ms)

        view_top_ms = current_ms - int(200 * self._scale)
        # 用最小速度扩展可视窗口，确保低速音符从屏幕顶部进入
        min_speed = max(self._min_note_speed, 1.0)
        view_bottom_ms = current_ms + self.view_window_ms * (10.0 / min_speed)

        lane_colors = get_lane_colors(self.keys)
        playing = self.player is not None and not self.player._paused

        # 扩展起始索引以包含 body 仍在可见区域的长 hold
        adjusted_top = view_top_ms - self._max_hold_duration_ms
        start_idx = bisect.bisect_left(self._sorted_times, adjusted_top)
        end_idx = bisect.bisect_right(self._sorted_times, view_bottom_ms)

        for pos in range(start_idx, end_idx):
            idx = self._note_indices_by_time[pos]
            note = self.notes[idx]
            note_time = note["time"]
            col = note["column"]
            note_type = note.get("type", "hit")
            end_time = note.get("end_time")

            if col < 0 or col >= self.keys:
                col = col % self.keys

            note_color = lane_colors[col % len(lane_colors)]

            if note_type == "hold" and end_time:
                visible = not (end_time < view_top_ms or note_time > view_bottom_ms)
            else:
                visible = not (note_time < view_top_ms or note_time > view_bottom_ms)
            if not visible:
                continue

            def time_to_y(t: float, speed: float = 10.0) -> float:
                dt = (t - current_ms) / 1000.0
                effective_speed = self.fall_speed * (speed / 10.0)
                return self.judge_line_y - dt * effective_speed

            s = self._scale
            x0 = self.lane_offset_x + col * self.lane_width + int(4*s)
            x1 = self.lane_offset_x + (col + 1) * self.lane_width - int(4*s)
            note_speed = note.get("speed", 10.0)

            if note_type == "hold" and end_time:
                y_head = time_to_y(note_time, note_speed)
                y_tail = time_to_y(end_time, note_speed)
                y_head = max(int(-20*s), min(self.canvas_height + int(20*s), y_head))
                y_tail = max(int(-20*s), min(self.canvas_height + int(20*s), y_tail))

                item = self.create_rectangle(
                    x0 + int(4*s), y_head, x1 - int(4*s), y_tail,
                    fill=note_color, outline=note_color, width=1, stipple="gray25"
                )
                self._note_items.append(item)
                item = self.create_rectangle(
                    x0, y_head - int(4*s), x1, y_head + int(4*s),
                    fill=note_color, outline=lighten_color(note_color, 0.3), width=max(1, int(1*s))
                )
                self._note_items.append(item)
            else:
                y = time_to_y(note_time, note_speed)
                if int(-20*s) <= y <= self.canvas_height + int(20*s):
                    outline_color = lighten_color(note_color, 0.3)
                    item = self.create_rectangle(
                        x0 + int(8*s), y - int(6*s), x1 - int(8*s), y + int(6*s),
                        fill=note_color, outline=outline_color, width=max(1, int(1*s))
                    )
                    self._note_items.append(item)

            if not playing:
                continue

            if idx not in self._hit_set:
                if current_ms >= note_time and current_ms <= note_time + 80:
                    self._hit_set.add(idx)
                    self._spawn_hit_effect(col)

            if note_type == "hold" and end_time:
                hold_end_key = ("hold_end", idx)
                if hold_end_key not in self._hit_set:
                    if current_ms >= end_time and current_ms <= end_time + 80:
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
        self.coords(self._timeline_fg, timeline_x0, self._timeline_y - 8, x, self._timeline_y + 8)

    def _spawn_hit_effect(self, col: int, is_hold_end: bool = False):
        cx = self.lane_offset_x + col * self.lane_width + self.lane_width // 2
        cy = self.judge_line_y
        lane_colors = get_lane_colors(self.keys)
        base_color = lane_colors[col % len(lane_colors)]
        color = base_color if not is_hold_end else "#ffffff"
        s = self._scale

        # 外圈扩散环
        self._hit_effects.append({
            "cx": cx, "cy": cy,
            "size": self.lane_width // 4,
            "alpha": 0.8,
            "alpha_speed": 0.06,
            "expand_speed": int(4*s),
            "color": color,
            "ring": True,
        })
        # 内圈闪光
        self._hit_effects.append({
            "cx": cx, "cy": cy,
            "size": int(6*s),
            "alpha": 1.0,
            "alpha_speed": 0.15,
            "expand_speed": int(1*s),
            "color": "#ffffff",
            "ring": False,
        })

    def _update_hit_effects(self):
        new_effects = []
        s = self._scale
        for eff in self._hit_effects:
            eff["alpha"] -= eff["alpha_speed"]
            eff["size"] += eff["expand_speed"]
            if eff["alpha"] <= 0:
                continue
            new_effects.append(eff)

            if eff["ring"]:
                # 外圈：扩散圆环
                width = max(1, int(3 * eff["alpha"] * s))
                item = self.create_oval(
                    eff["cx"] - eff["size"], eff["cy"] - eff["size"],
                    eff["cx"] + eff["size"], eff["cy"] + eff["size"],
                    outline=eff["color"], width=width,
                )
                self._effect_items.append(item)
            else:
                # 内圈：实心闪点
                r = int(eff["size"] * eff["alpha"])
                if r > 0:
                    item = self.create_oval(
                        eff["cx"] - r, eff["cy"] - r,
                        eff["cx"] + r, eff["cy"] + r,
                        fill=eff["color"], outline="",
                    )
                    self._effect_items.append(item)

        self._hit_effects = new_effects

    # ------------------ 公共接口 ------------------

    def set_notes(self, notes: List[Dict[str, Any]], duration_ms: float):
        """设置新的音符数据"""
        self.notes = notes
        self.duration_ms = duration_ms
        self._sorted_times = sorted(n["time"] for n in notes)
        self._note_indices_by_time = sorted(range(len(notes)), key=lambda i: notes[i]["time"])
        self._max_hold_duration_ms = max(
            (n.get("end_time", n["time"]) - n["time"] for n in notes if n.get("type") == "hold" and n.get("end_time")),
            default=0.0
        )
        self._min_note_speed = min(
            (n.get("speed", 10.0) for n in notes), default=10.0
        ) if notes else 10.0
        self._hit_set.clear()
        self._hit_effects.clear()

    def reset(self):
        """重置预览状态"""
        self._stop_loop()
        self._hit_set.clear()
        self._hit_effects.clear()
        if self.player:
            self.player.stop()
        self._render_frame(0.0)
        self.itemconfig(self._play_text, text="▶ 播放")
