"""
Mustaff 图形界面

基于 tkinter 的 GUI，支持：
- 选择音频文件
- 设置轨道数、难度名等参数
- 生成谱面并查看预览图
"""

import os
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Dict, Any, List

# Matplotlib 嵌入
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..analyzer import AudioAnalyzer
from ..mapper import BeatMapper
from ..exporters.osu_mania import OsuManiaExporter
from ..exporters.json_exporter import JsonExporter
from ..preview import generate_preview
from .preview_player import PreviewCanvas, AudioPlayer

try:
    import pygame
except ImportError:
    pygame = None


class MustaffGUI:
    """Mustaff 主窗口"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mustaff - 自动生成音游谱面")
        self.root.geometry("1000x700")
        self.root.minsize(900, 650)

        self.input_path: Optional[str] = None
        self.current_notes: list = []
        self.current_keys: int = 4
        self.current_duration_ms: float = 0.0
        self.current_title: str = ""
        self.current_audio_path: Optional[str] = None
        self._preview_mode = False
        self._audio_player: Optional[AudioPlayer] = None
        self._progress_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None

        self._build_ui()

    def _build_ui(self):
        # 主布局：左侧参数面板 + 右侧预览面板
        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        # ===== 左侧面板 =====
        left_frame = ttk.Frame(main_paned, width=320)
        main_paned.add(left_frame, weight=0)

        # 文件选择
        file_frame = ttk.LabelFrame(left_frame, text="音频文件", padding=10)
        file_frame.pack(fill="x", padx=5, pady=5)
        self.file_label = ttk.Label(file_frame, text="未选择文件", foreground="gray")
        self.file_label.pack(side="left", fill="x", expand=True)
        ttk.Button(file_frame, text="浏览...", command=self._browse_file).pack(side="right")

        # 参数设置
        param_frame = ttk.LabelFrame(left_frame, text="谱面参数", padding=10)
        param_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(param_frame, text="轨道数 (Keys):").grid(row=0, column=0, sticky="w", pady=3)
        self.keys_var = tk.IntVar(value=4)
        keys_combo = ttk.Combobox(param_frame, textvariable=self.keys_var, values=[1, 2, 3, 4, 5, 6, 7, 8, 9], width=8, state="readonly")
        keys_combo.grid(row=0, column=1, sticky="w", pady=3)

        ttk.Label(param_frame, text="难度名:").grid(row=1, column=0, sticky="w", pady=3)
        self.diff_var = tk.StringVar(value="Auto-generated")
        ttk.Entry(param_frame, textvariable=self.diff_var, width=20).grid(row=1, column=1, sticky="w", pady=3)

        ttk.Label(param_frame, text="密度过滤 (ms):").grid(row=2, column=0, sticky="w", pady=3)
        self.density_var = tk.DoubleVar(value=30.0)
        ttk.Spinbox(param_frame, from_=10, to=200, increment=5, textvariable=self.density_var, width=10).grid(row=2, column=1, sticky="w", pady=3)

        ttk.Label(param_frame, text="长按阈值 (0-1):").grid(row=3, column=0, sticky="w", pady=3)
        self.ln_var = tk.DoubleVar(value=0.7)
        ttk.Spinbox(param_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.ln_var, width=10).grid(row=3, column=1, sticky="w", pady=3)

        ttk.Label(param_frame, text="输出格式:").grid(row=4, column=0, sticky="w", pady=3)
        self.format_var = tk.StringVar(value="both")
        fmt_combo = ttk.Combobox(param_frame, textvariable=self.format_var, values=["osu", "json", "both"], width=10, state="readonly")
        fmt_combo.grid(row=4, column=1, sticky="w", pady=3)

        # 输出目录
        out_frame = ttk.LabelFrame(left_frame, text="输出目录", padding=10)
        out_frame.pack(fill="x", padx=5, pady=5)
        self.out_label = ttk.Label(out_frame, text=os.getcwd(), foreground="gray")
        self.out_label.pack(side="left", fill="x", expand=True)
        self.output_dir = os.getcwd()
        ttk.Button(out_frame, text="更改...", command=self._browse_out).pack(side="right")

        # 进度条
        prog_frame = ttk.LabelFrame(left_frame, text="进度", padding=5)
        prog_frame.pack(fill="x", padx=5, pady=5)
        self._status_label = ttk.Label(prog_frame, text="就绪", foreground="gray")
        self._status_label.pack(anchor="w")
        self._progress_bar = ttk.Progressbar(prog_frame, mode="determinate", maximum=100)
        self._progress_bar.pack(fill="x", pady=2)

        # 生成按钮
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill="x", padx=5, pady=10)
        self.gen_btn = ttk.Button(btn_frame, text="生成谱面", command=self._generate)
        self.gen_btn.pack(fill="x", pady=2)
        self.preview_btn = ttk.Button(btn_frame, text="刷新预览", command=self._refresh_preview, state="disabled")
        self.preview_btn.pack(fill="x", pady=2)
        self.play_preview_btn = ttk.Button(btn_frame, text="▶ 播放预览", command=self._switch_to_preview, state="disabled")
        self.play_preview_btn.pack(fill="x", pady=2)
        self.back_btn = ttk.Button(btn_frame, text="← 返回静态预览", command=self._switch_to_static, state="disabled")
        self.back_btn.pack(fill="x", pady=2)

        # 日志
        log_frame = ttk.LabelFrame(left_frame, text="日志", padding=5)
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_text = tk.Text(log_frame, height=10, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # ===== 右侧预览面板 =====
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        preview_label_frame = ttk.LabelFrame(right_frame, text="谱面预览", padding=5)
        preview_label_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # 容器框架用于切换两种预览
        self._preview_container = ttk.Frame(preview_label_frame)
        self._preview_container.pack(fill="both", expand=True)

        # Matplotlib 静态预览
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor="#0f0f1a")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#0f0f1a")
        self.ax.text(
            0.5, 0.5, "点击「生成谱面」后预览将显示在这里",
            transform=self.ax.transAxes,
            ha="center", va="center",
            fontsize=14, color="gray",
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color("#333")

        self.static_canvas = FigureCanvasTkAgg(self.fig, master=self._preview_container)
        self.static_widget = self.static_canvas.get_tk_widget()
        self.static_widget.pack(fill="both", expand=True)
        self.static_canvas.draw()

        # 交互式音游预览（初始隐藏）
        self.preview_canvas: Optional[PreviewCanvas] = None

    def _log(self, message: str):
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.root.update_idletasks()

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[
                ("音频文件", "*.mp3 *.wav *.flac *.ogg *.m4a"),
                ("MP3", "*.mp3"),
                ("WAV", "*.wav"),
                ("FLAC", "*.flac"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            self.input_path = path
            self.file_label.config(text=os.path.basename(path), foreground="black")
            self._log(f"已选择: {path}")

    def _browse_out(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_dir = path
            self.out_label.config(text=path)

    def _generate(self):
        if not self.input_path:
            messagebox.showwarning("提示", "请先选择一个音频文件")
            return
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showwarning("提示", "正在生成中，请稍候...")
            return

        # 重置进度
        self._progress_bar["value"] = 0
        self._status_label.config(text="准备开始...", foreground="black")
        self.gen_btn.config(state="disabled")
        self.preview_btn.config(state="disabled")
        self.play_preview_btn.config(state="disabled")
        self._log("=" * 40)
        self._log("开始分析...")

        # 清空队列
        while not self._progress_queue.empty():
            try:
                self._progress_queue.get_nowait()
            except queue.Empty:
                break

        # 启动后台线程
        self._worker_thread = threading.Thread(
            target=self._generate_worker,
            args=(
                self.input_path,
                self.keys_var.get(),
                self.density_var.get(),
                self.ln_var.get(),
                self.format_var.get(),
                self.diff_var.get(),
                self.output_dir,
            ),
            daemon=True,
        )
        self._worker_thread.start()
        self._poll_progress()

    def _generate_worker(self, input_path: str, keys: int, density: float,
                         ln_threshold: float, fmt: str, difficulty: str, output_dir: str):
        """后台线程执行生成任务（分析阶段使用 analyzer 内置并行）"""
        def report(step: int, total: int, msg: str):
            self._progress_queue.put({"type": "progress", "step": step, "total": total, "msg": msg})

        try:
            report(0, 100, "加载音频...")
            analyzer = AudioAnalyzer()
            analyzer.load(input_path)

            # 使用进度回调的分析
            def on_progress(pct: int, msg: str):
                report(pct, 100, msg)

            analyzer.analyze(progress_callback=on_progress)

            report(75, 100, "映射音符...")
            features = analyzer.get_note_features()
            mapper = BeatMapper(
                keys=keys,
                density_filter_ms=density,
                ln_threshold_ratio=ln_threshold,
            )
            notes = mapper.map_notes(features)

            report(85, 100, "导出文件...")
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            title = base_name
            artist = "Unknown Artist"
            exported = []

            # 并行导出多格式（I/O 密集，线程并行有效）
            export_futs = []
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2) as ex:
                if fmt in ("osu", "both"):
                    _osu_exporter = OsuManiaExporter(
                        notes=notes, bpm=analyzer.tempo, keys=keys,
                        title=title, artist=artist, version=difficulty,
                        audio_filename=os.path.basename(input_path),
                    )
                    _osu_path = os.path.join(output_dir, f"{base_name}.osu")
                    export_futs.append(("osu", _osu_path, ex.submit(_osu_exporter.export, _osu_path)))

                if fmt in ("json", "both"):
                    _json_exporter = JsonExporter(
                        notes=notes, bpm=analyzer.tempo, keys=keys,
                        title=title, artist=artist, version=difficulty,
                    )
                    _json_path = os.path.join(output_dir, f"{base_name}.json")
                    export_futs.append(("json", _json_path, ex.submit(_json_exporter.export, _json_path)))

                for name, path, fut in export_futs:
                    fut.result()
                    exported.append(f"{name}: {path}")

            # 发送完成消息
            self._progress_queue.put({
                "type": "done",
                "notes": notes,
                "keys": keys,
                "duration_ms": int(analyzer.duration * 1000),
                "title": base_name,
                "audio_path": input_path,
                "bpm": analyzer.tempo,
                "onset_count": len(analyzer.onset_times) if analyzer.onset_times is not None else 0,
                "duration_sec": analyzer.duration,
                "note_count": len(notes),
                "exported": exported,
            })

        except Exception as e:
            self._progress_queue.put({"type": "error", "msg": str(e)})

    def _poll_progress(self):
        """主线程轮询进度队列，更新 UI"""
        try:
            while True:
                item = self._progress_queue.get_nowait()
                if item["type"] == "progress":
                    pct = int(item["step"] / item["total"] * 100)
                    self._progress_bar["value"] = pct
                    self._status_label.config(text=item["msg"], foreground="black")
                    self._log(f"[{pct}%] {item['msg']}")
                elif item["type"] == "done":
                    self._progress_bar["value"] = 100
                    self._status_label.config(text="生成完成！", foreground="green")

                    # 保存状态
                    self.current_notes = item["notes"]
                    self.current_keys = item["keys"]
                    self.current_duration_ms = item["duration_ms"]
                    self.current_title = item["title"]
                    self.current_audio_path = item["audio_path"]

                    # 日志
                    self._log(f"BPM: {item['bpm']:.1f}")
                    self._log(f"Onset 数: {item['onset_count']}")
                    self._log(f"时长: {item['duration_sec']:.2f}s")
                    self._log(f"生成音符数: {item['note_count']}")
                    for exp in item["exported"]:
                        self._log(f"导出 {exp}")

                    # 显示静态预览
                    self._draw_preview()
                    self.gen_btn.config(state="normal")
                    self.preview_btn.config(state="normal")
                    self.play_preview_btn.config(state="normal")
                    self._log("[Done] 生成完成！")
                    return  # 停止轮询
                elif item["type"] == "error":
                    self._progress_bar["value"] = 0
                    self._status_label.config(text="生成失败", foreground="red")
                    self._log(f"[Error] {item['msg']}")
                    messagebox.showerror("错误", item["msg"])
                    self.gen_btn.config(state="normal")
                    return
        except queue.Empty:
            pass

        # 继续轮询
        self.root.after(50, self._poll_progress)

    def _refresh_preview(self):
        if not self.current_notes:
            messagebox.showinfo("提示", "请先生成谱面")
            return
        self._draw_preview()

    def _draw_preview(self):
        """在右侧画布绘制预览图"""
        self.ax.clear()

        notes = self.current_notes
        keys = self.current_keys
        duration_ms = self.current_duration_ms
        title = self.current_title

        # 颜色
        hit_color = "#4FC3F7"
        hold_color = "#FF8A65"
        lane_colors = ["#1a1a2e", "#16213e"]

        for i in range(keys):
            self.ax.axhspan(i - 0.5, i + 0.5, color=lane_colors[i % 2], alpha=0.3, zorder=0)
        for i in range(keys + 1):
            self.ax.axhline(i - 0.5, color="#444", linewidth=0.5, zorder=1)

        hit_times = []
        hit_cols = []
        hold_segments = []

        for note in notes:
            col = note["column"]
            t = note["time"] / 1000.0
            note_type = note.get("type", "hit")
            if col < 0 or col >= keys:
                col = col % keys
            if note_type == "hold":
                end_t = (note.get("end_time") or note["time"] + 200) / 1000.0
                hold_segments.append((t, end_t, col))
            else:
                hit_times.append(t)
                hit_cols.append(col)

        if hit_times:
            self.ax.scatter(hit_times, hit_cols, c=hit_color, s=15, alpha=0.9, zorder=3, edgecolors="white", linewidths=0.3)

        for start_t, end_t, col in hold_segments:
            self.ax.barh(col, width=end_t - start_t, left=start_t, height=0.6,
                         color=hold_color, alpha=0.85, zorder=2, edgecolor="white", linewidth=0.3)

        self.ax.set_yticks(range(keys))
        self.ax.set_yticklabels([f"Key {i+1}" for i in range(keys)], color="white")
        self.ax.set_ylim(-0.6, keys - 0.4)
        self.ax.invert_yaxis()

        if duration_ms:
            self.ax.set_xlim(0, duration_ms / 1000.0)
        elif notes:
            max_t = max(n["time"] for n in notes) / 1000.0
            self.ax.set_xlim(0, max_t * 1.05)

        self.ax.set_xlabel("Time (s)", fontsize=10, color="white")
        self.ax.set_facecolor("#0f0f1a")
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.spines["bottom"].set_color("#555")
        self.ax.spines["left"].set_color("#555")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

        hit_count = len(hit_times)
        hold_count = len(hold_segments)
        info = f"Keys: {keys}  |  Notes: {hit_count} hits + {hold_count} holds = {len(notes)} total"
        self.ax.set_title(f"{title} [{keys}K]\n{info}", fontsize=11, color="white", pad=8)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=hit_color, edgecolor="white", label="Hit"),
            Patch(facecolor=hold_color, edgecolor="white", label="Hold (LN)"),
        ]
        self.ax.legend(handles=legend_elements, loc="upper right", facecolor="#1a1a2e",
                       edgecolor="#555", labelcolor="white", fontsize=8)

        self.fig.tight_layout()
        self.static_canvas.draw()

    def _switch_to_preview(self):
        """切换到交互式音游预览"""
        if not self.current_notes or not self.current_audio_path:
            messagebox.showinfo("提示", "请先生成谱面")
            return
        if not pygame:
            messagebox.showerror("错误", "未安装 pygame，无法播放预览。\n请运行: pip install pygame")
            return

        # 隐藏静态预览
        self.static_widget.pack_forget()

        # 销毁旧的 preview_canvas
        if self.preview_canvas is not None:
            self.preview_canvas.destroy()

        # 创建音频播放器
        if self._audio_player is None:
            self._audio_player = AudioPlayer()

        # 创建新的预览画布
        self.preview_canvas = PreviewCanvas(
            self._preview_container,
            notes=self.current_notes,
            keys=self.current_keys,
            duration_ms=self.current_duration_ms,
            audio_player=self._audio_player,
        )
        self.preview_canvas.pack(fill="both", expand=True)

        # 加载音频
        ok = self._audio_player.load(self.current_audio_path, self.current_duration_ms)
        if not ok:
            messagebox.showerror("错误", "音频加载失败")
            self._switch_to_static()
            return

        self._preview_mode = True
        self.play_preview_btn.config(state="disabled")
        self.back_btn.config(state="normal")
        self.preview_canvas.focus_set()

    def _switch_to_static(self):
        """切换回静态预览"""
        if self.preview_canvas is not None:
            self.preview_canvas.reset()
            self.preview_canvas.destroy()
            self.preview_canvas = None

        if self._audio_player is not None:
            self._audio_player.stop()

        self.static_widget.pack(fill="both", expand=True)
        self._preview_mode = False
        self.play_preview_btn.config(state="normal")
        self.back_btn.config(state="disabled")

    def run(self):
        self.root.mainloop()


def run_gui():
    """启动 GUI"""
    app = MustaffGUI()
    app.run()


if __name__ == "__main__":
    run_gui()
