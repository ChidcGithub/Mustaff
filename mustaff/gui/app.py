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
from typing import Optional
import numpy as np

import sv_ttk

# Matplotlib 嵌入
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.font_manager as fm
for _fname in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc",
               "/System/Library/Fonts/PingFang.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
    if os.path.exists(_fname):
        fm.fontManager.addfont(_fname)
        matplotlib.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_fname).get_name()] + matplotlib.rcParams["font.sans-serif"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..analyzer import AudioAnalyzer
from ..mapper import BeatMapper
from ..exporters.osu_mania import OsuManiaExporter
from ..exporters.json_exporter import JsonExporter
from ..importers.json_importer import JsonImporter
from ..importers.osu_importer import OsuImporter
from ..importers.csv_importer import CsvImporter
from ..exporters.csv_exporter import CsvExporter
from ..colors import lane_colors
from .preview_player import PreviewCanvas, AudioPlayer

try:
    import pygame
except ImportError:
    pygame = None

# DPI awareness — 必须在 tk.Tk() 创建前设置
if os.name == "nt":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


class MustaffGUI:
    """Mustaff 主窗口"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mustaff")

        self._scale = self._detect_scale()

        sv_ttk.set_theme("light")
        self._scale_fonts()
        base_w, base_h = 1000, 700
        self.root.geometry(f"{int(base_w * self._scale)}x{int(base_h * self._scale)}")
        self.root.minsize(int(900 * self._scale), int(650 * self._scale))
        self.root.state("zoomed")

        self.input_path: Optional[str] = None
        self.current_notes: list = []
        self.current_keys: int = 4
        self.current_duration_ms: float = 0.0
        self.current_title: str = ""
        self.current_audio_path: Optional[str] = None
        self.current_bpm: float = 0.0
        self.current_artist: str = "Unknown Artist"
        self.output_dir: str = os.getcwd()
        self._preview_mode = False
        self._audio_player: Optional[AudioPlayer] = None
        self._progress_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._taskbar = None
        self._hwnd = None
        self._init_taskbar()

        self._build_ui()

    def _detect_scale(self) -> float:
        import platform
        system = platform.system()
        if system == "Windows":
            import ctypes
            try:
                hdc = ctypes.windll.user32.GetDC(0)
                dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
                ctypes.windll.user32.ReleaseDC(0, hdc)
            except Exception:
                dpi = 96
        else:
            try:
                dpi = int(float(self.root.tk.call("tk", "scaling")) * 72)
            except Exception:
                dpi = 96
        scale = dpi / 96.0
        self.root.tk.call("tk", "scaling", scale)
        return scale

    def _scale_fonts(self):
        """缩放 sv_ttk 负数字体和精灵图元素尺寸"""
        import tkinter.font as tkfont
        s = self._scale
        for name in self.root.tk.call("font", "names"):
            try:
                font_obj = tkfont.nametofont(name)
                size = font_obj.cget("size")
                if isinstance(size, int) and size < 0:
                    new_size = max(8, int(abs(size) * s))
                    font_obj.configure(size=-new_size)
            except Exception:
                pass
        for elem, opts in {
            "Combobox.arrow":        {"-width": int(34 * s)},
            "Spinbox.uparrow":       {"-width": int(34 * s), "-height": int(16 * s)},
            "Spinbox.downarrow":     {"-width": int(34 * s), "-height": int(16 * s)},
            "Scale.slider":          {"-width": int(20 * s), "-height": int(20 * s)},
            "Menubutton.indicator":  {"-width": int(10 * s)},
            "OptionMenu.indicator":  {"-width": int(10 * s)},
        }.items():
            try:
                self.root.tk.call("ttk::style", "element", "configure", elem,
                                  *sum(opts.items(), ()))
            except Exception:
                pass

    def _init_taskbar(self):
        """初始化 Windows 任务栏进度条 (ITaskbarList3 via comtypes)"""
        import sys
        if sys.platform != "win32":
            return
        self._taskbar = None
        self._hwnd = self.root.winfo_id()
        self._TBPF_NORMAL = 0x2
        self._TBPF_NOPROGRESS = 0x0
        try:
            import comtypes
            import comtypes.client
            from comtypes import GUID, COMMETHOD, HRESULT
            from ctypes.wintypes import HWND

            class ITaskbarList3(comtypes.IUnknown):
                _iid_ = GUID("{EA1AFB91-9E28-4B86-90E9-9E9F8A5EEFAF}")
                _methods_ = [
                    COMMETHOD([], HRESULT, "HrInit"),
                    COMMETHOD([], HRESULT, "AddTab", (["in"], HWND, "hwnd")),
                    COMMETHOD([], HRESULT, "DeleteTab", (["in"], HWND, "hwnd")),
                    COMMETHOD([], HRESULT, "ActivateTab", (["in"], HWND, "hwnd")),
                    COMMETHOD([], HRESULT, "SetActiveAlt", (["in"], HWND, "hwnd")),
                    COMMETHOD([], HRESULT, "MarkFullscreenWindow",
                              (["in"], HWND, "hwnd"), (["in"], comtypes.c_int, "fFullscreen")),
                    COMMETHOD([], HRESULT, "SetProgressValue",
                              (["in"], HWND, "hwnd"),
                              (["in"], ctypes.c_ulonglong, "ullCompleted"),
                              (["in"], ctypes.c_ulonglong, "ullTotal")),
                    COMMETHOD([], HRESULT, "SetProgressState",
                              (["in"], HWND, "hwnd"),
                              (["in"], comtypes.c_int, "tbpFlags")),
                ]

            self._taskbar = comtypes.client.CreateObject(
                "{56FDF344-FD6D-11d0-958A-006097C9A090}", interface=ITaskbarList3)
            self._taskbar.HrInit()
        except Exception:
            self._taskbar = None

    def _set_taskbar_progress(self, pct: int):
        """更新 Windows 任务栏进度条"""
        if not self._taskbar:
            return
        try:
            self._taskbar.SetProgressState(self._hwnd, self._TBPF_NORMAL)
            self._taskbar.SetProgressValue(self._hwnd, pct, 100)
        except Exception:
            pass

    def _clear_taskbar(self):
        """清除 Windows 任务栏进度条"""
        if not self._taskbar:
            return
        try:
            self._taskbar.SetProgressState(self._hwnd, self._TBPF_NOPROGRESS)
        except Exception:
            pass

    def _show_toast(self, title: str, msg: str):
        """显示 Windows 气泡通知"""
        import sys
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes

            NIM_ADD = 0x0
            NIM_MODIFY = 0x1
            NIM_DELETE = 0x2
            NIF_INFO = 0x10
            NIF_ICON = 0x2
            NIF_MESSAGE = 0x1

            class NOTIFYICONDATAW(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("hWnd", wintypes.HWND),
                    ("uID", wintypes.UINT),
                    ("uFlags", wintypes.UINT),
                    ("uCallbackMessage", wintypes.UINT),
                    ("hIcon", wintypes.HICON),
                    ("szTip", ctypes.c_wchar * 128),
                    ("dwState", wintypes.DWORD),
                    ("dwStateMask", wintypes.DWORD),
                    ("szInfo", ctypes.c_wchar * 256),
                    ("uTimeoutOrVersion", wintypes.UINT),
                    ("szInfoTitle", ctypes.c_wchar * 64),
                    ("dwInfoFlags", wintypes.DWORD),
                    ("guidItem", ctypes.c_wchar * 39),
                    ("hBalloonIcon", wintypes.HICON),
                ]

            shell32 = ctypes.windll.shell32
            user32 = ctypes.windll.user32

            nid = NOTIFYICONDATAW()
            nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
            nid.hWnd = self._hwnd
            nid.uID = 1
            nid.uFlags = NIF_ICON | NIF_MESSAGE
            nid.uCallbackMessage = 0x400
            nid.hIcon = user32.LoadIconW(None, 1)
            nid.szTip = "Mustaff"
            shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(nid))

            nid.uFlags = NIF_INFO
            nid.szInfo = msg
            nid.szInfoTitle = title
            nid.dwInfoFlags = 0x1
            shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(nid))

            def remove_icon():
                import time
                time.sleep(5)
                try:
                    shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(nid))
                except Exception:
                    pass
            threading.Thread(target=remove_icon, daemon=True).start()
        except Exception:
            pass

    def _build_ui(self):
        s = self._scale

        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=int(5*s), pady=int(5*s))

        left_frame = ttk.Frame(main_paned, width=int(320*s))
        main_paned.add(left_frame, weight=0)

        log_frame = ttk.LabelFrame(left_frame, text="日志", padding=int(5*s))
        log_frame.pack(side="bottom", fill="both", padx=int(5*s), pady=int(5*s))
        self.log_text = tk.Text(log_frame, height=max(10, int(10*s)), state="disabled", wrap="word")
        self.log_text.pack(fill="x")
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        self._left_canvas = tk.Canvas(left_frame, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=self._left_canvas.yview)
        self._scrollable_inner = ttk.Frame(self._left_canvas)

        self._scrollable_inner.bind("<Configure>", lambda e: self._left_canvas.configure(scrollregion=self._left_canvas.bbox("all")))
        self._canvas_window = self._left_canvas.create_window((0, 0), window=self._scrollable_inner, anchor="nw")
        self._left_canvas.configure(yscrollcommand=left_scrollbar.set)

        self._left_canvas.bind("<Configure>", self._on_canvas_resize)

        self._left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")

        self._left_canvas.bind("<Enter>", lambda e: self._bind_mousewheel())
        self._left_canvas.bind("<Leave>", lambda e: self._unbind_mousewheel())

        inner = self._scrollable_inner

        file_frame = ttk.LabelFrame(inner, text="音频文件", padding=int(10*s))
        file_frame.pack(fill="x", padx=int(5*s), pady=int(5*s))
        file_frame.columnconfigure(0, weight=1)
        self.file_label = ttk.Label(file_frame, text="未选择文件", foreground="gray")
        self.file_label.grid(row=0, column=0, sticky="ew", padx=(0, int(5*s)))
        ttk.Button(file_frame, text="浏览...", command=self._browse_file).grid(row=0, column=1)

        import_frame = ttk.LabelFrame(inner, text="导入谱面", padding=int(10*s))
        import_frame.pack(fill="x", padx=int(5*s), pady=int(5*s))
        import_frame.columnconfigure(0, weight=1)
        self.import_label = ttk.Label(import_frame, text="未导入谱面", foreground="gray")
        self.import_label.grid(row=0, column=0, sticky="ew", padx=(0, int(5*s)))
        ttk.Button(import_frame, text="导入...", command=self._import_chart).grid(row=0, column=1)

        param_frame = ttk.LabelFrame(inner, text="谱面参数", padding=int(10*s))
        param_frame.pack(fill="x", padx=int(5*s), pady=int(5*s))

        ttk.Label(param_frame, text="轨道数 (Keys):").grid(row=0, column=0, sticky="w", pady=int(3*s))
        self.keys_var = tk.IntVar(value=4)
        keys_combo = ttk.Combobox(param_frame, textvariable=self.keys_var, values=[1, 2, 3, 4, 5, 6, 7, 8, 9], width=8, state="readonly")
        keys_combo.grid(row=0, column=1, sticky="w", pady=int(3*s))

        ttk.Label(param_frame, text="难度名:").grid(row=1, column=0, sticky="w", pady=int(3*s))
        self.diff_var = tk.StringVar(value="Auto-generated")
        ttk.Entry(param_frame, textvariable=self.diff_var, width=20).grid(row=1, column=1, sticky="w", pady=int(3*s))

        ttk.Label(param_frame, text="密度过滤 (ms):").grid(row=2, column=0, sticky="w", pady=int(3*s))
        self.density_var = tk.DoubleVar(value=30.0)
        ttk.Spinbox(param_frame, from_=10, to=200, increment=5, textvariable=self.density_var, width=10).grid(row=2, column=1, sticky="w", pady=int(3*s))

        ttk.Label(param_frame, text="长按阈值 (0-1):").grid(row=3, column=0, sticky="w", pady=int(3*s))
        self.ln_var = tk.DoubleVar(value=0.7)
        ttk.Spinbox(param_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.ln_var, width=10).grid(row=3, column=1, sticky="w", pady=int(3*s))

        # ===== 高级选项 =====
        adv_frame = ttk.LabelFrame(inner, text="高级选项", padding=int(10*s))
        adv_frame.pack(fill="x", padx=int(5*s), pady=int(2*s))

        self._adv_content = ttk.Frame(adv_frame)
        self._adv_content.pack(fill="x")
        row = 0

        ttk.Label(self._adv_content, text="Onset 灵敏度:").grid(row=row, column=0, sticky="w", pady=int(2*s))
        self.onset_sens_var = tk.DoubleVar(value=0.5)
        ttk.Scale(self._adv_content, from_=0.1, to=1.0, variable=self.onset_sens_var,
                  orient="horizontal", length=int(120*s)).grid(row=row, column=1, padx=int(5*s), pady=int(2*s))
        ttk.Label(self._adv_content, textvariable=self.onset_sens_var, width=4).grid(row=row, column=2)
        row += 1

        self.backtrack_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self._adv_content, text="Onset 回溯定位", variable=self.backtrack_var).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=int(2*s))
        row += 1

        self.multi_band_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self._adv_content, text="多频段 Onset 检测", variable=self.multi_band_var).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=int(2*s))
        row += 1

        self.multi_process_pitch_var = tk.BooleanVar(value=False)
        row += 1

        self.snap_to_beat_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self._adv_content, text="节拍吸附", variable=self.snap_to_beat_var).grid(
            row=row, column=0, sticky="w", pady=int(2*s))
        self.snap_res_var = tk.StringVar(value="8")
        snap_combo = ttk.Combobox(self._adv_content, textvariable=self.snap_res_var,
                                  values=["4", "8", "16", "32"], width=6, state="readonly")
        snap_combo.grid(row=row, column=1, sticky="w", pady=int(2*s))
        ttk.Label(self._adv_content, text="分音符").grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Label(self._adv_content, text="复杂度:").grid(row=row, column=0, sticky="w", pady=int(2*s))
        self.complexity_var = tk.DoubleVar(value=1.0)
        ttk.Scale(self._adv_content, from_=0.5, to=2.0, variable=self.complexity_var,
                  orient="horizontal", length=int(120*s)).grid(row=row, column=1, padx=int(5*s), pady=int(2*s))
        ttk.Label(self._adv_content, textvariable=self.complexity_var, width=4).grid(row=row, column=2)
        row += 1

        ttk.Label(self._adv_content, text="最小 BPM:").grid(row=row, column=0, sticky="w", pady=int(2*s))
        self.min_bpm_var = tk.DoubleVar(value=50.0)
        ttk.Spinbox(self._adv_content, from_=20, to=180, increment=5,
                    textvariable=self.min_bpm_var, width=8).grid(row=row, column=1, sticky="w", pady=int(2*s))
        row += 1

        ttk.Label(self._adv_content, text="最大 BPM:").grid(row=row, column=0, sticky="w", pady=int(2*s))
        self.max_bpm_var = tk.DoubleVar(value=200.0)
        ttk.Spinbox(self._adv_content, from_=100, to=300, increment=5,
                    textvariable=self.max_bpm_var, width=8).grid(row=row, column=1, sticky="w", pady=int(2*s))
        row += 1

        ttk.Label(self._adv_content, text="长音倾向:").grid(row=row, column=0, sticky="w", pady=int(2*s))
        self.ln_tendency_var = tk.DoubleVar(value=0.5)
        ttk.Scale(self._adv_content, from_=0.0, to=1.0, variable=self.ln_tendency_var,
                  orient="horizontal", length=int(120*s)).grid(row=row, column=1, padx=int(5*s), pady=int(2*s))
        ttk.Label(self._adv_content, textvariable=self.ln_tendency_var, width=4).grid(row=row, column=2)
        row += 1

        ttk.Label(self._adv_content, text="能量对比度:").grid(row=row, column=0, sticky="w", pady=int(2*s))
        self.contrast_var = tk.DoubleVar(value=1.0)
        ttk.Scale(self._adv_content, from_=0.1, to=3.5, variable=self.contrast_var,
                  orient="horizontal", length=int(120*s)).grid(row=row, column=1, padx=int(5*s), pady=int(2*s))
        ttk.Label(self._adv_content, textvariable=self.contrast_var, width=4).grid(row=row, column=2)
        row += 1

        out_frame = ttk.LabelFrame(inner, text="输出目录", padding=int(10*s))
        out_frame.pack(fill="x", padx=int(5*s), pady=int(5*s))
        out_frame.columnconfigure(0, weight=1)
        self.out_label = ttk.Label(out_frame, text=os.getcwd(), foreground="gray")
        self.out_label.grid(row=0, column=0, sticky="ew", padx=(0, int(5*s)))
        ttk.Button(out_frame, text="更改...", command=self._browse_out).grid(row=0, column=1)

        prog_frame = ttk.LabelFrame(inner, text="进度", padding=int(5*s))
        prog_frame.pack(fill="x", padx=int(5*s), pady=int(5*s))
        self._status_label = ttk.Label(prog_frame, text="就绪", foreground="gray")
        self._status_label.pack(anchor="w")
        self._progress_bar = ttk.Progressbar(prog_frame, mode="determinate", maximum=100)
        self._progress_bar.pack(fill="x", pady=int(2*s))

        btn_frame = ttk.Frame(inner)
        btn_frame.pack(fill="x", padx=int(5*s), pady=int(10*s))
        self.gen_btn = ttk.Button(btn_frame, text="生成谱面", command=self._generate)
        self.gen_btn.pack(fill="x", pady=int(2*s))
        self.preview_btn = ttk.Button(btn_frame, text="刷新预览", command=self._refresh_preview, state="disabled")
        self.preview_btn.pack(fill="x", pady=int(2*s))
        self.play_preview_btn = ttk.Button(btn_frame, text="▶ 播放预览", command=self._switch_to_preview, state="disabled")
        self.play_preview_btn.pack(fill="x", pady=int(2*s))
        self.back_btn = ttk.Button(btn_frame, text="← 返回静态预览", command=self._switch_to_static, state="disabled")
        self.back_btn.pack(fill="x", pady=int(2*s))

        export_row = ttk.Frame(btn_frame)
        export_row.pack(fill="x", pady=int(2*s))
        self.format_var = tk.StringVar(value="all")
        ttk.Combobox(export_row, textvariable=self.format_var, values=["osu", "json", "csv", "all"], width=6, state="readonly").pack(side="left")
        self.export_btn = ttk.Button(export_row, text="导出", command=self._export, state="disabled")
        self.export_btn.pack(side="right", fill="x", expand=True, padx=(int(4*s), 0))

        about_frame = ttk.Frame(inner)
        about_frame.pack(fill="x", padx=int(5*s), pady=int(5*s))
        ttk.Label(about_frame, text="Mustaff v0.5.0", foreground="gray",
                  font=("", 7)).pack(anchor="center")
        ttk.Label(about_frame, text="by ChidcGithub", foreground="gray",
                  font=("", 7)).pack(anchor="center")
        github_link = tk.Label(about_frame, text="GitHub",
                               fg="#0078d4", cursor="hand2",
                               font=("", 7, "underline"))
        github_link.pack(anchor="center")
        github_link.bind("<Button-1>", lambda e: self._open_url("https://github.com/ChidcGithub/Mustaff"))

        # ===== 右侧预览面板 =====
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        preview_label_frame = ttk.LabelFrame(right_frame, text="谱面预览", padding=int(5*s))
        preview_label_frame.pack(fill="both", expand=True, padx=int(5*s), pady=int(5*s))

        # 容器框架用于切换两种预览
        self._preview_container = ttk.Frame(preview_label_frame)
        self._preview_container.pack(fill="both", expand=True)

        # Matplotlib 静态预览
        self.fig = Figure(figsize=(8, 5), dpi=max(100, int(100 + 50*(s - 1))), facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("white")
        self.ax.text(
            0.5, 0.5, "点击「生成谱面」后预览将显示在这里",
            transform=self.ax.transAxes,
            ha="center", va="center",
            fontsize=14, color="gray",
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color("#cccccc")

        self.static_canvas = FigureCanvasTkAgg(self.fig, master=self._preview_container)
        self.static_widget = self.static_canvas.get_tk_widget()
        self.static_widget.pack(fill="both", expand=True)
        self.static_canvas.draw()

        # 交互式音游预览（初始隐藏）
        self.preview_canvas: Optional[PreviewCanvas] = None

    def _on_canvas_resize(self, event):
        self._left_canvas.itemconfig(self._canvas_window, width=event.width)

    def _bind_mousewheel(self):
        self._left_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._left_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self._left_canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self):
        self._left_canvas.unbind_all("<MouseWheel>")
        self._left_canvas.unbind_all("<Button-4>")
        self._left_canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if event.num == 4:
            self._left_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._left_canvas.yview_scroll(1, "units")
        else:
            self._left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _open_url(self, url: str):
        import webbrowser
        webbrowser.open(url)

    def _log(self, message: str):
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

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
            basename = os.path.basename(path)
            display = basename if len(basename) <= 40 else basename[:37] + "..."
            self.file_label.config(text=display, foreground="black")
            self._log(f"已选择: {path}")

    def _browse_out(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_dir = path
            self.out_label.config(text=path)

    def _ask_time_unit(self):
        """弹窗让用户选择 CSV 时间单位，返回 'seconds' 或 None（取消）"""
        result = {"value": None}
        dialog = tk.Toplevel(self.root)
        dialog.title("选择时间单位")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        s = self._scale
        ttk.Label(dialog, text="CSV 中时间的单位：").pack(padx=int(12*s), pady=int(8*s))

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(padx=int(12*s), pady=int(4*s))

        def choose(unit):
            result["value"] = unit
            dialog.destroy()

        ttk.Button(btn_frame, text="秒 (seconds)", command=lambda: choose("seconds")).pack(side="left", padx=int(4*s))
        ttk.Button(btn_frame, text="毫秒 (milliseconds)", command=lambda: choose("milliseconds")).pack(side="left", padx=int(4*s))

        self.root.wait_window(dialog)
        return result["value"]

    def _import_chart(self):
        path = filedialog.askopenfilename(
            title="导入谱面文件",
            filetypes=[
                ("谱面文件", "*.json *.osu *.csv"),
                ("JSON", "*.json"),
                ("OSU", "*.osu"),
                ("CSV", "*.csv"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".json":
                importer = JsonImporter(path)
            elif ext == ".osu":
                importer = OsuImporter(path)
            elif ext == ".csv":
                time_unit = self._ask_time_unit()
                if time_unit is None:
                    return
                importer = CsvImporter(path, time_unit=time_unit)
            else:
                messagebox.showerror("错误", f"不支持的文件格式: {ext}")
                return

            info = importer.get_info()
            if not info["notes"]:
                messagebox.showwarning("提示", "谱面中没有音符")
                return

            self.current_notes = info["notes"]
            self.current_keys = info["keys"]
            self.current_bpm = info["bpm"]
            self.current_title = info["title"]
            self.current_artist = info["artist"]
            self.current_duration_ms = max(n["time"] for n in info["notes"]) + 2000

            self.keys_var.set(info["keys"])
            self.diff_var.set(info["version"])
            basename = os.path.basename(path)
            self.import_label.config(
                text=(basename if len(basename) <= 40 else basename[:37] + "..."),
                foreground="black",
            )
            self._log(f"导入谱面: {path}")
            self._log(f"  标题: {info['title']} - {info['artist']}")
            self._log(f"  BPM: {info['bpm']}, Keys: {info['keys']}, 音符数: {len(info['notes'])}")

            # 选择配对音频文件（可选）
            audio_path = filedialog.askopenfilename(
                title="选择配对音频文件（可选，用于播放预览）",
                filetypes=[
                    ("音频文件", "*.mp3 *.wav *.flac *.ogg *.m4a"),
                    ("所有文件", "*.*"),
                ],
            )
            if audio_path:
                self.input_path = audio_path
                self.current_audio_path = audio_path
                self.file_label.config(text=os.path.basename(audio_path), foreground="black")
                self._log(f"配对音频: {audio_path}")

            self.preview_btn.config(state="normal")
            self.play_preview_btn.config(state="normal")
            self.export_btn.config(state="normal")
            self._draw_preview()
            self._log("[Done] 谱面导入完成！")

        except Exception as e:
            messagebox.showerror("导入失败", str(e))
            self._log(f"[Error] 导入失败: {e}")

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
        self.export_btn.config(state="disabled")
        self._log("=" * 40)
        self._log("开始分析...")
        self._set_taskbar_progress(0)

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
                self.diff_var.get(),
                self.onset_sens_var.get(),
                self.backtrack_var.get(),
                self.multi_band_var.get(),
                self.multi_process_pitch_var.get(),
                self.snap_to_beat_var.get(),
                int(self.snap_res_var.get()),
                self.min_bpm_var.get(),
                self.max_bpm_var.get(),
                self.complexity_var.get(),
                self.ln_tendency_var.get(),
                self.contrast_var.get(),
            ),
            daemon=True,
        )
        self._worker_thread.start()
        self._poll_progress()

    def _generate_worker(self, input_path: str, keys: int, density: float,
                         ln_threshold: float, difficulty: str,
                         onset_sensitivity: float = 0.5, backtrack: bool = False,
                         multi_band: bool = False, multi_process_pitch: bool = False,
                         snap_to_beat: bool = False,
                         snap_resolution: int = 8, min_bpm: float = 50.0,
                         max_bpm: float = 200.0, complexity: float = 1.0,
                         ln_tendency: float = 0.5, contrast: float = 1.0):
        """后台线程执行生成任务"""
        def report(step: int, total: int, msg: str):
            self._progress_queue.put({"type": "progress", "step": step, "total": total, "msg": msg})

        try:
            report(0, 100, "加载音频...")
            analyzer = AudioAnalyzer(
                onset_sensitivity=onset_sensitivity,
                backtrack=backtrack,
                multi_band=multi_band,
                min_bpm=min_bpm,
                max_bpm=max_bpm,
                multi_process_pitch=multi_process_pitch,
            )
            analyzer.load(input_path)

            def on_progress(pct: int, msg: str):
                report(pct, 100, msg)

            analyzer.analyze(progress_callback=on_progress)

            report(90, 100, "映射音符...")
            features = analyzer.get_note_features()

            beat_subdivisions = None
            if snap_to_beat:
                beat_subdivisions = analyzer.get_beat_subdivisions(resolution=snap_resolution)

            mapper = BeatMapper(
                keys=keys,
                density_filter_ms=density,
                ln_threshold_ratio=ln_threshold,
                snap_to_beat=snap_to_beat,
                snap_resolution=snap_resolution,
                complexity=complexity,
                ln_tendency=ln_tendency,
                contrast=contrast,
            )
            notes = mapper.map_notes(
                features, beat_subdivisions=beat_subdivisions,
                rms_full=analyzer.rms, pitches_full=analyzer.pitches,
            )

            report(100, 100, "生成完成")
            base_name = os.path.splitext(os.path.basename(input_path))[0]

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
            })

        except Exception as e:
            self._progress_queue.put({"type": "error", "msg": str(e)})

    def _poll_progress(self):
        """主线程轮询进度队列，更新 UI"""
        try:
            for _ in range(5):
                item = self._progress_queue.get_nowait()
                if item["type"] == "progress":
                    pct = int(item["step"] / item["total"] * 100)
                    self._progress_bar["value"] = pct
                    self._status_label.config(text=item["msg"], foreground="black")
                    self._log(f"[{pct}%] {item['msg']}")
                    self._set_taskbar_progress(pct)
                elif item["type"] == "done":
                    self._progress_bar["value"] = 100
                    self._status_label.config(text="生成完成！", foreground="green")

                    # 保存状态
                    self.current_notes = item["notes"]
                    self.current_keys = item["keys"]
                    self.current_duration_ms = item["duration_ms"]
                    self.current_title = item["title"]
                    self.current_audio_path = item["audio_path"]
                    self.current_bpm = item["bpm"]

                    # 日志
                    self._log(f"BPM: {item['bpm']:.1f}")
                    self._log(f"Onset 数: {item['onset_count']}")
                    self._log(f"时长: {item['duration_sec']:.2f}s")
                    self._log(f"生成音符数: {item['note_count']}")

                    # 显示静态预览
                    self._draw_preview()
                    self.gen_btn.config(state="normal")
                    self.preview_btn.config(state="normal")
                    self.play_preview_btn.config(state="normal")
                    self.export_btn.config(state="normal")
                    self._clear_taskbar()
                    self._show_toast("Mustaff", f"生成完成！{item['note_count']} 个音符")
                    self._log("[Done] 生成完成！")
                    return  # 停止轮询
                elif item["type"] == "error":
                    self._progress_bar["value"] = 0
                    self._status_label.config(text="生成失败", foreground="red")
                    self._log(f"[Error] {item['msg']}")
                    self._clear_taskbar()
                    self._show_toast("Mustaff", f"生成失败：{item['msg']}")
                    messagebox.showerror("错误", item["msg"])
                    self.gen_btn.config(state="normal")
                    return
        except queue.Empty:
            pass

        self.root.update_idletasks()

        # 继续轮询
        self.root.after(50, self._poll_progress)

    def _refresh_preview(self):
        if not self.current_notes:
            messagebox.showinfo("提示", "请先生成谱面")
            return
        self._draw_preview()

    def _export(self):
        if not self.current_notes:
            messagebox.showinfo("提示", "请先生成谱面")
            return

        fmt = self.format_var.get()

        # CSV 导出需要选择时间单位
        csv_time_unit = "seconds"
        if fmt in ("csv", "all"):
            chosen = self._ask_time_unit()
            if chosen is None:
                return
            csv_time_unit = chosen

        base_name = self.current_title
        difficulty = self.diff_var.get()
        audio_basename = os.path.basename(self.current_audio_path) if self.current_audio_path else ""
        exported = []

        self.export_btn.config(state="disabled")
        self._log("导出中...")

        def do_export():
            nonlocal exported
            from concurrent.futures import ThreadPoolExecutor
            export_futs = []

            with ThreadPoolExecutor(max_workers=3) as ex:
                if fmt in ("osu", "all"):
                    _osu_exporter = OsuManiaExporter(
                        notes=self.current_notes, bpm=self.current_bpm, keys=self.current_keys,
                        title=base_name, artist=self.current_artist, version=difficulty,
                        audio_filename=audio_basename,
                    )
                    _osu_path = os.path.join(self.output_dir, f"{base_name}.osu")
                    export_futs.append(("osu", _osu_path, ex.submit(_osu_exporter.export, _osu_path)))

                if fmt in ("json", "all"):
                    _json_exporter = JsonExporter(
                        notes=self.current_notes, bpm=self.current_bpm, keys=self.current_keys,
                        title=base_name, artist=self.current_artist, version=difficulty,
                    )
                    _json_path = os.path.join(self.output_dir, f"{base_name}.json")
                    export_futs.append(("json", _json_path, ex.submit(_json_exporter.export, _json_path)))

                if fmt in ("csv", "all"):
                    _csv_exporter = CsvExporter(
                        notes=self.current_notes, bpm=self.current_bpm, keys=self.current_keys,
                        title=base_name, artist=self.current_artist, version=difficulty,
                        time_unit=csv_time_unit,
                    )
                    _csv_path = os.path.join(self.output_dir, f"{base_name}.csv")
                    export_futs.append(("csv", _csv_path, ex.submit(_csv_exporter.export, _csv_path)))

                for name, path, fut in export_futs:
                    fut.result()
                    exported.append(f"{name}: {path}")

            self.root.after(0, _on_done)

        def _on_done():
            for exp in exported:
                self._log(f"导出 {exp}")
            self._log("[Done] 导出完成！")
            self.export_btn.config(state="normal")

        def _on_error(e):
            self._log(f"[Error] 导出失败: {e}")
            self.export_btn.config(state="normal")

        def run_export():
            try:
                do_export()
            except Exception as e:
                self.root.after(0, lambda: _on_error(e))

        threading.Thread(target=run_export, daemon=True).start()

    def _draw_preview(self):
        self.ax.clear()

        notes = self.current_notes
        keys = self.current_keys
        duration_ms = self.current_duration_ms
        title = self.current_title
        s = self._scale

        import numpy as np
        from matplotlib.patches import Patch

        lane_colors_bg = ["#f0f0f0", "#e8e8e8"]
        lane_colors_notes = lane_colors(keys)

        for i in range(keys):
            self.ax.axhspan(i - 0.5, i + 0.5, color=lane_colors_bg[i % 2], alpha=0.3, zorder=0)
        for i in range(keys + 1):
            self.ax.axhline(i - 0.5, color="#cccccc", linewidth=0.5, zorder=1)

        max_time_s = duration_ms / 1000.0 if duration_ms else 0.0
        if max_time_s <= 0 and notes:
            max_time_s = max(n["time"] for n in notes) / 1000.0 * 1.05

        if max_time_s > 0:
            grid_spacing = 1 if max_time_s <= 30 else (5 if max_time_s <= 120 else 10)
            for t in np.arange(0, max_time_s + grid_spacing, grid_spacing):
                self.ax.axvline(t, color="#cccccc", linewidth=0.3, linestyle="--", alpha=0.4, zorder=0)

        hit_by_col: dict = {}
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
            self.ax.scatter(
                times, cols,
                c=color, s=max(20, int(35*s)), alpha=0.9, zorder=3,
                edgecolors="white", linewidths=max(0.3, 0.5*s),
            )

        for start_t, end_t, col in hold_segments:
            color = lane_colors_notes[col % len(lane_colors_notes)]
            self.ax.barh(
                col, width=end_t - start_t, left=start_t,
                height=0.65, color=color, alpha=0.7, zorder=2,
                edgecolor=color, linewidth=0.5,
            )

        self.ax.set_yticks(range(keys))
        self.ax.set_yticklabels([f"Key {i+1}" for i in range(keys)], color="black")
        self.ax.set_ylim(-0.6, keys - 0.4)
        self.ax.invert_yaxis()

        if max_time_s > 0:
            self.ax.set_xlim(0, max_time_s)

        self.ax.set_xlabel("Time (s)", fontsize=10, color="black")
        self.ax.set_facecolor("white")
        self.ax.tick_params(colors="black", labelsize=8)
        self.ax.xaxis.label.set_color("black")
        self.ax.yaxis.label.set_color("black")
        self.ax.spines["bottom"].set_color("#cccccc")
        self.ax.spines["left"].set_color("#cccccc")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

        hit_count = sum(len(t) for t, _ in hit_by_col.values())
        hold_count = len(hold_segments)
        info = f"Keys: {keys}  |  Notes: {hit_count} hits + {hold_count} holds = {len(notes)} total"
        self.ax.set_title(f"{title} [{keys}K]\n{info}", fontsize=11, color="black", pad=int(8*s))

        legend_elements = []
        for i in range(min(keys, len(lane_colors_notes))):
            legend_elements.append(
                Patch(facecolor=lane_colors_notes[i], edgecolor="white", label=f"Lane {i+1}")
            )
        legend_elements.append(
            Patch(facecolor=lane_colors_notes[0], edgecolor="white", alpha=0.5, label="Hold (LN)")
        )
        legend = self.ax.legend(
            handles=legend_elements, loc="upper right",
            facecolor="white", edgecolor="#cccccc",
            labelcolor="black", fontsize=7, ncol=2,
        )
        try:
            legend.legend_handles[-1].set_alpha(0.5)
        except (AttributeError, IndexError):
            try:
                legend.legendHandles[-1].set_alpha(0.5)
            except (AttributeError, IndexError):
                pass

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
            scale=self._scale,
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
