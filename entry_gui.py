"""
Mustaff GUI 入口脚本（用于 PyInstaller 打包）
在 tkinter 创建前设置 DPI Awareness，确保高分辨率屏幕渲染清晰。
"""
# PyInstaller 打包后，multiprocessing 子进程（如"多进程音高检测"的进程池）
# 会重新执行本脚本。freeze_support() 必须最先调用：
# 子进程在此进入 multiprocessing 工作循环后直接退出，
# 否则会继续向下执行 run_gui()，表现为分析时弹出多个 GUI 窗口。
# 该调用在开发环境（未打包）下是空操作，无副作用。
import multiprocessing

multiprocessing.freeze_support()

import ctypes
import sys

if sys.platform == "win32":
    # PerMonitorV2 = per-monitor DPI with automatic scaling
    # Windows 10 1607+ (Anniversary Update)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

from mustaff.gui.app import run_gui

if __name__ == "__main__":
    run_gui()
