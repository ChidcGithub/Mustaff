"""
Mustaff GUI 入口脚本（用于 PyInstaller 打包）
在 tkinter 创建前设置 DPI Awareness，确保高分辨率屏幕渲染清晰。
"""
import ctypes
import sys

if sys.platform == "win32":
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
