"""
Mustaff CLI 入口脚本（用于 PyInstaller 打包）
"""
# PyInstaller 打包后，multiprocessing 子进程（--multi-process-pitch 的进程池）
# 会重新执行本脚本。freeze_support() 必须最先调用，
# 否则每个子进程都会重新跑一遍完整 CLI 命令并再次创建进程池，
# 导致重复分析、重复输出甚至进程级联。开发环境下为空操作。
import multiprocessing

multiprocessing.freeze_support()

from mustaff.cli import main

if __name__ == "__main__":
    main()
