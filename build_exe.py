"""
Mustaff EXE 打包脚本

打包两个可执行文件：
- mustaff-cli.exe: 命令行工具
- mustaff-gui.exe: 图形界面

使用 onedir 模式（目录形式），启动更快、体积更合理。
"""

import os
import sys
import shutil
import subprocess

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(ROOT, "dist")
BUILD_DIR = os.path.join(ROOT, "build")

# 清理旧构建
for d in [DIST_DIR, BUILD_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)

# 通用 PyInstaller 参数（onedir 模式）
COMMON_ARGS = [
    "--noconfirm",
    "--clean",
]

# 需要排除的大型无用模块（减少体积和打包时间）
EXCLUDES = [
    "torch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "tensorboard",
    "tensorboardX",
    "jax",
    "jaxlib",
    "flax",
    "optuna",
    "wandb",
    "mlflow",
    "pytest",
    "mypy",
    "pylint",
    "flake8",
    "black",
    "isort",
    "jupyter",
    "IPython",
    "notebook",
    "sphinx",
    "tkinter.test",
    "test",
    "tests",
    "matplotlib.tests",
    "numpy.tests",
    "scipy.tests",
    "sklearn.tests",
]

def build_cli():
    """打包 CLI"""
    print("\n" + "=" * 50)
    print("Building mustaff-cli.exe...")
    print("=" * 50)

    cmd = ["pyinstaller"] + COMMON_ARGS
    cmd.append("--console")
    cmd.append("--name=mustaff-cli")

    for ex in EXCLUDES:
        cmd.append(f"--exclude-module={ex}")

    # 收集数据文件
    cmd.append("--collect-data=librosa")
    cmd.append("--collect-data=sklearn")
    cmd.append("--collect-data=matplotlib")
    cmd.append("--collect-data=soundfile")
    cmd.append("--collect-data=pygame")

    cmd.append(os.path.join(ROOT, "entry_cli.py"))

    subprocess.run(cmd, check=True)
    print("mustaff-cli.exe built successfully!")


def build_gui():
    """打包 GUI"""
    print("\n" + "=" * 50)
    print("Building mustaff-gui.exe...")
    print("=" * 50)

    cmd = ["pyinstaller"] + COMMON_ARGS
    cmd.append("--windowed")
    cmd.append("--name=mustaff-gui")

    for ex in EXCLUDES:
        cmd.append(f"--exclude-module={ex}")

    cmd.append("--collect-data=librosa")
    cmd.append("--collect-data=sklearn")
    cmd.append("--collect-data=matplotlib")
    cmd.append("--collect-data=soundfile")
    cmd.append("--collect-data=pygame")

    cmd.append(os.path.join(ROOT, "entry_gui.py"))

    subprocess.run(cmd, check=True)
    print("mustaff-gui.exe built successfully!")


def copy_dist():
    """整理输出目录"""
    print("\n" + "=" * 50)
    print("Organizing output...")
    print("=" * 50)

    final_dir = os.path.join(ROOT, "Mustaff-Windows")
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    os.makedirs(final_dir, exist_ok=True)

    # 复制 CLI 目录内容
    cli_src = os.path.join(DIST_DIR, "mustaff-cli")
    gui_src = os.path.join(DIST_DIR, "mustaff-gui")

    # 把 exe 和依赖复制到同一个目录下
    for src_dir, exe_name in [(cli_src, "mustaff-cli.exe"), (gui_src, "mustaff-gui.exe")]:
        if os.path.exists(src_dir):
            for item in os.listdir(src_dir):
                src = os.path.join(src_dir, item)
                dst = os.path.join(final_dir, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

    # 复制 README
    readme_src = os.path.join(ROOT, "README.md")
    if os.path.exists(readme_src):
        shutil.copy2(readme_src, os.path.join(final_dir, "README.md"))

    print(f"Output directory: {final_dir}")
    print("Top-level files:")
    for f in sorted(os.listdir(final_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    build_cli()
    build_gui()
    copy_dist()
    print("\n" + "=" * 50)
    print("All builds completed!")
    print("=" * 50)
