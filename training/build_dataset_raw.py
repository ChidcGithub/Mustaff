"""
数据集构建（raw v4）：.osz → 每拍原始波形矩阵

每个谱面集产出一个 raw_beats/<set_id>.npy：
  shape [n_units, RAW_LEN] int16，n_units = len(beat_times) - 1
  行 b = 拍 b→b+1 的音频段，先按曲集做抗混叠低通，再时间重采样到 RAW_LEN 点
  振幅按曲集 p95(|y|) 归一到 RAW_NORM_SCALE（dataset 侧除回 RAW_NORM_SCALE）

与 features npz 的分工：beat_times / fd_idx / labels / cols / holds / stars 仍读
原 npz（dataset_v3 的标签管线原样复用），本文件只补音频原始波形——
存成未压缩 .npy 以便训练时 np.load(mmap_mode='r') 零拷贝随机窗口读取。

波形处理唯一实现在 raw_audio.warp_beats_raw（与推理共用，保证同口径）。

用法：
  python -m training.build_dataset_raw             # 构建全部缺失（已存在且形状对的跳过）
  python -m training.build_dataset_raw --limit 5   # 试跑前 5 个
  python -m training.build_dataset_raw --sets 1000065 1000359
"""

import argparse
import json
import os
import re
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from mustaff.analyzer import AudioAnalyzer
from .dataset import DATA_DIR, NPZ_DIR
from .raw_audio import RAW_LEN, warp_beats_raw

ROOT = os.path.dirname(os.path.abspath(__file__))
OSZ_DIR = os.path.join(DATA_DIR, "osz")
RAW_DIR = os.path.join(DATA_DIR, "raw_beats")

AUDIO_EXTS = (".mp3", ".ogg", ".wav", ".flac", ".m4a")


def _find_audio_name(zf: zipfile.ZipFile) -> str:
    """与 build_dataset.build_one 同口径找主音频：先看 .osu 的 AudioFilename，退最大音频文件"""
    names = zf.namelist()
    declared = []
    for n in names:
        if n.lower().endswith(".osu"):
            try:
                m = re.search(r"^AudioFilename:\s*(.+?)\s*$",
                              zf.read(n).decode("utf-8-sig", errors="replace"), re.M)
                if m:
                    declared.append(m.group(1).strip())
            except Exception:
                continue
    name_lower = {n.lower(): n for n in names}
    for d in declared:
        if d.lower() in name_lower:
            return name_lower[d.lower()]
    candidates = [n for n in names if os.path.splitext(n)[1].lower() in AUDIO_EXTS]
    if not candidates:
        return ""
    return max(candidates, key=lambda n: zf.getinfo(n).file_size)


def build_one_raw(set_id: str) -> str:
    """一个谱面集 → raw_beats/<id>.npy；返回输出路径或空串（跳过原因见日志）"""
    npz_path = os.path.join(NPZ_DIR, f"{set_id}.npz")
    out_path = os.path.join(RAW_DIR, f"{set_id}.npy")
    osz_path = os.path.join(OSZ_DIR, f"{set_id}.osz")
    if not os.path.exists(npz_path):
        return ""
    if not os.path.exists(osz_path):
        print(f"  [Skip] {set_id}: 缺 .osz", flush=True)
        return ""

    with np.load(npz_path) as z:
        if "beat_times" not in z.files or len(z["beat_times"]) < 4:
            return ""
        beat_times = z["beat_times"].astype(np.float64)
        duration = float(json.loads(bytes(z["meta"]).decode("utf-8")).get("duration", 0.0))
    n_units = len(beat_times) - 1
    if n_units < 4 or duration <= 0:
        return ""

    if os.path.exists(out_path):  # 已存在则校验形状
        try:
            shape = np.load(out_path, mmap_mode="r").shape
            if shape == (n_units, RAW_LEN):
                return out_path
        except Exception:
            pass
        os.remove(out_path)  # 形状不符（如 RAW_LEN 变更）→ 重建

    with zipfile.ZipFile(osz_path) as zf:
        audio_name = _find_audio_name(zf)
        if not audio_name:
            print(f"  [Skip] {set_id}: zip 内无音频", flush=True)
            return ""
        audio_bytes = zf.read(audio_name)

    ext = os.path.splitext(audio_name)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y = AudioAnalyzer(sr=22050).load(tmp_path).y
    except Exception as e:
        print(f"  [Warn] 音频解码失败 {set_id}: {e}", flush=True)
        return ""
    finally:
        os.remove(tmp_path)

    # 与 build_one 完全一致的对齐口径：截断时长用 npz meta.duration（特征构建时的值）
    y = y[: int(duration * 22050)].astype(np.float32)
    if len(y) < 22050:
        print(f"  [Skip] {set_id}: 音频过短", flush=True)
        return ""

    rows = warp_beats_raw(y, beat_times)
    if rows.max() == 0:
        print(f"  [Skip] {set_id}: 静音或波形为空", flush=True)
        return ""

    np.save(out_path, rows)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--sets", nargs="*", default=None, help="只构建指定 set id")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    if args.sets:
        set_ids = args.sets
    else:
        set_ids = [os.path.splitext(f)[0] for f in sorted(os.listdir(NPZ_DIR))
                   if f.endswith(".npz")]
        if args.limit > 0:
            set_ids = set_ids[: args.limit]

    print(f"待构建 {len(set_ids)} 个（{args.workers} 进程）→ {RAW_DIR}", flush=True)
    n_ok, n_done = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(build_one_raw, sid): sid for sid in set_ids}
        for fut in as_completed(futs):
            n_done += 1
            try:
                if fut.result():
                    n_ok += 1
            except Exception as e:
                print(f"  [Error] {futs[fut]}: {e}", flush=True)
            if n_done % 50 == 0 or n_done == len(set_ids):
                print(f"[{n_done}/{len(set_ids)}] 成功 {n_ok}", flush=True)

    total_gb = sum(os.path.getsize(os.path.join(RAW_DIR, f)) for f in os.listdir(RAW_DIR)) / 1e9
    print(f"[Done] raw 波形 {n_ok}/{len(set_ids)} 个，目录总大小 {total_gb:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
