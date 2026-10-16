"""
数据集构建（mert v5）：.osz → MERT 预训练表征 → 每拍嵌入矩阵

每个谱面集产出一个 mert_beats/<set_id>.npy：
  shape [n_units, 8, 768] float16，n_units = len(beat_times) - 1
  行 b = 拍 b→b+1：音频按 24kHz 送入冻结的 MERT-v1-95M（30s 分块推理），
  13 层 hidden states 均值池化 → 75fps 帧序列 → 按拍网格重采样 8 步。

MERT 输出代表「预训练编码器学到的音乐表征」——placement/selection 的
Transformer/LSTM 结构与 v3 完全一致，仅输入换源（n_feats=768）。
与 v3 的 82 维手工特征、v4-raw 的从零卷积前端构成三方 A/B。

用法：
  python -m training.build_dataset_mert             # 全部缺失
  python -m training.build_dataset_mert --limit 3   # 试跑测速
  python -m training.build_dataset_mert --no-hf-mirror   # 直连 huggingface.co
"""

import argparse
import json
import os
import re
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from .dataset import DATA_DIR, NPZ_DIR

ROOT = os.path.dirname(os.path.abspath(__file__))
OSZ_DIR = os.path.join(DATA_DIR, "osz")
OUT_DIR = os.path.join(DATA_DIR, "mert_beats")

AUDIO_EXTS = (".mp3", ".ogg", ".wav", ".flac", ".m4a")
SR_MERT = 24000
FRAME_MS_MERT = 1000.0 / 75.0   # MERT 卷积总步长 320 @24kHz → 75 fps
CHUNK_S = 30                    # 分块推理长度（避免 21k 帧 attention 爆显存）
D_MERT = 768
N_LAYERS = 13

_mert_cache: dict = {}


def _get_mert(device: str):
    """加载冻结 MERT（进程内缓存；多进程时每 worker 各自加载一次）

    权重默认走本地缓存（HF_HUB_OFFLINE=1，避免长跑中被远端限流打断）；
    首次下载由 snapshot_download 完成。
    """
    if "model" in _mert_cache:
        return _mert_cache["model"], _mert_cache["fe"]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    name = "m-a-p/MERT-v1-95M"
    model = Wav2Vec2Model.from_pretrained(name, trust_remote_code=True)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
    model.eval().to(device)
    if device.startswith("cuda"):
        model = model.half()
    _mert_cache["model"] = model
    _mert_cache["fe"] = fe
    return model, fe


def _find_audio_name(zf: zipfile.ZipFile) -> str:
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


@torch.no_grad()
def _mert_frames(y24: np.ndarray, device: str) -> np.ndarray:
    """24kHz 波形 → [T_frames, 768] float32（13 层均值池化，30s 分块）"""
    model, fe = _get_mert(device)
    chunk = CHUNK_S * SR_MERT
    outs = []
    for s in range(0, len(y24), chunk):
        seg = y24[s: s + chunk].astype(np.float32)
        if len(seg) < SR_MERT // 2:
            seg = np.pad(seg, (0, SR_MERT // 2 - len(seg)))
        inputs = fe(seg, sampling_rate=SR_MERT, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.float16):
                hs = model(**inputs, output_hidden_states=True).hidden_states
        else:
            hs = model(**inputs, output_hidden_states=True).hidden_states
        # hs: tuple of 13 × [1, T, 768]（含输入投影层）→ 均值池化
        stacked = torch.stack([h[0] for h in hs], dim=0).float()  # [13, T, 768]
        outs.append(stacked.mean(dim=0).cpu().numpy())
    return np.concatenate(outs, axis=0)  # [T, 768]


def _frames_to_beats(frames: np.ndarray, beat_times_ms: np.ndarray,
                     steps: int = 8) -> np.ndarray:
    """帧序列 [T, D]（75fps）→ 按拍重采样 [n_units, steps, D]（与 beats_to_units 同逻辑）"""
    n_units = len(beat_times_ms) - 1
    times = np.arange(frames.shape[0]) * FRAME_MS_MERT
    out = np.zeros((n_units, steps, frames.shape[1]), dtype=np.float16)
    for b in range(n_units):
        t0, t1 = beat_times_ms[b], beat_times_ms[b + 1]
        ts = t0 + (t1 - t0) * (np.arange(steps) + 0.5) / steps
        idx = np.clip(np.searchsorted(times, ts), 0, frames.shape[0] - 1)
        out[b] = frames[idx].astype(np.float16)
    return out


def build_one_mert(set_id: str, device: str = "cuda") -> str:
    npz_path = os.path.join(NPZ_DIR, f"{set_id}.npz")
    out_path = os.path.join(OUT_DIR, f"{set_id}.npy")
    osz_path = os.path.join(OSZ_DIR, f"{set_id}.osz")
    if not os.path.exists(npz_path) or not os.path.exists(osz_path):
        return ""

    with np.load(npz_path) as z:
        if "beat_times" not in z.files or len(z["beat_times"]) < 4:
            return ""
        beat_times = z["beat_times"].astype(np.float64)
        duration = float(json.loads(bytes(z["meta"]).decode("utf-8")).get("duration", 0.0))
    n_units = len(beat_times) - 1
    if n_units < 4 or duration <= 0:
        return ""

    if os.path.exists(out_path):
        try:
            if np.load(out_path, mmap_mode="r").shape == (n_units, 8, D_MERT):
                return out_path
        except Exception:
            pass
        os.remove(out_path)

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
        from mustaff.analyzer import AudioAnalyzer
        # 与推理同口径：先解码到 22050（推理侧 AudioAnalyzer 的固定采样率），
        # 再 librosa.resample 到 24kHz——避免单次 24k 解码与推理二次重采样的分布偏移
        y22 = AudioAnalyzer(sr=22050).load(tmp_path).y
    except Exception as e:
        print(f"  [Warn] 音频解码失败 {set_id}: {e}", flush=True)
        return ""
    finally:
        os.remove(tmp_path)

    import librosa
    y24 = librosa.resample(y22.astype(np.float32), orig_sr=22050, target_sr=SR_MERT)
    y24 = y24[: int(duration * SR_MERT)].astype(np.float32)
    if len(y24) < SR_MERT:
        print(f"  [Skip] {set_id}: 音频过短", flush=True)
        return ""

    frames = _mert_frames(y24, device)
    rows = _frames_to_beats(frames, beat_times)
    np.save(out_path, rows)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="MERT 推理吃 GPU，多进程反而争抢，默认 1")
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--online", action="store_true",
                    help="允许 HF 在线校验（默认离线，用本地缓存）")
    ap.add_argument("--device", default="cuda" if _cuda_ok() else "cpu")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.online:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
    if args.sets:
        set_ids = args.sets
    else:
        set_ids = [os.path.splitext(f)[0] for f in sorted(os.listdir(NPZ_DIR))
                   if f.endswith(".npz")]
        if args.limit > 0:
            set_ids = set_ids[: args.limit]

    print(f"待构建 {len(set_ids)} 个（{args.workers} 进程，device={args.device}）→ {OUT_DIR}",
          flush=True)
    t0 = __import__("time").time()
    n_ok, n_done = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(build_one_mert, sid, args.device): sid for sid in set_ids}
        for fut in as_completed(futs):
            n_done += 1
            try:
                if fut.result():
                    n_ok += 1
            except Exception as e:
                print(f"  [Error] {futs[fut]}: {e}", flush=True)
            if n_done % 20 == 0 or n_done == len(set_ids):
                dt = __import__("time").time() - t0
                eta = dt / n_done * (len(set_ids) - n_done)
                print(f"[{n_done}/{len(set_ids)}] 成功 {n_ok}  "
                      f"均速 {dt/n_done:.1f}s/个  ETA {eta/60:.0f}min", flush=True)

    total_gb = sum(os.path.getsize(os.path.join(OUT_DIR, f))
                   for f in os.listdir(OUT_DIR)) / 1e9
    print(f"[Done] MERT 按拍嵌入 {n_ok}/{len(set_ids)} 个，目录总大小 {total_gb:.1f} GB",
          flush=True)


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
