"""
数据集构建：.osz → 音频特征 + 谱面标签

每个谱面集（set）产出一个 .npz：
  features:      [T, 65] float16  log-mel(64) + onset_strength(1)，hop=512@22050（≈23.2ms/帧）
  chart_<bid>:   dict 数组前缀，每个 4K 难度一组：
    labels_<bid> [T]   uint8      该帧是否有音符头（±0 帧对齐）
    cols_<bid>   [N]   uint8      每个音符的列（0-3）
    frames_<bid> [N]   int32      每个音符的帧号（与 cols 对齐）
    holds_<bid>  [N]   int32      hold 结束帧，hit 为 -1
    stars_<bid>  []    float32    星级
  meta:          json 字符串（set_id/title/artist/bpm/offset/duration）

帧长 ≈23ms，23ms 内多个同列音符只记第一个（4K 谱面极少出现）。
同一帧多列音符在 labels 上都是 1，列信息保留在 cols/frames 中。

用法：
  python -m training.build_dataset          # 构建全部
  python -m training.build_dataset --limit 10
"""

import argparse
import json
import os
import zipfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import librosa

from mustaff.analyzer import AudioAnalyzer

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OSZ_DIR = os.path.join(DATA_DIR, "osz")
OUT_DIR = os.path.join(DATA_DIR, "npz")
INDEX_PATH = os.path.join(DATA_DIR, "sets_index.json")
SPLIT_PATH = os.path.join(DATA_DIR, "split.json")

SR = 22050
HOP = 512          # ≈23.2ms @22050
N_MELS = 64
MAX_DURATION_S = 360.0  # 超过 6 分钟的歌截断
MIN_NOTES = 50


def extract_features(y: np.ndarray) -> np.ndarray:
    """log-mel + onset 强度 → [T, 65] float16"""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    mel = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max).T  # [T, 64]
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)  # [T]
    onset_env = onset_env / (onset_env.max() + 1e-8)

    T = min(log_mel.shape[0], onset_env.shape[0])
    feats = np.concatenate(
        [log_mel[:T], onset_env[:T, None]], axis=1
    )
    return feats.astype(np.float16)


def parse_osu_text(text: str) -> dict:
    """轻量解析 .osu：模式、音频文件名、CircleSize、HitObjects"""
    info = {"mode": 0, "audio": "", "cs": 4.0, "notes": []}
    section = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section == "General" and line.startswith("Mode:"):
            info["mode"] = int(line.split(":", 1)[1].strip())
        elif section == "General" and line.startswith("AudioFilename:"):
            info["audio"] = line.split(":", 1)[1].strip()
        elif section == "Difficulty" and line.startswith("CircleSize:"):
            info["cs"] = float(line.split(":", 1)[1].strip())
        elif section == "HitObjects":
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                x, t, ntype = int(parts[0]), int(parts[2]), int(parts[3])
            except ValueError:
                continue
            col = min(3, int(x * 4 / 512))  # 4K: x → 列
            end_t = t
            if ntype & 128:  # hold
                try:
                    end_t = int(parts[5].split(":")[0])
                except (ValueError, IndexError):
                    end_t = t + 100
            info["notes"].append((t, col, end_t if end_t > t else -1))
    return info


def build_one(osz_path: str, set_id: str, index_entry: dict) -> str:
    """处理一个 .osz，返回输出路径或空字符串"""
    with zipfile.ZipFile(osz_path) as zf:
        names = zf.namelist()
        name_lower = {n.lower(): n for n in names}

        # 收集可用的 4K mania 谱面
        charts = {}  # 文件名 -> parsed
        for n in names:
            if not n.lower().endswith(".osu"):
                continue
            try:
                text = zf.read(n).decode("utf-8-sig", errors="replace")
            except Exception:
                continue
            parsed = parse_osu_text(text)
            if parsed["mode"] != 3 or int(round(parsed["cs"])) != 4:
                continue
            if len(parsed["notes"]) < MIN_NOTES:
                continue
            charts[n] = parsed

        if not charts:
            return ""

        # 找音频
        audio_name = next(iter(charts.values()))["audio"]
        audio_path_in_zip = name_lower.get(audio_name.lower())
        if not audio_path_in_zip:
            # 模糊匹配：找最大的音频文件
            audio_candidates = [n for n in names if os.path.splitext(n)[1].lower() in (".mp3", ".ogg", ".wav")]
            if not audio_candidates:
                return ""
            audio_path_in_zip = max(audio_candidates, key=lambda n: zf.getinfo(n).file_size)

        audio_bytes = zf.read(audio_path_in_zip)

    # 解码音频（写到临时文件给 librosa）
    ext = os.path.splitext(audio_path_in_zip)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y = AudioAnalyzer(sr=SR).load(tmp_path).y  # 内含 miniaudio 兜底解码
    except Exception as e:
        print(f"  [Warn] 音频解码失败 {set_id}: {e}")
        return ""
    finally:
        os.remove(tmp_path)

    duration = librosa.get_duration(y=y, sr=SR)
    if duration > MAX_DURATION_S:
        y = y[: int(MAX_DURATION_S * SR)]
        duration = MAX_DURATION_S

    feats = extract_features(y)
    T = feats.shape[0]
    frame_dur = HOP / SR

    out = {"features": feats}
    n_charts = 0
    for bid, parsed in charts.items():
        notes = sorted(parsed["notes"], key=lambda x: x[0])
        frames, cols, holds = [], [], []
        labels = np.zeros(T, dtype=np.uint8)
        for t_ms, col, end_ms in notes:
            f = int(t_ms / 1000.0 / frame_dur)
            if f >= T:
                continue
            frames.append(f)
            cols.append(col)
            if end_ms > 0:
                holds.append(min(T - 1, int(end_ms / 1000.0 / frame_dur)))
            else:
                holds.append(-1)
            labels[f] = 1
        if len(frames) < MIN_NOTES:
            continue
        # 难度代理：NPS（每秒音符数）× 系数压到近似星级范围，摆脱对官方星级的依赖
        nps = len(frames) / max(duration, 1.0)
        stars = float(np.clip(nps * 1.5, 0.5, 12.0))
        out[f"labels_{bid}"] = labels
        out[f"frames_{bid}"] = np.array(frames, dtype=np.int32)
        out[f"cols_{bid}"] = np.array(cols, dtype=np.uint8)
        out[f"holds_{bid}"] = np.array(holds, dtype=np.int32)
        out[f"stars_{bid}"] = np.array(stars, dtype=np.float32)
        n_charts += 1

    if n_charts == 0:
        return ""

    out["meta"] = np.frombuffer(json.dumps({
        "set_id": set_id,
        "title": index_entry.get("title", ""),
        "artist": index_entry.get("artist", ""),
        "bpm": index_entry.get("bpm", 0),
        "charts": n_charts,
        "duration": duration,
        "frames": T,
    }).encode("utf-8"), dtype=np.uint8)

    out_path = os.path.join(OUT_DIR, f"{set_id}.npz")
    np.savez_compressed(out_path, **out)
    return out_path


def make_split():
    """按 set 划分 train/val/test = 80/10/10"""
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".npz"))
    rng = np.random.RandomState(42)
    files_arr = np.array(files)
    rng.shuffle(files_arr)
    n = len(files_arr)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    split = {
        "train": files_arr[:n_train].tolist(),
        "val": files_arr[n_train:n_train + n_val].tolist(),
        "test": files_arr[n_train + n_val:].tolist(),
    }
    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)
    print(f"Split: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = {str(k): v for k, v in json.load(f).items()}

    files = sorted(f for f in os.listdir(OSZ_DIR) if f.endswith(".osz"))
    if args.limit > 0:
        files = files[: args.limit]

    tasks = []
    for fn in files:
        set_id = os.path.splitext(fn)[0]
        if set_id not in index:
            continue
        if os.path.exists(os.path.join(OUT_DIR, f"{set_id}.npz")):
            continue
        tasks.append((os.path.join(OSZ_DIR, fn), set_id, index[set_id]))

    print(f"待构建 {len(tasks)} 个（{args.workers} 进程）", flush=True)
    n_ok = 0
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(build_one, p, sid, entry): sid for p, sid, entry in tasks}
        for fut in as_completed(futs):
            n_done += 1
            try:
                if fut.result():
                    n_ok += 1
            except Exception as e:
                print(f"  [Error] {futs[fut]}: {e}", flush=True)
            if n_done % 10 == 0:
                print(f"[{n_done}/{len(tasks)}] 成功 {n_ok}", flush=True)

    print(f"[Done] 成功构建 {n_ok}/{len(tasks)} 个谱面集", flush=True)
    if n_ok >= 5:
        make_split()


if __name__ == "__main__":
    main()
