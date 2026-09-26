"""
收集 osu!mania 4K 谱面集（主源：sayobot.cn 腾讯云 CDN，国内高速）

流程：
1. 通过 sayobot beatmaplist API 分页收集 ranked/approved/qualified/loved 的
   mania 谱面集（sid 列表 + 元数据）
2. 从 txy1.sayobot.cn 并行下载 .osz（novideo 版本，含音频与谱面）
3. catboy.best 作为备用源（国外，较慢）

难度条件不依赖官方星级：build_dataset 用谱面 NPS（每秒音符数）作为难度代理。

用法：
  python -m training.collect_sets --max-sets 200
  python -m training.collect_sets --download-only
"""

import argparse
import json
import os
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

LIST_API = "https://api.sayobot.cn/beatmaplist"
DL_SAYOBOT = "https://txy1.sayobot.cn/beatmaps/download/novideo/{sid}"
DL_CATBOY = "https://catboy.best/d/{sid}"

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OSZ_DIR = os.path.join(DATA_DIR, "osz")
INDEX_PATH = os.path.join(DATA_DIR, "sets_index.json")

APPROVED_OK = {1, 2, 3, 4}  # ranked / approved / qualified / loved

# 搜索关键词（T=4 搜索模式）：前两个词直接命中绝大多数 mania 谱面集，
# 后面是曲师/风格词，扩充曲库多样性
KEYWORDS = [
    "4K", "mania", "4k", "7K", "6K", "5K",
    "camellia", "xi", "t+pazolite", "goreshit", "kurokotei", "silentroom",
    "s3rl", "nanahira", "touhou", "vocaloid", "hardcore", "speedcore",
    "j-core", "anime", "kpop", "etterna", "stepmania", "bms", "dan",
    "jumpstream", "handstream", "jack", "technical", "chordjack",
]


def api_get(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_list(limit: int = 100, offset: int = 0, list_type: int = 2, keyword: str = "",
               mode: int = 8, cls: int = 7) -> dict:
    """beatmaplist: L=数量 O=起点 T=类型(1hot/2new/4search) K=关键词 M=8mania C=7(ranked+qualified+loved)"""
    params = {"L": limit, "O": offset, "T": list_type, "M": mode, "C": cls}
    if keyword:
        params["K"] = keyword
        params["T"] = 4
    url = LIST_API + "?" + urllib.parse.urlencode(params)
    return api_get(url)


def harvest(target: int = 1000) -> dict:
    """分页收集 mania 谱面集元数据 → {sid: record}"""
    os.makedirs(DATA_DIR, exist_ok=True)
    index = {}
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            index = {str(k): v for k, v in json.load(f).items()}
        print(f"已加载索引: {len(index)}", flush=True)

    def _add(resp: dict) -> int:
        new = 0
        for item in resp.get("data", []):
            if item.get("approved") not in APPROVED_OK:
                continue
            if not (item.get("modes", 0) & 8):  # 必须含 mania 难度
                continue
            sid = str(item["sid"])
            if sid in index:
                continue
            index[sid] = {
                "id": item["sid"],
                "title": item.get("title", ""),
                "artist": item.get("artist", ""),
                "creator": item.get("creator", ""),
                "status": item.get("approved", 0),
                "play_count": item.get("play_count", 0),
            }
            new += 1
        return new

    # 关键词搜索分页收集（sayobot 的 M 参数不生效，mania 靠客户端 modes 位过滤）
    for kw in KEYWORDS:
        if len(index) >= target:
            break
        offset = 0
        while len(index) < target:
            try:
                resp = fetch_list(limit=100, offset=offset, keyword=kw)
            except Exception as e:
                print(f"[Warn] 搜索 {kw!r} 失败: {e}", flush=True)
                break
            new = _add(resp)
            endid = resp.get("endid", 0)
            print(f"[搜索 {kw!r}] offset={offset} +{new}（累计 {len(index)}）", flush=True)
            if endid == 0 or endid == offset:
                break
            offset = endid
            time.sleep(0.3)

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(f"[Done] 索引共 {len(index)} 个谱面集", flush=True)
    return index


def download_one(sid: int) -> bool:
    path = os.path.join(OSZ_DIR, f"{sid}.osz")
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        return True
    for url_tpl, name in ((DL_SAYOBOT, "sayobot"), (DL_CATBOY, "catboy")):
        try:
            req = urllib.request.Request(url_tpl.format(sid=sid), headers=UA)
            with urllib.request.urlopen(req, timeout=600) as resp, open(path, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(path) > 10000:
                print(f"  [OK {name}] {sid} ({os.path.getsize(path)/1e6:.1f}MB)", flush=True)
                return True
        except Exception:
            pass
        if os.path.exists(path):
            os.remove(path)
    print(f"  [Warn] 双源均失败 {sid}", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sets", type=int, default=200, help="最多下载多少个谱面集")
    ap.add_argument("--harvest-target", type=int, default=1500, help="索引收集目标数")
    ap.add_argument("--download-only", action="store_true", help="跳过收集，只下载")
    ap.add_argument("--workers", type=int, default=8, help="并行下载线程数")
    args = ap.parse_args()

    os.makedirs(OSZ_DIR, exist_ok=True)

    if args.download_only and os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            index = {str(k): v for k, v in json.load(f).items()}
    else:
        index = harvest(args.harvest_target)

    # 优先玩的人多的谱面集
    candidates = sorted(index.values(), key=lambda r: r.get("play_count", 0), reverse=True)
    n_total = len([f for f in os.listdir(OSZ_DIR) if f.endswith(".osz")])
    todo = [r for r in candidates
            if not os.path.exists(os.path.join(OSZ_DIR, f"{r['id']}.osz"))]
    todo = todo[: max(0, args.max_sets - n_total)]
    print(f"已有 {n_total} 个 .osz，本次并行下载 {len(todo)} 个（{args.workers} 线程）", flush=True)

    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_one, rec["id"]): rec for rec in todo}
        for fut in as_completed(futs):
            if fut.result():
                n_ok += 1
                if n_ok % 10 == 0:
                    print(f"[{n_ok}/{len(todo)}]", flush=True)

    print(f"[Done] 本次下载 {n_ok}/{len(todo)}，累计 {n_total + n_ok} 个 .osz", flush=True)


if __name__ == "__main__":
    main()
