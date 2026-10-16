"""
用训练好的模型生成谱面（推理）

generate_notes_ml(y, sr, keys, stars) → notes 列表（与 BeatMapper 输出同构，
hit 带 end_time=None，hold 带 end_time）

v3（优先，若存在 placement_v3_best.pt + selection_v3_best.pt）：
  节拍网格 placement（每拍 48 时隙事件多标签，Transformer）
  → 256 组合 selection（4列×{无,音符,长音头,长音尾}，nucleus + n-gram 惩罚）
  → 头尾配对出长音 → BeatMapper._resolve_overlaps 收尾

v2（回退）：帧级 PlacementNet 多任务 + SelectionNet 选列（见 generate_notes_ml_v2）
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .features import SR, HOP, extract_features, estimate_beats
from .models import PlacementNet, SelectionNet, HISTORY, N_FEATS
from .train_placement import peak_pick, MIN_HOLD_FRAMES
from .models_v3 import (
    SLOTS_PER_BEAT, N_COMBO, BOS, SEL_HISTORY,
    PlacementTransformer, SelectionLSTM, beats_to_units,
)
from .dataset_v3 import normalize_feats
from .features import BEATS_PER_BAR, FRAME_MS
from .models_raw import FEAT_DIM_RAW, PlacementRaw, raw_selection_model
from .models_mert import D_MERT
from .raw_audio import RAW_NORM_SCALE, warp_beats_raw

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(ROOT, "output")

_CTX_W = 32
_FRAME_MS = HOP / SR * 1000.0

_model_cache: Dict[str, Any] = {}


def _load_models(model_dir: str, keys: int, device: str):
    if "placement" not in _model_cache:
        pm = PlacementNet()
        pm.load_state_dict(torch.load(os.path.join(model_dir, "placement_best.pt"), map_location=device))
        pm.to(device).eval()
        _model_cache["placement"] = pm
    key = f"selection_{keys}"
    if key not in _model_cache:
        sm = SelectionNet(keys=keys)
        sm.load_state_dict(torch.load(os.path.join(model_dir, "selection_best.pt"), map_location=device))
        sm.to(device).eval()
        _model_cache[key] = sm
    return _model_cache["placement"], _model_cache[key]


_avail_cache: Dict[Tuple[str, str], tuple] = {}


def _cached_backend_check(name: str, model_dir: str,
                          weight_files: Tuple[str, ...], check_impl) -> bool:
    """按（权重路径, mtime）缓存后端校验结果，避免每次 torch.load 全量权重

    任一权重缺失直接判不可用；权重被替换（mtime 变化）时缓存自动失效重检。
    """
    paths = [os.path.join(model_dir, w) for w in weight_files]
    try:
        fp = tuple((p, os.path.getmtime(p)) for p in paths)
    except OSError:
        return False
    key = (name, model_dir)
    hit = _avail_cache.get(key)
    if hit is not None and hit[0] == fp:
        return hit[1]
    try:
        ok = bool(check_impl())
    except Exception:
        ok = False
    _avail_cache[key] = (fp, ok)
    return ok


def _v3_check_impl(model_dir: str) -> bool:
    p = os.path.join(model_dir, "placement_v3_best.pt")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    return sd["inp.weight"].shape[1] == 8 * N_FEATS


def v3_available(model_dir: str = DEFAULT_MODEL_DIR) -> bool:
    """v3 节拍网格模型就绪（维度与当前特征匹配）"""
    return _cached_backend_check(
        "v3", model_dir, ("placement_v3_best.pt", "selection_v3_best.pt"),
        lambda: _v3_check_impl(model_dir))


def v2_available(model_dir: str = DEFAULT_MODEL_DIR) -> bool:
    """v2 帧级（回退）模型就绪"""

    def _impl() -> bool:
        p1 = os.path.join(model_dir, "placement_best.pt")
        p2 = os.path.join(model_dir, "selection_best.pt")
        sd = torch.load(p1, map_location="cpu", weights_only=True)
        if sd["conv.0.weight"].shape[1] != N_FEATS:
            return False
        sd = torch.load(p2, map_location="cpu", weights_only=True)
        return sd["ctx_encoder.net.0.weight"].shape[1] == N_FEATS

    return _cached_backend_check(
        "v2", model_dir, ("placement_best.pt", "selection_best.pt"), _impl)


def models_available(model_dir: str = DEFAULT_MODEL_DIR) -> bool:
    """任一可用代际（v3 → mert → raw → v2）"""
    return any(b["available"] for b in list_backends(model_dir))


def list_backends(model_dir: str = DEFAULT_MODEL_DIR) -> List[Dict[str, Any]]:
    """枚举全部 ML 后端及可用性（顺序即 auto 的回退优先级；GUI/CLI 模型识别用）

    返回 [{"id", "label", "available", "weights"}, ...]
    """
    return [
        {"id": "v3", "label": "v3 节拍网格（手工特征）",
         "available": v3_available(model_dir),
         "weights": ["placement_v3_best.pt", "selection_v3_best.pt"]},
        {"id": "mert", "label": "v5 MERT 预训练编码器（实验）",
         "available": mert_available(model_dir),
         "weights": ["placement_mert_best.pt", "selection_mert_best.pt"]},
        {"id": "raw", "label": "v4 原始波形前端（实验）",
         "available": raw_available(model_dir),
         "weights": ["placement_raw_best.pt", "selection_raw_best.pt"]},
        {"id": "v2", "label": "v2 帧级（回退）",
         "available": v2_available(model_dir),
         "weights": ["placement_best.pt", "selection_best.pt"]},
    ]


def resolve_backend(backend: str = "auto", model_dir: str = DEFAULT_MODEL_DIR) -> str:
    """解析后端选择：显式指定 > 环境变量 MUSTAFF_ML_BACKEND > auto（按优先级）

    显式/环境指定的后端不可用时抛 RuntimeError（附检测详情）。
    """
    backend = (backend or "auto").strip().lower()
    env = os.environ.get("MUSTAFF_ML_BACKEND", "").strip().lower()
    if backend in ("", "auto") and env:
        backend = env
    backends = list_backends(model_dir)
    by_id = {b["id"]: b for b in backends}
    if backend not in ("", "auto"):
        if backend not in by_id:
            raise RuntimeError(f"未知 ML 后端: {backend}（可选: auto/{'/'.join(by_id)}）")
        if not by_id[backend]["available"]:
            missing = [w for w in by_id[backend]["weights"]
                       if not os.path.exists(os.path.join(model_dir, w))]
            status = " ".join(f"{b['id']}={'✓' if b['available'] else '✗'}" for b in backends)
            raise RuntimeError(
                f"ML 后端 {backend} 不可用（缺 {', '.join(missing) or '权重校验未通过'}）；"
                f"当前检测: {status}")
        return backend
    for b in backends:
        if b["available"]:
            return b["id"]
    raise RuntimeError("未找到任何可用的 ML 模型（training/output/*.pt），请先运行训练流程")


def sample_top_p(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0,
                 recent: Optional[List[int]] = None, rep_penalty: float = 1.07) -> int:
    """nucleus 采样 + 对最近出现过的列做重复惩罚（ITGPT 式）"""
    logits = logits / max(temperature, 1e-4)
    if recent:
        penalty = torch.ones_like(logits)
        for c in set(recent):
            penalty[c] = rep_penalty
        # 大于 0 的 logit 除、小于 0 的乘，都是压低概率
        logits = torch.where(logits > 0, logits / penalty, logits * penalty)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # 保留累计概率不超过 top_p 的最小集合（至少保留 1 个）
    keep = cumsum - sorted_probs < top_p
    keep[0] = True
    filt = sorted_probs * keep.float()
    filt = filt / filt.sum()
    return int(sorted_idx[torch.multinomial(filt, 1)].item())


# ---------------- v3：节拍网格 + 组合选择 ----------------

_CTX_BEATS = 2

# 组合状态掩码/偏置表（生成时约束用）：state 2=长音头 3=长音尾
_COMBO_STATES = torch.arange(N_COMBO, dtype=torch.long)
_STATES_4 = torch.stack([(_COMBO_STATES >> (2 * c)) & 3 for c in range(4)], dim=1)  # [256,4]
HAS_HOLD_HEAD = (_STATES_4 == 2).any(dim=1)   # [256] 含长音头的组合
STATE23_MASK = ((_STATES_4 == 2) | (_STATES_4 == 3)).any(dim=1)  # 含长音状态的组合（联合采样时屏蔽）


def _load_models_v3(model_dir: str, device: str):
    if "placement_v3" not in _model_cache:
        pm = PlacementTransformer()
        sd = torch.load(os.path.join(model_dir, "placement_v3_best.pt"),
                        map_location=device, weights_only=True)
        pm.load_state_dict(sd, strict=False)  # 旧 checkpoint 无 diag 头，宽松加载
        pm.to(device).eval()
        _model_cache["placement_v3"] = pm
    if "selection_v3" not in _model_cache:
        sm = SelectionLSTM()
        sm.load_state_dict(torch.load(os.path.join(model_dir, "selection_v3_best.pt"),
                                      map_location=device, weights_only=True))
        sm.to(device).eval()
        _model_cache["selection_v3"] = sm
    return _model_cache["placement_v3"], _model_cache["selection_v3"]


def _ngram_penalty(logits: torch.Tensor, history: List[int],
                   window: int = 20, base: float = 1.07) -> torch.Tensor:
    """ITGPT 式 n-gram 重复惩罚：前 20 步内，会构成长度 l∈[4,8] 重复 n-gram 的
    候选组合按 base^(l-3) 压低"""
    penalty = torch.ones_like(logits)
    ctx = history[-window:]
    for l in range(4, 9):
        if len(ctx) < l - 1:
            continue
        tail = tuple(ctx[-(l - 1):])
        seen = {tuple(ctx[i: i + l]) for i in range(0, len(ctx) - l + 1)}
        p = base ** (l - 3)
        for c in range(N_COMBO):
            if tail + (c,) in seen:
                penalty[c] *= p
    return torch.where(logits > 0, logits / penalty, logits * penalty)


def _pick_events(probs: np.ndarray, beat_times: np.ndarray, threshold: float):
    """事件提取：阈值 + 拍内相邻时隙 NMS（保留概率高的）→ [(beat, slot, time_ms, prob)]"""
    events = []
    n_units = probs.shape[0]
    for b in range(n_units):
        row = probs[b]
        cand = [s for s in range(SLOTS_PER_BEAT) if row[s] > threshold]
        kept = []
        for s in sorted(cand, key=lambda s: -row[s]):
            if all(abs(s - k) > 1 for k in kept):
                kept.append(s)
        for s in sorted(kept):
            t = beat_times[b] + (beat_times[b + 1] - beat_times[b]) * s / SLOTS_PER_BEAT
            events.append((b, s, t, float(row[s])))
    return events


def _adaptive_threshold(probs: np.ndarray, beat_times: np.ndarray, stars: float) -> float:
    """按目标密度反推阈值：目标 NPS = 1.5 × stars（6★→9, 10★→15），
    阈值限制在 [0.32, 0.80] 内防止为了凑数往安静段硬塞音符"""
    target_nps = max(1.0, 1.5 * stars)
    duration_s = beat_times[-1] / 1000.0
    target = target_nps * duration_s * 1.1  # 事件含长音尾，乘 1.1 补偿
    lo, hi = 0.32, 0.80
    for _ in range(12):
        mid = (lo + hi) / 2
        n = sum(int((row > mid).sum()) for row in probs)  # 粗计（跳过 NMS，够准）
        if n > target:
            lo = mid
        else:
            hi = mid
    return hi


def _v3_select_and_emit(
    events, units, beat_times, fd_idx, bpm, stars, keys, selection, *,
    temperature, top_p, head_thr, tail_thr, min_hold_ms, seed, device,
):
    """事件列表 + 拍级音频单元 → 自回归选组合 → 音符列表

    v3 特征单元（8×82）与 v4-raw 前端嵌入（8×D）共用此逻辑：
    selection 的 audio_proj 输入维度与各自的 S*F 匹配，其余完全一致。
    """
    n_units = units.shape[0]
    # 预计算每个事件的音频上下文与条件向量
    S, F = units.shape[1], units.shape[2]
    audio_ctx = np.zeros((len(events), (2 * _CTX_BEATS + 1) * S * F), dtype=np.float32)
    cond = np.zeros((len(events), 9), dtype=np.float32)
    for i, (b, s, t, _) in enumerate(events):
        lo = b - _CTX_BEATS
        w = np.zeros((2 * _CTX_BEATS + 1, S, F), dtype=np.float32)
        src_lo, src_hi = max(0, lo), min(n_units, b + _CTX_BEATS + 1)
        w[src_lo - lo: src_hi - lo] = units[src_lo:src_hi]
        audio_ctx[i] = w.reshape(-1)
        rad = s / SLOTS_PER_BEAT * 2 * np.pi
        bar_idx = (b - fd_idx) % BEATS_PER_BAR
        ioi = t - events[i - 1][2] if i > 0 else 0.0
        cond[i] = [np.clip(bpm / 300.0, 0, 2), stars / 10.0,
                   np.sin(rad), np.cos(rad),
                   float(bar_idx == 0), float(bar_idx == 1),
                   float(bar_idx == 2), float(bar_idx == 3),
                   np.log1p(ioi) / 8.0]

    # 自回归选组合（逐步前向，LSTM 隐状态随步携带，O(N) 总开销）
    if seed is not None:
        torch.manual_seed(seed)
    history: List[int] = []
    open_hold: Dict[int, float] = {}
    notes: List[Dict[str, Any]] = []
    hidden = None
    event_times = np.array([e[2] for e in events])
    state23_mask = STATE23_MASK.to(device)
    # 每个事件是否具备长音收尾位：[t+min_hold, t+2拍] 内存在后续事件（双指针 O(N)）
    tail_ok = np.zeros(len(events), dtype=bool)
    for i, (b, s, t, _) in enumerate(events):
        beat_len = beat_times[b + 1] - beat_times[b]
        j = int(np.searchsorted(event_times, t + min_hold_ms))
        tail_ok[i] = j < len(event_times) and event_times[j] <= t + 2 * beat_len

    def _emit_hit(t, col):
        notes.append({"time": int(round(t)), "column": col, "type": "hit",
                      "end_time": None, "speed": 10.0})

    def _emit_hold(t0, t1, col):
        if t1 - t0 >= MIN_HOLD_FRAMES * FRAME_MS:
            notes.append({"time": int(round(t0)), "column": col, "type": "hold",
                          "end_time": int(round(t1)), "speed": 10.0})
        else:
            _emit_hit(t0, col)

    for i, (b, s, t, _) in enumerate(events):
        prev = [BOS] * (SEL_HISTORY - len(history)) + history[-SEL_HISTORY:]
        a_t = torch.from_numpy(audio_ctx[i: i + 1])[None].to(device)
        p_t = torch.tensor([prev], dtype=torch.long, device=device)[:, None, :]
        c_t = torch.from_numpy(cond[i: i + 1])[None].to(device)
        logits, hidden = selection(a_t, p_t, c_t, hidden)
        logits = logits[0, 0]

        # 列级边缘概率（把 256 联合分布按列聚合）——长音头/尾决策用，低方差不级联
        probs_full = torch.softmax(logits, dim=-1).reshape(4, 4, 4, 4)
        marg = [probs_full.sum(dim=tuple(a for a in range(4) if a != 3 - col))
                for col in range(4)]  # 每列 [4]：P(无/音符/长音头/长音尾)

        # 音符/和弦：只在 {无,音符} 子空间里采样（长音交给边缘规则，避免级联失稳）
        masked = logits.masked_fill(state23_mask, float("-inf"))
        if temperature > 0:
            m_logits = _ngram_penalty(masked / temperature, history)
            probs_c = torch.softmax(m_logits, dim=-1)
            sp, si = torch.sort(probs_c, descending=True)
            cum = torch.cumsum(sp, dim=0)
            keep = cum - sp < top_p
            keep[0] = True
            filt = sp * keep.float()
            filt = filt / filt.sum()
            combo = int(si[torch.multinomial(filt, 1)].item())
        else:
            combo = int(masked.argmax().item())
        history.append(combo)
        states = [(combo >> (2 * col)) & 3 for col in range(4)]
        beat_len = beat_times[b + 1] - beat_times[b]

        # 先收尾：边缘 P(尾) 超阈 / 同列来了新音符 / 超 2 拍强制收
        for col in range(4):
            if col in open_hold and (float(marg[col][3]) > tail_thr
                                     or states[col] == 1
                                     or t - open_hold[col] > 2 * beat_len):
                _emit_hold(open_hold.pop(col), t, col)
        # 再开头：边缘 P(头) 超阈且有收尾位（同事件先收后开 → 支持长音连打）
        for col in range(4):
            if col not in open_hold and tail_ok[i] and float(marg[col][2]) > head_thr:
                open_hold[col] = t
                states[col] = 0  # 该列普通音符被长音头取代
        for col in range(4):
            if states[col] == 1:
                _emit_hit(t, col)

    last_t = beat_times[-1]
    for col, t0 in open_hold.items():  # 曲末未闭合的长音补尾
        _emit_hold(t0, last_t, col)

    notes.sort(key=lambda n: n["time"])
    from mustaff.mapper import BeatMapper
    return BeatMapper(keys=keys)._resolve_overlaps(notes)


@torch.no_grad()
def generate_notes_ml_v3(
    y: np.ndarray,
    sr: int = SR,
    keys: int = 4,
    stars: float = 5.0,
    model_dir: str = DEFAULT_MODEL_DIR,
    threshold: Optional[float] = None,  # None=按星级自适应密度（推荐）；显式值=固定阈值
    temperature: float = 0.85,  # 0=联合 argmax（最保守）；0.8~0.9 兼顾稳定与变化
    top_p: float = 0.9,
    head_thr: float = 0.02,   # 长音头边缘概率阈值（列级聚合，val 实测 ~7% 长音）
    tail_thr: float = 0.15,   # 长音尾边缘概率阈值（模型尾信号弱，主靠同列新音符/2拍上限收尾）
    min_hold_ms: float = 180.0,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """v3 推理：拍网格事件 → 256 组合 → 头尾配对出长音"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    placement, selection = _load_models_v3(model_dir, device)

    y = y.astype(np.float32)
    beat_times, fd_idx, bpm = estimate_beats(y)
    feats = normalize_feats(extract_features(y, beat_times, fd_idx, bpm).astype(np.float32))
    units, bar_phase = beats_to_units(feats, beat_times, fd_idx)
    n_units = units.shape[0]
    if n_units < 4:
        return []

    cond_t = torch.tensor([[np.clip(bpm / 300.0, 0, 2), stars / 10.0]],
                          dtype=torch.float32, device=device)
    x = torch.from_numpy(units)[None].to(device)
    bp = torch.from_numpy(bar_phase)[None].to(device)
    probs = torch.sigmoid(placement(x, bp, cond_t)[0])[0].cpu().numpy()  # [B,48]

    if threshold is None:
        threshold = _adaptive_threshold(probs, beat_times, stars)
    events = _pick_events(probs, beat_times, threshold)
    if not events:
        return []

    return _v3_select_and_emit(events, units, beat_times, fd_idx, bpm, stars, keys,
                               selection, temperature=temperature, top_p=top_p,
                               head_thr=head_thr, tail_thr=tail_thr,
                               min_hold_ms=min_hold_ms, seed=seed, device=device)


# ---------------- v2：帧级（回退路径） ----------------

@torch.no_grad()
def generate_notes_ml_v2(
    y: np.ndarray,
    sr: int = SR,
    keys: int = 4,
    stars: float = 5.0,
    model_dir: str = DEFAULT_MODEL_DIR,
    threshold: float = 0.5,
    hold_threshold: float = 0.5,
    hold_max_ratio: float = 0.15,
    temperature: float = 1.0,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """从音频波形生成音符（hit + hold）"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    placement, selection = _load_models(model_dir, keys, device)

    y = y.astype(np.float32)
    beat_times, fd_idx, bpm = estimate_beats(y)
    feats = extract_features(y, beat_times, fd_idx, bpm).astype(np.float32)
    T = feats.shape[0]
    stars_t = torch.tensor([stars / 10.0], dtype=torch.float32, device=device)

    # 1) placement：分块前向拼成整首歌的三路输出
    probs = np.zeros(T, dtype=np.float32)
    holdps = np.zeros(T, dtype=np.float32)
    lens = np.zeros(T, dtype=np.float32)
    step = 2048
    for s in range(0, T, step):
        chunk = feats[s: s + step]
        pad = step - chunk.shape[0]
        if pad > 0:
            chunk = np.pad(chunk, ((0, pad), (0, 0)))
        x = torch.from_numpy(chunk)[None].to(device)
        nl, hl, lp = placement(x, stars_t)
        probs[s: s + step] = torch.sigmoid(nl)[0].cpu().numpy()[: T - s]
        holdps[s: s + step] = torch.sigmoid(hl)[0].cpu().numpy()[: T - s]
        lens[s: s + step] = lp[0].cpu().numpy()[: T - s]

    frames = peak_pick(probs, threshold=threshold, min_dist=2)
    if len(frames) == 0:
        return []

    # 长音判定阈值：0.5 绝对下限 + 分位数上限（hold 头区分度弱时防"全曲长音"，
    # 每首歌长音比例封顶 hold_max_ratio，贴合人类谱面 ~13% 的先验）
    hp = holdps[frames]
    hold_thr = max(hold_threshold, float(np.quantile(hp, 1.0 - hold_max_ratio)))

    # 2) selection：自回归选列（nucleus 采样 + 近 4 列重复惩罚）
    half = _CTX_W // 2

    def _ctx(frame: int) -> np.ndarray:
        lo = max(0, frame - half)
        hi = min(T, frame + half)
        w = feats[lo:hi]
        if w.shape[0] < _CTX_W:
            pad_before = max(0, half - frame)
            w = np.pad(w, ((pad_before, _CTX_W - w.shape[0] - pad_before), (0, 0)))
        return w

    notes: List[Dict[str, Any]] = []
    history = [keys] * HISTORY  # <BOS> 填充
    for f in frames:
        f = int(f)
        ctx = torch.from_numpy(_ctx(f))[None, None].to(device)  # [1,1,W,82]
        prev = torch.tensor([list(history[-HISTORY:])], dtype=torch.long, device=device)[:, None, :]  # [1,1,HISTORY]
        logits = selection(ctx, prev, stars_t)[0, 0]  # [keys]
        col = sample_top_p(logits, top_p=0.9, temperature=temperature,
                           recent=[c for c in history[-HISTORY:] if c < keys])
        history.append(col)

        time_ms = int(round(f * _FRAME_MS))
        note_type = "hit"
        end_time = None
        if holdps[f] > hold_thr:
            dur_frames = int(np.expm1(lens[f]))
            if dur_frames >= MIN_HOLD_FRAMES:
                note_type = "hold"
                end_time = int(round((f + dur_frames) * _FRAME_MS))
        notes.append({
            "time": time_ms,
            "column": col,
            "type": note_type,
            "end_time": end_time,
            "speed": 10.0,
        })

    # 3) 合法性收尾（长音覆盖冲突消解）
    from mustaff.mapper import BeatMapper
    notes = BeatMapper(keys=keys)._resolve_overlaps(notes)
    return notes


def generate_notes_ml(y: np.ndarray, sr: int = SR, keys: int = 4, stars: float = 5.0,
                      model_dir: str = DEFAULT_MODEL_DIR, backend: str = "auto",
                      **kwargs) -> List[Dict[str, Any]]:
    """统一入口：按 backend 选择生成后端（auto|v3|mert|raw|v2）

    auto 优先级 v3 → mert → raw → v2（实验后端默认不启用，详见
    training/output/eval_ab_*.json 的 A/B 结论）；环境变量 MUSTAFF_ML_BACKEND
    在 backend=auto 时生效，显式 backend 参数优先于环境变量。
    """
    chosen = resolve_backend(backend, model_dir)
    fn = {"v3": generate_notes_ml_v3, "mert": generate_notes_ml_mert,
          "raw": generate_notes_ml_raw, "v2": generate_notes_ml_v2}[chosen]
    return fn(y, sr=sr, keys=keys, stars=stars, model_dir=model_dir, **kwargs)


# ---------------- v4-raw：原始波形前端（与 v3 同网格，音频表征不同） ----------------

def raw_available(model_dir: str = DEFAULT_MODEL_DIR) -> bool:
    """v4-raw（原始波形前端）模型就绪：placement 键前缀 + selection 输入维度均校验"""

    def _impl() -> bool:
        sd = torch.load(os.path.join(model_dir, "placement_raw_best.pt"),
                        map_location="cpu", weights_only=True)
        if not any(k.startswith("frontend.") for k in sd):
            return False
        ss = torch.load(os.path.join(model_dir, "selection_raw_best.pt"),
                        map_location="cpu", weights_only=True)
        w = ss.get("audio_proj.0.weight")
        # selection(n_feats=128) 的 audio_proj 输入维度 = (2·ctx+1)·steps·D = 5·8·128
        return w is not None and w.shape[1] == 5 * 8 * FEAT_DIM_RAW

    return _cached_backend_check(
        "raw", model_dir, ("placement_raw_best.pt", "selection_raw_best.pt"), _impl)


def _load_models_raw(model_dir: str, device: str):
    if "placement_raw" not in _model_cache:
        pm = PlacementRaw()
        sd = torch.load(os.path.join(model_dir, "placement_raw_best.pt"),
                        map_location=device, weights_only=True)
        pm.load_state_dict(sd)
        pm.to(device).eval()
        _model_cache["placement_raw"] = pm
    if "selection_raw" not in _model_cache:
        sm = raw_selection_model()
        sm.load_state_dict(torch.load(os.path.join(model_dir, "selection_raw_best.pt"),
                                      map_location=device, weights_only=True))
        sm.to(device).eval()
        _model_cache["selection_raw"] = sm
    return _model_cache["placement_raw"], _model_cache["selection_raw"]


@torch.no_grad()
def generate_notes_ml_raw(
    y: np.ndarray,
    sr: int = SR,
    keys: int = 4,
    stars: float = 5.0,
    model_dir: str = DEFAULT_MODEL_DIR,
    threshold: Optional[float] = None,
    temperature: float = 0.85,
    top_p: float = 0.9,
    head_thr: float = 0.02,
    tail_thr: float = 0.15,
    min_hold_ms: float = 180.0,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """v4-raw 推理：原始波形按拍重采样（与训练数据同口径）→ 卷积前端 →
    v3 同款节拍网格 placement → 组合 selection。事件提取/自适应阈值/长音收尾全部复用。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    placement, selection = _load_models_raw(model_dir, device)

    y = np.asarray(y, dtype=np.float32)
    beat_times, fd_idx, bpm = estimate_beats(y)
    if len(beat_times) < 5:
        return []

    beats_i16 = warp_beats_raw(y, beat_times)          # [T, RAW_LEN] int16
    n_units = beats_i16.shape[0]
    x = torch.from_numpy(beats_i16.astype(np.float32) / RAW_NORM_SCALE)[None].to(device)
    bar_phase = (np.mod(np.arange(n_units) - fd_idx, 4) / 4.0).astype(np.float32)
    bp = torch.from_numpy(bar_phase)[None].to(device)
    cond_t = torch.tensor([[np.clip(bpm / 300.0, 0, 2), stars / 10.0]],
                          dtype=torch.float32, device=device)
    emb = placement.frontend(x)[0].float().cpu().numpy()   # [T, 8, D]
    if device.startswith("cuda"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = placement.transformer(
                torch.from_numpy(emb)[None].to(device), bp, cond_t)[0]
    else:
        logits = placement.transformer(
            torch.from_numpy(emb)[None].to(device), bp, cond_t)[0]
    probs = torch.sigmoid(logits.float())[0].cpu().numpy()  # [T, 48]

    if threshold is None:
        threshold = _adaptive_threshold(probs, beat_times, stars)
    events = _pick_events(probs, beat_times, threshold)
    if not events:
        return []
    return _v3_select_and_emit(events, emb, beat_times, fd_idx, bpm, stars, keys,
                               selection, temperature=temperature, top_p=top_p,
                               head_thr=head_thr, tail_thr=tail_thr,
                               min_hold_ms=min_hold_ms, seed=seed, device=device)


# ---------------- v5-mert：冻结预训练编码器前端（MERT） ----------------

def mert_available(model_dir: str = DEFAULT_MODEL_DIR) -> bool:
    """v5-mert（预训练编码器）模型就绪：placement 键前缀 + selection 输入维度均校验"""

    def _impl() -> bool:
        sd = torch.load(os.path.join(model_dir, "placement_mert_best.pt"),
                        map_location="cpu", weights_only=True)
        if not any(k.startswith("proj.") for k in sd):
            return False
        ss = torch.load(os.path.join(model_dir, "selection_mert_best.pt"),
                        map_location="cpu", weights_only=True)
        w = ss.get("audio_proj.0.weight")
        # selection(n_feats=768) 的 audio_proj 输入维度 = (2·ctx+1)·steps·D = 5·8·768
        return w is not None and w.shape[1] == 5 * 8 * D_MERT

    return _cached_backend_check(
        "mert", model_dir, ("placement_mert_best.pt", "selection_mert_best.pt"), _impl)


def _load_models_mert(model_dir: str, device: str):
    if "placement_mert" not in _model_cache:
        from .models_mert import MertProjection
        pm = MertProjection()
        sd = torch.load(os.path.join(model_dir, "placement_mert_best.pt"),
                        map_location=device, weights_only=True)
        pm.load_state_dict(sd)
        pm.to(device).eval()
        _model_cache["placement_mert"] = pm
    if "selection_mert" not in _model_cache:
        from .models_mert import mert_selection_model
        sm = mert_selection_model()
        sm.load_state_dict(torch.load(os.path.join(model_dir, "selection_mert_best.pt"),
                                      map_location=device, weights_only=True))
        sm.to(device).eval()
        _model_cache["selection_mert"] = sm
    return _model_cache["placement_mert"], _model_cache["selection_mert"]


@torch.no_grad()
def generate_notes_ml_mert(
    y: np.ndarray,
    sr: int = SR,
    keys: int = 4,
    stars: float = 5.0,
    model_dir: str = DEFAULT_MODEL_DIR,
    threshold: Optional[float] = None,
    temperature: float = 0.85,
    top_p: float = 0.9,
    head_thr: float = 0.02,
    tail_thr: float = 0.15,
    min_hold_ms: float = 180.0,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """v5-mert 推理：波形重采样 24kHz → 冻结 MERT 帧（30s 分块）→ 按拍 8×768 单元 →
    与 v3 同款节拍网格 placement → 组合 selection。需 transformers + MERT 权重。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    placement, selection = _load_models_mert(model_dir, device)

    y = np.asarray(y, dtype=np.float32)
    beat_times, fd_idx, bpm = estimate_beats(y)
    if len(beat_times) < 5:
        return []

    import librosa
    y24 = librosa.resample(y, orig_sr=sr, target_sr=24000) if int(sr) != 24000 else y
    from .build_dataset_mert import _frames_to_beats, _mert_frames
    frames = _mert_frames(y24, device)                # [T_frames, 768]
    units = _frames_to_beats(frames, beat_times).astype(np.float32)  # [T, 8, 768]
    n_units = units.shape[0]
    if n_units < 4:
        return []

    bar_phase = (np.mod(np.arange(n_units) - fd_idx, 4) / 4.0).astype(np.float32)
    x = torch.from_numpy(units)[None].to(device)
    bp = torch.from_numpy(bar_phase)[None].to(device)
    cond_t = torch.tensor([[np.clip(bpm / 300.0, 0, 2), stars / 10.0]],
                          dtype=torch.float32, device=device)
    if device.startswith("cuda"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = placement(x, bp, cond_t)[0]
    else:
        logits = placement(x, bp, cond_t)[0]
    probs = torch.sigmoid(logits.float())[0].cpu().numpy()  # [T, 48]

    if threshold is None:
        threshold = _adaptive_threshold(probs, beat_times, stars)
    events = _pick_events(probs, beat_times, threshold)
    if not events:
        return []
    return _v3_select_and_emit(events, units, beat_times, fd_idx, bpm, stars, keys,
                               selection, temperature=temperature, top_p=top_p,
                               head_thr=head_thr, tail_thr=tail_thr,
                               min_hold_ms=min_hold_ms, seed=seed, device=device)
