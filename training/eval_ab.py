"""
A/B 评估：v3.1（手工特征）vs v4-raw（原始波形前端），同口径对比

placement：val/test 集固定阈值 0.5 的 slot F1（±2 slot）与事件数量比
selection：teacher-forced combo 精确准确率 vs 众数先验

用法：
  python -m training.eval_ab --split test --max-sets 150
  python -m training.eval_ab --split val
"""

import argparse
import json
import os

import torch

from .dataset import load_split
from .dataset_mert import SelectionDatasetMERT
from .dataset_raw import SelectionDatasetRaw
from .dataset_v3 import SelectionDatasetV3
from .models_mert import MertProjection, mert_selection_model
from .models_raw import PlacementRaw, raw_selection_model
from .models_v3 import PlacementTransformer, SelectionLSTM
from . import train_placement_mert as tpmert
from . import train_placement_raw as tpraw
from . import train_placement_v3 as tpv3
from . import train_selection_mert as tsmert
from . import train_selection_raw as tsraw
from . import train_selection_v3 as tsv3

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")


def _load(model, path, strict=True):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=strict)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--max-sets", type=int, default=100, help="placement 评估的曲集数上限")
    args = ap.parse_args()

    split = load_split()[args.split]
    res = {"split": args.split, "n_files": len(split)}

    # ---- placement ----
    m_v3 = _load(PlacementTransformer(), os.path.join(OUT_DIR, "placement_v3_best.pt"),
                 strict=False).to(args.device).eval()
    f1, ratio = tpv3.evaluate(m_v3, split, args.device, max_sets=args.max_sets)
    res["placement_v3"] = {"slot_f1_tol2": round(f1, 4), "count_ratio": round(ratio, 3)}
    print(f"[placement v3 ] {args.split} slotF1={f1:.4f}  数量比={ratio:.3f}", flush=True)

    m_raw = _load(PlacementRaw(), os.path.join(OUT_DIR, "placement_raw_best.pt")
                  ).to(args.device).eval()
    f1r, ratior = tpraw.evaluate(m_raw, split, args.device, max_sets=args.max_sets)
    res["placement_raw"] = {"slot_f1_tol2": round(f1r, 4), "count_ratio": round(ratior, 3)}
    print(f"[placement raw] {args.split} slotF1={f1r:.4f}  数量比={ratior:.3f}", flush=True)

    mert_ckpt = os.path.join(OUT_DIR, "placement_mert_best.pt")
    if os.path.exists(mert_ckpt):
        m_mert = _load(MertProjection(), mert_ckpt).to(args.device).eval()
        f1m, ratiom = tpmert.evaluate(m_mert, split, args.device, max_sets=args.max_sets)
        res["placement_mert"] = {"slot_f1_tol2": round(f1m, 4), "count_ratio": round(ratiom, 3)}
        print(f"[placement mert] {args.split} slotF1={f1m:.4f}  数量比={ratiom:.3f}", flush=True)

    # ---- selection ----
    s_v3 = _load(SelectionLSTM(), os.path.join(OUT_DIR, "selection_v3_best.pt"),
                 strict=False).to(args.device).eval()
    ds_v3 = SelectionDatasetV3(split, mirror=False)
    acc, prior = tsv3.evaluate(s_v3, ds_v3, args.device)
    res["selection_v3"] = {"combo_acc": round(acc, 4), "majority_prior": round(prior, 4)}
    print(f"[selection v3 ] {args.split} combo_acc={acc:.4f} (先验 {prior:.4f})", flush=True)

    s_raw = _load(raw_selection_model(), os.path.join(OUT_DIR, "selection_raw_best.pt")
                  ).to(args.device).eval()
    ds_raw = SelectionDatasetRaw(split, mirror=False)
    accr, priorr = tsraw.evaluate(s_raw, ds_raw, args.device)
    res["selection_raw"] = {"combo_acc": round(accr, 4), "majority_prior": round(priorr, 4)}
    print(f"[selection raw] {args.split} combo_acc={accr:.4f} (先验 {priorr:.4f})", flush=True)

    mert_sel_ckpt = os.path.join(OUT_DIR, "selection_mert_best.pt")
    if os.path.exists(mert_sel_ckpt):
        s_mert = _load(mert_selection_model(), mert_sel_ckpt).to(args.device).eval()
        ds_mert = SelectionDatasetMERT(split, mirror=False)
        accm, priorm = tsmert.evaluate(s_mert, ds_mert, args.device)
        res["selection_mert"] = {"combo_acc": round(accm, 4), "majority_prior": round(priorm, 4)}
        print(f"[selection mert] {args.split} combo_acc={accm:.4f} (先验 {priorm:.4f})", flush=True)

    out_path = os.path.join(OUT_DIR, f"eval_ab_{args.split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"[Done] 结果已存 {out_path}", flush=True)


if __name__ == "__main__":
    main()
