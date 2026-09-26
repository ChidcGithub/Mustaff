"""临时：统计数据集中长音占比与时长分布"""
import os
import numpy as np
from training.dataset import NPZ_DIR, load_split

split = load_split()
files = split["train"][:80]  # 抽样 80 个 set 足够估计
n_notes = 0
n_holds = 0
durs = []
for fn in files:
    path = os.path.join(NPZ_DIR, fn)
    with np.load(path) as z:
        for k in z.files:
            if not k.startswith("labels_"):
                continue
            bid = k[len("labels_"):]
            frames = z[f"frames_{bid}"]
            holds = z[f"holds_{bid}"]
            is_hold = holds > frames
            n_notes += len(frames)
            n_holds += int(is_hold.sum())
            durs.extend((holds[is_hold] - frames[is_hold]).tolist())

durs = np.array(durs)
print(f"notes={n_notes}  holds={n_holds}  hold_ratio={n_holds/max(1,n_notes):.3f}")
if len(durs):
    print(f"dur frames: p10={np.percentile(durs,10):.0f} p50={np.percentile(durs,50):.0f} "
          f"p90={np.percentile(durs,90):.0f} max={durs.max()}")
    print(f"dur ms (p50): {np.percentile(durs,50)*23.2:.0f}ms")
