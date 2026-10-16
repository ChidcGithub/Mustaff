# 音频表征 A/B 实验报告：手工特征 vs 原始波形 vs 预训练编码器

> 2026-10-16 · Mustaff 项目 · 回答的问题：**「让模型直接读音频 raw 数据而不是提取特征，会不会更聪明？」**
>
> **最终结论：在当前数据规模（2,912 曲集）下，端到端质量由 placement 决定，v3.1 手工特征仍然最强；但「模型直接读音频」并非没有价值——冻结 MERT 预训练编码器让 selection 反超了 v3.1（0.4361 vs 0.4289），并显著优于从零学习的波形前端（0.5359 vs 0.4806 placement）。** 两代新后端均保留为实验选项，默认仍是 v3。

---

## 1. 实验设计（三轮同口径 A/B）

三者共享完全相同的节拍网格结构（每拍 48 时隙事件 placement + 256 组合 selection）、
标签管线、训练超参（25/15 epochs、bf16 AMP、EMA 0.999、warmup+cosine、scheduled sampling、镜像增广），**唯一变量是音频表征**：

| 后端 | 音频输入 | 前端 | placement 可训参数 |
|---|---|---|---|
| v3.1（基线） | 82 维手工特征（64 mel + 12 chroma + onset/RMS/相位）按拍重采样 8×82 | 无（特征即输入） | 2.70M |
| v4-raw | 每拍原始波形 4096 点（22.05kHz、p95 归一、抗混叠低通） | 从零 1D 卷积（SampleCNN 式，0.2M） | 2.95M |
| v5-mert | 每拍 MERT-v1-95M 嵌入 8×768（13 层均值池化，75fps→按拍 8 步，冻结） | Linear 投影 + LayerNorm | 3.02M |

数据：3,305 个 `.osz` 解码 → `MustaffData/raw_beats/`（13.2GB）与 `mert_beats/`（19.9GB），
2912/2912 全量；训练/推理全程同口径（重采样链、滤波、分块、归一均单一实现复用）。

## 2. 结果（test 集 150 曲集，固定阈值 0.5）

### placement（事件放置）

| 指标 | v3.1 手工特征 | v5-mert | v4-raw |
|---|---|---|---|
| slot F1（±2 时隙 ≈ ±17-25ms） | **0.6776** | 0.5359 | 0.4806 |
| 事件数量比（1.0 为完美） | **1.130** | 1.608 | 2.124 |

### selection（列分配）

| 指标 | v5-mert | v3.1 | v4-raw |
|---|---|---|---|
| combo 精确准确率 | **0.4361** | 0.4289 | 0.4117 |

（先验基线 0.14；val 集排序一致：mert 0.4400 > v3 0.4327 > raw 0.4219）

### 生成质量冒烟（verify_gen，6★，同一首歌）

| | v3 | v5-mert | v4-raw |
|---|---|---|---|
| 音符数 / 长音比例 | 1013 / 11.0% | 1431 / 8.0% | 1476 / 8.5% |
| 列分布 | [271,239,245,258] | [329,408,394,300] | [363,369,383,361] |
| 长音时长 p50 | 687 ms | 466 ms | 474 ms |
| 和弦数 | 42 | 135 | 68 |

三路输出均可玩；mert 和弦更多（其嵌入上下文更鼓励多列同时发声）。

## 3. 结果解读

1. **placement 的瓶颈不是信息，而是先验与数据量。** raw 与 mert 的训练曲线都呈「ep2-3 冲高（0.48-0.55）→ 下滑 → 退火回 0.48-0.50」的过拟合形态，而 v3.1 平台后一路爬到 0.67。更丰富的输入（波形 4096 维/拍、MERT 6144 维/拍 vs 特征 656 维/拍）在 2.9k 曲集上主要增加了记忆容量而非泛化能力。
2. **预训练确实比从零学强**：mert 比 raw 的 placement 高 0.055，且数量比从 2.12 收敛到 1.61（校准更好）——预训练表征把「音乐结构」免费带进来了，但 95M 冻结编码器的通用音乐表征仍不如为「onset→时隙」这个具体任务量身定制的 82 维特征。
3. **selection 的音频依赖很浅，但 mert 仍赢了**：音频上下文只占 selection 输入的一小部分（大头是 16 步组合历史 + 相位/IOI 条件），可 MERT 嵌入承载的和声/段落信息仍带来 +0.7pt 的稳定增益。
4. **混合管线是现成的后续方向**：placement 与 selection 是独立两级，v3-placement（0.678）+ mert-selection（0.436）的「取长组合」在推理上可行（同一估计拍网格下两级单元可比），预期拿到两段各自的最优；本次未实现，代码结构已支持。

## 4. 独立审查与修复

实施过程中由独立审阅者对 14 个文件做了静态审查：**无高严重度问题，4 条中严重度全部修复**：

1. MERT 特征构建与推理的重采样链不一致（训练侧 24k 直解码 vs 推理侧 22.05k→24k 二次重采样）→ 训练侧改为与推理完全相同的解码+重采样链，特征全量重建；
2. raw 嵌入缓存仅按形状校验、重训 placement 后会复用旧缓存 → 缓存目录绑定 ckpt mtime，变更自动清空重建；
3. raw 抗混叠截止频率依赖全曲最长拍、对 librosa 丢拍敏感 → 加 2.5×中位数封顶（训练/推理同规则），受影响的 72 个曲集已重建；
4. 三方评估口径不一致（raw selection 用 bf16、raw placement 分块前向）→ 统一为 fp32 + 整曲前向后重测。

## 5. 变更清单

**新文件**（主仓库 `training/`）：
- `raw_audio.py`（波形→按拍矩阵唯一实现）、`build_dataset_raw.py`、`models_raw.py`、`dataset_raw.py`、`train_placement_raw.py`、`train_selection_raw.py`
- `build_dataset_mert.py`（MERT 特征构建，含 --online/--offline HF 控制）、`models_mert.py`、`dataset_mert.py`、`train_placement_mert.py`、`train_selection_mert.py`
- `eval_ab.py`（三方评估，`python -m training.eval_ab --split test --max-sets 150`）

**改动**：`generate.py`（抽出 `_v3_select_and_emit` 供三代后端共用；新增 raw/mert 分支；`v2_available`/`list_backends`/`resolve_backend` 后端注册表与解析，可用性校验按（权重路径, mtime）缓存，placement+selection 双维度校验；`MUSTAFF_ML_BACKEND` 与显式 backend 参数双通道）；`mustaff/cli.py`（新增 `--ml-backend auto|v3|mert|raw|v2`，生成前回显四后端检测结果）；`mustaff/gui/app.py`（「ML 后端」下拉框 + 重新检测按钮 + 配置持久化 + worker 传 backend）；`README.md` / `README_CN.md`（后端表与用法）。

**产物**：`training/output/{placement,selection}_{raw,mert}_best.pt` + 各自 `*_metrics.json`、`eval_ab_test.json`、训练/构建日志。

**依赖**：`transformers`（新装，5.20.0）；MERT-v1-95M 权重本地缓存（~1.4GB，hf-mirror 下载）。mert 后端**推理时**也需要 transformers + 该权重（构建时已有）；打包发布需一并纳入。

## 6. 复现

```powershell
# 特征构建（各约 40 分钟）
python -m training.build_dataset_raw --workers 8
python -m training.build_dataset_mert            # 首次需 $env:HF_ENDPOINT='https://hf-mirror.com' 下载权重
# 训练（4060 Laptop）
python -m training.train_placement_raw  --epochs 25 --batch 16   # ~30 min
python -m training.train_placement_mert --epochs 25 --batch 16   # ~33 min
python -m training.train_selection_raw  --epochs 15 --batch 16   # ~18 min（含嵌入缓存）
python -m training.train_selection_mert --epochs 15 --batch 16   # ~60 min
# 评估与冒烟
python -m training.eval_ab --split test --max-sets 150
$env:MUSTAFF_ML_BACKEND='mert'; python -m training.verify_gen 6.0
```

## 7. 如果想让「模型直接读音频」真正超过 v3

按预期收益排序：
1. **混合后端**（零训练成本）：v3 placement + mert selection，见 §3.4；
2. **可学习层混合**：MERT 13 层现在是均值池化，改为可学习加权（softmax over layers）通常再涨 1-3pt，需重算特征或在线前向；
3. **微调 MERT 顶层**（LoRA/最后两层解冻）：数据量仍是约束，但配合强增广可能突破 0.60；
4. **扩数据**：collect_sets 直接可扩到 1 万+ 曲集，对所有后端都是正收益，对波形/编码器路线收益更大；
5. 官方星级标签替换 NPS 代理（当前 stars 是 NPS×1.5 的近似，噪声不小）。

## 8. 回退与边界

- 回退：删除 `training/output/*_{raw,mert}_best.pt` 即回到 v3.1 原状；`MUSTAFF_ML_BACKEND=v3` 可随时强制。
- `training/data` junction 曾因数据目录移动失效，已重建指向 `高二\Tools\MustaffData`。
- mert 推理多一次 MERT 前向（~1-2s/曲 GPU）+ librosa 重采样；打包体积 +1.4GB 权重 + transformers。
- 嵌入缓存/特征缓存与对应 ckpt/构建脚本绑定，重训后按提示重建（脚本已内置自动校验）。
