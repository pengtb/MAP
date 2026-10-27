# 06 — 3.3 论文指标级复现

## 该计划简述

在公开 benchmark 上重跑训练/评估，对齐论文的 six 指标（top-2000 HVG 与 top-50 DEG 两套基因集上的 Pearson delta、Direction Acc、PDS、MSE、Wasserstein）。

## 已有资源（标明位置）

| 资源 | 位置 | 说明 |
|---|---|---|
| 官方 checkpoint（评估对照） | `/data-tangc/models/map_repr/predictor/epoch_3.pt` | 可直接评估 |
| 图源数据 | Zotero `Q3ZZX7MA`（xlsx） | 每个药物/细胞系/方法的精确数值 |
| 训练脚本 | `MAP/train.py`（Tahoe）、`train_sciplex.py`、`train_nips.py` | 后两者缺数据加载器 |
| 评估脚本 | `MAP/evaluate/eval_tahoe.py`、`eval_multi_cell.py`、`eval_combosciplex.py` | `--de_topk 50` |
| 环境 | `.venv`（torch 2.8.0+cu128） | `flash_attn`/`apex`/`deepspeed` 未装 |

## 已完成部分

- ✅ 论文口径已确认：pseudobulk 分辨率、5 次独立运行的 95% CI、`set_size=24`、`de_topk=50`、bf16
- ✅ 确认 DEG 检验实现用 Welch t-test，而论文写 Wilcoxon+FDR → 指标**不可直接与论文数字对比**，需注明口径
- ✅ 确认为防泄漏，held-out 药物实体需从 MAP-KG 中剔除

## 待获取资源

| 资源 | 目标位置 | 来源 | 影响 |
|---|---|---|---|
| SciPlex3 | `/data-tangc2/raw/sciplex3/` | GEO `GSM4150378` | 建议首选（体量最小） |
| OP3 | `/data-tangc2/raw/op3/` | Kaggle（需 token） | 可选 |
| Tahoe-100M | `/data-tangc2/raw/tahoe100/` | HF `tahoebio/Tahoe-100M`，~300 GB | 1 GbE 下约 10 小时 |
| 2000 HVG 有序基因列表 | 待定 | 作者 | **阻塞**基因级指标（`E_hvg_multi_celllines.py` 用未设种子 `random.sample`，不可重生） |
| `MAP/` 5 个未发布文件 | — | 作者 | **阻塞** OP3/SciPlex3/Combo 训练路径 |

## 待完成实现

1. ⬜ 选定数据集（建议 SciPlex3），下载到 `/data-tangc2/raw/`
2. ⬜ 跑 `preprocess/A→G`（注意 `control_indices` vs `control_indices_UC` 命名差异）
3. ⬜ `train.py`（Tahoe，8 卡 torchrun，bs=1，grad_accum=2）或对应脚本
4. ⬜ `evaluate/eval_*.py` 产出指标，与 xlsx 图源数据逐项对比
5. ⬜ 记录口径差异（t-test vs Wilcoxon）与复现偏差