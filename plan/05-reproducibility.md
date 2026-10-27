# 05 — 3.2 可复现性与生物学合理性

## 该计划简述

确认 demo 推理在相同输入下可复现，并检验预测结果随条件的合理变化；建立后续自训练模型的对照基线。

## 已有资源（标明位置）

| 资源 | 位置 | 用途 |
|---|---|---|
| 官方 checkpoint | `/data-tangc/models/map_repr/predictor/epoch_3.pt` | 推理模型 |
| demo 最小数据集 | `MAP/data/minimal_dataset.zip` | 对照组（含 control cells） |
| 图源数据 | Zotero `Q3ZZX7MA`（xlsx，Fig_1~Fig_5b） | 论文精确数值对照基准 |
| ckpt 训练参数 | `epoch_3.pt['args']` | 6 细胞系、`set_size=24`、`de_topk=50`、bf16 等口径 |

## 已完成部分

- ✅ 已确认 `demo.py` 用**未设种子的** `np.random.choice` 选择 control cells → 同输入默认**不可复现**，必须固定 seed 或重复多次统计
- ✅ 已拿到论文的 6 个实验细胞系：`CVCL_0023 / CVCL_0480 / CVCL_0069 / CVCL_0131 / CVCL_1098 / CVCL_1056`
- ✅ 已拿到图源 xlsx 的精确数值，可直接作为对照

## 待获取资源

| 资源 | 说明 |
|---|---|
| 其余 5 个细胞系的 demo 输入 | `minimal_dataset.zip` 只含 `CVCL_0023`；要跨细胞系需自行预处理（依赖原始数据集）或向作者索取 |
| 2000 HVG 有序基因列表 | 仅当需要把输出维度映射回基因名时（3.2 可先不做） |

## 待完成实现

1. ⬜ 固定 seed（或在 demo 里加 `--seed`）后重复推理，验证逐位一致
2. ⬜ 换 `drug_smiles` / `drug_conc` 各跑 2–3 组，检查剂量单调性与方向合理性
3. ⬜ 与 `minimal_dataset` 的对照谱做 Pearson / top-50 DEG 方向一致性（若提供 ground truth）
4. ⬜ 记录一组"基准数值 + 环境"到 `plan/05` 附录或 `results/`，作为阶段二自训模型的对照