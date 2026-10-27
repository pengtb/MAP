# 07 — 阶段二：MAP-KG 知识预训练

## 该计划简述

用 MAP-KG 的 3 个 CSV 重训知识编码器（对齐分子结构、蛋白序列、文本描述），产出可用于替代 `mapkg_encoder_v3.pt` 的权重，并做知识规模/类型消融（对应论文 Fig_5a/5b）。

## 已有资源（标明位置）

| 资源 | 位置 | 说明 |
|---|---|---|
| 预训练代码 | `MAP-KG/`（`train_resume.py`、`data/dataload.py`、`model/model.py`） | 本地 import 全部可解析（issue #5 核实） |
| 预处理代码 | `preprocess/`（9 个脚本） | 完整 |
| 参考权重（对照） | `/data-tangc/models/map_repr/mapkg_encoder/mapkg_encoder_v3.pt` | 419 张量：`text_encoder` 200 / `smiles_encoder` 176 / `fusion_module` 12 / `smiles_projector` 10 / `gene_projector` 10 / `text_projector` 10 / `logit_scale` 1 |
| 图源数据 | Zotero `Q3ZZX7MA` 的 `Fig_5a` / `Fig_5b` | 知识规模与类型消融的精确数值 |

## 已完成部分

- ✅ 确认 `train_resume.py` 只读 3 个 CSV，路径为 `data/selected_data_csvs/`（**不是** README 写的 `selected_csvs`）
- ✅ 确认发布数据与论文计数存在偏差（总边 −3.44%，issue #5），需作为基线差异记录
- ✅ 确认论文的预训练配置：4×A100、MAP-KG 全量约 25 h 收敛；本机 8×B200 资源更充裕

## 待获取资源

| 资源 | 目标位置 | 来源 | 体量 |
|---|---|---|---|
| `DRUG_merged_drugs_with_residuals.csv` | `MAP-KG/data/selected_data_csvs/` | HF `RainGate/MAP-KG` | 30.9 MB |
| `GENE_tahoe_filtered.csv` | 同上 | 同上 | 0.69 MB |
| `DRUG-GENE_filtered_by_existing_drugs_and_genes.csv` | 同上 | 同上 | 36.2 MB |
| BioBERT（Φ_text） | `/data-tangc/models/map_repr/biobert/` | HF `dmis-lab/biobert-base-cased-v1.1` | 415 MiB |
| ESM-2 650M（Φ_seq） | `/data-tangc/models/map_repr/esm2_protein/` | HF `facebook/esm2_t33_650M_UR50D` | 2.43 GiB |
| 可选：`GENE-GENE_filtered_by_existing_genes.csv` | 同上 | 同上 | 30.7 MB（当前代码未使用） |

## 待完成实现

1. ⬜ 下载 3 个 CSV 与两个编码器权重
2. ⬜ 改 `train_resume.py:27-29` 的 CSV 路径
3. ⬜ 跑 `MAP-KG/train_resume.sh`，产物落 `MAP-KG/checkpoints/{exp}`
4. ⬜ 导出知识嵌入，替换 `configs/se600m.yaml` 的知识侧输入，与官方权重做对照
5. ⬜ 定性检查：药物/基因嵌入 UMAP 是否按 MOA / 基因家族聚类（对照论文 Fig_5 的 DBI / Silhouette 数值）