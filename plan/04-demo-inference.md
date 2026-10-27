# 04 — 3.1 demo 前向（最小闭环）

## 该计划简述

用官方 checkpoint 在 CVCL_0023（A-549）上跑通一次推理，产出 `[B,S,2000]` 预测表达谱，确认权重、词表、预处理输入的链路完整。

## 已有资源（标明位置）

| 资源 | 位置 | 状态 |
|---|---|---|
| 主 checkpoint | `/data-tangc/models/map_repr/predictor/epoch_3.pt` | ✅ |
| MAP-KG 分子编码器 | `/data-tangc/models/map_repr/mapkg_encoder/mapkg_encoder_v3.pt` | ✅ |
| 词表 | `/data-tangc/models/map_repr/molecule_stm/bart_vocab.txt` | ✅ |
| 基因嵌入表 | `/data-tangc/models/map_repr/esm2_gene/Homo_sapiens...ESM2.pt` | ✅ |
| demo 输入（A-549） | `MAP/data/minimal_dataset.zip` → `preprocessed/CVCL_0023/{CVCL_0023_meta.pkl, control_indices.json}`、`preprocessed_se_inputs_memmap/CVCL_0023/{se_shape.json, se_gene_ids.npy, se_expr.npy}` | ✅ 需解压 |
| STATE 骨干 `se600m.safetensors` | `/data-tangc/models/map_repr/state_se600m/` | ✅ 2,860,276,816 B（2.66 GiB），sha256 校验通过，238 张量 / 16 层 |
| 运行环境 | `.venv`（Python 3.11.14 + torch 2.8.0+cu128） | ✅ |

## 已完成部分

- ✅ `demo.py` 的必需参数已核实：只有 `--ckpt` 必填，其余（cell_line / drug_smiles / drug_conc / set_size / data_base_dir / se_inputs_base_dir / se_config / se_ckpt / output_dir）均可指定
- ✅ 输入格式核实：`load_control_bulk` 只读 `{cl}_meta.pkl`、`control_indices.json`、`se_shape.json`、`se_gene_ids.npy`、`se_expr.npy`
- ✅ 确认 `hvg_info` 可为 `None`（ckpt 训练时即 `hvg_info=null`），因此 3.1 不需要 HVG 列表
- ✅ 确认 `gene_embidx_mapping.torch` / `valid_genes_masks` 在 `.py` 中零引用 → 不需要

- ✅ **2026-10-27 首次跑通**（`demo.py`，rc=0）：输出 `[1,24,2048]` 预测嵌入 + `[1,24,2000]` 预测表达
- ✅ 额外发现并修复一处上游遗漏：`model.py::forward` 读取 `self.pert_model_hvg_ids_already_set`，但 `__init__` 从未初始化（`hvg_info=None` 时仍会先求值左操作数）→ 已在 `__init__` 补 `False` 与 `hvg_ids=None`
- ✅ 输出校验：无 NaN/Inf；`pred_hvg` 范围 0–2.96（均值 0.13，分位数 0/0/0/0.128/1.5/2.96，符合稀疏表达谱）；`pred_emb` 单细胞 L2≈0.44 且各细胞不同
- ✅ 计时实测：**wall 20.05 s**（含两次权重加载）、**峰值显存 12,732 MiB**、**maxrss 8.15 GiB**

## 待获取资源

无。`se600m.safetensors` 已下载并通过 sha256 校验（见「已有资源」）。

## 已完成实现

1. ✅ 解压 `MAP/data/minimal_dataset.zip` → `/data-tangc/dev/ptb/map-repro/work/minimal_dataset/`
2. ✅ 完成 `03-reimplementation.md` 的 P1/P1′/P2/P3 补丁 + 本次的 `model.py` 标志位修复
3. ✅ 下载并校验 `se600m.safetensors`
4. ✅ 实际运行命令（GPU0，其余卡被他人占用）：

```bash
cd /data-tangc/dev/ptb/map-repro/MAP
export MAP_MOLECULE_CKPT=/data-tangc/models/map_repr/mapkg_encoder/mapkg_encoder_v3.pt
export MAP_BART_VOCAB=/data-tangc/models/map_repr/molecule_stm/bart_vocab.txt
export CUDA_VISIBLE_DEVICES=0
../.venv/bin/python demo.py \
  --ckpt /data-tangc/models/map_repr/predictor/epoch_3.pt \
  --cell_line CVCL_0023 --drug_smiles "CC1=NC=C(C(=C1O)CO)CO" \
  --drug_conc 0.5 --set_size 24 \
  --se_ckpt /data-tangc/models/map_repr/state_se600m/se600m.safetensors \
  --se_config configs/se600m.yaml \
  --data_base_dir ../work/minimal_dataset/preprocessed \
  --se_inputs_base_dir ../work/minimal_dataset/preprocessed_se_inputs_memmap \
  --output_dir ../work/demo_output
```

5. ✅ 验收结果：

| 项 | 结果 |
|---|---|
| 权重加载 | `strict=True` → `<All keys matched successfully>`（499/499） |
| MolSTM 编码器 | 从 `mapkg_encoder_v3.pt` 抽取 `smiles_encoder._model.*`（176 张量），Missing/Unexpected 均为空 |
| SMILES 规范化 | `CC1=NC=C(C(=C1O)CO)CO` → `Cc1ncc(CO)c(CO)c1O` |
| 输出形状 | `input_ctrl_gene_ids`/`input_ctrl_expressions`/`output_pred_embeddings` 均 `(1,24,2048)`；`output_pred_hvg` `(1,24,2000)` |
| 数值 | 无 NaN/Inf；`pred_hvg` 0–2.96（均值 0.13）；`pred_emb` L2≈0.44 |
| 耗时/资源 | wall **20.05 s**、峰值显存 **12.7 GiB**、maxrss **8.15 GiB**、GPU 0 |

产物：`work/demo_output/{input_ctrl_gene_ids,input_ctrl_expressions,output_pred_embeddings,output_pred_hvg}.npy` + `meta.json`
