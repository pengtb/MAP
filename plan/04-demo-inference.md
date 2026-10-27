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

## 待获取资源

无。`se600m.safetensors` 已下载并通过 sha256 校验（见「已有资源」）。

## 待完成实现

1. ⬜ 解压 `MAP/data/minimal_dataset.zip` 到工作目录
2. ⬜ 完成 `03-reimplementation.md` 的 P1/P2/P3 补丁
3. ✅ 已下载 `se600m.safetensors`（sha256 与 HF 元数据一致）
4. ⬜ 运行：
   ```
   python demo.py --ckpt .../predictor/epoch_3.pt --cell_line CVCL_0023 \
     --drug_smiles "CC1=NC=C(C(=C1O)CO)CO" --drug_conc 0.5 --set_size 24 \
     --se_ckpt .../state_se600m/se600m.safetensors --se_config configs/se600m.yaml \
     --data_base_dir <解压>/preprocessed \
     --se_inputs_base_dir <解压>/preprocessed_se_inputs_memmap \
     --output_dir ./demo_output
   ```
5. ⬜ 验收：`strict=True` 加载无 missing/unexpected；输出形状 `[1,24,2000]`；无 NaN/Inf；记录显存峰值与耗时