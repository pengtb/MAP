# 03 — 缺失代码的重新实现

## 该计划简述

公开仓库 `MAP/` 缺 6 个文件，且已有代码与发布 checkpoint 的键名口径不一致。本步骤补齐这些实现，使 checkpoint 能以 `strict=True` 加载。

## 已有资源（标明位置）

| 资源 | 位置 | 用途 |
|---|---|---|
| 主 checkpoint | `/data-tangc/models/map_repr/predictor/epoch_3.pt` | 提供 499 个张量作为对齐目标 |
| MAP-KG 编码器 | `/data-tangc/models/map_repr/mapkg_encoder/mapkg_encoder_v3.pt` | 提供 `smiles_encoder.*` 176 键（与 ckpt 完全一致） |
| 词表 | `/data-tangc/models/map_repr/molecule_stm/bart_vocab.txt` | MolSTM 分词器 |
| 现有实现 | `MAP/model/pert.py` | 可复用的组件组装逻辑（cell_projector / gene_tokens_projector / transformer_backbone / project_out / gene_decoder） |
| 组件库 | `MAP/model/components.py`、`gene_decoders.py` | `ResidualProjector` / `CellProjector` / `ProjectOut` / `LatentToGeneDecoder` |

## 已完成部分

- ✅ 反推出 checkpoint 中 `pert_model` 的 260 键结构（7 个子模块）
- ✅ 定位到与原实现的唯一差异：`KGSmilesEncoder_v3` 封装（`smiles_encoder._model.*` + `smiles_projector.*`）在 ckpt 中被拆成平级的 `smiles_encoder`（176）+ `drug_projector`（10）
- ✅ 确认 `PrimeKG_ver1` 分支从未发布（全仓 grep 仅 `demo.py:80`、`train_sciplex.py:118` 两处默认值），实现应落在 `pert_ca.py`
- ✅ 确认 `model.py` 虽 import 了 `PerturbationEncoder_ca`，但实际实例化的是 `PerturbationEncoder`（死引用）

## 待获取资源

无（所需权重与词表均已在本地）。

## 实现记录（已完成）

| # | 文件 | 具体改动 |
|---|---|---|
| ✅ P1 | `MAP/model/pert_ca.py`（新建） | 类 `PerturbationEncoder_ca`：组装顺序同 `pert.py`，但药物分支拆为 `self.smiles_encoder = MolSTM_Extractor(...)`（→176 键）+ `self.drug_projector = ResidualProjector(256→1024)`（→10 键）；`drug_vec = drug_projector(smiles_encoder(smiles))` 且冻结；`smile_encoder` 接受 `{'PrimeKG_ver1','MAP-KG'}`；`forward/encode_cells/encode_genes/encode_drug` 与 `pert.py` 对齐（需通读 `pert.py:75+`） |
| ✅ P2 | `MAP/model/mega_molbart/STMencoder_ddp.py`（仅此一份被引用；`STMencoder.py` 无 import，未改） | ① 硬编码 `ckpt_path` → 环境变量 `MAP_MOLECULE_CKPT`，并支持指向 `mapkg_encoder_v3.pt`（抽取 `smiles_encoder.` 子字典去前缀）；② `vocab_path` → `MAP_BART_VOCAB`；③ `_load_model()` 支持注入 state_dict |
| ✅ P3 | `MAP/configs/se600m.yaml` | `embeddings.current: esm2-cellxgene`；`all_embeddings` → `.../esm2_gene/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt`；清理其余 26 处 `/large_storage`、`/path/to` 占位；`nlayers` 不动（代码硬编码 16） |
| — P4 | 无 | `demo.py` 的 `smile_encoder="PrimeKG_ver1"` 与 ckpt 的 `args` 一致，**不改** |

**明确不改**：`MAP/demo.py`、`MAP/model/pert.py`。

**实际需要改的第三处**：`MAP/model/model.py` 第 56 行把 `PerturbationEncoder(` 换成 `PerturbationEncoder_ca(`（1 行），否则实例化的仍是 `pert.py` 里键名不匹配的类。

**实测补充（写补丁时发现，之前未记录）**：
- `transformer_backbone` 实际是 **4 层**（ckpt `layers.0..3`），与 `ST_small_transformer_backbone_kwargs` 一致；
- `head_dim` 必须设 **128**（ckpt q/k/v/o=(1024,1024)、`rotary_emb.inv_freq=(64,)`），出厂配置的 84 会导致形状不匹配；
- `STMencoder_ddp.py` 的 `torch.load` 需显式 `weights_only=False`（torch 2.8 默认 True，该 ckpt 含 numpy 标量）。

**验收方式**：补丁后用 `epoch_3.pt` 试 `load_state_dict(strict=True)`，要求 499 键全命中（不跑前向）。