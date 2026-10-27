# 02 — 资源获取与落位

## 该计划简述

把复现所需的全部权重、词表、嵌入表与数据集取到集群上，并按"每个 encoder / 基础模型一个子目录"的约定落位。

## 已有资源（标明位置）

| 资源 | 位置 | 大小 | 来源 |
|---|---|---|---|
| 预测器 checkpoint | `/data-tangc/models/map_repr/predictor/epoch_3.pt` | 3.94 GiB | 作者 Google Drive |
| MAP-KG 预训练编码器 | `/data-tangc/models/map_repr/mapkg_encoder/mapkg_encoder_v3.pt` | 1.05 GiB | 作者 Google Drive |
| MoleculeSTM 词表 | `/data-tangc/models/map_repr/molecule_stm/bart_vocab.txt` | 5,495 B | GitHub `chao1224/MoleculeSTM` |
| 基因 ESM-2 嵌入表 | `/data-tangc/models/map_repr/esm2_gene/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` | 392 MiB | 作者 Google Drive |
| demo 最小数据集 | `MAP/data/minimal_dataset.zip`（仓库内） | 162 KB | 上游仓库 |
| 图源数据（数值基准） | Zotero 附件 `Q3ZZX7MA`（`42256_2026_1286_MOESM4_ESM.xlsx`） | 45 KB | Nature MI 补充材料 |
| STATE SE-600M 骨干 | `/data-tangc/models/map_repr/state_se600m/se600m.safetensors` | 2.66 GiB | HF `arcinstitute/SE-600M`（经 hf-mirror） |

## 已完成部分

- ✅ Google Drive 3 个文件已下载并完成结构核验（`epoch_3.pt` 499 张量；`mapkg_encoder_v3.pt` 419 张量；ESM-2 嵌入 19790×5120）
- ✅ 按 encoder / 基础模型重新组织为子目录（见 `plan/README.md` 的目录约定）
- ✅ `bart_vocab.txt` 已解决：523 token、index 272 = `C`、`MolEncTokenizer` 加载 `len=523`，与 ckpt `smiles_encoder._model.emb.weight=(523,256)` 吻合
- ✅ `molecule_model.pth` 不再需要：`mapkg_encoder_v3.pt` 的 `smiles_encoder.*` 与 ckpt 内嵌权重 176/176 完全一致
- ✅ `se600m.safetensors` 已下载并校验：2,860,276,816 B（2.66 GiB），sha256 `3ebec58bd9e9c07f0a76de25b900dad13d015768fbff1e68e5dcbdfb3b2245de` 与 HF LFS etag 完全一致；safetensors 含 238 张量（`transformer_encoder.layers.0..15`，16 层）
- ✅ 网络约束已摸清：HF 用 `hf-mirror.com`；GitHub git 走 ghfast 镜像，push 用 SSH deploy key 直连；下载走 1 GbE（实测 8–10 MB/s）

## 待获取资源

| 资源 | 目标位置 | 来源 | 阻塞程度 |
|---|---|---|---|
| BioBERT（MAP-KG 文本编码器） | `/data-tangc/models/map_repr/biobert/` | HF `dmis-lab/biobert-base-cased-v1.1` | 仅阶段二需要 |
| ESM-2 650M（蛋白序列编码器） | `/data-tangc/models/map_repr/esm2_protein/` | HF `facebook/esm2_t33_650M_UR50D` | 仅阶段二需要 |
| MAP-KG 三个 CSV | `MAP-KG/data/selected_data_csvs/` | HF `RainGate/MAP-KG`（98.5 MB） | 仅阶段二需要 |
| 2000 HVG 有序基因列表 | 待定 | 作者（issue #3/#7 未回） | 阻塞 3.3 基因级指标 |
| `MAP/` 5 个未发布文件 | — | 作者 | 阻塞 OP3/SciPlex3/Combo 路径 |

## 待完成实现

- ⬜ 下载 `se600m.safetensors` 到 `state_se600m/`
- ⬜ 视阶段二需要下载 BioBERT / ESM-2 650M / MAP-KG CSV
- ⬜ 所有资源路径尚未接入代码：`configs/se600m.yaml` 的 `all_embeddings`、`STMencoder_ddp.py` 的 ckpt/vocab 仍是占位路径 → 见 `03-reimplementation.md`