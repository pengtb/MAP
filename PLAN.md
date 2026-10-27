# PLAN — MAP 复现计划

> 论文：Feng et al., *A knowledge-driven framework for predicting single-cell responses for unprofiled drugs*, **Nature Machine Intelligence** 8(9):1478-1491 (2026) · DOI `10.1038/s42256-026-01286-w`
> 上游代码：https://github.com/MAGIC-AI4Med/MAP （HEAD `629ebdc`，已 unshallow）
> 本机位置：`/data-tangc/dev/ptb/map-repro`（b200，hostname `s33b01n02`）
> 分支：`repo`（跟踪 `origin/repo`，remote = `git@github.com:pengtb/MAP.git`，deploy key）
> 最后更新：2026-10-27（Drive 权重到位、checkpoint 结构与训练参数核验完成）

---

## 0. 两个阶段

| 阶段 | 名称 | 主要目的 | 依赖官方权重 |
|---|---|---|---|
| 一 | 复现权重 | 用官方 checkpoint 跑通推理并核对 | ✅ 已获得（`/data-tangc/models/map_repr/`） |
| 二 | 复现知识图谱学习 | 从 MAP-KG 数据重训知识编码器 | 否 |

阶段一拆三个子任务：1.1 demo 前向 / 1.2 可复现性与合理性 / 1.3 论文指标级复现。

---

## 1. 已有资产（2026-10-27 实测）

### 1.1 官方权重（`/data-tangc/models/map_repr/`，共 5.4 GB）

| 文件 | 大小 | 结构核验 |
|---|---|---|
| `epoch_3.pt` | 4,227,878,838 B（3.94 GiB） | 6 个顶层键：`epoch / global_step / model_state_dict / optimizer_state_dict / scheduler_state_dict / args`；`epoch=2`、`global_step=155847`；`model_state_dict` 共 **499** 张量 = `pert_model` **260** + `se` **239** |
| `mapkg_encoder_v3.pt` | 1,126,763,160 B（1.05 GiB） | 419 张量：`text_encoder` 200 / `smiles_encoder` 176 / `fusion_module` 12 / `smiles_projector` 10 / `gene_projector` 10 / `text_projector` 10 / `logit_scale` 1 → MAP-KG 预训练编码器 |
| `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` | 410,886,729 B（392 MiB） | dict：**19790** 个基因 → 5120 维（与 `configs/se600m.yaml` 的 `size:5120 / num:19790` 一致） |

**`epoch_3.pt['args']` 记录的原始训练参数**（`args` 是 dict）：

```json
{
  "exp_name": "train",
  "cell_lines": ["CVCL_0023","CVCL_0480","CVCL_0069","CVCL_0131","CVCL_1098","CVCL_1056"],
  "smile_encoder": "PrimeKG_ver1",
  "hvg_info": null,
  "set_size": 24, "UC": false, "batch_size": 1,
  "hvg_loss_weight": 0.1, "epochs": 400, "lr": 1e-05,
  "num_warmup_steps": 10000, "gradient_accumulation_steps": 2,
  "save_every_k_epochs": 3, "eval_every_steps": 500, "de_topk": 50,
  "amp": true, "amp_dtype": "bf16"
}
```

三条推论：
- `smile_encoder = "PrimeKG_ver1"` → 训练时用的就是该分支名，**不是笔误**（`demo.py` 无需改动）；
- `hvg_info = null` → 该 checkpoint **未使用 HVG 信息**，故 1.1/1.2 不需要 2000 HVG 列表；
- 6 个 CVCL 编号即论文 Tahoe 实验的训练/评估细胞系。

### 1.2 代码仓库

- `MAP/`：预测器（demo/train/eval）；`MAP-KG/`：知识图谱预训练；`preprocess/`：A→G 预处理
- 已 unshallow（85+ commits，含 tag `v1.0.0`、`v.1.0.1`）

### 1.3 运行环境

- `.venv`（Python 3.11.14，uv 建）+ `torch 2.8.0+cu128`（`sm_100` 已验证，bf16 matmul 正常）
- 依赖：numpy 2.4.6 / pandas 2.3.3 / scipy 1.17.1 / sklearn 1.9.1 / rdkit 2026.09.1 / omegaconf 2.3.1 / safetensors 0.8.0 / transformers 5.20.2 / scanpy 1.11.5 / anndata 0.12.19 / lightning / requests
- 可导入：`model.se`、`model.pert`、`model.smiles_kg_encoder_v3`、`model.mega_molbart.STMencoder_ddp`、`data.utils`；`model.model` 因缺 `pert_ca.py` 仍失败

---

## 2. 仍缺的东西

| # | 项 | 状态 | 说明 |
|---|---|---|---|
| M1 | `bart_vocab.txt` | ❌ **唯一硬阻塞** | MolSTM BART 词表；`transformer_backbone.embed_tokens=(32000,1024)`、`smiles_encoder._model.emb=(523,256)` 均绑定该词表口径，换词表会静默错位 |
| M2 | `MAP/model/pert_ca.py` | 🔄 缺失但**已反推出口径**（见 §4） | 不是简单 re-export；承载 `PrimeKG_ver1` 分支 |
| M3 | `se600m.safetensors` | ⬜ 待下载 | HF `arcinstitute/SE-600M`（2.66 GiB），走 hf-mirror |
| M4 | `gene_embidx_mapping.torch` | ❓ 大概率不需要 | config 的 `ds_emb_mapping`；demo 路径只读 meta.pkl / control_indices / se_* |
| M5 | 2000 HVG 有序基因列表 | ⚠️ 仅影响 1.3 | 该 ckpt 训练时 `hvg_info=null`，1.1/1.2 不受影响；1.3 的基因级指标仍需，且**不可重生**（`E_hvg_multi_celllines.py` 用未设种子 `random.sample`） |
| M6 | `molecule_model.pth` | ✅ 已解决 | `mapkg_encoder_v3.pt` 的 `smiles_encoder.*` 与 ckpt 内嵌权重**完全一致**（176/176，零形状差异） |
| M7 | `MAP/` 5 个未发布文件 | ❌ 缺 | `ds_nips_lora_se.py`、`ds_sciplex_lora_se.py`、`ds_combosciplex_lora_se.py`、`model_combosciplex_avg_emb.py`、`model_combosciplex_two_tokens.py`；只影响 OP3/SciPlex3/Combo 路径 |

---

## 3. 阶段一：复现权重

### 3.1 demo 前向（最小闭环）

**需要的数据**

| 数据 | 状态 | 位置 |
|---|---|---|
| `epoch_3.pt` | ✅ 已有 | `/data-tangc/models/map_repr/` |
| `mapkg_encoder_v3.pt` | ✅ 已有 | 同上 |
| ESM2 基因嵌入（`all_embeddings`） | ✅ 已有 | 同上 |
| `se600m.safetensors` | ⬜ 需下载 | 同上（HF 取） |
| `bart_vocab.txt` | ❌ 缺（M1） | — |
| CVCL_0023 demo 输入 | ✅ 仓库自带 | `MAP/data/minimal_dataset.zip` |

**实现方法**

1. 按 §4 完成 `pert_ca.py` 等重实现与改动；
2. 下载 `se600m.safetensors`；
3. 拿到 `bart_vocab.txt` 后运行：
   `demo.py --ckpt <epoch_3.pt> --cell_line CVCL_0023 --drug_smiles ... --drug_conc 0.5 --se_ckpt <se600m.safetensors> --se_config configs/se600m.yaml --data_base_dir <minimal_dataset/preprocessed> --se_inputs_base_dir <minimal_dataset/preprocessed_se_inputs_memmap>`

**预期目标**：`strict=True` 加载通过；输出 `[B,S,2000]` 预测谱无 NaN/Inf；记录显存与耗时。

### 3.2 可复现性与生物学合理性

- 固定 seed 重复推理（`demo.py` 用未设种子的 `np.random.choice` 选 control cells → 需固定或重复统计方差）
- 换 cell line / SMILES / 剂量各跑一组，检查单调性与生物学一致性
- 与 `minimal_dataset` 对照（若有 ground truth）做 Pearson / top-50 DEG 方向一致性

### 3.3 论文指标级复现

- 数据：Tahoe-100M（~300 GB，默认不下）或 SciPlex3（GEO）/ OP3（Kaggle）
- 流程：`preprocess/A→G` → `train.py` / `train_sciplex.py` / `train_nips.py` → `evaluate/eval_*.py`
- 受阻：HVG 列表（M5，仅基因级指标）；SciPlex/OP3/Combo 路径缺 5 个代码文件（M7）

---

## 4. 重新实现（本仓库缺失或必须改动的部分）

> 背景：公开仓库 `MAP/` 下缺 6 个文件，且已有代码与发布 checkpoint 的**键名口径不一致**。本节把"要自己写/改什么"固定下来，避免反复试错。

### 4.1 缺口与判定依据

| 缺口 | 判定依据 | 结论 |
|---|---|---|
| `pert_ca.py` 缺失 | `model.py:8` 有 `from model.pert_ca import PerturbationEncoder_ca`，但该文件在 main 与 `v.1.0.1` 都不存在 | 必须自写；该类实际未被 `model.py` 实例化（死引用），**只要 import 成功 + 结构匹配 ckpt 即可** |
| `PrimeKG_ver1` 无实现 | 全仓 grep 只有 `demo.py:80`、`train_sciplex.py:118` 两处默认值；`pert.py:117` 只认 `MAP-KG`，其余分支 `exit()` | 该分支属于 `pert_ca.py`；**`demo.py` 不要改** |
| 键名不一致 | ckpt 中 `pert_model.smiles_encoder.*`（176）+ `pert_model.drug_projector.*`（10），**无 `kg_` 前缀**；`pert.py` 用 `self.kg_smiles_encoder`（封装成 `smiles_encoder._model.*` + `smiles_projector.*`） | 需按 ckpt 的拆分方式实现 |
| `molecule_model.pth` 缺失 | `mapkg_encoder_v3.pt` 的 `smiles_encoder.*` 与 ckpt 的 `pert_model.smiles_encoder.*` **176/176 完全一致，零形状差异** | 不需要原文件，从 `mapkg_encoder_v3.pt` 抽取即可 |

### 4.2 `pert_model` 的 checkpoint 结构（260 键，反推结果）

| 子模块 | 键数 | 形状样例 | 对应现有实现 |
|---|---|---|---|
| `smiles_encoder` | 176 | `_model.emb.weight=(523,256)`、`_model.encoder.layers.{0..3}.*`、`_model.decoder.layers.{0..3}.*` | `MolSTM_Extractor`（= `KGSmilesEncoder_v3.smiles_encoder`） |
| `drug_projector` | 10 | `mlp.0.weight=(512,256)`、`residual_proj.*`、`gate.*` | `ResidualProjector(256→1024)`（= `KGSmilesEncoder_v3.smiles_projector`，仅改名/提级） |
| `transformer_backbone` | 42 | `embed_tokens.weight=(32000,1024)`、`layers.{0..9}.self_attn.q/k/v_proj`、`norm.*` | Llama backbone（`get_transformer_backbone('llama')`） |
| `gene_tokens_projector` | 10 | `mlp.0.weight=(2048,7168)` | `ResidualProjector(2048+5120→1024)` |
| `gene_decoder` | 10 | `decoder.0.weight=(512,2048)` | `LatentToGeneDecoder` |
| `cell_projector` | 8 | `projector.0.weight=(2048,2048)` | `CellProjector(n_layers=4, d_cell=2048, d_out=1024)` |
| `project_out` | 4 | `net.0.weight=(1024,1024)`、`net.3.weight=(2048,1024)` | `ProjectOut` |

**与 `pert.py::PerturbationEncoder` 的唯一结构差异**：

```python
# pert.py（现版本，键名对不上）
self.kg_smiles_encoder = KGSmilesEncoder_v3(d_smiles=256, d_model=1024)
#   └─ smiles_encoder (MolSTM_Extractor, 176)
#   └─ smiles_projector (ResidualProjector, 10)

# pert_ca.py（checkpoint 的真实结构）
self.smiles_encoder = MolSTM_Extractor(...)        # → pert_model.smiles_encoder._model.* (176)
self.drug_projector = ResidualProjector(256, 1024) # → pert_model.drug_projector.*         (10)
```

### 4.3 要写的 `MAP/model/pert_ca.py`

- 类名 `PerturbationEncoder_ca`；
- 组装顺序与 `pert.py` 一致：`cell_projector` → `smiles_encoder` + `drug_projector` → `gene_tokens_projector` → `transformer_backbone` → `project_out` → `gene_decoder`；
- 药物编码：`drug_vec = self.drug_projector(self.smiles_encoder(smiles))`，冻结该路径参数（与 `pert.py` 的 `requires_grad=False` 一致）；
- `smile_encoder` 参数接受 `{'PrimeKG_ver1', 'MAP-KG'}`（两者走同一实现，仅分支名不同）；
- `forward` / `encode_cells` / `encode_genes` / `encode_drug` 的行为需与 `pert.py` 对齐（写作前需通读 `pert.py:75` 之后的实现）。

### 4.4 要改的 `MAP/model/mega_molbart/STMencoder_ddp.py`（及 `STMencoder.py`）

| 现状 | 改动 |
|---|---|
| `ckpt_path = "/mnt/petrelfs/fengjinghao/molecule_model.pth"`（作者机器路径） | 改为环境变量 `MAP_MOLECULE_CKPT`，并允许指向 `mapkg_encoder_v3.pt`（自动抽取 `smiles_encoder.` 子字典并去掉该前缀） |
| `vocab_path = ".../MoleculeSTM/bart_vocab.txt"` | 改为 `MAP_BART_VOCAB` 环境变量 |
| `_load_model()` 无条件 `torch.load(ckpt_path)` | 允许从环境变量注入的 state_dict 构造（走上面同一分支） |

### 4.5 要改的 `MAP/configs/se600m.yaml`

- `embeddings.current: esm2-cellxgene`；
- 该分支的 `all_embeddings` → `/data-tangc/models/map_repr/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt`；
- 其余 26 处 `/large_storage/...`、`/path/to/...` 清空或指向占位；
- 不动 `nlayers`（代码里硬编码 16，与 ckpt 一致）。

### 4.6 明确**不改**的部分

| 文件 | 原因 |
|---|---|
| `MAP/demo.py` | 其 `smile_encoder="PrimeKG_ver1"` 与 ckpt 的 `args` 一致，改了反而错 |
| `MAP/model/pert.py` | 保持上游原样，`pert_ca.py` 独立实现；避免污染对照基线 |
| `MAP/model/model.py` | 只依赖 `pert_ca` 的 import 存在；不需要改逻辑 |

---

## 5. 阶段二：复现知识图谱学习

**数据**：HF `RainGate/MAP-KG` 的 3 个 CSV（98.5 MB）→ `MAP-KG/data/selected_data_csvs/`（注意不是 README 写的 `selected_csvs`）
**代码**：`MAP-KG/` 与 `preprocess/` 的本地 import 均完整（issue #5 核实）
**流程**：改 `train_resume.py` 路径 → `train_resume.sh` → 产物在 `MAP-KG/checkpoints/{exp}`
**产出**：知识编码器 checkpoint，可替换 `mapkg_encoder_v3.pt` 做对照
**预期**：训练收敛；嵌入 UMAP 按 MOA/基因家族聚类

---

## 6. 存储与网络

| 位置 | 容量/现状 | 用途 |
|---|---|---|
| `/data-tangc`（md0 RAID0） | 14 T，剩 ~860 G | 代码、权重（`models/map_repr`）、实验输出 |
| `/data-tangc2`（LVM） | 14 T，剩 ~3.9 T | 大数据 raw/processed |
| `/lustre3/tangc_pkuhpc` | 剩 ~5.7 T | 跨节点共享 |
| `/lustre1` | 账面超配额（实测写入未受阻） | 不推荐 |

网络：b200 出口 **1 GbE**，实测 8–10 MB/s；HF 需 `HF_ENDPOINT=https://hf-mirror.com`；GitHub git 走 ghfast 镜像（push 用 SSH deploy key 直连）。

---

## 7. 已知阻塞汇总

| 阻塞 | 影响 | 解法 |
|---|---|---|
| `bart_vocab.txt` 缺失 | 1.1 完全无法启动（分词器构造即失败） | 向作者索取（issue #5/#6；作者未回复） |
| `pert_ca.py` 缺失 + 键名差异 | 1.1 import 失败 / 加载不匹配 | 按 §4 自写 |
| 2000 HVG 列表缺失 | 1.3 基因级指标不可复现（1.1/1.2 不受影响） | 向作者索取（issue #3/#7） |
| `MAP/` 缺 5 个文件 | OP3/SciPlex3/Combo 路径不可用 | 走 Tahoe 路径，或向作者索取 |
| Drive 不可达（已绕过） | — | 用户已手动下载到 `/data-tangc/models/map_repr/` |

---

## 8. 执行顺序

1. 按 §4 完成重实现（`pert_ca.py`、`STMencoder_ddp.py`、`configs/se600m.yaml`）
2. 下 `se600m.safetensors`（HF）
3. **加载验证**（不跑前向）：用 `epoch_3.pt` 试 `load_state_dict`，确认 499 键全部命中
4. 拿到 `bart_vocab.txt` → 跑 demo（3.1）
5. 3.2 → 3.3
6. 阶段二（无外部阻塞，可并行）