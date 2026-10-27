# PLAN — MAP 复现计划

> 论文：Feng et al., *A knowledge-driven framework for predicting single-cell responses for unprofiled drugs*, **Nature Machine Intelligence** 8(9):1478-1491 (2026) · DOI `10.1038/s42256-026-01286-w`
> 上游代码：https://github.com/MAGIC-AI4Med/MAP （HEAD `629ebdc`，已 unshallow）
> 本机位置：`/data-tangc/dev/ptb/map-repro`（b200，hostname `s33b01n02`）
> 分支：`repo`（跟踪 `origin/repo`，remote = `git@github.com:pengtb/MAP.git`，deploy key）
> 最后更新：2026-10-27（Drive 权重到位后）

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
| `epoch_3.pt` | 4,227,878,838 B（3.94 GiB） | dict，6 个顶层键：`epoch / global_step / model_state_dict / optimizer_state_dict / scheduler_state_dict / args`；`model_state_dict` 共 **499** 张量：`pert_model.*` **260** + `se.*` **239** |
| `mapkg_encoder_v3.pt` | 1,126,763,160 B（1.05 GiB） | `model_state_dict` 共 **419** 张量：`text_encoder` 200 / `smiles_encoder` 176 / `fusion_module` 12 / `smiles_projector` 10 / `gene_projector` 10 / `text_projector` 10 / `logit_scale` 1 → **就是 `pert.py` 里 `best_model.pt` 所需的 MAP-KG 预训练编码器** |
| `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` | 410,886,729 B（392 MiB） | dict，**19790** 个基因 symbol → 5120 维张量（与 `configs/se600m.yaml` 的 `size: 5120 / num: 19790` 一致） |

### 1.2 代码仓库

- `MAP/`：预测器（demo/train/eval）；`MAP-KG/`：知识图谱预训练；`preprocess/`：A→G 预处理
- 已 unshallow（85+ commits，含 tag `v1.0.0`、`v.1.0.1`）

### 1.3 运行环境

- `.venv`（Python 3.11.14，uv 建）+ `torch 2.8.0+cu128`（`sm_100` 已验证，bf16 matmul 正常）
- 依赖：numpy 2.4.6 / pandas 2.3.3 / scipy 1.17.1 / sklearn 1.9.1 / rdkit 2026.09.1 / omegaconf 2.3.1 / safetensors 0.8.0 / transformers 5.20.2 / scanpy 1.11.5 / anndata 0.12.19 / lightning / requests
- `model.se`、`model.pert`、`data.utils`、`mega_molbart` 均可导入；`model.model` 仍因缺 `pert_ca.py` 无法导入

---

## 2. 仍缺的东西

| # | 项 | 状态 | 说明 |
|---|---|---|---|
| 1 | `bart_vocab.txt` | ❌ **硬阻塞** | MoleculeSTM 分词器词表；checkpoint 的 `smiles_encoder._model.emb.weight` 为 (32000,1024)，**不可用其他 BART 词表替代**。不在 Drive 里 → 需向作者索取（issue #5/#6 已提） |
| 2 | `MAP/model/pert_ca.py` | ❌ 缺，且**不是简单 re-export** | checkpoint 里是 `pert_model.smiles_encoder.*`（176 个键，**无 `kg_` 前缀**），而 `pert.py` 用 `self.kg_smiles_encoder` → 直接 re-export 会导致 `strict=True` 加载失败。需写一个把属性命名为 `smiles_encoder` 的等价类 |
| 3 | `se600m.safetensors` | ⬜ 未下载 | HF `arcinstitute/SE-600M`（2.66 GiB），走 hf-mirror |
| 4 | `gene_embidx_mapping.torch` | ❓ 用途待确认 | `configs/se600m.yaml` 的 `ds_emb_mapping`；demo 路径可能不需要（`load_control_bulk` 只读 meta.pkl / control_indices / se_* 文件） |
| 5 | 2000 HVG 有序基因列表 | ❌ 缺，且不可重生 | `preprocess/E_hvg_multi_celllines.py` 用未设种子的 `random.sample`；影响 1.3 的基因级指标 |
| 6 | `molecule_model.pth` | ⬜ 未获得，但有替代方案 | `KGSmilesEncoder_v3` 构造时 `MolSTM_Extractor(pretrained=True)` 会无条件 `torch.load(ckpt_path)`。而 `mapkg_encoder_v3.pt` 已含完整 `smiles_encoder.*`（176 键）→ 打补丁让 ckpt_path 接受该文件并抽取子字典即可，无需 `molecule_model.pth` |
| 7 | `MAP/` 下 5 个未发布文件 | ❌ 缺 | `ds_nips_lora_se.py`、`ds_sciplex_lora_se.py`、`ds_combosciplex_lora_se.py`、`model_combosciplex_avg_emb.py`、`model_combosciplex_two_tokens.py` → 只影响 OP3/SciPlex3/ComboSciPlex 路径（1.3），Tahoe 路径不受影响 |

---

## 3. 阶段一：复现权重

### 1.1 demo 前向（最小闭环）

**需要的数据**（已具备部分）

| 数据 | 状态 | 位置 |
|---|---|---|
| `epoch_3.pt` | ✅ 已有 | `/data-tangc/models/map_repr/epoch_3.pt` |
| `mapkg_encoder_v3.pt`（KG 分子编码器） | ✅ 已有 | 同上 |
| ESM2 基因嵌入（`all_embeddings`） | ✅ 已有 | 同上 |
| `se600m.safetensors`（SE 骨干） | ⬜ 需从 HF 下 | → `/data-tangc/models/map_repr/` |
| `bart_vocab.txt` | ❌ 缺 | — |
| CVCL_0023 demo 输入 | ✅ 仓库自带 | `MAP/data/minimal_dataset.zip` |

**实现方法**

1. 写 `MAP/model/pert_ca.py`：等价 `PerturbationEncoder`，但内部属性命名为 `smiles_encoder`（匹配 checkpoint 键名），复用 `pert.py` 的其余组件。
2. 下拉 `se600m.safetensors`（hf-mirror）。
3. 改 `configs/se600m.yaml`：`all_embeddings` 指向本地 ESM2 文件；`embeddings.current` 用 `esm2-cellxgene`；其余 26 处外部路径清理。
4. `demo.py` 的 `smile_encoder` 改 `"MAP-KG"`（`pert.py::encode_drug` 只认该值）。
5. 拿到 `bart_vocab.txt` 后跑：
   `demo.py --ckpt <epoch_3.pt> --cell_line CVCL_0023 --drug_smiles ... --drug_conc 0.5 --se_ckpt <se600m.safetensors> --se_config configs/se600m.yaml --data_base_dir <解压的 minimal_dataset/preprocessed> --se_inputs_base_dir <.../preprocessed_se_inputs_memmap>`

**预期目标**：权重 `strict=True` 加载通过；输出 `[B,S,2000]` 预测表达谱无 NaN；记录显存/耗时。

### 1.2 可复现性与生物学合理性

- 固定 seed 重复推理（`demo.py` 用未设种子的 `np.random.choice` 选 control cells → 需固定或重复统计方差）
- 换 cell line / SMILES / 剂量各跑一组，检查单调性与生物学一致性
- 用 `minimal_dataset` 对照（若有 ground truth）做 Pearson / top-50 DEG 方向一致性

### 1.3 论文指标级复现

- 数据：Tahoe-100M（~300 GB，默认不下）或 SciPlex3（GEO）/ OP3（Kaggle）
- 流程：`preprocess/A→G` → `train.py` / `train_sciplex.py` / `train_nips.py` → `evaluate/eval_*.py`
- 受阻：HVG 列表缺失（不可重生）；SciPlex/OP3/Combo 路径缺 5 个代码文件

---

## 4. 阶段二：复现知识图谱学习

**数据**：HF `RainGate/MAP-KG` 的 3 个 CSV（98.5 MB）→ `MAP-KG/data/selected_data_csvs/`（注意不是 README 写的 `selected_csvs`）
**代码**：`MAP-KG/` 与 `preprocess/` 的本地 import 均完整（issue #5 核实）
**流程**：改 `train_resume.py` 路径 → `train_resume.sh` → 产物在 `MAP-KG/checkpoints/{exp}`
**产出**：知识编码器 checkpoint，可替换 `mapkg_encoder_v3.pt` 做对照
**预期**：训练收敛；嵌入 UMAP 按 MOA/基因家族聚类

---

## 5. 存储与网络

| 位置 | 容量/现状 | 用途 |
|---|---|---|
| `/data-tangc`（md0 RAID0） | 14 T，剩 ~860 G | 代码、权重（`models/map_repr`）、实验输出 |
| `/data-tangc2`（LVM） | 14 T，剩 ~3.9 T | 大数据 raw/processed |
| `/lustre3/tangc_pkuhpc` | 剩 ~5.7 T | 跨节点共享 |
| `/lustre1` | 账面超配额（实测写入未受阻） | 不推荐 |

网络：b200 出口 **1 GbE**，实测 8–10 MB/s；HF 需 `HF_ENDPOINT=https://hf-mirror.com`；GitHub git 走 ghfast 镜像（push 用 SSH deploy key 直连）。

---

## 6. 已知阻塞汇总

| 阻塞 | 影响 | 解法 |
|---|---|---|
| `bart_vocab.txt` 缺失 | 1.1 完全无法启动（分词器构造即失败） | 向作者索取（issue #5/#6；作者未回复） |
| `pert_ca.py` 缺失 + 键名 `kg_` 前缀差异 | 1.1 import 失败 / 加载不匹配 | 本地写等价类（属性命名 `smiles_encoder`） |
| 2000 HVG 列表缺失 | 1.3 基因级指标不可复现 | 向作者索取（issue #3/#7） |
| `MAP/` 缺 5 个文件 | OP3/SciPlex3/Combo 路径不可用 | 仅走 Tahoe 路径，或向作者索取 |
| Drive 不可达（已绕过） | — | 用户已手动下载到 `/data-tangc/models/map_repr/` |

---

## 7. 执行顺序

1. 补丁：`pert_ca.py`（属性命名）+ `STMencoder_ddp.py` 路径参数化 + config 本地化 + demo 的 `smile_encoder`
2. 下 `se600m.safetensors`（HF）
3. 试加载 `epoch_3.pt`（先不跑前向，只验证 state_dict 匹配）→ 定位剩余 key 差异
4. 拿到 `bart_vocab.txt` → 跑 demo（1.1）
5. 1.2 → 1.3
6. 阶段二（无外部阻塞，可并行）