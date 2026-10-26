# PLAN — MAP 复现计划

> 论文：Feng et al., *A knowledge-driven framework for predicting single-cell responses for unprofiled drugs*, **Nature Machine Intelligence** 8(9):1478-1491 (2026) · DOI `10.1038/s42256-026-01286-w`
> 上游代码：https://github.com/MAGIC-AI4Med/MAP （本地镜像克隆，HEAD `629ebdc`）
> 本机位置：`/data-tangc/dev/ptb/map-repro`（b200，hostname `s33b01n02`）
> 分支：`repo`（跟踪 `origin/repo`，remote 走镜像 `ghfast.top`）
> 提交身份：`pengtb <ptbjj@126.com>`（仓库本地配置）

---

## 0. 两个阶段

| 阶段 | 名称 | 主要目的 | 依赖官方权重 | 预估数据量 |
|---|---|---|---|---|
| 一 | 复现权重（验证预训练产物可用） | 用官方 checkpoint 跑通推理并核对指标 | 是（仅 Google Drive 提供，当前不可达） | ~8-11 GB（不含 benchmark） |
| 二 | 复现知识图谱学习（MAP-KG 对比预训练） | 从 MAP-KG 数据重训知识编码器 | 否 | ~200 MB + 骨干权重 |

阶段一按"由易到难"拆成三个子任务（1.1 / 1.2 / 1.3），1.3 才需要真正下载 benchmark 原始数据。

---

## 1. 数据总清单

存放位置约定：
- `R = /data-tangc/dev/ptb/map-repro`（代码 + 小数据 + 输出）
- `W = /data-tangc/models/map`（权重，符合 `/data-tangc/readme.md` 的 models/ 约定）
- `D = /data-tangc2/map-repro`（大数据原始与预处理产物；该盘剩 ~3.9 T）

| # | 数据 | 大小 | 来源 | 是否下载 | 存放 |
|---|---|---|---|---|---|
| 1 | MAP 官方权重（知识编码器 + 预测器 ckpt） | 未知（Google Drive 无法枚举） | README 给出的 Drive 文件夹 | **必须**（阻塞项） | `W/official/` |
| 2 | STATE SE-600M `model.safetensors` | 2.66 GiB | HF `arcinstitute/SE-600M` | 是 | `W/state/` |
| 3 | STATE `se600m_epoch*.ckpt`（各 10.76 GiB，共 2 个） | 21.5 GiB | 同上 | **暂不下载**（仅当代码强制要 .ckpt） | — |
| 4 | STATE `protein_embeddings.pt` | 0.38 GiB | 同上 | 视需要（`preprocess` 可能要） | `W/state/` |
| 5 | BioBERT `biobert-base-cased-v1.1` | 415 MiB | HF `dmis-lab/biobert-base-cased-v1.1` | 是（知识编码器文本分支） | `W/biobert/` |
| 6 | ESM-2 `esm2_t33_650M_UR50D` | 2.43 GiB | HF `facebook/esm2_t33_650M_UR50D` | 是（蛋白序列分支） | `W/esm2/` |
| 7 | 分子编码器（MegaMolBART，代码目录 `mega_molbart`；论文正文写作 MoleculeSTM） | ~0.3-0.6 GB（估） | 待定（NVIDIA NGC / 论文 Drive） | 是（分子分支） | `W/mol/` |
| 8 | ESM2 基因嵌入表 `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` | 未测 | `configs/se600m.yaml` 引用，**仓库不含** | **待确认来源** | `W/emb/` |
| 9 | `gene_embidx_mapping.torch` / `valid_gene_index_*.torch` | 未测 | 同上，`configs/se600m.yaml` 引用，**仓库不含** | **待确认来源** | `W/emb/` |
| 10 | MAP-KG 预处理 CSV × 3（dataload 实际只读 3 个） | 98.5 MB | HF `RainGate/MAP-KG` | 是（阶段二必需） | `R/MAP-KG/data/selected_data_csvs/` |
| 11 | MAP-KG 第 4 个文件 `GENE-GENE_filtered_by_existing_genes.csv` | 30.7 MB | 同上 | 可选（当前代码未使用） | 同上 |
| 12 | demo 最小数据集 `minimal_dataset.zip` | 162 KB | 仓库自带 `MAP/data/` | 已具备 | `R/MAP/data/` |
| 13 | SciPlex3（GEO `GSM4150378`） | 未测 | NCBI GEO | 仅 1.3 需要，**集群可达性未测** | `D/raw/sciplex3/` |
| 14 | ComboSciPlex（GEO `GSE206741`） | 未测 | NCBI GEO | 仅 1.3 可选 | `D/raw/combosciplex/` |
| 15 | OP3（NeurIPS 竞赛） | 未测 | Kaggle（需账号 token） | 仅 1.3 可选 | `D/raw/op3/` |
| 16 | Tahoe-100M | ~300 GB（3388 个 parquet 分片） | HF `tahoebio/Tahoe-100M` | 仅 1.3 可选，**默认不下** | `D/raw/tahoe100/` |
| 17 | Supplementary Table 21 + 14 个原始知识源 | 未测 | Nature 补充材料 / 各库（部分需许可） | **不在本计划范围**（仅"从零重建图谱"才需要） | — |

> 注意 #10 的路径陷阱：README 写 `MAP-KG/data/selected_csvs/`，而 `MAP-KG/train_resume.py` 实际读 `data/selected_data_csvs/`。以代码为准。

---

## 2. 阶段一：复现权重

### 1.1 权重装载 + demo 前向（最小闭环）

**需要的数据**

| 数据 | 来源 | 下载 | 存放 |
|---|---|---|---|
| MAP 官方 `*.pt`（含 `model_state_dict`） | Google Drive（README） | 是（阻塞） | `W/official/` |
| `se600m.safetensors` | HF `arcinstitute/SE-600M` | 是 | `W/state/` |
| `configs/se600m.yaml`（需改本地路径） | 仓库自带 | 否 | `R/MAP/configs/` |
| ESM2 基因嵌入 + 映射表（#8/#9） | 待确认 | 待定 | `W/emb/` |
| 单个 cell line 的预处理输入（`{cl}_meta.pkl`、`control_indices.json`、`*_embeddings.npy`、`*_hvg.npy`） | 见下方说明 | 待定 | `D/processed/` |
| SE 输入 memmap（`se_shape.json`、`se_gene_ids.npy`、`se_expr.npy`） | `preprocess/G_extract_se_input_for_lora.py` 产出 | 视情况 | `D/se_inputs/` |

> 关键发现：`demo.py` 不能只靠权重跑。它同时要求 `--data_base_dir`（按 cell line 的 memmap + meta）与 `--se_inputs_base_dir`（SE memmap），这些是 `preprocess/F`、`preprocess/G` 的产物。因此 1.1 有两条路：
> - **A（省事）**：官方 Drive 目录里若附带预处理资产，直接下载 → 1.1 无需任何 benchmark 原始数据；
> - **B（自造）**：只下含目标 cell line（demo 默认 `CVCL_0023` = A-549）的那部分数据，跑 `preprocess/A→G` 生成输入。SciPlex3 含 A549，是 1.3 之前成本最低的选择。

**实现方法（概述）**

1. 建 conda 环境（Python 3.8 / torch 2.1.2 / scanpy 1.9.8 / `requirements.txt`；b200 无 `module`，用 conda 直装）。
2. 把 `MAP/configs/se600m.yaml` 里 27 处 `/large_storage/...`、`/path/to/...` 全部改成本地路径，并把 `embeddings.current` 设成实际可用的一项。
3. 跑：
   `python MAP/demo.py --ckpt $W/official/<ckpt>.pt --cell_line CVCL_0023 --drug_smiles "<smiles>" --drug_conc 0.5 --se_ckpt $W/state/model.safetensors --se_config MAP/configs/se600m.yaml --data_base_dir ... --se_inputs_base_dir ... --output_dir ./demo_output`
4. 记录：`strict=True` 是否通过（`load_state_dict` 严格匹配）、显存峰值、单次推理耗时、输出文件内容。

**预期目标**

- 权重严格加载成功（无 missing / unexpected keys）；
- 产出 `demo_output/` 下的预测表达谱，无 NaN/Inf，数值范围与 log1p 归一化一致；
- 固化一份"环境 + 路径 + 命令"的记录，保证换机可重复。

---

### 1.2 单条件推理的可复现性与生物学合理性

**需要的数据**

与 1.1 相同，另加：

| 数据 | 来源 | 下载 | 存放 |
|---|---|---|---|
| `minimal_dataset.zip`（示例/对照） | 仓库自带 | 否 | `R/MAP/data/` |
| 2-3 个额外 cell line / 药物 SMILES / 剂量组合 | 公开信息（SMILES 可变体自选） | 否 | 记录在 `logs/` |

**实现方法（概述）**

1. 固定随机种子（注意 `demo.py` 用 `np.random.choice` 选 control cells，**默认无 seed**，需固定或重复 5 次统计方差）。
2. 同输入跑多次 → 检查输出是否确定性一致（或不一致时量化差异）。
3. 换 `--cell_line` / `--drug_smiles` / `--drug_conc` 各跑一组 → 观察预测谱是否随剂量单调变化、是否随细胞系变化。
4. 与 `minimal_dataset` 提供（若有）的对照谱做 Pearson / top-50 DEG 方向一致性比较。

**预期目标**

- 同输入结果可复现（固定 seed 后逐位一致）；
- 不同剂量/细胞系产生可解释差异（非随机噪声）；
- 记录一组"基准数值 + 运行环境"，作为后续自训练模型的对照基线。

---

### 1.3 论文指标级复现（需 benchmark 数据）

**需要的数据**

| 数据 | 来源 | 下载 | 存放 |
|---|---|---|---|
| SciPlex3（最小、建议首选） | GEO `GSM4150378` | 待测可达性 | `D/raw/sciplex3/` |
| OP3 | Kaggle（需 token） | 待测 | `D/raw/op3/` |
| Tahoe-100M（默认不下） | HF `tahoebio/Tahoe-100M` | 可选 | `D/raw/tahoe100/` |
| 预处理产物 + SE 输入 | `preprocess/A→G` 生成 | 本地生成 | `D/processed/`、`D/se_inputs/` |
| 官方权重或自训练 ckpt | Drive / 阶段二产物 | 视情况 | `W/official/` 或 `R/checkpoints/` |

**实现方法（概述）**

1. 按 README：把数据解压到预处理目录，**按字母序**执行 `preprocess/`（A_fetch_cell_line → B_watch → C_uni_multi_process → Cs_split → D_extract_se_emb → E_hvg → F_gemini_opt → G/G2 提取 SE 输入）。
2. 训练/评估：
   - SciPlex3：`MAP/train_sciplex.py`
   - OP3：`MAP/train_nips.py`
   - Tahoe：`MAP/train.py`（`torchrun --nproc_per_node=8`，bs=1，set_size=24，100 epoch，lr=1e-5，bf16）
   - 评估：`MAP/evaluate/eval_multi_cell.py`、`eval_tahoe.py`、`eval_combosciplex.py`（参数：`--data_base_dir --se_inputs_base_dir --se_ckpt --checkpoint --de_topk 50`）
3. 对齐论文指标口径：top-50 DEG 的 Pearson delta、方向准确率、扰动判别分数、MSE、Wasserstein（pseudobulk 分辨率、top-2000 HVG 与 top-50 DEG 两套基因集）。

**预期目标**

- 复现论文报告量级（示例：unseen drug 设定下 top-50 DEG Pearson delta 相对最强基线 +11.8%；同细胞系 A-172 上 ≥0.95 等）；
- 产出可复现的评估脚本与结果表（`logs/` + 一份 `RESULTS.md`）。

---

## 3. 阶段二：复现知识图谱学习（MAP-KG 对比预训练）

**需要的数据**

| 数据 | 来源 | 下载 | 存放 |
|---|---|---|---|
| `DRUG_merged_drugs_with_residuals.csv`（30.9 MB） | HF `RainGate/MAP-KG` | 是 | `R/MAP-KG/data/selected_data_csvs/` |
| `GENE_tahoe_filtered.csv`（0.69 MB） | 同上 | 是 | 同上 |
| `DRUG-GENE_filtered_by_existing_drugs_and_genes.csv`（36.2 MB） | 同上 | 是 | 同上 |
| `GENE-GENE_filtered_by_existing_genes.csv`（30.7 MB） | 同上 | 可选 | 同上 |
| BioBERT / ESM-2 / 分子编码器权重 | HF / NGC | 是 | `W/biobert`、`W/esm2`、`W/mol` |

**实现方法（概述）**

1. 下载 3 个 CSV（走 hf-mirror），放到 `MAP-KG/data/selected_data_csvs/`（注意与 README 的 `selected_csvs` 不同）。
2. 检查 `MAP-KG/train_resume.py` 顶部的路径（第 27-29 行）与超参，改成实际路径。
3. 运行 `MAP-KG/train_resume.sh`（= `python -u train_resume.py > logs/train_resume.txt`），产物落 `MAP-KG/checkpoints/{exp}`，每 10 epoch 存一次。
4. 验证：把学到的 KG 嵌入导出为 `KGemb.pt`，接到 `configs/se600m.yaml` 的 `embeddings.gene-kg-bert.all_embeddings`（size 1024 / num 16476），用于后续阶段一模型的替代输入。
5. 复用论文的定性检查：对药物/基因嵌入做 UMAP，看是否按 MOA / 基因家族聚类（DBI 下降、silhouette 上升）。

**预期目标**

- 训练收敛（train loss 稳定下降，epoch 间 checkpoint 可加载）；
- 产出可用的实体嵌入（药物、基因各一套），能替代官方知识编码器参与阶段一推理；
- 形成"官方权重 vs 自训权重"的对照（阶段一 1.2 的基线可复用）。

---

## 4. 环境与依赖

| 项 | 值 / 说明 |
|---|---|
| 硬件 | b200：8 × NVIDIA B200（179 GiB/卡）、288 核、2.2 TiB 内存、驱动 595.91.07 |
| Python | 3.8（README）；torch 2.1.2；scanpy 1.9.8 |
| 依赖清单 | `requirements.txt`（pip freeze，含大量无关包，建议按需精简） |
| 加速 | `train.sh` 使用 bf16 + gradient accumulation 2 + 8 卡 torchrun |
| 注意 | b200 **无 `module` 命令**；conda 用 `~/miniconda3` 或 `/lustre3/tangc_pkuhpc/conda` |
| 环境位置 | 建议 `R/env`（项目内独立 venv/conda env），避免污染他人环境 |

---

## 5. 网络与下载方式

| 目标 | 状态 | 方式 |
|---|---|---|
| HuggingFace | 官方域名被 DNS 污染，镜像可用 | `export HF_ENDPOINT=https://hf-mirror.com` + `huggingface-cli download` |
| GitHub | 直连 git 端点超时；镜像可用 | 全局 `url.<ghfast>.insteadOf` 已生效；本仓库 remote 直接写镜像地址 |
| Google Drive（官方权重） | **不通** | 需用官方网页工作台（`https://10.100.1.89` → 文件传输）上传，或请作者换源 |
| Kaggle（OP3） | 未测 | 需注册 + API token |
| NCBI GEO（SciPlex3/ComboSciPlex） | 未测 | `curl`/`wget`（建议加 `-4`） |
| Nature 补充材料 | 未测 | 浏览器/`curl` |

通用注意：**所有下载加 `-4`**（两节点无 IPv6 默认路由，DNS 会返 AAAA 导致卡十几秒）。

---

## 6. 存储与配额

| 位置 | 容量 | 现状 | 用途 |
|---|---|---|---|
| `/data-tangc`（b200 本地 md0） | 14 T | 94% 用，剩 868 G | 代码 `dev/ptb/map-repro`、权重 `models/map`、实验输出 |
| `/data-tangc2`（b200 本地 LVM） | 14 T | 73% 用，剩 3.9 T | 原始大数据 `raw/`、预处理 `processed/`、`se_inputs/` |
| `/vllm` | 14 T | 81% 用 | **不用**：`/vllm/tangc_pkuhpc` 目录读时报 I/O error（待管理员修） |
| `/lustre1` | 2.5 P | 账面超配额 1.19 T（实测写入未受阻） | **不推荐** |
| `/lustre3` | 3.0 P | 剩 5.7 T | 跨节点共享备选（代码/结果备份） |

预估占用：阶段二 ~200 MB + 骨干 6-9 GB；阶段一 1.1/1.2 ~8-11 GB；1.3 视数据集（SciPlex3 数十 GB，Tahoe ~300 GB）。

---

## 7. 已知阻塞与待确认

| # | 事项 | 影响 | 需要谁 |
|---|---|---|---|
| B1 | MAP 官方权重仅 Google Drive，集群不可达 | 阶段一 1.1 无法开始 | 用户（网页工作台上传）或作者换源 |
| B2 | `configs/se600m.yaml` 引用的 ESM2 基因嵌入 / 映射表来源不明（仓库不含） | 阶段一、二都可能卡住 | 查 Drive 是否附带；或从 STATE 官方资产推导 |
| T1 | STATE 是否需要 `.ckpt`（10.76 GiB）而非 `.safetensors` | 影响下载量（+21.5 GiB） | 读 `MAP/model/se.py` 确认 |
| T2 | 分子编码器型号：论文写 MoleculeSTM，代码是 `mega_molbart` | 影响权重选择 | 读 `MAP-KG/model/model.py` 与 `MAP/model/smiles_kg_encoder_v3.py` |
| T3 | GEO / Kaggle / Nature 集群可达性 | 决定 1.3 能否进行 | 实测 |
| T4 | 补充材料（Table 21、消融细节）不在本地 PDF | 只影响"从零重建图谱" | 用户下载 SI |

---

## 8. 执行顺序

1. **环境**：建 conda env，跑通 `import torch` + GPU 可见性（无数据依赖）。
2. **阶段二（可并行、无阻塞）**：下 3 个 CSV → 改路径 → `train_resume.sh` → 产 KG 嵌入。
3. **阶段一 1.1**：等 B1/B2 解决后跑 demo（若 Drive 附带预处理资产，则与阶段二同步下载）。
4. **阶段一 1.2**：固定 seed 做可复现性 + 生物学合理性检查。
5. **阶段一 1.3**：按需选一套 benchmark（建议 SciPlex3），走 `preprocess/A→G` + 训练 + 评估。

---

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-26 | 建仓、改 remotes（`origin`=镜像 fork，`upstream`=镜像上游）、建 `repo` 分支并预建跟踪、设仓库级提交身份 |
| 2026-10-26 | 建立本计划文件 |
