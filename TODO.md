# TODO — MAP 复现任务清单

> 位置：`/data-tangc/dev/ptb/map-repro/TODO.md`
> 关联：`PLAN.md`（总体计划）；本文只记"已完成"与"待处理/风险"
> 最后更新：2026-10-26

图例：✅ 完成 ｜ 🔄 进行中 ｜ ⬜ 待做 ｜ ⚠️ 风险 ｜ ❌ 阻塞（等外部输入）

---

## 1. 已完成

### 1.1 仓库与 git

- ✅ 克隆仓库到 `/data-tangc/dev/ptb/map-repro`（HEAD `629ebdc`，`--depth 1`）
- ✅ 建分支 `repo`，并预建跟踪关系 `branch.repo.remote=origin`
- ✅ remote：`origin` = `https://ghfast.top/https://github.com/pengtb/MAP.git`（fetch/push），`upstream` = 上游镜像
- ✅ 仓库级提交身份：`pengtb <ptbjj@126.com>`
- ✅ 删除本地 `insteadOf` 规则（改为 origin 直接写镜像地址）
- ✅ 已写 `PLAN.md`

### 1.2 运行环境

- ✅ 用 uv（0.9.26）建项目 venv：`/data-tangc/dev/ptb/map-repro/.venv`，**Python 3.11.14**
- ✅ `torch 2.8.0+cu128` + `torchvision 0.23.0+cu128`
- ✅ GPU 真机验证：
  ```
  cap       (10, 0)                                  # B200 / sm_100
  archs     ['sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
  matmul    8192^2 bf16 @ 8192^2 → ok，0.181s，6.1 TFLOP/s（单次冷启动，非 benchmark）
  ```
- ✅ 已装依赖（版本）：numpy 2.4.6 / pandas 2.3.3 / scipy 1.17.1 / scikit-learn 1.9.1 / rdkit 2026.09.1 / omegaconf 2.3.1 / safetensors 0.8.0 / transformers 5.20.2 / scanpy 1.11.5 / anndata 0.12.19 / lightning + torchmetrics / requests 2.34.2
- ✅ 导入测试：`model.se` / `model.pert` / `model.smiles_kg_encoder_v3` / `model.mega_molbart.STMencoder_ddp` / `data.utils` 均可导入
- ❌ `model.model` / `demo` 仍无法导入 → 缺 `model/pert_ca.py`（见 §3）

### 1.3 调研结论（重要）

- ✅ GitHub issues 已查：缺文件问题被 **3 次**报告（#4 2026-09-10、#5 2026-09-15、#6 2026-09-21），**均 OPEN 且作者未回复**
- ✅ 作者仅在更早的 #1、#2 回复过：#1 → ESM2 嵌入文件在 Drive；#2 → 已提供 `MAP/data/minimal_dataset.zip`，并说明**先用 `epoch_3.pt` 作为初步 checkpoint**
- ✅ **`all_embeddings` 可不用 Drive**：issue #5 核实 `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt`（410,886,729 B）与 HF `arcinstitute/SE-600M` 的 `protein_embeddings.pt` **字节相同**（SHA256 `a210e1cc…7de9748`）
- ✅ **阶段二可用**：issue #5 核实 `MAP-KG/` 与 `preprocess/` 的所有本地 import 均可解析（缺文件只集中在 `MAP/`）
- ✅ 带宽实测：下载 8–10 MB/s（pytorch/anaconda/tuna/hf-mirror 均 ~8.4–10.3 MB/s）
- ✅ 网络约束：HF 官方域名被污染（用 `hf-mirror.com`）；GitHub git 直连不通（走 ghfast 镜像）；Google Drive 不可达（需人工下载）

---

## 2. 等用户从 Google Drive 取回的文件

> README 里的 Drive 链接：`https://drive.google.com/drive/folders/1cV0ZTk92PguKS2nyii6dLV0IfqoSDHsQ`

| 文件 | 必要性 | 备注 |
|---|---|---|
| `epoch_3.pt` | ❌ 必须 | 主模型 checkpoint（作者指定为"初步 checkpoint"） |
| `se600m.safetensors` | 可自取 | 也可直接从 HF `arcinstitute/SE-600M` 下（2.66 GiB） |
| `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` | 可自取 | 用 HF 的 `protein_embeddings.pt` 替代（SHA256 一致） |
| `gene_embidx_mapping.torch` | ❓ 确认是否在 Drive | `configs/se600m.yaml` 引用，仓库不含 |
| **`bart_vocab.txt`** | ❌ 关键阻塞 | MoleculeSTM 分词器；checkpoint `emb.weight=(32000,1024)`，**不可替代** |
| `molecule_model.pth` | 可能不需要 | issue #5：smiles 编码器权重已在 `epoch_3.pt` 内 |
| `hvg_gene_ids` / `top2000_hvg_info_seurat_v3_batch_aware.pkl` | 影响 1.3 | 2000 HVG 有序列表（无法重生，见 §4） |

取回后放到：`/data-tangc/models/map/official/`（权重）与 `/data-tangc/models/map/emb/`（嵌入/映射）。

---

## 3. 需要在本地打的补丁（代码/配置）

| # | 位置 | 要做什么 | 何时做 |
|---|---|---|---|
| P1 | `MAP/model/pert_ca.py`（新建） | 先加一个 re-export shim：`from model.pert import PerturbationEncoder as PerturbationEncoder_ca`；再用 `epoch_3.pt` 验证键是否匹配（issue #5 判断是"漏传"） | 拿到 ckpt 后 |
| P2 | `MAP/model/mega_molbart/STMencoder_ddp.py:16-17`（还有 `STMencoder.py`） | 两个硬编码作者路径改成环境变量/参数：`MAP_MOLECULE_CKPT`、`MAP_BART_VOCAB` | 立即（无依赖） |
| P3 | `MAP/configs/se600m.yaml` | 27 处 `/large_storage/...`、`/path/to/...` 改为本地路径；`embeddings.current` 指向本地 `all_embeddings` | 立即 |
| P4 | `MAP/demo.py:80` | `smile_encoder="PrimeKG_ver1"` → `"MAP-KG"`（`pert.py` 的 `encode_drug` 只认 `MAP-KG`，否则直接 `exit()`） | 拿到 ckpt 后 |
| P5 | `preprocess/Cs_split_unseen_combination.py` 输出 | 产出 `control_indices_UC.json`，而 demo 读 `control_indices.json` → 需 copy/改名（仅走预处理时需要） | 1.3 时 |
| P6 | `MAP-KG/train_resume.py:27-29` | CSV 路径指向 `data/selected_data_csvs/`（注意不是 README 写的 `selected_csvs`） | 阶段二开工时 |

---

## 4. 风险登记（可能出问题的地方）

| # | 风险 | 影响 | 处置建议 |
|---|---|---|---|
| R1 | `transformers 5.20.2` ≫ 代码写时的 4.30 | 5.x 有 API 破坏（如 LLaMA 相关）；可能 import/forward 报错 | 报错即降级到 `transformers==4.46.3` 一类 4.x 末版 |
| R2 | `numpy 2.4.6` ≫ requirements 的 1.24.4 | 老代码若用 `np.float_`/`np.bool8` 会在运行时崩 | 按报错逐个替换（不改库版本） |
| R3 | torch 2.8 vs 论文 2.1.2 | `torch.load` 默认 `weights_only=True`（代码里部分已显式传 False）；少数 API 行为差异 | 现有代码已显式 `weights_only=False`，风险较低 |
| R4 | `flash_attn` / `apex` / `deepspeed` **未安装** | 若训练/评估路径真的用到会 ImportError；这些需匹配 cu128+sm_100 编译 | 先跑 demo/推理，训练前再处理 |
| R5 | **2000 HVG 有序列表缺失且无法重生**（`preprocess/E_hvg_multi_celllines.py` 用未设种子的 `random.sample`） | 1.3 的基因级指标（top-50 DEG Pearson 等）**不可复现** | 只能向作者索取（issue #3/#7 已提） |
| R6 | DEG 检验实现与论文不一致（代码用 Welch t-test，论文写 Wilcoxon+FDR） | 即使指标算出，也**不能与论文数字直接对比** | 明确记录口径，必要时自行实现 Wilcoxon 版本 |
| R7 | MAP-KG 发布数据与论文计数有偏差（issue #5 实测：drug +900、gene −1、drug-gene −7,828、gene-gene −16,078，总边 −3.44%） | 阶段二训练出的嵌入与论文略有差异 | 记录基线数字，不复现论文数值时以此解释 |
| R8 | `MAP/` 下缺 5 个文件（`ds_nips_lora_se.py`、`ds_sciplex_lora_se.py`、`ds_combosciplex_lora_se.py`、`model_combosciplex_avg_emb.py`、`model_combosciplex_two_tokens.py`） | `train_nips.py` / `train_sciplex.py` / `train_combosciplex.py` 及其 eval **全部不可运行** | 1.3 只能走 Tahoe 路径（`train.py` + `eval_tahoe.py`），或向作者索取 |
| R9 | `/vllm/tangc_pkuhpc` 目录读取报 I/O error | 与本项目无关，但**不要把数据/环境放 /vllm** | 已避开；需要时找管理员修 |
| R10 | `/lustre1` 账面超配额 1.19 TB（实测写入未受阻） | 放大数据有风险（管理员随时可能开启强制） | 大数据放 `/data-tangc2`（剩 3.9 T） |
| R11 | 集群 git 直连 GitHub 不通 | 无法 `git push`/`pull` 直连 | 用镜像地址；推送需凭据（待定） |

---

## 5. 下一步（按优先级）

| 优先级 | 事项 | 依赖 |
|---|---|---|
| **P0** | 阶段二：下 MAP-KG 三个 CSV（hf-mirror，98.5 MB）→ 改 P6 → `MAP-KG/train_resume.sh` | 无依赖，**现在就能做** |
| P1 | 下 SE-600M 的 `protein_embeddings.pt`（作 `all_embeddings`）+ `se600m.safetensors` | 无依赖 |
| P2 | 打补丁 P2/P3（路径参数化、config 本地化） | 无依赖 |
| P3 | 1.1：拿 `epoch_3.pt` + `bart_vocab.txt` → 打 P1/P4 → 跑 `demo.py` | 等 Drive |
| P4 | 1.2：固定 seed 的复现性与生物学合理性检查 | 依赖 P3 |
| P5 | 1.3：Tahoe 路径（`train.py` + `eval_tahoe.py`）+ 预处理；SciPlex3/OP3 路径受 R8 阻塞 | 依赖 HVG 列表（R5）与 300 GB 数据 |

---

## 6. 常用命令备忘

```bash
# 激活环境（或直接用 .venv/bin/python）
source /data-tangc/dev/ptb/map-repro/.venv/bin/activate

# uv 安装时固定缓存到本地盘（默认在 GPFS home，慢）
export UV_CACHE_DIR=/data-tangc/dev/ptb/.uv-cache
uv pip install --python .venv/bin/python <pkgs>

# uv 下载 Python 需走镜像（GitHub 直连不通）
UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone \
  uv venv --python 3.11 .venv

# HuggingFace 下载走镜像
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 7. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-26 | 建立本文件；记录环境搭建、issue 调研、缺失文件与 11 项风险 |