# TODO — MAP 复现任务清单

> 位置：`/data-tangc/dev/ptb/map-repro/TODO.md`｜关联 `PLAN.md`（§4 为重新实现规格）
> 最后更新：2026-10-27（补入 checkpoint 结构核验与 `args` 结论，P4 撤销）

图例：✅ 完成 ｜ 🔄 进行中 ｜ ⬜ 待做 ｜ ⚠️ 风险 ｜ ❌ 阻塞（等外部输入）

---

## 1. 已完成

### 1.1 仓库与 git
- ✅ 克隆 + 分支 `repo`（跟踪 `origin/repo`）；`origin`=`git@github.com:pengtb/MAP.git`（deploy key），`upstream`=ghfast 镜像
- ✅ 仓库级身份 `pengtb <ptbjj@126.com>`；`.secrets/` 已忽略（`.gitignore` 本地维护、不提交）
- ✅ 已 unshallow（85+ commits，含 tag `v1.0.0`、`v.1.0.1`）
- ✅ `PLAN.md`/`TODO.md` 已提交推送（最新 `a5c8a2e`）

### 1.2 运行环境
- ✅ `.venv`（Python 3.11.14，uv 创建）+ `torch 2.8.0+cu128`
- ✅ GPU 验证：`cap=(10,0)`、`archs` 含 `sm_100`、8192² bf16 matmul 成功
- ✅ 依赖：numpy/pandas/scipy/sklearn/rdkit/omegaconf/safetensors/transformers/scanpy/anndata/lightning/requests
- ✅ 可导入：`model.se` / `model.pert` / `model.smiles_kg_encoder_v3` / `model.mega_molbart.STMencoder_ddp` / `data.utils`
- ❌ `model.model` 导入失败（缺 `model/pert_ca.py`，待 §3 P1）

### 1.3 官方权重（`/data-tangc/models/map_repr/`，5.4 GB）
- ✅ `epoch_3.pt`（3.94 GiB）：499 张量 = `pert_model` 260 + `se` 239；`epoch=2`、`global_step=155847`
- ✅ `mapkg_encoder_v3.pt`（1.05 GiB）：419 张量，含 `smiles_encoder` 176 / `smiles_projector` 10 / `text_encoder` 200 / `fusion_module` 12 等
- ✅ `Homo_sapiens...ESM2.pt`（392 MiB）：dict，19790 基因 × 5120 维

### 1.4 已完成的核验（PLAN §1.1、§4 的事实来源）
- ✅ 导出 `epoch_3.pt['args']`：6 个细胞系（CVCL_0023/0480/0069/0131/1098/1056）、`smile_encoder="PrimeKG_ver1"`、**`hvg_info=null`**、`set_size=24`、`epochs=400`、`lr=1e-5`、`de_topk=50`、amp bf16
- ✅ 全仓 grep `PrimeKG`：仅 `demo.py:80` 与 `train_sciplex.py:118` 两处默认值，**无实现**；`pert.py:117` 只认 `MAP-KG`，其余 `exit()`
- ✅ 键比对：`mapkg_encoder_v3.pt` 的 `smiles_encoder.*` 与 ckpt 的 `pert_model.smiles_encoder.*` **176/176 完全一致，零形状差异**（`emb.weight=(523,256)`）
- ✅ `pert_model` 260 键的 7 个子模块已列出（`smiles_encoder` 176 / `transformer_backbone` 42 / `drug_projector` 10 / `gene_tokens_projector` 10 / `gene_decoder` 10 / `cell_projector` 8 / `project_out` 4）
- ✅ 差异定位：`pert.py` 用 `KGSmilesEncoder_v3` 封装（`smiles_encoder._model.*` + `smiles_projector.*`），ckpt 拆成平级的 `smiles_encoder` + `drug_projector`

### 1.5 调研
- ✅ issues：#4/#5/#6 均报缺文件，作者未回复；#1/#2 作者答复过（Drive 链接、minimal dataset、`epoch_3.pt` 为初步 ckpt）
- ✅ `MAP-KG/` 与 `preprocess/` 代码完整；缺文件集中在 `MAP/`
- ✅ 带宽实测 8–10 MB/s（b200 出口为 1 GbE）

---

## 2. 仍然缺的东西

| # | 项 | 状态 | 影响 / 处置 |
|---|---|---|---|
| M1 | **`bart_vocab.txt`** | ❌ **唯一硬阻塞** | 分词器构造即失败；`transformer_backbone.embed_tokens=(32000,1024)` 与 `smiles_encoder._model.emb=(523,256)` 都绑定该词表，换表会静默错位 → 只能向作者索取 |
| M2 | `MAP/model/pert_ca.py` | 🔄 缺失但规格已定 | 见 §3 P1；不再是"不明结构" |
| M3 | `se600m.safetensors` | ⬜ 待下载 | HF `arcinstitute/SE-600M`（2.66 GiB），走 hf-mirror |
| M4 | `gene_embidx_mapping.torch` | ❓ 大概率不需要 | demo 路径只读 meta.pkl / control_indices / se_* |
| M5 | 2000 HVG 有序列表 | ⚠️ 仅影响 1.3 | ckpt 训练时 `hvg_info=null`，1.1/1.2 不需要；1.3 基因级指标仍需且**不可重生** |
| M6 | `molecule_model.pth` | ✅ 已解决 | 用 `mapkg_encoder_v3.pt` 的 `smiles_encoder.*`（同一权重） |
| M7 | `MAP/` 5 个未发布文件 | ❌ 缺 | 仅影响 OP3/SciPlex3/Combo（1.3）；Tahoe 路径可用 |

---

## 3. 要打的补丁

| # | 位置 | 做什么 | 依赖 |
|---|---|---|---|
| P1 | `MAP/model/pert_ca.py`（新建） | 写 `PerturbationEncoder_ca`：组装顺序同 `pert.py`，但把药物分支拆成 `self.smiles_encoder = MolSTM_Extractor(...)`（→ 176 键）+ `self.drug_projector = ResidualProjector(256→1024)`（→ 10 键）；`smile_encoder` 接受 `{'PrimeKG_ver1','MAP-KG'}`；`forward/encode_*` 与 `pert.py` 对齐 | 无（需先通读 `pert.py:75+`） |
| P2 | `MAP/model/mega_molbart/STMencoder_ddp.py`（及 `STMencoder.py`） | ①`ckpt_path` → 环境变量 `MAP_MOLECULE_CKPT`，允许指向 `mapkg_encoder_v3.pt`（抽取 `smiles_encoder.` 子字典去前缀）；②`vocab_path` → `MAP_BART_VOCAB`；③`_load_model()` 支持注入 state_dict | 无 |
| P3 | `MAP/configs/se600m.yaml` | `embeddings.current: esm2-cellxgene`；`all_embeddings` → `/data-tangc/models/map_repr/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt`；清理其余 26 处外部路径；不动 `nlayers` | 无 |
| ~~P4~~ | ~~`MAP/demo.py`~~ | **已撤销**：ckpt 的 `args.smile_encoder="PrimeKG_ver1"` 与 demo 一致，**不要改** | — |
| P5 | `preprocess/Cs_split_unseen_combination.py` 输出 | 产出 `control_indices_UC.json`，demo 读 `control_indices.json` → 需改名/复制（仅走预处理时需要） | 1.3 |
| P6 | `MAP-KG/train_resume.py:27-29` | CSV 路径指向 `data/selected_data_csvs/` | 阶段二 |

**明确不改**：`MAP/demo.py`、`MAP/model/pert.py`、`MAP/model/model.py`（理由见 PLAN §4.6）。

---

## 4. 风险登记

| # | 风险 | 影响 | 处置 |
|---|---|---|---|
| R1 | `transformers 5.20.2` ≫ 代码时代的 4.30 | 5.x API 破坏可能报错 | 报错即降级 4.46.x |
| R2 | `numpy 2.4.6` ≫ 要求的 1.24.4 | `np.float_` 等已移除 | 按报错替换 |
| R3 | torch 2.8 vs 论文 2.1.2 | `torch.load` 默认行为差异 | 代码已显式 `weights_only=False` |
| R4 | `flash_attn`/`apex`/`deepspeed` 未装 | 训练/评估路径可能 ImportError | 先跑推理，训练前再处理 |
| R5 | HVG 列表缺失且不可重生 | **仅** 1.3 基因级指标不可复现（1.1/1.2 不受影响） | 向作者索取（#3/#7） |
| R6 | DEG 检验用 Welch t-test，论文写 Wilcoxon+FDR | 指标口径不一致 | 记录说明；必要时自实现 |
| R7 | MAP-KG 发布数据与论文计数偏差（总边 −3.44%） | 阶段二嵌入略偏 | 作为基线差异记录 |
| R8 | `MAP/` 缺 5 个文件 | OP3/SciPlex/Combo 路径不可用 | 走 Tahoe 路径或索取 |
| R9 | `/vllm/tangc_pkuhpc` 读 I/O error | 该目录数据取不出 | 避开；找管理员修 |
| R10 | `/lustre1` 账面超配额 1.19 TB | 放大数据有风险 | 大数据放 `/data-tangc2` |
| R11 | b200 出口仅 1 GbE（实测 8–10 MB/s） | 300 GB 级数据集 ~10 小时 | tmux 长任务；优先 SciPlex3 |
| R12 | ckpt 与 `pert.py` 键名结构不一致（`smiles_encoder`+`drug_projector` vs `kg_smiles_encoder.*`） | `strict=True` 加载会失败 | 由 P1 解决 |
| R13 | ~~`molecule_model.pth` 缺失~~ | — | **已解除**：`mapkg_encoder_v3.pt` 权重一致，由 P2 接入 |
| R14 | `bart_vocab.txt` 绑定两处词表维度 | 拿不到就完全无法启动 | 只能索取；不可用替代词表 |

---

## 5. 下一步

| 优先级 | 事项 | 依赖 |
|---|---|---|
| **P0** | 通读 `pert.py:75+`（forward / encode_cells / encode_genes / encode_drug），写出 P1 的 `pert_ca.py` | 无 |
| **P0** | 打 P2、P3 | 无 |
| **P1** | 下 `se600m.safetensors`（hf-mirror，2.66 GiB） | 无 |
| **P1** | **加载验证**：用 `epoch_3.pt` 试 `load_state_dict`，确认 499 键全命中（不跑前向） | P0 补丁 |
| **P2** | 拿 `bart_vocab.txt` → 跑 `demo.py`（1.1） | M1 |
| **P3** | 1.2 可复现性（固定 seed、换条件、与 minimal_dataset 对照） | 1.1 |
| **P4** | 阶段二：MAP-KG 3 个 CSV → `train_resume.sh` | 无阻塞，可并行 |
| **P5** | 1.3（Tahoe 路径） | HVG 列表 + 300 GB 数据 |

---

## 6. 常用命令

```bash
source /data-tangc/dev/ptb/map-repro/.venv/bin/activate
export UV_CACHE_DIR=/data-tangc/dev/ptb/.uv-cache
export HF_ENDPOINT=https://hf-mirror.com
UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone  # uv 装 Python 需指定
```

---

## 7. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-26 | 建文件：环境搭建、issue 调研、缺失文件与风险 |
| 2026-10-27 | Drive 权重到位（3 文件 5.4 GB）并完成结构核验；撤销 P4、R5 降级、M6/R13 解除；补入 `args` 与 260 键结构结论 |