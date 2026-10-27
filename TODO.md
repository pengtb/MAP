# TODO — MAP 复现任务清单

> 位置：`/data-tangc/dev/ptb/map-repro/TODO.md`｜关联 `PLAN.md`
> 最后更新：2026-10-27（Drive 权重到位并完成结构核验后）

图例：✅ 完成 ｜ 🔄 进行中 ｜ ⬜ 待做 ｜ ⚠️ 风险 ｜ ❌ 阻塞（等外部输入）

---

## 1. 已完成

### 1.1 仓库与 git
- ✅ 克隆 + 分支 `repo`（跟踪 `origin/repo`），remote：`origin`=`git@github.com:pengtb/MAP.git`（deploy key），`upstream`=ghfast 镜像
- ✅ 仓库级身份 `pengtb <ptbjj@126.com>`；`.secrets/` 已忽略（`.gitignore` 本地维护、不提交）
- ✅ 已 unshallow（85+ commits，含 tag `v1.0.0`、`v.1.0.1`）
- ✅ 首次推送成功；`PLAN.md`/`TODO.md` 已提交推送

### 1.2 运行环境
- ✅ `.venv`（Python 3.11.14，uv 创建）+ `torch 2.8.0+cu128`
- ✅ GPU 验证：`cap=(10,0)`、`archs` 含 `sm_100`、8192² bf16 matmul 成功
- ✅ 依赖：numpy/pandas/scipy/sklearn/rdkit/omegaconf/safetensors/transformers/scanpy/anndata/lightning/requests
- ✅ 可导入：`model.se` / `model.pert` / `model.smiles_kg_encoder_v3` / `model.mega_molbart.STMencoder_ddp` / `data.utils`
- ❌ `model.model` 导入失败（缺 `model/pert_ca.py`）

### 1.3 官方权重（2026-10-27 到位，`/data-tangc/models/map_repr/`，5.4 GB）
- ✅ `epoch_3.pt`（4,227,878,838 B）：`model_state_dict` 499 张量 = `pert_model` 260 + `se` 239；顶层键还有 `epoch/global_step/optimizer_state_dict/scheduler_state_dict/args`
- ✅ `mapkg_encoder_v3.pt`（1,126,763,160 B）：419 张量，前缀 `text_encoder` 200 / `smiles_encoder` 176 / `fusion_module` 12 / `smiles_projector` 10 / `gene_projector` 10 / `text_projector` 10 / `logit_scale` 1 → **即 `pert.py` 需要的 KG 预训练编码器**
- ✅ `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt`（410,886,729 B）：dict，19790 基因 × 5120 维 → 可直接作 `all_embeddings`

### 1.4 调研
- ✅ issues：缺文件问题被 3 次报告（#4/#5/#6），作者未回复；#1/#2 作者答复过（Drive 链接、minimal dataset、`epoch_3.pt` 为初步 ckpt）
- ✅ `MAP-KG/` 与 `preprocess/` 代码完整；缺文件集中在 `MAP/`
- ✅ 带宽实测 8–10 MB/s；b200 出口为 1 GbE（硬件层面上限）

---

## 2. 仍然缺的东西

| # | 项 | 状态 | 影响 / 处置 |
|---|---|---|---|
| M1 | **`bart_vocab.txt`** | ❌ **硬阻塞** | 分词器构造即失败；`emb.weight=(32000,1024)` 不可替代 → 只能向作者索取（issue #5/#6 已提，未回复） |
| M2 | **`MAP/model/pert_ca.py`** | ❌ 缺 | 但 checkpoint 键为 `pert_model.smiles_encoder.*`（176 个，**无 `kg_` 前缀**），而 `pert.py` 命名是 `kg_smiles_encoder` → **不能简单 re-export**，需写属性名为 `smiles_encoder` 的等价类 |
| M3 | `se600m.safetensors` | ⬜ 待下载 | HF `arcinstitute/SE-600M`（2.66 GiB），走 `HF_ENDPOINT=https://hf-mirror.com` |
| M4 | `gene_embidx_mapping.torch` | ❓ 待确认是否真需要 | config 的 `ds_emb_mapping`；demo 路径（`load_control_bulk`）只读 meta.pkl / control_indices / se_* → 很可能不需要 |
| M5 | 2000 HVG 有序基因列表 | ❌ 缺且不可重生 | `preprocess/E_hvg_multi_celllines.py` 用未设种子 `random.sample`；阻塞 1.3 基因级指标 |
| M6 | `molecule_model.pth` | ✅ 有替代方案 | `mapkg_encoder_v3.pt` 已含 `smiles_encoder.*` 176 键 → 改 `STMencoder_ddp.py` 允许从该文件抽取子字典即可，无需原文件 |
| M7 | `MAP/` 5 个未发布文件（`ds_nips_lora_se.py`、`ds_sciplex_lora_se.py`、`ds_combosciplex_lora_se.py`、`model_combosciplex_avg_emb.py`、`model_combosciplex_two_tokens.py`） | ❌ 缺 | 仅影响 OP3/SciPlex3/ComboSciPlex（1.3）；Tahoe 路径不受影响 |

---

## 3. 要打的补丁

| # | 位置 | 做什么 | 依赖 |
|---|---|---|---|
| P1 | `MAP/model/pert_ca.py`（新建） | 等价 `PerturbationEncoder`，把 smiles 编码器属性命名为 **`smiles_encoder`**（匹配 ckpt 的 `pert_model.smiles_encoder.*`），其余组件复用 `pert.py` | 无 |
| P2 | `MAP/model/mega_molbart/STMencoder_ddp.py`（及 `STMencoder.py`） | 两个硬编码作者路径改成环境变量；并允许 `ckpt_path` 指向 `mapkg_encoder_v3.pt`（自动抽取 `smiles_encoder.` 子字典） | 无 |
| P3 | `MAP/configs/se600m.yaml` | `all_embeddings` → `/data-tangc/models/map_repr/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt`；`embeddings.current: esm2-cellxgene`；清理其余 26 处外部路径 | 无 |
| P4 | `MAP/demo.py` | `smile_encoder="PrimeKG_ver1"` → `"MAP-KG"`（`pert.py::encode_drug` 只认 `MAP-KG`，否则 `exit()`） | 无 |
| P5 | `preprocess/Cs_split_unseen_combination.py` 输出 | 产出 `control_indices_UC.json`，demo 读 `control_indices.json` → 需改名/复制（仅走预处理时需要） | 1.3 |
| P6 | `MAP-KG/train_resume.py:27-29` | CSV 路径指向 `data/selected_data_csvs/` | 阶段二 |

---

## 4. 风险登记

| # | 风险 | 影响 | 处置 |
|---|---|---|---|
| R1 | `transformers 5.20.2` ≫ 代码时代的 4.30 | 5.x API 破坏可能报错 | 报错即降级到 4.46.x |
| R2 | `numpy 2.4.6` ≫ 要求的 1.24.4 | `np.float_` 等已移除 | 按报错替换 |
| R3 | torch 2.8 vs 论文 2.1.2 | `torch.load` 默认行为差异 | 代码已显式 `weights_only=False`，风险低 |
| R4 | `flash_attn`/`apex`/`deepspeed` 未装 | 训练/评估路径可能 ImportError | 先跑推理，训练前再处理 |
| R5 | HVG 2000 列表缺失且不可重生 | 1.3 基因级指标不可复现 | 向作者索取（#3/#7） |
| R6 | DEG 检验用 Welch t-test，论文写 Wilcoxon+FDR | 指标口径不一致 | 记录说明；必要时自实现 |
| R7 | MAP-KG 发布数据与论文计数偏差（总边 −3.44%） | 阶段二嵌入略偏 | 作为基线差异记录 |
| R8 | `MAP/` 缺 5 个文件 | OP3/SciPlex/Combo 路径不可用 | 走 Tahoe 路径或索取 |
| R9 | `/vllm/tangc_pkuhpc` 读 I/O error | 与该目录相关的数据取不出 | 避开；找管理员修 |
| R10 | `/lustre1` 账面超配额 1.19 TB | 放大数据有风险 | 大数据放 `/data-tangc2` |
| R11 | b200 出口仅 **1 GbE**（实测 8–10 MB/s） | 300 GB 级数据集需 ~10 小时 | 大文件用 tmux 长任务；优先 SciPlex3 |
| R12 | ckpt 与代码键名不一致（`pert_model.smiles_encoder.*` vs `kg_smiles_encoder`） | `strict=True` 加载会失败 | 由 P1 解决 |
| R13 | `molecule_model.pth` 未获得 | 构造 `KGSmilesEncoder_v3` 时 `torch.load` 失败 | 由 P2 用 `mapkg_encoder_v3.pt` 替代 |
| R14 | `epoch_3.pt` 里 `args` 可能含原始训练配置 | 可用于反推 HVG/细胞系设置 | 下一步导出查看 |

---

## 5. 下一步

| 优先级 | 事项 | 依赖 |
|---|---|---|
| **P0** | 导出 `epoch_3.pt['args']` 看原始训练参数（可能含 hvg / cell_lines 线索） | 无 |
| **P0** | 打 P1/P2/P3/P4 四个补丁 | 无 |
| **P1** | 下 `se600m.safetensors`（hf-mirror，2.66 GiB） | 无 |
| **P1** | 试加载 `epoch_3.pt`（只验证 state_dict 匹配，不跑前向） | P0 补丁 |
| **P2** | 拿 `bart_vocab.txt` → 跑 `demo.py`（1.1） | M1 硬阻塞 |
| **P3** | 1.2 可复现性检查（固定 seed、换条件） | 1.1 |
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
| 2026-10-27 | Drive 权重到位（3 文件 5.4 GB）并完成结构核验；更新缺失项、补丁与风险（新增 R12–R14） |