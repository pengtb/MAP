# TODO — MAP 复现任务清单

> 位置：`/data-tangc/dev/ptb/map-repro/TODO.md`｜关联 `PLAN.md`（§4 为重新实现规格）
> 最后更新：2026-10-27（补入 checkpoint 结构核验与 `args` 结论，P4 撤销）

> 文档结构（2026-10-27 起）：分步计划见 `plan/`（`01`~`07`），架构见 `design/architecture.md`；原 `PLAN.md` 已拆分并删除。
> 资源根目录：`/data-tangc/models/map_repr/{predictor, mapkg_encoder, molecule_stm, state_se600m, esm2_gene}`（另有 `biobert/`、`esm2_protein/` 待填充）。

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
- ✅ `transformers` 定版 **4.43.4**（5.x 下双向注意力 override 失效；4.43 的 SDPA 用 `causal_mask is None` 决定 `is_causal`）
- ✅ 可导入：`model.se` / `model.pert` / `model.smiles_kg_encoder_v3` / `model.mega_molbart.STMencoder_ddp` / `data.utils`
- ❌ `model.model` 导入失败（缺 `model/pert_ca.py`，待 §3 P1）

### 1.3 官方权重（`/data-tangc/models/map_repr/`，5.4 GB）
- ✅ `epoch_3.pt`（3.94 GiB）：499 张量 = `pert_model` 260 + `se` 239；`epoch=2`、`global_step=155847`
- ✅ `mapkg_encoder_v3.pt`（1.05 GiB）：419 张量，含 `smiles_encoder` 176 / `smiles_projector` 10 / `text_encoder` 200 / `fusion_module` 12 等
- ✅ `Homo_sapiens...ESM2.pt`（392 MiB）：dict，19790 基因 × 5120 维

### 1.4 已完成的核验（详见 plan/02-resources.md、plan/03-reimplementation.md）
- ✅ 导出 `epoch_3.pt['args']`：6 个细胞系（CVCL_0023/0480/0069/0131/1098/1056）、`smile_encoder="PrimeKG_ver1"`、**`hvg_info=null`**、`set_size=24`、`epochs=400`、`lr=1e-5`、`de_topk=50`、amp bf16
- ✅ 全仓 grep `PrimeKG`：仅 `demo.py:80` 与 `train_sciplex.py:118` 两处默认值，**无实现**；`pert.py:117` 只认 `MAP-KG`，其余 `exit()`
- ✅ 键比对：`mapkg_encoder_v3.pt` 的 `smiles_encoder.*` 与 ckpt 的 `pert_model.smiles_encoder.*` **176/176 完全一致，零形状差异**（`emb.weight=(523,256)`）
- ✅ `pert_model` 260 键的 7 个子模块已列出（`smiles_encoder` 176 / `transformer_backbone` 42 / `drug_projector` 10 / `gene_tokens_projector` 10 / `gene_decoder` 10 / `cell_projector` 8 / `project_out` 4）
- ✅ 差异定位：`pert.py` 用 `KGSmilesEncoder_v3` 封装（`smiles_encoder._model.*` + `smiles_projector.*`），ckpt 拆成平级的 `smiles_encoder` + `drug_projector`

### 1.5 调研
- ✅ issues：#4/#5/#6 均报缺文件，作者未回复；#1/#2 作者答复过（Drive 链接、minimal dataset、`epoch_3.pt` 为初步 ckpt）
- ✅ `MAP-KG/` 与 `preprocess/` 代码完整；缺文件集中在 `MAP/`
- ✅ 带宽实测 8–10 MB/s（b200 出口为 1 GbE）
- ✅ 已读论文正文与 SI（Zotero 附件）：确认复用边界（BioBERT / MoleculeSTM / ESM-2 / STATE SE-600M / 基线 / 数据集 / 14 知识源），详见 design/architecture.md 与 plan/02-resources.md
- ✅ 已读图源数据 xlsx（`Q3ZZX7MA`，11 个 sheet 对应 Fig_1~Fig_5b，含每药每细胞系每方法的精确数值）→ 作为数值对照基准（见 plan/02-resources.md）
- ✅ **`bart_vocab.txt` 已解决**：取自 `chao1224/MoleculeSTM`，实测 523 token、index272=`C`、`MolEncTokenizer` 加载 `len=523`，与 ckpt 的 `emb.weight=(523,256)` 吻合 → 已放 `/data-tangc/models/map_repr/molecule_stm/bart_vocab.txt`

---

## 2. 仍然缺的东西

| # | 项 | 状态 | 影响 / 处置 |
|---|---|---|---|
| M1 | **`bart_vocab.txt`** | ✅ **已解决** | 来自 `chao1224/MoleculeSTM`（5,495 B，523 token，chem start=272），已放到 `/data-tangc/models/map_repr/molecule_stm/`；残留风险：token 顺序靠同源性推断，需 demo 输出做生物学间接验证 |
| M2 | `MAP/model/pert_ca.py` | 🔄 缺失但规格已定 | 见 §3 P1；不再是"不明结构" |
| M3 | `se600m.safetensors` | ✅ 已下载 | `state_se600m/`，2,860,276,816 B（2.66 GiB），sha256 `3ebec58bd9e9c07f…` 与 HF LFS etag 一致；238 张量 / 16 层 |
| M4 | `gene_embidx_mapping.torch` | ❓ 大概率不需要 | demo 路径只读 meta.pkl / control_indices / se_* |
| M5 | 2000 HVG 有序列表 | ⚠️ 仅影响 1.3 | ckpt 训练时 `hvg_info=null`，1.1/1.2 不需要；1.3 基因级指标仍需且**不可重生** |
| M6 | `molecule_model.pth` | ✅ 已解决 | 用 `mapkg_encoder_v3.pt` 的 `smiles_encoder.*`（同一权重） |
| M7 | `MAP/` 5 个未发布文件 | ❌ 缺 | 仅影响 OP3/SciPlex3/Combo（1.3）；Tahoe 路径可用 |

---

## 3. 要打的补丁

| # | 位置 | 做什么 | 依赖 |
|---|---|---|---|
| ✅ P1 | `MAP/model/pert_ca.py`（新建） | 写 `PerturbationEncoder_ca`：组装顺序同 `pert.py`，但把药物分支拆成 `self.smiles_encoder = MolSTM_Extractor(...)`（→ 176 键）+ `self.drug_projector = ResidualProjector(256→1024)`（→ 10 键）；`smile_encoder` 接受 `{'PrimeKG_ver1','MAP-KG'}`；`forward/encode_*` 与 `pert.py` 对齐 | 无（需先通读 `pert.py:75+`） |
| ✅ P2 | `MAP/model/mega_molbart/STMencoder_ddp.py`（`STMencoder.py` 未被引用，不改） | ①`ckpt_path` → 环境变量 `MAP_MOLECULE_CKPT`，允许指向 `mapkg_encoder_v3.pt`（抽取 `smiles_encoder.` 子字典去前缀）；②`vocab_path` → `MAP_BART_VOCAB`；③`_load_model()` 支持注入 state_dict | 无 |
| ✅ P3 | `MAP/configs/se600m.yaml` | `embeddings.current: esm2-cellxgene`；`all_embeddings` → `/data-tangc/models/map_repr/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt`；清理其余 26 处外部路径；不动 `nlayers` | 无 |
| ✅ P5 | `MAP/model/model.py`（追加） | `forward()` 读 `self.pert_model_hvg_ids_already_set`，但 `__init__` 未定义 → 补 `False` 与 `hvg_ids=None` | 无 |
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
| **R15** | **药物信号偏弱（未决）** | 同 cells/seed 下，吡哆醇 vs 布洛芬 vs 咖啡因的输出最大差仅 **2.98e-4 / 3.44e-4**；而 gene tokens 置零影响约 2%（均值 0.1528→0.1495）。可能是 log1p 空间的正常量级，也可能是 drug token 与 gene token 量级失衡（drug vec 范数 ~50） | 待查：先测 gene token 实际范数并与 drug 对比；必要时确认原训练是否对 drug token 做归一化。判据可参考图源 `Fig_3c` |
| **R16** | 剂量对输出无影响 | 0.1/0.5/5.0 μM 输出 md5 相同。非 bug：`encode_conc=False`，`concs` 从未进入 forward，ckpt `args` 相同 | 记录即可；剂量条件化属扩展而非复现 |
| **R17** | 双向注意力依赖 `_update_causal_mask` | 上游实现随 transformers 版本变化（5.x 直接删除），换环境会**静默**退化为因果注意力、指标失真且不报错 | 已用「全零加性 mask」加固；升级 transformers 后必须重跑 `plan/05` 的 sanity（zero-drug 与 real 应不同） |

---

## 5. 下一步

| 优先级 | 事项 | 依赖 |
|---|---|---|
| ~~P0~~ | ~~通读 `pert.py:75+` 并写 `pert_ca.py`~~ | ✅ 已完成 |
| **P0** | 打 P2、P3 | 无 |
| ~~P1~~ | ~~下 `se600m.safetensors`~~ | ✅ 已完成（sha256 校验通过） |
| ~~P1~~ | ~~加载验证~~ | ✅ 已完成：`strict=True` → `<All keys matched successfully>`，499/499 |
| ~~P2~~ | ~~跑 `demo.py`（1.1）~~ | ✅ 已完成（rc=0，20.05 s / 12.7 GiB，输出无 NaN） |
| **P3** | 1.2 可复现性（固定 seed、换条件、与 minimal_dataset 对照） | 1.1 |
| **P0** | **R15 量级排查**：测 gene token 与 drug token 的实际范数，判断 drug 是否被稀释或量级失衡（约 30 min） | 无 |
| **P1** | 阶段二：下 MAP-KG 3 个 CSV + BioBERT + ESM-2 650M → `MAP-KG/train_resume.sh` | 无阻塞 |
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
| 2026-10-27 | 读正文/SI/图源 xlsx：新增复用边界（PLAN §1.4）与图源数据（§1.5）；**M1 解除**（MoleculeSTM 词表 523 token 验证通过） |
| 2026-10-27 | 目录重组：权重按 encoder/基础模型分子目录（`predictor/`、`mapkg_encoder/`、`molecule_stm/`、`state_se600m/`、`esm2_gene/`）；`PLAN.md` 拆分为 `plan/01~07 + README`；新增 `design/architecture.md` |
| 2026-10-27 | `se600m.safetensors` 经 hf-mirror 下载完成并校验（2,860,276,816 B（2.66 GiB），sha256 与 HF etag 一致）；M3/P1 关闭 |
| 2026-10-27 | demo 跑通：新增 `pert_ca.py`、修 `model.py` 两处（类选择 + 标志位）、`STMencoder_ddp.py` 路径与子字典、config 1 行；`strict=True` 499/499；demo 20.05 s / 12.7 GiB |
| 2026-10-27 | 3.2 完成：`demo.py` 加 `--seed`；**修复双向注意力失效**（transformers 4.4x 强制 causal）；`transformers` 定版 4.43.4；修复 `hvg_info` 标志位；9 个 run 结果见 `plan/05`。新发现：剂量无效（`encode_conc=False`，代码事实）、药物信号仅 ~3e-4（待查） |

---

## 8. 交接（后续 agent / 协作者看这里）

### 8.1 环境与关键变量

```bash
source /data-tangc/dev/ptb/map-repro/.venv/bin/activate     # Python 3.11.14 + torch 2.8.0+cu128 + transformers 4.43.4
export MAP_MOLECULE_CKPT=/data-tangc/models/map_repr/mapkg_encoder/mapkg_encoder_v3.pt
export MAP_BART_VOCAB=/data-tangc/models/map_repr/molecule_stm/bart_vocab.txt
export CUDA_VISIBLE_DEVICES=0        # GPU 4-7 常被他人占用（各 167 GB）
```

### 8.2 一键复现 demo

```bash
cd /data-tangc/dev/ptb/map-repro/MAP
python demo.py --ckpt /data-tangc/models/map_repr/predictor/epoch_3.pt \
  --cell_line CVCL_0023 --drug_smiles "Cc1ncc(CO)c(CO)c1O" --drug_conc 0.5 --set_size 24 --seed 0 \
  --se_ckpt /data-tangc/models/map_repr/state_se600m/se600m.safetensors \
  --se_config configs/se600m.yaml \
  --data_base_dir ../work/minimal_dataset/preprocessed \
  --se_inputs_base_dir ../work/minimal_dataset/preprocessed_se_inputs_memmap \
  --output_dir ../work/demo_output
```

### 8.3 本仓库改过的文件（相对上游）

| 文件 | 改动摘要 |
|---|---|
| `MAP/model/pert_ca.py` | 新增：键名对齐 ckpt 的编码器 + 全零加性 mask（双向注意力）+ 持久 rotary buffer + `MAP_DEBUG_DRUG` 调试钩子 |
| `MAP/model/model.py` | 实例化 `PerturbationEncoder_ca`；初始化 `pert_model_hvg_ids_already_set` / `hvg_ids` |
| `MAP/model/mega_molbart/STMencoder_ddp.py` | `MAP_MOLECULE_CKPT` / `MAP_BART_VOCAB` 环境变量；从 `mapkg_encoder_v3.pt` 抽取 `smiles_encoder._model.*`；`weights_only=False` |
| `MAP/configs/se600m.yaml` | `all_embeddings` 指向本地 ESM2 嵌入表 |
| `MAP/demo.py` | `--seed`；保存 `input_ctrl_row_indices.npy`；meta 记录 seed |

### 8.4 已知坑（踩过的）

1. `transformers` 必须 **4.43.4**；5.x 会让双向注意力静默失效（输出与药物无关）。
2. `torch.load` 需显式 `weights_only=False`（ckpt 含 numpy 标量）。
3. `inv_freq` 在 4.43 是非持久 buffer，需在 `pert_ca.py` 重新注册，否则 `strict=True` 报 4 个 unexpected 键。
4. demo 的 `se_gene_ids.npy` / `se_expr.npy` 是**裸 memmap（无 npy 头）**，必须按 `shape/dtype` 读，不能用 `np.load`。
5. 不传 `--seed` 时 `np.random.choice` 每次抽到的 control cells 不同。
6. 进程内连续多次 forward 有 ~3e-3 的 kernel warm-up 抖动；比较不同条件务必**每次单独起进程**。

### 8.5 产物位置

| 路径 | 内容 |
|---|---|
| `work/minimal_dataset/` | 解压后的 demo 输入 |
| `work/demo_output/` | 单次 demo 输出（4 个 npy + meta.json） |
| `work/exp32d/` | 3.2 的 9 个 run（无 seed ×3、seed0 ×2、剂量 ×2、药物 ×2） |
| `/data-tangc/models/map_repr/` | 全部权重，按 encoder / 基础模型分子目录 |
