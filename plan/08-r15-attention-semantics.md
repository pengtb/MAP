# 08 — R15：发布 ckpt 训练时的注意力语义判定（因果 vs 双向）

## 该计划简述

plan/05 测得药物信号极弱：换药 maxdiff ≈ 3e-4，gene token 置零 ≈ 2%。Lead 的静态核查（见下）指向一个假设：

- **H1**：发布的 `epoch_3.pt` 是在**因果注意力**下训练的。drug / gene token 训练中拿不到梯度，ckpt 本身与药物无关；当前"双向修复"后测到的药物效应来自未训练的权重。

另外两种结果：

- **H2**：训练时确为双向。
- **H3**：无法判定。

本步骤用 ckpt 自带的证据在 H1 / H2 / H3 之间判定，据此决定后续推断语义。本步骤**只判定，不改模型代码**。

## 已有资源（标明位置）

| 资源 | 位置 | 用途 |
|---|---|---|
| 主 checkpoint | `/data-tangc/models/map_repr/predictor/epoch_3.pt` | 判定对象（`model_state_dict`；可能含 `optimizer_state_dict`） |
| MAP-KG 编码器 | `/data-tangc/models/map_repr/mapkg_encoder/mapkg_encoder_v3.pt` | `smiles_projector.*` 与 `drug_projector.*` 对照 |
| 训练脚本 | `MAP/train.py`（L72-86 loss；L236 优化器；L374-379 保存） | 优化器类型、保存内容 |
| 主干代码 | `MAP/model/transformer_encoder.py`（`LlamaBidirectionalModel`、`NoRoPE`）、`MAP/model/pert_ca.py` | 构造同结构模块做初始化基准 |
| 组件 | `MAP/model/components.py`（`ResidualProjector` 等） | 同上 |
| 运行环境 | `.venv`（Python 3.11.14、torch 2.8.0+cu128、transformers 4.43.4） | 全部计算 |
| demo 输入 / 既有结果 | `work/minimal_dataset/`、`work/exp32d/`、`plan/05` | 功能对照 |

## 已完成部分（Lead 静态核查，2026-10-27）

| # | 证据 | 来源 |
|---|---|---|
| E1 | `requirements.txt:277` 定 `transformers==4.30.1` | 仓库 |
| E2 | 4.30.1 / 4.31.0：`LlamaRotaryEmbedding` 用 `register_buffer("inv_freq", inv_freq)`（持久）；`LlamaModel.forward` 每次调用 `_prepare_decoder_attention_mask` 生成因果 mask；没有 `_update_causal_mask`。4.37.2：`inv_freq` 已改为 `persistent=False`，仍没有 `_update_causal_mask`。4.38.0 起才有 `_update_causal_mask`。4.43.4：`LlamaModel` 有模型级 `self.rotary_emb`，经 `position_embeddings` 传入各层 | 各 tag 的 transformers `models/llama/modeling_llama.py` |
| E3 | ckpt 每层都有 `self_attn.rotary_emb.inv_freq`（不补就报 4 个 unexpected 键，TODO §8.4-3）。推断链：保存 ckpt 的版本里 `inv_freq` 是持久 buffer → 该版本没有 `_update_causal_mask` → `transformer_encoder.py:91` 的 override 不生效 → 训练时为因果注意力（推断） | E2 + ckpt |
| E4 | `transformer_encoder.py:128` 注释掉了 `cache_position=cache_position`；4.30.1 的 `LlamaModel.forward` 没有该参数 | 仓库（旁证） |
| E5 | 前向只取 `last_hidden_state[:, 0]`（`pert_ca.py` forward），`train.py:72-86` 的 loss 只用这一路。若为因果，位置 0 只看得到自己，drug / gene token、`q_proj`、`k_proj`、`gene_tokens_projector` 的梯度恒为 0 | 仓库 |
| E6 | `train.py:236` 用 `optim.Adam(..., lr=args.lr)`，**无 weight decay**，所以梯度恒为 0 的参数一步不动，仍是初始化样本。`train.py:374-379` 保存时带 `optimizer_state_dict` | 仓库 |
| E7 | 4.43.4 下 `NoRoPE` 替换了模型级 rotary，所以当前推断不用 RoPE；4.30.1 下 `NoRoPE` 不被调用，训练时 RoPE 开着。因果语义下位置 0 的旋转角为 0，不受 RoPE 影响 | E2 + 仓库 |

H1 的可检验预测：

- **P1**：若 ckpt 含优化器状态，4 层的 `q_proj` / `k_proj` 与 `gene_tokens_projector` 的 `exp_avg_sq` 全为 0；对照组（`v_proj`、`o_proj`、MLP、`cell_projector`、`project_out`、`gene_decoder`）> 0。
- **P2**：上述参数的尺度无关统计量与同代码新初始化一致，对照组不一致。
- **P3**：在因果语义下，输出与药物完全无关（逐位相同）。

## 待获取资源

无（全部在 b200 本地）。

## 待完成实现

约束：

- remote-tmux 仅用于执行命令和在 b200 上直接创建/编辑文件；禁止用 `run.sh + base64` 在本地与 b200 之间传文件，禁止 rclone。
- 如确需在本地与 b200 之间传文件，只用 scp/sftp；免交互认证不可用时停止并报告 Lead，不反复尝试登录。
- 不修改 remote-tmux skill；不下载阶段二资源。
- 分析脚本放 `analysis/r15/`，产物放 `results/r15/`，原始前向输出放 `work/r15/`，日志写仓库根 `RUNLOG.md`。
- 不改 `MAP/` 下任何文件，不 commit / push。
- GPU 只用 0 号。
- 每个功能对比条件单独起进程（进程内连续前向有 ~3e-3 抖动，见 TODO §8.4-6）。
- 长任务使用新的任务名，并记录 `send.sh` 返回的 job ID；用 `job-status.sh <job-id>` 读取记录的 rc。launch accepted 或 pane 末尾本身不等于完成。
- 每一步在 RUNLOG 记录命令、起止时间、执行脚本 SHA-256、job ID、记录的 rc 和产物清单。

步骤：

1. **S0 环境记录**：在 `RUNLOG.md` 新建本任务条目。记录 `git rev-parse HEAD`、`git status --short`、Python / torch / transformers 版本、GPU 编号。
2. **S1 ckpt 结构与优化器状态**：列出 `epoch_3.pt` 顶层键。若有 `optimizer_state_dict`：
   - 参数顺序以 `model_state_dict` 键顺序为准，用当前模型的 `named_buffers()` 剔除 buffer。
   - 把这个顺序与 `state` 各下标 `exp_avg` 的形状序列、`param_groups[0]['params']` 的长度逐一对齐，得到 下标 → 参数名 的映射。对齐不唯一或不一致即停止，并写明原因。
   - 对每个参数输出 `exp_avg_sq` 的 `max`、非零元素数、元素总数，以及 `exp_avg` 的 `max|·|`。
   - 产物：`results/r15/optimizer_state.csv`、`results/r15/param_alignment.txt`（逐行：下标、参数名、形状）。
   - 若没有优化器状态：在 RUNLOG 写明，跳过 S1。
3. **S2 权重统计**：
   - 范围：`pert_model` 下除 `smiles_encoder.*` 与 `embed_tokens` 外的所有二维 `weight`。
   - 统计量：mean、std、超额峰度 κ、`max|w|`、σ_max、MP 比值 r = σ_max /(std·(√m+√n))。
   - 基准：用与 `pert_ca.py` 相同的构造参数（含 `head_dim=128`），以 `torch.manual_seed(0..4)` 各新建一份初始化，算同样的统计量。
   - 脚本须固定种子与线程数，保证复跑逐字节一致。
   - 产物：`results/r15/weight_stats.csv`，列为 `param, source∈{ckpt,init0..init4}, shape, mean, std, kurt, maxabs, smax, r`。
4. **S3 drug_projector 来源**：逐张量比较 `pert_model.drug_projector.*` 与 `mapkg_encoder_v3.pt` 的 `smiles_projector.*`，记录形状、是否完全相等、`max|Δ|`。产物：`results/r15/drug_projector_vs_mapkg.csv`。结果只作佐证，不进入判定规则（原训练中它是否可训练未知）。
5. **S4 功能对照**：
   - 环境：GPU0，`demo.py` 原样，`--seed 0 --drug_conc 0.5 --set_size 24 --cell_line CVCL_0023`。三种药（吡哆醇、布洛芬、咖啡因）的 SMILES 取自 `work/exp32d/{seed0_a,drug_ibu,drug_caf}/meta.json`。
   - 条件 a：现状（双向）。
   - 条件 b：因果。用运行时包装（monkeypatch）让主干的 `_update_causal_mask` 返回标准因果加性 mask（上三角为 dtype 最小值），**不改源码**。
   - 输出：每个条件内三种药两两的 `max|Δ|`；两个条件之间（同药）的 `max|Δ|` 与 Pearson。
   - 六个前向必须全部 recorded rc=0；驱动脚本遇错停止，不允许部分失败后仍返回整体 rc=0。每次保存独立日志和 `diag.json`，确认条件 a 的位置 0 可见全部 2049 个位置，条件 b 的位置 0/1 分别仅可见 1/2 个位置。
   - 产物：`results/r15/functional.json`，原始输出放 `work/r15/`。
6. **S5 结论**：按下方判定规则写 `results/r15/verdict.md`，写明 H1 / H2 / H3、触发判定的具体数值和 S3 佐证，并补 RUNLOG。Implement 交付后不得自行宣布完成，由 Review 按本文件复核。

### 判定规则（预注册）

分组：

- **对照组 C**：4 层的 `v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`，以及 `cell_projector`、`project_out`、`gene_decoder` 的全部二维权重。
- **目标组 T**：4 层的 `q_proj`、`k_proj`（共 8 个），以及 `gene_tokens_projector` 的全部二维权重。

判据一，**优化器判据**（S1 有数据时优先使用）：

- 判为 H1：T 中每个参数的 `exp_avg_sq` 非零元素数为 0，且 C 中每个参数的非零元素数 > 0。
- 判为 H2：8 个 `q_proj` / `k_proj` 中 ≥ 6 个的非零元素数 > 0。
- 其余判为 H3。

判据二，**统计判据**（S1 无数据时使用）：

- "像初始化"的定义：对每个参数，以 5 份初始化的 [min, max] 为基准带。κ 的容差取 `max(3×(max−min), 0.02)`，r 的容差取 `max(3×(max−min), 0.01)`。κ 与 r 都落在"基准带 ± 容差"内，记为像初始化。
- 检验有效的前提：C 中 ≥ 50% 的参数**不**像初始化。检验无效则判为 H3。
- 检验有效时：T 全部像初始化，判为 H1；8 个 q/k 中 ≥ 6 个不像初始化，判为 H2；其余判为 H3。

判据三，**功能判据**（必须满足，否则 S4 作废重做）：条件 b 下三种药两两的 `max|Δ|` 必须为 `0.0`。

## 验收方式

在 b200 仓库根 `/data-tangc/dev/ptb/map-repro` 执行，全部满足才算交付：

1. 产物存在且非空：`test -s RUNLOG.md && test -s results/r15/weight_stats.csv && test -s results/r15/drug_projector_vs_mapkg.csv && test -s results/r15/functional.json && test -s results/r15/verdict.md`。若 ckpt 含优化器状态，还须 `test -s results/r15/optimizer_state.csv && test -s results/r15/param_alignment.txt`。
2. 复跑一致：`.venv/bin/python analysis/r15/weight_stats.py --out-dir /tmp/r15_rerun && cmp results/r15/weight_stats.csv /tmp/r15_rerun/weight_stats.csv`。
3. 覆盖完整：`weight_stats.csv` 中 `source=ckpt` 的行与 S2 范围内的 ckpt 键一一对应，每个键另有 `init0..init4` 五行。
4. 功能对照：
   - 六个前向的 recorded rc 均为 0，六个 `diag.json` 均满足上述 mask 检查。
   - `functional.json` 中条件 b 的三组换药 `max|Δ|` 均为 `0.0`。
   - 条件 a 的三个 `output_pred_hvg.npy` 与 `work/exp32d/{seed0_a,drug_ibu,drug_caf}/` 中同名文件 md5 相同。不同则须在 RUNLOG 说明原因，由 Review 判定是否可接受。
5. 代码与结论：`git status --short MAP/` 与 S0 记录相比无新增改动；`verdict.md` 写明 H1 / H2 / H3 及触发判定的数值。

## 失败回退

- **S1 对齐失败**：停止 S1，在 RUNLOG 写明原因，改用统计判据，不猜映射。
- **统计检验无效或结论为 H3**：补做 S6，用 hook 计算条件 a 下位置 0 对各 key 的注意力熵（每层每头），与 log 2049 ≈ 7.625 比较。结果交回 Lead，由 Lead 与用户定下一步（包括是否向作者索要训练环境信息）。
- **GPU0 被占或 OOM**：在 RUNLOG 记录并等待，不擅自改用 GPU 4–7。
- **后续处置**：Review 通过后由 Lead / 用户决定，不在本任务内执行：
  - H1：发布 ckpt 视为与药物无关；另开任务把默认推断改为因果，双向保留为显式开关；3.3 注明口径；整理证据向作者提 issue；同步更新 TODO R15/R17、`plan/05`、`design/architecture.md`。
  - H2：回到原 R15 方案（量 token 范数、看位置 0 对 drug 的注意力、对照 `Fig_3c`）。

## 完成结果与复核（2026-10-27）

状态：✅ 已完成，结论 **H1——发布 `epoch_3.pt` 训练时主干为因果注意力**。本步骤未修改
`MAP/` 源码，也未改变当前默认推断；现有全零加性 mask 仅作为双向对照。

核心证据：

- 优化器判据：T 12/12 的 `exp_avg_sq` 非零元素数为 0、`exp_avg` maxabs 为 0；C 29/29
  非零。8 个 q/k 中有非零更新的为 0/8。
- 统计交叉校核：C 28/29 不像初始化，满足检验有效条件；T 12/12 像初始化，q/k 0/8
  不像初始化，同样指向 H1。
- 功能判据：因果条件 b 的三组换药 `max|Δ|` 全为 0.0；双向条件 a 与 `work/exp32d`
  基线 MD5 全同。六个有效前向 recorded rc 均为 0，mask diag 通过。
- 映射证明：Lead 独立 DP 重算确认完整 499 键有 4,738,802,796 个保序形状放置；
  `pert_model` 减 buffer 后为 12 个，唯一歧义为组外一维 bias slot 7，T/C 名称均固定。
  SE 排除由训练代码默认 `freeze_se=True` 和 optimizer 的 `requires_grad` 过滤直接证明，
  238/239 位相等仅作旁证；`_bin_indices_cached` 是 int64 buffer。

复核历史：独立 Review 首轮给出 F1/F2，Implement 已完成修订。第二轮 Review 因 302.AI
余额不足（HTTP 401）未能运行，Lead 按用户明确授权接替，独立重算 DP、核查训练代码、
job/rc、哈希和 RUNLOG 后通过。详见 `RUNLOG.md` 的 Lead re-review 小节；其中保留了关于
SE 排除措辞和少量历史时间精度的非数值残余说明。

产物：`analysis/r15/`、`results/r15/`、`work/r15/`、`RUNLOG.md`。原始 `work/r15/`
保留在 b200，不纳入 Git。后续默认推断语义调整、双向显式开关及作者 issue 均为另开任务，
不由本步骤自动执行。
