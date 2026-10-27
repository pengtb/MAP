# 05 — 3.2 可复现性与生物学合理性

## 该计划简述

在 demo 前向（`plan/04`）跑通后，量化三件事：
1. 同一输入是否可复现（`demo.py` 原来用未设种子的 `np.random.choice` 选 control cells）；
2. 固定 seed 后是否逐位一致；
3. 输出是否随条件（剂量、药物）发生合理变化。

## 已有资源（标明位置）

| 资源 | 位置 | 用途 |
|---|---|---|
| 主 checkpoint | `/data-tangc/models/map_repr/predictor/epoch_3.pt` | 推理模型 |
| STATE 骨干 | `/data-tangc/models/map_repr/state_se600m/se600m.safetensors` | SE 编码器 |
| MolSTM 编码器 + 词表 | `mapkg_encoder/`、`molecule_stm/` | 药物编码 |
| 基因嵌入表 | `esm2_gene/` | `pe_embedding` |
| demo 输入 | `work/minimal_dataset/`（由仓库 zip 解压，CVCL_0023，32 个 control 细胞） | 24-cell pseudo-bulk |
| 实验产物 | `work/exp32d/`（9 个 run） | 结果对比 |

## 已完成部分

- ✅ 给 `demo.py` 增加 `--seed`（同时设 `random` / `np.random` / `torch` / `torch.cuda`），并在输出目录额外保存 `input_ctrl_row_indices.npy`、meta 中记录 `seed` 与 `control_row_indices`
- ✅ **修复上游双向注意力失效**：`transformers 4.4x` 的 `LlamaSdpaAttention` 里写死了
  `is_causal = True if causal_mask is None and q_len > 1 else False`，
  仓库用"`_update_causal_mask` 返回 None"实现双向注意力的做法**完全失效**（等于因果注意力，位置 0 只看得到自己）。已在 `pert_ca.py` 中改为返回**全零加性 mask**
- ✅ 环境定版：`transformers==4.43.4`（5.20 下 `_update_causal_mask` 已不存在；4.43 下 rotary 每层存在但 `inv_freq` 是非持久 buffer，已在 `pert_ca.py` 里重新注册为持久 buffer 以匹配 ckpt）
- ✅ 修掉 `model.py::forward` 读未初始化属性 `pert_model_hvg_ids_already_set` 的问题
- ✅ 跑完 9 个 run 的实验矩阵（`work/exp32d/`）

## 实验结果（2026-10-27，GPU0）

| 组 | 设置 | 结果 |
|---|---|---|
| **A 无 seed** | 同输入 3 次（抽样不同） | 每次抽到的 cell 行不同；输出相关 **0.99996–0.99999**，最大绝对差 **0.27–0.35** |
| **B seed=0** | 同输入 2 次 | **逐位一致**（md5 相同、maxdiff = 0.0） |
| **C 剂量** | 0.1 / 0.5 / 5.0 μM（seed=0） | **完全一致**（md5 相同）——浓度不参与计算 |
| **D 药物** | 吡哆醇 / 布洛芬 / 咖啡因（seed=0, 0.5 μM） | 输出不同，但幅度极小：maxdiff **2.98e-4 / 3.44e-4** |

### 解读

1. **可复现性已解决**：`--seed` 固定后逐位一致；不设 seed 时差异来自 control-cell 抽样，量级 ~0.3（占值域 0–2.33 的 ~13%）。
2. **剂量无效是代码事实**：`PerturbationEncoder` 的 `encode_conc=False`，`concs` 从未进入 forward；ckpt 的 `args` 也印证同一设置。所以 demo 的 `--drug_conc` 只是被记录、不影响结果。
3. **药物信号偏弱（待查）**：把 drug 向量置零与真实值相比，输出差异仅 ~3e-4；把 gene tokens 置零则为 ~2%（均值 0.1528→0.1495）。药物 token 在 2049 长序列中被稀释，需在 3.3（指标级）或与论文图源数值对比时判断这是否符合发布权重应有的行为。

## 待获取资源

| 资源 | 用途 | 状态 |
|---|---|---|
| 每个条件（药物×剂量×细胞系）的**真实**表达谱 | 做"预测 vs 实测"的绝对对比 | ❌ demo fixture 只含 control cells，无 ground truth |
| 2000 HVG 有序基因列表 | 把 `[B,S,2000]` 映射回基因名、算 top-50 DEG | ❌ 缺（属 3.3 阻塞项） |

## 待完成实现

- ⬜ 追查药物信号偏弱：对照 `Fig_3c` 图源数值（同一药物在 A-549 的 Pearson）判断量级是否合理；必要时检查 drug token 是否应与 gene token 做同样的归一化
- ⬜ 若拿到 HVG 列表与真实表达谱，补做 top-50 DEG 方向一致性与 Pearson delta
