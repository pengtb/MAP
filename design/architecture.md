# MAP 架构说明（据仓库代码 + checkpoint 实测结构）

> 资源根目录：`/data-tangc/models/map_repr/`
> 本文所有形状来自 `epoch_3.pt` / `mapkg_encoder_v3.pt` 的实测键与代码里的 `forward`。

## 1. 两个阶段

```
阶段二（知识侧，可独立复现）                      阶段一（预测器，已发布权重）
──────────────────────────────                ──────────────────────────────
MAP-KG（14 源, 187k drugs / 23k genes）        控制细胞 bulk → STATE SE-600M → {gene tokens T, cell emb h}
   │                                            │
   ├─ Φ_text  = BioBERT        ┐                ├─ cell_projector(h)          → cell token
   ├─ Φ_mol   = MoleculeSTM    ├→ 对比对齐      ├─ gene_tokens_projector([T; ESM2]) → gene tokens
   └─ Φ_seq   = ESM-2 650M     ┘   → 融合       └─ Φ_mol(冻结)+drug_projector → drug token
        ↓                                                      ↓
   知识编码器 mapkg_encoder_v3.pt                     [CLS | Drug | 2047 Genes] → Llama 4 层
                                                              ↓
                                                    project_out → gene_decoder → 2000 HVG 表达
```

## 2. 模块与资源对应（关键表）

| 模块 | 代码位置 | 形状/规模 | 权重来源（文件） |
|---|---|---|---|
| 基因 ESM-2 嵌入表 `pe_embedding` | `MAP/model/se.py`、`model.py:44` | `19790 × 5120` | `esm2_gene/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` |
| 细胞基础模型 SE（STATE SE-600M） | `MAP/model/se.py`（`StateEmbeddingModel`） | d_model 2048，16 层，16 头（代码硬编码；config 写 8） | `state_se600m/se600m.safetensors`（加载时排除 `pe_embedding`） |
| MolSTM 分子编码器 | `MAP/model/mega_molbart/` | BART，vocab 523，d=256，enc/dec 各 4 层 | `predictor/epoch_3.pt` 的 `pert_model.smiles_encoder._model.*`（176 键）；等价来源 `mapkg_encoder/mapkg_encoder_v3.pt` |
| 分词器 | `MAP/model/mega_molbart/tokenizer.py` | 523 token，chem start = 272 | `molecule_stm/bart_vocab.txt` |
| 药物投影 `drug_projector` | `MAP/model/components.py::ResidualProjector` | 256 → 1024（hidden 512） | `epoch_3.pt` 的 `pert_model.drug_projector.*`（10 键） |
| 细胞投影 `cell_projector` | `CellProjector(n_layers=4, d_cell=2048, d_out=1024)` | 2048 → 1024 | `pert_model.cell_projector.*`（8 键） |
| 基因 token 投影 `gene_tokens_projector` | `ResidualProjector(7168→1024)` | 2048+5120 → 1024 | `pert_model.gene_tokens_projector.*`（10 键） |
| 序列主干 `transformer_backbone` | `MAP/model/transformer_encoder.py::get_transformer_backbone('llama')` | 1024 维、**4 层**（ckpt `layers.0..3`，每层 10 张量 = 42 键）；`head_dim=128`（由 ckpt 的 q/k/v/o=(1024,1024) 与 `rotary_emb.inv_freq=(64,)` 反推，出厂默认 84 不匹配）；`embed_tokens` 32000×1024 且被清零冻结 | `pert_model.transformer_backbone.*`（42 键） |
| 输出投影 `project_out` | `ProjectOut(1024→2048)` | 1024 → 2048 | `pert_model.project_out.*`（4 键） |
| 表达解码 `gene_decoder` | `LatentToGeneDecoder(latent 2048, gene 2000, hidden [512,1024])` | 2048 → 2000 | `pert_model.gene_decoder.*`（10 键） |

## 3. 预测器前向（`demo.py` → `MAPmodel.forward` → `PerturbationEncoder_ca.forward`，`pert_ca.py`）

```
输入: ctrl_src [B,S,L] (基因 id), ctrl_counts [B,S,L] (表达), smiles [B], concs [B]
 1) src = pe_embedding(gene_ids)                 → [B*S, L, 5120]
 2) esm_tokens = src[:,1:,:]                     → [B*S, L-1, 5120]      # 留作基因侧原始特征
 3) src = normalize(src) ; 首位置换成 cls_token ; 末尾追加 dataset_token
 4) SE forward(src, counts) → gene_output[B*S,2049,2048], embedding[B*S,2048], _
 5) gene_output = gene_output[:,1:-1,:]          → [B*S, 2047, 2048]     # 去掉 cls 与 dataset token
 6) cell_tokens  = cell_projector(embedding)     → [B*S, 1024]
 7) drug_tokens  = drug_projector(Φ_mol(smiles)) → [B, 1024] → expand [B*S, 1024]
 8) gene_tokens  = gene_tokens_projector(cat([gene_output, esm_tokens], -1))
                                                 → [B*S, 2047, 1024]
 9) combined = cat([cell_token, drug_token, gene_tokens], dim=1)
                                                 → [B*S, 2049, 1024]     # 与 max_position_embeddings=2049 对齐
10) res = transformer_backbone(inputs_embeds=combined).last_hidden_state[:,0]
11) pred_tokens = project_out(cell_token + res)  → [B*S, 2048]
12) pred_hvgs   = gene_decoder(pred_tokens)      → [B*S, 2000] → [B, S, 2000]
```

**关键点**：`combined` 长度恰为 **2049**（1 cell + 1 drug + 2047 gene），与主干配置 `max_position_embeddings=2049` 一致；主干只用 `inputs_embeds`，因此 `embed_tokens` 的 32000 词表不参与计算。

## 4. 知识预训练侧（阶段二）

`mapkg_encoder_v3.pt` 的 419 个张量给出该模块的组成：

| 前缀 | 键数 | 含义 |
|---|---|---|
| `text_encoder.*` | 200 | BioBERT（Φ_text） |
| `smiles_encoder.*` | 176 | MoleculeSTM BART（Φ_mol） |
| `fusion_module.*` | 12 | 跨模态融合 |
| `smiles_projector.*` | 10 | 药物侧投影 |
| `gene_projector.*` | 10 | 基因侧投影（来自 Φ_seq / ESM-2） |
| `text_projector.*` | 10 | 文本侧投影 |
| `logit_scale` | 1 | 对比学习温度 |

论文口径：冻结 ESM-2 与 MoleculeSTM，各接 **4 层可训练 MLP**；通过对比学习把三类模态对齐到统一嵌入空间，产出机制感知的药物/基因表示。

## 5. checkpoint 键 → 模块（`epoch_3.pt`，499 张量）

| 顶层前缀 | 键数 | 组成 |
|---|---|---|
| `pert_model.*` | 260 | `smiles_encoder` 176 / `transformer_backbone` 42 / `drug_projector` 10 / `gene_tokens_projector` 10 / `gene_decoder` 10 / `cell_projector` 8 / `project_out` 4 |
| `se.*` | 239 | SE 主干（cls_token / dataset_token / layers / …）；`pe_embedding` 由外部嵌入表提供 |

## 6. 复现注意

| 项 | 说明 |
|---|---|
| `embed_tokens` 被清零 | `get_transformer_backbone` 里 `embed_tokens.weight.zero_()` 且 `requires_grad=False` → 32000 词表与结果无关 |
| `nlayers` 冲突 | `configs/se600m.yaml` 写 8，代码硬编码 16，ckpt 与代码一致（16）→ 以代码为准 |
| `hvg_info=None` | 发布 ckpt 训练时未用 HVG 信息 → 3.1/3.2 不需要 HVG 列表 |
| **训练因果 / 当前双向对照** | `transformers 4.4x` 的 SDPA 里 `is_causal = True if causal_mask is None and q_len > 1 else False`，仓库用 `_update_causal_mask` 返回 `None` 的做法在这些版本中等同因果注意力。`plan/08` 已由优化器、权重统计和功能对照判定发布 ckpt **训练时为因果注意力（H1）**：位置 0 只看自己，drug/gene 路径未获有效训练。`pert_ca.py` 的全零加性 mask（`[B,1,q,q]`）目前仅提供双向对照，其 ~3e-4 换药差异不应作为药物响应证据；默认推断改为因果、双向改为显式开关另开任务 |
| 标志位未初始化 | `model.py::forward` 读 `pert_model_hvg_ids_already_set`，上游 `__init__` 未定义（`hvg_info=None` 也会触发）→ 已补 `False` / `hvg_ids=None` |
| 键名差异 | ckpt 用 `pert_model.smiles_encoder.*` + `pert_model.drug_projector.*`；现有 `pert.py` 用 `kg_smiles_encoder.smiles_encoder.*` → 由 `pert_ca.py` 解决（已实现，见 `plan/03-reimplementation.md`） |
| RoPE 键 | ckpt 每层带 `self_attn.rotary_emb.inv_freq`：4.31.0 及以前它是持久 buffer，4.37.2 已改为非持久，所以新版 `state_dict` 里没有它。`pert_ca.py` 把它重新注册为持久 buffer（初值 0，`strict=True` 加载后为 ckpt 值），只为通过严格加载。4.43.4 下 `LlamaModel` 用模型级 `self.rotary_emb` 生成 `position_embeddings`，而 `LlamaBidirectionalModel` 把它换成 `NoRoPE`（cos=1、sin=0），所以当前推断**不使用 RoPE**，每层 `inv_freq` 不参与计算。4.30.1 中 `NoRoPE` 不会被调用，训练时 RoPE 是开着的（见 `plan/08`） |
| 冻结策略 | ESM-2 / MoleculeSTM 冻结；SE 在 `freeze_se=True` 时冻结 |