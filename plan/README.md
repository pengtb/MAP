# plan/ — MAP 复现分步计划

> 本目录取代原先的 `PLAN.md`。每个步骤一个文件，统一按五段结构写：
> **该计划简述 / 已有资源（标明位置）/ 已完成部分 / 待获取资源 / 待完成实现**

## 步骤索引

| 文件 | 步骤 | 状态 |
|---|---|---|
| `01-environment.md` | 运行环境（venv / torch / 依赖 / GPU 验证） | ✅ 已完成 |
| `02-resources.md` | 资源获取与落位（官方权重、词表、骨干、数据集） | 🔄 阶段一资源已齐；阶段二编码器（BioBERT / ESM-2 650M）待下 |
| `03-reimplementation.md` | 缺失代码的重新实现（`pert_ca.py` 等补丁） | ✅ 已完成（`strict=True` 全键匹配） |
| `04-demo-inference.md` | 3.1 demo 前向（最小闭环） | ✅ 已完成（rc=0，20.05 s / 12.7 GiB） |
| `05-reproducibility.md` | 3.2 可复现性与生物学合理性 | ✅ 已完成（seed 逐位一致；弱药物信号根因见 `08`） |
| `06-benchmark.md` | 3.3 论文指标级复现 | ⬜ 待做（受 HVG 列表与缺失文件限制） |
| `07-mapkg-pretraining.md` | 阶段二：MAP-KG 知识预训练 | ⬜ 待做（无外部阻塞；按用户要求暂缓下载） |
| `08-r15-attention-semantics.md` | R15：发布 ckpt 训练时的注意力语义判定（因果 vs 双向） | ✅ 已完成（H1 因果训练；Lead 复审通过） |

## 全局约定

| 项 | 值 |
|---|---|
| 代码仓库 | `/data-tangc/dev/ptb/map-repro` |
| 运行环境 | `/data-tangc/dev/ptb/map-repro/.venv`（Python 3.11.14） |
| 资源根目录 | `/data-tangc/models/map_repr/`（子目录按 encoder / 基础模型划分） |
| 大数据目录 | `/data-tangc2/`（raw / processed） |
| 任务清单 | `TODO.md`（执行视角，与本目录互补） |
| 架构说明 | `design/architecture.md` |

## 资源目录约定（`/data-tangc/models/map_repr/`）

| 子目录 | 内容 | 对应模块 |
|---|---|---|
| `predictor/` | `epoch_3.pt` | MAP 预测器主 checkpoint |
| `mapkg_encoder/` | `mapkg_encoder_v3.pt` | MAP-KG 预训练编码器（知识侧） |
| `molecule_stm/` | `bart_vocab.txt` | MoleculeSTM 分词器词表（523 token） |
| `state_se600m/` | `se600m.safetensors`（已就绪，sha256 校验通过） | STATE SE-600M 细胞基础模型 |
| `esm2_gene/` | `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` | 基因 ESM-2 嵌入表（19790×5120） |
| `biobert/` | 待获取 | MAP-KG 文本编码器 Φ_text |
| `esm2_protein/` | 待获取 | MAP-KG 蛋白序列编码器 Φ_seq |

## 常用命令

```bash
source /data-tangc/dev/ptb/map-repro/.venv/bin/activate
export UV_CACHE_DIR=/data-tangc/dev/ptb/.uv-cache
export HF_ENDPOINT=https://hf-mirror.com
```