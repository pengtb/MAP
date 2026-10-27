# 01 — 运行环境

## 该计划简述

在 b200 上建立可复现的 Python 环境，使 `torch` 能在 Blackwell（sm_100）上执行，并满足 MAP 代码的依赖。

## 已有资源（标明位置）

| 资源 | 位置 | 说明 |
|---|---|---|
| Python 解释器 | `/data-tangc/dev/ptb/map-repro/.venv/bin/python` | CPython 3.11.14（uv 管理） |
| torch | 同上 site-packages | `2.8.0+cu128`，`archs` 含 `sm_100` |
| torchvision | 同上 | `0.23.0+cu128` |
| 依赖 | 同上 | numpy 2.4.6 / pandas 2.3.3 / scipy 1.17.1 / sklearn 1.9.1 / rdkit 2026.09.1 / omegaconf 2.3.1 / safetensors 0.8.0 / transformers 4.43.4（定版） / scanpy 1.11.5 / anndata 0.12.19 / lightning / torchmetrics / requests 2.34.2 |
| uv | `~/.local/bin/uv`（0.9.26） | 建环境与装包 |

## 已完成部分

- ✅ 用 uv 建 `.venv`（Python 3.11.14）；GH 直连被墙，改用 `UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone`
- ✅ 从 `download.pytorch.org/whl/cu128` 安装 torch 2.8.0 / torchvision 0.23.0
- ✅ GPU 验证：`torch.cuda.get_device_capability(0)=(10,0)`、`archs` 含 `sm_100`、8192² bf16 matmul 成功（0.181 s）
- ✅ 导入验证：`model.se` / `model.pert` / `model.smiles_kg_encoder_v3` / `model.mega_molbart.STMencoder_ddp` / `data.utils` 均可导入
- ✅ 依赖安装（见上表）

## 待获取资源

无（环境层面自足）。

## 待完成实现

- ✅ `model.model` 可导入（`pert_ca.py` 已实现，见 `03-reimplementation.md`）
- ✅ `transformers` 已定版 4.43.4：5.x 删除了 `LlamaModel._update_causal_mask`（仓库的双向注意力 override 变成死代码）；4.4x 的 SDPA 由 `causal_mask is None` 决定 `is_causal`（详见 plan/05）
- ⚠️ `requirements.txt` 定的训练环境是 `transformers==4.30.1`，与 4.43.4 的注意力 mask / RoPE 行为不同；训练时的实际语义见 `plan/08`
- ⚠️ `numpy 2.4.6` 仍新于代码时代，遇 `np.float_` 等按需替换
- ⬜ `flash_attn` / `apex` / `deepspeed` 未安装（训练路径可能需要，推理路径不需要）