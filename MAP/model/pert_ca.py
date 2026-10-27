# Perturbation encoder matching the released epoch_3.pt checkpoint.
#
# The released checkpoint stores the drug branch as two *sibling* modules:
#     pert_model.smiles_encoder.*    (MolSTM BART, 176 tensors)
#     pert_model.drug_projector.*    (ResidualProjector 256->1024, 10 tensors)
# while MAP/model/pert.py wraps them inside KGSmilesEncoder_v3, producing
#     pert_model.kg_smiles_encoder.smiles_encoder.*  /  .smiles_projector.*
# so load_state_dict(strict=True) cannot succeed with pert.PerturbationEncoder.
#
# This module reproduces the checkpoint layout and implements the
# 'PrimeKG_ver1' branch that demo.py / train_sciplex.py select by default.
import os

import torch
import torch.nn as nn

from model.gene_decoders import LatentToGeneDecoder
from model.transformer_encoder import (
    get_transformer_backbone,
    ST_small_transformer_backbone_kwargs,
)
from model.mega_molbart.STMencoder_ddp import MolSTM_Extractor
from model.components import ResidualProjector, CellProjector, ProjectOut


class PerturbationEncoder_ca(nn.Module):
    def __init__(
        self,
        dim_emb=1024,
        hvg_dim=2000,
        encode_conc=False,
        num_heads=8,
        dropout=0.1,
        is_hvg_space=False,
        detach_decoder=False,
        transformer_backbone_key='llama',
        smile_encoder='PrimeKG_ver1',
        mol_ckpt=None,
        bart_vocab=None,
        backbone_kwargs=None,
    ):
        super().__init__()
        self.dim_emb = dim_emb
        self.hvg_dim = hvg_dim
        self.encode_conc = encode_conc
        self.num_heads = num_heads
        self.dropout = dropout
        self.is_hvg_space = is_hvg_space
        self.detach_decoder = detach_decoder
        self.transformer_backbone_key = transformer_backbone_key
        self.smile_encoder = smile_encoder

        self.cell_projector = CellProjector(
            n_layers=4, d_cell=2048, d_out=self.dim_emb, dropout=self.dropout)

        # ---- drug branch: two sibling modules (checkpoint layout) ----
        mol_ckpt = mol_ckpt or os.environ.get("MAP_MOLECULE_CKPT")
        bart_vocab = bart_vocab or os.environ.get("MAP_BART_VOCAB")
        self.smiles_encoder = MolSTM_Extractor(
            ckpt_path=mol_ckpt,
            vocab_path=bart_vocab,
        )
        self.drug_projector = ResidualProjector(
            input_dim=256, output_dim=self.dim_emb, hidden_dim=512)

        self.smiles_encoder.eval()
        for param in self.smiles_encoder.parameters():
            param.requires_grad = False
        for param in self.drug_projector.parameters():
            param.requires_grad = False

        self.gene_tokens_projector = ResidualProjector(
            input_dim=2048 + 5120,
            output_dim=self.dim_emb,
            hidden_dim=2048,
        )

        kw = ST_small_transformer_backbone_kwargs.copy()
        kw["max_position_embeddings"] = 2049
        kw["hidden_size"] = self.dim_emb
        # checkpoint: q/k/v/o_proj = (1024,1024) with 8 heads -> head_dim 128,
        # rotary inv_freq is (64,) = head_dim/2 (the shipped default 84 does not match).
        kw["head_dim"] = 128
        if backbone_kwargs:
            kw.update(backbone_kwargs)
        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            key=self.transformer_backbone_key,
            kwargs=kw,
        )
        self._ensure_rotary_buffers(kw["head_dim"])
        self._force_bidirectional_attention()
        self._dbg_drug = False

        self.project_out = ProjectOut(
            hidden_dim=self.dim_emb,
            output_dim=self.cell_projector.d_cell,
            dropout=self.dropout,
        )

        self.gene_decoder = LatentToGeneDecoder(
            latent_dim=2048,
            gene_dim=self.hvg_dim,
            hidden_dims=[512, 1024],
            dropout=self.dropout,
        )

    def _ensure_rotary_buffers(self, head_dim):
        """The checkpoint carries per-layer `self_attn.rotary_emb.inv_freq`.

        Newer transformers keeps RoPE at the model level, so those keys would be
        reported as unexpected. Add unused buffers with the same names/sizes so a
        strict load stays possible; NoRoPE already makes RoPE a no-op.
        """
        layers = getattr(self.transformer_backbone, "layers", None)
        if layers is None:
            return
        size = max(1, head_dim // 2)
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            rotary = getattr(attn, "rotary_emb", None)
            if rotary is None:
                holder = nn.Module()
                holder.register_buffer("inv_freq", torch.zeros(size))
                attn.add_module("rotary_emb", holder)
            else:
                # transformers registers rotary inv_freq as a *non-persistent* buffer,
                # so it never appears in state_dict while the released checkpoint has it.
                try:
                    rotary.register_buffer("inv_freq", torch.zeros(size), persistent=True)
                except Exception:
                    pass

    def _force_bidirectional_attention(self):
        """Llama's SDPA/flash paths pass `is_causal=self.is_causal` with a hard-coded
        True, so overriding `_update_causal_mask` alone does not make attention
        bidirectional. Flip the flag on every attention module.
        """
        layers = getattr(self.transformer_backbone, "layers", None) or []
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "is_causal"):
                attn.is_causal = False
        if hasattr(self.transformer_backbone, "is_causal"):
            self.transformer_backbone.is_causal = False

        # transformers 4.4x computes:
        #   is_causal = True if causal_mask is None and q_len > 1 else False
        # so returning None from _update_causal_mask still forces causal attention.
        # Return an all-zero additive mask instead -> full (bidirectional) attention.
        import types

        def _zero_additive_mask(self_, attention_mask, input_tensor, cache_position,
                                past_key_values, output_attentions=False):
            q = input_tensor.shape[1]
            return torch.zeros(input_tensor.shape[0], 1, q, q,
                               dtype=input_tensor.dtype, device=input_tensor.device)

        self.transformer_backbone._update_causal_mask = types.MethodType(
            _zero_additive_mask, self.transformer_backbone)

    def forward(self, gene_tokens, cls_tokens, esm_tokens, smiles, concs):
        B = len(smiles)
        BS = cls_tokens.shape[0]
        S = BS // B

        cell_tokens = self.encode_cells(cls_tokens)          # [B*S, dim_emb]
        drug_tokens = self.encode_drug(smiles)               # [B, dim_emb]
        gene_tokens = self.encode_genes(gene_tokens, esm_tokens)  # [B*S, 2047, dim_emb]

        drug_tokens = drug_tokens.unsqueeze(1).expand(B, S, -1).reshape(BS, -1)

        combined_tokens = torch.cat([
            cell_tokens.unsqueeze(1),
            drug_tokens.unsqueeze(1),
            gene_tokens,
        ], dim=1)                                            # [B*S, 2049, dim_emb]

        res_pred = self.transformer_backbone(inputs_embeds=combined_tokens).last_hidden_state
        res_pred = res_pred[:, 0, :]

        pred_tokens = self.project_out(cell_tokens + res_pred)
        pert_cell_counts_preds = self.gene_decoder(pred_tokens)

        pred_tokens = pred_tokens.reshape(B, S, -1)
        pert_cell_counts_preds = pert_cell_counts_preds.reshape(B, S, -1)
        return pred_tokens, pert_cell_counts_preds

    def encode_drug(self, smiles):
        assert ('' not in smiles) and ('xxx' not in smiles) and (None not in smiles), smiles
        if self.smile_encoder in ('MAP-KG', 'PrimeKG_ver1'):
            with torch.no_grad():
                out = self.drug_projector(self.smiles_encoder(smiles))
                if os.environ.get("MAP_DEBUG_DRUG"):
                    import hashlib, numpy as _np
                    _h = hashlib.md5(_np.ascontiguousarray(
                        out.detach().cpu().numpy()).tobytes()).hexdigest()[:10]
                    print(f"[debug-drug] smiles={smiles} md5={_h} norm={float(out.norm()):.4f}", flush=True)
                return out
        raise ValueError(f"Unsupported smile_encoder: {self.smile_encoder}")

    def encode_cells(self, ctrl_embs):
        return self.cell_projector(ctrl_embs)

    def encode_genes(self, gene_tokens, esm_tokens):
        combined = torch.cat([gene_tokens, esm_tokens], dim=-1)
        return self.gene_tokens_projector(combined)