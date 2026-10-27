"""One demo.py forward under condition a (as shipped, bidirectional) or b (causal).

Condition b is a runtime monkeypatch only, applied lazily on the first MAPmodel.forward
(never on __init__, because Lightning save_hyperparameters() parses __init__ frames):
the backbone _update_causal_mask returns the standard causal additive mask (strict upper
triangle = finfo(dtype).min) and the attention modules get is_causal=True. No file under
MAP/ is touched.

A forward pre-hook on layer0.self_attn captures the additive mask that the attention
module actually receives, so the mask claim is measured, not assumed. Writes <diag> and
exits 3 when the mask check for the requested condition fails.
"""
import argparse
import json
import os
import sys
import types

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO, "MAP")
if MAP_DIR not in sys.path:
    sys.path.insert(0, MAP_DIR)

CKPT = "/data-tangc/models/map_repr/predictor/epoch_3.pt"
SE_CKPT = "/data-tangc/models/map_repr/state_se600m/se600m.safetensors"
SE_CONFIG = "configs/se600m.yaml"
DATA_BASE = "../work/minimal_dataset/preprocessed"
SE_INPUTS = "../work/minimal_dataset/preprocessed_se_inputs_memmap"

CAPTURED = {}


def make_causal_updater():
    def _causal(self_, attention_mask, input_tensor, cache_position, past_key_values,
                output_attentions=False):
        q = int(input_tensor.shape[1])
        minv = torch.finfo(input_tensor.dtype).min
        m = torch.full((q, q), minv, dtype=input_tensor.dtype, device=input_tensor.device)
        m = torch.triu(m, diagonal=1)
        return m[None, None]

    return _causal


def install(condition):
    from model.model import MAPmodel

    orig_forward = MAPmodel.forward

    def apply_once(self):
        backbone = self.pert_model.transformer_backbone
        if condition == "b":
            for layer in backbone.layers:
                if hasattr(layer.self_attn, "is_causal"):
                    layer.self_attn.is_causal = True
            backbone._update_causal_mask = types.MethodType(make_causal_updater(), backbone)
        attn0 = backbone.layers[0].self_attn

        def hook(module, hargs, hkwargs):
            CAPTURED["mask"] = hkwargs.get("attention_mask", None)
            CAPTURED["is_causal_flag"] = getattr(module, "is_causal", None)

        attn0.register_forward_pre_hook(hook, with_kwargs=True)
        self._r15_installed = True

    def patched_forward(self, ctrl_src, ctrl_counts, smiles, concs=None):
        if not getattr(self, "_r15_installed", False):
            apply_once(self)
        return orig_forward(self, ctrl_src=ctrl_src, ctrl_counts=ctrl_counts,
                            smiles=smiles, concs=concs)

    MAPmodel.forward = patched_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["a", "b"])
    ap.add_argument("--drug-tag", required=True)
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--se-ckpt", default=SE_CKPT)
    ap.add_argument("--se-config", default=SE_CONFIG)
    ap.add_argument("--data-base-dir", default=DATA_BASE)
    ap.add_argument("--se-inputs-base-dir", default=SE_INPUTS)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    diag_dir = os.path.dirname(os.path.abspath(args.diag))
    if diag_dir:
        os.makedirs(diag_dir, exist_ok=True)

    import demo

    install(args.condition)

    sys.argv = [
        "demo.py",
        "--ckpt", args.ckpt,
        "--cell_line", "CVCL_0023",
        "--drug_smiles", args.smiles,
        "--drug_conc", "0.5",
        "--set_size", "24",
        "--seed", "0",
        "--se_ckpt", args.se_ckpt,
        "--se_config", args.se_config,
        "--data_base_dir", args.data_base_dir,
        "--se_inputs_base_dir", args.se_inputs_base_dir,
        "--output_dir", args.out_dir,
    ]
    print("[forward] condition=%s drug=%s smiles=%s"
          % (args.condition, args.drug_tag, args.smiles), flush=True)
    demo.main()

    mask = CAPTURED.get("mask", "missing")
    info = {
        "condition": args.condition,
        "drug_tag": args.drug_tag,
        "smiles_arg": args.smiles,
        "attn_is_causal_flag": CAPTURED.get("is_causal_flag"),
        "mask_present": isinstance(mask, torch.Tensor),
        "seq_len": None,
        "mask_shape": None,
        "mask_dtype": None,
        "mask_kind": None,
        "mask_min": None,
        "mask_max": None,
        "visible_row0": None,
        "visible_row1": None,
        "visible_rows": None,
    }

    ok = False
    if isinstance(mask, torch.Tensor):
        info["mask_shape"] = [int(s) for s in mask.shape]
        info["mask_dtype"] = str(mask.dtype)
        info["seq_len"] = int(mask.shape[-1])
        row = mask[0, 0]
        vis = (row > -1e30).sum(dim=1).tolist()
        info["visible_rows"] = [int(v) for v in vis]
        info["visible_row0"] = int(vis[0])
        info["visible_row1"] = int(vis[1])
        mmin = float(mask.min())
        mmax = float(mask.max())
        info["mask_min"] = mmin
        info["mask_max"] = mmax
        if mmax == 0.0 and mmin == 0.0:
            info["mask_kind"] = "zero_additive_bidirectional"
        elif mmin < 0.0:
            info["mask_kind"] = "additive_with_blocked_entries"
        else:
            info["mask_kind"] = "unknown"
        if args.condition == "a":
            ok = info["visible_row0"] == 2049 and info["visible_row1"] == 2049
            info["expect"] = "row0 == 2049 (all positions visible)"
        else:
            ok = info["visible_row0"] == 1 and info["visible_row1"] == 2
            info["expect"] = "row0 == 1, row1 == 2 (strictly causal)"
        del mask
    else:
        info["expect"] = "additive mask must be present"
        info["mask_kind"] = "none"

    info["mask_check_pass"] = bool(ok)
    with open(args.diag, "w") as f:
        json.dump(info, f, indent=2, sort_keys=True)
    print("[diag] " + json.dumps({k: info[k] for k in [
        "condition", "drug_tag", "mask_kind", "visible_row0", "visible_row1", "mask_check_pass"]}), flush=True)
    if not ok:
        print("[diag] MASK CHECK FAILED for condition " + args.condition, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
