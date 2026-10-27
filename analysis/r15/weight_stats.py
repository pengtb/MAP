"""S2: weight statistics for the released checkpoint vs fresh initializations (plan/08).

Scope: every 2-D `weight` under `pert_model` except `pert_model.smiles_encoder.*` and
`pert_model.transformer_backbone.embed_tokens.weight`.

Per tensor: mean, std (torch default, unbiased), excess kurtosis kurt, max|w|,
sigma_max, and the Marchenko-Pastur ratio r = sigma_max / (std * (sqrt(m)+sqrt(n))).

Baseline: five fresh PerturbationEncoder_ca initializations seeded torch.manual_seed(0..4)
with the constructor arguments used by model.py, including head_dim=128.

Determinism: single torch thread, float64 reductions, repr() float formatting, no
timestamps; a rerun into another --out-dir must be byte identical.

Names are full checkpoint keys (prefix pert_model.). Writes <out-dir>/weight_stats.csv:
  param,source,shape,mean,std,kurt,maxabs,smax,r
Read-only with respect to MAP/.
"""
import argparse
import csv
import gc
import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO, "MAP")
if MAP_DIR not in sys.path:
    sys.path.insert(0, MAP_DIR)

DEFAULT_CKPT = "/data-tangc/models/map_repr/predictor/epoch_3.pt"
MAPKG_CKPT = "/data-tangc/models/map_repr/mapkg_encoder/mapkg_encoder_v3.pt"
BART_VOCAB = "/data-tangc/models/map_repr/molecule_stm/bart_vocab.txt"
os.environ.setdefault("MAP_MOLECULE_CKPT", MAPKG_CKPT)
os.environ.setdefault("MAP_BART_VOCAB", BART_VOCAB)
PFX = "pert_model."
EXCLUDED_PREFIXES = (PFX + "smiles_encoder.",)
EXCLUDED_EXACT = (PFX + "transformer_backbone.embed_tokens.weight",)
SOURCES = ["ckpt", "init0", "init1", "init2", "init3", "init4"]


def in_scope(name, shape):
    if name in EXCLUDED_EXACT:
        return False
    for p in EXCLUDED_PREFIXES:
        if name.startswith(p):
            return False
    if not name.endswith(".weight"):
        return False
    return len(shape) == 2


def build_model(seed):
    from model.pert_ca import PerturbationEncoder_ca

    torch.manual_seed(seed)
    return PerturbationEncoder_ca(
        dim_emb=1024,
        encode_conc=False,
        num_heads=8,
        dropout=0.1,
        is_hvg_space=False,
        detach_decoder=False,
        transformer_backbone_key="llama",
        smile_encoder="PrimeKG_ver1",
    )


def tensor_stats(w):
    x = w.detach().to(torch.float64)
    m, n = int(x.shape[0]), int(x.shape[1])
    mean = x.mean()
    xc = x - mean
    m2 = (xc * xc).mean()
    kurt = (xc ** 4).mean() / (m2 * m2) - 3.0
    std = x.std()
    maxabs = x.abs().max()
    smax = torch.linalg.svdvals(x)[0]
    denom = float(std) * (math.sqrt(m) + math.sqrt(n))
    r = float(smax) / denom if denom != 0.0 else float("nan")
    return {
        "mean": float(mean),
        "std": float(std),
        "kurt": float(kurt),
        "maxabs": float(maxabs),
        "smax": float(smax),
        "r": r,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results", "r15"))
    ap.add_argument("--sources", default=",".join(SOURCES))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    print("[S2] ckpt: " + args.ckpt, flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    names = []
    shapes = {}
    shape_tuples = {}
    for k in state.keys():
        if not k.startswith(PFX):
            continue
        t = state[k]
        if len(tuple(t.shape)) != 2:
            continue
        if in_scope(k, tuple(t.shape)):
            names.append(k)
            shape_tuples[k] = tuple(int(d) for d in t.shape)
            shapes[k] = "x".join(str(int(d)) for d in t.shape)
    print("[S2] in-scope 2-D weights: %d" % len(names), flush=True)

    stats = {s: {} for s in sources}
    for k in names:
        st = tensor_stats(state[k])
        st["shape"] = shapes[k]
        stats["ckpt"][k] = st
    print("[S2] ckpt stats done", flush=True)
    del ckpt
    del state
    gc.collect()

    for s in sources:
        if s == "ckpt":
            continue
        seed = int(s[len("init"):])
        model = build_model(seed)
        msd = model.state_dict()
        for k in names:
            short = k[len(PFX):]
            if short not in msd:
                raise SystemExit("[S2] init state missing key " + short)
            t = msd[short]
            got = tuple(int(d) for d in t.shape)
            if got != shape_tuples[k]:
                raise SystemExit("[S2] shape mismatch for %s: %s vs %s" % (k, got, shape_tuples[k]))
            st = tensor_stats(t)
            st["shape"] = shapes[k]
            stats[s][k] = st
        del msd
        del model
        gc.collect()
        print("[S2] %s stats done" % s, flush=True)

    out = os.path.join(args.out_dir, "weight_stats.csv")
    rows = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "source", "shape", "mean", "std", "kurt", "maxabs", "smax", "r"])
        for k in names:
            for s in sources:
                st = stats[s][k]
                w.writerow([
                    k, s, st["shape"],
                    repr(st["mean"]), repr(st["std"]), repr(st["kurt"]),
                    repr(st["maxabs"]), repr(st["smax"]), repr(st["r"]),
                ])
                rows += 1
    print("[S2] wrote %s rows=%d (params=%d sources=%d)"
          % (out, rows, len(names), len(sources)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
