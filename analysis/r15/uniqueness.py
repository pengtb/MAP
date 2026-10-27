"""Order-preserving shape-fit analysis for plan/08 F1 (R15 repair round 1).

The optimizer in epoch_3.pt was built from every trainable parameter of the whole model
(train.py:236), so the 239 `se.*` keys of model_state_dict cannot simply be dropped. This
module computes, with exact integer dynamic programming and no sampling or enumeration cap:

  * how many order-preserving placements of the optimizer exp_avg shape sequence exist
    inside a given candidate key sequence;
  * for every optimizer slot, the exact set of parameter names that slot can take in some
    placement (feasibility from prefix and suffix counts);
  * the bit-identity evidence that the `se.*` block never changed during training, which is
    what justifies excluding it.

Pure functions only, no side effects, no writes.
"""
import torch
from safetensors.torch import load_file


def fit_tables(key_shapes, target_shapes):
    """pre[j][k] = ways to place target[:j] within key[:k]; suf mirrors it from the right.

    A placement is order preserving (strictly increasing key indices) and matches shapes
    exactly. All counts are exact Python integers.
    """
    K = len(key_shapes)
    n = len(target_shapes)
    pre = [[0] * (K + 1) for _ in range(n + 1)]
    for k in range(K + 1):
        pre[0][k] = 1
    for j in range(1, n + 1):
        tj = target_shapes[j - 1]
        row = pre[j]
        prev = pre[j - 1]
        for k in range(1, K + 1):
            v = row[k - 1]
            if key_shapes[k - 1] == tj:
                v += prev[k - 1]
            row[k] = v
    suf = [[0] * (K + 1) for _ in range(n + 1)]
    for k in range(K + 1):
        suf[n][k] = 1
    for j in range(n - 1, -1, -1):
        tj = target_shapes[j]
        row = suf[j]
        nxt = suf[j + 1]
        for k in range(K - 1, -1, -1):
            v = row[k + 1]
            if key_shapes[k] == tj:
                v += nxt[k + 1]
            row[k] = v
    return pre, suf, pre[n][K]


def slot_candidates(key_names, key_shapes, target_shapes, pre, suf):
    """For each optimizer slot, every key name that slot can take in some placement."""
    n = len(target_shapes)
    out = []
    for j in range(n):
        tj = target_shapes[j]
        cand = []
        for i in range(len(key_shapes)):
            if key_shapes[i] != tj:
                continue
            if pre[j][i] > 0 and suf[j + 1][i + 1] > 0:
                cand.append(key_names[i])
        out.append(cand)
    return out


def se_frozen_evidence(ckpt_state, se_path, prefix="se."):
    """Compare every `se.*` tensor of the checkpoint against the se600m initial weights.

    The name mapping mirrors MAPmodel.load_se_ckpt in MAP/model/model.py (encoder.0/1
    weight/bias are stored under both names across upstream versions).
    """
    sf = load_file(se_path)
    remap = {}
    for k, v in sf.items():
        if not k.startswith("pe_embedding"):
            remap[k] = v
    pairs = [
        ("encoder.0.weight", "gene_embedding_layer.0.weight"),
        ("encoder.0.bias", "gene_embedding_layer.0.bias"),
        ("encoder.1.weight", "gene_embedding_layer.1.weight"),
        ("encoder.1.bias", "gene_embedding_layer.1.bias"),
    ]
    for a, b in pairs:
        if a in sf and b not in remap:
            remap[b] = sf[a]
        if b in sf and a not in remap:
            remap[a] = sf[b]

    reverse = {}
    for a, b in pairs:
        reverse[b] = a
        reverse[a] = b

    keys = [k for k in ckpt_state.keys() if k.startswith(prefix)]
    identical = []
    differing = []
    no_ref = []
    for k in keys:
        short = k[len(prefix):]
        ref = sf.get(short)
        if ref is None:
            ref = remap.get(short)
        if ref is None and short in reverse:
            ref = sf.get(reverse[short])
        if ref is None:
            no_ref.append(short)
            continue
        cur = ckpt_state[k].cpu()
        ref = ref.cpu()
        if tuple(cur.shape) != tuple(ref.shape):
            differing.append((short, "shape %s vs %s" % (tuple(cur.shape), tuple(ref.shape))))
            continue
        if cur.dtype != ref.dtype:
            differing.append((short, "dtype %s vs %s" % (cur.dtype, ref.dtype)))
            continue
        if torch.equal(cur, ref):
            identical.append(short)
        else:
            d = float((cur.double() - ref.double()).abs().max())
            differing.append((short, d))
    return {
        "se_keys": len(keys),
        "identical": identical,
        "differing": differing,
        "no_reference": no_ref,
        "safetensors_keys": len(sf),
    }
