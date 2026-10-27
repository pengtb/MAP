"""Aggregate the six S4 forwards into results/r15/functional.json (plan/08 S4).

Reads work/r15/{a,b}_{seed0_a,drug_ibu,drug_caf}/ outputs plus each diag.json.
Computes, for pred_hvg and pred_embeddings:
  - within-condition pairwise max|delta| between the three drugs
  - between-condition per-drug max|delta| and Pearson r
  - md5 of output_pred_hvg.npy versus work/exp32d/*/output_pred_hvg.npy
  - the measured mask checks from each diag.json
Also embeds work/r15/forward_rc.json (recorded per-forward exit codes) when present.
Read-only with respect to MAP/ and the forward outputs.
"""
import argparse
import hashlib
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRUGS = [("seed0_a", "pyridoxine"), ("drug_ibu", "ibuprofen"), ("drug_caf", "caffeine")]
CONDS = ["a", "b"]


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(out_dir, name):
    return np.load(os.path.join(out_dir, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r15-dir", default=os.path.join(REPO, "work", "r15"))
    ap.add_argument("--exp32d-dir", default=os.path.join(REPO, "work", "exp32d"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "r15", "functional.json"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    arrays = {}
    diags = {}
    for c in CONDS:
        for tag, _common in DRUGS:
            d = os.path.join(args.r15_dir, "%s_%s" % (c, tag))
            arrays[(c, tag, "hvg")] = load(d, "output_pred_hvg.npy")
            arrays[(c, tag, "emb")] = load(d, "output_pred_embeddings.npy")
            with open(os.path.join(d, "diag.json")) as f:
                diags["%s_%s" % (c, tag)] = json.load(f)

    def maxabs(x, y):
        return float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))

    def pearson(x, y):
        a = x.reshape(-1).astype(np.float64)
        b = y.reshape(-1).astype(np.float64)
        return float(np.corrcoef(a, b)[0, 1])

    result = {
        "meta": {
            "cell_line": "CVCL_0023",
            "seed": 0,
            "drug_conc_uM": 0.5,
            "set_size": 24,
            "dtype_compared": "float64",
            "note": "condition a = shipped bidirectional; condition b = runtime causal monkeypatch",
        },
        "drugs": [{"tag": t, "name": n} for t, n in DRUGS],
        "within_condition": {},
        "between_condition": {},
        "mask_checks": {},
        "md5": {"condition_a_vs_exp32d": {}, "condition_b_vs_exp32d": {}},
    }

    for c in CONDS:
        block = {"hvg": {}, "emb": {}}
        for kind, key in (("hvg", "hvg"), ("emb", "emb")):
            for i in range(len(DRUGS)):
                for j in range(i + 1, len(DRUGS)):
                    t1 = DRUGS[i][0]
                    t2 = DRUGS[j][0]
                    name = "%s|%s" % (t1, t2)
                    block[key][name] = maxabs(arrays[(c, t1, kind)], arrays[(c, t2, kind)])
        result["within_condition"][c] = block

    for tag, _n in DRUGS:
        entry = {}
        for kind in ("hvg", "emb"):
            x = arrays[("a", tag, kind)]
            y = arrays[("b", tag, kind)]
            entry[kind] = {"maxabs": maxabs(x, y), "pearson": pearson(x, y)}
        result["between_condition"][tag] = entry

    for k, d in diags.items():
        result["mask_checks"][k] = {
            "mask_kind": d.get("mask_kind"),
            "mask_shape": d.get("mask_shape"),
            "visible_row0": d.get("visible_row0"),
            "visible_row1": d.get("visible_row1"),
            "mask_check_pass": d.get("mask_check_pass"),
            "attn_is_causal_flag": d.get("attn_is_causal_flag"),
        }

    for c in CONDS:
        for tag, _n in DRUGS:
            new = os.path.join(args.r15_dir, "%s_%s" % (c, tag), "output_pred_hvg.npy")
            old = os.path.join(args.exp32d_dir, tag, "output_pred_hvg.npy")
            nm = md5_of(new)
            om = md5_of(old) if os.path.exists(old) else None
            key = "condition_%s_vs_exp32d" % c
            result["md5"][key]["%s_%s" % (c, tag)] = {
                "run_md5": nm,
                "exp32d_md5": om,
                "match": (om is not None and nm == om),
            }

    rc_path = os.path.join(args.r15_dir, "forward_rc.json")
    if os.path.exists(rc_path):
        with open(rc_path) as f:
            result["recorded_forward_rc"] = json.load(f)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print("[S4] wrote " + args.out)
    print("[S4] within-condition b hvg maxabs: " + json.dumps(result["within_condition"]["b"]["hvg"], sort_keys=True))
    print("[S4] within-condition a hvg maxabs: " + json.dumps(result["within_condition"]["a"]["hvg"], sort_keys=True))
    for tag, _n in DRUGS:
        print("[S4] a vs b %s hvg: %s" % (tag, json.dumps(result["between_condition"][tag]["hvg"], sort_keys=True)))
    for k, v in result["md5"]["condition_a_vs_exp32d"].items():
        print("[S4] md5 %s match=%s" % (k, v["match"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
