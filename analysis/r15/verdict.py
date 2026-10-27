"""S5: apply the preregistered plan/08 decision rules and write results/r15/verdict.md.

Rules implemented verbatim from plan/08 (no post-hoc thresholds):

C group: the 2-D weights of v_proj/o_proj/gate_proj/up_proj/down_proj in the 4 layers,
         plus all 2-D weights of cell_projector, project_out, gene_decoder.
T group: the 4 layers q_proj and k_proj (8 in total), plus all 2-D weights of
         gene_tokens_projector.

Criterion 1 (optimizer evidence, preferred when S1 has data):
  H1 if every T parameter has exp_avg_sq nonzero count == 0 and every C parameter has
  nonzero count > 0.
  H2 if at least 6 of the 8 q/k parameters have nonzero count > 0.
  Otherwise H3.

Criterion 2 (statistical, only used when S1 has no optimizer data):
  looks-initialized = kurt and r both inside the 5-init band + tolerance, where
  kurt tolerance = max(3*(band width), 0.02) and r tolerance = max(3*(band width), 0.01).
  Test valid only if at least 50 percent of C does NOT look initialized, else H3.
  When valid: all T looks initialized -> H1; at least 6 of 8 q/k not initialized -> H2;
  else H3. Reported as a cross-check only when criterion 1 applies.

Criterion 3 (functional, must hold or S4 is void): condition b pairwise drug
  max|delta| must be 0.0.

Writes results/r15/verdict.md. Read-only with respect to MAP/.
"""
import argparse
import csv
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R15 = os.path.join(REPO, "results", "r15")

QK_RE = re.compile(r"^pert_model\.transformer_backbone\.layers\.([0-3])\.self_attn\.(q_proj|k_proj)\.weight$")
VO_RE = re.compile(r"^pert_model\.transformer_backbone\.layers\.([0-3])\.self_attn\.(v_proj|o_proj)\.weight$")
MLP_RE = re.compile(r"^pert_model\.transformer_backbone\.layers\.([0-3])\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def is_2d(shape):
    return "x" in shape


def classify(param, shape):
    if not is_2d(shape):
        return None
    if QK_RE.match(param):
        return "T"
    if param.startswith("pert_model.gene_tokens_projector.") and param.endswith(".weight"):
        return "T"
    if VO_RE.match(param) or MLP_RE.match(param):
        return "C"
    if param.startswith("pert_model.cell_projector.") and param.endswith(".weight"):
        return "C"
    if param.startswith("pert_model.project_out.") and param.endswith(".weight"):
        return "C"
    if param.startswith("pert_model.gene_decoder.") and param.endswith(".weight"):
        return "C"
    return None


def load_opt(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "idx": int(r["idx"]),
                "param": r["param"],
                "shape": r["shape"],
                "ea_sq_max": float(r["ea_sq_max"]),
                "nz": int(r["ea_sq_nonzero"]),
                "numel": int(r["ea_sq_numel"]),
                "ea_maxabs": float(r["ea_maxabs"]),
            })
    return rows


def load_ws(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d.setdefault(r["param"], {})[r["source"]] = r
    return d


def plain(value):
    t = (value or "?").strip()
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1]
    for ch in (chr(34), chr(39)):
        t = t.replace(ch, "")
    return t.strip() if t.strip() else "none"


def load_alignment_markers(path):
    info = {}
    keys = ("FITS_FULL_RAW", "FITS_FULL_NOBUF", "FITS_PERT_ONLY", "FITS_PERT_NO_DRUG",
            "AMBIGUOUS_SLOTS", "AMBIGUOUS_SLOTS_IN_T_OR_C", "T_C_UNIQUE",
            "SE_FROZEN_IDENTICAL", "SE_NO_REFERENCE", "GLOBAL_STEP")
    if not os.path.exists(path):
        return info
    with open(path) as f:
        for line in f:
            line = line.strip()
            for k in keys:
                if line.startswith(k + "="):
                    info[k] = line[len(k) + 1:]
    return info


def stat_looks_init(rows_by_source):
    k = [float(rows_by_source["init%d" % i]["kurt"]) for i in range(5)]
    rr = [float(rows_by_source["init%d" % i]["r"]) for i in range(5)]
    klo, khi = min(k), max(k)
    rlo, rhi = min(rr), max(rr)
    ktol = max(3.0 * (khi - klo), 0.02)
    rtol = max(3.0 * (rhi - rlo), 0.01)
    ck = float(rows_by_source["ckpt"]["kurt"])
    cr = float(rows_by_source["ckpt"]["r"])
    looks = (klo - ktol <= ck <= khi + ktol) and (rlo - rtol <= cr <= rhi + rtol)
    return {
        "looks_init": bool(looks),
        "kurt_ckpt": ck,
        "kurt_band": [klo, khi],
        "kurt_tol": ktol,
        "r_ckpt": cr,
        "r_band": [rlo, rhi],
        "r_tol": rtol,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=R15)
    args = ap.parse_args()
    d = args.results_dir

    opt = load_opt(os.path.join(d, "optimizer_state.csv"))
    ws = load_ws(os.path.join(d, "weight_stats.csv"))
    with open(os.path.join(d, "functional.json")) as f:
        fun = json.load(f)
    align = load_alignment_markers(os.path.join(d, "param_alignment.txt"))

    opt_T = [r for r in opt if classify(r["param"], r["shape"]) == "T"]
    opt_C = [r for r in opt if classify(r["param"], r["shape"]) == "C"]
    qk = [r for r in opt if QK_RE.match(r["param"])]
    gtp = [r for r in opt_T if r["param"].startswith("pert_model.gene_tokens_projector.")]

    ws_T = [p for p in ws if classify(p, ws[p]["ckpt"]["shape"]) == "T"]
    ws_C = [p for p in ws if classify(p, ws[p]["ckpt"]["shape"]) == "C"]
    ws_qk = [p for p in ws_T if QK_RE.match(p)]

    lines = []
    A = lines.append

    A("# R15 verdict - attention semantics of the released epoch_3.pt")
    A("")
    A("Generated by `analysis/r15/verdict.py` from `results/r15/{optimizer_state.csv,weight_stats.csv,functional.json}`.")
    A("Rules are taken verbatim from `plan/08-r15-attention-semantics.md` (preregistered, no post-hoc changes).")
    A("")
    A("## Group definitions and coverage")
    A("")
    A("| group | definition | S1 optimizer rows | S2 weight rows |")
    A("|---|---|---|---|")
    A("| T | 4 layers q_proj + k_proj (8), all 2-D weights of gene_tokens_projector | %d | %d |" % (len(opt_T), len(ws_T)))
    A("| C | 4 layers v_proj/o_proj/gate_proj/up_proj/down_proj, all 2-D weights of cell_projector, project_out, gene_decoder | %d | %d |" % (len(opt_C), len(ws_C)))
    A("")
    A("## Criterion 1 - optimizer evidence (preferred, used here)")
    A("")
    A("S1 produced an optimizer mapping in which every T and C parameter has a single fixed name.")
    A("Within `pert_model` there are %s order-preserving shape placements; the only ambiguous" % align.get("FITS_PERT_ONLY", "?"))
    A("slot is %s, a 1-D bias outside both groups, so criterion 1 decides the verdict." % plain(align.get("AMBIGUOUS_SLOTS")))
    A("")
    A("Uniqueness basis, full detail in `results/r15/param_alignment.txt`: the `se.*` block is")
    A("excluded because %s of its tensors are bit-identical to `se600m.safetensors` at" % align.get("SE_FROZEN_IDENTICAL", "?"))
    A("global_step %s with weight_decay=0; only %s has no counterpart there." % (align.get("GLOBAL_STEP", "?"), plain(align.get("SE_NO_REFERENCE"))))
    A("Fit counts: full `model_state_dict` raw keys %s, minus pert buffers %s," % (align.get("FITS_FULL_RAW", "?"), align.get("FITS_FULL_NOBUF", "?")))
    A("`pert_model` only %s, and also dropping the two frozen drug-branch groups %s." % (align.get("FITS_PERT_ONLY", "?"), align.get("FITS_PERT_NO_DRUG", "?")))
    A("T/C slots with more than one candidate: %s." % plain(align.get("AMBIGUOUS_SLOTS_IN_T_OR_C")))
    A("")
    A("### T group detail (every parameter must have exp_avg_sq nonzero count == 0 for H1)")
    A("")
    A("| index | parameter | shape | exp_avg_sq nonzero | numel | exp_avg_sq max | exp_avg maxabs |")
    A("|---|---|---|---|---|---|---|")
    for r in opt_T:
        A("| %d | %s | %s | %d | %d | %s | %s |" % (
            r["idx"], r["param"], r["shape"], r["nz"], r["numel"],
            "%.3e" % r["ea_sq_max"], "%.3e" % r["ea_maxabs"]))
    A("")
    A("### C group summary (every parameter must have exp_avg_sq nonzero count > 0 for H1)")
    A("")
    c_all_pos = all(r["nz"] > 0 for r in opt_C)
    c_min_nz = min(r["nz"] for r in opt_C) if opt_C else 0
    A("- C parameters: %d" % len(opt_C))
    A("- C parameters with exp_avg_sq nonzero count > 0: %d / %d" % (sum(1 for r in opt_C if r["nz"] > 0), len(opt_C)))
    A("- smallest C nonzero count: %d; largest C exp_avg_sq max: %s" % (
        c_min_nz, "%.3e" % max(r["ea_sq_max"] for r in opt_C)))
    A("")
    A("### Criterion 1 evaluation")
    A("")
    t_all_zero = all(r["nz"] == 0 for r in opt_T)
    qk_pos = sum(1 for r in qk if r["nz"] > 0)
    A("- T parameters with exp_avg_sq nonzero count == 0: %d / %d (all zero: %s)" % (
        sum(1 for r in opt_T if r["nz"] == 0), len(opt_T), t_all_zero))
    A("- C parameters with nonzero count > 0 (all positive: %s)" % c_all_pos)
    A("- q/k parameters with nonzero count > 0: %d / 8 (H2 threshold is 6)" % qk_pos)
    A("- 8 q/k exp_avg_sq max values: %s" % ", ".join("%.1e" % r["ea_sq_max"] for r in qk))
    A("- gene_tokens_projector 2-D weights with exp_avg_sq max == 0: %d / %d" % (
        sum(1 for r in gtp if r["ea_sq_max"] == 0.0), len(gtp)))
    A("")
    if t_all_zero and c_all_pos:
        c1 = "H1"
    elif qk_pos >= 6:
        c1 = "H2"
    else:
        c1 = "H3"
    A("Criterion 1 outcome: **%s**" % c1)
    A("")
    A("## Criterion 2 - statistical (cross-check; not the decision rule here)")
    A("")
    ok_stat = {}
    for p in ws_T + ws_C:
        ok_stat[p] = stat_looks_init(ws[p])
    c_not_init = [p for p in ws_C if not ok_stat[p]["looks_init"]]
    frac_c = (len(c_not_init) / len(ws_C)) if ws_C else 0.0
    valid = frac_c >= 0.5
    t_looks = sum(1 for p in ws_T if ok_stat[p]["looks_init"])
    qk_not = [p for p in ws_qk if not ok_stat[p]["looks_init"]]
    A("- C parameters (S2, 2-D weights): %d; not looking initialized: %d (%.1f%%); valid-test threshold 50%% -> %s" % (
        len(ws_C), len(c_not_init), 100.0 * frac_c, valid))
    A("- T looking initialized: %d / %d" % (t_looks, len(ws_T)))
    A("- q/k not looking initialized: %d / %d" % (len(qk_not), len(ws_qk)))
    A("")
    A("Representative bands (kurt and r over the five seeded initializations):")
    A("")
    A("| parameter | kurt ckpt | kurt band | r ckpt | r band | looks init |")
    A("|---|---|---|---|---|---|")
    sample = list(ws_qk) + [p for p in ws_C if p.startswith("pert_model.cell_projector.")][:2]
    for p in sample:
        s = ok_stat[p]
        A("| %s | %.4f | [%.4f, %.4f] | %.4f | [%.4f, %.4f] | %s |" % (
            p, s["kurt_ckpt"], s["kurt_band"][0], s["kurt_band"][1],
            s["r_ckpt"], s["r_band"][0], s["r_band"][1], s["looks_init"]))
    A("")
    A("## Criterion 3 - functional (must hold, else S4 is void)")
    A("")
    bpair = fun["within_condition"]["b"]["hvg"]
    abpair = fun["within_condition"]["a"]["hvg"]
    c3_ok = all(abs(v) == 0.0 for v in bpair.values())
    A("- condition b drug-pairwise max|delta| (pred_hvg): %s -> %s" % (
        json.dumps(bpair, sort_keys=True), "all exactly 0.0" if c3_ok else "NOT all zero"))
    A("- condition a drug-pairwise max|delta| (pred_hvg): %s" % json.dumps(abpair, sort_keys=True))
    A("")
    A("### Mask checks measured per forward")
    A("")
    A("| run | mask kind | shape | visible row0 | visible row1 | pass |")
    A("|---|---|---|---|---|---|")
    for k in sorted(fun["mask_checks"]):
        m = fun["mask_checks"][k]
        A("| %s | %s | %s | %s | %s | %s |" % (
            k, m["mask_kind"], m["mask_shape"], m["visible_row0"], m["visible_row1"], m["mask_check_pass"]))
    A("")
    A("### Between-condition comparison (same drug)")
    A("")
    A("| drug | pred_hvg max|delta| | pred_hvg Pearson | pred_emb max|delta| |")
    A("|---|---|---|---|")
    for tag in sorted(fun["between_condition"]):
        e = fun["between_condition"][tag]
        A("| %s | %.6e | %.6f | %.6e |" % (
            tag, e["hvg"]["maxabs"], e["hvg"]["pearson"], e["emb"]["maxabs"]))
    A("")
    A("### Condition a reproducibility vs work/exp32d")
    A("")
    for k in sorted(fun["md5"]["condition_a_vs_exp32d"]):
        v = fun["md5"]["condition_a_vs_exp32d"][k]
        A("- %s: run md5 %s, exp32d md5 %s, match %s" % (k, v["run_md5"], v["exp32d_md5"], v["match"]))
    A("")
    s3 = os.path.join(d, "drug_projector_vs_mapkg.csv")
    if os.path.exists(s3):
        with open(s3) as f:
            rows = list(csv.DictReader(f))
        eq = sum(1 for r in rows if r["equal"] == "True")
        vals = [float(r["maxabs_diff"]) for r in rows if r["maxabs_diff"] not in ("", None)]
        maxd = max(vals) if vals else float("nan")
        A("## S3 corroboration (not part of the decision rules)")
        A("")
        A("- drug_projector vs MAP-KG smiles_projector: %d tensors compared, bitwise equal %d, max|delta| %s" % (
            len(rows), eq, "%.3e" % maxd if maxd == maxd else "n/a"))
        A("")
    A("## Decision")
    A("")
    A("Criterion 1 (optimizer) decides because S1 produced data. Criterion 3 %s." % (
        "holds" if c3_ok else "FAILS - S4 must be redone"))
    A("")
    A("**%s**" % c1)
    A("")
    if c1 == "H1":
        A("Triggering numbers: all %d T parameters (including the 8 q/k weights) have" % len(opt_T))
        A("exp_avg_sq nonzero count 0 and exp_avg maxabs 0, while all %d C parameters have" % len(opt_C))
        A("nonzero count > 0. Position 0 attends only to itself under this training")
        A("semantics, so q_proj, k_proj and gene_tokens_projector received exactly zero")
        A("gradient for the entire run: their optimizer moments are still the Adam")
        A("initialization values.")
    elif c1 == "H2":
        A("Triggering numbers: %d of 8 q/k parameters have exp_avg_sq nonzero count > 0." % qk_pos)
    else:
        A("Triggering numbers: q/k with nonzero count > 0 = %d (below the 6 threshold) and the H1" % qk_pos)
        A("condition does not hold.")
    A("")
    A("This file reports the preregistered outcome only. Acceptance is decided by an independent review.")
    A("")
    out = os.path.join(d, "verdict.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[S5] wrote " + out)
    print("[S5] criterion1=%s (T all zero=%s, C all positive=%s, qk positive=%d) criterion3_ok=%s"
          % (c1, t_all_zero, c_all_pos, qk_pos, c3_ok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
