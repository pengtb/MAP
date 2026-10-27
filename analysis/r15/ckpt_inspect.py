"""S1: checkpoint structure and optimizer-state evidence for R15 (plan/08).

Reads epoch_3.pt, takes the parameter order from model_state_dict key order (plan/08:
the saved order is authoritative because transformers 4.43.4 registers modules in a
different order than the training-time version did), removes buffers via the
constructed module named_buffers(), aligns that order with optimizer_state_dict
param_groups / state shapes, and writes:
  results/r15/optimizer_state.csv
  results/r15/param_alignment.txt

Names are full checkpoint keys (prefix pert_model.) so they map literally onto
model_state_dict entries. Read-only with respect to MAP/.
Exit code 3 = alignment not unique or not matching: plan/08 then stops S1 and uses the
statistical criterion instead of guessing a mapping.
"""
import argparse
import csv
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO, "MAP")
if MAP_DIR not in sys.path:
    sys.path.insert(0, MAP_DIR)

DEFAULT_CKPT = "/data-tangc/models/map_repr/predictor/epoch_3.pt"
SE_CKPT = "/data-tangc/models/map_repr/state_se600m/se600m.safetensors"
PFX = "pert_model."


def build_model():
    from model.pert_ca import PerturbationEncoder_ca

    torch.manual_seed(0)
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


def tshape(t):
    return "x".join(str(int(s)) for s in t.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results", "r15"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log("[S1] ckpt: " + args.ckpt)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit("[S1] checkpoint top level is %s, expected dict" % type(ckpt).__name__)

    top = list(ckpt.keys())
    log("top-level keys (%d): %s" % (len(top), repr(top)))
    for k in top:
        v = ckpt[k]
        extra = ""
        if hasattr(v, "__len__"):
            try:
                extra = " len=%d" % len(v)
            except Exception:
                extra = ""
        log("  %-26s %s%s" % (k, type(v).__name__, extra))

    if "model_state_dict" not in ckpt:
        raise SystemExit("[S1] model_state_dict missing")
    state = ckpt["model_state_dict"]
    opt = ckpt.get("optimizer_state_dict")

    model = build_model()
    params = list(model.named_parameters())
    req = {n: bool(p.requires_grad) for n, p in params}
    param_names = [n for n, _ in params]
    buffer_names = [n for n, _ in model.named_buffers()]
    buffer_set = set(buffer_names)

    log("")
    log("constructed module: named_parameters=%d named_buffers=%d" % (len(param_names), len(buffer_names)))
    log("buffers: " + repr(buffer_names))

    ck_names = [k[len(PFX):] for k in state.keys() if k.startswith(PFX)]
    ck_bufs = [n for n in ck_names if n in buffer_set]
    ck_params = [n for n in ck_names if n not in buffer_set]
    log("ckpt pert_model keys=%d; buffers removed=%d -> %s" % (len(ck_names), len(ck_bufs), repr(ck_bufs)))
    log("ckpt parameter keys (saved order, buffers removed)=%d" % len(ck_params))

    set_equal = set(ck_params) == set(param_names)
    log("ckpt param key SET == module named_parameters SET: %s" % set_equal)
    log("ckpt saved order == named_parameters order: %s" % (ck_params == param_names))
    log("ORDER SOURCE (plan/08): model_state_dict key order with buffers removed.")
    if not set_equal:
        only_ck = [n for n in ck_params if n not in set(param_names)]
        only_md = [n for n in param_names if n not in set(ck_params)]
        log("  only in ckpt: " + repr(only_ck))
        log("  only in model: " + repr(only_md))
    elif ck_params != param_names:
        first = next(i for i in range(len(ck_params)) if ck_params[i] != param_names[i])
        log("  first order divergence at index %d: ckpt=%s module=%s"
            % (first, ck_params[first], param_names[first]))

    ck_shape = {}
    for n in ck_params:
        ck_shape[n] = tuple(int(s) for s in state[PFX + n].shape)

    fail_reason = None
    if not set_equal:
        fail_reason = "ckpt parameter key set differs from the reconstructed module"
    elif len(ck_bufs) != len(buffer_set):
        fail_reason = "buffer set mismatch between ckpt and module"

    mapping = None
    if opt is None:
        log("")
        log("NO optimizer_state_dict in ckpt -> plan/08 S1 says skip S1 and use the statistical criterion.")
    elif fail_reason is None:
        groups = opt.get("param_groups", [])
        ostate = opt.get("state", {})
        log("")
        log("optimizer param_groups=%d, state entries=%d" % (len(groups), len(ostate)))
        ids = []
        for gi, g in enumerate(groups):
            pid = list(g.get("params", []))
            ids.extend(pid)
            log("  group %d: params=%d lr=%s betas=%s eps=%s weight_decay=%s"
                % (gi, len(pid), g.get("lr"), g.get("betas"), g.get("eps"), g.get("weight_decay")))

        opt_shapes = []
        unreadable = False
        for i in ids:
            if i not in ostate or "exp_avg" not in ostate[i]:
                unreadable = True
                break
            opt_shapes.append(tuple(int(s) for s in ostate[i]["exp_avg"].shape))

        cand_all = list(ck_params)
        cand_train = [n for n in ck_params if req.get(n, False)]
        log("candidate A all ckpt params (saved order): %d" % len(cand_all))
        log("candidate B requires_grad ckpt params (saved order): %d" % len(cand_train))
        log("optimizer index count: %d" % len(ids))

        def check(cand):
            if unreadable:
                return False, "optimizer state entry unreadable"
            if len(cand) != len(ids):
                return False, "count %d != optimizer %d" % (len(cand), len(ids))
            for j, n in enumerate(cand):
                if ck_shape[n] != opt_shapes[j]:
                    return False, "index %d name %s ckpt %s != opt %s" % (j, n, ck_shape[n], opt_shapes[j])
            return True, "all %d shapes match in order" % len(cand)

        ok_a, why_a = check(cand_all)
        ok_b, why_b = check(cand_train)
        log("candidate A (all params): %s -- %s" % (ok_a, why_a))
        log("candidate B (requires_grad only): %s -- %s" % (ok_b, why_b))

        excl = {
            "smiles_encoder": [n for n in ck_params if n.startswith("smiles_encoder.")],
            "drug_projector": [n for n in ck_params if n.startswith("drug_projector.")],
            "embed_tokens": [n for n in ck_params if n == "transformer_backbone.embed_tokens.weight"],
        }
        named_excluded = set()
        for g in sorted(excl):
            named_excluded.update(excl[g])
            log("frozen-by-name group %s: %d parameters" % (g, len(excl[g])))
        cand_named = [n for n in ck_params if n not in named_excluded]
        ok_c, why_c = check(cand_named)
        log("candidate C (all params minus the three module-name groups): %s -- %s" % (ok_c, why_c))
        log("candidate C length=%d, excluded total=%d" % (len(cand_named), len(named_excluded)))

        if ok_a and ok_b:
            if cand_all == cand_train:
                mapping = cand_all
                log("both candidate checks pass on the identical list -> mapping unique")
            else:
                fail_reason = "two distinct candidate lists both match; mapping is not unique"
        elif ok_a:
            mapping = cand_all
            log("only candidate A matches -> the optimizer covered every parameter")
        elif ok_b:
            mapping = cand_train
            log("only candidate B matches -> the optimizer covered requires_grad parameters only")
        else:
            fail_reason = "neither candidate matches the optimizer state"

        if mapping is not None:
            if len(set(mapping)) != len(mapping):
                fail_reason = "mapping contains duplicate parameter names"
                mapping = None
            elif len(mapping) != len(ids):
                fail_reason = "mapping length differs from optimizer index count"
                mapping = None
            else:
                log("RESULT: index -> parameter mapping over %d entries (uniqueness of the T and C names is analysed in the F1 section below)" % len(mapping))
            mapped = set(mapping)
            not_mapped = [n for n in ck_params if n not in mapped]
            log("parameters outside the optimizer mapping: %d" % len(not_mapped))
            log("outside-set == the three named frozen groups: %s" % (set(not_mapped) == named_excluded))
            log("chosen mapping == candidate C (name-defined): %s" % (mapping == cand_named))

    if mapping is not None:
        ostate = opt["state"]
        ids = []
        for g in opt.get("param_groups", []):
            ids.extend(list(g.get("params", [])))
        opt_csv = os.path.join(args.out_dir, "optimizer_state.csv")
        with open(opt_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "param", "shape", "ea_sq_max", "ea_sq_nonzero", "ea_sq_numel", "ea_maxabs"])
            for j, n in enumerate(mapping):
                st = ostate[ids[j]]
                ea = st["exp_avg"].to(torch.float64)
                easq = st["exp_avg_sq"].to(torch.float64)
                w.writerow([
                    j,
                    PFX + n,
                    tshape(state[PFX + n]),
                    repr(float(easq.max())) if easq.numel() else "nan",
                    int((easq != 0).sum()),
                    int(easq.numel()),
                    repr(float(ea.abs().max())) if ea.numel() else "nan",
                ])
        log("wrote " + opt_csv)

    # ---- F1 (repair round 1): uniqueness proof over the full model_state_dict ----
    if mapping is not None and opt is not None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import uniqueness
        from verdict import classify

        opt_state = opt["state"]
        opt_ids = []
        for g in opt.get("param_groups", []):
            opt_ids.extend(list(g.get("params", [])))
        target_shapes = [tuple(int(s) for s in opt_state[i]["exp_avg"].shape) for i in opt_ids]
        wd = [g.get("weight_decay") for g in opt.get("param_groups", [])]

        all_keys = list(state.keys())
        all_shapes = [tuple(int(s) for s in state[k].shape) for k in all_keys]

        def scoped(only_pert, drop_buffers, skip_mods):
            ks, shs = [], []
            for k, sh in zip(all_keys, all_shapes):
                short = k[len(PFX):] if k.startswith(PFX) else None
                if only_pert and short is None:
                    continue
                if drop_buffers and short is not None and short in buffer_set:
                    continue
                if short is not None and any(short.startswith(m) for m in skip_mods):
                    continue
                ks.append(short if short is not None else k)
                shs.append(sh)
            return ks, shs

        log("")
        log("===== F1 uniqueness analysis over the full model_state_dict (repair round 1) =====")
        log("optimizer slots=%d ; weight_decay per group=%s" % (len(target_shapes), wd))
        scopes = [
            ("full_state_dict_raw_keys", scoped(False, False, ())),
            ("full_state_dict_minus_pert_buffers", scoped(False, True, ())),
            ("pert_model_only_minus_buffers", scoped(True, True, ())),
            ("pert_model_minus_smiles_encoder_minus_drug_projector",
             scoped(True, True, ("smiles_encoder.", "drug_projector."))),
        ]
        tables = {}
        for label, (ks, shs) in scopes:
            pre_t, suf_t, total = uniqueness.fit_tables(shs, target_shapes)
            tables[label] = (ks, shs, pre_t, suf_t, total)
            log("order-preserving shape fits: %-52s n_keys=%-4d fits=%d" % (label, len(ks), total))

        ev = uniqueness.se_frozen_evidence(state, SE_CKPT)
        log("se.* frozen evidence vs se600m.safetensors: se_keys=%d identical=%d differing=%d no_reference=%s"
            % (ev["se_keys"], len(ev["identical"]), len(ev["differing"]), repr(ev["no_reference"])))
        if ev["differing"]:
            log("  differing entries: %s" % repr(ev["differing"][:10]))
        log("ckpt global_step=%s ; weight_decay=%s and a zero gradient leave a parameter untouched,"
            % (ckpt.get("global_step"), wd))
        log("so bit-identity with the se600m initial file proves the se.* block was frozen.")

        def grp(name, shape):
            full = name if name.startswith(PFX) else PFX + name
            return classify(full, "x".join(str(d) for d in shape))

        justified = "pert_model_only_minus_buffers"
        ks, shs, pre_t, suf_t, total = tables[justified]
        cand = uniqueness.slot_candidates(ks, shs, target_shapes, pre_t, suf_t)
        amb = [j for j in range(len(cand)) if len(cand[j]) > 1]
        tc_amb = [j for j in amb if any(grp(n, target_shapes[j]) in ("T", "C") for n in cand[j])]
        t_slots = [j for j in range(len(cand)) if cand[j] and grp(cand[j][0], target_shapes[j]) == "T"]
        c_slots = [j for j in range(len(cand)) if cand[j] and grp(cand[j][0], target_shapes[j]) == "C"]
        log("")
        log("justified scope: se.* excluded on that bit-identity -> %s" % justified)
        log("  fits=%d ; ambiguous slots (%d)=%s" % (total, len(amb), amb))
        for j in amb:
            log("  slot %d shape=%s candidates=%s"
                % (j, "x".join(str(d) for d in target_shapes[j]), cand[j]))
        log("  T slots=%d C slots=%d ; T/C slots with more than one candidate: %d"
            % (len(t_slots), len(c_slots), len(tc_amb)))
        log("  chosen mapping is a valid placement in this scope: %s"
            % all(mapping[j] in cand[j] for j in range(len(mapping))))
        log("  every T and C slot has exactly one candidate: %s" % (len(tc_amb) == 0))

        narrow = "pert_model_minus_smiles_encoder_minus_drug_projector"
        ks2, shs2, pre2, suf2, total2 = tables[narrow]
        cand2 = uniqueness.slot_candidates(ks2, shs2, target_shapes, pre2, suf2)
        amb2 = [j for j in range(len(cand2)) if len(cand2[j]) > 1]
        log("")
        log("narrower scope (also drops the two frozen drug-branch groups): %s" % narrow)
        log("  fits=%d ; ambiguous slots=%s" % (total2, amb2))

        log("")
        log("FITS_FULL_RAW=%d" % tables["full_state_dict_raw_keys"][4])
        log("FITS_FULL_NOBUF=%d" % tables["full_state_dict_minus_pert_buffers"][4])
        log("FITS_PERT_ONLY=%d" % total)
        log("FITS_PERT_NO_DRUG=%d" % total2)
        log("AMBIGUOUS_SLOTS=%s" % amb)
        log("AMBIGUOUS_SLOTS_IN_T_OR_C=%s" % tc_amb)
        log("T_C_UNIQUE=%s" % (len(tc_amb) == 0))
        log("SE_FROZEN_IDENTICAL=%d/%d" % (len(ev["identical"]), ev["se_keys"]))
        log("SE_NO_REFERENCE=%s" % repr(ev["no_reference"]))
        log("GLOBAL_STEP=%s" % ckpt.get("global_step"))

    align = os.path.join(args.out_dir, "param_alignment.txt")
    with open(align, "w") as f:
        f.write("\n".join(lines) + "\n")
        f.write("\n")
        if mapping is None:
            f.write("STATUS: no alignment\n")
            if fail_reason:
                f.write("REASON: " + fail_reason + "\n")
        else:
            f.write("STATUS: aligned\n")
            f.write("NAMES: full checkpoint keys (model_state_dict), prefix pert_model.\n")
            f.write("COLUMNS: idx<TAB>param<TAB>shape\n")
            for j, n in enumerate(mapping):
                f.write("%d\t%s\t%s\n" % (j, PFX + n, tshape(state[PFX + n])))
    print("[S1] wrote " + align, flush=True)

    if mapping is None and opt is not None and fail_reason:
        print("[S1] ALIGNMENT FAILED: " + fail_reason, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
