"""S3: provenance check of pert_model.drug_projector against the MAP-KG smiles_projector.

Plan/08 S3: compare each tensor of pert_model.drug_projector.* with the matching
smiles_projector.* tensor of mapkg_encoder_v3.pt (found under model_state_dict) and
record shape, exact equality and max|delta|. The result is corroborating evidence only;
it is not part of the preregistered decision rules.

Writes results/r15/drug_projector_vs_mapkg.csv. Read-only with respect to MAP/.
"""
import argparse
import csv
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED = "/data-tangc/models/map_repr/predictor/epoch_3.pt"
MAPKG = "/data-tangc/models/map_repr/mapkg_encoder/mapkg_encoder_v3.pt"
PFX = "pert_model.drug_projector."
KG_PFX = "smiles_projector."


def tshape(t):
    return "x".join(str(int(s)) for s in t.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-ckpt", default=PRED)
    ap.add_argument("--mapkg-ckpt", default=MAPKG)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results", "r15"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[S3] loading predictor ckpt", flush=True)
    pred = torch.load(args.pred_ckpt, map_location="cpu", weights_only=False)
    pred_sd = pred["model_state_dict"]
    drug = {k[len(PFX):]: v for k, v in pred_sd.items() if k.startswith(PFX)}
    print("[S3] drug_projector tensors: %d" % len(drug), flush=True)
    del pred
    del pred_sd

    print("[S3] loading mapkg ckpt", flush=True)
    kg = torch.load(args.mapkg_ckpt, map_location="cpu", weights_only=False)
    kg_sd = kg["model_state_dict"]
    smiles = {k[len(KG_PFX):]: v for k, v in kg_sd.items() if k.startswith(KG_PFX)}
    print("[S3] smiles_projector tensors: %d" % len(smiles), flush=True)
    del kg
    del kg_sd

    out = os.path.join(args.out_dir, "drug_projector_vs_mapkg.csv")
    rows = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "mapkg_key", "shape_drug", "shape_mapkg", "shape_match", "equal", "maxabs_diff"])
        for suffix in sorted(drug.keys()):
            a = drug[suffix]
            b = smiles.get(suffix)
            if b is None:
                w.writerow([PFX + suffix, "MISSING", tshape(a), "", "False", "False", ""])
                rows += 1
                continue
            same_shape = tuple(a.shape) == tuple(b.shape)
            if same_shape:
                af = a.to(torch.float64)
                bf = b.to(torch.float64)
                diff = (af - bf).abs().max()
                maxabs = repr(float(diff))
                equal = bool(torch.equal(a.to(torch.float32), b.to(torch.float32))) and bool(diff == 0)
            else:
                maxabs = ""
                equal = False
            w.writerow([
                PFX + suffix, KG_PFX + suffix,
                tshape(a), tshape(b), str(same_shape), str(equal), maxabs,
            ])
            rows += 1
    print("[S3] wrote %s rows=%d" % (out, rows), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
