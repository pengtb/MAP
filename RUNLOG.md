# RUNLOG - R15 attention semantics (clean rerun)

Task: `plan/08-r15-attention-semantics.md`, steps S0-S5 (S6 only if the preregistered
rules require it). Decision H1 / H2 / H3 is produced by this run only; no number is
reused from the deleted previous R15 artifacts.

Hard constraints in force: no edits under `MAP/`, no commit/push, no plan/TODO edits,
GPU0 only, one process per functional condition, artifacts fixed at `analysis/r15/`,
`results/r15/`, `work/r15/`, `RUNLOG.md`.

## S0 - environment record

Run start: 2026-10-27T19:40:48+08:00 (b200 local time)

| item | value |
|---|---|
| host | s33b01n02 (b200) |
| repo | /data-tangc/dev/ptb/map-repro |
| git rev-parse HEAD | 5f659927e15c0b9f038df3871f197608292d22f4 |
| git status --short MAP/ | empty (no changes) |
| git status --short (whole tree) | pre-existing doc modifications (TODO.md, design/architecture.md, plan/01,02,03,05,README) plus untracked .gitignore, plan/08, analysis/ - all pre-existing or created by this task; none under MAP/ |
| python (.venv) | 3.11.14 |
| torch | 2.8.0+cu128 |
| transformers | 4.43.4 |
| numpy | 2.4.6 |
| CUDA_VISIBLE_DEVICES | 0 |
| GPU0 at start | 46638 MiB used / 183359 MiB total (NVIDIA B200) |
| GPU4-7 at start | 166886 MiB used each (left untouched) |
| main ckpt | /data-tangc/models/map_repr/predictor/epoch_3.pt (4227878838 bytes) |
| MAP-KG encoder ckpt | /data-tangc/models/map_repr/mapkg_encoder/mapkg_encoder_v3.pt (1126763160 bytes) |

Environment used for every compute step:

```
source .venv/bin/activate
export MAP_MOLECULE_CKPT=/data-tangc/models/map_repr/mapkg_encoder/mapkg_encoder_v3.pt
export MAP_BART_VOCAB=/data-tangc/models/map_repr/molecule_stm/bart_vocab.txt
export CUDA_VISIBLE_DEVICES=0
```

`git status --short MAP/` will be re-checked at the end and compared to this S0 state.

## S1 - checkpoint structure and optimizer state

Artifacts: `results/r15/optimizer_state.csv` (69 data rows), `results/r15/param_alignment.txt`.
Script: `analysis/r15/ckpt_inspect.py`.

| attempt | script sha256 | job id | recorded rc | outcome |
|---|---|---|---|---|
| 1 | 2138a5864eba0174a68cfb944603dbec354c8b0fa358f0f59bd3599cd72fffdb | 9a7ba87cdf614ef082ba1d09ab88a0e4 | 3 | alignment failed: script aligned on `named_parameters()` order |
| 2 | 4e808f8ee30e5e45c97db9e6ba54def1bec2515444cb205494df7b791c43c7cc | 4a4aef381287480698160c1ad9ca6956 | 0 | aligned, 69-entry mapping (the "unique" wording was imprecise; corrected and proven in "Repair round 1") |

Attempt 1 detail: the checkpoint key **set** already matched the reconstructed module
exactly (255 parameters), but the **saved order** differs from the transformers 4.43.4
`named_parameters()` order, first divergence at index 209
(`...layers.0.mlp.down_proj.weight` vs `...layers.0.mlp.up_proj.weight`). plan/08 requires
the `model_state_dict` key order, so the script was corrected to use it and re-run.
Attempt 1 is kept on record; no number from it was used.

Findings (`work/r15/s1.log`, `results/r15/param_alignment.txt`):

- top-level keys: `epoch`, `global_step`, `model_state_dict` (499), `optimizer_state_dict` (2),
  `scheduler_state_dict` (7), `args` (22) - so the optimizer state **is** present.
- `pert_model` keys 260; 5 buffers removed (`smiles_encoder._model.pos_emb` plus the four
  `self_attn.rotary_emb.inv_freq`) -> 255 parameters.
- ckpt parameter key set == module `named_parameters` set: True.
- optimizer: 1 param group, 69 params, `lr=9.998783363402833e-06`, `betas=(0.9, 0.999)`,
  `eps=1e-08`, `weight_decay=0` (so a zero-gradient parameter can never move).
- candidate A (all 255 params): rejected, count 255 != 69.
- candidate B (69 `requires_grad` params in saved order): all 69 shapes match in order.
- index -> parameter mapping over 69 entries; CSV rows 69. The word "unique" here was
  imprecise; the proof and the exact statement are in "Repair round 1".

## S2 - weight statistics

Script: `analysis/r15/weight_stats.py`. Two fixes before the run:
`--out-dir` support (for the acceptance rerun) and self-contained defaults for
`MAP_MOLECULE_CKPT` / `MAP_BART_VOCAB`, because the plan/08 acceptance command does not
export them.

SHA-256: `b8a282561e21bbb4ac181e7b2d5495b393c8da47b7ff4edbe0ac88f234bccafb`.
Job: `01dec32cb3f4461c89f8467944420968`.

S2 run: job `01dec32cb3f4461c89f8467944420968`, recorded **rc=0**, log `work/r15/s2.log`.
Result: 45 in-scope 2-D weights x 6 sources = 270 data rows (`results/r15/weight_stats.csv`).
Statistic definitions used: std is torch default (unbiased), kurt is the excess kurtosis with
population moments, `smax` is the largest singular value of the float64 matrix, and
`r = smax / (std * (sqrt(m)+sqrt(n)))`. Computation is single-threaded float64.

Rerun check (acceptance 2), exactly the plan/08 command:

```
.venv/bin/python analysis/r15/weight_stats.py --out-dir /tmp/r15_rerun
cmp results/r15/weight_stats.csv /tmp/r15_rerun/weight_stats.csv
```

Job `21c08ff315644a0a9fc63444104fa4b3`, recorded **rc=0**; pane shows `RERUN_RC=0` and
`CMP_RC=0`; both files sha256 `91b3fd60bffcad9844aaf3653700be781c480f1fbdfa3b3689995a82f792ce55`.
Log `work/r15/s2_rerun.log`. Byte identical.

## S4 - functional comparison

Runner: `analysis/r15/run_demo_condition.py`, sha256
`8cdb8f7891d6d963b86223df7309470b08d85b56251da9a88728e4e6e62f0550`.
One process per (condition x drug), `CUDA_VISIBLE_DEVICES=0`, `demo.py` unmodified
(`--seed 0 --drug_conc 0.5 --set_size 24 --cell_line CVCL_0023`). Condition b patches only
at runtime, on the first `MAPmodel.forward`: `_update_causal_mask` returns the causal additive
mask (strict upper triangle = dtype min) and `is_causal=True`.

| forward | job id | recorded rc | diag mask kind | visible row0 / row1 |
|---|---|---|---|---|
| a / seed0_a | 0115bf3a492b48fa8c5d143cac0873c8 | 0 | zero_additive_bidirectional | 2049 / 2049 |
| a / drug_ibu | f6c7f5b54d8a46ed82d2d6eacaec1654 | 0 | zero_additive_bidirectional | 2049 / 2049 |
| a / drug_caf | 5144141795eb4e17ad30c158bbdaf67f | 0 | zero_additive_bidirectional | 2049 / 2049 |
| b / seed0_a | f8c02bd102c1406b96d2a8c4e8c00146 | 0 | additive_with_blocked_entries | 1 / 2 |
| b / drug_ibu | 85db1d2ae48c4cadb433744a73a5f341 | 0 | additive_with_blocked_entries | 1 / 2 |
| b / drug_caf | a026a56cc7a441879890a15954462326 | 0 | additive_with_blocked_entries | 1 / 2 |

Failed attempt kept on record: job `283765d98dcf4f69b3614188d8bd4123`, **rc=1**. The first
runner version wrapped `MAPmodel.__init__` with `*args/**kwargs`, which broke the Lightning
`save_hyperparameters` frame parser (`KeyError: a`). It produced no artifacts and no numbers
are used from it. The runner was rewritten to patch at forward time (never on `__init__`).

Condition b visible-row profile is `[1, 2, 3, ...]`, i.e. strictly causal.

Aggregation: `analysis/r15/s4_aggregate.py`, sha256
`ad902fd4b65ad6a8c78cbbeb668e53eda9addbae92b25102592e1e25be8119c2`;
artifact `results/r15/functional.json`; per-forward recorded rc also embedded via
`work/r15/forward_rc.json`.

Key numbers (pred_hvg):

- condition b drug-pairwise max|delta|: seed0_a|drug_ibu `0.0`, seed0_a|drug_caf `0.0`,
  drug_ibu|drug_caf `0.0` - all exactly zero.
- condition a drug-pairwise max|delta|: `2.981424331665039e-04`, `3.440380096435547e-04`,
  `3.439188003540039e-04` (reproduces the weak ~3e-4 signal of plan/05).
- condition a vs b, same drug: max|delta| about `1.543`, Pearson about `0.884` for all three.
- condition a `output_pred_hvg.npy` md5 equals `work/exp32d/*/output_pred_hvg.npy` for all
  three drugs (`4efcdf75cea1b47b17363cd12e2e3be4`, `5a8611e6829a52142506b220f84a7091`,
  `91dca0696d0f70a122dfe39340eef19d`). `output_pred_embeddings.npy` md5 also matches.

## S1c - strengthened alignment proof (authoritative S1 run)

The aligned run above chose the mapping through the reconstructed `requires_grad` flags.
To rule out a flag-dependent coincidence, `ckpt_inspect.py` was extended with candidate C,
whose exclusion set is defined purely by module name, and re-run.

| item | value |
|---|---|
| script sha256 | `a36cfe1f1ac30631d50046a5e971926d2a09ddf17e06f6853dee89951865e679` |
| job | `d0f35e9d93ed439db58b45861164f648` |
| recorded rc | 0 |
| `results/r15/optimizer_state.csv` sha256 | `dd2d0d7651315ce877f99cae53e49616c843b07a51bd6b3f7815b9a142a45442` (unchanged by the re-run) |

Evidence:

- frozen by module name: `smiles_encoder` 175 params, `drug_projector` 10, `embed_tokens` 1
  -> 186 excluded, 255 - 186 = 69.
- candidate C (name-defined, 69 params) matches the optimizer `exp_avg` shape sequence exactly.
- parameters outside the mapping: 186; outside-set == the three named groups: **True**.
- chosen mapping == candidate C: **True**.
- mapping: 69 entries. The exclusion is therefore module-name defined, not an
  artefact of the reconstructed flags. "Unique" was imprecise; see "Repair round 1".

## S3 - drug_projector provenance

| item | value |
|---|---|
| script | `analysis/r15/drug_projector_vs_mapkg.py`, sha256 `f003bf4d057568ffedca61c0f055c5f311448ceaceaf388ffcc386aa0cf9b575` |
| job | `717d373cabc94e14ae49b5d3b2f8ac72` |
| recorded rc | 0 |
| artifact | `results/r15/drug_projector_vs_mapkg.csv`, 10 data rows, sha256 `6adcc2ca8f9b9f765f77e050cfe6f7d36249e6d7b78bb882ff6cdd46ec77f1e9` |
| log | `work/r15/s3.log` |

Result: all 10 tensors have the same shape as the MAP-KG `smiles_projector`, **0** are bitwise
equal, max|delta| `1.939`. The released `drug_projector` is not a copy of the MAP-KG
projector; it is a separate (and, per S1, frozen) branch. Corroborating evidence only, it is
not part of the decision rules.

## S5 - verdict

| item | value |
|---|---|
| script | `analysis/r15/verdict.py`, sha256 `6781246f337c935ebf6431c23c2315dd40dc7ab0b639b43d7a92e4d6743f406f` |
| command | `.venv/bin/python analysis/r15/verdict.py` |
| artifact | `results/r15/verdict.md`, sha256 `e8632e7d97cc05cfec36150fa5c6cb6fea43545b97f70457945433a4b0c6dce3` |

Rules applied exactly as preregistered, no threshold change. Criterion 1 decided because S1
produced data:

- **H1**: all 12 T parameters (8 q/k weights plus the 4 `gene_tokens_projector` 2-D weights)
  have `exp_avg_sq` nonzero count **0** and `exp_avg` maxabs **0.0**, while all 29 C
  parameters have nonzero count **> 0** (smallest 524288). q/k with nonzero count > 0: 0 / 8.
- Criterion 2 cross-check also returns H1: test valid (96.6 percent of C not looking
  initialized), T 12/12 look initialized, q/k not initialized 0/8.
- Criterion 3 holds: condition b drug-pairwise max|delta| exactly `0.0`.

S6 was **not** required: the verdict is H1 (not H3) and the statistical test is valid, so the
plan/08 fallback does not trigger.

## Acceptance checks

Run on b200 at 2026-10-27T19:54:46+08:00.

1. non-empty artifacts: all seven present (`RUNLOG.md`, `weight_stats.csv`,
   `drug_projector_vs_mapkg.csv`, `functional.json`, `verdict.md`, `optimizer_state.csv`,
   `param_alignment.txt`) - pass.
2. rerun determinism: `RERUN_RC=0`, `CMP_RC=0`, both files sha256
   `91b3fd60bffcad9844aaf3653700be781c480f1fbdfa3b3689995a82f792ce55` - pass.
3. coverage: 45 params x 6 sources = 270 rows, no parameter missing a source - pass.
   Set decomposition: T 12 + C 29 + `drug_projector` 2-D weights 4 = 45.
4. functional: 6/6 diag pass (condition a rows 2049/2049, condition b rows 1/2), condition b
   pairwise all `0.0`, 3/3 condition-a md5 equal to `work/exp32d`, 6/6 recorded forward rc = 0
   - pass.
5. `git status --short MAP/` empty (0 lines), HEAD `5f659927e15c0b9f038df3871f197608292d22f4`
   unchanged, `verdict.md` states H1 with the triggering numbers - pass.

## Script SHA-256 (final)

```
a36cfe1f1ac30631d50046a5e971926d2a09ddf17e06f6853dee89951865e679  analysis/r15/ckpt_inspect.py
f003bf4d057568ffedca61c0f055c5f311448ceaceaf388ffcc386aa0cf9b575  analysis/r15/drug_projector_vs_mapkg.py
8cdb8f7891d6d963b86223df7309470b08d85b56251da9a88728e4e6e62f0550  analysis/r15/run_demo_condition.py
ad902fd4b65ad6a8c78cbbeb668e53eda9addbae92b25102592e1e25be8119c2  analysis/r15/s4_aggregate.py
6781246f337c935ebf6431c23c2315dd40dc7ab0b639b43d7a92e4d6743f406f  analysis/r15/verdict.py
b8a282561e21bbb4ac181e7b2d5495b393c8da47b7ff4edbe0ac88f234bccafb  analysis/r15/weight_stats.py
```

## Artifact SHA-256

```
6adcc2ca8f9b9f765f77e050cfe6f7d36249e6d7b78bb882ff6cdd46ec77f1e9  results/r15/drug_projector_vs_mapkg.csv
2240afba11c1b3cc132c1e0150ea3615e31f6e10b9a04a06f43f53771b83af5a  results/r15/functional.json
dd2d0d7651315ce877f99cae53e49616c843b07a51bd6b3f7815b9a142a45442  results/r15/optimizer_state.csv
0c458cda67dba7121bad36be2b34cd26676f2aef47db1717e2238ad03b2736e4  results/r15/param_alignment.txt
e8632e7d97cc05cfec36150fa5c6cb6fea43545b97f70457945433a4b0c6dce3  results/r15/verdict.md
91b3fd60bffcad9844aaf3653700be781c480f1fbdfa3b3689995a82f792ce55  results/r15/weight_stats.csv
```

## tmux sessions left for Review (not cleaned)

`mjt_r15clean_s1`, `mjt_r15clean_s1b`, `mjt_r15clean_s1c`, `mjt_r15clean_s2`,
`mjt_r15clean_s2b`, `mjt_r15clean_s3`, `mjt_r15clean_s4a1`, `mjt_r15clean_s4a1b`,
`mjt_r15clean_s4a2`, `mjt_r15clean_s4a3`, `mjt_r15clean_s4b1`, `mjt_r15clean_s4b2`,
`mjt_r15clean_s4b3`.

BLOCKED: none.

Implement does not declare acceptance. Independent Review decides against plan/08.

## Repair round 1 - response to Review findings F1 and F2

Scope: F1 and F2 only. No GPU forward was re-run. `results/r15/functional.json` is
byte-identical to round 1 (`2240afba11c1b3cc132c1e0150ea3615e31f6e10b9a04a06f43f53771b83af5a`)
and `results/r15/optimizer_state.csv` is byte-identical
(`dd2d0d7651315ce877f99cae53e49616c843b07a51bd6b3f7815b9a142a45442`). The decision **H1** is
unchanged; only the wording of the uniqueness claim and the trace records changed.

### F1 - uniqueness of the S1 mapping

Problem accepted as stated: `ckpt_inspect.py` only looked at `pert_model.*`, ignored the 239
`se.*` keys of the 499-key `model_state_dict`, and called the mapping "unique" after testing
three hand-picked candidate lists. That is not a proof.

Fix, files changed:

| file | change | sha256 before | sha256 after |
|---|---|---|---|
| `analysis/r15/uniqueness.py` | new: exact integer DP (no sampling, no enumeration cap) for order-preserving shape placements, per-slot feasibility, and the `se.*` bit-identity check | (new) | `0296b1a15c1f30eb526b4e977b7c55c3ba7bb650d341590f67505aa52adf5336` |
| `analysis/r15/ckpt_inspect.py` | analyses the full 499-key `model_state_dict`; adds `se.*` frozen evidence, per-slot candidate sets and fit counts; neutral RESULT wording | `4e808f8ee30e5e45c97db9e6ba54def1bec2515444cb205494df7b791c43c7cc` | `c9be89bbf3d5daca73398109a81d9125fc17e166e48cc501658bb2e109885141` |
| `analysis/r15/verdict.py` | reads the F1 markers from `param_alignment.txt` and states the result precisely | `6781246f337c935ebf6431c23c2315dd40dc7ab0b639b43d7a92e4d6743f406f` | `17fa7dc2e637ca42a176ca50b536ae76b8c534dd14c655879f0f4608df0bcfc2` |
| `results/r15/param_alignment.txt` | regenerated with the proof section and markers | `0c458cda67dba7121bad36be2b34cd26676f2aef47db1717e2238ad03b2736e4` | `cef24238a47b136b8aa3324841a3c360253d992bd101774acbf2cbd46cfb103d` |
| `results/r15/verdict.md` | regenerated with the accurate wording | `e8632e7d97cc05cfec36150fa5c6cb6fea43545b97f70457945433a4b0c6dce3` | `8b27412ec961b53732ae9d0036960723e9143ba6c9b1f5cf3b63a4b8813e53bb` |

F1 jobs (recorded rc from `job-status.sh`):

| purpose | job id | start | end | rc |
|---|---|---|---|---|
| F1 analysis, first run | `f0f46e756cd546b0a3a5adc71907aaef` | 20:20:11 | 20:20:19.12 | 0 |
| F1 analysis, after T/C prefix fix | `ae250e42a3814e9a8dff66b0396340bf` | 20:21:38 | 20:21:52.73 | 0 |
| F1 analysis, final wording | `2db0b1dd6308492c82191a51a662689a` | 20:22:23 | 20:22:31.73 | 0 |

Disclosure: the first F1 run (`f0f46e75`) printed `T slots=0 C slots=0`, because `classify()`
was called on names without the `pert_model.` prefix. That made the T/C claim vacuously true
and it was rejected, not used. After the prefix fix the same job reports `T slots=12 C slots=29`.

Computed result, reproduced by `analysis/r15/ckpt_inspect.py` (also in
`work/r15/s1_repair.log` and as `KEY=value` markers at the end of `param_alignment.txt`):

| scope | keys | order-preserving shape placements |
|---|---|---|
| full `model_state_dict`, raw keys | 499 | 4738802796 |
| full, minus the 5 `pert_model` buffers | 494 | 4738802796 |
| `pert_model` only, minus buffers | 255 | 12 |
| also dropping `smiles_encoder` and `drug_projector` | 70 | 1 |

Why `se.*` may be excluded - bit-identity, not an assumption: `se.*` has 239 tensors; 238 are
bit-identical to `se600m.safetensors` (name mapping mirrors `MAPmodel.load_se_ckpt`), 0 differ,
and only `_bin_indices_cached` has no counterpart in that file. The checkpoint carries
`global_step=155847` and the optimizer has `weight_decay=0`, so a parameter that never moved
must have had zero gradient: the whole `se.*` block was frozen.

Per-slot feasibility in that justified scope (`pert_model`, se excluded): 12 placements, exactly
one ambiguous slot, **slot 7** (a 1024-long 1-D bias). T slots = 12 and C slots = 29; T/C slots
with more than one candidate = none. So every T and C weight has one fixed name in every
possible placement; the ambiguity is confined to one 1-D bias outside both groups. The narrower
scope that also drops the two frozen drug-branch groups has exactly 1 placement.

Wording now used (also in `verdict.md`): "an optimizer mapping in which every T and C parameter
has a single fixed name; within `pert_model` there are 12 order-preserving shape placements and
the only ambiguous slot is 7, a 1-D bias outside both groups".

Reproduction command:

```
.venv/bin/python -u analysis/r15/ckpt_inspect.py
```

The earlier "unique mapping" statements in this log (the S1 finding list and the S1c proof) are
annotated in place and point here; they were not silently rewritten.

### F2 - per-step trace records

Missing records were added below. All times are **reconstructed afterwards** (labelled
"事后重建"), not written live at the time: the command comes from the preserved
`~/.rtmux_cmd_<job-id>.sh`, the start time from the first `date -Is` line in the preserved
tmux pane (`capture-pane`), and the end time from the mtime of `~/.rtmux_done_<job-id>`.
No timestamp was invented; where no record exists it is stated as unavailable.

Long jobs (tmux `send.sh`), **事后重建**:

| step | core command | start | end | job id | rc |
|---|---|---|---|---|---|
| S1 attempt 1 | `.venv/bin/python -u analysis/r15/ckpt_inspect.py` | 19:41:38 | 19:41:45 | `9a7ba87cdf614ef082ba1d09ab88a0e4` | 3 |
| S1 attempt 2 | `.venv/bin/python -u analysis/r15/ckpt_inspect.py` | 19:42:27 | 19:42:40 | `4a4aef381287480698160c1ad9ca6956` | 0 |
| S2 | `.venv/bin/python -u analysis/r15/weight_stats.py` | 19:43:39 | 19:47:23 | `01dec32cb3f4461c89f8467944420968` | 0 |
| S4 a/seed0_a attempt 1 | `../.venv/bin/python -u ../analysis/r15/run_demo_condition.py --condition a --drug-tag seed0_a` | 19:45:33 | 19:45:37 | `283765d98dcf4f69b3614188d8bd4123` | 1 |
| S4 a/seed0_a | same, `--drug-tag seed0_a` | 19:46:14 | 19:46:35 | `0115bf3a492b48fa8c5d143cac0873c8` | 0 |
| S4 a/drug_ibu | `--condition a --drug-tag drug_ibu` | 19:47:34 | 19:47:57 | `f6c7f5b54d8a46ed82d2d6eacaec1654` | 0 |
| S2 rerun (acceptance 2) | `.venv/bin/python analysis/r15/weight_stats.py --out-dir /tmp/r15_rerun` then `cmp` | 19:47:59 | 19:51:48 | `21c08ff315644a0a9fc63444104fa4b3` | 0 |
| S4 a/drug_caf | `--condition a --drug-tag drug_caf` | 19:48:04 | 19:48:22 | `5144141795eb4e17ad30c158bbdaf67f` | 0 |
| S4 b/seed0_a | `--condition b --drug-tag seed0_a` | 19:49:13 | 19:49:33 | `f8c02bd102c1406b96d2a8c4e8c00146` | 0 |
| S4 b/drug_ibu | `--condition b --drug-tag drug_ibu` | 19:50:07 | 19:50:28 | `85db1d2ae48c4cadb433744a73a5f341` | 0 |
| S4 b/drug_caf | `--condition b --drug-tag drug_caf` | 19:51:07 | 19:51:26 | `a026a56cc7a441879890a15954462326` | 0 |
| S3 | `.venv/bin/python -u analysis/r15/drug_projector_vs_mapkg.py` | 19:52:26 | 19:52:32 | `717d373cabc94e14ae49b5d3b2f8ac72` | 0 |
| S1c | `.venv/bin/python -u analysis/r15/ckpt_inspect.py` | 19:53:10 | 19:53:21 | `d0f35e9d93ed439db58b45861164f648` | 0 |

Every forward was launched from `MAP/` with `CUDA_VISIBLE_DEVICES=0`,
`MAP_MOLECULE_CKPT` and `MAP_BART_VOCAB` set; the exact wrapper, including the full
`--smiles`, `--out-dir` and `--diag` arguments, is preserved verbatim in
`~/.rtmux_cmd_<job-id>.sh`. Recorded rc is always from `bash scripts/job-status.sh <job-id>`.

Short steps (`run.sh`, recorded rc is the `run.sh` exit code), executed live during this
repair so the record is not reconstructed:

| step | command | start | end | runner | rc |
|---|---|---|---|---|---|
| S4 aggregation re-run | `.venv/bin/python analysis/r15/s4_aggregate.py` | 2026-10-27T20:22:02 | 2026-10-27T20:22:02 | `run.sh` | 0 |
| S5 verdict re-run | `.venv/bin/python analysis/r15/verdict.py` | 2026-10-27T20:22:02 | 2026-10-27T20:22:02 | `run.sh` | 0 |
| S5 verdict final | `.venv/bin/python analysis/r15/verdict.py` | 2026-10-27T20:23:33 | 2026-10-27T20:23:33 | `run.sh` | 0 |

S0 and the round-1 acceptance battery were also `run.sh` steps, not tmux jobs, so they have no
job id. Their times are in this log (S0 at 19:40:48, acceptance battery at 19:54:46) and their
`run.sh` exit code was 0. Exact per-command end times for those two were not recorded and are
not reconstructed here.

Repair step records (live): F1 jobs are listed in the F1 table above; the S4 aggregation and S5
verdict commands are the two `run.sh` rows above, both rc=0.

### Post-repair artifact hashes

| artifact | round 1 | after repair |
|---|---|---|
| `results/r15/optimizer_state.csv` | `dd2d0d76...` | `dd2d0d7651315ce877f99cae53e49616c843b07a51bd6b3f7815b9a142a45442` (unchanged) |
| `results/r15/weight_stats.csv` | `91b3fd60...` | `91b3fd60bffcad9844aaf3653700be781c480f1fbdfa3b3689995a82f792ce55` (unchanged) |
| `results/r15/drug_projector_vs_mapkg.csv` | `6adcc2ca...` | `6adcc2ca8f9b9f765f77e050cfe6f7d36249e6d7b78bb882ff6cdd46ec77f1e9` (unchanged) |
| `results/r15/functional.json` | `2240afba...` | `2240afba11c1b3cc132c1e0150ea3615e31f6e10b9a04a06f43f53771b83af5a` (unchanged) |
| `results/r15/param_alignment.txt` | `0c458cda...` | `cef24238a47b136b8aa3324841a3c360253d992bd101774acbf2cbd46cfb103d` (proof added) |
| `results/r15/verdict.md` | `e8632e7d...` | `8b27412ec961b53732ae9d0036960723e9143ba6c9b1f5cf3b63a4b8813e53bb` (wording corrected) |

Final script hashes:

```
c9be89bbf3d5daca73398109a81d9125fc17e166e48cc501658bb2e109885141  analysis/r15/ckpt_inspect.py
f003bf4d057568ffedca61c0f055c5f311448ceaceaf388ffcc386aa0cf9b575  analysis/r15/drug_projector_vs_mapkg.py
8cdb8f7891d6d963b86223df7309470b08d85b56251da9a88728e4e6e62f0550  analysis/r15/run_demo_condition.py
ad902fd4b65ad6a8c78cbbeb668e53eda9addbae92b25102592e1e25be8119c2  analysis/r15/s4_aggregate.py
0296b1a15c1f30eb526b4e977b7c55c3ba7bb650d341590f67505aa52adf5336  analysis/r15/uniqueness.py
17fa7dc2e637ca42a176ca50b536ae76b8c534dd14c655879f0f4608df0bcfc2  analysis/r15/verdict.py
b8a282561e21bbb4ac181e7b2d5495b393c8da47b7ff4edbe0ac88f234bccafb  analysis/r15/weight_stats.py
```

Not done in this round: no GPU forward re-run, no change under `MAP/`, no plan/TODO edit, no
commit, no download, no session cleanup. The verdict remains **H1**.

### Sessions after the repair round

Left for Review, not cleaned: the 13 `mjt_r15clean_*` sessions from round 1 plus the three
repair sessions `mjt_r15repair_s1`, `mjt_r15repair_s1b`, `mjt_r15repair_s1c`.
`mjt_r15review_chk` and `mjt_r15review_chk2` belong to the reviewer and were not touched.

BLOCKED: none. Implement does not declare acceptance; the same Review re-checks this revision.

## Lead re-review after Review provider failure (2026-10-27)

VERDICT: pass. The independent Review completed round 1 with F1/F2, and Implement repaired
both findings. The same Review could not perform round 2 because the provider returned
HTTP 401 (insufficient 302.AI balance). With the user's explicit authorization, Lead took
over the second read-only review. No GPU forward was re-run.

F1 closure evidence:

- Lead independently recomputed the order-preserving shape-fit DP from `epoch_3.pt` without
  calling `analysis/r15/uniqueness.py`: full raw keys 499 -> 4,738,802,796 fits; full minus
  five pert buffers -> the same count; pert_model minus buffers 255 -> 12 fits; additionally
  dropping smiles_encoder and drug_projector 70 -> 1 fit.
- In the 255-key scope, only slot 7 is ambiguous (12 candidate 1-D bias names). T=12 and C=29
  slots each have exactly one fixed name, and every actual T/C row in optimizer_state.csv
  equals that sole DP candidate. The first repair run with T=0/C=0 was discarded; the final
  repair job `2db0b1dd6308492c82191a51a662689a` recorded rc=0 and prints T=12/C=29.
- SE exclusion is independently established by the training path, not by bit-identity alone:
  `MAPmodel.__init__` defaults `freeze_se=True` (`MAP/model/model.py:19,54-56`), `train.py`
  constructs MAPmodel without overriding it, and the optimizer accepts only requires_grad
  parameters (`MAP/train.py:236`). The 238/239 bit-identical tensors corroborate this; the
  remaining `_bin_indices_cached` is an int64 registered buffer (`MAP/model/se.py:232`).
- Residual wording note: phrases in the generated artifacts saying bit-identity "proves" SE
  exclusion should be read together with the direct freeze_se code evidence above. Bit-
  identity alone would not distinguish exclusion from inclusion with zero gradient. This
  does not change any mapping or H1 number.

F2 closure evidence:

- RUNLOG now contains the S1-S5 command/start/end/job-id/rc records. Historical records are
  explicitly labelled reconstructed afterwards and cite the cmd/pane/done-file sources;
  S4 aggregation and S5 were re-run as short run.sh steps with rc=0.
- All three repair jobs have COMPLETED rc=0 via job-status.sh. The final artifact SHA-256
  values match the post-repair table. optimizer_state.csv, weight_stats.csv,
  drug_projector_vs_mapkg.csv and functional.json are unchanged from round 1.
- S0 and the acceptance battery honestly disclose that exact per-command end times were not
  recorded and were not fabricated. Exact long-job wrappers remain in ~/.rtmux_cmd_<job-id>.sh
  as identified in RUNLOG. These are residual trace limitations, not numerical discrepancies.

Final checks by Lead: `git status --short MAP/` is empty; HEAD remains
`5f659927e15c0b9f038df3871f197608292d22f4`. The preregistered result is H1. Current
bidirectional inference remains a comparison mode only; changing the default inference
semantics is a separate user-approved task.
