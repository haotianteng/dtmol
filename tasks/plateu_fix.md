# Training-loss plateau — root cause and fixes

> **Status (2026-05-06):** root-causes identified, four fixes attempted (A–E), three on the merge path. Fix E (single-crystal subset training) demonstrably learns rotation prediction (`ratio_tr 1.00 → 0.35`, `cos_tr → +0.85` over 2000 steps on `fcc_Al_2x2x2`). Per-atom perturbation lags but trends positive. Multi-type generalisation is the next open problem.

## TL;DR — the canonical setup that learns

```bash
# branch: debug/training-convergence
# state-of-this-doc commit: ab36b2c (saved checkpoint + eval scripts)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NO_DROPOUT=1 SYSTEM_ID_PREFIX=fcc_Al_2x2x2 \
  python scripts/debug_plateau.py \
    --max-batches 2000 --lr 1e-4 --device cuda \
    --save-folder dtmol/models/bindingpose_<DATE>_fixE_fcc_Al
```

**Saved checkpoint from this run** (committed at `ab36b2c`):
```
dtmol/models/bindingpose_20260506_fixE_fcc_Al/ckpt-2000.pt   (716MB)
dtmol/models/bindingpose_20260506_fixE_fcc_Al/checkpoint     (index)
```

**Visualise the trained denoise behaviour:**
```bash
python scripts/eval_denoise_trajectory.py \
  --ckpt-dir dtmol/models/bindingpose_20260506_fixE_fcc_Al \
  --system-id-prefix fcc_Al_2x2x2 \
  --out-dir ralph/test_results/fixE_fcc_Al \
  --t-targets 1000 2500 3500 4500 --traj-steps 20 --device cuda
```
Outputs `tweedie_panels.png`, `reverse_trajectory.png`, `report.json`.

**Honest read of the eval (`ralph/test_results/fixE_fcc_Al/report.json`):**
- Tweedie one-step RMSD reduction: 0.1% at t=1000 to 1.4% at t=4500. Direction is partially learned but magnitude is ~15% of target, so the Tweedie step `x0_est = x_t − sigma_t · eps_pred` only nudges back ~15% of the noise.
- Reverse trajectory of 20 steps starts at RMSD=2.30 (pure noise → struct distance) and *diverges* to ~4.2. The model isn't yet a working denoiser — it has a useful but undersized score signal. **trrot rotation** is what learns first (cos_tr peaks ~+0.85 at step ~2000); per-atom **perturbation** lags (cos_p ~+0.3, ratio_p ~0.95).

The branch carries:
- **Fix A** — zero-init the geometric projection in both heads (`linear3.weight`/`out_proj2.weight`). Without it the random `y_branch` makes initial loss > baseline and Adam first kills `y_branch` toward 0 (regress-to-baseline) instead of rotating it toward target.
- **Fix B** — lift the DiT zero-init on `TransformerDecoderWithPair.final_layer.linear`. The DiT trick assumes the zero-inited linear is the *terminal* output projection, but our diffusion heads run a multi-layer MLP downstream, so pinning `decoder_rep ≡ 0` at init starves all head MLP weights of gradient.
- **Fix E** — `SYSTEM_ID_PREFIX` filter on the dataset + a looping dataloader so a small subset still gets `max_batches` steps. The multi-record gradient signal averaging across 17 crystal types is what blocks learning; restrict to one type and the signal is consistent enough to converge.
- `--dropout 0` is the default in `dtmol/dtmol_train_test.py`; `NO_DROPOUT=1` is the equivalent runtime switch in the diagnostic. Dropout in train mode injects gradient noise that destabilises the score-matching bootstrap.

## Why training plateaued

Three compounding problems, in the order we found them:

1. **Heads defeat the DiT zero-init.** `pred = Sigmoid(MLP(decoder_rep)) * linear3(node_rep_y)`. With Sigmoid gate, init pred ≈ 0.5 · y_branch, where `y_branch` is random direction. Loss > baseline, and the gradient pushes Adam to *shrink* `y_branch` (kill the head) before learning to align direction. Once at zero, signal vanishes.
   - **Fix A**: zero-init `linear3.weight` so pred = 0 at init.
   - Confirmed via `scripts/debug_plateau.py` baseline vs Fix A on multi-record (small but real ratio_p improvement).
2. **DiT zero-init pins decoder_rep at 0.** `final_layer.linear` is zero-inited so `decoder_rep ≡ 0`, which means the head MLP (`linear1 → act → LN → linear2`) sees zero input and only its biases get gradient — weights stay at Xavier init forever. Slow ramp.
   - **Fix B**: lift the zero-init. With Fix A's y-branch zero-init handling pred = 0 at init, the decoder's final_layer is free to start from a normal Xavier init.
3. **Multi-record gradient signal averages out.** Even with A+B applied, multi-record training over 17+ crystal lattices stalls (`ratio_p 0.99`). The architecture is *fundamentally not learning* the score function across diverse inputs.
   - The decisive evidence: `scripts/debug_overfit_one.py` shows the same network drives loss from baseline to `ratio=0.28, cos=+0.85` on a *fixed* batch in 1000 steps. So the architecture works. The bottleneck is per-step gradient variance across mixed crystal types.
   - **Fix E**: filter to one `system_id` prefix (74 records of `fcc_Al_2x2x2`). With consistent geometry the gradient signal stops averaging out.

## Fix-by-fix results

500-step multi-record diagnostic on `dtmol/data/datamix_unit_cell.yaml` with `lr=1e-4`, `NO_DROPOUT=1`. Aggregates over the last 50 steps.

| variant | branch | ratio_p ↓ | \|cos_p\| ↑ | ratio_tr ↓ | \|cos_tr\| ↑ | merged |
|---|---|---|---|---|---|---|
| baseline | `ralph/dataset-integration` | 1.000 | 0.056 | 1.002 | 0.337 | n/a |
| A: zero-init y-branch + dropout=0 default | `fix/zero-init-y-branch` | 0.988 | 0.113 | 1.000 | 0.341 | ✓ |
| A + B: also lift decoder.final_layer.linear zero-init | `fix/lift-decoder-final-zeroinit` | 0.994 | 0.122 | 1.000 | 0.358 | ✓ |
| A + B + C: curriculum sigma_pert_max 1.5→0.3 | `fix/curriculum-sigma` | 0.999 | 0.059 | 1.001 | 0.354 | ✗ no help |
| A + B + D: drop multiplicative gate | `fix/drop-gate` | 0.993 | 0.127 | 1.000 | 0.341 | ✗ neutral, kept for ref |
| A + B + E: single-crystal `fcc_Al_2x2x2` (500 step) | `fix/single-crystal` | 0.960 | 0.198 | 0.999 | 0.326 | ✓ |

**Fix E 2000-step trajectory on `fcc_Al_2x2x2`** (74 records looped):

| step | ratio_tr | cos_tr | ratio_p | cos_p |
|---|---|---|---|---|
|   99 | 1.00 | −0.03 | 1.00 | +0.06 |
|  999 | 0.98 | +0.13 | 0.99 | +0.10 |
| 1499 | 0.80 | +0.50 | 0.95 | +0.23 |
| 1899 | 0.66 | +0.73 | 1.01 | −0.01 |
| **1998** | **0.35** | **+0.85** | 0.96 | +0.24 |

Rotation prediction reaches the same level as the overfit-one ceiling. Per-atom perturbation lags because each atom needs its own direction.

## Branch & commit map

```
master
└── ralph/dataset-integration
    └── debug/training-convergence       ← all the work; HEAD = 4dafb9c
        ├── fix/zero-init-y-branch       (Fix A; merged into debug)
        ├── fix/lift-decoder-final-zeroinit  (Fix B; merged into debug)
        ├── fix/single-crystal            (Fix E; merged into debug)
        ├── fix/drop-gate                 (Fix D; not merged, kept for reference)
        └── fix/curriculum-sigma          (Fix C; not merged, did not help)
```

To reproduce any state precisely:
```bash
# Pre-fix baseline
git checkout 1e64581       # debug branch — diagnostics added but no fixes
# Fix A only
git checkout d987bef       # fix/zero-init-y-branch
# Fix A + B
git checkout 4ec3a50       # fix/lift-decoder-final-zeroinit
# Fix A + B + D
git checkout 20a4415       # fix/drop-gate
# Fix A + B + E (current debug HEAD)
git checkout 4dafb9c       # debug/training-convergence
```

## Diagnostics added in this work

- `scripts/debug_plateau.py` — multi-record diagnostic. Per-step (target_rms, pred_rms, cos_sim, ratio) and aggregates over the last 50 steps. Env switches: `NO_DROPOUT=1`, `ZERO_INIT_HEADY=1`, `SYSTEM_ID_PREFIX=<prefix>`. Flags: `--max-batches`, `--lr`, `--batch-size`, `--save-folder`.
- `scripts/debug_overfit_one.py` — caches one batch and tries to overfit. Pass criterion: `ratio → 0` and `|cos| → 1`. The decisive sanity check that proved the architecture works.
- `scripts/eval_denoise_trajectory.py` — loads a saved checkpoint and visualises (a) Tweedie one-step denoising at multiple `t` values and (b) a full reverse-diffusion trajectory of ~20 steps. Outputs `tweedie_panels.png`, `reverse_trajectory.png`, `report.json` under `--out-dir`.
- All run logs under `ralph/debug_logs/` (`baseline_*`, `fixA_*`, `fixB_*`, `fixC_*`, `fixD_*`, `fixE_*`, `overfit_*`).

## Next problems to attack

1. **Multi-type generalisation.** Take the Fix E `fcc_Al_2x2x2` checkpoint as a warm start and broaden `SYSTEM_ID_PREFIX` to cover more types (e.g. all `fcc_*`, then add `bcc_*`, then arbitrary). Track whether `ratio_tr/p` regress when we add types — if learning sticks, curriculum is the path; if it collapses, we need an architectural fix that handles multi-modal score targets (e.g. type-conditioned heads).
2. **Per-atom perturbation lag.** On Fix E `fcc_Al`, `cos_p` only reached ~+0.24 after 2000 steps while `cos_tr` reached +0.85. Per-atom predictions are intrinsically harder; possible levers: longer training, widen `linear3` from `(1, 8)` to `(out, hidden)` so the projection has more capacity, or replace the scalar gate with a normalisation-aware additive head.
3. **Production-script port.** Push the `SYSTEM_ID_PREFIX` filter into `UnifiedDatasetConfig` so `dtmol/dtmol_train_test.py` can run the Fix E setup directly (currently only the diagnostic supports it).
4. **trrot translation 3-DOF on single-mol data.** Translation is structurally unlearnable on single-molecule records (SE3 input is pairwise displacements which are translation-invariant). Either zero out trrot translation loss when `single_molecule_mask=True`, or feed `full_coor[:,0,:]` (mol centroid) as an absolute-position reference token.
5. **Wider `linear3`.** Current shape `(out_dim/3, input_dim2=8)` — only 8 parameters carry the per-atom score direction. Compare with widening `input_dim2` (controlled by SE3 stack output channels).

## How to validate a fix

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NO_DROPOUT=1 \
  python scripts/debug_plateau.py --max-batches 500 --lr 1e-4 --device cuda \
  > ralph/debug_logs/<run-name>.log 2>&1
sed -n '/AGGREGATE/,/Legend/p' ralph/debug_logs/<run-name>.log
```

"Decent" target on `unit_cell` after this work:
- **trrot**: `ratio_tr < 0.5`, `cos_tr > +0.7` over the last 50 steps (achieved at 2000 steps on Fix E).
- **perturbation**: `ratio_p < 0.5`, `cos_p > +0.5` (still open).
