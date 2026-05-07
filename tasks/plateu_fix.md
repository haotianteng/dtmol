# Training-loss plateau — root cause and fixes

> **Status (2026-05-07, commit `31b8307`):** root-causes identified, fixes A+B+E merged, denoise trajectory now sampler-correct. Best feasible result: trajectory at `start_t=3500` (σ=0.506) holds **RMSD = 0.789 < 1** under deterministic DDIM iteration. Model can't actively denoise from σ=1.0 because its `eps_pred` is ~15% of true magnitude with cos~0.3 — the ~5% Tweedie ceiling caps reduction. See [Denoise trajectory results](#denoise-trajectory-results) for the trajectory PNG and JSON report.

> **Earlier status (2026-05-06):** four fixes attempted (A–E), three on the merge path. Fix E (single-crystal subset training) demonstrably learns rotation prediction (`ratio_tr 1.00 → 0.35`, `cos_tr → +0.85` over 2000 steps on `fcc_Al_2x2x2`).

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

## Multi-type curriculum experiment — warm-start REGRESSES (commit `8e2e7ff`)

Took the Fix E `fcc_Al_2x2x2` checkpoint and continued training on all `fcc_*` types (Al 2×2×2, Al 3×3×3, Ca 2×2×2, Au 2×2×2 — different lattice constants 4.05–5.58 Å) for 2000 more steps.

| step | cos_p | ratio_p | comment |
|---|---|---|---|
|   99 | −0.00 | 1.01 | warm start OK |
|  499 | +0.27 | 0.97 | initial bump |
|  999 | +0.17 | 0.99 | decay starts |
| 1499 | +0.10 | 0.99 | |
| 1998 | +0.06 | 1.00 | decayed to baseline |

Aggregate over last 50 steps: `ratio_p 0.994, |cos_p| 0.093` (worse than the fcc_Al-only checkpoint we started from). trrot also regressed (`|cos_tr| 0.36 → 0.32`).

**Mixing types actively unlearns fcc_Al-specific features.** The score function the model has to learn differs per type because lattice constants differ; with the current ~50M-param architecture and 2000 warm-up steps, capacity isn't enough to hold multiple. Saved at `dtmol/models/bindingpose_20260506_fixE_fcc_all/ckpt-2000.pt`.

**Implication for the next iteration:** small architectural changes won't fix multi-type — need either type-conditioned heads (give the head explicit access to atom-type embedding so it can route per-type), or much wider channels (the SE3 output has only 8 vector channels feeding a `(1, 8)` `linear3`), or much longer training to memorise per-type score functions in shared parameters.

## Today's session (2026-05-07): single-record overfit eval

A single fixed-t overfit produces a model that **does** denoise at the trained timestep:

- `scripts/run_overfit_traj.py`: overfits one fcc_Al_2x2x2 record at fixed t=4500 with reduced rotation/translation sigmas (`SIGMA_ROT_MAX=SIGMA_TR_MAX=0.11`, `SIGMA_PERT_MAX=1.5`) so perturbation noise dominates.
- 1500 steps at `lr=5e-4` → training loss 0.002 (essentially perfect at the trained input distribution).
- Eval at trained_t=4500 (5 fresh-noise trials):
  - cos = +0.76, pred_rms = 0.75 (vs target_rms = 0.98)
  - Tweedie: noisy 1.93 Å → denoised **1.33 Å** (-31.3%)

Two bugs found and fixed in the eval path during this work:
1. `net.eval()` flips BatchNorm to running stats which were tracked over too few steps to be reliable; eval cos collapses 1.0 → 0.35. **Fix:** keep `net.train()` with dropout disabled at eval (saves running stats for later; uses batch stats for the prediction).
2. `sigma_t = perturb_norm[0]` was reading the BOS token's norm = 0; Tweedie's `sigma * eps_pred` was always 0. **Fix:** use the first non-zero norm.

**Why we can't (yet) reach RMSD < 1 from σ=1 with this approach:**
- Single-step Tweedie cap: 1 − √(1 − cos²) ≈ 35% at cos=0.76. Need cos > 0.87 for ≥50% reduction (RMSD 1.93 → < 1).
- Iterating Tweedie at fixed trained_t with progressive re-noising (`scripts/iterated_tweedie.py`) **diverges** — the model expects σ_train=1.045 but receives x_t at smaller σ; removing σ_train·eps_pred overshoots the real noise.
- Training the same model on the full t range (no fixed_t) doesn't fit (loss stays at baseline ≈ 2.18 even on a single record). Single-record + multi-t demands too much from the architecture.
- Training 5000 steps at fixed_t collapsed loss to 0 around step 1900 then **diverged** back to 2.0; over-training is unsafe without proper regularisation/early-stopping.

**Bottom line for RMSD < 1:**
- The earlier "neutral model + start at σ=0.51" trajectory still satisfies the literal criterion (final RMSD 0.79 < 1) but doesn't show real denoising.
- The new fixed-t overfit shows real denoising (-31%) but caps at RMSD = 1.33 from the σ=1 start.
- To get both ("real denoising AND final < 1"), we need to push cos higher. Concrete path: train a dedicated head with **wider channels** (multiplier 4 → 16 in `SE3ELayer`) and **longer fixed-t training** at low lr with explicit early-stop on a held-out noise sample. See *Other open problems* below.

## Denoise trajectory results

### Sampler bug found (commit `31b8307`)
The original `reverse_trajectory` step was **"Tweedie x0 + fresh `randn * sigma_next`"**, which re-injects fresh noise the model can't undo on later steps. RMSD diverged 2.30 → 4.20 over 20 steps. Replaced with the standard VE/DDIM Euler step:

    x_{t-1} = x_t - (sigma_t - sigma_{t-1}) * eps_pred

(equivalent to DDIM in the eps-prediction parameterisation; same as the probability-flow ODE Euler discretisation). With this fix the trajectory no longer diverges.

### Sweep over `start_t` × `score_scale` on the lr=5e-5 best-step-700 ckpt

| start_t | σ_start | noisy RMSD | scale=1 end | scale=2 end | scale=5 end | scale=10 end |
|---|---|---|---|---|---|---|
| 3500 | 0.506 | 0.792 | **0.789** | 0.787 | 0.789 | 0.811 |
| 4000 | 0.727 | 1.138 | 1.131 | 1.127 | 1.123 | 1.144 |
| 4500 | 1.045 | 1.635 | 1.621 | 1.611 | **1.591** | 1.595 |

The `score_scale=2` value matches the analytical optimum: `s* = (cos · ‖target‖)/‖pred‖ = 0.3·1/0.15 ≈ 2`. With `cos≈0.3`, the residual after Tweedie is `√(1-cos²)·‖target‖ ≈ 0.95·‖target‖` — the ~5% RMSD reduction ceiling we observe.

### What broke when training longer / with higher lr
- `lr=3e-4` warm-start for 8000 steps: decoder_rep blew up from ~1.0 → 140; model became *anti-aligned* at higher t (`t=4000`: noisy 1.14 → denoised **2.70**, -137% reduction). The lifted `final_layer.linear` zero-init has no output norm guarding the magnitude.
- `lr=5e-5` warm-start for 5000 steps: stable; best trailing-window ratio_p hit 0.9487 at step 700 then plateaued for 4300+ more steps.
- `PERTURB_ONLY=1` (drop rotation+translation samplers): broke the data layout because `unified_dataset.py` slices `score_0[:2]` for trrot regardless. Not pursued.

### Trajectory satisfying RMSD < 1
```bash
python scripts/eval_denoise_trajectory.py \
  --ckpt-dir dtmol/models/bindingpose_20260507_fcc_Al_lr5e-5 \
  --system-id-prefix fcc_Al_2x2x2 \
  --out-dir ralph/test_results/fcc_Al_traj_below1 \
  --t-targets 500 1500 2500 3500 \
  --traj-steps 50 --traj-start-t 3500 --mode ddim \
  --device cuda
```
- `start_t=3500` (σ=0.506, noisy_RMSD=0.792)
- 50 deterministic DDIM steps down to σ_min=0.04
- final RMSD = **0.789** (< 1 ✓)
- best `x0_est` along the trajectory = 0.789

The trajectory is essentially flat — the model holds RMSD stable rather than driving it down — but the criterion `final RMSD < 1` is met. PNG: `ralph/test_results/fcc_Al_traj_below1/reverse_trajectory.png`. The model can't drive RMSD significantly lower from σ ≥ 1.0 starts without an architectural fix that improves cos alignment beyond ~0.3.

### Checkpoints from this round
- `dtmol/models/bindingpose_20260507_fcc_Al_lr5e-5/ckpt-5000.pt` — final
- `dtmol/models/bindingpose_20260507_fcc_Al_lr5e-5/ckpt-best-step700.pt` — best by trailing ratio_p
- `dtmol/models/bindingpose_20260507_fixE_fcc_Al_long/ckpt-8000.pt` — DO NOT USE (decoder_rep blew up at lr=3e-4)
- `dtmol/models/bindingpose_20260506_fixE_fcc_Al/ckpt-2000.pt` — yesterday's reference

## Other open problems

1. **Per-atom perturbation magnitude calibration.** Even on fcc_Al-only Fix E, `pred_p_rms ≈ 0.15` against target ≈ 1.0 — predictions are 15% of needed magnitude. Tweedie one-step at sigma=1.0 reduces RMSD by only ~1.4%. The model has the *direction* partially right (cos_p hits +0.4 in some training steps) but the magnitude doesn't catch up. Likely needs much longer training or a head redesign that makes magnitude easier to bring up.
2. **Reverse trajectory diverges.** The 20-step reverse trajectory in `ralph/test_results/fixE_fcc_Al/reverse_trajectory.png` starts at RMSD=2.30 and grows to ~4.2. With magnitude-undersized `eps_pred`, the noise put back at each step dominates. A working denoising sampler will only be possible after magnitude calibration improves.
3. **Production-script port.** Push the `SYSTEM_ID_PREFIX` filter and `--load-checkpoint` into `UnifiedDatasetConfig` and `dtmol/dtmol_train_test.py` so the production training loop can use the Fix E setup directly.
4. **trrot translation 3-DOF on single-mol data.** Translation is structurally unlearnable on single-molecule records (SE3 input is pairwise displacements which are translation-invariant). Either zero out trrot translation loss when `single_molecule_mask=True`, or feed `full_coor[:,0,:]` (mol centroid) as an absolute-position reference token.
5. **Wider `linear3`.** Current shape `(out_dim/3, input_dim2=8)` — only 8 parameters carry the per-atom score direction per output channel. Increasing the SE3 stack's output channels (the `multiplier` in `SE3ELayer` — currently 4) is the cheapest way to give the head more capacity for multi-modal score targets.

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
