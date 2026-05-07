# US-015 Validation Report — Attempt 1

## Acceptance Criteria Checklist

1. **New file dtmol/data/tests/test_train_integration.py** — PASS (file exists)
2. **Test builds DatasetMixer from datamix_default.yaml, skipping missing LMDBs; requires >= 2 datasets** — PASS (fixture `available` skips if < 2, `_available_datasets()` filters by existence)
3. **Builds encoder/decoder using pretrain dicts, creates DiffusionTrainer, runs train_step on 3 batches with batch_size=2 on CPU** — PASS (`test_three_batches_finite_loss` runs 3 batches, batch_size=2, device="cpu")
4. **Loss is finite (not NaN, not Inf) for all batches** — PASS (asserts `torch.isfinite(loss)` and `not torch.isnan(loss)`)
5. **Loss dict contains keys: total_loss, diffusion_loss, force_loss, trrot_loss, pert_loss** — PASS (`test_loss_dict_keys` checks exact set)
6. **When Tier A data present and lambda_force > 0, tier_a_force_loss appears** — PASS (`test_tier_a_force_loss_present` passed)
7. **When Tier B data present and lambda_fd_force > 0, tier_b_force_loss appears** — SKIP (expected: MISATO records lack force data per implementor notes)
8. **Separate test function: legacy mode regression** — PASS (`TestLegacyMode::test_legacy_one_step_finite_loss` passed)
9. **Tests skip gracefully if pretrain dicts missing** — PASS (`_skip_if_no_pretrain()` called in fixtures)
10. **pytest ... -v --timeout=180 exits 0** — PASS (exit code 0, 5 passed, 1 skipped in 75.68s)
11. **Typecheck passes** — PASS (mypy: Success, no issues found)

## Test Results

- `pytest dtmol/data/tests/test_train_integration.py -v --timeout=180`: **5 passed, 1 skipped** (75.68s)
- `mypy --ignore-missing-imports dtmol/data/tests/test_train_integration.py`: **Success**
- Full regression suite (`pytest dtmol/data/tests/ -v --timeout=180`): **154 passed, 8 skipped** (132.74s) — no regressions

## Verdict: TEST_PASSED
