# ACTRN v2 experiment protocol (fixed before the registered run, 16 October 2026)

The complete protocol was executed once with `python code/actrn_experiments.py --workers 32`.
Outputs are in `data/processed/actrn_v2/` (`manifest.json` records environment, seeds,
timings and output hashes). The 16 July 2026 results in `data/processed/actrn_v1_20260716/`
were produced by an earlier protocol whose field split let training labels overlap the test
period; they are kept for reference only.

## Common settings

- Environment: see `requirements.lock.txt` (Python 3.11.15). One thread per process.
- Saturating action feature: `1 - exp(-U_kj / s_j)`, where `s_j` is the mean positive injection
  of injector `j` in the training data; used in both the benchmark and the field case.
- Tuning: every model chooses one of eight pre-specified configurations (grids in
  `code/actrn_experiments.py::GRIDS`) by validation NMAE in physical units, separately for each
  benchmark repeat and each field origin. The selected model is evaluated on the test data
  without refitting. ACTRN also uses the validation data for early stopping. No selection step
  reads test data.
- Ranking weight 3.0 on the benchmark; no ranking term in the field case.

## Known-truth benchmark

- Five repeats (data seed `20260716 + 1009 r`, model seed `42 + r`), 48 reservoirs each, grouped
  split 32/8/8 by reservoir; four decision states per test reservoir with 80 equal-budget
  candidates shared by all models.
- Models: Ridge, random forest, HGB and MLP on raw inputs; the same four on the ACTRN
  representation `[x_H, x_U]`; ACTRN.
- Ablations (reusing the full ACTRN configuration selected in the same repeat): no ranking
  term; no multiplicative interaction `z_I`; raw action input `vec(U)` only.
- Robustness: the simulator's support response `1 - exp(-1.9 a)` is replaced by `a / (a + 0.55)`
  and the raw baselines, feature-matched baselines and ACTRN are rerun.
- Statistics: relative regret (%) as the main decision metric; for ACTRN against every other
  model, exact Wilcoxon tests over the five repeats, Wilcoxon tests over the 160 decision
  contexts, and bootstrap 95% intervals resampling contexts within repeats (10,000 draws,
  seed 20261016).

## Field case (Volve)

- 400 weekly decision windows (2008-06-22 to 2016-02-14), 10-week history, 8-week horizon.
- Four rolling origins with the same 60-window test blocks as v1. Each test block is preceded by
  a 30-window validation block; eight windows are removed before every validation and test
  block so that all targets used for fitting or selection precede the block they inform. All
  models use identical windows; clipping bounds come from training targets only.
- Decision example: latest window with four-week median field oil of at least 20,000 m3.
  Candidates are historical injection blocks that end before the decision date, within 5% of
  the median total of the latest 52 such blocks, rescaled to that total. The prediction under
  the implemented schedule is compared with the observed production.

## Exploratory analysis added after the run

`code/actrn_field_exploratory.py` evaluates the four baselines on the ACTRN representation in
the field case under the same blocks, grids and selection rule. It is labelled exploratory.
