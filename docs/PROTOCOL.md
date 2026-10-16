# ACTRN experiment protocol (fixed before the registered run, 16 October 2026)

The complete protocol was executed once with `python code/actrn_experiments.py --workers 40`.
Outputs are in `data/processed/actrn_v3/` (`manifest.json` records environment, git commit,
seeds, timings and output hashes). Results of earlier model designs and protocols are kept in
`data/processed/actrn_v2/` and `data/processed/actrn_v1_20260716/` for reference only.

## Model (`code/actrn_model.py`)

- Forecast `Y_hat = f(x_H) + g(x_H, x_U)` with level `f = x_H W_B + b` and response
  `g = x_U W_A + (z_H * z_U) W_I + z_A W_Z`, where `z_H = tanh(x_H W_H + b_H)`,
  `z_U = tanh(x_U W_U + b_U)` and `z_A = tanh(x_U W_P + b_P)`.
- `x_H`: flattened history, terminal state, mean, standard deviation, least-squares slope, and
  response proxies (within-window correlation of each injector rate with the increments of every
  other series at lags 0 and 1). `x_U`: absolute, cumulative, incremental and saturating
  (`1 - exp(-U_kj / s_j)`, `s_j` = mean positive training injection of injector `j`) schedule.
- With several schedules per state (benchmark): `W_A` by ridge on context-centred data (penalty
  in {0.01, 0.1, 1, 10} chosen by the validation within-context loss); nonlinear paths by Adam on
  the within-context centred loss, mini-batches of 21 contexts, zero-initialised output weights,
  early stopping on the validation within-context loss; level `W_B` by ridge on `Y - g_hat`
  (penalty chosen by validation NMAE).
- Without such contrasts (field): `W_B` and `W_A` by joint ridge; nonlinear paths fitted to the
  residual by mean squared error with early stopping.
- Five initialisation seeds (model seed + 1000 k) averaged.

## Common settings

- Environment: see `requirements.lock.txt` (Python 3.11.15). One thread per process.
- Tuning: every model chooses one of eight pre-specified configurations (grids in
  `code/actrn_experiments.py::GRIDS`) by validation NMAE in physical units, separately for each
  benchmark repeat and each field origin, and is evaluated on the test data without refitting.
  No selection step reads test data.
- Model development used benchmark draws with separate seeds (`910000 + 1009 r`) that are not
  part of the reported repeats; field design choices were compared on validation blocks only.

## Known-truth benchmark

- Ten repeats (data seed `20260716 + 1009 r`, model seed `42 + r`, `r = 0..9`), 48 reservoirs
  each, grouped split 32/8/8 by reservoir; four decision states per test reservoir with 80
  equal-budget candidates shared by all models.
- Models: Ridge, random forest, HGB and MLP on raw inputs; the same four on the ACTRN
  representation `[x_H, x_U]`; ACTRN.
- Ablations (reusing the full ACTRN configuration selected in the same repeat): A1 pooled
  training instead of within-context contrasts; A2 no interaction path; A3 no response proxies;
  A4 raw action input `vec(U)`; A5 linear response `x_U W_A` only; A6 concatenation MLP of the
  same width instead of `z_H * z_U`.
- Robustness: the simulator's support response `1 - exp(-1.9 a)` is replaced by `a / (a + 0.55)`
  and the raw baselines, feature-matched baselines and ACTRN are rerun.
- Statistics: relative regret (%) as the main decision metric; for ACTRN against every other
  model, exact Wilcoxon tests over repeats, Wilcoxon tests over the 320 decision contexts, and
  bootstrap 95% intervals resampling contexts within repeats (10,000 draws, seed 20261016),
  for all repeats and separately for repeats 0-4 and 5-9.

## Field case (Volve)

- 400 weekly decision windows (2008-06-22 to 2016-02-14), 10-week history, 8-week horizon.
- Four rolling origins with 60-window test blocks. Each test block is preceded by a 30-window
  validation block; eight windows are removed before every validation and test block so that
  all targets used for fitting or selection precede the block they inform. All models use
  identical windows; clipping bounds come from training targets only.
- Models: persistence; Ridge, random forest, HGB and MLP on raw inputs and on the ACTRN
  representation; ACTRN.
- Decision example: latest window with four-week median field oil of at least 20,000 m3.
  Candidates are historical injection blocks that end before the decision date, within 5% of
  the median total of the latest 52 such blocks, rescaled to that total. The prediction under
  the implemented schedule is compared with the observed production.
