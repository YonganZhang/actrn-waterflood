# ACTRN experiment protocol (fixed before the registered run, 16 October 2026)

The complete protocol was executed once with `python code/actrn_experiments.py --workers 44`.
Outputs are in `data/processed/actrn_v4/` (`manifest.json` records environment, git commit,
seeds, timings and output hashes). Results of earlier model designs and protocols are kept in
`data/processed/actrn_v3/`, `actrn_v2/` and `actrn_v1_20260716/` for reference only.

## Model (`code/actrn_model.py`)

- Forecast `Y_hat = f(x_H) + g(x_H, x_U)` with level `f = x_H W_B + b` and response
  `g = x_U W_A + (z_H * z_U) W_I + z_A W_Z`, where `z_H = tanh(x_H W_H + b_H)`,
  `z_U = tanh(x_U W_U + b_U)` and `z_A = tanh(x_U W_P + b_P)`.
- `x_H`: flattened history, terminal state, mean, standard deviation, least-squares slope, and
  response proxies: within-window correlations of each injector rate with the increments of
  every other series (lags 0 and 1) and within-window ridge coefficients of those increments on
  the current and previous rates of all injectors. `x_U`: absolute, cumulative, incremental and
  saturating (`1 - exp(-U_kj / s_j)`) schedule.
- With several schedules per state (benchmark): `W_A` by ridge on context-centred data; nonlinear
  paths by Adam on the within-context centred loss, mini-batches of 21 contexts, zero-initialised
  output weights, early stopping on the validation centred loss; level `W_B` by ridge on `Y - g_hat`.
  Without such contrasts (field): joint ridge for `W_B`, `W_A`; nonlinear paths fitted to the
  residual by mean squared error. Five initialisation seeds averaged.

## Benchmark

- Settings: base population with saturating response `1 - exp(-1.9 a)` (main), base population
  with `a / (a + 0.55)` (robustness), channelised population (each producer connected to one
  injector by a high-conductivity channel) with the main response.
- Twenty repeats `r = 10..29` (data seed `20260716 + 1009 r`, model seed `42 + r`); repeats 0-9
  were used by earlier protocols. 48 reservoirs per repeat, grouped split 32/8/8; four decision
  states per validation and test reservoir with 80 equal-budget candidates (test libraries: data
  seed + 1776; validation libraries: data seed + 2999).
- Every model family (ACTRN; Ridge, random forest, HGB, MLP on raw inputs and on `[x_H, x_U]`)
  chooses among the same menu: its eight grid configurations and their averaged prediction. The
  choice minimises validation regret, ties broken by validation NMAE; selection by validation NMAE
  is also reported.
- Ablations (reusing the full model's candidate) on base and channelised populations: pooled
  training; no interaction path; no response proxies; correlation proxies only; raw action input;
  linear response only; concatenation layer instead of `z_H * z_U`.
- Data efficiency: ACTRN, its linear response, Ridge and HGB on `[x_H, x_U]` with training data
  reduced to 16 reservoirs, 3 schedules per state, or 8 reservoirs (nested subsets).
- Model-free state dependence: regret of a random candidate and of the best state-blind ranking
  (candidates ranked by their mean true value over all other test states of the repeat).
- Metrics: relative regret (primary), gain capture `100 (V_sel - V_mean) / (V_best - V_mean)`,
  exact-optimum and top-decile selection, NMAE, NRMSE, R2.
- Statistics: ACTRN against every other family; exact Wilcoxon signed-rank test over the 20 repeat
  means (primary); hierarchical bootstrap (repeats, then reservoirs within repeats, after averaging
  the four states of a reservoir; 10,000 draws, seed 20261016); context-level Wilcoxon (secondary).

## Field case (Volve)

- 400 weekly decision windows (2008-06-22 to 2016-02-14), 10-week history, 8-week horizon; four
  rolling origins with 60-window test blocks, each preceded by a 30-window validation block, with
  eight windows removed before every validation and test block; clipping bounds from training
  targets only.
- Same model families and nine-candidate menu, chosen by validation NMAE.
- Decision example: latest window with four-week median field oil of at least 20,000 m3;
  candidates are historical injection blocks that end before the decision date, within 5% of the
  median total of the latest 52 such blocks, rescaled to that total.

## Extension: deep sequence baselines and hyperparameter sensitivity (fixed 16 October 2026)

Run once by `code/actrn_extension.py` on the same repeats (r = 10-29), settings and seeds; no model
of the registered run is refitted, and the ACTRN rows of the statistics are the registered results.

- Deep sequence baselines: Elman RNN, LSTM, Transformer encoder, Autoformer (series decomposition
  and auto-correlation, encoder-decoder) and Reformer (reformer-pytorch 1.4.4). RNN, LSTM,
  Transformer and Reformer read 18 time steps (10 history steps with all channels, 8 future steps
  with the injection rates, a future flag and zeros for the other channels) and predict from the
  last 8 steps; Autoformer encodes the history, receives the injection rates as known covariates
  and starts its decoder from the latest 5 oil steps.
- Training: mean squared error, Adam (weight decay 1e-4), mini-batches of 128, at most 300 epochs,
  early stopping after 30 epochs without validation improvement, best weights restored; one CPU
  thread per job.
- Grids (eight configurations each): benchmark d_model {32, 64} x lr {1e-3, 3e-3} x dropout
  {0, 0.1}; field d_model {16, 32} x lr {1e-3, 3e-3} x dropout {0.1, 0.3}. The same nine-candidate
  menu and selection rules as all other families.
- ACTRN sensitivity, one hyperparameter at a time around the default (latent 16, learning rate
  0.002, l2 1e-3, five seeds): latent {4, 8, 16, 32, 64}; l2 {1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
  ensemble {1, 2, 5, 10}; learning rate {5e-4, 1e-3, 2e-3, 5e-3}; base/exp and channelised/exp,
  all 20 repeats, described without selection.
- Statistics against ACTRN as for the registered run.

## Amendment: Autoformer correction (fixed 16 October 2026)

A code review after the extension run found that the first Autoformer implementation selected the
auto-correlation delays from the mean over the whole batch also in evaluation, so that a
prediction depended on the other examples evaluated with it. The corrected implementation follows
the official code (github.com/thuml/Autoformer): delays shared across the mini-batch in training
and selected per example in evaluation, seasonal layer normalisation in encoder and decoder,
decomposition of the full oil history before taking the decoder start, bias-free 1x1-convolution
feed-forward layers with GELU, and bias-free circular token embedding plus covariate embedding.
Kept adaptations: moving-average kernel 5, a one-channel target decoder (the official model
decodes all variables and keeps the target afterwards), injection rates in place of calendar
features, default initialisation of the token convolution. Grids, training, candidate menu, selection rules, settings, repeats, seeds and
the field protocol are unchanged. Only Autoformer is rerun, once
(`code/actrn_extension.py --kinds Autoformer --skip-sensitivity`), into
`data/processed/actrn_v4_ext_autoformer/`; its rows supersede the Autoformer rows of
`data/processed/actrn_v4_ext/`, which are kept as provenance. Reformer keeps the reference
behaviour of reformer-pytorch, which draws new hash rotations at every forward pass.
Mean-squared-error reductions are reported as the mean over repeats (or field origins) of RMSE
squared.
