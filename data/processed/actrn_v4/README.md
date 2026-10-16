# Registered ACTRN results (16 October 2026)

Produced once by `python code/actrn_experiments.py --workers 44 --out data/processed/actrn_v4`
under `docs/PROTOCOL.md` (log: `run.log`; environment, git commit, seeds and output hashes:
`manifest.json`). `code_dirty_paths` in the manifest lists one pre-existing deletion of an unused
legacy plotting script; every file the run imports was committed at the recorded commit.

- `benchmark_runs.csv`, `benchmark_summary.csv`: selected candidate per repeat, setting and model
  family (settings: base/exp, base/hill, channelised/exp; repeats 10-29) and ablations.
- `benchmark_tuning.csv`: all nine candidates (eight configurations and their average) with
  validation and test scores; `selected` marks the candidate chosen by validation regret.
- `benchmark_summary_nmae_selection.csv`: the same summary when candidates are chosen by
  validation NMAE.
- `benchmark_decisions.csv`: every test decision context of every selected model, with regret and
  gain capture.
- `benchmark_state_dependence.csv`: model-free regret of a random candidate and of the best
  state-blind ranking for every test decision context.
- `benchmark_learning_curve.csv`: reduced-training-data experiment.
- `benchmark_stats.json`: ACTRN against every other family (exact Wilcoxon over repeat means,
  hierarchical bootstrap intervals, context-level Wilcoxon).
- `field_*`: leakage-free rolling-origin field evaluation and the decision example
  (`field_case_backtest.csv` compares the prediction under the implemented schedule with observations).
