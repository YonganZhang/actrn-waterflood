# Registered ACTRN v2 results (16 October 2026)

Produced once by `python code/actrn_experiments.py --workers 32 --out data/processed/actrn_v2`
under `docs/PROTOCOL.md` (log: `run.log`; environment, seeds and output hashes: `manifest.json`).
`code_dirty: true` in the manifest reflects working-tree edits made while the run was in progress
(an output-identical refactor of the budget allocation and figure scripts); the running process
did not load them.

- `benchmark_*`: known-truth benchmark (`response = exp` main, `hill` robustness); `*_tuning.csv`
  lists every candidate configuration and marks the selected one.
- `field_*`: leakage-free rolling-origin field evaluation and the decision example
  (`field_case_backtest.csv` compares the prediction under the implemented schedule with observations).
- `exploratory_field_engineered_metrics.csv`: exploratory analysis added after the run
  (`code/actrn_field_exploratory.py`).
