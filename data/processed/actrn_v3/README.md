# Registered ACTRN results (16 October 2026)

Produced once by `python code/actrn_experiments.py --workers 40 --out data/processed/actrn_v3`
under `docs/PROTOCOL.md` (log: `run.log`; environment, git commit, seeds and output hashes:
`manifest.json`). `code_dirty_paths` in the manifest lists one pre-existing deletion of an unused
legacy plotting script; every file the run imports was committed at the recorded commit.

- `benchmark_*`: known-truth benchmark, ten repeats (`response = exp` main, `hill` robustness);
  `benchmark_tuning.csv` lists every candidate configuration and marks the selected one;
  `benchmark_stats.json` holds the paired statistics for all repeats and for repeats 0-4 and 5-9.
- `field_*`: leakage-free rolling-origin field evaluation and the decision example
  (`field_case_backtest.csv` compares the prediction under the implemented schedule with observations).
