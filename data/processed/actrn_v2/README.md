# ACTRN v2 results (16 October 2026) — earlier model design, kept for reference

Produced once by the v2 runner (git tag `actrn-v2-registered` in the project history) with the
same leakage-free field split and equal-budget tuning as the reported results, but with the
earlier ACTRN design (fusion layer, pooled trajectory loss with a pairwise ranking term, no
response proxies) and five benchmark repeats. The results reported in the manuscript are in
`data/processed/actrn_v3/`.

- `benchmark_*`: known-truth benchmark (`response = exp` main, `hill` robustness).
- `field_*`: leakage-free rolling-origin field evaluation and decision example.
- `exploratory_field_engineered_metrics.csv`: field baselines on the ACTRN representation,
  evaluated after the v2 run.
- `manifest.json` records environment, seeds and output hashes; `code_dirty: true` reflects an
  output-identical refactor made in the working tree while the run was in progress.
