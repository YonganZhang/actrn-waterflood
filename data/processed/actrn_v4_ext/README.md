# Registered ACTRN v4 extension (16 October 2026)

Produced once by `python code/actrn_extension.py --out data/processed/actrn_v4_ext --workers 44`
under the extension section of `docs/PROTOCOL.md`
(commit, seeds and output hashes: `manifest.json`; `run.log` holds the closing summary). The run
uses the repeats r = 10-29 and the seeds of `data/processed/actrn_v4/` and refits no v4 model; the
ACTRN rows in the statistics are the registered v4 results. Deep models used PyTorch 2.14.1 (CPU)
and reformer-pytorch 1.4.4 (`code/requirements-actrn.lock.txt`). `code_dirty_paths` lists the table
and figure scripts, which the run does not import, and one pre-existing deletion of an unused legacy
plotting script.

**Superseded rows:** the Autoformer rows in every file here come from an implementation that
selected its auto-correlation delays over the whole batch also in evaluation. They are kept as
provenance only; the manuscript uses the corrected rerun in `data/processed/actrn_v4_ext_autoformer/`
(amendment in `docs/PROTOCOL.md`). The other four sequence models and the sensitivity runs are unaffected.

- `sequence_runs.csv`, `sequence_summary.csv`: RNN, LSTM, Transformer, Autoformer and Reformer,
  selected candidate per repeat and setting (base/exp, base/hill, channelised/exp), chosen by
  validation regret from the same nine-candidate menu as the v4 families.
- `sequence_tuning.csv`: all candidates with validation and test scores.
- `sequence_summary_nmae_selection.csv`: the same summary when candidates are chosen by validation
  NMAE.
- `sequence_decisions.csv`: every test decision context of every selected sequence model.
- `sequence_stats.json`: registered ACTRN against each sequence model (exact Wilcoxon over repeat
  means, hierarchical bootstrap intervals, context-level Wilcoxon), as in v4.
- `sensitivity_hyperparameters.csv`: ACTRN with one hyperparameter changed at a time around the
  default (latent dimension, l2 penalty, ensemble size, learning rate); base/exp and
  channelised/exp, all 20 repeats.
- `sequence_field_tuning.csv`, `sequence_field_metrics.csv`: leakage-free rolling-origin field
  evaluation of the sequence models (selection by validation NMAE).
