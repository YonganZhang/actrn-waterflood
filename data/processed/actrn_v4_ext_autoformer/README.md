# Corrected Autoformer, registered rerun (16 October 2026)

Produced once by
`python code/actrn_extension.py --kinds Autoformer --skip-sensitivity --out data/processed/actrn_v4_ext_autoformer --workers 44`
under the amendment in `docs/PROTOCOL.md` (commit, seeds and output hashes: `manifest.json`; `run.log` holds the closing
summary). `code_dirty_paths` lists table, figure and release files, which the run does not import,
and one pre-existing deletion of an unused legacy plotting script. A first start of the same command
was stopped within seconds, before any job finished, because its log was being written into this
directory; nothing from it was kept.

The corrected Autoformer selects its auto-correlation delays per example in evaluation (the first
implementation used the batch-shared training form there) and follows the official encoder,
decoder and embedding design. Grids, training, candidate menu, selection rules, settings, repeats
r = 10-29 and seeds are those of `data/processed/actrn_v4_ext/`, whose Autoformer rows this
directory supersedes. The file layout is the same as there (`sequence_*`), restricted to Autoformer;
`sequence_stats.json` compares the registered v4 ACTRN with the corrected Autoformer.
