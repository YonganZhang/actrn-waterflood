# Field decision example at the implemented budget (16 October 2026)

Produced once by
`python code/actrn_field_case.py --budget implemented --out data/processed/actrn_v4_case`
under the field-case amendment in `docs/PROTOCOL.md` (commit, seed and output hashes:
`manifest.json`; `run.log` holds the printed summary). `code_dirty_paths` lists one pre-existing deletion of an unused legacy plotting script.

The decision date (24 May 2015), training blocks, model seed 747 and the validation-NMAE choice
among the nine candidates are those of the v4 example in `data/processed/actrn_v4/`. Before this run,
the same script with `--budget recent_median` reproduced the v4 `field_case_*.csv` files byte for
byte and the v4 summary exactly. The only change is the common total injection of the candidate
library: the total of the implemented schedule (528,095 m3) instead of the median of the latest 52
historical blocks, so that the selected and the implemented schedules are compared at the same
budget. The file layout is that of the v4 example; `field_case_summary.json` adds the budget rule,
the number of candidates predicted above the implemented schedule, and the F-4 and first-half shares
of the selected and implemented schedules.
