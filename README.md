# ACTRN: Action-Conditioned Temporal Response Networks for waterflood injection decisions

Code, protocol and registered results for the manuscript *Action-Conditioned Temporal Response
Networks for Waterflood Injection Decision Optimisation* (Zhang, Yu, Li and Sun).

## Contents

| Path | Content |
|---|---|
| `code/actrn_model.py` | ACTRN: level-response decomposition, interaction and action response paths, contrastive response learning |
| `code/action_conditioned_decision.py` | ACTRN input representations, baselines, known-truth benchmark generator, field-case window builder, decision scoring |
| `code/actrn_experiments.py` | The registered experiment protocol: tuning, ablations, feature-matched baselines, robustness benchmark, leakage-free field evaluation, statistics |
| `code/make_actrn_tables.py`, `code/render_actrn_figures.py` | Manuscript tables and Figures 2 and 4-7 from the registered results |
| `code/preprocess.py`, `code/fieldwide_effects.py`, `code/baseline_crm.py`, `code/baseline_network.py`, `code/render_p17_full_figure_suite.py` | Volve preprocessing, weekly aggregation, descriptive CRM/correlation summaries and the field-state figure |
| `data/processed/` | Derived Volve tables used by the field case and context figures |
| `data/processed/actrn_v3/` | Registered results reported in the manuscript |
| `data/processed/actrn_v2/`, `data/processed/actrn_v1_20260716/` | Results of earlier model designs and protocols, kept for reference |
| `docs/PROTOCOL.md` | Experiment protocol fixed before the registered run |
| `tests/` | Regression tests (budget feasibility, leakage-free field blocks, feature scaling, response proxies, contrastive invariance, ablation wiring) |

## Reproduce

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python code/actrn_experiments.py --workers 8 --out results_rerun   # several CPU-hours
.venv/bin/python code/render_actrn_figures.py --results data/processed/actrn_v3
```

`--quick` runs one benchmark repeat with two configurations per model for a smoke test.
Results are deterministic for a fixed environment; tree-ensemble baselines can differ slightly
across scikit-learn versions.

## Data

The field case uses the daily production data of the Volve field released by Equinor ASA
(<https://www.equinor.com/energy/volve-data-sharing>) under the Equinor Open Data Licence;
the derived weekly tables in `data/processed/` contain data from Equinor ASA. The raw file is not
redistributed: place `Volve production data.xlsx` in `data/raw/volve/` and run
`code/preprocess.py` followed by `code/fieldwide_effects.py` to rebuild the derived tables.
Surface-wellhead positions are from the Norwegian Offshore Directorate FactPages.

## Licence

Code: MIT (see `LICENSE`). Data remain under the terms of their original providers.
