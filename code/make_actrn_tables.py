"""Write the manuscript tables and an in-text number sheet from the registered v3 results.

Usage (from the project root):
    .venv/bin/python code/make_actrn_tables.py

Outputs ``paper/fuel_2026/tables/*.tex`` (input by ``main_body.tex``) and
``paper/fuel_2026/tables/numbers.json`` (every number quoted in the text).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "processed" / "actrn_v3"
OUT = ROOT / "paper" / "fuel_2026" / "tables"
RAW = ["Ridge", "Random forest", "HGB", "MLP"]
ENG = [f"{m} (engineered)" for m in RAW]
ABL = ["ACTRN w/o contrastive learning", "ACTRN w/o interaction", "ACTRN w/o response proxies",
       "ACTRN raw action input", "ACTRN linear response", "ACTRN concatenation interaction"]
NAME = {"Ridge": "Ridge", "Random forest": "Random forest", "HGB": "HGB", "MLP": "MLP",
        "ACTRN": "ACTRN", "Persistence": "Persistence",
        "ACTRN w/o contrastive learning": "A1: pooled training (no contrasts)",
        "ACTRN w/o interaction": "A2: no interaction path $z_I$",
        "ACTRN w/o response proxies": "A3: no response proxies $\\rho(H_t)$",
        "ACTRN raw action input": "A4: raw action input $\\operatorname{vec}(U_t)$",
        "ACTRN linear response": "A5: linear response $x_U W_A$ only",
        "ACTRN concatenation interaction": "A6: concatenation MLP instead of $z_H \\odot z_U$"}
NAME.update({f"{m} (engineered)": f"{m} + ACTRN features" for m in RAW})


def ms(mean, sd, digits):
    return f"${mean:.{digits}f}\\pm{sd:.{digits}f}$"


def table(rows, header, cols, caption, label, spec, wide=False, notes=None):
    env = "widetable" if wide else "narrowtable"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{adjustbox}{max width=\\linewidth}",
             f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\",
             "\\midrule"]
    for r in rows:
        lines.append(r if r.startswith("\\") else r + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if notes:
        lines.append(f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\scriptsize {notes}}}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(RES / "benchmark_summary.csv").set_index(["response", "family"])
    stats = json.loads((RES / "benchmark_stats.json").read_text())
    field = pd.read_csv(RES / "field_rolling_metrics.csv")
    case = json.loads((RES / "field_case_summary.json").read_text())

    def row(resp, fam, cols):
        cells = [NAME[fam]]
        for col, d in cols:
            cells.append(ms(s.loc[(resp, fam), f"{col}_mean"], s.loc[(resp, fam), f"{col}_std"], d))
        return " & ".join(cells)

    pred_cols = [("nmae_pct", 2), ("nrmse_pct", 2), ("r2", 3)]
    dec_cols = [("decision_regret_pct", 2), ("optimal_selection_pct", 1),
                ("top_decile_selection_pct", 1)]
    fams = RAW + ENG + ["ACTRN"]
    pred_rows = [row("exp", f, pred_cols) for f in RAW] + ["\\midrule"] + \
                [row("exp", f, pred_cols) for f in ENG] + ["\\midrule"] + [row("exp", "ACTRN", pred_cols)]
    (OUT / "tab_prediction.tex").write_text(table(
        pred_rows, ["Model", "NMAE (\\%)", "NRMSE (\\%)", "$R^2$"], pred_cols,
        "Known-truth trajectory prediction on the group-held-out test reservoirs (mean $\\pm$ standard deviation over ten repeats). Models marked ``+ ACTRN features'' receive the same history and action representation $[x_H, x_U]$ as ACTRN.",
        "tab:prediction-summary", "lrrr"))
    dec_rows = [row("exp", f, dec_cols) for f in RAW] + ["\\midrule"] + \
               [row("exp", f, dec_cols) for f in ENG] + ["\\midrule"] + [row("exp", "ACTRN", dec_cols)]
    (OUT / "tab_decision.tex").write_text(table(
        dec_rows, ["Model", "Regret (\\%)", "Exact optimum (\\%)", "Top decile (\\%)"],
        dec_cols,
        "Known-truth schedule selection over 320 decision contexts with 80 equal-budget candidates each (mean $\\pm$ standard deviation over ten repeats). Regret is the relative shortfall of the selected candidate's true cumulative oil from the best candidate, as a percentage of the best.",
        "tab:known-truth", "lrrr"))
    abl_rows = [row("exp", f, pred_cols[:1] + dec_cols[:2]) for f in ["ACTRN"] + ABL]
    abl_rows += ["\\midrule", "\\multicolumn{4}{l}{\\emph{Alternative saturation benchmark}} \\\\"]
    abl_rows += [row("hill", f, pred_cols[:1] + dec_cols[:2]) for f in
                 ["HGB", "HGB (engineered)", "Ridge (engineered)", "ACTRN"]]
    (OUT / "tab_ablation.tex").write_text(table(
        abl_rows, ["Model", "NMAE (\\%)", "Regret (\\%)", "Exact optimum (\\%)"], None,
        "Component ablations of ACTRN on the known-truth benchmark and results on the alternative saturation benchmark (mean $\\pm$ standard deviation over ten repeats; regret is relative regret). Each ablation changes one component and reuses the configuration selected for the full model in the same repeat.",
        "tab:ablation", "lrrr"))
    fp = field.pivot(index="family", columns="fold", values="nmae_pct")
    fsum = field.groupby("family")[["nmae_pct", "nrmse_pct", "r2"]].mean()
    frows = []
    for group in (["Persistence"], RAW, ENG, ["ACTRN"]):
        for f in group:
            frows.append(" & ".join([NAME[f]] + [f"{fp.loc[f, k]:.1f}" for k in range(4)] +
                                    [f"{fsum.loc[f, 'nmae_pct']:.1f}", f"${fsum.loc[f, 'r2']:.2f}$"]))
        frows.append("\\midrule")
    frows.pop()
    dates = field.drop_duplicates("fold").sort_values("fold")
    origin_note = "; ".join(f"origin {int(r.fold) + 1}: {r.test_start} to {r.test_end}"
                            for r in dates.itertuples())
    (OUT / "tab_field.tex").write_text(table(
        frows, ["Model", "O1", "O2", "O3", "O4", "Mean NMAE", "Mean $R^2$"], None,
        "Leakage-free rolling-origin field forecasting on the Volve record: NMAE (\\%) by origin and means over the four origins.",
        "tab:field", "lrrrrrr", wide=True,
        notes=f"Test blocks of 60 weekly decision windows ({origin_note})."))

    st = stats["exp"]
    numbers = {
        "actrn": {k: round(float(s.loc[("exp", "ACTRN"), f"{k}_mean"]), 3) for k in
                  ("nmae_pct", "nrmse_pct", "r2", "decision_regret_pct", "optimal_selection_pct",
                   "top_decile_selection_pct")},
        "families": {f: {k: round(float(s.loc[("exp", f), f"{k}_mean"]), 3) for k in
                         ("nmae_pct", "r2", "decision_regret_pct", "optimal_selection_pct",
                          "top_decile_selection_pct")} for f in fams + ABL},
        "hill": {f: {k: round(float(s.loc[("hill", f), f"{k}_mean"]), 3) for k in
                     ("nmae_pct", "decision_regret_pct", "optimal_selection_pct")}
                 for f in RAW + ENG + ["ACTRN"]},
        "stats_exp": {sub: {o: {"ctx": st[sub][o]["context_regret_pct"],
                                "regret": st[sub][o]["decision_regret_pct"],
                                "nmae": st[sub][o]["nmae_pct"]}
                            for o in st[sub]} for sub in st},
        "stats_hill": {sub: {o: {"ctx": stats["hill"][sub][o]["context_regret_pct"],
                                 "regret": stats["hill"][sub][o]["decision_regret_pct"]}
                             for o in stats["hill"][sub]} for sub in stats["hill"]},
        "field_mean": fsum.round(3).to_dict(orient="index"),
        "field_by_fold_nmae": fp.round(2).to_dict(orient="index"),
        "field_case": case,
        "within_context_variance_share": within_context_share(),
    }
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps({k: numbers[k] for k in ("actrn", "field_mean")}, indent=1))



def within_context_share() -> dict:
    """Share of the variance of cumulative training oil that lies between schedules of one state."""
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT / "code"))
    from actrn_experiments import FRESH_REPEATS, REGISTERED_REPEATS, benchmark_data

    shares = []
    for r in REGISTERED_REPEATS + FRESH_REPEATS:
        _, _, _, y, context_ids, split, _ = benchmark_data(r, "exp")
        value = y[split["train"]].sum(axis=1)
        _, inv = np.unique(context_ids[split["train"]], return_inverse=True)
        means = np.bincount(inv, weights=value) / np.bincount(inv)
        shares.append(float(np.mean((value - means[inv]) ** 2) / np.var(value)))
    return {"mean": float(np.mean(shares)), "min": float(min(shares)), "max": float(max(shares))}


# ----------------------------------------------------------------- supplementary tables
def _fmt_params(p: dict) -> str:
    alias = {"latent_dim": "latent", "learning_rate": "lr", "l2": "$\\ell_2$",
             "alpha": "$\\alpha$", "max_depth": "depth", "min_samples_leaf": "min leaf",
             "max_leaf_nodes": "leaf nodes", "hidden_layer_sizes": "hidden",
             "learning_rate_init": "lr"}
    out = []
    for k, v in p.items():
        if isinstance(v, (list, tuple)):
            v = "(" + ", ".join(str(x) for x in v) + ")"
        elif v is None:
            v = "none"
        out.append(f"{alias.get(k, k)} {v}")
    return "; ".join(out)


def supplementary() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "code"))
    from actrn_experiments import GRIDS

    lines = ["\\begin{longtable}{p{0.16\\linewidth}p{0.38\\linewidth}p{0.38\\linewidth}}",
             "\\caption{Tuning grids (eight configurations per model). Selection used validation NMAE.}"
             "\\label{tab:s-grids}\\\\", "\\toprule", "Model & Benchmark & Field \\\\", "\\midrule"]
    for model in ("ACTRN", "Ridge", "Random forest", "HGB", "MLP"):
        def axes(grid):
            alias = {"latent_dim": "latent", "learning_rate": "lr", "l2": "$\\ell_2$",
                     "alpha": "$\\alpha$", "max_depth": "max depth",
                     "min_samples_leaf": "min leaf", "max_leaf_nodes": "max leaf nodes",
                     "hidden_layer_sizes": "hidden", "learning_rate_init": "lr"}

            def show(v):
                if v is None:
                    return "none"
                if isinstance(v, (list, tuple)):
                    return "(" + ", ".join(str(x) for x in v) + ")"
                return f"{v:g}" if isinstance(v, float) else str(v)

            parts = []
            for k in grid[0]:
                vals = sorted({g[k] for g in grid}, key=lambda v: (v is None, v))
                parts.append(f"{alias[k]} " + "\\{" + ", ".join(show(v) for v in vals) + "\\}")
            return "; ".join(parts)
        lines.append(f"{model} & {axes(GRIDS['benchmark'][model])} & {axes(GRIDS['field'][model])} \\\\")
    lines += ["\\bottomrule", "\\end{longtable}",
              "\\noindent Fixed settings: ACTRN ensemble of five initialisation seeds; benchmark 900 epochs, "
              "patience 80, mini-batches of 21 states, ridge penalties $\\{0.01, 0.1, 1, 10\\}$ for $W_A$ "
              "(within-state validation loss) and $\\{0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000\\}$ for $W_B$ "
              "(validation NMAE); field 1200 epochs, patience 100, mini-batches of 128 windows and joint ridge "
              "penalties $\\{1, 3, 10, 30, 100, 300, 1000, 3000\\}$; random forest 240 trees with maximum feature "
              "fraction 0.8; HGB 180 iterations with $\\ell_2$ regularisation 1.0; MLP at most 700 iterations "
              "with early stopping on 15\\% of the training data and patience 35.\n"]
    (OUT / "supp_grids.tex").write_text("\n".join(lines) + "\n")

    runs = pd.read_csv(RES / "benchmark_runs.csv")
    rows = ["\\begin{longtable}{p{0.31\\linewidth}rp{0.28\\linewidth}rr}",
            "\\caption{Configurations selected per repeat and test results on the main benchmark.}"
            "\\label{tab:s-repeats}\\\\", "\\toprule",
            "Model & Repeat & Selected configuration & NMAE (\\%) & Regret (\\%) \\\\",
            "\\midrule",
            "\\endhead"]
    for fam in ["ACTRN"] + ABL + RAW + ENG:
        sub = runs[(runs.response == "exp") & (runs.family == fam)].sort_values("repeat")
        for r in sub.itertuples():
            rows.append(f"{NAME[fam]} & {r.repeat + 1} & {_fmt_params(json.loads(r.params))} & "
                        f"{r.nmae_pct:.2f} & {r.decision_regret_pct:.2f} \\\\")
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    rows.append("\\end{longtable}")
    (OUT / "supp_repeats.tex").write_text("\n".join(rows) + "\n")

    field = pd.read_csv(RES / "field_rolling_metrics.csv")
    rows = ["\\begin{longtable}{p{0.24\\linewidth}rrrrp{0.29\\linewidth}}",
            "\\caption{Field configurations selected per origin and test metrics.}\\label{tab:s-field}\\\\",
            "\\toprule", "Model & Origin & NMAE (\\%) & NRMSE (\\%) & $R^2$ & Selected configuration \\\\",
            "\\midrule", "\\endhead"]
    for fam in ["Persistence"] + RAW + ENG + ["ACTRN"]:
        for r in field[field.family == fam].sort_values("fold").itertuples():
            cfg = _fmt_params(json.loads(r.params)) if r.params != "{}" else "--"
            rows.append(f"{NAME[fam]} & {int(r.fold) + 1} & {r.nmae_pct:.1f} & {r.nrmse_pct:.1f} & "
                        f"${r.r2:.2f}$ & {cfg} \\\\")
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    rows.append("\\end{longtable}")
    (OUT / "supp_field.tex").write_text("\n".join(rows) + "\n")



def supplementary_field_blocks() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "code"))
    from actrn_experiments import FIELD_FOLD_ENDS, FIELD_TEST, field_blocks
    from action_conditioned_decision import build_field_case_dataset

    _, _, _, dates = build_field_case_dataset()
    d = lambda i: str(dates[i].date())  # noqa: E731
    rows = ["\\begin{tabular}{lllll}", "\\toprule",
            "Origin & Training decision dates & Validation decision dates & Test decision dates & Last training target week \\\\",
            "\\midrule"]
    for k, end in enumerate(FIELD_FOLD_ENDS):
        start = end - FIELD_TEST
        b = field_blocks(start)
        tr, va = b["train"], b["validation"]
        last_target = str((dates[tr[-1]] + pd.Timedelta(weeks=8)).date())
        rows.append(f"{k + 1} & {d(tr[0])} -- {d(tr[-1])} ({len(tr)}) & {d(va[0])} -- {d(va[-1])} (30) & "
                    f"{d(start)} -- {d(end - 1)} (60) & {last_target} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "supp_field_blocks.tex").write_text("\n".join(rows) + "\n")
    bt = pd.read_csv(RES / "field_case_backtest.csv")
    dec = pd.read_csv(RES / "field_case_decision.csv")
    rows = ["\\begin{tabular}{rrrrrr}", "\\toprule",
            "Week & Selected F-4 & Selected F-5 & Predicted (selected) & Predicted (implemented) & Observed (implemented) \\\\",
            "\\midrule"]
    for (_, b), (_, s) in zip(bt.iterrows(), dec.iterrows()):
        rows.append(f"{int(b.step)} & {s.injector_1_m3:,.0f} & {s.injector_2_m3:,.0f} & "
                    f"{s.predicted_field_oil_m3:,.0f} & {b.predicted_under_implemented_m3:,.0f} & "
                    f"{b.observed_field_oil_m3:,.0f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "supp_field_case.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
    supplementary()
    supplementary_field_blocks()
