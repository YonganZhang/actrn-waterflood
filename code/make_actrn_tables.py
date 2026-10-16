"""Write the manuscript tables and an in-text number sheet from the registered v4 results.

Usage (from the project root):
    .venv/bin/python code/make_actrn_tables.py

Outputs ``paper/fuel_2026/tables/*.tex`` (input by ``main_body.tex`` and ``supplementary.tex``)
and ``paper/fuel_2026/tables/numbers.json`` (every number quoted in the text).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "processed" / "actrn_v4"
OUT = ROOT / "paper" / "fuel_2026" / "tables"
sys.path.insert(0, str(ROOT / "code"))

RAW = ["Ridge", "Random forest", "HGB", "MLP"]
ENG = [f"{m} (engineered)" for m in RAW]
FAMILIES = RAW + ENG + ["ACTRN"]
ABL = ["ACTRN w/o contrastive learning", "ACTRN w/o interaction", "ACTRN w/o response proxies",
       "ACTRN correlation proxies only", "ACTRN raw action input", "ACTRN linear response",
       "ACTRN concatenation interaction"]
SETTINGS = {"base": ("base", "exp"), "hill": ("base", "hill"), "channel": ("channelised", "exp")}
NAME = {"Ridge": "Ridge", "Random forest": "Random forest", "HGB": "HGB", "MLP": "MLP",
        "ACTRN": "ACTRN", "Persistence": "Persistence",
        "ACTRN w/o contrastive learning": "A1: pooled training (no contrasts)",
        "ACTRN w/o interaction": "A2: no interaction path $z_I$",
        "ACTRN w/o response proxies": "A3: no response proxies $\\rho(H_t)$",
        "ACTRN correlation proxies only": "A3b: correlation proxies only",
        "ACTRN raw action input": "A4: raw action input $\\operatorname{vec}(U_t)$",
        "ACTRN linear response": "A5: linear response $x_U W_A$ only",
        "ACTRN concatenation interaction": "A6: concatenation layer instead of $z_H \\odot z_U$"}
NAME.update({f"{m} (engineered)": f"{m} + ACTRN features" for m in RAW})
SHORT = {"Ridge": "Ridge", "Random forest": "RF", "HGB": "HGB", "MLP": "MLP", "ACTRN": "ACTRN"}
SHORT.update({f"{m} (engineered)": f"{SHORT[m]}+F" for m in RAW})
LC_NAME = {"ACTRN": "ACTRN", "ACTRN linear response": "A5 linear",
           "Ridge (engineered)": "Ridge+F", "HGB (engineered)": "HGB+F"}
SUBSET_LABEL = {"32x18x6": "full: 32 reservoirs",
                "16x18x6": "half: 16 reservoirs", "32x18x3": "half: 3 schedules per state",
                "8x18x6": "quarter: 8 reservoirs"}


def ms(mean, sd, digits):
    return f"${mean:.{digits}f}\\pm{sd:.{digits}f}$"


def table(rows, header, caption, label, spec, wide=False, notes=None, pre=None):
    env = "widetable" if wide else "narrowtable"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{adjustbox}{max width=\\linewidth}",
             f"\\begin{{tabular}}{{{spec}}}", "\\toprule"] + (pre or []) + \
            [" & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(r if r.startswith("\\") else r + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if notes:
        lines.append(f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\scriptsize {notes}}}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


def load():
    s = pd.read_csv(RES / "benchmark_summary.csv").set_index(["population", "response", "family"])
    return {
        "summary": s,
        "nmae_sel": pd.read_csv(RES / "benchmark_summary_nmae_selection.csv").set_index(
            ["population", "response", "family"]),
        "stats": json.loads((RES / "benchmark_stats.json").read_text()),
        "dependence": pd.read_csv(RES / "benchmark_state_dependence.csv"),
        "curve": pd.read_csv(RES / "benchmark_learning_curve.csv"),
        "runs": pd.read_csv(RES / "benchmark_runs.csv"),
        "field": pd.read_csv(RES / "field_rolling_metrics.csv"),
        "case": json.loads((RES / "field_case_summary.json").read_text()),
    }


def dependence_reference(dep, population, response):
    d = dep[(dep.population == population) & (dep.response == response)]
    capture = 100 * (1 - d.state_blind_regret_pct / d.random_regret_pct)
    rep = d.assign(capture=capture).groupby("repeat")[["random_regret_pct",
                                                       "state_blind_regret_pct", "capture"]].mean()
    return {"random_regret": (rep.random_regret_pct.mean(), rep.random_regret_pct.std()),
            "blind_regret": (rep.state_blind_regret_pct.mean(), rep.state_blind_regret_pct.std()),
            "blind_capture": (rep.capture.mean(), rep.capture.std())}


def main_tables(R) -> dict:
    s = R["summary"]

    def cell(setting, fam, col, d):
        key = SETTINGS[setting] + (fam,)
        return ms(s.loc[key, f"{col}_mean"], s.loc[key, f"{col}_std"], d)

    pred_cols = [("nmae_pct", 2), ("nrmse_pct", 2), ("r2", 3)]
    rows = []
    for group in (RAW, ENG, ["ACTRN"]):
        rows += [" & ".join([NAME[f]] + [cell("base", f, c, d) for c, d in pred_cols])
                 for f in group] + ["\\midrule"]
    (OUT / "tab_prediction.tex").write_text(table(
        rows[:-1], ["Model", "NMAE (\\%)", "NRMSE (\\%)", "$R^2$"],
        "Known-truth trajectory prediction on the group-held-out test reservoirs of the base "
        "population (mean $\\pm$ standard deviation over 20 repeats). Models marked ``+ ACTRN "
        "features'' receive the same history and action representation $[x_H, x_U]$ as ACTRN.",
        "tab:prediction-summary", "lrrr"))

    dec_cols = [("decision_regret_pct", 2), ("gain_capture_pct", 1), ("optimal_selection_pct", 1)]
    rows = []
    for group in (RAW, ENG, ["ACTRN"]):
        rows += [" & ".join([NAME[f]] + [cell(st, f, c, d) for st in ("base", "channel")
                                         for c, d in dec_cols]) for f in group] + ["\\midrule"]
    refs = {st: dependence_reference(R["dependence"], *SETTINGS[st]) for st in ("base", "channel")}
    rows.append("\\multicolumn{7}{l}{\\emph{Model-free references}} \\\\")
    rows.append(" & ".join(["Random candidate"] + sum(
        ([ms(*refs[st]["random_regret"], 2), "$0$", "$1.3$"] for st in ("base", "channel")), [])))
    rows.append(" & ".join(["State-blind reference ranking"] + sum(
        ([ms(*refs[st]["blind_regret"], 2), ms(*refs[st]["blind_capture"], 1), "--"]
         for st in ("base", "channel")), [])))
    (OUT / "tab_decision.tex").write_text(table(
        rows, ["Model", "Regret (\\%)", "Gain capture (\\%)", "Exact optimum (\\%)",
               "Regret (\\%)", "Gain capture (\\%)", "Exact optimum (\\%)"],
        "Known-truth schedule selection over 640 decision contexts per population, each with 80 "
        "equal-budget candidates (mean $\\pm$ standard deviation over 20 repeats). Regret is the "
        "shortfall of the selected candidate's true cumulative oil from the best candidate, as a "
        "percentage of the best; gain capture is the share of the attainable gain over the average "
        "candidate that the selection realises. The state-blind reference ranks the candidates "
        "of a state by their mean true value over all other test states of the repeat.",
        "tab:known-truth", "lrrrrrr", wide=True,
        pre=["& \\multicolumn{3}{c}{Base population} & \\multicolumn{3}{c}{Channelised population} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}"]))

    abl_cols = [("nmae_pct", 2), ("decision_regret_pct", 2), ("gain_capture_pct", 1)]
    rows = [" & ".join([NAME[f]] + [cell(st, f, c, d) for st in ("base", "channel")
                                    for c, d in abl_cols]) for f in ["ACTRN"] + ABL]
    rows.insert(1, "\\midrule")
    (OUT / "tab_ablation.tex").write_text(table(
        rows, ["Model", "NMAE (\\%)", "Regret (\\%)", "Gain capture (\\%)",
               "NMAE (\\%)", "Regret (\\%)", "Gain capture (\\%)"],
        "Component ablations of ACTRN (mean $\\pm$ standard deviation over 20 repeats). Each "
        "ablation changes one component and reuses the candidate selected for the full model in "
        "the same repeat.", "tab:ablation", "lrrrrrr", wide=True,
        pre=["& \\multicolumn{3}{c}{Base population} & \\multicolumn{3}{c}{Channelised population} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}"]))

    field = R["field"]
    fp = field.pivot(index="family", columns="fold", values="nmae_pct")
    fsum = field.groupby("family")[["nmae_pct", "nrmse_pct", "r2"]].mean()
    rows = []
    for group in (["Persistence"], RAW, ENG, ["ACTRN"]):
        rows += [" & ".join([NAME[f]] + [f"{fp.loc[f, k]:.1f}" for k in range(4)] +
                            [f"{fsum.loc[f, 'nmae_pct']:.1f}", f"${fsum.loc[f, 'r2']:.2f}$"])
                 for f in group] + ["\\midrule"]
    dates = field.drop_duplicates("fold").sort_values("fold")
    note = "; ".join(f"origin {int(r.fold) + 1}: {r.test_start} to {r.test_end}"
                     for r in dates.itertuples())
    (OUT / "tab_field.tex").write_text(table(
        rows[:-1], ["Model", "O1", "O2", "O3", "O4", "Mean NMAE", "Mean $R^2$"],
        "Leakage-free rolling-origin field forecasting on the Volve record: NMAE (\\%) by origin "
        "and means over the four origins.", "tab:field", "lrrrrrr", wide=True,
        notes=f"Test blocks of 60 weekly decision windows ({note})."))
    return {"refs": refs, "field_by_fold": fp, "field_mean": fsum}


def numbers(R, extra) -> dict:
    s, st = R["summary"], R["stats"]

    def fam(setting, f):
        key = SETTINGS[setting] + (f,)
        return {c: round(float(s.loc[key, f"{c}_mean"]), 4) for c in
                ("nmae_pct", "nrmse_pct", "r2", "decision_regret_pct", "gain_capture_pct",
                 "optimal_selection_pct", "top_decile_selection_pct")}

    curve = R["curve"]
    lc = (curve.groupby(["population", "family", "train_subset"])["decision_regret_pct"].mean()
          .round(4).unstack("train_subset"))
    out = {
        "families": {st_: {f: fam(st_, f) for f in FAMILIES + (ABL if st_ != "hill" else [])}
                     for st_ in SETTINGS},
        "stats": {st_: {o: {m: dict(e[m]) for m in ("nmae_pct", "decision_regret_pct",
                                                    "gain_capture_pct")}
                        for o, e in st["/".join(SETTINGS[st_])].items()} for st_ in SETTINGS},
        "references": {k: {kk: [round(float(x), 4) for x in vv] for kk, vv in v.items()}
                       for k, v in extra["refs"].items()},
        "learning_curve": {p: lc.loc[p].to_dict(orient="index") for p in lc.index.levels[0]},
        "field_mean": extra["field_mean"].round(3).to_dict(orient="index"),
        "field_by_fold_nmae": extra["field_by_fold"].round(2).to_dict(orient="index"),
        "field_case": R["case"],
        "within_context_variance_share": within_context_share(),
    }
    (OUT / "numbers.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def within_context_share() -> dict:
    """Share of the variance of cumulative training oil that lies between schedules of one state."""
    from actrn_experiments import REPEATS, benchmark_data

    shares = []
    for r in REPEATS:
        _, _, _, y, context_ids, split, _, _ = benchmark_data(r, "base", "exp")
        value = y[split["train"]].sum(axis=1)
        _, inv = np.unique(context_ids[split["train"]], return_inverse=True)
        means = np.bincount(inv, weights=value) / np.bincount(inv)
        shares.append(float(np.mean((value - means[inv]) ** 2) / np.var(value)))
    return {"mean": float(np.mean(shares)), "min": float(min(shares)), "max": float(max(shares))}


# ----------------------------------------------------------------- supplementary tables
def _fmt_params(p) -> str:
    if p == "average of all configurations":
        return "average of the eight configurations"
    p = json.loads(p)
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


def _longtable(spec, caption, label, header, body):
    rows = [f"\\begin{{longtable}}{{{spec}}}", f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
            "\\toprule", header + " \\\\", "\\midrule", "\\endhead"] + body
    if rows[-1] == "\\midrule":
        rows = rows[:-1]
    return "\n".join(rows + ["\\bottomrule", "\\end{longtable}"]) + "\n"


def supplementary(R) -> None:
    from actrn_experiments import GRIDS

    def axes(grid):
        alias = {"latent_dim": "latent", "learning_rate": "lr", "l2": "$\\ell_2$",
                 "alpha": "$\\alpha$", "max_depth": "max depth", "min_samples_leaf": "min leaf",
                 "max_leaf_nodes": "max leaf nodes", "hidden_layer_sizes": "hidden",
                 "learning_rate_init": "lr"}

        def show(v):
            if v is None:
                return "none"
            if isinstance(v, (list, tuple)):
                return "(" + ", ".join(str(x) for x in v) + ")"
            return f"{v:g}" if isinstance(v, float) else str(v)

        return "; ".join(f"{alias[k]} " + "\\{" + ", ".join(
            show(v) for v in sorted({g[k] for g in grid}, key=lambda v: (v is None, v))) + "\\}"
            for k in grid[0])

    body = [f"{m} & {axes(GRIDS['benchmark'][m])} & {axes(GRIDS['field'][m])} \\\\"
            for m in ("ACTRN", "Ridge", "Random forest", "HGB", "MLP")]
    text = _longtable("p{0.16\\linewidth}p{0.38\\linewidth}p{0.38\\linewidth}",
                      "Tuning grids. Each model family chooses among its eight configurations and "
                      "their averaged prediction, by validation regret on the benchmark and by "
                      "validation NMAE in the field case.", "tab:s-grids",
                      "Model & Benchmark & Field", body)
    text += ("\\noindent Fixed settings: ACTRN ensemble of five initialisation seeds; benchmark 900 "
             "epochs, patience 80, mini-batches of 21 states, ridge penalties $\\{0.01, 0.1, 1, 10\\}$ "
             "for $W_A$ (within-state validation loss) and $\\{0.1, 0.3, 1, 3, 10, 30, 100, 300, "
             "1000\\}$ for $W_B$ (validation NMAE); field 1200 epochs, patience 100, mini-batches of "
             "128 windows and joint ridge penalties $\\{1, 3, 10, 30, 100, 300, 1000, 3000\\}$; random "
             "forest 240 trees with maximum feature fraction 0.8; HGB 180 iterations with $\\ell_2$ "
             "regularisation 1.0; MLP at most 700 iterations with early stopping on 15\\% of the "
             "training data and patience 35.\n")
    (OUT / "supp_grids.tex").write_text(text)

    runs = R["runs"]
    out = []
    for st, title in (("base", "base population"), ("channel", "channelised population"),
                      ("hill", "alternative saturation response")):
        p, r = SETTINGS[st]
        sub = runs[(runs.population == p) & (runs.response == r) & runs.family.isin(FAMILIES)]
        piv = sub.pivot(index="repeat", columns="family", values="decision_regret_pct")[FAMILIES]
        body = [f"{int(rep)} & " + " & ".join(f"{v:.2f}" for v in row) + " \\\\"
                for rep, row in piv.iterrows()]
        out.append(_longtable("r" + "r" * len(FAMILIES),
                              f"Relative regret (\\%) per registered repeat index $r$, {title}.",
                              f"tab:s-repeats-{st}",
                              "Repeat & " + " & ".join(SHORT[f] for f in FAMILIES), body))
    (OUT / "supp_repeats.tex").write_text("\n".join(out))

    s, n = R["summary"], R["nmae_sel"]
    body = []
    for st in ("base", "channel", "hill"):
        for f in FAMILIES:
            key = SETTINGS[st] + (f,)
            label = {"base": "base", "channel": "channelised", "hill": "base, Hill"}[st]
            body.append(f"{label if f == FAMILIES[0] else ''} & {SHORT[f]} & "
                        f"{s.loc[key, 'decision_regret_pct_mean']:.2f} & "
                        f"{n.loc[key, 'decision_regret_pct_mean']:.2f} & "
                        f"{s.loc[key, 'nmae_pct_mean']:.2f} & {n.loc[key, 'nmae_pct_mean']:.2f} \\\\")
        body.append("\\midrule")
    (OUT / "supp_selection.tex").write_text(_longtable(
        "llrrrr", "Mean regret and NMAE when every family selects its candidate by validation "
        "regret (registered rule) or by validation NMAE.", "tab:s-selection",
        "& & \\multicolumn{2}{c}{Regret (\\%)} & \\multicolumn{2}{c}{NMAE (\\%)} \\\\ "
        "Setting & Model & regret rule & NMAE rule & regret rule & NMAE rule", body))

    hill = [" & ".join([NAME[f]] + [ms(s.loc[SETTINGS["hill"] + (f,), f"{c}_mean"],
                                       s.loc[SETTINGS["hill"] + (f,), f"{c}_std"], d)
                                    for c, d in (("nmae_pct", 2), ("decision_regret_pct", 2),
                                                 ("gain_capture_pct", 1))]) + " \\\\"
            for f in FAMILIES]
    (OUT / "supp_hill.tex").write_text(_longtable(
        "lrrr", "Base population with the alternative saturation response $a/(a+0.55)$ "
        "(mean $\\pm$ standard deviation over 20 repeats).", "tab:s-hill",
        "Model & NMAE (\\%) & Regret (\\%) & Gain capture (\\%)", hill))

    curve = R["curve"]
    agg = curve.groupby(["population", "family", "train_subset"]).agg(
        regret=("decision_regret_pct", "mean"), sd=("decision_regret_pct", "std"),
        nmae=("nmae_pct", "mean"), rows=("train_rows", "first")).reset_index()
    body = []
    for pop in ("base", "channelised"):
        for f in ["ACTRN", "ACTRN linear response", "Ridge (engineered)", "HGB (engineered)"]:
            for sub in ("32x18x6", "16x18x6", "32x18x3", "8x18x6"):
                r = agg[(agg.population == pop) & (agg.family == f) & (agg.train_subset == sub)]
                if r.empty:
                    continue
                r = r.iloc[0]
                body.append(f"{pop if (f == 'ACTRN' and sub == '32x18x6') else ''} & "
                            f"{LC_NAME[f] if sub == '32x18x6' else ''} & {SUBSET_LABEL[sub]} & "
                            f"{int(r.rows)} & {ms(r.regret, r.sd, 2)} & {r.nmae:.2f} \\\\")
        body.append("\\midrule")
    (OUT / "supp_learning.tex").write_text(_longtable(
        "llp{0.27\\linewidth}rrr", "Decision regret with reduced training data (mean $\\pm$ "
        "standard deviation over 20 repeats). Subsets are nested in the full training set; "
        "validation and test data are unchanged.", "tab:s-learning",
        "Population & Model & Training data & Examples & Regret (\\%) & NMAE (\\%)", body))

    field = R["field"]
    body = []
    for f in ["Persistence"] + FAMILIES:
        for r in field[field.family == f].sort_values("fold").itertuples():
            cfg = _fmt_params(r.params) if r.params != "{}" else "--"
            body.append(f"{NAME[f]} & {int(r.fold) + 1} & {r.nmae_pct:.1f} & {r.nrmse_pct:.1f} & "
                        f"${r.r2:.2f}$ & {cfg} \\\\")
        body.append("\\midrule")
    (OUT / "supp_field.tex").write_text(_longtable(
        "p{0.24\\linewidth}rrrrp{0.29\\linewidth}",
        "Field candidates selected per origin and test metrics.", "tab:s-field",
        "Model & Origin & NMAE (\\%) & NRMSE (\\%) & $R^2$ & Selected candidate", body))


def supplementary_field_blocks() -> None:
    from actrn_experiments import FIELD_FOLD_ENDS, FIELD_TEST, field_blocks
    from action_conditioned_decision import build_field_case_dataset

    _, _, _, dates = build_field_case_dataset()
    d = lambda i: str(dates[i].date())  # noqa: E731
    rows = ["\\begin{tabular}{lllll}", "\\toprule",
            "Origin & Training decision dates & Validation decision dates & Test decision dates "
            "& Last training target week \\\\", "\\midrule"]
    for k, end in enumerate(FIELD_FOLD_ENDS):
        start = end - FIELD_TEST
        b = field_blocks(start)
        tr, va = b["train"], b["validation"]
        last_target = str((dates[tr[-1]] + pd.Timedelta(weeks=8)).date())
        rows.append(f"{k + 1} & {d(tr[0])} -- {d(tr[-1])} ({len(tr)}) & {d(va[0])} -- "
                    f"{d(va[-1])} (30) & {d(start)} -- {d(end - 1)} (60) & {last_target} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "supp_field_blocks.tex").write_text("\n".join(rows) + "\n")
    bt = pd.read_csv(RES / "field_case_backtest.csv")
    dec = pd.read_csv(RES / "field_case_decision.csv")
    rows = ["\\begin{tabular}{rrrrrr}", "\\toprule",
            "Week & Selected F-4 & Selected F-5 & Predicted (selected) & Predicted (implemented) "
            "& Observed (implemented) \\\\", "\\midrule"]
    for (_, b), (_, s) in zip(bt.iterrows(), dec.iterrows()):
        rows.append(f"{int(b.step)} & {s.injector_1_m3:,.0f} & {s.injector_2_m3:,.0f} & "
                    f"{s.predicted_field_oil_m3:,.0f} & {b.predicted_under_implemented_m3:,.0f} & "
                    f"{b.observed_field_oil_m3:,.0f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "supp_field_case.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    results = load()
    extra = main_tables(results)
    numbers(results, extra)
    supplementary(results)
    supplementary_field_blocks()
    print(json.dumps({k: results["case"][k] for k in ("selected_source_block", "library_size")}))
