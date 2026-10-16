"""Write the manuscript tables and an in-text number sheet from the registered v4 results.

Deep sequence baselines and the sensitivity runs come from the v4 extension (ADR-0005); the
Autoformer rows come from its corrected rerun (ADR-0006), which supersedes the Autoformer rows of
the extension. The field decision example comes from its equal-budget run (ADR-0007).

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
EXT = ROOT / "data" / "processed" / "actrn_v4_ext"
AUTOFORMER = ROOT / "data" / "processed" / "actrn_v4_ext_autoformer"
CASE = ROOT / "data" / "processed" / "actrn_v4_case"
OUT = ROOT / "paper" / "fuel_2026" / "tables"
sys.path.insert(0, str(ROOT / "code"))

RAW = ["Ridge", "Random forest", "HGB", "MLP"]
SEQ = ["RNN", "LSTM", "Transformer", "Autoformer", "Reformer"]
ENG = [f"{m} (engineered)" for m in RAW]
FAMILIES = RAW + SEQ + ENG + ["ACTRN"]
BASELINES = RAW + SEQ + ENG
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
NAME.update({m: m for m in SEQ})
SHORT = {"Ridge": "Ridge", "Random forest": "RF", "HGB": "HGB", "MLP": "MLP", "ACTRN": "ACTRN",
         "RNN": "RNN", "LSTM": "LSTM", "Transformer": "Transf.", "Autoformer": "Autof.",
         "Reformer": "Reform."}
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
        raw = r.rstrip().endswith("\\\\") or r.strip() in ("\\midrule", "\\toprule") \
            or r.startswith("\\cmidrule")
        lines.append(r if raw else r + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if notes:
        lines.append(f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\scriptsize {notes}}}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


def read_ext(name: str) -> pd.DataFrame:
    """Extension CSV with the Autoformer rows replaced by the corrected rerun (ADR-0006)."""
    ext = pd.read_csv(EXT / name)
    return pd.concat([ext[ext.family != "Autoformer"], pd.read_csv(AUTOFORMER / name)],
                     ignore_index=True)


def read_ext_stats() -> dict:
    """Paired statistics of ACTRN against the sequence models, Autoformer from ADR-0006."""
    stats = json.loads((EXT / "sequence_stats.json").read_text())
    for setting, entries in json.loads((AUTOFORMER / "sequence_stats.json").read_text()).items():
        stats[setting].update(entries)
    return stats


def load():
    idx = ["population", "response", "family"]
    both = lambda name, ext: pd.concat([pd.read_csv(RES / name), read_ext(ext)],  # noqa: E731
                                       ignore_index=True)
    stats = json.loads((RES / "benchmark_stats.json").read_text())
    for setting, entries in read_ext_stats().items():
        stats[setting].update(entries)
    field = pd.read_csv(RES / "field_rolling_metrics.csv")
    seq_field = read_ext("sequence_field_metrics.csv").merge(
        field[["fold", "test_start", "test_end"]].drop_duplicates("fold"), on="fold")
    return {
        "summary": both("benchmark_summary.csv", "sequence_summary.csv").set_index(idx),
        "nmae_sel": both("benchmark_summary_nmae_selection.csv",
                         "sequence_summary_nmae_selection.csv").set_index(idx),
        "stats": stats,
        "dependence": pd.read_csv(RES / "benchmark_state_dependence.csv"),
        "curve": pd.read_csv(RES / "benchmark_learning_curve.csv"),
        "runs": both("benchmark_runs.csv", "sequence_runs.csv"),
        "sensitivity": pd.read_csv(EXT / "sensitivity_hyperparameters.csv"),
        "field": pd.concat([field, seq_field], ignore_index=True),
        "case": json.loads((CASE / "field_case_summary.json").read_text()),
    }


def dependence_reference(dep, population, response):
    d = dep[(dep.population == population) & (dep.response == response)]
    capture = 100 * (1 - d.state_blind_regret_pct / d.random_regret_pct)
    rep = d.assign(capture=capture).groupby("repeat")[["random_regret_pct",
                                                       "state_blind_regret_pct", "capture"]].mean()
    return {"random_regret": (rep.random_regret_pct.mean(), rep.random_regret_pct.std()),
            "blind_regret": (rep.state_blind_regret_pct.mean(), rep.state_blind_regret_pct.std()),
            "blind_capture": (rep.capture.mean(), rep.capture.std())}


METRICS = [("nmae_pct", 2, "low"), ("nrmse_pct", 2, "low"), ("r2", 3, "high"),
           ("decision_regret_pct", 2, "low"), ("gain_capture_pct", 1, "high"),
           ("optimal_selection_pct", 1, "high")]


def improvement(s, setting, group):
    """ACTRN relative to the best model of ``group`` on each metric: % reduction for errors,
    percentage-point gain for scores."""
    out = {}
    for col, _, direction in METRICS:
        values = s.loc[SETTINGS[setting]].loc[group, f"{col}_mean"]
        actrn = s.loc[SETTINGS[setting] + ("ACTRN",), f"{col}_mean"]
        if direction == "low":
            ref = values.min()
            out[col] = {"best": values.idxmin(), "value": float(ref),
                        "change": float(100 * (1 - actrn / ref)), "unit": "%"}
        else:
            ref = values.max()
            out[col] = {"best": values.idxmax(), "value": float(ref),
                        "change": float(actrn - ref), "unit": "" if col == "r2" else "pt"}
    return out


def main_tables(R) -> dict:
    s = R["summary"]

    def cell(setting, fam, col, d):
        key = SETTINGS[setting] + (fam,)
        return ms(s.loc[key, f"{col}_mean"], s.loc[key, f"{col}_std"], d)

    refs = {st: dependence_reference(R["dependence"], *SETTINGS[st]) for st in ("base", "channel")}
    gains = {st: {"vs_conventional_and_deep": improvement(s, st, RAW + SEQ),
                  "vs_with_features": improvement(s, st, ENG),
                  "vs_all": improvement(s, st, BASELINES)} for st in ("base", "channel")}
    cols = [("nmae_pct", 2, "low"), ("r2", 3, "high"), ("decision_regret_pct", 2, "low"),
            ("gain_capture_pct", 1, "high")]
    rows = []
    for label, group in (("Conventional models", RAW), ("Deep sequence models", SEQ),
                         ("Conventional models with ACTRN features", ENG)):
        rows.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{label}}}}} \\\\")
        rows += [" & ".join([NAME[f]] + [cell(st, f, c, d) for st in ("base", "channel")
                                         for c, d, _ in cols]) for f in group]
        rows.append("\\midrule")
    rows.append(" & ".join(["\\textbf{ACTRN}"] + [cell(st, "ACTRN", c, d)
                                                   for st in ("base", "channel") for c, d, _ in cols]))
    rows.append("\\midrule")
    for key, label in (("vs_conventional_and_deep", "Gain over best conventional/deep model"),
                       ("vs_with_features", "Gain over best model with ACTRN features")):
        rows.append(" & ".join([f"\\emph{{{label}}}"] + [
            (f"${gains[st][key][c]['change']:.1f}\\%$" if u == "low" else
             f"${gains[st][key][c]['change']:+.3f}$" if c == "r2" else
             f"${gains[st][key][c]['change']:+.1f}$")
            for st in ("base", "channel") for c, _, u in cols]))
    rows.append("\\midrule")
    rows.append(" & ".join(["Random candidate"] + [x for st in ("base", "channel") for x in
                                                  ("--", "--", ms(*refs[st]["random_regret"], 2), "$0$")]))
    rows.append(" & ".join(["State-blind reference ranking"] + [
        x for st in ("base", "channel") for x in
        ("--", "--", ms(*refs[st]["blind_regret"], 2), ms(*refs[st]["blind_capture"], 1))]))
    (OUT / "tab_comparison.tex").write_text(table(
        rows, ["Model"] + ["NMAE (\\%)", "$R^2$", "Regret (\\%)", "Gain capture (\\%)"] * 2,
        "Prediction and selection performance in the comparative experiment (mean $\\pm$ SD, "
        "20 repeats).", "tab:comparison", "lrrrrrrrr", wide=True,
        pre=["& \\multicolumn{4}{c}{Base population} & \\multicolumn{4}{c}{Channelised population} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}"],
        notes="Gain: reduction (\\%) in NMAE and regret; difference in $R^2$; percentage points in "
              "gain capture. State-blind reference: candidates ranked by their mean true value over "
              "the other test states."))
    for stale in ("tab_comparison_base.tex", "tab_comparison_channel.tex"):
        (OUT / stale).unlink(missing_ok=True)

    abl_cols = [("nmae_pct", 2), ("decision_regret_pct", 2), ("gain_capture_pct", 1)]
    rows = [" & ".join([NAME[f]] + [cell(st, f, c, d) for st in ("base", "channel")
                                    for c, d in abl_cols]) for f in ["ACTRN"] + ABL]
    rows.insert(1, "\\midrule")
    (OUT / "tab_ablation.tex").write_text(table(
        rows, ["Model", "NMAE (\\%)", "Regret (\\%)", "Gain capture (\\%)",
               "NMAE (\\%)", "Regret (\\%)", "Gain capture (\\%)"],
        "Ablation results (mean $\\pm$ SD, 20 repeats).", "tab:ablation", "lrrrrrr", wide=True,
        notes="Each variant changes one component and reuses the candidate selected for the full "
              "model.",
        pre=["& \\multicolumn{3}{c}{Base population} & \\multicolumn{3}{c}{Channelised population} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}"]))

    field = R["field"]
    fp = field.pivot(index="family", columns="fold", values="nmae_pct")
    fsum = field.groupby("family")[["nmae_pct", "nrmse_pct", "r2"]].mean()
    rows = []
    for group in (["Persistence"], RAW, SEQ, ENG, ["ACTRN"]):
        rows += [" & ".join([NAME[f]] + [f"{fp.loc[f, k]:.1f}" for k in range(4)] +
                            [f"{fsum.loc[f, 'nmae_pct']:.1f}", f"{fsum.loc[f, 'nrmse_pct']:.1f}",
                             f"${fsum.loc[f, 'r2']:.2f}$"])
                 for f in group] + ["\\midrule"]
    dates = field.drop_duplicates("fold").sort_values("fold")
    note = "; ".join(f"origin {int(r.fold) + 1}: {r.test_start} to {r.test_end}"
                     for r in dates.itertuples())
    (OUT / "supp_field_nmae.tex").write_text("\n".join(
        ["\\begin{table}[h]", "\\centering",
         "\\caption{Field NMAE (\\%) by rolling origin (O1--O4) and means over the four origins.}",
         "\\label{tab:s-field}", "\\begin{adjustbox}{max width=\\linewidth}",
         "\\begin{tabular}{lrrrrrrr}", "\\toprule",
         "Model & O1 & O2 & O3 & O4 & Mean NMAE & Mean NRMSE & Mean $R^2$ \\\\", "\\midrule"]
        + [r if r == "\\midrule" else r + " \\\\" for r in rows[:-1]]
        + ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}",
           f"\\par\\smallskip{{\\footnotesize Test blocks of 60 weekly decision windows ({note}).}}",
           "\\end{table}"]) + "\n")
    for stale in ("tab_prediction.tex", "tab_decision.tex", "tab_field.tex"):
        (OUT / stale).unlink(missing_ok=True)
    return {"refs": refs, "field_by_fold": fp, "field_mean": fsum, "gains": gains}


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
        "gains": extra["gains"],
        "mse_gains": mse_gains(R["runs"]),
        "field_mse_vs_persistence_pct": field_mse_reduction(R["field"]),
        "sensitivity": sensitivity_summary(R["sensitivity"]),
        "fit_minutes": {st_: {f: round(v, 2) for f, v in d.items() if f in FAMILIES}
                        for st_, d in fit_minutes(R["runs"]).items()},
        "selection_rule": {st_: {"regret_rule": {f: round(float(R["summary"].loc[
            SETTINGS[st_] + (f,), "decision_regret_pct_mean"]), 4) for f in FAMILIES},
            "nmae_rule": {f: round(float(R["nmae_sel"].loc[SETTINGS[st_] + (f,),
                                                            "decision_regret_pct_mean"]), 4)
                          for f in FAMILIES}} for st_ in SETTINGS},
    }
    (OUT / "numbers.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def mse_gains(runs) -> dict:
    """Reduction of the mean squared error (mean over repeats of the test MSE, RMSE squared)
    relative to the lowest-MSE model of each baseline group."""
    out = {}
    for setting, key in SETTINGS.items():
        sub = runs[(runs.population == key[0]) & (runs.response == key[1])]
        mse = (sub.assign(mse=sub.rmse ** 2).groupby("family")["mse"].mean())
        out[setting] = {}
        for label, group in (("vs_conventional_and_deep", RAW + SEQ), ("vs_with_features", ENG)):
            best = mse.loc[group].idxmin()
            out[setting][label] = {"best": best, "reduction_pct": float(100 * (1 - mse["ACTRN"] / mse[best]))}
    return out


def fit_minutes(runs) -> dict:
    """Median over repeats of the time to fit all candidates of a family, in minutes."""
    out = {}
    for setting in ("base", "channel"):
        key = SETTINGS[setting]
        sub = runs[(runs.population == key[0]) & (runs.response == key[1])]
        out[setting] = (sub.groupby("family")["fit_seconds_all_configs"].median() / 60).to_dict()
    return out


def field_mse_reduction(field) -> float:
    """Reduction of the field MSE (mean over origins of RMSE squared) of ACTRN against persistence."""
    mse = field.assign(mse=field.rmse ** 2).groupby("family")["mse"].mean()
    return float(100 * (1 - mse["ACTRN"] / mse["Persistence"]))


SENS_DEFAULT = {"latent_dim": 16, "l2": 1e-3, "ensemble": 5, "learning_rate": 0.002}


def sensitivity_summary(sens) -> dict:
    """One-at-a-time ACTRN sensitivity: mean test regret and NMAE per value of each axis."""
    out = {}
    for axis in SENS_DEFAULT:
        sub = sens
        for k, v in SENS_DEFAULT.items():
            if k != axis:
                sub = sub[np.isclose(sub[f"cfg_{k}"], v)]
        g = sub.groupby(["population", f"cfg_{axis}"])[["decision_regret_pct", "nmae_pct"]].mean()
        out[axis] = {pop: {str(v): {m: round(float(g.loc[(pop, v), m]), 4) for m in g.columns}
                           for v in g.loc[pop].index} for pop in g.index.levels[0]}
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
def _longtable(spec, caption, label, header, body):
    rows = [f"\\begin{{longtable}}{{{spec}}}", f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
            "\\toprule", header + " \\\\", "\\midrule", "\\endfirsthead",
            "\\caption[]{(continued)}\\\\", "\\toprule", header + " \\\\", "\\midrule",
            "\\endhead"] + body
    if rows[-1] == "\\midrule":
        rows = rows[:-1]
    return "\n".join(rows + ["\\bottomrule", "\\end{longtable}"]) + "\n"


def supplementary(R) -> None:
    from actrn_experiments import GRIDS

    def axes(grid):
        alias = {"latent_dim": "latent", "learning_rate": "lr", "l2": "$\\ell_2$",
                 "alpha": "$\\alpha$", "max_depth": "max depth", "min_samples_leaf": "min leaf",
                 "max_leaf_nodes": "max leaf nodes", "hidden_layer_sizes": "hidden",
                 "learning_rate_init": "lr", "d_model": "width", "dropout": "dropout"}

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
            for m in ("ACTRN", "Ridge", "Random forest", "HGB", "MLP") + tuple(SEQ)]
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
             "training data and patience 35; deep sequence models trained by mean squared error "
             "with Adam (weight decay $10^{-4}$), mini-batches of 128, at most 300 epochs and early "
             "stopping after 30 epochs without validation improvement, two layers, four attention "
             "heads for the attention-based models, Autoformer moving-average kernel 5 and label "
             "length 5, Reformer bucket size 4 with two hashing rounds.\n")
    (OUT / "supp_grids.tex").write_text(text)

    cost = fit_minutes(R["runs"])
    body = [f"{NAME[f]} & {cost['base'][f]:.1f} & {cost['channel'][f]:.1f} \\\\" for f in FAMILIES]
    (OUT / "supp_cost.tex").write_text("\n".join(
        ["\\begin{table}[h]", "\\centering",
         "\\caption{Time to fit the nine candidates of each model family and predict the "
         "validation and test data (median minutes per repeat, one CPU thread).}\\label{tab:s-cost}",
         "\\begin{tabular}{lrr}", "\\toprule", "Model & Base population & Channelised population \\\\",
         "\\midrule"] + body + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]) + "\n")

    s, n = R["summary"], R["nmae_sel"]
    body = []
    for st in ("base", "channel"):
        for f in FAMILIES:
            key = SETTINGS[st] + (f,)
            label = {"base": "base", "channel": "channelised"}[st]
            body.append(f"{label if f == FAMILIES[0] else ''} & {SHORT[f]} & "
                        f"{s.loc[key, 'decision_regret_pct_mean']:.2f} & "
                        f"{n.loc[key, 'decision_regret_pct_mean']:.2f} & "
                        f"{s.loc[key, 'nmae_pct_mean']:.2f} & {n.loc[key, 'nmae_pct_mean']:.2f} \\\\")
        body.append("\\midrule")
    (OUT / "supp_selection.tex").write_text(_longtable(
        "llrrrr", "Mean regret and NMAE under the two tuning rules (validation regret, the default, "
        "or validation NMAE).", "tab:s-selection",
        "& & \\multicolumn{2}{c}{Regret (\\%)} & \\multicolumn{2}{c}{NMAE (\\%)} \\\\ "
        "Setting & Model & regret rule & NMAE rule & regret rule & NMAE rule", body))

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
        "llp{0.27\\linewidth}rrr", "Regret and NMAE with reduced training data (mean $\\pm$ SD, "
        "20 repeats).", "tab:s-learning",
        "Population & Model & Training data & Examples & Regret (\\%) & NMAE (\\%)", body))

    for stale in ("supp_repeats.tex", "supp_sensitivity.tex", "supp_hill.tex", "supp_field.tex"):
        (OUT / stale).unlink(missing_ok=True)


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
    (OUT / "supp_field_case.tex").unlink(missing_ok=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    results = load()
    extra = main_tables(results)
    numbers(results, extra)
    supplementary(results)
    supplementary_field_blocks()
    print(json.dumps({k: results["case"][k] for k in ("selected_source_block", "library_size")}))
