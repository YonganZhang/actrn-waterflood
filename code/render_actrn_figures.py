"""Render the ACTRN manuscript data figures (Figures 2-8) from registered v4 outputs.

Reads ``data/processed/actrn_v4`` (written by ``actrn_experiments.py``) plus the Volve
context tables, and writes PDF/PNG pairs to ``paper/fuel_2026/figures``.  Figure 3 is
the field-state figure from ``render_p17_full_figure_suite.py``; Figure 1 is the editable
PowerPoint schematic.  No fallback or placeholder data are permitted.

Usage (from the project root):
    .venv/bin/python code/render_actrn_figures.py [--results data/processed/actrn_v4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import font_manager
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIGURES = ROOT / "paper" / "fuel_2026" / "figures"
sys.path.insert(0, str(ROOT / "code"))



def _register_times_new_roman() -> None:
    """Matplotlib skips upper-case .TTF files; register fontconfig's Times New Roman faces."""
    import subprocess
    listing = subprocess.run(["fc-list", ":family=Times New Roman", "file"], capture_output=True,
                             text=True).stdout
    for line in listing.splitlines():
        path = line.split(":")[0].strip()
        if path:
            font_manager.fontManager.addfont(path)


_register_times_new_roman()

from action_conditioned_decision import allocate_budget, make_reservoir, rollout  # noqa: E402
import render_p17_full_figure_suite as context  # noqa: E402
from make_actrn_tables import read_ext  # noqa: E402

BLUE, SKY, YELLOW, CORAL = "#376795", "#72BCD5", "#FFD06F", "#E76254"
NAVY, PALE, GREY, DARK = "#1E466E", "#AADCE0", "#A9A9A9", "#333333"
ORANGE = "#EF8A47"
RAW = ["Ridge", "Random forest", "HGB", "MLP"]
SEQ = ["RNN", "LSTM", "Transformer", "Autoformer", "Reformer"]
ENG = [f"{m} (engineered)" for m in RAW]
ORDER = RAW + SEQ + ENG + ["ACTRN"]
BASELINES = RAW + SEQ + ENG
N_REPEATS = 20
EXTENSION = PROC / "actrn_v4_ext"
LABEL = {"Ridge": "Ridge", "Random forest": "RF", "HGB": "HGB", "MLP": "MLP",
         "RNN": "RNN", "LSTM": "LSTM", "Transformer": "Transf.", "Autoformer": "Autof.",
         "Reformer": "Reform.", "ACTRN": "ACTRN", "Persistence": "Persistence"}
LABEL.update({f"{m} (engineered)": f"{LABEL[m]}+F" for m in RAW})
BASE_COLOR = {"Ridge": GREY, "Random forest": SKY, "HGB": YELLOW, "MLP": ORANGE,
              "RNN": "#B39DDB", "LSTM": "#7E57C2", "Transformer": "#66BB6A",
              "Autoformer": "#2E7D32", "Reformer": "#A5D6A7",
              "ACTRN": BLUE, "Persistence": "#BDBDBD"}
COLOR = {**BASE_COLOR, **{f"{m} (engineered)": BASE_COLOR[m] for m in RAW}}


def _font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Times New Roman", "Nimbus Roman", "Liberation Serif"):
        if name in names:
            return name
    raise RuntimeError("No Times-compatible serif font is installed")


FONT = _font()
STYLE = {
    "font.family": "serif", "font.serif": [FONT], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    "axes.unicode_minus": True,
}
plt.rcParams.update(STYLE)


def panel(ax, letter, x=-0.14, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=12, fontweight="bold", color="#111111", fontfamily=FONT)


def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D7D7D7", linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)


def save(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontfamily(FONT)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def read_results(results: Path) -> dict:
    expected = {"benchmark_runs.csv": None, "benchmark_decisions.csv": None,
                "benchmark_summary.csv": None, "benchmark_state_dependence.csv": None,
                "benchmark_learning_curve.csv": None, "field_rolling_metrics.csv": 40,
                "field_case_decision.csv": 8, "field_case_backtest.csv": 8,
                "field_case_candidates.csv": None}
    frames = {}
    for name, rows in expected.items():
        frame = pd.read_csv(results / name)
        if rows is not None and len(frame) != rows:
            raise ValueError(f"{name}: expected {rows} rows, got {len(frame)}")
        frames[name.removesuffix(".csv")] = frame
    ext = {name: read_ext(f"{name}.csv") for name in
           ("sequence_runs", "sequence_decisions", "sequence_summary", "sequence_field_metrics")}
    ext["sensitivity_hyperparameters"] = pd.read_csv(EXTENSION / "sensitivity_hyperparameters.csv")
    frames["benchmark_runs"] = pd.concat([frames["benchmark_runs"], ext["sequence_runs"]],
                                         ignore_index=True)
    frames["benchmark_decisions"] = pd.concat([frames["benchmark_decisions"],
                                               ext["sequence_decisions"]], ignore_index=True)
    frames["benchmark_summary"] = pd.concat([frames["benchmark_summary"],
                                             ext["sequence_summary"]], ignore_index=True)
    persistence = frames["field_rolling_metrics"][["fold", "test_start", "test_end"]]
    seq_field = ext["sequence_field_metrics"].merge(persistence.drop_duplicates("fold"), on="fold")
    frames["field_rolling_metrics"] = pd.concat([frames["field_rolling_metrics"], seq_field],
                                                ignore_index=True)
    frames["sensitivity"] = ext["sensitivity_hyperparameters"]
    frames["selection_summary"] = pd.concat(
        [pd.read_csv(results / "benchmark_summary_nmae_selection.csv"),
         read_ext("sequence_summary_nmae_selection.csv")], ignore_index=True)
    runs = frames["benchmark_runs"]
    runs = runs[runs["response"] == "exp"]
    for population, sub in runs.groupby("population"):
        missing = set(ORDER) - set(sub["family"])
        if missing or sub.groupby("family").size().loc[ORDER].ne(N_REPEATS).any():
            raise ValueError(f"{population}: benchmark runs incomplete, missing {sorted(missing)}")
    frames["benchmark_runs"] = runs
    dec = frames["benchmark_decisions"]
    frames["benchmark_decisions"] = dec[dec["response"] == "exp"]
    summary = frames["benchmark_summary"]
    frames["benchmark_summary"] = summary[summary["response"] == "exp"]
    frames["case"] = json.loads((results / "field_case_summary.json").read_text())
    return frames


# ----------------------------------------------------------------- Figure 2
def response_surface_data():
    """Fixed-budget response surface for one state, using the benchmark's action bounds."""
    reservoir = make_reservoir(np.random.default_rng(20260716))
    pressure = np.array([0.38, 0.57, 0.44])
    water = np.array([0.12, 0.21, 0.17])
    horizon, budget = 8, 12.8
    t_grid = np.linspace(-1.0, 1.0, horizon)

    def plan(split, timing):
        w = np.exp(-timing * t_grid)
        return allocate_budget(np.column_stack([split * w, (1.0 - split) * w]), budget)

    allocation = np.linspace(0.15, 0.85, 36)
    timing = np.linspace(-1.2, 1.2, 36)
    surface = np.array([[rollout(reservoir, pressure, water, plan(s, t), age=31).sum()
                         for s in allocation] for t in timing])
    examples = [(p := plan(s, t), rollout(reservoir, pressure, water, p, age=31))
                for s, t in ((0.30, -0.8), (0.50, 0.0), (0.70, 0.8))]
    return allocation, timing, surface, examples


def figure_benchmark_design():
    allocation, timing, surface, examples = response_surface_data()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(3.4, 8.2), constrained_layout=True)
    levels = np.linspace(surface.min(), surface.max(), 15)
    cs = ax_a.contourf(100 * allocation, timing, surface, levels=levels, cmap="YlGnBu_r")
    ax_a.contour(100 * allocation, timing, surface, levels=levels[::2], colors="white",
                 linewidths=0.35, alpha=0.65)
    cb = fig.colorbar(cs, ax=ax_a, orientation="horizontal", fraction=0.09, pad=0.16)
    cb.set_label("Cumulative oil (simulator units)")
    cb.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax_a.set_xlabel("Allocation to injector 1 (%)")
    ax_a.set_ylabel("Front-loading index")
    panel(ax_a, "a")
    steps = np.arange(1, 9)
    colors = ["#D55E00", "#009E73", "#0072B2"]
    for i, ((p, _), c) in enumerate(zip(examples, colors), 1):
        ax_b.plot(steps, p[:, 0], color=c, marker="o", ms=3.5, label=f"C{i}, injector 1")
        ax_b.plot(steps, p[:, 1], color=c, ls="--", marker="s", ms=3.2, label=f"C{i}, injector 2")
    ax_b.set_ylim(0, 2.55)
    ax_b.set_xlabel("Planning step")
    ax_b.set_ylabel("Injection (simulator units)")
    ax_b.set_xticks(steps)
    ax_b.legend(ncol=2, frameon=False, loc="upper center", fontsize=7.5)
    style_axis(ax_b)
    panel(ax_b, "b")
    for i, ((_, prod), c) in enumerate(zip(examples, colors), 1):
        ax_c.plot(steps, prod, color=c, marker="o", ms=3.5, label=f"Candidate {i}")
    ax_c.set_xlabel("Planning step")
    ax_c.set_ylabel("Oil production (simulator units)")
    ax_c.set_xticks(steps)
    ax_c.legend(frameon=False)
    style_axis(ax_c)
    panel(ax_c, "c")
    save(fig, "actrn_benchmark_design")


# ----------------------------------------------------------------- Figure 4
def operating_region(fig, ax, letter):
    oof = pd.read_csv(PROC / "fieldwide_oof_predictions.csv", parse_dates=["date"])
    weekly = pd.read_parquet(PROC / "fieldwide_weekly.parquet")
    op = oof.merge(weekly[["inj_F5_m3"]].rename_axis("date").reset_index(), on="date",
                   how="left", validate="one_to_one")
    cmap = mcolors.LinearSegmentedColormap.from_list("wbr", ["#FFFFFF", BLUE, CORAL])
    hb = ax.hexbin(op["A_F4_m3"] / 1e3, op["inj_F5_m3"] / 1e3,
                   C=op["field_oil_cum_h8_m3"] / 1e3, reduce_C_function=np.mean,
                   gridsize=18, mincnt=3, cmap=cmap, linewidths=0.25, edgecolors="white")
    ax.set_xlabel(r"F-4 injection ($10^3$ m$^3$ week$^{-1}$)")
    ax.set_ylabel(r"F-5 injection ($10^3$ m$^3$ week$^{-1}$)")
    cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r"Observed 8-week field oil ($10^3$ m$^3$)")
    style_axis(ax, None)
    panel(ax, letter)


def figure_field_state():
    """Figure 3 from the field-state renderer, written into the manuscript figure folder."""
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        context.OUT, context.FIG = Path(tmp), Path(tmp) / "figures"
        context.SERIF_FONT = FONT
        plt.rcParams.update({"font.serif": [FONT]})
        produced = context.render_figure_2()
        for path in produced:
            if path.suffix in (".pdf", ".png"):
                shutil.copy(path, FIGURES / f"actrn_field_state{path.suffix}")
    plt.rcParams.update(STYLE)


def figure_field_context():
    crm = pd.read_csv(PROC / "crm_lambda.csv", index_col=0)
    corr = pd.read_csv(PROC / "network_corr.csv", index_col=0)
    wellheads = pd.read_csv(PROC / "volve_surface_wellheads.csv")
    ref = (float(wellheads["latitude_deg"].mean()), float(wellheads["longitude_deg"].mean()))
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.2, 6.6), constrained_layout=True)
        grid = fig.add_gridspec(2, 2)
        ax = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
        context._pattern_network(ax[0], crm, "a", r"CRM-P $\lambda$", signed=False,
                                 surface_reference=ref)
        context._pattern_network(ax[1], corr, "b", r"Pearson $r$", signed=True,
                                 surface_reference=ref)
        context._surface_wellhead_map(ax[2], wellheads, "c")
        (x0, x1), (y0, y1) = ax[2].get_xlim(), ax[2].get_ylim()
        ax[2].set_xlim(x0 - 1.0, x1 + 1.0)   # keep edge labels inside the map frame
        ax[2].set_ylim(y0 - 1.0, y1 + 1.0)
        operating_region(fig, ax[3], "d")
        for a in ax[:3]:
            a.set_box_aspect(1.0)
        fig.legend(handles=[
            Line2D([0], [0], marker="^", color="none", markerfacecolor=context.RED,
                   markeredgecolor="#7A211B", markersize=8, label="Injection well"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor=context.BLUE,
                   markeredgecolor="#173B61", markersize=8, label="Production well")],
            loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
        save(fig, "actrn_field_context")


# ----------------------------------------------------------------- Figures 5-6
def box_strip(ax, runs, metric, order):
    data = [runs.loc[runs["family"] == f, metric].to_numpy() for f in order]
    box = ax.boxplot(data, positions=np.arange(len(order)), widths=0.55, patch_artist=True,
                     showfliers=False, medianprops={"color": DARK, "linewidth": 1.0},
                     whiskerprops={"color": "#777", "linewidth": 0.7},
                     capprops={"color": "#777", "linewidth": 0.7})
    for patch, f in zip(box["boxes"], order):
        patch.set_facecolor(COLOR[f])
        patch.set_alpha(0.8 if f == "ACTRN" else (0.25 if f in ENG else 0.5))
        patch.set_edgecolor(COLOR[f])
        if f in ENG:
            patch.set_hatch("////")
    for i, (f, v) in enumerate(zip(order, data)):
        ax.scatter(i + np.linspace(-0.15, 0.15, len(v)), v, s=14, color=COLOR[f],
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([LABEL[f] for f in order], rotation=40, ha="right")
    style_axis(ax)


def paired(ax, runs, metric, other, ylabel):
    piv = runs.pivot(index="repeat", columns="family", values=metric)
    for r in piv.index:
        ax.plot([0, 1], [piv.loc[r, other], piv.loc[r, "ACTRN"]], color="#B8B8B8", lw=0.9)
    ax.scatter(np.zeros(len(piv)), piv[other], color=COLOR[other], edgecolor=DARK, lw=0.3, s=26,
               hatch="////" if other in ENG else None)
    ax.scatter(np.ones(len(piv)), piv["ACTRN"], color=BLUE, s=26)
    ax.scatter([0, 1], [piv[other].mean(), piv["ACTRN"].mean()], marker="D", s=46,
               color=[DARK, NAVY], zorder=4)          # diamonds = means (stated in caption)
    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([LABEL[other], "ACTRN"])
    ax.set_ylabel(ylabel)
    style_axis(ax)


def strongest(summary, metric, population="base", families=None):
    s = summary[(summary["population"] == population) & (summary["response"] == "exp")
                & summary["family"].isin(families or BASELINES)]
    return str(s.loc[s[f"{metric}_mean"].idxmin(), "family"])


def figure_prediction(runs, summary):
    runs = runs[runs["population"] == "base"]
    other = strongest(summary, "nmae_pct")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.9), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1.5, 0.55, 1.5]})
    box_strip(a, runs, "nmae_pct", ORDER)
    a.set_ylabel("NMAE (%)")
    panel(a, "a")
    paired(b, runs, "nmae_pct", other, "NMAE (%)")
    panel(b, "b", x=-0.3)
    box_strip(c, runs, "r2", ORDER)
    c.set_ylabel("$R^2$")
    panel(c, "c")
    save(fig, "actrn_prediction_performance")
    return other


def regret_means(ax, runs, dep):
    """Mean relative regret per family (log scale) with the model-free references."""
    rep = runs.groupby("family")["decision_regret_pct"]
    means, sds = rep.mean().loc[ORDER], rep.std().loc[ORDER]
    x = np.arange(len(ORDER))
    for i, f in enumerate(ORDER):
        ax.errorbar(i, means[f], yerr=sds[f], fmt="o", ms=7 if f == "ACTRN" else 5,
                    color=COLOR[f], mfc="white" if f in ENG else COLOR[f], capsize=2.5, lw=1.0,
                    zorder=3)
    per_repeat = dep.groupby("repeat")[["state_blind_regret_pct", "random_regret_pct"]].mean()
    for col, ls, label in (("random_regret_pct", ":", "random candidate"),
                           ("state_blind_regret_pct", "--", "state-blind reference")):
        level = per_repeat[col].mean()
        ax.axhline(level, color=CORAL if col.startswith("state") else GREY, lw=1.0, ls=ls)
        ax.text(-0.3, level * 1.06, label, color=CORAL if col.startswith("state") else "#777777",
                fontsize=7.5, ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.set_ylim(0.08, 9)
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[f] for f in ORDER], rotation=40, ha="right")
    ax.set_ylabel("Mean relative regret (%)")
    style_axis(ax)


def figure_decision(runs, summary, decisions, dependence):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.6, 0.75]})
    comparators = {}
    for row, (population, title) in enumerate((("base", "Base population"),
                                               ("channelised", "Channelised population"))):
        sub = runs[runs["population"] == population]
        dep = dependence[(dependence["population"] == population)
                         & (dependence["response"] == "exp")]
        regret_means(axes[row, 0], sub, dep)
        axes[row, 0].set_title(title, fontsize=9.5, loc="left", x=0.02)
        panel(axes[row, 0], "ac"[row], x=-0.1)
        other = strongest(summary, "decision_regret_pct", population)
        comparators[population] = other
        paired(axes[row, 1], sub, "decision_regret_pct", other, "Mean relative regret (%)")
        panel(axes[row, 1], "bd"[row], x=-0.38)
    save(fig, "actrn_decision_performance")
    return comparators


def figure_sensitivity(curve, sens, summary, selection):
    """Sensitivity experiment: training data, four hyperparameters and the selection rule."""
    pops = (("base", "Base", BLUE, "o"), ("channelised", "Channelised", CORAL, "s"))
    best = {p: summary[(summary.population == p) & (summary.response == "exp")
                       & summary.family.isin(BASELINES)].set_index("family")
            ["decision_regret_pct_mean"].min() for p, *_ in pops}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), constrained_layout=True)
    ax = axes.ravel()

    def reference(a):
        for p, label, color, _ in pops:
            a.axhline(best[p], color=color, lw=0.9, ls="--", alpha=0.8)
        a.set_ylim(0, None)
        style_axis(a)

    size = {"8x18x6": 864, "16x18x6": 1728, "32x18x6": 3456}
    for p, label, color, marker in pops:
        mean = curve[(curve.population == p) & (curve.family == "ACTRN")].groupby(
            "train_subset")["decision_regret_pct"].mean()
        ax[0].plot(list(size.values()), [mean[k] for k in size], color=color, marker=marker,
                   ms=4.5, lw=1.4, label=label)
        ax[0].scatter([1728], [mean["32x18x3"]], marker=marker, facecolor="white",
                      edgecolor=color, s=26, zorder=4)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks([864, 1728, 3456])
    ax[0].set_xticklabels(["864", "1728", "3456"])
    ax[0].set_xlabel("Training trajectories")
    reference(ax[0])
    axes_spec = (("latent_dim", "Latent dimension $d$", True),
                 ("l2", r"$\ell_2$ penalty $\lambda_2$", True),
                 ("ensemble", "Ensemble size", False),
                 ("learning_rate", "Learning rate", True))
    default = {"latent_dim": 16, "l2": 1e-3, "ensemble": 5, "learning_rate": 0.002}
    for a, (axis, xlabel, logx) in zip(ax[1:5], axes_spec):
        others = [k for k in default if k != axis]
        sub = sens.copy()
        for k in others:
            sub = sub[np.isclose(sub[f"cfg_{k}"], default[k])]
        for p, label, color, marker in pops:
            m = sub[sub.population == p].groupby(f"cfg_{axis}")["decision_regret_pct"].mean()
            a.plot(m.index, m.values, color=color, marker=marker, ms=4.5, lw=1.4, label=label)
        if logx:
            a.set_xscale("log", base=2 if axis == "latent_dim" else 10)
        if axis == "latent_dim":
            a.set_xticks([4, 8, 16, 32, 64])
            a.set_xticklabels(["4", "8", "16", "32", "64"])
        if axis == "ensemble":
            a.set_xticks([1, 2, 5, 10])
        if axis == "learning_rate":
            a.set_xticks([5e-4, 1e-3, 2e-3, 5e-3])
            a.set_xticklabels(["0.0005", "0.001", "0.002", "0.005"])
            a.xaxis.set_minor_formatter(mticker.NullFormatter())
        a.axvline(default[axis], color="#999999", lw=0.7, ls=":")
        a.set_xlabel(xlabel)
        reference(a)
    sel = selection.set_index(["population", "response", "family"])
    reg = summary.set_index(["population", "response", "family"])
    x = np.arange(2)
    for i, (p, label, color, _) in enumerate(pops):
        bl = [f for f in BASELINES]
        vals_regret = [reg.loc[(p, "exp", "ACTRN"), "decision_regret_pct_mean"],
                       reg.loc[(p, "exp"), "decision_regret_pct_mean"].loc[bl].min()]
        vals_nmae = [sel.loc[(p, "exp", "ACTRN"), "decision_regret_pct_mean"],
                     sel.loc[(p, "exp"), "decision_regret_pct_mean"].loc[bl].min()]
        ax[5].bar(x + (i - 0.5) * 0.38, [vals_regret[0], vals_nmae[0]], width=0.36,
                  color=color, label=f"{label}: ACTRN")
        ax[5].scatter(x + (i - 0.5) * 0.38, [vals_regret[1], vals_nmae[1]], marker="_",
                      s=260, color=DARK, lw=1.6, zorder=4,
                      label="best baseline" if i == 0 else None)
    ax[5].set_xticks(x)
    ax[5].set_xticklabels(["by validation\nregret", "by validation\nNMAE"])
    ax[5].set_xlabel("Candidate selection rule")
    ax[5].set_ylim(0, None)
    style_axis(ax[5])
    for a, letter in zip(ax, "abcdef"):
        a.set_ylabel("Mean relative regret (%)")
        panel(a, letter, x=-0.3)
    handles, labels = ax[1].get_legend_handles_labels()
    handles += [Line2D([0], [0], color="#777777", ls="--", lw=0.9),
                Line2D([0], [0], marker="_", color=DARK, ls="none", ms=12, mew=1.6)]
    labels += ["best baseline (full data, default rule)", "best baseline under the same rule"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               frameon=False, fontsize=7.5)
    save(fig, "actrn_sensitivity")


# ----------------------------------------------------------------- Figure 7
def figure_field(metrics, decision, backtest, case):
    mean = metrics.groupby("family")["nmae_pct"].mean()
    best = [mean.loc[group].idxmin() for group in (RAW, SEQ, ENG)]
    order = ["Persistence"] + best + ["ACTRN"]     # panel b shows every model
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    a, b, c, d = axes.ravel()
    for f in order:
        sub = metrics[metrics["family"] == f].sort_values("fold")
        a.plot(sub["fold"] + 1, sub["nmae_pct"], marker="o", ms=4,
               color="#555555" if f == "Persistence" else COLOR[f],
               ls="--" if f == "Persistence" else (":" if f in ENG else "-"),
               mfc="white" if f in ENG else None,
               lw=2.0 if f == "ACTRN" else 1.2, alpha=1.0 if f == "ACTRN" else 0.85,
               label=LABEL[f])
    a.set_xlabel("Rolling origin")
    a.set_ylabel("NMAE (%)")
    a.set_xticks([1, 2, 3, 4])
    a.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    style_axis(a)
    panel(a, "a", y=1.18)
    avg = metrics.groupby("family", as_index=False)["nmae_pct"].mean().sort_values("nmae_pct")
    pos = np.arange(len(avg))
    b.hlines(pos, 0, avg["nmae_pct"], color=[COLOR[f] for f in avg["family"]], lw=2.0, alpha=0.8)
    b.scatter(avg["nmae_pct"], pos, s=[44 if f == "ACTRN" else 30 for f in avg["family"]],
              color=["white" if f in ENG else COLOR[f] for f in avg["family"]],
              edgecolor=[COLOR[f] for f in avg["family"]], linewidth=1.2, zorder=3)
    b.set_yticks(pos)
    b.set_yticklabels([LABEL[f] for f in avg["family"]])
    b.set_xlabel("Mean NMAE across origins (%)")
    b.set_xlim(left=0)
    style_axis(b, "x")
    panel(b, "b", x=-0.3)
    steps = decision["step"].to_numpy()
    i1, i2 = decision["injector_1_m3"].to_numpy() / 1e3, decision["injector_2_m3"].to_numpy() / 1e3
    c.bar(steps, i1, color=BLUE, width=0.7, label="Injector F-4")
    c.bar(steps, i2, bottom=i1, color=SKY, width=0.7, label="Injector F-5")
    c.set_xlabel("Planning step (week)")
    c.set_ylabel(r"Selected injection ($10^3$ m$^3$)")
    c.set_xticks(steps)
    c.legend(frameon=False, loc="upper left", fontsize=8)
    style_axis(c)
    panel(c, "c")
    d.plot(steps, decision["predicted_field_oil_m3"] / 1e3, color=CORAL, marker="D", ms=4,
           label="Predicted, selected schedule")
    d.plot(steps, backtest["predicted_under_implemented_m3"] / 1e3, color=BLUE, ls="--",
           marker="o", ms=3.5, label="Predicted, implemented schedule")
    d.plot(steps, backtest["observed_field_oil_m3"] / 1e3, color=DARK, ls="none", marker="s",
           ms=4, label="Observed, implemented schedule")
    d.axhline(case["case_level_m3"] / 1e3, color=GREY, lw=1.0, ls=":", label="Persistence level")
    d.set_xlabel("Planning step (week)")
    d.set_ylabel(r"Field oil ($10^3$ m$^3$ week$^{-1}$)")
    d.set_xticks(steps)
    top = max(decision["predicted_field_oil_m3"].max(), backtest["predicted_under_implemented_m3"].max())
    d.set_ylim(0, 1.6 * top / 1e3)                     # head-room for the legend
    d.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    style_axis(d)
    panel(d, "d")
    save(fig, "actrn_field_decision")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(PROC / "actrn_v4"))
    args = ap.parse_args()
    r = read_results(Path(args.results))
    figure_benchmark_design()
    figure_field_state()
    figure_field_context()
    pred_other = figure_prediction(r["benchmark_runs"], r["benchmark_summary"])
    dec_other = figure_decision(r["benchmark_runs"], r["benchmark_summary"],
                                r["benchmark_decisions"], r["benchmark_state_dependence"])
    figure_sensitivity(r["benchmark_learning_curve"], r["sensitivity"], r["benchmark_summary"],
                       r["selection_summary"])
    figure_field(r["field_rolling_metrics"], r["field_case_decision"],
                 r["field_case_backtest"], r["case"])
    print(json.dumps({"figures": str(FIGURES), "prediction_comparator": pred_other,
                      "decision_comparator": dec_other}))


if __name__ == "__main__":
    main()
