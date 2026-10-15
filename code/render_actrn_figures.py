"""Render the ACTRN manuscript data figures (Figures 2 and 4-7) from registered v2 outputs.

Reads ``data/processed/actrn_v2`` (written by ``actrn_experiments.py``) plus the Volve
context tables, and writes PDF/PNG pairs to ``paper/fuel_2026/figures``.  Figure 3 is
the field-state figure from ``render_p17_full_figure_suite.py``; Figure 1 is the editable
PowerPoint schematic.  No fallback or placeholder data are permitted.

Usage (from the project root):
    .venv/bin/python code/render_actrn_figures.py [--results data/processed/actrn_v2]
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

from action_conditioned_decision import allocate_budget, make_reservoir, rollout  # noqa: E402
import render_p17_full_figure_suite as context  # noqa: E402

BLUE, SKY, YELLOW, CORAL = "#376795", "#72BCD5", "#FFD06F", "#E76254"
NAVY, PALE, GREY, DARK = "#1E466E", "#AADCE0", "#A9A9A9", "#333333"
ORANGE = "#EF8A47"
RAW = ["Ridge", "Random forest", "HGB", "MLP"]
ENG = [f"{m} (engineered)" for m in RAW]
ORDER = RAW + ENG + ["ACTRN"]
LABEL = {"Ridge": "Ridge", "Random forest": "RF", "HGB": "HGB", "MLP": "MLP",
         "ACTRN": "ACTRN", "Persistence": "Persistence"}
LABEL.update({f"{m} (engineered)": f"{LABEL[m]}+F" for m in RAW})
BASE_COLOR = {"Ridge": GREY, "Random forest": SKY, "HGB": YELLOW, "MLP": ORANGE,
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
                "benchmark_summary.csv": None, "field_rolling_metrics.csv": 24,
                "field_case_decision.csv": 8, "field_case_backtest.csv": 8,
                "field_case_candidates.csv": None}
    frames = {}
    for name, rows in expected.items():
        frame = pd.read_csv(results / name)
        if rows is not None and len(frame) != rows:
            raise ValueError(f"{name}: expected {rows} rows, got {len(frame)}")
        frames[name.removesuffix(".csv")] = frame
    runs = frames["benchmark_runs"]
    runs = runs[runs["response"] == "exp"]
    missing = set(ORDER) - set(runs["family"])
    if missing or runs.groupby("family").size().loc[ORDER].ne(5).any():
        raise ValueError(f"benchmark runs incomplete: missing {sorted(missing)}")
    frames["benchmark_runs"] = runs
    dec = frames["benchmark_decisions"]
    frames["benchmark_decisions"] = dec[dec["response"] == "exp"]
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


def strongest(summary, metric):
    s = summary[(summary["response"] == "exp") & summary["family"].isin(RAW + ENG)]
    return str(s.loc[s[f"{metric}_mean"].idxmin(), "family"])


def figure_prediction(runs, summary):
    other = strongest(summary, "nmae_pct")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1.35, 0.6, 1.35]})
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


def figure_decision(runs, summary, decisions):
    other = strongest(summary, "decision_regret_pct")
    s = summary[summary["response"] == "exp"].set_index("family").loc[ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    a, b, c, d = axes.ravel()
    data = [decisions.loc[decisions["family"] == f, "decision_regret_pct"].to_numpy()
            for f in ORDER]
    v = a.violinplot(data, positions=np.arange(len(ORDER)), showmedians=True,
                     showextrema=False, widths=0.85)
    for body, f in zip(v["bodies"], ORDER):
        body.set_facecolor(COLOR[f])
        body.set_edgecolor(COLOR[f])
        body.set_alpha(0.85 if f == "ACTRN" else (0.3 if f in ENG else 0.55))
    v["cmedians"].set_color(DARK)
    a.set_xticks(np.arange(len(ORDER)))
    a.set_xticklabels([LABEL[f] for f in ORDER], rotation=40, ha="right")
    a.set_ylabel("Relative regret (%)")
    style_axis(a)
    panel(a, "a")
    paired(b, runs, "decision_regret_pct", other, "Mean relative regret (%)")
    panel(b, "b")
    for ax, col, ylabel, letter in ((c, "optimal_selection_pct", "Exact optimum selection (%)", "c"),
                                    (d, "top_decile_selection_pct", "Top-decile selection (%)", "d")):
        for i, f in enumerate(ORDER):
            ax.errorbar(i, s.loc[f, f"{col}_mean"], yerr=s.loc[f, f"{col}_std"], fmt="o",
                        ms=6 if f == "ACTRN" else 4.5, color=COLOR[f],
                        mfc=COLOR[f] if f not in ENG else "white", capsize=2.5, lw=1.0)
        ax.set_xticks(np.arange(len(ORDER)))
        ax.set_xticklabels([LABEL[f] for f in ORDER], rotation=40, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        style_axis(ax)
        panel(ax, letter)
    save(fig, "actrn_decision_performance")
    return other


# ----------------------------------------------------------------- Figure 7
def figure_field(metrics, decision, backtest, case):
    order = ["Persistence", "Ridge", "Random forest", "HGB", "MLP", "ACTRN"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    a, b, c, d = axes.ravel()
    for f in order:
        sub = metrics[metrics["family"] == f].sort_values("fold")
        a.plot(sub["fold"] + 1, sub["nmae_pct"], marker="o", ms=4,
               color="#555555" if f == "Persistence" else COLOR[f],
               ls="--" if f == "Persistence" else "-",
               lw=2.0 if f == "ACTRN" else 1.0, alpha=1.0 if f == "ACTRN" else 0.8, label=LABEL[f])
    a.set_xlabel("Rolling origin")
    a.set_ylabel("NMAE (%)")
    a.set_xticks([1, 2, 3, 4])
    a.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    style_axis(a)
    panel(a, "a", y=1.18)
    avg = metrics.groupby("family", as_index=False)["nmae_pct"].mean().sort_values("nmae_pct")
    pos = np.arange(len(avg))
    b.hlines(pos, 0, avg["nmae_pct"], color=[COLOR[f] for f in avg["family"]], lw=2.0, alpha=0.8)
    b.scatter(avg["nmae_pct"], pos, s=[44 if f == "ACTRN" else 30 for f in avg["family"]],
              color=[COLOR[f] for f in avg["family"]], zorder=3)
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
    d.set_ylim(0, 1.55 * top / 1e3)                    # head-room for the legend
    d.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    style_axis(d)
    panel(d, "d")
    save(fig, "actrn_field_decision")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(PROC / "actrn_v2"))
    args = ap.parse_args()
    r = read_results(Path(args.results))
    figure_benchmark_design()
    figure_field_context()
    pred_other = figure_prediction(r["benchmark_runs"], r["benchmark_summary"])
    dec_other = figure_decision(r["benchmark_runs"], r["benchmark_summary"],
                                r["benchmark_decisions"])
    figure_field(r["field_rolling_metrics"], r["field_case_decision"],
                 r["field_case_backtest"], r["case"])
    print(json.dumps({"figures": str(FIGURES), "prediction_comparator": pred_other,
                      "decision_comparator": dec_other}))


if __name__ == "__main__":
    main()
