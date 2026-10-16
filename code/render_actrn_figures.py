"""Render the ACTRN manuscript data figures (Figures 2-7 and Supplementary Fig. S1).

Reads ``data/processed/actrn_v4`` (written by ``actrn_experiments.py``), the extension,
corrected-Autoformer and field-case results, the Volve context tables and the Natural Earth
coastline subset, and writes PDF/PNG pairs to ``paper/fuel_2026/figures``.  Supplementary
Fig. S1 is the field-state figure from ``render_p17_full_figure_suite.py``; Figure 1 is the
editable PowerPoint schematic.  No fallback or placeholder data are permitted.

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
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
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
CASE = PROC / "actrn_v4_case"                  # field decision example (ADR-0007)
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
        # the field decision example comes from its equal-budget run (ADR-0007)
        frame = pd.read_csv((CASE if name.startswith("field_case") else results) / name)
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
    frames["case"] = json.loads((CASE / "field_case_summary.json").read_text())
    return frames


# ----------------------------------------------------------------- Figure 2
CANDIDATE_COLORS = ["#D55E00", "#009E73", "#0072B2"]
CANDIDATE_PLANS = ((0.30, -0.8), (0.50, 0.0), (0.70, 0.8))   # (allocation, front-loading)


def response_surface_data(n=60):
    """Fixed-budget response surface for one state, using the benchmark's action bounds."""
    reservoir = make_reservoir(np.random.default_rng(20260716))
    pressure = np.array([0.38, 0.57, 0.44])
    water = np.array([0.12, 0.21, 0.17])
    horizon, budget = 8, 12.8
    t_grid = np.linspace(-1.0, 1.0, horizon)

    def plan(split, timing):
        w = np.exp(-timing * t_grid)
        return allocate_budget(np.column_stack([split * w, (1.0 - split) * w]), budget)

    allocation = np.linspace(0.15, 0.85, n)
    timing = np.linspace(-1.2, 1.2, n)
    surface = np.array([[rollout(reservoir, pressure, water, plan(s, t), age=31).sum()
                         for s in allocation] for t in timing])
    examples = [(p := plan(s, t), rollout(reservoir, pressure, water, p, age=31))
                for s, t in CANDIDATE_PLANS]
    return allocation, timing, surface, examples


def _surface_panel(fig, ax, allocation, timing, surface, cbar="right"):
    levels = np.linspace(surface.min(), surface.max(), 15)
    cs = ax.contourf(100 * allocation, timing, surface, levels=levels, cmap="YlGnBu_r",
                     extend="both")
    ax.contour(100 * allocation, timing, surface, levels=levels[::2], colors="white",
               linewidths=0.35, alpha=0.65)
    for i, ((split, front), color) in enumerate(zip(CANDIDATE_PLANS, CANDIDATE_COLORS), 1):
        ax.scatter(100 * split, front, s=34, marker="o", color=color, edgecolor="white",
                   linewidth=0.9, zorder=4)
        ax.annotate(f"C{i}", (100 * split, front), xytext=(5, 4), textcoords="offset points",
                    color=DARK, fontsize=8, fontweight="bold")
    if cbar == "right":
        cb = fig.colorbar(cs, ax=ax, fraction=0.06, pad=0.03)
    else:
        cb = fig.colorbar(cs, ax=ax, orientation="horizontal", fraction=0.09, pad=0.16)
    cb.set_label("Cumulative oil (simulator units)")
    cb.ax.tick_params(labelsize=7.5)
    cb.formatter = mticker.FormatStrFormatter("%.1f")
    cb.update_ticks()
    ax.set_xlabel("Allocation to injector 1 (%)")
    ax.set_ylabel("Front-loading index")


def _schedule_panel(ax, examples):
    steps = np.arange(1, 9)
    for (p, _), c in zip(examples, CANDIDATE_COLORS):
        ax.plot(steps, p[:, 0], color=c, marker="o", ms=3.4)
        ax.plot(steps, p[:, 1], color=c, ls="--", marker="s", ms=3.0)
    ax.set_ylim(0, 2.45)
    ax.set_xlabel("Planning step")
    ax.set_ylabel("Injection (simulator units)")
    ax.set_xticks(steps)
    ax.legend([Line2D([0], [0], color=DARK, marker="o", ms=3.4),
               Line2D([0], [0], color=DARK, ls="--", marker="s", ms=3.0)],
              ["Injector 1", "Injector 2"], frameon=False, loc="upper left",
              bbox_to_anchor=(0.02, 1.0), ncol=2, handlelength=2.0, columnspacing=1.0)
    style_axis(ax)


def _response_panel(ax, examples):
    steps = np.arange(1, 9)
    prod = np.array([p for _, p in examples])
    ax.fill_between(steps, prod.min(axis=0), prod.max(axis=0), color=PALE, alpha=0.35, lw=0)
    for i, (row, c) in enumerate(zip(prod, CANDIDATE_COLORS), 1):
        ax.plot(steps, row, color=c, marker="o", ms=3.4, label=f"Candidate C{i}")
    ax.set_xlabel("Planning step")
    ax.set_ylabel("Oil production (simulator units)")
    ax.set_xticks(steps)
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)


def figure_benchmark_design():
    allocation, timing, surface, examples = response_surface_data()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1.25, 1, 1]})
    _surface_panel(fig, a, allocation, timing, surface)
    _schedule_panel(b, examples)
    _response_panel(c, examples)
    for ax, letter in zip((a, b, c), "abc"):
        panel(ax, letter, x=-0.24)
    save(fig, "actrn_benchmark_design")


# ----------------------------------------------------------------- Figure 3
def operating_region(fig, ax, letter):
    oof = pd.read_csv(PROC / "fieldwide_oof_predictions.csv", parse_dates=["date"])
    weekly = pd.read_parquet(PROC / "fieldwide_weekly.parquet")
    op = oof.merge(weekly[["inj_F5_m3"]].rename_axis("date").reset_index(), on="date",
                   how="left", validate="one_to_one")
    cmap = mcolors.LinearSegmentedColormap.from_list("ukiyoe", context.UKIYOE[::-1])
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
    """Supplementary Fig. S1 from the field-state renderer, written into the figure folder."""
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


def _north_sea_locator(ax, lat0, lon0):
    """Regional inset: Natural Earth 1:50m coastlines with the Volve field location."""
    countries = json.loads((PROC / "naturalearth_north_sea_countries.geojson").read_text())
    for feature in countries["features"]:
        for polygon in feature["geometry"]["coordinates"]:
            ring = np.asarray(polygon[0])
            ax.fill(ring[:, 0], ring[:, 1], facecolor="#E3E3E3", edgecolor="#9A9A9A", lw=0.35)
    ax.scatter([lon0], [lat0], marker="*", s=80, color=CORAL, edgecolor=DARK, lw=0.4, zorder=5)
    ax.text(lon0, lat0 + 0.55, "Volve", fontsize=7, ha="center", va="bottom", zorder=6,
            fontweight="bold")
    for name, lon, lat in (("Norway", 8.6, 61.6), ("North Sea", 3.4, 56.2)):
        ax.text(lon, lat, name, fontsize=6.2, color="#4D4D4D", ha="center", va="center",
                style="italic")
    ax.set_xlim(-4.0, 12.0)
    ax.set_ylim(53.5, 62.5)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(58.0)))
    ax.set_facecolor("#EAF4F8")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#7A8A90")
        spine.set_linewidth(0.6)


def wellhead_lonlat_map(ax, wellheads, letter, icons=True, locator=True):
    """Official ED50 surface-wellhead positions on longitude/latitude axes (Fig. 3c)."""
    coords = wellheads.set_index("well").loc[context.WELLS]
    lon = coords["longitude_deg"].astype(float)
    lat = coords["latitude_deg"].astype(float)
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError("Surface-wellhead registry contains non-finite coordinates")
    if set(coords["geodetic_datum"]) != {"ED50"}:
        raise ValueError("Surface-wellhead map is locked to ED50 coordinates")
    lat0, lon0 = float(lat.mean()), float(lon.mean())
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    ax.set_facecolor("#F3FAFC")
    ax.set_axisbelow(True)
    ax.grid(color="#C7DCE4", linewidth=0.6, linestyle=(0, (2, 3)))
    for name in context.WELLS:
        x, y = float(lon.loc[name]), float(lat.loc[name])
        injector = name in context.WELLS[:2]
        if icons:
            icon = context.INJECTION_ICON if injector else context.PRODUCTION_ICON
            ax.add_artist(AnnotationBbox(OffsetImage(icon, zoom=0.024), (x, y), frameon=False,
                                         pad=0, zorder=6))
        else:
            ax.scatter(x, y, s=150, color=CORAL if injector else ORANGE,
                       edgecolor=NAVY if injector else "#7A3B1E", lw=0.9, zorder=6)
    offsets = {"15/9-F-4": (-24, -14), "15/9-F-5_inj": (8, -20), "15/9-F-1 C": (-26, 8),
               "15/9-F-11": (-6, 17), "15/9-F-12": (24, -9), "15/9-F-14": (22, 7),
               "15/9-F-15 D": (22, -14)}
    for name in context.WELLS:
        ax.annotate(context.SHORT[name], xy=(float(lon.loc[name]), float(lat.loc[name])),
                    xytext=offsets[name], textcoords="offset points", ha="center", va="center",
                    fontweight="bold", fontsize=8,
                    bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.86},
                    arrowprops={"arrowstyle": "-", "color": "#6E7D82", "lw": 0.6}, zorder=7)
    # equal metres on both axes; the extra room to the east holds the regional inset
    x_min, x_max = float(lon.min()) - 3.0 / m_per_deg_lon, float(lon.max()) + 6.5 / m_per_deg_lon
    y_min, y_max = float(lat.min()) - 3.0 / m_per_deg_lat, float(lat.max()) + 4.0 / m_per_deg_lat
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect(m_per_deg_lat / m_per_deg_lon)            # metres equal in both directions
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.5f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.5f"))
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel(r"Longitude ($^\circ$E, ED50)")
    ax.set_ylabel(r"Latitude ($^\circ$N, ED50)")
    metres = 5.0
    sx = x_min + 0.06 * (x_max - x_min)
    sy = y_min + 0.07 * (y_max - y_min)
    tick = 0.012 * (y_max - y_min)
    ax.plot([sx, sx + metres / m_per_deg_lon], [sy, sy], color=NAVY, lw=2.0)
    for xx in (sx, sx + metres / m_per_deg_lon):
        ax.plot([xx, xx], [sy - tick, sy + tick], color=NAVY, lw=1.0)
    ax.text(sx + 0.5 * metres / m_per_deg_lon, sy + 2.2 * tick, "5 m", ha="center", va="bottom",
            fontsize=8)
    ax.annotate("N", xy=(0.06, 0.95), xytext=(0.06, 0.80), xycoords=ax.transAxes,
                ha="center", va="center", color=NAVY, fontweight="bold", fontsize=8.5,
                arrowprops={"arrowstyle": "-|>", "color": NAVY, "lw": 1.0}, zorder=9)
    if locator:
        _north_sea_locator(ax.inset_axes([0.58, 0.58, 0.41, 0.41]), lat0, lon0)
    panel(ax, letter)


def figure_field_context(icons=True, stem="actrn_field_context"):
    crm = pd.read_csv(PROC / "crm_lambda.csv", index_col=0)
    corr = pd.read_csv(PROC / "network_corr.csv", index_col=0)
    wellheads = pd.read_csv(PROC / "volve_surface_wellheads.csv")
    ref = (float(wellheads["latitude_deg"].mean()), float(wellheads["longitude_deg"].mean()))
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
        grid = fig.add_gridspec(2, 2)
        ax = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
        context._pattern_network(ax[0], crm, "a", r"CRM-P $\lambda$", signed=False,
                                 surface_reference=ref)
        context._pattern_network(ax[1], corr, "b", r"Pearson $r$", signed=True,
                                 surface_reference=ref)
        wellhead_lonlat_map(ax[2], wellheads, "c", icons=icons)
        operating_region(fig, ax[3], "d")
        for a in ax[:2]:
            a.set_box_aspect(1.0)
        handles = [Line2D([0], [0], marker="^", color="none", markerfacecolor=context.RED,
                          markeredgecolor="#7A211B", markersize=8, label="Injection well"),
                   Line2D([0], [0], marker="^", color="none", markerfacecolor=context.BLUE,
                          markeredgecolor="#173B61", markersize=8, label="Production well")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                   frameon=False)
        save(fig, stem)


# ----------------------------------------------------------------- Figures 4-5
GROUPS = (("ACTRN", ["ACTRN"]), ("With ACTRN features", ENG), ("Deep sequence", SEQ),
          ("Conventional", RAW))
LONG = {"Ridge": "Ridge", "Random forest": "Random forest", "HGB": "HGB", "MLP": "MLP",
        "RNN": "RNN", "LSTM": "LSTM", "Transformer": "Transformer", "Autoformer": "Autoformer",
        "Reformer": "Reformer", "ACTRN": "ACTRN"}
LONG.update({f"{m} (engineered)": f"{m} + F" for m in RAW})


def model_rows():
    """Row position of each model (ACTRN on top) and the span of each model group."""
    pos, spans, y = {}, [], 0.0
    for name, families in GROUPS:
        start = y
        for f in families:
            pos[f] = y
            y += 1.0
        spans.append((name, start, y - 1.0))
        y += 1.0
    return pos, spans


def group_bands(ax, spans, label=True):
    for k, (name, start, end) in enumerate(spans):
        if k % 2 == 1:
            ax.axhspan(start - 1.0, end + 0.5, color="#F1F3F5", zorder=0, lw=0)
        if label and name != "ACTRN":
            ax.text(0.99, start - 0.78, name, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=7, style="italic", color="#6B6B6B")


def _model_axis(ax, pos, labels=True):
    ax.set_yticks(list(pos.values()))
    ax.set_yticklabels([LONG[f] for f in pos] if labels else [])
    if labels:
        for tick, f in zip(ax.get_yticklabels(), pos):
            if f == "ACTRN":
                tick.set_fontweight("bold")
    ax.set_ylim(max(pos.values()) + 0.7, min(pos.values()) - 0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#D7D7D7", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)


def hbox_strip(ax, runs, metric, pos, spans, labels=True):
    families = list(pos)
    data = [runs.loc[runs["family"] == f, metric].to_numpy() for f in families]
    group_bands(ax, spans, label=labels)
    box = ax.boxplot(data, positions=[pos[f] for f in families], vert=False, widths=0.62,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": DARK, "linewidth": 1.0},
                     whiskerprops={"color": "#777777", "linewidth": 0.7},
                     capprops={"color": "#777777", "linewidth": 0.7})
    for patch, f in zip(box["boxes"], families):
        patch.set_facecolor(COLOR[f])
        patch.set_alpha(0.85 if f == "ACTRN" else (0.3 if f in ENG else 0.5))
        patch.set_edgecolor(COLOR[f])
        if f in ENG:
            patch.set_hatch("////")
    for f, v in zip(families, data):
        jitter = np.linspace(-0.2, 0.2, len(v))
        ax.scatter(v, pos[f] + jitter, s=9, color=COLOR[f], edgecolor="white", linewidth=0.3,
                   zorder=3)
    _model_axis(ax, pos, labels)


def paired(ax, runs, metric, other, xlabel_unused=None, ylabel=None):
    piv = runs.pivot(index="repeat", columns="family", values=metric)
    for r in piv.index:
        ax.plot([0, 1], [piv.loc[r, other], piv.loc[r, "ACTRN"]], color="#B8B8B8", lw=0.8)
    ax.scatter(np.zeros(len(piv)), piv[other], color=COLOR[other], edgecolor=DARK, lw=0.3, s=20,
               hatch="////" if other in ENG else None)
    ax.scatter(np.ones(len(piv)), piv["ACTRN"], color=BLUE, s=20)
    ax.scatter([0, 1], [piv[other].mean(), piv["ACTRN"].mean()], marker="D", s=40,
               color=[DARK, NAVY], zorder=4)          # diamonds = means (stated in caption)
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([LONG[other], "ACTRN"])
    if ylabel:
        ax.set_ylabel(ylabel)
    style_axis(ax)


def strongest(summary, metric, population="base", families=None):
    s = summary[(summary["population"] == population) & (summary["response"] == "exp")
                & summary["family"].isin(families or BASELINES)]
    return str(s.loc[s[f"{metric}_mean"].idxmin(), "family"])


def figure_prediction(runs, summary):
    pos, spans = model_rows()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.9), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.45, 1.0, 0.72]})
    comparators = {}
    for row, population in enumerate(("base", "channelised")):
        sub = runs[runs["population"] == population]
        hbox_strip(axes[row, 0], sub, "nmae_pct", pos, spans)
        axes[row, 0].set_xlabel("NMAE (%)")
        hbox_strip(axes[row, 1], sub, "r2", pos, spans, labels=False)
        axes[row, 1].set_xlabel("$R^2$")
        other = strongest(summary, "nmae_pct", population)
        comparators[population] = other
        paired(axes[row, 2], sub, "nmae_pct", other, ylabel="NMAE (%)")
        for col, letter in enumerate("abc" if row == 0 else "def"):
            panel(axes[row, col], letter, x=-0.42 if col == 0 else -0.2)
    save(fig, "actrn_prediction_performance")
    return comparators


def _references(dependence, population):
    dep = dependence[(dependence["population"] == population) & (dependence["response"] == "exp")]
    per_repeat = dep.groupby("repeat")[["state_blind_regret_pct", "random_regret_pct"]].mean()
    return per_repeat.mean()


def hviolin(ax, decisions, pos, spans, refs, labels=True):
    group_bands(ax, spans, label=labels)
    for f in pos:
        v = decisions.loc[decisions["family"] == f, "decision_regret_pct"].to_numpy()
        parts = ax.violinplot(v, positions=[pos[f]], vert=False, widths=0.82, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(COLOR[f])
            body.set_edgecolor(COLOR[f])
            body.set_alpha(0.85 if f == "ACTRN" else (0.3 if f in ENG else 0.5))
            if f in ENG:
                body.set_hatch("////")
        ax.scatter([v.mean()], [pos[f]], marker="D", s=16, color=DARK, zorder=4)
    for key, ls, color, text in (("state_blind_regret_pct", "--", CORAL, "state-blind"),
                                 ("random_regret_pct", ":", "#7A7A7A", "random")):
        ax.axvline(refs[key], color=color, lw=0.9, ls=ls, zorder=1)
        ax.text(refs[key], 1.005, text, transform=ax.get_xaxis_transform(), color=color,
                fontsize=7, ha="center", va="bottom")
    ax.set_xscale("function", functions=(lambda v: np.sqrt(np.clip(v, 0, None)), np.square))
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0", "1", "2", "5", "10", "20"])
    _model_axis(ax, pos, labels)


def selection_rates(ax, runs, pos, spans):
    group_bands(ax, spans, label=False)
    for f in pos:
        sub = runs[runs["family"] == f]
        for metric, marker, face in (("optimal_selection_pct", "o", COLOR[f]),
                                     ("top_decile_selection_pct", "s", "white")):
            ax.errorbar(sub[metric].mean(), pos[f] + (-0.17 if marker == "o" else 0.17),
                        xerr=sub[metric].std(), fmt=marker, ms=4.2, color=COLOR[f], mfc=face,
                        mec=COLOR[f], lw=0.8, capsize=1.6, zorder=3)
    ax.set_xlim(0, 105)
    _model_axis(ax, pos, labels=False)


def figure_decision(runs, summary, decisions, dependence):
    pos, spans = model_rows()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.1), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.45, 1.0, 0.72]})
    comparators = {}
    for row, population in enumerate(("base", "channelised")):
        sub = runs[runs["population"] == population]
        dec = decisions[decisions["population"] == population]
        hviolin(axes[row, 0], dec, pos, spans, _references(dependence, population))
        axes[row, 0].set_xlabel("Relative regret per decision (%)")
        selection_rates(axes[row, 1], sub, pos, spans)
        axes[row, 1].set_xlabel("Selection rate (%)")
        other = strongest(summary, "decision_regret_pct", population)
        comparators[population] = other
        paired(axes[row, 2], sub, "decision_regret_pct", other, ylabel="Mean relative regret (%)")
        for col, letter in enumerate("abc" if row == 0 else "def"):
            panel(axes[row, col], letter, x=-0.42 if col == 0 else -0.2)
    axes[0, 1].legend([Line2D([0], [0], marker="o", color=DARK, ls="none", ms=4.2),
                       Line2D([0], [0], marker="s", color=DARK, mfc="white", ls="none", ms=4.2)],
                      ["exact optimum", "top decile"], loc="lower center",
                      bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.5)
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
        for p, _, color, _ in pops:
            a.axhline(best[p], color=color, lw=0.9, ls="--", alpha=0.85, zorder=1)
        a.set_ylim(0, 0.92)
        style_axis(a)

    size = {"8x18x6": 864, "16x18x6": 1728, "32x18x6": 3456}
    for p, _, color, marker in pops:
        mean = curve[(curve.population == p) & (curve.family == "ACTRN")].groupby(
            "train_subset")["decision_regret_pct"].mean()
        ax[0].plot(list(size.values()), [mean[k] for k in size], color=color, marker=marker,
                   ms=4.5, lw=1.4)
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
        sub = sens.copy()
        for k in (k for k in default if k != axis):
            sub = sub[np.isclose(sub[f"cfg_{k}"], default[k])]
        for p, _, color, marker in pops:
            m = sub[sub.population == p].groupby(f"cfg_{axis}")["decision_regret_pct"].mean()
            a.plot(m.index, m.values, color=color, marker=marker, ms=4.5, lw=1.4)
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
    for i, (p, _, color, _) in enumerate(pops):
        actrn = [reg.loc[(p, "exp", "ACTRN"), "decision_regret_pct_mean"],
                 sel.loc[(p, "exp", "ACTRN"), "decision_regret_pct_mean"]]
        base = [reg.loc[(p, "exp"), "decision_regret_pct_mean"].loc[BASELINES].min(),
                sel.loc[(p, "exp"), "decision_regret_pct_mean"].loc[BASELINES].min()]
        offset = (i - 0.5) * 0.42
        ax[5].bar(x + offset - 0.1, actrn, width=0.2, color=color)
        ax[5].bar(x + offset + 0.1, base, width=0.2, color="white", edgecolor=color, hatch="////",
                  lw=0.8)
    ax[5].set_xticks(x)
    ax[5].set_xticklabels(["Validation\nregret", "Validation\nNMAE"])
    ax[5].set_xlabel("Selection rule")
    ax[5].set_ylim(0, 0.92)
    style_axis(ax[5])
    for a, letter in zip(ax, "abcdef"):
        a.set_ylabel("Mean relative regret (%)")
        panel(a, letter, x=-0.3)
    handles = [Line2D([0], [0], color=BLUE, marker="o", ms=4.5, lw=1.4),
               Line2D([0], [0], color=CORAL, marker="s", ms=4.5, lw=1.4),
               Line2D([0], [0], color=BLUE, ls="--", lw=0.9),
               Line2D([0], [0], color=CORAL, ls="--", lw=0.9),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=DARK, hatch="////", lw=0.8)]
    labels = ["ACTRN, base", "ACTRN, channelised", "Best baseline, base",
              "Best baseline, channelised", "Best baseline (f)"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5,
               frameon=False, fontsize=7.5)
    save(fig, "actrn_sensitivity")


# ----------------------------------------------------------------- Figure 7
def figure_field(metrics, decision, backtest, case):
    mean = metrics.groupby("family")["nmae_pct"].mean()
    best = [mean.loc[group].idxmin() for group in (RAW, SEQ, ENG)]
    order = ["Persistence"] + best + ["ACTRN"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    a, b, c, d = axes.ravel()
    for f in order:
        sub = metrics[metrics["family"] == f].sort_values("fold")
        a.plot(sub["fold"] + 1, sub["nmae_pct"], marker="o", ms=4,
               color="#555555" if f == "Persistence" else COLOR[f],
               ls="--" if f == "Persistence" else (":" if f in ENG else "-"),
               mfc="white" if f in ENG else None,
               lw=2.0 if f == "ACTRN" else 1.2, alpha=1.0 if f == "ACTRN" else 0.85,
               label=LONG.get(f, f))
    a.set_xlabel("Rolling origin")
    a.set_ylabel("NMAE (%)")
    a.set_xticks([1, 2, 3, 4])
    a.legend(ncol=2, frameon=False, loc="upper right", fontsize=7.5)
    style_axis(a)
    panel(a, "a")
    avg = metrics.groupby("family", as_index=False)["nmae_pct"].mean().sort_values(
        "nmae_pct", ascending=False)
    pos = np.arange(len(avg))
    cap = 60.0
    for y, (f, v) in zip(pos, zip(avg["family"], avg["nmae_pct"])):
        shown = min(v, cap)
        b.hlines(y, 0, shown, color=COLOR[f], lw=2.0, alpha=0.8)
        b.scatter(shown, y, s=44 if f == "ACTRN" else 28, color="white" if f in ENG else COLOR[f],
                  edgecolor=COLOR[f], lw=1.2, zorder=3, marker=">" if v > cap else "o")
        if v > cap:
            b.text(cap - 1.0, y + 0.42, f"{v:.0f}", ha="right", va="center", fontsize=7)
    persistence = mean.loc["Persistence"]
    b.axvline(persistence, color="#555555", lw=0.8, ls="--", zorder=1)
    b.set_yticks(pos)
    b.set_yticklabels([LONG.get(f, f) for f in avg["family"]], fontsize=7.5)
    for tick, f in zip(b.get_yticklabels(), avg["family"]):
        if f == "ACTRN":
            tick.set_fontweight("bold")
    b.set_xlabel("Mean NMAE across origins (%)")
    b.set_xlim(0, cap + 2)
    style_axis(b, "x")
    panel(b, "b", x=-0.36)
    steps = decision["step"].to_numpy()
    schedules = [("selected", decision["injector_1_m3"], decision["injector_2_m3"], -0.2, None),
                 ("implemented", backtest["implemented_injector_1_m3"],
                  backtest["implemented_injector_2_m3"], 0.2, "////")]
    for name, f4, f5, shift, hatch in schedules:
        f4, f5 = f4.to_numpy() / 1e3, f5.to_numpy() / 1e3
        fill = {"edgecolor": "white", "linewidth": 0.6} if hatch else {}
        c.bar(steps + shift, f4, color=BLUE, width=0.38, hatch=hatch, label=f"F-4, {name}", **fill)
        c.bar(steps + shift, f5, bottom=f4, color=SKY, width=0.38, hatch=hatch,
              label=f"F-5, {name}", **fill)
    c.set_xlabel("Planning step (week)")
    c.set_ylabel(r"Injection ($10^3$ m$^3$ week$^{-1}$)")
    c.set_xticks(steps)
    top_c = max((decision["injector_1_m3"] + decision["injector_2_m3"]).max(),
                (backtest["implemented_injector_1_m3"] + backtest["implemented_injector_2_m3"]).max())
    c.set_ylim(0, 1.45 * top_c / 1e3)
    c.legend(frameon=False, loc="upper left", fontsize=7.5, ncol=2)
    style_axis(c)
    panel(c, "c")
    d.fill_between(steps, backtest["predicted_under_implemented_m3"] / 1e3,
                   decision["predicted_field_oil_m3"] / 1e3, color=CORAL, alpha=0.12, lw=0)
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
    d.set_ylim(0, 1.6 * top / 1e3)
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
