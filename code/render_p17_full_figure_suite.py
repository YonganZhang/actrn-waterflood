"""Render the P17 data-figure suite (Figures 2--7).

Figure 1 is a generated scientific schematic governed by
``_paper/figure-suite-contract.md``.  This renderer is deliberately strict:
every data input must exist with the expected schema, and there is no
placeholder or synthetic fallback. Profile v2_4 is the canonical LaTeX source;
earlier profiles preserve the review history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import font_manager as fm
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ICON_DIR = ROOT / "paper" / "assets" / "icons"
INJECTION_ICON = plt.imread(ICON_DIR / "oil-well-energy-fuel-industry-red.png")
PRODUCTION_ICON = plt.imread(ICON_DIR / "oil-well-energy-fuel-industry-blue.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the strict P17 data-figure suite."
    )
    parser.add_argument(
        "--profile",
        choices=("v1", "v2", "v2_1", "v2_2", "v2_3", "v2_4"),
        default="v2_4",
        help=(
            "v1 retains the first suite; v2 retains the enlarged-font review; "
            "v2_1 adds data-aware missingness and refined panel encodings; "
            "v2_2 diversifies the descriptive network, screening, and "
            "producer-contribution panels; v2_3 replaces the categorical "
            "two-column network with an oilfield-pattern schematic layout; "
            "v2_4 promotes CRM-P/Pearson to panels a/b and adds an official "
            "ED50 surface-wellhead longitude/latitude panel."
        ),
    )
    return parser.parse_args()


ARGS = _parse_args() if __name__ == "__main__" else argparse.Namespace(profile="v2_4")
PROFILE = ARGS.profile
REFINED_PROFILES = {"v2_1", "v2_2", "v2_3", "v2_4"}
DIVERSE_PROFILES = {"v2_2", "v2_3", "v2_4"}
OUT = ROOT / "_outputs" / {
    "v1": "figure-suite-review-20260715",
    "v2": "figure-suite-review-v2-20260715",
    "v2_1": "figure-suite-review-v2-1-20260715",
    "v2_2": "figure-suite-review-v2-2-20260715",
    "v2_3": "figure-suite-review-v2-3-20260715",
    "v2_4": "figure-suite-review-v2-4-20260715",
}[PROFILE]
FIG = OUT / "figures"

FIGURE_MODES = {
    "A": (3.5, 3.5),
    "B": (7.2, 3.5),
    "C": (3.5, 5.5),
    "D": (7.2, 7.2),
    "E": (7.2, 2.5),
}

UKIYOE = [
    "#E76254",
    "#EF8A47",
    "#F7AA58",
    "#FFD06F",
    "#FFE6B7",
    "#AADCE0",
    "#72BCD5",
    "#528FAD",
    "#376795",
    "#1E466E",
]
BLUE = "#376795"
SKY = "#72BCD5"
YELLOW = "#FFD06F"
RED = "#E76254"
NAVY = "#1E466E"
GREY = "#D9D9D9"
LIGHT_GREY = "#F1F1F1"

FONT_PROFILES = {
    "v1": {
        "base": 8.0,
        "axis_label": 8.0,
        "tick": 7.0,
        "legend": 7.0,
        "panel": 10.0,
        "annotation": 6.5,
    },
    "v2": {
        "base": 10.0,
        "axis_label": 11.0,
        "tick": 9.5,
        "legend": 9.5,
        "panel": 14.0,
        "annotation": 9.0,
    },
    "v2_1": {
        "base": 10.0,
        "axis_label": 11.0,
        "tick": 9.5,
        "legend": 9.5,
        "panel": 14.0,
        "annotation": 9.0,
    },
    "v2_2": {
        "base": 10.0,
        "axis_label": 11.0,
        "tick": 9.5,
        "legend": 9.5,
        "panel": 14.0,
        "annotation": 9.0,
    },
    "v2_3": {
        "base": 10.0,
        "axis_label": 11.0,
        "tick": 9.5,
        "legend": 9.5,
        "panel": 14.0,
        "annotation": 9.0,
    },
    "v2_4": {
        "base": 10.0,
        "axis_label": 11.0,
        "tick": 9.5,
        "legend": 9.5,
        "panel": 14.0,
        "annotation": 9.0,
    },
}
FONT_SIZES = FONT_PROFILES[PROFILE]
FONT_AUDIT: list[dict] = []

_available_fonts = {font.name for font in fm.fontManager.ttflist}
SERIF_FONT = next(
    name
    for name in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif")
    if name in _available_fonts
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [SERIF_FONT],
        "mathtext.fontset": "stix",
        "font.size": FONT_SIZES["base"],
        "axes.labelsize": FONT_SIZES["axis_label"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "legend.fontsize": FONT_SIZES["legend"],
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.5,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

WELLS = [
    "15/9-F-4",
    "15/9-F-5_inj",
    "15/9-F-1 C",
    "15/9-F-11",
    "15/9-F-12",
    "15/9-F-14",
    "15/9-F-15 D",
]
PRODUCERS = [
    "15/9-F-1 C",
    "15/9-F-11",
    "15/9-F-12",
    "15/9-F-14",
    "15/9-F-15 D",
]
SHORT = {
    "15/9-F-4": "F-4",
    "15/9-F-5_inj": "F-5",
    "15/9-F-1 C": "F-1C",
    "15/9-F-11": "F-11",
    "15/9-F-12": "F-12",
    "15/9-F-14": "F-14",
    "15/9-F-15 D": "F-15D",
}

INPUTS = [
    PROC / "volve_panel.parquet",
    PROC / "volve_surface_wellheads.csv",
    PROC / "fieldwide_weekly.parquet",
    PROC / "fieldwide_folds.csv",
    PROC / "fieldwide_oof_predictions.csv",
    PROC / "fieldwide_effects.csv",
    PROC / "fieldwide_design.json",
    PROC / "fieldwide_diagnostics.json",
    PROC / "crm_lambda.csv",
    PROC / "network_corr.csv",
    PROC / "p17_audit_manifest.json",
    PROC / "synthetic_known_truth_identified_hgb_summary.json",
    PROC / "synthetic_known_truth_identified_hgb_realizations.csv",
    PROC / "synthetic_known_truth_identified_ridge_summary.json",
    PROC / "synthetic_known_truth_identified_ridge_realizations.csv",
    PROC / "synthetic_known_truth_identified_ols_summary.json",
    PROC / "synthetic_known_truth_identified_ols_realizations.csv",
    PROC / "synthetic_known_truth_hidden_policy_ols_summary.json",
    PROC / "synthetic_known_truth_hidden_policy_ols_realizations.csv",
    ROOT / "_paper" / "figure-03-review-caption-v2-4.md",
]

FIGURE_1 = FIG / "figure_01_fail_closed_architecture.png"
FIGURE_1_PROMPT = OUT / "figure_01_prompt.md"


def require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{source} missing columns: {sorted(missing)}")


def panel(ax, letter: str) -> None:
    ax.text(
        -0.10,
        1.10,
        letter,
        transform=ax.transAxes,
        fontsize=FONT_SIZES["panel"],
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111111",
    )


def panel_frame(ax) -> None:
    """Give each panel a quiet boundary so the multi-panel figure reads as a unit."""
    ax.add_patch(
        Rectangle(
            (-0.08, -0.08),
            1.16,
            1.16,
            transform=ax.transAxes,
            fill=False,
            edgecolor="#B7C1CC",
            linewidth=0.9,
            joinstyle="round",
            zorder=20,
            clip_on=False,
        )
    )


def normalize_fonts(fig) -> None:
    """Normalize every visible text object before export."""
    for ax in fig.get_axes():
        ax.xaxis.label.set_fontsize(FONT_SIZES["axis_label"])
        ax.yaxis.label.set_fontsize(FONT_SIZES["axis_label"])
        ax.xaxis.label.set_fontfamily(SERIF_FONT)
        ax.yaxis.label.set_fontfamily(SERIF_FONT)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(FONT_SIZES["tick"])
            tick.set_fontfamily(SERIF_FONT)
        legend = ax.get_legend()
        if legend is not None:
            for item in legend.get_texts():
                item.set_fontsize(FONT_SIZES["legend"])
                item.set_fontfamily(SERIF_FONT)
        for item in ax.texts:
            item.set_fontfamily(SERIF_FONT)
            if item.get_text() in set("abcdefg"):
                item.set_fontsize(FONT_SIZES["panel"])
                item.set_fontweight("bold")
            elif item.get_fontsize() < FONT_SIZES["annotation"]:
                item.set_fontsize(FONT_SIZES["annotation"])


def audit_fonts(fig, stem: str) -> None:
    """Fail loudly if any visible scientific text falls below the profile floor."""
    observed: dict[str, list[float]] = {
        "axis_label": [],
        "tick": [],
        "legend": [],
        "panel": [],
        "annotation": [],
    }
    for ax in fig.get_axes():
        for label in (ax.xaxis.label, ax.yaxis.label):
            if label.get_text().strip():
                observed["axis_label"].append(float(label.get_fontsize()))
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            if tick.get_visible() and tick.get_text().strip():
                observed["tick"].append(float(tick.get_fontsize()))
        legend = ax.get_legend()
        if legend is not None:
            observed["legend"].extend(
                float(item.get_fontsize())
                for item in legend.get_texts()
                if item.get_text().strip()
            )
        for item in ax.texts:
            if not item.get_visible() or not item.get_text().strip():
                continue
            bucket = "panel" if item.get_text() in set("abcdefg") else "annotation"
            observed[bucket].append(float(item.get_fontsize()))

    minimums = {
        bucket: min(values) if values else None for bucket, values in observed.items()
    }
    for bucket, minimum in minimums.items():
        if minimum is not None and minimum + 1e-9 < FONT_SIZES[bucket]:
            raise AssertionError(
                f"{stem}: {bucket} minimum {minimum} pt below "
                f"{FONT_SIZES[bucket]} pt floor"
            )
    FONT_AUDIT.append(
        {
            "figure": stem,
            "profile": PROFILE,
            "figure_width_in": float(fig.get_size_inches()[0]),
            "minimum_font_sizes_pt": minimums,
            "required_font_sizes_pt": {
                key: FONT_SIZES[key]
                for key in ("axis_label", "tick", "legend", "panel", "annotation")
            },
        }
    )


def save(fig, stem: str) -> list[Path]:
    normalize_fonts(fig)
    audit_fonts(fig, stem)
    FIG.mkdir(parents=True, exist_ok=True)
    outputs = [FIG / f"{stem}.pdf", FIG / f"{stem}.png"]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight", dpi=600)
    plt.close(fig)
    return outputs


def _monthly_axis(ax, periods: pd.PeriodIndex) -> None:
    years = sorted(set(periods.year))
    display_years = years if PROFILE == "v1" else [year for year in years if year % 2 == 0]
    if not display_years:
        display_years = years
    ticks = []
    labels = []
    for year in display_years:
        positions = np.flatnonzero(periods.year == year)
        if len(positions):
            ticks.append(float(positions[0]))
            labels.append(str(year))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=0)


def _monthly_matrix(
    panel_frame: pd.DataFrame,
    value: str,
    wells: list[str],
    periods: pd.PeriodIndex,
    active_fraction: bool = False,
) -> np.ndarray:
    frame = panel_frame.loc[panel_frame["well_id"].isin(wells)].copy()
    frame["month"] = frame["DATEPRD"].dt.to_period("M")
    if active_fraction:
        volume_cols = ["BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL", "BORE_WI_VOL"]
        frame["value"] = frame[volume_cols].fillna(0.0).sum(axis=1).gt(0).astype(float)
        grouped = frame.groupby(["well_id", "month"])["value"].sum()
        matrix = np.full((len(wells), len(periods)), np.nan)
        for i, well in enumerate(wells):
            for j, month in enumerate(periods):
                days = month.days_in_month
                matrix[i, j] = float(grouped.get((well, month), 0.0)) / float(days)
        return matrix

    frame["value"] = pd.to_numeric(frame[value], errors="coerce").replace(0.0, np.nan)
    grouped = frame.groupby(["well_id", "month"])["value"].median()
    matrix = np.full((len(wells), len(periods)), np.nan)
    for i, well in enumerate(wells):
        for j, month in enumerate(periods):
            matrix[i, j] = grouped.get((well, month), np.nan)
    return matrix


def render_figure_2() -> list[Path]:
    panel_frame = pd.read_parquet(PROC / "volve_panel.parquet")
    required = {
        "DATEPRD",
        "well_id",
        "BORE_OIL_VOL",
        "BORE_GAS_VOL",
        "BORE_WAT_VOL",
        "BORE_WI_VOL",
        "AVG_DOWNHOLE_TEMPERATURE",
        "AVG_DOWNHOLE_PRESSURE",
    }
    require_columns(panel_frame, required, "volve_panel.parquet")
    panel_frame["DATEPRD"] = pd.to_datetime(panel_frame["DATEPRD"])
    start = panel_frame["DATEPRD"].min().to_period("M")
    end = panel_frame["DATEPRD"].max().to_period("M")
    periods = pd.period_range(start, end, freq="M")

    activity = _monthly_matrix(panel_frame, "", WELLS, periods, active_fraction=True)
    temperature = _monthly_matrix(
        panel_frame, "AVG_DOWNHOLE_TEMPERATURE", PRODUCERS, periods
    )
    pressure = _monthly_matrix(panel_frame, "AVG_DOWNHOLE_PRESSURE", PRODUCERS, periods)
    producer_activity = activity[[WELLS.index(well) for well in PRODUCERS], :]

    cmap_activity = mcolors.LinearSegmentedColormap.from_list(
        "activity",
        ["#FFFFFF", "#AADCE0", "#376795", "#1E466E"]
        if PROFILE in REFINED_PROFILES
        else ["#F1F1F1", "#AADCE0", "#376795", "#1E466E"],
    )
    cmap_temp = mcolors.LinearSegmentedColormap.from_list(
        "temperature", ["#FFE6B7", "#FFD06F", "#EF8A47", "#E76254"]
    )
    cmap_pressure = mcolors.LinearSegmentedColormap.from_list(
        "pressure", ["#AADCE0", "#72BCD5", "#528FAD", "#1E466E"]
    )
    if PROFILE in REFINED_PROFILES:
        cmap_activity.set_bad("#FFFFFF")
        cmap_temp.set_bad((1.0, 1.0, 1.0, 0.0))
        cmap_pressure.set_bad((1.0, 1.0, 1.0, 0.0))
    else:
        for cmap in (cmap_activity, cmap_temp, cmap_pressure):
            cmap.set_bad("#ECECEC")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=FIGURE_MODES["D"],
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.15, 1.0, 1.0]},
    )
    payload = [
        (axes[0], activity, WELLS, cmap_activity, 0.0, 1.0, "Active fraction", "a", None),
        (
            axes[1],
            temperature,
            PRODUCERS,
            cmap_temp,
            float(np.nanpercentile(temperature, 2)),
            float(np.nanpercentile(temperature, 98)),
            r"Downhole temperature ($^\circ$C)",
            "b",
            (producer_activity > 0.0) & ~np.isfinite(temperature),
        ),
        (
            axes[2],
            pressure,
            PRODUCERS,
            cmap_pressure,
            float(np.nanpercentile(pressure, 2)),
            float(np.nanpercentile(pressure, 98)),
            "Downhole pressure (bar)",
            "c",
            (producer_activity > 0.0) & ~np.isfinite(pressure),
        ),
    ]
    for ax, matrix, wells, cmap, vmin, vmax, label, letter, active_missing in payload:
        im = ax.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if PROFILE in REFINED_PROFILES and active_missing is not None:
            for row, column in np.argwhere(active_missing):
                ax.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1.0,
                        1.0,
                        facecolor="#F6F6F6",
                        edgecolor="#B8B8B8",
                        linewidth=0.15,
                        hatch="////",
                        zorder=2,
                    )
                )
        ax.set_yticks(np.arange(len(wells)))
        ax.set_yticklabels([SHORT[well] for well in wells])
        _monthly_axis(ax, periods)
        ax.set_xlabel("Calendar month")
        cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.025)
        cb.set_label(label)
        if PROFILE in REFINED_PROFILES and letter == "b":
            ax.legend(
                handles=[
                    Patch(
                        facecolor="#F6F6F6",
                        edgecolor="#B8B8B8",
                        hatch="////",
                        label="Active month; no valid sensor",
                    )
                ],
                loc="lower left",
                bbox_to_anchor=(0.0, 1.01),
                frameon=False,
            )
        panel(ax, letter)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(12.5)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        cb.ax.tick_params(labelsize=11.5)
        cb.ax.yaxis.label.set_size(13)
    outputs = save(fig, "figure_02_field_state_heatmaps")
    if PROFILE in REFINED_PROFILES:
        coverage_rows = []
        for sensor_name, sensor_matrix in (
            ("temperature", temperature),
            ("pressure", pressure),
        ):
            observed = np.isfinite(sensor_matrix)
            active = producer_activity > 0.0
            for row, well in enumerate(PRODUCERS):
                observed_periods = periods[observed[row]]
                coverage_rows.append(
                    {
                        "well": SHORT[well],
                        "sensor": sensor_name,
                        "calendar_months": int(len(periods)),
                        "active_months": int(active[row].sum()),
                        "observed_months": int(observed[row].sum()),
                        "observed_within_active_months": int((observed[row] & active[row]).sum()),
                        "active_missing_sensor_months": int((~observed[row] & active[row]).sum()),
                        "observed_outside_active_months": int((observed[row] & ~active[row]).sum()),
                        "first_observed_month": str(observed_periods.min()) if len(observed_periods) else None,
                        "last_observed_month": str(observed_periods.max()) if len(observed_periods) else None,
                    }
                )
        coverage_path = OUT / "figure_02_sensor_coverage_audit.json"
        coverage_path.write_text(
            json.dumps(
                {
                    "schema_version": "p17-figure-02-sensor-coverage/v1",
                    "zero_values_treated_as_missing": True,
                    "no_interpolation": True,
                    "rows": coverage_rows,
                },
                indent=2,
            )
            + "\n"
        )
        outputs.append(coverage_path)
    return outputs


def _heatmap(
    fig,
    ax,
    frame: pd.DataFrame,
    cmap,
    vmin: float,
    vmax: float,
    letter: str,
    colorbar_label: str,
    colorbar_ax=None,
) -> None:
    matrix = frame.to_numpy(dtype=float)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(frame.shape[1]))
    ax.set_xticklabels([str(x).replace("15/9-", "") for x in frame.columns], rotation=28, ha="right")
    ax.set_yticks(np.arange(frame.shape[0]))
    ax.set_yticklabels([str(x).replace("15/9-", "").replace("_inj", "") for x in frame.index])
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=6.5)
    if colorbar_ax is None:
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    else:
        cb = fig.colorbar(im, cax=colorbar_ax)
    cb.set_label(colorbar_label)
    panel(ax, letter)


def _bipartite_network(
    ax,
    frame: pd.DataFrame,
    letter: str,
    colorbar_label: str,
    *,
    signed: bool,
) -> None:
    """Draw a fixed categorical injector-producer network (not a well map)."""
    matrix = frame.to_numpy(dtype=float)
    if matrix.shape != (2, 5) or not np.isfinite(matrix).all():
        raise ValueError("Bipartite network input must be a finite 2x5 matrix")
    if not signed and np.any(matrix < 0.0):
        raise ValueError("Unsigned network input contains negative weights")

    injector_x, producer_x = 0.20, 0.80
    injector_y = np.asarray([0.69, 0.47])
    producer_y = np.linspace(0.86, 0.30, frame.shape[1])
    magnitude_max = float(np.max(np.abs(matrix)))
    if magnitude_max <= 0.0:
        raise ValueError("Bipartite network has no non-zero edge")

    if signed:
        limit = max(abs(float(np.min(matrix))), abs(float(np.max(matrix))))
        norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        cmap = mcolors.LinearSegmentedColormap.from_list("network_signed", UKIYOE[::-1])
    else:
        norm = mcolors.Normalize(vmin=0.0, vmax=magnitude_max)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "network_unsigned", ["#AADCE0", "#72BCD5", "#528FAD", "#376795", "#1E466E"]
        )

    # Edges are undirected descriptive links: no arrows and no embedded labels.
    for i, y_left in enumerate(injector_y):
        for j, y_right in enumerate(producer_y):
            value = float(matrix[i, j])
            width = 0.65 + 4.35 * abs(value) / magnitude_max
            ax.plot(
                [injector_x, producer_x],
                [y_left, y_right],
                color=cmap(norm(value)),
                linewidth=width,
                alpha=0.72,
                solid_capstyle="round",
                zorder=1,
            )

    ax.scatter(
        np.full(len(injector_y), injector_x),
        injector_y,
        s=260,
        marker="s",
        facecolor=NAVY,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    ax.scatter(
        np.full(len(producer_y), producer_x),
        producer_y,
        s=240,
        marker="o",
        facecolor="#FFE6B7",
        edgecolor="#9C7519",
        linewidth=0.9,
        zorder=3,
    )
    for y, name in zip(injector_y, frame.index):
        label = SHORT.get(str(name), str(name).replace("15/9-", "").replace("_inj", ""))
        ax.text(injector_x - 0.07, y, label, ha="right", va="center")
    for y, name in zip(producer_y, frame.columns):
        label = SHORT.get(str(name), str(name).replace("15/9-", "").replace("_inj", ""))
        ax.text(producer_x + 0.07, y, label, ha="left", va="center")
    ax.text(injector_x, 0.98, "Injectors", ha="center", va="top", fontweight="bold")
    ax.text(producer_x, 0.98, "Producers", ha="center", va="top", fontweight="bold")

    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array(matrix)
    colorbar_ax = ax.inset_axes([0.23, 0.075, 0.54, 0.035])
    cb = ax.figure.colorbar(scalar, cax=colorbar_ax, orientation="horizontal")
    cb.set_label(colorbar_label, labelpad=1.5)
    cb.ax.tick_params(length=2.5, pad=1.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    panel(ax, letter)


def _pattern_network(
    ax,
    frame: pd.DataFrame,
    letter: str,
    colorbar_label: str,
    *,
    signed: bool,
    surface_reference: tuple[float, float] | None = None,
) -> None:
    """Draw an oilfield-pattern schematic without claiming mapped well coordinates."""
    matrix = frame.to_numpy(dtype=float)
    if matrix.shape != (2, 5) or not np.isfinite(matrix).all():
        raise ValueError("Pattern network input must be a finite 2x5 matrix")
    if not signed and np.any(matrix < 0.0):
        raise ValueError("Unsigned pattern network contains negative weights")
    magnitude_max = float(np.max(np.abs(matrix)))
    if magnitude_max <= 0.0:
        raise ValueError("Pattern network has no non-zero edge")

    injector_names = [SHORT.get(str(name), str(name).replace("15/9-", "")) for name in frame.index]
    producer_names = [SHORT.get(str(name), str(name).replace("15/9-", "")) for name in frame.columns]
    expected_injectors = ["F-4", "F-5"]
    expected_producers = ["F-1C", "F-11", "F-12", "F-14", "F-15D"]
    if injector_names != expected_injectors or producer_names != expected_producers:
        raise ValueError(
            "Pattern layout is locked to F-4/F-5 and the five registered producers; "
            f"got {injector_names} and {producer_names}"
        )

    # Two overlapping injection-pattern cells. These are normalized schematic
    # positions, not wellhead, bottom-hole, UTM, or reservoir-grid coordinates.
    positions = {
        "F-4": (-0.48, 0.08),
        "F-5": (0.42, -0.05),
        "F-1C": (-0.78, 0.70),
        "F-11": (0.00, 0.80),
        "F-12": (-0.08, 0.22),
        "F-14": (0.77, 0.55),
        "F-15D": (0.38, -0.72),
    }
    label_offsets = {
        "F-4": (-0.12, -0.22),
        "F-5": (0.12, -0.22),
        "F-1C": (-0.14, 0.18),
        "F-11": (0.00, 0.22),
        "F-12": (-0.18, 0.20),
        "F-14": (0.14, 0.18),
        "F-15D": (0.20, 0.00),
    }

    if signed:
        limit = max(abs(float(np.min(matrix))), abs(float(np.max(matrix))))
        norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        cmap = mcolors.LinearSegmentedColormap.from_list("pattern_signed", UKIYOE[::-1])
    else:
        norm = mcolors.Normalize(vmin=0.0, vmax=magnitude_max)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "pattern_unsigned", ["#AADCE0", "#72BCD5", "#528FAD", "#376795", "#1E466E"]
        )

    # A faint producer hull gives the plan-view reading of a field pattern but
    # carries no statistical value and is deliberately excluded from the legend.
    hull_names = ["F-1C", "F-11", "F-14", "F-15D", "F-12", "F-1C"]
    hull = np.asarray([positions[name] for name in hull_names], dtype=float)
    ax.plot(
        hull[:, 0],
        hull[:, 1],
        color="#B9B9B9",
        linewidth=0.85,
        linestyle=(0, (4, 4)),
        zorder=0,
    )

    for i, injector in enumerate(injector_names):
        x_left, y_left = positions[injector]
        for j, producer in enumerate(producer_names):
            x_right, y_right = positions[producer]
            value = float(matrix[i, j])
            width = 0.65 + 4.35 * abs(value) / magnitude_max
            ax.plot(
                [x_left, x_right],
                [y_left, y_right],
                color=cmap(norm(value)),
                linewidth=width,
                alpha=0.72,
                solid_capstyle="round",
                zorder=1,
            )

    # Petroleum SVG symbols: red and blue pumpjacks. Colour encodes the
    # operational role; shape is intentionally identical across well types.
    for injector in injector_names:
        x, y = positions[injector]
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(INJECTION_ICON, zoom=0.040),
                (x, y),
                frameon=False,
                pad=0,
                zorder=6,
            )
        )
    for producer in producer_names:
        x, y = positions[producer]
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(PRODUCTION_ICON, zoom=0.040),
                (x, y),
                frameon=False,
                pad=0,
                zorder=6,
            )
        )
    for name in [*injector_names, *producer_names]:
        x, y = positions[name]
        dx, dy = label_offsets[name]
        ax.text(
            x + dx,
            y + dy,
            name,
            ha="center",
            va="center",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.88},
            zorder=10,
        )

    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array(matrix)
    colorbar_ax = ax.inset_axes([0.23, 0.055, 0.54, 0.035])
    cb = ax.figure.colorbar(scalar, cax=colorbar_ax, orientation="horizontal")
    cb.set_label(colorbar_label, labelpad=1.5)
    cb.ax.tick_params(length=2.5, pad=1.5)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("auto")
    for grid_value in (-0.75, -0.25, 0.25, 0.75):
        ax.axvline(grid_value, color="#D6E1E8", linewidth=0.65, linestyle=(0, (2, 3)), zorder=0)
        ax.axhline(grid_value, color="#D6E1E8", linewidth=0.65, linestyle=(0, (2, 3)), zorder=0)
    ax.axis("off")
    panel_frame(ax)
    panel(ax, letter)


def _operating_topography(fig, ax, operating: pd.DataFrame, letter: str) -> None:
    """Render the observed joint-injection operating region without interpolation."""
    topography_cmap = mcolors.LinearSegmentedColormap.from_list(
        "topography_white_blue_red", ["#FFFFFF", "#376795", "#E76254"]
    )
    hb = ax.hexbin(
        operating["A_F4_m3"] / 1000.0,
        operating["inj_F5_m3"] / 1000.0,
        C=operating["field_oil_cum_h8_m3"] / 1000.0,
        reduce_C_function=np.mean,
        gridsize=18,
        mincnt=3,
        cmap=topography_cmap,
        linewidths=0.25,
        edgecolors="white",
    )
    ax.set_xlabel(r"F-4 injection ($10^3$ m$^3$/week)")
    ax.set_ylabel(r"F-5 injection ($10^3$ m$^3$/week)")
    cb = fig.colorbar(hb, ax=ax, pad=0.018, fraction=0.045)
    cb.set_label(r"Observed H8 field oil ($10^3$ m$^3$)")
    panel(ax, letter)


def _surface_wellhead_map(ax, wellheads: pd.DataFrame, letter: str) -> None:
    """Plot official surface-wellhead geography without implying reservoir targets."""
    required = {
        "well",
        "role",
        "coordinate_scope",
        "geodetic_datum",
        "latitude_deg",
        "longitude_deg",
        "utm_zone",
        "utm_northing_m",
        "utm_easting_m",
        "source_url",
    }
    require_columns(wellheads, required, "volve_surface_wellheads.csv")
    if wellheads["well"].duplicated().any():
        raise ValueError("Surface-wellhead registry contains duplicate well rows")
    if set(wellheads["well"]) != set(WELLS):
        raise ValueError("Surface-wellhead registry must contain exactly the seven plotted wells")
    if set(wellheads["coordinate_scope"]) != {"surface_wellhead"}:
        raise ValueError("Only surface-wellhead coordinates are permitted in panel c")
    if set(wellheads["geodetic_datum"]) != {"ED50"} or set(wellheads["utm_zone"]) != {31}:
        raise ValueError("Surface-wellhead map is locked to ED50 / UTM zone 31")
    if set(wellheads.loc[wellheads["role"] == "injector", "well"]) != set(WELLS[:2]):
        raise ValueError("Surface-wellhead injector roles do not match the analysis registry")

    coords = wellheads.set_index("well").loc[WELLS]
    lon = coords["longitude_deg"].astype(float)
    lat = coords["latitude_deg"].astype(float)
    if not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise ValueError("Surface-wellhead registry contains non-finite coordinates")

    # Convert the official ED50 longitude/latitude to local east/north offsets
    # (metres) from the wellhead centre; the clustered positions are platform
    # surface slots, and the scale bar makes that physical scale explicit.
    lat0 = float(lat.mean())
    lon0 = float(lon.mean())
    east_m = (lon - lon0) * 111_320.0 * np.cos(np.deg2rad(lat0))
    north_m = (lat - lat0) * 111_320.0

    ax.set_facecolor("#F3FAFC")
    ax.set_axisbelow(True)
    ax.grid(color="#C7DCE4", linewidth=0.65, linestyle=(0, (2, 3)))

    for name in WELLS[:2]:
        x, y = float(east_m.loc[name]), float(north_m.loc[name])
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(INJECTION_ICON, zoom=0.025),
                (x, y),
                frameon=False,
                pad=0,
                zorder=6,
            )
        )
    for name in PRODUCERS:
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(PRODUCTION_ICON, zoom=0.025),
                (float(east_m.loc[name]), float(north_m.loc[name])),
                frameon=False,
                pad=0,
                zorder=6,
            )
        )

    label_offsets = {
        "15/9-F-4": (-23, -15),
        "15/9-F-5_inj": (6, -21),
        "15/9-F-1 C": (-28, 7),
        "15/9-F-11": (-5, 18),
        "15/9-F-12": (18, 10),
        "15/9-F-14": (23, 6),
        "15/9-F-15 D": (24, -14),
    }
    for name in WELLS:
        ax.annotate(
            SHORT[name],
            xy=(float(east_m.loc[name]), float(north_m.loc[name])),
            xytext=label_offsets[name],
            textcoords="offset points",
            ha="center",
            va="center",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.86},
            arrowprops={"arrowstyle": "-", "color": "#6E7D82", "lw": 0.65},
            zorder=7,
        )

    x_pad, y_pad = 2.5, 2.5
    x_min, x_max = float(east_m.min() - x_pad), float(east_m.max() + x_pad)
    y_min, y_max = float(north_m.min() - y_pad), float(north_m.max() + y_pad)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.set_xlabel(r"East offset from field centre (m, ED50)")
    ax.set_ylabel(r"North offset from field centre (m, ED50)")
    ax.text(0.02, 1.03, f"ED50 lat {lat0:.5f}°, lon {lon0:.5f}°",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)

    # Five-metre scale bar in the local east-north metric frame plotted above.
    metres = 5.0
    scale_x = x_min + 0.07 * (x_max - x_min)
    scale_y = y_min + 0.075 * (y_max - y_min)
    ax.plot([scale_x, scale_x + metres], [scale_y, scale_y], color=NAVY, linewidth=2.0)
    ax.plot([scale_x, scale_x], [scale_y - 0.35, scale_y + 0.35], color=NAVY, linewidth=1.0)
    ax.plot(
        [scale_x + metres, scale_x + metres],
        [scale_y - 0.35, scale_y + 0.35],
        color=NAVY,
        linewidth=1.0,
    )
    ax.text(scale_x + metres / 2.0, scale_y + 0.65, "5 m", ha="center", va="bottom")

    compass = Circle((0.90, 0.84), 0.075, transform=ax.transAxes, facecolor="white", edgecolor=NAVY, lw=0.9, zorder=8)
    ax.add_patch(compass)
    ax.annotate("N", xy=(0.90, 0.905), xytext=(0.90, 0.84), xycoords=ax.transAxes,
                ha="center", va="center", color=NAVY, fontweight="bold",
                arrowprops={"arrowstyle": "-|>", "color": NAVY, "lw": 1.1}, zorder=9)
    ax.text(0.90, 0.775, "S", transform=ax.transAxes, ha="center", va="center", fontsize=8, zorder=9)
    ax.text(0.965, 0.84, "E", transform=ax.transAxes, ha="center", va="center", fontsize=8, zorder=9)
    ax.text(0.835, 0.84, "W", transform=ax.transAxes, ha="center", va="center", fontsize=8, zorder=9)
    panel(ax, letter)


def render_figure_3() -> list[Path]:
    oof = pd.read_csv(PROC / "fieldwide_oof_predictions.csv", parse_dates=["date"])
    require_columns(
        oof,
        {"date", "A_F4_m3", "field_oil_cum_h8_m3"},
        "fieldwide_oof_predictions.csv",
    )
    weekly = pd.read_parquet(PROC / "fieldwide_weekly.parquet")
    if not isinstance(weekly.index, pd.DatetimeIndex):
        weekly.index = pd.to_datetime(weekly.index)
    require_columns(weekly.reset_index(), {"inj_F5_m3"}, "fieldwide_weekly.parquet")
    operating = oof.merge(
        weekly[["inj_F5_m3"]].rename_axis("date").reset_index(),
        on="date",
        how="left",
        validate="one_to_one",
    )
    if operating["inj_F5_m3"].isna().any():
        raise ValueError("F-5 weekly injection missing after one-to-one date join")

    crm = pd.read_csv(PROC / "crm_lambda.csv", index_col=0)
    corr = pd.read_csv(PROC / "network_corr.csv", index_col=0)
    if crm.shape != (2, 5) or corr.shape != (2, 5):
        raise ValueError(f"Expected 2x5 CRM/Pearson matrices; got {crm.shape} and {corr.shape}")

    diverging = mcolors.LinearSegmentedColormap.from_list("ukiyoe_div", UKIYOE[::-1])
    wellheads: pd.DataFrame | None = None
    if PROFILE == "v2_4":
        wellheads = pd.read_csv(PROC / "volve_surface_wellheads.csv")
        surface_reference = (
            float(wellheads["latitude_deg"].mean()),
            float(wellheads["longitude_deg"].mean()),
        )
        # Keep the heterogeneous context panels as a compact single-column
        # 2x2 plate; the manuscript places this figure in one column.
        fig = plt.figure(figsize=(3.35, 6.2), constrained_layout=True)
        grid = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[1, 0])
        ax_c = fig.add_subplot(grid[2, 0])
        ax_d = fig.add_subplot(grid[3, 0])
        _pattern_network(
            ax_a,
            crm,
            "a",
            r"CRM-P $\lambda$",
            signed=False,
            surface_reference=surface_reference,
        )
        _pattern_network(
            ax_b,
            corr,
            "b",
            r"Pearson $r$",
            signed=True,
            surface_reference=surface_reference,
        )
        _surface_wellhead_map(ax_c, wellheads, "c")
        _operating_topography(fig, ax_d, operating, "d")
        for _ax in (ax_a, ax_b, ax_c, ax_d):
            _ax.set_box_aspect(1.0)
        fig.legend(
            handles=[
                Line2D([0], [0], marker="^", color="none", markerfacecolor=RED,
                       markeredgecolor="#7A211B", markersize=8, label="Injection well"),
                Line2D([0], [0], marker="^", color="none", markerfacecolor=BLUE,
                       markeredgecolor="#173B61", markersize=8, label="Production well"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=2,
            frameon=False,
            fancybox=True,
            edgecolor="#B7C1CC",
            facecolor="white",
            borderpad=0.45,
            handletextpad=0.5,
        )
    else:
        fig = plt.figure(figsize=FIGURE_MODES["D"], constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
        ax_top = fig.add_subplot(grid[0, :])
        _operating_topography(fig, ax_top, operating, "a")

    if PROFILE == "v2_4":
        pass
    elif PROFILE == "v2_3":
        ax_b = fig.add_subplot(grid[1, 0])
        ax_c = fig.add_subplot(grid[1, 1])
        _pattern_network(ax_b, crm, "b", r"CRM-P $\lambda$", signed=False)
        _pattern_network(ax_c, corr, "c", r"Pearson $r$", signed=True)
    elif PROFILE == "v2_2":
        ax_b = fig.add_subplot(grid[1, 0])
        ax_c = fig.add_subplot(grid[1, 1])
        _bipartite_network(ax_b, crm, "b", r"CRM-P $\lambda$", signed=False)
        _bipartite_network(ax_c, corr, "c", r"Pearson $r$", signed=True)
    elif PROFILE == "v2_1":
        grid_b = grid[1, 0].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.05)
        grid_c = grid[1, 1].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.05)
        ax_b, cb_b = fig.add_subplot(grid_b[0, 0]), fig.add_subplot(grid_b[0, 1])
        ax_c, cb_c = fig.add_subplot(grid_c[0, 0]), fig.add_subplot(grid_c[0, 1])
        _heatmap(fig, ax_b, crm, diverging, -0.30, 0.30, "b", r"CRM-P $\lambda$", cb_b)
        _heatmap(fig, ax_c, corr, diverging, -0.50, 0.50, "c", r"Pearson $r$", cb_c)
    else:
        _heatmap(fig, fig.add_subplot(grid[1, 0]), crm, diverging, -0.30, 0.30, "b", r"CRM-P $\lambda$")
        _heatmap(fig, fig.add_subplot(grid[1, 1]), corr, diverging, -0.50, 0.50, "c", r"Pearson $r$")
    stem = (
        "figure_03_well_networks_geography"
        if PROFILE == "v2_4"
        else "figure_03_operating_topography_baselines"
    )
    outputs = save(fig, stem)
    if PROFILE in DIVERSE_PROFILES:
        audit_path = OUT / "figure_03_network_encoding_audit.json"
        if PROFILE == "v2_4":
            assert wellheads is not None
            layout = "primary_schematic_pattern_networks_plus_surface_geography"
            position_semantics = {
                "panels_a_b": "normalized_pattern_space_not_mapped_coordinates",
                "panel_c": "official_ed50_surface_wellhead_coordinates",
                "panel_d": "observed_joint_injection_operating_space",
            }
            official_coordinates_used = True
            coordinate_note = (
                "Panel c uses official Norwegian Offshore Directorate surface-wellhead "
                "coordinates. These positions cluster at the Maersk Inspirer facility "
                "and are not reservoir target or bottom-hole coordinates."
            )
        else:
            layout = (
                "schematic_overlapping_injection_pattern_space"
                if PROFILE == "v2_3"
                else "fixed_bipartite_non_geographic"
            )
            position_semantics = (
                "normalized_pattern_space_not_mapped_coordinates"
                if PROFILE == "v2_3"
                else "fixed_categorical_non_geographic"
            )
            official_coordinates_used = False
            coordinate_note = (
                "Norwegian Offshore Directorate wellhead positions cluster at the "
                "Maersk Inspirer facility and do not supply the reservoir target or "
                "bottom-hole coordinates required for a true subsurface well map."
            )
        audit_path.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "p17-figure-03-network-encoding/v3"
                        if PROFILE == "v2_4"
                        else "p17-figure-03-network-encoding/v2"
                    ),
                    "layout": layout,
                    "position_semantics": position_semantics,
                    "network_panels": ["a", "b"] if PROFILE == "v2_4" else ["b", "c"],
                    "surface_geography_panel": "c" if PROFILE == "v2_4" else None,
                    "operating_topography_panel": "d" if PROFILE == "v2_4" else "a",
                    "official_wellhead_coordinates_used": official_coordinates_used,
                    "official_wellhead_coordinate_note": coordinate_note,
                    "reservoir_target_or_bottom_hole_coordinates_used": False,
                    "directional_arrows": False,
                    "causal_interpretation_permitted": False,
                    "crm_p_lambda": crm.to_dict(orient="index"),
                    "pearson_r": corr.to_dict(orient="index"),
                    "surface_wellheads": (
                        wellheads[
                            [
                                "well",
                                "role",
                                "npd_id",
                                "coordinate_scope",
                                "geodetic_datum",
                                "latitude_deg",
                                "longitude_deg",
                                "utm_zone",
                                "utm_northing_m",
                                "utm_easting_m",
                                "source_url",
                            ]
                        ].to_dict(orient="records")
                        if PROFILE == "v2_4" and wellheads is not None
                        else None
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        outputs.append(audit_path)
    if PROFILE == "v2_4":
        caption_path = OUT / "review_caption_figure_03_v2_4.md"
        caption_path.write_text(
            (ROOT / "_paper" / "figure-03-review-caption-v2-4.md").read_text()
        )
        outputs.append(caption_path)
    return outputs


def render_figure_4() -> list[Path]:
    folds = pd.read_csv(PROC / "fieldwide_folds.csv", parse_dates=["date"])
    require_columns(folds, {"date", "fold", "A_F4_m3"}, "fieldwide_folds.csv")
    design = json.loads((PROC / "fieldwide_design.json").read_text())
    required_design = {
        "n_weeks",
        "n_features",
        "feature_screening",
        "raw_sample_window_offsets_weeks",
        "required_minimum_train_test_center_distance_weeks",
    }
    missing = required_design - set(design)
    if missing:
        raise KeyError(f"fieldwide_design.json missing keys: {sorted(missing)}")

    fig = plt.figure(figsize=FIGURE_MODES["D"], constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.25])

    ax = fig.add_subplot(grid[0, :])
    offsets = np.arange(-8, 9)
    if PROFILE in REFINED_PROFILES:
        ax.set_xlim(-8.5, 8.5)
        ax.set_ylim(-0.65, 2.65)
        for offset in range(-8, 0):
            ax.add_patch(
                Rectangle(
                    (offset - 0.47, 2.0 - 0.33),
                    0.94,
                    0.66,
                    facecolor=SKY,
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
        for offset in range(1, 9):
            ax.add_patch(
                Rectangle(
                    (offset - 0.47, 0.0 - 0.33),
                    0.94,
                    0.66,
                    facecolor=RED,
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
        ax.axvspan(-0.22, 0.22, color=YELLOW, alpha=0.22, linewidth=0, zorder=0)
        ax.scatter(
            [0],
            [1],
            s=760,
            marker="D",
            facecolor=YELLOW,
            edgecolor="#9C7519",
            linewidth=0.9,
            zorder=3,
        )
        ax.set_xticks(offsets)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels([r"History $H_t$", r"Decision $A_t$, $B_t$", r"Outcome $Y_t(h)$"])
        ax.grid(axis="x", color="#E8E8E8", linewidth=0.4, zorder=-1)
    else:
        support = np.zeros((3, len(offsets)), dtype=int)
        support[0, offsets <= -1] = 1
        support[1, offsets == 0] = 2
        support[2, offsets >= 1] = 3
        cmap = mcolors.ListedColormap(["#FFFFFF", SKY, YELLOW, RED])
        ax.imshow(support, cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(offsets)))
        ax.set_xticklabels([str(x) for x in offsets])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels([r"History $H_t$", r"$A_t$, $B_t$", r"Outcome $Y_t(h)$"])
    ax.set_xlabel("Week offset from decision time $t$")
    if PROFILE not in REFINED_PROFILES:
        ax.set_xticks(np.arange(-0.5, len(offsets), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
    ax.legend(
        handles=[
            Patch(facecolor=SKY, label="Pretreatment features"),
            Patch(facecolor=YELLOW, label="Treatment/co-treatment"),
            Patch(facecolor=RED, label="Post-treatment outcomes"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
    )
    panel(ax, "a")

    ax = fig.add_subplot(grid[1, 0])
    purge = pd.Timedelta(weeks=16)
    fold_colors = [BLUE, SKY, YELLOW, "#EF8A47", RED]
    for fold_id, color in zip(sorted(folds["fold"].unique()), fold_colors):
        group = folds.loc[folds["fold"].eq(fold_id)]
        start, end = group["date"].min(), group["date"].max()
        ax.barh(
            fold_id,
            (end + purge - (start - purge)).days,
            left=mdates.date2num(start - purge),
            height=0.58,
            color=LIGHT_GREY,
            edgecolor="#9A9A9A",
            linewidth=0.4,
            hatch="////",
        )
        ax.barh(
            fold_id,
            (end - start).days + 7,
            left=mdates.date2num(start),
            height=0.58,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"Block {i + 1}" for i in range(5)])
    ax.set_xlabel("Decision week")
    ax.set_ylabel("Contiguous test block")
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, label="Test weeks"),
            Patch(facecolor=LIGHT_GREY, edgecolor="#9A9A9A", hatch="////", label="16-week purge"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        frameon=False,
    )
    panel(ax, "b")

    screening = design["feature_screening"]
    raw = int(screening["raw_feature_count"])
    retained = int(screening["retained_feature_count"])
    mostly_missing = len(screening["mostly_missing_features"])
    constant = len(screening["constant_features"])
    if retained + mostly_missing + constant != raw:
        raise ValueError("Feature-screening counts do not close to raw feature count")
    ax = fig.add_subplot(grid[1, 1])
    if PROFILE in DIVERSE_PROFILES:
        x = np.arange(4)
        width = 0.64
        ax.bar(x[0], raw, width, color=NAVY, edgecolor="white", linewidth=0.6)
        ax.bar(
            x[1],
            mostly_missing,
            width,
            bottom=raw - mostly_missing,
            color=YELLOW,
            edgecolor="#9C7519",
            linewidth=0.6,
            hatch="///",
        )
        ax.bar(
            x[2],
            constant,
            width,
            bottom=retained,
            color=RED,
            edgecolor="#A13F35",
            linewidth=0.6,
            hatch="///",
        )
        ax.bar(x[3], retained, width, color=BLUE, edgecolor="white", linewidth=0.6)
        connector_levels = [raw, raw - mostly_missing, retained]
        for left, level in zip(range(3), connector_levels):
            ax.plot(
                [x[left] + width / 2, x[left + 1] - width / 2],
                [level, level],
                color="#777777",
                linewidth=0.8,
                linestyle="--",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Raw input", "Missing filter", "Constant filter", "Retained"],
            rotation=22,
            ha="right",
        )
        ax.set_ylabel("Feature count")
        ax.set_xlabel("Screening stage")
        ax.set_ylim(0, raw * 1.10)
        ax.legend(
            handles=[
                Patch(facecolor=BLUE, label="Available set"),
                Patch(facecolor=YELLOW, hatch="///", label="Removed"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
        )
        ax.grid(axis="y", color=GREY, linewidth=0.5)
    else:
        categories = ["Raw", "Retained", "Mostly missing", "Constant"]
        values = [raw, retained, mostly_missing, constant]
        colors = [NAVY, BLUE, YELLOW, RED]
        ax.barh(np.arange(4), values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels(categories)
        ax.invert_yaxis()
        ax.set_xlabel("Feature count")
        ax.grid(axis="x", color=GREY, linewidth=0.5)
    panel(ax, "c")
    outputs = save(fig, "figure_04_time_contract_purging")
    if PROFILE in DIVERSE_PROFILES:
        audit_path = OUT / "figure_04_feature_screening_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "schema_version": "p17-figure-04-feature-screening/v1",
                    "raw_feature_count": raw,
                    "removed_mostly_missing_count": mostly_missing,
                    "removed_constant_count": constant,
                    "retained_feature_count": retained,
                    "closure_equation": f"{raw} - {mostly_missing} - {constant} = {retained}",
                    "closes": retained + mostly_missing + constant == raw,
                },
                indent=2,
            )
            + "\n"
        )
        outputs.append(audit_path)
    return outputs


def render_figure_5() -> list[Path]:
    effects = pd.read_csv(PROC / "fieldwide_effects.csv")
    required = {
        "outcome",
        "well",
        "horizon_weeks",
        "theta",
        "ci95_low",
        "ci95_high",
        "q_value_bh",
        "interpretation_status",
    }
    require_columns(effects, required, "fieldwide_effects.csv")
    if set(effects["interpretation_status"]) != {"UNSTABLE_EXPLORATORY_ONLY"}:
        raise ValueError("Unexpected interpretation status in fieldwide_effects.csv")

    field = effects.loc[effects["outcome"].eq("field_oil")].sort_values("horizon_weeks")
    producer = effects.loc[effects["outcome"].str.startswith("oil_")].copy()
    producer["well_short"] = producer["well"].map(
        {"F1C": "F-1C", "F11": "F-11", "F12": "F-12", "F14": "F-14", "F15D": "F-15D"}
    )
    order = ["F-1C", "F-11", "F-12", "F-14", "F-15D"]
    pivot = producer.pivot(index="well_short", columns="horizon_weeks", values="theta").reindex(order)
    if pivot.shape != (5, 8) or pivot.isna().any().any():
        raise ValueError("Producer projection matrix must be complete 5x8")
    horizons = pivot.columns.to_numpy(dtype=int)
    field_aligned = field.set_index("horizon_weeks").reindex(horizons)
    if field_aligned["theta"].isna().any():
        raise ValueError("Field projection is incomplete over producer horizons")
    producer_sum = pivot.sum(axis=0).to_numpy(dtype=float)
    field_total = field_aligned["theta"].to_numpy(dtype=float)
    decomposition_max_abs_diff = float(np.max(np.abs(producer_sum - field_total)))
    if decomposition_max_abs_diff > 1e-12:
        raise ValueError(
            "Producer point estimates do not close to the field projection: "
            f"max abs diff={decomposition_max_abs_diff:.3e}"
        )

    fig = plt.figure(figsize=FIGURE_MODES["D"], constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.15],
        width_ratios=[1.0, 1.0] if PROFILE == "v1" else [1.25, 0.95],
    )
    ax = fig.add_subplot(grid[0, :])
    yerr = np.vstack([field["theta"] - field["ci95_low"], field["ci95_high"] - field["theta"]])
    if PROFILE in REFINED_PROFILES:
        ax.fill_between(
            field["horizon_weeks"],
            field["ci95_low"],
            field["ci95_high"],
            color=SKY,
            alpha=0.30,
            linewidth=0,
            label="95% confidence interval",
        )
        ax.plot(
            field["horizon_weeks"],
            field["theta"],
            "o-",
            color=NAVY,
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=5,
            label="Point estimate",
        )
        ax.legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
        )
    else:
        ax.errorbar(
            field["horizon_weeks"],
            field["theta"],
            yerr=yerr,
            fmt="o-",
            color=BLUE,
            markerfacecolor="white",
            markeredgewidth=0.9,
            markersize=4,
            capsize=2.5,
        )
    ax.axhline(0, color="#222222", linewidth=0.7, linestyle="--")
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Projection horizon (weeks)")
    ax.set_ylabel(
        "Cumulative field-oil projection\n" + r"(m$^3$/m$^3$ F-4)"
        if PROFILE in REFINED_PROFILES
        else r"Cumulative field oil projection (m$^3$/m$^3$ F-4)"
    )
    ax.grid(axis="y", color=GREY, linewidth=0.5)
    panel(ax, "a")

    ax = fig.add_subplot(grid[1, 0])
    if PROFILE in DIVERSE_PROFILES:
        producer_colors = [NAVY, SKY, YELLOW, "#EF8A47", RED]
        positive_base = np.zeros(len(horizons), dtype=float)
        negative_base = np.zeros(len(horizons), dtype=float)
        for producer_name, color in zip(order, producer_colors):
            values = pivot.loc[producer_name].to_numpy(dtype=float)
            positive = np.clip(values, 0.0, None)
            negative = np.clip(values, None, 0.0)
            positive_upper = positive_base + positive
            negative_lower = negative_base + negative
            ax.fill_between(
                horizons,
                positive_base,
                positive_upper,
                facecolor=color,
                edgecolor="white",
                linewidth=0.55,
                alpha=0.86,
            )
            ax.fill_between(
                horizons,
                negative_base,
                negative_lower,
                facecolor=color,
                edgecolor="white",
                linewidth=0.55,
                alpha=0.86,
            )
            positive_base = positive_upper
            negative_base = negative_lower
        signed_stack = positive_base + negative_base
        if not np.allclose(signed_stack, field_total, rtol=0.0, atol=1e-12):
            raise AssertionError("Signed contribution stack does not equal field total")
        ax.plot(
            horizons,
            field_total,
            color="#222222",
            linewidth=1.35,
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            label="Field total",
            zorder=4,
        )
        ax.axhline(0, color="#222222", linewidth=0.7)
        ax.set_xticks(horizons)
        lower = min(-0.08, float(np.min(negative_base)) * 1.25)
        upper = max(0.10, float(np.max(positive_base)) * 1.10)
        ax.set_ylim(lower, upper)
        ax.set_xlabel("Projection horizon (weeks)")
        ax.set_ylabel("Producer contribution\n" + r"(m$^3$/m$^3$ F-4)")
        ax.legend(
            handles=[
                *[
                    Patch(facecolor=color, label=producer_name)
                    for producer_name, color in zip(order, producer_colors)
                ],
                Line2D(
                    [0],
                    [0],
                    color="#222222",
                    marker="o",
                    markerfacecolor="white",
                    label="Field total",
                ),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=False,
        )
        ax.grid(axis="y", color=GREY, linewidth=0.5)
    else:
        vmax = float(np.nanmax(np.abs(pivot.to_numpy())))
        cmap = mcolors.LinearSegmentedColormap.from_list("producer_effects", UKIYOE[::-1])
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels(range(1, 9))
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels(order)
        ax.set_xlabel("Projection horizon (weeks)")
        ax.set_ylabel("Producer")
        for i in range(5):
            for j in range(8):
                ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=6.2)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
        cb.set_label(r"Projection (m$^3$/m$^3$ F-4)")
    panel(ax, "b")

    h8 = producer.loc[producer["horizon_weeks"].eq(8)].copy()
    h8["well_short"] = pd.Categorical(h8["well_short"], categories=order, ordered=True)
    h8 = h8.sort_values("well_short")
    ax = fig.add_subplot(grid[1, 1])
    y = np.arange(len(h8))[::-1]
    xerr = np.vstack([h8["theta"] - h8["ci95_low"], h8["ci95_high"] - h8["theta"]])
    ax.errorbar(
        h8["theta"],
        y,
        xerr=xerr,
        fmt="o",
        color=RED,
        markeredgecolor="white",
        markeredgewidth=0.4,
        capsize=2.5,
    )
    ax.axvline(0, color="#222222", linewidth=0.7, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(h8["well_short"].astype(str))
    ax.set_xlabel(r"H8 projection (m$^3$/m$^3$ F-4)")
    ax.set_ylabel("Producer")
    ax.grid(axis="x", color=GREY, linewidth=0.5)
    panel(ax, "c")
    outputs = save(fig, "figure_05_dynamic_projections")
    if PROFILE in DIVERSE_PROFILES:
        audit_path = OUT / "figure_05_contribution_closure_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "schema_version": "p17-figure-05-contribution-closure/v1",
                    "interpretation": "point_estimate_decomposition_only",
                    "uncertainty_decomposition_permitted": False,
                    "horizon_weeks": horizons.tolist(),
                    "producer_sum": producer_sum.tolist(),
                    "field_total": field_total.tolist(),
                    "max_abs_difference": decomposition_max_abs_diff,
                    "tolerance": 1e-12,
                    "closes": decomposition_max_abs_diff <= 1e-12,
                },
                indent=2,
            )
            + "\n"
        )
        outputs.append(audit_path)
    return outputs


def render_figure_6() -> list[Path]:
    diagnostics = json.loads((PROC / "fieldwide_diagnostics.json").read_text())
    folds = diagnostics["fold_nuisance_diagnostics"]
    sensitivity = diagnostics["sensitivity"]
    if len(folds) != 5 or len(sensitivity) != 4:
        raise ValueError("Expected five fold diagnostics and four sensitivity specifications")
    oof = pd.read_csv(PROC / "fieldwide_oof_predictions.csv", parse_dates=["date"])
    require_columns(
        oof,
        {"fold", "A_F4_resid", "field_oil_resid"},
        "fieldwide_oof_predictions.csv",
    )

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_MODES["D"], constrained_layout=True)
    ax = axes[0, 0]
    theta = np.asarray([row["primary_theta"] for row in folds])
    x = np.arange(1, 6)
    colors = [BLUE if value >= 0 else RED for value in theta]
    ax.vlines(x, 0, theta, color=colors, linewidth=1.2)
    ax.scatter(x, theta, color=colors, s=28, zorder=3, edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="#222222", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xlabel("Contiguous test block")
    ax.set_ylabel(r"H8 projection (m$^3$/m$^3$ F-4)")
    ax.grid(axis="y", color=GREY, linewidth=0.5)
    panel(ax, "a")

    primary = {
        "learner": diagnostics["learner"],
        "purge_weeks": diagnostics["purge_weeks"],
        **diagnostics["primary_field_effect"],
    }
    specs = [primary, *sensitivity]
    labels = [f"{row['learner'].upper()}-p{row['purge_weeks']}" for row in specs]
    y = np.arange(len(specs))[::-1]
    estimate = np.asarray([row["theta"] for row in specs])
    low = np.asarray([row["ci95_low"] for row in specs])
    high = np.asarray([row["ci95_high"] for row in specs])
    xerr = np.vstack([estimate - low, high - estimate])
    ax = axes[0, 1]
    for index, color in enumerate([BLUE, SKY, YELLOW, "#EF8A47", RED]):
        ax.errorbar(
            estimate[index],
            y[index],
            xerr=xerr[:, index : index + 1],
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            capsize=2.5,
        )
    ax.axvline(0, color="#222222", linewidth=0.7, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"H8 projection (m$^3$/m$^3$ F-4)")
    ax.grid(axis="x", color=GREY, linewidth=0.5)
    panel(ax, "b")

    ax = axes[1, 0]
    treatment_r2 = np.asarray([row["treatment_r2"] for row in folds])
    outcome_r2 = np.asarray([row["primary_outcome_r2"] for row in folds])
    width = 0.36
    ax.bar(x - width / 2, treatment_r2, width, color=SKY, label="Treatment")
    ax.bar(x + width / 2, outcome_r2, width, color=RED, label="H8 field oil")
    ax.axhline(0, color="#222222", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xlabel("Contiguous test block")
    ax.set_ylabel(r"Out-of-fold $R^2$")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color=GREY, linewidth=0.5)
    panel(ax, "c")

    ax = axes[1, 1]
    fold_colors = [BLUE, SKY, YELLOW, "#EF8A47", RED]
    for fold_id, color in enumerate(fold_colors):
        rows = oof.loc[oof["fold"].eq(fold_id)]
        ax.scatter(
            rows["A_F4_resid"] / 1000.0,
            rows["field_oil_resid"] / 1000.0,
            s=12,
            alpha=0.48,
            color=color,
            edgecolors="none",
            label=f"Block {fold_id + 1}",
        )
    ax.axhline(0, color="#222222", linewidth=0.6)
    ax.axvline(0, color="#222222", linewidth=0.6)
    ax.set_xlabel(r"F-4 treatment residual ($10^3$ m$^3$)")
    ax.set_ylabel(r"H8 field-oil residual ($10^3$ m$^3$)")
    if PROFILE in REFINED_PROFILES:
        ax.legend(
            frameon=False,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
        )
    else:
        ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(color="#E7E7E7", linewidth=0.4)
    panel(ax, "d")
    return save(fig, "figure_06_credibility_diagnostics")


def _synthetic_runs() -> list[tuple[str, Path, dict]]:
    definitions = [
        ("Identified HGB", "identified_hgb"),
        ("Identified Ridge", "identified_ridge"),
        ("Identified OLS", "identified_ols"),
        ("Hidden-policy OLS", "hidden_policy_ols"),
    ]
    runs = []
    for label, suffix in definitions:
        rows_path = PROC / f"synthetic_known_truth_{suffix}_realizations.csv"
        summary_path = PROC / f"synthetic_known_truth_{suffix}_summary.json"
        if not rows_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Missing canonical synthetic run: {suffix}")
        runs.append((label, rows_path, json.loads(summary_path.read_text())))
    return runs


def render_figure_7() -> list[Path]:
    runs = _synthetic_runs()
    manifest = json.loads((PROC / "p17_audit_manifest.json").read_text())
    configs = manifest["known_truth_calibration"]["canonical_configurations"]
    config_keys = ["identified_hgb", "identified_ridge", "identified_ols", "hidden_policy_ols"]

    fig = plt.figure(figsize=FIGURE_MODES["D"], constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], width_ratios=[1.55, 0.75])
    ax = fig.add_subplot(grid[0, :])
    markers = ["o", "s", "^", "D"]
    colors = [BLUE, SKY, YELLOW, RED]
    all_values: list[float] = []
    for (label, rows_path, _), color, marker in zip(runs, colors, markers):
        rows = pd.read_csv(rows_path)
        require_columns(rows, {"truth", "dml_theta"}, rows_path.name)
        if len(rows) != 60:
            raise ValueError(f"{rows_path.name} must contain 60 independent realizations")
        all_values.extend(rows["truth"].tolist())
        all_values.extend(rows["dml_theta"].tolist())
        ax.scatter(
            rows["truth"],
            rows["dml_theta"],
            s=14,
            alpha=0.42,
            color=color,
            marker=marker,
            edgecolors="none",
            label=label,
        )
    lower, upper = float(np.min(all_values)), float(np.max(all_values))
    pad = 0.05 * (upper - lower)
    ax.plot(
        [lower - pad, upper + pad],
        [lower - pad, upper + pad],
        color="#222222",
        linestyle="--",
        linewidth=0.7,
    )
    ax.set_xlim(lower - pad, upper + pad)
    ax.set_ylim(lower - pad, upper + pad)
    ax.set_xlabel(r"Analytic closed-loop truth (m$^3$/m$^3$)")
    ax.set_ylabel(r"DML estimate (m$^3$/m$^3$)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(color="#E1E1E1", linewidth=0.4)
    panel(ax, "a")

    ax = fig.add_subplot(grid[1, 0])
    labels = ["Id. HGB", "Id. Ridge", "Id. OLS", "Hidden OLS"]
    coverage = np.asarray([configs[key]["coverage_95"] for key in config_keys])
    fpr = np.asarray([configs[key]["null_false_positive_rate"] for key in config_keys])
    coverage_ci = np.asarray([configs[key]["coverage_exact_binomial_95_interval"] for key in config_keys])
    fpr_ci = np.asarray([configs[key]["null_fpr_exact_binomial_95_interval"] for key in config_keys])
    x = np.arange(4)
    ax.errorbar(
        x - 0.10,
        coverage,
        yerr=np.vstack([coverage - coverage_ci[:, 0], coverage_ci[:, 1] - coverage]),
        fmt="o",
        color=BLUE,
        markerfacecolor="white",
        capsize=2.5,
        label="95% coverage",
    )
    ax.errorbar(
        x + 0.10,
        fpr,
        yerr=np.vstack([fpr - fpr_ci[:, 0], fpr_ci[:, 1] - fpr]),
        fmt="s",
        color=RED,
        markerfacecolor="white",
        capsize=2.5,
        label="Null FPR",
    )
    ax.axhline(0.90, color=BLUE, linestyle="--", linewidth=0.8)
    ax.axhline(0.10, color=RED, linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Realisation proportion")
    if PROFILE == "v1":
        ax.legend(frameon=False, loc="center left")
    else:
        ax.legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
        )
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.4)
    panel(ax, "b")

    ax = fig.add_subplot(grid[1, 1])
    gate_labels = [
        "Observed response",
        "Specification robustness",
        "Known-truth calibration",
        "Causal decision use",
        "Economic optimisation",
    ]
    if PROFILE in REFINED_PROFILES:
        observed_checks = list(manifest["observed_credibility"]["checks"].values())
        specification_checks = list(manifest["specification_stability"]["checks"].values())
        known_truth_checks = [
            configs[key]["validation_status"].startswith("PASS") for key in config_keys
        ]
        check_groups = [
            observed_checks,
            specification_checks,
            known_truth_checks,
            [manifest["causal_reporting_permitted"]],
            [manifest["economic_reporting_permitted"]],
        ]
        pass_share = np.asarray([np.mean(group) for group in check_groups], dtype=float)
        fail_share = 1.0 - pass_share
        y = np.arange(len(gate_labels))
        ax.barh(
            y,
            pass_share,
            color=BLUE,
            edgecolor="white",
            linewidth=0.5,
            label="Criteria met",
        )
        ax.barh(
            y,
            fail_share,
            left=pass_share,
            color=RED,
            edgecolor="white",
            linewidth=0.5,
            label="Criteria unmet",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.set_yticks(y)
        ax.set_yticklabels(gate_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Share of evaluation criteria")
        ax.legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=1,
        )
        ax.grid(axis="x", color="#E1E1E1", linewidth=0.4)
    else:
        gate_values = np.asarray(
            [
                manifest["observed_credibility"]["passes"],
                manifest["specification_stability"]["passes"],
                manifest["known_truth_calibration"]["passes"],
                manifest["causal_reporting_permitted"],
                manifest["economic_reporting_permitted"],
            ],
            dtype=int,
        )[:, None]
        gate_cmap = mcolors.ListedColormap([RED, BLUE])
        im = ax.imshow(gate_values, aspect="auto", cmap=gate_cmap, vmin=0, vmax=1)
        ax.set_xticks([0])
        ax.set_xticklabels(["Machine gate"])
        ax.set_yticks(np.arange(len(gate_labels)))
        ax.set_yticklabels(gate_labels)
        cb = fig.colorbar(im, ax=ax, pad=0.05, fraction=0.12, ticks=[0, 1])
        cb.ax.set_yticklabels(["Fail / closed", "Pass / open"])
    panel(ax, "c")
    return save(fig, "figure_07_known_truth_gates")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(outputs: list[Path]) -> None:
    provenance_inputs = [*INPUTS]
    if FIGURE_1.exists() or FIGURE_1_PROMPT.exists():
        if not FIGURE_1.exists() or not FIGURE_1_PROMPT.exists():
            raise FileNotFoundError(
                "Figure 1 and its generation record must be present together"
            )
        provenance_inputs.append(FIGURE_1_PROMPT)
        outputs = [FIGURE_1, *outputs]
    font_audit_path = OUT / "font_audit.json"
    font_audit_path.write_text(json.dumps(FONT_AUDIT, indent=2) + "\n")
    outputs = [*outputs, font_audit_path]
    payload = {
        "schema_version": f"p17-figure-suite-preview/{PROFILE}",
        "status": (
            "INTEGRATED_IN_LATEX_FIGURES_2_TO_7"
            if PROFILE == "v2_4"
            else "REVIEW_ONLY_NOT_INTEGRATED_IN_LATEX"
        ),
        "scientific_status": "AUDIT_CLOSED_CAUSAL_AND_ECONOMIC_REPORTING_WITHHELD",
        "renderer": str(Path(__file__).relative_to(ROOT)),
        "renderer_command": f"python3 code/render_p17_full_figure_suite.py --profile {PROFILE}",
        "font_profile": FONT_SIZES,
        "font": SERIF_FONT,
        "inputs": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in provenance_inputs
        ],
        "outputs": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in outputs
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    missing = [path for path in INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required figure inputs: {missing}")
    print(f"Times-compatible serif: {SERIF_FONT}")
    outputs: list[Path] = []
    for number, renderer in [
        (2, render_figure_2),
        (3, render_figure_3),
        (4, render_figure_4),
        (5, render_figure_5),
        (6, render_figure_6),
        (7, render_figure_7),
    ]:
        rendered = renderer()
        outputs.extend(rendered)
        print(f"Figure {number}: {', '.join(path.name for path in rendered)}")
    write_manifest(outputs)
    print(f"Manifest: {OUT / 'figure_manifest.json'}")


if __name__ == "__main__":
    main()
