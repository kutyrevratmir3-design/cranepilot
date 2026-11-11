"""Render the manuscript figures from the committed retrospective replay tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, StrMethodFormatter


INK = "#242C32"
TEAL = "#006B7D"
ORANGE = "#A74724"
GREY = "#616B75"
MODELS = {
    "selected": ("CranePilot", TEAL, "-", "o"),
    "dense_matched_arx": ("Dense ARX", ORANGE, "--", "s"),
    "bank_simplex_1": ("Direct bank", GREY, ":", "^"),
}
AGES = [0, 20, 30, 50, 70, 100, 150, 200, 250]
HORIZONS = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50]
REPLAY = "dense_reference_20261108"
HORIZON_AUDIT = "dense_horizons_20261109"
OFFSET_REPLAY = "timestamp_offset_20261109"
OFFSET_AUDIT = "timestamp_offset_report_20261109"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.65,
    "axes.edgecolor": GREY,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "legend.handlelength": 2.7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def panel_title(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", pad=9)


def line(ax, x, values, key):
    label, color, style, marker = MODELS[key]
    ax.plot(
        x, values, label=label, color=color, linestyle=style, marker=marker,
        linewidth=1.55, markersize=4.25, markeredgewidth=1.0,
        markerfacecolor=color if key == "selected" else "white",
        zorder=4 if key == "selected" else 3,
    )


def legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.985), borderaxespad=0.0, columnspacing=1.8)


def line_axis(ax, xlabel, ylabel, ylim, ystep):
    ax.set(xlabel=xlabel, ylabel=ylabel, ylim=ylim)
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.grid(axis="y", color="#DFE4E7", linewidth=0.45)
    ax.set_axisbelow(True)


def save(fig, output, stem):
    fig.savefig(output / f"{stem}.pdf",
                metadata={"Creator": "CranePilot figure script", "CreationDate": None})
    plt.close(fig)


def forecast_by_activity(results, output):
    scores = pd.read_csv(results / REPLAY / "summary.csv")
    scores = scores[(scores["noise_quantum"] == 0) & scores["method"].isin(MODELS)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    fig.subplots_adjust(left=0.095, right=0.976, bottom=0.225, top=0.765, wspace=0.38)
    for ax, letter, title, group, limit, step in [
        (axes[0], "A", "All origins (7 records)", "all", 1.5, 0.5),
        (axes[1], "B", "Drive-active origins (6 records)", "drive_active", 5.0, 1.0),
    ]:
        subset = scores[scores["group"] == group]
        values = subset.pivot(index="age_ms", columns="method", values="mae_deg").loc[AGES]
        for model in MODELS:
            line(ax, AGES, values[model], model)
        line_axis(ax, "Angle age (ms)", "Mean absolute error (°)", (0, limit), step)
        ax.set(xlim=(-10, 260), xticks=[0, 50, 100, 150, 200, 250])
        panel_title(ax, letter, title)
    legend(fig, axes[0])
    save(fig, output, "forecast_by_activity")


def forecast_by_horizon(paired, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.976, bottom=0.225, top=0.765, wspace=0.22)
    for ax, letter, age in zip(axes, ["A", "B"], [0, 250], strict=True):
        subset = paired[(paired["noise_quantum"] == 0) & (paired["age_ms"] == age)]
        values = subset.set_index("horizon_s").loc[HORIZONS]
        for model in MODELS:
            line(ax, HORIZONS, values[model], model)
        line_axis(ax, "Forecast horizon (s)", "", (0, 2.62), 0.5)
        ax.set(xlim=(0, 1.57), xticks=[0.1, 0.5, 1.0, 1.5])
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
        panel_title(ax, letter, f"Angle age {age} ms")
    axes[0].set_ylabel("Mean absolute error (°)")
    legend(fig, axes[0])
    save(fig, output, "forecast_by_horizon")


def forecast_error_map(paired, output):
    delta = "selected_minus_dense_matched_arx"
    bound = np.ceil(paired[delta].abs().max() * 2) / 2
    palette = LinearSegmentedColormap.from_list(
        "model_delta", ["#195C81", "#F4F4F2", "#AA4D2C"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.45), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.973, bottom=0.34, top=0.88, wspace=0.17)
    for ax, letter, noise, title in [
        (axes[0], "A", 0, "No added angle noise"),
        (axes[1], "B", 1, "Added noise of ±1 encoder quantum"),
    ]:
        matrix = paired[paired["noise_quantum"] == noise].pivot(
            index="age_ms", columns="horizon_s", values=delta
        ).loc[AGES, HORIZONS]
        colors = ax.pcolormesh(
            matrix.to_numpy(), vmin=-bound, vmax=bound, cmap=palette,
            edgecolors="white", linewidth=0.55,
        )
        for i, j in np.argwhere(matrix.to_numpy() > 0):
            ax.add_patch(Rectangle(
                (j + 0.04, i + 0.04), 0.92, 0.92, fill=False,
                edgecolor=ORANGE, linewidth=1.4,
            ))
            ax.text(j + 0.5, i + 0.5, f"{matrix.iloc[i, j]:+.3f}",
                    ha="center", va="center", fontsize=8.0, color=INK)
        ax.set(
            ylim=(len(AGES), 0),
            xticks=np.arange(len(HORIZONS)) + 0.5,
            xticklabels=["0.10", "0.20", "0.35", "0.50", "0.75", "1.00", "1.50"],
            yticks=np.arange(len(AGES)) + 0.5,
            yticklabels=AGES,
            xlabel="Forecast horizon (s)",
        )
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.tick_params(length=0)
        panel_title(ax, letter, title)
    axes[0].set_ylabel("Angle age (ms)")
    color_axis = fig.add_axes((0.26, 0.147, 0.48, 0.045))
    bar = fig.colorbar(colors, cax=color_axis, orientation="horizontal",
                       ticks=[-bound, -bound / 2, 0, bound / 2, bound])
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor("face")
    bar.outline.set_visible(False)
    bar.ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    bar.set_label("CranePilot MAE minus dense ARX MAE (°)", labelpad=4)
    save(fig, output, "forecast_error_map")


def timestamp_sensitivity(results, output):
    common = pd.read_csv(results / OFFSET_REPLAY / "common_interior_offset_means.csv")
    common["offset_ms"] = common["offset_raw"] * 10
    common = common.pivot(index="offset_ms", columns="method", values="mae_deg").sort_index()
    short = pd.read_csv(results / OFFSET_AUDIT / "short_horizon_at_250ms.csv")
    short = short.set_index("offset_ms").sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    fig.subplots_adjust(left=0.095, right=0.976, bottom=0.225, top=0.765, wspace=0.38)
    for ax, letter, title, subset, limit, step in [
        (axes[0], "A", "Five interior angle ages", common, 1.60, 0.4),
        (axes[1], "B", "250 ms angle age, 0.1 s horizon", short, 1.0, 0.2),
    ]:
        ax.axvline(0, color="#AEB7BE", linewidth=0.9, zorder=2)
        for model in ["selected", "dense_matched_arx"]:
            line(ax, subset.index, subset[model], model)
        line_axis(ax, "Reported age minus true age (ms)",
                  "Mean absolute error (°)", (0, limit), step)
        ax.set(xticks=subset.index)
        ax.margins(x=0.09)
        panel_title(ax, letter, title)
    legend(fig, axes[0])
    save(fig, output, "timestamp_sensitivity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--output", type=Path, default=root / "paper" / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    paired = pd.read_csv(args.results / HORIZON_AUDIT / "paired_horizon_summary.csv")
    paired["horizon_s"] = paired["horizon_s"].round(2)
    forecast_by_activity(args.results, args.output)
    forecast_by_horizon(paired, args.output)
    forecast_error_map(paired, args.output)
    timestamp_sensitivity(args.results, args.output)


if __name__ == "__main__":
    main()
