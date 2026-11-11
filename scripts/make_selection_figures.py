"""Render method-selection and measured-input figures for the CranePilot paper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

INK = "#242C32"
TEAL = "#006B7D"
ORANGE = "#A74724"
GREY = "#616B75"
GRID = "#DFE4E7"
AGES_MS = [0, 20, 50, 100, 150, 200]
DATED = "dated_forecast_20261108"
DENSE = "dense_reference_20261108"

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
    "legend.fontsize": 8.3,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def save(fig, output, stem):
    fig.savefig(output / f"{stem}.pdf",
                metadata={"Creator": "CranePilot figure script", "CreationDate": None})
    plt.close(fig)


def validation_selection(results, output):
    dated = pd.read_csv(results / DATED / "validation_summary.csv").set_index("method")
    dense = pd.read_csv(results / DENSE / "balanced_summary.csv")
    dense = dense[dense.role == "validation"]
    dense_age = dense.groupby(["age_ms", "method"]).mae_deg.mean().unstack()
    overall = dated.balanced_mae_deg.to_dict()
    overall["dense_matched_arx"] = dense_age.dense_matched_arx.mean()

    selection = json.loads((results / DATED / "frozen_development/selection.json").read_text())
    scores = pd.concat([
        pd.DataFrame(rows).assign(case=case)
        for case, rows in selection["case_scores"].items()
    ])
    scores["age_ms"] = 10 * scores.age_raw
    by_case = scores.groupby(["method", "age_ms", "case"]).score.mean()
    dated_age = by_case.groupby(["age_ms", "method"]).mean().unstack()
    if list(dated_age.index) != AGES_MS or list(dense_age.index) != AGES_MS:
        raise ValueError("Unexpected validation ages")
    np.testing.assert_allclose(dated_age.mix_bank_10, dense_age.selected)
    np.testing.assert_allclose(dated_age.mean().loc[dated.index], dated.balanced_mae_deg)

    fig, (rank, curve) = plt.subplots(1, 2, figsize=(7.0, 4.08),
                                     gridspec_kw={"width_ratios": [1.02, 1]})
    fig.subplots_adjust(left=0.295, right=0.985, bottom=0.225, top=0.89, wspace=0.44)
    variants = [
        ("mix_bank_10", "Bank blend, 100 ms", TEAL, "o"),
        ("mix_bank_20", "Bank blend, 200 ms", GREY, "o"),
        ("mix_bank_5", "Bank blend, 50 ms", GREY, "o"),
        ("mix_free_20", "Free-mode blend, 200 ms", GREY, "o"),
        ("mix_horizon_10", "Horizon blend, 100 ms", GREY, "o"),
        ("dense_matched_arx", "Dense ARX", ORANGE, "s"),
        ("matched_arx", "Earlier matched ARX", ORANGE, "s"),
        ("direct_linear_fixed", "Best individual expert", GREY, "o"),
        ("bank_simplex_1", "Direct bank alone", GREY, "^"),
        ("parent_quad", "Quadratic completion", GREY, "D"),
    ]
    variants.sort(key=lambda variant: overall[variant[0]])
    for index, (method, label, color, marker) in enumerate(variants):
        value = overall[method]
        rank.plot(value, index, marker=marker, markersize=5.1, linestyle="none",
                  markerfacecolor=color if method == "mix_bank_10" else "white",
                  markeredgecolor=color, markeredgewidth=1.05, zorder=4)
        rank.annotate(f"{value:.3f}", (value, index), xytext=(5, 0),
                      textcoords="offset points", va="center", fontsize=8.3,
                      color=color)
    rank.set(
        yticks=np.arange(len(variants)),
        yticklabels=[label for _, label, _, _ in variants],
        ylim=(len(variants) - 0.42, -0.58),
        xlim=(1.52, 1.99),
        xlabel="Balanced MAE (°)",
    )
    rank.tick_params(axis="y", length=0, pad=6)
    rank.spines["left"].set_visible(False)
    rank.get_yticklabels()[0].set_color(TEAL)
    rank.get_yticklabels()[0].set_fontweight("bold")
    rank.xaxis.set_major_locator(MultipleLocator(0.1))
    rank.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    rank.grid(axis="x", linewidth=0.55, color=GRID)
    rank.set_axisbelow(True)
    rank.set_title("A  Selected validation comparisons", loc="left", pad=9)

    for values, label, color, linestyle, marker in [
        (dated_age.mix_bank_10, "Selected bank blend", TEAL, "-", "o"),
        (dense_age.dense_matched_arx, "Dense ARX", ORANGE, "--", "s"),
        (dated_age.bank_simplex_1, "Direct bank alone", GREY, ":", "^"),
        (dated_age.parent_quad, "Quadratic completion", INK, "-.", "D"),
    ]:
        curve.plot(values.index, values, label=label, color=color, linestyle=linestyle,
                   linewidth=1.55, marker=marker, markersize=4.1,
                   markerfacecolor=color if marker == "o" else "white",
                   markeredgewidth=0.95)
    curve.set(xlim=(-8, 208), xticks=[0, 50, 100, 150, 200], xlabel="Angle age (ms)",
              ylim=(1.23, 3.03), ylabel="Balanced MAE (°)")
    curve.yaxis.set_major_locator(MultipleLocator(0.4))
    curve.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    curve.set_axisbelow(True)
    curve.grid(axis="y", color=GRID, linewidth=0.55)
    curve.set_title("B  Same score by angle age", loc="left", pad=9)
    handles, labels = curve.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.55, 0.027),
               ncol=4, columnspacing=1.25, handlelength=2.25)
    save(fig, output, "dated_validation_selection")


def dated_inputs(dataset, output):
    scenario = 9
    origin_sample = 525
    age_samples = 25
    horizon_samples = np.array([10, 20, 35, 50, 75, 100, 150])
    last_angle = origin_sample - age_samples
    history_angle = np.arange(last_angle - 400, last_angle + 1)
    history_support = np.arange(origin_sample - 400, origin_sample + 1)
    withheld = np.arange(last_angle, origin_sample + horizon_samples.max() + 1)
    targets = origin_sample + horizon_samples

    with np.load(dataset, allow_pickle=False) as data:
        time = data["time"] - data["time"][origin_sample]
        angle = np.rad2deg(data[f"s{scenario}_theta"])
        trolley = data[f"s{scenario}_x"]
        length = data[f"s{scenario}_L"]
    np.testing.assert_allclose(time[history_angle[[0, -1]]], [-4.25, -0.25])
    np.testing.assert_allclose(time[history_support[[0, -1]]], [-4, 0])
    np.testing.assert_allclose(time[targets], horizon_samples * 0.01)
    fig, (sway, position, cable) = plt.subplots(
        3, 1, figsize=(7.0, 3.92), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 0.90, 0.90]},
    )
    fig.subplots_adjust(left=0.113, right=0.988, bottom=0.202, top=0.84, hspace=0.31)
    for ax in (sway, position, cable):
        ax.axvspan(-0.25, 0, color="#F8EDE5", zorder=0)
        ax.axvspan(0, 1.53, color="#F1F4F6", zorder=0)
        ax.axvline(-0.25, color="#BE9D8B", linewidth=0.8, zorder=1)
        ax.axvline(0, color=GREY, linewidth=0.8, zorder=1)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    sway.plot(time[history_angle], angle[history_angle], color=TEAL, linewidth=1.55,
              label="Available measurement", zorder=3)
    sway.plot(time[withheld], angle[withheld], color=GREY, linewidth=1.2, linestyle="--",
              label="Withheld encoder angle", zorder=2)
    sway.scatter(time[targets], angle[targets], s=19, facecolors="white",
                 edgecolors=ORANGE, linewidths=1.0, zorder=4, label="Scoring target")
    for ax, signal in [(position, trolley), (cable, length)]:
        ax.plot(time[history_support], signal[history_support],
                color=TEAL, linewidth=1.55, zorder=3)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
    sway.yaxis.set_major_locator(MultipleLocator(10))
    sway.set_ylabel("Sway (°)")
    position.set_ylabel("Trolley (m)")
    cable.set_ylabel("Cable (m)")
    cable.set(xlabel="Time relative to forecast origin (s)", xlim=(-4.37, 1.53),
              xticks=[-4, -3, -2, -1, 0, 0.5, 1, 1.5])

    text_transform = sway.get_xaxis_transform()
    sway.text(-4.31, 1.29, "Record 9   Trolley motion and hoisting",
              transform=text_transform, fontsize=10.5, fontweight="bold")
    sway.annotate("Latest angle", xy=(-0.25, 1.02), xytext=(-0.48, 1.10),
                  xycoords=text_transform, textcoords=text_transform,
                  ha="right", va="bottom", fontsize=8.4,
                  arrowprops={"arrowstyle": "-", "color": GREY, "linewidth": 0.65})
    sway.annotate("Forecast origin", xy=(0, 1.02), xytext=(0.18, 1.10),
                  xycoords=text_transform, textcoords=text_transform,
                  ha="left", va="bottom", fontsize=8.4,
                  arrowprops={"arrowstyle": "-", "color": GREY, "linewidth": 0.65})
    handles, labels = sway.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.56, 0.031),
               ncol=3, columnspacing=1.55, handlelength=2.4)
    save(fig, output, "dated_input_example")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="Repository results directory")
    parser.add_argument("--dataset", required=True, type=Path, help="Processed Inteco dataset.npz")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    validation_selection(args.results, args.output)
    dated_inputs(args.dataset, args.output)


if __name__ == "__main__":
    main()
