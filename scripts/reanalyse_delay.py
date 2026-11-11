"""Reaggregate frozen Inteco predictions for the delayed-angle manuscript.

Input errors have already been averaged over forecast origins and then
over axes within each physical record. This script gives all seven
horizons and all contributing physical records equal weight.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SOURCE = Path("dense_horizons_20261109/paired_record_horizons.csv")
RECOMBINED = SOURCE.with_name("recombined_age_summary.csv")
MODELS = {
    "selected": "blend",
    "parent_quad": "completion",
    "bank_simplex_1": "bank",
    "dense_matched_arx": "dense_arx",
    "matched_arx": "earlier_arx",
}
METHODS = list(MODELS.values())
AGE_NOISE = ["age_ms", "noise_quantum"]
AGES = [0, 20, 30, 50, 70, 100, 150, 200, 250]
HORIZONS = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50]
RECORDS = [9, 13, 21, 22, 23, 32, 33]
HOISTING = {9, 13, 21, 32, 33}
LARGE_GAIN_RECORDS = {13, 32, 33}
N_BOOTSTRAP = 20_000
SEED = 20261111
INK = "#242C32"
TEAL = "#006B7D"
ORANGE = "#A74724"
GREY = "#616B75"
BLUE = "#375C88"

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
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def load_records(results):
    keys = [*AGE_NOISE, "scenario", "horizon_s"]
    source = pd.read_csv(results / SOURCE, usecols=[*keys, *MODELS]).rename(columns=MODELS)
    source["horizon_s"] = source["horizon_s"].round(2)
    expected = pd.MultiIndex.from_product([AGES, [0, 1], RECORDS, HORIZONS], names=keys)
    actual = pd.MultiIndex.from_frame(source[keys])
    if actual.has_duplicates or not actual.sort_values().equals(expected):
        raise ValueError("Expected one row per age, noise level, record and horizon")
    if not np.isfinite(source[METHODS]).all().all():
        raise ValueError("All frozen scores must be finite")
    zero = source.loc[source.age_ms == 0, ["blend", "completion", "bank"]].to_numpy()
    if not np.all(zero == zero[:, :1]):
        raise ValueError("The zero-age fallback no longer matches exactly")
    return source.sort_values(keys).reset_index(drop=True)


def equal_record_tables(source, results):
    age_records = source.groupby([*AGE_NOISE, "scenario"], as_index=False)[METHODS].mean()
    age_records["hoisting_in_record"] = age_records["scenario"].isin(HOISTING)
    age = age_records.groupby(AGE_NOISE, as_index=False)[METHODS].mean()
    frozen_age = pd.read_csv(results / RECOMBINED)
    frozen_age = frozen_age.pivot(index=AGE_NOISE, columns="method", values="mae_deg")
    frozen_age = frozen_age.rename(columns=MODELS)[METHODS]
    np.testing.assert_allclose(age.set_index(AGE_NOISE), frozen_age, atol=1e-12, rtol=0)
    return age_records, age


def paired_stats(age_records):
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(RECORDS), size=(N_BOOTSTRAP, len(RECORDS)))
    rows = []
    for (age, noise), group in age_records.groupby(AGE_NOISE):
        group = group.sort_values("scenario")
        for comparator in METHODS[1:]:
            delta = (group["blend"] - group[comparator]).to_numpy()
            low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
            rows.append({
                "age_ms": age,
                "noise_quantum": noise,
                "comparator": comparator,
                "n_records": len(delta),
                "blend_mean_mae_deg": group["blend"].mean(),
                "comparator_mean_mae_deg": group[comparator].mean(),
                "blend_minus_comparator_mean_deg": delta.mean(),
                "blend_minus_comparator_median_deg": np.median(delta),
                "bootstrap_percentile_2_5_deg": low,
                "bootstrap_percentile_97_5_deg": high,
                "blend_better_records": np.count_nonzero(delta < 0),
                "tied_records": np.count_nonzero(delta == 0),
            })
    return pd.DataFrame(rows)


def horizon_stats(source):
    keys = [*AGE_NOISE, "horizon_s"]
    stats = source.groupby(keys)[METHODS].mean()
    for comparator in METHODS[1:]:
        delta = source["blend"] - source[comparator]
        stats[f"blend_minus_{comparator}_deg"] = stats["blend"] - stats[comparator]
        stats[f"blend_better_than_{comparator}_records"] = (
            source.assign(win=delta < 0).groupby(keys)["win"].sum()
        )
    return stats.reset_index()


def earlier_arx_contributions(age_records):
    summary = []
    for (age, noise), group in age_records.groupby(AGE_NOISE):
        gains = group["earlier_arx"] - group["blend"]
        large = group["scenario"].isin(LARGE_GAIN_RECORDS)
        hoist = group["hoisting_in_record"]
        mean_all = gains.mean()
        summary.append({
            "age_ms": age,
            "noise_quantum": noise,
            "earlier_arx_minus_blend_mean_deg": mean_all,
            "earlier_arx_minus_blend_median_deg": gains.median(),
            "n_hoisting_records": int(hoist.sum()),
            "earlier_arx_minus_blend_mean_hoisting_deg": gains[hoist].mean(),
            "n_non_hoisting_records": int((~hoist).sum()),
            "earlier_arx_minus_blend_mean_non_hoisting_deg": gains[~hoist].mean(),
            "earlier_arx_minus_blend_mean_13_32_33_deg": gains[large].mean(),
            "earlier_arx_minus_blend_mean_other_four_deg": gains[~large].mean(),
            "fraction_net_gain_from_13_32_33": gains[large].sum() / gains.sum() if mean_all else None,
        })
    return pd.DataFrame(summary)


def line(ax, ages, scores, label, color, style, marker):
    ax.plot(ages, scores, label=label, color=color, ls=style, lw=1.5,
            marker=marker, ms=4, markerfacecolor=color if label == "Blend" else "white",
            zorder=4)


def age_axis(ax):
    ax.set(xlim=(-8, 258), xticks=[0, 50, 100, 150, 200, 250], xlabel="Angle age (ms)")
    ax.grid(axis="y", color="#DFE4E7", lw=0.45)
    ax.set_axisbelow(True)


def save(fig, output, name):
    fig.savefig(output / name, metadata={"Creator": "CranePilot delayed-angle reanalysis",
                                          "CreationDate": None})
    plt.close(fig)


def compensation_figure(age, comparisons, output):
    clean = age[age.noise_quantum == 0]
    fig, (absolute, relative) = plt.subplots(1, 2, figsize=(7.0, 3.28))
    fig.subplots_adjust(left=.092, right=.973, bottom=.17, top=.74, wspace=.36)
    pos = clean[clean.age_ms > 0]
    for method, label, color, style, marker in [
        ("blend", "Blend", TEAL, "-", "o"),
        ("completion", "Quadratic completion", ORANGE, "--", "s"),
        ("bank", "Direct bank", GREY, ":", "^"),
    ]:
        line(absolute, pos.age_ms, pos[method], label, color, style, marker)
    absolute.scatter(0, clean.iloc[0].blend, color=INK, s=24, marker="D", zorder=5,
                     label="Synchronous fallback")
    absolute.set_title("A  Error of each method", loc="left", pad=8)
    absolute.set(ylabel="Mean absolute error (°)", ylim=(.43, 1.55))
    age_axis(absolute)
    handles, labels = absolute.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(.5,.98),
               columnspacing=1.2, handlelength=2.3)

    subset = comparisons[(comparisons.noise_quantum == 0) &
                         (comparisons.age_ms.isin([20, 50, 100, 150, 200, 250]))]
    relative.axhline(0, lw=.85, color=GREY, zorder=2)
    for method, color, marker, shift in [
        ("completion", ORANGE, "s", -3),
        ("bank", GREY, "^", 3),
    ]:
        rows = subset[subset.comparator == method]
        value = rows.blend_minus_comparator_mean_deg
        bounds = np.stack([value - rows.bootstrap_percentile_2_5_deg,
                           rows.bootstrap_percentile_97_5_deg - value])
        relative.errorbar(rows.age_ms + shift, value, yerr=bounds, fmt=marker,
                          color=color, markerfacecolor="white", ms=4, capsize=2,
                          elinewidth=.85, linestyle="none", zorder=4)
    relative.set_title("B  Paired change from each component", loc="left", pad=8)
    relative.set(ylabel="Blend MAE minus component MAE (°)",
                 ylim=(-1.53, .23), yticks=[-1.5, -1, -.5, 0])
    age_axis(relative)
    save(fig, output, "delay_compensation.pdf")


def arx_figure(age, age_records, output):
    clean = age[age.noise_quantum == 0]
    fig, (absolute, records) = plt.subplots(1, 2, figsize=(7.0, 3.28),
                                            gridspec_kw={"width_ratios": [1, 1.11]})
    fig.subplots_adjust(left=.09, right=.972, bottom=.18, top=.75, wspace=.44)
    for method, label, color, style, marker in [
        ("blend", "Blend", TEAL, "-", "o"),
        ("earlier_arx", "Earlier ARX", BLUE, "--", "s"),
        ("dense_arx", "Dense ARX", ORANGE, ":", "^"),
    ]:
        line(absolute, clean.age_ms, clean[method], label, color, style, marker)
    absolute.set_title("A  ARX references by angle age", loc="left", pad=8)
    absolute.set(ylabel="Mean absolute error (°)", ylim=(.43, 1.42))
    age_axis(absolute)
    handles, labels = absolute.get_legend_handles_labels()
    absolute.legend(handles, labels, loc="lower center", bbox_to_anchor=(.50, 1.18),
                    ncol=3, columnspacing=.85, handlelength=2.0)

    longest = age_records[(age_records.age_ms == 250) & (age_records.noise_quantum == 0)]
    scores = longest.earlier_arx - longest.blend
    ys = np.arange(len(longest))
    records.axvline(0, lw=.85, color=GREY, zorder=2)
    records.hlines(ys, np.minimum(0, scores), np.maximum(0, scores), color=BLUE,
                   lw=1.15, zorder=3)
    for present, fill in [(True, BLUE), (False, "white")]:
        mask = longest.hoisting_in_record.to_numpy() == present
        records.scatter(scores.to_numpy()[mask], ys[mask], s=30, color=fill,
                        edgecolors=BLUE, linewidths=1.1, zorder=4)
    records.set_title("B  Paired difference at 250 ms", loc="left", pad=8)
    records.set(yticks=ys, yticklabels=longest.scenario, ylim=(6.52, -.52),
                xlim=(-.43, 2.08), xticks=[0, .5, 1, 1.5, 2],
                xlabel="Earlier ARX MAE minus blend MAE (°)", ylabel="Record ID")
    records.grid(axis="y", color="#DFE4E7", lw=0.45)
    records.set_axisbelow(True)
    dots = [
        Line2D([0], [0], marker="o", ms=4.7, linestyle="none", markeredgecolor=BLUE,
               markerfacecolor=fill, label=label)
        for fill, label in [(BLUE, "Hoisting in record"), ("white", "No hoisting")]
    ]
    records.legend(handles=dots, loc="lower center", bbox_to_anchor=(.50, 1.18),
                   ncol=2, columnspacing=.85, handlelength=.8)
    save(fig, output, "arx_reference_replay.pdf")


def write_outputs(source, age_records, age, comparisons, output):
    strata = age_records.groupby([*AGE_NOISE, "hoisting_in_record"])[METHODS].mean()
    strata["n_records"] = age_records.groupby([*AGE_NOISE, "hoisting_in_record"]).size()
    for name, table in {
        "record_age_mae.csv": age_records,
        "age_mae.csv": age,
        "paired_age_differences.csv": comparisons,
        "age_horizon_mae.csv": horizon_stats(source),
        "hoisting_strata_mae.csv": strata.reset_index(),
        "earlier_arx_gain_concentration.csv": earlier_arx_contributions(age_records),
    }.items():
        table.to_csv(output / name, index=False, float_format="%.12g")

    settings = {
        "source": SOURCE.as_posix(),
        "source_cross_check": RECOMBINED.as_posix(),
        "all_origins": True,
        "zero_age_bank_bypassed": True,
        "age_selection_max_ms": 200,
        "bootstrap": {
            "replicates": N_BOOTSTRAP,
            "seed": SEED,
            "resample_unit": "paired physical record",
            "percentiles": [2.5, 97.5],
            "interpretation": "Descriptive sensitivity for seven reused records; not confirmatory inference.",
        },
        "hoisting_record_ids": sorted(HOISTING),
        "post_hoc_grouped_gain_record_ids": sorted(LARGE_GAIN_RECORDS),
    }
    (output / "analysis_metadata.json").write_text(json.dumps(settings, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = load_records(args.results)
    age_records, age = equal_record_tables(source, args.results)
    comparisons = paired_stats(age_records)
    write_outputs(source, age_records, age, comparisons, args.output)
    compensation_figure(age, comparisons, args.output)
    arx_figure(age, age_records, args.output)
    print("Verified", len(source), "paired horizon rows and", len(age_records), "paired record-age rows")


if __name__ == "__main__":
    main()
