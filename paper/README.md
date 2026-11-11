# Crane Payload-Sway Forecasting with Delayed Angle Measurements

Ratmir Kutyrev · Manuscript for review · 11 November 2026

[Read the PDF](CranePilot_Manuscript.pdf) · [LaTeX source](main.tex) · [Earlier manuscript source](https://github.com/kutyrevratmir3-design/cranepilot/tree/f3107fe70e7b68856eb6233b4da463b3857f44cf/paper)

This revision adds the dated-angle forecasting method, the expanded ARX comparison and the November 9 timing analysis. It reports the short-horizon loss at 250 ms angle age alongside the average improvement over seven horizons. The study replays existing laboratory records. Independent physical validation of the dated forecaster is pending.

## Build

Run the following from the repository root, or open this directory as an Overleaf project with main.tex as the main document.

    cd paper
    latexmk -pdf main.tex

The four vector PDFs are included. To regenerate them from the saved CSVs after installing the repository dependencies, run this command from the repository root.

    python scripts/make_manuscript_figures.py

| Figure | Inputs |
| --- | --- |
| Error by angle age and drive activity | [Dense ARX replay summary](../results/dense_reference_20261108/summary.csv) |
| Error by forecast horizon | [Paired horizon summary](../results/dense_horizons_20261109/paired_horizon_summary.csv) |
| Difference in error across ages and horizons | [Paired horizon summary](../results/dense_horizons_20261109/paired_horizon_summary.csv) |
| Sensitivity to the reported angle age | [Common-age averages](../results/timestamp_offset_20261109/common_interior_offset_means.csv) and [250 ms short-horizon results](../results/timestamp_offset_report_20261109/short_horizon_at_250ms.csv) |

See the [research notes](../docs/README.md) for the fitting, replay and independent-confirmation protocols.
