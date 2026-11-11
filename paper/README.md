# Crane Payload-Sway Forecasting with Delayed Angle Measurements

Ratmir Kutyrev · Revised manuscript for review · 11 November 2026

[Read the PDF](CranePilot_Manuscript.pdf) · [LaTeX source](main.tex) · [Earlier GitHub version](https://github.com/kutyrevratmir3-design/cranepilot/tree/fcea3efa59aeb603c8723ef7d851e798be566c6a/paper)

This revision compares the blend directly with quadratic completion and the direct modal bank. It reports both archived ARX references, paired record-level differences and a separate retrospective comparison with timestamp-aware Kalman and local autoregressive references. All results use previously inspected laboratory recordings.

## Build

From the repository root, run:

    cd paper
    latexmk -pdf main.tex

The checked-in PDF is the export of the same manuscript compiled in Overleaf. The source in this directory splits the preamble and body into separate files for easier review.

## Figures and results

| Material | Source |
| --- | --- |
| Inteco rig photograph | [Alhassan et al., 2025, Fig. 2](https://doi.org/10.1016/j.dib.2025.111822), CC BY-NC 4.0 |
| Dated input example and validation candidates | [Figure script](../scripts/make_selection_figures.py) and [Inteco data v2](https://doi.org/10.17632/x69hyyv3zw.2) |
| Incremental delay comparison and both ARX references | [Paired reanalysis and figure recipe](../results/delayed_angle_review_20261111/README.md) |
| Timestamp sensitivity | [Common-age averages](../results/timestamp_offset_20261109/common_interior_offset_means.csv) and [250 ms short-horizon results](../results/timestamp_offset_report_20261109/short_horizon_at_250ms.csv) |
| Kalman, local AR and simple baselines | [Configuration, validation scores and replay](../results/delayed_baselines_20261111/README.md) |

For the measured input and validation figures, use the processed Inteco file from the [dataset import](../src/cranepilot/data.py):

    python scripts/make_selection_figures.py --results results --dataset data/processed/dataset.npz --output paper/figures

The [prospective protocol](../docs/FRESH_CONFIRMATION_20261108.md) documents requirements for genuinely new physical recordings.
