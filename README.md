# CranePilot

Forecasting tower-crane payload sway from 0.10 to 1.50 seconds ahead when angle measurements arrive after support measurements. The repository contains the manuscript, frozen predictors, and records of the experiments.

[Read the manuscript (PDF)](paper/CranePilot_Manuscript.pdf) · [Source and figure data](paper/README.md) · [Research index](docs/README.md)

## What the replay shows

Angle age is the time from acquisition of the latest available sway-angle measurement to the forecast origin. The blend combines a forecast based on quadratic completion with forecasts from a bank of modal experts. At zero age, all three use the original synchronous predictor. Its error is already lower than that of both autoregressive references with exogenous inputs (ARX).

| Angle age and score | Blend | Quadratic completion | Direct bank | Earlier ARX | Dense ARX |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 ms, seven horizons | 0.525° | 0.525° | 0.525° | 1.074° | 1.071° |
| 100 ms, seven horizons | 0.566° | 0.667° | 0.626° | 1.077° | 1.183° |
| 250 ms, seven horizons | 0.742° | 1.479° | 0.692° | 1.246° | 1.316° |
| 250 ms, 100 ms horizon | 0.700° | 1.733° | 0.583° | 0.666° | 0.653° |

Values are mean absolute angle errors with no added noise, weighting each of the seven recordings equally. The blend has lower mean error than quadratic completion at every tested positive age. The bank alone has lower mean error at 200 and 250 ms. The earlier ARX has lower mean error than the dense ARX at the tested ages from 50 to 250 ms. At 250 ms, the blend improves on the earlier ARX on three hoisting recordings (13, 32 and 33). On the other four recordings, the mean error increases.

The blend was selected on three validation recordings, one per motion family, with ages through 200 ms. The Kalman filter and other added references were evaluated retrospectively using the same seven previously examined Inteco recordings. Angle age was simulated. Independent validation on new physical recordings is pending.

[Paired reanalysis](results/delayed_angle_review_20261111/README.md) · [Delayed-measurement baselines](results/delayed_baselines_20261111/README.md) · [Confirmation protocol](docs/FRESH_CONFIRMATION_20261108.md)

## Run the checks

The repository's continuous integration uses Python 3.12 and CPU PyTorch. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/delayed_baselines.py check
```

The [forecast API](docs/DATED_BUNDLE_20261108.md) takes a known measurement age and four seconds of history per channel. Its [frozen forecast bundle](artifacts/dated_bundle_20261108) and [dense ARX bundle](artifacts/dense_reference_bundle_20261108) are included. The inference API and unit tests can run without downloading the source recordings.

## Repository layout

| Path | Contents |
| --- | --- |
| [`paper/`](paper) | LaTeX, bibliography, and manuscript figures |
| [`src/cranepilot/`](src/cranepilot) | Forecasting models, data import, and benchmarks |
| [`configs/`](configs) | Experiment settings and record-level splits |
| [`artifacts/`](artifacts) | Frozen model bundles |
| [`results/`](results) | Saved predictions, metrics, and verification reports |
| [`docs/`](docs/README.md) | Evaluation protocols and research reports |

## Data and citation

The source is the [33-record Inteco tower-crane dataset, version 2](https://doi.org/10.17632/x69hyyv3zw.2) (CC BY 4.0). See the [dataset paper](https://doi.org/10.1016/j.dib.2025.111822) for the experimental setup and the [research index](docs/README.md) for the forecast evaluation protocols. Cite the dataset authors and [CranePilot](CITATION.cff) when reusing this work. The code is available under the [MIT license](LICENSE).
