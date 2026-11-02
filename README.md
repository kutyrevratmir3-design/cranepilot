# CranePilot

**Physics-Guided Prediction and Control under Cable-Length Shift**

Research by Ratmir Kutyrev, extending his previous experimental payload-sway
forecasting study. This repository separates measured-data forecasting from
simulation-only controller evaluation. It is an evolving research implementation,
not a finished industrial controller or an accepted journal paper.

## Research question

Can a physically calibrated oscillator and a structured learned correction
transfer across cable lengths more efficiently than a black-box predictor?
Does any improvement survive strong physical baselines and closed-loop tests?

The projection-based model is an exploratory candidate, with explicit
predecessors in orthogonal Gaussian processes and physics-guided neural
feedforward control. A claim of methodological novelty requires further
literature comparison and evidence; it is not established by this README.

## Data and scope

Experiments use the existing 33-scenario, 100 Hz Inteco tower-crane dataset:
[Mendeley version 2](https://doi.org/10.17632/x69hyyv3zw.2), CC BY 4.0.
See [the dataset article](https://doi.org/10.1016/j.dib.2025.111822).
The raw XLSX records are never edited. Import records their SHA-256 checksums.

The dataset has already informed this research and previous audits. Its splits
are retrospective, not an untouched prospective test. Nearby time windows are
not independent trials. Existing logs cannot experimentally validate a newly
designed controller. No new hardware data collection is assumed or required.

## Setup

```sh
python -m venv .venv
# Activate the environment using your operating system's normal command.
python -m pip install -e .
```

The working machine can also use compatible cached wheels in an isolated
environment. Those machine-specific caches and paths are excluded from Git.
The checked-in dependency snapshot records the actually used package versions.

## Reproduce the first experiment

Download and extract all dataset XLSX files, including the time file.

```sh
cranepilot import-data --source /path/to/extracted/dataset --output data/processed
python -m unittest discover -s tests -v
cranepilot benchmark --config configs/extrapolation_pilot.json --output artifacts/runs/extrapolation_pilot
python scripts/make_report.py --run artifacts/runs/extrapolation_pilot
```

The pilot trains on trolley scenarios 3, 5, and 6; validates on scenario 4;
and evaluates separately on scenarios 1, 2, 7, and 8. Whole scenarios remain
disjoint. Hyperparameters in this pilot are fixed development choices, not a
claim of optimal or nested-cross-validated performance.

Newest measurement is `theta[k]`; horizon `H` means exactly `theta[k+H]` at
`H*0.01` seconds. Every method uses the declared finite observation history.
Frequency/length calibration uses training records only. Evaluation of the
pilot uses a 0.02 s origin stride; it is not described as every-sample scoring.

## Models

- Persistence, ordinary least squares, and a length-varying linear baseline.
- Ideal and train-calibrated harmonic oscillator forecasts.
- Python gradient-boosted trees and a three-seed MLP ensemble.
- Capped-size fixed-kernel GP residual reference.
- Phase-coordinate neural residual and its sensitivity-projection ablation.

The oscillator is a finite-history harmonic forecasting baseline, not a full
nonlinear command-conditioned plant model. Its frequency is estimated from a
training calibration tail. That tail is not automatically called free decay.
The active/quiet diagnostic is based on backward differences at the forecast
origin and does not assert that the future horizon remains quiet.

Python neural/tree implementations are new fits with a documented common
protocol, not bit-identical MATLAB fits. Legacy OLS reproduction is checked
separately. Saved historical MATLAB NN errors must not be mixed into new
same-protocol model comparisons.

## Research artifacts

- `src/cranepilot`: import, physics, models, metrics, benchmark, simulation.
- `configs`: frozen, inspectable scenario splits and model choices.
- `tests`: causality, horizon indexing, analytic physics and numerical checks.
- `docs`: experimental contract, data audit, research checkpoint.
- `paper`: English LaTeX manuscript and generated result tables/figures.
- `results`: selected reproducible summaries, distinct from raw run archives.

Each run keeps its configuration hash, input manifest hash, environment,
calibration, predictions, training logs and completion status. A run is not
considered complete merely because a process started. Metrics include all
declared scenarios; failures must be retained and discussed.

## Claims and limitations

Do not report the earlier aspirational 72%, 28%, or 41% improvements as results.
Do not call offline forecasting a hardware controller experiment. Do not infer
industrial safety or full-scale similarity from a laboratory pendulum.
Performance comparisons belong to the exact retained run and information set.

## Citation and authorship

See `CITATION.cff` and the manuscript bibliography. Cite the original data
creators as well as any reused method or software. The original research and
its experimental analysis belong to Ratmir Kutyrev; this project builds on that
work. AI-assisted software and manuscript preparation will be disclosed in the
paper, with human review and responsibility for the final submission.

