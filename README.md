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
- Command-conditioned recurrent actuator and phase-exact passive-memory model.

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

The current English manuscript compiles in the linked Overleaf project and
is retained locally as `results/latest/CranePilot_Research_Draft.pdf`.
The chronological adaptation pilot in `docs/VALIDATED_ADAPTATION.md` accepts
no real-record updates; its equality with the physical prior is abstention,
not evidence that adaptation improves forecasting.

## Passive-memory development pilot

PEPM combines a causal initial-state encoder, analytic oscillator rotation,
and a dissipative latent-memory closure. The implementation checks its discrete
storage balance, exact base rotation, independent-solver convergence and
finite-difference gradients. The fixed-length forecast uses prescribed future
reference commands and past measured trolley positions; it never consumes
future measured trolley positions or accelerations.

At 0.1 s the retained four-scenario mean MAE is 0.228 degrees for PEPM,
0.245 for action-conditioned physics, and 0.216 for a same-information GP
residual. At 1 s PEPM does not improve action-conditioned physics. This is
a completed development pilot with a mixed result, not established superiority.
See [the method and evidence](docs/PASSIVE_ROLLOUT.md).

After importing the public data, reproduce this pilot using fresh output paths:

```sh
python scripts/run_passive_rollout.py --actuator-only --output artifacts/runs/actuator_development
python scripts/run_passive_rollout.py --output artifacts/runs/passive_rollout_pilot --actuator-checkpoint artifacts/runs/actuator_development/actuator.pt
python scripts/run_rollout_references.py
python scripts/report_passive_rollout.py
```

Retained runs keep their configurations, environment, predictions, training
logs and completion status. The new innovation campaign also serializes
train-only calibration and guards evaluation with source/configuration hashes. A run is not
considered complete merely because a process started. Metrics include all
declared scenarios; failures must be retained and discussed.

## Observer and driven-innovation campaign

Nine further fits compare a refined causal angle/frequency observer, an explicit
learned forcing port and the old architecture under matching training rules.
Three seeds are averaged per variant; no best evaluation seed is selected.
Mean MAE across seven horizons and four scenarios is 0.51193 deg for the driven
variant versus 0.54267 for original PEPM, 0.52376 for the matched-training control,
and 0.51668 for GP. GP remains stronger at several intermediate horizons;
this small aggregate difference does not establish broad superiority.
The learned forcing is an extra energy port, not identified physical force.

```sh
python scripts/run_innovation_development.py
python scripts/run_innovation_development.py --evaluate
python scripts/run_uniform_control.py
python scripts/report_innovation.py
```

See [the full results](docs/INNOVATION_RESULTS.md). The separately compiled
11-page paper predates this newest campaign. Tests now total 28.

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

