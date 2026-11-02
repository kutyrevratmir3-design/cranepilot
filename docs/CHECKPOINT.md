# Checkpoint 2

2 November 2026. Working research, not a completed journal submission.

## What is actually complete

Python package with isolated dependencies, verified import of 33 records,
legacy OLS replication, causal scenario-separated forecasting, physical and
learned models, prefix calibration reference, bounded-input MPC simulation,
tests, generated figures and an English LaTeX research draft.

The full neural development benchmark completed 38 scenario-axis folds and
342 network fits. All-family physical baselines additionally used 3- and
41-sample histories. The simulation-only MPC pilot completed 16 paired
contexts (32 episodes). Seventeen scientific/numerical tests passed.

## Retained runs

| Run | Status | Scope |
|---|---|---|
| extrapolation_pilot | complete | 11 models, four held-out cable-length scenarios, three neural seeds |
| loso_baselines | complete | 38 scenario-axis folds, two histories, corrected horizon indexing |
| neural_loso | complete | 38 folds, 11 models, 342 neural fits |
| adaptation_pilot | complete | 0, 1, 3, 5, 10-period causal-prefix budgets; common scored tail |
| validated_adaptation | complete | fixed chronological prefix gate, development scenario 4 followed by four evaluation scenarios; all updates abstained |
| control_pilot | complete | synthetic nonlinear plant, two MPC models, exact-state assumption |

Raw predictions, model checkpoints and full local run provenance are retained
under artifacts/runs. Selected metrics are versioned under results/latest.
The release archive contains all source/configuration files, processed public
data, run metadata and metrics, plus selected initial prediction/episode traces.
Large all-family prediction arrays are reproducible locally and excluded from
the compact handoff archive.

## Main findings

In the first extrapolation split at 1 s, calibrated physics obtains MAE 0.7578
deg and projected NN residual 0.9074 deg. At 0.1 s, the same scores are 0.2999
and 0.2193 deg. There is no established across-horizon victory of the candidate.
GP residuals are strong comparators on transverse motion in all-family LOSO.
The simple causal-prefix adaptation method worsens the mean score; retain it
as a negative reference and improve estimator reliability before advancing.

The additional forecast-oriented prefix gate keeps zero-shot performance
(1 s MAE 0.672523 deg at every budget) by accepting no updates. Of the 16
nonzero-budget evaluation cases, nine have insufficient prefix data and seven
fail validation. The chronological gate is implemented and tested, but this
result is abstention, not adaptation success. Its acceptance thresholds were
not retuned after seeing the pilot. See docs/VALIDATED_ADAPTATION.md.

The MPC simulation mean residual RMS is 0.08050 deg for nominal and 0.06394 deg
for calibrated MPC in the specified 16 synthetic contexts. This does not show
neural MPC performance or hardware validation. The manuscript states this.

## Immediate next work

1. Audit reference-to-actual actuator dynamics on TRAIN/VALIDATION records;
   prefix-only harmonic adaptation currently has no demonstrated benefit.
2. Diagnose forcing, parameter identifiability and prefix regime changes before
   adding more adaptive-model complexity. Keep the conservative gate as a reference.
3. Audit reference-to-actual actuator dynamics before action-conditioned learning.
4. Extend coupled-axis/hoisting modelling while retaining the physical prior.
5. Integrate a validated action-conditioned model in MPC; add estimated-state,
   input-shaping and adaptive-physics comparisons on independent simulation.

See RESEARCH_PLAN.md for the detailed decision rules and remaining stages.

## External state

Local Git repository is initialized. GitHub authentication is working for
kutyrevratmir3-design, but the visible app installation currently exposes only
KrafterDen/pazzle. No crane files were put into that unrelated repository.
The available connector has no create-repository operation. A distinct empty
cranepilot repository with write access is required to publish this project.

Overleaf is authenticated in the in-app browser. The project is named
CranePilot - Physics-Guided Prediction and Simulation-Based Control.
cranepilot_main.tex, cranepilot_refs.bib and five cp_ figures have been uploaded;
cranepilot_main.tex is now the selected main document. The original article
and its figures are preserved. Initial compilation succeeded with zero errors
and one microtype/showhyphens warning. A layout review identified floats
preceding their section heading; figures and tables now retain source order
using fixed placement in this development draft.
The updated manuscript additionally records the all-abstention adaptation pilot.
The final nine-page PDF has been rendered and visually reviewed, with all five
figures present, resolved citations, zero compile errors and one non-content
microtype warning. The downloaded compiled manuscript is
results/latest/CranePilot_Research_Draft.pdf.
The earlier three-page checkpoint PDF remains a separate reportlab summary.
GitHub creation via browser currently requires the user's sign-in; its tab is
left open. No GitHub repository has been created or published yet.

## Commands

```sh
python -m unittest discover -s tests -v
cranepilot benchmark --config configs/extrapolation_pilot.json --output artifacts/runs/extrapolation_pilot
cranepilot benchmark --config configs/neural_loso.json --output artifacts/runs/neural_loso
python scripts/run_adaptation_pilot.py
python scripts/run_validated_adaptation.py
python scripts/run_control_pilot.py --episodes 16
python scripts/make_report.py --run artifacts/runs/extrapolation_pilot
python scripts/make_checkpoint_pdf.py
```

For historical OLS replication, additionally provide the retained MATLAB CSV
via scripts/reproduce_legacy.py --legacy-results PATH.

