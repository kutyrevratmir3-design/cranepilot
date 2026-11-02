# Checkpoint 1

2 November 2026. Working research, not a completed journal submission.

## What is actually complete

Python package with isolated dependencies, verified import of 33 records,
legacy OLS replication, causal scenario-separated forecasting, physical and
learned models, prefix calibration reference, bounded-input MPC simulation,
tests, generated figures and an English LaTeX research draft.

The full neural development benchmark completed 38 scenario-axis folds and
342 network fits. All-family physical baselines additionally used 3- and
41-sample histories. The simulation-only MPC pilot completed 16 paired
contexts (32 episodes). Thirteen scientific/numerical tests passed.

## Retained runs

| Run | Status | Scope |
|---|---|---|
| extrapolation_pilot | complete | 11 models, four held-out cable-length scenarios, three neural seeds |
| loso_baselines | complete | 38 scenario-axis folds, two histories, corrected horizon indexing |
| neural_loso | complete | 38 folds, 11 models, 342 neural fits |
| adaptation_pilot | complete | 0, 1, 3, 5, 10-period causal-prefix budgets; common scored tail |
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

The MPC simulation mean residual RMS is 0.08050 deg for nominal and 0.06394 deg
for calibrated MPC in the specified 16 synthetic contexts. This does not show
neural MPC performance or hardware validation. The manuscript states this.

## Immediate next work

1. Diagnose prefix-fit uncertainty and identifiability on TRAIN/VALIDATION data.
2. Build a constrained nonlinear/integral adaptive-physics baseline with
   prequential prefix validation and rejection of unreliable updates.
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

Overleaf's supplied edit link opens a sign-in page in the connected in-app
browser. The page is left open for the user's sign-in. The remote title and
files have NOT been changed. The source paper/main.tex and supporting files
are ready locally and will be synced once the session is authenticated.
The built-in local LaTeX compiler is unavailable in this session; the LaTeX
sources were checked for includes, citation keys and balanced braces, but have
not been successfully compiled. The separate three-page checkpoint PDF is a
visually inspected reportlab report, not the compiled LaTeX manuscript.

## Commands

```sh
python -m unittest discover -s tests -v
cranepilot benchmark --config configs/extrapolation_pilot.json --output artifacts/runs/extrapolation_pilot
cranepilot benchmark --config configs/neural_loso.json --output artifacts/runs/neural_loso
python scripts/run_adaptation_pilot.py
python scripts/run_control_pilot.py --episodes 16
python scripts/make_report.py --run artifacts/runs/extrapolation_pilot
python scripts/make_checkpoint_pdf.py
```

For historical OLS replication, additionally provide the retained MATLAB CSV
via scripts/reproduce_legacy.py --legacy-results PATH.

