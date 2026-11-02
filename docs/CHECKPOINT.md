# Checkpoint 3

2 November 2026. Working research, not a completed journal submission.

## What is actually complete

Python package with isolated dependencies, verified import of 33 records,
legacy OLS replication, causal scenario-separated forecasting, physical and
learned models, prefix calibration reference, bounded-input MPC simulation,
tests, generated figures and an English LaTeX research draft.

The full neural development benchmark completed 38 scenario-axis folds and
342 network fits. All-family physical baselines additionally used 3- and
41-sample histories. The simulation-only MPC pilot completed 16 paired
contexts (32 episodes). Twenty-eight scientific/numerical tests passed.
The command-conditioned passive-memory pilot and same-information GP/MLP
references have also completed. See PASSIVE_ROLLOUT.md for the derivation,
information contract, ablations and numerical verification.

## Retained runs

| Run | Status | Scope |
|---|---|---|
| extrapolation_pilot | complete | 11 models, four held-out cable-length scenarios, three neural seeds |
| loso_baselines | complete | 38 scenario-axis folds, two histories, corrected horizon indexing |
| neural_loso | complete | 38 folds, 11 models, 342 neural fits |
| adaptation_pilot | complete | 0, 1, 3, 5, 10-period causal-prefix budgets; common scored tail |
| validated_adaptation | complete | fixed chronological prefix gate, development scenario 4 followed by four evaluation scenarios; all updates abstained |
| control_pilot | complete | synthetic nonlinear plant, two MPC models, exact-state assumption |
| actuator_development | complete | causal native-reference drive model, train 3/5/6, validation 4 |
| passive_rollout_pilot | complete | memory 0/4, three seeds each, frozen causal drive model |
| rollout_references | complete | same-information GP and three-seed MLP residual references |

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

The reference/actual audit and first command-conditioned model are complete.
The learned drive gives 2.43 mm validation MAE at 1 s versus 4.86 mm for
constant velocity, while losing at shorter horizons. The four-memory model
obtains 0.228 degrees at 0.1 s versus 0.245 for action-conditioned physics;
the same-information GP is stronger at 0.216. At 1 s the passive model is
0.765 versus 0.757 for action-conditioned physics. This is a mixed result,
not a successful superiority claim. The memory-free observer ablation is
included. The new learned model has not been evaluated inside MPC.

1. Diagnose causal drive/state-estimation error on development records;
   compare against the retained GP, without retuning on evaluation scenarios.
2. Test whether multiple passive memory modes are identifiable from the
   quantized, short observation packet before increasing model complexity.
3. Extend coupled-axis/hoisting modelling only after a revised frozen protocol.
4. Integrate an independently validated action-conditioned model in MPC;
   add estimated-state, input-shaping and adaptive-physics comparisons.

See RESEARCH_PLAN.md for the detailed decision rules and remaining stages.

## External state

The public repository is https://github.com/kutyrevratmir3-design/cranepilot.
Publication uses the authenticated browser because the connector cannot write
this repository and local Git network access is unavailable. Public web commits
and the retained local Git history are separate; neither history is force-pushed.
Public web commits contain the source, configurations, tests and manuscript
figures. The new campaign, documentation and complete selected metrics are
being synchronized through the same browser workflow.

Overleaf is authenticated. The selected main document is cranepilot_main.tex;
the original article and figures are preserved. The eleven-page PDF including the passive-memory section,
same-information comparisons and sixth figure compiled with zero errors and
was rendered and visually reviewed. The newer innovation campaign is documented
in INNOVATION_RESULTS.md; it is not yet included in that compiled PDF. The compiled manuscript
is retained as results/latest/CranePilot_Research_Draft.pdf. See overleaf_build.json
for the exact verified build state. The earlier checkpoint PDF is historical.

## Commands

```sh
python -m unittest discover -s tests -v
cranepilot benchmark --config configs/extrapolation_pilot.json --output artifacts/runs/extrapolation_pilot
cranepilot benchmark --config configs/neural_loso.json --output artifacts/runs/neural_loso
python scripts/run_adaptation_pilot.py
python scripts/run_validated_adaptation.py
python scripts/run_control_pilot.py --episodes 16
python scripts/run_passive_rollout.py --output artifacts/runs/passive_rollout_pilot
python scripts/run_rollout_references.py
python scripts/report_passive_rollout.py
python scripts/make_report.py --run artifacts/runs/extrapolation_pilot
python scripts/make_checkpoint_pdf.py
```

For historical OLS replication, additionally provide the retained MATLAB CSV
via scripts/reproduce_legacy.py --legacy-results PATH.


## Completed observer/innovation campaign

Nine new fits (three fixed seeds per variant/control) completed after the first
passive-memory pilot. Mean seven-horizon MAE: old PEPM 0.54267 deg; matched
training control 0.52376; refined observer 0.51626; driven innovation 0.51193;
GP reference 0.51668. The tiny aggregate difference from GP is not convincing
evidence of superiority. See INNOVATION_RESULTS.md for exact horizons and limits.
