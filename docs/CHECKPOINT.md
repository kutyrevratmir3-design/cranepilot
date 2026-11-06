# Checkpoint 5

6 November 2026. Working research, not a completed journal submission.

## Latest completed stage

`temporal_20261106` completed 36 further angular fits (four variants, three
seeds, three folds), using verified actors/calibration from identical retained
folds. Validation selection was locked before evaluation. Primary keeps the
incumbent (0.50050 deg). Short transfer selects dense/horizon (0.43858 versus
0.43213 incumbent, a regression). Long transfer selects temporal/horizon
(0.49712 versus 0.51264, a 3.03% macro improvement), but active-origin error
in that fold increases. All models, predictions and selection scores remain.

The fixed 576-case angular-only nonlinear free-sway diagnostic also completed.
It identifies a weakness in the incumbent's corrections: undamped exact-history
error 0.23573 deg versus physical 0.01213. No model was tuned on this diagnostic.
RK4 step-halving difference is 1.36e-9 deg. Forty-nine tests pass. History-only
inference matches retained evaluation traces within 1.63e-5 deg across all
18 model/fold comparisons (three packets each).

The separate Aalto archive DOI 10.5281/zenodo.6907569 was downloaded, verified
and audited: 31,303 exported rows, eight industrial cycles. An alignment audit
found many exported angles matching subsequent raw events. A new causal raw
event reconstruction retains ten signals, missing values and log ages. No
models were trained or scored on Aalto. Geometry, angle provenance and command
semantics still require resolution. See AALTO_DATA_AUDIT.md.

Current evidence: RESEARCH_2026_11_06.md, results/latest/*20261106*,
artifacts/runs/temporal_20261106 and the new evidence archive. The pretraining
source/config/design commit is 678b754. All work remains LOCAL ONLY, with no
GitHub or Overleaf changes. Earlier experiment details below are historical.

## What is actually complete

Python package with isolated dependencies, verified import of 33 records,
legacy OLS replication, causal scenario-separated forecasting, physical and
learned models, prefix calibration reference, bounded-input MPC simulation,
tests, generated figures and an English LaTeX research draft.

The full neural development benchmark completed 38 scenario-axis folds and
342 network fits. All-family physical baselines additionally used 3- and
41-sample histories. The simulation-only MPC pilot completed 16 paired
contexts (32 episodes). Forty-one scientific/numerical tests passed.
The command-conditioned passive-memory pilot and same-information GP/MLP
references have also completed. See PASSIVE_ROLLOUT.md for the derivation,
information contract, ablations and numerical verification.

The further 3 November campaign completed 36 angular fits, two fold-specific
actors and 56 GP candidate fits. The selected force-aware robust NN obtains
0.50050 deg primary macro MAE versus 0.51193 for the previous driven model.
Two additional extrapolation-only scores are 0.43213 (shorter) and 0.51264
(longer). Same-fold GRU scores are 0.44140 and 0.57330. See MODEL_CARD.md
and RESEARCH_2026_11_03.md; all methods and limitations remain in the tables.

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

1. Test residual constraints that preserve simple physical limits, with a newly
   frozen follow-up protocol rather than tuning on the completed cleanroom cases.
2. Make validation robust across motion regimes; long-transfer macro gains
   currently hide active-window regressions, and one validation record can
   select the worse short-transfer variant.
3. Resolve the Aalto measurement/geometry/command contract, lock cycle/payload
   holdouts and use the causal reconstruction at its native rate. Reproduce
   additional identification methods; ReSMILE has only been inspected.
4. Only then integrate an action-conditioned learned model in MPC and compare
   it with estimated-state physical control and input shaping.

The causal forced observer and robust loss help in the retained primary score.
Phase-normalized and measurement-bin-anchored models did not improve it.
The earlier driven NN remains stronger in the active-origin 1.5 s subset.
Warm CPU flow compilation reduces inference time without changing weights.
No new hardware control result is established.

## External state

The public repository is https://github.com/kutyrevratmir3-design/cranepilot.
Publication uses the authenticated browser because the connector cannot write
this repository and local Git network access is unavailable. Public web commits
and the retained local Git history are separate; neither history is force-pushed.
Public web commits contain the previous source, configurations, tests, metrics
and manuscript figures. GitHub CI passed for that previously published state.
The 3 November campaign is LOCAL ONLY under the user's publication preference;
no new GitHub writes were performed. The current tests pass locally.

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

## New retained runs

- forceaware_campaign: failed before training; technical cause recorded.
- forceaware_campaign_v2: complete, 12 neural fits and 16 GP fits.
- phase_normalized_campaign: complete, six phase-model fits plus three GRU fits.
- measurement_consistent_campaign: complete, three nominal-bin-anchor fits.
- cross_length_campaign: complete, 12 angular fits, two actors and 16 GP fits.
- zero_prior_gp_campaign: complete, 24 GP fits with a physical zero-residual prior.

The compact research_20261103_evidence.zip retains neural weights, actors,
prediction traces, configurations, source, tests, metrics and a content manifest.
Large GP joblib checkpoints remain local and are reproducible from scripts.
