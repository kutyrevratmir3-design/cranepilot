# Observer and driven-innovation development results

2 November 2026. Nine additional fits completed, three fixed seeds per variant.
All comparisons use existing retrospective records; no new physical trials.

## What changed

The previous model fixed its initial angle to the latest quantized sample and
used a fixed train-calibrated frequency. A new causal observer can adjust the
angle by up to 0.02 rad and the effective frequency by up to 15 percent. Both
adjustments depend only on the same 41-sample past packet. The actual fitted
validation frequency ratios stay approximately within 0.9788--1.0282.

The driven variant adds a 12-state GRU that predicts a bounded dimensionless
forcing correction at every 50 ms rollout step. It consumes predicted angular
state, prescribed command, frozen actuator acceleration and frequency. It never
consumes future measured sway or trolley position. Its correction enters the
analytic oscillator as an explicit extra energy port. The passive core keeps
its numerical storage balance with the total supplied input; the full driven
model is not claimed passive relative to the physical actuator alone. Learned
forcing is not an identified force or a hardware safety result.

The observer variant has 7728 parameters; the driven variant has 8821. Both
retain four passive memory states. The original drive checkpoint is frozen.

## Training and evaluation contract

TRAIN scenarios 3/5/6; validation 4; evaluation 1/2/7/8. All three seeds 0/1/2
are averaged, without evaluation-seed selection. Training uses uniform 0.25 s
origins, maximum 100 epochs, AdamW at 0.001 and patience 15. MSE trains all
30 rollout steps; early stopping uses validation MAE at the seven fixed
horizons. There is no evaluation-based horizon blending or threshold search.

The original architecture is also refit with the new sampling, training and
selection rules. That matched-training control is necessary: gains over the
old run combine architectural and training-protocol changes. It retains the
same latent-memory penalty because its innovation penalty is identically zero.

The primary runner completes and saves validation before a separate evaluation
command is permitted. It checks configuration/source hashes and the frozen
actuator checkpoint. The calibration, environment, logs, weights, predictions,
diagnostics and complete metrics are retained locally. The control is a
separate three-seed run; no variant is omitted from the report.

## Complete mean scenario MAE, degrees

| Model | 0.1 s | 0.2 s | 0.35 s | 0.5 s | 0.75 s | 1 s | 1.5 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Action-conditioned physics | .24480 | .37000 | .47796 | .54582 | .76220 | .75723 | .72596 |
| Original PEPM | .22812 | .33268 | .47501 | .54943 | .71604 | .76539 | .73203 |
| Matched-training PEPM | .22422 | .31017 | .44276 | .53406 | .70108 | .74644 | .70757 |
| Refined observer | .27076 | .33903 | .44440 | .50593 | .68929 | .70451 | .65992 |
| Observer + driven innovation | .21551 | .31377 | .46696 | .53061 | .66245 | .72362 | .67058 |
| Same-information GP | .21553 | .29071 | .43393 | .50119 | .64625 | .78495 | .74424 |

Mean across all seven declared horizons: original PEPM 0.54267 degrees;
matched control 0.52376; observer 0.51626; driven 0.51193; GP 0.51668.
The driven variant improves the old PEPM mean by 5.67 percent, but the
matched-training control already improves it by 3.49 percent. Thus the
architecture's incremental advantage over that control is about 2.26 percent.
These are descriptive changes, not statistically established population gains.

The 0.1 s difference between driven and GP is effectively a tie at the shown
precision. GP is stronger at 0.2--0.75 s overall; the new variants improve the
longer horizons. The observer-only model gives the lowest overall 1--1.5 s
MAE among these models but worsens 0.1 s. Driven innovation improves active
long-horizon motion relative to the observer-only variant, so their benefits
are regime dependent. It beats GP in only 13 of the 28 scenario/horizon pairs.
The small aggregate advantage over GP does not establish general superiority.

In the pre-existing active-at-origin diagnostic at 1.5 s, mean scenario MAE is
2.20940 degrees for original PEPM, 2.29415 for matched-training PEPM, 2.08983
for the observer, 1.64148 for driven innovation and 2.28850 for GP. The driven
variant's reduction from original PEPM is 25.7 percent in this specific subset.
Active means the backward-only trolley velocity fit exceeds 0.005 m/s at the
origin; it does not label the whole future horizon. These are subsets of the
same four physical records, not independent experiments or controller trials.

This campaign therefore supports causal state/forcing refinement as a useful
direction, without proving that greater complexity always helps. A follow-up
must isolate angle versus frequency correction and verify robustness on an
independently defined simulation protocol, rather than tuning on evaluation.
The model has not been placed inside MPC. The compiled 11-page paper predates
this campaign; this document and CSVs are the current record of these results.

## Verification and reproduction

All 28 tests pass. Added tests verify that changing future measured targets
does not change a forecast, that nonzero learned forcing participates in the
storage balance with finite gradients, and that frequency corrections remain
bounded. These numerical properties do not establish predictive superiority.

```sh
python scripts/run_innovation_development.py
python scripts/run_innovation_development.py --evaluate
python scripts/run_uniform_control.py
python scripts/report_innovation.py
```

The first and third commands refuse existing output directories. Full evidence
is under artifacts/runs/innovation_development and uniform_passive_control.
Versioned summaries, training logs and provenance are under results/latest/
innovation_* and uniform_control_*. The report includes every scenario,
all/active/quiet errors, MAE and RMSE. The comparison plot is retained under
paper/figures/innovation_comparison.png and .svg.
