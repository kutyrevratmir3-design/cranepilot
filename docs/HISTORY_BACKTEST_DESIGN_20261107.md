# Conditional backtesting inside the same observed history

Freeze this protocol and all new experimental sources before validation scoring.
The failed forced-adaptation study is retained. No new neural/actor training,
no extra history, no current-default change, no APHYNITY changes or publication.

## Causal information and objective

The complete observed input remains41 angle/position samples at10ms, newest at
the current forecast origin. Reverse to chronological order inside the method.
For a candidate frequency/damping, compute the same forced particular solution
and two homogeneous basis columns as the frozen forced observer. Fit coefficients
using only the earliest prefix of21 samples (0.2s) or31 samples (0.3s). Score
prediction of the remaining20 or10 angle samples, conditional on their already
observed trolley positions. No suffix angle enters the prefix coefficient fit.
Prefix state must not use endpoint velocity estimated from all41 samples.

This is conditional history backtesting, not an actuator forecast. Tail trolley
positions are known by the actual forecast origin and are legitimate for causal
parameter estimation, but this score cannot test future actuator-input errors.
Only after choosing frequency/damping, refit the state on all41 observations
using the unchanged forced_fit and original backward21sample endpoint velocity,
then forecast the original30 future steps from the shared frozen actor.

## Fixed variants and controls

Frequency grid [.75,.85,1,1.15,1.25] times nominal; damping[0,.1,.2,.4]/s.
Score = withheld-angle MSE + Q²/41*((ratio-1)/.15)²; no target after origin.
Four strategies: bt21; bt31; bt_multi (equal average of two split MSEs);
bt_guarded (same multi score, adapt only if gain against nominal ratio1,d0
is at least10% of nominal score AND at least Q²/41, with strict improvement).
Thresholds are heuristics, not confidence or hypothesis-test bounds.
The latest10 samples participate in both multi splits; splits are not independent.
All final states refit all41, including fallback. A lower backtest score does
not guarantee improvement at the operational horizons0.1..1.5s.

Three new hybrids use only bt_guarded physics A, previous nominal physics P and
soft network S: A+(S-P), A+g_old*(S-P), A+g_old*(S-A). The old nominal-observer
RMS defines g_old withtau.5. Never use the optimized adaptive residual as trust.
Controls: incumbent, soft_symmetry, previous combined, nominal physics, previous
coarse in-sample adaptive and fine in-sample adaptive. Total13 methods, report all.
The previous candidate computes other controls internally for source reuse;
these are not new choices and remain documented in its completed parent study.

## Selection and acceptance

Same folds, actor/calibration provenance, stride10, seven horizons, equal whole-
scenario weighting and active threshold.005m/s as prior stage. All measured
records have been examined historically: no untouched-test claim. Choose one
method per fold using designated validation only: overall<=incumbent, active
and quiet<=1.01incumbent, minimum overall MAE. Common strategy must satisfy every
fold and minimize mean normalized overall validation MAE; incumbent fallback.
Hash selections before evaluation or new simulation. No revision after scores.

Acceptance of common strategy remains: primary all/active<=soft_symmetry;
transfer all/active<=1.01incumbent; primary loadstress all/active<=1.01incumbent.
Report local selections and every method irrespective of acceptance. A win over
the previously failed adaptive baseline alone is not a useful-model claim.

## Fresh simulation within the existing generator family

Use unchanged forced_adaptive_simulator with new lengths .19/.47/.97, amplitudes
2.5/6.5degrees, phases .8/2.6, frequencyratios .86/1.02/1.22, damping .06/.23/s,
new noise seed9112026. Both effective omega_true²/g and geometric1/L force laws,
zero/nonzero trolley drive, futuregain1/1.15, exact/noisyquantized angles.
1152 cases/fold; three folds have their own calibration-dependent truth. These
are paired computational cases, not independent hardware repetitions or a new
simulator family. Inputs exclude true parameters and future actual state.

Shared prescribed acceleration bypasses actor only for angular simulation.
Measured trolley positions inside observed history are exact in the simulator;
only angle gets noise/quantization. Separate force law, drive amplitude, unknown
futuregain, measurement, length and horizon. Save casep95 and subgroup failures.
The futuregain pair has identical histories and prescribed inputs. This must
produce identical forecasts up to the fixed numerical tolerance.

## Verification and evidence

Unit tests must show suffix angles cannot change a prefix-based prediction;
later x cannot change prefix predictions; chosen parameters never depend on
future acceleration; refit equals frozen forced_fit for selected parameters;
zero-excitation fallback; independent forced solution, reflection and batching.
Verify parent six control forecasts, history-only packet reconstruction, poisoned
target invariance, simulation1ms/.5ms convergence and futuregain pairs, all at
unchanged5e-5degree tolerance. Retain failures under new output paths if necessary.
Log parameter choices, backtest versus full-fit residuals, nominal score, preguard
choices, adaptation acceptance and adjacent switching. Overlapping packets and
multisplit scores are descriptive, not independent replications.
