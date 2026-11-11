# Frozen predictors on off-grid mismatch — 7 November 2026

This protocol is written after inspection of `combined_stress_20261107_replayfix`.
The adaptive autonomous physical baseline strongly improved the grid-aligned
frequency mismatch cases in that run. Those true ratios coincided with entries
in its fixed frequency grid. The present post-hoc hypothesis test asks whether
its improvement survives true frequencies and damping between its grid nodes.
It is new simulation evidence for that hypothesis, not an originally reserved
confirmation dataset, independent hardware data, or a new untouched real test.

## Fixed methods and inputs

Use the six existing predictors from `CombinedCandidate` without modification:
incumbent, soft_symmetry, incumbent_gate, combined, fixed_half and physical.
Add the existing `adaptive_free_forecast` with its unchanged frequency ratios
0.75/0.85/1/1.15/1.25, damping coefficients 0/0.1/0.2/0.4 and penalty. No new
method or hyperparameter is selected. Its causal per-packet least-squares
state/grid estimation is part of the frozen baseline, not a campaign-level fit.
No optimizer, weight training, recalibration or gate change is permitted.

Use all three existing fold calibrations and their corresponding model weights.
The same nominal length and 41 measured past angles are available to each
predictor. True frequency, damping, noiseless future and simulation condition
labels are never input to a predictor. Trolley and command histories are zero;
all methods use zero future acceleration in this autonomous diagnostic.
Adaptive physics is not extrapolated to the forced measured-data benchmark.

## Newly fixed simulation cases

Lengths 0.17/0.41/0.83/1.19 m, amplitudes 2/8 degrees, phases 1.1/2.9 rad,
true-to-nominal frequency ratios 0.79/0.93/1.08/1.21, damping 0.07/0.27 per
second. All four frequency ratios and both damping coefficients lie off the
fixed baseline grid. There is no future forcing in this test.

The Cartesian product gives 128 clean trajectories per fold, with four
observation modes: exact, quantized, noisy quantized (0.25 encoder bins standard
deviation), and biased quantized (0.5 bin bias). Fixed noise seed 7112027 gives
512 observation cases per fold. Noise is paired across dynamics conditions for
the same length/amplitude/phase. Truth is the existing nonlinear simulator;
changing parameter values does not create an independent simulator family.

Integrate from -0.4 to +1.5 s with RK4 step 0.001 s. Repeat at 0.0005 s and require
maximum target disagreement no greater than 0.00005 degrees, without changing
this tolerance in response to scores. The half-step check assesses integration,
not model accuracy. Targets are clean actual angles and history is newest first.

## Evidence, comparison and stopping

Before producing any output, verify the completed parent's guarded inputs and
all result hashes, and require this configuration, design, runner and existing
combined-source/tests to match the local Git commit. Save the parent provenance
SHA-256, source/weight hashes, environment, timings, predictions, metadata,
per-packet adaptive choices and gate weights. Failed execution retains a failed
provenance and does not overwrite prior outputs.

Report errors at every case/horizon for all seven methods and all three folds.
Group by fold, condition, observation mode and method. Mean, median and p95
refer to each case's mean absolute error across the seven existing horizons;
individual correlated horizons are not counted as independent observations.
Retain the full table even if the off-grid result reverses the previous finding.
There is no score-based rerun, selection, threshold adjustment or model promotion.

The common simulator grid evaluated under three trained calibrations does not
supply three independent physical experiments. Results support only sensitivity
to these specified structural and observation mismatches. No GitHub/Overleaf
publication and no modification of the separate APHYNITY experiment are allowed.
