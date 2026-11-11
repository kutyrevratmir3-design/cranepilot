# Offline identification and direct delay-lift research

Freeze new code, tests, this design and configuration in a local commit before
fitting or validation. This is a substantial family change after repeated failure
of packet-wise dynamics adaptation, motivated by the two companion literature
reviews. It is not an exact reproduction of WENDy, G3P-SR, SMILE or Koopman.

## Three experimental families

1. Global weak-form physical identification: learn only c, kscale and bscale on
complete TRAIN records, independently controlling damping, restoring stiffness
and acceleration coupling. Equation theta'' + c theta' + kscale omega(L)^2 theta
= -bscale omega(L)^2/g x''. Fixed compact supports1/2/4s, phi=(1-u^2)^4;
integration by parts removes pointwise theta'/theta''/x''. Normalize rows by
the L2 norm of phi'', equal record-duration weights, global target RMS, ridge
toward(0,1,1) at0/.01/.1. Bounds c[0,2], kscale[.5,1.5], bscale[.2,2].
No EIV/IRLS weighting: this is a weak-form least-squares adaptation, not WENDy.
At deployment only the two current state coefficients use the41 observations;
the three dynamic parameters remain fixed. The original actor supplies forcing.

2. Direct multihorizon LPV ridge: global mapping from delay features to the
30-step residual against nominal forced physics. Nine lags(0,5,...40) of theta,
relative trolley position and reference, current velocity and refined oscillator
q,p plus30 future input values make60 dynamic features v. Use(v, v*l, v*l^2),
l=omega/8-1. Offline shared coefficients replace per-packet parameter estimation.

3. Direct quadratic delay lift: same60 variables, v and v*l plus1950 total
features including unique pair products divided by sqrt(60). No intercept or
standalone length feature; zero dynamic input has zero correction. This is a
direct polynomial predictor, not a recursively propagated Koopman operator.
No stability guarantee beyond the finite1.5s evaluation horizon.

For each direct family compare frozen-actor acceleration versus prescribed
future commands as the30 input values. Both retain the same nominal physical
baseline, so the command variant bypasses the actor only in its learned residual,
not completely. This is a controlled feature comparison, not causal proof that
the actor is the bottleneck. Future actual positions/angles never enter features.
The complete prescribed command plan is known at origin; direct horizons can
depend on that complete plan, with no additional prefix-causality guarantee.

Train-only RMS feature scaling without centering; equal total weight per TRAIN
record. Fit all30 angular residual outputs simultaneously with a ridge SVD path.
Objective mean_record(mean_window(mean_30(error^2))) + alpha||W||_F^2;
alpha .0001/.01/1. Exactly12 direct and3 weak fits per fold, deterministic.
Three folds mean45 fitted models total, plus4 retained controls per fold.
Train stride25, evaluation stride10; no augmented validation or current-test
parameter adaptation. No stochastic seeds for deterministic ridge fitting.

## Fixed comparisons and selection

Controls: incumbent, soft symmetry, combined tau.5, nominal physical. Same
TRAIN/validation/evaluation/load IDs as frozen temporal_20261106 source,
same train-only length calibration, same frozen actor and original weights.
Deployment remains41samples/.4s past,30steps at50ms, same seven headline horizons,
active threshold .005m/s. This stage fits new regression weights only.

Per-fold validation: all MAE<=incumbent, active/quiet<=1.01 incumbent; among
eligible choose minimum all MAE. Common strategy must be eligible on all folds
and minimize mean fold-normalized all MAE. Always include incumbent fallback.
Equal whole-scenario weighting; no claims treating overlapping windows as
independent. Freeze selection and all checkpoint hashes before evaluation.
All real records have historical exposure; results are retrospective.

For adoption, common choice must match/beat soft symmetry primary all/active,
and be within1% of incumbent for transfer extrapolation and primary loadstress
all/active. As in the retained reports, the headline evaluation uses lengths
outside each fold's training range, for all three folds. Report interpolation
separately, all methods and any regression.
No default model replacement based on a local or isolated metric improvement.

## Synthetic diagnostic and verification

Fresh within-generator forced grid with lengths.21/.53/1.03, amplitudes3/8deg,
phases.6/2.5, frequency ratios.88/1.06/1.24, damping.04/.19, zero/nonzero drive,
two force laws, futureacceleration gain1/1.15, exact/noisyquantized angles,
noise seed10112026.1152cases/fold. Only angular methods with actual supplied
prescribed acceleration can be scored: exclude the six command variants since
this simulator has no actuator/reference model and uses zero placeholders.
This limitation is explicit; simulation cannot validate those residual models.
All simulation is after selection, no acceptance tuning on it.

Verify1ms versus.5ms integration, hidden futuregain pair invariance, all four
parent controls, independent history-only packet construction, poisoned-target
invariance, saved fit reload and train-only normalization. Unit tests include
weak parameter recovery and prediction against known dynamics, independent
coupling, delay-lift linear recovery, translation and zero-input behavior.
Retain failed runs; preserve old source/data/weights and separate APHYNITY.

