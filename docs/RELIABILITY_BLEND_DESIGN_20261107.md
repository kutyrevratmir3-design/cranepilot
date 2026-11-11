# Causal physics/neural blending — 7 November 2026

Predeclare a second, distinct family while broad_search_20261107 is training
and before either family's real test or reserved-grid results are opened.
The original model, actor, length calibration and three seeds remain frozen.
This family uses no optimizer and does not modify the broad-search sources.

Let r be the RMS residual of the causal forced oscillator observer on the 41
past angle samples. Let Q = 2*pi/4096 rad, the recorded nominal encoder step.
For a fixed scale tau in encoder-step units, define

`g = 1 - exp(-(r / (tau * Q))^2)`

and `forecast = physics + g * (incumbent - physics)`.
Both forecasts use exactly the same past-derived state, calibrated frequency
and frozen predicted acceleration. At zero past discrepancy the full learned
forecast correction vanishes; at large discrepancy the original network is
recovered. This gates the complete output, unlike the angle-only bound in the
training screen. It does not preserve the original latent storage identity.

The six thresholds 0.05/0.1/0.25/0.5/1/2 are fixed before comparison. Include
the unchanged incumbent and pure physical predictor as controls. RMS is a
model discrepancy, not a probability of model validity or a calibrated sensor
uncertainty. Low discrepancy on a short history need not predict good future
physical behavior, especially with commanded forcing, noise or unobserved modes.
Quantization may produce nonzero discrepancy in exact physical motion.

Use the original three folds and original validation records. A candidate is
eligible only when each of overall, active and quiet validation MAE is within
1% of the incumbent. Among eligible candidates, minimize the average error on
the exposed November 6 synthetic parameter grid (all damping/history modes),
generated with that fold's train-only calibration. That grid is now explicitly
development evidence. Incumbent is the fallback. Also lock one common threshold
eligible across all folds, minimizing mean normalized development error.

This is a physics-prioritized screen with a 1% validation tolerance; it is
different from the accuracy-first continuation screen and must be labelled
separately. The tolerance is operational, not a statistical noninferiority claim.
The six-threshold search still reuses the small validation set; no untouched
real-world generalization claim follows.

Commit this config/runner/gate/tests/design before selection. Seal selections
and all source/data/weight hashes before either campaign's held-out evaluation.
Then score all variants on all original test/stress records, report interpolation
separately, and use the same already reserved primary synthetic grid. Both
families are locked before that grid is opened; no changes after confirmation.
Retain every threshold, every seed forecast, gate statistics, and negative results.
The broad training run remains under its own independent frozen protocol.
