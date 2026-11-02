# Research plan after the first implementation

Updated 2 November 2026. This supersedes hardware-dependent parts of the earlier
proposal. No new physical measurements can be collected.

## Target

A defensible contribution in physics-guided forecasting and simulation-based
control under cable-length/configuration shift, supported by the existing
experimental measurements, strong baselines, and public reproducibility.
No claim of experimentally validating a newly designed controller.

## Completed

- Isolated working Python environment; SciPy, sklearn, CPU PyTorch, matplotlib.
- All 33 XLSX scenarios imported and source checksums verified.
- Historical OLS reproduction: all 24 values, maximum discrepancy about 1e-13 deg.
- Causal, corrected-index forecast benchmark; train-only physical calibration.
- 11 models in the four-length extrapolation experiment.
- 38 scenario/axis LOSO folds, histories of 3 and 41 for physical baselines.
- 38 neural folds and 342 network fits with three fixed seeds.
- Separate paired synthetic MPC pilot: 16 contexts, 32 episodes.
- Causal-prefix adaptation pilot, including its negative result.
- Scientific-contract and numerical tests, generated figures and LaTeX sources.

## Findings that change the research direction

Calibrated physics already solves much of the fixed-length trolley forecasting
problem. The initial projection penalty does not consistently outperform the
unprojected residual or GP baseline. Short causal-prefix ARX calibration is
not yet reliable and can degrade an already good physical prior.

The next contribution must address a measured failure mode: trustworthy local
adaptation under quantization and forcing, or physically action-conditioned
coupled dynamics. More black-box model complexity alone is not justified.

## Next sequence

1. Audit frequency-fit reliability and prior/local-estimator uncertainty using
   training and validation scenarios, keeping evaluation records retrospective.
2. Build a stronger adaptive baseline: constrained nonlinear/integral fitting
   with a causal state observer, prequential validation within the prefix, and
   an explicit rule for refusing an unreliable update.
3. Audit the input path. The data documentation reports references in metres
   and radians, but they are not actuator forces. Identify what can be inferred
   from recorded reference/actual motion before designing command-aware MPC.
4. Separate geometry/frequency mismatch from rotating-base coupling. Compare
   joint-axis physical features against independent angle-only forecasts.
5. Implement action-conditioned rollouts and compare adaptive physics, GP
   residual and the candidate under the same inputs and calibration budget.
6. Extend to the imported hoisting and alternate-load records, clearly noting
   their small number and confounding of mass, shape and trajectories.
7. Broaden simulation to an independently implemented physical object and an
   estimated-state controller. Add LQR and input shaping, fair controller tuning,
   numerical convergence, observed constraint violations and complete failures.
8. Fix a confirmatory simulation protocol and complete the final ablations.
9. Rewrite the paper around the actual contribution; verify all references,
   reproducibility, scope and journal requirements. Obtain human author review.

## Decision rules

Retain negative experiments. Do not select a seed or a scenario after seeing
test results. Do not call the initial 16-context campaign industrial validation.
Do not claim that a useful baseline implementation is a new method.
If a candidate cannot beat adaptive physics, GP residuals or a simpler model
under a fair protocol, reject that candidate and preserve its results.

## External dependencies

Overleaf requires sign-in in the connected in-app browser before remote edits
and renaming can be verified. The local manuscript remains editable meanwhile.
GitHub access is authenticated, but the visible installation currently exposes
only KrafterDen/pazzle; do not put crane research into that unrelated project.
A separate empty cranepilot repository with write access is needed for publication.

