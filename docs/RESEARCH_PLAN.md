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
- Causal native-reference drive identification and command-conditioned rollouts.
- Passive-memory model with memory-free ablation, three seeds each.
- Same-information GP and MLP residual references; 25 numerical/scientific tests.

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

## Current method decision

The passive-memory construction has an exact discrete storage balance and a
verified differentiable implementation, but its retained pilot does not beat
the GP residual across horizons. Keep its negative and mixed results. The
reference-to-actual drive audit is complete for the initial fixed-length pilot;
short-horizon drive error and causal state estimation remain development targets.
See PASSIVE_ROLLOUT.md before defining any new experiment. Existing evaluation
records are retrospective, not a fresh confirmatory test.

## External state

GitHub repository: https://github.com/kutyrevratmir3-design/cranepilot.
Overleaf is authenticated and the original manuscript is preserved alongside
the development draft. See CHECKPOINT.md and results/latest/overleaf_build.json
for publication and compilation status.

## Observer/innovation follow-up completed

Nine further fits and a matched-training control are complete; see
INNOVATION_RESULTS.md. Driven innovation helps active long-horizon motion but
does not dominate GP across horizons. Next isolate initial-angle versus frequency
correction on development data and define independent simulation tests before
claiming controller performance. Do not select horizon mixtures on evaluation.

## Completed 3 November stage and next decision

The force-aware robust model leads the retained macro error in the primary
length-extrapolation protocol and two additional training ranges. The stage
completed 36 angular fits, two fresh fold-specific actors, 56 GP candidates
and 41 passing checks. Phase normalization and measurement-bin anchoring
did not improve the primary model. See RESEARCH_2026_11_03.md and MODEL_CARD.md.

The next priority is evidence breadth, not another unbounded hyperparameter
search on the same four primary records: reproduce released LPV/forecasting
methods, add a matched temporal-convolution reference, and fix an independent
simulation/public-data protocol before selecting a final method. Keep all
previous baselines, interpolation/extrapolation separation, seed averaging
and original train-only actuator/calibration boundaries. The alternate-load
stress group and the active-origin 1.5 s subset still have different winners.
Do not call this a global state-of-the-art model or a validated learned MPC.

The current stage stays local; no GitHub publication is authorized.

## Completed 6 November stage and next decision

The matched TCN comparison and 36-fit temporal/objective ablation are complete.
One validation-selected model improves long-transfer macro MAE by 3.03%, but
worsens its active-window error. The short-transfer selection regresses 1.49%;
the primary incumbent remains the default. Do not replace it using an
evaluation-selected winner. All tested alternatives and 49 passing checks are
retained in RESEARCH_2026_11_06.md.

The independently implemented, fixed free-sway diagnostic revealed that learned
corrections can spoil a simple physical limit. Next investigate a residual
parameterization/regularizer that preserves that limit, while protecting active
motion performance. Define fresh confirmation cases before any further tuning;
the completed cleanroom set is now development evidence.

The separate Aalto industrial data have been acquired and causally reconstructed.
Their exported 4 Hz grid often associates an angle with a subsequent raw event;
the new importer instead uses only logged events available at the grid time.
Before training, resolve angle provenance, suspension geometry and native
command semantics; fix whole-cycle/payload splits and a native-rate information
contract. Do not upsample to claim 100 Hz observations or merge these results
with the Inteco benchmark. See AALTO_DATA_AUDIT.md.

ReSMILE was inspected but not implemented, and no new learned-controller or
hardware result has been established. Publication remains local only.
