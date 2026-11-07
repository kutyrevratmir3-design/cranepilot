# Drive-conditioned modal state estimation: preregistration

Motivation from frozen qualification: old-evaluation drive-active blendMAE
1.7264° vsARX1.8055°, both-active3.1570°vs3.2151°. The large settled-motion
gain does not extend automatically to active operation. A long unforced fit can
also absorb earlier forcing into a poor current state. Test a specific physical
correction before trying another neural architecture.

Related primary work: [Muntwiler et al.2022](https://proceedings.mlr.press/v168/muntwiler22a.html)
uses dynamics and uncertain measurements for moving-horizon state estimation;
[Cao et al.2023](https://doi.org/10.3390/jmse11010004) studies MHE/MPC for
shipboard cranes. [LPV crane identification2026](https://ieeexplore.ieee.org/abstract/document/11631165/)
emphasises state-space structure and held-out load conditions. These motivate
input-aware estimation; our small linear local fit is not their complete
algorithm, a full constrainedMHE implementation, or their experimental result.
No new raw data link was verified for the2026paper; its abstract is not a dataset.

## Locked physical approximation

Same existingTRAIN/VAL/old-evaluation roles and post4sorigins as qualification.
All old recordings are already open; every score is retrospective development.
No active/native-controller success or independent confirmation is claimed.

Take the frozen modal estimator's past-only primary frequency w,damping d and
optional secondary mode. When sway amplitude is unidentifiable use the existing
train-calibrated nominal frequency. Fit current modal amplitudes/intercept with
additional forced-response basis functions from measured past x/J. Effective
length is the same train-calibrated current length; varying-length equations
are not silently introduced.

For a support coordinate u, use transfer −s²/(s²+d*s+lambda), lambda=w²+d²/4.
Implement z''+d*z'+lambda*z=u and output lambda*z+d*z'−u. This is a local forced
oscillator basis for observed support displacement and avoids double-differencing
noisy measured positions. Unknown initial state is absorbed by free modal columns.
For Jdot², x*Jdot² and xdot*Jdot use the same oscillator's force-to-angle response.
These are limited geometry-motivated nuisance channels; their gains are learned
from TRAIN, not assumed to identify physical torque or exact crane geometry.

Integrate each known past input with an exact first-order hold state transition.
Remove input's initial level; fit an intercept and modal initial state. Divide
response by current calibrated effective length. Past derivatives for quadratic
channels use only the supplied history (central differences inside it, backward
differences at its end). No sample beyond current forecast time is read.

Residualize forced and target histories against the same free modal columns.
Fit one TRAIN coupling prior percase with ridge.001 after TRAIN-only channel
scaling, record- and drive-regime-balanced weights. Five priors. Online either
keep this prior(fixed) or fit a ridge correction toward it with penalty1(adapt).
Both consume current past history only. Retain all coefficients and condition
diagnostics. Their identifiability/stability is not guaranteed by this objective.

Two future support assumptions: hold current position, or continue a local
five-sample least-squares velocity. Linear paths clip to TRAIN observed position
min/max extended to include the current position; no instantaneous clamp of an
unseen current value. These are explicit assumptions, not known realised motion.
The mathematical helper can condition on a supplied hypothetical support path;
real-data runners must NEVER supply measured future paths. Future measured L is
never supplied. A real command-conditioned actuator remains a separate problem.

## Stronger matched reference and selection

Refit ARX on the same post4sTRAIN windows, with two representations: angle/L/omega
LPV; or IO-LPV adding full relative support histories, measured-drive nonlinear
channels, past native references and past length variation. Direct angle-change
or residual over the frozen free-modal forecast; six ridge values.24fits percase,
120total. Select percase by VAL balanced score. Fit one3s model inheriting that
choice, and choose3s/4s or the previous frozenARX by VAL, never old-evaluation.
No deliberately weakened settled-only comparator.

All fitting uses the same available4s history and original TRAIN records. Each
TRAINrecord weights its present quiet/drive-active groups equally. Validation
score: equal groups within case, equal fivecases; seven forecast horizons equal.
No L-changing VALrecords are invented. Report every regime separately.

Four physical variants(hold/linear × fixed/adapt), and four causal soft blends
with the strengthened ARX. Each blend predicts the last observed1s from its3s
prefix, using its own assumed support continuation, not the observed support
motion in that held-out past second. ARX backtest uses its frozen3s version.
Same heuristic softweight as before, including encoderquantum floor. Include
strongARX, oldfrozenblend, and oldfreemodal as fallbacks:11global choices.
One global minimum byVAL only, frozen before any old-evaluation replay.

Development gate: balancedMAE≥10%lower than matchedARX,≥4/5casewins, and quiet
meanMAE≤1.05reference. Gate failure is retained; do not switch to a different
variant after seeing old-evaluation scores. The completed out-of-scope
qualification remains unchanged. Existing public APIs/defaults/APHYNITY remain
unchanged. New author logs are still required for the active thread goal.
