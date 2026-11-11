# Fixed combination and structural-mismatch falsification

Freeze before evaluation: existing soft_symmetry checkpoints for every fold,
unchanged causal discrepancy gate tau=0.5. No fitting, search or post-score repair.
This follows exposed results; all measured-record scores are retrospective.
The current default and APHYNITY remain unchanged.

Compare incumbent, soft_symmetry, physical, incumbent_gate, combined, fixed_half
(50/50 soft/physical). Same actors, causal 41-sample histories, commands and
train-only calibrations. Fixed_half separates gate value from ordinary averaging.
Use original validation/evaluation/load-stress records, stride10, active past
velocity >=0.005m/s. Equal scenario then equal horizon averages; exclude
interpolation records from cross-length headlines. Report all/active/quiet,
all seven horizons, all individual scenarios, and seed forecasts.

The earlier free-sway generator used the same calibrated frequency for truth and
forecast. This new grid explicitly changes that assumption. Five new lengths,
three amplitudes (up to12degrees), three phases, five conditions: nominal,
frequency0.85x, frequency1.15x, damping0.2/s, unknown future trolley acceleration
0.15m/s² at t=[0.25,0.5)s. Truth is nonlinear sin(theta), includes cos(theta)
in forcing; predictor sees neither truth frequency nor disturbance. Nominal and
hidden-impulse histories must be identical and produce identical forecasts.
Failure on an unannounced future disturbance is an information limitation, not
evidence that a different causal model can predict this hidden event.

Four measurement modes: exact; encoder quantization; noise std0.25encoderbin
then quantization; fixed bias0.5bin then quantization. Targets remain clean.
225 trajectories ×4 histories =900 cases per fold. Frequency truth is perturbed
relative to each fold's own training calibration: the three fold simulations
are distinct and must not be described as a common-target fold comparison.
Report 20 condition×measurement groups per fold, case p95 over seven-horizon MAE,
length/horizon breakdowns, gate distributions, worst groups. Compare paired
method differences; synthetic cases are not hardware repetitions.

Autonomous simulation supplies zero prescribed acceleration to *all* angular
models, bypassing the frozen actor. It is not an end-to-end actuator test.
An additional autonomous-only baseline fits damped sin/cos over past history:
frequency ratios [.75,.85,1,1.15,1.25], damping[0,.1,.2,.4], residual MSE plus
Q²/41*((ratio-1)/.15)². No intercept. The penalty is fixed before scoring;
this is a simple locally specified comparator, not a published-method replication.
It receives the same 41 samples and calibration; cannot see future truth.

Integration at1ms must agree with0.5ms within0.00005degrees including forcing;
use interval-consistent handling of discontinuity. Scientific unit tests,
parent-checkpoint replay (tolerance0.00005degrees), history-only equivalence,
target-invariance and hidden-disturbance input identity precede interpretation.
Commit code/config/design/tests before evaluation; save guarded hashes and
predictions. A computational failure is retained, fixed under a new commit and
new output directory with unchanged selection/acceptance thresholds.

Interpret a broad successful tradeoff only if combined all/active MAE <=incumbent
in primary, <=1.01incumbent in both transfer splits, AND each of20 simulation
groups per fold <=incumbent. Report these gates separately and show parent and
adaptive-physics comparisons even if incumbent is beaten. These fixed operational
thresholds are not statistical noninferiority bounds. Do not promote by pooled
means, retune after seeing new stress, claim passivity for the blended output,
or call this fresh hardware/SOTA evidence. Any failed subgroup remains visible.


## Numerical revision before new simulation results

First attempt `combined_stress_20261107` stopped on original-model replay for
primary record2: compiled float32 flow differed from retained original flow by
0.000052934degrees, above unchanged0.00005degree tolerance. Preserve failed run.
For the revised run use the original InnovationForecaster implementation for
incumbent and its gated comparator; soft_symmetry and combined stay unchanged.
No metrics or simulation scores were interpreted to choose this revision.
Hidden-input forecast equality uses the same numerical tolerance because paired
histories can cross differently sized BLAS chunks; input histories remain
bitwise equal. Parent reference files are now verified against parent hashes.
All thresholds, weights, simulator parameters and candidate remain unchanged.
