# Causal local modal identification: preregistration

This stage changes the *task*, openly: residual sway forecasting in settled
drive/length intervals, with four seconds of past measurements. It does not
replace the .4s-history active-input forecast, prove hardware control, or claim
novelty for sinusoidal least squares. The motivation is transfer under payload
and effective-length mismatch: estimate frequency/state from recent history
instead of extrapolating a fixed training-record relation.

Variable projection separates linear amplitudes from nonlinear frequency and
damping parameters. Background: [NIST, Variable Projection for Nonlinear Least
Squares Problems](https://www.nist.gov/publications/variable-projection-nonlinear-least-squares-problems)
and [Askham/Kutz, optimized DMD](https://arxiv.org/abs/1704.02343). Our simple grid
and local refinement are an implementation choice, not a reproduction of either
paper's full algorithm. Adaptive harmonic fitting with .4s history already exists
in the repository; the new question concerns identifiability with 4s history,
offsets, multiple modes, and equal-history high-order ARX comparators.

## Locked data contract

Config is authoritative. Original raw signals at100Hz, unchanged. Inputs are
81 past samples at50ms spacing, newest first, covering4s: target-axis angle,
length and every available position/reference channel. Both methods receive
the same available signals and TRAIN records. No whole-record test mean, future
measured position/length/angle, future frequency fit or future-derived filter.
Forecast horizons .10,.20,.35,.50,.75,1.00,1.50s. Origins every100ms from25s.

Eligibility uses only input information: past4s length range≤2mm, every past
position range≤2mm(trolley) or2mrad(jib), and known future reference constant
through1.5s (range≤1e−9 native units). No future *measured* channel enters this
gate. It labels a settled-input prediction task; it does not prove zero physical
forcing. Report coverage and excluded origins for every scenario; never remove
an eligible window based on a prediction error or amplitude. An empty confirmation
scenario prevents the seven-scenario success claim.

TRAIN/VALIDATION are historical development records: trolley TRAIN1,2,3,5,6,7,8,
10,11,12 / VAL4; jib TRAIN14,15,16,18,19,20 / VAL17; simult TRAIN24,25,26,27,
29,30,31 / VAL28. Theta and beta separately, giving five family/axis cases. Only
fixed-length TRAIN records calibrate nominal omega=sqrt(g/(aL+b)), using existing
LengthCalibration. Every model can use that calibration.

Candidate confirmation IDs9,13,21,22,23,32,33 are sealed. A metadata-only audit
found no saved model training/validation/score use, but the complete public33
records were imported before and unrecorded access cannot be excluded. These
are **previously unscored conditional holdouts**, not newly collected independent
experiments. The five hoisting records are evaluated only when the *past* gate
allows; this is not a forecast of changing future length. Author-provided new
recordings remain necessary for a stronger independence claim. The loader must
open only explicitly allowed NPZ keys; the eager full-dataset loader is forbidden.

## Models and fixed development search

For each case, a strong ARX reference selects among direct or calibrated-harmonic
residual regression, with/without theta-history×omega and×omega² LPV terms;
six ridge penalties,24fits. It uses all81samples of every allowed input channel.
Intercept and training-only standardization; equal training-record weights.
Seven target horizons are independently fitted. Selection only by VAL MAE.

Local autoregression reference: orders2,4,8 and ridge1e−8,1e−4,.01. Fit coefficients
only to the current81sample history, after centering by that history's mean and
scaling by max(std,oneencoderbin). Clip recursive characteristic root magnitudes
to1 (stable projection), then recursively predict. Also include TLS-AR2 with the
same centering and stable projection. These are classical causal comparators,
not TRAIN models. Select local-AR setting per case on VAL only.

Modal candidates: fixed calibrated omega with zero damping; one-mode; two-mode.
For one-mode, project recent history on [1,exp(−d*t/2)cos(wt),exp(−d*t/2)sin(wt)],
t∈[−4,0]. Search89frequency ratios .5..1.6 times train-calibrated omega and
damping[0,.04,.12,.3]/s. Choose lowest past SSE, then a clipped parabola refinement
in frequency at the chosen damping; accept refinement only if past SSE decreases.
No future target selects frequency/damping. Amplitude below oneencoderbin uses
the history mean rather than an unidentifiable oscillatory extrapolation.

Two-mode starts with that one-mode fit, then fits a joint intercept/two-sinusoid
basis. Secondary101ratios .5..3 of nominal, zero secondary damping, separation
≥.2nominal. Accept extra mode only if SSE reduction exceeds
sigma²*(3log81+2log101), sigma²=max(quantum²/12,SSE1/(81−3)). This is a fixed
complexity heuristic, not a calibrated probabilistic confidence test. No search
over sensor noise or penalty on confirmation. Fixed-frequency candidate estimates
offset and state from the same4s and serves as an ablation. Select modal variant
per case only by VAL MAE; retain all outcomes and forecasts.

Score mean MAE in degrees over seven horizons, then equal cases. Development
gate: ≥10% reduction against strongARX, ≥4/5case wins. Require selected modal
also to beat selected localAR in mean before opening holdout. If gate fails,
do not read holdout targets and do not report success. Hash config, source,
selection and saved reference weights before any confirmation access.

If eligible: one confirmation on all7scenarios, no refitting/reselection. Average
axes per scenario, then equal scenarios. Require ≥10%MAE reduction vs strongARX,
≥5/7scenario wins, paired whole-scenario bootstrap upper95%change bound<0,
and no empty scenario. Report localAR comparison, every axis/condition/horizon
and coverage even if unsuccessful. This is a conditional public-dataset result;
it cannot alone establish broad SOTA, fresh-rig independence or closed-loop gain.
Do not modify the active goal to erase these limits.

Commit code/config/tests/design before fitting. Preserve originals, all negatives,
and the main/APHYNITY separation. No new author messages are authorized.
