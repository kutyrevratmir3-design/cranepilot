# Known sensor age: dated state fit, head completion and matched ARX

Preregistered after the prior50ms stress scores were visible. All33recordings
already opened; every result is retrospective/synthetic qualification. We do not
estimate an unknown delay from future targets or claim new hardware validation.

Motivation: [Bar-Shalom etal2004](https://ieeexplore.ieee.org/document/1292140/)
addresses incorporation of older measurements into a current state estimate.
[Dubois/Bertrand/Eudes2018](https://w3.onera.fr/copernic/sites/default/files/2018_icstcc.pdf)
studies a linear moving-horizon estimator for delayed multi-rate sensors. These
motivate timestamp-aware estimation, not a reproduction of their estimators,
MSE guarantees or controller stability theorems.

## Information contract and fair origins

Selected angle history is401actual past samples ending at current_time−age;
L/support/native-reference histories are401samples ending now, all100Hz.
Age is known from metadata, integer0..15samples, and supplied to every method,
including the stronger ARX. Unknown age is unsupported. In free sway
A*cos(w*(t−d)+phi) reveals phi−w*d, not phi and d separately: one cannot infer
the absolute phase/delay from that angle history alone without extra information.

Same oldTRAIN/VAL/evaluation roles. Origins4.25s onward, stride1s, so every
age0..150ms has genuine recorded prehistory. Never invent/pad older sensor data.
Seven unchanged target horizons .1..1.5s from the current, not sensor, time.
Regimes come only from current-clock past401samples of x/J/L. No amplitude/error
exclusions. This differs from the prior .25s-origin/paddedlag stress grid;
compare only freshly matched scores within this study.

TRAIN/VAL ages0,20,50,100ms, clean or independent uniform angle jitter±encoderq.
Old evaluation additionally30,70,150ms (not used for selection). Common seeded
perturbations for all methods; target remains original current-clockfutureangle.
Noisy observations overlap across origins but are packetwise perturbations,
not a coherent newly measured sensor process or independent trials.

## Causal adapters and missing observations

Move already observed delayed angle samples to their actual current-clock
positions. Only the newest age samples are missing. Fill those by hold; linear
least squares over .10/.25s; quadratic over .25s; free-modal3s; or forced3s
fixed/adaptive state projection. Age0 is exact identity for every adapter.

Modal parameters come from the frozen frequency/damping search using last3s
of the delayed angle and nominal length at that sensor time. Forced fits align
past support/L with the same measurement timestamps, residualize against modal
initial-state columns and use unchanged TRAIN priors. Past state integration
uses exact FOH at50ms; propagate across the known historical sensor gap at10ms
using support positions actually measured by current time. Nonlinear nuisance
force channels are held at their last sensor-clock level across the short gap.
Effective frequency/length are held over≤150ms; no variable-length theorem.
No support sample after current time is read. No new physical prior is fitted.

Run the frozen four-expert bank on the aligned angle. In causal weighting,
exclude prefix-validation heads whose target timestamps are later than the
latest real angle observation: completed head values are predictions, not truth.
Recompute both soft priors and simplex weights using only valid heads. Keep
six heads at age0; five at positive supported ages. Quiet routing remains
legacy over the same valid evidence, not the old mixed-clock noisy score.
The unaware comparator deliberately ignores age and remains unchanged.

## Stronger reference and selection

Refit120ARX candidates: angle/ioLPV × direct/delayed-modal-residual ×6ridge,
fivecases. SameTRAINrecordings/conditions/origins; weights equal records and
quiet/drive groups, all age/noise combinations equally represented. Add known
age, phase sin(w*age),cos(w*age)−1 and their interactions with relative angle
histories. All feature scaling TRAIN-only; no trialID/noise label as a feature.
Delayed-modal-residual baseline forecasts h+age from the dated angle, not h.
Save all fits/scores, including negative results.

Per-case VAL chooses the best among all120search outputs appropriate to that
case and both frozen ARXs under every matched adapter (including unaware).
This avoids attributing access to timestamps solely to our method. The
reference is fixed globally across validation ages/noises within eachcase.

One global adapter chosen by equal-five-case validation MAE, equal age/noise
conditions, quiet/drive groups equal within eachcondition/case. All eight are
retained. Gate additionally requires≥20%gain vsunaware at positive ages,
≥3%gain vshold at positive ages,≥5%gain vsmatchedARX and≥4/5ARXcasewins.
Failure is retained without reselection. Freeze numerical code/config before
fitting, weights/selection before old evaluation.

On old evaluation score selected, unaware, timestamp-hold, selected matchedARX,
selected augmentedARX, and dated free-modal, at every declared age/noise. Keep
all losses/regimes/horizons/coverage. Do not search all adapters on old scores.
Retain a separate reusable bundle only as experimental, without changing any
previous default/API/bundle. Verify raw-to-forecast replay, age0identity,
missing-head masking, original1281files, CPUlatency and zero-input bias.
New suitable physical records are still required for the independent goal.
