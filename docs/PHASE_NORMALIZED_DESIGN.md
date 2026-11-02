# Phase-normalized angular model: pre-evaluation design

3 November 2026. This design and configuration are fixed before evaluation of
the current force-aware campaign. A draft is staged under .local while that
campaign's guarded source snapshot is active. The subsequent campaign has now completed; see the outcome below.

The current observer networks consume a fixed duration in seconds and native
command levels. The same 0.4 s packet spans different fractions of a swing
cycle for different cable lengths. Native input patterns can also correlate
with length. The next model instead observes a common one-radian interval of
past oscillator phase, interpolated causally from the existing 41 observations.
No older measurements or future measurements are added.

Its angular encoder sees phase-sampled angles and dimensionless trolley
displacement omega^2*(x-x_origin)/g, plus the forced observer's q and p and
their discrepancies from the raw state. Frequency does not enter the encoder
as a separate length label. Physics still supplies omega for stepping.

An optional recurrent disturbance uses bounded latent states with a positive
leak rate per unit oscillator phase. Its update is a convex combination of
previous bounded state and a bounded learned target, using exp(-lambda*omega*dt).
The extra force enters the oscillator's work balance explicitly. This is not
passivity relative to the real actuator or identification of a real force.

For the angular surrogate, scaling omega by 1/s and dt by s with identical
dimensionless histories, states and acceleration samples preserves the forecast
angles. Unit tests will check that statement independently of experimental
accuracy. It does not apply to the separate native-command actuator. Constant
forces within steps, small-angle physics and fixed cable length remain limits.

Two variants, with and without the learned force port, use three fixed seeds
and the same robust loss and train/validation/evaluation split as the current
campaign. Both will be reported, regardless of which performs better. The
configuration is configs/phase_normalized_campaign.json. The required phase
history is checked explicitly; unsupported lengths must not silently extrapolate
the observation packet. No best-model claim follows from this construction.

## Completed outcome

Six phase-normalized fits and three direct GRU reference fits completed.
The phase variants did not improve the validation or primary extrapolation
result. Their exact angular-similarity property passed independent tests but
did not guarantee better prediction. Both negative variants remain in the
complete report. No further architecture parameters were chosen from their
evaluation errors.
