# Delayed-measurement references for the Inteco replay

This post-hoc comparison reuses the published 100 Hz Inteco records, the existing record roles and the dated-angle packet grid. The settings were chosen on the three existing validation records before computing the new reference predictions on the seven previously examined replay records. Each family–axis is fitted and selected independently. The result is retrospective.

## Timestamp-aware Kalman reference

For one sway axis, the latent state contains swing about a local equilibrium, angular velocity and a drifting angle offset. The observation is the sum of swing and offset. A small-angle oscillator supplies the time-varying state equation:

    δ̇ = v
    v̇ = −(g/ℓeff) δ − [d + clip(2 ℓ̇eff/ℓeff, −2, 2)] v + u
    ḃ = 0
    y = δ + b + measurement noise

The effective length is a positive affine calibration of measured length. The calibration fits dominant frequencies on the fixed-length TRAIN tails (35–90 s). The deterministic drive term is a TRAIN-only ridge fit to the smoothed angular-acceleration residual. It uses measured support acceleration and applicable rotation/coupling terms, scaled by effective length. At inference the support derivatives are backward 0.20 s quadratic fits. Native command-reference channels are not used.

The discrete filter uses a second-order Taylor state transition, white angular-acceleration process noise, a small random walk on the offset, and a fixed measurement variance of one encoder increment squared divided by 12. The filter receives the known angle age and does not receive the injected jitter label. It starts with zero swing and rate and initializes the offset to the oldest measured angle. Angle corrections end at the last received angle's acquisition timestamp, t − age. The filter then propagates the estimated state through the measured length and support history until t. For the first few angle observations that predate the oldest support sample, the model holds that oldest measured length and sets drive and length rate to zero. Beyond t, length stays at its last measured value, and forcing is either zero or exponentially decays from its last estimated value with a 100 ms time constant. Both future options use only information available at t.

This is a linear time-varying delayed-observation Kalman reference for the benchmark's ordered, uniform-age packets. It filters in acquisition order and propagates across the known gap, which is sufficient for this observation pattern. It is not an implementation of general out-of-sequence measurement fusion or an exact reproduction of a named estimator from the literature.

The frozen design covers 18 candidates per case (3 damping values × 3 process-noise scales × 2 forcing continuations). The fixed seed, six validation ages, both jitter levels and balanced validation objective match the original protocol. The five selected settings are held across replay ages. All cases selected damping 0.1/s and acceleration-process standard deviation 0.03 rad/s^(3/2). The two transverse-angle cases selected decaying forcing and the three longitudinal-angle cases selected zero forcing. An exploratory pilot used the injected jitter variance to set measurement noise. The final protocol removes that information to match the other estimators. Repeating selection on the existing validation records retained the same settings.

## Other references

- Local AR uses the same 2.5 s angle span as the modal bank, sampled at 20 Hz. Each prediction fits a local centered ridge AR model. Validation selects the order and ridge from a 4/8-lag grid; any unstable pole radius is shrunk to at most 0.999. Forecasts are evaluated at age plus horizon, with linear interpolation between the model's forecast steps.
- The calibrated oscillator fits offset, sine and cosine coefficients on the same 2.5 s span. Its frequency uses TRAIN-calibrated effective length at the last angle sample. It forecasts at age plus horizon with that frequency held.
- Persistence repeats the last available angle. Zero predicts zero at all horizons.

## Reproduce

Install the repository dependencies and run the commands below from the repository root. Unpack the 34 XLSX files from [Inteco data v2](https://doi.org/10.17632/x69hyyv3zw.2) into INTECO_XLSX. Use a new directory for NEW_RESULTS. The development stage hashes the protocol, code, configuration and data before saving the validation scores and selection. Replay validates those hashes before scoring the frozen references.

    python -m cranepilot.cli import-data --source INTECO_XLSX --output data/processed
    python scripts/delayed_baselines.py check
    python scripts/delayed_baselines.py develop --repo . --dataset data/processed/dataset.npz --out NEW_RESULTS
    python scripts/delayed_baselines.py replay --repo . --dataset data/processed/dataset.npz --out NEW_RESULTS
    python scripts/verify_archived_targets.py --repo . --dataset data/processed/dataset.npz --out NEW_RESULTS/archived_target_check.json

Replay matches the archived origin indices and past-motion regime labels exactly for all 216 record–axis–age–noise packets. The target-timestamp check recomputes the archived dense-ARX horizon MAEs using the same 142,128 encoder target and forecast pairs. The largest MAE discrepancy is 6.22 × 10⁻¹⁵ degrees. The summary uses the manuscript's aggregation order: mean over origins, equal axes within each record, equal physical records, and equal horizons for the age table. The implementation checks against a standard scalar-measurement Joseph-form Kalman update, includes a stationary gap test, checks stationary local AR, and checks that edits to measurements beyond either timestamp do not change the corresponding forecast inputs.

The saved results include the design lock, validation scores, selected settings, age and horizon summaries, and the target check. A full replay also generates replay_metrics.csv with the per-axis and regime aggregates. Its checksum is recorded in replay_manifest.json.

## Sources

- Dataset by Alhassan et al. (2025). [Data in Brief](https://doi.org/10.1016/j.dib.2025.111822) and [Inteco data v2](https://doi.org/10.17632/x69hyyv3zw.2).
- Delayed-measurement estimation by Larsen et al. (1998). [Incorporation of Time Delayed Measurements in a Discrete-time Kalman Filter](https://doi.org/10.1109/CDC.1998.761918).
- Related crane estimator by Kaneko et al. (2009). [Anti Sway Crane Control Based on Sway Angle Observer with Sensor-Delay Correction](https://doi.org/10.1541/ieejias.129.555).
