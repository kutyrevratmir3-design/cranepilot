# Local research checkpoint: force-aware robust forecasting

Updated 6 November 2026. Primary default: `forced_driven_huber`, three fixed seeds.
This stage has not been published. Code retains the project's MIT license;
the original experimental dataset requires its CC BY 4.0 attribution.

The latest temporal/objective campaign did not replace this primary default.
The separately trained `temporal_horizon` model (10,677 angular parameters per
seed) improves the long-transfer fold from 0.51264 to 0.49712 deg, while its
active-origin mean worsens from 0.93185 to 0.96782 deg. The validation-selected
short-transfer `dense_horizon` model regresses from 0.43213 to 0.43858 deg.
See RESEARCH_2026_11_06.md for every model and the locked selection. Do not
present three independently trained fold models as one universally improved model.

The fixed synthetic check also limits the generalization claim. In the
undamped nonlinear free-sway subset with exact past measurements and known
zero future acceleration, this incumbent has 0.23573 deg mean error versus
0.01213 for the physical forecast. That check bypasses the actuator and uses
the train-calibrated effective length. It exposes behavior outside the recorded
training regimes; it is neither an industrial benchmark nor a controller test.

## Intended task and inputs

Conditional forecasting of fixed-length tower-crane trolley-axis sway. Supply
41 observations at 100 Hz, newest first: angle in radians, trolley x in metres,
and recorded native reference. Supply current cable length in metres and 30
prescribed reference samples at 50 ms, starting with the origin command.
The input does not contain future measured trolley motion or future sway.

The output is seven angles at 0.1, 0.2, 0.35, 0.5, 0.75, 1.0 and 1.5 s.
The ensemble also reports seed spread. It is not a calibrated confidence interval.
The native reference is not identified as voltage, force or exact trolley position.

## Construction

A separately trained recurrent actuator predicts trolley acceleration. A causal
forced-oscillator observer estimates initial angular state using the past position
response, without taking its noisy second derivative. A neural context encoder
refines state/frequency, initializes four passive memory states and drives a
bounded recurrent forcing correction. There are 8821 parameters per angular
seed, excluding the shared actuator. The three angular seeds are averaged.

The small-angle oscillator has analytic rotation, with two Cayley closure
half-steps per 50 ms step. The passive closure obeys a normalized storage
balance. Learned forcing is an additional input port; it is not measured force,
proof of passivity relative to the physical actuator, or a control safety claim.
Robust smooth-L1 loss trains the rollouts; validation MAE chooses stopping time.

The inference-only compiled flow precomputes D, M = D R D and B = D b once
per window, then applies z_next = M z + B(-a/g + f_learned). It uses the same
weights and fixed coefficients as the reference calculation. Float64 tests
check their agreement to 1e-12, and retained trained forecasts agree within
about 1.1e-5 degrees on the history-only smoke fixture.

## Data and selection

Original data: DOI 10.17632/x69hyyv3zw.2 and article 10.1016/j.dib.2025.111822.
Primary TRAIN 3/5/6, validation 4, evaluation 1/2/7/8. Training lengths are
approximately 0.296--0.596 m; evaluation lengths extend to 0.099--0.809 m.
The model was selected as the new neural variant with lowest validation MAE,
without choosing an evaluation seed or horizon mixture. These records have
been repeatedly used in development; evaluation is retrospective.

Additional range tests retrain both calibration and actuator within each fold:

| Test | Train | Validation | Extrapolation-only evaluation |
|---|---|---|---|
| Shorter lengths | 4/6/8 | 5 | 1/2/3 |
| Longer lengths | 1/3/5 | 4 | 6/7/8 |

Interpolation records in each extra fold are excluded from the extrapolation
headline. Their errors remain in the full CSV. Multiple appearances of a record,
overlapping windows and neural seeds do not create independent physical trials.

## Results, mean scenario/horizon MAE in degrees

| Test | Force-aware NN | Direct GRU | Zero-prior GP | Physical reference |
|---|---:|---:|---:|---:|
| Primary extrapolation | 0.50050 | 0.54202 | 0.52519 | 0.55420 |
| Shorter-length extrapolation | 0.43213 | 0.44140 | 1.70365 | 0.46275 |
| Longer-length extrapolation | 0.51264 | 0.57330 | 0.59624 | 0.65663 |

The earlier 512-point GP gives 0.51668 degrees in the primary protocol; it
still wins at some shorter/intermediate horizons. The previous driven NN
gives 0.51193, so this stage's primary gain is 2.23 percent. The original
PEPM gives 0.54267, making the cumulative gain 7.77 percent. These are
descriptive benchmark gains, not established population effects or a
best-in-literature claim. The four-record bootstrap interval still includes zero.

The earlier driven NN remains stronger in the active-origin 1.5 s subset:
1.64148 versus 1.71053 degrees for this variant. New phase-normalized and
measurement-bin-anchored variants are retained as negative results. Additional
load/shape records 10/11/12 are reported separately and have different winners;
the candidate is not uniformly best under those shifts.

Warm single-packet CPU median is approximately 21.1 ms for preprocessing,
the shared actor and all three compiled neural seeds; reference flow takes
59.0 ms and the 1024-point GP about 5.2 ms. Model loading, I/O and sensor
acquisition are excluded. This microbenchmark is not a real-time guarantee.

## Local use

After extracting the compact evidence archive into the project root:

```sh
python scripts/predict_cranepilot.py --run artifacts/runs/forceaware_campaign_v2 --variant forced_driven_huber --input examples/history_packet.npz --output forecast.csv --fast
```

Input NPZ keys: `theta`, `x`, `reference` of shape (N,41), `length_m` of
shape (N,) and `future_reference` of shape (N,30). The first future reference
must match the origin reference. This command requires no processed dataset.
Do not overwrite an existing forecast file; use a fresh output name.

## Unsupported conclusions and remaining work

The model is not validated for hoisting, rotating/combined axes, unknown future
commands, industrial-scale deployment, arbitrary length ranges, or hardware
control. It has not been evaluated inside MPC. It is a development checkpoint,
with 41 passing numerical/information tests and no guarantee that physical
assumptions or predictive accuracy hold outside the stated task.

See RESEARCH_2026_11_03.md for all variants, confidence limitations, retained
failures and exact baselines. Further work should reproduce additional published
methods under this contract and test independent simulation/public data before
making a broad best-in-niche claim.
