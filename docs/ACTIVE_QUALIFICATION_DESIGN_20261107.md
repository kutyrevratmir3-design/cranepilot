# Frozen-model qualification during motion and hoisting

User resumed research after the independent-data blocker. This stage improves
scope evidence without pretending that old recordings become fresh. The full
goal remains unmet until a suitable independent test is available.

Current reported gain belongs to4s settled-input prediction after25s. Before
developing more methods, measure the frozen predictors' behavior when drive or
cable length moves. It is explicitly an **out-of-scope qualification**: no fitting,
threshold/model selection, default promotion, API relaxation or hardware claim.

Use all33already opened Inteco records, with the original roles retained.
Primary descriptive comparison is only on the seven old-evaluation records;
TRAIN and VAL scores are separately labelled diagnostic. Five family/axis model
copies keep their original train-only calibration and saved long/shortARX choices.
No APHYNITY data/code is used.

Origins every250ms from4s through the last complete1.5s forecast. Every method
receives the same available4s angle/length/position/reference history, with81
angle/length samples at50ms for prediction and401raw100Hz samples for regime
labelling. Seven horizons remain .10,.20,.35,.50,.75,1.00,1.50s. The initial4s
have no complete recorded history and are explicitly unevaluated; do not zero-pad
an invented prehistory or claim startup performance.

Four disjoint regimes, computed from past-only ranges with previous thresholds:
drive_active iff x range>2mm or J range>2mrad; length_active iff L range>2mm.
The four combinations include low_past_motion, not a proof of zero forcing.
No window is excluded for either flag or for amplitude/error. All available
origins are evaluated. A separate indicator reconstructs the old numerical
scope: origin≥25s, both past ranges small, future recorded reference constant
through1.5s. This indicator is descriptive only; it does not establish that
future feedback commands are knowable. Predictions and primary regime labels
cannot use future measured angles, length, positions or reference values.

Frozen models: simpleARX selected previously on VAL (3s/4s as frozen percase),
full4stwo-mode modal, past-error soft blend, fixed-frequency harmonic ablation,
and persistence. The research-only kernel wrapper evaluates these outside the
public API's gate; the public API keeps returningNaN outside its declared task.
No predictor receives true future realised motion/length or an oracle actuator.
All predictions, model weights and backtest evidence are retained.

Report coverage, every scenario/axis/horizon, each old role and each regime.
Average horizons within an axis-record, axes within a scenario, scenarios
equally within each role/regime. Never treat overlapping windows as independent
trials or pool long records to inflate wins. Counts must show missing regimes;
no same-population inference across differently covered regimes. These are
domain checks, not a new best-method selection or independent superiority test.

Commit code/config/tests/design before running. Verify source/bundle/data hashes,
replay forecasts and metrics, test future-target poisoning and past-only regime
indexing. Preserve negatives and originals. A poor moving-length result may
motivate a separately registered physics model using past length and causal
future-length extrapolation; do not quietly supply measured future length.
