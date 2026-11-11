# Stronger dense ARX reference, without primary reselection

This protocol is written after all dated-forecast old-record scores were seen.
All 33 Inteco recordings are already opened. This stage can improve comparator
fairness and reproducibility; it cannot establish independent confirmation.
No new author data are read, no private raw records are published, and the
primary `mix_bank_10` source, bundle, weights, choices and forecasts stay frozen.

## Why another reference challenge

The physical coupling priors and original drive ARXs were fit on the same
TRAIN packets: origins from 4.0 s, stride 25 raw samples. The current dated
reference offers the previously chosen active ARX, not every saved dense ARX.
For jib theta, that choice is the older quiet-trained fallback; its original
dense active models are omitted. Trolley offers its chosen dense 3 s model,
not the complete dense 4 s grid. Other cases retain a dense 4 s selected model.
The newer age-augmented references use stride 100. Thus common source records
are established, but complete dense supervised-packet coverage is not.

## Fixed training and validation access

Fit exactly 120 new augmented ARXs: five cases, angle/io LPV, direct or
dated-modal residual, six ridge values. Use the existing dated feature,
TRAIN scaling, fit, calibration and balanced-record/quiet-drive helpers.
TRAIN origins are 4.25 s onward, stride 25; ages 0/20/50/100/150/200 ms and
noise 0 or uniform history jitter ±the existing encoder quantum. Use the same
raw_history generator and seed 20261108; each packet/noise history is simulated,
not a new physical recording. The changed TRAIN grid changes which random
draw belongs to a shared origin; no cross-grid noise identity is claimed.

VAL and old replay keep the original dated grid exactly: 4.25 s, stride 100.
This preserves the population used to choose the current primary/reference.
The new reference has the same raw histories, ages, noise and seven horizons
as the frozen primary. No old replay record is read during reference selection.

The support audit records roles, source XLSX hashes, calibration, raw origin
indices and hashes for each TRAIN record. It verifies that new dense origins
are exactly the original dense-prior origins at/after 4.25 s. Original dense
ARXs retain the omitted first 4.0 s origin per record. Actual delayed history
must exist even at the 250 ms evaluation boundary; no padding is permitted.
Raw source histories used by the new fits are within the original dense
TRAIN record/sample support. A stronger reference receives explicit delay/noise
augmentation; identical transformed examples across all learners are not claimed.
The support report also stores the actual 81 sampled angle offsets for every
training age. Delaying angles by 20 ms changes their sample phase relative to
the original 50 ms grid; interval containment is not a claim that every original
learner consumed every 100 Hz angle value. Raw TRAIN files and availability,
dense current origins and target support are the shared access definition.

## Complete reference pool and fixed choice

For every case, retain these 304 declared options (some may be duplicates):

- 24 newly fit dense age-augmented ARXs;
- all 24 previous dated augmented and all 24 latency augmented ARXs;
- all 24 original dense 4 s drive ARXs under each of the eight existing head
  adapters, regardless of which reference the old parent selected;
- the saved dense drive 3 s ARX, legacy 3 s ARX and legacy 4 s ARX, each under
  all eight adapters, regardless of the old parent's chosen history length;
- the previous 16 named frozen/active expert options under the eight adapters.

No old physical prior or ARX is refit. Direct and modal-residual models use
their original feature/base implementations, including the original 3 s slicing.
All head adapters use the frozen extended implementation for age 0..25.
Current support/native references and dated angle availability are identical
for every option. Unknown units/clocks cannot be repaired by filling channels
with zeros; a future external intake must establish task suitability first.

Choose one reference per case and each of six TRAIN ages, averaging balanced
VAL MAE across the two noise conditions. Exact score ties use lexicographic
option keys. Other ages use the nearest TRAIN age, lower age on distance ties.
Save all option specifications, source/model hashes, 120 new weights, all VAL
option scores, all 30 choices and selected forecasts. Never change the primary
or its acceptance gate because a stronger reference wins or loses.

## Freeze and replay

Commit this config, design, numerical source, verifier, tests and comparator
API before fitting. Preserve attempts with exclusive output directories and
failure provenance. Commit a byte-identical frozen development mirror before
opening any old replay records. Rebuild all 120 TRAIN fits and every VAL score
and choice before packaging the comparator.

Old replay uses precisely the existing dated ages 0/20/50/100/150/200/30/70/250 ms,
two noise levels, scenarios and origins. Preserve every original NPZ array,
including all primary candidates, diagnostics, labels and former comparators;
add the new `dense_matched_arx` forecast and selection metadata. Recompute the
frozen primary from raw histories and assert parity rather than selecting a
better candidate after this replay. Recompute all metrics and match original
metrics before attributing a reference change.

The separate comparator bundle contains only selected reference weights and
specifications. It pins the exact original dated bundle manifest and numerical
source; it does not duplicate or modify that primary bundle. Inference may
read the two specified bundles, never raw/processed datasets or prior runs.
`FrozenDenseReference.forecast_reference(...)` requires an explicitly supplied
known integer age 0..25. The independent-data intake must pin both bundles
before any fresh forecast errors are calculated.

API: `FrozenDenseReference(bundle=..., primary_bundle=...).forecast(history,
family=..., axis=..., angle_age_samples=...)` returns a dictionary containing
`forecast_rad`, `reference_key`, `matched_train_age_raw` and `horizons_s`.
`forecast_reference` is an equivalent alias. Both require explicit known age;
neither supplies or changes the primary forecast.

After source review/commit, use the restored main Python launcher from the
main repository: `sh ../../handoff/restoration_20261107/main-python.sh` followed by:

1. `-m scripts.dense_reference` (TRAIN/VAL only).
2. `-m scripts.verify_dense_reference --development-only`.
3. Copy the complete run to `results/dense_reference_20261108/frozen_development`
   and commit that numerical mirror; existing attempts must never be overwritten.
4. `-m scripts.build_dense_reference_bundle`.
5. `-m scripts.dense_reference --evaluate` (old records only, unchanged primary).
6. `-m scripts.verify_dense_reference` (source/raw/two-bundle parity and all metrics).

Verify source/raw/portable parity on every VAL and old replay batch, all stored
primary arrays, every metric, all selected reference choices and all source
hashes. Keep the 1,281 protected original files and 122 pre-existing EOL-only
Git differences unchanged. Publication covers new source/results/weights;
no original external data or private correspondence is added to packages.
Fresh-trial provenance, metadata eligibility, cluster hierarchy and the primary
independent acceptance criteria remain separate unresolved requirements.
