# Independent industrial data candidate: Aalto Ilmatar

6 November 2026. Acquisition and schema audit only; no trained models or
forecast scores on these records.

Primary records: https://zenodo.org/api/records/6907569 and
https://research.aalto.fi/en/datasets/driving-smart-crane-with-various-loads/.
Dataset DOI: 10.5281/zenodo.6907569, version 1.0.2, CC BY 4.0.
Credit: Riku Ala-Laurinaho and Tuomas Keski-Heikkila, with Kari Tammi.

The official 5,855,928-byte archive was downloaded and checked against the
publisher's MD5 `2ad915f66f38ba7a5a61f8e70183b09c`. Its SHA256 is
`78269ff76e330910ed1789a20874bfccfcbbe26b38aac3d4f0bad37187d88952`.
The unmodified source is retained in `data/raw/aalto_6907569/` and is not mixed
with the original Inteco training data. Download/audit can be reproduced with
`python -m scripts.audit_aalto --download`.

## What the files establish

The combined table has 31,303 rows and 15 columns across eight cycles. Payloads
are 0, 120, 500 and 1000 kg, each with anti-sway control on/off. Columns include
bridge/trolley position and rope angle, hoist position, speed feedback, motor
torque, tared load and alarm status. No entries are missing in that table.

All eight cycles have a regular 0.25 s exported grid. The raw logs are
event-based; the median interval between logged angle changes is about
0.297 s. The regular table is therefore not evidence of independent fresh
measurements at every grid point. Around 19--28 percent of trolley-angle rows
repeat the preceding value, depending on the cycle.

The raw tag maps contain `RopeAngle.Angle_rad`, speed requests and a
`SwayControl_SlingLength_mm` setting. This is stronger evidence for the angle
unit than guessing from its range. A tag name alone does not establish how the
angle was sensed or estimated, or the total effective suspended length.

Original raw and exported timestamps contain year 2103. They are preserved;
the archive's publication date is not a defensible replacement timestamp.
Within-cycle relative time is regular in the combined table. The per-cycle
and raw-signal audits are retained under `results/latest/aalto_20261106_*`.

## Timestamp alignment finding and reconstruction

Across cycles, roughly 77.5% of trolley-angle rows and 78.3% of bridge-angle
rows equal the next raw event while differing from the previous available
event (absolute tolerance 1e-9 rad). The mean of per-cycle median look-aheads
is about 0.126 s. For example, cycle 1's bridge angle at 15:27:16.500 is
0.001592 rad, matching an event at 15:27:16.585; the preceding event has
0.002721 rad. Exact timestamp/value examples and per-cycle counts are retained.

This finding is consistent with assigning subsequent-bin events to the bin's
start; the original export code was not supplied, so that mechanism is an
inference. The table must not be treated as instantaneously available data at
its printed timestamp without resolving this. It is not evidence of wrongdoing
by its creators: the export may have been intended for offline analysis.

`scripts/prepare_aalto.py` now reconstructs ten signals directly from raw logs.
It uses the last finite, GOOD-status event whose source and server timestamps
are both at or before the grid time. It retains missing initial observations
and the time since each last logged event. Log age is not automatically sensor
age, because stationary signals need not generate new events. Only Date/Cycle
from the combined table are reused, never its exported signal values.

The resulting `data/processed/aalto_6907569/observations.npz` and provenance are
local and separate from Inteco. Three tests cover future-value mutation,
delayed availability, unsorted events, duplicates and missing initial history.
This makes a causal dataset candidate available; it does not resolve geometry,
angle provenance or the future-command contract, and no model was fit to it.

## Why it cannot be fed directly to the current model

| Current Inteco contract | Aalto evidence | Required work |
|---|---|---|
| 41 samples at 100 Hz, 0.4 s history | Exported table at 4 Hz; raw angle changes around 0.3 s | Define a native-rate contract; interpolation cannot manufacture 100 Hz evidence |
| Measured fixed cable length | Hoist position and a sling-length setting | Verify geometry, offsets and changing suspended length |
| Prescribed native future reference | Combined table has no reference; raw speed requests exist | Establish semantics and a causal command contract |
| Encoder angle as evaluation target | OPC UA angle tag in radians | Verify sensing/estimation and filtering provenance |
| Causal features from raw samples | Export preprocessing is undocumented in the archive README | Reconstruct from raw events with backward as-of alignment and explicit signal age |

In particular, speed feedback is not automatically metres per second, hoist
position is not automatically cable length, and realized future speed requests
are not automatically commands known at the prediction origin. Future measured
positions or accelerations must not be inserted as ordinary model inputs.

## Next external evaluation

First resolve the above measurement/command/geometry contract, then lock entire
cycles or payload pairs as evaluation groups before any model fitting. An
angle-only native-rate benchmark may be possible sooner than the existing
command-conditioned model. Keep its results separate from Inteco length
extrapolation. Do not infer that a schema audit demonstrates transfer quality.
