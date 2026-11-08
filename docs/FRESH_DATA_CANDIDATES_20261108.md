# Additional physical-data lead and eligibility decision

This search advances access checking, not independent performance evidence.
No new full-profile physical record has been scored or used for fitting.

## AAU gantry crane: real files, insufficient joint channels

The [pinned repository](https://github.com/elomarjc/gantry-crane-anti-sway-control/tree/18bda9674252dfebd74bdeadbc0bbea770d9669f)
describes a Spring 2023 Aalborg University laboratory 2D gantry crane using an
Arduino Mega, positioning potentiometers and a separate angle sensor. This
is a different rig from the NU Inteco source. The GitHub repository metadata
did not identify a top-level data license; public access alone was not treated
as permission to redistribute its raw files.

Only README/source, schemas, headers and time columns were inspected. MAT
files were checked in memory with `whosmat`; response values were not printed,
scored or fitted. Original response freshness was preserved for any later
properly qualified task.

| Asset | Observed schema |
|---|---|
| Acceptance test 2 MAT | `angle`, `t`, each 3612×1 |
| Acceptance test 3 MAT | `Positionup`, `Positiondown`, `t`, each 14933×1 |
| Acceptance test 4 MAT | `Position`, `t`, each 7114×1 |
| Technical requirement 2 CSV | Time, Angle, matched degrees; time 0–36.11 s, 10 ms spacing |
| Experimental X step CSV | Time, upper position, Match; 0–10 s, 10 ms spacing |
| Experimental Y step CSV | Separate upper/lower position timelines |
| Two experimental ANGLE XLSX | Time, Angle, Match headers; sheet dimension 4745×16 |

[Figure_maker.m](https://github.com/elomarjc/gantry-crane-anti-sway-control/blob/18bda9674252dfebd74bdeadbc0bbea770d9669f/Main/data/Acceptance%20test/Figure_maker.m)
uses separate acceptance files for angle and positions. Its X figure uses a
moving average including later samples; that plotting operation cannot be
adopted as causal inference preprocessing.

[Firmware](https://github.com/elomarjc/gantry-crane-anti-sway-control/blob/18bda9674252dfebd74bdeadbc0bbea770d9669f/Main/include/functions.h)
sets a nominal 10 ms controller timer, converts received angle degrees to
radians and zeroes small angles within 1 degree. It is not established that
this exact branch/filter produced each exported file. A controller timer is
not proof of fresh synchronous angle acquisition or known delivery age.

The MAT2/3/4 Git blobs match the pinned tree. Their SHA256 values are,
respectively, `c83fd190d5f135cd5c659358376a580afbab1e194bef4e2130640a64d6190722`,
`976e194a8feafd3a168be2bb93c99434cd1da94515788484a754e0f41c871d1e`,
`c2fec58fd44ae1422c4ed4535338554ee8c2e1d61c537dd4deeabe3622813734`.
Hashes confirm asset identity, not independent physical trials.

Decision: **not eligible for the current frozen full-profile confirmation**.
No synchronous angle/L/support/native-reference packet was established. Vertical
position is not automatically suspended rope length; a gantry setpoint or motor
voltage is not automatically the Inteco native reference channel. Joining
separate experiments or inventing missing zero channels would change the task.
An angle-only transfer experiment would require another explicit, preregistered
task and adequate physical metadata; it cannot stand in for the present goal.

## Other bounded checks

- Targeted connected-Drive metadata searches found previous manuscripts and
  papers, no suitable new numerical crane logs. No private file IDs, content or
  correspondence are included here; nothing was changed in Drive.
- Same-author public searches still lead to the old33 collection and the
  already documented on-request papers. The repeat search does not prove that
  no other data exist.
- [Zenodo 17181719](https://zenodo.org/records/17181719) and
  [10533297](https://zenodo.org/records/10533297) expose PDF assets rather than
  a verified new raw payload-sway corpus.
- The [November 2026 container-crane record](https://data.mendeley.com/datasets/rpkkfhmft8/1)
  describes seismic shake-table work. The required joint suspended-load angle,
  drive and real-length channels were not verified; no numerical responses
  were downloaded or relabeled as a suitable forecast benchmark.

The concrete outstanding input remains new raw physical trials with source,
rig/session/condition provenance, channel meanings/units and clock information.
No messages were sent to authors. The runnable intake/protocol gate is in
`FRESH_CONFIRMATION_20261108.md`; it does not manufacture independent evidence.
