# A locked path to independent confirmation

This is executable preparation for new physical records. It is not a report
of a successful independent experiment. No suitable new full-profile records
have been received. The selected predictor remains `mix_bank_10`; the stronger
comparator is the separately frozen dense ARX reference. No new fit or model
selection is permitted on incoming confirmation records.

## What is frozen before scores

`configs/fresh_confirmation_20261108.json` declares the sole primary metric,
packet origins, seven horizons, overlap policy, bootstrap rule and acceptance
thresholds. The primary mean gives equal weight to conditions, physical trials
within condition, axes within trial, and horizons/origins within axis. It does
not exclude windows by amplitude, prediction error or drive regime.

This confirmation policy requires at least five declared conditions and five
independent clusters. Repeats are desirable; the software never counts axes
or overlapping forecast windows as independent trials. The statistical gate
requires at least 10% lower MAE than the matched ARX, a strict majority of
condition wins and a negative upper 97.5th percentile of paired bootstrap
differences. These thresholds are fixed now, not chosen from new errors.

The bootstrap hierarchy is resolved from metadata at lock time. Disjoint
condition clusters use conditions then whole clusters within condition. If
clusters span conditions, whole clusters are sampled while condition strata
remain fixed. A draw missing a condition is assigned an infinite adverse
difference for the upper bound; it is not silently discarded. Cluster IDs and
source independence require evidence: naming them does not establish it.

## Private intake and the two clocks

Incoming raw bytes, source-review notes, normalized files, metadata, numerical
screens and forecasts/targets stay in a Git-ignored private directory, by
default `data/private_confirmation_20261108`. Research-use permission is
separate from permission to publish raw data. Publication defaults false.
The public card contains a cryptographic commitment to the private lock,
counts, policy and model/source hashes; no raw records, identifiers or paths.

`scripts/fresh_record_guard.py` documents the manifest schema. Each entry is
one physical trial. Trolley needs theta/L/x/xr; jib and simultaneous trials
require both theta and beta and their corresponding support/reference channels.
Units, real suspended length, quantization, filters and native reference
compatibility need explicit evidence. No missing channel is replaced with zero,
and a gantry position setpoint or voltage is not equated with Inteco xr by name.
An opaque native export unit is allowed only with explicit matching export
semantics/scale evidence. Unknown training-reference semantics remain a real
compatibility question for the source review.

The root lock additionally binds references to
`configs/inteco_native_contract_20261108.json` by exact SHA256, unit and semantic
code. That contract explicitly retains the unverified servo scaling of the
original exports. It does not turn a declaration into proof of voltage, force
or position meaning. A different verified conversion needs its own frozen
contract before new scores, rather than silently changing the current one.

The private source review is JSON with `source_id`, `rig_id`, canonical
`metadata_sha256`, `training_contract_sha256`, a nonempty `evidence` list,
`normalization_kind` (`author_supplied_si` or `frozen_causal_adapter`),
`normalization_evidence`, `original_source_files` (private relative paths to
SHA256), and `adapter_hashes` (committed repo source paths to SHA256). A causal
adapter requires actual code commitments. Original source bytes are checked
against the seen registry too. Evidence authenticity and the conversion's
physical/causal validity still need the source audit; JSON assertions alone
cannot certify either.

Two scopes avoid an unnecessary claim about physical arrival latency:

| Scope | Meaning and evidence |
|---|---|
| synchronous_acquisition_offline | Certified common acquisition grid, explicit zero acquisition alignment. Actual arrival delay may remain unknown. Only offline forecasting at that acquisition origin is claimed. |
| known_arrival_age | Actual acquisition/arrival timestamps and current nonangle availability evidence. The known scalar angle age is an integer from 0 to 25 samples. |

Unknown actual lag is never silently set to zero. The first scope is explicitly
different from claiming zero network/sensor latency. Filters must be known and
causal; their delay treatment is declared. A different actual clock needs a
separately frozen conversion adapter. Mechanical normalization is permitted
only with a source-defined mapping fixed before model errors; no future-looking
smoothing, full-record target centering or target-fitted calibration.
Known-arrival metadata supplies `arrival.nonangle_time_columns` for every
nonangle input, plus the shared angle `time_column` or per-axis
`angle_time_columns`. The loader checks those arrays against the declared
acquisition grid and age; positive support age does not meet this profile's
current-support contract. Offline acquisition-only scope does not require or
claim physical arrival times.

The numerical loader accepts normalized NPZ arrays using the manifest's column
mapping. It checks finite aligned 1D channels, positive L, a 100 Hz grid,
actual arrival age where claimed, genuine histories and all seven target tails.
Origins start at raw index 425 and step by 100. Both methods receive exactly the
same arrays and indices. Failed trials remain in the coverage report and fail
the complete-coverage criterion. They are not quietly removed from the mean.

## Prior use and potential overlap

The seen registry covers the main 33-record source, Aalto 6907569 archive and
its causal angle reconstruction, and all 20 previously opened
BehavioralTrajGenCrane camera records. It includes file/member hashes and
numerical angle fingerprints. The new gantry candidate was inspected only for
metadata/schema, so it is not falsely labeled used-for-performance here.
Shared clock grids, calibration files, license text and prescribed input
profiles are recorded as auxiliary assets. Reusing them does not by itself
reuse a measured trial. Only response-bearing files trigger the exact-byte
freshness block; every signal still needs its source/availability audit.

Full exact and offset/sign-invariant numerical hashes detect renamed/reformatted
copies. A 256-sample block index at seen stride16 is compared with every candidate
offset. Constant blocks are skipped for duplicate search only. Short unaligned
crops, changed rounding, resampling, noise or other transformations may evade
this screen. A hit is potential overlap requiring resolution before scoring;
a nonmatch never proves that a physical trial is independent. New session/source
provenance and the concrete review remain necessary.

## Commands and immutable order

Run from the main repository using the restored Python launcher. APHYNITY and
all older source/bundles remain unchanged.

```sh
# Old, already opened corpora only; creates a private overlap index.
sh ../../handoff/restoration_20261107/main-python.sh -m scripts.fresh_confirmation registry

# After real files/metadata/source evidence arrive; all paths are explicit.
sh ../../handoff/restoration_20261107/main-python.sh -m scripts.fresh_confirmation lock \
  --private-root data/private_confirmation_20261108 \
  --manifest data/private_confirmation_20261108/manifest.json \
  --index data/private_confirmation_20261108/known_seen.json \
  --review data/private_confirmation_20261108/source_review.json \
  --public-card docs/preregistrations/fresh_confirmation_COMMITMENT.json
```

The lock refuses incomplete declarations, reused source bytes or insufficient
declared clusters. Before any candidate numerical parsing, commit the generated
public card. Then `screen --lock PRIVATE_LOCK --public-card PUBLIC_CARD` checks
schema and potential overlap; `score` with the same arguments runs the paired
frozen APIs. Input bytes, model/source hashes and the public commitment are
checked again. All scoring outputs stay under the private lock directory.
Each scoring attempt gets a new immutable directory; a rerun cannot overwrite
forecasts or a failed attempt. These commands do not authorize publication of
private data or contact with authors.

The software reports a statistical gate and evidence coverage. It never
certifies authentic physical independence or declares the overall research
goal achieved by itself. Actual new-record provenance, fair TRAIN access,
preregistration and every result still need the full completion audit.
