# Timestamp-offset run: metadata-only recovery

The fixed replay at commit `8ea1fa38902640d65e645530c71ecd95fa4d1594`
finished writing all 84 forecast batches and all eight metric/routing CSVs.
The process then exited with code 1 when the terminal provenance JSON tried
to serialize NumPy integer labels for the common true ages. The initial
running record and original error log are preserved under
`results/timestamp_offset_recovery_20261109/`.

`scripts/finalize_timestamp_offset.py` reconstructed the terminal metadata
from the unchanged saved arrays. It required the original terminal marker
and error, the full preregistered batch list, exact archived targets and
zero-offset parity, and independent recomputation of each CSV. It then
verified that all result bytes remained identical and replaced only the
provenance JSON. The exact elapsed time held only in process memory is not
reconstructed. No forecast was retried and no model or method was changed.

The separate full audit in
`results/timestamp_offset_audit_20261109/verification.json` checks the
recovered metadata against all batches and tables. This is a saved-array
audit; the nonzero-offset forecasts were not independently rerun.

To reproduce in an isolated working copy with the frozen inputs and no prior
timestamp output directory, use the metadata conversion wrapper:

```sh
sh ../../handoff/restoration_20261107/main-python.sh -m scripts.finalize_timestamp_offset --replay
```

The wrapper calls the same locked producer and converts the output age labels
to JSON integers. It refuses to overwrite an existing run. The current
artifact retains the original process failure and the metadata-only recovery.
