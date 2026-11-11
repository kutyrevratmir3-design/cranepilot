# Separate dated-angle forecast profile

This experimental API uses `artifacts/dated_bundle_20261108`. It does not
replace previous bundles or the original neural API. It needs the main
repository and its Python dependencies, but no raw/processed dataset or
ancestral run directories during inference. Bundle and source hashes are
checked at construction.

```python
from cranepilot.dated_pilot import FrozenDatedPilot

pilot = FrozenDatedPilot()
result = pilot.forecast(
    history,
    family="trolley",       # trolley, jib, simult
    axis="theta",           # beta also supported for jib and simult
    angle_age_samples=15,   # known acquisition age: 150 ms at 100 Hz
    include_reference=True,
)
forecast_rad = result["forecast_rad"]   # (N, 7), radians
```

Each history is a finite `(N, 401)` NumPy array, newest first, sampled at 100 Hz.
The selected angle ends at `t - angle_age_samples * .01`; other channels end
at `t`. Supply `L`, the selected angle, and:

| Family | Required support and original reference channels |
|---|---|
| trolley | `x`, `xr` |
| jib | `J`, `Jr` |
| simult | `x`, `xr`, `J`, `Jr` |

`L` must be positive. Use the same signal units and conventions as the source
Inteco recordings. The physical meaning/units of native `xr`/`Jr` are still
awaiting author confirmation; they must not be relabeled as acceleration or
voltage without evidence.

Age must be a known integer from 0 to 25, common to this batch. Boolean,
fractional and unknown ages, future/target fields, nonfinite or misaligned
histories are rejected. Provide genuine prehistory: the oldest angle is
`4 + age*.01` seconds before t. At age=0 all candidates exactly preserve
`FrozenRegimePilot`; the separately reported bare dated-modal baseline does
not receive that guard.

Targets are at `[.10, .20, .35, .50, .75, 1., 1.5]` seconds after **t**, rather
than after the last angle acquisition. `include_reference=True` returns
`matched_arx_rad`, `augmented_arx_rad` and `bare_dated_modal_rad`. The matched
reference uses frozen case/age VAL choices; unseen ages use the nearest
TRAIN age, with lower age winning a distance tie. It does not select on
forecast error or infer unknown latency.

`include_candidates=True` also exposes all 21 fixed candidates and the
past-error/weight diagnostics. The implementation currently computes the
complete bank even for a selected-only request. CPU timings in the study
therefore apply to that implementation, with no real-time guarantee.

Future support and rope length are assumed continuations. Received support
through the past sensor gap is measured; no actual future trajectory is
supplied. Synthetic delay replay keeps recorded control actions unchanged
and does not simulate the resulting closed loop. Actual hardware channel
availability needs acquisition/arrival timestamps and filter metadata.

See `DATED_FORECAST_2026_11_08_RU.md` for retained gains, losses and limitations,
`DELAY_CONTROL_LITERATURE_20261108_RU.md` for related work, and
`results/dated_forecast_20261108` for the full frozen grid and verification.
