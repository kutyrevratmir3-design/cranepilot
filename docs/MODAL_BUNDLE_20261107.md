# Frozen modal predictor and simple ARX reference

Experimental bundle: `artifacts/modal_bundle_20261107`,12files,107738bytes.
It includes task/selection metadata, five train-only length calibrations, five
validation-selected reference ARX checkpoints, and hashes of required source.
No raw dataset, training, ancestral run tree or neural weights are required for
inference. Main-version Python dependencies and repository source are required.
APHYNITY and the retained neural bundle are separate and unchanged.

```python
from cranepilot.modal_pilot import FrozenModalPilot
pilot = FrozenModalPilot()
result = pilot.forecast(
    history, future_reference,
    family="trolley", axis="theta", known_plan=True,
)
# result["forecast_rad"] and result["reference_arx_rad"]: (N,7)
# result["eligible"]: (N,), forecasts are NaN outside the declared task.
```

`history`: dict of finite float(N,401)arrays at100Hz, newest first: L(m), target
angle theta/beta(rad), position x(m) and/or J(rad), native reference xr/Jr.
Families: trolley/theta; jib/theta,beta; simult/theta,beta. For simultaneous
motion, supply both position/reference pairs. Length must be positive.
`future_reference`: corresponding xr/Jr arrays(N,151), current first, with
their actual native units documented. These must be a plan known at forecast
time. Future feedback-controller outputs cannot be silently treated as known.
The first future reference must match the current history reference.

Forecast horizons .10,.20,.35,.50,.75,1.00,1.50s. Kernel uses81angles at50ms
spacing, covering4s. Eligibility uses full401past samples: length range≤2mm,
position range≤2mm/2mrad, known future reference constant≤1e−9 native units.
The gate never reads future measured length/position/angle. It does not prove
zero physical forcing, constant future length or safe closed-loop control.
The source evaluation additionally starts origins at25s; callers must retain
that protocol rule when reproducing its scores.

Frozen selected candidate: two-mode past-only projection in allfive cases.
Reference: post-hoc strengthened, validation-selected simple ARX; it is included
to avoid presenting the weaker absolute-coordinate ARX as the only baseline.
The result on seven formerly unscored public conditions is .261421° vs .492306°,
−46.9%,6/7wins; whole-scenario interval includes0. These are now retrospective
records and do not establish independent new-data superiority.

Observed single-packet kernel latency on this Mac is median11.3–11.6ms, p95
11.4–12.0ms across fivecases. Measurement is not a hard real-time guarantee.
Source/checkpoint hashes are checked when loading; altered source is rejected.
New author logs require metadata/units/clock and known-plan verification before
this interface can be applied. See AUTHOR_DATA_INTAKE_20261107_RU.md.
