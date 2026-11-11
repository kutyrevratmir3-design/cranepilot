# Causal recent-history model selection: preregistration

Status: every old33scenario record and the previous seven conditional holdout
scores are already open. This is reliability development, not a fresh test. It
cannot close the active independent-confirmation goal. Old source/config/models,
the portable modal bundle, original defaults and APHYNITY remain unchanged.

Motivation: the frozen modal model loses scenario21 to a simpler ARX; the mean
advantage has a wide interval. Before relying on a single physical fit, test
whether recent causal prediction errors indicate which model is locally reliable.
The repository already contains forced41sample history-backtest diagnostics.
This new stage asks a different, explicit question: model-class selection with4s
history, a two-mode estimator, and equal-data long/short simpleARX references.
No claim to invent adaptive supervision or backtesting.

## Inputs and internal backtest

The task, TRAIN/VAL records, calibration, origin gate and seven scoring horizons
are unchanged from MODAL_HISTORY_DESIGN. All methods can see the same4s packet
of81past angle/length samples at50ms. Public-record origin gate still inspects
the full401raw100Hz past samples. No future measured gate is introduced.

At forecast time0, form a61sample prefix covering[−4,−1]s. Run the frozen
two-mode algorithm on it, estimating frequency/damping/state from that prefix
only. Nominal omega uses the length at−1s and the unchanged TRAIN calibration.
Compare its forecasts .10,.20,.35,.50,.75,1.00s with already observed angles
at−.90,−.80,−.65,−.50,−.25,0s. These are inputs at time0, not future targets.
Never use the1.5s backtest head (it would extend beyond time0).

Fit exactly one3sARX percase on the same TRAIN records and eligible origins,
using the earlier simpleARX representation/form/alpha without a new grid.
Its3s prefix forecast is compared with those same six observed angles.
History-only backtest MAEs em,er are in radians. The3s model also forecasts
from the newest61samples at time0. For a stronger comparison, choose3s or
the earlier4sARX percase by original validation MAE; keep both outcomes.

## Fixed operator pool and freeze

Reference: selected3s/4sARX. Modal: unchanged4stwo-mode forecast.
Hard thresholds[.8,1,1.2] use modal iff em≤threshold*er, otherwise reference.
Soft: modal weight=er²/(em²+er²+quantum²/12), complement to reference. This is
a fixed heuristic reliability weight, not a probability or uncertainty guarantee.
The same operator is selected globally for allfive cases by equal-case VAL MAE,
with the reference winning exact ties. Six variants total; retain every score,
forecast, weight, backtest error and allfive new3sARX fits.

Freeze weights, choice and source hashes before retrospective scoring of the
seven old conditions. New reference receives at least as much validation choice
as before; it cannot be intentionally weakened. Report all7conditions, axes,
horizons and failures; whole-scenario bootstrap is descriptive. No tuning or
default/bundle replacement based on these retrospective results. Even a strong
old-data gain requires a separately frozen test on new appropriate measurements.

Development gate remains≥10%MAE reduction and≥4/5case wins versus the strengthened
reference. If it fails, retain the failure. Unit tests must check prefix indexing,
ignored future targets, weight bounds and exact endpoints before fitting; commit
source/config/design/tests first. This stage is a predictor experiment, not a
switched-controller stability or hardware claim.
