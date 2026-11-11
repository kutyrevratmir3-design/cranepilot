# Pre-use audit of the prospective confirmation gate

No real incoming physical trial was scored, screened numerically or locked
during development of this gate. All scorer tests used synthetic arrays and
fake predictors. The actual private registry contains only previously opened
main33, Aalto6907569 and Behavioral20 data. Readiness is not confirmation.

A final independent code review found four concrete gaps after the initial
302-test readiness check. They were fixed before any real candidate intake:

| Gap | Final behavior |
|---|---|
| A pruned registry could retain the correct policy hash. | Exact private-index SHA, corpus coverage, policy and old source hashes are bound by a committed public seal. A different index is rejected before candidate parsing. |
| Distinct locks could verify stale ledger state simultaneously. | One atomic directory lease covers re-verification, numerical re-screen and full-cohort reservation before the first predictor call. A busy/stale lease fails closed. |
| Prior ledger commitments were recorded but not rechecked. | Missing, modified, nonregular or escaped previously committed events block verification. Added unrelated events are allowed. |
| Copied waves in one incoming cohort could inflate declared clusters. | Each physical record's angles are also checked against earlier records in the same cohort. Axes inside one physical trial remain a single trial. |

New synthetic regressions cover registry pruning/source changes, offset copies
with different NPZ serialization and IDs, valid distinct signals with shared
support, lease ownership/cleanup, deleted/changed prior events, and an exposure
arriving between initial score checks and final reservation. The last case
must prevent both predictor calls, including when only numerical overlap reveals
the reuse. Existing failed-coverage, privacy, immutable-attempt, clock and input
parity checks remain in the suite.

The first hardening-test attempt failed because a fixture created the new seal's
directory before the fixture's existing directory creation. The fixture was
corrected to permit the existing directory. Both failed and passing attempt
logs are retained. This failure involved no physical records or fitted models.

`readiness_initial_302tests.json` and `final_unit_tests.log` retain the initial
software check. `readiness.json` and its named hardened unit report describe
the final gate. Model weights and numerical retrospective results are unchanged.

Residual limits remain explicit: hashes and declarations cannot authenticate
independent physical trials; resampling/noise may evade overlap fingerprints;
filter causality and native reference units require concrete source evidence.
Keep the private exposure ledger on device transfers. Its absence is not proof
that trials were unused. Interrupted reservations require investigation and can
consume a cohort conservatively even before a forecast returns.

The research goal remains open. No independent superiority, real-time control
or hardware result is claimed by this audit.
