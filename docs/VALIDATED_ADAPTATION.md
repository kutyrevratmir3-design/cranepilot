# Chronological prefix acceptance: development result

2 November 2026. Retrospective pilot; no new experimental measurements.

The earlier prefix estimator worsened 1 s MAE from 0.672523 to 1.099353
degrees at a ten-period calibration budget. This motivates a check of whether
prefix estimates can be validated before they replace the physical prior.

## Fixed protocol

`configs/validated_adaptation_pilot.json` freezes the settings. The prior
length relationship uses scenarios 3, 5 and 6. Development scenario 4 is
processed before the four evaluation scenarios 1, 2, 7 and 8; the settings
are not revised from this run's results. All those records have already been
used in earlier work, so this is not an untouched confirmatory test.

Each budget ends after 0, 1, 3, 5 or 10 periods of the prior oscillator.
Frequency-only and damped physical candidates minimize multi-step forecast
residuals on the first 60% of that angle prefix. Their predictions are checked
on the last 40%, with complete observation histories and targets after the
training portion. No target after the prefix is used in parameter selection.

Acceptance requires at least two disjoint one-second future-target blocks,
at least 5% improvement in mean validation MAE, no worse error at any of the
0.1, 0.5 and 1 s horizons, and wins on at least 75% of the blocks. Dense origins
are used for per-horizon MAE; their overlap is not treated as independence.
The block rule is a heuristic, not a confidence interval or safety guarantee.
An accepted candidate would be retained without refitting after selection.

All evaluation forecasts use the same origins beginning at 25 s. The ungated
reference uses measured trolley motion as well as sway in the prefix; the
new candidate uses sway only. They therefore do not have identical information
sets. The chosen gates and objective are development choices, not a new-method
novelty claim.

## Retained result

No update is accepted. Among 16 nonzero-budget evaluation cases, nine lack
sufficient training/validation prefix data and seven fail validation. The
development scenario also accepts no updates. At every calibration budget,
the validated method is exactly the zero-shot prior:

| Horizon | Zero-shot / validated MAE (degrees) | Ungated MAE at ten periods (degrees) |
|---|---:|---:|
| 0.1 s | 0.258590 | 0.321382 |
| 0.5 s | 0.540297 | 0.901403 |
| 1.0 s | 0.672523 | 1.099353 |

Avoid interpreting equality after universal rejection as successful adaptation.
Do not report a percentage gain against the ungated reference as an improvement
over calibrated physics. A synthetic analytic-signal test demonstrates that
the implementation can accept and recover a genuine frequency shift; it
does not establish that real prefix data identify that shift reliably.

## Reproduction and evidence

Run `python scripts/run_validated_adaptation.py` in the project environment.
The output directory must not exist, to preserve earlier evidence. The run
retains settings and source hashes, data-manifest hash, Git status, fitted
candidates, chronological-prefix validation diagnostics and scenario metrics.
Selected copies are in `results/latest/validated_adaptation_*`.

Tests cover analytic frequency recovery, post-budget mutation invariance,
rejection when a training-only regime fails later, and short-prefix abstention.
All 17 project tests pass. The next useful investigation is the actuator/input
path and changing forced-motion regimes, rather than relaxing this gate until
the already-seen evaluation scores improve.
