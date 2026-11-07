# One conditional confirmation after modal development

All signal processing, eligibility, horizons, model variants, ARX forms/ridge
grid and acceptance criteria are frozen by MODAL_HISTORY_DESIGN_20261107.md and
its config (commit d1f722a). No holdout error was available during selection.
Development: mean MAE .428822907° vs strong ARX .582517335° (−26.3845%), 4/5case
wins; localAR .893746539°. All five selected modal variants are two-mode.
simult/beta is worse (.742927 vs .532586°); keep this failure visible.

Before any newly scored signal read, freeze the development provenance,
selection, weights, verification and source hashes in the confirmation config.
The runner verifies them before using a lazy, explicit-ID loader. Only IDs
9,13,21,22,23,32,33 are allowed. No global eager load. No re-selection or fitting
on holdout targets. Past-only modal/localAR estimation is inference under the
frozen procedure. Do not change it after scores are observed.

Primary comparison: selected modal vs frozen strongARX, exactly the same windows
and input history. Fixed calibration, one-mode, selected localAR and persistence
are prespecified secondary references/ablations. No best-of-holdout selection.
Report every case, horizon and eligibility count, including any empty case.

Acceptance defined in the earlier config: equal axes per scenario, equal7
scenarios; ≥10% meanMAE reduction, ≥5/7scenario wins, paired whole-scenario
bootstrap(seed7112030,10000draws) upper95%change bound<0 and no empty case.
Overlapping forecast windows are NOT independent test samples. Report intervals
as descriptive across seven public scenarios, not a guarantee over new rigs.

The audit supports *previously unscored public conditions*. Full33record import
and possible unrecorded access occurred previously; fresh independent collection
is not established. The settled-input gate does not establish unforced motion
or constant future measured length. This result concerns residual-sway forecasts
with4s history. It does not prove superiority on the original .4s active task,
native-command closed loop, or competing papers' own metrics. Keep the broad
research goal open pending suitable author-provided/new raw recordings.
