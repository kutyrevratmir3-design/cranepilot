# Frozen-weight physical-limit audit

7 November 2026. Written before running the new diagnostic.

The restored incumbent passes inference replay but its retained free-sway
diagnostic has much larger undamped error than the physical reference. Before
changing training, isolate which learned paths contribute to this discrepancy
and which paths are useful on real active motion. This is a bounded development
audit on the current InnovationForecaster, separate from the APHYNITY pilot.

Use primary-fold frozen seeds 0/1/2, frozen actuator and original train-only
calibration. Verify original guarded/checkpoint SHA256 hashes before execution.
No optimizer, retraining, hyperparameter search or model selection is performed.
Keep the incumbent as the deployed/default predictor regardless of these results.

## Interventions

Fix ten variants in `configs/limit_audit_20261107.json`: unchanged incumbent;
six one-component removals; observer-only; dynamics-only; and all learned
corrections removed. Components are initial angle shift, additive normalized
momentum shift, frequency adjustment, initial memory, passive closure (all
damping/exchange/decay coefficients), and learned forcing. Zeroing frequency
adjustment retains the existing momentum normalization rule. Zeroing initial
memory does not prevent subsequent memory excitation by the closure.

Forward hooks and temporary method overrides modify inference only, never the
stored state_dict. Recompute the recurrent forcing on each intervened trajectory;
do not replay a force trace from a different trajectory. These are interventions
on jointly trained, co-adapted weights, not independently retrained ablations,
additive feature importance, or identification of the real plant's causal terms.
The all-off forecast must agree with the analytical forced oscillator. The
unchanged forecast must reproduce the saved validation and synthetic traces.

## Data boundaries

Recorded development only: train 3/5/6 and validation 4, stride 10, 41 history
samples, seven original horizons, native future commands and the same learned
actuator for every variant. No primary test or alternate-load record is scored.
The dataset container contains other records; only these four are windowed.
Targets are stripped from model inputs. Report each scenario and horizon, and
active/quiet subsets using only past-derived |velocity| >= 0.005 m/s.
Aggregate train scores with equal scenario/horizon weight. Validation is one
record; neither windows nor seeds are independent physical trials.

Synthetic development: reuse the already exposed 576-case November 6 set with
zero true acceleration, bypassing the actuator. Preserve exact/quantized history,
length, amplitude, phase and damping breakdowns. Reproduce the original target
array and check RK4 step halving. This is not fresh generalization evidence.

A separate parameter grid is reserved in the config before the diagnostic.
It is not generated or evaluated here. It can be used once for a future locked
candidate, with a matched frozen incumbent; a new full training/selection
protocol is still required. It is a new grid of the same simulator, not an
independent physical-system confirmation or a hardware test.

## Outputs and decision

Save per-seed predictions, ensemble case errors, per-scenario/subset/horizon
metrics, component statistics, exact input/checkpoint/code hashes and environment
to a fresh directory. Refuse overwrite. Publish every fixed variant and negative
result. No winner is selected for deployment. Use the tradeoffs to formulate one
specific next training hypothesis and its required falsification conditions.

The APHYNITY source motivates controlling augmentation, but its separate local
pilot did not improve on matched controls; no literature superiority is claimed.
Primary reference: [Yin et al., ICLR 2021](https://arxiv.org/abs/2010.04456).
This audit is our diagnostic of the existing model, not a reproduction of that
paper's optimization method. All work remains local.
