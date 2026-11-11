# Broad matched strategy screen — 7 November 2026

Predeclared before fitting. Local only. This continues the restored November 6
InnovationForecaster; it does not modify the separate APHYNITY pilot. The aim
is better fixed-length sway forecasting with known future native commands,
including active motion and physical-limit behavior. A broad SOTA or real-crane
controller claim is not supported by these experiments alone.

## Motivation and sources

The frozen-weight audit found a useful but physically damaging initial-angle
correction: removing it reduced undamped free-sway error while worsening real
validation. Removing learned forcing alone did not fix that limit. Test several
distinct inductive biases and learning objectives, with a matched continuation
control and retained TCN/temporal/physical comparisons.

- [Wang, Walters and Yu, Incorporating Symmetry into Deep Dynamics Models](https://arxiv.org/abs/2002.03061)
  motivates enforcing transformation behavior. Here a two-element sign-reflection
  projection is applied to angular forecasts; this is not their fluid-model reproduction.
- [Approximately Equivariant Networks for Imperfectly Symmetric Dynamics](https://arxiv.org/abs/2201.11969)
  motivates including a relaxed alternative because measurements and actuators
  need not obey perfect symmetry. Our relaxation is a fixed 50% blend, not their architecture.
- [Sagawa et al., Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731)
  motivates reporting and protecting minority regimes. Our equal active/quiet
  loss is fixed group balancing, not their adaptive group-DRO optimization.
- [Yin et al., APHYNITY](https://arxiv.org/abs/2010.04456) motivates controlling
  physical augmentation. Here nonlinear synthetic supervision is tested, not
  the original constrained minimum-augmentation algorithm.
- [Györök et al., Orthogonal projection-based regularization, L4DC 2025](https://arxiv.org/abs/2501.05842)
  was considered as an additional identification direction. It is not implemented
  or claimed as a beaten baseline in this screen; physical calibration stays frozen.

## Nine fixed strategies, 81 training runs

Each strategy starts from the same corresponding incumbent fold/seed checkpoint.
All three folds and seeds 0/1/2 are mandatory, yielding nine runs per strategy.

| Strategy | Change |
|---|---|
| continued_control | unchanged architecture, fresh optimizer and matched continuation budget |
| residual_bound | delta_q = tanh(h) × min(0.02 rad, past forced-observer residual RMS) |
| odd_symmetry | f_odd(x,a) = [f(x,a) - f(-x,-a)] / 2 |
| soft_symmetry | 50% original output + 50% odd projection |
| regime_balanced | equal active/quiet mean SmoothL1 training loss |
| physics_aug | original architecture plus fixed-weight nonlinear free-sway supervision |
| bound_physics | residual bound plus synthetic supervision |
| odd_physics | exact odd projection plus synthetic supervision |
| balanced_physics | equal regime loss plus synthetic supervision |

No parameter grid or post-test changes within this campaign. Every strategy and
seed remains in the report, including initial-checkpoint selections and failures.

Reflection negates angular context (including signed position/reference history),
initial q/p, future reference and supplied acceleration; frequency and residual
RMS stay unchanged. It is an angular-model property given acceleration. The
frozen real actuator is not forced to be equivariant, so this does not guarantee
end-to-end reflection symmetry. Exact rest is preserved by the projected angular
model, but general free-sway energy conservation is not guaranteed. Soft symmetry
and the residual bound alone give no exact equilibrium guarantee.

## Training and equal-budget meaning

Keep original primary, short-transfer and long-transfer splits, actors and
train-only length calibrations from temporal_20261106 provenance. Never mix them.
Original dataset/source/actor/angular hashes are verified. Development windows:
stride 25 for training, 10 for validation, same 41-sample causal history and
30-step prescribed-command contract. Measured future x/theta are labels only.
Activity is |past-derived trolley velocity| >= 0.005 m/s.

Each run uses a fresh AdamW optimizer, learning rate 0.0003, weight decay 0.0001,
gradient clipping 2, maximum 50 epochs, patience 12, batch size 256. Within a seed,
the same real minibatch order is used across strategies. Loss uses all 30 steps,
angle scale 0.1 rad, SmoothL1 beta 0.1 and the old 0.01 memory/forcing penalties.
The odd branches regularize both internal paths, avoiding cancellation of their
penalties. No actuator is trained. The compiled flow is differentiable and tested
against original outputs and parameter gradients in float64; fitting uses float32.

This matches maximum real-data passes and optimizer updates, not FLOPs or total
data: the odd variants evaluate two paths and augmented variants receive extra
simulated labels. Report actual epochs and wall-clock fit times. Initialization
weights are identical within fold/seed, while changed architecture can change
the initial prediction. Do not describe this as training all models from scratch.

Synthetic training uses only five lengths evenly spanning each fold's measured
training range. Fixed amplitudes 0.75/3/6 degrees, phases 0.2/1.2/2.2/3.2,
damping 0/0.05 per second, exact/quantized histories: 240 cases per fold. The
nonlinear RK4 generator supplies labels; history remains the only observed input.
Trolley/reference and prescribed acceleration are zero, so the actuator is
bypassed. Draw 64 synthetic cases per optimizer step with an independent fixed
RNG; add 0.1 × synthetic SmoothL1 loss. These are explicitly training examples,
not physical-limit evaluation cases. Their amplitudes and phases differ from
the reserved confirmation grid; this is still the same simulator family.

## Selection locked before test evaluation

For each seed, score checkpoints by the mean of two ratios: overall validation
MAE / original same-seed overall MAE, and active validation MAE / original
same-seed active MAE. Epoch -1 is eligible; no invented improvement is reported
if continued learning fails. All variants use this same stopping criterion.

Average three seed forecasts before ensemble scoring. A strategy is eligible
only if overall validation MAE does not exceed the original incumbent, and
active and quiet MAE are each at most 1.01 × the incumbent. Choose the eligible
strategy with lowest overall MAE, with incumbent always available as fallback.
Also declare one common strategy across all folds: eligible in every fold and
lowest mean normalized overall validation error. This refers to a training
strategy applied to separate fold models, not a single universal checkpoint.

The one-record validation sets and correlated windows limit robustness of this
screen. The 1% allowances are operational screening thresholds, not statistical
noninferiority confidence bounds. With nine candidates, validation overfitting
remains possible. Seed selection, horizon mixing and test-based tuning are forbidden.

Only after all folds finish, write selection.json and checkpoint hashes, then
permit a separate evaluation command. The complete original evaluation groups
and alternate-load stress records are scored for every strategy. Extrapolation
headlines exclude the additional folds' interpolation records. Retain per-scenario,
per-horizon, active/quiet metrics, all individual-seed predictions and old baselines.
Those real records have been reused historically: evaluation is retrospective.

Open the previously reserved grid from limit_audit_20261107_float64.json once,
after the selections are locked, for the primary fold only. Its 432 cases are
216 trajectories × exact/quantized histories. Report damping and history modes,
and verify RK4 step halving. No candidate or weights may change after seeing it.
It is additional simulation confirmation, not an independent hardware domain.

## Reproducibility and decision

Commit config/design/model/runner/tests locally before fitting. The runner refuses
existing output paths and uncommitted experimental sources. Save source/data
hashes, original provenance, environments, training traces, selected epochs,
validation predictions, selection manifest and all weights. Evaluation checks
the same code/input/selection hashes. Replay every final checkpoint after evaluation.

A successful method must show its tradeoffs against both the unchanged incumbent
and the continued control. Better simulator behavior alone does not justify
replacement. Neither does a lower mean that hides unacceptable active-regime
or length failures. Keep the current default unchanged until the full report
is reviewed against these predefined boundaries. No GitHub or Overleaf writes.
