# Phase-exact passive memory: implemented development result

2 November 2026. Experimental forecasting and simulation-only control remain
separate. The new neural forecaster has not been evaluated inside MPC.

## Scientific question

Does a passive latent-memory closure improve cable-length transfer beyond
an otherwise identical causal observer and a strong GP residual? The retained
pilot does not establish that advantage. This remains an exploratory method,
with a verified numerical construction and mixed experimental predictions.

### Relevant prior work

- [Moradi et al., Output Error port-Hamiltonian NN](https://arxiv.org/abs/2502.14432):
  externally driven dissipative models, causal state encoding and section loss.
- [Gyorok et al., orthogonal augmentation](https://arxiv.org/abs/2501.05842):
  regularizing learned augmentation against the sensitivities of nonlinear physics.
- [Gyorok et al., orthogonal-by-construction augmentation](https://arxiv.org/abs/2511.01321):
  separating the representations of physical and learned input-output models.
- [Linares et al., controlled oscillation modelling](https://doi.org/10.1016/j.physd.2026.135341):
  power-preserving numerical integration inside port-Hamiltonian learning.
- [Menier et al., iLED](https://arxiv.org/abs/2309.05812): interpretable dynamics
  with memory motivated by Mori-Zwanzig and Koopman theory.
- [Yu et al., OnsagerNet](https://doi.org/10.1103/PhysRevFluids.6.114402):
  learning structured conservative and dissipative dynamics.
- [Canizares et al., SympFlow](https://arxiv.org/abs/2412.16787): learned
  compositions of Hamiltonian flow maps.
- [Zhou et al., crane PINN forecasting](https://doi.org/10.1016/j.aei.2025.103269):
  physics-informed sway prediction and simulation-to-experiment transfer.
- [Mao et al., hybrid crane MPC](https://doi.org/10.1109/JAS.2025.126002):
  physical model, GP correction and predictive control.
- [Kusznir and Smoczek, learned crane NMPC](https://doi.org/10.3390/app14125112).
- [CoDIT 2026 crane LPV identification](https://doi.org/10.1109/CoDIT70676.2026.11631165).
- [Original experimental dataset article](https://doi.org/10.1016/j.dib.2025.111822).

The general building blocks already have predecessors. The candidate-specific
question is whether a causal, phase-scaled memory model and a separate drive
model transfer usefully on these measurements. The small-angle restriction,
few independent records and prior use of the evaluation scenarios limit claims.

## Model construction

For fixed length, train-only calibration gives omega = sqrt(g/(a L+b)).
Set q = theta, p = theta_dot/omega, tau = omega*t, and augment the state with
four latent memory coordinates r. Their closure matrix is

```math
C=\begin{bmatrix}0&0&0\\0&-d&c^T\\0&-c&-\Lambda\end{bmatrix},
\qquad d\ge0,\quad\Lambda=\operatorname{diag}(\lambda_i)>0.
```

A two-layer network predicts bounded exchange coefficients and nonnegative
losses from causal context; coefficients are held fixed for a forecast window.
The observer sees 41 raw 100 Hz samples of angle, trolley position and command.
Current angle is retained as measured. A bounded correction is applied to
the initial phase velocity, with initial memory norm limited in proportion to
the physical state. This is not a measured hidden-mode identification.

The storage E = ||z|| squared / 2 satisfies, for prescribed dimensionless force f,

```math
\frac{dE}{d\tau}=-d p^2-r^T\Lambda r+p f,\qquad f=-\hat a_x/g.
```

Two Cayley closure half-steps surround exact forced oscillator rotation.
For each closure step of duration delta, with midpoint z_bar,

```math
z^+=(I-\delta C/2)^{-1}(I+\delta C/2)z,
\qquad E(z^+)-E(z)=-\delta(d\bar p^2+\bar r^T\Lambda\bar r).
```

The central rotation supplies work f*(q_after-q_before). This gives a discrete
balance for the entire composition and no unforced increase of total storage
in exact arithmetic. With closure disabled, rotation has exact physical phase.
Memory can still change the complete model's modes, and observer resets can
inject storage. No closed-loop or hardware safety result follows.

Eliminating fixed-coefficient r gives a passive convolution kernel
K(t)=sum(c_i squared * exp(-lambda_i*t)) plus a decaying initial-condition
term. This is a restricted memory class. Oscillatory hidden modes or fixed
structural timescales may require a different closure, to be evaluated separately.

## Inputs and actuator

The article labels xr as a displacement reference. The recorded pulse levels
and actual position do not establish exact trajectory tracking or a direct
acceleration input. The implementation therefore retains the reference in its
native scale and trains a small recurrent drive model on multi-step x error.
It does not relabel the reference as voltage or force.

Train scenarios are 3, 5 and 6; validation is scenario 4; evaluation is 1, 2,
7 and 8. All have fixed cable length. Hoisting is explicitly rejected by the
window builder. Initial trolley velocity uses a backward-only position fit.
Future commands are presumed prescribed in this conditional forecast task.
Future measured x and sway occur only among the targets.

The drive is frozen before angular training. On scenario 4, its 1 s position
MAE is 0.00243 m, versus 0.00486 m for constant velocity. At 0.1 s it is worse:
0.00025 m versus 0.00016 m. These are development observations, not a claim
that the drive path is fully identified.

## Retained experiment

Forecasts integrate at 50 ms and are scored every 0.1 s. All models share
origins. Training uses 0.1 s origins before 20 s and 0.5 s origins afterward;
this is an explicit training sampling choice. References are sampled at the
start of each 50 ms integration interval. This is not a continuous-input model.

PEPM and its zero-memory ablation use the same observer and predicted drive
accelerations. Three fixed seeds are averaged for each. PEPM has 7662
parameters; the zero-memory observer has 7266. The GP and direct MLP consume
the same available observations and prescribed commands, with the deterministic
drive rollout added to their features. They are direct forecast references,
not recursive plant models. The GP uses 512 training points and selects one
of four fixed length scales by validation MAE; it is not an unlimited GP search.

| Predictor | 0.1 s MAE | 0.5 s MAE | 1.0 s MAE |
|---|---:|---:|---:|
| Angle-only harmonic | 0.30055 | 0.57708 | 0.75884 |
| Action-conditioned physics | 0.24480 | 0.54582 | 0.75723 |
| Same observer without memory | 0.23502 | 0.54855 | 0.76697 |
| PEPM with four memory states | 0.22812 | 0.54943 | 0.76539 |
| GP action residual | 0.21553 | 0.50119 | 0.78495 |
| MLP action residual | 0.23302 | 0.52858 | 0.75616 |

Entries are degrees, averaged equally over four scenarios. The angle-only
row has less information. Its comparison with PEPM cannot isolate architecture.
The grid differs from the earlier 0.02 s benchmark, so their reported scores
must not be treated as the same experiment.

PEPM's short-horizon improvement over action-conditioned physics is real in
this retained pilot, but the GP is better at several short/intermediate
horizons. Long-horizon superiority is absent. The memory's improvement over
the same observer without memory is small and horizon dependent. No new
hyperparameters were selected from these evaluation scores.

All real records have been used in prior work. These are retrospective
development results. Multiple seeds and overlapping windows do not create
new independent physical trials. The initial nonlinear controller simulations
remain a separate physics-only experiment; PEPM is not yet inside MPC.

## Numerical and information checks

The project suite has 25 passing tests, including exact base rotation,
unforced storage contraction, forced balance, independent DOP853 convergence,
finite-difference gradients, batch independence, future-measurement mutation,
and exclusion of hoisting from the fixed-length protocol. The largest float32
storage-balance residual in the retained forecast runs is 8.4e-8, in normalised
storage units. Float64 analytic tests use a tighter tolerance. None measures
experimental accuracy by itself.

## Reproduction

See README for the four commands. Complete logs, weights and predictions are
retained locally under artifacts/runs/{actuator_development,passive_rollout_pilot,
rollout_references}. Public CSV summaries and source are retained under
results/latest and the source directories. Output directories cannot be
overwritten accidentally. The local run provenance records package-source
hashes; the post-run source snapshot additionally retains script hashes.

## Next scientific decision

Do not market the passive architecture as a new state-of-the-art predictor.
First test input-path uncertainty, constant versus context-conditioned memory,
and nonlinear/coupled dynamics using training/validation evidence. If a simpler
observer or GP explains the gain, retain that result and narrow the manuscript's
contribution. New independent simulation can support a confirmatory control
claim; existing recordings cannot validate hardware behavior of a new controller.
