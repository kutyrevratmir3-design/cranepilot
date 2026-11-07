# Forced causal adaptation after the autonomous baseline finding

Freeze this design, config, candidate/physics/simulation/runner and tests locally
before measured validation or new simulation scoring. No neural or actor fitting.
All old records are retrospective; selection uses only designated validation
records, but neither validation nor evaluation is untouched historically.

The observer solves theta''+d theta'+omega²theta=-(omega²/g)x'' through the
position transformation z=theta+(omega²/g)x with exact piecewise-linear history
propagation. This avoids explicitly differentiating position twice. Fit two
homogeneous coefficients over the 41 past samples; endpoint velocity is the
same backward-only 21-sample position slope as the original model. Forecast
with exact damped transition and shared frozen-actor acceleration at50ms.
Adapting omega also changes the effective forcing gain; call it parameter
adaptation, not isolated frequency identification. The inferred damping is an
effective fit parameter and need not equal physical dissipation.

Controls: incumbent, soft_symmetry, previous combination, nominal physics.
New fixed strategies: damping-only; coarse frequency+damping; coarse with
16x frequency-prior penalty; fine21point frequency grid with same penalty;
three hybrids based on coarse adaptation. All damping grids[0,.1,.2,.4].
Score per packet: history MSE +lambda*Q²/41*((ratio-1)/.15)²; no future target.
Hybrids (S=soft, P=nominal physics, A=adaptive physics): A+(S-P),
A+g_old*(S-P), A+g_old*(S-A). g_old uses nominal observer RMS and tau=.5,
never the residual minimized during adaptation. Report selected ratios,
damping, residuals, score gaps and boundary hits, including failure cases.

All11 methods receive identical real history, future prescribed command and
actor. Actor uses no future actual trajectory. Per-fold choice: candidates with
validation all<=incumbent and active/quiet<=1.01incumbent, then minimum overall
MAE. Incumbent is fallback. Also lock common eligible strategy minimizing mean
normalized validation MAE across all3folds. Selection is saved and hashed before
evaluation and new simulation. No tuning or replacement after those results.

Real evaluation keeps original temporal splits,stride10, active threshold.005m/s,
seven horizons and equal scenario weighting. Report interpolation separately;
cross-length headline excludes it. Acceptance for the common chosen strategy:
primary all/active <=soft_symmetry; transfer all/active<=1.01incumbent; primary
load stress all/active<=1.01incumbent. These are operational gates, not statistical
confidence bounds. Also report local selections and all individual candidates.

Fresh nonlinear forced simulation: new lengths, amplitudes, phases, off-grid
frequencies and damping from config. Sinusoidal trolley motion in past (.7Hz,
phase.3), ZOH prescribed future acceleration. Both effective coupling omega_true²/g
and geometric1/L are tested. Truth future acceleration gain1 or1.15; predictors
receive unscaled schedule and never future true states. Histories and input
commands are identical within gain pairs. Force amplitude0 and.012m distinguish
free and forced motion; do not pool them for a claim about forced prediction.
Angle modes exact and noise.25bin then quantization. True angles remain targets.
288base trajectories ×2gain ×2measurement=1152cases/fold; targets differ between
fold calibrations. Direct acceleration is an angular-model diagnostic, not an
end-to-end actor evaluation. Every method shares this information advantage.

Scientific checks: d0 nominal reproduces old observer/rollout tolerance5e-5deg;
independent forced solution; autonomous equivalence; reflection and packet
independence; targets poisoned without changing prediction; old neural replay;
identical history/prescribed schedule across gain pairs; integrator .001 vs.0005s
within5e-5deg. Retain failed attempts with explicit cause and numerical fixes,
never loosen tolerances or tune methods on failures. Report per condition,force
law,drive,measurement,gain,horizon,length and casep95, not only a pooled score.

All original artifacts and APHYNITY are preserved. No public posting, no claim
of hardware generalization/SOTA/passivity/confidence from blended outputs.
