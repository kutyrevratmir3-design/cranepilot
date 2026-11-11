# Matched fine-tuning and control-inspired objectives, 7 November 2026

Freeze this design/config/new implementation/tests in a LOCAL commit before fitting.
This stage follows global_research_20261107 and the two new literature reviews.
The work changes optimization/which parameters adapt, not the input budget.
The existing LinearMPC scaffold does not use this neural predictor: this experiment
cannot establish any improvement in closed-loop control, safety or hardware.

## Source and fixed comparison

Every new variant starts from the same fold/seed soft_symmetry checkpoint from
broad_search_20261107, with exactly symmetry_mix=.5. Architecture and frozen actor
unchanged. Three original seed checkpoints, three folds;10 strategies=90 fits.
Fixed controls: incumbent, soft_symmetry, combined, physical, previous_local.
The last is the previous global study's per-fold frozen validation choice
(combined/LPV actor ridge.01/incumbent), protecting the new short-transfer gain.
No new random initialization. Variants sharing a seed use identical batch orders.

Same TRAIN/validation/evaluation/load IDs, train-only length calibration, train
stride25 and evalstride10. All forecast inputs:41 observations at10ms and30 known
command values at50ms. Actual future theta/x are supervision or metrics only.
Data have historical exposure. This is a retrospective screen, not a fresh test.
Equal record weighting in reporting, seven HORIZONS_S from frozen rollout module.
No extra synthetic training, actor fitting or architecture changes.

## Ten fixed strategies

All AdamW variants use lr.0003, decay.0001, max60epochs, patience15, batch256,
gradient norm clipping2. Base: smoothL1((prediction-target)/.1,beta=.1), averaged
over30 outputs/windows, plus exactly the prior .01 initial-memory-square and
.01 innovation-force-square penalties. Additional coefficients below are fixed
heuristics; their optimality is not claimed. AdamW and L2-SP penalties coexist.

1. continued: matched all-parameter continuation control.
2. observer_only: train initial and observer heads only (231 parameters).
3. innovation_only: train hidden_initial, innovation GRU and force_output only
(1093parameters). Encoder, state heads and passive core frozen.
4. l2sp: all parameters, +.01 SUM squared displacement from initial parameters.
This anchor is toward the useful checkpoint; separate weight decay remains.
5. function_anchor: +.5 mean squared normalized output difference from cached
starting-model predictions on TRAIN only. No teacher validation/evaluation target.
6. sam: gradient perturbation with global norm rho=.01; second pass supplies AdamW
gradient. Restore original weights before actual optimizer step even on failure.
Two backward passes per update; report actual compute rather than calling equal
epochs equal effort.
7. ema: ordinary all-parameter updates; exponential weight average decay.95 after
every update, initialized from starting weights. Only EMA snapshots compete for
checkpoint selection. No batchnorm state exists in this model.
8. phase_space: +.25 smoothL1(diff(prediction-target)/(omega*.05*.1), beta=.1)
and +.25 smoothL1(final error/.1,beta=.1). These are sampled increment/terminal
surrogates inspired by control costs, not exact angular-rate observations or
closed-loop policy gradients. Quantization can contaminate difference targets.
9. group_dro: persistent exponential group weights (eta=.05) for each TRAIN
scenario x active/quiet group. Minibatch group means, q renormalized over observed
groups; missing groups receive no fabricated zero-loss update. Report this
minibatch adaptation explicitly. Prior regularizers added once.
10. lbfgs_polish: full TRAIN batch from the already Adam-trained soft checkpoint;
L-BFGS lr.3, max_iter1, max_eval4, history10, strong-Wolfe search, max30outerepochs,
patience10. No gradient clipping in line-search closure. Report actual closure
and backward counts; max_eval is not a strict hard cap inside strong-Wolfe.
L-BFGS has no AdamW-style weight decay; the shared explicit memory/force
regularizers remain. Its full-batch order is fixed across outer steps.
This is a second-order polish experiment, not exact reproduction of the Koopman
paper's joint optimizer/schedule/architecture experiments.

Full parameter count8821. State/forcing partial adaptation serves as the small
model's parameter-efficient comparison; no unmotivated large-model LoRA claim.
No new future measured signal or online update across evaluation packets.

## Checkpoint selection and validation seal

Per seed: criterion=.5*(validation all MAE/starting-seed all MAE +
validation active MAE/starting-seed active MAE). Initial epoch-1 eligible.
All strategies share criterion and fallback; log selected epoch, epochs executed,
trainable/frozen counts, update/backward/closure counts, actual elapsed time.
An unchanged epoch-1 checkpoint is a rejected update, not successful tuning.

Each strategy is the mean of its three fixed seed predictions. Per-fold select
minimum validation all MAE among methods with all<=incumbent, active/quiet<=1.01
incumbent. Common strategy must qualify on every fold, then minimum mean
fold-normalized validation all MAE. Incumbent always a fallback; previous_local
is a retained selection rule, not a common single architecture.
Freeze all checkpoint SHA values and selection before evaluation/simulation.

Adoption gate for common strategy all/active: primary<=soft_symmetry;
short transfer<=1.01previous_local; long transfer<=1.01incumbent; primary load
<=1.01incumbent. Headline evaluation always lengths outside the fold's TRAIN
range; interpolation separate. Report every strategy/local choice and seed
variation, including regressions and unchanged checkpoints, regardless of gate.
No significance claim from overlapping packets or three seeded fits.

## Control-informed diagnostics and fresh simulation

In addition to fixed MAE: record mean per-packet maximum absolute error,
95th percentile of per-packet30step MAE, last10step MAE, and normalized increment
MAE |diff(error)/(omega*.05)|. Aggregate record-level diagnostics equally; do not
reinterpret mean record quantiles as pooled quantiles. They are predictor
diagnostics, not constraint violations, costs or settling times of a controller.

Fresh same-family forced angular simulation: lengths.24/.59/1.07,
amplitudes2/7.5deg, phase.7/2.1, omega ratios.84/1.03/1.2, damping.05/.21,
both force laws, zero/driven trolley, futuregain1/1.15, exact/noisyquantized
angles, seed11112026.1152cases/fold, opened only after selection.
All models get shared prescribed acceleration; actor bypassed only here.
Previous_local uses actor-based LPV on short fold, never a command-placeholder
model. Same simulator family is not independent hardware evidence.

## Verification and retained evidence

Run existing and new unit tests before training. Check freezing, anchor/EMA
mathematics, SAM restoration and failure paths, group weight evolution, phase
surrogate and L-BFGS closure behavior. Reload every selected saved seed checkpoint
against validation predictions. Replay five parent controls, poison targets,
rebuild independent history-only packets, test simulation integration refinement
and futuregain invariance. After completion replay every real forecast and all
seed arrays from weights. Original1281files main/APH snapshot remains unchanged.
No GitHub publication, no external message, no APHYNITY mixing.

