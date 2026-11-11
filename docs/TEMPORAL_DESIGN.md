# Temporal observation and loss alignment

6 November 2026. Written before this campaign's training/evaluation.

The retained force-aware model improves average performance but loses to the
earlier driven model on short horizons and the active-origin 1.5 s subset.
The next experiment changes two specific mechanisms: temporal representation
of the 0.4 s observation history, and alignment of the training loss with the
seven scored forecast horizons. Changing the objective can change tradeoffs;
it does not establish uniformly better dynamics.

## Fixed comparisons

- Incumbent: retained dense encoder, Huber loss at every 50 ms step.
- Dense/horizon: identical architecture, Huber loss at the seven scored steps.
- Temporal/uniform: causal dilated encoder, original all-step loss.
- Temporal/horizon: causal dilated encoder, seven-step loss.
- TCN reference: same causal history encoder, convolutional residual decoder
  over the forced physical forecast; seven-step loss, no storage identity.

The temporal encoder has four two-convolution residual blocks, 16 channels,
kernel size 3, dilations 1/2/4/8, channel dropout 0.1 and a 61-sample receptive
field over the available 41 samples. All padding is on the past side. The
physics core, latent forcing, state bounds and actuator remain unchanged.
The reference uses three decoder blocks with dilations 1/2/4 and sees only
commands/accelerations up to each predicted step, plus past observations.

Every new model gets three fixed seeds on each of the three existing splits:
36 angular fits. Actors and calibration are reused only from the identical
verified fold. Early stopping uses validation; one physics candidate per fold
is selected using ensemble validation MAE, with the incumbent in the pool.
This selection is written before opening this campaign's evaluation outputs.
Every variant remains in the final report, including any negative results.

The same eight recorded lengths have already influenced development. These
are retrospective comparisons, even with a locked configuration and separate
validation. Overlapping windows and repeated folds are not independent trials.
Primary alternate-load records 10/11/12 remain a separate stress group.

## Synthetic falsification check

`configs/cleanroom_20261106.json` fixes 576 cases before training: independent
nonlinear free-sway simulations over six lengths, four amplitudes, four phases,
three damping values, exact/quantized histories. Future acceleration is known
zero for every angular model. This tests whether learned corrections invent
spurious motion or spoil a simple known limit. It deliberately bypasses the
actuator and uses the train-calibrated effective length; success would not be
evidence of end-to-end or hardware generalization. Do not tune on these cases.

## Primary sources and scope

The TCN reference is an independent, task-adapted implementation of causal
dilated residual convolutions described by Bai, Kolter and Koltun (2018):
https://arxiv.org/abs/1803.01271 and https://github.com/locuslab/TCN.
Unlike their released building block it omits weight normalization and uses
channel dropout. It is a matched-information reference, not an exact
reproduction of their benchmark results or a new architecture claim.

The released ReSMILE identification toolbox was also inspected:
https://github.com/meco-group/resmile. It requires MATLAB Control System
Toolbox, OptiSpline/CasADi, YALMIP and Mosek, and fits interpolated state-space
models from local estimates. It has NOT been reproduced here and is NOT a
method we claim to beat. A convenient substitute must not be labelled ReSMILE.
