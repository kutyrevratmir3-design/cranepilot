# CranePilot fixed-length transfer benchmark

3 November 2026. Development benchmark, not evidence of best-in-literature status.

## Exact niche

Forecast a tower-crane payload's trolley-axis sway using 41 past samples of
angle, actual trolley position and native reference at 100 Hz, current fixed
cable length, and a prescribed future reference sampled at 50 ms. Forecast
horizons are 0.1, 0.2, 0.35, 0.5, 0.75, 1.0 and 1.5 s. No future measured
position, acceleration or angle is supplied. Unknown future operator actions,
hoisting and multi-axis motion are different tasks.

TRAIN: scenarios 3/5/6, lengths approximately 0.296/0.503/0.596 m. Validation:
scenario 4 at 0.401 m. Primary evaluation: scenarios 1/2/7/8 at approximately
0.099/0.196/0.712/0.809 m. Thus the primary transfer is outside the training
length range in both directions. Additional load/shape scenarios 10/11/12
at approximately 0.496 m are a separately reported stress group; they do not
constitute new length extrapolation and their trajectories also differ.

The public dataset is DOI 10.17632/x69hyyv3zw.2; original article DOI
10.1016/j.dib.2025.111822. All records have informed previous work. Repeated
development on these records does not produce an untouched confirmatory test.

## Fair scoring

Primary score: mean scenario MAE at each horizon, then equal mean over all
seven declared horizons. Also report every scenario, RMSE, worst-scenario
mean, active/quiet diagnostics and latency. The existing active label uses
a backward position-velocity fit exceeding 0.005 m/s at the origin. It says
nothing about whether the future horizon is unforced. Overlapping windows,
multiple horizons and neural seeds are not independent physical experiments.

All primary comparisons use origins starting at sample 40 with stride 10 and
complete 150-sample future targets. Reference models get the same available
information. Derived features may differ; no measured future is permitted.
Calibration/scaling fits TRAIN only, stopping and kernel choice use validation.
Three fixed neural seeds are averaged; no best evaluation seed is selected.
Full future commands are a conditional input assumption, not a prediction of
an unknown human operator. Input-conditioned models must remain distinguished
from angle-only comparisons.

## Current comparison set

- Train-calibrated harmonic and action-conditioned physical models.
- Original passive-memory PEPM and a matched-training control.
- Causal refined observer and recurrent driven-innovation model.
- Direct MLP and capped GP residual references.
- Current force-aware observer/robust-loss ablations: fixed protocol in
  configs/forceaware_campaign.json, all variants retained.
- Stronger 1024-point RBF and Matern GP references with fixed validation grids,
  including a compact physical-feature representation.

Broader hardware or published-paper numbers cannot be ranked against this
score without matching information sets, data splits and prediction timing.
The implementation is public in an existing repository, but the current
campaign is kept local under the user's publication preference.

## Literature context checked 3 November 2026

The following are relevant precedents, not reproduced results in this benchmark:

- Zhou et al., physics-informed load-sway prediction and transfer learning,
  https://doi.org/10.1016/j.aei.2025.103269.
- Data-driven LPV identification explicitly accounting for cable length and
  load-held-out validation, https://doi.org/10.1109/CoDIT70676.2026.11631165.
- Mao et al., physical dynamics plus GP residual and disturbance observer in
  experimentally evaluated MPC, https://ieeexplore.ieee.org/document/11595100.

Combining physics, an observer and a learned correction is not sufficient
novelty. To support a best-model claim, retain a strong matched baseline suite,
compare released implementations where reproducible, broaden transfer cases,
and define independent tests before selecting a final architecture. Until that
is done, claim only the exact retained benchmark result and its limits.
