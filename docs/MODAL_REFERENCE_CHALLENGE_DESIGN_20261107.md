# Post-hoc comparator robustness challenge

Timing is explicit: the seven conditional holdout errors have already been
observed. The frozen primary comparison is retained, and these records can
only be called retrospective for this stage. No new confirmation claim.

Observed concern: several full-input ARX transfer errors are large. Absolute
drive/reference levels may be irrelevant nuisance correlations. More input
features do not automatically make a stronger reference. Before writing a
scientific conclusion, challenge the modal gain against simpler physically
invariant ARX representations with the SAME81past angles and TRAIN/VAL records.

Four representations: theta_only (81 raw angles); angle_lpv (theta, omega,
theta×omega and theta×omega²); difference (theta−theta_current, current length,
omega); difference_lpv adds the two frequency interaction groups to difference.
No absolute x,J,xr,Jr levels. Direct difference models predict changes relative
to theta_current; other direct forms predict absolute angle. Each can instead
predict a residual over the same frozen calibrated harmonic forecast. Training
standardization and equal-record weights, six ridge values.48fits percase,
240fits total, all retained. Original frozen strongARX remains eligible fallback.

Select only by original validation MAE percase. Freeze these choices and hashes
before scoring old holdout records again. Modal weights/algorithm, task, origin
gate, metrics and input history do not change. Report whether the modal advantage
survives the stronger reference, even if it disappears. Do not relabel a losing
reference, cherry-pick its representations or replace the original primary table.

This challenge does not repair the lack of fresh independent author recordings.
Its purpose is to remove an avoidable weak-baseline objection before freezing
the method for those recordings.
