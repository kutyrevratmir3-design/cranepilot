# Experimental contract

## Fixed scope

The only measured dataset is Mendeley version 2, DOI 10.17632/x69hyyv3zw.2.
There is no new hardware collection. Existing records support retrospective
forecasting and identification. New-controller results are simulation-only.
The manuscript must preserve this distinction in its title, abstract and tables.

## Causality and information

The current measured angle is included. Forecast H is exactly H*0.01 seconds
after that measurement. A model's input record stops at its prediction origin.
No centered test smoother, test-tail frequency estimate or future actual
trolley trajectory is part of the operational forecasting information set.

Calibration of omega(L) uses only training scenario tails between 35 and 90 s.
These tails are descriptive identification intervals; they are not all assumed
to be physically unforced. Report spectral fit quality and actuator activity.
No historical all-data calibration coefficients are reused.

## Scenario separation

Whole scenarios are disjoint between train, validation and evaluation. Test
results are retrospective because this dataset has already been examined.
Do not claim an untouched test. Input scaling uses training data only; neural
early stopping uses the designated validation scenario only.

The first neural pilot uses three predefined seeds and fixed development
hyperparameters. Its projection penalty is an exploratory candidate. A final
claim needs additional validation across scenario folds, calibration budgets,
independent simulation mismatch and fair adaptive physical baselines.

## Comparison and reporting

All predictors within a history configuration have the same maximum available
angle history. Forecast models are compared on common targets and origins.
The seven-horizon neural predictor is multi-output. It is not represented as
the original scalar MATLAB network. The tree implementation also differs.

The active/quiet split uses a causal movement threshold at the forecast origin.
It is not a statement that the complete future horizon is unforced. Report the
whole-record score alongside regimes and individual held-out scenarios.

Equal scenario weighting is used for headline averages. Never count individual
overlapping windows as independent experiments. Three training seeds do not
make three independent physical measurements of a cable length.

## Authorship and provenance

The project builds on Ratmir Kutyrev's previous study and saved experimental
analysis. Preserve this provenance. Attribute the public measurements to their
creators, and disclose AI assistance in research code and manuscript preparation.
The human author must review the final claims before journal submission.

## Journal readiness

Passing code tests does not establish scientific novelty, controller safety,
industrial transfer or journal acceptance. Negative ablations are retained.
A verified failure to beat adaptive physics is a research finding, not a reason
to conceal the baseline or silently change the metric.

