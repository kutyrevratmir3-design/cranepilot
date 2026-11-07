# External real-camera benchmark, 2026-11-07

Preregistered source a52db9d182678407476c207bacee22efe69c6248.
External author repo https://github.com/ikhemakhem/BehavioralTrajGenCrane,
commit1aa26b5c358e885325f75cc0e89ef613ff4b26f3. Author validation_data records
were independently partitioned for this study: TRAIN1–12,VALIDATION13–16,TEST17–20.
Source schema/calibration and only TRAINrecord1 were inspected before preregistration.
All test predictions were generated after training/validation selection SHA freeze.

The independent test falsified frozen angular transfer on this domain adapter;
no model/default change followed. Locally fitted ARX/PEM/LSTM and frozen networks
have different training access. This is not an exact reproduction of Zhou PINN,
CoDIT SMILE LPV or author behavioral control, and not an actual closed-loop test.

Commands from main repository root through restored Python3.12 launcher:

```sh
python -m scripts.fetch_external_crane
python -m scripts.run_external_benchmark --stage train
python -m scripts.run_external_benchmark --stage evaluate
python -m scripts.verify_external_benchmark
python -m scripts.analyze_external_benchmark
python -m scripts.report_external_benchmark
python -m scripts.package_external_benchmark
```

Use `sh ../../handoff/restoration_20261107/main-python.sh` in place of python.
Scripts preserve earlier attempts. Runner accepts --output for a fresh run;
verifier/report/analyzer/package RUN/OUT constants refer to the recorded stage.

`frozen_run` is a byte-for-byte copy of the run: all checkpoints, selection,
source manifest, metrics and derived numerical predictions. NPZ contains origins
and model forecasts, no author MAT/source files. The persistence forecast encodes
the observed current angle after training-only centering. Original raw MAT files
stay in ignored local cache and must be downloaded/verified via the pinned script
for replay. No source MATLAB is executed. Author code has no listed license;
this evidence archive excludes original MAT/source files and credits the source.

Camera-to-angle geometry is evaluated per frame. Author whole-record mean,
filtfilt, vision_data filtered angles and input-file simulated yd are not used.
Inputs are past observed angles/encoder positions and known prescribed profiles;
raw velocity/acceleration fields are prescribed values, not independent sensors.
Targets/timestamps are unavailable to predictors. Physical timestamps are irregular:
nominal.05s index steps differ from actual elapsed time. 9native20Hz observations
are held causally to41entries only for frozen network input, with no new precision.

`posthoc_*` files diagnose clock scaling and response to a synthetic zero packet;
they do not fit/evaluate a corrected predictor. First report draft retained before
adding the explicit negative-transfer summary and explanatory diagnostics.
All179tests passed;14784forecasts and all record/horizon metrics verified. Four
TESTrecords at one fixed rope/load are not a population-level SOTA result.
