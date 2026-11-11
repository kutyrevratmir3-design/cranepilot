# Structural transfer evidence, 2026-11-07

Preregistration: 540477f7be091293d3c58ceb80230df51c0bd00f.
Frozen run contains 54 development + 864 evaluation episodes and all 146880 decisions.
Configuration, plant, controller, tests, verifier and benchmark were committed before development.
No offline tuning, no neural training, APHYNITY untouched.

Commands executed from project root through the restored main Python 3.12 launcher:

```sh
python -m unittest discover -s tests -v
python -m scripts.run_structural_transfer --stage development
python -m scripts.run_structural_transfer --stage evaluate
python -m scripts.verify_structural_transfer
python -m scripts.benchmark_structural_transfer
python -m scripts.analyze_structural_transfer
python -m scripts.report_structural_transfer
python -m scripts.package_structural_transfer
```

Use `sh ../../handoff/restoration_20261107/main-python.sh` in place of `python`.
Scripts reject overwriting frozen attempts. Runner/verifier accept fresh --output paths;
benchmark/report/analyzer/package use the stage paths declared in source. Main source tree is required.

`frozen_run` is a byte-for-byte copy of artifacts/runs/structural_transfer_20261107.
Raw latent states have 6 entries [x,v,theta,theta_dot,a_motor,z]. Metric states have
5 entries [x,v,theta,theta_dot,net_acceleration], including friction. Raw samples 0..40
cover physical warmup -.4..0. Issued commands and measurement-only histories are saved;
hidden delayed inputs are never passed to a controller. Time-varying targets are explicit.
`id_*` traces are audited via sequential causal reconstruction, not consumed as replay inputs.

`acceptance.json` contains every global and family gate. `posthoc_*` files describe
additional distribution/strong-reference contrasts, never used for tuning/selection.
`report_initial_draft.*` retain the first complete report; final report adds explicit
strong-reference tradeoffs and shares cluster resamples between identical policies.
Independent metrics reconstruct friction/net acceleration without original helpers.
The full optimizer is not reexecuted by the replay verifier. Warm batch1 tests do rerun
solves from restored pre-tick state. Wall-clock timing does not alter simulated dynamics.

The evidence ZIP stays local; source/config/docs and full synthetic traces are tracked.
This is an acceleration-command simulation, not native actuator/hardware qualification.
