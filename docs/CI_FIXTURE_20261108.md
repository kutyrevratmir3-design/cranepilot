# Restore the frozen test fixture in clean checkouts

GitHub run [42470306321](https://github.com/kutyrevratmir3-design/cranepilot/actions/runs/42470306321)
at research commit `51219c0` failed in the unit-test step. The job log shows two
`setUpClass` errors, in `test_matched_adaptation` and `test_shared_response`:
`artifacts/runs/external_benchmark_20261107/selection.json` was missing.
Those tests call the historical `get_config()` and need its already frozen
oscillator parameters. The local file existed, so local tests did not reveal
this clean-checkout failure.

The fix tracks that exact 1109-byte JSON, despite the general run-directory
ignore rule. SHA256:
`33a9542cf2b6f5a4d2171ba9c2f3ad7aa783699f6000ed688b7581b278fd8153`.
It is byte-identical to the already public
`results/external_benchmark_20261107/frozen_run/selection.json`. It contains
selected settings and validation summaries, no raw records or correspondence.
No fit, parameter, historical source, test expectation or workflow is changed.

A clean `git archive` checkout is tested using the restored Python 3.12
environment. This verifies missing-file and source-hash behavior without
relying on the local ignored runs. Actual Linux CI remains a separate check.
