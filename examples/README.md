# History-only forecast example

The packet contains three past-only observation windows from public validation
scenario 4 at origins 4, 8.5 and 15 s. Histories have 41 newest-first values at
100 Hz. Future reference commands have 30 values at 50 ms, starting at the
origin command. There are no future measured angles or positions in the NPZ.

Angle is radians, x and length are metres, and reference retains its native
recorded scale. This is a usage/parity example, not an additional independent
accuracy test. history_forecast.csv is the compiled three-seed model output;
seed spread is not calibrated confidence.

Original data DOI: 10.17632/x69hyyv3zw.2, CC BY 4.0. Dataset article:
10.1016/j.dib.2025.111822. Cite the original creators when reusing the data.

## November 6 history-only entry point

The same `history_packet.npz` also works with the new multi-fold campaign:

```sh
python -m scripts.predict_temporal --input examples/history_packet.npz --output examples/NEW_FORECAST.csv
```

`history_forecast_20261106.csv` is the retained primary selected forecast, which
still uses the incumbent. Pass `--fold long_transfer --variant temporal_horizon`
for the separately trained experimental long-transfer model. These are different
training folds, not interchangeable checkpoints of one universally better model.
No future measured positions or angles are used; seed spread is not a calibrated
confidence interval. See the model card and November 6 report.
