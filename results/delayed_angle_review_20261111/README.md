# Delayed-angle retrospective reanalysis

Run from the root of the repository with its plotting dependencies installed.

    python scripts/reanalyse_delay.py --output results/delayed_angle_review_20261111

The source table already averages over forecast origins, then over axes within a physical record. The reanalysis averages the seven forecast horizons and then the physical records with equal weight. It verifies all 882 original paired horizon rows, checks that the zero-age fallbacks coincide, and reproduces the archived aggregate means to within 10⁻¹² degrees. No weights, references, inputs or forecasts are changed.

- age_mae.csv gives mean MAE at every age and noise condition.
- record_age_mae.csv gives the seven individual record averages at each age and noise condition.
- age_horizon_mae.csv gives the equal-record means and paired record wins at each age, horizon and noise condition.
- paired_age_differences.csv gives the mean and median of the paired differences (blend minus each reference), record wins, and descriptive percentile intervals. The interval resamples the seven physical records as paired units 20,000 times with replacement (seed 20261111). These are sensitivity summaries of previously inspected records.
- hoisting_strata_mae.csv separates records containing hoisting from the other two records. This classification is by recording, so each stratum still includes all its forecast origins.
- earlier_arx_gain_concentration.csv compares the contribution of records 13, 32 and 33 with the remaining four. This is a post-hoc grouping. A fraction greater than one means the sum of improvements in the three records exceeds the full seven-record net improvement.

Both figures use the condition without added angle noise. In delay_compensation.pdf, the zero-age point is the shared synchronous fallback. No standalone bank prediction is made at zero age. In panel B, positive differences favour the component and negative differences favour the blend. The two interval series are shifted three milliseconds horizontally for visibility. In arx_reference_replay.pdf, positive record differences favour the blend. Solid circles mark records that contain hoisting.
