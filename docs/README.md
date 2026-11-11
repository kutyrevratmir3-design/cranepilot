# Research notes

The [manuscript](../paper/README.md) describes the current delayed-angle study. The [paired reanalysis](../results/delayed_angle_review_20261111/README.md) compares the blend with its components and both ARX references. The [reference study](../results/delayed_baselines_20261111/README.md) adds a timestamp-aware Kalman filter and other baselines using the same previously examined recordings.

## Delayed angle measurements

| Study | Protocol or report | Saved evidence |
| --- | --- | --- |
| Dated-angle forecast | [Input contract and API](DATED_BUNDLE_20261108.md), [results (RU)](DATED_FORECAST_2026_11_08_RU.md) | [Forecast replay](../results/dated_forecast_20261108) |
| Dense ARX comparison | [Design](DENSE_REFERENCE_DESIGN_20261108.md), [results (RU)](DENSE_REFERENCE_2026_11_08_RU.md) | [Model selection and replay](../results/dense_reference_20261108) |
| Error by forecast horizon | [Horizon report (RU)](DENSE_HORIZONS_2026_11_09_RU.md) | [Paired results by record](../results/dense_horizons_20261109/paired_record_horizons.csv) |
| Error in the supplied angle age | [Design](TIMESTAMP_OFFSET_DESIGN_20261109.md), [results (RU)](TIMESTAMP_OFFSET_2026_11_09_RU.md) | [Offset replay](../results/timestamp_offset_20261109), [replay audit](../results/timestamp_offset_audit_20261109/verification.json) |
| Prospective confirmation | [Frozen intake and scoring protocol](FRESH_CONFIRMATION_20261108.md), [session coverage plan (RU)](FRESH_SESSION_DESIGN_20261109_RU.md) | [Planning calculation](../results/fresh_session_design_20261109/planning.json) |

Reports marked **RU** are in Russian.

## Earlier experiments

These archived reports describe the methods and publication status as recorded at each stage.

- The original cable-length transfer study is documented in the [archived benchmark card](BENCHMARK_CARD.md) and [model card](MODEL_CARD.md).
- [Passive rollout](PASSIVE_ROLLOUT.md), [driven innovation](INNOVATION_RESULTS.md), and [chronological adaptation](VALIDATED_ADAPTATION.md) describe precursor models.
- [November 3 campaign](RESEARCH_2026_11_03.md), [November 6 campaign](RESEARCH_2026_11_06.md), and [November 7 limit audit](RESEARCH_2026_11_07.md) retain earlier comparisons and counterexamples.
- [Aalto data audit](AALTO_DATA_AUDIT.md) and [external benchmark report (RU)](EXTERNAL_BENCHMARK_2026_11_07_RU.md) document separate data sources and their limitations.
