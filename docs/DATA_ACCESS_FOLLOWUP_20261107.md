# Follow-up data audit after modal campaign

The previous goal turn made progress: it completed modal forecasts, challenged
the reference, verified a portable bundle and retained all negatives. Completion
remains unproven: the stronger ARX comparison has an interval spanning zero and
no newly acquired appropriate measurements have arrived in the worktree.

Additional primary-source leads checked:

| Source | Evidence and decision |
|---|---|
| [YWYucas/IRV-Datasets](https://github.com/YWYucas/IRV-Datasets/tree/f71bace80879067a05edbd655addbf338da771fe) | Pinned commit16July2025. LOC-1 workbook,580data rows, headers `freq/HZ,zeta/-,Y-acceleration/\|g\|`. README defines parameter configurations and peak vibration, not a chronological angular series. Header-only numerical-file audit with Git-blob/SHA256 verification retained in results/data_access_followup_20261107/irv_schema_audit.json. Not suitable for rolling prediction confirmation. No response rows fitted/evaluated; raw spreadsheet not republished. |
| [YWYucas/Datasets](https://github.com/YWYucas/Datasets) | Primary README also defines frequency/damping and maximum-vibration configuration tables for beam/arm platforms. Not an angular forecast time series. |
| [Haarhoff dissertation repository](https://zenodo.org/records/5841504), DOI10.5281/zenodo.5841504 | Zenodo describes code/data for crane-CMG-robot modeling and simulation, and explicitly frames physical construction-site realization as future work. This does not establish an independent physical payload-sway dataset. The290MBarchive was not downloaded or relabeled as measured data. |
| [Shehu et al.2026, ISA](https://www.sciencedirect.com/science/article/pii/S0019057826001990), DOI10.1016/j.isatra.2026.04.019 | Correct primary PII endpoint located. Abstract confirms experimental control results. Full/data-availability endpoint returned403; no raw numerical supplement verified. Do not infer data are downloadable or on-request solely from the abstract. |
| [Mendeley version1](https://data.mendeley.com/datasets/x69hyyv3zw/1) | Same experiment collection/authors/platform, published25March2025, preceding the already-used version2. A version change is not evidence of independent trials. Not treated as a new holdout. |

Targeted public search for Talapiden/Alhassan data still points to the existing
Mendeley collection or on-request papers documented in SAME_RIG_DATA_ACCESS.
The user can ask the authors; no agent messages or requests were sent. Metadata
from unrelated crane imagery, logistics simulation, or parameter-response tables
cannot replace new suitable measured angular trajectories.

Continue useful reliability development on opened records, with its retrospective
status explicit. It cannot by itself close the independent-confirmation requirement.
