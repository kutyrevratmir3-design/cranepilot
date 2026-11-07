# Same-platform data access audit, 7 November 2026

Requested authors: Kulash Talapiden and Ahmad Bala Alhassan. Goal: new raw measurements from the Inteco/PCMC platform, distinct from the already reused 33-scenario source. Search performed on publisher pages, university records, public repository metadata and Mendeley. No messages sent and no access controls bypassed.

| Source | Verified access finding | Suitability |
|---|---|---|
| [Talapiden et al.,2026, combined NN input shaping/FLC](https://journals.sagepub.com/doi/10.1177/10775463251411778), DOI10.1177/10775463251411778 | Publisher explicitly says data available on request to corresponding author; no open numerical supplement found. Corresponding authors listed as Zaharuddin Mohamed and Ton Duc Do. | Strong candidate: tower crane with trolley/jib/hoisting, cylindrical and distributed payloads, affiliated NU PCMC researchers. Physical identity of the user's exact rig is not established by authorship alone. |
| [Alhassan et al.,2025, comprehensive experimental dataset](https://doi.org/10.1016/j.dib.2025.111822) and [Mendeley version2](https://data.mendeley.com/datasets/x69hyyv3zw/2) | Public version2 dated16May2025;33scenarios,99s at100Hz. This is the existing main dataset. | Valuable development data, NOT new confirmation. A new URL/version wrapper cannot make reused signals independent. |
| [Shehu/Alhassan/Talapiden et al.,2026, composite super-twisting](https://doi.org/10.1016/j.isatra.2026.04.019) | Article/abstract/experimental claims confirmed; primary full-text/data endpoint did not return an accessible dataset during this search. No public raw-file link verified. | Candidate for an author-provided independent set. Exact platform match and data availability remain unconfirmed. |
| [Alhassan et al.,2024, hybrid input shaping/FLC](https://onlinelibrary.wiley.com/doi/10.1111/exsy.13484) | Publisher explicitly says supporting data are available from corresponding author on reasonable request. | Related tower-crane work; university/rig identity differs or is unresolved. Do not equate it to the user's physical setup. |
| Talapiden et al., IROS2026 comparison of adaptive shapers | Title/author lead found in program indexes; authoritative complete article, data supplement and usable raw files not recovered. | Search lead only, not a confirmed downloadable dataset. |

Supplementary public search found camera/assembly/LiDAR and hydraulic-stress crane datasets. They do not directly supply this project's payload-sway/command contract; they are not substitutes merely because they concern cranes. The previously used external rotary-camera source has only20rawvalidation records and29processedtraining records in its pinned tree; no additional raw recording files were found there. The processed29 cannot be assumed causal without provenance of their filtering.

## Concrete data specification for an author request

This is a technical request brief, not a sent email. Ask for RAW numerical recordings from the above control experiments, distinct from x69hyyv3zw versions1/2, preferably including repeated trials per condition:

- Time stamps and acquisition/control sample periods, plus any missing/dropped sample flags.
- Measured sway theta/beta, trolley x, jib J, cable L; angles in radians and positions in metres with units documented.
- Command/reference channels and actuator outputs, with their actual meaning, units, baseline and delay (position target vs velocity/voltage/PWM).
- Prescribed future reference trajectory, if available, separated from measured future motion.
- Payload mass/shape, rig/sensor identifiers and calibration, controller/shaper identity and gains, initial conditions and external disturbance labels.
- Information about anti-alias filtering, resampling, zero-phase filtering or any other offline smoothing; unfiltered signals preferred.
- Multiple fixed lengths/payloads and some active manoeuvres, including data from physical trials not previously released in the33scenario dataset.
- Permission/license for research use and sharing code/aggregate metrics, and source citation; numerical data need not be republished if that is not permitted.

Use new recordings as sealed confirmation only after a model and comparator are frozen. Existing records and already opened external TEST17–20 stay development/retrospective. Do not transcribe plots into fake high-rate raw measurements. Requesting data from people requires the user's explicit instruction to send; ordinary literature search does not authorize contact.
