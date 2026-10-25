# Real rehabilitation data requirements

**Use only institution-approved, de-identified or pseudonymized records under an approved data-use protocol.** Do not upload or commit patient records to this repository.

The supported flat export has one record per index patient. Required fields are a pseudonymous `Patient_ID`, age in years, injury type, severity, and enrollment/index date. Optional baseline predictors should include injury-region-appropriate primary ROM and strength measures, pain scale and instrument, balance/function measure, gait speed, comorbidities, and demographics with an explicit codebook. Outcomes should be independently recorded final PROMs (FIM, VAS pain, DASH, KOOS/LEFS as applicable), measured final ROM/strength, discharge date/duration, and any adverse events. Include a separate data dictionary identifying instrument/version, units, collection timepoint, source, allowed values, missing-value codes, and outcome-recorded dates.

Longitudinal adaptive-loop evaluation additionally requires patient-week records with dates, raw values, treatment/protocol actually received, pathway assignment, deviations/safety events, and final outcome links. Without those linkage/timing fields, model metrics may be calculated but adaptive efficacy and pathway effects cannot be established. The current validator rejects obvious direct-identifier column names; institutional de-identification review is still required.

## Public data leads and fit to this project

- **Osteoarthritis Initiative (OAI):** NIH's archive contains longitudinal knee-OA clinical, demographic, patient-reported, and imaging data. Access requires an NDA account and applicable data-use terms. It may support selected knee-OA natural-history/PROM analyses after checking the codebook and protocol, but it is not a general post-operative orthopaedic rehabilitation cohort and does not automatically supply the pathway exposure/adverse-event links required here. [OAI study portal](https://nda.nih.gov/oai), [access portal](https://nda.nih.gov/oai/query-download).
- **KneE-PAD:** Public EMG/IMU exercise recordings and execution-variation labels are available through Zenodo. It is useful for a separately scoped wearable signal or exercise-quality experiment, not for final recovery-percentage model training or treatment-effect validation. Review its license, metadata, and study protocol before use. [Zenodo dataset](https://zenodo.org/records/12112951), [data descriptor](https://doi.org/10.1038/s41597-025-04963-4).

These are leads, not datasets already obtained, vetted, mapped, or validated by this project. Do not combine incompatible cohorts or claim that either one meets the real-world rehabilitation validation request without a prespecified compatibility review.
