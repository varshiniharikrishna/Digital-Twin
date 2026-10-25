# Related work — literature map (draft)

This section identifies relevant methods and datasets; it is not a systematic review and does not establish novelty, clinical benefit, or patentability. Each reference supports only the topic stated.

## Patient-specific orthopaedic models and digital twins

- Johnson et al. developed a patient-specific biomechanical hip model for Legg–Calvé–Perthes disease, using MRI, musculoskeletal simulation, sensitivity analysis, and comparison with upright MRI. It is a mechanics-based, single-patient use case and differs from this project's outcome-prediction/state-update prototype. [Article](https://doi.org/10.1007/s11548-025-03553-4).
- Böhme et al. reported five orthopaedic trauma cases integrating patient imaging, motion capture, musculoskeletal modeling, and finite-element analysis. This illustrates a patient-specific biomechanical workflow; it does not validate the present model's rehabilitation forecasts. [Article](https://doi.org/10.1186/s13018-025-05830-1).

## Mixed-type patient similarity and cohort retrieval

- Gower defined a general similarity coefficient that accommodates mixed variable types. This is the mathematical basis for the project's numeric-range/categorical-mismatch distance, while the project's outcome-derived weights and retrieval threshold are additional exploratory design choices. [Gower, *Biometrics* (1971)](https://doi.org/10.2307/2528823).
- Wang et al. described an EHR system using patient similarity for treatment recommendations. That work motivates evaluating retrieved cohorts, but does not validate the feature set, threshold, or rehabilitation use in this repository. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/25762458/).

## Explainability

- Lundberg and Lee introduced SHAP as a unified additive feature-attribution framework. The implementation here uses SHAP where installed and checks additivity; attribution is a model explanation, not a causal explanation or proof of clinical validity. [NeurIPS paper](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).

## Wearable sensing and rehabilitation

- Kasnesis et al. released KneE-PAD, a dataset of EMG/IMU rehabilitation-exercise recordings; the published curated set identifies execution variations for 31 participants. It can support exercise/posture or sensor-ingest experiments, not recovery-percentage prediction or treatment-effect evaluation. [Dataset record](https://zenodo.org/records/12112951), [data descriptor](https://doi.org/10.1038/s41597-025-04963-4).
- A 2023 study validated a two-IMU knee-kinematics approach against an optoelectronic system for home-based rehabilitation movements. It supports exploring sensor measurement pipelines, not assuming a sensor's readings are interchangeable with clinician outcomes. [Article](https://doi.org/10.1016/j.sciaf.2023.e01676).

## Federated learning

- Dayan et al. evaluated federated clinical-outcome prediction across 20 institutions for COVID-19. It demonstrates a multi-site application, but the task, data modalities, governance, and scale differ from this repository's local simulated FedAvg example. [Nature Medicine article](https://doi.org/10.1038/s41591-021-01506-3).

## Relevance and gap

These references establish that the individual ideas have prior art and illustrate materially different, often more data-rich approaches. This repository's current claims must stay narrow: its Gower metric, state transition, wearable ingestion, XAI, and federated-learning code are prototypes; its bundled CSVs are synthetic; no cited study validates this system or demonstrates benefit to rehabilitation patients. A patent or journal submission needs a broader, database-searched, date-bounded review and attorney/domain-expert assessment.
