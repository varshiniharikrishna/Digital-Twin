# Patent claims draft — DRAFT – requires patent attorney review

**1. System (independent).** A computer-implemented rehabilitation analytics system comprising one or more processors and memory storing instructions that cause the system to: maintain a time-indexed patient state vector comprising recorded rehabilitation measurements and static patient covariates; generate an outcome estimate using a model trained on records separated from a later evaluation cohort; retrieve comparison records using a weighted, training-fitted per-feature Gower distance; compare a predicted observation with a subsequent observation using a standardized, direction-aware residual threshold; update the patient state by a prediction-observation shrinkage update; and present an updated estimate and an explanation identifying the measurements contributing to a trigger, wherein the system labels unsupported pathway effects as assumptions.

**2. Method (independent).** A computer-implemented method comprising: receiving a de-identified patient record; checking the record against a declared schema; deriving baseline-only model inputs; calculating a similarity score `s=1-Σ_i w_i d_i/Σ_i w_i` using a scaler and weights fitted on training records; retrieving up to K records whose score exceeds a training-selected threshold; receiving a time-indexed measurement; calculating adverse standardized deviation; updating a state estimate and uncertainty by `K=P/(P+R)`, `x'=x+K(y-x)`; and returning a recalibrated forecast with a confidence/insufficient-cohort indication.

**3. Non-transitory computer-readable medium (independent).** A non-transitory computer-readable medium storing instructions to perform the method of claim 2.

**Dependent claims (proposed).** 4. The system of claim 1, wherein numeric distances use ranges fitted only on the training partition and categorical distances use mismatch indicators. 5. The system of claim 1, wherein similarity weights are derived from training-only feature/outcome association, with a uniform-weight fallback recorded in metadata. 6. The system of claim 1, wherein retrieval reports low confidence if fewer than K_min records exceed threshold τ. 7. The system of claim 1, wherein an adverse deviation is a direction-adjusted residual divided by a metric-specific residual scale and requires persistence across consecutive observations. 8. The system of claim 1, wherein the state transition uses a per-measure exponential asymptote and patient-specific recovery rate. 9. The system of claim 1, wherein pathway ranking uses outcome-linked observed exposures and penalizes limited sample support, and otherwise marks effect estimates as assumed. 10. The system of claim 1, wherein wearable observations are quality-screened and aggregated to a weekly interval. 11. The system of claim 1, wherein an explanation includes per-feature contributions to a prediction and to a neighbor similarity score.

## Traceability and validation boundary

| Feature | Code | Status |
|---|---|---|
| Baseline feature/target preparation | `data_prep.py`, `leakage_audit.py` | Software implemented; real schema validation pending external records |
| Similarity and confidence threshold | `similarity.py` | Training-fitted prototype; needs external validation |
| Deviation rule | `deviation_analysis.py`, `simulation.py` | Engineering scales until residual calibration/clinical labels exist |
| State update and transition | `twin_state.py` | Parametric proof-of-concept; comparative report generated only if weekly data supports it |
| Pathway objective | `pathway_ranking.py` | No outcome-linked pathway data in bundled schema; no effect claim |
| Recalibration | `simulation.py`, `eval_adaptive_loop.py` | Forecast replay only; no causal benefit evidence |
| XAI/wearable/federated | `explainability.py`, `sensors/`, `federated_demo.py` | Prototype/optional; see reports and limitations |

Unvalidated: clinical outcomes, causal treatment benefit, real-world generalization, real-data privacy certification, regulatory classification, group fairness, prospectively measured utility and all treatment-effect parameters. These claims must be narrowed or supported before filing.
