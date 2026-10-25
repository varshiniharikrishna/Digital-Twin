# Draft technical claims for review

**Working draft only.** These claims are limited to the behavior implemented in this repository. They do not establish novelty, non-obviousness, enablement, inventorship, ownership, freedom to operate, or patentability. Have a patent professional compare them with the full disclosure, prior art, and the applicable jurisdiction before filing. Synthetic demonstration data are not experimental proof of clinical performance.

## Claims

1. **A computer-implemented method for updating a patient-specific rehabilitation state**, comprising: receiving a baseline record including patient, injury, and assessment features; encoding the baseline record as a state vector; deriving one or more engineered features from the state vector; generating, with one or more trained regression models, a recovery estimate and a recovery-duration estimate; receiving a time-indexed rehabilitation observation comprising at least one pain, range-of-motion, strength, balance, or walking-speed value; replacing a corresponding observed component of the state vector while retaining unobserved state components; recomputing the one or more engineered features from the updated state vector; and generating updated recovery and recovery-duration estimates from the updated state vector.

2. The method of claim 1, wherein the state update for component `j` is `x_(t+1,j)=o_(t+1,j)` when a valid observation for component `j` is received and `x_(t+1,j)=x_(t,j)` otherwise, followed by deterministic recomputation of derived features including at least one of a quadriceps-to-hamstring strength ratio, pain-to-range-of-motion ratio, mobility index, comorbidity score, or estimated recovery duration.

3. The method of claim 1, further comprising calculating patient similarity over numeric and categorical features, wherein numeric distance is the absolute feature difference divided by the range of that feature in a reference training population, categorical distance is zero for a match and one for a mismatch, missing feature comparisons are omitted, and overall distance is a weighted combination of mean numeric distance and mean categorical distance; wherein the implemented weights are 0.60 and 0.40, respectively.

4. The method of claim 1, further comprising ranking one or more pathway templates by a retrieval score calculated as `0.70 × mean similarity of up to ten retrieved reference patients + 0.30 × cohort compatibility`, wherein cohort compatibility is the mean of exact injury-severity and age-band matches, and wherein the retrieval score is not represented as a treatment-effect estimate.

5. The method of claim 1, further comprising comparing an observed pain value, range-of-motion value, or strength value with a corresponding predicted value, calculating signed percentage deviation as `100 × (observed − predicted) / max(|predicted|, ε)`, and generating a review flag when pain deviation exceeds a configurable positive threshold or range-of-motion or strength deviation is below the negative threshold, where `ε` is a positive denominator floor.

6. The method of claim 5, wherein the default configurable threshold is 15 percent and candidate thresholds are evaluated by reporting flag counts at 5, 10, 15, 20, and 25 percent, without characterizing those counts as clinical sensitivity or specificity absent adjudicated reference labels.

7. The method of claim 1, wherein generating a local feature attribution for a recovery estimate comprises applying a SHAP permutation explainer to the fitted recovery model relative to a background sample from the training feature population.

8. A system comprising one or more processors and memory storing instructions that, when executed, cause the system to perform the method of any one of claims 1-7.

## Scope and support limitations

The implementation does not identify treatment effects: pathway templates are not joined to patient-level treatment exposure, outcomes can be a derived strength-ROM proxy, the included records are synthetic, and no clinical calibration, bias evaluation, or prospective outcome study has been completed. The experimental language in these claims is limited to feature/state processing, model prediction, retrieval, and software-trigger behavior. Any claim to improved recovery, clinical reliability, or adaptive patient benefit requires appropriate real-world evidence and separate claim support.
