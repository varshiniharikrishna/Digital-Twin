# Similarity metric specification — DRAFT

Feature space is `F=N ∪ C`; there is no configured ordinal feature set yet. `N` is the numeric subset and `C` the categorical subset in `config.py`. For each candidate feature `i`, `d_i=|x_i-y_i|/(max_train_i-min_train_i)` clipped to `[0,1]` for numeric values (a constant training feature uses 0 when equal and 1 otherwise); categorical mismatch is 0/1. Missing feature pairs are omitted and weights renormalized.

`d(x,y) = Σ_i w_i d_i / Σ_i w_i`, `s(x,y)=1-d(x,y) ∈ [0,1]`.

The range scaler is fit on training data and serialized with `SimilarityModel`. Feature weights are normalized absolute feature/outcome Spearman association learned on training data only; categorical levels use the largest one-vs-rest association. If outcomes/associations are absent, uniform weights are used and the fallback is recorded. This is exploratory predictive association, not a causal or clinical weight. `τ` is selected from a training-pair outcome-agreement curve using a 25th percentile of similarities among pairs within 20% of training target IQR. That agreement definition is an engineering choice and needs validation. Retrieval returns at most `K=5` patients with `s≥τ`; when fewer than `K_min=3` pass, result metadata is `low-confidence / insufficient cohort`, never fabricated fallback neighbors.

The CLI experiment compares learned feature weights, uniform weights, and old 0.6/0.4 component weights. See `reports/similarity/`. Neighbor outcome estimates use a similarity-weighted mean. Pathway results require observed patient-level pathway assignment and outcome linkage; template-level success-rate fields alone are not used.

Known limitations: weights/cutoff can be unstable in small or synthetic data; no external validation; no missingness-aware uncertainty calibration; outcome agreement is not clinically adjudicated.
