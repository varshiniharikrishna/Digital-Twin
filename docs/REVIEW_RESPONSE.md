# Review response — DRAFT – requires patent attorney and domain-expert review

| # | Reviewer item | Files/reports | Status |
|---:|---|---|---|
| 1 | Real de-identified data | `data_adapter.py`, `schema/patient_schema.yaml`, `validate_real_data.py`, `reports/real_data/` | Needs real data / governance |
| 2 | Recovery model R² ≥ .65 | `model_benchmark.py`, `reports/model_benchmark/` | Partial; report actual R², synthetic is not clinical |
| 3 | Similarity metric | `similarity.py`, `docs/SPEC_similarity.md`, `reports/similarity/` | Partial; requires external validation |
| 4 | Deviation thresholds | `deviation_analysis.py`, `docs/SPEC_deviation.md`, `reports/deviation/` | Partial; no adjudicated deterioration label |
| 5 | Pathway ranking | `pathway_ranking.py`, `docs/SPEC_pathway_ranking.md` | Needs linked treatment/outcome data |
| 6 | Mathematical digital twin | `twin_state.py`, `docs/SPEC_digital_twin.md`, `reports/digital_twin/` | Prototype; external validation needed |
| 7 | Ablation | `ablation.py`, `reports/ablation/` | Partial; causal pathway/simulation isolation unsupported |
| 8 | Adaptive loop | `eval_adaptive_loop.py`, `reports/adaptive_loop/` | Forecast replay only; no patient benefit evidence |
| 9 | XAI | `explainability.py`, Streamlit Explain page, `reports/xai/` | Partial; optional SHAP and additivity checked |
| 10 | Ethics/privacy/regulatory | `privacy.py`, `docs/ETHICS_PRIVACY_REGULATORY.md` | Draft; requires expert/legal review |
| 11 | Wearables | `sensors/`, `docs/WEARABLES.md`, `reports/wearable_robustness/` | Simulated prototype only |
| 12 | Related work | `docs/RELATED_WORK.md` | Needs verified citations |
| 13 | Formal claims | `docs/PATENT_CLAIMS_DRAFT.md` | Draft; attorney review |
| 14 | TRL | `docs/TRL_ASSESSMENT.md` | Provisional TRL 3; expert review |
| 15 | Commercialization | `docs/COMMERCIALIZATION.md` | Draft; market data to be sourced |
