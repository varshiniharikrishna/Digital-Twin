# Pathway ranking — DRAFT

For a candidate pathway `p`, the intended objective is

`J(p|x)=α E[recovery_gain|x,p] - β E[weeks|x,p] - γ P(pain_flare or adverse deviation|x,p) + λ LCB_evidence(p)`.

Each empirical term must be normalized on training-only data. `pathway_ranking.py` uses normalized mean gain, duration/52, observed risk with a Wilson lower bound, and `n/(n+n_min)` evidence support. Configurable weights default to 0.40/0.20/0.20/0.20 and must be prespecified/sensitivity-tested. No treatment effect is estimated unless actual patient-level pathway exposure is joined to outcomes. In the current bundled schema that linkage is absent, so all intervention effects remain marked `assumed (expert prior)` and no causal pathway recommendation is made. Candidate suitability heuristics are demonstration-only.
