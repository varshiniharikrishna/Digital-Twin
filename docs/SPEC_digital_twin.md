# Digital twin state — DRAFT

State `x_t=[pain, ROM, quadriceps strength, hamstring strength, gait speed, balance]`; static covariates `z` include age, sex/gender as available, injury type, severity and BMI. Per metric parameters `θ_m=(a_m,k_m)` define an interpretable asymptotic transition `f_m(x_t,u_t,z;θ)=a_m+(x_(m,t)-a_m)exp(-k_m Δt)`. `u_t` is a recorded treatment/protocol input; when no exposure data is present it is absent, not replaced by a claimed effect. Observation model is `y_t=h(x_t)+ε_t`, initially `h=identity`.

The update is scalar shrinkage: `K=P/(P+R)`, `x_(t|t)=x_(t|t-1)+K(y_t-h(x_(t|t-1)))`, `P'=(1-K)P`. With at least three observations, per-metric asymptote/rate are refit by bounded nonlinear least squares. `fit_hierarchical_priors` estimates cohort-median starting asymptote/rate parameters from training-patient longitudinal data only; the app/server pass those priors into a twin. `twin_trajectory_eval.py` evaluates early-week fitting against later observed weeks and a training-endpoint asymptotic comparator.

A twin is a time-indexed state with transition, observation, parameters, uncertainty and update history; a static patient table is not. This implementation is a proof-of-concept and the hand-coded app trajectory is still not a physiologically validated simulator. No patient benefit claim is supported.
