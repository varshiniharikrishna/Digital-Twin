"""Fail-closed audit for model feature declarations and derived features."""
import re
import config

POST_BASELINE = re.compile(r"(week|weekly|actual_|final_|discharge|outcome|recovery_pct_target|availability|timestamp|observed_after)", re.I)
SAFE_ENGINEERED = set()

def audit_feature_names(feature_names=None):
    names = list(config.ALL_FEATURES if feature_names is None else feature_names)
    findings = [name for name in names if POST_BASELINE.search(str(name)) and name not in SAFE_ENGINEERED]
    if findings:
        raise ValueError("Potential post-baseline leakage feature(s): " + ", ".join(map(str, findings)))
    return {"passed": True, "features_checked": names, "post_baseline_features": []}

if __name__ == "__main__":
    print(audit_feature_names())
