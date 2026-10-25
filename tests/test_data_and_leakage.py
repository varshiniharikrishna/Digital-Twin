import pandas as pd
import pytest
from data_adapter import DataValidationError, validate_real_records, to_tables, load_real_records
from leakage_audit import audit_feature_names

def valid_frame():
    return pd.DataFrame({"Patient_ID":["PSEUDO-1","PSEUDO-2"],"Age":[31,47],"Injury_Type":["ACL Tear","Rotator Cuff Tear"],"Severity":["Moderate","Mild"],"Enrollment_Date":["2023-01-01","2024-02-01"],"Primary_ROM_deg":[75,80],"Primary_Strength_pct":[45,50],"FIM_Final":[90,100]})

def test_valid_real_schema_maps_tables_without_fabricating_targets():
    records=validate_real_records(valid_frame());tables=to_tables(records)
    assert len(tables["Patient"])==2
    assert pd.isna(tables["Final_Outcome"].get("Final_Recovery_Pct",pd.Series(dtype=float))).all() if "Final_Recovery_Pct" in tables["Final_Outcome"] else True
    assert tables["Weekly_Progress"].empty

def test_adapter_rejects_identifiers_and_bad_age():
    f=valid_frame();f["patient_email"]="a@example.test"
    with pytest.raises(DataValidationError,match="identifiers"):validate_real_records(f)
    f=valid_frame();f.loc[0,"Age"]=190
    with pytest.raises(DataValidationError,match="Age"):validate_real_records(f)

def test_missing_path_never_generates_data(tmp_path):
    with pytest.raises(FileNotFoundError,match="No synthetic data"):load_real_records(tmp_path/"missing.csv")

def test_leakage_audit_rejects_weekly_feature():
    with pytest.raises(ValueError,match="leakage"):audit_feature_names(["Age","Week_1_Strength"])
