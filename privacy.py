"""Privacy utilities: identifier rejection, salted pseudonyms, audit events, k-anonymity."""
import hashlib, hmac, json, re
from pathlib import Path
import pandas as pd
SENSITIVE=re.compile(r"name|email|phone|mobile|address|mrn|medical.?record|ssn|date.?of.?birth|(^|_)dob($|_)",re.I)

def reject_direct_identifiers(frame):
    bad=[str(c) for c in frame.columns if SENSITIVE.search(str(c))]
    if bad: raise ValueError("Potential direct identifiers present: "+", ".join(bad))
    return True

def pseudonymize(identifier, salt):
    if not salt: raise ValueError("A deployment-managed secret salt is required.")
    return hmac.new(str(salt).encode(),str(identifier).encode(),hashlib.sha256).hexdigest()

def append_audit_event(path, user_id, action, subject_token=None, metadata=None):
    """Append minimal event without accepting free-text clinical values or direct IDs."""
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    allowed={"timestamp", "request_id", "model_version", "role"}
    clean={k:v for k,v in (metadata or {}).items() if k in allowed and isinstance(v,(str,int,float,bool))}
    event={"user_id":str(user_id),"action":str(action),"subject_token":str(subject_token) if subject_token else None,"metadata":clean}
    with p.open("a") as f:f.write(json.dumps(event)+"\n")

def k_anonymity(frame, quasi_identifiers, k=5):
    cols=[c for c in quasi_identifiers if c in frame]
    if not cols:return {"k":int(k),"minimum_group_size":0,"passed":False,"reason":"no quasi-identifiers available"}
    minimum=int(frame.groupby(cols,dropna=False).size().min()) if not frame.empty else 0
    return {"k":int(k),"minimum_group_size":minimum,"passed":minimum>=k,"columns":cols}
