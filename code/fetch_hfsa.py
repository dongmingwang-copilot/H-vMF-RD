"""Fetch the exact official HFSA source files used by the comparison."""
from pathlib import Path
import hashlib,json,requests
out=Path(__file__).resolve().parent/"third_party/hfsa"
record=json.loads((out/"SOURCE.json").read_text())
session=requests.Session();session.trust_env=False
for name,entry in record["files"].items():
    p=out/name
    if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==entry["sha256"]:continue
    response=session.get(entry["url"],timeout=60);response.raise_for_status()
    assert hashlib.sha256(response.content).hexdigest()==entry["sha256"],name
    p.write_bytes(response.content)
print("HFSA source hashes verified")
