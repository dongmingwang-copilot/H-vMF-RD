"""Fetch the pinned official CFRG source; upstream files are not bundled."""
from pathlib import Path
import argparse,hashlib,json,requests

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    parent=Path(__file__).resolve().parent/"third_party/cfrg"
    record=json.loads((parent/"SOURCE.json").read_text())
    destination=parent/"upstream"
    base="https://raw.githubusercontent.com/EnquanYang2022/3CAD/"+record["commit"]+"/"
    with requests.Session() as session:
        session.trust_env=False
        for entry in record["files"]:
            relative=Path(entry["path"])
            target=destination/relative
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid upstream source path")
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==entry["sha256"]:
                continue
            response=session.get(base+entry["path"],timeout=(20,60))
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest()!=entry["sha256"]:
                raise ValueError("Source checksum mismatch: "+entry["path"])
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(response.content)
    print("CFRG source hashes verified:",record["commit"])
if __name__=="__main__":main()
