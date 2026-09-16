"""Prepare public datasets on the cloud without altering official splits."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile
import zlib

import numpy as np
from PIL import Image
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def request(url, **kwargs):
    for attempt in range(6):
        try:
            result = requests.get(url, timeout=(20, 90), **kwargs)
            result.raise_for_status()
            return result
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 15))


def prepare_ad2():
    dataset = DATA / "mvtec_ad2"
    dataset.mkdir(parents=True, exist_ok=True)
    revision = "9a6d209dc83082a4ab0761c4792de957f56b053f"
    endpoint = os.environ.get('HF_ENDPOINT','https://huggingface.co').rstrip('/')
    base = f"{endpoint}/datasets/pjramg/MVTecAD2_FO/resolve/{revision}/"
    source_manifest = dataset / "samples_original.json"
    records = json.loads(source_manifest.read_text())["samples"] if source_manifest.exists() else request(base + "samples.json").json()["samples"]
    print("AD2 original split counts", Counter((r["category_label"]["label"], r["folder_type"]) for r in records), flush=True)

    def convert(row):
        category = row["category_label"]["label"]
        split = row["folder_type"]
        label = int(row["defect_label"]["label"] != "good")
        uid = row["_id"]["$oid"]
        receipt = dataset / 'prepared_records' / (uid + '.json')
        if receipt.exists():
            return json.loads(receipt.read_text())
        stem = f"{category}/{split}/{uid}"
        image_path = dataset / (stem + ".png")
        mask_path = dataset / (stem + "_mask.png")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = dataset / "raw" / row["filepath"]
        assert raw_path.resolve().is_relative_to((dataset/'raw').resolve())
        image_bytes = raw_path.read_bytes() if raw_path.exists() else request(base + row["filepath"]).content
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        native_width, native_height = image.size
        # Keep native-resolution reference masks; resize only model input images.
        if "segmentation" in row:
            binary = row["segmentation"]["mask"]["$binary"]["base64"]
            array = np.load(io.BytesIO(zlib.decompress(base64.b64decode(binary))), allow_pickle=False)
            mask = np.asarray(array > 0, dtype=np.uint8)
            assert mask.shape == (native_height, native_width), (uid, mask.shape, image.size)
            Image.fromarray(mask * 255).save(mask_path)
            assert bool(mask.any()) == bool(label), uid
        elif label:
            raise ValueError(f"Missing public mask: {uid}")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        image.save(image_path)
        result = {"id": "ad2_" + uid, "dataset": "mvtec_ad2", "category": category,
                "split": split, "label": label, "domain": row["shift_type"],
                "image": str(image_path), "mask": str(mask_path) if label else None,
                "native_width": native_width, "native_height": native_height,
                "source_path": row["filepath"], "source_sha256": hashlib.sha256(image_bytes).hexdigest()}
        with Image.open(image_path) as saved:
            saved.verify()
        receipt.parent.mkdir(exist_ok=True)
        receipt.write_text(json.dumps(result))
        if raw_path.exists():
            raw_path.unlink()
        return result

    manifest = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(convert, row) for row in records]
        for index, future in enumerate(as_completed(futures), 1):
            manifest.append(future.result())
            if index % 100 == 0:
                print(f"AD2 {index}/{len(records)}", flush=True)
    manifest.sort(key=lambda r: r["id"])
    assert len(manifest) == 3914
    assert len({r["source_path"] for r in manifest}) == len(manifest)
    temporary = dataset / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(dataset / 'manifest.json')
    print("AD2 complete", len(manifest), flush=True)


def prepare_3cad(archive_path=None):
    import gdown
    archive = Path(archive_path) if archive_path else DATA / '_downloads' / '3cad.zip'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        gdown.download(id="1BIX0H8TZp0wmrAnXPw8_aCAIX1j1Fzwz", output=str(archive), quiet=False, resume=True)
    print("3CAD archive bytes", archive.stat().st_size, flush=True)
    dataset = DATA / "3cad"
    dataset.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        entries = [n for n in bundle.namelist() if Path(n).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        print("3CAD image entries", len(entries), entries[:12], flush=True)
        manifest = []
        names = set(entries)
        def convert(entry):
            parts = Path(entry).parts
            if "train" not in parts and "test" not in parts:
                return None
            split = "train" if "train" in parts else "test"
            pivot = parts.index(split)
            category = parts[pivot - 1]
            label = int(parts[pivot + 1] != "good")
            image_bytes = bundle.read(entry)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            width, height = image.size
            relative = "/".join(parts[pivot - 1:])
            image_path = dataset / relative
            assert image_path.resolve().is_relative_to(dataset.resolve())
            image_path.parent.mkdir(parents=True, exist_ok=True)
            valid = image_path.exists()
            if valid:
                try:
                    with Image.open(image_path) as saved:
                        saved.verify()
                except Exception:
                    valid = False
            if not valid:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                temporary = image_path.with_suffix('.tmp.png')
                image.save(temporary)
                temporary.replace(image_path)
            mask_path = None
            if label:
                parent = "/".join(parts[:pivot] + ("ground_truth",) + parts[pivot + 1:-1])
                stem = Path(entry).stem
                candidates = [f"{parent}/{stem}_mask.png", f"{parent}/{stem}.png", f"{parent}/{Path(entry).name}"]
                mask_entry = next((p for p in candidates if p in names), None)
                if mask_entry is None:
                    raise ValueError(f"Missing 3CAD mask: {entry}")
                mask = np.asarray(Image.open(io.BytesIO(bundle.read(mask_entry))).convert("L")) > 0
                assert mask.shape == (height, width)
                assert mask.any(), mask_entry
                mask_path = image_path.with_name(image_path.stem + "_mask.png")
                if not mask_path.exists():
                    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            return {"id": "3cad_" + hashlib.sha256(entry.encode()).hexdigest()[:20],
                             "dataset": "3cad", "category": category, "split": split,
                             "label": label, "domain": "regular", "image": str(image_path),
                             "mask": str(mask_path) if mask_path else None,
                             "native_width": width, "native_height": height,
                             "source_path": entry, "source_sha256": hashlib.sha256(image_bytes).hexdigest()}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(convert, entry) for entry in entries]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    manifest.append(result)
                    if len(manifest) % 500 == 0:
                        print("3CAD converted", len(manifest), flush=True)
        assert len(manifest) == 27039, len(manifest)
        assert sum(r["split"] == "train" for r in manifest) == 10493
        temporary = dataset / 'manifest.json.tmp'
        temporary.write_text(json.dumps(sorted(manifest, key=lambda r: r["id"]), indent=2))
        temporary.replace(dataset / 'manifest.json')
    print("3CAD complete", len(manifest), flush=True)
    # The source archive is retained until image verification has completed.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["3cad", "ad2"])
    parser.add_argument('--archive',help='Optional path to the official 3CAD archive')
    args = parser.parse_args()
    prepare_3cad(args.archive) if args.dataset == "3cad" else prepare_ad2()
