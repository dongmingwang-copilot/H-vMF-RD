"""Measure the exact FP16-SDPA/BF16 inference path used for 896 evaluation."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from common import ROOT, WORK, split_rows
from cache import Images
from encoder_sdpa import enable_sdpa
from network import EfficientRD, anomaly_map
sys.path.insert(0, str(ROOT / "code"))
from cache_features import load_encoder, encode_groups


def load_models():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    checkpoint = WORK / "runs/3cad_spatial_highres_s17/model_12000.pt"
    encoder = enable_sdpa(load_encoder("cuda"))
    model = EfficientRD().cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    return encoder, model, checkpoint


def predict_maps(encoder, model, images):
    with torch.autocast("cuda", dtype=torch.float16):
        teacher = encode_groups(encoder, images).to(torch.float16).float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return anomaly_map(teacher, model(teacher)).cpu().numpy()


def validate_saved_predictions():
    encoder, model, checkpoint = load_models()
    output = checkpoint.parent / "eval_3cad_12000"
    metadata = json.loads((output / "prediction_metadata.json").read_text())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metadata["checkpoint_sha256"] == digest
    assert metadata["encoder"] == "fp16SDPA"
    rows = split_rows("3cad", "test")
    stored_rows = json.loads((output / "rows.json").read_text())
    assert [r["id"] for r in rows] == [r["id"] for r in stored_rows]
    dataset = Images(rows[:16], 896)
    images = torch.stack([dataset[i] for i in range(16)]).cuda()
    with torch.inference_mode():
        actual = predict_maps(encoder, model, images)
    with np.load(output / "predictions.npz") as archive:
        expected = archive["maps"][:16]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    return {"checkpoint_sha256": digest, "images": 16,
            "row_ids": [r["id"] for r in rows[:16]],
            "bitwise_equal": bool(np.array_equal(actual, expected)),
            "max_absolute_difference": float(np.max(np.abs(actual - expected))),
            "rtol": 1e-5, "atol": 1e-6}


def measure(batch):
    encoder, model, checkpoint = load_models()
    rows = sorted(split_rows("3cad", "validation"),
                  key=lambda r: hashlib.sha256(("latency:" + r["id"]).encode()).hexdigest())[:16]
    dataset = Images(rows, 896)
    images = torch.stack([dataset[i] for i in range(batch)]).cuda()
    elapsed = []
    with torch.inference_mode():
        for i in range(150):
            if i == 50:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            raw = predict_maps(encoder, model, images)
            for arr in raw:
                dense = cv2.resize(arr, (896, 896))
                smoothed = cv2.GaussianBlur(dense, (0, 0), sigmaX=8., sigmaY=8.,
                                           borderType=cv2.BORDER_REFLECT)
                score = float(smoothed.max())
            torch.cuda.synchronize()
            duration = time.perf_counter() - start
            if i >= 50:
                elapsed.append(duration)
            del raw, dense, smoothed
    return {"method": "spatial_highres", "batch": batch, "size": 896,
            "encoder": "fp16SDPA", "decoder": "bf16; linear-attention accumulation float32",
            "seconds": elapsed, "gaussian_sigma": 8.,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "hardware": torch.cuda.get_device_name(), "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda, "warmup_batches": 50, "measured_batches": 100}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=[1, 16])
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.validate and args.batch is None:
        parser.error("--batch is required for timing")
    result = validate_saved_predictions() if args.validate else measure(args.batch)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ["seconds", "row_ids"]}), flush=True)


if __name__ == "__main__":
    main()
