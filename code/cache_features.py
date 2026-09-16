import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE / "third_party" / "dinomaly"))
from dinov2.models.vision_transformer import vit_small


class Images(Dataset):
    def __init__(self, rows, size):
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]["image"]) as image:
            return self.transform(image.convert("RGB"))


def load_encoder(device):
    weights = ROOT / "pretrained" / "dinov2_vits14_reg4_pretrain.pth"
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        torch.hub.download_url_to_file("https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth", str(weights))
    encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-8, num_register_tokens=4,
                        interpolate_antialias=False, interpolate_offset=0.1)
    encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return encoder.eval().to(device).requires_grad_(False)


def encode_groups(encoder, images):
    x = encoder.prepare_tokens(images)
    layers = []
    for index, block in enumerate(encoder.blocks[:10]):
        x = block(x)
        if index >= 2:
            layers.append(x)
    return torch.stack([torch.stack(layers[:4]).mean(0), torch.stack(layers[4:]).mean(0)], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["3cad", "mvtec_ad2"])
    parser.add_argument("--size", type=int, default=280)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(8)
    weights = ROOT / "pretrained" / "dinov2_vits14_reg4_pretrain.pth"
    encoder = load_encoder('cuda')
    manifest_path = ROOT / "data" / args.dataset / "manifest.json"
    rows = json.loads(manifest_path.read_text())
    out = ROOT / "data" / args.dataset / f"features_{args.size}"
    out.mkdir(exist_ok=True)
    shape = (len(rows), 2, (args.size // 14) ** 2 + 5, 384)
    metadata = {"shape": list(shape), "dtype": "float16", "size": args.size,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "encoder_layers": [[2, 3, 4, 5], [6, 7, 8, 9]], "num_prefix_tokens": 5}
    if (out / "metadata.json").exists():
        assert json.loads((out / "metadata.json").read_text()) == metadata
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    start = int((out / "progress.txt").read_text()) if (out / "progress.txt").exists() else 0
    features = np.memmap(out / "features.dat", dtype=np.float16, mode="r+" if start else "w+", shape=shape)
    loader = DataLoader(Images(rows[start:], args.size), batch_size=args.batch, shuffle=False,
                        num_workers=8, pin_memory=True, persistent_workers=True)
    cursor = start
    timer = time.time()
    with torch.inference_mode():
        for image in loader:
            packed = encode_groups(encoder, image.cuda(non_blocking=True))
            assert torch.isfinite(packed).all()
            count = image.shape[0]
            features[cursor:cursor + count] = packed.cpu().numpy().astype(np.float16)
            cursor += count
            if cursor % 256 < args.batch or cursor == len(rows):
                features.flush()
                (out / "progress.txt").write_text(str(cursor))
                print(args.dataset, cursor, "/", len(rows), "images/sec", round((cursor - start) / (time.time() - timer), 2), flush=True)
    (out / "COMPLETE").write_text(str(cursor))


if __name__ == "__main__":
    main()
