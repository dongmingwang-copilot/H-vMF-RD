import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"code/third_party/dinomaly"))
from dinov2.models.vision_transformer import vit_small

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
