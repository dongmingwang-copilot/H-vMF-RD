"""Apply a trained directional RD model to an image at its native resolution."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from cache_features import Images, load_encoder, encode_groups
from model import DirectionalRD


def predict_image(checkpoint_path, image_path, device, score='default'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    size = config['size']
    encoder = load_encoder(device)
    decoder = DirectionalRD(config.get('base_variant',config['variant'])).to(device).eval()
    decoder.load_state_dict(checkpoint['model'], strict=True)
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    native_size = image.size
    prepared = image.copy()
    prepared.thumbnail((1024,1024), Image.Resampling.LANCZOS)
    transform = Images([],size).transform
    with torch.inference_mode():
        features = encode_groups(encoder,transform(prepared).unsqueeze(0).to(device)).half().float()
        context = torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
        with context:
            output = decoder(features)
            if 'statistics' in checkpoint:
                if score!='default':
                    raise ValueError('Residual checkpoints use their fitted residual score')
                from residual_rd import residual_score
                with torch.autocast(device.type,enabled=False):
                    raw = residual_score(features,output,checkpoint['statistics'],config['variant'])[0].cpu().numpy()
            else:
                raw = decoder.anomaly_map(features,output,score=score)[0].cpu().numpy()
    resized = cv2.resize(raw,(size,size),interpolation=cv2.INTER_LINEAR)
    smoothed = cv2.GaussianBlur(resized,(0,0),sigmaX=4.,sigmaY=4.,borderType=cv2.BORDER_REFLECT)
    native = cv2.resize(smoothed,native_size,interpolation=cv2.INTER_LINEAR)
    return {'variant':config['variant'],'score':score,'input_size':size,'image_score':float(smoothed.max()),
            'raw_map':raw,'native_map':native}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu',choices=['cuda','cpu'])
    parser.add_argument('--score',default='default',choices=['default','cosine','angular_standardized','angular_deviation'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = predict_image(args.checkpoint,args.image,torch.device(args.device),args.score)
    args.output.mkdir(parents=True,exist_ok=True)
    np.save(args.output/'anomaly_map.npy',result.pop('native_map'))
    np.save(args.output/'token_map.npy',result.pop('raw_map'))
    result.update(image=str(args.image),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  gaussian_sigma=4.,feature_cache_precision='float16')
    (args.output/'prediction.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
