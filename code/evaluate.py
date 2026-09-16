"""Cache low-resolution predictions and evaluate native public annotations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from model import DirectionalRD
from train import CachedFeatures
from metrics import PixelMetrics, image_auroc
from pipeline import prediction_complete

ROOT = Path(__file__).resolve().parent.parent


def category_metrics(payload):
    prediction_path, rows, positions, category, pro_limit, bins = payload
    cv2.setNumThreads(1)
    arrays = np.load(prediction_path)
    metadata = json.loads((Path(prediction_path).parent/'prediction_metadata.json').read_text())
    size = metadata['size']
    maps, scores = arrays['maps'], arrays['scores']
    lower = min(float(maps[p].min()) for p in positions)
    upper = max(float(maps[p].max()) for p in positions)
    accumulator = PixelMetrics(lower, upper, bins=bins)
    labels = []
    for position in positions:
        row = rows[position]
        resized = cv2.resize(maps[position], (size, size), interpolation=cv2.INTER_LINEAR)
        smoothed = cv2.GaussianBlur(resized, (0, 0), sigmaX=4., sigmaY=4., borderType=cv2.BORDER_REFLECT)
        if row['mask']:
            mask = np.asarray(Image.open(row['mask']).convert('L')) > 0
        else:
            mask = np.zeros((row['native_height'], row['native_width']), dtype=np.uint8)
        assert mask.shape == (row['native_height'], row['native_width'])
        accumulator.add(smoothed, mask)
        labels.append(row['label'])
    result = accumulator.summary(pro_limit)
    result['i_auroc'] = image_auroc(labels, scores[positions])
    result.update(category=category, images=len(positions), anomalous_images=sum(labels))
    return result


def predict(checkpoint_path, target, output, size, score_mode):
    torch.set_num_threads(8)
    config = json.loads((checkpoint_path.parent/'config.json').read_text())
    if config['variant'] == 'patchcore':
        from patchcore_runner import predict as predict_patchcore
        return predict_patchcore(checkpoint_path.parent,target,output)
    if config['variant'] in {'rd','rdpp'}:
        return predict_cnn(checkpoint_path,target,output)
    data = CachedFeatures(target, size, 'test')
    rows = [data.rows[i] for i in data.indices]
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    variant = checkpoint['config']['variant']
    model = DirectionalRD(checkpoint['config'].get('base_variant',variant)).cuda()
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    maps, scores, indices = [], [], []
    loader = DataLoader(data, batch_size=32, num_workers=4, pin_memory=True)
    with torch.inference_mode():
        for features, ids in loader:
            features = features.cuda(non_blocking=True).float()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(features)
                if 'statistics' in checkpoint:
                    from residual_rd import residual_score
                    with torch.autocast('cuda',enabled=False):
                        raw = residual_score(features,result,checkpoint['statistics'],variant).cpu().numpy()
                else:
                    raw = model.anomaly_map(features, result, score=score_mode).cpu().numpy()
            for raw_map in raw:
                resized = cv2.resize(raw_map, (size, size), interpolation=cv2.INTER_LINEAR)
                smoothed = cv2.GaussianBlur(resized, (0, 0), sigmaX=4., sigmaY=4., borderType=cv2.BORDER_REFLECT)
                maps.append(raw_map)
                scores.append(float(smoothed.max()))
            indices.extend(ids.tolist())
    assert indices == data.indices
    np.savez_compressed(output / 'predictions.npz', maps=np.asarray(maps, dtype=np.float32),
                        scores=np.asarray(scores), indices=np.asarray(indices))
    (output / 'rows.json').write_text(json.dumps(rows))
    (output / 'prediction_metadata.json').write_text(json.dumps({
        'checkpoint': str(checkpoint_path), 'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'target': target, 'score': score_mode, 'size': size, 'gaussian_sigma': 4., 'stored_maps': 'raw token maps',
        'image_score': 'maximum of smoothed input-resolution map',
        'feature_manifest_sha256': data.meta['manifest_sha256']}, indent=2))


def predict_cnn(checkpoint_path,target,output):
    from baselines import CNNReverse, BaselineImages, selected_rows
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model=CNNReverse(checkpoint['config']['variant']).cuda().eval()
    result=model.load_state_dict(checkpoint['model'],strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith('encoder.') for k in result.missing_keys)
    rows=selected_rows(target,'test')
    loader=DataLoader(BaselineImages(rows),batch_size=16,num_workers=4,pin_memory=True)
    maps,scores=[],[]
    with torch.inference_mode():
        for images,_ in loader:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                raw=model.anomaly_map(images.cuda()).cpu().numpy()
            for array in raw:
                resized=cv2.resize(array,(256,256),interpolation=cv2.INTER_LINEAR)
                smoothed=cv2.GaussianBlur(resized,(0,0),sigmaX=4.,sigmaY=4.,borderType=cv2.BORDER_REFLECT)
                maps.append(array)
                scores.append(float(smoothed.max()))
    np.savez_compressed(output/'predictions.npz',maps=np.asarray(maps,dtype=np.float32),scores=np.asarray(scores))
    (output/'rows.json').write_text(json.dumps(rows))
    (output/'prediction_metadata.json').write_text(json.dumps({'size':256,'gaussian_sigma':4.,
        'image_score':'maximum of smoothed input-resolution map','stored_maps':'raw feature maps',
        'target':target,'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()},indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--target', required=True, choices=['3cad', 'mvtec_ad2'])
    parser.add_argument('--size', type=int, default=280)
    parser.add_argument('--score', default='default', choices=['default', 'cosine', 'angular_standardized', 'angular_deviation'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--bins', type=int, default=262144)
    parser.add_argument('--predict-only', action='store_true')
    parser.add_argument('--metrics-only', action='store_true')
    args = parser.parse_args()
    run = ROOT / 'results' / args.run
    output = run / ('eval_' + args.target + ('_' + args.score if args.score != 'default' else ''))
    output.mkdir(parents=True, exist_ok=True)
    if not prediction_complete(output):
        assert not args.metrics_only, 'Complete predictions, row identifiers, and metadata must be generated first'
        predict(run / 'model.pt', args.target, output, args.size, args.score)
    if args.predict_only:
        return
    rows = json.loads((output / 'rows.json').read_text())
    categories = sorted({r['category'] for r in rows})
    tasks = [(str(output / 'predictions.npz'), rows, [i for i,r in enumerate(rows) if r['category'] == category],
              category, 0.05 if args.target == 'mvtec_ad2' else 0.3, args.bins) for category in categories]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = []
        for result in pool.map(category_metrics, tasks):
            results.append(result)
            print(json.dumps(result), flush=True)
    summary = {key: float(np.mean([r[key] for r in results])) for key in ['i_auroc','p_auroc','p_ap','aupro']}
    report = {'run': args.run, 'target': args.target, 'mean': summary, 'categories': results,
              'elapsed_seconds': time.time()-started, 'bins': args.bins,
              'aggregation': 'category macro-average; region pooling within each category'}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
