import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from metrics import PixelMetrics,image_auroc

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
