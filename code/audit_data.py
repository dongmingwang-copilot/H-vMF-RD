"""Validate public manifests and retained native masks before evaluation."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from PIL import Image

root = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/'code'))
from baselines import selected_rows
for dataset in sys.argv[1:] or ['3cad', 'mvtec_ad2']:
    path = root / 'data' / dataset / 'manifest.json'
    rows = json.loads(path.read_text())
    ids = [r['id'] for r in rows]
    assert len(set(ids)) == len(ids)
    hashes = defaultdict(list)
    for r in rows:
        hashes[r['source_sha256']].append(r)
        assert r['label'] == 0 or r['split'] in ['test', 'test_public']
    duplicate_groups = [g for g in hashes.values() if len(g)>1]
    cross_split = [g for g in duplicate_groups if len({r['split'] for r in g}) > 1]

    def verify(row):
        try:
            with Image.open(row['image']) as image:
                assert max(image.size) <= 1024
                image.verify()
            if row['mask']:
                with Image.open(row['mask']) as mask:
                    assert mask.size == (row['native_width'], row['native_height'])
                    mask.verify()
        except Exception as error:
            return {'id': row['id'], 'image': row['image'], 'source_path': row['source_path'], 'error': str(error)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        bad = [r for r in pool.map(verify, rows) if r is not None]
    train = selected_rows(dataset,'train')
    validation = selected_rows(dataset,'validation')
    test = selected_rows(dataset,'test')
    hashes_by_split=[{r['source_sha256'] for r in split} for split in [train,validation,test]]
    assert not hashes_by_split[0] & hashes_by_split[1], 'Optimization and validation share image content'
    assert not hashes_by_split[0] & hashes_by_split[2], 'Optimization and test share image content'
    assert not hashes_by_split[1] & hashes_by_split[2], 'Validation and test share image content'
    used={r['id'] for r in train+validation}
    excluded=[r['id'] for r in rows if r['split']=='train' and r['id'] not in used]
    report = {'dataset': dataset, 'images':len(rows),
              'split_counts': dict(Counter(r['split'] for r in rows)),
              'categories': dict(Counter(r['category'] for r in rows)),
              'native_shapes': dict(Counter(str((r['native_width'],r['native_height'])) for r in rows)),
              'duplicate_hash_groups': len(duplicate_groups),
              'cross_split_duplicates': [[r['id'] for r in group] for group in cross_split],
              'verified_images_and_masks': not bad, 'invalid': bad,
              'optimization_images':len(train),'validation_images':len(validation),
              'excluded_training_ids':excluded,'usable_training_pool':len(train)+len(validation)}
    (root / 'research' / (dataset+'_audit.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps({key:report[key] for key in ['dataset','images','split_counts','optimization_images','validation_images','verified_images_and_masks','invalid']}), flush=True)
    assert not bad, 'Invalid prepared files'
