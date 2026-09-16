"""Collect all measured research variants without dropping unsuccessful runs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = []
for path in sorted((ROOT/'results').glob('*/eval_*/metrics.json')):
    report = json.loads(path.read_text())
    config = json.loads((path.parent.parent/'config.json').read_text())
    prediction = json.loads((path.parent/'prediction_metadata.json').read_text())
    rows.append({'source':config['dataset'],'variant':config['variant'],'seed':config['seed'],
                 'target':report['target'],'score':prediction.get('score','default'),
                 **{key:value*100 for key,value in report['mean'].items()}})
with (ROOT/'research/all_measured_results.csv').open('w',newline='') as stream:
    writer = csv.DictWriter(stream,fieldnames=['source','variant','seed','target','score','i_auroc','p_auroc','p_ap','aupro'])
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows,indent=2))
