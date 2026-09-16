"""Download the pinned, publicly accessible MVTec AD 2 mirror."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import threading
import time
import requests

ROOT=Path(__file__).resolve().parent.parent
REVISION='9a6d209dc83082a4ab0761c4792de957f56b053f'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    destination=ROOT/'data/mvtec_ad2'
    (destination/'raw/data').mkdir(parents=True,exist_ok=True)
    endpoint=os.environ.get('HF_ENDPOINT','https://huggingface.co').rstrip('/')
    base=f'{endpoint}/datasets/pjramg/MVTecAD2_FO/resolve/{REVISION}/'
    metadata=destination/'samples_original.json'
    if not metadata.exists():
        response=requests.get(base+'samples.json',timeout=(20,120))
        response.raise_for_status()
        records=response.json()['samples']
        assert len(records)==3914
        temporary=metadata.with_suffix('.json.part')
        temporary.write_bytes(response.content)
        temporary.replace(metadata)
    records=json.loads(metadata.read_text())['samples']
    assert len(records)==3914
    state=threading.local()

    def transfer(row):
        target=destination/'raw'/row['filepath']
        assert target.resolve().is_relative_to((destination/'raw').resolve())
        receipt=destination/'prepared_records'/(row['_id']['$oid']+'.json')
        if target.exists() or receipt.exists():
            return
        if not hasattr(state,'session'):
            state.session=requests.Session()
        for attempt in range(6):
            try:
                with state.session.get(base+row['filepath'],stream=True,timeout=(20,120)) as response:
                    response.raise_for_status()
                    temporary=target.with_suffix(target.suffix+'.part')
                    with temporary.open('wb') as output:
                        for block in response.iter_content(1024*1024):
                            output.write(block)
                    temporary.replace(target)
                return
            except requests.RequestException:
                if attempt==5:
                    raise
                time.sleep(min(2**attempt,20))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(transfer,row) for row in records]
        for count,future in enumerate(as_completed(futures),1):
            future.result()
            if count%100==0 or count==len(records):
                print('Public AD2 images available',count,'/',len(records),flush=True)
    (destination/'STREAM_COMPLETE').write_text(str(len(records)))


if __name__=='__main__':
    main()
