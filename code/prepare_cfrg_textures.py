from pathlib import Path
import argparse
argparse.ArgumentParser(description='Download and verify the official DTD texture source for CFRG.').parse_args()
import requests,hashlib,tarfile,time,json,shutil
ROOT=Path(__file__).resolve().parent.parent
folder=ROOT/'data/texture_source';folder.mkdir(parents=True,exist_ok=True)
archive=folder/'dtd-r1.0.1.tar.gz'
url='https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
expected='fff73e5086ae6bdbea199a49dfb8a4c1'
if not (folder/'COMPLETE').exists():
 if not archive.exists():
  partial=archive.with_suffix('.partial')
  with requests.get(url,stream=True,timeout=(20,60)) as response:
   response.raise_for_status()
   with partial.open('wb') as handle:
    total=0;start=time.time()
    for chunk in response.iter_content(1024*1024):
     handle.write(chunk);total+=len(chunk)
     if total%(50*1024*1024)<1024*1024:print(json.dumps({'downloaded_mb':total/2**20,'seconds':time.time()-start}),flush=True)
  partial.replace(archive)
 md5=hashlib.md5()
 with archive.open('rb') as handle:
  for b in iter(lambda:handle.read(8*1024*1024),b''):md5.update(b)
 assert md5.hexdigest()==expected,md5.hexdigest()
 count=0
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar:
   if member.isfile() and member.name.startswith('dtd/images/') and member.name.lower().endswith('.jpg'):
    target=folder/member.name
    assert folder.resolve() in target.resolve().parents
    target.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
    count+=1
 assert count==5640,count
 (folder/'COMPLETE').write_text(json.dumps({'url':url,'md5':expected,'images':count,'used_for':'CFRG training texture source; no test images'}))
 archive.unlink()
print('DTD_READY',flush=True)
