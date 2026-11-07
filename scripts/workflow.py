"""Shared logging, device selection and immutable standalone run setup."""
import hashlib,importlib.util,json,os,shlex,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def device_default():return os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
def architecture(value,device,dry=False):
    if value and value!='auto':
        if not value.isdigit() or int(value)<80:raise ValueError('SM80 or newer is required for the selected Tensor Core kernels')
        return value
    if dry:return 'auto'
    text=subprocess.check_output(['nvidia-smi','-i',device,'--query-gpu=compute_cap','--format=csv,noheader'],text=True).strip()
    if len(text.splitlines())!=1:raise ValueError('Select exactly one GPU')
    result=text.replace('.','');return architecture(result,device)
class Workflow:
    def __init__(self,out,device,dry=False,profile=True):
        self.out=Path(out).resolve();self.dry=dry;self.env=dict(os.environ,CUDA_VISIBLE_DEVICES=device)
        if self.out.exists():raise ValueError('Choose a fresh output directory')
        if not dry:
            for package in ['numpy','matplotlib']:
                if importlib.util.find_spec(package) is None:raise ValueError('Missing Python dependency: '+package)
            for name in ['cmake','nvcc','nvidia-smi','compute-sanitizer']+(['ncu'] if profile else []):
                if not shutil.which(name):raise ValueError('Missing tool: '+name)
            (self.out/'logs').mkdir(parents=True)
    def run(self,label,argv):
        argv=list(map(str,argv));print(label+': '+shlex.join(argv),flush=True)
        if self.dry:return
        started=datetime.now(timezone.utc).isoformat()
        with (self.out/'logs'/f'{label}.txt').open('w') as stream:
            result=subprocess.run(argv,cwd=ROOT,env=self.env,stdout=stream,stderr=subprocess.STDOUT)
        with (self.out/'commands.jsonl').open('a') as stream:
            stream.write(json.dumps({'label':label,'argv':argv,'cwd':str(ROOT),'started_utc':started,'exit_code':result.returncode,'CUDA_VISIBLE_DEVICES':self.env['CUDA_VISIBLE_DEVICES']})+'\n')
        if result.returncode:raise RuntimeError(label+' failed; see '+str(self.out/'logs'/f'{label}.txt'))

def device_info(device):
    fields=['uuid','name','driver_version','compute_cap']
    values=subprocess.check_output(['nvidia-smi','-i',device,'--query-gpu='+','.join(fields),'--format=csv,noheader'],text=True).strip().split(',')
    if len(values)!=len(fields):raise ValueError('Expected one physical GPU')
    return dict(zip(fields,[s.strip() for s in values]))
