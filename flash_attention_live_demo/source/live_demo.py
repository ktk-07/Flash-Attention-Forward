from pathlib import Path
import os,sys,json,csv,subprocess,shutil,hashlib,time,uuid,statistics
from datetime import datetime,timezone

REPO_FILES=[
 'part_a/common/attention_types.hpp','part_a/common/input_generation.hpp',
 'part_a/cpu_reference/sampled_reference.cpp',
 'part_a/gpu_baseline/matmul_tiled.cu','part_a/gpu_baseline/softmax_parallel.cu','part_a/gpu_baseline/matmul_tensorcore.cu',
 'part_b/include/flash_attention/dense_flash_attention.h','part_b/src/dense_flash_attention.cu',
 'part_c/include/flash_attention/types.h','part_c/include/flash_attention/flash_attention_2_tensor_core.h',
 'part_c/src/flash_attention_2_tensor_core.cu']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path,label):
    path=Path(path)
    try:
        raw=path.read_text(encoding='utf-8-sig')
        if not raw.strip():raise ValueError('file is empty')
        value=json.loads(raw)
        if not isinstance(value,dict):raise ValueError('expected a JSON object')
        return value
    except (OSError,ValueError) as e:
        raise RuntimeError(f'{label}: cannot read valid JSON from {path}: {e}') from e

def write_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)
def command(args,env=None,timeout=60):
    r=subprocess.run([str(a) for a in args],env=env,capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError('Command failed: '+str(args[0])+'\n'+r.stdout[-2000:]+r.stderr[-2000:])
    return r.stdout.strip()

def compiler_environment():
    env=os.environ.copy()
    if os.name!='nt' or shutil.which('cl',path=env.get('PATH')):return env
    vswhere=Path(env.get('ProgramFiles(x86)',r'C:\Program Files (x86)'))/'Microsoft Visual Studio/Installer/vswhere.exe'
    if not vswhere.is_file():raise RuntimeError('Install Visual Studio C++ Build Tools, or start JupyterLab from an x64 Native Tools prompt.')
    # CUDA 12.x uses the supported VS2022 host rather than a newer VS18 preview.
    installation=command([vswhere,'-latest','-products','*','-version','[17.0,18.0)','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'])
    if not installation:raise RuntimeError('Visual Studio 2022 C++ Build Tools are required for this CUDA 12.x notebook build.')
    script=Path(installation)/'VC/Auxiliary/Build/vcvars64.bat'
    if not script.is_file():raise RuntimeError('Visual Studio x64 C++ toolchain not found.')
    r=subprocess.run('call "'+str(script)+'" >nul && set',shell=True,capture_output=True,text=True)
    if r.returncode:raise RuntimeError('Cannot initialize Visual Studio: '+r.stderr)
    for line in r.stdout.splitlines():
        if '=' in line:
            k,v=line.split('=',1)
            if k:env[k]=v
    return env

def prepare(repo,work,source):
    repo,work,source=map(Path,(repo,work,source));env=compiler_environment()
    for tool in ['cmake','nvcc','nvidia-smi']:
        if not shutil.which(tool,path=env.get('PATH')):raise RuntimeError(tool+' is required. See the presenter preparation notes.')
    report=command(['nvidia-smi','--id=0','--query-gpu=name,uuid,compute_cap','--format=csv,noheader'],env)
    name,gpu_uuid,cap=[v.strip() for v in report.splitlines()[0].split(',')]
    arch=cap.replace('.','')
    if int(arch)<80:raise RuntimeError('The PTX demonstration requires SM80 or newer.')
    hashes={n:sha(repo/n) for n in REPO_FILES}
    driver_hashes={n:sha(source/n) for n in ['demo_runner.cu','CMakeLists.txt','live_demo.py']}
    nvcc=command(['nvcc','--version'],env)
    identity={'repo_sources':hashes,'demo_sources':driver_hashes,'arch':arch,'nvcc':nvcc}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    build=work/'build'/key;build.mkdir(parents=True,exist_ok=True)
    executable=build/('demo_runner.exe' if os.name=='nt' else 'demo_runner')
    manifest_path=build/'build_manifest.json'
    cached=False
    if executable.is_file() and manifest_path.is_file():
        try:
            old=read_json(manifest_path,'Cached build manifest')
            cached=old.get('identity')==identity and old.get('executable_sha256')==sha(executable)
        except RuntimeError:
            print('Cached build manifest is incomplete; rebuilding and replacing it atomically.',flush=True)
    if not cached:
        generator='Ninja' if shutil.which('ninja',path=env.get('PATH')) else ('NMake Makefiles' if os.name=='nt' else 'Unix Makefiles')
        args=['cmake','-S',str(source),'-B',str(build),'-G',generator,
              '-DATTENTION_REPO='+repo.as_posix(),'-DCMAKE_BUILD_TYPE=Release',
              '-DCMAKE_CUDA_ARCHITECTURES='+arch,'-DCMAKE_POLICY_DEFAULT_CMP0141=NEW',
              '-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded']
        if os.name=='nt':args.append('-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler')
        log=build/'build.log'
        print('Preparing repository CUDA kernels for SM'+arch+'; first build may take several minutes.',flush=True)
        with log.open('w',encoding='utf-8') as f:
            for cmd in [args,['cmake','--build',str(build),'--parallel','2']]:
                f.write(json.dumps(cmd)+'\n');f.flush()
                r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
                if r.returncode:raise RuntimeError('CUDA build failed. Read '+str(log)+'\n'+log.read_text(encoding='utf-8',errors='replace')[-5000:])
        if not executable.is_file():raise RuntimeError('Build did not produce demo_runner.')
        write_json(manifest_path,{'identity':identity,'executable_sha256':sha(executable),'built_utc':datetime.now(timezone.utc).isoformat()})
    # Use a real output file instead of assuming stdout contains only JSON.
    device_file=build/('device_'+uuid.uuid4().hex+'.json')
    proc=subprocess.run([str(executable),'--device-json',str(device_file)],env=env,capture_output=True,text=True,timeout=60)
    diagnostic=f'GPU query executable: {executable}\nExit code: {proc.returncode}\nstdout: {proc.stdout[-2000:]!r}\nstderr: {proc.stderr[-2000:]!r}'
    if proc.returncode:raise RuntimeError('CUDA device query failed.\n'+diagnostic)
    try:
        device=read_json(device_file,'CUDA device query')
        for field in ['name','sm','total_bytes','driver','runtime']:
            if field not in device:raise RuntimeError('CUDA device JSON is missing '+field)
    except RuntimeError as e:raise RuntimeError(str(e)+'\n'+diagnostic) from e
    device['uuid']=gpu_uuid;device['reported_by_nvidia_smi']=name
    if str(device['sm'])!=arch:raise RuntimeError('CUDA device architecture differs from the selected nvidia-smi GPU.')
    git={}
    for label,args in [('commit',['rev-parse','HEAD']),('status',['status','--porcelain'])]:
        try:git[label]=command(['git','-C',repo,*args],timeout=10)
        except Exception:git[label]='unavailable; repository file SHA-256 hashes retained'
    session=work/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);session.mkdir(parents=True)
    provenance={'device':device,'build_manifest':read_json(manifest_path,'Build provenance'),'git':git,
                'repository':str(repo),'session_utc':datetime.now(timezone.utc).isoformat(),
                'historical_gpu':'RTX 3070 Ti Laptop GPU','live_results_are_new':True}
    write_json(session/'provenance.json',provenance)
    return {'repo':repo,'work':work,'exe':executable,'env':env,'device':device,'session':session,'cached_build':cached}

def run_case(context,n=512,seed=20261113,mode='ptx',b=4,h=8,repeats=20):
    if mode not in ('ptx','dense'):raise ValueError('Choose ptx or dense.')
    if not (16<=int(n)<=2048):raise ValueError('N must be between 16 and 2048 for this short demo.')
    output=context['session']/(mode+'_N'+str(n)+'_'+uuid.uuid4().hex[:8]+'.json')
    args=[context['exe'],mode,b,h,n,64,seed,repeats,output]
    started=time.perf_counter()
    proc=subprocess.run([str(a) for a in args],env=context['env'],capture_output=True,text=True,timeout=180)
    if not output.is_file():raise RuntimeError(proc.stdout+'\n'+proc.stderr)
    try:data=read_json(output,'Live CUDA result')
    except RuntimeError as e:raise RuntimeError(str(e)+f'\nExit code: {proc.returncode}\nstdout: {proc.stdout[-2000:]!r}\nstderr: {proc.stderr[-2000:]!r}') from e
    data['wall_seconds']=time.perf_counter()-started
    data['created_utc']=datetime.now(timezone.utc).isoformat();data['result_path']=str(output)
    data['gpu_uuid']=context['device']['uuid'];data['build_manifest']='provenance.json'
    data['command']=[str(a) for a in args]
    write_json(output,data)
    if proc.returncode or not data['passed']:raise RuntimeError('Fresh GPU validation failed; raw evidence retained at '+str(output)+'\n'+proc.stderr)
    for r in data['records']:
        if len(r['samples_ms'])!=repeats or not all(x>0 for x in r['samples_ms']):raise RuntimeError('Incomplete/invalid CUDA-event samples.')
    return data

def median(record):return statistics.median(record['samples_ms'])
