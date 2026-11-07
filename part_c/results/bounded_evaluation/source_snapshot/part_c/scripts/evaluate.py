#!/usr/bin/env python3
"""Bounded launch tuning, frozen evaluation and matched vector-load comparison."""
import argparse,csv,hashlib,json,math,os,platform,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
VARIANTS={"skip":5,"vector":5,"wmma":4,"ptx":4}
CORE_CONFIGS=[[32,32,4,3,5],[32,32,4,2,4],[16,32,4,2,4],[32,16,4,2,4],[32,32,8,2,4]]
TC_CONFIGS=[[32,32,2],[16,32,1],[32,16,2],[16,16,1]]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2)+"\n")
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analyze-only',action='store_true');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.analyze_only:analyze(out);return
    if (out/'plan.json').exists():raise SystemExit('Choose a new output directory: existing studies are immutable.')
    plan={'seed':5206,'generator':'Part A SplitMix64, Q then K then V','mode':'causal','layout':'BHNd contiguous','warmups':5,'repeats':20,
          'core_configs':CORE_CONFIGS,'tensor_configs':TC_CONFIGS,'tuning_N':[256,1024],'main_N':[256,512,1024,2048,4096],'d':[32,64,128],
          'selection':'per variant and d: geometric mean of median latency over tuning N; tie by config index; freeze before main',
          'family_selection':'skip vs vector; wmma vs ptx, using tuning only; not an exhaustive kernel/config search',
          'correctness':'FP64 causal oracle; full small cases; 3 query rows/head on large cases plus all-output finite checks',
          'memory':'isolated process; explicit live Q/K/V/O ledger; cudaMemGetInfo before/after allocation and first launch is an observed delta, not true peak',
          'timing':'CUDA events around forward launch only; no allocation/copy/reference/resource queries; unprofiled; sequential cases; default clocks/cache',
          'limitations':['Two tuning lengths also occur in main grid; 512/2048/4096 held out from config selection.','No speedup comparison with historical Part A measurements from a different GPU.','FP16 Tensor Core and FP32 CUDA Core are different precision paths.']}
    dump(out/'plan.json',plan)
    binary=a.build.resolve()/'part_c_evaluation'
    sources={}
    for path in sorted((ROOT/'part_c').rglob('*')):
        if path.is_file() and 'results' not in path.parts and path.suffix in {'.cu','.cpp','.h','.py','.sh','.txt','.md'}:
            rel=path.relative_to(ROOT);target=out/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes());sources[str(rel)]=digest(path)
    dump(out/'manifest.json',{'binary':str(binary),'binary_sha256':digest(binary),'sources':sources,'python':sys.version,'platform':platform.platform(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    for name,cmd in [('gpu',['nvidia-smi']),('compiler',['nvcc','--version'])]:
        r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/f'{name}.txt').write_text(r.stdout)
        if r.returncode:raise RuntimeError(f'{cmd} failed')
    subprocess.run([str(a.build.resolve()/'part_c_input_contract')],check=True,stdout=(out/'input_contract.txt').open('w'))
    records=[]
    def case(experiment,kernel,config,s,mode='benchmark',pattern='random'):
        tag=f'{experiment}_{kernel}_c{config}_'+ '_'.join(map(str,s))+'_'+pattern
        file=out/'cases'/f'{tag}.csv';cmd=[str(binary),kernel,str(config),*map(str,s),pattern,mode,str(file)]
        start=time.time();r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        with (out/'commands.jsonl').open('a') as f:f.write(json.dumps({'argv':cmd,'exit_code':r.returncode,'elapsed_seconds':time.time()-start,'output':r.stdout,'experiment':experiment})+'\n')
        print(f'{experiment}: {r.stdout.strip()}',flush=True)
        if r.returncode:raise RuntimeError(f'case failed: {tag}; see commands.jsonl')
        rows=list(csv.DictReader(file.open()))
        for row in rows:row['experiment']=experiment;records.append(row)
        return statistics.median(float(row['forward_ms']) for row in rows)
    # Every configuration is checked before any selection; tails cover both 16/32 tiles.
    for kernel,count in VARIANTS.items():
        for config in range(count):
            for d in [32,64,128]:
                for n in [1,17,33,65]:case('validation',kernel,config,[1,1,n,d],'validate')
            for pattern in ['uniform','constant_v','rising','stress']:case('validation',kernel,config,[2,3,97,64],'validate',pattern)
    timings={};frozen={}
    for kernel,count in VARIANTS.items():
        frozen[kernel]={}
        for d in [32,64,128]:
            scores=[]
            for config in range(count):
                values=[case('tuning',kernel,config,[4,8,n,d]) for n in [256,1024]]
                score=math.exp(statistics.mean(math.log(x) for x in values));scores.append((score,config));timings[kernel,d,config]=score
            score,config=min(scores);frozen[kernel][str(d)]={'config':config,'geomean_ms':score}
    winners={}
    for family,variants in [('cuda_core',['skip','vector']),('tensor_core',['wmma','ptx'])]:
        winners[family]={str(d):min(variants,key=lambda k:frozen[k][str(d)]['geomean_ms']) for d in [32,64,128]}
    dump(out/'frozen_selection.json',{'variants':frozen,'families':winners,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'plan_sha256':digest(out/'plan.json')})
    # Complete progression at defaults, then tuned finalists. Cached equality is explicit.
    for d in [32,64,128]:
        for n in [256,512,1024,2048,4096]:
            for kernel in ['fa1','scalar','warp','tiled','skip','vector','wmma','ptx']:case('default_main',kernel,0,[4,8,n,d])
            for kernel in VARIANTS:
                cfg=frozen[kernel][str(d)]['config']
                if cfg:case('frozen_main',kernel,cfg,[4,8,n,d])
            cfg=frozen['skip'][str(d)]['config']
            # Match *all* launch knobs so this isolates scalar vs float4 loading.
            for kernel in ['skip','vector']:case('matched_load',kernel,cfg,[4,8,n,d])
            for family in winners:
                kernel=winners[family][str(d)];cfg=frozen[kernel][str(d)]['config']
                case('memory',kernel,cfg,[4,8,n,d],'memory')
    fields=list(records[0]);
    with (out/'measurements.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    analyze(out)

def analyze(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows=list(csv.DictReader((out/'measurements.csv').open()));frozen=json.loads((out/'frozen_selection.json').read_text());groups={}
    for r in rows:groups.setdefault((r['experiment'],r['kernel'],int(r['config']),int(r['N']),int(r['d'])),[]).append(r)
    summaries=[]
    for key,values in groups.items():
        r=values[0].copy();times=[float(v['forward_ms']) for v in values];r.update(median_ms=statistics.median(times),q1_ms=float(np.quantile(times,.25)),q3_ms=float(np.quantile(times,.75)),repeats=len(times) if r['mode']=='benchmark' else 0)
        r.pop('repeat');r.pop('forward_ms');summaries.append(r)
    with (out/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    def find(exp,k,d,n,c=None):
        found=[r for r in summaries if r['experiment']==exp and r['kernel']==k and int(r['d'])==d and int(r['N'])==n and (c is None or int(r['config'])==c)]
        if len(found)!=1:raise ValueError((exp,k,d,n,c,len(found)))
        return found[0]
    def final(k,d,n):
        c=frozen['variants'][k][str(d)]['config']
        return find('frozen_main' if c else 'default_main',k,d,n,c)
    figs=out/'figures';figs.mkdir(exist_ok=True)
    def save(fig,name):fig.tight_layout();fig.savefig(figs/f'{name}.png',dpi=180);fig.savefig(figs/f'{name}.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    ns=[256,512,1024,2048,4096]
    for name,variants,base in [('cuda_core_progression',['fa1','scalar','warp','tiled','skip','vector'],'fa1'),('tensor_core_progression',['wmma','ptx'],'wmma')]:
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,d in zip(axes,[32,64,128]):
            for k in variants:ax.plot(ns,[float(find('default_main',base,d,n)['median_ms'])/float(find('default_main',k,d,n)['median_ms']) for n in ns],marker='o',label=k)
            ax.set(title=f'd={d}, fixed defaults',xlabel='Sequence length',ylabel=f'Speedup over {base}');ax.set_xscale('log',base=2);ax.grid(alpha=.3)
        axes[-1].legend();save(fig,name)
    fig,axes=plt.subplots(1,3,figsize=(15,4));comparison=[]
    for ax,d in zip(axes,[32,64,128]):
        c=frozen['families']['cuda_core'][str(d)];t=frozen['families']['tensor_core'][str(d)]
        for k in [c,t]:ax.plot(ns,[float(final(k,d,n)['median_ms']) for n in ns],marker='o',label=f'{k} cfg {frozen["variants"][k][str(d)]["config"]}')
        for n in ns:comparison.append({'d':d,'N':n,'cuda_core':c,'tensor_core':t,'cuda_ms':final(c,d,n)['median_ms'],'tensor_ms':final(t,d,n)['median_ms'],'tensor_speedup':float(final(c,d,n)['median_ms'])/float(final(t,d,n)['median_ms'])})
        ax.set(title=f'd={d}: FP32 core / FP16 tensor',xlabel='Sequence length',ylabel='Forward latency (ms)',yscale='log');ax.legend();ax.grid(alpha=.3)
    save(fig,'frozen_family_comparison')
    with (out/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    fig,axes=plt.subplots(1,3,figsize=(15,4));ablations=[]
    for ax,d in zip(axes,[32,64,128]):
        speed=[]
        for n in ns:
            s=float(find('matched_load','skip',d,n)['median_ms']);v=float(find('matched_load','vector',d,n)['median_ms']);speed.append(s/v)
            ablations.append({'d':d,'N':n,'config':frozen['variants']['skip'][str(d)]['config'],'scalar_ms':s,'vector_ms':v,'vector_speedup':s/v})
        ax.plot(ns,speed,marker='o');ax.axhline(1,color='gray',linestyle='--');ax.set(title=f'd={d}, matched launch configuration',xlabel='Sequence length',ylabel='Scalar / vector latency');ax.grid(alpha=.3)
    save(fig,'matched_vector_load')
    with (out/'matched_ablation.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(ablations[0]));w.writeheader();w.writerows(ablations)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,d in zip(axes,[32,64,128]):
        for family in ['cuda_core','tensor_core']:
            k=frozen['families'][family][str(d)];ax.plot(ns,[int(find('memory',k,d,n)['explicit_live_device_bytes'])/2**20 for n in ns],marker='o',label=k)
        ax.plot(ns,[4*8*n*n*8/2**20 for n in ns],linestyle='--',label='FP32 S+P storage model')
        ax.set(title=f'd={d}',xlabel='Sequence length',ylabel='Explicit live device storage (MiB)',yscale='log');ax.legend();ax.grid(alpha=.3)
    save(fig,'device_memory')
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,(k,count) in zip(axes,VARIANTS.items()):
        data=[[math.exp(statistics.mean(math.log(float(find('tuning',k,d,n,c)['median_ms'])) for n in [256,1024])) for d in [32,64,128]] for c in range(count)]
        norm=np.asarray(data)/np.min(data,axis=0);im=ax.imshow(norm,aspect='auto',vmin=1);ax.set(title=k,xlabel='Head dimension',ylabel='Configuration');ax.set_xticks(range(3),[32,64,128]);fig.colorbar(im,ax=ax,label='Latency / best')
    save(fig,'tuning_heatmap')
    failures=[r for r in rows if r['status']!='PASS'];validation=[r for r in summaries if r['experiment']=='validation']
    fingerprints={}
    for r in rows:
        key=tuple(r[x] for x in ['B','H','N','d','pattern']);value=(r['input_digest'],r['rounded_input_digest'])
        if key in fingerprints and fingerprints[key]!=value:raise ValueError('input hash mismatch')
        fingerprints[key]=value
        expected=int(r['B'])*int(r['H'])*int(r['N'])*int(r['d'])*(10 if r['input_precision']=='FP16' else 16)
        if int(r['explicit_live_device_bytes'])!=expected:raise ValueError('allocation ledger mismatch')
    manifest=json.loads((out/'manifest.json').read_text());source_ok=all(digest(out/'source_snapshot'/path)==sha for path,sha in manifest['sources'].items())
    audit={'passed':not failures and source_ok,'validation_cases':len(validation),'benchmark_cases':sum(r['mode']=='benchmark' for r in summaries),'samples':sum(r['mode']=='benchmark' for r in rows),'memory_cases':sum(r['mode']=='memory' for r in summaries),'source_snapshot_hashes_match':source_ok,'input_hashes_consistent':True,'allocation_ledger_matches':True,'plan_hash_matches_frozen':digest(out/'plan.json')==frozen['plan_sha256']}
    dump(out/'audit.json',audit)
    report=['# Part C bounded evaluation','',f'Correctness: {len(validation)} full small/tail/pattern checks; {audit["benchmark_cases"]} sampled benchmark checks; {audit["memory_cases"]} sampled memory checks. All outputs checked finite.',
            '', 'Selection was frozen after tuning and before the final grid. Family winners are selected only among skip/vector and WMMA/PTX; fixed-default progression includes the earlier variants. Tuning lengths 256 and 1024 overlap the reporting grid; 512, 2048 and 4096 are held out.', '', '## Frozen configurations','', '| d | CUDA core | Tensor core |','|---|---|---|']
    for d in [32,64,128]:
        c=frozen['families']['cuda_core'][str(d)];t=frozen['families']['tensor_core'][str(d)];report.append(f'| {d} | {c}, config {frozen["variants"][c][str(d)]["config"]} | {t}, config {frozen["variants"][t][str(d)]["config"]} |')
    report+=['','## Held-out N=4096','', '| d | CUDA ms | Tensor ms | Core / tensor |','|---|---|---|---|']
    for r in comparison:
        if r['N']==4096:report.append(f'| {r["d"]} | {float(r["cuda_ms"]):.3f} | {float(r["tensor_ms"]):.3f} | {r["tensor_speedup"]:.2f}x |')
    report+=['','## Interpretation limits','','The matched-load study changes scalar versus float4 loading with the same launch configuration. Other progression steps combine changes and are not isolated causal ablations. Config tuning changes resource use without changing kernel algorithms.','', 'Memory plots show live explicit Q/K/V/O allocations, not DRAM traffic or total process memory. FP32 core uses 16 bytes/element; FP16 inputs with FP32 output use 10. The S+P line is an analytical materialized-storage model, not a measured Part A result. cudaMemGetInfo deltas include allocation granularity and concurrent device activity; only three observations are made. Resource occupancy is an API estimate, not achieved Nsight occupancy.','', 'Tensor Core paths use FP16 Q/K/V and round probabilities for PV; CUDA Core uses FP32. Error is checked against the corresponding rounded-input FP64 oracle. These are precision-labelled performance comparisons.','', 'See plan.json, commands.jsonl, cases/, measurements.csv, summary.csv, comparison.csv, matched_ablation.csv, manifest.json, frozen_selection.json and audit.json. Historical profiling remains a separate fixed-default study; its counters do not describe tuned configurations.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    if not audit['passed'] or not audit['plan_hash_matches_frozen']:raise RuntimeError('audit failed')
if __name__=='__main__':main()
