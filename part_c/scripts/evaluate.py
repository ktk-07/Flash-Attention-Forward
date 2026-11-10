#!/usr/bin/env python3
"""Bounded launch tuning, frozen evaluation and matched vector-load comparison."""
import argparse,csv,hashlib,json,math,os,platform,statistics,subprocess,sys,time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/part-c-matplotlib')
ROOT=Path(__file__).resolve().parents[2]
VARIANTS={"skip":5,"vector":5,"wmma":4,"ptx":4}
CORE_CONFIGS=[[32,32,4,3,5],[32,32,4,2,4],[16,32,4,2,4],[32,16,4,2,4],[32,32,8,2,4]]
TC_CONFIGS=[[32,32,2],[16,32,1],[32,16,2],[16,16,1]]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2)+"\n")
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--grid',choices=['full','smoke'],default='full');p.add_argument('--analyze-only',action='store_true');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.analyze_only:analyze(out);return
    if (out/'plan.json').exists():raise SystemExit('Choose a new output directory: existing studies are immutable.')
    plan={'seed':5206,'generator':'Part A SplitMix64, Q then K then V','mode':'causal','layout':'BHNd contiguous','warmups':5,'repeats':20,
          'core_configs':CORE_CONFIGS,'tensor_configs':TC_CONFIGS,'tuning_N':[256,1024],'main_N':[256,512,1024,2048,4096],'d':[32,64,128],
          'selection':'per variant and d: geometric mean of median latency over tuning N; tie by config index; freeze before main',
          'family_selection':'skip vs vector; wmma vs ptx, using tuning only; not an exhaustive kernel/config search',
          'correctness':'FP64 causal oracle; full small cases; 3 query rows/head on large cases plus all-output finite checks',
          'memory':'isolated process; explicit live Q/K/V/O plus FA1 m/l ledger; cudaMemGetInfo before/after allocation and first launch is an observed delta, not true peak',
          'timing':'CUDA events around forward launch only; no allocation/copy/reference/resource queries; unprofiled; sequential cases; default clocks/cache',
          'limitations':['Two tuning lengths also occur in main grid; 512/2048/4096 held out from config selection.','No speedup comparison with historical Part A measurements from a different GPU.','FP16 Tensor Core and FP32 CUDA Core are different precision paths.']}
    plan.update(kernels=['fa1','fa1_wmma','scalar','warp','tiled','skip','vector','wmma','ptx'],memory_baselines=['fa1','fa1_wmma'],fa1_baselines=['fa1','fa1_wmma'],fa1_wmma_implementation='kv_outer_wmma_v1',fa1_implementation='kv_outer_shared_fp32_v2',fa1_workspace_floats_per_query=2,reference_backend='part_a_shared_fp64',shared_oracle_linked=True,grid=a.grid,B=4,H=8,validation_N=[1,17,33,65])
    if a.grid=='smoke':plan.update(B=1,H=1,d=[64],main_N=[33,65],tuning_N=[17,33],validation_N=[1,17,33])
    dump(out/'plan.json',plan)
    binary=a.build.resolve()/'part_c_evaluation'
    sources={}
    for path in sorted((ROOT/'part_c').rglob('*')):
        if path.is_file() and not path.resolve().is_relative_to(out) and 'results' not in path.parts and 'extra' not in path.relative_to(ROOT/'part_c').parts and path.suffix in {'.cu','.cpp','.h','.py','.sh','.txt','.md','.json'}:
            rel=path.relative_to(ROOT);target=out/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes());sources[str(rel)]=digest(path)
    # Contract-test headers are read-only Part A dependencies, not a Part A run.
    for name in ['part_a/common/input_generation.hpp','part_a/common/attention_types.hpp','part_a/cpu_reference/attention_reference.cpp','part_a/cpu_reference/sampled_reference.cpp','part_a/cpu_reference/CMakeLists.txt']:
        path=ROOT/name;target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes());sources[name]=digest(path)
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
            for d in plan['d']:
                for n in plan['validation_N']:case('validation',kernel,config,[1,1,n,d],'validate')
            for pattern in ['uniform','constant_v','rising','stress']:case('validation',kernel,config,[2,3,97,64],'validate',pattern)
    timings={};frozen={}
    for kernel,count in VARIANTS.items():
        frozen[kernel]={}
        for d in plan['d']:
            scores=[]
            for config in range(count):
                values=[case('tuning',kernel,config,[plan['B'],plan['H'],n,d]) for n in plan['tuning_N']]
                score=math.exp(statistics.mean(math.log(x) for x in values));scores.append((score,config));timings[kernel,d,config]=score
            score,config=min(scores);frozen[kernel][str(d)]={'config':config,'geomean_ms':score}
    winners={}
    for family,variants in [('cuda_core',['skip','vector']),('tensor_core',['wmma','ptx'])]:
        winners[family]={str(d):min(variants,key=lambda k:frozen[k][str(d)]['geomean_ms']) for d in plan['d']}
    dump(out/'frozen_selection.json',{'variants':frozen,'families':winners,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'plan_sha256':digest(out/'plan.json')})
    # Complete progression at defaults, then tuned finalists. Cached equality is explicit.
    for d in plan['d']:
        for n in plan['main_N']:
            for kernel in plan['kernels']:case('default_main',kernel,0,[plan['B'],plan['H'],n,d])
            for kernel in VARIANTS:
                cfg=frozen[kernel][str(d)]['config']
                if cfg:case('frozen_main',kernel,cfg,[plan['B'],plan['H'],n,d])
            cfg=frozen['skip'][str(d)]['config']
            # Match *all* launch knobs so this isolates scalar vs float4 loading.
            for kernel in ['skip','vector']:case('matched_load',kernel,cfg,[plan['B'],plan['H'],n,d])
            for kernel in plan['memory_baselines']:case('memory',kernel,0,[plan['B'],plan['H'],n,d],'memory')
            for family in winners:
                kernel=winners[family][str(d)];cfg=frozen[kernel][str(d)]['config']
                case('memory',kernel,cfg,[plan['B'],plan['H'],n,d],'memory')
    fields=list(records[0]);
    with (out/'measurements.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    analyze(out)

def analyze(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plan=json.loads((out/'plan.json').read_text());
    ds=plan['d'];ns=plan['main_N'];
    rows=list(csv.DictReader((out/'measurements.csv').open()));frozen=json.loads((out/'frozen_selection.json').read_text());groups={}
    for r in rows:groups.setdefault((r['experiment'],r['kernel'],int(r['config']),int(r['N']),int(r['d']),r['pattern'],int(r['B']),int(r['H'])),[]).append(r)
    summaries=[]
    for key,values in groups.items():
        r=values[0].copy();
        # These baseline branches do not expose resource queries; raw defaults
        # are placeholders, never measured zero-valued resource usage.
        if r['kernel'] in ['scalar','warp'] or (r['kernel']=='fa1' and plan.get('fa1_workspace_floats_per_query',0)==0):
            for col in ['registers_per_thread','shared_bytes','local_bytes_per_thread','active_blocks_per_sm','occupancy_estimate_percent']:r[col]=''
        times=[float(v['forward_ms']) for v in values];r.update(median_ms=statistics.median(times),q1_ms=float(np.quantile(times,.25)),q3_ms=float(np.quantile(times,.75)),repeats=len(times) if r['mode']=='benchmark' else 0)
        r.update(iqr_ms=r['q3_ms']-r['q1_ms'],sequences_per_second=1000*int(r['B'])/r['median_ms'] if r['median_ms'] else '',tokens_per_second=1000*int(r['B'])*int(r['N'])/r['median_ms'] if r['median_ms'] else '')
        if r['mode']=='benchmark' and (len(values)!=20 or sorted(int(v['repeat']) for v in values)!=list(range(20))):raise ValueError('Missing/duplicate timing repetitions')
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
    ns=plan['main_N']
    for name,variants,base in [('cuda_core_progression',['fa1','scalar','warp','tiled','skip','vector'],'fa1'),('tensor_core_progression',(['fa1_wmma','wmma','ptx'] if 'fa1_wmma' in plan.get('kernels',[]) else ['wmma','ptx']),'fa1_wmma' if 'fa1_wmma' in plan.get('kernels',[]) else 'wmma')]:
        fig,axes=plt.subplots(1,len(ds),figsize=(5*len(ds),4),squeeze=False)
        axes=axes[0]
        for ax,d in zip(axes,ds):
            for k in variants:ax.plot(ns,[float(find('default_main',base,d,n)['median_ms'])/float(find('default_main',k,d,n)['median_ms']) for n in ns],marker='o',label=k)
            ax.set(title=f'd={d}, fixed defaults',xlabel='Sequence length',ylabel=f'Speedup over {base}');ax.set_xscale('log',base=2);ax.grid(alpha=.3)
        axes[-1].legend();save(fig,name)
    fig,axes=plt.subplots(1,len(ds),figsize=(5*len(ds),4),squeeze=False)
    axes=axes[0]
    for ax,d in zip(axes,ds):
        for k in VARIANTS:
            ax.plot(ns,[float(find('default_main',k,d,n)['median_ms'])/float(final(k,d,n)['median_ms']) for n in ns],marker='o',label=k)
        ax.axhline(1,color='gray',linestyle='--');ax.set(title=f'd={d}: frozen vs default',xlabel='Sequence length',ylabel='Default / tuned latency');ax.legend();ax.grid(alpha=.3)
    save(fig,'frozen_tuning_speedup')
    fig,axes=plt.subplots(1,len(ds),figsize=(5*len(ds),4),squeeze=False);comparison=[]
    axes=axes[0]
    for ax,d in zip(axes,ds):
        c=frozen['families']['cuda_core'][str(d)];t=frozen['families']['tensor_core'][str(d)]
        for k in [c,t]:ax.plot(ns,[float(final(k,d,n)['median_ms']) for n in ns],marker='o',label=f'{k} cfg {frozen["variants"][k][str(d)]["config"]}')
        for n in ns:comparison.append({'d':d,'N':n,'cuda_core':c,'tensor_core':t,'cuda_ms':final(c,d,n)['median_ms'],'tensor_ms':final(t,d,n)['median_ms'],'tensor_speedup':float(final(c,d,n)['median_ms'])/float(final(t,d,n)['median_ms'])})
        ax.set(title=f'd={d}: FP32 core / FP16 tensor',xlabel='Sequence length',ylabel='Forward latency (ms)',yscale='log');ax.legend();ax.grid(alpha=.3)
    save(fig,'frozen_family_comparison')
    with (out/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    fig,axes=plt.subplots(1,len(ds),figsize=(5*len(ds),4),squeeze=False);ablations=[]
    axes=axes[0]
    for ax,d in zip(axes,ds):
        speed=[]
        for n in ns:
            s=float(find('matched_load','skip',d,n)['median_ms']);v=float(find('matched_load','vector',d,n)['median_ms']);speed.append(s/v)
            ablations.append({'d':d,'N':n,'config':frozen['variants']['skip'][str(d)]['config'],'scalar_ms':s,'vector_ms':v,'vector_speedup':s/v})
        ax.plot(ns,speed,marker='o');ax.axhline(1,color='gray',linestyle='--');ax.set(title=f'd={d}, matched launch configuration',xlabel='Sequence length',ylabel='Scalar / vector latency');ax.grid(alpha=.3)
    save(fig,'matched_vector_load')
    with (out/'matched_ablation.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(ablations[0]));w.writeheader();w.writerows(ablations)
    fig,axes=plt.subplots(1,len(ds),figsize=(5*len(ds),4),squeeze=False)
    axes=axes[0]
    for ax,d in zip(axes,ds):
        for k in plan.get('memory_baselines',[]):ax.plot(ns,[int(find('memory',k,d,n)['explicit_live_device_bytes'])/2**20 for n in ns],marker='o',label=k+' (incl. m/l)')
        for family in ['cuda_core','tensor_core']:
            k=frozen['families'][family][str(d)];ax.plot(ns,[int(find('memory',k,d,n)['explicit_live_device_bytes'])/2**20 for n in ns],marker='o',label=k)
        ax.plot(ns,[plan.get('B',4)*plan.get('H',8)*n*n*8/2**20 for n in ns],linestyle='--',label='FP32 S+P storage model')
        ax.set(title=f'd={d}',xlabel='Sequence length',ylabel='Explicit live device storage (MiB)',yscale='log');ax.legend();ax.grid(alpha=.3)
    save(fig,'device_memory')
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,(k,count) in zip(axes,VARIANTS.items()):
        data=[[math.exp(statistics.mean(math.log(float(find('tuning',k,d,n,c)['median_ms'])) for n in plan['tuning_N'])) for d in plan['d']] for c in range(count)]
        norm=np.asarray(data)/np.min(data,axis=0);im=ax.imshow(norm,aspect='auto',vmin=1);ax.set(title=k,xlabel='Head dimension',ylabel='Configuration');ax.set_xticks(range(len(ds)),ds);fig.colorbar(im,ax=ax,label='Latency / best')
    save(fig,'tuning_heatmap')
    failures=[r for r in rows if r['status']!='PASS'];validation=[r for r in summaries if r['experiment']=='validation']
    fingerprints={}
    for r in rows:
        key=tuple(r[x] for x in ['B','H','N','d','pattern']);value=(r['input_digest'],r['rounded_input_digest'])
        if key in fingerprints and fingerprints[key]!=value:raise ValueError('input hash mismatch')
        fingerprints[key]=value
        expected=int(r['B'])*int(r['H'])*int(r['N'])*int(r['d'])*(10 if r['input_precision']=='FP16' else 16)
        if r['kernel'] in ['fa1','fa1_wmma']:expected+=4*plan.get('fa1_workspace_floats_per_query',0)*int(r['B'])*int(r['H'])*int(r['N'])
        if int(r['explicit_live_device_bytes'])!=expected:raise ValueError('allocation ledger mismatch')
    manifest=json.loads((out/'manifest.json').read_text());source_ok=all(digest(out/'source_snapshot'/path)==sha for path,sha in manifest['sources'].items())
    audit={'passed':not failures and source_ok,'validation_cases':len(validation),'benchmark_cases':sum(r['mode']=='benchmark' for r in summaries),'samples':sum(r['mode']=='benchmark' for r in rows),'memory_cases':sum(r['mode']=='memory' for r in summaries),'source_snapshot_hashes_match':source_ok,'input_hashes_consistent':True,'allocation_ledger_matches':True,'plan_hash_matches_frozen':digest(out/'plan.json')==frozen['plan_sha256']}
    # Record the analysis version separately from the measurement-time snapshot.
    analysis=out/'analysis';analysis.mkdir(exist_ok=True)
    (analysis/'evaluate.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'analysis_manifest.json',{'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__,'script_sha256':digest(Path(__file__)), 'measurements_sha256':digest(out/'measurements.csv'), 'case_csv_sha256':{str(path.relative_to(out)):digest(path) for path in sorted((out/'cases').glob('*.csv'))}})
    expected={'validation':sum(VARIANTS.values())*(len(plan['d'])*len(plan.get('validation_N',[1,17,33,65]))+4),'tuning':sum(VARIANTS.values())*len(plan['d'])*len(plan['tuning_N']),'default_main':len(plan.get('kernels',['fa1','scalar','warp','tiled','skip','vector','wmma','ptx']))*len(plan['d'])*len(ns),'matched_load':2*len(plan['d'])*len(ns),'memory':(2+len(plan.get('memory_baselines',[])))*len(plan['d'])*len(ns)}
    for exp,count in expected.items():
        if sum(r['experiment']==exp for r in summaries)!=count:raise ValueError(f'Incomplete experiment: {exp}')
    expected_frozen=len(ns)*sum(frozen['variants'][k][str(d)]['config']!=0 for k in VARIANTS for d in plan['d'])
    if sum(r['experiment']=='frozen_main' for r in summaries)!=expected_frozen:raise ValueError('Incomplete frozen grid')
    if len(summaries)!=sum(expected.values())+expected_frozen:raise ValueError('Unexpected study cases')
    dump(out/'audit.json',audit)
    report=['# Part C bounded evaluation','',f'Correctness: {len(validation)} full small/tail/pattern checks; {audit["benchmark_cases"]} sampled benchmark checks; {audit["memory_cases"]} sampled memory checks. All outputs checked finite.',
            '', f"Selection was frozen after tuning and before the main grid. Family winners are selected only among skip/vector and WMMA/PTX; both FA1 baselines use fixed config 0. Tuning lengths: {plan['tuning_N']}; reporting lengths: {ns}; held-out reporting lengths: {sorted(set(ns)-set(plan['tuning_N']))}.", '', '## Frozen configurations','', '| d | CUDA core | Tensor core |','|---|---|---|']
    for d in plan['d']:
        c=frozen['families']['cuda_core'][str(d)];t=frozen['families']['tensor_core'][str(d)];report.append(f'| {d} | {c}, config {frozen["variants"][c][str(d)]["config"]} | {t}, config {frozen["variants"][t][str(d)]["config"]} |')
    report+=['',f'## Largest reporting N={max(ns)}','', '| d | CUDA ms | Tensor ms | Core / tensor |','|---|---|---|---|']
    for r in comparison:
        if r['N']==max(ns):report.append(f'| {r["d"]} | {float(r["cuda_ms"]):.3f} | {float(r["tensor_ms"]):.3f} | {r["tensor_speedup"]:.2f}x |')
    report+=['',f'## Tuning effect at N={max(ns)}','','| d | Family | Kernel | Default ms | Frozen ms | Default / frozen | Reg/thread default → frozen | Local bytes/thread default → frozen |','|---|---|---|---|---|---|---|---|']
    for d in plan['d']:
        for family in ['cuda_core','tensor_core']:
            k=frozen['families'][family][str(d)];base=find('default_main',k,d,max(ns));tuned=final(k,d,max(ns))
            b,t=float(base['median_ms']),float(tuned['median_ms'])
            report.append(f'| {d} | {family} | {k} | {b:.3f} | {t:.3f} | {b/t:.2f}x | {base["registers_per_thread"]} → {tuned["registers_per_thread"]} | {base["local_bytes_per_thread"]} → {tuned["local_bytes_per_thread"]} |')
    report+=['',f'## Memory at N={max(ns)}','','| d | Kernel | Explicit live buffers MiB | Observed device delta MiB |','|---|---|---|---|']
    for d in plan['d']:
        memory_kernels=plan.get('memory_baselines',[])+[frozen['families'][family][str(d)] for family in ['cuda_core','tensor_core']]
        for k in memory_kernels:
            r=find('memory',k,d,max(ns))
            report.append(f'| {d} | {k} | {int(r["explicit_live_device_bytes"])/2**20:.1f} | {int(r["observed_device_delta_bytes"])/2**20:.1f} |')
    if plan.get('fa1_workspace_floats_per_query'):
        report+=['','FA1 in this run is the CUDA-core Algorithm-1 schedule: K/V outer, Q inner, one CTA/head, shared Q/K/V/S/O, global m/l. Its live allocation adds 8*B*H*N bytes; it is not the historical query-parallel scalar baseline. The WMMA FA1 baseline uses the same K/V-outer state schedule, half Q/K/V/P and FP32 accumulators/softmax/O. Both baselines add global m/l; neither is the official FA1 library. FA2-style kernels are unchanged. Do not reuse old FA1 progression ratios.']
    report+=['','## Interpretation limits','','The matched-load study changes scalar versus float4 loading with the same launch configuration. Other progression steps combine changes and are not isolated causal ablations. Config tuning changes resource use without changing kernel algorithms.','', 'Resource queries are available for tiled/skip/vector/WMMA/PTX. Raw scalar/warp zero defaults are unavailable placeholders; summary.csv leaves those fields blank. FA1 resources are also unavailable in this study.', '', 'Memory plots show live explicit Q/K/V/O allocations plus FA1 m/l where recorded, not DRAM traffic or total process memory. FP32 core uses 16 bytes/element; FP16 inputs with FP32 output use 10. The S+P line is an analytical materialized-storage model, not a measured Part A result. cudaMemGetInfo deltas include allocation granularity and concurrent device activity; only three observations are made. Resource occupancy is an API estimate, not achieved Nsight occupancy.','', 'Tensor Core paths use FP16 Q/K/V and round probabilities for PV; CUDA Core uses FP32. Error is checked against the corresponding rounded-input FP64 oracle. These are precision-labelled performance comparisons.','', 'Cases run in a fixed sequential order on one laptop GPU with unmanaged clocks. Twenty repetitions characterize within-case spread, not independent-session uncertainty; close winners may change on a fresh run. No confidence interval or universal winner is claimed.', '', 'See plan.json, commands.jsonl, cases/, measurements.csv, summary.csv, comparison.csv, matched_ablation.csv, manifest.json, frozen_selection.json and audit.json. Historical profiling remains a separate fixed-default study; its counters do not describe tuned configurations.']
    if plan.get('fa1_workspace_floats_per_query'):
        report=[line.replace('FA1 resources are also unavailable in this study.','FA1 resource queries include dynamic shared storage.') for line in report]
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    if not audit['passed'] or not audit['plan_hash_matches_frozen']:raise RuntimeError('audit failed')
if __name__=='__main__':main()
