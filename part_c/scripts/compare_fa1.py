#!/usr/bin/env python3
"""Small, explicitly non-official FA1/FA2 pilot; same inputs/device, native evidence retained."""
import argparse,csv,json,math,statistics,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import results_schema as schema
from workflow import Workflow,architecture,device_default,device_info,dump,sha
KERNELS=['fa1','skip','fa1_wmma','wmma','ptx']
FILES=['part_c/src/flash_attention_1.cu','part_c/src/flash_attention_1_tensor_core.cu','part_c/src/flash_attention_2_cuda_core.cu','part_c/src/flash_attention_2_tensor_core.cu','part_c/tests/test_support.h','part_c/tests/test_support.cpp','part_c/tests/kernel_runner.cpp','part_c/tests/evaluation_runner.cpp','part_c/tests/flash_attention_1_test.cpp','part_c/CMakeLists.txt','part_a/common/input_generation.hpp','part_a/common/attention_types.hpp','part_a/cpu_reference/attention_reference.cpp','part_a/cpu_reference/sampled_reference.cpp','part_a/cpu_reference/CMakeLists.txt','part_c/scripts/compare_fa1.py','part_c/scripts/inspect_binary.py','part_c/tests/kernel_bodies.json','scripts/results_schema.py','scripts/workflow.py']
FILES += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'part_c/include/flash_attention').glob('*.h'))]
def write_comparisons(path,rows):
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def quantile(values,p):
    values=sorted(values);x=(len(values)-1)*p;lo=int(x);hi=min(lo+1,len(values)-1)
    return values[lo]+(x-lo)*(values[hi]-values[lo])
def analyze(out):
    out=Path(out).resolve();raw=out/'raw/part_c/fa1_pilot';plan=json.loads((raw/'plan.json').read_text())
    for name,value in plan['sources'].items():
        if sha(raw/'source_snapshot'/name)!=value:raise ValueError('Changed source snapshot: '+name)
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        for name,value in old['native_inputs'].items():
            if sha(out/name)!=value:raise ValueError('Native evidence changed: '+name)
    tables={name:[] for name in schema.HEADERS};summary=[];by_shape={};inputs={};inventory={}
    configs={}
    for case in plan['cases']:
        file=raw/'cases'/case['file'];rows=schema.read(file)
        if len(rows)!=20 or {int(r['repeat']) for r in rows}!=set(range(20)):raise ValueError('Missing/duplicate repetitions')
        row=rows[0];variant=row['kernel'];n=int(row['N']);d=int(row['d']);heads=int(row['B'])*int(row['H']);tc=variant in ['fa1_ptx','fa1_wmma','wmma','ptx'];fa1=variant in ['fa1','fa1_wmma','fa1_ptx']
        if variant!=case['kernel'] or any(int(row[k])!=case[k] for k in ['B','H','N','d']) or row['config']!='0' or row['pattern']!='random':raise ValueError('Unexpected case metadata')
        if row['mode']!='benchmark' or row['input_precision']!=('FP16' if tc else 'FP32') or row['accumulator_precision']!='FP32' or row['output_precision']!='FP32':raise ValueError('Pilot numerical contract differs from implementation')
        for r in rows:
            for name,value in row.items():
                if name not in ['repeat','forward_ms'] and r[name]!=value:raise ValueError('Case metadata changes between samples')
        if row['status']!='PASS' or int(row['failures']) or int(row['nonfinite']):raise ValueError('Failed output validation')
        key=tuple(row[k] for k in ['B','H','N','d'])
        if key in inputs and inputs[key]!=row['input_digest']:raise ValueError('Base inputs differ')
        inputs[key]=row['input_digest']
        ctx={name:'' for name in schema.COMMON}
        ctx.update(experiment='fa1_pilot',precision=row['input_precision'],B=row['B'],H=row['H'],N=row['N'],d=row['d'],mode='causal',variant=variant,run_id=plan['run_id'],part='part_c',config=f'{variant}:pilot_default:d{d}',pattern='random',layout='BHNd_contiguous_row_major',input_precision=row['input_precision'],accumulator_precision='FP32',output_precision='FP32',generator='splitmix64-fp32-v1',seed='5206',input_digest=row['input_digest'],rounded_input_digest=row['rounded_input_digest'] if tc else '')
        ctx['case_id']=schema.case_id(ctx)
        scope=row['reference_scope'];tables['correctness'].append({**ctx,'stage':'O',**{k:row[k] for k in ['atol','rtol','max_absolute','max_relative','failures','status','checked_elements','nonfinite','reference_scope']},'oracle_input_precision':'FP16-rounded' if tc else 'FP32','checked_query_rows':'all' if scope=='full' else json.dumps(sorted({0,n//2,n-1}),separators=(',',':'))})
        values=[float(r['forward_ms']) for r in rows];med=statistics.median(values);q1=quantile(values,.25);q3=quantile(values,.75)
        expected=n*d*heads*(10 if tc else 16)+(8*heads*n if fa1 else 0)
        if int(row['explicit_live_device_bytes'])!=expected:raise ValueError('Allocation ledger mismatch')
        record={**ctx,**{k:row[k] for k in ['atol','rtol','max_absolute','max_relative']},'warmup':'5','repeats':'20','forward_median_ms':str(med),'forward_q1_ms':str(q1),'forward_q3_ms':str(q3),'forward_iqr_ms':str(q3-q1),'sequences_per_second':str(1000*int(row['B'])/med),'tokens_per_second':str(1000*int(row['B'])*n/med),'peak_workspace_allocated_bytes':str(expected),'validation_status':'PASS','reference_scope':scope}
        tables['summary'].append(record);summary.append(record);by_shape[variant,*key]=record
        tables['samples'] += [{**ctx,'repeat':r['repeat'],'forward_ms':r['forward_ms']} for r in rows]
        configs[ctx['config']]={k:row[k] for k in ['block_q','block_kv','warps','thread_q','thread_k']}
        tables['resources'].append({**ctx,'stage':'forward','blocks':str(heads if fa1 else heads*math.ceil(n/int(row['block_q']))),'threads':str(32*int(row['warps'])),**{k:row[k] for k in ['registers_per_thread','shared_bytes','local_bytes_per_thread','active_blocks_per_sm']},'theoretical_occupancy_percent':row['occupancy_estimate_percent']})
        tables['memory'].append({**ctx,'explicit_live_device_bytes':str(expected),'auxiliary_global_bytes':str(8*heads*n if fa1 else 0),'observed_device_delta_bytes':row['observed_device_delta_bytes'],'measurement':'explicit_live_QKVO_plus_FA1_state_if_applicable_not_process_peak'})
        tables['work'].append({**ctx,'valid_attention_pairs':str(heads*n*(n+1)//2),'evidence_kind':'analytical_valid_pairs_not_executed_instructions'})
        inventory[str(file.relative_to(out))]=sha(file)
    expected_cases={(kernel,*[str(x) for x in shape]) for kernel in plan.get('kernels',KERNELS) for shape in plan['shapes']}
    if set(by_shape)!=expected_cases:raise ValueError('Incomplete pilot grid')
    audit=schema.validate(tables);comparisons=[];tradeoffs=[]
    pairs=[('fa1','skip','matched_FP32'),('fa1_wmma','wmma','matched_FP16_QKV_FP32_accum_softmax_output_normalisation_schedule_differs'),('fa1_wmma','ptx','matched_FP16_QKV_FP32_accum_softmax_output_and_different_MMA_mapping')]
    if 'fa1_ptx' in plan.get('kernels',[]):pairs.append(('fa1_ptx','ptx','matched_FP16_same_PTX_MMA_mapping_fixed_defaults'))
    for shape in plan['shapes']:
        key=[str(x) for x in shape]
        for baseline,candidate,contract in pairs+[('fa1','fa1_wmma','cross_precision_hardware_plus_input_precision_not_FA2_speedup')]:
            a=by_shape[baseline,*key];b=by_shape[candidate,*key]
            if a['input_digest']!=b['input_digest']:raise ValueError('Comparison input mismatch')
            if a['precision']==b['precision']=='FP16' and a['rounded_input_digest']!=b['rounded_input_digest']:raise ValueError('Effective FP16 inputs differ')
            row={'B':key[0],'H':key[1],'N':key[2],'d':key[3],'mode':'causal','baseline':baseline,'candidate':candidate,'baseline_median_ms':a['forward_median_ms'],'baseline_q1_ms':a['forward_q1_ms'],'baseline_q3_ms':a['forward_q3_ms'],'candidate_median_ms':b['forward_median_ms'],'candidate_q1_ms':b['forward_q1_ms'],'candidate_q3_ms':b['forward_q3_ms'],'speedup':float(a['forward_median_ms'])/float(b['forward_median_ms']),'contract':contract,'input_digest':a['input_digest']}
            (tradeoffs if baseline=='fa1' and candidate=='fa1_wmma' else comparisons).append(row)
    for name,rows in tables.items():schema.write(out/f'{name}.csv',name,rows)
    write_comparisons(out/'comparison.csv',comparisons);write_comparisons(out/'precision_tradeoffs.csv',tradeoffs)
    (out/'configs').mkdir(exist_ok=True);dump(out/'configs/part_c.json',{'configurations':configs,'selection':'one fixed config per implementation; no tuning or reporting-grid winner selection'})
    audit.update(grid='small_pilot' if len(plan['shapes'])==2 else 'full_shape_pilot',official_FA1_measured=False,all_cases_validated=True,shared_cpu_reference_linked=True,profiling_requested=False)
    dump(out/'audit.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'plots/part_c';folder.mkdir(parents=True,exist_ok=True)
    ds=sorted({shape[3] for shape in plan['shapes']});fig,axes=plt.subplots(1,len(ds),figsize=(6*len(ds),4),squeeze=False)
    for ax,d in zip(axes[0],ds):
        for baseline,candidate,_ in pairs:
            selected=sorted([r for r in comparisons if int(r['d'])==d and r['baseline']==baseline and r['candidate']==candidate],key=lambda r:int(r['N']))
            ax.plot([int(r['N']) for r in selected],[r['speedup'] for r in selected],marker='o',label=f'{baseline} / {candidate}')
        ax.axhline(1,color='gray',linestyle='--');ax.set(title=f'Causal, B4/H8/d{d}; defaults',xlabel='Sequence length',ylabel='FA1 latency / FA2-style latency');ax.legend(fontsize=7);ax.grid(alpha=.25)
    fig.suptitle('Own FA1/FA2 implementations; no official-performance parity claim',fontsize=10);fig.tight_layout()
    for extension in ['png','pdf']:fig.savefig(folder/f'fa1_fa2_speedup.{extension}',dpi=160,metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Software':'attention-fa1-pilot'})
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));x=list(range(len(summary)))
    ax.errorbar(x,[float(r['forward_median_ms']) for r in summary],yerr=[[float(r['forward_median_ms'])-float(r['forward_q1_ms']) for r in summary],[float(r['forward_q3_ms'])-float(r['forward_median_ms']) for r in summary]],fmt='o',capsize=3)
    ax.set_xticks(x,[f"{r['variant']}/N{r['N']}/d{r['d']}" for r in summary],rotation=65,ha='right',fontsize=7);ax.set(ylabel='Forward latency (ms)',yscale='log',title='Median and IQR; FP32 and FP16 series remain distinct');ax.grid(axis='y',alpha=.25);fig.tight_layout()
    for extension in ['png','pdf']:fig.savefig(folder/f'pilot_latency.{extension}',dpi=160,metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Software':'attention-fa1-pilot'})
    plt.close(fig)
    report=['# FA1/FA2 pilot','','All kernels are this project\'s implementations. The official FA1 package was not measured and performance parity is not established. The default small grid is a pilot, not the final experiment. Configurations are fixed defaults; no speedup-based case filtering or winner selection.','','| N | d | Baseline | Candidate | Speedup |','|---|---|---|---|---|']
    for row in comparisons:report.append(f"| {row['N']} | {row['d']} | {row['baseline']} | {row['candidate']} | {row['speedup']:.3f}x |")
    report += ['','FP32 FA1/skip is a matched FP32 comparison. FA1 WMMA versus FA2 WMMA/PTX uses identical FP16-rounded inputs, FP32 accumulation/state/output and FP16 probability operands. Normalised versus unnormalised intermediate state and probability-rounding history can differ, so bitwise equality is not expected. WMMA versus PTX changes fragment mapping as well as scheduling.','', 'precision_tradeoffs.csv separately records FP32-to-Tensor-Core FA1 ratios; these are not FA2 improvements. Samples use five warmups and twenty CUDA-event repetitions outside allocation/transfers/reference work. IQR describes within-case spread only. Every timed case has sampled FP64 checks and all-output finiteness. Small dedicated tests additionally check full output, m/l and causal isolation.','', 'Review source hashes, binaries, command logs and cases under raw/part_c/fa1_pilot/. Nsight counters are not collected by this timing-only pilot; resource occupancy is theoretical. The standard workflow uses FP32 FA1 and tuned PTX FA1; WMMA FA1 is supplementary. This explicit pilot retains WMMA and uses fixed config 0 for all variants. Historical saved data remain unchanged.','', 'Offline: bash part_c/run.sh --fa1-pilot --analyze-only '+str(out)]
    (out/'ANALYSIS.md').write_text('\n'.join(report)+'\n')
    inventory={str(path.relative_to(out)):sha(path) for path in sorted(raw.rglob('*')) if path.is_file()}
    dump(out/'manifest.json',{'schema_version':schema.VERSION,'part':'part_c','experiment':'fa1_pilot','run_id':plan['run_id'],'grid':plan['grid'],'device':plan['device'],'shared_cpu_reference_linked':True,'official_FA1_measured':False,'profile_requested':False,'sources':plan['sources'],'binary_sha256':plan['binary_sha256'],'native_inputs':inventory,'numerical_contracts':{'FP32':'FP32 QKV/accum/state/output','FP16':'FP16 QKV/P operands; FP32 accum/state/output'},'processing_sha256':sha(Path(__file__))})
    dump(out/'figure_index.json',{'figures':['plots/part_c/fa1_fa2_speedup','plots/part_c/pilot_latency'],'formats':['PNG','PDF'],'data':['summary.csv','comparison.csv','precision_tradeoffs.csv']})
    return audit

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('results')/datetime.now(timezone.utc).strftime('fa1_pilot_%Y%m%dT%H%M%SZ'));p.add_argument('--build',type=Path,default=Path('/tmp/fa1-pilot-build'));p.add_argument('--arch',default='auto');p.add_argument('--device',default=device_default());p.add_argument('--grid',choices=['small','full'],default='small');p.add_argument('--dry-run',action='store_true');p.add_argument('--analyze-only',type=Path)
    a=p.parse_args()
    if a.analyze_only:print(json.dumps(analyze(a.analyze_only),indent=2));return
    if not a.device or ',' in a.device:p.error('Select one GPU')
    fresh_kernels=KERNELS+['fa1_ptx']
    arch=architecture(a.arch,a.device,a.dry_run);out=a.output.resolve();build=a.build.resolve();w=Workflow(out,a.device,a.dry_run,False);raw=out/'raw/part_c/fa1_pilot'
    try:
        w.run('gpu',['nvidia-smi','-q']);w.run('configure',['cmake','-S',ROOT/'part_c','-B',build,'-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_CUDA_ARCHITECTURES={arch}']);w.run('build',['cmake','--build',build,'--parallel','2'])
        for flag,label in [([], 'fa1_fp32'),(['--tensorcore'],'fa1_wmma'),(['--ptx'],'fa1_ptx')]:w.run(label+'_invariants',[build/'part_c_fa1_test',*flag,'--output',raw/'correctness'/label])
        w.run('all_kernel_regression',[build/'part_c_runner','--suite','correctness','--output',raw/'correctness/all_kernels'])
        for tool in ['memcheck','racecheck','synccheck','initcheck']:w.run('fa1_wmma_'+tool,['compute-sanitizer','--tool',tool,'--error-exitcode','3',build/'part_c_fa1_test','--tensorcore','--smoke','--output',raw/'safety'/tool])
        w.run('binary_inspection',[sys.executable,ROOT/'part_c/scripts/inspect_binary.py',build/'part_c_runner',raw/'binary_inspection'])
        shapes=[[4,8,n,d] for d in ([64] if a.grid=='small' else [32,64,128]) for n in ([256,1024] if a.grid=='small' else [256,512,1024,2048,4096])];cases=[]
        if not a.dry_run:
            functions=schema.read(raw/'binary_inspection/tensor_core_instructions.csv');new=[r for r in functions if 'flash_attention_v1_wmma' in r['symbol']]
            if len(new)!=18 or not all(int(r['static_hmma_instruction_count'])>0 for r in new):raise ValueError('FA1 WMMA Tensor Core instructions were not verified')
            ptx=[r for r in functions if 'flash_attention_v1_ptx_mma' in r['symbol']]
            if len(ptx)!=40 or not all(int(r['static_hmma_instruction_count'])>0 for r in ptx):raise ValueError('FA1 PTX Tensor Core instructions were not verified')
            raw.mkdir(parents=True,exist_ok=True);sources={}
            for name in FILES:
                path=ROOT/name;dest=raw/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());sources[name]=sha(path)
            plan={'run_id':out.name,'grid':a.grid,'device':device_info(a.device),'architecture':arch,'shapes':shapes,'kernels':fresh_kernels,'config':0,'sources':sources,'binary_sha256':sha(build/'part_c_evaluation'),'cases':cases,'order':'ascending N per d; kernel order rotated per shape','tuning':'none; fixed defaults; small pilot first','profile_requested':False,'official_FA1':'not measured; own CUDA-core/WMMA/PTX implementations only'}
            dump(raw/'plan.json',plan)
        for index,shape in enumerate(shapes):
            order=fresh_kernels[index%len(fresh_kernels):]+fresh_kernels[:index%len(fresh_kernels)]
            for kernel in order:
                name=kernel+'_'+'_'.join(map(str,shape))+'.csv';w.run(name[:-4],[build/'part_c_evaluation',kernel,0,*shape,'random','benchmark',raw/'cases'/name]);cases.append({'file':name,'kernel':kernel,**dict(zip(['B','H','N','d'],shape))})
        if not a.dry_run:
            dump(raw/'plan.json',plan);audit=analyze(out);dump(out/'run_status.json',{'status':'completed_small_pilot' if a.grid=='small' else 'completed_full_shape_pilot','audit_passed':audit['passed'],'official_FA1_measured':False});print(out/'ANALYSIS.md')
    except Exception as error:
        if not a.dry_run:dump(out/'run_status.json',{'status':'failed','error':str(error)})
        raise
if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as error:print('ERROR: '+str(error),file=sys.stderr);sys.exit(1)
