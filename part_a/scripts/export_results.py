#!/usr/bin/env python3
"""Part A adapter using the shared Part A-based report schema; native data retained."""
import argparse,csv,json,math,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import results_schema as schema
from workflow import dump,sha
from ncu_csv import captures,metric_rows,metric_aliases

# Report counters only
REPORT_METRICS=set("""dram__bytes_read.sum dram__bytes_write.sum
    dram__throughput.avg.pct_of_peak_sustained_elapsed dram__bytes.sum.per_second
    gpu__time_duration.sum sm__warps_active.avg.pct_of_peak_sustained_active
    smsp__warps_eligible.avg.per_cycle_active
    sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
    l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum
    l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum
    memory_l1_wavefronts_shared memory_l1_wavefronts_shared_ideal
    derived__memory_l1_wavefronts_shared_excessive
    smsp__pcsamp_warps_issue_stalled_long_scoreboard
    smsp__pcsamp_warps_issue_stalled_short_scoreboard
    smsp__pcsamp_warps_issue_stalled_barrier
    smsp__pcsamp_warps_issue_stalled_mio_throttle""".split())
REPORT_NAMES={name for metric in REPORT_METRICS for name in metric_aliases(metric)}
def report_profiles(rows):return (row for row in rows if row['metric'] in REPORT_NAMES)

def context(row,run_id,phase):
    result={k:'' for k in schema.COMMON}
    result.update(**{k:row[k] for k in ['precision','B','H','N','d','mode','variant','tile','softmax_threads']})
    result.update(experiment=phase,run_id=run_id,part='part_a',config=f"{row['variant']}:tile{row['tile']}:sm{row['softmax_threads']}:d{row['d']}",pattern=row['pattern'],layout='BHNd_contiguous_row_major',input_precision=row['precision'],accumulator_precision='FP32',output_precision='FP32',generator='splitmix64-fp32-v1',seed='5206',input_digest=row['input_digest'],rounded_input_digest=row['effective_input_digest'] if row['precision']=='FP16' else '')
    result['case_id']=schema.case_id(result);return result

def identity(row):return tuple(row[k] for k in ['precision','B','H','N','d','mode','variant','tile','softmax_threads'])
def publish(out):
    out=Path(out).resolve();native=out/'raw/part_a';plan=json.loads((native/'plan.json').read_text())
    for name,value in plan['sources'].items():
        if sha(native/'source_snapshot'/name)!=value:raise ValueError('Part A source snapshot changed: '+name)
    old_manifest=out/'manifest.json'
    if old_manifest.exists():
        for name,value in json.loads(old_manifest.read_text()).get('raw_files',{}).items():
            path=out/name
            if path.suffix in ['.ncu-rep','.ncu-repz'] and not path.exists():continue
            if sha(path)!=value:raise ValueError('Part A native evidence changed: '+name)
    tables={k:[] for k in schema.HEADERS};timings={}
    configs={}
    for item in plan['outputs']:
        folder=native/item['path'];phase=item['phase']
        if (folder/'correctness.csv').exists():
            for row in schema.read(folder/'correctness.csv'):
                ctx=context(row,plan['run_id'],'correctness')
                tables['correctness'].append({**ctx,**{k:row[k] for k in ['stage','atol','rtol','max_absolute','max_relative','failures','status','checked_elements','nonfinite']},'oracle_input_precision':'FP16-rounded' if row['precision']=='FP16' else 'FP32','reference_scope':'full','checked_query_rows':'all'})
            continue
        local={}
        for row in schema.read(folder/'summary.csv'):
            ctx=context(row,plan['run_id'],phase);local[identity(row)]=(ctx,row)
            if row['status']!='PASS' or int(row['failures']) or int(row['nonfinite']):raise ValueError('Part A output failed validation')
            configs[ctx['config']]={'variant':row['variant'],'tile':int(row['tile']),'softmax_threads':int(row['softmax_threads']),'d':int(row['d'])}
            checked='all' if row['reference_scope']=='full' else json.dumps(sorted({0,int(row['N'])//2,int(row['N'])-1}),separators=(',',':'))
            tables['correctness'].append({**ctx,'stage':'O',**{k:row[k] for k in ['atol','rtol','max_absolute','max_relative','failures','status','checked_elements','nonfinite','reference_scope']},'oracle_input_precision':'FP16-rounded' if row['precision']=='FP16' else 'FP32','checked_query_rows':checked})
            fields=[k for k in schema.EXTRA['summary'] if k in row]
            record={**ctx,**{k:row[k] for k in fields},'validation_status':'PASS','reference_scope':row['reference_scope']}
            record['tokens_per_second']=str(float(row['sequences_per_second'])*int(row['N']))
            tables['summary'].append(record)
            size=2 if row['precision']=='FP16' else 4;heads=int(row['B'])*int(row['H']);n=int(row['N']);d=int(row['d'])
            expected=(10*heads*n*d+6*heads*n*n) if size==2 else (16*heads*n*d+8*heads*n*n)
            if int(row['peak_workspace_allocated_bytes'])!=expected:raise ValueError('Part A allocation ledger mismatch')
            tables['memory'].append({**ctx,'explicit_live_device_bytes':str(expected),'auxiliary_global_bytes':str(heads*n*n*(4+size)),'observed_device_delta_bytes':'','measurement':'persistent_QKVSPO_explicit_allocation_ledger_not_device_wide_peak'})
            tables['work'].append({**ctx,'valid_attention_pairs':str(heads*(n*(n+1)//2 if row['mode']=='causal' else n*n)),'scheduled_tiles':'','skipped_tiles':'','evidence_kind':'analytical_valid_attention_pairs_not_executed_instruction_counts'})
            if phase=='proposal':timings[identity(row)]=(ctx,row)
        for row in schema.read(folder/'samples.csv'):
            ctx,_=local[identity(row)]
            if row['input_digest']!=ctx['input_digest']:raise ValueError('Sample input fingerprint changed')
            tables['samples'].append({**ctx,**{k:row[k] for k in ['repeat','forward_ms','qk_ms','softmax_ms','pv_ms']}})
        if (folder/'resources.csv').exists():
            for row in schema.read(folder/'resources.csv'):
                ctx,_=local[identity(row)]
                tables['resources'].append({**ctx,**{k:row[k] for k in schema.EXTRA['resources']}})
    profile_count=0
    for meta in sorted((native/'profiles').glob('*/identity.json')):
        case=meta.parent;item=json.loads(meta.read_text());ctx,row=timings[tuple(item['identity'])]
        if item['binary_sha256']!=plan['binary_sha256']:raise ValueError('Profile binary mismatch')
        resources=schema.read(case/'program/resources.csv')
        if len(resources)!=3 or any(identity(resource)!=identity(row) for resource in resources):raise ValueError('Profile program configuration mismatch')
        data=captures(case/'raw.csv')
        if len(data)!=3:raise ValueError('Expected QK, softmax, PV captures')
        for (values,_),resource in zip(data,resources):
            dims=lambda name:[int(value.strip()) for value in values[name].strip('()').split(',')]
            if math.prod(dims('Grid Size'))!=int(resource['blocks']) or math.prod(dims('Block Size'))!=int(resource['threads']):raise ValueError('Profiler launch geometry differs from program configuration')
        for stage,(values,units) in zip(['QK','softmax','PV'],data):
            tables['profiles'].extend(report_profiles(metric_rows(ctx,values,units,item['rep'],str((case/'raw.csv').relative_to(out)),stage)));profile_count+=1
    audit=schema.validate(tables)
    audit.update(shared_cpu_reference_linked=True,profile_captures=profile_count,grid=plan['grid'])
    for name,rows in tables.items():schema.write(out/f'{name}.csv',name,rows)
    (out/'configs').mkdir(exist_ok=True);dump(out/'configs/part_a.json',{'configurations':configs,'frozen_selection':plan['selection']})
    dump(out/'audit.json',audit)
    dump(out/'manifest.json',{'schema_version':schema.VERSION,'part':'part_a','run_id':plan['run_id'],'grid':plan['grid'],'exporter':'part_a_v1','shared_cpu_reference_linked':True,'binary_sha256':plan['binary_sha256'],'sources':plan['sources'],'device':plan['device'],'profile_requested':plan['profile_requested'],'implementation_registry':{'fp32':'FP32 inputs/accumulation/softmax/output','wmma':'FP16 Q/K/V/P; FP32 S/accumulation/softmax/O','ptx':'FP16 Q/K/V/P; FP32 S/accumulation/softmax/O'},'raw_files':{str(p.relative_to(out)):sha(p) for p in sorted(native.rglob('*')) if p.is_file()}})
    (out/'ANALYSIS.md').write_text('# Part A\n\nAll timings link to shared FP64 oracle validation. N=256,d=64 uses full O comparison. Other large outputs use three sampled query rows per head plus all-output finiteness checks. Both Tensor Core variants use FP16 Q/K/V/P and FP32 S, accumulation, softmax arithmetic and O. Materialized normalized P rounding differs from Part C tile-wise unnormalized P rounding. Stage timings are independent passes, not components to sum into a forward median. All native outputs and profiler exports remain under raw/part_a/.\n')
    return audit

def save_report_results(out,target):
    out=Path(out);target=Path(target);target.mkdir(parents=True,exist_ok=True)
    for name in schema.HEADERS:
        if name=='profiles':schema.write(target/'profiles.csv','profiles',report_profiles(schema.read(out/'profiles.csv')))
        else:shutil.copy2(out/f'{name}.csv',target/f'{name}.csv')
    figures=target/'figures';figures.mkdir(exist_ok=True)
    for path in figures.iterdir():
        if path.is_file():path.unlink()
    for path in (out/'plots/part_a').glob('*'):
        if path.suffix in ['.png','.pdf']:shutil.copy2(path,figures/path.name)
    plan=json.loads((out/'raw/part_a/plan.json').read_text())
    text=['Part A mixed-precision evaluation',f"run={plan['run_id']}",f"grid={plan['grid']}",json.dumps(plan['device'],indent=2),
          'Tensor Core: FP16 Q/K/V/P; FP32 S/accumulation/softmax/O',
          'A rounds normalized P; C rounds unnormalized tile probabilities',
          'warmup=5; repeats=20; timing=CUDA events; preparation and comparison excluded',
          'Full O: small shapes and N=256,d=64; other large shapes: queries 0,N/2,N-1 per head',
          'Allocation: FP32=16BHNd+8BHN^2 bytes; mixed=10BHNd+6BHN^2 bytes',
          'Frozen configurations:',json.dumps(plan['selection'],indent=2),
          'Raw run records remain local; published CSVs and figures are from this run']
    example=next((out/'raw/part_a').glob('main_optimized_*/environment.txt'))
    text.extend(['Build and environment:',example.read_text()])
    settings=out/'logs/gpu_settings.txt'
    if settings.exists():text.extend(['GPU settings before the run:',settings.read_text()])
    (target/'environment.txt').write_text('\n'.join(text)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(publish(a.output),indent=2))
if __name__=='__main__':main()
