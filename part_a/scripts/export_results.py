#!/usr/bin/env python3
"""Part A adapter using the shared Part A-based report schema; native data retained."""
import argparse,csv,json,math,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import results_schema as schema
from workflow import dump,sha
from ncu_csv import captures,metric_rows

def context(row,run_id,phase):
    result={k:'' for k in schema.COMMON}
    result.update(**{k:row[k] for k in ['precision','B','H','N','d','mode','variant','tile','softmax_threads']})
    result.update(experiment=phase,run_id=run_id,part='part_a',config=f"{row['variant']}:tile{row['tile']}:sm{row['softmax_threads']}:d{row['d']}",pattern=row['pattern'],layout='BHNd_contiguous_row_major',input_precision=row['precision'],accumulator_precision=row['precision'],output_precision=row['precision'],generator='splitmix64-fp32-v1',seed='5206',input_digest=row['input_digest'],rounded_input_digest=row['effective_input_digest'] if row['precision']=='FP16' else '')
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
            expected=size*(4*heads*n*d+2*heads*n*n)
            if int(row['peak_workspace_allocated_bytes'])!=expected:raise ValueError('Part A allocation ledger mismatch')
            tables['memory'].append({**ctx,'explicit_live_device_bytes':str(expected),'auxiliary_global_bytes':str(2*heads*n*n*size),'observed_device_delta_bytes':'','measurement':'persistent_QKVSPO_explicit_allocation_ledger_not_device_wide_peak'})
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
            tables['profiles'].extend(metric_rows(ctx,values,units,item['rep'],str((case/'raw.csv').relative_to(out)),stage));profile_count+=1
    audit=schema.validate(tables)
    audit.update(shared_cpu_reference_linked=True,profile_captures=profile_count,grid=plan['grid'])
    for name,rows in tables.items():schema.write(out/f'{name}.csv',name,rows)
    (out/'configs').mkdir(exist_ok=True);dump(out/'configs/part_a.json',{'configurations':configs,'frozen_selection':plan['selection']})
    dump(out/'audit.json',audit)
    dump(out/'manifest.json',{'schema_version':schema.VERSION,'part':'part_a','run_id':plan['run_id'],'grid':plan['grid'],'exporter':'part_a_v1','shared_cpu_reference_linked':True,'binary_sha256':plan['binary_sha256'],'sources':plan['sources'],'device':plan['device'],'profile_requested':plan['profile_requested'],'implementation_registry':{'fp32':'FP32 inputs/accumulation/softmax/output','tensorcore':'FP16 inputs/WMMA accumulators/softmax/output'},'raw_files':{str(p.relative_to(out)):sha(p) for p in sorted(native.rglob('*')) if p.is_file()}})
    (out/'ANALYSIS.md').write_text('# Part A\n\nAll timings link to shared FP64 oracle validation. Large outputs use three sampled query rows per head plus all-output finiteness checks. Stage timings are independent passes, not components to sum into a forward median. All native outputs and profiler exports remain under raw/part_a/.\n')
    return audit

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(publish(a.output),indent=2))
if __name__=='__main__':main()
