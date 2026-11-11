"""Reconstruct the seven public tables solely from retained Part B evidence."""
import argparse
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import results_schema as schema
from workflow import dump,sha
from ncu_csv import captures,metric_rows

def config_name(c):return f"{c['variant']}_w{c['warps']}_k{c['key_tile']}"

def export(out):
    out=Path(out);manifest=json.loads((out/'manifest.json').read_text())
    index=json.loads((out/'raw/part_b/index.json').read_text())
    # Provenance guards apply to offline regeneration as well as initial export.
    for relative,digest in index['evidence_hashes'].items():
        if sha(out/relative)!=digest:raise ValueError('Native evidence changed: '+relative)
    tables={name:[] for name in schema.HEADERS};contexts={}
    for entry in index['cases']:
        r=json.loads((out/entry['path']).read_text());c=r
        context={key:'' for key in schema.COMMON}
        context.update(experiment=entry['experiment'],precision='FP32',B=r['B'],H=r['H'],N=r['N'],d=r['d'],mode='dense',
            variant=entry['public_variant'],tile=r['key_tile'],run_id=manifest['run_id'],part='part_b',config=config_name(c),pattern=r['pattern'],
            layout='BHNd_contiguous_row_major',input_precision='FP32',accumulator_precision='FP32',output_precision='FP32',
            generator='splitmix64-fp32-v1',seed=5206,input_digest=r['input_digest'])
        context={k:str(v) for k,v in context.items()};context['case_id']=schema.case_id(context);contexts[entry['path']]=context
        scope=r['reference_scope']
        tables['correctness'].append({**context,'stage':'O','atol':1e-4,'rtol':1e-4,
            **{k:r[k] for k in ['max_absolute','max_relative','failures','status','checked_query_rows','checked_elements','nonfinite']},
            'oracle_input_precision':'FP32','reference_scope':scope})
        samples=r['samples_ms']
        if r['status']!='PASS' or not samples:continue
        values=sorted(samples)
        def quantile(p):
            x=(len(values)-1)*p;lo=int(x);return values[lo]+(x-lo)*(values[min(lo+1,len(values)-1)]-values[lo])
        median=statistics.median(values)
        tables['samples'].extend({**context,'repeat':i,'forward_ms':x} for i,x in enumerate(samples))
        tables['summary'].append({**context,'atol':1e-4,'rtol':1e-4,'max_absolute':r['max_absolute'],'max_relative':r['max_relative'],
            'warmup':5,'repeats':20,'forward_median_ms':median,'forward_q1_ms':quantile(.25),'forward_q3_ms':quantile(.75),
            'forward_iqr_ms':quantile(.75)-quantile(.25),'sequences_per_second':1000*r['B']/median,
            'tokens_per_second':1000*r['B']*r['N']/median,'peak_workspace_allocated_bytes':r['explicit_live_device_bytes'],
            'setup_host_ms':r['setup_host_ms'],'validation_status':'PASS','reference_scope':scope})
        tables['resources'].append({**context,'stage':'forward',**{k:r[k] for k in schema.EXTRA['resources'] if k!='stage'}})
        tables['memory'].append({**context,**{k:r[k] for k in ['explicit_live_device_bytes','auxiliary_global_bytes','observed_device_delta_bytes']},
            'measurement':'four_live_cudaMalloc_buffers; device_delta_context_initialized_WDDM_may_be_unreliable'})
        query_tiles=(r['N']+r['warps']-1)//r['warps']
        key_tiles=r['N'] if r['variant']=='streaming' else (r['N']+r['key_tile']-1)//r['key_tile']
        tables['work'].append({**context,'valid_attention_pairs':r['B']*r['H']*r['N']**2,
            'scheduled_tiles':r['B']*r['H']*query_tiles*key_tiles,'skipped_tiles':0,'evidence_kind':'analytical_dense_launch_loop_counts'})
    for profile in index['profiles']:
        context=contexts[profile['case']]
        for data,units in captures(out/profile['csv']):
            tables['profiles'].extend(metric_rows(context,data,units,profile['repeat'],profile['csv'],'forward'))
    # Round-trip through the fixed schemas: identity and statistic checks use CSV strings.
    for name,rows in tables.items():schema.write(out/(name+'.csv'),name,rows)
    tables={name:schema.read(out/(name+'.csv')) for name in schema.HEADERS}
    audit=schema.validate(tables)
    failed=[r for r in tables['correctness'] if r['status']=='FAIL']
    safety=index['sanitizers']
    expected_tools={'memcheck','racecheck','synccheck','initcheck'}
    selected=json.loads((out/'configs/part_b.json').read_text())['selected']
    safety_ok=all(expected_tools <= {x['tool'] for x in safety if x['config']==config_name(c) and x['exit_code']==0} for c in selected.values())
    captures_count=len(index['profiles'])
    audit.update(passed=audit['passed'] and not failed and safety_ok and (not manifest['profile_requested'] or captures_count>=6),
        shared_cpu_reference_linked=manifest['shared_cpu_reference_linked'],profile_captures=captures_count,
        sanitizer_checks=len(safety),frozen_sanitizers_passed=safety_ok,failed_checks=len(failed),
        final_report_grid=manifest['grid']=='full',profiling_complete=manifest['profile_requested'] and captures_count>=6)
    dump(out/'audit.json',audit)
    plot(out,tables)
    return audit

def plot(out,tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'plots/part_b';folder.mkdir(parents=True,exist_ok=True)
    main=[r for r in tables['summary'] if r['experiment']=='proposal']
    figures=[]
    for metric,label,filename in [('forward_median_ms','Complete forward latency (ms)','latency'),('peak_workspace_allocated_bytes','Explicit live device buffers (bytes)','memory')]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for d in sorted({r['d'] for r in main},key=int):
            for variant in sorted({r['variant'] for r in main}):
                rows=sorted([r for r in main if r['d']==d and r['variant']==variant],key=lambda r:int(r['N']))
                ax.plot([int(r['N']) for r in rows],[float(r[metric]) for r in rows],marker='o',label=f'd={d}, {variant}')
        ax.set(xlabel='Sequence length',ylabel=label);ax.grid(alpha=.25);ax.legend();fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(folder/f'{filename}.{ext}',dpi=160)
        plt.close(fig);figures.append('plots/part_b/'+filename)
    tuning=[r for r in tables['summary'] if r['experiment']=='tuning']
    schema.write(folder/'plotted_main.csv','summary',main);schema.write(folder/'plotted_tuning.csv','summary',tuning)
    dump(out/'figure_index.json',{'figures':figures})
    selected=[r for r in main if r['variant']=='selected']
    text='# Part B dense FlashAttention evaluation\n\n'
    text+='FP32 contiguous BHNd inputs, FP32 accumulation/output. The shared C++ FP64 oracle validates every accepted timing case outside timing. Full small-case checks and sampled large-case checks are distinguished in correctness.csv. All outputs are checked for finiteness.\n\n'
    text+='| N | d | Frozen configuration | Median ms | IQR ms | Explicit bytes |\n|---|---|---|---|---|---|\n'
    for r in selected:text+=f"| {r['N']} | {r['d']} | {r['config']} | {float(r['forward_median_ms']):.6f} | {float(r['forward_iqr_ms']):.6f} | {r['peak_workspace_allocated_bytes']} |\n"
    text+='\nThe allocation ledger is 4 × B × H × N × d × sizeof(float), covering Q/K/V/O, with zero auxiliary global allocation. It is linear in N. Shared K/V/probability tiles and registers are per-block resources. Device free-memory deltas are observational and may be unreliable under Windows WDDM.\n\n'
    text+='## Controlled loading comparison\n\nSame W4/BK32 scalar and float4 kernels on the same generated tensors. Ratios below compare median latencies from separate sequential cases; close ratios are not statistically conclusive.\n\n| N | d | Scalar ms | float4 ms | Scalar / float4 |\n|---|---|---|---|---|\n'
    for scalar in [r for r in main if r['variant']=='tiled']:
        vector=next(r for r in main if r['variant']=='vector' and all(r[k]==scalar[k] for k in ['B','H','N','d','input_digest']))
        x,y=float(scalar['forward_median_ms']),float(vector['forward_median_ms'])
        text+=f"| {scalar['N']} | {scalar['d']} | {x:.6f} | {y:.6f} | {x/y:.3f} |\n"
    checks=tables['correctness'];full=sum(r['reference_scope']=='full' for r in checks);sampled=len(checks)-full
    text+=f'\nValidation records: {full} full and {sampled} sampled; {sum(int(r["checked_elements"]) for r in checks):,} oracle-compared elements. Failed records: {sum(r["status"]!="PASS" for r in checks)}. All timed cases have matching validation records and input fingerprints.\n\n'
    text+='The streaming and tiled kernels share the warp-per-query mapping. Tiling changes both K/V reuse and online update granularity; this comparison does not isolate one change. Tiled versus float4 with identical warps/key tile is the loading ablation. Candidate choices are frozen using geometric mean latency on tuning lengths only, with lexical ties. Main lengths include held-out lengths.\n\n'
    frozen=json.loads((out/'configs/part_b.json').read_text())
    if 'scores' in frozen:
        text+='## Tuning and bottleneck interpretation\n\n| d | Frozen winner | Tuning geometric mean ms | Slowest candidate | Tuning geometric mean ms |\n|---|---|---|---|---|\n'
        for d,ranking in frozen['scores'].items():
            fast,slow=ranking[0],ranking[-1]
            text+=f'| {d} | {fast["config"]} | {fast["geomean_ms"]:.6f} | {slow["config"]} | {slow["geomean_ms"]:.6f} |\n'
        text+='\nLarger shared tiles need not win: they increase per-block storage and can reduce occupancy. Cooperative loading adds barriers, probability-tile communication and warp reductions. Global-cache reuse can make the streaming baseline competitive. These are source/resource-based explanations to investigate; without fresh counters, the measurements do not establish DRAM bandwidth, bank conflicts or a definitive hardware bottleneck.\n\n'
    text+='Five warmups and twenty CUDA-event samples measure complete fused forward only. Allocation, transfers, oracle, resource queries and file I/O are excluded. Fixed order, display-GPU contention, unmanaged clocks and within-case variation limit close-winner conclusions. No same-machine A/B/C performance comparison is claimed by this standalone run.\n\n'
    text+='Fresh profiling requested: '+str(json.loads((out/'manifest.json').read_text())['profile_requested'])+'. See audit.json for actual captures and safety checks. Missing counters must not be interpreted as zero traffic. AI-assisted implementation and analysis require member review and disclosure.\n'
    (out/'ANALYSIS.md').write_text(text,encoding='utf-8')
    (out/'REPORT.md').write_text(text,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analyze-only',required=True,type=Path);a=p.parse_args();result=export(a.analyze_only);print(json.dumps(result,indent=2));sys.exit(0 if result['passed'] else 1)
