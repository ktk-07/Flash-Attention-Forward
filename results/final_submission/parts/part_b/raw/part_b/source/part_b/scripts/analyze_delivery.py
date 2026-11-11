#!/usr/bin/env python3
"""Verify retained Part B evidence and regenerate report assets without CUDA."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import results_schema as schema

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=ROOT/'part_b/results/notebook_full_20261110_044740')
    p.add_argument('--output',type=Path);p.add_argument('--verify-only',action='store_true')
    args=p.parse_args();bundle=args.bundle.resolve()
    index=json.loads((bundle/'raw/part_b/index.json').read_text())
    for name,digest in index['evidence_hashes'].items():
        if sha(bundle/name)!=digest:raise ValueError('Native evidence changed: '+name)
    manifest=json.loads((bundle/'manifest.json').read_text())
    snapshot=bundle/'raw/part_b/source'
    for name,digest in manifest['source_hashes'].items():
        if sha(snapshot/name)!=digest:raise ValueError('Source snapshot changed: '+name)
    tables={name:schema.read(bundle/(name+'.csv')) for name in schema.HEADERS}
    validation=schema.validate(tables)
    if any(row['status']!='PASS' for row in tables['correctness']):raise ValueError('Numerical failure in supplied data')
    native=[json.loads((bundle/entry['path']).read_text()) for entry in index['cases']]
    if any(row.get('materialised_failures',0) for row in native):raise ValueError('A/B comparison failure')
    audit=json.loads((bundle/'audit.json').read_text())
    verification={'schema_valid':validation['passed'],'native_hashes_verified':True,'source_snapshot_hashes_verified':True,
        'numerical_records':len(tables['correctness']),'samples':len(tables['samples']),
        'full_checks':sum(r['reference_scope']=='full' for r in tables['correctness']),
        'sampled_checks':sum(r['reference_scope']=='sampled' for r in tables['correctness']),
        'direct_A_B_checks':sum(r.get('materialised_compared',False) for r in native),
        'overall_evidence_audit_passed':audit['passed'],'profile_captures':audit['profile_captures'],
        'sanitizer_checks':audit['sanitizer_checks']}
    print(json.dumps(verification,indent=2))
    if args.verify_only:return
    out=(args.output or bundle/'supplementary/part_b/reanalysis').resolve();out.mkdir(parents=True,exist_ok=True)
    main_rows=[r for r in tables['summary'] if r['experiment']=='proposal']
    selected=[r for r in main_rows if r['variant']=='selected']
    dump_csv(out/'selected.csv',[{k:r[k] for k in ['B','H','N','d','config','forward_median_ms','forward_iqr_ms','tokens_per_second','peak_workspace_allocated_bytes','reference_scope']} for r in selected])
    matched=[]
    for scalar in [r for r in main_rows if r['variant']=='tiled']:
        vector=next(r for r in main_rows if r['variant']=='vector' and all(r[k]==scalar[k] for k in ['B','H','N','d','input_digest']))
        x,y=float(scalar['forward_median_ms']),float(vector['forward_median_ms'])
        matched.append({'N':int(scalar['N']),'d':int(scalar['d']),'scalar_ms':x,'vector_ms':y,'speedup_scalar_over_vector':x/y,'latency_reduction_percent':100*(1-y/x)})
    dump_csv(out/'matched_loading.csv',matched)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'streaming':'#7d8597','tiled':'#c17a16','vector':'#2a9d8f','selected':'#173e74'}
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,d in zip(axes,[32,64,128]):
        for variant in ['streaming','tiled','vector','selected']:
            rows=sorted([r for r in main_rows if int(r['d'])==d and r['variant']==variant],key=lambda r:int(r['N']))
            n=[int(r['N']) for r in rows];med=[float(r['forward_median_ms']) for r in rows]
            ax.plot(n,med,label=variant,color=colors[variant],marker='o',markersize=3,linestyle='--' if variant=='selected' else '-',linewidth=2 if variant=='selected' else 1)
            ax.fill_between(n,[float(r['forward_q1_ms']) for r in rows],[float(r['forward_q3_ms']) for r in rows],color=colors[variant],alpha=.12)
        ax.set(title=f'd = {d}',xlabel='Sequence length N',ylabel='Complete forward latency (ms)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Dense FP32, B=4/H=8 - medians with Q1-Q3 bands',fontsize=12)
    fig.tight_layout();fig.savefig(out/'part_b_latency.png',dpi=220,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for d,color in zip([32,64,128],['#173e74','#2a9d8f','#c17a16']):
        rows=sorted([r for r in matched if r['d']==d],key=lambda r:r['N'])
        axes[0].plot([r['N'] for r in rows],[r['speedup_scalar_over_vector'] for r in rows],marker='o',label=f'd={d}',color=color)
        rows=sorted([r for r in selected if int(r['d'])==d],key=lambda r:int(r['N']))
        axes[1].plot([int(r['N']) for r in rows],[int(r['peak_workspace_allocated_bytes'])/2**20 for r in rows],marker='o',label=f'd={d}',color=color)
    axes[0].axhline(1,color='gray',linestyle='--',linewidth=.8);axes[0].set(xlabel='Sequence length N',ylabel='Scalar median / float4 median',title='Matched W4/BK32 loading (>1 favours float4)')
    axes[1].set(xlabel='Sequence length N',ylabel='Explicit live allocation (MiB)',title='Measured Q/K/V/O storage; zero auxiliary global')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'part_b_loading_memory.png',dpi=220,bbox_inches='tight');plt.close(fig)
    frozen=json.loads((bundle/'configs/part_b.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,3.3))
    for ax,d in zip(axes,[32,64,128]):
        rank=frozen['scores'][str(d)];ax.bar(range(len(rank)),[r['geomean_ms'] for r in rank],color=['#173e74']+['#bcc9d9']*(len(rank)-1))
        ax.set_xticks(range(len(rank)),[r['config'].replace('_w','\nW').replace('_k','/BK') for r in rank],fontsize=7)
        ax.set(title=f'd={d}',ylabel='Tuning geometric mean (ms)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Frozen selection from N=256/1024; lowest geometric mean wins',fontsize=11)
    fig.tight_layout();fig.savefig(out/'part_b_tuning.png',dpi=220,bbox_inches='tight');plt.close(fig)
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print('Report assets:',out)
    print('The retained overall audit is unchanged; no CUDA or safety/profiler checks were run.')

if __name__=='__main__':main()
