#!/usr/bin/env python3
"""Plot validated canonical tables, preserving plotted data and comparison contracts."""
import argparse,csv,json,math,os,sys
os.environ.setdefault("MPLCONFIGDIR","/tmp/attention-matplotlib")
from pathlib import Path
import results_schema as schema
from workflow import dump,sha
from ncu_csv import metric_aliases

def write_rows(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metric_records(rows,name):
    aliases=metric_aliases(name);selected={}
    for row in rows:
        if row['metric'] not in aliases:continue
        key=tuple(row.get(k,'') for k in ['case_id','stage','profile_repeat'])
        old=selected.get(key)
        rank=lambda r:(r['availability']!='available',aliases.index(r['metric']))
        if old is None or rank(row)<rank(old):selected[key]=row
    return list(selected.values())

def generate(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out).resolve();tables={k:schema.read(out/f'{k}.csv') for k in schema.HEADERS};schema.validate(tables)
    rows=[r for r in tables['summary'] if r['experiment']=='proposal'];parts=sorted({r['part'] for r in rows});index=[];comparison=[]
    for part in parts:
        folder=out/'plots'/part;folder.mkdir(parents=True,exist_ok=True)
        subset=[r for r in rows if r['part']==part];write_rows(folder/'main_plot_data.csv',subset)
        def save(fig,name,description):
            fig.tight_layout()
            for ext in ['png','pdf']:fig.savefig(folder/f'{name}.{ext}',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Software':'attention-evaluation'},dpi=160)
            plt.close(fig);index.append({'part':part,'figure':str((folder/name).relative_to(out)),'description':description})
        modes=sorted({r['mode'] for r in subset});ds=sorted({int(r['d']) for r in subset})
        if part=='part_a':
            fig,axes=plt.subplots(1,2,figsize=(8,3.2))
            for ax,mode in zip(axes,['dense','causal']):
                for variant in ['naive','tiled','parallel','optimized','wmma','ptx']:
                    line=sorted([r for r in subset if r['mode']==mode and r['d']=='64' and r['variant']==variant],key=lambda r:int(r['N']))
                    if not line:continue
                    ax.errorbar([int(r['N']) for r in line],[float(r['forward_median_ms']) for r in line],yerr=[[float(r['forward_median_ms'])-float(r['forward_q1_ms']) for r in line],[float(r['forward_q3_ms'])-float(r['forward_median_ms']) for r in line]],marker='o',label=variant,capsize=2)
                ax.set(title=mode,xlabel='Sequence length',ylabel='Forward latency (ms)',yscale='log');ax.set_xticks([256,1024,2048,4096]);ax.grid(alpha=.25);ax.legend(fontsize=9)
            save(fig,'materialised_comparison','Part A d=64 progression, medians and interquartile ranges. Tensor Core variants use FP16 operands and FP32 accumulation/output.')

        for precision in ['FP32','FP16']:
            current=[r for r in subset if r['precision']==precision]
            if not current:continue
            fig,axes=plt.subplots(len(modes),len(ds),figsize=(5*len(ds),3.8*len(modes)),squeeze=False)
            for i,mode in enumerate(modes):
                for j,d in enumerate(ds):
                    ax=axes[i,j];selected=[r for r in current if r['mode']==mode and int(r['d'])==d]
                    for variant in sorted({r['variant'] for r in selected}):
                        line=sorted([r for r in selected if r['variant']==variant],key=lambda r:int(r['N']))
                        ax.errorbar([int(r['N']) for r in line],[float(r['forward_median_ms']) for r in line],yerr=[[float(r['forward_median_ms'])-float(r['forward_q1_ms']) for r in line],[float(r['forward_q3_ms'])-float(r['forward_median_ms']) for r in line]],marker='o',label=variant,capsize=2)
                    ax.set(title=f'{part}, {mode}, d={d}, {precision}',xlabel='Sequence length',ylabel='Forward latency (ms)',yscale='log');ax.grid(alpha=.25);ax.legend(fontsize=8)
            save(fig,f'kernel_progression_{precision.lower()}','All final-grid variants; median and interquartile range. Precision contracts remain separate.')
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for mode in modes:
            for precision in ['FP32','FP16']:
                group=sorted([r for r in subset if r['mode']==mode and r['precision']==precision and int(r['d'])==64],key=lambda r:(r['variant'],int(r['N'])))
                for variant in sorted({r['variant'] for r in group}):
                    line=[r for r in group if r['variant']==variant];label=f'{mode}/{variant}/{precision}'
                    axes[0].plot([int(r['N']) for r in line],[float(r['tokens_per_second']) for r in line],marker='o',label=label)
                    axes[1].plot([int(r['N']) for r in line],[int(r['peak_workspace_allocated_bytes'])/2**20 for r in line],marker='o',label=label)
        axes[0].set(xlabel='Sequence length (d=64)',ylabel='Tokens / second');axes[1].set(xlabel='Sequence length (d=64)',ylabel='Explicit live device buffers (MiB)')
        for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=6)
        save(fig,'throughput_and_memory','Measured forward throughput; explicit allocation ledger, not a device-wide peak.')
        fig,ax=plt.subplots(figsize=(9,4));errors=[]
        for mode in modes:
            for variant in sorted({r['variant'] for r in subset}):
                line=sorted([r for r in subset if r['mode']==mode and r['variant']==variant and r['d']=='64'],key=lambda r:int(r['N']))
                if not line:continue
                ax.plot([int(r['N']) for r in line],[float(r['max_absolute']) for r in line],marker='o',label=f'{mode}/{variant}/{line[0]["precision"]}')
                errors.extend({k:r[k] for k in ['mode','variant','precision','N','d','max_absolute','max_relative','reference_scope']} for r in line)
        ax.set(xlabel='Sequence length (d=64)',ylabel='Maximum checked absolute error',title='Numerical error: sampled scopes are labelled in CSV');ax.legend(fontsize=6);ax.grid(alpha=.25)
        write_rows(folder/'numerical_error_data.csv',errors);save(fig,'numerical_error','Errors from exact per-case CPU validation; full or sampled scope retained in CSV.')
        tuning=[r for r in tables['summary'] if r['part']==part and r['experiment']=='tuning'];write_rows(folder/'tuning_plot_data.csv',tuning)
        if tuning:
            fig,ax=plt.subplots(figsize=(max(9,len({r['config'] for r in tuning})*.35),4))
            groups={}
            for r in tuning:groups.setdefault((r['mode'],r['config']),[]).append(float(r['forward_median_ms']))
            keys=sorted(groups);ax.bar(range(len(keys)),[math.exp(sum(map(math.log,groups[k]))/len(groups[k])) for k in keys]);ax.set_xticks(range(len(keys)),[f'{a}/{b}' for a,b in keys],rotation=90,fontsize=6);ax.set(ylabel='Geometric mean tuning latency (ms)',yscale='log')
            save(fig,'tuning','Bounded candidate sweep, grouped by mode and configuration. Selection rules and frozen configurations are retained.')
        profiled=[r for r in tables['profiles'] if r['part']==part and r['availability']=='available']
        if part=='part_a':
            reads=metric_records(profiled,'dram__bytes_read.sum');writes=metric_records(profiled,'dram__bytes_write.sum')
            groups={}
            for kind,records in [('read',reads),('write',writes)]:
                for r in records:
                    key=(r['variant'],r['mode'],r['N'],r['stage'])
                    groups.setdefault(key,{}).setdefault(kind,[]).append(float(r['value']))
            keys=sorted(k for k,v in groups.items() if 'read' in v and 'write' in v)
            if keys:
                med=lambda a:sorted(a)[len(a)//2]/2**20
                read=[med(groups[k]['read']) for k in keys];write=[med(groups[k]['write']) for k in keys]
                fig,ax=plt.subplots(figsize=(13,4.8));x=list(range(len(keys)))
                ax.bar(x,read,label='DRAM read');ax.bar(x,write,bottom=read,label='DRAM write')
                ax.set_xticks(x,['/'.join(k) for k in keys],rotation=90,fontsize=6)
                ax.set(ylabel='DRAM traffic per kernel capture (MiB)',title='Separate Nsight stage captures; cache control=all');ax.legend();ax.grid(axis='y',alpha=.25)
                save(fig,'dram_traffic','Median read/write traffic over three instrumented captures per stage. These cold-cache captures are separate from normal forward timing.')
        metrics=['sm__warps_active.avg.pct_of_peak_sustained_active','smsp__warps_eligible.avg.per_cycle_active','smsp__pcsamp_warps_issue_stalled_mio_throttle']
        selected_metrics=[metric for metric in metrics if any(r['metric']==metric for r in profiled)]
        if selected_metrics:
            fig,axes=plt.subplots(len(selected_metrics),1,figsize=(13,4*len(selected_metrics)),squeeze=False);profile_data=[]
            for ax,metric in zip(axes[:,0],selected_metrics):
                groups={}
                for r in profiled:
                    if r['metric']==metric:groups.setdefault((r['variant'],r['mode'],r['N'],r['stage'],r['unit']),[]).append(float(r['value']))
                keys=sorted(groups);values=[sorted(groups[k])[len(groups[k])//2] for k in keys]
                for key,value in zip(keys,values):profile_data.append(dict(zip(['variant','mode','N','stage','unit'],key),metric=metric,value=value))
                ax.bar(range(len(keys)),values);ax.set_xticks(range(len(keys)),['/'.join(k[:-1]) for k in keys],rotation=90,fontsize=6);ax.set(title=metric,ylabel=keys[0][-1] if keys else '');ax.grid(axis='y',alpha=.2)
            write_rows(folder/'profiler_plot_data.csv',profile_data);save(fig,'profiler_counters','Median raw Nsight counter across captures. MIO sample counts are not percentages or runtime fractions. Missing counters are omitted and retained in profiles.csv with reasons.')
        stages=[r for r in subset if r.get('qk_median_ms')]
        if stages:
            write_rows(folder/'stage_plot_data.csv',stages)
            fig,ax=plt.subplots(figsize=(10,4));stage_rows=[r for r in stages if r['variant'] in ['optimized','wmma','ptx'] and r['d']=='64'];x=list(range(len(stage_rows)));bottom=[0.]*len(x)
            for stage in ['qk','softmax','pv']:
                values=[float(r[stage+'_median_ms']) for r in stage_rows];ax.bar(x,values,bottom=bottom,label=stage);bottom=[a+b for a,b in zip(bottom,values)]
            ax.set_xticks(x,[f"{r['mode']}/{r['variant']}/N{r['N']}" for r in stage_rows],rotation=90,fontsize=7);ax.set(ylabel='Independent stage median (ms)',title='Stage breakdown; sum is not the measured forward median');ax.legend()
            save(fig,'stage_breakdown','Independent stage passes; fused stages are unavailable, not zero.')
    # Frozen C family winners; avoid selecting minima on the reporting grid.
    c_registry=out/'configs/part_c.json';c_data=json.loads(c_registry.read_text()) if c_registry.exists() else {}
    def choose_c(precision,d):
        family=c_data.get('frozen_selection',{}).get('families',{})
        key='cuda_core' if precision=='FP32' else 'tensor_core'
        value=family.get(key,{}).get(str(d))
        if isinstance(value,dict):return value.get('kernel')
        if isinstance(value,str):return value
        return None
    # Cross-part comparisons use explicit common workload and precision.
    for mode,left,right in [('causal','part_a','part_c'),('dense','part_a','part_b')]:
        if left not in parts or right not in parts:continue
        for precision in ['FP32','FP16']:
            arows=[r for r in rows if r['part']==left and r['mode']==mode and r['precision']==precision and r['variant'] in (['optimized'] if precision=='FP32' else ['wmma','ptx'])]
            for a in arows:
                matches=[r for r in rows if r['part']==right and r['mode']==mode and r['precision']==precision and all(r[k]==a[k] for k in (['B','H','N','d','pattern','input_digest']+(['rounded_input_digest'] if precision=='FP16' else [])))]
                if right=='part_c':matches=[r for r in matches if r['variant']==(choose_c(precision,int(a['d'])) if precision=='FP32' else a['variant'])]
                if right=='part_b':matches=[r for r in matches if r['variant']=='selected']
                if len(matches)!=1:continue
                b=matches[0]
                if any(a[k]!=b[k] for k in ['input_precision','accumulator_precision','output_precision']):raise ValueError('Cross-part precision metadata differs')
                comparison.append({'mode':mode,'precision':precision,'B':a['B'],'H':a['H'],'N':a['N'],'d':a['d'],'baseline_part':left,'baseline_variant':a['variant'],'target_part':right,'target_variant':b['variant'],'baseline_ms':a['forward_median_ms'],'baseline_q1_ms':a['forward_q1_ms'],'baseline_q3_ms':a['forward_q3_ms'],'target_ms':b['forward_median_ms'],'target_q1_ms':b['forward_q1_ms'],'target_q3_ms':b['forward_q3_ms'],'speedup':float(a['forward_median_ms'])/float(b['forward_median_ms']),'input_digest':a['input_digest'],'rounded_input_digest':a['rounded_input_digest'],'numerical_contract':'matched_FP32' if precision=='FP32' else 'FP16_inputs_FP32_accum_output_different_P_rounding_order'})
    for mode,left,right in [('causal','part_a','part_c'),('dense','part_a','part_b')]:
        if left in parts and right in parts:
            for precision in (['FP32','FP16'] if right=='part_c' else ['FP32']):
                count=sum(r['part']==left and r['mode']==mode and r['precision']==precision and r['variant'] in (['optimized'] if precision=='FP32' else ['wmma','ptx']) for r in rows)
                actual=sum(r['target_part']==right and r['precision']==precision for r in comparison)
                if actual!=count:raise ValueError(f'Missing frozen cross-part comparison: {right}/{precision}, expected {count}, found {actual}')
    folder=out/'plots/combined';folder.mkdir(parents=True,exist_ok=True);write_rows(out/'comparison.csv',comparison)
    if comparison:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        groups={}
        for row in comparison:groups.setdefault((row['target_part'],row['precision'],row['d'],row['baseline_variant']),[]).append(row)
        for key,group in sorted(groups.items()):
            group.sort(key=lambda r:int(r['N']));label='/'.join(key)
            axes[0].plot([int(r['N']) for r in group],[float(r['speedup']) for r in group],marker='o',label=label)
            axes[1].errorbar([int(r['N']) for r in group],[float(r['target_ms']) for r in group],yerr=[[float(r['target_ms'])-float(r['target_q1_ms']) for r in group],[float(r['target_q3_ms'])-float(r['target_ms']) for r in group]],marker='o',label=label,capsize=2)
            axes[1].plot([int(r['N']) for r in group],[float(r['baseline_ms']) for r in group],linestyle='--',label='A/'+key[3]+'/'+key[1]+'/d'+key[2])
        axes[0].axhline(1,color='gray',linestyle='--');axes[0].set(xlabel='Sequence length',ylabel='Part A latency / fused latency');axes[1].set(xlabel='Sequence length',ylabel='Forward latency (ms); A dashed, fused solid',yscale='log')
        for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=7)
        fig.suptitle('Same inputs and FP32 accumulation/output; FP16 probability rounding order differs',fontsize=10)
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(folder/f'cross_part.{ext}',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Software':'attention-evaluation'},dpi=160)
        plt.close(fig);index.append({'part':'combined','figure':'plots/combined/cross_part','description':'Frozen selections on identical inputs. FP16 comparison matches accumulator/output precision; probability rounding order differs.'})
        memory_data=[];fig,ax=plt.subplots(figsize=(9,4))
        for part in parts:
            for precision in ['FP32','FP16']:
                line=[r for r in rows if r['part']==part and r['d']=='64' and r['mode']=='causal' and r['precision']==precision]
                # Allocation size is identical within each part/precision/shape.
                line=list({r['N']:r for r in line}.values());line.sort(key=lambda r:int(r['N']))
                if not line:continue
                ax.plot([int(r['N']) for r in line],[int(r['peak_workspace_allocated_bytes'])/2**20 for r in line],marker='o',label=f'{part}/{precision}')
                memory_data.extend({k:r[k] for k in ['part','precision','B','H','N','d','peak_workspace_allocated_bytes']} for r in line)
        ax.set(xlabel='Sequence length (causal, d=64)',ylabel='Explicit live device buffers (MiB)');ax.legend();ax.grid(alpha=.25);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(folder/f'materialised_fused_memory.{ext}',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Software':'attention-evaluation'},dpi=160)
        plt.close(fig);write_rows(folder/'materialised_fused_memory.csv',memory_data);index.append({'part':'combined','figure':'plots/combined/materialised_fused_memory','description':'Same-input explicit allocation evidence; not total process memory.'})

    if 'part_b' in parts and 'part_c' in parts:
        work_comparison=[]
        def traffic(row):
            amounts=[]
            for metric in ['dram__bytes_read.sum','dram__bytes_write.sum']:
                scale={'byte':1,'Kbyte':1000,'Mbyte':1000**2,'Gbyte':1000**3,'Tbyte':1000**4}
                raw=[p for p in metric_records(tables['profiles'],metric) if p['case_id']==row['case_id'] and p['availability']=='available']
                if any(p['unit'] not in scale for p in raw):return ''
                values=sorted(float(p['value'])*scale[p['unit']] for p in raw)
                if not values:return ''
                amounts.append(values[len(values)//2])
            return sum(amounts)
        for b in [r for r in rows if r['part']=='part_b' and r['precision']=='FP32' and r['variant']=='selected']:
            matches=[r for r in rows if r['part']=='part_c' and r['precision']=='FP32' and r['variant']==choose_c('FP32',int(b['d'])) and all(r[k]==b[k] for k in ['B','H','N','d','pattern','input_digest'])]
            if len(matches)!=1:raise ValueError('Missing same-input dense/causal workload comparison')
            c=matches[0];heads=int(b['B'])*int(b['H']);n=int(b['N'])
            work_comparison.append({'B':b['B'],'H':b['H'],'N':b['N'],'d':b['d'],'input_digest':b['input_digest'],'dense_ms':b['forward_median_ms'],'causal_ms':c['forward_median_ms'],'dense_valid_pairs':heads*n*n,'causal_valid_pairs':heads*n*(n+1)//2,'dense_dram_bytes':traffic(b),'causal_dram_bytes':traffic(c),'interpretation':'different_attention_workloads_not_output_equivalence_or_same_operation_speedup'})
        write_rows(folder/'dense_causal_work.csv',work_comparison)
        fig,axes=plt.subplots(1,3,figsize=(15,4));current=sorted([r for r in work_comparison if r['d']=='64'],key=lambda r:int(r['N']))
        for prefix in ['dense','causal']:
            axes[0].plot([int(r['N']) for r in current],[float(r[prefix+'_ms']) for r in current],marker='o',label=prefix)
            axes[1].plot([int(r['N']) for r in current],[r[prefix+'_valid_pairs'] for r in current],marker='o',label=prefix)
            available=[r for r in current if r[prefix+'_dram_bytes']!='']
            if available:axes[2].plot([int(r['N']) for r in available],[r[prefix+'_dram_bytes']/2**20 for r in available],marker='o',label=prefix)
        for ax,label in zip(axes,['Latency (ms)','Analytical valid attention pairs','Measured DRAM traffic (MiB)']):
            ax.set(xlabel='Sequence length (d=64)',ylabel=label);ax.grid(alpha=.25)
            if ax.lines:ax.legend()
            else:ax.text(.5,.5,'No compatible counters',transform=ax.transAxes,ha='center')
        fig.suptitle('Dense B versus causal C: different attention workloads');fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(folder/f'dense_causal_work.{ext}',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Software':'attention-evaluation'},dpi=160)
        plt.close(fig);index.append({'part':'combined','figure':'plots/combined/dense_causal_work','description':'Distinct dense/causal workloads on identical inputs; analytical pairs and measured traffic remain separate.'})
    dump(out/'figure_index.json',{'figures':index,'parts_present':parts,'unavailable_parts':[p for p in ['part_a','part_b','part_c'] if p not in parts],'comparison_rows':len(comparison),'inputs':{f'{k}.csv':sha(out/f'{k}.csv') for k in schema.HEADERS}})
    return index
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);generate(p.parse_args().output)
