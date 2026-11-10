"""Matched FA1/FA2 baseline comparisons from canonical tables; no GPU or winner search."""
import csv,json,math
from pathlib import Path

def generate(out,tables,selection):
    out=Path(out);rows=[r for r in tables['summary'] if r['part']=='part_c' and r['experiment']=='proposal']
    tc_baseline='fa1_ptx' if any(r['variant']=='fa1_ptx' for r in rows) else 'fa1_wmma'
    if not any(r['variant']==tc_baseline for r in rows):return [] # Historical eight-kernel studies.
    if not selection.get('families'):raise ValueError('Missing frozen FA2 selection for FA1 comparisons')
    baselines=[r for r in rows if r['variant'] in ['fa1',tc_baseline]]
    shapes={tuple(r[k] for k in ['B','H','N','d','mode','pattern']) for r in baselines}
    matched=[];cross=[]
    def find(variant,key):
        found=[r for r in rows if r['variant']==variant and tuple(r[k] for k in ['B','H','N','d','mode','pattern'])==key]
        if len(found)!=1:raise ValueError(f'Missing or duplicate FA1 comparison: {variant}/{key}')
        return found[0]
    for key in sorted(shapes):
        d=key[3];core=selection['families']['cuda_core'][d]
        pairs=[('fa1',core,'FP32_inputs_accumulation_softmax_output')]
        if tc_baseline=='fa1_ptx':pairs.append((tc_baseline,'ptx','FP16_QKV_P;FP32_accumulation_softmax_output;same_m16n8k16_mapping;independently_tuned;schedule_and_normalisation_differ'))
        else:pairs += [(tc_baseline,'wmma','FP16_QKV_P;FP32_accumulation_softmax_output;normalisation_schedule_differs'),(tc_baseline,'ptx','FP16_QKV_P;FP32_accumulation_softmax_output;MMA_mapping_and_normalisation_differ')]
        pairs.append((tc_baseline,core,'cross_precision_FA1_FP16_vs_FA2_FP32_not_equivalent_precision'))
        for baseline,candidate,contract in pairs:
            a,b=find(baseline,key),find(candidate,key)
            for record in [a,b]:
                expected='FP16' if record['variant'] in ['fa1_ptx','fa1_wmma','wmma','ptx'] else 'FP32'
                if any(record[field]!=expected for field in ['precision','input_precision']) or any(record[field]!='FP32' for field in ['accumulator_precision','output_precision']):raise ValueError('FA1 comparison numerical contract differs')
            if a['input_digest']!=b['input_digest']:raise ValueError('FA1 comparison base inputs differ')
            if a['precision']==b['precision']=='FP16' and a['rounded_input_digest']!=b['rounded_input_digest']:raise ValueError('FA1 comparison effective FP16 inputs differ')
            row={**dict(zip(['B','H','N','d','mode','pattern'],key)),'baseline':baseline,'baseline_config':a['config'],'candidate':candidate,'candidate_config':b['config'],'baseline_precision':a['precision'],'candidate_precision':b['precision'],'baseline_ms':a['forward_median_ms'],'baseline_q1_ms':a['forward_q1_ms'],'baseline_q3_ms':a['forward_q3_ms'],'candidate_ms':b['forward_median_ms'],'candidate_q1_ms':b['forward_q1_ms'],'candidate_q3_ms':b['forward_q3_ms'],'speedup':float(a['forward_median_ms'])/float(b['forward_median_ms']),'contract':contract,'input_digest':a['input_digest'],'baseline_case_id':a['case_id'],'candidate_case_id':b['case_id']}
            (cross if baseline==tc_baseline and candidate==core else matched).append(row)
    for name,data in [('fa1_fa2_comparison.csv',matched),('fa1_cross_precision.csv',cross)]:
        with (out/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/attention-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'plots/part_c';folder.mkdir(parents=True,exist_ok=True);figures=[]
    for name,data in [('fa1_fa2_comparison',matched),('fa1_cross_precision',cross)]:
        ds=sorted({int(r['d']) for r in data});fig,axes=plt.subplots(1,len(ds),figsize=(5*len(ds),4),squeeze=False)
        for ax,d in zip(axes[0],ds):
            for pair in sorted({(r['baseline'],r['candidate']) for r in data if int(r['d'])==d}):
                line=sorted([r for r in data if int(r['d'])==d and (r['baseline'],r['candidate'])==pair],key=lambda r:int(r['N']))
                ax.plot([int(r['N']) for r in line],[float(r['speedup']) for r in line],marker='o',label=' / '.join(pair))
            ax.axhline(1,color='gray',linestyle='--');ax.set(title=f'Causal d={d}',xlabel='Sequence length',ylabel='FA1 median / FA2 median');ax.grid(alpha=.25);ax.legend(fontsize=7)
        fig.suptitle('Own FA1/FA2 implementations; matched precision' if name=='fa1_fa2_comparison' else 'Cross precision: FA1 Tensor Core vs FA2 CUDA core; not equivalent accuracy',fontsize=10);fig.tight_layout()
        for extension in ['png','pdf']:fig.savefig(folder/f'{name}.{extension}',dpi=160,metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Software':'attention-fa1-comparison'})
        plt.close(fig);figures.append({'part':'part_c','figure':'plots/part_c/'+name,'description':'CSV includes exact case IDs, precision labels, medians and quartiles; no official-library parity claim.'})
    (out/'fa1_comparison_index.json').write_text(json.dumps({'figures':figures,'data':['fa1_fa2_comparison.csv','fa1_cross_precision.csv'],'official_FA1_measured':False},indent=2)+'\n')
    return figures
