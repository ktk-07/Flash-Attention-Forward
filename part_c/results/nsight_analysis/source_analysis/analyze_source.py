"""Export reproducible CUDA/SASS stall correlation from representative NCU reports."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();out=a.run.resolve()
plan=json.loads((out/'plan.json').read_text());root=Path(__file__).resolve().parents[2]
dest=out/'source_analysis';dest.mkdir(exist_ok=True)
script=Path(__file__).resolve();shutil.copy2(script,dest/'analyze_source.py')
(dest/'processing.json').write_text(json.dumps({'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'capture_rep':0,'purpose':'Representative source correlation; aggregate metric repeat statistics remain separate'},indent=2)+'\n')
source_files=[out/'source_snapshot'/name for name in plan['source_hashes'] if name.endswith('.cu')]
metrics=','.join('smsp__pcsamp_warps_issue_stalled_'+s for s in ['short_scoreboard','long_scoreboard','mio_throttle'])
ops=[];top=[];export_audits=[]
def write(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
for b,h,n,d in plan['shapes']:
    for kernel in plan['kernels']:
        label=f'{kernel}_n{n}_d{d}_r0';case=out/'cases'/label
        cmd=['ncu','--import',str(case/'profile.ncu-rep'),'--page','source','--print-source','cuda,sass','--resolve-source-file',','.join(map(str,source_files)),'--metrics',metrics,'--csv']
        with (out/'commands.jsonl').open('a') as f:f.write(json.dumps({'label':'source_'+label,'argv':cmd,'cwd':str(root),'started_utc':datetime.now(timezone.utc).isoformat()})+'\n')
        with (dest/(label+'.csv')).open('w') as stdout,(dest/(label+'.log')).open('w') as stderr:
            result=subprocess.run(cmd,stdout=stdout,stderr=stderr)
        with (out/'commands.jsonl').open('a') as f:f.write(json.dumps({'label':'source_'+label,'exit_code':result.returncode})+'\n')
        if result.returncode:raise SystemExit('Source export failed: '+label)
        with (dest/(label+'.csv')).open() as f:rows=list(csv.reader(f))
        file='';line='';code='';columns=None;instructions=[]
        for row in rows:
            if not row:continue
            if row[0]=='File Path':file=row[1];continue
            if row[0]=='Line No':columns=row;continue
            if columns is None or len(row)!=len(columns):continue
            if row[0]:line=row[0];code=row[1]
            if not row[2].startswith('0x'):continue
            values={name:int(float(row[i].replace(',','') or 0)) for i,name in enumerate(columns) if name.startswith('stall_')}
            inst=row[3].strip()
            match=re.search(r'\b(LDL|STL|LDS|STS|SHFL|MUFU|HMMA|FFMA|FADD|FMUL|LDG|STG)\b',inst)
            op=match.group(1) if match else 'other'
            instructions.append({'kernel':kernel,'N':n,'file':file,'line':line,'cuda':code,'address':row[2],'sass':inst,'operation':op,**values})
        if not instructions:raise SystemExit('No correlated source/SASS instructions for '+label)
        # Optimised/inlined code can map one PC to several CUDA source lines.
        # Count each instruction only once when computing stall percentages.
        unique={}
        for r in instructions:
            previous=unique.get(r['address'])
            if previous:
                for name in ['stall_short_sb','stall_long_sb','stall_mio']:
                    if previous.get(name,0)!=r.get(name,0):
                        raise ValueError('Inconsistent repeated PC metrics: '+label)
                if Path(r['file']).name.startswith('flash_attention_') and not Path(previous['file']).name.startswith('flash_attention_'):
                    unique[r['address']]=r
            else:unique[r['address']]=r
        export_audits.append({'kernel':kernel,'N':n,'exported_instruction_rows':len(instructions),'unique_pcs':len(unique),'duplicate_source_mappings':len(instructions)-len(unique)})
        instructions=list(unique.values())
        totals=defaultdict(int)
        for r in instructions:
            for name in ['stall_short_sb','stall_long_sb','stall_mio']:totals[name]+=r.get(name,0)
        for name,total in totals.items():
            if not total:continue
            groups=defaultdict(int)
            for r in instructions:groups[r['operation']]+=r.get(name,0)
            for op,count in groups.items():ops.append({'kernel':kernel,'N':n,'stall':name,'operation_at_sampled_pc':op,'samples':count,'percent_within_this_stall':100*count/total})
            for r in sorted(instructions,key=lambda x:x.get(name,0),reverse=True)[:8]:
                if r.get(name,0):top.append({k:r[k] for k in ['kernel','N','file','line','cuda','address','sass','operation']}|{'stall':name,'samples':r[name],'percent_within_this_stall':100*r[name]/total})
write(dest/'operation_stalls.csv',ops,['kernel','N','stall','operation_at_sampled_pc','samples','percent_within_this_stall'])
write(dest/'export_audit.csv',export_audits,['kernel','N','exported_instruction_rows','unique_pcs','duplicate_source_mappings'])
write(dest/'hotspots.csv',top,['kernel','N','file','line','cuda','address','sass','operation','stall','samples','percent_within_this_stall'])
lines=['# CUDA/SASS correlation of sampled stalls','',
'Representative repetition 0 for every kernel and workload. All source exports resolve the captured source snapshots; no live source modification is needed. The aggregate report uses three independent captures per case.','',
'A sampled PC identifies the instruction associated with a waiting warp; inspect its data dependencies and neighbouring producers before assigning a cause. In particular, a short-scoreboard sample at an arithmetic instruction can arise from a preceding shuffle or shared-memory load. It does not mean that the arithmetic instruction itself accessed shared memory. Percentages below are within the named stall category, not kernel runtime. Sampling and attribution can vary between captures. Optimised/inlined source mappings are approximate; duplicate mappings to the same PC are counted once, with consistency checks and project source preferred over library headers. Counts are in export_audit.csv.','',
'| Kernel | N | Stall | % within category | CUDA line | SASS at sampled PC |','|---|---:|---|---:|---|---|']
for k in plan['kernels']:
    for n in [s[2] for s in plan['shapes']]:
        for name in ['stall_short_sb','stall_long_sb','stall_mio']:
            rows=[r for r in top if r['kernel']==k and r['N']==n and r['stall']==name]
            for r in rows[:2]:lines.append(f"| {k} | {n} | {name} | {r['percent_within_this_stall']:.1f} | {Path(r['file']).name}:{r['line']} | `{r['sass'].replace('|','/')}` |")
lines+=['','Full correlated source/SASS CSV exports, all eight top PCs per category, and operation-group distributions are in this directory. The main `plan.json` and `source_snapshot/` identify the exact code; `processing.json` records this processing script hash.','',
'```bash',f'python3 part_c/scripts/analyze_source.py {a.run.as_posix()}','```','']
(dest/'SOURCE_ANALYSIS.md').write_text('\n'.join(lines))
print(f'Source-correlated {len(plan["kernels"])*len(plan["shapes"])} representative captures')
