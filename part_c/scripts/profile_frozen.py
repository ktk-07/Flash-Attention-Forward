#!/usr/bin/env python3
"""Profile both fixed FA1 baselines and frozen FA2 family winners, separately from timing."""
import argparse,csv,hashlib,json,math,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('--build',type=Path,default=Path('/tmp/part_c-evaluation-build'));p.add_argument('--analyze-only',action='store_true');p.add_argument('--smoke',action='store_true',help='profile the small study shapes; test evidence, not full-grid profiling');a=p.parse_args()
out=a.study/'profiling';selection=json.loads((a.study/'frozen_selection.json').read_text());study_plan=json.loads((a.study/'plan.json').read_text()) if not a.analyze_only else {};binary=a.build.resolve()/'part_c_evaluation'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def family_targets():
    return [{'role':family,'kernel':selection['families'][family]['64'],'config':selection['variants'][selection['families'][family]['64']]['64']['config']} for family in ['cuda_core','tensor_core']]
if not a.analyze_only:
    if out.exists():raise SystemExit('Profiling directory exists; use --analyze-only or a fresh study')
    out.mkdir();(out/'cases').mkdir();(out/'profile_frozen.py').write_bytes(Path(__file__).read_bytes())
    targets=[{'role':k,'kernel':k,'config':0} for k in study_plan.get('fa1_baselines',[])]+family_targets()
    lengths=study_plan['main_N'] if a.smoke else [256,4096]
    if any(n not in study_plan['main_N'] for n in lengths):raise ValueError('Profile lengths must exist in the measured study; use --smoke for a smoke study')
    plan={'targets':targets,'B':study_plan.get('B',4),'H':study_plan.get('H',8),'selection_sha256':sha(a.study/'frozen_selection.json'),'binary_sha256':sha(binary),'study_binary_match':sha(binary)==json.loads((a.study/'manifest.json').read_text())['binary_sha256'],'launch_skip':7,'explanation':'one warm-module forward, one validation forward, five warmups, then first timed forward; all reset work stays in forward','repeats':3,'N':lengths,'d':64,'cache_control':'all','clock_control':'none','smoke':a.smoke,'fa1_total_shared_bytes':bool(study_plan.get('fa1_baselines'))}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');assert plan['study_binary_match']
    def run(cmd,stdout):
        with stdout.open('w') as f:r=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT)
        with (out/'commands.jsonl').open('a') as f:f.write(json.dumps({'argv':list(map(str,cmd)),'exit_code':r.returncode})+'\n')
        if r.returncode:raise RuntimeError(f'Failed; see {stdout}')
    run(['ncu','--version'],out/'ncu_version.txt');run(['nvidia-smi','-q'],out/'gpu.txt')
    for target in targets:
        k,config=target['kernel'],target['config']
        for n in lengths:
            for rep in range(plan['repeats']):
                case=out/'cases'/f'{k}_c{config}_n{n}_r{rep}';case.mkdir()
                run(['ncu','--set','full','--launch-skip','7','--launch-count','1','--cache-control','all','--clock-control','none','--export',case/'profile',binary,k,config,plan['B'],plan['H'],n,64,'random','benchmark',case/'validation.csv'],case/'capture.txt')
                run(['ncu','--import',case/'profile.ncu-rep','--page','raw','--csv','--print-units','base'],case/'raw.csv')
                (case/'identity.json').write_text(json.dumps({'kernel':k,'config':config,'N':n,'d':64,'rep':rep})+'\n')
plan=json.loads((out/'plan.json').read_text());assert sha(a.study/'frozen_selection.json')==plan['selection_sha256']
targets=plan.get('targets',family_targets());lengths=plan.get('N',[256,4096]);repeats=plan.get('repeats',3)
metrics={'occupancy_pct':'sm__warps_active.avg.pct_of_peak_sustained_active','eligible_warps_per_scheduler':'smsp__warps_eligible.avg.per_cycle_active','registers_per_thread':'launch__registers_per_thread','shared_bytes':'launch__shared_mem_per_block_static','dram_read_bytes':'dram__bytes_read.sum','dram_write_bytes':'dram__bytes_write.sum','local_load_sectors':'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','local_store_sectors':'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum','shared_actual':'memory_l1_wavefronts_shared','shared_ideal':'memory_l1_wavefronts_shared_ideal','shared_excessive':'derived__memory_l1_wavefronts_shared_excessive','tensor_active_pct':'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active'}
if plan.get('fa1_total_shared_bytes'):metrics.update(shared_static_bytes='launch__shared_mem_per_block_static',shared_dynamic_bytes='launch__shared_mem_per_block_dynamic')
def numeric(value):
    try:
        result=float(str(value).replace(',',''));return result if math.isfinite(result) else float('nan')
    except (ValueError,TypeError):return float('nan')
def median(group,column):
    values=[r[column] for r in group];return statistics.median(values) if values and all(math.isfinite(v) for v in values) else float('nan')
def fmt(value,digits=2):return f'{value:.{digits}f}' if math.isfinite(value) else 'n/a'
records=[]
for case in sorted((out/'cases').iterdir()):
    with (case/'validation.csv').open() as f:checks=list(csv.DictReader(f))
    assert len(checks)==20 and all(r['status']=='PASS' for r in checks)
    with (case/'raw.csv').open() as f:raw=list(csv.reader(f))
    assert len(raw)==3 and raw[0][0]=='ID'
    data=dict(zip(raw[0],raw[2]));r=json.loads((case/'identity.json').read_text());first=checks[0]
    assert all(str(r[k])==first[k] for k in ['kernel','config','N','d'])
    b,h=int(first['B']),int(first['H']);qtiles=1 if r['kernel'] in ['fa1','fa1_wmma'] else (r['N']+int(first['block_q'])-1)//int(first['block_q'])
    assert tuple(int(v) for v in data['Grid Size'].strip('()').split(','))==(b*h,qtiles,1)
    assert tuple(int(v) for v in data['Block Size'].strip('()').split(','))==(32*int(first['warps']),1,1)
    for alias,key in metrics.items():r[alias]=numeric(data.get(key,''))
    if plan.get('fa1_total_shared_bytes'):r['shared_bytes']=r['shared_static_bytes']+r['shared_dynamic_bytes']
    if all(math.isfinite(r[k]) for k in ['shared_actual','shared_ideal','shared_excessive']):assert r['shared_actual']-r['shared_ideal']==r['shared_excessive']
    prefix='smsp__pcsamp_warps_issue_stalled_';states={k:numeric(v) for k,v in data.items() if k.startswith(prefix) and not k.endswith('_not_issued')};denom=sum(states.values())
    for state in ['long_scoreboard','short_scoreboard','mio_throttle']:r[state+'_pct']=100*states.get(prefix+state,float('nan'))/denom if math.isfinite(denom) and denom>0 else float('nan')
    records.append(r)
expected={(t['kernel'],t['config'],n,plan.get('d',64),rep) for t in targets for n in lengths for rep in range(repeats)}
assert len(records)==len(expected) and {(r['kernel'],r['config'],r['N'],r['d'],r['rep']) for r in records}==expected
with (out/'metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows({k:('' if isinstance(v,float) and not math.isfinite(v) else v) for k,v in r.items()} for r in records)
lines=['# FA1 baselines and frozen FA2 profiling','',f"B={plan.get('B',4)},H={plan.get('H',8)},d=64; {repeats} captures at N={lengths}. Smoke={plan.get('smoke',False)}. Profiler times are excluded from benchmark summaries.",'','Shared bytes include static+dynamic storage for new FA1-aware plans. Missing counters/zero sample denominators remain unavailable.','', '| Kernel | Config | N | Occupancy % | Eligible warps | Shared KiB | Excess wavefronts | Long scoreboard % | Short scoreboard % |','|---|---|---|---|---|---|---|---|---|']
for key in sorted({(r['kernel'],r['config'],r['N']) for r in records}):
    group=[r for r in records if (r['kernel'],r['config'],r['N'])==key]
    lines.append(f'| {key[0]} | {key[1]} | {key[2]} | {fmt(median(group,"occupancy_pct"))} | {fmt(median(group,"eligible_warps_per_scheduler"))} | {fmt(median(group,"shared_bytes")/1024)} | {fmt(median(group,"shared_excessive"),0)} | {fmt(median(group,"long_scoreboard_pct"))} | {fmt(median(group,"short_scoreboard_pct"))} |')
lines+=['','Use these measured resource/traffic/scheduler values to test bottleneck hypotheses; sampled warp states are not runtime fractions. Explicit allocation is not traffic. Only d=64 is profiled. The two FA1 baselines are this project\'s implementations, not official FA1 parity measurements.','', 'Regenerate offline: `python3 part_c/scripts/profile_frozen.py '+str(a.study)+' --analyze-only`.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
(out/'audit.json').write_text(json.dumps({'validated_captures':len(records),'expected_captures':len(expected),'targets':targets,'all_pass':True,'raw_report_sha256':{str(x.relative_to(out)):sha(x) for x in out.glob('cases/*/profile.ncu-rep')}},indent=2)+'\n')
processing=out/'processing';processing.mkdir(exist_ok=True);(processing/'profile_frozen.py').write_bytes(Path(__file__).read_bytes());(processing/'manifest.json').write_text(json.dumps({'script_sha256':sha(Path(__file__)), 'raw_csv_sha256':{str(x.relative_to(out)):sha(x) for x in sorted(out.glob('cases/*/raw.csv'))}},indent=2)+'\n')
print(out/'REPORT.md')
