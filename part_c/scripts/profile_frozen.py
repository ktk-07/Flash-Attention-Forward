#!/usr/bin/env python3
"""Profile frozen d=64 family winners separately from latency measurements."""
import argparse,csv,hashlib,json,statistics,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('--build',type=Path,default=Path('/tmp/part_c-evaluation-build'));p.add_argument('--analyze-only',action='store_true');a=p.parse_args()
out=a.study/'profiling';selection=json.loads((a.study/'frozen_selection.json').read_text());binary=a.build.resolve()/'part_c_evaluation'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
if not a.analyze_only:
    if out.exists():raise SystemExit('Profiling directory exists; use --analyze-only or a fresh study')
    out.mkdir();(out/'cases').mkdir();(out/'profile_frozen.py').write_bytes(Path(__file__).read_bytes())
    (out/'plan.json').write_text(json.dumps({'selection_sha256':sha(a.study/'frozen_selection.json'),'binary_sha256':sha(binary),'study_binary_match':sha(binary)==json.loads((a.study/'manifest.json').read_text())['binary_sha256'],'launch_skip':7,'explanation':'one warm-module launch, one validation launch, five warmups, then first timed launch; profiler timings excluded from main study','repeats':3,'N':[256,4096],'d':64,'cache_control':'all','clock_control':'none'},indent=2)+'\n')
    assert json.loads((out/'plan.json').read_text())['study_binary_match']
    def run(cmd,stdout):
        with stdout.open('w') as f:r=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT)
        with (out/'commands.jsonl').open('a') as f:f.write(json.dumps({'argv':list(map(str,cmd)),'exit_code':r.returncode})+'\n')
        if r.returncode:raise RuntimeError(f'Failed; see {stdout}')
    run(['ncu','--version'],out/'ncu_version.txt');run(['nvidia-smi','-q'],out/'gpu.txt')
    for family in ['cuda_core','tensor_core']:
        k=selection['families'][family]['64'];config=selection['variants'][k]['64']['config']
        for n in [256,4096]:
            for rep in range(3):
                case=out/'cases'/f'{k}_c{config}_n{n}_r{rep}';case.mkdir()
                run(['ncu','--set','full','--launch-skip','7','--launch-count','1','--cache-control','all','--clock-control','none','--export',case/'profile',binary,k,config,4,8,n,64,'random','benchmark',case/'validation.csv'],case/'capture.txt')
                run(['ncu','--import',case/'profile.ncu-rep','--page','raw','--csv','--print-units','base'],case/'raw.csv')
                (case/'identity.json').write_text(json.dumps({'kernel':k,'config':config,'N':n,'d':64,'rep':rep})+'\n')
assert sha(a.study/'frozen_selection.json')==json.loads((out/'plan.json').read_text())['selection_sha256']
metrics={'occupancy_pct':'sm__warps_active.avg.pct_of_peak_sustained_active','eligible_warps_per_scheduler':'smsp__warps_eligible.avg.per_cycle_active','registers_per_thread':'launch__registers_per_thread','shared_bytes':'launch__shared_mem_per_block_static','dram_read_bytes':'dram__bytes_read.sum','dram_write_bytes':'dram__bytes_write.sum','local_load_sectors':'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','local_store_sectors':'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum','shared_actual':'memory_l1_wavefronts_shared','shared_ideal':'memory_l1_wavefronts_shared_ideal','shared_excessive':'derived__memory_l1_wavefronts_shared_excessive','tensor_active_pct':'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active'}
records=[]
for case in sorted((out/'cases').iterdir()):
    with (case/'validation.csv').open() as f:checks=list(csv.DictReader(f))
    assert len(checks)==20 and all(r['status']=='PASS' for r in checks)
    with (case/'raw.csv').open() as f:raw=list(csv.reader(f))
    assert len(raw)==3 and raw[0][0]=='ID'
    data=dict(zip(raw[0],raw[2]));r=json.loads((case/'identity.json').read_text())
    assert tuple(int(v) for v in data['Grid Size'].strip('()').split(','))==(32,(r['N']+int(checks[0]['block_q'])-1)//int(checks[0]['block_q']),1)
    assert tuple(int(v) for v in data['Block Size'].strip('()').split(','))==(32*int(checks[0]['warps']),1,1)
    for alias,key in metrics.items():r[alias]=float(data[key].replace(',',''))
    assert r['shared_actual']-r['shared_ideal']==r['shared_excessive']
    prefix='smsp__pcsamp_warps_issue_stalled_';states={k:float(v.replace(',','')) for k,v in data.items() if k.startswith(prefix) and not k.endswith('_not_issued')};denom=sum(states.values())
    for state in ['long_scoreboard','short_scoreboard','mio_throttle']:r[state+'_pct']=100*states[prefix+state]/denom if denom else 0
    records.append(r)
assert len(records)==12 and len({(r['kernel'],r['config'],r['N'],r['rep']) for r in records})==12
with (out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
lines=['# Frozen winner profiling','','B=4,H=8,d=64; three independent full Nsight captures at each N, medians below. These are fresh captures of the selected configurations. Instrumented timings are excluded from main results.','', '| Kernel | Config | N | Occupancy % | Eligible warps/scheduler | Shared actual/ideal | Excessive wavefronts | Long scoreboard % | Short scoreboard % |','|---|---|---|---|---|---|---|---|---|']
for key in sorted({(r['kernel'],r['config'],r['N']) for r in records}):
    group=[r for r in records if (r['kernel'],r['config'],r['N'])==key];m={col:statistics.median(r[col] for r in group) for col in metrics};long=statistics.median(r['long_scoreboard_pct'] for r in group);short=statistics.median(r['short_scoreboard_pct'] for r in group)
    lines.append(f'| {key[0]} | {key[1]} | {key[2]} | {m["occupancy_pct"]:.2f} | {m["eligible_warps_per_scheduler"]:.2f} | {m["shared_actual"]/m["shared_ideal"]:.3f} | {m["shared_excessive"]:.0f} | {long:.2f} | {short:.2f} |')
lines+=['','## Traffic and resources at N=4096','','| Kernel | Registers/thread | Shared KiB/block | DRAM read+write GiB | Local L1 sector traffic GiB | Tensor active % |','|---|---|---|---|---|---|']
for k in sorted({r['kernel'] for r in records}):
    group=[r for r in records if r['kernel']==k and r['N']==4096]
    m={col:statistics.median(r[col] for r in group) for col in metrics}
    lines.append(f'| {k} | {m["registers_per_thread"]:.0f} | {m["shared_bytes"]/1024:.2f} | {(m["dram_read_bytes"]+m["dram_write_bytes"])/2**30:.2f} | {32*(m["local_load_sectors"]+m["local_store_sectors"])/2**30:.2f} | {m["tensor_active_pct"]:.2f} |')
lines+=['','The CUDA-core winner still generates substantial local-memory transactions despite a smaller per-thread local allocation than its default configuration. Smaller allocation size does not imply fewer executed transactions. Its measured occupancy is near the API resource ceiling, yet local-memory work remains an optimisation opportunity. PTX has no measured local sector traffic here; its long-scoreboard samples and shared excessive wavefronts identify different remaining constraints. These observations do not establish how much a future optimisation would accelerate either kernel.','']
lines+=['','Warp-state percentages describe sampled states, not runtime fractions. API resource estimates are in the main study CSVs; occupancy above is measured. Positive source excessive wavefronts support studying shared layouts, but do not predict the speedup from a layout change. DRAM bytes and local sectors are traffic, separate from the explicit allocation ledger. Clock management is disabled and caches are flushed by the profiler. Only d=64 is profiled here; do not extrapolate counters to other dimensions.','', 'Regenerate offline: `python3 part_c/scripts/profile_frozen.py '+str(a.study)+' --analyze-only`.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
(out/'audit.json').write_text(json.dumps({'validated_captures':len(records),'all_pass':True,'raw_report_sha256':{str(x.relative_to(out)):sha(x) for x in out.glob('cases/*/profile.ncu-rep')}},indent=2)+'\n')
print(out/'REPORT.md')

processing=out/'processing';processing.mkdir(exist_ok=True)
(processing/'profile_frozen.py').write_bytes(Path(__file__).read_bytes())
(processing/'manifest.json').write_text(json.dumps({'script_sha256':sha(Path(__file__)), 'raw_csv_sha256':{str(x.relative_to(out)):sha(x) for x in sorted(out.glob('cases/*/raw.csv'))}},indent=2)+'\n')
