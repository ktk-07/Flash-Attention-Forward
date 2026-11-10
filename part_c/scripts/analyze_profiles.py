"""Regenerate all profiling tables/figures/findings from saved NCU CSVs."""
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import statistics

os.environ.setdefault('MPLCONFIGDIR','/tmp/part-c-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();out=a.run
plan=json.loads((out/'plan.json').read_text())
K=plan['kernels']
METRICS={
 'duration_ns':'gpu__time_duration.sum',
 'achieved_occupancy_pct':'sm__warps_active.avg.pct_of_peak_sustained_active',
 'eligible_warps_per_scheduler':'smsp__warps_eligible.avg.per_cycle_active',
 'issue_active_pct':'smsp__issue_active.avg.pct_of_peak_sustained_active',
 'registers_per_thread':'launch__registers_per_thread',
 'static_shared_bytes':'launch__shared_mem_per_block_static',
 'dynamic_shared_bytes':'launch__shared_mem_per_block_dynamic',
 'occupancy_limit_blocks':'launch__occupancy_limit_blocks',
 'occupancy_limit_registers':'launch__occupancy_limit_registers',
 'occupancy_limit_shared':'launch__occupancy_limit_shared_mem',
 'occupancy_limit_warps':'launch__occupancy_limit_warps',
 'dram_read_bytes':'dram__bytes_read.sum',
 'dram_write_bytes':'dram__bytes_write.sum',
 'dram_util_pct':'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
 'l1tex_util_pct':'l1tex__throughput.avg.pct_of_peak_sustained_active',
 'sm_throughput_pct':'sm__throughput.avg.pct_of_peak_sustained_elapsed',
 'tensor_active_pct':'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
 'source_shared_wavefronts':'memory_l1_wavefronts_shared',
 'source_shared_ideal_wavefronts':'memory_l1_wavefronts_shared_ideal',
 'source_shared_excessive_wavefronts':'derived__memory_l1_wavefronts_shared_excessive',
 'shared_load_conflicts':'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
 'shared_store_conflicts':'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
 'shared_load_wavefronts':'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
 'shared_store_wavefronts':'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
 'local_load_sectors':'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
 'local_store_sectors':'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
 'global_load_sectors':'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
 'global_store_sectors':'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum',
 'branch_uniform_pct':'smsp__sass_average_branch_targets_threads_uniform.pct',
 'divergent_branches':'smsp__sass_branch_targets_threads_divergent.sum',
 'active_threads_per_warp':'smsp__thread_inst_executed_per_inst_executed.ratio',
 'pred_on_threads_per_warp':'smsp__thread_inst_executed_pred_on_per_inst_executed.ratio',
 'sm_frequency_hz':'sm__cycles_elapsed.avg.per_second',
 'fadd_count':'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum',
 'fmul_count':'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum',
 'ffma_count':'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum',
}
def number(s):
    try:
        v=float(s.replace(',',''));return v if math.isfinite(v) else None
    except (ValueError,AttributeError):return None
def write(path,rows,fields=None):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
samples=[];long=[];rules=[];missing=[]
for case in sorted((out/'cases').iterdir()):
    meta=json.loads((case/'complete.json').read_text())
    identity={k:meta[k] for k in ['kernel','B','H','N','d','rep']}
    with (case/'raw.csv').open() as f:raw=list(csv.reader(f))
    if len(raw)!=3 or raw[0][0]!='ID':raise ValueError(f'Expected exactly one profiled kernel: {case}')
    names,units,values=raw
    data=dict(zip(names,values));unitmap=dict(zip(names,units))
    expected_y=(meta['N']+(4 if meta['kernel']=='warp' else 32)-1)//(4 if meta['kernel']=='warp' else 32)
    if meta['kernel'] in ['fa1','fa1_wmma'] and plan.get('fa1_grid')=='one_CTA_per_batch_head':expected_y=1
    actual_grid=tuple(int(v) for v in data['Grid Size'].strip('()').split(','))
    if actual_grid!=(meta['B']*meta['H'],expected_y,1):raise ValueError(f'Wrong profiled grid: {case}')
    with (case/'validation/correctness.csv').open() as f:checks=list(csv.DictReader(f))
    if len(checks)!=1 or checks[0]['status']!='PASS':raise ValueError(f'Unvalidated profile: {case}')
    row=dict(identity)
    for alias,name in METRICS.items():
        row[alias]=number(data.get(name))
        if row[alias] is None:missing.append({**identity,'metric':name})
    states={}
    prefix='smsp__pcsamp_warps_issue_stalled_'
    for name,value in data.items():
        if name.startswith(prefix) and not name.endswith('_not_issued'):
            states[name[len(prefix):]]=number(value)
    denominator=sum(v for v in states.values() if v is not None)
    row['warp_state_samples']=denominator
    for state in ['mio_throttle','short_scoreboard','long_scoreboard','barrier','math_pipe_throttle','wait','not_selected','selected','lg_throttle','branch_resolving']:
        row['state_'+state+'_pct']=100*states[state]/denominator if denominator and states.get(state) is not None else None
    with (case/'details.csv').open() as f:details=list(csv.DictReader(f))
    theoretical=[r for r in details if r['Section Name']=='Occupancy' and r['Metric Name']=='Theoretical Occupancy']
    row['theoretical_occupancy_pct']=number(theoretical[0]['Metric Value']) if theoretical else None
    for rule in details:
        if rule.get('Rule Description'):rules.append({**identity,'rule':rule['Rule Name'],'type':rule['Rule Type'],'description':rule['Rule Description']})
    dram=(row['dram_read_bytes'] or 0)+(row['dram_write_bytes'] or 0)
    row['useful_attention_flops']=4*meta['B']*meta['H']*(meta['N']*(meta['N']+1)//2)*meta['d']
    row['useful_flops_per_dram_byte']=row['useful_attention_flops']/dram if dram and row['dram_read_bytes'] is not None and row['dram_write_bytes'] is not None else None
    row['local_l1_sector_bytes']=32*(row['local_load_sectors']+row['local_store_sectors']) if row['local_load_sectors'] is not None and row['local_store_sectors'] is not None else None
    row['measured_scalar_fp32_flops']=2*row['ffma_count']+row['fadd_count']+row['fmul_count'] if all(row[x] is not None for x in ['ffma_count','fadd_count','fmul_count']) else None
    actual,ideal,excess=(row[x] for x in ['source_shared_wavefronts','source_shared_ideal_wavefronts','source_shared_excessive_wavefronts'])
    if None not in (actual,ideal,excess) and abs(actual-ideal-excess)>0.5:raise ValueError('Shared wavefront identity failed')
    row['source_shared_wavefront_expansion']=actual/ideal if ideal else None
    samples.append(row)
    for name,value in data.items():
        if name not in names[:11]:long.append({**identity,'metric':name,'unit':unitmap[name],'raw_value':value,'numeric_value':number(value)})
expected=len(K)*len(plan['shapes'])*plan['repetitions']
if len(samples)!=expected:raise ValueError(f'Expected {expected} profiles, got {len(samples)}')
write(out/'metric_samples.csv',samples);write(out/'all_metrics.csv',long)
write(out/'rules.csv',rules,['kernel','B','H','N','d','rep','rule','type','description'])
write(out/'unavailable_metrics.csv',missing,['kernel','B','H','N','d','rep','metric'])
groups=defaultdict(list)
for r in samples:groups[(r['kernel'],r['N'])].append(r)
summary=[]
for (kernel,n),group in groups.items():
    row={'kernel':kernel,'N':n,'B':4,'H':8,'d':64,'profiles':len(group)}
    for col in samples[0]:
        if col in ['kernel','B','H','N','d','rep']:continue
        vals=[r[col] for r in group if r[col] is not None]
        row[col]=statistics.median(vals) if vals else None
        row[col+'_min']=min(vals) if vals else None;row[col+'_max']=max(vals) if vals else None
    summary.append(row)
write(out/'metric_summary.csv',summary)
lookup={(r['kernel'],r['N']):r for r in summary}
figdir=out/'figures';figdir.mkdir(exist_ok=True)
def save(fig,name):
    fig.savefig(figdir/(name+'.png'),dpi=180);fig.savefig(figdir/(name+'.pdf'));plt.close(fig)
def value(kernel,n,metric):
    x=lookup[kernel,n][metric];return float('nan') if x is None else x
for metric,title in [('achieved_occupancy_pct','Achieved occupancy (%)'),('eligible_warps_per_scheduler','Eligible warps per scheduler'),('pred_on_threads_per_warp','Average predicated-on threads per warp'),('useful_flops_per_dram_byte','Useful attention FLOPs / measured DRAM byte')]:
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,n in zip(axes,[256,4096]):
        vals=[value(k,n,metric) for k in K];ax.bar(K,vals)
        ax.set(title=f'N={n}',ylabel=title);ax.tick_params(axis='x',rotation=40)
        if metric=='achieved_occupancy_pct':ax.plot(K,[value(k,n,'theoretical_occupancy_pct') for k in K],color='black',marker='_',linestyle='none',label='Theoretical ceiling');ax.legend()
        if metric=='useful_flops_per_dram_byte':ax.set_yscale('log')
    fig.suptitle('Nsight Compute: B=4 H=8 d=64; medians of '+str(plan['repetitions'])+' independent profiles; caches flushed')
    save(fig,metric)
fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
state_cols=['mio_throttle','short_scoreboard','long_scoreboard','barrier','math_pipe_throttle','wait','not_selected','selected','lg_throttle','branch_resolving']
for ax,n in zip(axes,[256,4096]):
    bottom=[0.]*len(K)
    for state in state_cols:
        vals=[value(k,n,'state_'+state+'_pct') for k in K];ax.bar(K,vals,bottom=bottom,label=state);bottom=[b+(0 if math.isnan(v) else v) for b,v in zip(bottom,vals)]
    residual=[max(0,100-b) for b in bottom];ax.bar(K,residual,bottom=bottom,label='other states')
    ax.set(title=f'N={n}',ylabel='% of sampled warp states (includes issuing warps)',ylim=(0,100));ax.tick_params(axis='x',rotation=40)
axes[1].legend(fontsize=7,bbox_to_anchor=(1,1));fig.suptitle('Warp-state sample distribution — not percentages of total kernel runtime')
save(fig,'warp_states')
fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
for ax,n in zip(axes,[256,4096]):
    ax.plot(K,[(value(k,n,'dram_read_bytes')+value(k,n,'dram_write_bytes'))/2**20 for k in K],marker='o',label='DRAM read + write MiB')
    ax.plot(K,[value(k,n,'local_l1_sector_bytes')/2**20 for k in K],marker='s',label='Local L1 sector bytes / MiB')
    ax.set(title=f'N={n}',ylabel='Traffic (MiB), different memory levels',yscale='symlog');ax.tick_params(axis='x',rotation=40);ax.legend(fontsize=8)
fig.suptitle('Spill/local traffic is not allocated workspace or necessarily DRAM traffic')
save(fig,'memory_traffic')

def fmt(v):return 'unavailable' if v is None else f'{v:.3g}'
lines=['# Reproducible Nsight Compute analysis','',f"{len(samples)} captured kernels: {len(K)} Part C variants, B=4,H=8,d=64,N=256/4096, {plan['repetitions']} independent profiles each. No Part A integration. All captures passed sampled CPU-FP64 validation before profiling. Full small-case correctness validation is also saved.",'',
'## Measurement contract','',
'Each process performs one validated kernel launch, five warmups, then one target launch. Nsight uses `--launch-skip 6 --launch-count 1`. The exporter verifies the captured grid and the validation record. The `full` section set collects scheduler, occupancy, memory, source/branch, instruction, and roofline information. Explicit FP32 FADD/FMUL/FFMA counters are added. Raw `.ncu-rep`, raw-metric CSV, detail/rule CSV, all commands, environment, snapshots, and SHA256 hashes are retained.','',
'`--cache-control all` flushes caches between replay passes. `--clock-control none` leaves GPU clocks unmanaged; measured frequencies and repeat ranges are saved. Profiler durations are not substituted for the normal warm-cache CUDA-event benchmark. Profile repeats measure counter/sampling variability; they do not make individual counter values bitwise deterministic.','',
'Stall percentages below use each state\'s samples divided by the sum of the all-samples warp-state counts, including selected/issuing and not-selected warps. They are not percentages of kernel wall time. Missing metrics remain unavailable, never fabricated as zero.','',
'## Occupancy, scheduling, and memory','',
'| N | Kernel | Theoretical / achieved occupancy % | Eligible warps / scheduler | Registers/thread | Static shared KiB | DRAM MiB | Local L1 sector MiB | Uniform branches % | Predicated-on threads/warp |',
'|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for n in [256,4096]:
    for k in K:
        r=lookup[k,n];dram=(r['dram_read_bytes']+r['dram_write_bytes'])/2**20 if r['dram_read_bytes'] is not None and r['dram_write_bytes'] is not None else None
        lines.append(f"| {n} | {k} | {fmt(r['theoretical_occupancy_pct'])} / {fmt(r['achieved_occupancy_pct'])} | {fmt(r['eligible_warps_per_scheduler'])} | {fmt(r['registers_per_thread'])} | {fmt(r['static_shared_bytes']/1024) if r['static_shared_bytes'] is not None else 'unavailable'} | {fmt(dram)} | {fmt(r['local_l1_sector_bytes']/2**20) if r['local_l1_sector_bytes'] is not None else 'unavailable'} | {fmt(r['branch_uniform_pct'])} | {fmt(r['pred_on_threads_per_warp'])} |")
lines+=['','## Warp-state evidence and bank conflicts','',
'| N | Kernel | MIO throttle % | Short scoreboard % | Long scoreboard % | Barrier % | Math-pipe throttle % | Shared load/store conflicts |',
'|---:|---|---:|---:|---:|---:|---:|---:|']
for n in [256,4096]:
    for k in K:
        r=lookup[k,n];lines.append(f"| {n} | {k} | {fmt(r['state_mio_throttle_pct'])} | {fmt(r['state_short_scoreboard_pct'])} | {fmt(r['state_long_scoreboard_pct'])} | {fmt(r['state_barrier_pct'])} | {fmt(r['state_math_pipe_throttle_pct'])} | {fmt(r['shared_load_conflicts'])} / {fmt(r['shared_store_conflicts'])} |")
lines+=['','## Source shared-memory wavefronts','','Aggregate L1 bank-conflict counters can include arbitration. Diagnose layout conflicts using source-level actual minus ideal wavefronts; zero excess is no source-proven bank conflict. No-shared-memory kernels have no expansion ratio.','','| N | Kernel | Actual | Ideal | Excessive | Actual / ideal |','|---|---|---|---|---|---|']
for n in [256,4096]:
    for k in K:
        r=lookup[k,n];lines.append(f"| {n} | {k} | {fmt(r['source_shared_wavefronts'])} | {fmt(r['source_shared_ideal_wavefronts'])} | {fmt(r['source_shared_excessive_wavefronts'])} | {fmt(r['source_shared_wavefront_expansion'])} |")
lines+=['','NVIDIA explains arbitration in [shared-memory bank conflict metrics](https://forums.developer.nvidia.com/t/shared-memory-bank-conflicts-and-nsight-metric/115731/15).','','## Arithmetic intensity and operation counts','',
'Useful attention work is `4*B*H*d*N*(N+1)/2` FLOPs (causal QK plus PV, excluding softmax). Dividing it by measured DRAM read+write bytes gives an algorithm-useful DRAM intensity. This includes the effects of caching and spills in measured traffic and does not count masked/padded work. It is not the executed Tensor Core FLOP count or a substitute for the appropriate hardware roofline in the raw report. Measured scalar FP32 operations are `2*FFMA + FADD + FMUL`; for Tensor Core kernels these exclude matrix operations. Compare intensity at the same memory level.','',
'## Reading the results and choosing experiments','',
'1. **Global-memory reuse:** compare `fa1` with `tiled/skip` at matching shapes. Check plan.json: the old FA1 is query-parallel scalar; the rewritten FA1 is K/V-outer with shared Q/K/V/S/O and global m/l. Shared staging, loop order and register reuse change the access pattern; use DRAM and L1/local traffic plus issue metrics to determine which layer improved. A lower DRAM byte count by itself does not prove that bandwidth was limiting.','',
'2. **MIO and shared memory:** MIO throttle means pressure on the MIO instruction path, which also serves special math and dynamic branching. Short scoreboard signals dependencies that can involve shared memory but is not proof of bank conflicts. Use source-level excessive wavefronts to diagnose layout conflicts; aggregate conflict counters alone are insufficient. Reducing redundant shared-memory operations/register reuse is a separate hypothesis from eliminating bank conflicts.','',
'3. **Latency hiding:** achieved occupancy is the measured resident-warp fraction; theoretical occupancy is the launch/resource ceiling. Neither equals utilisation or eligible warps. Inspect the register/shared/block/warp limits and eligible warps together. Higher occupancy alone is not a performance objective.','',
'4. **Local traffic:** local load/store sectors indicate thread-local memory accesses, including spilled or dynamically indexed arrays. They do not all reach DRAM. Correlate with compiler stack/local resources and SASS/source lines before attributing every local transaction to register spilling. Scalar/register-array indexing is a concrete candidate for compile-time unrolling/specialisation experiments.','',
'5. **Divergence and predication:** uniform-branch percentage only describes branch behaviour. Predicated-off lanes can waste work even with 100% uniform branches. Compare active and predicated-on threads per warp, the causal/tail guards, and grid size. A causal mask does not automatically imply that divergence is the bottleneck.','',
'6. **Vectorisation:** compare `skip/tiled` against `vector` using global load instruction/sector counts, local traffic, resources, and issue rates. Fewer load instructions or 128-bit loads need not shorten the critical path. The current fixed launch mapping may leave loader lanes idle; an alternative cooperative loader is an experiment, not an established improvement.','',
'7. **WMMA versus PTX:** compare shared traffic/wavefronts, scoreboard/MIO states, local traffic, and tensor utilisation. PTX keeps score/output state in registers and avoids some WMMA shared-memory round trips, but occupancy/register trade-offs must be measured. FP16-input Tensor Core results must not be presented as a precision-isolated speedup over FP32 CUDA-core kernels.','',
'8. **Double buffering:** consider only after identifying a useful load/compute overlap opportunity. Additional K/V buffers consume shared memory and may lower occupancy. Swizzling and double buffering are not implemented by this profiling workflow.','',
'These are observation-driven experiment candidates, not proven causes or promises of speedup. Establish causality with one-change variants, revalidation, ordinary CUDA-event timing, and repeat profiling. Existing kernel progression versions can change several features simultaneously.','',
'## Reproduce','',
'```bash',f"bash part_c/profile.sh --output part_c/results/my_profile --build /tmp/part_c-profile-build --repeats {plan['repetitions']}",
'# Regenerate all tables, plots and this report without accessing the GPU:',f'python3 part_c/scripts/analyze_profiles.py {out.as_posix()}', '```','',
'Use a fresh output directory. `--resume` requires matching source hashes and the same plan. Build/runtime/tool versions and GPU state are in `logs/`; exact argv, working directory, UTC times and exit codes are in `commands.jsonl`. Source snapshots are under `source_snapshot/`. The initial capability probe is separate and is not used as a representative workload.','',
'## Artifacts','',
'- `cases/*/profile.ncu-rep`: open in Nsight Compute GUI for source/SASS correlation and rooflines.','- `cases/*/raw.csv` and `details.csv`: original exports.','- `all_metrics.csv`: lossless long-form metric values and units.','- `metric_samples.csv`: derived per-profile values and sampling denominators.','- `metric_summary.csv`: medians and min/max across independent profiles.','- `rules.csv`: Nsight advisory text, preserved as suggestions rather than proof.','- `unavailable_metrics.csv`: explicit metric availability.','- `figures/`: occupancy, scheduler eligibility, lane activity, warp states, memory traffic, and useful DRAM intensity (PNG/PDF).','',
'Definitions follow NVIDIA\'s [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) and [Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/).','']
(out/'ANALYSIS.md').write_text('\n'.join(lines))
print(f'Analysed {len(samples)} validated profiles; missing metric entries: {len(missing)}')

# Retain the actual processor for future deterministic offline audits.
import shutil,hashlib
processing=out/'processing';processing.mkdir(exist_ok=True)
script=Path(__file__).resolve()
if script!=processing/'analyze_profiles.py':shutil.copy2(script,processing/'analyze_profiles.py')
(processing/'manifest.json').write_text(json.dumps({'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest()},indent=2)+'\n')
