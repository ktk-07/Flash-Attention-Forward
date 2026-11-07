"""Generate workload-specific findings from metrics and correlated source exports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();out=a.run
with (out/'metric_summary.csv').open() as f:summary=list(csv.DictReader(f))
with (out/'all_metrics.csv').open() as f:raw=list(csv.DictReader(f))
with (out/'source_analysis/hotspots.csv').open() as f:hot=list(csv.DictReader(f))
rows={(x['kernel'],int(x['N'])):x for x in summary}
def val(k,key,n=4096):return float(rows[k,n][key])
def dram(k):return (val(k,'dram_read_bytes')+val(k,'dram_write_bytes'))/2**30
def local(k):return val(k,'local_l1_sector_bytes')/2**30
def extra(k,metric):
    values=[float(x['numeric_value']) for x in raw if x['kernel']==k and x['N']=='4096' and x['metric']==metric and x['numeric_value']]
    return statistics.median(values) if values else float('nan')
lines=['# Measured findings and next optimisation experiments','',
'These findings use B=4,H=8,N=4096,d=64 unless stated otherwise, and medians over three Nsight captures. See ANALYSIS.md for the complete small/large tables, metric definitions, ranges, and capture conditions. Warp-state percentages are fractions of sampled warp states, not fractions of wall time. The candidates below have not been implemented or established as speedups.','',
'## 1. The selected CUDA-core kernel is not simply reducing total DRAM traffic','',
f"The FA1-style baseline uses {val('fa1','registers_per_thread'):.0f} registers/thread, with {val('fa1','theoretical_occupancy_pct'):.2f}% theoretical and {val('fa1','achieved_occupancy_pct'):.2f}% achieved occupancy. It has only {val('fa1','eligible_warps_per_scheduler'):.2f} eligible warps/scheduler and {val('fa1','state_long_scoreboard_pct'):.2f}% long-scoreboard samples.",'',
f"The selected `skip` version achieves {val('skip','achieved_occupancy_pct'):.2f}% occupancy out of a {val('skip','theoretical_occupancy_pct'):.2f}% ceiling and {val('skip','eligible_warps_per_scheduler'):.2f} eligible warps/scheduler. This supports improved cooperative scheduling/latency hiding. However, measured total DRAM read+write traffic is {dram('skip'):.2f} GiB versus {dram('fa1'):.2f} GiB for the baseline under this cold-cache profiling protocol. We must not claim that this implementation's speedup is demonstrated to come from lower total DRAM traffic.",'',
f"`skip` generates {local('skip'):.2f} GiB of local L1 sector traffic versus {local('fa1'):.2f} GiB for the baseline. These are repeated local-memory transactions, not that much allocated memory, and not all reach DRAM. Dynamically indexed per-thread state/accumulator arrays are a concrete optimisation target; inspect the exported LDL/STL instructions before calling every access register spilling.",'',
'## 2. The busy SM statistic is not FP32 arithmetic utilisation','',
f"For `skip`, the LSU instruction pipeline reaches {extra('skip','sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active'):.2f}% of its active-cycle peak, while the FMA pipeline is at {extra('skip','sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active'):.2f}%. DRAM throughput is {val('skip','dram_util_pct'):.2f}% of the profiler's elapsed-cycle peak metric. A high aggregate SM-throughput percentage must not be interpreted as saturated FP32 arithmetic.",'',
f"The useful QK+PV contract is {val('skip','useful_attention_flops')/1e9:.2f} GFLOPs, excluding softmax. Measured scalar FP32 FADD/FMUL/FFMA work totals {val('skip','measured_scalar_fp32_flops')/1e9:.2f} GFLOPs versus {val('fa1','measured_scalar_fp32_flops')/1e9:.2f} GFLOPs in the baseline. This includes reductions/bookkeeping and masked/padded work; it is not all avoidable overhead. More executed FLOPs can artificially increase an arithmetic-intensity numerator without improving useful efficiency.",'',
'## 3. Why float4 did not automatically win','',
f"`vector` reduces registers/thread from {val('skip','registers_per_thread'):.0f} to {val('vector','registers_per_thread'):.0f}, but both allocate {val('skip','static_shared_bytes')/1024:.0f} KiB of static shared memory and have the same {val('skip','theoretical_occupancy_pct'):.2f}% theoretical occupancy ceiling. Achieved occupancy is {val('skip','achieved_occupancy_pct'):.2f}% versus {val('vector','achieved_occupancy_pct'):.2f}%; local sector traffic remains {local('vector'):.2f} GiB.",'',
f"MIO-throttle samples fall from {val('skip','state_mio_throttle_pct'):.2f}% to {val('vector','state_mio_throttle_pct'):.2f}%, but short-scoreboard samples rise from {val('skip','state_short_scoreboard_pct'):.2f}% to {val('vector','state_short_scoreboard_pct'):.2f}%. Eligible warps/scheduler change from {val('skip','eligible_warps_per_scheduler'):.2f} to {val('vector','eligible_warps_per_scheduler'):.2f}. The loader changed without removing the common thread-local traffic and reduction/dependency structure. This is a supported explanation for limited benefit, not proof that vector loads themselves caused every timing difference.",'',
'The source loader gives four adjacent features to each lane. At d=64, only 16 lanes per warp have a complete float4 to load for each row. Testing a flat cooperative loader across multiple rows is a possible experiment; it does not guarantee improvement. Keep the existing scalar and vector variants as controls.','',
'## 4. Higher occupancy alone does not select the fastest kernel','',
f"The warp-per-query kernel reaches {val('warp','achieved_occupancy_pct'):.2f}% occupancy, higher than `skip` at {val('skip','achieved_occupancy_pct'):.2f}%, but it has fewer eligible warps/scheduler ({val('warp','eligible_warps_per_scheduler'):.2f} versus {val('skip','eligible_warps_per_scheduler'):.2f}), {val('warp','state_long_scoreboard_pct'):.2f}% long-scoreboard samples, and {dram('warp'):.2f} GiB of DRAM traffic. The working data/reuse pattern and dependency chain matter alongside resident warp count.",'',
'## 5. WMMA versus PTX has a specific shared-memory/resource trade-off','',
f"WMMA uses {val('wmma','static_shared_bytes')/1024:.2f} KiB shared memory per block versus PTX's {val('ptx','static_shared_bytes')/1024:.2f} KiB. Their theoretical occupancy ceilings are {val('wmma','theoretical_occupancy_pct'):.2f}% and {val('ptx','theoretical_occupancy_pct'):.2f}%; achieved occupancy is {val('wmma','achieved_occupancy_pct'):.2f}% and {val('ptx','achieved_occupancy_pct'):.2f}%. PTX uses more registers/thread ({val('ptx','registers_per_thread'):.0f} versus {val('wmma','registers_per_thread'):.0f}) while avoiding some shared output/state round trips.",'',
f"WMMA has {val('wmma','state_short_scoreboard_pct'):.2f}% short-scoreboard samples versus PTX's {val('ptx','state_short_scoreboard_pct'):.2f}%, and {val('wmma','shared_load_wavefronts')/val('ptx','shared_load_wavefronts'):.2f} times as many shared-load wavefronts. Tensor-pipeline active-cycle utilisation is {val('wmma','tensor_active_pct'):.2f}% versus {val('ptx','tensor_active_pct'):.2f}%. Both still have substantial headroom; none of these individual ratios is a measured causal speedup.",'',
f"Source-level actual/ideal shared wavefront expansion is {val('wmma','source_shared_wavefront_expansion'):.3f}x for WMMA, {val('ptx','source_shared_wavefront_expansion'):.3f}x for PTX, and {val('warp','source_shared_wavefront_expansion'):.3f}x for warp-per-query. These have positive excessive wavefronts and motivate controlled layout experiments. Tiled, skip and vector have {val('skip','source_shared_excessive_wavefronts'):.0f} excessive wavefronts: there is no source-proven bank conflict in these CUDA-core layouts. Aggregate L1 conflict counters include arbitration and do not justify swizzling them. None of these ratios is a percentage of runtime.",'',

'## 6. Source locations support targeted experiments','',
'The table below gives the highest sampled PC per selected category from representative repetition 0. Optimised/inlined line attribution can be approximate. Duplicate source mappings to the same PC were deduplicated and checked; inspect surrounding instructions and dependencies in the full exports.','',
'| Kernel | Category | % within this category | CUDA location | SASS |',
'|---|---|---:|---|---|']
for k,stall in [('fa1','stall_long_sb'),('warp','stall_long_sb'),('skip','stall_short_sb'),('skip','stall_long_sb'),('vector','stall_long_sb'),('wmma','stall_short_sb'),('ptx','stall_long_sb')]:
    candidates=[x for x in hot if x['kernel']==k and x['N']=='4096' and x['stall']==stall]
    if candidates:
        x=max(candidates,key=lambda r:float(r['samples']));lines.append(f"| {k} | {stall} | {float(x['percent_within_this_stall']):.2f} | {Path(x['file']).name}:{x['line']} | `{x['sass'].replace('|','/')}` |")
lines+=['',
'In the CUDA-core kernels, exported source includes short-scoreboard samples at shuffle-result arithmetic and scale/reduction instructions, as well as local state traffic. In WMMA, the shared `out_acc` update is prominent. For PTX, long-scoreboard source samples concentrate at the K/V load-to-shared path: a store may be waiting on a preceding global load. This makes input-load overlap a candidate for PTX, not a blanket justification for double buffering every kernel.','',
'## 7. Divergence is measured, not assumed','',
f"At the large shape, branch uniformity is {val('fa1','branch_uniform_pct'):.2f}% in the baseline, {val('skip','branch_uniform_pct'):.2f}% in `skip`, and {val('vector','branch_uniform_pct'):.2f}% in `vector`. Average predicated-on threads/warp are {val('fa1','pred_on_threads_per_warp'):.2f}, {val('skip','pred_on_threads_per_warp'):.2f}, and {val('vector','pred_on_threads_per_warp'):.2f} respectively. Thus even fully uniform branches do not imply all lanes perform useful work. The evidence does not justify calling branch divergence the dominant large-shape bottleneck.",'',
'## Recommended order of experiments','',
'1. **CUDA-core local state and reductions:** specialise/unroll small array indexing where useful, inspect LDL/STL and register allocation, and reduce repeated warp reductions or row-statistic work where mathematically valid. Monitor shared/register occupancy limits after every change.','2. **Tensor Core shared layouts:** try one controlled padding or swizzle variant, verifying both writes and reads plus fragment operand mapping. Recheck bank conflicts and ordinary timings.','3. **PTX input pipeline:** evaluate asynchronous copies/double buffering only with a concrete load/compute schedule. Account for extra shared memory and its occupancy cost.','4. **Bounded launch/tile tuning:** vary warps/query/KV tile sizes under a predefined search. These historical profiles use fixed educational launch configurations; the separate bounded evaluation studies other configurations and must not inherit these counters.','5. **Alternative vector loader:** test cooperative loading across rows rather than silently replacing the existing baseline.','',
'Each experiment needs numerical validation, sanitizer checks where memory/synchronisation changes, ordinary repeated CUDA-event timing, and a fresh profiling directory. The current workflow analyses kernels; it does not implement these optimisations. No new speedup is claimed here.','',
'## Reproduce these findings','',
'```bash',f'python3 part_c/scripts/interpret_profiles.py {a.run.as_posix()}','```','',
'This report is generated from `metric_summary.csv`, `all_metrics.csv`, and the deduplicated `source_analysis/hotspots.csv`. These observations are limited to the two profiled shapes on the recorded RTX 3070 Ti Laptop GPU, with unmanaged clocks, cache flushing, instrumentation/replay, and the sampling limitations documented in ANALYSIS.md. Ordinary benchmark timing remains separate.','']
(out/'FINDINGS.md').write_text('\n'.join(lines))
snapshot=out/'interpretation';snapshot.mkdir(exist_ok=True)
script=Path(__file__).resolve();shutil.copy2(script,snapshot/'interpret_profiles.py')
inputs=[out/name for name in ['metric_summary.csv','all_metrics.csv','source_analysis/hotspots.csv']]
(snapshot/'manifest.json').write_text(json.dumps({'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'input_sha256':{str(x.relative_to(out)):hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs}},indent=2)+'\n')
print(out/'FINDINGS.md')
