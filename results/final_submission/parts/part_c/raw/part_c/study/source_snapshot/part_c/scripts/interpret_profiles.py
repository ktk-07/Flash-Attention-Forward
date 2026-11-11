"""Generate evidence-led findings for every captured kernel, including both FA1 baselines."""
import argparse,csv,hashlib,json,math,shutil,statistics
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();out=a.run
read=lambda path:list(csv.DictReader(path.open()))
summary=read(out/'metric_summary.csv');plan=json.loads((out/'plan.json').read_text())
raw=read(out/'all_metrics.csv');hot_path=out/'source_analysis/hotspots.csv';hot=read(hot_path) if hot_path.exists() else []
def val(row,key):
    try:value=float(row.get(key,''));return value if math.isfinite(value) else float('nan')
    except (ValueError,TypeError):return float('nan')
def fmt(value,digits=2):return f'{value:.{digits}f}' if math.isfinite(value) else 'n/a'
lines=['# Measured profiling evidence','','Each row describes the captured configuration, not the fastest reporting-grid implementation. Counter values are medians across captures; unavailable values remain n/a. Sampled warp states are not wall-time fractions. Profiler-instrumented times are excluded from benchmark speedups.','','| Kernel | N | Reg/thread | Static/dynamic shared KiB | Theoretical/achieved occupancy % | Eligible warps | DRAM GiB | Local sector GiB | Tensor % | Excess shared wavefronts |','|---|---|---|---|---|---|---|---|---|---|']
for row in sorted(summary,key=lambda r:(int(r['N']),r['kernel'])):
    lines.append(f"| {row['kernel']} | {row['N']} | {fmt(val(row,'registers_per_thread'),0)} | {fmt(val(row,'static_shared_bytes')/1024)} / {fmt(val(row,'dynamic_shared_bytes')/1024)} | {fmt(val(row,'theoretical_occupancy_pct'))} / {fmt(val(row,'achieved_occupancy_pct'))} | {fmt(val(row,'eligible_warps_per_scheduler'))} | {fmt((val(row,'dram_read_bytes')+val(row,'dram_write_bytes'))/2**30)} | {fmt(val(row,'local_l1_sector_bytes')/2**30)} | {fmt(val(row,'tensor_active_pct'))} | {fmt(val(row,'source_shared_excessive_wavefronts'),0)} |")
lines+=['','## Interpretation by mechanism','','- Compare `fa1` with the FP32 FA2-style kernels and `fa1_ptx` with PTX FA2 separately. WMMA FA1, where present in an older or supplementary plan, is a separate implementation comparison. Fresh FA1 kernels use K/V-outer traversal, one CTA/head and global m/l plus normalised output updates. Legacy plans without the FA1 version flags describe the earlier query-parallel scalar baseline.','- CTA/warp organisation and shared/register capacity determine residency ceilings; measured occupancy and eligible warps describe different aspects of execution. A high occupancy value alone does not establish faster execution.','- Explicit allocations are capacity, DRAM/local/shared transactions are traffic, and local transactions do not all reach DRAM. Register allocation alone cannot establish executed spill traffic.','- Actual minus ideal source shared wavefronts can support a layout investigation. Aggregate conflict counters, MIO samples and short-scoreboard samples alone do not establish bank conflicts. Wider shared instructions can have multiple ideal wavefronts.','- WMMA/PTX and the two FA1 variants change more than one mechanism. Inspect source/SASS dependencies and use unprofiled matched-input timing before claiming a causal speedup.','- Float4 loading is a candidate, not an automatic improvement. A matched launch/configuration study isolates it more closely than broad progression plots.','','## Source correlation','','| Kernel | N | Stall | Share within this stall % | Source | SASS at sampled PC |','|---|---|---|---|---|---|']
for kernel in plan['kernels']:
    for n in sorted({int(r['N']) for r in summary}):
        for stall in ['stall_short_sb','stall_long_sb','stall_mio']:
            candidates=[r for r in hot if r['kernel']==kernel and int(r['N'])==n and r['stall']==stall]
            if candidates:
                row=max(candidates,key=lambda r:float(r['samples']))
                lines.append(f"| {kernel} | {n} | {stall} | {float(row['percent_within_this_stall']):.2f} | {Path(row['file']).name}:{row['line']} | `{row['sass'].replace('|','/')}` |")
if not hot:lines+=['','Source correlation is not supplied. Generate it with analyze_source.py and retained binary Nsight reports; no source hotspot is invented here.']
lines+=['','## Candidate experiments','','Use these data to choose controlled tests of local state placement, shared layout, transfer overlap or instruction scheduling. None is implemented by this analysis script. New kernels require oracle/sanitizer checks and fresh ordinary timings. Preserve negative results and label cross-precision comparisons. No universal bank-conflict, spill or speedup claim is inferred from one counter.','']
(out/'FINDINGS.md').write_text('\n'.join(lines))
folder=out/'interpretation';folder.mkdir(exist_ok=True);script=Path(__file__).resolve()
if script!=folder/'interpret_profiles.py':shutil.copy2(script,folder/'interpret_profiles.py')
inputs=[out/'metric_summary.csv',out/'all_metrics.csv']+([hot_path] if hot_path.exists() else [])
(folder/'manifest.json').write_text(json.dumps({'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'input_sha256':{str(x.relative_to(out)):hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs}},indent=2)+'\n')
print(out/'FINDINGS.md')
