#!/usr/bin/env python3
"""Export native Part C studies to Part A-based report tables, without data loss.

Copies native evidence into a self-contained report bundle. Historical/default
profiles can be included as supplementary evidence; they are never relabelled
as counters for the tuned configurations or their different input stream.
"""
import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import results_schema as schema

GENERATOR = 'splitmix64-fp32-v1'
KERNELS = ['fa1', 'scalar', 'warp', 'tiled', 'skip', 'vector', 'wmma', 'ptx']
PHASES = {'validation': 'correctness', 'default_main': 'proposal_default', 'frozen_main': 'proposal', 'tuning': 'tuning', 'matched_load': 'ablation_vector_load', 'memory': 'memory'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def inventory(folder):
    return {str(p.relative_to(folder)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(folder.rglob('*')) if p.is_file()}


def check_declared_files(folder, declared):
    missing=[]
    for name, item in declared.items():
        path=folder/name
        if not path.exists():
            generated_figure = 'figures' in path.parts and path.suffix in ['.png','.pdf','.svg']
            if path.suffix not in ['.ncu-rep','.ncu-repz','.log'] and not generated_figure:
                raise ValueError(f'Required native evidence is missing: {name}')
            missing.append(name)
        elif sha(path)!=item['sha256']:
            raise ValueError(f'Native evidence changed since export: {name}')
    return missing


def common(row, run_id, plan, selection):
    phase = PHASES[row['experiment']]
    if row['experiment'] == 'default_main' and row['kernel'] in selection['variants']:
        if selection['variants'][row['kernel']][row['d']]['config'] == int(row['config']):
            phase = 'proposal'
    result = {field: '' for field in schema.COMMON}
    result.update(experiment=phase, precision=row['input_precision'], **{field: row[field] for field in ['B','H','N','d','pattern','input_precision','accumulator_precision','output_precision','input_digest','rounded_input_digest']})
    result.update(mode='causal', variant=row['kernel'], run_id=run_id, part='part_c', config=f"{row['kernel']}:c{row['config']}:d{row['d']}", layout='BHNd_contiguous_row_major', generator=GENERATOR, seed=str(plan['seed']))
    result['case_id'] = schema.case_id(result)
    return result


def verify_native(study):
    plan = json.loads((study / 'plan.json').read_text())
    if plan['seed'] != 5206 or plan['mode'] != 'causal' or 'SplitMix64' not in plan['generator']:
        raise ValueError('Exporter requires the common-input bounded study, not legacy mt19937 runs')
    selection = json.loads((study / 'frozen_selection.json').read_text())
    if selection['plan_sha256'] != sha(study / 'plan.json'):
        raise ValueError('Frozen selection/plan mismatch')
    manifest = json.loads((study / 'manifest.json').read_text())
    for name, expected in manifest['sources'].items():
        if sha(study / 'source_snapshot' / name) != expected:
            raise ValueError(f'Measurement source snapshot changed: {name}')
    rows = schema.read(study / 'summary.csv'); samples = schema.read(study / 'measurements.csv')
    analysis = json.loads((study / 'analysis_manifest.json').read_text())
    if sha(study / 'analysis/evaluate.py') != analysis['script_sha256']:
        raise ValueError('Captured plot processor hash mismatch')
    if sha(study / 'measurements.csv') != analysis['measurements_sha256']:
        raise ValueError('Raw measurement hash mismatch')
    for name, expected in analysis['case_csv_sha256'].items():
        if sha(study / name) != expected:
            raise ValueError(f'Native raw case changed: {name}')
    keys = ['experiment','kernel','config','B','H','N','d','pattern']
    groups = {}
    for sample in samples:
        groups.setdefault(tuple(sample[k] for k in keys), []).append(sample)
    if {tuple(row[k] for k in keys) for row in rows} != set(groups):
        raise ValueError('Native summary/raw case sets disagree')
    unavailable = {'registers_per_thread','shared_bytes','local_bytes_per_thread','active_blocks_per_sm','occupancy_estimate_percent'}
    for row in rows:
        group = groups[tuple(row[k] for k in keys)]; first = group[0]
        for field, value in first.items():
            if field in ['repeat','forward_ms']:
                continue
            if any(sample[field] != value for sample in group):
                raise ValueError(f'Inconsistent raw case metadata: {field}')
            if row['kernel'] in ['fa1','scalar','warp'] and field in unavailable:
                continue  # Native summary already marks unavailable query fields blank.
            if row[field] != value:
                raise ValueError(f'Native summary metadata differs from raw rows: {field}')
    return plan, selection, manifest, rows, samples


def make_tables(study, run_id):
    plan, selection, manifest, rows, samples = verify_native(study)
    tables = {name: [] for name in schema.HEADERS}
    native_keys = lambda row: tuple(row[k] for k in ['experiment','kernel','config','B','H','N','d','pattern'])
    contexts, configs, timing_index = {}, {}, {}
    for row in rows:
        ctx = common(row, run_id, plan, selection); key = native_keys(row)
        if key in contexts:
            raise ValueError('Duplicate native case')
        contexts[key] = ctx
        if row['status'] != 'PASS' or int(row['failures']) or int(row['nonfinite']):
            raise ValueError('Native study contains failed validation; no accepted export produced')
        query_rows = 'all' if row['reference_scope'] == 'full' else json.dumps(sorted({0, int(row['N'])//2, int(row['N'])-1}), separators=(',', ':'))
        tables['correctness'].append({**ctx, 'stage': 'O', **{k: row[k] for k in ['atol','rtol','max_absolute','max_relative','failures','status','reference_scope','checked_elements','nonfinite']}, 'oracle_input_precision': 'FP16-rounded' if row['input_precision'] == 'FP16' else 'FP32', 'checked_query_rows': query_rows})
        params = {k: int(row[k]) for k in ['block_q','block_kv','warps','thread_q','thread_k']}
        if ctx['config'] in configs and configs[ctx['config']]['launch'] != params:
            raise ValueError('Config ID maps to different launch parameters')
        configs[ctx['config']] = {'variant': row['kernel'], 'd': int(row['d']), 'native_config': int(row['config']), 'launch': params}
        count = math.prod(int(row[k]) for k in ['B','H','N','d'])
        if int(row['explicit_live_device_bytes']) != count * (10 if row['input_precision'] == 'FP16' else 16):
            raise ValueError('Explicit allocation ledger does not match the recorded precision')
        if row['mode'] == 'benchmark':
            tables['summary'].append({**ctx, **{k: row[k] for k in ['atol','rtol','max_absolute','max_relative','repeats','sequences_per_second','tokens_per_second','reference_scope']}, 'warmup': str(plan['warmups']), 'forward_median_ms': row['median_ms'], 'forward_q1_ms': row['q1_ms'], 'forward_q3_ms': row['q3_ms'], 'forward_iqr_ms': row['iqr_ms'], 'validation_status': row['status'], 'peak_workspace_allocated_bytes': row['explicit_live_device_bytes']})
            blocks = int(row['B'])*int(row['H'])*math.ceil(int(row['N'])/params['block_q'])
            tables['resources'].append({**ctx, 'stage': 'forward', 'blocks': str(blocks), 'threads': str(32*params['warps']), **{k: row[k] for k in ['registers_per_thread','shared_bytes','local_bytes_per_thread','active_blocks_per_sm']}, 'theoretical_occupancy_percent': row['occupancy_estimate_percent']})
            n = int(row['N']); heads = int(row['B'])*int(row['H'])
            work = {**ctx, 'valid_attention_pairs': str(heads*n*(n+1)//2), 'scheduled_tiles': '', 'skipped_tiles': '', 'evidence_kind': 'analytical_valid_attention_pairs'}
            if row['kernel'] in ['tiled','skip','vector','wmma','ptx']:
                bq, bk = params['block_q'], params['block_kv']
                scheduled = heads*sum(math.ceil(min(n, q+bq)/bk) for q in range(0,n,bq))
                dense = heads*math.ceil(n/bq)*math.ceil(n/bk)
                work.update(scheduled_tiles=str(scheduled), skipped_tiles=str(dense-scheduled), evidence_kind='analytical_pairs_and_source_derived_CTA_KV_tile_schedule_not_instruction_counts')
            tables['work'].append(work)
            if ctx['experiment'] == 'proposal':
                timing_index[tuple(row[k] for k in ['kernel','config','B','H','N','d','pattern'])] = (ctx, row)
        elif row['mode'] == 'memory':
            tables['memory'].append({**ctx, 'explicit_live_device_bytes': row['explicit_live_device_bytes'], 'auxiliary_global_bytes': '0', 'observed_device_delta_bytes': row['observed_device_delta_bytes'], 'measurement': 'persistent_QKVO_ledger;device_delta_observed_at_three_points_not_total_process_peak'})
    for row in samples:
        if row['mode'] == 'benchmark':
            ctx = contexts.get(native_keys(row))
            if not ctx:
                raise ValueError('Raw timing case has no native summary')
            tables['samples'].append({**ctx, 'repeat': row['repeat'], 'forward_ms': row['forward_ms']})
    expected = {(k, str(n), str(d)) for k in KERNELS for n in plan['main_N'] for d in plan['d']}
    actual = {(r['kernel'],r['N'],r['d']) for r in rows if r['experiment']=='default_main'}
    if actual != expected:
        raise ValueError('Incomplete default main grid')
    profile_count, binary_count, missing_reports = import_profiles(study, manifest, timing_index, tables)
    audit = schema.validate(tables)
    audit.update(native_cases=len(rows), profile_captures=profile_count, profile_csv_hashes_verified=bool(profile_count), verified_binary_profile_reports=binary_count, missing_binary_profile_reports=missing_reports, native_values_retained=True, fused_stage_timings_unavailable=True, shared_cpu_reference_linked=bool(plan.get('shared_oracle_linked',False)), final_abc_integration_complete=False)
    return tables, configs, selection, audit


def import_profiles(study, manifest, timing_index, tables):
    profile = study / 'profiling'
    if not (profile / 'plan.json').exists():
        return 0, 0, []
    plan = json.loads((profile / 'plan.json').read_text())
    if plan['binary_sha256'] != manifest['binary_sha256'] or plan['selection_sha256'] != sha(study / 'frozen_selection.json'):
        raise ValueError('Profile belongs to another binary/selection')
    audit = json.loads((profile / 'audit.json').read_text())
    verified = 0; missing = []
    for name, expected in audit['raw_report_sha256'].items():
        path = profile / name
        if not path.exists():
            missing.append({'path':name,'expected_sha256':expected,'reason':'binary report not supplied; saved CSV evidence remains available'})
        elif sha(path) != expected:
            raise ValueError('Raw Nsight report hash mismatch')
        else:
            verified += 1
    processing = json.loads((profile / 'processing/manifest.json').read_text())
    for name, expected in processing['raw_csv_sha256'].items():
        if sha(profile / name) != expected:
            raise ValueError('Raw Nsight CSV hash mismatch')
    reproducibility = study / 'reproducibility_audit.json'
    if reproducibility.exists():
        expected = json.loads(reproducibility.read_text())['processed_sha256'].get('profiling/metrics.csv')
        if expected and sha(profile / 'metrics.csv') != expected:
            raise ValueError('Derived profiler metrics changed')
    derived = {(r['kernel'],r['config'],r['N'],r['d'],r['rep']): r for r in schema.read(profile / 'metrics.csv')}
    cases = sorted((profile / 'cases').iterdir())
    if len(cases) != 12:
        raise ValueError('Expected twelve frozen-winner captures')
    for case in cases:
        identity = json.loads((case / 'identity.json').read_text())
        checks = schema.read(case / 'validation.csv')
        if not checks or any(r['status'] != 'PASS' for r in checks):
            raise ValueError('Unvalidated profile')
        first = checks[0]
        if any(str(identity[k]) != first[k] for k in ['kernel','config','N','d']):
            raise ValueError('Profiler identity disagrees with validation workload')
        key = tuple(first[k] for k in ['kernel','config','B','H','N','d','pattern'])
        if key not in timing_index:
            raise ValueError('Profile does not match a frozen timing case')
        ctx, reference = timing_index[key]
        for field in ['input_digest','rounded_input_digest','block_q','block_kv','warps','thread_q','thread_k']:
            if any(r[field] != reference[field] for r in checks):
                raise ValueError(f'Profile fingerprint/config mismatch: {field}')
        with (case / 'raw.csv').open(newline='') as stream:
            raw = list(csv.reader(stream))
        if len(raw) != 3 or raw[0][0] != 'ID':
            raise ValueError('Expected one raw profiled kernel')
        names, units, values = raw
        symbol = values[names.index('Kernel Name')]
        raw_data = dict(zip(names, values))
        state_values = []
        for field, value in raw_data.items():
            if field.startswith('smsp__pcsamp_warps_issue_stalled_') and not field.endswith('_not_issued'):
                try: state_values.append(float(value.replace(',', '')))
                except ValueError: state_values.append(float('nan'))
        sampled_states_available = bool(state_values) and all(math.isfinite(x) for x in state_values) and sum(state_values)>0
        evidence = case/('profile.ncu-rep' if (case/'profile.ncu-rep').exists() else 'raw.csv')
        source = 'raw/part_c/study/' + str(evidence.relative_to(study))
        for name, unit, value in zip(names[11:], units[11:], values[11:]):
            try:
                parsed = float(value.replace(',', ''))
                available = math.isfinite(parsed)
            except ValueError:
                available = False
            tables['profiles'].append({**ctx, 'kernel': symbol, 'metric': name, 'unit': unit, 'value': value.replace(',', '') if available else '', 'stage': 'forward', 'profile_repeat': str(identity['rep']), 'availability': 'available' if available else 'unavailable', 'reason': '' if available else f'Nsight exported a missing/non-numeric value: {value!r}; exact original retained in raw.csv', 'source_report': source})
        record = derived[(first['kernel'],first['config'],first['N'],first['d'],str(identity['rep']))]
        for name, value in record.items():
            if name in ['kernel','config','N','d','rep']:
                continue
            unit = '%' if name.endswith('_pct') else ('byte' if name.endswith('_bytes') else 'sector' if name.endswith('_sectors') else 'wavefront' if name.startswith('shared_') else '')
            try: available = math.isfinite(float(value))
            except ValueError: available = False
            if name in ['long_scoreboard_pct','short_scoreboard_pct','mio_throttle_pct']:
                available = available and sampled_states_available
            tables['profiles'].append({**ctx, 'kernel': symbol, 'metric': 'derived::'+name, 'unit': unit, 'value': value if available else '', 'stage': 'forward', 'profile_repeat': str(identity['rep']), 'availability': 'available' if available else 'unavailable', 'reason': '' if available else 'derived counter/sampling denominator unavailable; native value retained in raw', 'source_report': source})
    return len(cases), verified, missing


def regenerate_study_plots(study, output):
    """Run the captured CPU plot processor in isolation, never over raw evidence."""
    processor=study/'analysis/evaluate.py'
    if not processor.exists():
        raise ValueError('Captured native plot processor is required for reproduction')
    with tempfile.TemporaryDirectory(prefix='part-c-native-plots-') as directory:
        work=Path(directory)
        for name in ['measurements.csv','manifest.json','plan.json','frozen_selection.json']:
            shutil.copy2(study/name,work/name)
        for name in ['cases','source_snapshot']:
            shutil.copytree(study/name,work/name)
        command=[sys.executable,str(processor),'--build','/tmp/unused-build','--output',str(work),'--analyze-only']
        result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if result.returncode:
            raise ValueError('Native plot regeneration failed: '+result.stdout)
        output.mkdir(parents=True,exist_ok=True)
        figures=sorted((work/'figures').glob('*'))
        if len(figures)!=14:raise ValueError('Expected seven PNG/PDF study figure pairs')
        for path in figures:shutil.copy2(path,output/path.name)
    return sha(processor)


def plot_insights(study, output, profiles):
    """Extra profiler plots; native reports/metrics/source exports stay intact."""
    path = study / 'profiling/metrics.csv'
    if not path.exists():
        return []
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/part-c-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import statistics
    grouped = {}
    for r in profiles:
        if not r['metric'].startswith('derived::'):continue
        key=(r['variant'],r['N'],r['d'],r['profile_repeat'])
        row=grouped.setdefault(key,{'kernel':r['variant'],'N':r['N']})
        row[r['metric'].removeprefix('derived::')]=r['value'] if r['availability']=='available' else ''
    rows=list(grouped.values());variants=sorted({r['kernel'] for r in rows})
    def median(k,n,col):
        values=[float(r[col]) for r in rows if r['kernel']==k and int(r['N'])==n and r.get(col,'')!='']
        return statistics.median(values) if values else float('nan')
    output.mkdir(parents=True, exist_ok=True); names=[]
    for name, columns, ylabel in [('occupancy',['occupancy_pct'],'Achieved occupancy (%)'),('eligible_warps',['eligible_warps_per_scheduler'],'Eligible warps per scheduler'),('tensor_activity',['tensor_active_pct'],'Tensor pipeline (% of active-cycle peak)'),('warp_states',['long_scoreboard_pct','short_scoreboard_pct','mio_throttle_pct'],'Share of sampled warp states (%)'),('shared_wavefronts',['shared_actual','shared_ideal','shared_excessive'],'Shared wavefront count'),('traffic',['dram_read_bytes','dram_write_bytes','local_load_sectors','local_store_sectors'],'Traffic (GiB; distinct memory levels)')]:
        fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,n in zip(axes,[256,4096]):
            width=.8/len(columns)
            for i,col in enumerate(columns):
                vals=[median(k,n,col) for k in variants]
                if name=='traffic':vals=[v*(32 if col.endswith('_sectors') else 1)/2**30 for v in vals]
                positions=[j+(i-(len(columns)-1)/2)*width for j in range(len(variants))]
                ax.bar(positions,vals,width=width,label=col)
                for position,value in zip(positions,vals):
                    if not math.isfinite(value):ax.text(position,0,'n/a',rotation=90,fontsize=7)
            ax.set_xticks(range(len(variants)),variants);ax.set(title=f'N={n}, d=64',ylabel=ylabel);ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
        fig.suptitle('Frozen winners: medians of 3 Nsight captures; profiler timing excluded')
        for extension in ['png','pdf']:
            target=output/f'{name}.{extension}';options={'dpi':160} if extension=='png' else {'metadata':{'CreationDate':None,'ModDate':None}}
            fig.savefig(target,**options);names.append(target.name)
        plt.close(fig)
    return names


def publish(study, out, supplementary=(), in_place=False, regenerate=False):
    study, out = Path(study).resolve(), Path(out).resolve()
    if study == out or out.is_relative_to(study):
        raise ValueError('Output must not be inside the native study')
    if regenerate:
        previous = json.loads((out / 'manifest.json').read_text())
        if previous.get('exporter') != 'part_c_to_part_a_v1':
            raise ValueError('Not an owned Part C standardized bundle')
        run_id = previous['run_id']
        check_declared_files(study,previous['native_inventory'])
        for name,declared in previous['supplementary_inventory'].items():
            check_declared_files(out/name,declared)
        supplementary = [out/path for path in previous['supplementary_datasets']]
    else:
        run_id = out.name
        if out.exists() and not (in_place and study.is_relative_to(out/'raw') and set(p.name for p in out.iterdir()) <= {'raw','logs','commands.jsonl','device.json','run_status.json'}):
            raise ValueError('Choose a fresh output directory; existing exports are not overwritten')
    # Fully validate before copying/publishing any accepted table.
    tables, configs, selection, audit = make_tables(study, run_id)
    before = inventory(study)
    out.mkdir(parents=True, exist_ok=True)
    native = out / 'raw/part_c/study'
    if study != native:
        native.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(study, native)
    if inventory(native) != before:
        raise ValueError('Native evidence copy is not byte-identical')
    declared_native={**(previous['native_inventory'] if regenerate else {}),**before}
    historical=[]
    for source in supplementary:
        source=Path(source).resolve();target=out/'raw/part_c/historical_profiles'/source.name
        if source != target:
            if target.exists():raise ValueError('Duplicate supplementary dataset name')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,target)
            if inventory(source)!=inventory(target):raise ValueError('Supplementary evidence changed during copy')
        historical.append(str(target.relative_to(out)))
    for name, rows in tables.items():schema.write(out/f'{name}.csv',name,rows)
    registry=out/'configs';registry.mkdir(exist_ok=True)
    dump(registry/'part_c.json',{'configurations':configs,'frozen_selection':selection})
    for name in ['plan.json','frozen_selection.json']:
        shutil.copy2(native/name, registry/name)
    plots=out/'plots/part_c';plots.mkdir(parents=True,exist_ok=True)
    native_plot_processor_sha256=regenerate_study_plots(native,plots)
    extra=out/'supplementary/part_c';extra.mkdir(parents=True,exist_ok=True)
    generated=plot_insights(native,extra/'plots',tables['profiles'])
    source_copy=out/'analysis';source_copy.mkdir(exist_ok=True)
    for source in [Path(__file__).resolve(),Path(schema.__file__).resolve()]:
        if source != source_copy/source.name:
            shutil.copy2(source,source_copy/source.name)
    native_manifest=json.loads((native/'manifest.json').read_text())
    symbols={
        'fa1':'flash_attention_v1_baseline','scalar':'flash_attention_v2_scalar_baseline',
        'warp':'flash_attention_v2_warp_per_query','tiled':'flash_attention_v2_cuda_core_tiled',
        'skip':'tile_skip::casual_flash_attention_forward_gpu_optimised_for_cuda_core',
        'vector':'vector_load::casual_flash_attention_forward_gpu_optimised_for_cuda_core',
        'wmma':'flash_attention_wmma_fused','ptx':'flash_attention_v2_tensor_core_ptx_mma_fully_fused'}
    implementations={}
    for variant,symbol in symbols.items():
        tensor=variant in ['wmma','ptx']
        filename='flash_attention_1.cu' if variant=='fa1' else 'flash_attention_2_tensor_core.cu' if tensor else 'flash_attention_2_cuda_core.cu'
        relative='part_c/src/'+filename
        implementations[variant]={'symbol':symbol,'source':'raw/part_c/study/source_snapshot/'+relative,'source_sha256':native_manifest['sources'][relative], 'input_precision':'FP16' if tensor else 'FP32','accumulator_precision':'FP32','output_precision':'FP32','softmax_precision':'FP32','probability_operand_precision':'FP16' if tensor else 'FP32'}
    manifest={'schema_version':schema.VERSION,'native_plot_processor_sha256':native_plot_processor_sha256,'exporter':'part_c_to_part_a_v1','grid':json.loads((native/'plan.json').read_text()).get('grid','full'),'run_id':run_id,'part':'part_c','implementations':implementations,'export_kind':'format_adaptation_not_a_new_GPU_measurement','measurement_started_utc':json.loads((native/'manifest.json').read_text()).get('started_utc',''),'native_binary_sha256':json.loads((native/'manifest.json').read_text())['binary_sha256'],'native_source':'raw/part_c/study','supplementary_datasets':historical,'native_inventory':declared_native,'supplementary_inventory':{name:inventory(out/name) for name in historical},'source_columns_preserved_in_raw':list(schema.read(native/'measurements.csv')[0]),'experiment_mapping':PHASES,'selected_default_config_promoted_to_proposal':True,'mapping':{'kernel':'variant','mode':'native execution mode replaced by causal attention mode; native value retained in raw','median_ms':'forward_median_ms','q1_ms':'forward_q1_ms','q3_ms':'forward_q3_ms','iqr_ms':'forward_iqr_ms'},'profiling':{'captures':audit['profile_captures'],'all_raw_metric_names_units_and_original_values_retained':True,'verified_binary_reports':audit['verified_binary_profile_reports'],'missing_binary_reports':audit['missing_binary_profile_reports'],'instrumented_times_excluded_from_samples':True,'historical_profiles_are_supplementary_not_relabelled_as_current':historical},'reference_backend':json.loads((native/'plan.json').read_text()).get('reference_backend','historical Part C FP64 oracle'),'final_abc_integration_complete':False,'processing_sha256':{s.name:sha(s) for s in source_copy.iterdir() if s.is_file()}}
    dump(out/'manifest.json',manifest)
    audit['external_native_files_missing']=check_declared_files(native,declared_native)
    audit['native_inventory_verified']=inventory(study)==before
    audit['schema_headers_fixed']=True
    dump(out/'audit.json',audit)
    links=['# Part C supplementary evidence','','All native data, MIO/scoreboard counters, Nsight reports, logs, source snapshots and insights are retained byte for byte under [raw study](../../raw/part_c/study/). Only public column names are adapted.','', 'The profiles table contains all raw Nsight metric columns plus the native derived metrics. Unavailable counters remain explicit. Sampled warp-state percentages are not percentages of runtime. Shared actual/ideal/excessive wavefronts diagnose source excess; aggregate conflict counters alone include arbitration. Local sector traffic is not allocated storage.','','## Native reports','','- [Measured study](../../raw/part_c/study/REPORT.md)']
    if audit['profile_captures']:links.append('- [Frozen profiler findings](../../raw/part_c/study/profiling/REPORT.md)')
    for name in historical:links.append(f'- [Supplementary historical dataset](../../{name}/): different input stream/configurations; not joined to current timing cases.')
    links+=['','## Additional generated profiler plots','']+[f'- [{name}](plots/{name})' for name in generated]
    (extra/'README.md').write_text('\n'.join(links)+'\n')
    figure_index=out/'plots/README.md';figure_index.parent.mkdir(exist_ok=True)
    figure_index.write_text('# Part C figure index\n\nThe seven native study figures are in [part_c/](part_c/). Additional profiler charts and preserved insights are indexed in [supplementary/part_c/](../supplementary/part_c/README.md).\n')
    (out/'ANALYSIS.md').write_text('# Part C report export\n\nThis bundle uses Part A column names with explicit precision, fingerprints and validation scope. All unprofiled timings and numerical checks are unchanged. See [native measured findings](raw/part_c/study/REPORT.md) and [supplementary profiler insights](supplementary/part_c/README.md).\n\nFused stage times, setup timing and inappropriate naive speedups are blank. Allocation fields measure explicit live buffers. Work.csv uses analytical valid pairs and source-derived CTA tile schedules, not measured executed FLOPs. The manifest records the actual oracle backend. Historical bundles keep their original reference provenance; fresh runs use the shared Part A CPU library.\n')
    if not audit['native_inventory_verified']:raise ValueError('Source changed during export')
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--supplementary-profile',type=Path,action='append',default=[])
    parser.add_argument('--in-place',action='store_true',help='publish into an existing wrapper output containing only raw/logs')
    parser.add_argument('--analyze-only',type=Path,metavar='BUNDLE',help='regenerate owned public tables/plots from immutable copied raw data')
    args=parser.parse_args()
    if args.analyze_only:
        if args.study or args.output or args.supplementary_profile:parser.error('--analyze-only takes only the existing bundle')
        out=args.analyze_only.resolve();study=out/'raw/part_c/study'
        audit=publish(study,out,regenerate=True)
    else:
        if not args.study or not args.output:parser.error('--study and --output are required')
        audit=publish(args.study,args.output,args.supplementary_profile,args.in_place)
    target = args.analyze_only.resolve() if args.analyze_only else args.output.resolve()
    with (target/'commands.jsonl').open('a') as stream:
        stream.write(json.dumps({'operation':'analyze-only' if args.analyze_only else 'export','argv':[sys.executable,*sys.argv],'exit_code':0,'note':'CPU-only adaptation; original measured commands are retained under raw/part_c/study/commands.jsonl'})+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError) as error:
        print(f'ERROR: {error}',file=sys.stderr);sys.exit(1)
