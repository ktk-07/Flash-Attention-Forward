#!/usr/bin/env python3
"""Run existing part workflows on one GPU and collect traceable, separate results.

Shared CPU-oracle linkage and full Part A per-row validation remain integration
work. This runner does not promote unchecked rows to passing measurements or
produce cross-part speedup plots before those requirements are fulfilled.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['part'])
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def best_a_config(rows):
    groups = {}
    for row in rows:
        key = (int(row['tile']), int(row['softmax_threads']))
        value = float(row['forward_median_ms'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid Part A tuning latency')
        groups.setdefault(key, []).append(value)
    if set(groups) != {(8, 256), (16, 256), (32, 256), (16, 128)}:
        raise ValueError('Incomplete Part A candidate grid')
    expected = {(n, mode) for n in ('256', '1024') for mode in ('dense', 'causal')}
    for key in groups:
        subset = [r for r in rows if (int(r['tile']), int(r['softmax_threads'])) == key]
        if len(subset) != 4 or {(r['N'], r['mode']) for r in subset} != expected:
            raise ValueError('Incomplete Part A tuning cases')
    return min(groups, key=lambda key: (statistics.mean(math.log(x) for x in groups[key]), key))


def normalize(part, row):
    """Map only recorded fields. Missing measurements remain missing."""
    a = part == 'part_a'
    precision = row.get('precision', 'FP32') if a else row['input_precision']
    config = f"tile{row['tile']}_softmax{row['softmax_threads']}" if a else row['config']
    result = {
        'part': part, 'experiment': row['experiment'],
        'kernel': row['variant'] if a else row['kernel'], 'config': config,
        **{key: row[key] for key in ['B', 'H', 'N', 'd']},
        'mode': row['mode'] if a else 'causal', 'pattern': row.get('pattern', 'random'),
        'input_precision': precision,
        'output_precision': precision if a else row['output_precision'],
        'accumulator_precision': precision if a else row['accumulator_precision'],
        'input_digest': row.get('input_digest', ''),
        'rounded_input_digest': row.get('rounded_input_digest', ''),
    }
    identity = json.dumps(result, sort_keys=True)
    result['case_id'] = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return result


def validate_gpu_identity(out, parts):
    names = {}
    for part in parts:
        if part == 'part_c':
            values = {r['gpu'] for r in read_csv(out / 'raw/part_c/main/summary.csv')}
            if len(values) != 1 or not next(iter(values)):
                raise ValueError('Part C contains missing or mixed GPU identities')
            names[part] = values.pop()
        else:
            relative = 'main_fp32/environment.txt' if part == 'part_a' else 'environment.txt'
            lines = (out / 'raw' / part / relative).read_text().splitlines()
            values = [line.split('=', 1)[1] for line in lines if line.startswith('gpu=')]
            if len(values) != 1 or not values[0]:
                raise ValueError(f'Missing/ambiguous GPU identity: {part}')
            names[part] = values[0]
    if len(set(names.values())) > 1:
        raise ValueError(f'GPU mismatch: {names}')
    return names


def aggregate(out, parts):
    gpu_names = validate_gpu_identity(out, parts)
    summaries, samples, correctness, resources, memory = [], [], [], [], []
    for part in parts:
        base = out / 'raw' / part
        if part == 'part_a':
            for name in ['correctness_fp32', 'correctness_fp16']:
                for row in read_csv(base / name / 'correctness.csv'):
                    correctness.append({**normalize(part, row), **{k: row[k] for k in ['stage', 'atol', 'rtol', 'max_absolute', 'max_relative', 'failures', 'status']}, 'reference_scope': 'full'})
            stages = ['tuning', 'ablation', 'main_fp32', 'main_fp16']
            # Preserve native merged files for Part A's own plotting script.
            for filename in ['summary.csv', 'samples.csv', 'resources.csv']:
                native = []
                for stage in stages:
                    path = base / stage / filename
                    if path.exists():
                        native.extend(read_csv(path))
                write_csv(base / filename, native)
            for row in read_csv(base / 'summary.csv'):
                checked = row.get('max_absolute', '') != '' and row.get('max_relative', '') != ''
                common = normalize(part, row)
                status = 'PASS' if checked else 'UNCHECKED'
                summaries.append({**common, 'median_ms': row['forward_median_ms'], 'q1_ms': row['forward_q1_ms'], 'q3_ms': row['forward_q3_ms'], 'iqr_ms': row['forward_iqr_ms'], 'repeats': row['repeats'], 'warmups': row['warmup'], 'validation_status': status, 'reference_scope': 'full' if checked else 'not_checked'})
                correctness.append({**common, 'stage': 'O', **{k: row.get(k, '') for k in ['atol', 'rtol', 'max_absolute', 'max_relative']}, 'status': status, 'reference_scope': 'full' if checked else 'not_checked'})
                memory.append({**common, 'explicit_live_device_bytes': row['peak_workspace_allocated_bytes'], 'observed_device_delta_bytes': '', 'measurement': 'persistent_QKVSPO_allocation_ledger'})
            for row in read_csv(base / 'samples.csv'):
                samples.append({**normalize(part, row), 'repeat': row['repeat'], 'forward_ms': row['forward_ms']})
            for row in read_csv(base / 'resources.csv'):
                resources.append({**normalize(part, row), **{k: row[k] for k in ['stage', 'registers_per_thread', 'shared_bytes', 'local_bytes_per_thread', 'active_blocks_per_sm', 'theoretical_occupancy_percent']}})
        elif part == 'part_c':
            for row in read_csv(base / 'main' / 'summary.csv'):
                common = normalize(part, row)
                correctness.append({**common, 'stage': 'O', **{k: row[k] for k in ['atol', 'rtol', 'max_absolute', 'max_relative', 'checked_elements', 'nonfinite', 'failures', 'status', 'reference_scope']}})
                if row['mode'] == 'benchmark':
                    summaries.append({**common, **{k: row[k] for k in ['median_ms', 'q1_ms', 'q3_ms', 'iqr_ms', 'repeats']}, 'warmups': 5, 'validation_status': row['status'], 'reference_scope': row['reference_scope']})
                    resources.append({**common, 'stage': 'forward', **{k: row[k] for k in ['registers_per_thread', 'shared_bytes', 'local_bytes_per_thread', 'active_blocks_per_sm']}, 'theoretical_occupancy_percent': row['occupancy_estimate_percent']})
                elif row['mode'] == 'memory':
                    memory.append({**common, **{k: row[k] for k in ['explicit_live_device_bytes', 'observed_device_delta_bytes']}, 'measurement': 'QKVO_ledger_and_sampled_device_delta'})
            for row in read_csv(base / 'main' / 'measurements.csv'):
                if row['mode'] == 'benchmark':
                    samples.append({**normalize(part, row), 'repeat': row['repeat'], 'forward_ms': row['forward_ms']})
        else:
            # The future B adapter must supply these normalized files and IDs.
            for filename, destination in [('summary.csv', summaries), ('samples.csv', samples), ('correctness.csv', correctness), ('resources.csv', resources), ('memory.csv', memory)]:
                for row in read_csv(base / filename):
                    if row.get('part') != 'part_b' or row.get('mode') != 'dense' or not row.get('case_id'):
                        raise ValueError('Part B must supply normalized dense rows with part/case_id')
                    destination.append(row)
    groups = {}
    for row in samples:
        value = float(row['forward_ms'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid forward timing')
        groups.setdefault(row['case_id'], []).append(row)
    if len({r['case_id'] for r in summaries}) != len(summaries):
        raise ValueError('Duplicate summary case IDs')
    if set(groups) != {row['case_id'] for row in summaries}:
        raise ValueError('Timing samples and summaries have different case sets')
    # These entrypoints always request the complete fixed main grid.
    ns, ds = ['256', '512', '1024', '2048', '4096'], ['32', '64', '128']
    for part in parts:
        if part == 'part_a':
            expected = {(n, d, mode, kernel, precision) for n in ns for d in ds for mode in ['dense', 'causal'] for kernel, precision in [('naive', 'FP32'), ('optimized', 'FP32'), ('tensorcore', 'FP16')]}
            rows = [r for r in summaries if r['part'] == part and r['experiment'] == 'proposal']
        elif part == 'part_c':
            expected = {(n, d, 'causal', kernel, 'FP16' if kernel in ['wmma', 'ptx'] else 'FP32') for n in ns for d in ds for kernel in ['fa1', 'scalar', 'warp', 'tiled', 'skip', 'vector', 'wmma', 'ptx']}
            rows = [r for r in summaries if r['part'] == part and r['experiment'] == 'default_main']
        else:
            required_shapes = {(n, d) for n in ns for d in ds}
            rows = [r for r in summaries if r['part'] == part and r['input_precision'] == 'FP32' and r['B'] == '4' and r['H'] == '8']
            if {(r['N'], r['d']) for r in rows} != required_shapes:
                raise ValueError('Part B has no complete FP32 dense main grid')
            continue
        actual = {(r['N'], r['d'], r['mode'], r['kernel'], r['input_precision']) for r in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError(f'Incomplete/duplicate main grid: {part}')
    for row in summaries:
        group = groups.get(row['case_id'], [])
        if len(group) != 20 or {int(r['repeat']) for r in group} != set(range(20)):
            raise ValueError(f"Missing/duplicate timing samples: {row['case_id']}")
        times = sorted(float(r['forward_ms']) for r in group)
        def quantile(p):
            pos = p * (len(times) - 1); lo = int(pos); hi = min(lo + 1, len(times) - 1)
            return times[lo] + (pos - lo) * (times[hi] - times[lo])
        for field, p in [('median_ms', .5), ('q1_ms', .25), ('q3_ms', .75)]:
            if not math.isclose(float(row[field]), quantile(p), rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f'Summary/sample disagreement: {field}')
        if not math.isclose(float(row['iqr_ms']), quantile(.75) - quantile(.25), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError('Incorrect IQR')
        row['sequences_per_second'] = 1000 * int(row['B']) / float(row['median_ms'])
        row['tokens_per_second'] = row['sequences_per_second'] * int(row['N'])
    for row in correctness:
        if row.get('status') not in ['PASS', 'FAIL', 'UNSUPPORTED', 'UNCHECKED']:
            raise ValueError('Unknown correctness status')
        if row['status'] == 'FAIL' or (row['status'] == 'PASS' and any(int(row.get(k, '') or 0) for k in ['failures', 'nonfinite'])):
            raise ValueError('Failed/contradictory correctness rows; no accepted aggregate generated')
    for rows in [summaries, samples, correctness, resources, memory]:
        for row in rows:
            row.update(run_id=out.name, generator='splitmix64-fp32-v1', seed=5206)
    for name, rows in [('summary', summaries), ('samples', samples), ('correctness', correctness), ('resources', resources), ('memory', memory)]:
        write_csv(out / f'{name}.csv', rows)
    unchecked = sum(r['validation_status'] != 'PASS' for r in summaries)
    report = {'reported_gpu_names': gpu_names, 'summary_rows': len(summaries), 'timing_samples': len(samples), 'unchecked_timing_cases': unchecked, 'statistics_recomputed': True, 'final_integration_ready': False, 'remaining_work': ['Part B final adapter and shared-oracle integration', 'Part A per-row correctness, effective input fingerprints and common tuning grid', 'One shared CPU-oracle agreement gate for all parts'], 'cross_part_speedup_plots_generated': False}
    write_json(out / 'aggregation_audit.json', report)
    comparison = out / 'plots' / 'comparison'; comparison.mkdir(parents=True, exist_ok=True)
    (comparison / 'README.md').write_text('# Cross-part comparison pending\n\nNo cross-part speedups are generated yet. See aggregation_audit.json: shared-oracle integration and complete Part A per-row validation are outstanding. Part B may also be absent from an explicitly selected A/C run.\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', default='all', help='all, or an explicit comma-separated subset: a,c')
    parser.add_argument('--output', type=Path, default=Path('results') / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ'))
    parser.add_argument('--build-root', type=Path, default=Path('/tmp/flash-attention-all-build'))
    parser.add_argument('--arch', default='86', help='numeric CUDA architecture shared by all builds')
    parser.add_argument('--dry-run', action='store_true', help='print commands without building, running, or creating results')
    args = parser.parse_args()
    parts = ['part_a', 'part_b', 'part_c'] if args.parts == 'all' else [f'part_{p.removeprefix("part_")}' for p in args.parts.split(',')]
    if not parts or len(set(parts)) != len(parts) or any(p not in ['part_a', 'part_b', 'part_c'] for p in parts):
        parser.error('--parts must be all or unique entries from a,b,c')
    if not args.arch.isdigit() or int(args.arch) < 80:
        parser.error('--arch must be a numeric SM80-or-newer architecture for Part C PTX MMA')
    # All required entrypoints are checked before creating output or building.
    required = {'part_a': ROOT / 'part_a/CMakeLists.txt', 'part_b': ROOT / 'part_b/run.sh', 'part_c': ROOT / 'part_c/evaluate.sh'}
    for part in parts:
        if not required[part].is_file():
            parser.error(f'{part} is unavailable ({required[part]}). Supply its adapter or explicitly select available parts with --parts a,c. No all-parts run was started.')
    out = args.output.resolve(); build = args.build_root.resolve()
    if out.exists():
        parser.error(f'output already exists; use a fresh directory: {out}')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/flash-attention-matplotlib')
    def run(label, argv, env=None):
        argv = list(map(str, argv))
        overrides = {k: env[k] for k in ['PART_C_BUILD_DIR', 'PART_C_CUDA_ARCHITECTURES'] if env and k in env}
        prefix = ' '.join(f'{k}={shlex.quote(v)}' for k, v in overrides.items())
        command_text = f'{prefix} {shlex.join(argv)}'.strip()
        print(f'{label}: {command_text}', flush=True)
        if args.dry_run:
            return
        start = datetime.now(timezone.utc).isoformat()
        with (out / 'logs' / f'{label}.txt').open('w') as stream:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        with (out / 'commands.jsonl').open('a') as stream:
            stream.write(json.dumps({'label': label, 'argv': argv, 'cwd': str(ROOT), 'started_utc': start, 'exit_code': result.returncode, 'environment_overrides': {k: env[k] for k in ['PART_C_BUILD_DIR', 'PART_C_CUDA_ARCHITECTURES'] if env and k in env}}) + '\n')
        if result.returncode:
            raise RuntimeError(f'{label} failed ({result.returncode}); see {out / "logs" / (label + ".txt")}')
    manifest = {'requested_parts': parts, 'architecture': args.arch, 'seed': 5206, 'generator': 'Part A SplitMix64 Q then K then V', 'warmups': 5, 'repeats': 20, 'status': 'running', 'final_integration_ready': False, 'started_utc': datetime.now(timezone.utc).isoformat(), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'unset; CUDA device 0'), 'source_sha256': {}}
    if not args.dry_run:
        tools = ['cmake', 'nvcc', 'nvidia-smi', 'c++']
        if 'part_c' in parts:
            tools.append('compute-sanitizer')
        for tool in tools:
            if not shutil.which(tool):
                raise RuntimeError(f'Missing dependency: {tool}')
        matplotlib = __import__('matplotlib'); numpy = __import__('numpy')
        manifest['analysis_environment'] = {'python': sys.version, 'matplotlib': matplotlib.__version__, 'numpy': numpy.__version__}
        (out / 'logs').mkdir(parents=True)
        (out / 'configs').mkdir()
        for part in parts:
            for source in sorted((ROOT / part).rglob('*')):
                if source.is_file() and not any(x in source.relative_to(ROOT / part).parts for x in ['results', 'extra', 'build', '__pycache__']) and source.suffix in ['.cu', '.cpp', '.hpp', '.h', '.py', '.sh', '.json', '.txt']:
                    manifest['source_sha256'][str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        if 'part_c' in parts:
            for name in ['input_generation.hpp', 'attention_types.hpp']:
                source = ROOT / 'part_a/common' / name
                manifest['source_sha256'][str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        for source in [ROOT / 'run_all.sh', Path(__file__).resolve()]:
            manifest['source_sha256'][str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        write_json(out / 'manifest.json', manifest)
    try:
        for label, cmd in [('gpu', ['nvidia-smi', '-q']), ('nvcc', ['nvcc', '--version']), ('cmake', ['cmake', '--version']), ('host_compiler', ['c++', '--version'])]:
            run(label, cmd)
        binaries = {}
        # Build every requested part before starting GPU experiments.
        for part in parts:
            if part == 'part_b':
                continue
            dest = build / part
            cmd = ['cmake', '-S', ROOT / part, '-B', dest, '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_CUDA_ARCHITECTURES={args.arch}']
            if part == 'part_a':
                cmd += ['-DPART_A_ENABLE_CUDA=ON', '-DPART_A_PROFILE_INFO=ON']
            run(part + '_configure', cmd); run(part + '_build', ['cmake', '--build', dest, '--parallel', '2'])
            binaries[part] = dest / ('attention' if part == 'part_a' else 'part_c_evaluation')
        if not args.dry_run:
            manifest['binary_sha256'] = {part: hashlib.sha256(path.read_bytes()).hexdigest() for part, path in binaries.items()}
            write_json(out / 'manifest.json', manifest)
        for part in parts:
            base = out / 'raw' / part
            if part == 'part_a':
                binary = binaries[part]
                for label, variants in [('correctness_fp32', 'all'), ('correctness_fp16', 'tensorcore')]:
                    run('a_' + label, [binary, '--grid', 'correctness', '--mode', 'both', '--variants', variants, '--output', base / label])
                run('a_tuning', [binary, '--grid', 'tuning', '--mode', 'both', '--variants', 'optimized', '--output', base / 'tuning'])
                tile, threads = ('<tuned_tile>', '<tuned_threads>') if args.dry_run else best_a_config(read_csv(base / 'tuning/summary.csv'))
                if not args.dry_run:
                    write_json(out / 'configs/part_a.json', {'tile': tile, 'softmax_threads': threads, 'selection': 'geometric mean across both modes and N256/1024, d64; fixed before proposal grid', 'tensorcore': {'tile': 16, 'softmax_threads': 256, 'selection': 'existing fixed baseline, not tuned'}})
                for label, grid, variants, t, sm in [('ablation', 'ablation', 'all', tile, threads), ('main_fp32', 'proposal', 'main', tile, threads), ('main_fp16', 'proposal', 'tensorcore', 16, 256)]:
                    run('a_' + label, [binary, '--grid', grid, '--mode', 'both', '--variants', variants, '--tile', t, '--softmax-threads', sm, '--output', base / label])
            elif part == 'part_c':
                cb = build / part
                run('c_correctness', [cb / 'part_c_runner', '--suite', 'correctness', '--output', base / 'regression'])
                for tool in ['memcheck', 'racecheck', 'synccheck', 'initcheck']:
                    run('c_' + tool, ['compute-sanitizer', '--tool', tool, '--error-exitcode', '3', cb / 'part_c_runner', '--suite', 'smoke', '--output', base / 'safety' / tool])
                env = dict(os.environ, PART_C_BUILD_DIR=str(cb), PART_C_CUDA_ARCHITECTURES=args.arch)
                run('c_study', ['bash', ROOT / 'part_c/evaluate.sh', base / 'main'], env)
                run('c_frozen_safety', ['bash', ROOT / 'part_c/evaluate.sh', '--verify', base / 'main'], env)
                if not args.dry_run:
                    shutil.copy2(base / 'main/frozen_selection.json', out / 'configs/part_c.json')
            else:
                run('b_workflow', ['bash', ROOT / 'part_b/run.sh', '--output', base, '--build', build / part, '--arch', args.arch])
        if args.dry_run:
            print('Then normalize CSVs, verify timing statistics, and generate per-part plots. No commands executed or output created. Shared-oracle integration remains pending.')
            return
        audit = aggregate(out, parts)
        manifest['reported_gpu_names'] = audit['reported_gpu_names']
        for part in parts:
            target = out / 'plots' / part
            if part == 'part_a':
                run('a_plots', [sys.executable, ROOT / 'part_a/plot_results.py', out / 'raw/part_a', '--output', target])
            else:
                source = out / 'raw' / part / ('main/figures' if part == 'part_c' else 'plots')
                if not source.is_dir():
                    raise RuntimeError(f'Missing per-part plots: {source}')
                shutil.copytree(source, target)
        manifest.update(status='completed_requested_workflows', completed_utc=datetime.now(timezone.utc).isoformat(), aggregation=audit)
        write_json(out / 'manifest.json', manifest)
        print(f'Completed {", ".join(parts)}: {out}. Final shared-oracle integration remains pending.')
    except Exception as error:
        if not args.dry_run:
            manifest.update(status='failed', error=str(error)); write_json(out / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
