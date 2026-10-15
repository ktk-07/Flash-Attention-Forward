"""Check measured samples and summarize results"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

FIELDS = ('B', 'H', 'N', 'd', 'mode', 'variant', 'tile', 'softmax_threads')


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def key(row):
    return tuple(row[name] for name in FIELDS)


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def check_close(actual, expected, label):
    if not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f'Incorrect {label}: {actual}; expected {expected}')


def audit_correctness(rows, checks):
    shapes = (('1', '1', '17', '7'), ('2', '2', '33', '33'),
              ('1', '1', '129', '64'), ('1', '1', '257', '128'))
    selections = {key(row)[4:] for row in rows}
    expected = {(shape + selection, 'representative_gate', stage)
                for shape in shapes for selection in selections for stage in ('S', 'P', 'O')}
    fingerprints = {key(row)[:4]: row['input_fingerprint'] for row in rows}
    for row in rows:
        exact = int(row['N']) <= 257
        label = 'exact_CPU_SPO' if exact else 'representative_CPU_gate+finite_O'
        if row['validation'] != label:
            raise ValueError('Incorrect validation scope')
        if exact:
            expected.update((key(row), scope, stage)
                            for scope in ('exact_timed_shape', 'after_timing')
                            for stage in ('S', 'P', 'O'))
    actual = [(key(row), row['scope'], row['stage']) for row in checks]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Missing, duplicate or unexpected correctness reports')
    for row in checks:
        shape = key(row)[:4]
        fingerprint = row['input_fingerprint']
        if not fingerprint.isdigit() or not 0 <= int(fingerprint) < 2**64:
            raise ValueError('Invalid correctness fingerprint')
        if fingerprints.setdefault(shape, fingerprint) != fingerprint:
            raise ValueError('Correctness inputs differ from timing or other checks')


def audit(directory):
    status_file = directory / 'status.txt'
    if status_file.exists():
        status = status_file.read_text(encoding='utf-8-sig').strip()
    else:
        statuses = [line.split('=', 1)[1] for line in
                    (directory / 'metadata.txt').read_text(encoding='utf-8-sig').splitlines()
                    if line.startswith('status=')]
        status = statuses[-1] if statuses else 'UNKNOWN'
    if status != 'PASS' or (directory / 'failure.txt').exists():
        raise ValueError('Benchmark run is incomplete')
    rows = read_csv(directory / 'summary.csv')
    samples = read_csv(directory / 'samples.csv')
    plan = read_csv(directory / 'plan.csv')
    checks = read_csv(directory / 'correctness.csv')
    if not rows or not samples or not checks:
        raise ValueError('Normal timing data and correctness evidence are required')
    groups = defaultdict(list)
    for sample in samples:
        groups[key(sample)].append(sample)
    row_keys = [key(row) for row in rows]
    if len(set(row_keys)) != len(row_keys) or len(plan) != len(rows) or set(row_keys) != set(map(key, plan)) or set(row_keys) != set(groups):
        raise ValueError('Plan, summaries and samples do not match')
    fingerprints = defaultdict(set)
    medians = {}
    for row in rows:
        group = groups[key(row)]
        repeats = int(row['repeats'])
        if not 10 <= repeats <= 30 or len(group) != repeats:
            raise ValueError('Unexpected repeat count')
        if not 5 <= int(row['warmup']) <= 10:
            raise ValueError('Unexpected warmup count')
        if {int(sample['repeat']) for sample in group} != set(range(repeats)):
            raise ValueError('Missing or duplicate repeat index')
        if any(sample['input_fingerprint'] != row['input_fingerprint'] for sample in group):
            raise ValueError('Sample fingerprint differs from summary')
        fingerprints[key(row)[:4]].add(row['input_fingerprint'])
        for stage in ('forward', 'qk', 'softmax', 'pv'):
            times = [float(sample[stage + '_ms']) for sample in group]
            if any(not math.isfinite(value) or value <= 0 for value in times):
                raise ValueError('Invalid timing sample')
            check_close(row[stage + '_median_ms'], quantile(times, .5), stage + ' median')
            check_close(row[stage + '_iqr_ms'], quantile(times, .75) - quantile(times, .25), stage + ' IQR')
            if stage == 'forward':
                check_close(row['forward_q1_ms'], quantile(times, .25), 'Q1')
                check_close(row['forward_q3_ms'], quantile(times, .75), 'Q3')
        b, h, n, d = map(int, key(row)[:4])
        median = float(row['forward_median_ms'])
        medians[key(row)] = median
        check_close(row['sequences_per_second'], 1000 * b / median, 'sequence throughput')
        check_close(row['tokens_per_second'], 1000 * b * n / median, 'token throughput')
        if int(row['peak_workspace_allocated_bytes']) != 16 * b * h * n * d + 8 * b * h * n * n:
            raise ValueError('Incorrect workspace allocation')
    if any(len(values) != 1 for values in fingerprints.values()):
        raise ValueError('Implementations used different inputs')
    for row in rows:
        identity = key(row)
        baseline = identity[:5] + ('naive',) + identity[6:]
        if row['speedup_vs_naive']:
            if baseline not in medians:
                raise ValueError('Speedup has no matched baseline')
            check_close(row['speedup_vs_naive'], medians[baseline] / medians[identity], 'speedup')
        elif baseline in medians:
            raise ValueError('Matched speedup is missing')
    if any(row['status'] != 'PASS' or int(row['failures']) != 0 for row in checks):
        raise ValueError('Correctness evidence contains failures')
    if any(not math.isfinite(float(row['max_absolute'])) or
           not math.isfinite(float(row['max_relative'])) for row in checks):
        raise ValueError('Nonfinite error report')
    audit_correctness(rows, checks)
    return rows, samples, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    rows, samples, checks = audit(args.benchmark)
    args.output.mkdir(parents=True, exist_ok=True)
    configurations = {key(row)[6:] for row in rows}
    # Frozen configuration coverage; default for mixed tuning
    tile, softmax_threads = next(iter(configurations)) if len(configurations) == 1 else ('16', '256')
    required = {(str(b), str(h), str(n), str(d), mode, variant, tile, softmax_threads)
                for b, h in [(4, 8)] for n in (256, 512, 1024, 2048, 4096)
                for d in (32, 64, 128) for mode in ('dense', 'causal')
                for variant in ('naive', 'optimized')}
    measured = set(map(key, rows))
    columns = list(FIELDS) + ['status']
    with (args.output / 'coverage.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for identity in sorted(required):
            writer.writerow(identity + ('MEASURED' if identity in measured else 'NOT_RUN',))
    report = ['# Measured attention results', '',
              f'Source data: `{args.benchmark.as_posix()}`', '',
              f'CSV checks passed for {len(rows)} configurations and {len(samples)} samples.',
              f'All {len(checks)} stage reports passed the fixed tolerance.',
              f'This run covers {len(required & measured)}/60 main-grid rows with tile {tile} and softmax {softmax_threads}.', '',
              '| B,H,N,d | Mode | Variant | Tile/threads | Median ms | IQR ms | Speedup |',
              '| --- | --- | --- | --- | ---: | ---: | ---: |']
    for row in rows:
        shape = ','.join(row[field] for field in FIELDS[:4])
        speedup = f"{float(row['speedup_vs_naive']):.3f}" if row['speedup_vs_naive'] else ''
        report.append(f"| {shape} | {row['mode']} | {row['variant']} | {row['tile']}/{row['softmax_threads']} | "
                      f"{float(row['forward_median_ms']):.6f} | {float(row['forward_iqr_ms']):.6f} | {speedup} |")
    report += ['', 'Workspace bytes describe allocated buffers. They do not measure DRAM traffic.',
               'The data represents warmed, resident-buffer execution. Profiler times are excluded.',
               'Large timed shapes use representative CPU validation and finite-output checks.',
               ('All 60 Part A main-grid rows were measured with one frozen configuration.'
                if required <= measured else 'Missing main-grid rows remain NOT_RUN.'),
               'Fused FlashAttention and Parts B/C remain outside this Part A measurement scope.']
    (args.output / 'README.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    if not args.no_plots:
        from plot_results import plot
        plot(args.benchmark / 'summary.csv', args.output / 'plots')
    print(f'PASS: {len(rows)} rows, {len(samples)} samples; analysis={args.output}')


if __name__ == '__main__':
    main()
