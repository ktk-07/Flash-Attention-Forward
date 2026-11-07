"""Version 1 report tables: Part A names plus shared provenance/validation fields.

Native files remain authoritative evidence. Empty cells mean unavailable or
not applicable, never a fabricated measurement. No GPU dependencies.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

VERSION = 1
BASE = 'experiment precision B H N d mode variant tile softmax_threads'.split()
META = 'run_id case_id part config pattern layout input_precision accumulator_precision output_precision generator seed input_digest rounded_input_digest'.split()
COMMON = BASE + META
EXTRA = {
    'correctness': 'stage atol rtol max_absolute max_relative failures status oracle_input_precision reference_scope checked_query_rows checked_elements nonfinite'.split(),
    'samples': 'repeat forward_ms qk_ms softmax_ms pv_ms'.split(),
    'summary': 'atol rtol max_absolute max_relative warmup repeats forward_median_ms forward_q1_ms forward_q3_ms forward_iqr_ms qk_median_ms qk_iqr_ms softmax_median_ms softmax_iqr_ms pv_median_ms pv_iqr_ms sequences_per_second speedup_vs_naive peak_workspace_allocated_bytes setup_host_ms tokens_per_second validation_status reference_scope'.split(),
    'resources': 'stage blocks threads registers_per_thread shared_bytes local_bytes_per_thread active_blocks_per_sm theoretical_occupancy_percent'.split(),
    'profiles': 'kernel metric unit value stage profile_repeat availability reason source_report'.split(),
    'memory': 'explicit_live_device_bytes auxiliary_global_bytes observed_device_delta_bytes measurement'.split(),
    'work': 'valid_attention_pairs scheduled_tiles skipped_tiles evidence_kind'.split(),
}
HEADERS = {name: COMMON + fields for name, fields in EXTRA.items()}


def read(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write(path, table, rows):
    path = Path(path)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS[table], extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def case_id(row):
    # Resource/profile/stage measurements share the timing case identity.
    identity = {key: row.get(key, '') for key in COMMON if key != 'case_id'}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]


def number(value, label):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Invalid numeric {label}: {value!r}') from error
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite {label}')
    return value


def validate(tables):
    """Check fixed schemas, identities, sample statistics and O-validation joins."""
    if set(tables) != set(HEADERS):
        raise ValueError('Missing or unexpected report table')
    for table, rows in tables.items():
        for row in rows:
            if set(row) - set(HEADERS[table]):
                raise ValueError(f'Unexpected fields in {table}')
            if row.get('case_id') != case_id(row):
                raise ValueError(f'Case identity mismatch in {table}')
            if row.get('mode') not in ['dense', 'causal']:
                raise ValueError('Execution phase was used as attention mode')
            if row.get('precision') != row.get('input_precision'):
                raise ValueError('Input precision labels disagree')
            for field in ['B', 'H', 'N', 'd']:
                if int(row[field]) <= 0:
                    raise ValueError('Nonpositive shape')
    checks = {}
    for row in tables['correctness']:
        if row['status'] not in ['PASS', 'FAIL', 'UNSUPPORTED']:
            raise ValueError('Unknown correctness status')
        if row['stage'] == 'O':
            if row['case_id'] in checks:
                raise ValueError('Duplicate output correctness record')
            checks[row['case_id']] = row
        if row['status'] == 'PASS':
            if int(row['failures']) or int(row['nonfinite']):
                raise ValueError('PASS contradicts numerical failures')
            if int(row['checked_elements']) <= 0:
                raise ValueError('PASS has no checked elements')
            number(row['max_absolute'], 'max_absolute')
            number(row['max_relative'], 'max_relative')
    groups = {}
    for row in tables['samples']:
        if number(row['forward_ms'], 'forward_ms') <= 0:
            raise ValueError('Nonpositive forward latency')
        groups.setdefault(row['case_id'], []).append(row)
    summaries = {r['case_id']: r for r in tables['summary']}
    if len(summaries) != len(tables['summary']) or set(groups) != set(summaries):
        raise ValueError('Duplicate/missing summary or sample cases')
    for key, row in summaries.items():
        check = checks.get(key)
        if not check or check['status'] != 'PASS' or row['validation_status'] != 'PASS':
            raise ValueError('Timing case lacks passing output validation')
        if check['reference_scope'] != row['reference_scope']:
            raise ValueError('Validation scopes disagree')
        effective = row['input_digest'] if row['precision'] == 'FP32' else row['rounded_input_digest']
        if len(effective) != 16 or any(c not in '0123456789abcdef' for c in effective):
            raise ValueError('Missing/noncanonical effective input fingerprint')
        group = groups[key]
        repeats = int(row['repeats'])
        if repeats != 20 or len(group) != repeats or {int(r['repeat']) for r in group} != set(range(repeats)):
            raise ValueError('Missing or duplicate timing repetitions')
        if int(row['warmup']) != 5:
            raise ValueError('Unexpected warmup count')
        values = sorted(float(r['forward_ms']) for r in group)
        def quantile(p):
            pos = (len(values) - 1) * p; lo = int(pos); hi = min(lo + 1, len(values) - 1)
            return values[lo] + (pos - lo) * (values[hi] - values[lo])
        expected = {'forward_median_ms': statistics.median(values), 'forward_q1_ms': quantile(.25), 'forward_q3_ms': quantile(.75), 'forward_iqr_ms': quantile(.75)-quantile(.25)}
        expected['sequences_per_second'] = 1000 * int(row['B']) / expected['forward_median_ms']
        expected['tokens_per_second'] = expected['sequences_per_second'] * int(row['N'])
        for field, value in expected.items():
            if not math.isclose(number(row[field], field), value, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f'Summary/sample disagreement: {field}')
        if any(row.get(field, '') != '' for field in ['qk_median_ms','softmax_median_ms','pv_median_ms']) and row['part'] == 'part_c':
            raise ValueError('Fused Part C stage timings must not be fabricated')
    for row in tables['profiles']:
        if row['availability'] not in ['available', 'unavailable', 'not_applicable']:
            raise ValueError('Unknown metric availability')
        if row['availability'] == 'available':
            number(row['value'], 'profile value')
        elif row['value'] != '' or not row['reason']:
            raise ValueError('Unavailable counter needs a blank value and reason')
        if row['case_id'] not in summaries:
            raise ValueError('Profile does not identify a measured timing case')
    for table in ['resources', 'work']:
        if any(row['case_id'] not in summaries for row in tables[table]):
            raise ValueError(f'{table} has an orphan timing case')
    for row in tables['memory']:
        if row['case_id'] not in checks or checks[row['case_id']]['status'] != 'PASS':
            raise ValueError('Memory case lacks validation')
        if number(row['explicit_live_device_bytes'], 'allocation') <= 0:
            raise ValueError('Invalid allocation ledger')
    return {'passed': True, 'schema_version': VERSION, 'rows': {name: len(rows) for name, rows in tables.items()}, 'statistics_recomputed': True, 'timing_validation_joins_verified': True, 'effective_input_fingerprints_present': True}
