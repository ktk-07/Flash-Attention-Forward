#!/usr/bin/env python3
"""CPU-only source audit, experiment validation and checksummed submission archives.

Never runs Git, builds CUDA, executes GPU kernels, publishes files or edits results.
Final archives require complete full-grid evidence; --validation-only permits smoke
bundles and marks the archive as a packaging test. Current private extra/ directories are excluded; pinned historical snapshots remain intact.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from urllib.parse import urlparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import results_schema as schema

PARTS = ['part_a', 'part_b', 'part_c']
CAPTURES = {'part_a': 108, 'part_b': 6, 'part_c': 24}
SOURCE_ROOTS = ['scripts', 'tests', *PARTS]
SOURCE_TOP = ['README.md', 'README.txt', 'run.sh', 'run_all.sh', '.gitignore']
EXCLUDED_DIRS = {'.git', '.venv', '__pycache__', 'extra', 'build', 'Build',
                 'CMakeFiles', 'Testing', '.cache', 'vcpkg_installed'}
EXCLUDED_FILES = {'CMakeCache.txt', 'cmake_install.cmake', 'Makefile',
                  'install_manifest.txt', 'compile_commands.json'}
BINARY_SUFFIXES = {'.o', '.obj', '.a', '.lib', '.so', '.dll', '.exe', '.out',
                   '.pdb', '.pyc', '.pyo', '.d', '.tmp', '.swp'}
PREFIX = 'Flash-Attention-CUDA'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe evidence/archive path: ' + name)
    return path


def source_files(root=ROOT):
    """Retain source and historical CSV fixtures, exclude scratch/build artifacts.

    Historical binary Nsight reports are optional source-fixture attachments.
    The SELECTED experiment's complete evidence, including these binaries, is
    added separately by pack() and is never filtered by this function.
    """
    root = Path(root)
    files = [root / name for name in SOURCE_TOP if (root / name).is_file()]
    for name in SOURCE_ROOTS:
        for path in sorted((root / name).rglob('*')):
            rel = path.relative_to(root)
            snapshot = 'results' in rel.parts and 'source_snapshot' in rel.parts
            # A recorded snapshot may legitimately contain a historical path
            # named extra/. Dropping it breaks the capture's source hashes.
            if '.git' in rel.parts:continue
            if not snapshot and any(p in EXCLUDED_DIRS or p.startswith('build-') for p in rel.parts):
                continue
            if not path.is_file() or path.is_symlink():
                continue
            if not snapshot and (path.name in EXCLUDED_FILES or path.suffix in BINARY_SUFFIXES):
                continue
            if path.name.endswith(('.ncu-rep', '.ncu-repz')):
                continue
            with path.open('rb') as stream:
                if stream.read(4) == b'\x7fELF':continue
            files.append(path)
    return sorted(set(files))


def missing_references(root=ROOT):
    pin = Path(root) / 'part_b/EXECUTION_REFERENCE.json'
    if not pin.exists():
        return ['part_b/EXECUTION_REFERENCE.json']
    return [name for name in load(pin)['bundles'].values()
            if not (Path(root) / relative(name) / 'manifest.json').is_file()]


def tables(folder):
    data = {}
    for name, header in schema.HEADERS.items():
        path = folder / (name + '.csv')
        import csv
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != header:
                raise ValueError('Wrong schema header: ' + str(path))
            data[name] = list(reader)
    schema.validate(data)
    return data


def check_run(folder, allow_smoke=False):
    folder = Path(folder).resolve()
    manifest = load(folder / 'manifest.json')
    audit = load(folder / 'aggregation_audit.json')
    status = load(folder / 'run_status.json')
    if set(manifest['requested_parts']) != set(PARTS) or not audit.get('all_three_parts_present'):
        raise ValueError('Submission evidence requires all three parts')
    if not audit.get('passed') or status.get('status') != 'completed_requested_parts':
        raise ValueError('The combined run did not complete and pass its audit')
    full = manifest.get('grid') == 'full' and audit.get('final_report_grid') is True
    if not full and not allow_smoke:
        raise ValueError('Smoke evidence is not a final full-grid submission; use --allow-smoke only for validation')
    data = tables(folder)
    if any(r['status'] == 'FAIL' for r in data['correctness']):
        raise ValueError('Failed numerical checks in the combined results')
    expected_tables = {f'parts/{part}/{name}.csv' for part in PARTS for name in schema.HEADERS}
    if set(manifest['source_tables']) != expected_tables:
        raise ValueError('Incomplete child-table provenance')
    for name, expected in manifest['source_tables'].items():
        if sha(folder / relative(name)) != expected:
            raise ValueError('Child table changed: ' + name)
    root_ids = {r['run_id'] for r in data['summary']}
    if len(root_ids) != 1:raise ValueError('Expected one combined run identity')
    root_id = next(iter(root_ids))
    reconstructed = {name: [] for name in schema.HEADERS}
    devices = set()
    captures = {}
    for part in PARTS:
        child = folder / 'parts' / part
        child_data = tables(child)
        child_audit = load(child / 'audit.json')
        child_manifest = load(child / 'manifest.json')
        if not child_audit.get('passed'):
            raise ValueError('Child audit failed: ' + part)
        if not child_manifest.get('shared_cpu_reference_linked', child_audit.get('shared_cpu_reference_linked')):
            raise ValueError('Shared oracle linkage missing: ' + part)
        device = load(child / 'device.json') if (child / 'device.json').exists() else child_manifest.get('device', {})
        if not device.get('uuid'):
            raise ValueError('Physical GPU identity missing: ' + part)
        devices.add(device['uuid'])
        child_status = load(child / 'run_status.json') if (child / 'run_status.json').exists() else {}
        requested = child_manifest.get('profile_requested', child_status.get('profiling_requested', False))
        captures[part] = child_audit.get('profile_captures', 0)
        if full and (not requested or captures[part] < CAPTURES[part]):
            raise ValueError('Missing requested full-run profiling evidence: ' + part)
        if full and not any(r['availability'] == 'available' for r in child_data['profiles']):
            raise ValueError('No available measured profiler counters: ' + part)
        for name, rows in child_data.items():
            for old in rows:
                row = dict(old, run_id=root_id)
                row['case_id'] = schema.case_id(row)
                if name == 'profiles':row['source_report'] = (Path('parts') / part / relative(row['source_report'])).as_posix()
                reconstructed[name].append(row)
        for row in child_data['profiles']:
            if not (child / relative(row['source_report'])).is_file():
                raise ValueError('Missing profiler evidence: ' + row['source_report'])
    if reconstructed != data:
        raise ValueError('Combined tables differ from the recorded child results')
    if len(devices) != 1 or next(iter(devices)) != audit.get('physical_gpu_uuid'):
        raise ValueError('GPU identities do not match')
    if not audit.get('shared_input_fingerprints_verified'):
        raise ValueError('Shared input fingerprints were not verified')
    ns, ds, b, h = ([256,512,1024,2048,4096],[32,64,128],4,8) if full else ([33,65],[64],1,1)
    main = [r for r in data['summary'] if r['experiment'] == 'proposal']
    for part in PARTS:
        modes = ['dense','causal'] if part == 'part_a' else ['dense'] if part == 'part_b' else ['causal']
        for precision in (['FP32'] if part == 'part_b' else ['FP32','FP16']):
            wanted = {(str(b),str(h),str(n),str(d),mode) for n in ns for d in ds for mode in modes}
            found = {tuple(r[k] for k in ['B','H','N','d','mode']) for r in main if r['part'] == part and r['precision'] == precision}
            if found != wanted:
                raise ValueError('Incomplete main workload: ' + part + '/' + precision)
    if full and not any(r['variant'] == 'fa1_ptx' for r in main):
        raise ValueError('Final evidence lacks the new PTX FA1 baseline')
    figure_index = load(folder / 'figure_index.json')
    if figure_index.get('inputs') != {name+'.csv': sha(folder / (name+'.csv')) for name in schema.HEADERS}:
        raise ValueError('Plots do not match the current canonical tables')
    figures = figure_index['figures']
    for item in figures:
        for ext in item.get('formats', ['png', 'pdf']):
            if not (folder / relative(item['figure'] + '.' + ext)).is_file():
                raise ValueError('Missing indexed figure: ' + item['figure'])
    for stem in ['cross_part', 'materialised_fused_memory', 'dense_causal_work']:
        for ext in ['png', 'pdf']:
            if not (folder / 'plots/combined' / (stem + '.' + ext)).is_file():
                raise ValueError('Missing combined comparison: ' + stem)
    return {'passed': True, 'full_report_evidence': full, 'scope': 'full' if full else 'smoke_validation_only',
            'parts': PARTS, 'rows': {k:len(v) for k,v in data.items()}, 'profile_captures': captures,
            'physical_gpu_uuid': next(iter(devices)), 'indexed_figures': len(figures)}


def source_check(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a fresh source-audit directory')
    source = output / 'source'
    source.mkdir(parents=True)
    inventory = {}
    for path in source_files():
        name = path.relative_to(ROOT)
        dest = source / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        inventory[name.as_posix()] = sha(dest)
    dump(output / 'source_manifest.json', inventory)
    checks = [
        ('integration', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']),
        ('part_c', [sys.executable, '-m', 'unittest', 'discover', '-s', 'part_c/tests', '-p', 'test_*.py', '-v']),
        ('part_b', [sys.executable, '-m', 'unittest', 'discover', '-s', 'part_b/tests', '-p', 'test_*.py', '-v']),
        ('pinned_b_full', [sys.executable, 'part_b/scripts/reproduce_notebook.py', '--grid', 'full', '--check-only']),
        ('pipeline_plan', ['bash', 'run.sh', '--dry-run', '--arch', '86', '--output', 'results/source_audit_preview']),
    ]
    results = []
    for name, argv in checks:
        with (output / (name + '.txt')).open('w') as stream:
            result = subprocess.run(argv, cwd=source, stdout=stream, stderr=subprocess.STDOUT)
        results.append({'check': name, 'argv': argv, 'exit_code': result.returncode, 'log': name + '.txt'})
    # File contents in the snapshot must survive the read-only checks.
    stable = all(sha(source / name) == digest for name,digest in inventory.items())
    report = {'passed': stable and all(r['exit_code'] == 0 for r in results),
              'source_copy_unchanged': stable, 'source_files': len(inventory), 'checks': results,
              'missing_pinned_reference_bundles': missing_references(source),
              'scope': 'Filesystem source snapshot; not a Git clean-checkout claim. No GPU execution.'}
    dump(output / 'audit.json', report)
    return report


def pack(folder, output, source_version, report=None, video_url=None, repository_url=None,
         profiles_output=None, validation_only=False, source_root=ROOT):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    profiles_output = Path(profiles_output).resolve() if profiles_output else None
    if output.exists() or output.with_name(output.name+'.sha256').exists() or (profiles_output and (profiles_output.exists() or profiles_output.with_name(profiles_output.name+'.sha256').exists())):
        raise ValueError('Archive outputs must not already exist')
    if output.suffix != '.zip' or (profiles_output and profiles_output.suffix != '.zip'):
        raise ValueError('Archive outputs must end in .zip')
    if output.is_relative_to(folder) or (profiles_output and profiles_output.is_relative_to(folder)):
        raise ValueError('Archive outputs must be outside the immutable result bundle')
    if profiles_output == output:
        raise ValueError('Source and profiler archives must have different paths')
    evidence = check_run(folder, allow_smoke=validation_only)
    missing = missing_references(source_root)
    if not validation_only:
        if missing:
            raise ValueError('Pinned reference bundles are missing; resolve with Yichun before final packaging: ' + ', '.join(missing))
        if not re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}', source_version):
            raise ValueError('Supply the complete reviewed commit ID; no Git command is run by this helper')
        if not report or not video_url or not repository_url:
            raise ValueError('Final packaging requires --report, --video-url and --repository-url')
    for label, url in [('video', video_url), ('repository', repository_url)]:
        if url and (urlparse(url).scheme not in ['https', 'http'] or not urlparse(url).netloc):
            raise ValueError('Invalid ' + label + ' URL')
    if report:
        report = Path(report).resolve()
        with report.open('rb') as stream:
            if report.suffix.lower() != '.pdf' or stream.read(5) != b'%PDF-':
                raise ValueError('The final report must be a PDF')
    entries = []
    for path in source_files(Path(source_root)):
        entries.append((path, PREFIX + '/' + path.relative_to(source_root).as_posix(), 'source'))
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            raise ValueError('Result evidence must use regular files: ' + str(path))
        if path.is_file():
            name = PREFIX + '/results/' + folder.name + '/' + path.relative_to(folder).as_posix()
            target = 'profiles' if profiles_output and path.name.endswith(('.ncu-rep', '.ncu-repz')) else 'source'
            entries.append((path, name, target))
    if report:
        entries.append((report, PREFIX + '/report.pdf', 'source'))
    names = [name for _,name,_ in entries]
    if len(names) != len(set(names)):
        raise ValueError('Archive members collide')
    meta = {'format_version': 1, 'validation_only': validation_only, 'source_version_supplied_by_operator': source_version,
            'source_version_verified_against_git': False, 'run_directory': 'results/' + folder.name,
            'evidence_check': evidence, 'missing_pinned_reference_bundles': missing,
            'report': 'report.pdf' if report else None, 'video_url': video_url, 'repository_url': repository_url,
            'profile_archive': profiles_output.name if profiles_output else None,
            'manual_review_required': ['Report contains actual results, limitations, citations, member contributions and AI disclosure.',
                                       'Video is 10–15 minutes and its link works without the team login.',
                                       'Public repository and supplied commit identify the reviewed source.',
                                       'Final source-audit tests pass after the missing replay bundle is resolved.'],
            'files': []}
    for path,name,target in entries:
        relative(name)
        meta['files'].append({'path': name, 'archive': target, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    output.parent.mkdir(parents=True, exist_ok=True)
    if profiles_output:
        profiles_output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(profiles_output, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path,name,target in entries:
                if target == 'profiles':archive.write(path,name)
        meta['profile_archive_sha256'] = sha(profiles_output)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path,name,target in entries:
            if target == 'source':archive.write(path,name)
        archive.writestr(PREFIX + '/submission_manifest.json', json.dumps(meta,indent=2)+'\n')
        archive.writestr(PREFIX + '/SHA256SUMS', ''.join(r['sha256']+'  '+r['path'].removeprefix(PREFIX+'/')+'\n' for r in meta['files']))
    verify_archive(output, profiles_output)
    for path in [output, profiles_output]:
        if path:path.with_name(path.name+'.sha256').write_text(sha(path)+'  '+path.name+'\n')
    return {'passed': True, 'validation_only': validation_only, 'archive': str(output),
            'profile_archive': str(profiles_output) if profiles_output else None,
            'files': len(entries), 'missing_pinned_reference_bundles': missing}


def verify_archive(archive, profiles=None):
    archive = Path(archive)
    sidecar = archive.with_name(archive.name+'.sha256')
    if sidecar.exists() and sidecar.read_text().split()[0] != sha(archive):
        raise ValueError('Source archive checksum mismatch')
    with zipfile.ZipFile(archive) as source:
        meta = json.loads(source.read(PREFIX + '/submission_manifest.json'))
        other = zipfile.ZipFile(profiles) if profiles else None
        try:
            if meta.get('profile_archive'):
                if not other:raise ValueError('Supply the companion profiler archive with --profiles')
                if sha(profiles) != meta['profile_archive_sha256']:raise ValueError('Profiler archive checksum mismatch')
            elif other:
                raise ValueError('This package does not declare a companion profiler archive')
            for row in meta['files']:
                relative(row['path'])
                selected = other if row['archive'] == 'profiles' else source
                digest = hashlib.sha256()
                size = 0
                with selected.open(row['path']) as stream:
                    for chunk in iter(lambda: stream.read(1024*1024), b''):
                        digest.update(chunk);size += len(chunk)
                if digest.hexdigest() != row['sha256'] or size != row['bytes']:
                    raise ValueError('Archive content changed: ' + row['path'])
            if len(source.namelist()) != len(set(source.namelist())) or (other and len(other.namelist()) != len(set(other.namelist()))):
                raise ValueError('Duplicate archive members')
            expected_source = {r['path'] for r in meta['files'] if r['archive'] == 'source'} | {PREFIX+'/submission_manifest.json',PREFIX+'/SHA256SUMS'}
            if set(source.namelist()) != expected_source:raise ValueError('Unexpected/missing source archive members')
            if other and set(other.namelist()) != {r['path'] for r in meta['files'] if r['archive'] == 'profiles'}:
                raise ValueError('Unexpected/missing profiler archive members')
        finally:
            if other:other.close()
    return {'passed': True, 'validation_only': meta['validation_only'], 'files': len(meta['files'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('source-check', help='Audit a clean filesystem source copy, CPU only')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('check', help='Read-only validation of the completed combined bundle')
    p.add_argument('--results', type=Path, required=True);p.add_argument('--allow-smoke', action='store_true')
    p = sub.add_parser('pack', help='Create a self-contained ZIP, optionally splitting raw profiler binaries')
    p.add_argument('--results', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-version', required=True);p.add_argument('--report', type=Path)
    p.add_argument('--video-url');p.add_argument('--repository-url');p.add_argument('--profiles-output', type=Path)
    p.add_argument('--validation-only', action='store_true')
    p = sub.add_parser('verify-archive');p.add_argument('archive', type=Path);p.add_argument('--profiles', type=Path)
    args = parser.parse_args()
    if args.command == 'source-check':result = source_check(args.output)
    elif args.command == 'check':result = check_run(args.results, args.allow_smoke)
    elif args.command == 'verify-archive':result = verify_archive(args.archive,args.profiles)
    else:result = pack(args.results,args.output,args.source_version,args.report,args.video_url,args.repository_url,args.profiles_output,args.validation_only)
    print(json.dumps(result,indent=2))
    return 0 if result.get('passed') else 1


if __name__ == '__main__':
    try:sys.exit(main())
    except (OSError,ValueError,KeyError,zipfile.BadZipFile) as error:
        print('ERROR: '+str(error),file=sys.stderr);sys.exit(1)
