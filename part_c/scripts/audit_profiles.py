#!/usr/bin/env python3
"""Check capture integrity and deterministic CPU-only reprocessing with retained processors."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();out=a.run
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
prior=out/'extra/before_source_wavefront_correction/reproducibility_audit.json'
prior_path=prior if prior.exists() else out/'reproducibility_audit.json'
prior=json.loads(prior_path.read_text()) if prior_path.exists() else None
plan=json.loads((out/'plan.json').read_text());binary=json.loads((out/'binary.json').read_text())
raw={str(path.relative_to(out)):sha(path) for path in out.glob('cases/*/profile.ncu-rep')}
if prior:assert raw==prior['raw_report_sha256'],'Raw profile changed'
assert all(sha(out/'source_snapshot'/path)==value for path,value in plan['source_hashes'].items()),'Capture source changed'
assert all(json.loads(path.read_text())['binary_sha256']==binary['sha256'] for path in out.glob('cases/*/complete.json'))
def processed():
    files=list(out.glob('*.csv'))+list(out.glob('*.md'))+list((out/'figures').glob('*.png'))
    return {str(path.relative_to(out)):sha(path) for path in files}
before=processed()
processors=[out/'processing/analyze_profiles.py',out/'interpretation/interpret_profiles.py']
expected_hashes=prior.get('processing_script_sha256',{}) if prior else {}
used_processors={}
for processor in processors:
    if not processor.exists():
        candidate=out/'source_snapshot/part_c/scripts'/processor.name
        if candidate.exists() and sha(candidate)==expected_hashes.get(processor.name):processor=candidate
        else:raise ValueError('Original processing script unavailable; preserve historical outputs and regenerate into a new copied bundle rather than applying current code: '+processor.name)
    used_processors[processor.name]=sha(processor)
    subprocess.run([sys.executable,str(processor),str(out)],check=True)
assert processed()==before,'Offline regeneration changed CSV/Markdown/PNG'
result={'validated_captures':len(raw),'snapshot_hashes_verified':True,'same_binary_all_captures':True,'raw_reports_unchanged':True,'offline_csv_markdown_png_regeneration_identical':True,'processed_file_sha256':before,'raw_report_sha256':raw,'processing_script_sha256':used_processors,'prior_audit':'extra/before_source_wavefront_correction/reproducibility_audit.json' if (out/'extra/before_source_wavefront_correction').exists() else None}
(out/'reproducibility_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: capture hashes, source snapshot, common binary and deterministic offline regeneration')
