#!/usr/bin/env python3
"""Check historical capture integrity and deterministic CPU-only reprocessing."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();out=a.run
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
prior=out/'extra/before_source_wavefront_correction/reproducibility_audit.json'
prior=json.loads((prior if prior.exists() else out/'reproducibility_audit.json').read_text())
plan=json.loads((out/'plan.json').read_text());binary=json.loads((out/'binary.json').read_text())
raw={str(path.relative_to(out)):sha(path) for path in out.glob('cases/*/profile.ncu-rep')}
assert raw==prior['raw_report_sha256'],'Raw profile changed'
assert all(sha(out/'source_snapshot'/path)==value for path,value in plan['source_hashes'].items()),'Capture source changed'
assert all(json.loads(path.read_text())['binary_sha256']==binary['sha256'] for path in out.glob('cases/*/complete.json'))
def processed():
    files=list(out.glob('*.csv'))+list(out.glob('*.md'))+list((out/'figures').glob('*.png'))
    return {str(path.relative_to(out)):sha(path) for path in files}
before=processed()
for script in ['analyze_profiles.py','interpret_profiles.py']:
    subprocess.run([sys.executable,str(Path(__file__).parent/script),str(out)],check=True)
assert processed()==before,'Offline regeneration changed CSV/Markdown/PNG'
result={'validated_captures':len(raw),'snapshot_hashes_verified':True,'same_binary_all_captures':True,'raw_reports_unchanged':True,'offline_csv_markdown_png_regeneration_identical':True,'processed_file_sha256':before,'raw_report_sha256':raw,'processing_script_sha256':{name:sha(Path(__file__).parent/name) for name in ['analyze_profiles.py','interpret_profiles.py']},'prior_audit':'extra/before_source_wavefront_correction/reproducibility_audit.json' if (out/'extra/before_source_wavefront_correction').exists() else None}
(out/'reproducibility_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: capture hashes, source snapshot, common binary and deterministic offline regeneration')
