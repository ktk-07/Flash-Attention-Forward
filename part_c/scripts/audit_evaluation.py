#!/usr/bin/env python3
"""Audit saved study rows and repeat CPU-only analysis; no CUDA execution."""
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);a=p.parse_args();out=a.study
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
commands=[json.loads(line) for line in (out/'commands.jsonl').read_text().splitlines()]
assert all(r['exit_code']==0 for r in commands)
experiments=[r['experiment'] for r in commands]
assert max(i for i,e in enumerate(experiments) if e=='validation')<min(i for i,e in enumerate(experiments) if e=='tuning')
assert max(i for i,e in enumerate(experiments) if e=='tuning')<min(i for i,e in enumerate(experiments) if e=='default_main')
combined=list(csv.DictReader((out/'measurements.csv').open()));original=[]
for command in commands:
    file=out/'cases'/Path(command['argv'][-1]).name
    for row in csv.DictReader(file.open()):row['experiment']=command['experiment'];original.append(row)
assert original==combined,'Combined measurements differ from raw cases'
manifest=json.loads((out/'manifest.json').read_text())
assert all(sha(out/'source_snapshot'/path)==value for path,value in manifest['sources'].items())
analysis=json.loads((out/'analysis_manifest.json').read_text())
assert sha(out/'measurements.csv')==analysis['measurements_sha256']
assert all(sha(out/path)==value for path,value in analysis['case_csv_sha256'].items())
def outputs():
    files=[out/name for name in ['summary.csv','comparison.csv','matched_ablation.csv','REPORT.md','audit.json']]+list((out/'figures').glob('*.png'))+list((out/'figures').glob('*.pdf'))
    if (out/'profiling/audit.json').exists():files.extend([out/'profiling/metrics.csv',out/'profiling/REPORT.md'])
    return {str(path.relative_to(out)):sha(path) for path in sorted(files)}
before=outputs();scripts=Path(__file__).resolve().parent;reruns=[]
analysis_commands=[[sys.executable,str(out/'analysis/evaluate.py'),'--build','/tmp/unused-build','--output',str(out),'--analyze-only']]
if (out/'profiling/audit.json').exists():
    analysis_commands.append([sys.executable,str(out/'profiling/processing/profile_frozen.py'),str(out),'--analyze-only'])
for cmd in analysis_commands:
    result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    reruns.append({'argv':cmd,'exit_code':result.returncode,'output':result.stdout});assert result.returncode==0,result.stdout
assert outputs()==before,'Offline regeneration changed an output'
if (out/'profiling/audit.json').exists():
    profile=json.loads((out/'profiling/audit.json').read_text())
    assert all(sha(out/'profiling'/path)==value for path,value in profile['raw_report_sha256'].items())
result={'passed':True,'case_count':len(commands),'raw_rows':len(combined),'raw_and_combined_rows_identical':True,'validation_before_tuning_before_main':True,'measurement_source_snapshots_verified':True,'offline_csv_markdown_png_pdf_regeneration_identical':True,'processed_sha256':before,'rerun_commands':reruns,'profiling_raw_reports_verified':(out/'profiling/audit.json').exists(),'known_initial_wrapper_error':'execution_notes.json'}
(out/'reproducibility_audit.json').write_text(json.dumps(result,indent=2)+'\n')
folder=out/'analysis';(folder/'audit_evaluation.py').write_bytes(Path(__file__).read_bytes())
print('PASS: raw rows, snapshots, experiment order, profile hashes and identical offline regeneration')
