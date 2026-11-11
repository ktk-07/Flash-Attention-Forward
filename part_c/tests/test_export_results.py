"""CPU-only schema/export regression tests; no CUDA execution or timing fixtures invented."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
NATIVE=ROOT/'part_c/results/bounded_evaluation'
spec=importlib.util.spec_from_file_location('part_c_export',ROOT/'part_c/scripts/export_results.py')
exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
schema=exporter.schema


@unittest.skipUnless((NATIVE/'measurements.csv').exists(),'saved native study fixture not supplied')
class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables,cls.configs,cls.selection,cls.audit=exporter.make_tables(NATIVE,'test_run')

    def test_part_a_headers_and_counts(self):
        self.assertEqual(schema.BASE,'experiment precision B H N d mode variant tile softmax_threads'.split())
        self.assertEqual(self.audit['rows']['summary'],318)
        self.assertEqual(self.audit['rows']['samples'],6360)
        self.assertEqual(self.audit['rows']['correctness'],636)
        self.assertEqual(self.audit['rows']['memory'],30)
        self.assertEqual(self.audit['profile_captures'],12)
        self.assertEqual(sum(r['experiment']=='proposal' for r in self.tables['summary']),60)

    def test_ptx_fa1_uses_frozen_configuration_only(self):
        row=next(r for r in schema.read(NATIVE/'summary.csv') if r['kernel']=='ptx' and r['experiment']=='default_main')
        row=dict(row,kernel='fa1_ptx',config='0')
        plan=json.loads((NATIVE/'plan.json').read_text());plan.update(fa1_baselines=['fa1','fa1_ptx'],fa1_workspace_floats_per_query=2)
        selection=copy.deepcopy(self.selection);selection['variants']['fa1_ptx']={row['d']:{'config':1}}
        default=exporter.common(row,'same_run',plan,selection)
        self.assertEqual(default['experiment'],'proposal_default')
        frozen=exporter.common(dict(row,experiment='frozen_main',config='1'),'same_run',plan,selection)
        self.assertEqual(frozen['experiment'],'proposal')
        self.assertIn('kv_outer_ptx_v1:c1',frozen['config'])
        self.assertNotEqual(default['case_id'],frozen['case_id'])

    def test_exact_native_latencies_and_fused_blanks(self):
        lookup={(r['kernel'],r['config'],r['B'],r['H'],r['N'],r['d'],r['pattern'],r['median_ms']) for r in schema.read(NATIVE/'summary.csv') if r['mode']=='benchmark'}
        for row in self.tables['summary']:
            index=self.configs[row['config']]['native_config']
            self.assertIn((row['variant'],str(index),row['B'],row['H'],row['N'],row['d'],row['pattern'],row['forward_median_ms']),lookup)
            self.assertEqual(row['mode'],'causal')
            self.assertEqual(row['precision'],row['input_precision'])
            self.assertEqual(row['tile'],'')
            for field in ['qk_median_ms','softmax_median_ms','pv_median_ms','setup_host_ms','speedup_vs_naive']:
                self.assertEqual(row.get(field,''),'')

    def test_every_raw_ncu_metric_and_mio_preserved(self):
        expected=0
        for path in (NATIVE/'profiling/cases').glob('*/raw.csv'):
            with path.open() as stream:expected+=len(next(csv.reader(stream)))-11
        raw=[r for r in self.tables['profiles'] if not r['metric'].startswith('derived::')]
        self.assertEqual(len(raw),expected)
        self.assertTrue(any('mio_throttle' in r['metric'] for r in raw))
        derived=[r for r in self.tables['profiles'] if r['metric']=='derived::mio_throttle_pct']
        self.assertEqual(len(derived),12)
        self.assertTrue(all(r['unit']=='%' for r in derived))

    def test_fa1_revision_has_distinct_configuration_identity(self):
        plan=json.loads((NATIVE/'plan.json').read_text())
        row=next(r for r in schema.read(NATIVE/'summary.csv') if r['kernel']=='fa1')
        old=exporter.common(row,'same_run',plan,self.selection)
        current=exporter.common(row,'same_run',{**plan,'fa1_workspace_floats_per_query':2},self.selection)
        self.assertNotEqual(old['config'],current['config'])
        self.assertNotEqual(old['case_id'],current['case_id'])
        self.assertIn('kv_outer_shared_v2',current['config'])

    def test_duplicate_repetitions_rejected(self):
        tables=copy.deepcopy(self.tables)
        tables['samples'][1]['repeat']=tables['samples'][0]['repeat']
        with self.assertRaisesRegex(ValueError,'timing repetitions'):schema.validate(tables)

    def test_wrong_summary_rejected(self):
        tables=copy.deepcopy(self.tables)
        tables['summary'][0]['forward_median_ms']='9999'
        with self.assertRaisesRegex(ValueError,'Summary/sample disagreement'):schema.validate(tables)

    def test_missing_correctness_rejected(self):
        tables=copy.deepcopy(self.tables);key=tables['summary'][0]['case_id']
        tables['correctness']=[r for r in tables['correctness'] if r['case_id']!=key]
        with self.assertRaisesRegex(ValueError,'passing output validation'):schema.validate(tables)

    def test_unknown_profile_not_zero(self):
        rows=[r for r in self.tables['profiles'] if r['availability']=='unavailable']
        self.assertTrue(rows)
        self.assertTrue(all(r['value']=='' and r['reason'] for r in rows))

    def test_schema_rejects_orphan_profile(self):
        tables=copy.deepcopy(self.tables)
        tables['profiles'][0]['config']='wrong_config'
        tables['profiles'][0]['case_id']=schema.case_id(tables['profiles'][0])
        with self.assertRaisesRegex(ValueError,'measured timing case'):schema.validate(tables)

    def test_wrapper_in_place_and_summary_tamper_detection(self):
        with tempfile.TemporaryDirectory(prefix='part-c-inplace-test-') as folder:
            out=Path(folder)/'report';native=out/'raw/part_c/study'
            native.parent.mkdir(parents=True)
            shutil.copytree(NATIVE,native,ignore=shutil.ignore_patterns('profiling','*.ncu-rep','*.log'))
            (out/'logs').mkdir();(out/'commands.jsonl').write_text('{}\n')
            before=exporter.inventory(native)
            result=exporter.publish(native,out,in_place=True)
            self.assertTrue(result['passed']);self.assertEqual(result['profile_captures'],0)
            self.assertEqual(exporter.inventory(native),before)
            rows=schema.read(native/'summary.csv');fields=list(rows[0]);rows[0]['max_absolute']='999'
            with (native/'summary.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'metadata differs from raw'):
                exporter.make_tables(native,'tamper_check')

    def test_self_contained_export_and_repeatable_cpu_analysis(self):
        with tempfile.TemporaryDirectory(prefix='part-c-export-test-') as folder:
            root=Path(folder);native=root/'native';out=root/'public';history=root/'historical_fixture'
            # Binary reports are deliberately omitted to exercise the committed
            # CSV-only workflow; their expected checksums must remain explicit.
            shutil.copytree(NATIVE,native,ignore=shutil.ignore_patterns('*.ncu-rep','*.log'))
            history.mkdir();(history/'source_notes.txt').write_text('Supplementary fixture, not current timing evidence.\n')
            before=exporter.inventory(native)
            audit=exporter.publish(native,out,[history])
            self.assertEqual(audit['verified_binary_profile_reports'],0)
            self.assertEqual(len(audit['missing_binary_profile_reports']),12)
            self.assertEqual(exporter.inventory(native),before)
            self.assertEqual(exporter.inventory(out/'raw/part_c/study'),before)
            self.assertEqual((out/'raw/part_c/historical_profiles/historical_fixture/source_notes.txt').read_text(),(history/'source_notes.txt').read_text())
            for name,header in schema.HEADERS.items():
                with (out/f'{name}.csv').open() as stream:self.assertEqual(next(csv.reader(stream)),header)
            tables={name:schema.read(out/f'{name}.csv') for name in schema.HEADERS}
            self.assertTrue(schema.validate(tables)['passed'])
            def report_hashes():
                files=list(out.glob('*.csv'))+list((out/'plots').rglob('*.png'))+list((out/'plots').rglob('*.pdf'))+list((out/'supplementary').rglob('*.png'))+list((out/'supplementary').rglob('*.pdf'))
                return {str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            saved=report_hashes()
            r=subprocess.run([sys.executable,str(out/'analysis/export_results.py'),'--analyze-only',str(out)],text=True,capture_output=True)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            self.assertEqual(saved,report_hashes())
            self.assertEqual(exporter.inventory(native),before)
            with self.assertRaisesRegex(ValueError,'fresh output'):exporter.publish(native,out)


if __name__=='__main__':unittest.main()
