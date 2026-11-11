"""CPU-only packaging fixtures; synthetic data/archives are never report evidence."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_submission as submission
import results_schema as schema
from run_all import aggregate
from test_integration import fixture
from workflow import dump


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.bundle = cls.base / 'synthetic_smoke'
        for part in ['part_a', 'part_b', 'part_c']:
            folder = cls.bundle / 'parts' / part
            folder.mkdir(parents=True)
            data = fixture('part_c' if part == 'part_b' else part)
            if part == 'part_b':
                for name, rows in data.items():
                    unique = {}
                    for row in rows:
                        if row['precision'] != 'FP32':continue
                        row.update(part=part, mode='dense', variant='selected', run_id=part, config='selected')
                        row['case_id'] = schema.case_id(row)
                        unique[json.dumps(row, sort_keys=True)] = row
                    data[name] = list(unique.values())
            for name, rows in data.items():schema.write(folder / (name + '.csv'), name, rows)
            dump(folder / 'manifest.json', {'schema_version': 1, 'shared_cpu_reference_linked': True, 'device': {'uuid': 'GPU-TEST-FIXTURE'}})
            dump(folder / 'audit.json', {'passed': True})
            if part == 'part_c':
                (folder / 'configs').mkdir()
                dump(folder / 'configs/part_c.json', {'frozen_selection': {'families': {'cuda_core': {'64': 'skip'}, 'tensor_core': {'64': 'wmma'}}}})
        aggregate(cls.bundle, ['part_a','part_b','part_c'], 'smoke')
        dump(cls.bundle / 'run_status.json', {'status': 'completed_requested_parts', 'parts': ['part_a','part_b','part_c'], 'grid': 'smoke'})
        cls.source = cls.base / 'source'
        cls.source.mkdir()
        (cls.source / 'README.md').write_text('Synthetic source fixture\n')
        for name in ['scripts', 'tests', 'part_a', 'part_b', 'part_c']:(cls.source / name).mkdir()
        (cls.source / 'part_c/src').mkdir()
        (cls.source / 'part_c/src/attention.cu').write_text('// synthetic source fixture\n')
        (cls.source / 'extra').mkdir()
        (cls.source / 'extra/main.tex').write_text('PRIVATE DRAFT MUST NEVER BE PACKAGED')
        (cls.source / 'part_c/build').mkdir()
        (cls.source / 'part_c/build/test.o').write_bytes(b'object fixture')
        (cls.source / 'part_c/test_binary').write_bytes(b'\x7fELFbinary fixture')
        snapshot=cls.source/'part_c/results/historical/source_snapshot/part_c/src/extra'
        snapshot.mkdir(parents=True)
        (snapshot/'README.md').write_text('Required historical source hash fixture')
        dump(cls.source / 'part_b/EXECUTION_REFERENCE.json', {'bundles': {'smoke': 'part_b/results/missing_fixture'}})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_source_filter_excludes_private_and_compiled_files(self):
        names = {p.relative_to(self.source).as_posix() for p in submission.source_files(self.source)}
        self.assertIn('part_c/src/attention.cu', names)
        self.assertNotIn('extra/main.tex', names)
        self.assertNotIn('part_c/build/test.o', names)
        self.assertNotIn('part_c/test_binary', names)
        self.assertIn('part_c/results/historical/source_snapshot/part_c/src/extra/README.md', names)
        self.assertEqual(submission.missing_references(self.source), ['part_b/results/missing_fixture'])

    def test_smoke_is_explicit_and_cannot_be_final_submission(self):
        report = submission.check_run(self.bundle, allow_smoke=True)
        self.assertTrue(report['passed'])
        self.assertFalse(report['full_report_evidence'])
        with self.assertRaisesRegex(ValueError, 'full-grid'):
            submission.check_run(self.bundle)

    def test_changed_child_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'changed'
            shutil.copytree(self.bundle, run)
            with (run / 'parts/part_b/summary.csv').open('a') as stream:stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'Child table changed'):
                submission.check_run(run, allow_smoke=True)

    def test_split_archive_retains_evidence_and_verifies_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            run = temp / 'smoke'
            shutil.copytree(self.bundle, run)
            profile = run / 'parts/part_c/raw/synthetic.ncu-rep'
            profile.parent.mkdir(parents=True)
            profile.write_bytes(b'Synthetic archive-routing fixture, not a real Nsight report')
            archive, profiles = temp / 'source.zip', temp / 'profiles.zip'
            result = submission.pack(run, archive, 'synthetic-workspace', profiles_output=profiles, validation_only=True, source_root=self.source)
            self.assertTrue(result['validation_only'])
            self.assertTrue(submission.verify_archive(archive, profiles)['passed'])
            with self.assertRaisesRegex(ValueError, 'companion'):
                submission.verify_archive(archive)
            with zipfile.ZipFile(archive) as src:
                names = src.namelist()
                self.assertFalse(any('main.tex' in name or name.startswith(submission.PREFIX+'/extra/') or name.startswith(submission.PREFIX+'/part_c/src/extra/') for name in names))
                meta = json.loads(src.read(submission.PREFIX + '/submission_manifest.json'))
                self.assertFalse(meta['source_version_verified_against_git'])
                self.assertTrue(meta['missing_pinned_reference_bundles'])
                bad = temp / 'tampered.zip'
                with zipfile.ZipFile(bad, 'w') as dst:
                    for name in names:
                        content = b'changed' if name.endswith('/README.md') else src.read(name)
                        dst.writestr(name, content)
            with self.assertRaisesRegex(ValueError, 'Archive content changed'):
                submission.verify_archive(bad, profiles)
            with self.assertRaisesRegex(ValueError, 'must not already exist'):
                submission.pack(run, archive, 'fixture', validation_only=True, source_root=self.source)

    def test_output_inside_result_bundle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside the immutable'):
            submission.pack(self.bundle, self.bundle / 'bad.zip', 'fixture', validation_only=True, source_root=self.source)


if __name__ == '__main__':
    unittest.main()
