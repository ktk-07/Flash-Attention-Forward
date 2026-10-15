"""Check correctness evidence coverage and recorded runs"""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('analyzer', ROOT / 'benchmarks/analyze_results.py')
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)
RUNS = ('2026-10-14T154423Z-0f768e', '2026-10-14T154428Z-18c60f',
        '2026-10-14T154516Z-a73070', '2026-10-14T154841Z-f210f0')


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'results/gpu-benchmarks' / RUNS[0] / 'benchmark'
        self.rows = analyzer.read_csv(self.directory / 'summary.csv')
        self.checks = analyzer.read_csv(self.directory / 'correctness.csv')

    def test_recorded_runs(self):
        for run, count in zip(RUNS, (8, 16, 60, 16)):
            with self.subTest(run=run):
                rows, _, _ = analyzer.audit(ROOT / 'results/gpu-benchmarks' / run / 'benchmark')
                self.assertEqual(len(rows), count)

    def test_missing_report(self):
        with self.assertRaisesRegex(ValueError, 'correctness reports'):
            analyzer.audit_correctness(self.rows, self.checks[1:])

    def test_duplicate_report(self):
        with self.assertRaisesRegex(ValueError, 'correctness reports'):
            analyzer.audit_correctness(self.rows, self.checks + self.checks[:1])

    def test_wrong_gate_shape(self):
        self.checks[0]['N'] = '18'
        with self.assertRaisesRegex(ValueError, 'correctness reports'):
            analyzer.audit_correctness(self.rows, self.checks)

    def test_wrong_gate_fingerprint(self):
        self.checks[0]['input_fingerprint'] = '0'
        with self.assertRaisesRegex(ValueError, 'inputs differ'):
            analyzer.audit_correctness(self.rows, self.checks)

    def test_timed_fingerprint_mismatch(self):
        checks = copy.deepcopy(self.checks)
        for check in checks:
            if check['scope'] == 'after_timing':
                check['input_fingerprint'] = '0'
        with self.assertRaisesRegex(ValueError, 'inputs differ'):
            analyzer.audit_correctness(self.rows, checks)

    def test_wrong_validation_scope(self):
        self.rows[0]['validation'] = 'exact_CPU_SPO' if int(self.rows[0]['N']) > 257 else 'finite_O'
        with self.assertRaisesRegex(ValueError, 'validation scope'):
            analyzer.audit_correctness(self.rows, self.checks)


if __name__ == '__main__':
    unittest.main()
