"""CPU-only guards for exact recorded-execution replay."""
import contextlib,io,json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'part_b/scripts'))
import reproduce_notebook as replay

class ReplayTests(unittest.TestCase):
    def test_both_recordings_pass_pinned_input_source_and_data_checks(self):
        for grid,cases,samples in [('smoke',83,360),('full',165,1800)]:
            pin,folder,manifest,index,tables=replay.load_reference(grid)
            self.assertEqual(len(index['cases']),cases)
            self.assertEqual(len(tables['samples']),samples)
            self.assertTrue(all(row['status']=='PASS' for row in tables['correctness']))
            self.assertFalse(json.loads((folder/'audit.json').read_text())['passed'])

    def test_source_edits_cannot_be_replayed_as_the_reference(self):
        pin=json.loads(replay.PIN_FILE.read_text())
        reference=ROOT/pin['bundles']['smoke']
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp);ref=target/pin['bundles']['smoke'];(ref/'raw/part_b').mkdir(parents=True)
            shutil.copy2(reference/'manifest.json',ref/'manifest.json')
            shutil.copy2(reference/'raw/part_b/index.json',ref/'raw/part_b/index.json')
            (target/'part_b/CMakeLists.txt').write_text('changed kernel linkage\n')
            with patch.object(replay,'ROOT',target),self.assertRaisesRegex(ValueError,'Measured B source changed'):
                replay.load_reference('smoke')

    def test_dry_run_is_cpu_only_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'must_not_exist'
            with patch.object(sys,'argv',['replay','--dry-run','--output',str(out)]),patch('subprocess.check_output',side_effect=AssertionError('GPU access')),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(replay.main(),0)
            self.assertFalse(out.exists())

if __name__=='__main__':unittest.main()
