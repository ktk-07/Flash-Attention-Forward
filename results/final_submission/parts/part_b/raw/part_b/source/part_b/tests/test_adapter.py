"""CPU-only contract checks. Synthetic fixtures never become benchmark evidence."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'part_b/scripts'))
sys.path.insert(0,str(ROOT/'scripts'))
from export_results import export
from run_standard import main
from workflow import dump,sha
import results_schema as schema

class AdapterTests(unittest.TestCase):
    def test_dry_run_never_calls_gpu_or_creates_output(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'absent'
            with patch.object(sys,'argv',['b','--dry-run','--output',str(target)]),patch('subprocess.check_output',side_effect=AssertionError('GPU access')),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(),0)
            self.assertFalse(target.exists())

    def test_offline_reextract_is_stable_and_rejects_modified_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'raw/part_b').mkdir(parents=True);(out/'configs').mkdir()
            r={'B':1,'H':1,'N':33,'d':64,'variant':'tiled','warps':4,'key_tile':32,'pattern':'random',
               'input_digest':'0123456789abcdef','status':'PASS','failures':0,'nonfinite':0,'max_absolute':0.,'max_relative':0.,
               'reference_scope':'full','checked_query_rows':33,'checked_elements':2112,'samples_ms':[1.+i*.01 for i in range(20)],
               'explicit_live_device_bytes':33792,'auxiliary_global_bytes':0,'observed_device_delta_bytes':0,'setup_host_ms':1,
               'blocks':9,'threads':128,'registers_per_thread':32,'shared_bytes':16896,'local_bytes_per_thread':0,'active_blocks_per_sm':4,'theoretical_occupancy_percent':33.3}
            relative='raw/part_b/case.json';dump(out/relative,r)
            dump(out/'configs/part_b.json',{'selected':{'64':{'variant':'tiled','warps':4,'key_tile':32}}})
            dump(out/'manifest.json',{'run_id':'temporary_fixture','grid':'smoke','profile_requested':False,'shared_cpu_reference_linked':True})
            dump(out/'raw/part_b/index.json',{'cases':[{'path':relative,'experiment':'proposal','public_variant':'selected'}],
                'profiles':[],'sanitizers':[],'evidence_hashes':{relative:sha(out/relative)}})
            # Patch only plots: table regeneration itself must require neither CUDA nor CMake.
            with patch('export_results.plot'):
                audit=export(out);self.assertFalse(audit['passed']);self.assertFalse(audit['frozen_sanitizers_passed'])
                before={name:sha(out/(name+'.csv')) for name in schema.HEADERS}
                export(out)
                self.assertEqual(before,{name:sha(out/(name+'.csv')) for name in schema.HEADERS})
                r['samples_ms'][0]=999;dump(out/relative,r)
                with self.assertRaisesRegex(ValueError,'Native evidence changed'):export(out)

    def test_root_recognizes_b_and_plans_all_parts(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'absent'
            result=subprocess.run([sys.executable,ROOT/'scripts/run_all.py','--dry-run','--arch','89','--output',target],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('configure_part_b',result.stdout);self.assertIn('part_b/run.sh',result.stdout.replace('\\','/'))
            self.assertFalse(target.exists())

if __name__=='__main__':unittest.main()
