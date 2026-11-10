"""CPU regression of pilot attribution/provenance; fixtures are synthetic, never report data."""
import csv,hashlib,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('fa1_pilot',ROOT/'part_c/scripts/compare_fa1.py');pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)

def fixture(out,broken=None):
    raw=out/'raw/part_c/fa1_pilot';(raw/'cases').mkdir(parents=True)
    plan={'run_id':'TEST_FIXTURE','grid':'small','shapes':[[4,8,256,64],[4,8,1024,64]],'sources':{},'device':{'uuid':'GPU-TEST-FIXTURE'},'binary_sha256':'test_fixture','cases':[]}
    times={'fa1':10.,'skip':2.,'fa1_wmma':1.,'wmma':.5,'ptx':.25}
    for b,h,n,d in plan['shapes']:
        for kernel in pilot.KERNELS:
            tc=kernel in ['fa1_wmma','wmma','ptx'];fa1=kernel in ['fa1','fa1_wmma'];name=f'{kernel}_{n}.csv'
            row={'kernel':kernel,'config':'0','B':str(b),'H':str(h),'N':str(n),'d':str(d),'pattern':'random','mode':'benchmark','input_precision':'FP16' if tc else 'FP32','accumulator_precision':'FP32','output_precision':'FP32','input_digest':'0123456789abcdef','rounded_input_digest':'fedcba9876543210','atol':'.002' if tc else '.0001','rtol':'.02' if tc else '.0001','max_absolute':'0','max_relative':'0','status':'PASS','failures':'0','nonfinite':'0','checked_elements':str(b*h*3*d),'reference_scope':'sampled_queries','explicit_live_device_bytes':str(b*h*n*d*(10 if tc else 16)+(8*b*h*n if fa1 else 0)),'observed_device_delta_bytes':'0','block_q':'32','block_kv':'32','warps':'2' if tc else '4','thread_q':'1','thread_k':'1','registers_per_thread':'32','shared_bytes':'4096','local_bytes_per_thread':'0','active_blocks_per_sm':'1','occupancy_estimate_percent':'4','forward_ms':str(times[kernel]),'repeat':'0'}
            if kernel=='wmma' and broken=='precision':row['output_precision']='FP16'
            if kernel=='wmma' and broken=='rounded':row['rounded_input_digest']='aaaaaaaaaaaaaaaa'
            rows=[{**row,'repeat':str(i)} for i in range(19 if broken=='repetitions' and kernel=='fa1' else 20)]
            with (raw/'cases'/name).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
            plan['cases'].append({'file':name,'kernel':kernel,'B':b,'H':h,'N':n,'d':d})
    (raw/'plan.json').write_text(json.dumps(plan))

class PilotTests(unittest.TestCase):
    def test_matched_and_cross_precision_ratios_are_separate_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);fixture(out);audit=pilot.analyze(out)
            self.assertTrue(audit['passed']);self.assertFalse(audit['official_FA1_measured'])
            rows=pilot.schema.read(out/'comparison.csv');tradeoffs=pilot.schema.read(out/'precision_tradeoffs.csv')
            self.assertEqual(len(rows),6);self.assertEqual(len(tradeoffs),2)
            self.assertTrue(all(float(r['speedup'])=={'skip':5,'wmma':2,'ptx':4}[r['candidate']] for r in rows))
            self.assertTrue(all(r['contract'].endswith('not_FA2_speedup') for r in tradeoffs))
            files=list(out.glob('*.csv'))+list((out/'plots').rglob('*.png'))+list((out/'plots').rglob('*.pdf'))
            before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files};pilot.analyze(out)
            self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest()==value for p,value in before.items()))
            native=next((out/'raw/part_c/fa1_pilot/cases').glob('*.csv'));native.write_text(native.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'Native evidence changed'):pilot.analyze(out)
    def test_rejects_mismatched_precision_inputs_and_repetition_count(self):
        for kind in ['precision','rounded','repetitions']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                out=Path(temp);fixture(out,kind)
                with self.assertRaises(ValueError):pilot.analyze(out)
    def test_dry_run_is_gpu_free_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'new';result=subprocess.run(['bash',ROOT/'part_c/run.sh','--fa1-pilot','--arch','86','--dry-run','--output',out],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
