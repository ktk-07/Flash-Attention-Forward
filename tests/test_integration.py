"""CPU-only integration rejection tests; all synthetic records are temporary test fixtures."""
import copy,csv,importlib.util,json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import results_schema as schema
from run_all import aggregate,load_tables
from ncu_csv import captures,metric_rows
from workflow import dump

def fixture(part):
    tables={name:[] for name in schema.HEADERS}
    for precision in ['FP32','FP16']:
        for mode in (['dense','causal'] if part=='part_a' else ['causal']):
            for n in [33,65]:
                ctx={key:'' for key in schema.COMMON};ctx.update(experiment='proposal',precision=precision,B='1',H='1',N=str(n),d='64',mode=mode,variant='optimized' if precision=='FP32' else 'tensorcore',run_id=part,part=part,config='test',pattern='random',layout='BHNd_contiguous_row_major',input_precision=precision,accumulator_precision='FP32',output_precision='FP32',generator='splitmix64-fp32-v1',seed='5206',input_digest='0123456789abcdef',rounded_input_digest='fedcba9876543210' if precision=='FP16' else '')
                ctx['case_id']=schema.case_id(ctx)
                tables['correctness'].append({**ctx,'stage':'O','atol':'.0001','rtol':'.0001','max_absolute':'0','max_relative':'0','failures':'0','status':'PASS','oracle_input_precision':precision,'reference_scope':'full','checked_query_rows':'all','checked_elements':str(n*64),'nonfinite':'0'})
                tables['summary'].append({**ctx,'max_absolute':'0','max_relative':'0','warmup':'5','repeats':'20','forward_median_ms':'1','forward_q1_ms':'1','forward_q3_ms':'1','forward_iqr_ms':'0','sequences_per_second':'1000','tokens_per_second':str(1000*n),'validation_status':'PASS','reference_scope':'full','peak_workspace_allocated_bytes':'100'})
                tables['samples'] += [{**ctx,'repeat':str(i),'forward_ms':'1','qk_ms':'','softmax_ms':'','pv_ms':''} for i in range(20)]
    expanded={name:[] for name in tables}
    for name,rows in tables.items():
        for row in rows:
            variants=(['naive','tiled','parallel','optimized'] if row['precision']=='FP32' else ['wmma','ptx']) if part=='part_a' else (['skip','vector'] if row['precision']=='FP32' else ['wmma','ptx'])
            for variant in variants:
                value=dict(row,variant=variant,config=variant);value['case_id']=schema.case_id(value);expanded[name].append(value)
    return expanded

class IntegrationTests(unittest.TestCase):
    def test_sample_statistics_and_validation_required(self):
        tables=fixture('part_a');schema.validate(tables)
        broken=copy.deepcopy(tables);broken['summary'][0]['forward_median_ms']='2'
        with self.assertRaisesRegex(ValueError,'Summary/sample'):schema.validate(broken)
        broken=copy.deepcopy(tables);broken['correctness'][0]['status']='FAIL'
        with self.assertRaisesRegex(ValueError,'passing output'):schema.validate(broken)
        broken=copy.deepcopy(tables);broken['samples'].pop()
        with self.assertRaisesRegex(ValueError,'repetitions'):schema.validate(broken)
    def test_aggregation_rejects_gpu_or_data_mismatch(self):
        for mismatch in ['uuid','input','oracle','grid','header','schema']:
            with self.subTest(mismatch=mismatch),tempfile.TemporaryDirectory() as temp:
                out=Path(temp)
                for part in ['part_a','part_c']:
                    folder=out/'parts'/part;folder.mkdir(parents=True);tables=fixture(part)
                    if part=='part_c' and mismatch=='input':
                        for rows in tables.values():
                            for r in rows:r['input_digest']='aaaaaaaaaaaaaaaa';r['case_id']=schema.case_id(r)
                    if part=='part_c' and mismatch=='grid':
                        tables={name:[r for r in rows if r['N']=='33'] for name,rows in tables.items()}
                    for name,rows in tables.items():schema.write(folder/f'{name}.csv',name,rows)
                    if part=='part_c' and mismatch=='header':(folder/'profiles.csv').write_text('wrong\n')
                    dump(folder/'manifest.json',{'schema_version':999 if mismatch=='schema' and part=='part_c' else 1,'shared_cpu_reference_linked':not (mismatch=='oracle' and part=='part_c'),'device':{'uuid':part if mismatch=='uuid' else 'GPU-test'}});dump(folder/'audit.json',{'passed':True})
                with self.assertRaises(ValueError):aggregate(out,['part_a','part_c'],'smoke')
    def test_three_part_contract_and_plots_with_temporary_fixtures(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            for part in ['part_a','part_b','part_c']:
                folder=out/'parts'/part;folder.mkdir(parents=True)
                tables=fixture('part_c' if part=='part_b' else part)
                if part=='part_b':
                    for name,rows in tables.items():
                        unique={}
                        for r in rows:
                            if r['precision']!='FP32':continue
                            r.update(part=part,mode='dense',variant='selected',run_id=part,config='selected');r['case_id']=schema.case_id(r)
                            unique[json.dumps(r,sort_keys=True)]=r
                        tables[name]=list(unique.values())
                for name,rows in tables.items():schema.write(folder/f'{name}.csv',name,rows)
                dump(folder/'manifest.json',{'schema_version':1,'shared_cpu_reference_linked':True,'device':{'uuid':'GPU-test-fixture'}});dump(folder/'audit.json',{'passed':True})
                if part=='part_c':
                    (folder/'configs').mkdir();dump(folder/'configs/part_c.json',{'frozen_selection':{'families':{'cuda_core':{'64':'skip'},'tensor_core':{'64':'wmma'}}}})
            audit=aggregate(out,['part_a','part_b','part_c'],'smoke')
            self.assertTrue(audit['all_three_parts_present']);self.assertFalse(audit['final_report_grid'])
            self.assertEqual(len(schema.read(out/'comparison.csv')),8)
            self.assertEqual(len(schema.read(out/'plots/combined/dense_causal_work.csv')),2)
            self.assertTrue((out/'plots/combined/dense_causal_work.png').is_file())
    def test_both_fa1_baselines_aggregate_and_require_matched_inputs(self):
        from fa1_comparison import generate
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            for part in ['part_a','part_c']:
                folder=out/'parts'/part;folder.mkdir(parents=True);tables=fixture(part)
                if part=='part_c':
                    for name,rows in tables.items():
                        rows.extend(dict(r,variant='fa1' if r['variant']=='skip' else 'fa1_wmma',config='baseline') for r in list(rows) if r['variant'] in ['skip','wmma'])
                        for row in rows:row['case_id']=schema.case_id(row)
                    (folder/'configs').mkdir();dump(folder/'configs/part_c.json',{'frozen_selection':{'families':{'cuda_core':{'64':'skip'},'tensor_core':{'64':'wmma'}}}})
                for name,rows in tables.items():schema.write(folder/f'{name}.csv',name,rows)
                dump(folder/'manifest.json',{'schema_version':1,'shared_cpu_reference_linked':True,'device':{'uuid':'GPU-test-fixture'},**({'fa1_baselines':['fa1','fa1_wmma']} if part=='part_c' else {})});dump(folder/'audit.json',{'passed':True})
            aggregate(out,['part_a','part_c'],'smoke')
            self.assertEqual(len(schema.read(out/'fa1_fa2_comparison.csv')),6)
            self.assertEqual(len(schema.read(out/'fa1_cross_precision.csv')),2)
            tables=load_tables(out/'parts/part_c');selection={'families':{'cuda_core':{'64':'skip'}}}
            broken=copy.deepcopy(tables)
            next(r for r in broken['summary'] if r['variant']=='fa1_wmma')['rounded_input_digest']='different'
            with self.assertRaisesRegex(ValueError,'effective FP16'):generate(out,broken,selection)
            broken=copy.deepcopy(tables)
            next(r for r in broken['summary'] if r['variant']=='fa1_wmma')['accumulator_precision']='FP16'
            with self.assertRaisesRegex(ValueError,'numerical contract'):generate(out,broken,selection)
            tables['summary']=[r for r in tables['summary'] if r['variant']!='fa1']
            schema.write(out/'parts/part_c/summary.csv','summary',tables['summary'])
            with self.assertRaises(ValueError):aggregate(out,['part_a','part_c'],'smoke')
    def test_ptx_fa1_baseline_aggregate_and_require_matched_inputs(self):
        from fa1_comparison import generate
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            for part in ['part_a','part_c']:
                folder=out/'parts'/part;folder.mkdir(parents=True);tables=fixture(part)
                if part=='part_c':
                    for name,rows in tables.items():
                        rows.extend(dict(r,variant='fa1' if r['variant']=='skip' else 'fa1_ptx',config='baseline') for r in list(rows) if r['variant'] in ['skip','wmma'])
                        for row in rows:row['case_id']=schema.case_id(row)
                    (folder/'configs').mkdir();dump(folder/'configs/part_c.json',{'frozen_selection':{'families':{'cuda_core':{'64':'skip'},'tensor_core':{'64':'wmma'}}}})
                for name,rows in tables.items():schema.write(folder/f'{name}.csv',name,rows)
                dump(folder/'manifest.json',{'schema_version':1,'shared_cpu_reference_linked':True,'device':{'uuid':'GPU-test-fixture'},**({'fa1_baselines':['fa1','fa1_ptx']} if part=='part_c' else {})});dump(folder/'audit.json',{'passed':True})
            aggregate(out,['part_a','part_c'],'smoke')
            self.assertEqual(len(schema.read(out/'fa1_fa2_comparison.csv')),4)
            self.assertEqual(len(schema.read(out/'fa1_cross_precision.csv')),2)
            tables=load_tables(out/'parts/part_c');selection={'families':{'cuda_core':{'64':'skip'}}}
            broken=copy.deepcopy(tables)
            next(r for r in broken['summary'] if r['variant']=='fa1_ptx')['rounded_input_digest']='different'
            with self.assertRaisesRegex(ValueError,'effective FP16'):generate(out,broken,selection)
            broken=copy.deepcopy(tables)
            next(r for r in broken['summary'] if r['variant']=='fa1_ptx')['accumulator_precision']='FP16'
            with self.assertRaisesRegex(ValueError,'numerical contract'):generate(out,broken,selection)
            tables['summary']=[r for r in tables['summary'] if r['variant']!='fa1']
            schema.write(out/'parts/part_c/summary.csv','summary',tables['summary'])
            with self.assertRaises(ValueError):aggregate(out,['part_a','part_c'],'smoke')
    def test_missing_b_fails_without_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            # Missing-B rejection must use a fixture now that B exists in-tree.
            repo=Path(temp)/'repo';shutil.copytree(ROOT/'scripts',repo/'scripts')
            for part in ['part_a','part_c']:
                (repo/part).mkdir();(repo/part/'run.sh').write_text('# fixture: never executed\n')
            target=Path(temp)/'new';r=subprocess.run([sys.executable,repo/'scripts/run_all.py','--output',target,'--dry-run'],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('part_b/run.sh is missing',r.stderr);self.assertFalse(target.exists())
    def test_dry_run_and_alias_no_gpu_or_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            for wrapper in ['run.sh','run_all.sh','part_a/run.sh','part_b/run.sh','part_c/run.sh']:
                target=Path(temp)/wrapper.replace('/','_');args=['bash',ROOT/wrapper,'--arch','86','--dry-run','--output',target]
                if wrapper in ['run.sh','run_all.sh']:args+=['--parts','a,c']
                result=subprocess.run(args,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr);self.assertFalse(target.exists())
    def test_frozen_profile_processing_with_missing_counters(self):
        source=ROOT/'part_c/results/bounded_evaluation'
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            for name in ['manifest.json','frozen_selection.json']:shutil.copy2(source/name,out/name)
            shutil.copytree(source/'profiling',out/'profiling',ignore=shutil.ignore_patterns('*.ncu-rep','*.ncu-repz'))
            target=next((out/'profiling/cases').glob('*/raw.csv'))
            with target.open() as stream:rows=list(csv.reader(stream))
            for metric in ['sm__warps_active.avg.pct_of_peak_sustained_active','memory_l1_wavefronts_shared_ideal','smsp__pcsamp_warps_issue_stalled_mio_throttle']:
                if metric in rows[0]:rows[2][rows[0].index(metric)]='n/a'
            with target.open('w',newline='') as stream:csv.writer(stream).writerows(rows)
            result=subprocess.run([sys.executable,ROOT/'part_c/scripts/profile_frozen.py',out,'--analyze-only'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue(any(row['occupancy_pct']=='' for row in schema.read(out/'profiling/metrics.csv')))
            self.assertIn('n/a',(out/'profiling/REPORT.md').read_text())
    def test_multi_kernel_nsight_and_unavailable_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'raw.csv'
            path.write_text('ID,Kernel Name,metric.x,metric.y\n,,%,cycle\n0,QK,50,n/a\n1,softmax,30,12\n2,PV,40,16\n')
            data=captures(path);self.assertEqual(len(data),3)
            rows=list(metric_rows({},*data[0],0,'raw.csv','QK'));self.assertEqual(rows[0]['value'],'50');self.assertEqual(rows[1]['availability'],'unavailable');self.assertEqual(rows[1]['value'],'')
if __name__=='__main__':unittest.main()
