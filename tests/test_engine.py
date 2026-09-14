import copy, json, tempfile, unittest
from pathlib import Path
from h2_eval.engine import (ValidationError, load_dataset, choose_tau, retained_rows,
                            aggregate, spearman, split_clips, bootstrap, strict_json, verify_sun, validate_config)

def record(signal, dice, clip='a', frame='1'):
    return (clip, frame, 'uac_base', 44, dice, signal, 10)

def receipt(seed=44):
    return {'family':'uac_base','seed':seed,'per_frame':[
        {'clip':'a','frame':'1','dice':1.0,'max_prob':0.1,'gt_pixels':0,'predicted_pixels':0,'case':'both_empty'},
        {'clip':'b','frame':'1','dice':0.0,'max_prob':0.1,'gt_pixels':10,'predicted_pixels':0,'case':'one_empty'}]}

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.cfg={'families':['uac_base'],'seeds':[44,45],
                  'expected':{'sun':{'frames_per_run':2,'clips_per_run':2}},
                  'bootstrap_seed':20260913,'bootstrap_replicates':40}
        for seed in (44,45):
            (self.path/f'{seed}.json').write_text(json.dumps(receipt(seed)))
    def tearDown(self): self.temp.cleanup()
    def bad(self, mutate):
        d=receipt();mutate(d);(self.path/'44.json').write_text(json.dumps(d))
        with self.assertRaises(ValidationError): load_dataset(self.path,'sun',self.cfg)
    def test_valid_and_empty_masks(self):
        data,v=load_dataset(self.path,'sun',self.cfg)
        self.assertEqual(len(data),4);self.assertTrue(v['identity_and_gt_aligned'])
    def test_duplicate_frame(self): self.bad(lambda d:d['per_frame'].append(d['per_frame'][0]))
    def test_bad_gt(self): self.bad(lambda d:d['per_frame'][1].update(gt_pixels=11))
    def test_illegal_value(self): self.bad(lambda d:d['per_frame'][1].update(dice=1.1))
    def test_bad_empty_rule(self): self.bad(lambda d:d['per_frame'][0].update(dice=0.0))
    def test_missing_run(self):
        (self.path/'45.json').unlink()
        with self.assertRaises(ValidationError):load_dataset(self.path,'sun',self.cfg)
    def test_duplicate_json_key(self):
        p=self.path/'bad';p.write_text('{"a":1,"a":2}')
        with self.assertRaises(ValidationError):strict_json(p)
    def test_tie_expansion_is_not_calibration_success(self):
        data=[record(0.1,1,frame='a'),record(0.1,0,frame='b')]
        tau,meta=choose_tau(data,.1,.3,'prefix')
        self.assertEqual(tau,.1);self.assertEqual(meta['tie_expansion_count'],1)
        self.assertEqual(meta['actual_calibration_risk'],.5)
        self.assertFalse(meta['actual_calibration_meets_target'])
        tau,meta=choose_tau(data,.1,.3,'full_tie_groups')
        self.assertIsNone(tau);self.assertEqual(meta['actual_calibration_retained'],0)
    def test_later_recovery_is_considered(self):
        data=[record(i/20,0 if i==0 else 1,frame=str(i)) for i in range(11)]
        tau,meta=choose_tau(data,.1,.3,'full_tie_groups')
        self.assertEqual(tau,.5);self.assertEqual(meta['actual_calibration_retained'],11)
    def test_no_feasible_and_zero_retention(self):
        data=[record(.1,0)]
        tau,meta=choose_tau(data,.1,.3,'prefix');self.assertIsNone(tau)
        rows=retained_rows(data,['a'],tau,.3,.15)
        self.assertIsNone(rows[0]['retained_risk']);self.assertIsNone(rows[0]['compliant'])
        self.assertIsNone(aggregate(rows)['pooled_risk'])
        result=bootstrap(rows,self.cfg)
        self.assertIsNone(result['pooled_risk_ci95']);self.assertEqual(result['zero_admitted_replicates'],40)
    def test_partitions(self):
        ids=['seq1','seq2','seq10','seq11','seq3']
        a,b=split_clips(ids);c,d=split_clips(ids,True)
        self.assertEqual(a,d);self.assertEqual(b,c);self.assertFalse(set(a)&set(b))
        self.assertEqual(a,['seq1','seq11','seq3'])
    def test_spearman_ties_and_undefined(self):
        self.assertAlmostEqual(spearman([1,1,2,3],[1,1,2,3]),1)
        self.assertAlmostEqual(spearman([1,2,3],[3,2,1]),-1)
        self.assertIsNone(spearman([1,1],[1,2]))
    def test_cluster_bootstrap_deterministic(self):
        data=[record(.1,0),record(.2,1,clip='b')]
        rows=retained_rows(data,['a','b'],.5,.3,.15)
        self.assertEqual(bootstrap(rows,self.cfg),bootstrap(rows,self.cfg))
    def test_zero_retention_replay(self):
        p=self.path/'reference.csv'
        p.write_text('host,arm,clip,retained_count,retained_crashes,total_count,retained_risk,tau\nuac_base,baseline,a,0,0,1,,0.01\n')
        rows=retained_rows([record(.1,0)],['a'],.01,.3,.15)
        result=verify_sun(rows,p,{'uac_base':{'K6':{'tau':.01,'retained_crash_rate':None}}},'uac_base',1e-12)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(rows[0]['status'],'zero_retained')
    def test_config_rejects_loose_replay(self):
        cfg=json.loads((Path(__file__).parents[1]/'protocol.json').read_text())
        validate_config(cfg)
        cfg['replay_tolerance']=.1
        with self.assertRaises(ValidationError):validate_config(cfg)
    def test_pg_positive_filter_and_alignment(self):
        for seed in (44,45):
            d=receipt(seed);frames=d.pop('per_frame')
            for r in frames:r['clip_id']=r.pop('clip')
            (self.path/f'{seed}.json').write_text(json.dumps({'summary':d,'frames':frames}))
        cfg=copy.deepcopy(self.cfg);cfg['expected']['pg']={'frames_per_run':2,'clips_per_run':2,'positive_clips':1}
        rows,v=load_dataset(self.path,'pg',cfg)
        self.assertEqual(len(rows),2);self.assertTrue(all(r[6]>0 for r in rows))

if __name__=='__main__':unittest.main()
