import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from generic_report import report_csv, write_report

class GenericReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
    def csv(self, text):
        path = self.root/'data.csv'; path.write_text(text, encoding='utf-8'); return path
    def test_third_dataset_two_models_zero_retention_and_bound(self):
        source = Path(__file__).parent/'generic_example/inputs.csv'
        rows, summary = report_csv(source, expected_runs=['repeat-red', 'repeat-blue'], synthetic_demo=True)
        self.assertEqual(len(rows), 3)
        a, b, c = rows
        self.assertEqual((a['total_count'], a['failure_count'], a['retained_count'], a['retained_failure_count']), (4, 2, 3, 1))
        self.assertEqual(a['risk_after'], 1/3)
        self.assertEqual(a['fixed_clip_count_risk_lower_bound'], 1/3)
        self.assertEqual(b['status'], 'zero_retained')
        self.assertIsNone(b['risk_after']); self.assertIsNone(b['fixed_clip_count_risk_lower_bound'])
        self.assertEqual(c['risk_after'], 0)
        self.assertEqual(summary['source']['rows'], 8)
        self.assertEqual(summary['groups'][0]['pooled_risk_after'], 1/3)
        self.assertEqual(summary['groups'][0]['zero_retained_clip_count'], 1)
        self.assertEqual(summary['source']['sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        dice_rows, _ = report_csv(source, dice_threshold=.3)
        self.assertEqual(dice_rows, rows)
    def test_lower_bound_does_not_assume_selector_is_optimal(self):
        path = self.csv('dataset,model,run,clip,frame,failure,retained\nD,M,r,C,a,true,true\nD,M,r,C,b,false,false\n')
        rows, _ = report_csv(path)
        self.assertEqual(rows[0]['risk_after'], 1)
        self.assertEqual(rows[0]['fixed_clip_count_risk_lower_bound'], 0)
    def test_duplicate_identity_rejected(self):
        row = 'D,M,r,C,a,true,true\n'
        path = self.csv('dataset,model,run,clip,frame,failure,retained\n'+row+row)
        with self.assertRaisesRegex(ValueError, 'duplicate frame-run'): report_csv(path)
    def test_partial_missing_run_rejected(self):
        path = self.csv('dataset,model,run,clip,frame,failure,retained\nD,M,r1,C,a,false,true\nD,M,r2,C,a,false,true\nD,M,r1,C,b,false,true\n')
        with self.assertRaisesRegex(ValueError, 'run coverage mismatch'): report_csv(path, expected_runs=['r1', 'r2'])
        with self.assertRaisesRegex(ValueError, 'duplicate expected'): report_csv(path, expected_runs=['r1', 'r1'])
    def test_invalid_values_and_headers(self):
        header = 'dataset,model,run,clip,frame,failure,retained\n'
        for body in ['D,M,r,C,a,yes,true\n', 'D,M,r,,a,true,true\n', 'D,M,r,C,a,true\n', 'D,M,r,C,a,true,true,extra\n']:
            with self.subTest(body=body), self.assertRaises(ValueError): report_csv(self.csv(header+body))
        for text in [header, 'dataset,model,run,clip,frame,retained,retained\nD,M,r,C,a,true,true\n']:
            with self.subTest(text=text), self.assertRaises(ValueError): report_csv(self.csv(text))
    def test_dice_requires_explicit_valid_threshold_and_consistency(self):
        header = 'dataset,model,run,clip,frame,dice,retained\n'
        path = self.csv(header+'D,M,r,C,a,0.2,true\n')
        with self.assertRaisesRegex(ValueError, 'missing columns'): report_csv(path)
        for threshold in [float('nan'), float('inf'), -.1, 1.1]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError): report_csv(path, dice_threshold=threshold)
        for dice in ['nan', 'inf', '-0.1', '1.1', 'text']:
            with self.subTest(dice=dice), self.assertRaises(ValueError): report_csv(self.csv(header+f'D,M,r,C,a,{dice},true\n'), dice_threshold=.3)
        path = self.csv('dataset,model,run,clip,frame,dice,failure,retained\nD,M,r,C,a,0.2,false,true\n')
        with self.assertRaisesRegex(ValueError, 'disagrees'): report_csv(path, dice_threshold=.3)
    def test_all_rejected_json_null_and_output_integrity(self):
        path = self.csv('dataset,model,run,clip,frame,failure,retained\nD,M,r,C,a,true,false\n')
        before = path.read_bytes(); rows, summary = report_csv(path)
        self.assertIsNone(summary['groups'][0]['pooled_risk_after'])
        out = self.root/'output'; write_report(out, rows, summary)
        self.assertEqual(path.read_bytes(), before)
        parsed = json.loads((out/'summary.json').read_text())
        self.assertIsNone(parsed['groups'][0]['pooled_lower_bound_given_each_clip_retained_count'])
        receipt = json.loads((out/'receipt.json').read_text())
        for name, sha in receipt['output_sha256'].items(): self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(), sha)
        with self.assertRaisesRegex(ValueError, 'absent or empty'): write_report(out, rows, summary)

if __name__ == '__main__': unittest.main()
