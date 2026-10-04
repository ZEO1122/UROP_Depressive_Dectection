import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.run_c import budget_auc, cost, coverage, proxy_slots, select, run


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(question='How do you sleep?', answer='I sleep poorly every day for three weeks', domain='sleep'),
                     dict(question='What is your work?', answer='I work at school', domain='none'),
                     dict(question='How do you sleep?', answer='I sleep poorly every day for three weeks', domain='sleep')]

    @patch('experiments.run_c.require_private_output', side_effect=lambda path: path)
    def test_end_to_end_rejects_test_and_outputs_proxy_only(self, mocked_guard):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(participant_id=1, split='train', qa_id='1', position=0, **self.rows[0]),
                    dict(participant_id=2, split='test', qa_id='2', position=0, **self.rows[0])]
            (root / 'qa.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'sealed test'):
                run(root, root / 'out')
            (root / 'qa.jsonl').write_text(json.dumps(rows[0])+'\n')
            report = run(root, root / 'out')
            self.assertEqual(report['status'], 'PROXY_ONLY')
            self.assertEqual(report['sealed_test_rows_excluded'], 0)
            outputs = [json.loads(s) for s in (root / 'out/c_selections_private.jsonl').read_text().splitlines()]
            self.assertTrue(outputs)
            self.assertTrue(all(r['split'] == 'train' and r['selected_words'] <= r['word_budget'] for r in outputs))
            worksheet = (root / 'out/c_human_review_blank_private.csv').read_text()
            self.assertIn('annotator_id', worksheet)

    def test_private_output_guard(self):
        with self.assertRaisesRegex(ValueError, 'Participant outputs'):
            run('/tmp/nonexistent', '/tmp/c_experiment_unsafe_output')

    def test_strict_budget_all_methods(self):
        for method in ('chronological','random','bm25','mmr','slot_gain_per_word'):
            for budget in range(50):
                selected = select(self.rows, budget, method)
                self.assertLessEqual(sum(cost(self.rows[i]) for i in selected), budget)
                self.assertEqual(len(selected),len(set(selected)))
            self.assertEqual(select(self.rows, 0, method), [])

    def test_duplicate_not_new_coverage(self):
        self.assertEqual(coverage(self.rows, [0]), coverage(self.rows, [0, 2]))

    def test_empty_denominator(self):
        self.assertEqual(coverage([dict(question='sleep daily for three years', answer='', domain='sleep')], [0]), (None,0))
        self.assertEqual(coverage([], []), (None,0))

    def test_question_not_evidence(self):
        row = dict(question='Have you slept badly for three years every day at work?', answer='maybe', domain='sleep')
        self.assertEqual(proxy_slots(row), set())

    def test_context_charged(self):
        row = dict(question='why',answer='yes',context='three more words')
        self.assertEqual(cost(row), 5)

    def test_auc_and_reproducibility(self):
        self.assertAlmostEqual(budget_auc([1,1,1,1]),1)
        self.assertIsNone(budget_auc([None]*4))
        self.assertEqual(select(self.rows,20,'random',42),select(self.rows,20,'random',42))

if __name__ == '__main__':
    unittest.main()
