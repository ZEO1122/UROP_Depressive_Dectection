import unittest
import numpy as np
from experiments.run_a import challenge, rules, metrics, TextModel, synthetic_metrics


class ExperimentATest(unittest.TestCase):
    def test_past_and_other_cannot_determine_current_self(self):
        for q in ['Did you have trouble sleeping last year?', 'Does your mother have trouble sleeping?']:
            for a in ['Yes.', 'No.']:
                self.assertEqual(rules({'question': q, 'answer': a}), 'unknown')

    def test_answer_polarity_and_healthy_question(self):
        self.assertEqual(rules({'question': 'Do you sleep well?', 'answer': 'Yes.'}), 'deny')
        self.assertEqual(rules({'question': 'Do you sleep well?', 'answer': 'No.'}), 'support')
        self.assertEqual(rules({'question': 'Do you have trouble sleeping?', 'answer': 'Maybe.'}), 'unknown')

    def test_challenge_pairs(self):
        rows = challenge()
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({r['qa_id'] for r in rows}), 16)
        for pair in {r['pair'] for r in rows}:
            group = {r['variant']: r for r in rows if r['pair'] == pair}
            self.assertEqual(group['original']['intended_label'], group['paraphrase']['intended_label'])
            self.assertEqual(group['past']['intended_label'], 'unknown')
            self.assertEqual(group['other']['intended_label'], 'unknown')
            self.assertNotEqual(group['original']['intended_label'], 'unknown')
        self.assertEqual(synthetic_metrics(rows, [r['intended_label'] for r in rows])['paired_correct_original_past'], 1.)

    def test_absent_class_metrics_finite_and_explicit(self):
        result = metrics(['support'], ['support'])
        self.assertAlmostEqual(result['macro_f1_fixed_3'], 1 / 3)
        self.assertEqual(result['macro_f1_present_labels'], 1.)
        self.assertTrue(np.isfinite(result['macro_f1_fixed_3']))

    def test_vocabulary_fits_train_only(self):
        rows = [{'question': 'How is sleep?', 'answer': 'Yes trouble', 'weak_label': 'support'},
                {'question': 'How is sleep?', 'answer': 'No fine', 'weak_label': 'deny'}]
        model = TextModel('qa_relation').fit(rows)
        model.predict([{'question': 'uniquetesttoken', 'answer': 'othernewtoken'}])
        self.assertNotIn('uniquetesttoken', model.vectorizer.vocabulary_)
        self.assertNotIn('othernewtoken', model.vectorizer.vocabulary_)


if __name__ == '__main__':
    unittest.main()
