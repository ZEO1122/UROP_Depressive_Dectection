"""Experiment A: train/dev weak-label agreement and frozen synthetic semantics checks.

Neither source provides independent clinical ground truth. No test-set records are read.
"""
import argparse
import csv
import hashlib
import json
import platform
import re
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from experiments.prepare import require_private_output

LABELS = ['support', 'deny', 'unknown']
SEED = 20260927


def semantics(row):
    q, a = row['question'].lower(), row['answer'].lower().strip()
    past = bool(re.search(r'\b(last year|previously|used to|in the past|as a child|when you were)\b', q))
    other = bool(re.search(r'\b(your (?:mother|father|friend|sister|brother)|he|she|they)\b', q))
    positive = bool(re.search(r'\b(sleep well|sleeping well|enjoy|enjoying|interested|interest in)\b', q))
    negative = bool(re.search(r'\b(difficulty|trouble|difficult|lost|loss|struggle|struggling|hard|problems|less|no longer)\b', q))
    ap = 'yes' if re.match(r'^(yes|yeah|yep|absolutely)\b', a) else 'no' if re.match(r'^(no|nope|not really)\b', a) else 'other'
    scope = 'past' if past else 'other' if other else 'current_self'
    polarity = 'symptom' if negative else 'healthy' if positive else 'unspecified'
    return scope, polarity, ap


def relation_features(rows):
    return [{f'scope={s}': 1., f'polarity={p}': 1., f'answer={a}': 1.,
             f'scope_answer={s}_{a}': 1., f'polarity_answer={p}_{a}': 1.,
             f'joint={s}_{p}_{a}': 1.} for s, p, a in map(semantics, rows)]


def rules(row):
    scope, polarity, a = semantics(row)
    if scope != 'current_self' or a == 'other' or polarity == 'unspecified':
        return 'unknown'
    return 'support' if (polarity == 'symptom') == (a == 'yes') else 'deny'


def challenge():
    """Frozen authored templates: all short answers, current SELF target throughout."""
    families = {
        'sleep': ['Are you having trouble sleeping these days?', 'Do you currently have difficulty sleeping?',
                  'Did you have trouble sleeping last year?', 'Does your mother have trouble sleeping these days?'],
        'interest': ['Have you lost interest in your activities these days?', 'Are you currently enjoying activities less?',
                     'Did you lose interest in your activities last year?', 'Has your mother lost interest in her activities these days?'],
    }
    rows = []
    for domain, questions in families.items():
        for answer, target in [('Yes.', 'support'), ('No.', 'deny')]:
            pair = f'{domain}_{answer[:2]}'
            for variant, question in zip(['original', 'paraphrase', 'past', 'other'], questions):
                rows.append(dict(qa_id=f'synthetic_{pair}_{variant}', pair=pair, domain=domain,
                                 question=question, answer=answer, variant=variant,
                                 intended_label=target if variant in ['original', 'paraphrase'] else 'unknown'))
    return rows


def metrics(y, pred):
    support = dict(Counter(y))
    # Fixed class universe: missing classes contribute zero, report present-only too.
    return {'n': len(y), 'macro_f1_fixed_3': float(f1_score(y, pred, labels=LABELS, average='macro', zero_division=0)),
            'macro_f1_present_labels': float(f1_score(y, pred, labels=[x for x in LABELS if support.get(x)], average='macro', zero_division=0)),
            'accuracy': float(np.mean(np.asarray(y) == np.asarray(pred))), 'target_support': support,
            'confusion_labels': LABELS, 'confusion_matrix': confusion_matrix(y, pred, labels=LABELS).tolist()}


class TextModel:
    def __init__(self, view):
        self.view = view
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000)
        self.relations = DictVectorizer()
        self.model = LogisticRegression(C=1, class_weight='balanced', max_iter=2000, random_state=SEED)

    def texts(self, rows):
        if self.view == 'answer_only':
            return [r['answer'] for r in rows]
        if self.view == 'question_only':
            return [r['question'] for r in rows]
        return ['question ' + r['question'] + ' answer ' + r['answer'] for r in rows]

    def fit(self, rows):
        x = self.vectorizer.fit_transform(self.texts(rows))
        if self.view == 'qa_relation':
            x = hstack([x, self.relations.fit_transform(relation_features(rows))])
        self.model.fit(x, [r['weak_label'] for r in rows])
        return self

    def predict(self, rows):
        x = self.vectorizer.transform(self.texts(rows))
        if self.view == 'qa_relation':
            x = hstack([x, self.relations.transform(relation_features(rows))])
        return self.model.predict(x).tolist()


def paired_bootstrap(rows, y, predictions, reference='qa_flat', repeats=2000):
    ids = sorted({r['participant_id'] for r in rows})
    arrays = {}
    for name, pred in predictions.items():
        arrays[name] = np.array([confusion_matrix([y[i] for i, r in enumerate(rows) if r['participant_id'] == pid],
                                                [pred[i] for i, r in enumerate(rows) if r['participant_id'] == pid], labels=LABELS)
                                 for pid in ids])
    def score(c):
        den = c.sum(axis=0) + c.sum(axis=1)
        return np.divide(2 * np.diag(c), den, out=np.zeros(3), where=den > 0).mean()
    rng = np.random.default_rng(SEED)
    diffs = {name: [] for name in arrays if name != reference}
    for _ in range(repeats):
        selected = rng.integers(0, len(ids), size=len(ids))
        base = score(arrays[reference][selected].sum(axis=0))
        for name in diffs:
            diffs[name].append(score(arrays[name][selected].sum(axis=0)) - base)
    return {'reference': reference, 'repeats': repeats, 'cluster': 'participant', 'seed': SEED,
            'differences_macro_f1_fixed_3': {name: {'point': float(score(arrays[name].sum(axis=0)) - score(arrays[reference].sum(axis=0))),
                                                               'ci95': np.quantile(values, [.025, .975]).tolist()} for name, values in diffs.items()}}


def synthetic_metrics(rows, predictions):
    result = metrics([r['intended_label'] for r in rows], predictions)
    grouped = {}
    for r, pred in zip(rows, predictions):
        grouped.setdefault(r['pair'], {})[r['variant']] = pred == r['intended_label']
    for variant in ['paraphrase', 'past', 'other']:
        result[f'paired_correct_original_{variant}'] = float(np.mean([g['original'] and g[variant] for g in grouped.values()]))
    altered = [(r, p) for r, p in zip(rows, predictions) if r['variant'] in ['past', 'other']]
    result['past_other_false_current_assertion_rate'] = float(np.mean([p != 'unknown' for _, p in altered]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = require_private_output(args.output_dir)
    raw = (args.data_dir / 'qa.jsonl').read_bytes()
    all_rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if any(r['split'] not in ['train', 'dev'] for r in all_rows):
        raise ValueError('Input must exclude test participants entirely')
    rows = [r for r in all_rows if r['domain'] in ['sleep', 'interest']]
    train, dev = ([r for r in rows if r['split'] == split] for split in ['train', 'dev'])
    if {r['participant_id'] for r in train} & {r['participant_id'] for r in dev}:
        raise ValueError('Participant split overlap')
    if not train or not dev or len({r['weak_label'] for r in train}) < 2:
        raise ValueError('Need train/dev eligible QA and at least two train classes')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = challenge()
    dev_pred, synthetic_pred = {}, {}
    for name in ['answer_only', 'question_only', 'qa_flat', 'qa_relation']:
        model = TextModel(name).fit(train)
        dev_pred[name], synthetic_pred[name] = model.predict(dev), model.predict(synthetic)
    dev_pred['relation_rules'] = [rules(r) for r in dev]
    synthetic_pred['relation_rules'] = [rules(r) for r in synthetic]
    majority = Counter(r['weak_label'] for r in train).most_common(1)[0][0]
    dev_pred['train_majority'] = [majority] * len(dev)
    synthetic_pred['train_majority'] = [majority] * len(synthetic)
    dev_pred['always_unknown'] = ['unknown'] * len(dev)
    synthetic_pred['always_unknown'] = ['unknown'] * len(synthetic)
    targets = [r['weak_label'] for r in dev]
    report = {'experiment': 'A', 'primary_independent_evaluation': 'blocked_missing_human_annotations',
              'data_sha256': hashlib.sha256(raw).hexdigest(), 'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__},
              'protocol': {'train_only_fit': True, 'test_access': False, 'C': 1, 'class_weight': 'balanced', 'seed': SEED,
                           'synthetic_training': False, 'independent_human_labels': False,
                           'absent_class_policy': 'Fixed 3-class macro-F1 assigns zero to absent classes; present-only separately reported.'},
              'limitations': ['Weak-label agreement is circular diagnostic, not symptom accuracy.',
                             'Synthetic targets authored alongside rules; rule success is a construction sanity check, not independent validation.',
                             'Only 16 short-answer synthetic cases from two template families; no clinical generalization claim.',
                             'Natural dev is exploratory; do not tune and then report it as untouched evaluation.'],
              'counts': {split: {'qa': len(group), 'participants': len({r['participant_id'] for r in group}), 'weak_labels': dict(Counter(r['weak_label'] for r in group))} for split, group in [('train', train), ('dev', dev)]},
              'natural_dev_WEAK_LABEL_AGREEMENT': {name: metrics(targets, pred) for name, pred in dev_pred.items()},
              'synthetic_CONSTRUCTION_TARGET': {name: synthetic_metrics(synthetic, pred) for name, pred in synthetic_pred.items()},
              'natural_paired_bootstrap': paired_bootstrap(dev, targets, dev_pred)}
    (args.output_dir / 'a_report.json').write_text(json.dumps(report, indent=2))
    with (args.output_dir / 'a_predictions.jsonl').open('w') as f:
        for dataset, items, preds in [('natural_dev_weak', dev, dev_pred), ('synthetic_construction', synthetic, synthetic_pred)]:
            for i, row in enumerate(items):
                out = {'dataset': dataset, 'qa_id': row['qa_id'], 'predictions': {name: p[i] for name, p in preds.items()}}
                if dataset.startswith('synthetic'):
                    out.update(row)
                else:
                    out.update(participant_id=row['participant_id'], weak_label=row['weak_label'])
                f.write(json.dumps(out) + '\n')
    # Blind review: no weak labels, model outputs, PHQ, or intended targets shown.
    if not (args.output_dir / 'a_review.csv').exists():
        with (args.output_dir / 'a_review.csv').open('w', newline='') as f:
            fields = ['qa_id', 'participant_id', 'question', 'answer', 'reviewer_id', 'current_self_status', 'time_scope', 'subject', 'evidence_span', 'notes']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in dev:
                writer.writerow({key: row.get(key, '') for key in fields})
    (args.output_dir / 'a_review_instructions.md').write_text('Two reviewers independently label current self symptom status: support / deny / unknown / conflict. Record time, subject, and verbatim evidence. Do not use PHQ, weak labels or predictions. Adjudicate after recording both independent judgments. These judgments are not present in this run. Synthetic cases need separate blinded naturalness and target validation before scientific evaluation.\n')
    print(json.dumps({'report': str(args.output_dir / 'a_report.json'), 'counts': report['counts'], 'status': report['primary_independent_evaluation']}))


if __name__ == '__main__':
    main()
