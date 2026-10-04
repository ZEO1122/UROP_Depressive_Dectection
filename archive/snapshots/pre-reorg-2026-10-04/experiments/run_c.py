"""Offline QA selection pilot. All coverage results are lexical proxies, not human gold."""
import argparse
import collections
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path

from experiments.prepare import require_private_output

BUDGETS = (0.1, 0.2, 0.3, 0.5)
SEEDS = (42, 43, 44, 45, 46)
QUERY = 'sleep sleeping insomnia tired enjoy interest pleasure activities weeks months years daily often work friends family'
PATTERNS = {
    'duration': r'\b(?:\d+|one|two|three|four|five|six|several|few)\s+(?:days?|weeks?|months?|years?)\b|\bsince\b',
    'frequency': r'\b(?:every|daily|always|sometimes|often|rarely|usually|never)\b|\b(?:once|twice)\b',
    'function': r'\b(?:work|school|job|friends|family|concentrat\w*|function\w*)\b',
    'report': r'\b(?:yes|yeah|no|not|sleep\w*|insomnia|enjoy\w*|interest\w*|pleasure|fun)\b',
}


def tokens(text):
    return re.findall(r"\b[\w']+\b", str(text).lower())


def package_text(row):
    # Upstream QA consists of the complete interviewer turn and all following
    # participant turns. Any additional supplied context is also charged.
    context = row.get('context', '')
    if isinstance(context, list):
        context = ' '.join(str(x) for x in context)
    return ' '.join(str(x) for x in (context, row.get('question', ''), row.get('answer', '')) if x)


def cost(row):
    return len(tokens(package_text(row)))


def proxy_slots(row):
    """Lexical availability only; no claim of symptom truth, sufficient context or clinical adequacy."""
    domain = row.get('domain')
    answer = str(row.get('answer', '')).lower()
    if domain not in ('sleep', 'interest') or not tokens(answer):
        return set()
    return {domain + ':' + slot for slot, pattern in PATTERNS.items()
            if re.search(pattern, answer)}


def coverage(rows, selected):
    available = set().union(*(proxy_slots(r) for r in rows))
    found = set().union(*(proxy_slots(rows[i]) for i in selected))
    return (len(found) / len(available) if available else None), len(available)


def bm25(rows):
    docs = [tokens(package_text(r)) for r in rows]
    if not docs:
        return []
    df = collections.Counter(t for doc in docs for t in set(doc))
    avg = sum(map(len, docs)) / len(docs) or 1
    query = set(tokens(QUERY))
    scores = []
    for doc in docs:
        counts = collections.Counter(doc)
        score = 0.0
        for term in query:
            tf = counts[term]
            if tf:
                idf = math.log(1 + (len(docs) - df[term] + .5) / (df[term] + .5))
                score += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * len(doc) / avg))
        scores.append(score)
    return scores


def select(rows, budget_words, method, seed=42):
    """Atomic packages only, strict integer word cap; no test annotations used."""
    remaining = max(0, int(budget_words))
    costs = [cost(r) for r in rows]
    chosen = []
    candidates = [i for i, c in enumerate(costs) if c > 0]
    scores = bm25(rows)
    maximum = max(scores, default=1) or 1
    normalized = [s / maximum for s in scores]
    sets = [set(tokens(package_text(r))) for r in rows]
    covered = set()
    if method == 'random':
        random.Random(seed).shuffle(candidates)
    elif method == 'bm25':
        candidates.sort(key=lambda i: (-scores[i], i))
    elif method not in ('chronological', 'mmr', 'slot_gain_per_word'):
        raise ValueError(method)
    while candidates:
        fitting = [i for i in candidates if costs[i] <= remaining]
        if not fitting:
            break
        if method == 'mmr':
            def score(i):
                redundancy = max((len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) for j in chosen), default=0)
                return .7 * normalized[i] - .3 * redundancy
            pick = max(fitting, key=lambda i: (score(i), -i))
        elif method == 'slot_gain_per_word':
            # Same lexical detector as the proxy evaluator: intentionally reported
            # as circular engineering smoke-test, never independent efficacy.
            pick = max(fitting, key=lambda i: ((len(proxy_slots(rows[i]) - covered) + .01 * normalized[i]) / costs[i], -i))
        else:
            pick = fitting[0]
        candidates.remove(pick)
        chosen.append(pick)
        remaining -= costs[pick]
        covered.update(proxy_slots(rows[pick]))
    return sorted(chosen)


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def budget_auc(values):
    if any(x is None for x in values):
        return None
    return sum((b - a) * (x + y) / 2 for a, b, x, y in zip(BUDGETS, BUDGETS[1:], values, values[1:])) / .4


def run(data_dir, output_dir):
    source = Path(data_dir) / 'qa.jsonl'
    output = require_private_output(Path(output_dir))
    output.mkdir(parents=True, exist_ok=True)
    grouped = collections.defaultdict(list)
    excluded = 0
    with source.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row['split'] not in ('train', 'dev'):
                raise ValueError('Only train/dev QA allowed; sealed test input rejected')
            grouped[(row['split'], int(row['participant_id']))].append(row)
    records = []
    for (split, pid), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r.get('position', 0))
        total = sum(cost(r) for r in rows)
        for method in ('chronological', 'random', 'bm25', 'mmr', 'slot_gain_per_word'):
            for seed in SEEDS if method == 'random' else (42,):
                for fraction in BUDGETS:
                    limit = math.floor(total * fraction)
                    selected = select(rows, limit, method, seed)
                    value, denominator = coverage(rows, selected)
                    spent = sum(cost(rows[i]) for i in selected)
                    assert spent <= limit
                    records.append(dict(split=split, participant_id=pid, method=method, seed=seed, budget_fraction=fraction,
                                        word_budget=limit, selected_words=spent, total_words=total,
                                        actual_fraction=spent / total if total else 0,
                                        proxy_coverage=value, available_proxy_slots=denominator,
                                        selected_qa_ids=[rows[i]['qa_id'] for i in selected]))
    with (output / 'c_selections_private.jsonl').open('w') as fh:
        for record in records:
            fh.write(json.dumps(record) + '\n')
    summary = []
    for split in ('train', 'dev'):
        for method in ('chronological', 'random', 'bm25', 'mmr', 'slot_gain_per_word'):
            subset = [r for r in records if r['split'] == split and r['method'] == method]
            by_pid = collections.defaultdict(list)
            for r in subset:
                by_pid[r['participant_id']].append(r)
            aucs, thresholds, threshold_words, unattained = [], [], [], 0
            for participant_rows in by_pid.values():
                curve = [mean([r['proxy_coverage'] for r in participant_rows if r['budget_fraction'] == b]) for b in BUDGETS]
                aucs.append(budget_auc(curve))
                if curve[0] is not None:
                    hit = next((b for b, c in zip(BUDGETS, curve) if c >= .8), None)
                    if hit is None:
                        unattained += 1
                    else:
                        thresholds.append(hit)
                        threshold_words.append(mean([r['selected_words'] for r in participant_rows if r['budget_fraction'] == hit]))
            n_eligible = sum(a is not None for a in aucs)
            summary.append(dict(split=split, method=method, n_participants=len(by_pid), n_with_proxy_information=n_eligible,
                                n_without_proxy_information=len(by_pid)-n_eligible,
                                coverage_curve={str(b): mean([r['proxy_coverage'] for r in subset if r['budget_fraction']==b]) for b in BUDGETS},
                                proxy_coverage_at_20=mean([r['proxy_coverage'] for r in subset if r['budget_fraction']==.2]),
                                normalized_auc_10_to_50=mean(aucs),
                                grid_budget_fraction_to_80_mean_attainers_only=mean(thresholds),
                                grid_selected_words_to_80_mean_attainers_only=mean(threshold_words),
                                not_reaching_80_by_50_fraction=unattained/n_eligible if n_eligible else None))
    report = dict(experiment='C', status='PROXY_ONLY', input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  primary_human_evaluation='UNAVAILABLE: independent adjudicated human annotations not supplied',
                  negation_temporal_subject_conflict_loss='UNAVAILABLE: requires independent human annotation',
                  evidence_precision='UNAVAILABLE: lexical detections are not evidence correctness labels',
                  warning='Retrieval engineering proxies only: pure keyword budget selection is not efficacy. No selector ranking or best-model claim is justified. Shared lexical detector for slot selector and evaluator induces same-lexicon bias. Coverage is an engineering diagnostic, not clinical utility or comparative efficacy.',
                  proxy_definition='8 binary domain:slot types: sleep/interest × report/duration/frequency/function; answer-side regex only; repeats count once; no information excluded from coverage denominator.',
                  package_limitation='Complete QA text retained; extra context charged if supplied. Referential completeness beyond QA requires human review.',
                  cost_definition='regex word tokens of question + complete answer + optional context; floor(total * budget); no overflow',
                  random_aggregation='5 fixed seeds averaged within participant; threshold uses mean random coverage curve',
                  threshold_definition='first tested budget reaching 0.8; grid-censored if not attained at 0.5; not exact minimal word cost',
                  auc_definition='trapezoidal area over nominal budget fractions .1,.2,.3,.5 divided by .4; actual utilization saved separately',
                  sealed_test_rows_excluded=excluded, budgets=BUDGETS, seeds=SEEDS, summary=summary)
    (output / 'c_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    if not (output / 'c_human_review_blank_private.csv').exists():
        with (output / 'c_human_review_blank_private.csv').open('w', newline='') as fh:
            fields = ['participant_id','split','qa_id','question','answer','domain_human','slot_human','fact_id_human','evidence_span_human','status_human','context_required_human','annotator_id','adjudicated']
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for (split, pid), rows in sorted(grouped.items()):
                for row in rows:
                    writer.writerow({k: row.get(k, '') for k in ('participant_id','split','qa_id','question','answer')})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    report = run(args.data_dir, args.output_dir)
    print(json.dumps({'status': report['status'], 'groups': len(report['summary']), 'output': args.output_dir}))


if __name__ == '__main__':
    main()
