"""Label-free ASR/official BERTScore inputs for declared positional intervention."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .hique_data import length_batches, question_table
from .prepare import require_private_output, split_ids


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def build_record(pid, split, rows, mapping, questions):
    """Last answer per canonical slot; all events retained for collision audit."""
    retained, events = {}, []
    root = None
    previous_question = None
    depth = 0
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if row['speaker'] != 'Ellie':
            continue
        slot = mapping[row['text']]['slot']
        followup = questions[slot]['type'].lower().startswith('follow')
        if not followup:
            root, depth = slot, 0
        else:
            depth += 1
        parent = root if followup and root is not None else slot
        orphan = followup and root is None
        if orphan:
            counts['orphan_followup'] += 1
        event = {'slot': slot, 'type': questions[slot]['type'], 'parent_topic_id': parent,
                 'previous_question_id': previous_question, 'followup_depth': depth,
                 'orphan_fallback': orphan, 'question_row': index,
                 'question_span': [row['start'], row['end']], 'retained': False}
        previous_question = slot
        if index + 1 >= len(rows) or rows[index + 1]['speaker'] != 'Participant':
            event['reason'] = 'no_following_response'
            events.append(event)
            continue
        response = rows[index + 1]
        start, end = float(response['start']), float(response['end'])
        if not response['text'].strip() or not np.isfinite([start, end]).all() or start < 0 or end <= start:
            event['reason'] = 'empty_or_invalid_response'
            events.append(event)
            continue
        event.update(text=response['text'].strip(), spans=[[start, end]], answer_row=index + 1,
                     retained=True)
        if slot in retained:
            events[retained[slot]]['retained'] = False
            counts['overwritten_occurrences'] += 1
            if events[retained[slot]]['parent_topic_id'] != parent:
                counts['cross_parent_collisions'] += 1
        retained[slot] = len(events)
        events.append(event)
    slots = [events[retained[s]].copy() for s in sorted(retained)]
    position_ids = list(range(85))
    presence = [False] * 85
    for slot in slots:
        position_ids[slot['slot']] = slot['parent_topic_id']
        presence[slot['slot']] = True
    counts['events'] = len(events)
    counts['retained_slots'] = len(slots)
    counts['retained_followups'] = sum(s['type'].lower().startswith('follow') for s in slots)
    return {'participant_id': pid, 'split': split, 'slots': slots, 'events': events,
            'position_ids': position_ids, 'question_presence': presence, 'audit': dict(counts)}


def prepare(base, zip_dir, model_dir, device):
    from bert_score import BERTScorer
    import torch
    torch.set_num_threads(4)
    rosters = split_ids(zip_dir)
    roster = {pid: split for split, ids in rosters.items() for pid in ids if pid != 440}
    questions = question_table(Path('Data/hique_reproduction/paper.html'))
    corpus = []
    for pid in sorted(roster):
        path = base / 'asr' / f'{pid}.json'
        data = json.loads(path.read_text())
        if data['participant_id'] != pid:
            raise ValueError('ASR participant mismatch')
        corpus.append((pid, data['rows'], digest(path)))
    strings = sorted({row['text'] for _, rows, _ in corpus for row in rows if row['speaker'] == 'Ellie'})
    fingerprint = hashlib.sha256(json.dumps({'asr': [(p, h) for p, _, h in corpus],
                  'model': json.loads((base / 'bertscore_model.json').read_text()),
                  'source': digest(__file__), 'questions': questions}, sort_keys=True).encode()).hexdigest()
    mapping_path = base / 'question_mapping_private.json'
    if mapping_path.exists():
        saved = json.loads(mapping_path.read_text())
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Mapping cache differs; preserve previous and use separate directory')
        mapping = saved['mapping']
    else:
        scorer = BERTScorer(model_type=str(model_dir.resolve()), num_layers=17, lang='en',
                            device=device, batch_size=8, idf=False, rescale_with_baseline=False,
                            use_fast_tokenizer=False)
        hyps = [s for s in strings for _ in questions]
        refs = [q['question'] for _ in strings for q in questions]
        print(f'BERTScore {len(strings)} distinct pseudo-questions x85; official scorer deduplicates embeddings', flush=True)
        _, _, scores = scorer.score(hyps, refs, batch_size=8, verbose=True)
        values = scores.numpy().reshape(len(strings), 85)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite BERTScore')
        mapping = {s: {'slot': int(np.argmax(v)), 'f1': float(v.max())} for s, v in zip(strings, values)}
        write(mapping_path, {'fingerprint': fingerprint, 'mapping': mapping,
              'hash': scorer.hash, 'model_provenance': json.loads((base / 'bertscore_model.json').read_text())})
        del scorer
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    records = [build_record(pid, roster[pid], rows, mapping, questions) for pid, rows, _ in corpus]
    (base / 'segments.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    write(base / 'questions.json', questions)
    write(base / 'manifest.json', {'participants': [{'participant_id': p, 'split': roster[p], 'asr_sha256': h}
                                                   for p, _, h in corpus], 'excluded_ids': [440]})
    audit: Counter[str] = Counter()
    for r in records:
        audit.update(r['audit'])
    write(base / 'preparation_summary.json', {'participants': len(records), 'unique_questions': len(strings),
          'audit': dict(audit), 'zero_answer_ids': [r['participant_id'] for r in records if not r['slots']],
          'participants_with_changed_positions': sum(r['position_ids'] != list(range(85)) for r in records),
          'changed_positions_total': sum(sum(p != i for i,p in enumerate(r['position_ids'])) for r in records),
          'source_sha256': digest(__file__), 'warning': 'Ellie is punctuation-inferred pseudo-speaker, not verified diarization'})
    print(dict(audit), flush=True)


def text_features(base, model_dir, device):
    import torch
    from transformers import RobertaModel, RobertaTokenizerFast  # type: ignore[attr-defined]
    torch.set_num_threads(4)
    rows = [json.loads(line) for line in (base / 'segments.jsonl').read_text().splitlines()]
    tokenizer = RobertaTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)
    model = RobertaModel.from_pretrained(str(model_dir), local_files_only=True, add_pooling_layer=False).to(device).eval()
    indices, texts = [], []
    for i, row in enumerate(rows):
        for slot in row['slots']:
            if slot['text']:
                indices.append((i, slot['slot']))
                texts.append(slot['text'])
    tokenized = tokenizer(texts, truncation=True, max_length=512, padding=False)
    lengths = [len(ids) for ids in tokenized['input_ids']]
    full_lengths = [len(ids) for ids in tokenizer(texts, truncation=False, verbose=False)['input_ids']]
    feature = np.zeros((len(rows), 85, 768), dtype='float32')
    mask = np.zeros((len(rows), 85), dtype=bool)
    batches = length_batches(lengths, token_budget=2048, max_batch=16)
    with torch.inference_mode():
        for n, batch in enumerate(batches):
            inputs = tokenizer.pad([{k: v[i] for k, v in tokenized.items()} for i in batch],
                                   padding=True, return_tensors='pt').to(device)
            embeddings = model(**inputs).last_hidden_state[:, 0].cpu().numpy()
            if not np.isfinite(embeddings).all():
                raise ValueError('Nonfinite RoBERTa features')
            for original, value in zip(batch, embeddings):
                i, slot = indices[original]
                feature[i, slot] = value
                mask[i, slot] = True
            if n % 30 == 0:
                print(f'Text batch {n+1}/{len(batches)}', flush=True)
    np.savez_compressed(base / 'text.npz', T=feature, mask_T=mask,
                        participant_ids=np.array([r['participant_id'] for r in rows]))
    write(base / 'text_summary.json', {'encoded_slots': len(texts), 'truncated_slots': sum(n > 512 for n in full_lengths),
          'source_sha256': digest(__file__), 'device': device, 'pooling': 'last_hidden_state[:,0]',
          'roberta_base_revision': 'e2da8e2f811d1448a5b465c236feacd80ffbac7b', 'frozen': True})


def assemble(base):
    records = [json.loads(line) for line in (base / 'segments.jsonl').read_text().splitlines()]
    with np.load(base / 'text.npz') as text:
        ids = np.array([r['participant_id'] for r in records])
        np.testing.assert_array_equal(ids, text['participant_ids'])
        arrays = {'T': text['T'], 'mask_T': text['mask_T'], 'participant_ids': ids,
                  'split': np.array([r['split'] for r in records]),
                  'position_ids': np.array([r['position_ids'] for r in records], dtype='int32'),
                  'question_presence': np.array([r['question_presence'] for r in records])}
    for key in ['A', 'V', 'mask_A', 'mask_V']:
        arrays[key] = np.array([np.load(base / 'av' / f'{pid}.npz')[key] for pid in ids])
    eligible = arrays['mask_A'].any(axis=1) & arrays['mask_V'].any(axis=1) & arrays['mask_T'].any(axis=1)
    excluded = ids[~eligible].tolist()
    write(base / 'eligibility.json', {'excluded_input_unavailable': excluded,
          'eligible_counts': {s: int(np.sum(eligible & (arrays['split'] == s))) for s in ['train', 'dev', 'test']},
          'rule': 'atleast one valid slot in each modality; no intersection requirement',
          'coverage': {str(pid): {m: int(arrays['mask_'+m][i].sum()) for m in 'AVT'} for i,pid in enumerate(ids)}})
    np.savez_compressed(base / 'features.npz', **{key: value[eligible] for key, value in arrays.items()})
    print(f'Assembled eligible={int(eligible.sum())}, excluded={excluded}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'text', 'assemble'], required=True)
    parser.add_argument('--base', type=Path, default=Path('Data/hique_reproduction_excluding440'))
    parser.add_argument('--zip-dir', type=Path, default=Path('Data/DAIC-WOZ'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.base = require_private_output(args.base)
    if args.stage == 'prepare':
        prepare(args.base, args.zip_dir, Path('.tmp/hique_weights/roberta-large'), args.device)
    elif args.stage == 'text':
        text_features(args.base, Path('.tmp/hique_weights/roberta-base'), args.device)
    else:
        assemble(args.base)


if __name__ == '__main__':
    main()
