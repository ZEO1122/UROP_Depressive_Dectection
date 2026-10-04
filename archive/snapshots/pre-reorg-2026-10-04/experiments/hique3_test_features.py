"""Frozen V3 test feature extension; never reads outcome labels."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.hique2_av import extract_one
from experiments.hique_data import length_batches
from experiments.hique3_features import load_tokenizer, token_blocks, pool_blocks
from experiments.hique3_prepare import read_source, build_record, select_record

VARIANTS = ('E1', 'E2', 'E3')
TEXT_VARIANTS = ('E0', *VARIANTS)
KEYS = ('A', 'V', 'T', 'mask_A', 'mask_V', 'mask_T', 'question_presence')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def read_records(base: Path) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for variant in TEXT_VARIANTS:
        rows = [json.loads(line) for line in (base / f'segments_{variant}.jsonl').read_text().splitlines() if line.strip()]
        rows.sort(key=lambda row: int(row['participant_id']))
        ids = [int(row['participant_id']) for row in rows]
        if len(ids) != 45 or len(set(ids)) != 45 or set(ids) & {440, 411, 480}:
            raise ValueError('fixed 45-person cohort required; never filter by feature availability')
        if {s: sum(row['split'] == s for row in rows) for s in ('train', 'dev', 'test')} != {'train': 0, 'dev': 0, 'test': 45}:
            raise ValueError('test45 only')
        for row in rows:
            slots = [int(s['slot']) for s in row['slots']]
            if len(slots) != len(set(slots)) or any(s < 0 or s >= 85 for s in slots):
                raise ValueError('duplicate or out-of-range slots')
        result[variant] = rows
    reference = [(r['participant_id'], r['split']) for r in result['E1']]
    manifest = json.loads((base / 'manifest.json').read_text())
    expected = sorted((int(r['participant_id']), r['split']) for r in manifest['participants'])
    if reference != expected:
        raise ValueError('segment IDs differ from frozen preparation manifest')
    if any([(r['participant_id'], r['split']) for r in rows] != reference for rows in result.values()):
        raise ValueError('condition cohorts differ')
    return result


def verify_training(training: Path) -> None:
    frozen = json.loads((training / 'runs/frozen_config.json').read_text())
    features = json.loads((training / 'features/frozen_inputs.json').read_text())
    for group in (frozen['input_file_hashes'], frozen['source_hashes'], features['input_sha256']):
        for name, expected in group.items():
            if digest(Path(name)) != expected:
                raise ValueError(f'Original training input changed: {name}')


def freeze(base: Path, baseline: Path, model_dir: Path, training: Path) -> dict[str, Any]:
    verify_training(training)
    inputs = [base / f'segments_{v}.jsonl' for v in TEXT_VARIANTS]
    inputs += [base / 'manifest.json', base / 'test_protocol.json', training / 'mapping.json',
               training / 'runs/frozen_config.json', baseline, Path(__file__)]
    inputs += sorted(p for p in model_dir.iterdir() if p.is_file())
    payload = {'input_sha256': {str(p): digest(p) for p in inputs},
               'training_verified': True, 'test_labels_read': False,
               'text_policy': 'same frozen V3 corrected BPE and CLS; E4 weighted blocks',
               'av_policy': 'same frozen extract_one; independent masks; E4 copies E3'}
    target = base / 'features/frozen_inputs.json'
    if target.exists() and json.loads(target.read_text()) != payload:
        raise ValueError('Frozen test inputs changed')
    if not target.exists():
        write(target, payload)
    return payload


def test_ids(root: Path) -> list[int]:
    with (root / 'test_split_Depression_AVEC2017.csv').open() as stream:
        rows = csv.DictReader(stream)
        ids = [int(r['participant_ID']) for r in rows if r['participant_ID'].strip()]
    if len(ids) != 47 or len(set(ids)) != 47:
        raise ValueError('Official test roster must have47 distinct IDs')
    result = sorted(set(ids) - {411, 480})
    if len(result) != 45 or 300 not in result or 440 in result:
        raise ValueError('Unexpected frozen common test cohort')
    return result


def prepare(base: Path, training: Path, root: Path) -> None:
    verify_training(training)
    if not (base / 'test_protocol.json').exists():
        raise ValueError('Parent must freeze evaluation protocol first')
    ids = test_ids(root)
    payload = json.loads((training / 'mapping.json').read_text())
    mapping = payload.get('mapping', payload)
    questions = json.loads(Path('Data/hique_reproduction/questions.json').read_text())
    events, manifest = [], []
    for pid in ids:
        rows, duration, sha = read_source(root, pid)
        record = build_record(pid, 'test', rows, duration, mapping, questions)
        events.append(record)
        manifest.append({'participant_id': pid, 'split': 'test', 'transcript_sha256': sha})
    def save_rows(name: str, rows: list[dict[str, Any]]) -> None:
        (base / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    save_rows('events.jsonl', events)
    for variant, policy in [('E1', 'last'), ('E2', 'first'), ('E3', 'all')]:
        selected = [select_record(r, policy) for r in events]
        if any(not r['slots'] for r in selected):
            raise ValueError('No usable slot: no silent test exclusion allowed')
        save_rows(f'segments_{variant}.jsonl', selected)
    old = [json.loads(line) for line in Path('Data/hique_reproduction_excluding440/segments.jsonl').read_text().splitlines() if line.strip()]
    lookup = {int(r['participant_id']): r for r in old}
    selected = [lookup[pid] for pid in ids]
    if any(r['split'] != 'test' or not r['slots'] for r in selected):
        raise ValueError('Invalid frozen E0 test records')
    save_rows('segments_E0.jsonl', selected)
    write(base / 'manifest.json', {'participants': manifest, 'mapping_sha256': digest(training / 'mapping.json'),
                                  'excluded': {'411': 'No V2 ASR mapped question', '480': 'No provided Ellie transcript'}})
    counts: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    for record in events:
        counts.update(record['audit'])
        unknown.update(e['normalized_question'] for e in record['events'] if e['mapping_status'] == 'unknown')
    write(base / 'preparation_summary.json', {'participants': len(ids), 'source_row_classifications': dict(counts),
        'unknown_question_occurrences': sum(unknown.values()), 'unknown_question_types': len(unknown),
        'unknown_questions_private': dict(unknown), 'test_labels_read': False, 'mapping_updated': False})


def extract_av(base: Path, records: dict[str, list[dict[str, Any]]], zip_dir: Path, workers: int) -> None:
    for variant in VARIANTS:
        rows = records[variant]
        output = base / 'features' / f'av_{variant}'
        output.mkdir(parents=True, exist_ok=True)
        tasks = [(r, str(zip_dir), str(output)) for r in rows]
        summaries = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for summary in pool.map(extract_one, tasks):
                summaries.append(summary)
                print(f'{variant} AV {len(summaries)}/{len(rows)} {summary}', flush=True)
        write(output / 'summary.json', summaries)


def extract_text(base: Path, records: dict[str, list[dict[str, Any]]], model_dir: Path, device: str) -> None:
    import torch
    from transformers import RobertaModel

    torch.set_num_threads(4)
    tokenizer = load_tokenizer(model_dir)
    # Deduplicate whole strings then exact token sequences across every condition.
    strings = sorted({s['text'] for rows in records.values() for r in rows for s in r['slots'] if s['text'].strip()})
    content = tokenizer(strings, add_special_tokens=False, truncation=False, verbose=False)['input_ids']
    from tokenizers import Tokenizer
    serialized = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
    for value, actual in zip(strings, content):
        if actual != serialized.encode(value, add_special_tokens=False).ids:
            raise ValueError('All-string serialized tokenizer parity failed')
        if tokenizer(value)['input_ids'] != serialized.encode(value).ids:
            raise ValueError('All-string special-token parity failed')
    tokens_by_text = dict(zip(strings, content))
    sequences: dict[tuple[int, ...], int] = {}
    plans: dict[str, list[tuple[int, int, list[int], list[int]]]] = {v: [] for v in (*TEXT_VARIANTS, 'E4')}
    audit: dict[str, Any] = {}
    for variant in plans:
        rows = records['E3' if variant == 'E4' else variant]
        stats = {'slots': 0, 'words_before': 0, 'content_tokens_before': 0, 'content_tokens_retained': 0,
                 'full_coverage_slots': 0, 'long_slots': 0, 'empty_token_slots': 0}
        for i, row in enumerate(rows):
            for slot in row['slots']:
                value = slot['text']
                if not value.strip():
                    continue
                tokens = tokens_by_text[value]
                if not tokens:
                    stats['empty_token_slots'] += 1
                    continue
                blocks = token_blocks(tokens) if variant == 'E4' else [tokens[:510]]
                indices = []
                for block in blocks:
                    sequence = tuple([tokenizer.bos_token_id, *block, tokenizer.eos_token_id])
                    if len(sequence) > 512:
                        raise ValueError('RoBERTa block exceeds512')
                    indices.append(sequences.setdefault(sequence, len(sequences)))
                plans[variant].append((i, int(slot['slot']), indices, [len(b) for b in blocks]))
                stats['slots'] += 1
                stats['words_before'] += len(value.split())
                stats['content_tokens_before'] += len(tokens)
                stats['content_tokens_retained'] += sum(map(len, blocks))
                stats['full_coverage_slots'] += int(variant == 'E4' or len(tokens) <= 510)
                stats['long_slots'] += int(len(tokens) > 510)
        audit[variant] = stats
    model = RobertaModel.from_pretrained(str(model_dir), local_files_only=True, add_pooling_layer=False).to(device).eval()  # type: ignore[arg-type]
    ordered = list(sequences)
    encoded = np.zeros((len(ordered), 768), dtype=np.float32)
    batches = length_batches([len(x) for x in ordered], token_budget=4096, max_batch=32)
    with torch.inference_mode():
        for n, batch in enumerate(batches):
            inputs = tokenizer.pad([{'input_ids': list(ordered[k]), 'attention_mask': [1] * len(ordered[k])} for k in batch],
                                   padding=True, return_tensors='pt').to(device)
            vectors = model(**inputs).last_hidden_state[:, 0].cpu().numpy()
            if not np.isfinite(vectors).all():
                raise ValueError('nonfinite RoBERTa')
            encoded[batch] = vectors
            if n % 20 == 0:
                print(f'Text {n+1}/{len(batches)} unique_blocks={len(ordered)}', flush=True)
    for variant, entries in plans.items():
        values = np.zeros((45, 85, 768), dtype=np.float32)
        mask = np.zeros((45, 85), dtype=bool)
        for i, s, indices, lengths in entries:
            values[i, s] = pool_blocks([encoded[k] for k in indices], lengths)
            mask[i, s] = True
        np.savez_compressed(base / 'features' / f'text_{variant}.npz', T=values, mask_T=mask)
    with np.load(base / 'features/text_E3.npz') as e3, np.load(base / 'features/text_E4.npz') as e4:
        # NpzFile access decompresses the complete member; materialize once.
        e3_text, e4_text = e3['T'], e4['T']
        for i, s, indices, _ in plans['E4']:
            if len(indices) == 1:
                np.testing.assert_array_equal(e3_text[i, s], e4_text[i, s])
        np.testing.assert_array_equal(e3['mask_T'], e4['mask_T'])
    audit.update({'device': device, 'unique_strings': len(strings), 'unique_encoded_blocks': len(ordered),
                  'source_sha256': digest(Path(__file__)), 'short_E3_E4_bit_identical': True,
                  'content_coverage_exact': True, 'frozen_model': True,
                  'all_unique_test_strings_parity_count': len(strings), 'tokenizer_class': type(tokenizer).__name__, 'tokenizer_serialized_parity_verified': True,
                  'E0_text_recomputed_correct_BPE': True,
                  'word_count_note': 'Whitespace words before encoding; retention is measured in content tokens, not partial-word estimates.'})
    write(base / 'features/text_audit.json', audit)


def validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    if len(arrays['participant_ids']) != 45 or set(arrays['split']) != {'test'}:
        raise ValueError('unexpected cohort')
    for modality, width in [('A', 88), ('V', 272), ('T', 768)]:
        values, mask = arrays[modality], arrays['mask_' + modality]
        if values.shape != (45, 85, width) or mask.shape != (45, 85):
            raise ValueError('invalid feature dimensions')
        if not np.isfinite(values).all() or np.any(values[~mask] != 0):
            raise ValueError('nonfinite or nonzero missing features')
        if not mask.any(axis=1).all():
            raise ValueError(f'participant has no {modality} input; do not silently filter cohort')


def assemble(base: Path, records: dict[str, list[dict[str, Any]]], baseline: Path) -> None:
    ids = np.array([r['participant_id'] for r in records['E1']], dtype=np.int32)
    splits = np.array([r['split'] for r in records['E1']])
    common = {'participant_ids': ids, 'split': splits, 'position_ids': np.tile(np.arange(85, dtype=np.int32), (45, 1))}
    with np.load(baseline, allow_pickle=False) as old:
        lookup = {int(pid): i for i, pid in enumerate(old['participant_ids'])}
        indices = [lookup[int(pid)] for pid in ids]
        np.testing.assert_array_equal(old['split'][indices], splits)
        arrays = {**common, **{key: old[key][indices].copy() for key in KEYS if key not in ('T', 'mask_T')}}
    with np.load(base / 'features/text_E0.npz') as corrected:
        arrays.update({key: corrected[key].copy() for key in ('T', 'mask_T')})
    validate_arrays(arrays)
    np.savez_compressed(base / 'features/E0.npz', **arrays)
    audit: dict[str, Any] = {'E0': {'exact_v2_subset': False, 'A_V_masks_presence_exact_v2_subset': True,
                                     'T_mask_T_recomputed': 'same selected ASR text, corrected RoBERTa BPE tokenizer'}}
    for variant in (*VARIANTS, 'E4'):
        origin = 'E3' if variant == 'E4' else variant
        arrays = dict(common)
        with np.load(base / 'features' / f'text_{variant}.npz') as text:
            arrays.update({key: text[key].copy() for key in ('T', 'mask_T')})
        collected: dict[str, list[np.ndarray]] = {key: [] for key in ('A', 'V', 'mask_A', 'mask_V')}
        for pid in ids:
            with np.load(base / 'features' / f'av_{origin}' / f'{pid}.npz') as av:
                for key in collected:
                    collected[key].append(av[key].copy())
        arrays.update({key: np.stack(value) for key, value in collected.items()})
        presence = np.zeros((45, 85), dtype=bool)
        for i, row in enumerate(records[origin]):
            for slot in row['slots']:
                presence[i, int(slot['slot'])] = True
        arrays['question_presence'] = presence
        validate_arrays(arrays)
        np.savez_compressed(base / 'features' / f'{variant}.npz', **arrays)
        audit[variant] = {'modality_slots': {m: int(arrays['mask_'+m].sum()) for m in 'AVT'}}
    with np.load(base / 'features/E3.npz') as e3, np.load(base / 'features/E4.npz') as e4:
        hashes = {}
        for key in ('A', 'V', 'mask_A', 'mask_V', 'mask_T', 'question_presence'):
            e3_value, e4_value = e3[key], e4[key]
            np.testing.assert_array_equal(e3_value, e4_value)
            hashes[key] = hashlib.sha256(e3_value.tobytes()).hexdigest()
        audit['E3_E4_identical_arrays_sha256'] = hashes
    audit['files_sha256'] = {v: digest(base / 'features' / f'{v}.npz') for v in ('E0', *VARIANTS, 'E4')}
    audit['cohort'] = {'train': 0, 'dev': 0, 'test': 45, 'participant_ids': ids.tolist()}
    write(base / 'features/assembly_audit.json', audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('Data/hique_input_study_v3/test_evaluation'))
    parser.add_argument('--training', type=Path, default=Path('Data/hique_input_study_v3'))
    parser.add_argument('--baseline', type=Path, default=Path('Data/hique_reproduction_excluding440/features.npz'))
    parser.add_argument('--model-dir', type=Path, default=Path('.tmp/hique_weights/roberta-base'))
    parser.add_argument('--zip-dir', type=Path, default=Path('Data/DAIC-WOZ'))
    parser.add_argument('--stage', choices=['prepare', 'av', 'text', 'assemble', 'all'], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not args.base.resolve().is_relative_to(Path('Data').resolve()):
        raise ValueError('participant features must stay under ignored Data/')
    (args.base / 'features').mkdir(parents=True, exist_ok=True)
    if args.stage in ('prepare', 'all'):
        prepare(args.base, args.training, args.zip_dir)
    if args.stage == 'prepare':
        return
    records = read_records(args.base)
    freeze(args.base, args.baseline, args.model_dir, args.training)
    if args.stage in ('av', 'all'):
        extract_av(args.base, records, args.zip_dir, args.workers)
    if args.stage in ('text', 'all'):
        extract_text(args.base, records, args.model_dir, args.device)
    if args.stage in ('assemble', 'all'):
        assemble(args.base, records, args.baseline)


if __name__ == '__main__':
    main()
