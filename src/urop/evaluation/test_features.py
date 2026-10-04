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

from urop.features.av import extract_one
from urop.features.text import extract_text
from urop.data.transcripts import read_source, build_record, select_record
from urop.io import digest, write_json as write, require_private_output
from urop.paths import DEFAULT_STUDY, DATA_ROOT

VARIANTS = ('E1', 'E2', 'E3')
TEXT_VARIANTS = ('E0', *VARIANTS)
KEYS = ('A', 'V', 'T', 'mask_A', 'mask_V', 'mask_T', 'question_presence')


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
    from urop.features import text, av
    from urop.data import transcripts, questions, intervals
    from urop import io, paths
    inputs += [Path(m.__file__) for m in (text, av, transcripts, questions, intervals, io, paths) if m.__file__]
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


def prepare(base: Path, training: Path, root: Path,
            questions_path: Path = DATA_ROOT / 'hique_reproduction/questions.json',
            baseline_segments: Path = DATA_ROOT / 'hique_reproduction_excluding440/segments.jsonl') -> None:
    verify_training(training)
    if not (base / 'test_protocol.json').exists():
        raise ValueError('Parent must freeze evaluation protocol first')
    ids = test_ids(root)
    payload = json.loads((training / 'mapping.json').read_text())
    mapping = payload.get('mapping', payload)
    questions = json.loads(questions_path.read_text())
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
    old = [json.loads(line) for line in baseline_segments.read_text().splitlines() if line.strip()]
    lookup = {int(r['participant_id']): r for r in old}
    selected = [lookup[pid] for pid in ids]
    if any(r['split'] != 'test' or not r['slots'] for r in selected):
        raise ValueError('Invalid frozen E0 test records')
    save_rows('segments_E0.jsonl', selected)
    write(base / 'manifest.json', {'participants': manifest, 'mapping_sha256': digest(training / 'mapping.json'),
                                  'questions_sha256': digest(questions_path),
                                  'baseline_segments_sha256': digest(baseline_segments),
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
    parser.add_argument('--base', type=Path, default=DEFAULT_STUDY / 'test_evaluation')
    parser.add_argument('--training', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--baseline', type=Path, default=Path('Data/hique_reproduction_excluding440/features.npz'))
    parser.add_argument('--questions', type=Path, default=DATA_ROOT / 'hique_reproduction/questions.json')
    parser.add_argument('--baseline-segments', type=Path, default=DATA_ROOT / 'hique_reproduction_excluding440/segments.jsonl')
    parser.add_argument('--model-dir', type=Path, default=Path('.tmp/hique_weights/roberta-base'))
    parser.add_argument('--zip-dir', type=Path, default=DATA_ROOT / 'DAIC-WOZ')
    parser.add_argument('--stage', choices=['prepare', 'av', 'text', 'assemble', 'all'], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.base = require_private_output(args.base)
    (args.base / 'features').mkdir(parents=True, exist_ok=True)
    if args.stage in ('prepare', 'all'):
        prepare(args.base, args.training, args.zip_dir, args.questions, args.baseline_segments)
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
