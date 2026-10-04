"""Train/dev-only input-study features: fixed cohort, independent A/V, lossless token blocks."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from urop.features.av import extract_one
from urop.features.text import extract_text
from urop.io import digest, write_json as write, require_private_output
from urop.paths import DEFAULT_STUDY, ROOT

VARIANTS = ('E1', 'E2', 'E3')
TEXT_VARIANTS = ('E0', *VARIANTS)
KEYS = ('A', 'V', 'T', 'mask_A', 'mask_V', 'mask_T', 'question_presence')










def read_records(base: Path) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for variant in TEXT_VARIANTS:
        rows = [json.loads(line) for line in (base / f'segments_{variant}.jsonl').read_text().splitlines() if line.strip()]
        rows.sort(key=lambda row: int(row['participant_id']))
        ids = [int(row['participant_id']) for row in rows]
        if len(ids) != 139 or len(set(ids)) != 139 or set(ids) & {440, 451, 458}:
            raise ValueError('fixed 139-person cohort required; never filter by feature availability')
        if {s: sum(row['split'] == s for row in rows) for s in ('train', 'dev', 'test')} != {'train': 107, 'dev': 32, 'test': 0}:
            raise ValueError('train107/dev32 only')
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


def freeze(base: Path, baseline: Path, model_dir: Path) -> dict[str, Any]:
    inputs = [base / f'segments_{v}.jsonl' for v in TEXT_VARIANTS]
    inputs += [base / 'manifest.json']
    inputs += [baseline, *sorted((ROOT / 'src/urop').rglob('*.py'))]
    inputs += sorted(p for p in model_dir.iterdir() if p.is_file())
    payload = {'version': 2, 'input_sha256': {str(p): digest(p) for p in inputs},
               'roberta_revision': 'e2da8e2f811d1448a5b465c236feacd80ffbac7b',
               'tokenizer_policy': 'AutoTokenizer(use_fast=True), validated against serialized tokenizer.json; E0 T recomputed with corrected BPE',
               'text_policy': 'E0/E1/E2/E3 first510 content tokens+BOS/EOS; E4 nonoverlap510 weighted CLS',
               'av_policy': 'urop.features.av.extract_one; independent masks; E4 copies E3',
               'test_used': False,
               'baseline_container_note': 'Existing V2 NPZ includes old test features; only train/dev ID slices are used. No test ZIP, transcript, label or prediction is opened.'}
    path = base / 'features' / 'frozen_inputs.json'
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError('frozen input/source/model changed; use new output directory')
    if not path.exists():
        write(path, payload)
    return payload


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
    if len(arrays['participant_ids']) != 139 or set(arrays['split']) != {'train', 'dev'}:
        raise ValueError('unexpected cohort')
    for modality, width in [('A', 88), ('V', 272), ('T', 768)]:
        values, mask = arrays[modality], arrays['mask_' + modality]
        if values.shape != (139, 85, width) or mask.shape != (139, 85):
            raise ValueError('invalid feature dimensions')
        if not np.isfinite(values).all() or np.any(values[~mask] != 0):
            raise ValueError('nonfinite or nonzero missing features')
        if not mask.any(axis=1).all():
            raise ValueError(f'participant has no {modality} input; do not silently filter cohort')


def assemble(base: Path, records: dict[str, list[dict[str, Any]]], baseline: Path) -> None:
    ids = np.array([r['participant_id'] for r in records['E1']], dtype=np.int32)
    splits = np.array([r['split'] for r in records['E1']])
    common = {'participant_ids': ids, 'split': splits, 'position_ids': np.tile(np.arange(85, dtype=np.int32), (139, 1))}
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
        presence = np.zeros((139, 85), dtype=bool)
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
    audit['cohort'] = {'train': 107, 'dev': 32, 'test': 0, 'participant_ids': ids.tolist()}
    write(base / 'features/assembly_audit.json', audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--baseline', type=Path, default=Path('Data/hique_reproduction_excluding440/features.npz'))
    parser.add_argument('--model-dir', type=Path, default=Path('.tmp/hique_weights/roberta-base'))
    parser.add_argument('--zip-dir', type=Path, default=Path('Data/DAIC-WOZ'))
    parser.add_argument('--stage', choices=['av', 'text', 'assemble', 'all'], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.base = require_private_output(args.base)
    (args.base / 'features').mkdir(parents=True, exist_ok=True)
    records = read_records(args.base)
    freeze(args.base, args.baseline, args.model_dir)
    if args.stage in ('av', 'all'):
        extract_av(args.base, records, args.zip_dir, args.workers)
    if args.stage in ('text', 'all'):
        extract_text(args.base, records, args.model_dir, args.device)
    if args.stage in ('assemble', 'all'):
        assemble(args.base, records, args.baseline)


if __name__ == '__main__':
    main()
