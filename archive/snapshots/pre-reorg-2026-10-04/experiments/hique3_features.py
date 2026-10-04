"""Train/dev-only V3 features: fixed cohort, independent A/V, lossless token blocks."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.hique2_av import extract_one
from experiments.hique_data import length_batches

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


def token_blocks(tokens: list[int], size: int = 510) -> list[list[int]]:
    if size <= 0:
        raise ValueError('positive block size required')
    result = [tokens[i:i + size] for i in range(0, len(tokens), size)]
    if [token for block in result for token in block] != tokens:
        raise ValueError('token coverage failure')
    return result


def pool_blocks(values: list[np.ndarray], lengths: list[int]) -> np.ndarray:
    if not values or len(values) != len(lengths) or min(lengths) <= 0:
        raise ValueError('nonempty blocks with positive content weights required')
    if len(values) == 1:
        return values[0].copy()  # Bit-identical to E3, without multiply/divide rounding.
    return np.average(np.stack(values).astype(np.float64), axis=0,
                      weights=np.asarray(lengths, dtype=np.float64)).astype(np.float32)


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
    inputs += [baseline, Path(__file__), Path('experiments/hique2_av.py'), Path('experiments/hique_av.py')]
    inputs += sorted(p for p in model_dir.iterdir() if p.is_file())
    payload = {'version': 2, 'input_sha256': {str(p): digest(p) for p in inputs},
               'roberta_revision': 'e2da8e2f811d1448a5b465c236feacd80ffbac7b',
               'tokenizer_policy': 'AutoTokenizer(use_fast=True), validated against serialized tokenizer.json; E0 T recomputed with corrected BPE',
               'text_policy': 'E0/E1/E2/E3 first510 content tokens+BOS/EOS; E4 nonoverlap510 weighted CLS',
               'av_policy': 'unchanged hique2_av.extract_one; independent masks; E4 copies E3',
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


def load_tokenizer(model_dir: Path) -> Any:
    """Reject silent character-tokenizer fallback using independent serialized BPE."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    serialized = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
    probe = 'Hello world, this is a simple test.'
    expected = [0, 31414, 232, 6, 42, 16, 10, 2007, 1296, 4, 2]
    if tokenizer(probe)['input_ids'] != expected:
        raise ValueError('RoBERTa BPE regression: known-text IDs differ')
    for value in (probe, 'hello world', 'helloworld', ' hello  world', "I am not sure.\nHow are you?"):
        for special in (False, True):
            actual = tokenizer(value, add_special_tokens=special)['input_ids']
            reference = serialized.encode(value, add_special_tokens=special).ids
            if actual != reference:
                raise ValueError('AutoTokenizer disagrees with serialized tokenizer.json')
        content = tokenizer(value, add_special_tokens=False)['input_ids']
        if [tokenizer.bos_token_id, *content, tokenizer.eos_token_id] != tokenizer(value)['input_ids']:
            raise ValueError('special-token construction mismatch')
    return tokenizer


def extract_text(base: Path, records: dict[str, list[dict[str, Any]]], model_dir: Path, device: str) -> None:
    import torch
    from transformers import RobertaModel

    torch.set_num_threads(4)
    tokenizer = load_tokenizer(model_dir)
    # Deduplicate whole strings then exact token sequences across every condition.
    strings = sorted({s['text'] for rows in records.values() for r in rows for s in r['slots'] if s['text'].strip()})
    content = tokenizer(strings, add_special_tokens=False, truncation=False, verbose=False)['input_ids']
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
        values = np.zeros((139, 85, 768), dtype=np.float32)
        mask = np.zeros((139, 85), dtype=bool)
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
                  'tokenizer_class': type(tokenizer).__name__, 'tokenizer_serialized_parity_verified': True,
                  'E0_text_recomputed_correct_BPE': True,
                  'word_count_note': 'Whitespace words before encoding; retention is measured in content tokens, not partial-word estimates.'})
    write(base / 'features/text_audit.json', audit)


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
    parser.add_argument('--base', type=Path, default=Path('Data/hique_input_study_v3'))
    parser.add_argument('--baseline', type=Path, default=Path('Data/hique_reproduction_excluding440/features.npz'))
    parser.add_argument('--model-dir', type=Path, default=Path('.tmp/hique_weights/roberta-base'))
    parser.add_argument('--zip-dir', type=Path, default=Path('Data/DAIC-WOZ'))
    parser.add_argument('--stage', choices=['av', 'text', 'assemble', 'all'], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not args.base.resolve().is_relative_to(Path('Data').resolve()):
        raise ValueError('participant features must stay under ignored Data/')
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
