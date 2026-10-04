"""Validated frozen RoBERTa embeddings with exact token coverage and block pooling."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from urop.io import digest, write_json as write

TEXT_VARIANTS = ('E0', 'E1', 'E2', 'E3')


def length_batches(lengths: list[int], token_budget: int = 4096, max_batch: int = 32) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    for index in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        if current and (len(current) >= max_batch or lengths[index] * (len(current) + 1) > token_budget):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches

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
        values = np.zeros((len(records['E3' if variant == 'E4' else variant]), 85, 768), dtype=np.float32)
        mask = np.zeros((len(records['E3' if variant == 'E4' else variant]), 85), dtype=bool)
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
                  'all_unique_strings_parity_count': len(strings),
                  'E0_text_recomputed_correct_BPE': True,
                  'word_count_note': 'Whitespace words before encoding; retention is measured in content tokens, not partial-word estimates.'})
    write(base / 'features/text_audit.json', audit)
