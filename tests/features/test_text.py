import numpy as np
import pytest

from urop.features.text import pool_blocks, token_blocks
from urop.features.pipeline import validate_arrays


@pytest.mark.parametrize('length', [0, 1, 509, 510, 511, 1020, 1021, 4097])
def test_nonoverlap_blocks_preserve_every_token(length):
    tokens = list(range(length))
    blocks = token_blocks(tokens)
    assert sum(blocks, []) == tokens
    assert all(0 < len(block) <= 510 for block in blocks)


def test_short_pool_is_bit_identical():
    value = np.random.default_rng(42).normal(size=768).astype(np.float32)
    assert pool_blocks([value], [510]).tobytes() == value.tobytes()


def test_pool_weights_content_length():
    result = pool_blocks([np.array([2, 4], dtype=np.float32), np.array([8, 10], dtype=np.float32)], [510, 170])
    np.testing.assert_array_equal(result, [3.5, 5.5])


def test_cohort_never_silently_filters():
    arrays = {'participant_ids': np.arange(139), 'split': np.array(['train'] * 107 + ['dev'] * 32)}
    for m, width in [('A', 88), ('V', 272), ('T', 768)]:
        arrays[m] = np.zeros((139, 85, width), dtype=np.float32)
        arrays['mask_'+m] = np.ones((139, 85), dtype=bool)
    validate_arrays(arrays)
    arrays['mask_V'][0] = False
    with pytest.raises(ValueError, match='do not silently filter'):
        validate_arrays(arrays)


def test_records_match_manifest_and_reject_test(tmp_path):
    import json
    from urop.features.pipeline import read_records

    rows = [{'participant_id': i, 'split': 'train' if i < 107 else 'dev',
             'slots': [{'slot': 1, 'text': 'answer', 'spans': [[1, 2]]}]} for i in range(139)]
    for variant in ('E0', 'E1', 'E2', 'E3'):
        (tmp_path / f'segments_{variant}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path / 'manifest.json').write_text(json.dumps({'participants': rows}))
    assert len(read_records(tmp_path)['E1']) == 139
    rows[-1]['split'] = 'test'
    (tmp_path / 'segments_E1.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='train107/dev32 only'):
        read_records(tmp_path)


def test_roberta_tokenizer_serialized_parity_and_word_spaces():
    from pathlib import Path
    from tokenizers import Tokenizer
    from urop.features.text import load_tokenizer

    model_dir = Path('.tmp/hique_weights/roberta-base')
    if not (model_dir / 'tokenizer.json').exists():
        pytest.skip('local frozen tokenizer fixture unavailable')
    tokenizer = load_tokenizer(model_dir)
    reference = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
    assert tokenizer('Hello world, this is a simple test.')['input_ids'] == [0, 31414, 232, 6, 42, 16, 10, 2007, 1296, 4, 2]
    spaced = tokenizer('hello world', add_special_tokens=False)['input_ids']
    joined = tokenizer('helloworld', add_special_tokens=False)['input_ids']
    assert spaced != joined
    assert spaced == reference.encode('hello world', add_special_tokens=False).ids
    long_text = 'Hello world, this is a simple test. ' * 80
    content = tokenizer(long_text, add_special_tokens=False)['input_ids']
    assert [tokenizer.bos_token_id, *content[:510], tokenizer.eos_token_id] == tokenizer(long_text, truncation=True, max_length=512)['input_ids']
    assert sum(token_blocks(content), []) == reference.encode(long_text, add_special_tokens=False).ids
    sequence = [tokenizer.bos_token_id, *spaced, tokenizer.eos_token_id]
    padded = tokenizer.pad([{'input_ids': sequence, 'attention_mask': [1] * len(sequence)},
                            {'input_ids': [0, 2], 'attention_mask': [1, 1]}],
                           padding=True, return_tensors='pt')
    assert padded['input_ids'].shape == (2, len(sequence))
    assert padded['attention_mask'][1].tolist() == [1, 1] + [0] * (len(sequence) - 2)


@pytest.mark.parametrize('corruption', [None, 'content', 'special'])
def test_extract_text_generic_cohort_deduplicates_and_preserves_short_blocks(tmp_path, monkeypatch, corruption):
    """Exercise orchestration without model downloads or participant data."""
    import json
    import torch
    import transformers
    import tokenizers
    from types import SimpleNamespace
    from urop.features import text

    calls = []

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        bos_token_id, eos_token_id = 0, 2

        def __call__(self, values, **kwargs):
            def encode(value):
                ids = [int(word) for word in value.split()]
                if corruption == 'content' and len(ids) > 510:
                    ids[-1] = 20
                if kwargs.get('add_special_tokens', True):
                    ids = [self.bos_token_id, *ids, self.eos_token_id]
                    if corruption == 'special' and len(ids) > 512:
                        ids[-1] = 20
                return ids
            return {'input_ids': encode(values) if isinstance(values, str) else [encode(value) for value in values]}

        def pad(self, rows, **kwargs):
            size = max(len(row['input_ids']) for row in rows)
            return Batch({key: torch.tensor([row[key] + [0] * (size - len(row[key])) for row in rows])
                          for key in ('input_ids', 'attention_mask')})

    class Model:
        frozen = False

        def to(self, device):
            return self

        def eval(self):
            self.frozen = True
            return self

        def __call__(self, **inputs):
            assert self.frozen and not torch.is_grad_enabled()
            calls.extend(tuple(row[mask.bool()].tolist()) for row, mask in
                         zip(inputs['input_ids'], inputs['attention_mask']))
            # Context-dependent deterministic output stands in for frozen CLS.
            vectors = inputs['input_ids'].sum(axis=1).float()[:, None, None]
            return SimpleNamespace(last_hidden_state=vectors.expand(-1, inputs['input_ids'].shape[1], 768))

    monkeypatch.setattr(text, 'load_tokenizer', lambda path: Tokenizer())
    monkeypatch.setattr(transformers.RobertaModel, 'from_pretrained', lambda *a, **kw: Model())
    # Independent Rust tokenizer supplies the expected IDs, not the fake encoder.
    vocabulary = {str(i): i for i in range(21)}
    reference = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token='1'))
    reference.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    reference.post_processor = tokenizers.processors.TemplateProcessing(
        single='0 $A 2', special_tokens=[('0', 0), ('2', 2)])
    reference.save(str(tmp_path / 'tokenizer.json'))
    rows = [{'participant_id': 1, 'slots': [{'slot': 1, 'text': '10 20'},
                                          {'slot': 2, 'text': ' '.join(['3'] * 511)}]},
            {'participant_id': 2, 'slots': [{'slot': 1, 'text': '10 20'}]}]
    (tmp_path / 'features').mkdir()
    if corruption is not None:
        with pytest.raises(ValueError, match='All-string .* parity failed'):
            text.extract_text(tmp_path, {v: rows for v in text.TEXT_VARIANTS}, tmp_path, 'cpu')
        assert not calls
        assert not list((tmp_path / 'features').iterdir())
        return
    text.extract_text(tmp_path, {v: rows for v in text.TEXT_VARIANTS}, tmp_path, 'cpu')
    assert len(calls) == len(set(calls)) == 3
    with np.load(tmp_path / 'features/text_E3.npz') as e3, np.load(tmp_path / 'features/text_E4.npz') as e4:
        assert e3['T'].shape == e4['T'].shape == (2, 85, 768)
        np.testing.assert_array_equal(e3['T'][:, 1], e4['T'][:, 1])
        np.testing.assert_array_equal(e3['mask_T'], e4['mask_T'])
        np.testing.assert_array_equal(e4['T'][~e4['mask_T']], 0)
        assert not np.array_equal(e3['T'][0, 2], e4['T'][0, 2])
    audit = json.loads((tmp_path / 'features/text_audit.json').read_text())
    assert audit['unique_strings'] == 2
    assert audit['all_unique_strings_parity_count'] == 2
    assert audit['E4']['content_tokens_retained'] == audit['E4']['content_tokens_before']
