import numpy as np
import pytest

from experiments.hique3_features import pool_blocks, token_blocks, validate_arrays


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
    from experiments.hique3_features import read_records

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
    from experiments.hique3_features import load_tokenizer

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
