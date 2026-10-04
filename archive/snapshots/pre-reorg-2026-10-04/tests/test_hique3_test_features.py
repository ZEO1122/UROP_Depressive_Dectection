import ast
from pathlib import Path

import numpy as np
import pytest

from experiments import hique3_features as original
from experiments import hique3_test_features as extension


def test_block_policies_identical():
    for length in (1, 509, 510, 511, 1021):
        tokens = list(range(length))
        assert extension.token_blocks(tokens) == original.token_blocks(tokens)
        blocks = extension.token_blocks(tokens)
        assert sum(blocks, []) == tokens
        vectors = [np.full(768, i + 0.123, dtype=np.float32) for i in range(len(blocks))]
        np.testing.assert_array_equal(extension.pool_blocks(vectors, list(map(len, blocks))),
                                      original.pool_blocks(vectors, list(map(len, blocks))))


def test_original_frozen_sources_not_edited_and_shared_tokenizer():
    assert extension.load_tokenizer is original.load_tokenizer
    assert extension.token_blocks is original.token_blocks
    assert extension.pool_blocks is original.pool_blocks
    source = Path(extension.__file__).read_text()
    assert 'full_test_split' not in source
    assert 'monkeypatch' not in source


def test_fixed_cohort_arrays_and_missing_zero():
    arrays = {'participant_ids': np.arange(45), 'split': np.array(['test'] * 45)}
    for modality, width in [('A', 88), ('V', 272), ('T', 768)]:
        arrays[modality] = np.zeros((45, 85, width), dtype=np.float32)
        arrays['mask_' + modality] = np.zeros((45, 85), dtype=bool)
        arrays['mask_' + modality][:, 0] = True
    extension.validate_arrays(arrays)
    arrays['A'][0, 1, 0] = 1
    with pytest.raises(ValueError, match='nonzero missing'):
        extension.validate_arrays(arrays)


def test_encoder_same_computation_ast():
    # New extraction duplicates original orchestration solely to remove hardcoded
    # train/dev count. Every inference/pooling operation must remain identical.
    def expressions(module):
        tree = ast.parse(Path(module.__file__).read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_text')
        selected = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                 {'blocks', 'sequence', 'model', 'ordered', 'encoded', 'batches', 'inputs', 'vectors'} for t in node.targets):
                selected.append(ast.dump(node.value))
        return selected
    assert expressions(extension) == expressions(original)
