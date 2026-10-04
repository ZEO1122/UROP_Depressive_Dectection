from argparse import Namespace
import json

import numpy as np
import pytest

from experiments import hique2_train as study


def feature_fixture(tmp_path):
    for split, pid in [('train', 301), ('dev', 451), ('test', 300)]:
        (tmp_path / f'{split}_split_Depression_AVEC2017.csv').write_text(f'Participant_ID\n{pid}\n')
    data = {'participant_ids': np.array([301, 451, 300]), 'split': np.array(['train', 'dev', 'test']),
            'position_ids': np.tile(np.arange(85, dtype=np.int32), (3, 1)), 'question_presence': np.ones((3, 85), bool)}
    data['position_ids'][:, 20] = 3
    for m, dim in [('A', 88), ('V', 272), ('T', 768)]:
        data[m] = np.full((3, 85, dim), 7, np.float32)
        data['mask_' + m] = np.ones((3, 85), bool)
    return data


def test_own_mask_only_no_scaler_or_old_exclusions(tmp_path):
    data = feature_fixture(tmp_path)
    data['mask_V'][:, 2] = False
    path = tmp_path / 'features.npz'
    np.savez(path, **data)
    loaded = study.load_features(path, tmp_path)
    assert loaded['participant_ids'].tolist() == [301, 451, 300]
    assert np.all(loaded['V'][:, 2] == 0)
    assert np.all(loaded['A'] == 7) and np.all(loaded['T'] == 7)
    assert 'common_mask' not in loaded
    data['participant_ids'][1] = 440
    np.savez(path, **data)
    with pytest.raises(ValueError, match='excluded'):
        study.load_features(path, tmp_path)


def test_h_f_inputs_share_features_and_augmentation_positions_unchanged(tmp_path):
    data = feature_fixture(tmp_path)
    index = np.array([True, True, True])
    labels = np.array([0, 1, 1])
    h, hy, log = study.augmented_inputs(data, index, 'H_AVT', labels, 42)
    f, fy, flog = study.augmented_inputs(data, index, 'F_AVT', labels, 42)
    assert log == flog
    np.testing.assert_array_equal(hy, fy)
    for a, b in zip(h[:-1], f[:-1]):
        np.testing.assert_array_equal(a, b)
    assert np.all(h[-1][:, 20] == 3) and np.all(f[-1][:, 20] == 20)
    for i in range(3, 7):
        assert np.sum(h[0][i, :, 0] == 0) == 10
        np.testing.assert_array_equal(h[0][i, :, 0] == 0, h[2][i, :, 0] == 0)
    assert np.all(data['A'] == 7)


def test_metric_set_and_ci_is_mean_seed_not_ensemble():
    y = np.array([0, 0, 1, 1])
    a = np.array([[.1, .1, .9, .9], [.9, .9, .1, .1]])
    b = np.tile([.1, .9, .1, .9], (2, 1))
    result = study.paired_bootstrap(y, a, b, repeats=50)
    assert result['delta_macro_f1'] == 0
    assert result == study.paired_bootstrap(y, a, b, repeats=50)
    m = study.metrics(y, b[0])
    for key in ['macro_precision', 'macro_recall', 'macro_f1', 'weighted_precision', 'weighted_recall', 'weighted_f1', 'gmean']:
        assert m[key] == .5


def test_hash_tracks_bytes_and_extra_provenance(tmp_path):
    names = ['train_split_Depression_AVEC2017.csv', 'dev_split_Depression_AVEC2017.csv', 'test_split_Depression_AVEC2017.csv', 'full_test_split.csv', 'protocol.json', 'manifest.json', 'segments.jsonl']
    for name in names:
        (tmp_path / name).write_text('bytes, not parseable labels')
    args = Namespace(features=tmp_path / 'features.npz', zip_dir=tmp_path, provenance_file=[tmp_path / 'segments.jsonl'])
    before = study.input_hashes(args)
    (tmp_path / 'segments.jsonl').write_text('changed')
    after = study.input_hashes(args)
    assert before != after


def test_evaluation_gates_outcomes_before_missing_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'require_private_output', lambda p: p)
    monkeypatch.setattr(study, 'input_hashes', lambda args: {})
    monkeypatch.setattr(study, 'source_hashes', lambda: {})
    features = tmp_path / 'features.npz'
    features.write_bytes(b'fake')
    config = {'input_file_hashes': {}, 'source_hashes': {}, 'feature_sha256': study.digest(features), 'conditions': list(study.CONDITIONS), 'seeds': list(study.SEEDS)}
    (tmp_path / 'frozen_config.json').write_text(json.dumps(config))
    called = []
    monkeypatch.setattr(study, 'load_labels', lambda *args: called.append(args))
    with pytest.raises(FileNotFoundError, match='training_complete'):
        study.evaluate(Namespace(output_dir=tmp_path, features=features, zip_dir=tmp_path))
    assert called == []


def test_exact_half_uses_softmax_argmax_tie_class_zero():
    m = study.metrics([0, 1], [.5, .5])
    assert m['confusion_matrix'] == [[1, 0], [1, 0]]
    assert m['auroc'] == .5
    result = study.paired_bootstrap([0, 1], np.array([[.5, .5]]), np.array([[.1, .1]]), repeats=10)
    assert result['delta_macro_f1'] == 0 and result['ci95'] == [0, 0]
