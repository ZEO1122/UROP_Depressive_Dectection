from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest

from urop.training import runner as study


def feature_fixture():
    data = {'participant_ids': np.array([301, 302, 303]),
            'split': np.array(['train', 'train', 'dev']),
            'question_presence': np.ones((3, 85), dtype=bool)}
    for m, dim in study.DIMENSIONS.items():
        data[m] = np.full((3, 85, dim), 7, dtype=np.float32)
        data['mask_' + m] = np.ones((3, 85), dtype=bool)
    return data


def test_roster_never_opens_test_even_when_test_files_exist(tmp_path, monkeypatch):
    for split, ids in [('train', range(1000, 1107)), ('dev', [*range(2000, 2032), 440, 451, 458])]:
        text = 'Participant_ID,PHQ8_Binary\n' + ''.join(f'{i},{i % 2}\n' for i in ids)
        (tmp_path / f'{split}_split_Depression_AVEC2017.csv').write_text(text)
    for name in ('test_split_Depression_AVEC2017.csv', 'full_test_split.csv'):
        (tmp_path / name).write_text('DO NOT OPEN')
    original = Path.open
    opened = []

    def guarded(path, *args, **kwargs):
        if path.name.startswith(('test_', 'full_test')):
            raise AssertionError('Test access attempted')
        opened.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)
    roster, labels = study.load_roster(tmp_path)
    assert [len(roster[s]) for s in ('train', 'dev')] == [107, 32]
    assert set(opened) == {'train_split_Depression_AVEC2017.csv', 'dev_split_Depression_AVEC2017.csv'}
    assert all(i not in roster['dev'] for i in (440, 451, 458))
    assert labels[1001] == 1


def test_hashes_never_open_test_and_reject_test_provenance(tmp_path, monkeypatch):
    names = [f'{s}_split_Depression_AVEC2017.csv' for s in ('train', 'dev')]
    names += [f'{c}.npz' for c in study.CONDITIONS] + ['protocol.json']
    for name in names:
        (tmp_path / name).write_text(name)
    args = Namespace(label_dir=tmp_path, features_dir=tmp_path, provenance_file=[tmp_path / 'protocol.json'])
    original = Path.open
    def guarded(path, *a, **kw):
        if path.name.startswith(('test_', 'full_test')):
            raise AssertionError('Test access attempted')
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', guarded)
    before = study.input_hashes(args)
    assert len(before) == 8
    (tmp_path / 'protocol.json').write_text('changed')
    assert before != study.input_hashes(args)
    args.provenance_file.append(tmp_path / 'full_test_split.csv')
    with pytest.raises(ValueError, match='Test files'):
        study.input_hashes(args)


def test_own_masks_preserve_other_modalities_and_raw_scale(tmp_path):
    data = feature_fixture()
    data['mask_V'][:, 7] = False
    path = tmp_path / 'E0.npz'
    np.savez(path, **data)
    loaded = study.load_features(path, {'train': [301, 302], 'dev': [303]})
    assert np.all(loaded['V'][:, 7] == 0)
    assert np.all(loaded['A'] == 7) and np.all(loaded['T'] == 7)
    data['split'][2] = 'test'
    np.savez(path, **data)
    with pytest.raises(ValueError, match='Only train and dev'):
        study.load_features(path, {'train': [301, 302], 'dev': [303]})


def test_reject_nonfinite_values_even_in_masked_slots(tmp_path):
    data = feature_fixture()
    data['mask_A'][0, 0] = False
    data['A'][0, 0, 0] = np.nan
    path = tmp_path / 'E0.npz'
    np.savez(path, **data)
    with pytest.raises(ValueError, match='finite'):
        study.load_features(path, {'train': [301, 302], 'dev': [303]})


def test_conditions_gate_participant_alignment_and_e3e4_av_identity(tmp_path):
    data = feature_fixture()
    roster = {'train': [301, 302], 'dev': [303]}
    for condition in study.CONDITIONS:
        np.savez(tmp_path / f'{condition}.npz', **data)
    study.load_conditions(tmp_path, roster)
    data['A'][0, 0, 0] = 8
    np.savez(tmp_path / 'E4.npz', **data)
    with pytest.raises(ValueError, match='E3/E4'):
        study.load_conditions(tmp_path, roster)
    data['A'][0, 0, 0] = 7
    swapped = {k: v[[1, 0, 2]] for k, v in data.items()}
    np.savez(tmp_path / 'E4.npz', **swapped)
    with pytest.raises(ValueError, match='identical participant order'):
        study.load_conditions(tmp_path, roster)


def test_augmentation_is_paired_but_reports_effective_removed_information():
    a = feature_fixture()
    b = feature_fixture()
    b['mask_V'][:] = False
    b['V'][:] = 0
    idx = np.array([True, True, False])
    y = np.array([0, 1])
    ax, ay, ar, ae = study.augmented_inputs(a, idx, y, 42)
    bx, by, br, be = study.augmented_inputs(b, idx, y, 42)
    assert ar == br and np.array_equal(ay, by)
    assert ae['per_modality']['V']['total_valid_slots_removed'] == 20
    assert be['per_modality']['V']['total_valid_slots_removed'] == 0
    assert np.array_equal(ax[0], bx[0])
    assert np.array_equal(ax[-1], np.tile(np.arange(85), (4, 1)))
    assert np.all(a['A'] == 7)


def test_freeze_covers_all_seeds_independently_of_stage(tmp_path, monkeypatch):
    args = Namespace(stage='initial')
    data = {c: feature_fixture() for c in study.CONDITIONS}
    monkeypatch.setattr(study, 'input_hashes', lambda args: {'input': 'hash'})
    monkeypatch.setattr(study, 'source_hashes', lambda: {'source': 'hash'})
    initial = study.frozen_config(args, data)
    args.stage = 'extend'
    assert initial == study.frozen_config(args, data)
    assert initial['seeds'] == [42, 13, 23, 37, 79]
    assert set(initial['participants']) == {'train', 'dev'}


def test_extension_requires_initial_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match='initial_complete'):
        study.verify_initial_gate(tmp_path, {})


def test_initial_gate_rejects_changed_summary(tmp_path, monkeypatch):
    for name in ('frozen_config.json', 'initial_results.json', 'controls.json'):
        (tmp_path / name).write_text('{}')
    hashes = {'E0_seed42': 'completehash'}
    manifest = {'config_sha256': study.digest(tmp_path / 'frozen_config.json'),
                'runs': hashes, 'results_sha256': study.digest(tmp_path / 'initial_results.json'),
                'controls_sha256': study.digest(tmp_path / 'controls.json')}
    (tmp_path / 'initial_complete.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(study, 'verify_stage', lambda *args: hashes)
    assert study.verify_initial_gate(tmp_path, {}) == hashes
    (tmp_path / 'initial_results.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='summary or controls changed'):
        study.verify_initial_gate(tmp_path, {})


def test_stage_rejects_unpaired_augmentation(tmp_path, monkeypatch):
    (tmp_path / 'frozen_config.json').write_text('{}')
    monkeypatch.setattr(study, 'verify_run', lambda *args: {})
    for condition in study.CONDITIONS:
        directory = tmp_path / f'{condition}_seed42'
        directory.mkdir()
        (directory / 'complete.json').write_text('{}')
        (directory / 'augmentation.json').write_text(condition)
    with pytest.raises(ValueError, match='Paired augmentation'):
        study.verify_stage(tmp_path, [42], {'participants': {}})


def test_paired_bootstrap_uses_mean_seed_f1_not_probability_ensemble():
    truth = np.array([0, 0, 1, 1])
    a = np.array([[.1, .1, .9, .9], [.9, .9, .1, .1]])
    b = np.tile([.1, .9, .1, .9], (2, 1))
    result = study.paired_bootstrap(truth, a, b, repeats=20)
    assert result['delta_macro_f1'] == 0


def test_verify_completed_run_detects_modified_weights_before_predictions(tmp_path):
    record = {'condition': 'E0', 'seed': 42, 'epochs': 100, 'config_sha256': 'hash',
              'artifact_hashes': {name: 'originalhash' for name in study.ARTIFACTS}}
    (tmp_path / 'complete.json').write_text(json.dumps(record))
    (tmp_path / 'best.weights.h5').write_text('changed')
    with pytest.raises(ValueError, match='artifact changed'):
        study.verify_run(tmp_path, 'E0', 42, 'hash')
