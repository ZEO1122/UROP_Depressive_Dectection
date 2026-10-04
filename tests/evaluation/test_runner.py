from argparse import Namespace
import json

import numpy as np
import pytest

from urop.evaluation import runner as study


def features(ids=(300, 301, 302)):
    n = len(ids)
    data = {'participant_ids': np.array(ids), 'split': np.array(['test'] * n),
            'position_ids': np.tile(np.arange(85), (n, 1)),
            'question_presence': np.ones((n, 85), bool)}
    for m, dim in study.train.DIMENSIONS.items():
        data[m] = np.full((n, 85, dim), 7, dtype=np.float32)
        data['mask_' + m] = np.ones((n, 85), bool)
    return data


@pytest.mark.parametrize('header', ['Participant_ID', 'participant_ID'])
def test_official_roster_excludes_only411480_and_preserves300(tmp_path, header):
    ids = list(range(300, 345)) + [411, 480]
    (tmp_path / 'test_split_Depression_AVEC2017.csv').write_text(header + '\n' + '\n'.join(map(str, ids)))
    assert study.load_test_roster(tmp_path) == list(range(300, 345))


def test_test_feature_gate_rejects_wrong_ids_nonfixed_positions_and_preserves_scale(tmp_path):
    data = features()
    data['mask_V'][:, 7] = False
    path = tmp_path / 'E0.npz'
    np.savez(path, **data)
    loaded = study.load_test_features(path, [300, 301, 302])
    assert np.all(loaded['V'][:, 7] == 0)
    assert np.all(loaded['A'] == 7) and np.all(loaded['T'] == 7)
    with pytest.raises(ValueError, match='ordered cohort'):
        study.load_test_features(path, [301, 300, 302])
    data['position_ids'][0, 1] = 0
    np.savez(path, **data)
    with pytest.raises(ValueError, match='nonfixed positions'):
        study.load_test_features(path, [300, 301, 302])


def test_labels_handle_blank_rows_and_preserve_binary_despite_score(tmp_path):
    path = tmp_path / 'labels.csv'
    path.write_text('Participant_ID,PHQ_Binary,PHQ_Score\n300,0,10\n301,1,2\n,,\n\n411,0,0\n480,1,11\n')
    assert study.load_test_labels(path, [300, 301]).tolist() == [0, 1]
    path.write_text(path.read_text() + '300,0,10\n')
    with pytest.raises(ValueError, match='Duplicate'):
        study.load_test_labels(path, [300, 301])


def test_evaluation_never_parses_outcomes_before_training_gate(tmp_path, monkeypatch):
    (tmp_path / 'frozen_evaluation.json').write_text('{}')
    monkeypatch.setattr(study, 'output_directory', lambda args: tmp_path)
    def incomplete(*args):
        raise ValueError('Incomplete 25 runs')
    monkeypatch.setattr(study, 'verify_training', incomplete)
    called = []
    monkeypatch.setattr(study, 'load_test_labels', lambda *args: called.append(True))
    with pytest.raises(ValueError, match='Incomplete 25 runs'):
        study.evaluate(Namespace(training_dir=tmp_path))
    assert called == [] and not (tmp_path / 'evaluation_manifest.json').exists()


def test_changed_frozen_evaluation_prevents_outcome_read(tmp_path, monkeypatch):
    (tmp_path / 'frozen_evaluation.json').write_text('{"hash": "old"}')
    monkeypatch.setattr(study, 'output_directory', lambda args: tmp_path)
    monkeypatch.setattr(study, 'snapshot', lambda args: ({'hash': 'new'}, {}, {}))
    called = []
    monkeypatch.setattr(study, 'load_test_labels', lambda *args: called.append(True))
    with pytest.raises(ValueError, match='inputs/implementation changed'):
        study.evaluate(Namespace())
    assert called == []


def test_exposure_manifest_precedes_outcome_parse_and_failure_is_recorded(tmp_path, monkeypatch):
    frozen = {'test_ids': [300], 'pretest_dev_selected_reference': 'E2'}
    (tmp_path / 'frozen_evaluation.json').write_text(json.dumps(frozen))
    monkeypatch.setattr(study, 'output_directory', lambda args: tmp_path)
    monkeypatch.setattr(study, 'snapshot', lambda args: (frozen, {}, {}))
    def inspect_before_read(*args):
        exposure = json.loads((tmp_path / 'evaluation_manifest.json').read_text())
        assert exposure['status'] == 'in_progress' and exposure['attempt'] == 1
        raise RuntimeError('technical failure')
    monkeypatch.setattr(study, 'load_test_labels', inspect_before_read)
    with pytest.raises(RuntimeError, match='technical failure'):
        study.evaluate(Namespace(label_dir=tmp_path))
    exposure = json.loads((tmp_path / 'evaluation_manifest.json').read_text())
    assert exposure['status'] == 'failed' and exposure['error_type'] == 'RuntimeError'


def test_existing_results_are_never_overwritten(tmp_path, monkeypatch):
    (tmp_path / 'test_results.json').write_text('original')
    monkeypatch.setattr(study, 'output_directory', lambda args: tmp_path)
    with pytest.raises(ValueError, match='refusing overwrite'):
        study.evaluate(Namespace())
    assert (tmp_path / 'test_results.json').read_text() == 'original'


def test_tokenizer_provenance_requires_actual_parity_and_e0_correction(tmp_path):
    path = tmp_path / 'audit.json'
    flags = {key: True for key in ['tokenizer_serialized_parity_verified', 'E0_text_recomputed_correct_BPE',
                                  'short_E3_E4_bit_identical', 'content_coverage_exact', 'frozen_model']}
    path.write_text(json.dumps(flags))
    study.verify_tokenizer_provenance(path)
    flags['tokenizer_serialized_parity_verified'] = False
    path.write_text(json.dumps(flags))
    with pytest.raises(ValueError, match='tokenizer_serialized_parity_verified'):
        study.verify_tokenizer_provenance(path)


def test_training_source_mismatch_blocks_checkpoint_and_outcome_reads(tmp_path, monkeypatch):
    config = {'conditions': list(study.train.CONDITIONS), 'seeds': list(study.train.SEEDS), 'source_hashes': {'file': 'old'}}
    (tmp_path / 'frozen_config.json').write_text(json.dumps(config))
    monkeypatch.setattr(study.train, 'source_hashes', lambda: {'file': 'changed'})
    with pytest.raises(ValueError, match='training sources changed'):
        study.verify_training(tmp_path)


def test_training_input_hash_mismatch_blocks_later_stages(tmp_path, monkeypatch):
    source = tmp_path / 'feature.npz'
    source.write_bytes(b'original')
    config = {'conditions': list(study.train.CONDITIONS), 'seeds': list(study.train.SEEDS),
              'source_hashes': {}, 'input_file_hashes': {str(source): study.digest(source)}}
    (tmp_path / 'frozen_config.json').write_text(json.dumps(config))
    source.write_bytes(b'changed')
    monkeypatch.setattr(study.train, 'source_hashes', lambda: {})
    with pytest.raises(ValueError, match='training input changed'):
        study.verify_training(tmp_path)


def test_summary_reports_all_conditions_seeds_and_four_paired_comparisons():
    truth = np.array([0, 0, 1, 1])
    probability = np.array([[.1, .1, .9, .9], [.9, .9, .1, .1], [.1, .9, .1, .9],
                            [.1, .1, .9, .9], [.1, .9, .1, .9]])
    probs = {c: probability for c in study.train.CONDITIONS}
    scores = {c: [{'seed': seed, 'metrics': study.metrics(truth, p)}
                  for seed, p in zip(study.train.SEEDS, probability)] for c in study.train.CONDITIONS}
    result = study.summarize(truth, scores, probs)
    assert len(result['conditions']) == 5 and len(result['comparisons']) == 4
    assert result['conditions']['E0']['seed42']['macro_f1'] == 1
    assert result['conditions']['E0']['per_seed_summary']['macro_f1']['mean'] == .6
    assert all(c['delta_macro_f1'] == 0 for c in result['comparisons'].values())


def test_resume_checks_saved_probability_integrity(tmp_path):
    p = tmp_path / 'test_predictions.npz'
    truth = np.array([0, 1])
    np.savez(p, participant_ids=[300, 301], truth=truth, probability=[.1, .9], checkpoint_sha256=np.array('weights'))
    record = {'evaluation_config_sha256': 'config', 'checkpoint_sha256': 'weights',
              'prediction_sha256': study.digest(p), 'metrics': study.metrics(truth, [.1, .9])}
    (tmp_path / 'complete.json').write_text(json.dumps(record))
    _, probability = study.saved_prediction(tmp_path, 'config', 'weights', [300, 301], truth)
    assert probability.tolist() == [.1, .9]
    p.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='predictions changed'):
        study.saved_prediction(tmp_path, 'config', 'weights', [300, 301], truth)
