"""Evaluate all frozen V3 checkpoints once on the predeclared reference test cohort."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from . import hique3_train as train
from .hique2_train import configure_tf, metrics, paired_bootstrap
from .hique_train import digest, write_json
from .prepare import require_private_output

TEST_EXCLUDED = {411, 480}


def verify_training(training_dir):
    """Complete training, immutable source/input bytes and every saved run are gates."""
    frozen = json.loads((training_dir / 'frozen_config.json').read_text())
    if frozen['conditions'] != list(train.CONDITIONS) or frozen['seeds'] != list(train.SEEDS):
        raise ValueError('Unexpected training conditions/seeds')
    if frozen['source_hashes'] != train.source_hashes():
        raise ValueError('Frozen training sources changed')
    for path, expected in frozen['input_file_hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Frozen training input changed: {path}')
    train.verify_initial_gate(training_dir, frozen)
    runs = train.verify_stage(training_dir, train.SEEDS, frozen)
    completed = json.loads((training_dir / 'extend_complete.json').read_text())
    if len(runs) != 25 or completed['runs'] != runs or completed['config_sha256'] != digest(training_dir / 'frozen_config.json'):
        raise ValueError('All 25 unchanged training runs must be complete')
    for filename, key in [('extend_results.json', 'results_sha256'), ('controls.json', 'controls_sha256')]:
        if digest(training_dir / filename) != completed[key]:
            raise ValueError('Frozen training summary/control changed')
    controls = json.loads((training_dir / 'controls.json').read_text())
    if set(controls['artifact_hashes']) != {f'{c}_question_control.npz' for c in train.CONDITIONS}:
        raise ValueError('Expected all five saved question controls')
    for filename, expected in controls['artifact_hashes'].items():
        if digest(training_dir / filename) != expected:
            raise ValueError('Saved control parameters changed')
    return frozen, controls


def load_test_roster(label_dir):
    with (label_dir / 'test_split_Depression_AVEC2017.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        id_column = next((name for name in reader.fieldnames or [] if name.strip().lower() == 'participant_id'), None)
        if id_column is None:
            raise ValueError('Official test roster lacks a participant ID column')
        ids = [int(row[id_column]) for row in reader if row.get(id_column, '').strip()]
    if len(ids) != 47 or len(set(ids)) != 47 or not TEST_EXCLUDED.issubset(ids):
        raise ValueError('Unexpected official test roster')
    expected = sorted(set(ids) - TEST_EXCLUDED)
    if len(expected) != 45 or 300 not in expected:
        raise ValueError('Expected common45 test cohort including300')
    return expected


def load_test_features(path, expected_ids):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    ids = data['participant_ids']
    n = len(expected_ids)
    if ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer) or ids.tolist() != expected_ids:
        raise ValueError('Test features must match the fixed ordered cohort')
    if data['split'].shape != (n,) or not np.all(data['split'] == 'test'):
        raise ValueError('Only test feature rows are allowed')
    presence = data['question_presence']
    if presence.shape != (n, 85) or not np.isin(presence, [0, 1]).all() or not presence.any(axis=1).all():
        raise ValueError('Invalid or empty test question presence')
    if 'position_ids' in data:
        if not np.array_equal(data['position_ids'], np.tile(np.arange(85), (n, 1))):
            raise ValueError('Test features contain nonfixed positions')
    for m, dim in train.DIMENSIONS.items():
        mask, values = data['mask_' + m], data[m]
        if mask.shape != (n, 85) or mask.dtype != bool or values.shape != (n, 85, dim):
            raise ValueError('Invalid test feature/mask schema')
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError('Nonfinite test features')
        cleaned = np.zeros(values.shape, dtype=np.float32)
        cleaned[mask] = values[mask]
        if not np.isfinite(cleaned).all():
            raise ValueError('Test features overflow float32')
        data[m] = cleaned  # Preserve training's raw scales and independent masks.
    if not (data['mask_A'] | data['mask_V'] | data['mask_T']).any(axis=1).all():
        raise ValueError('A test participant has no observed features')
    return data


def verify_tokenizer_provenance(path):
    audit = json.loads(path.read_text())
    for field in ('tokenizer_serialized_parity_verified', 'E0_text_recomputed_correct_BPE',
                  'short_E3_E4_bit_identical', 'content_coverage_exact', 'frozen_model'):
        if audit.get(field) is not True:
            raise ValueError(f'Missing corrected-tokenizer/feature verification: {field}')
    return audit


def snapshot(args):
    """No outcome parsing: hash label bytes only after all training/schema gates."""
    frozen, controls = verify_training(args.training_dir)
    protocol = json.loads(args.protocol.read_text())
    ids = load_test_roster(args.label_dir)
    if protocol['cohort']['test_ids'] != ids or protocol['cohort']['n'] != 45:
        raise ValueError('Protocol and official test cohort disagree')
    if protocol['conditions'] != list(train.CONDITIONS) or protocol['seeds'] != list(train.SEEDS):
        raise ValueError('Test protocol conditions/seeds changed')
    if protocol.get('prior_test_exposure') is not True:
        raise ValueError('Prior test exposure must be acknowledged')
    for filename, key in [('frozen_config.json', 'training_config_sha256'),
                          ('extend_complete.json', 'training_complete_sha256')]:
        if digest(args.training_dir / filename) != protocol[key]:
            raise ValueError('Test protocol refers to different training results')
    data = {c: load_test_features(args.features_dir / f'{c}.npz', ids) for c in train.CONDITIONS}
    for key in ('A', 'V', 'mask_A', 'mask_V', 'mask_T', 'question_presence'):
        if not np.array_equal(data['E3'][key], data['E4'][key]):
            raise ValueError(f'E3/E4 test inputs differ outside text: {key}')
    verify_tokenizer_provenance(args.tokenizer_provenance)
    paths = [args.protocol, args.tokenizer_provenance,
             args.label_dir / 'test_split_Depression_AVEC2017.csv',
             args.label_dir / 'full_test_split.csv']
    paths += [args.features_dir / f'{c}.npz' for c in train.CONDITIONS]
    paths += list(args.feature_code) + list(args.provenance_file)
    if not args.feature_code or any(Path(p).suffix != '.py' for p in args.feature_code):
        raise ValueError('Test feature implementation paths must be supplied')
    config = {'training_config_sha256': digest(args.training_dir / 'frozen_config.json'),
              'training_complete_sha256': digest(args.training_dir / 'extend_complete.json'),
              'training_dir': str(args.training_dir.resolve()),
              'evaluation_source_sha256': digest(__file__),
              'input_file_hashes': {str(Path(p).resolve()): digest(p) for p in paths},
              'conditions': list(train.CONDITIONS), 'seeds': list(train.SEEDS), 'test_ids': ids,
              'comparisons': [f'{a}_minus_{b}' for a, b in train.COMPARISONS],
              'pretest_dev_selected_reference': protocol['pretest_dev_selected_reference'],
              'normalization': frozen['normalization'], 'mask': frozen['mask'], 'threshold': frozen['threshold'],
              'prior_test_exposure': True, 'model_selection_after_test': False,
              'claim_limit': protocol['claim_limit']}
    return config, data, controls


def output_directory(args):
    output = require_private_output(args.output_dir)
    if output.is_relative_to(args.training_dir.resolve()) or args.training_dir.resolve().is_relative_to(output):
        raise ValueError('Test output must be separate from existing training runs')
    output.mkdir(parents=True, exist_ok=True)
    return output


def freeze(args):
    output = output_directory(args)
    if (output / 'test_results.json').exists() or (output / 'evaluation_manifest.json').exists():
        raise ValueError('Test evaluation already started; freeze cannot be changed')
    config, _, _ = snapshot(args)
    path = output / 'frozen_evaluation.json'
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError('Frozen test evaluation differs')
    else:
        write_json(path, config)
    return config


def load_test_labels(path, expected_ids):
    """Only called after evaluation gates and the prior-outcome exposure record."""
    labels = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if not row.get('Participant_ID', '').strip():
                continue
            pid, value = int(row['Participant_ID']), int(row['PHQ_Binary'])
            if pid in labels or value not in (0, 1):
                raise ValueError('Duplicate participant or nonbinary test label')
            labels[pid] = value
    if set(labels) != set(expected_ids) | TEST_EXCLUDED:
        raise ValueError('Full test labels do not cover the expected official roster')
    return np.array([labels[pid] for pid in expected_ids], dtype=int)


def saved_prediction(directory, config_sha, checkpoint_sha, ids, truth):
    record = json.loads((directory / 'complete.json').read_text())
    if record['evaluation_config_sha256'] != config_sha or record['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Saved test prediction has different provenance')
    path = directory / 'test_predictions.npz'
    if digest(path) != record['prediction_sha256']:
        raise ValueError('Saved test predictions changed')
    with np.load(path, allow_pickle=False) as archive:
        if archive['participant_ids'].tolist() != ids or not np.array_equal(archive['truth'], truth):
            raise ValueError('Saved test cohort/labels changed')
        if str(archive['checkpoint_sha256']) != checkpoint_sha:
            raise ValueError('Saved test checkpoint metadata changed')
        probability = archive['probability'].copy()
    if metrics(truth, probability) != record['metrics']:
        raise ValueError('Saved test metrics disagree with probabilities')
    return record, probability


def summarize(truth, scores, probabilities):
    result = {'conditions': {}, 'comparisons': {}}
    for condition in train.CONDITIONS:
        records = scores[condition]
        keys = [key for key, value in records[0]['metrics'].items() if isinstance(value, float)]
        result['conditions'][condition] = {
            'per_seed': [{'seed': r['seed'], **r['metrics']} for r in records],
            'seed42': next(r['metrics'] for r in records if r['seed'] == 42),
            'per_seed_summary': {key: {'mean': float(np.mean([r['metrics'][key] for r in records])),
                                       'std': float(np.std([r['metrics'][key] for r in records]))} for key in keys},
            'seed_averaged_probability_metrics_secondary': metrics(truth, probabilities[condition].mean(axis=0))}
    for a, b in train.COMPARISONS:
        result['comparisons'][f'{a}_minus_{b}'] = paired_bootstrap(truth, probabilities[a], probabilities[b], repeats=2000)
    return result


def evaluate(args):
    output = output_directory(args)
    if (output / 'test_results.json').exists():
        raise ValueError('Test evaluation already complete; refusing overwrite')
    frozen = json.loads((output / 'frozen_evaluation.json').read_text())
    current, data, controls = snapshot(args)
    if current != frozen:
        raise ValueError('Frozen evaluation inputs/implementation changed')
    config_sha = digest(output / 'frozen_evaluation.json')
    exposure_path = output / 'evaluation_manifest.json'
    previous = json.loads(exposure_path.read_text()) if exposure_path.exists() else {}
    if previous and previous['evaluation_config_sha256'] != config_sha:
        raise ValueError('Cannot resume a different evaluation after outcome exposure')
    exposure = {'status': 'in_progress', 'attempt': previous.get('attempt', 0) + 1,
                'started_utc': datetime.now(timezone.utc).isoformat(), 'evaluation_config_sha256': config_sha,
                'test_participants': frozen['test_ids'], 'prior_test_exposure': True,
                'model_selection_after_test': False, 'pretest_dev_selected_reference': frozen['pretest_dev_selected_reference']}
    write_json(exposure_path, exposure)
    try:
        truth = load_test_labels(args.label_dir / 'full_test_split.csv', frozen['test_ids'])
        tf = configure_tf()
        from . import hique2_network
        write_json(output / 'runtime.json', {'tensorflow': tf.__version__, 'gpu': [d.name for d in tf.config.list_physical_devices('GPU')],
                                           'training': False, 'checkpoint_selection': False})
        scores, probabilities, run_hashes = {}, {}, {}
        for condition in train.CONDITIONS:
            records, ps = [], []
            for seed in train.SEEDS:
                checkpoint = args.training_dir / f'{condition}_seed{seed}' / 'best.weights.h5'
                checkpoint_sha = digest(checkpoint)
                directory = output / f'{condition}_seed{seed}'
                directory.mkdir(exist_ok=True)
                if not (directory / 'complete.json').exists():
                    with train.preserve_upstream_manifest():
                        model = hique2_network.build_model('F_AVT', seed=seed, learning_rate=.0002, dropout=.5)
                    model.load_weights(str(checkpoint))
                    values = data[condition]
                    p = model.predict(train.model_inputs(values, np.ones(len(truth), bool)), batch_size=8, verbose=0)[:, 1]
                    np.savez(directory / 'test_predictions.npz', participant_ids=values['participant_ids'], truth=truth,
                             probability=p, checkpoint_sha256=np.array(checkpoint_sha))
                    record = {'condition': condition, 'seed': seed, 'evaluation_config_sha256': config_sha,
                              'checkpoint_source': str(checkpoint.resolve()), 'checkpoint_sha256': checkpoint_sha,
                              'prediction_sha256': digest(directory / 'test_predictions.npz'), 'metrics': metrics(truth, p)}
                    write_json(directory / 'complete.json', record)
                record, p = saved_prediction(directory, config_sha, checkpoint_sha, frozen['test_ids'], truth)
                if record['condition'] != condition or record['seed'] != seed:
                    raise ValueError('Test prediction identity changed')
                records.append(record)
                ps.append(p)
                run_hashes[directory.name] = digest(directory / 'complete.json')
            scores[condition], probabilities[condition] = records, np.asarray(ps)
        results = summarize(truth, scores, probabilities)
        results['controls'] = {}
        for condition in train.CONDITIONS:
            source = args.training_dir / f'{condition}_question_control.npz'
            with np.load(source, allow_pickle=False) as control:
                if control['coef'].shape != (1, 85) or control['intercept'].shape != (1,):
                    raise ValueError('Saved logistic control shape changed')
                z = data[condition]['question_presence'] @ control['coef'][0] + control['intercept'][0]
            probability = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
            majority = controls['conditions'][condition]['majority_class']
            if majority not in (0, 1):
                raise ValueError('Invalid saved majority class')
            np.savez(output / f'{condition}_question_control_test.npz', participant_ids=np.asarray(frozen['test_ids']),
                     truth=truth, probability=probability, coefficient_source_sha256=np.array(digest(source)))
            results['controls'][condition] = {'question_presence': metrics(truth, probability),
                                             'majority': metrics(truth, np.full(len(truth), majority))}
        if snapshot(args)[0] != frozen:
            raise ValueError('Frozen sources/inputs changed during inference')
        results.update(test_ids=frozen['test_ids'], prior_test_exposure=True, model_selection_after_test=False,
                       pretest_dev_selected_reference=frozen['pretest_dev_selected_reference'],
                       claim_limit=frozen['claim_limit'], evaluation_config_sha256=config_sha)
        write_json(output / 'test_results.json', results)
        exposure.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(), runs=run_hashes,
                        test_results_sha256=digest(output / 'test_results.json'))
        write_json(exposure_path, exposure)
    except Exception as exc:
        exposure.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                        failed_utc=datetime.now(timezone.utc).isoformat())
        write_json(exposure_path, exposure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', type=Path, required=True)
    parser.add_argument('--features-dir', type=Path, required=True)
    parser.add_argument('--label-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--tokenizer-provenance', type=Path, required=True)
    parser.add_argument('--feature-code', type=Path, action='append', required=True)
    parser.add_argument('--provenance-file', type=Path, action='append', default=[])
    parser.add_argument('--stage', choices=['freeze', 'evaluate'], required=True)
    args = parser.parse_args()
    (freeze if args.stage == 'freeze' else evaluate)(args)


if __name__ == '__main__':
    main()
