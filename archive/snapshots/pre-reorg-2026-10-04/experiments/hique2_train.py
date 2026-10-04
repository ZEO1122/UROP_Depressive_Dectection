"""Frozen three-condition HiQuE study excluding 440; test outcomes gated last."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .hique_train import augment, digest, load_labels, write_json
from .prepare import require_private_output, split_ids

CONDITIONS = ('H_AVT', 'F_AVT', 'H_T')
SEEDS = (42, 13, 23, 37, 79)


def metrics(truth, probability):
    probability = np.asarray(probability)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid predicted probabilities')
    truth = np.asarray(truth, dtype=int)
    predicted = probability > .5  # argmax([p0,p1]) resolves exact ties to class 0.
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    result = {'n': len(truth), 'macro_f1': float(f1_score(truth, predicted, labels=[0, 1], average='macro', zero_division=0)),
              'positive_f1': float(f1_score(truth, predicted, zero_division=0)),
              'sensitivity': float(tp / (tp + fn)) if tp + fn else None,
              'specificity': float(tn / (tn + fp)) if tn + fp else None,
              'auroc': float(roc_auc_score(truth, probability)) if len(set(truth)) == 2 else None,
              'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]}
    for average in ('macro', 'weighted'):
        result[average + '_precision'] = float(precision_score(truth, predicted, labels=[0, 1], average=average, zero_division=0))
        result[average + '_recall'] = float(recall_score(truth, predicted, labels=[0, 1], average=average, zero_division=0))
    result['weighted_f1'] = float(f1_score(truth, predicted, labels=[0, 1], average='weighted', zero_division=0))
    result['gmean'] = (float(np.sqrt(result['sensitivity'] * result['specificity']))
                       if result['sensitivity'] is not None and result['specificity'] is not None else None)
    return result


def paired_bootstrap(truth, a, b, repeats=2000):
    """Same participant resample for every paired seed, then average seed F1 deltas."""
    truth = np.asarray(truth, dtype=int)
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != len(truth):
        raise ValueError('Expected matched seed-by-participant probability arrays')
    def score(y, p):
        cm = np.bincount(y * 2 + (p > .5).astype(int), minlength=4).reshape(2, 2)
        denom = cm.sum(axis=0) + cm.sum(axis=1)
        return np.divide(2 * cm.diagonal(), denom, out=np.zeros(2), where=denom > 0).mean()
    def difference(idx):
        return float(np.mean([score(truth[idx], x[idx]) - score(truth[idx], y[idx]) for x, y in zip(a, b)]))
    rng = np.random.default_rng(91273)
    deltas = [difference(rng.integers(0, len(truth), len(truth))) for _ in range(repeats)]
    return {'delta_macro_f1': difference(np.arange(len(truth))), 'ci95': np.percentile(deltas, [2.5, 97.5]).tolist(),
            'repeats': repeats, 'unit': 'participant, same resample across fixed paired seeds',
            'prediction': 'mean of per-seed Macro-F1 differences; not ensemble probabilities'}


def load_features(path, root):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    ids = data['participant_ids']
    n = len(ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(set(ids)) != n or 440 in ids:
        raise ValueError('Invalid, duplicate, or excluded participant')
    if data['split'].shape != (n,):
        raise ValueError('Invalid split shape')
    official = split_ids(root)
    for pid, split in zip(ids, data['split']):
        if split not in official or pid not in official[split]:
            raise ValueError('Invalid official split')
    for split in official:
        if not np.any(data['split'] == split):
            raise ValueError('Empty split')
    for m, dim in [('A', 88), ('V', 272), ('T', 768)]:
        mask = data['mask_' + m]
        if data[m].shape != (n, 85, dim) or mask.shape != (n, 85) or mask.dtype != bool:
            raise ValueError(f'Invalid {m} feature or mask schema')
        if not np.isfinite(data[m][mask]).all():
            raise ValueError('Nonfinite observed features')
        values = np.zeros_like(data[m], dtype='float32')
        values[mask] = data[m][mask]
        data[m] = values  # No standardization and no cross-modality common mask.
    positions = data['position_ids']
    if positions.shape != (n, 85) or not np.issubdtype(positions.dtype, np.integer) or np.any((positions < 0) | (positions >= 85)):
        raise ValueError('Invalid hierarchical positions')
    if data['question_presence'].shape != (n, 85) or not np.isin(data['question_presence'], [0, 1]).all():
        raise ValueError('Invalid question presence')
    return data


def model_inputs(data, index, mode):
    positions = data['position_ids'][index] if mode.startswith('H_') else np.broadcast_to(np.arange(85), data['position_ids'][index].shape).copy()
    return [data[m][index] for m in mode.split('_')[1]] + [positions]


def augmented_inputs(data, index, mode, labels, seed):
    inputs = model_inputs(data, index, mode)
    x, y, record = augment(inputs[:-1], labels, seed)
    sources = np.asarray(record['source_train_indices'], dtype=int)
    x.append(np.concatenate([inputs[-1], inputs[-1][sources]]))
    return x, y, record


def input_hashes(args):
    paths = [args.zip_dir / name for name in ('train_split_Depression_AVEC2017.csv', 'dev_split_Depression_AVEC2017.csv', 'test_split_Depression_AVEC2017.csv', 'full_test_split.csv')]
    paths += [args.features.parent / 'protocol.json', args.features.parent / 'manifest.json']
    paths += list(args.provenance_file)
    return {str(path.resolve()): digest(path) for path in paths}


def configure_tf():
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    return tf


def source_hashes():
    from . import hique2_network, hique_network, hique_train, prepare
    return {str(Path(path).resolve()): digest(path) for path in
            (__file__, hique2_network.__file__, hique_network.__file__, hique_network.SOURCE, hique_train.__file__, prepare.__file__)}


def verify_artifacts(output, frozen):
    completion = json.loads((output / 'training_complete.json').read_text())
    if completion['runs'] != 15 or completion['config_sha256'] != digest(output / 'frozen_config.json'):
        raise ValueError('Incomplete or changed frozen training')
    for name in ('question_control.npz', 'majority_control.json'):
        if completion['control_hashes'][name] != digest(output / name):
            raise ValueError('Fitted control changed')
    for mode, seed in itertools.product(CONDITIONS, SEEDS):
        directory = output / f'{mode}_seed{seed}'
        record = json.loads((directory / 'complete.json').read_text())
        if record['epochs'] != 100 or record['mode'] != mode or record['seed'] != seed:
            raise ValueError('Incomplete run')
        for name, key in [('best.weights.h5', 'weights_sha256'), ('history.csv', 'history_sha256'), ('dev_predictions.npz', 'dev_sha256'), ('augmentation.json', 'augmentation_sha256')]:
            if record[key] != digest(directory / name):
                raise ValueError('Run artifact changed')

    for seed in SEEDS:
        h = output / f'H_AVT_seed{seed}' / 'augmentation.json'
        f = output / f'F_AVT_seed{seed}' / 'augmentation.json'
        if digest(h) != digest(f):
            raise ValueError('H/F augmentation mismatch')


def train(args):
    output = require_private_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = load_features(args.features, args.zip_dir)
    ids = data['participant_ids']
    ti, di = data['split'] == 'train', data['split'] == 'dev'
    labels = load_labels(args.zip_dir, 'train')
    y = np.array([labels[int(pid)] for pid in ids[ti]])
    yd = np.array([labels[int(pid)] for pid in ids[di]])
    if len(np.unique(y)) != 2:
        raise ValueError('Training requires both labels')
    frozen = {'feature_sha256': digest(args.features), 'input_file_hashes': input_hashes(args), 'source_hashes': source_hashes(),
              'conditions': list(CONDITIONS), 'seeds': list(SEEDS), 'epochs': 100, 'batch_size': 8, 'learning_rate': .0002, 'dropout': .5,
              'checkpoint': 'minimum dev cross-entropy', 'threshold': .5, 'threshold_rule': 'positive iff probability > 0.5; softmax argmax ties class 0', 'normalization': 'none', 'mask': 'modality-specific only',
              'batch_policy': 'ceil batches, retain last partial batch; explicit deviation from upstream floor steps',
              'primary_comparison': 'H_AVT minus F_AVT mean per-seed Macro-F1',
              'augmentation': 'positive tripling, 10 synchronized feature slots zeroed, positions unchanged',
              'participants': {s: ids[data['split'] == s].tolist() for s in ('train', 'dev', 'test')},
              'excluded_official_ids': {s: sorted(set(split_ids(args.zip_dir)[s]) - set(ids[data['split'] == s])) for s in ('train', 'dev', 'test')},
              'test_labels_accessed': False}
    freeze_path = output / 'frozen_config.json'
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != frozen:
        raise ValueError('Frozen configuration mismatch; use new output directory')
    write_json(freeze_path, frozen)
    control = LogisticRegression(C=1, max_iter=2000, random_state=13)
    control.fit(data['question_presence'][ti], y)
    np.savez(output / 'question_control.npz', coef=control.coef_, intercept=control.intercept_)
    write_json(output / 'question_control_dev.json', metrics(yd, control.predict_proba(data['question_presence'][di])[:, 1]))
    majority = int(np.bincount(y, minlength=2).argmax())
    write_json(output / 'majority_control.json', {'class': majority, 'train_counts': np.bincount(y, minlength=2).tolist(), 'dev': metrics(yd, np.full(len(yd), majority))})
    tf = configure_tf()
    from . import hique2_network
    write_json(output / 'runtime.json', {'tensorflow': tf.__version__, 'gpus': [d.name for d in tf.config.list_physical_devices('GPU')]})
    for mode, seed in itertools.product(CONDITIONS, SEEDS):
        directory = output / f'{mode}_seed{seed}'
        directory.mkdir(exist_ok=True)
        if (directory / 'complete.json').exists():
            record = json.loads((directory / 'complete.json').read_text())
            if record['weights_sha256'] != digest(directory / 'best.weights.h5') or record['history_sha256'] != digest(directory / 'history.csv') or record['dev_sha256'] != digest(directory / 'dev_predictions.npz') or record['augmentation_sha256'] != digest(directory / 'augmentation.json'):
                raise ValueError('Completed run changed')
            continue
        tf.keras.backend.clear_session()
        model = hique2_network.build_model(mode, seed=seed, learning_rate=.0002, dropout=.5)
        x, ya, record = augmented_inputs(data, ti, mode, y, seed)
        if mode == 'F_AVT':
            reference = output / f'H_AVT_seed{seed}' / 'augmentation.json'
            if json.loads(reference.read_text()) != record:
                raise ValueError('H/F augmentation mismatch')
        write_json(directory / 'augmentation.json', record)
        dx = model_inputs(data, di, mode)
        checkpoint = directory / 'best.weights.h5'
        start = time.monotonic()
        history = model.fit(x, tf.keras.utils.to_categorical(ya, 2), validation_data=(dx, tf.keras.utils.to_categorical(yd, 2)),
                            epochs=100, batch_size=8, shuffle=True, verbose=0,
                            callbacks=[tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss', save_best_only=True, save_weights_only=True), tf.keras.callbacks.CSVLogger(str(directory / 'history.csv'))])
        if len(history.history['loss']) != 100 or not np.isfinite(history.history['val_loss']).all() or not np.isfinite(history.history['loss']).all():
            raise ValueError('Incomplete or nonfinite training')
        model.load_weights(str(checkpoint))
        probability = model.predict(dx, batch_size=8, verbose=0)[:, 1]
        np.savez(directory / 'dev_predictions.npz', participant_ids=ids[di], truth=yd, probability=probability)
        result = {'mode': mode, 'seed': seed, 'seconds': time.monotonic() - start, 'epochs': 100,
                  'best_epoch': int(np.argmin(history.history['val_loss'])) + 1, 'dev': metrics(yd, probability),
                  'weights_sha256': digest(checkpoint), 'history_sha256': digest(directory / 'history.csv'), 'dev_sha256': digest(directory / 'dev_predictions.npz'), 'augmentation_sha256': digest(directory / 'augmentation.json')}
        write_json(directory / 'complete.json', result)
        print(json.dumps(result), flush=True)
    write_json(output / 'training_complete.json', {'config_sha256': digest(freeze_path), 'runs': 15,
               'control_hashes': {name: digest(output / name) for name in ('question_control.npz', 'majority_control.json')}})
    verify_artifacts(output, frozen)


def evaluate(args):
    output = require_private_output(args.output_dir)
    frozen = json.loads((output / 'frozen_config.json').read_text())
    if frozen['input_file_hashes'] != input_hashes(args) or frozen['source_hashes'] != source_hashes() or frozen['feature_sha256'] != digest(args.features):
        raise ValueError('Frozen inputs or implementation changed')
    if frozen['conditions'] != list(CONDITIONS) or frozen['seeds'] != list(SEEDS):
        raise ValueError('Frozen conditions changed')
    verify_artifacts(output, frozen)  # Must finish all 15 runs before reading test outcomes.
    if (output / 'test_results.json').exists():
        raise ValueError('Evaluation already complete')
    data = load_features(args.features, args.zip_dir)
    index = data['split'] == 'test'
    ids = data['participant_ids'][index]
    if ids.tolist() != frozen['participants']['test']:
        raise ValueError('Held-out cohort changed')
    exposure_path = output / 'evaluation_manifest.json'
    previous = json.loads(exposure_path.read_text()) if exposure_path.exists() else {}
    exposure = {'status': 'in_progress', 'attempt': previous.get('attempt', 0) + 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
                'config_sha256': digest(output / 'frozen_config.json'), 'test_label_file_sha256': digest(args.zip_dir / 'full_test_split.csv'),
                'test_participants': ids.tolist(), 'prior_study_test_exposure': True, 'model_selection_after_test': False}
    write_json(exposure_path, exposure)
    labels = load_labels(args.zip_dir, 'evaluate')
    truth = np.array([labels[int(pid)] for pid in ids])
    tf = configure_tf()
    from . import hique2_network
    results, all_probs = {}, {}
    for mode in CONDITIONS:
        scores, probabilities = [], []
        for seed in SEEDS:
            tf.keras.backend.clear_session()
            model = hique2_network.build_model(mode, seed=seed, learning_rate=.0002, dropout=.5)
            directory = output / f'{mode}_seed{seed}'
            model.load_weights(str(directory / 'best.weights.h5'))
            probability = model.predict(model_inputs(data, index, mode), batch_size=8, verbose=0)[:, 1]
            np.savez(directory / 'test_predictions.npz', participant_ids=ids, truth=truth, probability=probability)
            scores.append({'seed': seed, **metrics(truth, probability)})
            probabilities.append(probability)
        all_probs[mode] = np.asarray(probabilities)
        keys = [k for k, value in scores[0].items() if isinstance(value, float)]
        results[mode] = {'per_seed': scores, 'per_seed_summary': {k: {'mean': float(np.mean([s[k] for s in scores])), 'std': float(np.std([s[k] for s in scores]))} for k in keys if all(s[k] is not None for s in scores)},
                         'seed_averaged_probability_metrics_secondary': metrics(truth, all_probs[mode].mean(axis=0))}
    with np.load(output / 'question_control.npz') as control:
        z = data['question_presence'][index] @ control['coef'][0] + control['intercept'][0]
        probability = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
    np.savez(output / 'question_control_test.npz', participant_ids=ids, truth=truth, probability=probability)
    results['question_presence_control'] = metrics(truth, probability)
    majority = json.loads((output / 'majority_control.json').read_text())['class']
    results['majority_control'] = metrics(truth, np.full(len(truth), majority))
    results['primary_H_AVT_minus_F_AVT'] = paired_bootstrap(truth, all_probs['H_AVT'], all_probs['F_AVT'])
    results['secondary_H_AVT_minus_H_T'] = paired_bootstrap(truth, all_probs['H_AVT'], all_probs['H_T'])
    write_json(output / 'test_results.json', results)
    exposure.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(), test_results_sha256=digest(output / 'test_results.json'))
    write_json(exposure_path, exposure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--zip-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--provenance-file', type=Path, action='append', default=[])
    parser.add_argument('--phase', choices=['train', 'evaluate'], default='train')
    args = parser.parse_args()
    (train if args.phase == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
