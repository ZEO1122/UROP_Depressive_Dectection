"""Frozen train/dev HiQuE reproduction; held-out labels only in evaluate phase."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from .prepare import require_private_output, split_ids


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def input_hashes(features: Path, root: Path):
    """Hash held-out outcome bytes without parsing/using their values."""
    files = [root / name for name in ('train_split_Depression_AVEC2017.csv',
             'dev_split_Depression_AVEC2017.csv', 'test_split_Depression_AVEC2017.csv', 'full_test_split.csv')]
    files += [features.parent / 'protocol.json', features.parent / 'manifest.json']
    return {str(path.resolve()): digest(path) for path in files}


def load_labels(root: Path, phase: str) -> dict[int, int]:
    """Preserve published binary labels, including participant 409 discrepancy."""
    official = split_ids(root)
    result = {}
    for split in (('train', 'dev') if phase == 'train' else ('test',)):
        filename = 'full_test_split.csv' if split == 'test' else f'{split}_split_Depression_AVEC2017.csv'
        with (root / filename).open(newline='') as stream:
            for row in csv.DictReader(stream):
                if not row.get('Participant_ID', '').strip():
                    continue
                pid = int(row['Participant_ID'])
                if pid not in official[split]:
                    continue
                label = int(row['PHQ_Binary' if split == 'test' else 'PHQ8_Binary'])
                if label not in (0, 1) or pid in result:
                    raise ValueError('Invalid or duplicate binary label')
                result[pid] = label
    return result


def fit_scaler(values, mask):
    observed = values[mask]
    if len(observed) == 0 or not np.isfinite(observed).all():
        raise ValueError('No finite training observations')
    mean = observed.mean(axis=0, dtype=np.float64)
    std = observed.std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1
    return mean.astype('float32'), std.astype('float32')


def transform(values, mask, mean, std):
    result = np.zeros_like(values, dtype='float32')
    result[mask] = (values[mask] - mean) / std
    return result


def augment(inputs, labels, seed, slots_to_mask=10):
    """Two extra positive copies; synchronized slot removal across modalities."""
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    sources = np.tile(positive, 2)
    outputs = [np.concatenate([x, x[sources].copy()]) for x in inputs]
    removed = []
    for index in range(len(sources)):
        slots = rng.choice(inputs[0].shape[1], slots_to_mask, replace=False)
        removed.append(slots.tolist())
        for array in outputs:
            array[len(labels) + index, slots] = 0
    return outputs, np.concatenate([labels, labels[sources]]), {'seed': seed, 'source_train_indices': sources.tolist(), 'removed_slots': removed}


def metrics(truth, probability):
    truth = np.asarray(truth, dtype=int)
    probability = np.asarray(probability)
    pred = (probability >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    return {'n': len(truth), 'macro_f1': float(f1_score(truth, pred, labels=[0, 1], average='macro', zero_division=0)),
            'positive_f1': float(f1_score(truth, pred, zero_division=0)),
            'sensitivity': float(tp / (tp + fn)) if tp + fn else None,
            'specificity': float(tn / (tn + fp)) if tn + fp else None,
            'auroc': float(roc_auc_score(truth, probability)) if len(set(truth)) == 2 else None,
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]}


def paired_bootstrap(truth, a, b, repeats=2000):
    rng = np.random.default_rng(91273)
    deltas = []
    def macro(y, p):
        cm = np.bincount(y * 2 + (p >= .5).astype(int), minlength=4).reshape(2, 2)
        denominator = cm.sum(axis=0) + cm.sum(axis=1)
        return np.divide(2 * cm.diagonal(), denominator, out=np.zeros(2), where=denominator > 0).mean()
    for _ in range(repeats):
        idx = rng.integers(0, len(truth), len(truth))
        deltas.append(macro(truth[idx], a[idx]) - macro(truth[idx], b[idx]))
    return {'delta_macro_f1': metrics(truth, a)['macro_f1'] - metrics(truth, b)['macro_f1'],
            'ci95': np.percentile(deltas, [2.5, 97.5]).tolist(), 'repeats': repeats,
            'unit': 'participant', 'prediction': 'seed-averaged probability; not mean per-seed F1'}


def load_features(path, root):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    ids = data['participant_ids'].astype(int)
    official = split_ids(root)
    if len(set(ids)) != len(ids) or set(ids) & {300, 440, 451, 458, 480}:
        raise ValueError('Duplicated or excluded participant')
    for pid, split in zip(ids, data['split']):
        if split not in official or pid not in official[split]:
            raise ValueError('Invalid official split')
    mask = data['mask_A'] & data['mask_V'] & data['mask_T']
    eligible = mask.any(axis=1)
    excluded = ids[~eligible].tolist()
    for key in data:
        data[key] = data[key][eligible]
    data['common_mask'] = mask[eligible]
    for m, dim in [('A', 88), ('V', 272), ('T', 768)]:
        if data[m].shape != (eligible.sum(), 85, dim):
            raise ValueError(f'Invalid {m} dimensions')
        if not np.isfinite(data[m][data['common_mask']]).all():
            raise ValueError('Nonfinite observed features')
    return data, excluded


def configure_tf():
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    return tf


def train(args):
    output = require_private_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data, excluded = load_features(args.features, args.zip_dir)
    labels = load_labels(args.zip_dir, 'train')
    train_idx = data['split'] == 'train'
    dev_idx = data['split'] == 'dev'
    ids = data['participant_ids']
    y = np.array([labels[int(pid)] for pid in ids[train_idx]])
    yd = np.array([labels[int(pid)] for pid in ids[dev_idx]])
    modes = args.modalities.split(',')
    if any(not m or m != ''.join(c for c in 'AVT' if c in m) for m in modes):
        raise ValueError('Use canonical AVT modality order')
    seeds = list(map(int, args.seeds.split(',')))
    from . import hique_network
    frozen = {'feature_sha256': digest(args.features), 'trainer_sha256': digest(__file__),
              'input_file_hashes': input_hashes(args.features, args.zip_dir),
              'model_sha256': digest(hique_network.__file__), 'seeds': seeds, 'modalities': modes,
              'epochs': args.epochs, 'batch_size': 8, 'learning_rate': .0002, 'dropout': .5,
              'checkpoint': 'minimum dev cross-entropy', 'threshold': .5,
              'primary_comparison': 'AVT minus T', 'augmentation': 'positive tripling; ten synchronized slots masked',
              'participants': {s: ids[data['split'] == s].tolist() for s in ('train', 'dev', 'test')},
              'excluded_no_common_slots': excluded, 'test_labels_accessed': False}
    freeze_path = output / 'frozen_config.json'
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != frozen:
        raise ValueError('Frozen configuration mismatch; use a new output directory')
    write_json(freeze_path, frozen)
    normalized, scalers = {}, {}
    for m in 'AVT':
        mean, std = fit_scaler(data[m][train_idx], data['common_mask'][train_idx])
        normalized[m] = transform(data[m], data['common_mask'], mean, std)
        scalers[m + '_mean'], scalers[m + '_std'] = mean, std
    np.savez(output / 'scalers.npz', **scalers)
    control = LogisticRegression(C=1, max_iter=2000, random_state=13)
    control.fit(data['question_presence'][train_idx], y)
    np.savez(output / 'question_control.npz', coef=control.coef_, intercept=control.intercept_)
    write_json(output / 'question_control_dev.json', metrics(yd, control.predict_proba(data['question_presence'][dev_idx])[:, 1]))
    tf = configure_tf()
    for mode, seed in itertools.product(modes, seeds):
        directory = output / f'{mode}_seed{seed}'
        directory.mkdir(exist_ok=True)
        if (directory / 'complete.json').exists():
            continue
        tf.keras.backend.clear_session()
        model = hique_network.build_model(mode, seed=seed, learning_rate=.0002, dropout=.5)
        x, ya, augmentation = augment([normalized[m][train_idx] for m in mode], y, seed)
        write_json(directory / 'augmentation.json', augmentation)
        target = tf.keras.utils.to_categorical(ya, 2)
        dev_target = tf.keras.utils.to_categorical(yd, 2)
        dev_x = [normalized[m][dev_idx] for m in mode]
        checkpoint = directory / 'best.weights.h5'
        start = time.monotonic()
        history = model.fit(x, target, validation_data=(dev_x, dev_target), epochs=args.epochs,
                            batch_size=8, verbose=0, shuffle=True,
                            callbacks=[tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss', save_best_only=True, save_weights_only=True),
                                       tf.keras.callbacks.CSVLogger(str(directory / 'history.csv'))])
        elapsed = time.monotonic() - start
        model.load_weights(str(checkpoint))
        probabilities = model.predict(dev_x, batch_size=8, verbose=0)[:, 1]
        np.savez(directory / 'dev_predictions.npz', participant_ids=ids[dev_idx], truth=yd, probability=probabilities)
        result = {'mode': mode, 'seed': seed, 'seconds': elapsed, 'epochs': args.epochs,
                  'best_epoch': int(np.argmin(history.history['val_loss'])) + 1,
                  'dev': metrics(yd, probabilities), 'weights_sha256': digest(checkpoint)}
        write_json(directory / 'complete.json', result)
        print(json.dumps(result), flush=True)
    write_json(output / 'training_complete.json', {'config_sha256': digest(freeze_path), 'runs': len(modes) * len(seeds),
               'scalers_sha256': digest(output / 'scalers.npz'),
               'question_control_sha256': digest(output / 'question_control.npz')})


def evaluate(args):
    output = require_private_output(args.output_dir)
    frozen = json.loads((output / 'frozen_config.json').read_text())
    completion = json.loads((output / 'training_complete.json').read_text())
    if frozen['input_file_hashes'] != input_hashes(args.features, args.zip_dir):
        raise ValueError('Frozen protocol, split manifest or label files changed')
    if completion['config_sha256'] != digest(output / 'frozen_config.json') or frozen['feature_sha256'] != digest(args.features):
        raise ValueError('Frozen data/configuration changed')
    for name, key in [('scalers.npz', 'scalers_sha256'), ('question_control.npz', 'question_control_sha256')]:
        if completion[key] != digest(output / name):
            raise ValueError('Fitted preprocessing/control changed')
    from . import hique_network
    if frozen['model_sha256'] != digest(hique_network.__file__) or frozen['trainer_sha256'] != digest(__file__):
        raise ValueError('Frozen implementation changed')
    if (output / 'test_results.json').exists():
        raise ValueError('Held-out evaluation already completed; inspect saved results')
    data, _ = load_features(args.features, args.zip_dir)
    test_idx = data['split'] == 'test'
    ids = data['participant_ids'][test_idx]
    if ids.tolist() != frozen['participants']['test']:
        raise ValueError('Held-out cohort changed')
    # Check all artifacts before opening any held-out outcome.
    for mode, seed in itertools.product(frozen['modalities'], frozen['seeds']):
        d = output / f'{mode}_seed{seed}'
        if json.loads((d / 'complete.json').read_text())['weights_sha256'] != digest(d / 'best.weights.h5'):
            raise ValueError('Weights changed')
    exposure_path = output / 'evaluation_manifest.json'
    previous = json.loads(exposure_path.read_text()) if exposure_path.exists() else {}
    exposure = {'status': 'in_progress', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'attempt': previous.get('attempt', 0) + 1, 'config_sha256': digest(output / 'frozen_config.json'),
                'test_label_file_sha256': digest(args.zip_dir / 'full_test_split.csv'),
                'test_participants': ids.tolist(), 'model_selection_after_test': False}
    write_json(exposure_path, exposure)
    labels = load_labels(args.zip_dir, 'evaluate')
    truth = np.array([labels[int(pid)] for pid in ids])
    scaled = {}
    with np.load(output / 'scalers.npz') as scalers:
        for m in 'AVT':
            scaled[m] = transform(data[m][test_idx], data['common_mask'][test_idx], scalers[m + '_mean'], scalers[m + '_std'])
    tf = configure_tf()
    results, averaged = {}, {}
    for mode in frozen['modalities']:
        scores, probs = [], []
        for seed in frozen['seeds']:
            tf.keras.backend.clear_session()
            model = hique_network.build_model(mode, seed=seed, learning_rate=.0002, dropout=.5)
            directory = output / f'{mode}_seed{seed}'
            model.load_weights(str(directory / 'best.weights.h5'))
            probability = model.predict([scaled[m] for m in mode], batch_size=8, verbose=0)[:, 1]
            np.savez(directory / 'test_predictions.npz', participant_ids=ids, truth=truth, probability=probability)
            scores.append({'seed': seed, **metrics(truth, probability)})
            probs.append(probability)
        averaged[mode] = np.mean(probs, axis=0)
        results[mode] = {'per_seed': scores, 'seed_averaged_probability_metrics': metrics(truth, averaged[mode]),
                         'per_seed_summary': {k: {'mean': float(np.mean([s[k] for s in scores])), 'std': float(np.std([s[k] for s in scores]))}
                                              for k in ['macro_f1', 'positive_f1', 'sensitivity', 'specificity', 'auroc'] if all(s[k] is not None for s in scores)}}
    with np.load(output / 'question_control.npz') as control:
        z = data['question_presence'][test_idx] @ control['coef'][0] + control['intercept'][0]
        probability = 1 / (1 + np.exp(-z))
    np.savez(output / 'question_control_test.npz', participant_ids=ids, truth=truth, probability=probability)
    results['question_presence_control'] = metrics(truth, probability)
    results['comparisons'] = {f'{a}_minus_{b}': paired_bootstrap(truth, averaged[a], averaged[b])
                              for a, b in itertools.combinations(frozen['modalities'], 2)}
    if 'AVT' in averaged and 'T' in averaged:
        results['primary_AVT_minus_T'] = paired_bootstrap(truth, averaged['AVT'], averaged['T'])
    write_json(output / 'test_results.json', results)
    exposure.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                    test_results_sha256=digest(output / 'test_results.json'))
    write_json(exposure_path, exposure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--zip-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', default='13,23,37,42,79')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--modalities', default='A,V,T,AV,AT,VT,AVT')
    parser.add_argument('--phase', choices=['train', 'evaluate'], default='train')
    args = parser.parse_args()
    (train if args.phase == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
