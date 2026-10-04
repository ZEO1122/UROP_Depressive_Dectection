"""Paired E0--E4 input study on train/dev only; no test-data access."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression

from .hique2_train import configure_tf, metrics, paired_bootstrap
from .hique_train import augment, digest, write_json
from .prepare import require_private_output

CONDITIONS = ('E0', 'E1', 'E2', 'E3', 'E4')
SEEDS = (42, 13, 23, 37, 79)
COMPARISONS = (('E1', 'E0'), ('E2', 'E1'), ('E3', 'E1'), ('E4', 'E3'))
DIMENSIONS = {'A': 88, 'V': 272, 'T': 768}
DEV_EXCLUDED = {440, 451, 458}
ARTIFACTS = ('best.weights.h5', 'history.csv', 'train_predictions.npz',
             'dev_predictions.npz', 'augmentation.json', 'effective_masking.json',
             'feature_attention.json')


def reject_test_path(path):
    """Also reject explicitly supplied held-out outcome/feature provenance paths."""
    path = Path(path)
    if path.name.lower().startswith(('test_', 'full_test')) or any(part.lower() == 'test' for part in path.parts):
        raise ValueError('Test files are outside this train/dev study')
    return path


def load_roster(label_dir):
    """Read exactly two CSVs; split_ids/load_labels would also open a test CSV."""
    roster, labels = {}, {}
    for split in ('train', 'dev'):
        path = Path(label_dir) / f'{split}_split_Depression_AVEC2017.csv'
        participants = []
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if not row.get('Participant_ID', '').strip():
                    continue
                pid, label = int(row['Participant_ID']), int(row['PHQ8_Binary'])
                if pid in labels or label not in (0, 1):
                    raise ValueError('Duplicate participant or invalid binary label')
                labels[pid] = label
                if split != 'dev' or pid not in DEV_EXCLUDED:
                    participants.append(pid)
        roster[split] = sorted(participants)
    if len(roster['train']) != 107 or len(roster['dev']) != 32:
        raise ValueError('Expected fixed official cohort train107/dev32')
    if any(pid in roster['train'] for pid in DEV_EXCLUDED):
        raise ValueError('Excluded dev participant unexpectedly in train')
    return roster, labels


def load_features(path, roster):
    with np.load(reject_test_path(path), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    ids = data['participant_ids']
    n = len(ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(set(ids)) != n:
        raise ValueError('Participant IDs must be unique integers')
    if data['split'].shape != (n,) or set(data['split']) != {'train', 'dev'}:
        raise ValueError('Only train and dev splits are permitted')
    for split in ('train', 'dev'):
        if sorted(ids[data['split'] == split].tolist()) != sorted(roster[split]):
            raise ValueError('Feature cohort does not match fixed train/dev roster')
    presence = data['question_presence']
    if presence.shape != (n, 85) or not np.isin(presence, [0, 1]).all() or not presence.any(axis=1).all():
        raise ValueError('Invalid or empty question presence')
    for m, dim in DIMENSIONS.items():
        mask, values = data['mask_' + m], data[m]
        if mask.shape != (n, 85) or mask.dtype != bool or values.shape != (n, 85, dim):
            raise ValueError(f'Invalid {m} feature/mask shape')
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError('All feature values must be finite numeric values')
        cleaned = np.zeros(values.shape, dtype=np.float32)
        cleaned[mask] = values[mask]
        if not np.isfinite(cleaned).all():
            raise ValueError('Feature values overflow float32')
        data[m] = cleaned
    any_observed = data['mask_A'] | data['mask_V'] | data['mask_T']
    if not any_observed.any(axis=1).all():
        raise ValueError('Participant has no observed feature slots')
    return data


def load_conditions(features_dir, roster):
    data = {condition: load_features(Path(features_dir) / f'{condition}.npz', roster)
            for condition in CONDITIONS}
    first = data['E0']
    for condition in CONDITIONS[1:]:
        for key in ('participant_ids', 'split'):
            if not np.array_equal(first[key], data[condition][key]):
                raise ValueError('All conditions must use identical participant order and split')
    for key in ('A', 'V', 'mask_A', 'mask_V', 'mask_T', 'question_presence'):
        if not np.array_equal(data['E3'][key], data['E4'][key]):
            raise ValueError(f'E3/E4 must differ only in text representation: {key}')
    return data


def model_inputs(data, index):
    positions = np.broadcast_to(np.arange(85, dtype=np.int32), (int(index.sum()), 85)).copy()
    return [data[m][index] for m in DIMENSIONS] + [positions]


def augmented_inputs(data, index, labels, seed):
    inputs = model_inputs(data, index)
    x, y, record = augment(inputs[:-1], labels, seed)
    source = np.asarray(record['source_train_indices'], dtype=int)
    x.append(np.concatenate([inputs[-1], inputs[-1][source]]))
    effective = {'per_modality': {}}
    for m in DIMENSIONS:
        mask = data['mask_' + m][index]
        counts = [int(mask[source_id, slots].sum()) for source_id, slots in
                  zip(source, record['removed_slots'])]
        effective['per_modality'][m] = {'valid_slots_removed_per_copy': counts,
                                       'total_valid_slots_removed': sum(counts),
                                       'requested_slots_per_copy': 10}
    return x, y, record, effective


@contextmanager
def preserve_upstream_manifest():
    """Suppress only the legacy adapter's descriptive manifest write."""
    from . import hique_network
    target = hique_network.ROOT / 'Data/hique_reproduction/model_patch_manifest.json'
    before = digest(target) if target.exists() else None
    original = Path.write_text

    def guarded(path, text, *args, **kwargs):
        if path.resolve() == target.resolve():
            return len(text)
        return original(path, text, *args, **kwargs)

    with patch.object(Path, 'write_text', guarded):
        yield
    after = digest(target) if target.exists() else None
    if before != after:
        raise ValueError('Existing upstream patch manifest changed')


def input_hashes(args):
    paths = [args.label_dir / f'{s}_split_Depression_AVEC2017.csv' for s in ('train', 'dev')]
    paths += [args.features_dir / f'{c}.npz' for c in CONDITIONS]
    paths += list(args.provenance_file)
    return {str(reject_test_path(path).resolve()): digest(path) for path in paths}


def source_hashes():
    from . import hique2_network, hique2_train, hique_network, hique_train, prepare
    paths = [__file__, hique2_network.__file__, hique2_train.__file__,
             hique_network.__file__, hique_network.SOURCE, hique_train.__file__, prepare.__file__]
    return {str(Path(path).resolve()): digest(path) for path in paths}


def frozen_config(args, data):
    first = data['E0']
    return {'input_file_hashes': input_hashes(args), 'source_hashes': source_hashes(),
            'conditions': list(CONDITIONS), 'seeds': list(SEEDS),
            'epochs': 100, 'batch_size': 8, 'learning_rate': .0002, 'dropout': .5,
            'model': 'F_AVT; identical fixed positions 0..84 in every condition',
            'normalization': 'none', 'mask': 'per modality; zero missing; no shared mask',
            'checkpoint': 'minimum dev categorical crossentropy',
            'threshold': 'positive iff p1 > .5; exact softmax tie class0',
            'augmentation': 'train positives tripled, 10 synchronized slots; all conditions share source/slot indices',
            'stages': {'initial': list(SEEDS[:3]), 'extend': list(SEEDS[3:])},
            'comparisons': [f'{a}_minus_{b}' for a, b in COMPARISONS],
            'participants': {s: first['participant_ids'][first['split'] == s].tolist() for s in ('train', 'dev')},
            'excluded_dev': sorted(DEV_EXCLUDED), 'test_files_accessed': False,
            'inference_limit': 'exploratory dev, conditional on checkpoint selection using the same dev; not held-out inference'}


def verify_run(directory, condition, seed, config_sha, ids=None):
    record = json.loads((directory / 'complete.json').read_text())
    if record['condition'] != condition or record['seed'] != seed or record['epochs'] != 100 or record['config_sha256'] != config_sha:
        raise ValueError('Run identity or configuration mismatch')
    if set(record['artifact_hashes']) != set(ARTIFACTS):
        raise ValueError('Missing run artifact hashes')
    for filename, expected in record['artifact_hashes'].items():
        if digest(directory / filename) != expected:
            raise ValueError(f'Completed run artifact changed: {filename}')
    with (directory / 'history.csv').open(newline='') as stream:
        history = list(csv.DictReader(stream))
    if len(history) != 100 or [int(row['epoch']) for row in history] != list(range(100)):
        raise ValueError('Expected 100 complete epochs')
    losses = np.array([float(row['val_loss']) for row in history])
    all_values = np.array([[float(value) for value in row.values()] for row in history])
    if not np.isfinite(all_values).all() or record['best_epoch'] != int(losses.argmin()) + 1:
        raise ValueError('Best checkpoint epoch disagrees with history')
    for split in ('train', 'dev'):
        with np.load(directory / f'{split}_predictions.npz', allow_pickle=False) as pred:
            actual = metrics(pred['truth'], pred['probability'])
            if actual != record[split]:
                raise ValueError('Saved metrics do not match predictions')
            if ids is not None and pred['participant_ids'].tolist() != ids[split]:
                raise ValueError('Prediction cohort mismatch')
    return record


def verify_stage(output, seeds, frozen):
    config_sha = digest(output / 'frozen_config.json')
    hashes = {}
    for seed in seeds:
        reference = None
        for condition in CONDITIONS:
            directory = output / f'{condition}_seed{seed}'
            verify_run(directory, condition, seed, config_sha, frozen['participants'])
            aug = digest(directory / 'augmentation.json')
            if reference is not None and aug != reference:
                raise ValueError('Paired augmentation differs between conditions')
            reference = aug
            hashes[directory.name] = digest(directory / 'complete.json')
    return hashes


def verify_initial_gate(output, frozen):
    manifest = json.loads((output / 'initial_complete.json').read_text())
    hashes = verify_stage(output, SEEDS[:3], frozen)
    if manifest['config_sha256'] != digest(output / 'frozen_config.json') or manifest['runs'] != hashes:
        raise ValueError('Initial stage changed or incomplete')
    for filename, key in [('initial_results.json', 'results_sha256'), ('controls.json', 'controls_sha256')]:
        if digest(output / filename) != manifest[key]:
            raise ValueError('Initial summary or controls changed')
    return hashes


def write_controls(output, data, labels):
    record_path = output / 'controls.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        for filename, expected in record['artifact_hashes'].items():
            if digest(output / filename) != expected:
                raise ValueError('Fitted control changed')
        return record
    controls = {'conditions': {}, 'artifact_hashes': {}}
    for condition, values in data.items():
        ti, di = values['split'] == 'train', values['split'] == 'dev'
        y = np.array([labels[int(pid)] for pid in values['participant_ids'][ti]])
        yd = np.array([labels[int(pid)] for pid in values['participant_ids'][di]])
        model = LogisticRegression(C=1, max_iter=2000, random_state=13)
        model.fit(values['question_presence'][ti], y)
        p = model.predict_proba(values['question_presence'][di])[:, 1]
        filename = f'{condition}_question_control.npz'
        np.savez(output / filename, coef=model.coef_, intercept=model.intercept_,
                 participant_ids=values['participant_ids'][di], truth=yd, probability=p)
        controls['artifact_hashes'][filename] = digest(output / filename)
        majority = int(np.bincount(y, minlength=2).argmax())
        controls['conditions'][condition] = {'question_presence': metrics(yd, p),
            'majority': metrics(yd, np.full(len(yd), majority)), 'majority_class': majority,
            'train_counts': np.bincount(y, minlength=2).tolist()}
    write_json(record_path, controls)
    return controls


def feature_attention(model, inputs, data, train_index):
    result = {}
    features = [layer for layer in model.layers if type(layer).__name__ == 'FeatureEmbedding']
    positions = [layer for layer in model.layers if type(layer).__name__ == 'SuppliedPositionEmbedding']
    for i, m in enumerate(DIMENSIONS):
        embedded = features[i](inputs[i], training=False).numpy()
        position = positions[i].pos_emb(inputs[-1]).numpy()
        block = next(layer for layer in model.layers if type(layer).__name__ == 'TransformerBlock'
                     and layer.input._keras_history.layer is positions[i])
        _, attention = block.att(embedded + position, embedded + position, return_attention_scores=True)
        attention = attention.numpy()[:, 0]
        mask = data['mask_' + m][train_index]
        entropy = -(attention * np.log(np.maximum(attention, 1e-30))).sum(axis=-1) / np.log(85)
        result[m] = {'observed_train_slots': int(mask.sum()), 'total_train_slots': int(mask.size),
                     'embedding_rms': float(np.sqrt(np.mean(embedded**2))),
                     'position_rms': float(np.sqrt(np.mean(position**2))),
                     'attention_entropy_observed': float(entropy[mask].mean()) if mask.any() else None,
                     'attention_top1_above_0_999_observed': float((attention.max(-1)[mask] > .999).mean()) if mask.any() else None}
    return result


def aggregate(output, seeds, frozen, controls):
    verify_stage(output, seeds, frozen)
    result = {'seeds': list(seeds), 'conditions': {}, 'comparisons': {}, 'controls': controls['conditions'],
              'scope': frozen['inference_limit'], 'test_files_accessed': False}
    probabilities, truth = {}, None
    for condition in CONDITIONS:
        records, ps = [], []
        for seed in seeds:
            directory = output / f'{condition}_seed{seed}'
            record = json.loads((directory / 'complete.json').read_text())
            records.append({'seed': seed, 'best_epoch': record['best_epoch'], 'train': record['train'], 'dev': record['dev']})
            with np.load(directory / 'dev_predictions.npz', allow_pickle=False) as pred:
                if truth is not None and not np.array_equal(truth, pred['truth']):
                    raise ValueError('Labels differ in paired comparison')
                truth = pred['truth'].copy()
                ps.append(pred['probability'].copy())
        summary = {}
        for split in ('train', 'dev'):
            summary[split] = {key: {'mean': float(np.mean([r[split][key] for r in records])),
                                   'std': float(np.std([r[split][key] for r in records]))}
                              for key, value in records[0][split].items() if isinstance(value, float)}
        result['conditions'][condition] = {'per_seed': records, 'summary': summary}
        probabilities[condition] = np.asarray(ps)
    for a, b in COMPARISONS:
        result['comparisons'][f'{a}_minus_{b}'] = paired_bootstrap(truth, probabilities[a], probabilities[b], repeats=2000)
    return result


def train(args):
    output = require_private_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    roster, labels = load_roster(args.label_dir)
    data = load_conditions(args.features_dir, roster)
    frozen = frozen_config(args, data)
    path = output / 'frozen_config.json'
    if path.exists():
        if json.loads(path.read_text()) != frozen:
            raise ValueError('Frozen configuration mismatch; preserve prior output')
    else:
        if args.stage != 'initial':
            raise ValueError('Initial stage must complete before extension')
        write_json(path, frozen)
    if args.stage == 'extend':
        verify_initial_gate(output, frozen)
    controls = write_controls(output, data, labels)
    tf = configure_tf()
    from . import hique2_network
    runtime = output / 'runtime.json'
    if not runtime.exists():
        write_json(runtime, {'tensorflow': tf.__version__, 'gpu': [d.name for d in tf.config.list_physical_devices('GPU')]})
    stage_seeds = SEEDS[:3] if args.stage == 'initial' else SEEDS[3:]
    config_sha = digest(path)
    for seed in stage_seeds:
        expected_augmentation = None
        for condition in CONDITIONS:
            values = data[condition]
            ti, di = values['split'] == 'train', values['split'] == 'dev'
            y = np.array([labels[int(pid)] for pid in values['participant_ids'][ti]])
            yd = np.array([labels[int(pid)] for pid in values['participant_ids'][di]])
            x, ya, augmentation, effective = augmented_inputs(values, ti, y, seed)
            if expected_augmentation is not None and augmentation != expected_augmentation:
                raise ValueError('Augmentation pairing changed')
            expected_augmentation = augmentation
            directory = output / f'{condition}_seed{seed}'
            directory.mkdir(exist_ok=True)
            if (directory / 'complete.json').exists():
                verify_run(directory, condition, seed, config_sha, frozen['participants'])
                continue
            with preserve_upstream_manifest():
                model = hique2_network.build_model('F_AVT', seed=seed, learning_rate=.0002, dropout=.5)
            write_json(directory / 'augmentation.json', augmentation)
            write_json(directory / 'effective_masking.json', effective)
            dx, tx = model_inputs(values, di), model_inputs(values, ti)
            checkpoint = directory / 'best.weights.h5'
            start = time.monotonic()
            history = model.fit(x, tf.keras.utils.to_categorical(ya, 2),
                validation_data=(dx, tf.keras.utils.to_categorical(yd, 2)), epochs=100, batch_size=8, shuffle=True, verbose=0,
                callbacks=[tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss', save_best_only=True, save_weights_only=True),
                           tf.keras.callbacks.CSVLogger(str(directory / 'history.csv'))])
            if len(history.history['loss']) != 100 or not np.isfinite(history.history['loss']).all() or not np.isfinite(history.history['val_loss']).all():
                raise ValueError('Incomplete/nonfinite training')
            model.load_weights(str(checkpoint))
            record = {'condition': condition, 'seed': seed, 'epochs': 100, 'config_sha256': config_sha,
                      'best_epoch': int(np.argmin(history.history['val_loss'])) + 1,
                      'seconds': time.monotonic() - start, 'parameters': model.count_params()}
            for split, index, inputs, truth in [('train', ti, tx, y), ('dev', di, dx, yd)]:
                p = model.predict(inputs, batch_size=8, verbose=0)[:, 1]
                np.savez(directory / f'{split}_predictions.npz', participant_ids=values['participant_ids'][index], truth=truth, probability=p)
                record[split] = metrics(truth, p)
            write_json(directory / 'feature_attention.json', feature_attention(model, tx, values, ti))
            record['artifact_hashes'] = {name: digest(directory / name) for name in ARTIFACTS}
            write_json(directory / 'complete.json', record)
            verify_run(directory, condition, seed, config_sha, frozen['participants'])
            print(json.dumps({'condition': condition, 'seed': seed, 'best_epoch': record['best_epoch'], 'dev': record['dev']}), flush=True)
    all_seeds = SEEDS[:3] if args.stage == 'initial' else SEEDS
    hashes = verify_stage(output, all_seeds, frozen)
    if frozen_config(args, data) != frozen:
        raise ValueError('Inputs or sources changed during execution')
    results_path = output / f'{args.stage}_results.json'
    write_json(results_path, aggregate(output, all_seeds, frozen, controls))
    write_json(output / f'{args.stage}_complete.json', {'config_sha256': config_sha, 'runs': hashes,
        'results_sha256': digest(results_path), 'controls_sha256': digest(output / 'controls.json'), 'test_files_accessed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', type=Path, required=True)
    parser.add_argument('--label-dir', '--zip-dir', dest='label_dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--provenance-file', type=Path, action='append', required=True)
    parser.add_argument('--stage', choices=['initial', 'extend'], required=True)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
