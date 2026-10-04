"""Source fingerprints and read-only checks of completed experiment artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .io import digest
from .paths import ROOT


def source_hashes() -> dict[str, str]:
    return {str(p.resolve()): digest(p) for p in sorted((ROOT / 'src/urop').rglob('*.py'))}


def audit_historical_artifacts() -> dict:
    """Call with legacy sources materialized; never rewrite historical manifests."""
    cache: dict[str, str] = {}
    manifests: list[str] = []

    def check(path, expected):
        path = Path(path)
        if not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError(f'Invalid SHA256 in manifest for {path}')
        key = str(path.resolve())
        if key not in cache:
            cache[key] = digest(path)
        if cache[key] != expected:
            raise ValueError(f'Historical content changed: {path}')

    def load(path):
        manifests.append(str(path.relative_to(ROOT)))
        return json.loads(path.read_text())

    roots = [ROOT / 'Data/hique_reproduction/runs',
             ROOT / 'Data/hique_reproduction_excluding440/runs',
             ROOT / 'Data/hique_input_study_v3/runs',
             ROOT / 'Data/hique_input_study_v3/test_evaluation/runs']
    found = 0
    for runs in roots:
        freeze = runs / ('frozen_evaluation.json' if (runs / 'frozen_evaluation.json').exists() else 'frozen_config.json')
        if not freeze.exists():
            continue
        found += 1
        record = load(freeze)
        for key in ('input_file_hashes', 'source_hashes'):
            for path, expected in record.get(key, {}).items():
                check(path, expected)
        if 'trainer_sha256' in record:
            check(ROOT / 'experiments/hique_train.py', record['trainer_sha256'])
            check(ROOT / 'experiments/hique_network.py', record['model_sha256'])
        if 'feature_sha256' in record:
            check(runs.parent / 'features.npz', record['feature_sha256'])
        if 'evaluation_source_sha256' in record:
            check(ROOT / 'experiments/hique3_test.py', record['evaluation_source_sha256'])
        for path in sorted(runs.glob('*/complete.json')):
            run = load(path)
            for filename, expected in run.get('artifact_hashes', {}).items():
                check(path.parent / filename, expected)
            for field, filename in [('weights_sha256', 'best.weights.h5'), ('history_sha256', 'history.csv'),
                                    ('dev_sha256', 'dev_predictions.npz'), ('augmentation_sha256', 'augmentation.json'),
                                    ('prediction_sha256', 'test_predictions.npz')]:
                if field in run:
                    check(path.parent / filename, run[field])
            if 'checkpoint_source' in run:
                check(run['checkpoint_source'], run['checkpoint_sha256'])
        controls = runs / 'controls.json'
        if controls.exists():
            for filename, expected in load(controls)['artifact_hashes'].items():
                check(runs / filename, expected)
        for filename in ('initial_complete.json', 'extend_complete.json', 'training_complete.json', 'evaluation_manifest.json'):
            path = runs / filename
            if not path.exists():
                continue
            complete = load(path)
            if 'config_sha256' in complete:
                check(runs / 'frozen_config.json', complete['config_sha256'])
            if 'evaluation_config_sha256' in complete:
                check(runs / 'frozen_evaluation.json', complete['evaluation_config_sha256'])
            if 'test_results_sha256' in complete:
                check(runs / 'test_results.json', complete['test_results_sha256'])
            if 'results_sha256' in complete:
                check(runs / filename.replace('_complete', '_results'), complete['results_sha256'])
            if 'controls_sha256' in complete:
                check(runs / 'controls.json', complete['controls_sha256'])
            for field, target in [('scalers_sha256', 'scalers.npz'), ('question_control_sha256', 'question_control.npz')]:
                if field in complete:
                    check(runs / target, complete[field])
            for target, expected in complete.get('control_hashes', {}).items():
                check(runs / target, expected)
            if isinstance(complete.get('runs'), dict):
                for name, expected in complete['runs'].items():
                    check(runs / name / 'complete.json', expected)
    if not found:
        return {'status': 'unavailable', 'reason': 'No local historical run manifests found', 'checked_files': 0}
    return {'status': 'passed', 'run_groups': found, 'checked_files': len(cache),
            'manifests_read': len(manifests), 'mutated_artifacts': 0}
