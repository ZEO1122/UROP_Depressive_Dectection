"""Frozen source/input changes must fail without replacing prior metadata."""
import json

import pytest

from urop.features.pipeline import freeze
from urop.io import digest, require_private_output
from urop.paths import DATA_ROOT, ROOT


def test_digest_accepts_path_objects_and_frozen_json_strings(tmp_path):
    import hashlib

    path = tmp_path / 'source.py'
    data = b'preserved source bytes\n'
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert digest(path) == digest(str(path)) == expected
    assert path.read_bytes() == data


def test_paths_and_private_guard(tmp_path):
    assert (ROOT / 'src/urop/paths.py').is_file()
    assert require_private_output(DATA_ROOT / 'runs/example') == DATA_ROOT.resolve() / 'runs/example'
    with pytest.raises(ValueError, match='repository Data'):
        require_private_output(tmp_path)


def test_freeze_retains_existing_metadata_on_changed_input(tmp_path):
    base = tmp_path / 'study'
    (base / 'features').mkdir(parents=True)
    for variant in ('E0', 'E1', 'E2', 'E3'):
        (base / f'segments_{variant}.jsonl').write_text('{}\n')
    (base / 'manifest.json').write_text('{}\n')
    baseline = tmp_path / 'baseline.npz'
    baseline.write_bytes(b'baseline fixture')
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'tokenizer.json').write_text('{}\n')
    payload = freeze(base, baseline, model)
    frozen = base / 'features/frozen_inputs.json'
    before = frozen.read_bytes()
    assert payload == freeze(base, baseline, model)
    assert frozen.read_bytes() == before
    assert any('src/urop/features/text.py' in name for name in payload['input_sha256'])
    assert not any(name.startswith('experiments/') for name in payload['input_sha256'])
    baseline.write_bytes(b'changed fixture')
    with pytest.raises(ValueError, match='frozen input/source/model changed'):
        freeze(base, baseline, model)
    assert frozen.read_bytes() == before
    assert json.loads(before) == payload
