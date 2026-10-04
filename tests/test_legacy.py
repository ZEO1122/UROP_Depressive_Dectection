import hashlib
import json
from pathlib import Path

import pytest

from urop.legacy import materialized_sources, snapshot_files


def sample(tmp_path):
    root = tmp_path / 'repo'
    snap = root / 'archive/snapshot'
    snap.mkdir(parents=True)
    relative = 'experiments/old.py'
    source = snap / relative
    source.parent.mkdir()
    source.write_text('VALUE = 42\n')
    meta = {relative: {'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'bytes': source.stat().st_size}}
    (snap / 'manifest.json').write_text(json.dumps({'files': meta}))
    return root, snap, root / relative


def test_legacy_restores_exact_source_and_removes_only_created(tmp_path):
    root, snap, target = sample(tmp_path)
    with materialized_sources(root, snap):
        assert target.read_bytes() == (snap / 'experiments/old.py').read_bytes()
        assert (root / '.urop-legacy.lock').exists()
    assert not target.exists()
    assert not (root / '.urop-legacy.lock').exists()
    assert len(snapshot_files(snap)) == 1


def test_legacy_cleanup_after_exception_keeps_preexisting_file(tmp_path):
    root, snap, target = sample(tmp_path)
    target.parent.mkdir()
    target.write_bytes((snap / 'experiments/old.py').read_bytes())
    with pytest.raises(RuntimeError, match='command failed'):
        with materialized_sources(root, snap):
            raise RuntimeError('command failed')
    assert target.read_text() == 'VALUE = 42\n'
    assert not (root / '.urop-legacy.lock').exists()


def test_legacy_refuses_collision_without_overwriting(tmp_path):
    root, snap, target = sample(tmp_path)
    target.parent.mkdir()
    target.write_text('user work\n')
    with pytest.raises(ValueError, match='refusing overwrite'):
        with materialized_sources(root, snap):
            pytest.fail('must not execute')
    assert target.read_text() == 'user work\n'
    assert not (root / '.urop-legacy.lock').exists()


def test_legacy_retains_changed_temporary_source(tmp_path):
    root, snap, target = sample(tmp_path)
    with pytest.raises(RuntimeError, match='retained'):
        with materialized_sources(root, snap):
            target.write_text('modified during execution\n')
    assert target.read_text() == 'modified during execution\n'
    assert not (root / '.urop-legacy.lock').exists()


def test_legacy_detects_archive_tampering(tmp_path):
    root, snap, _ = sample(tmp_path)
    (snap / 'experiments/old.py').write_text('altered snapshot\n')
    with pytest.raises(ValueError, match='snapshot changed'):
        with materialized_sources(root, snap):
            pytest.fail('must not execute')
    assert not (root / '.urop-legacy.lock').exists()


def test_legacy_refuses_symlink_and_existing_lock(tmp_path):
    root, snap, target = sample(tmp_path)
    target.parent.mkdir()
    target.symlink_to(snap / 'experiments/old.py')
    with pytest.raises(ValueError, match='symlink'):
        with materialized_sources(root, snap):
            pass
    target.unlink()
    (root / '.urop-legacy.lock').write_text('other process\n')
    with pytest.raises(FileExistsError):
        with materialized_sources(root, snap):
            pass
    assert (root / '.urop-legacy.lock').read_text() == 'other process\n'
