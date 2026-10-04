"""Temporarily restore byte-identical historical source paths for audits or commands."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

from .io import digest
from .paths import ROOT

SNAPSHOT = ROOT / 'archive/snapshots/pre-reorg-2026-10-04'


def snapshot_files(snapshot=SNAPSHOT):
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    for relative, metadata in manifest['files'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or path.parts[0] not in ('experiments', 'tests'):
            raise ValueError(f'Unsafe snapshot path: {relative}')
        source = snapshot / path
        if source.is_symlink() or digest(source) != metadata['sha256']:
            raise ValueError(f'Historical snapshot changed: {relative}')
    return manifest['files']


@contextmanager
def materialized_sources(root=ROOT, snapshot=SNAPSHOT):
    """Never replace existing files; remove only matching files created by this call."""
    files = snapshot_files(snapshot)
    sources = {p: meta for p, meta in files.items() if Path(p).parts[0] == 'experiments'}
    for relative, metadata in sources.items():
        target = root / relative
        if target.parent.is_symlink() or target.is_symlink():
            raise ValueError(f'Refusing legacy symlink collision: {target}')
        if target.exists() and digest(target) != metadata['sha256']:
            raise ValueError(f'Existing file differs; refusing overwrite: {target}')
    lock = root / '.urop-legacy.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    created = []
    changed = []
    try:
        for relative, metadata in sources.items():
            target = root / relative
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive create protects against a concurrent user file appearing.
            with target.open('xb') as stream:
                stream.write((snapshot / relative).read_bytes())
            created.append((target, metadata['sha256']))
        yield
    finally:
        for target, expected in reversed(created):
            if target.is_symlink() or (target.exists() and digest(target) != expected):
                changed.append(str(target))  # Preserve edits instead of deleting them.
            elif target.exists():
                target.unlink()
        lock.unlink(missing_ok=True)
        if changed:
            raise RuntimeError('Legacy command changed temporary sources; retained for inspection: ' + ', '.join(changed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('verify', help='read-only snapshot and historical artifact hash audit')
    run = sub.add_parser('run', help='restore original paths for one explicitly requested historical command')
    run.add_argument('--python', default=sys.executable, help='interpreter with the original dependencies')
    run.add_argument('module', help='archived experiments module or pytest')
    run.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == 'verify':
        from .provenance import audit_historical_artifacts
        with materialized_sources():
            result = audit_historical_artifacts()
        print(json.dumps({'snapshot_files': len(snapshot_files()), **result}, indent=2))
        return 0
    modules = {'experiments.' + Path(p).stem for p in snapshot_files()
               if p.startswith('experiments/') and Path(p).name != '__init__.py'}
    if args.module not in modules | {'pytest'}:
        parser.error('Only archived experiments modules and pytest are accepted')
    extra = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    with materialized_sources():
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([args.python, '-m', args.module, *extra], cwd=ROOT, env=env, check=False)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
