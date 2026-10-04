"""Shared artifact hashing and private output guard."""
import hashlib
import json
from pathlib import Path
from typing import Any
from urop.paths import DATA_ROOT

def digest(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def require_private_output(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(DATA_ROOT.resolve()):
        raise ValueError('Participant outputs must stay under the repository Data directory')
    return resolved
