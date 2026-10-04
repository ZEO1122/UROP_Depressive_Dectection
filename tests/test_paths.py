from pathlib import Path

from urop import paths
from urop.io import require_private_output
from urop.provenance import source_hashes


def test_src_layout_resolves_repository_data_and_sources():
    root = Path(__file__).resolve().parents[1]
    assert paths.ROOT == root
    assert paths.DATA_ROOT == root / 'Data'
    assert paths.UPSTREAM_SOURCE == root / 'Data/hique_reproduction/upstream/code/fusionmodel.py'
    assert require_private_output(root / 'Data/runs/notebook') == root / 'Data/runs/notebook'
    hashes = source_hashes()
    assert str(root / 'src/urop/models/hique.py') in hashes
    assert str(root / 'src/urop/paths.py') in hashes
