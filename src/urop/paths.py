"""Repository paths; existing private artifacts are never relocated implicitly."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / 'Data'
DEFAULT_STUDY = Path('Data/runs/hique_input_study')
UPSTREAM_SOURCE = DATA_ROOT / 'hique_reproduction/upstream/code/fusionmodel.py'
