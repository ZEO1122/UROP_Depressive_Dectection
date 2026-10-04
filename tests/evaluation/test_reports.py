import json
import os
import subprocess
import sys

import pytest
from urop.training import report as dev_report
from urop.evaluation import report as test_report
from urop.paths import ROOT


def test_dev_report_rejects_changed_aggregate_before_writing(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    runs.mkdir()
    (runs / 'extend_complete.json').write_text(json.dumps({'results_sha256': 'wrong'}))
    (runs / 'extend_results.json').write_text('{}')
    monkeypatch.setattr(dev_report, 'require_private_output', lambda p: p)
    with pytest.raises(ValueError, match='Changed training results'):
        dev_report.render(tmp_path)
    assert not (tmp_path / 'results.csv').exists()


def test_test_report_rejects_incomplete_evaluation(tmp_path, monkeypatch):
    runs = tmp_path / 'test_evaluation/runs'
    runs.mkdir(parents=True)
    (runs / 'evaluation_manifest.json').write_text('{"status":"in_progress"}')
    monkeypatch.setattr(test_report, 'require_private_output', lambda p: p)
    with pytest.raises(ValueError, match='Incomplete'):
        test_report.render(tmp_path)
    assert not (tmp_path / 'test_evaluation/results.csv').exists()


@pytest.mark.parametrize('command', ['train', 'test-features', 'evaluate', 'report', 'test-report'])
def test_cli_help_is_passed_to_selected_command(command):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT / 'src'), env.get('PYTHONPATH', '')]))
    process = subprocess.run([sys.executable, '-m', 'urop', command, '--help'], text=True, capture_output=True, env=env)
    assert process.returncode == 0, process.stderr
    assert 'usage:' in process.stdout
