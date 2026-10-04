"""Render verified fixed-checkpoint reference test aggregates."""
import argparse
import json
from pathlib import Path

from urop.io import digest, require_private_output
from urop.paths import DEFAULT_STUDY
from urop.training.report import NAMES, comparison_lines, write_table


def render(base):
    base = require_private_output(base)
    test = base / 'test_evaluation'
    runs = test / 'runs'
    manifest = json.loads((runs / 'evaluation_manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['test_results_sha256'] != digest(runs / 'test_results.json'):
        raise ValueError('Incomplete or changed evaluation')
    result = json.loads((runs / 'test_results.json').read_text())
    dev = json.loads((base / 'runs/extend_results.json').read_text())
    dev_completion = json.loads((base / 'runs/extend_complete.json').read_text())
    if dev_completion['results_sha256'] != digest(base / 'runs/extend_results.json'):
        raise ValueError('Changed dev results')
    rows = []
    for condition in NAMES:
        summary = result['conditions'][condition]['per_seed_summary']
        row = {'condition': condition, 'description': NAMES[condition],
               'dev_macro_f1_mean': dev['conditions'][condition]['summary']['dev']['macro_f1']['mean']}
        for key in ('macro_f1', 'positive_f1', 'sensitivity', 'specificity', 'auroc', 'weighted_f1'):
            row[key + '_mean'] = summary[key]['mean']
            row[key + '_sd'] = summary[key]['std']
        rows.append(row)
    write_table(test / 'results.csv', rows)
    lines = ['# HiQuE 고정 모델 test 평가', '',
             f"참가자 {len(result['test_ids'])}명. Dev 사전 선택 참고 조건: {result['pretest_dev_selected_reference']}.",
             '기존 test 노출 이력이 있는 참고 benchmark 결과다. 재학습이나 test 기반 모델 선택을 하지 않은 고정 체크포인트 평가다.', '',
             '| 조건 | Dev Macro-F1 평균 | Test Macro-F1 평균 ± SD |', '|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['condition']} {row['description']} | {row['dev_macro_f1_mean']:.4f} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_sd']:.4f} |")
    lines += comparison_lines(result)
    lo, hi = result['comparisons']['E1_minus_E0']['ci95']
    lines += ['', ('주 비교 구간에 0이 포함되므로 우위가 확인됐다고 결론 내리지 않는다.' if lo <= 0 <= hi else
                   '주 비교 구간에 0이 포함되지 않지만 기존 test 노출이 있는 독립 확증 평가가 아니다.'),
              'SD는 시드 간 표준편차(ddof=0)이며 참가자 신뢰구간이 아니다. 참가자 재표집은 훈련 자료·설계 선택의 불확실성을 모두 반영하지 않는다.',
              '원 논문과 표본·전처리·모델 가정의 차이가 있다. PHQ 선별 라벨 예측이며 임상 진단 성능이 아니다.', '',
              '출처: test_protocol.json, runs/frozen_evaluation.json, runs/test_results.json 및 runs/evaluation_manifest.json.']
    (test / 'results_ko.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DEFAULT_STUDY)
    render(parser.parse_args().base)


if __name__ == '__main__':
    main()
