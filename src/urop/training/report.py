"""Render verified train/dev aggregates without modifying training artifacts."""
import argparse
import csv
import json
from pathlib import Path

from urop.io import digest, require_private_output
from urop.paths import DEFAULT_STUDY

NAMES = {'E0': 'ASR 기준선·수정된 토크나이저', 'E1': '제공 전사·마지막 답변',
         'E2': '제공 전사·첫 답변', 'E3': '전체 답변·512토큰 제한', 'E4': '전체 답변·블록 가중 평균'}


def write_table(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison_lines(result):
    lines = ['', '| 비교 | 시드별 Macro-F1 차이 평균 | 참가자 재표집 95% 구간 |',
             '|---|---:|---:|']
    for key, value in result['comparisons'].items():
        lo, hi = value['ci95']
        lines.append(f"| {key} | {value['delta_macro_f1']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ['', '동일 참가자를 모든 고정 시드·비교 조건에서 함께 재표집했다. 시드별 지표 차이를 평균하며 확률 앙상블 비교와 다르다.']
    return lines


def render(base):
    base = require_private_output(base)
    runs = base / 'runs'
    completion = json.loads((runs / 'extend_complete.json').read_text())
    if completion['results_sha256'] != digest(runs / 'extend_results.json'):
        raise ValueError('Changed training results')
    if completion['config_sha256'] != digest(runs / 'frozen_config.json'):
        raise ValueError('Changed training configuration')
    result = json.loads((runs / 'extend_results.json').read_text())
    config = json.loads((runs / 'frozen_config.json').read_text())
    rows = []
    for condition in config['conditions']:
        row = {'condition': condition, 'description': NAMES[condition]}
        for split in ('train', 'dev'):
            for key in ('macro_f1', 'positive_f1', 'sensitivity', 'specificity', 'auroc'):
                summary = result['conditions'][condition]['summary'][split][key]
                row[f'{split}_{key}_mean'] = summary['mean']
                row[f'{split}_{key}_sd'] = summary['std']
        rows.append(row)
    write_table(base / 'results.csv', rows)
    lines = ['# HiQuE 입력 비교: train/dev 결과', '',
             f"학습 {len(config['participants']['train'])}명 / 검증 {len(config['participants']['dev'])}명. 시드: {config['seeds']}.",
             '', '검증 집합은 체크포인트 선택에도 사용했다. 아래 값은 탐색적 결과이며 미노출 테스트 성능이나 논문 재현 성공을 의미하지 않는다.',
             '', '| 조건 | 학습 Macro-F1 평균 | 검증 Macro-F1 평균 ± SD |', '|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['condition']} {row['description']} | {row['train_macro_f1_mean']:.4f} | {row['dev_macro_f1_mean']:.4f} ± {row['dev_macro_f1_sd']:.4f} |")
    lines += comparison_lines(result)
    lines += ['', 'SD는 시드 간 표준편차(ddof=0)다. 체크포인트 선택에 사용한 dev의 재표집 구간은 선택 편향을 제거하지 않는다.',
              'E1−E0는 전사·화자·시간·질문 매핑을 함께 변경한 효과다. 모든 답변을 모으는 E3/E4는 질문 부모 맥락을 합치며, 이 고정 위치 실험은 원 논문 계층 구현과 동치가 아니다.',
              '', '설정과 출처: runs/frozen_config.json. 원본 예측·학습 기록·증강 기록은 해당 run에 보존된다.']
    (base / 'results_ko.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DEFAULT_STUDY)
    render(parser.parse_args().base)


if __name__ == '__main__':
    main()
