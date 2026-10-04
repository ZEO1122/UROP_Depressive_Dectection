"""Report adapted public-code HiQuE replication, preserving evaluation distinctions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .prepare import require_private_output


def render(base: Path) -> str:
    base = require_private_output(base)
    runs = base / 'runs'
    config = json.loads((runs / 'frozen_config.json').read_text())
    results = json.loads((runs / 'test_results.json').read_text())
    exposure = json.loads((runs / 'evaluation_manifest.json').read_text())
    if exposure['status'] != 'complete' or hashlib.sha256((runs / 'test_results.json').read_bytes()).hexdigest() != exposure['test_results_sha256']:
        raise ValueError('Evaluation is incomplete or result hash differs')
    text_info = json.loads((base / 'text_summary.json').read_text())
    prep = json.loads((base / 'preparation_summary.json').read_text())
    counts = {split: len(ids) for split, ids in config['participants'].items()}
    rows = []
    for mode in config['modalities']:
        summary = results[mode]['per_seed_summary']
        row = {'modalities': mode}
        for metric in ('macro_f1', 'positive_f1', 'sensitivity', 'specificity', 'auroc'):
            row[metric + '_mean'] = summary.get(metric, {}).get('mean')
            row[metric + '_seed_sd'] = summary.get(metric, {}).get('std')
        rows.append(row)
    with (base / 'modality_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# HiQuE 공개 코드 기반 수정 재현 결과', '',
             '**실제 PHQ-8 라벨로 학습·평가한 결과다. 기존 AI 주석 일치도 실험과 다르며, 임상 진단 성능은 아니다.**', '',
             '정확히 동일한 원 실험의 숫자 재현이 아니라, 공개 모델 구조를 수정하고 논문의 특징을 맞춘 adapted reproduction이다.', '',
             '## 실행 조건', '',
             '- 원 코드: https://github.com/JuHo-Jung/HiQuE, commit `24c553bf2666b442ae5b0e3490b998a5d4493559`.',
             f"- 동일 공통 cohort: train {counts['train']} / dev {counts['dev']} / test {counts['test']}명. 각 참가자는 한 split에만 존재한다.",
             f"- {len(config['modalities'])}개 모달리티 구성 × {len(config['seeds'])}개 seed = {len(config['modalities'])*len(config['seeds'])}회 학습. seed {config['seeds']}.",
             f"- 각 {config['epochs']} epochs, batch 8, Adam lr 0.0002, head dropout 0.5. Dev loss 최소 checkpoint, 분류 threshold 0.5.",
             '- PHQ8_Binary 원본 유지, 409의 score/binary 불일치를 자동 수정하지 않음. Test label은 사용자 제공 full_test_split.csv이며 외부 배포본 진위까지 인증하지는 않았다.',
             '- Train에서만 scaler를 fit하고 양성 복사본 두 개를 추가해 3배로 증강한다. 각 복사본의 질문 10개를 모든 modality에서 동일하게 가린다.',
             f"- Text: frozen RoBERTa-base CLS, {text_info['encoded_slots']}개 응답 slot; 512-token 초과 {text_info['truncated_at_512']}개를 사전 고정된 방식으로 자름.",
             '- Audio: openSMILE eGeMAPSv02 functionals 88. Visual: CLNF 2D 68개 좌표의 1fps 평균·분산 272.',
             '- 모든 조건에 동일한 T/A/V 공통 관측 slot mask를 사용한다. 질문 등장 여부만 쓰는 LR 대조군은 별도로 평가한다.',
             '- 원본 데이터·AI 주석은 변경하지 않았다. 새로운 모델은 AI 주석을 학습 target으로 사용하지 않았다.', '',
             '## Test 결과: seed별 점수의 평균', '',
             '아래 ±는 사전 고정 seed 간 표준편차(ddof=0)이며 참가자 신뢰구간이 아니다. 좋은 seed만 선택하지 않았다.', '',
             '| 입력 | Macro-F1 평균 ± SD | 양성 F1 평균 | 민감도 평균 | 특이도 평균 | AUROC 평균 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        def fmt(key):
            return '계산 불가' if row[key] is None else f"{row[key]:.4f}"
        lines.append(f"| {row['modalities']} | {fmt('macro_f1_mean')} ± {fmt('macro_f1_seed_sd')} | {fmt('positive_f1_mean')} | {fmt('sensitivity_mean')} | {fmt('specificity_mean')} | {fmt('auroc_mean')} |")
    control = results['question_presence_control']
    best = max(rows, key=lambda row: row['macro_f1_mean'])
    lines.extend(['', f"관측된 평균 Macro-F1은 **{best['modalities']} ({best['macro_f1_mean']:.4f})**에서 가장 높았다. 이는 test 결과의 기술적 요약이며 test에서 새 모델을 선택하거나 그 우위를 확증한 분석은 아니다."])
    lines.extend(['', f"질문 등장 여부만 사용하는 대조군: Macro-F1 **{control['macro_f1']:.4f}**, 양성 F1 {control['positive_f1']:.4f}. 실제 참가자 발화·음성·얼굴 특징을 입력하지 않은 진단용 대조이며, 질문 경로의 영향을 점검한다.", '',
                  '## 사전 지정한 주요 비교: AVT − T', ''])
    difference = results['primary_AVT_minus_T']
    lower, upper = difference['ci95']
    if lower > 0:
        interpretation = '이 표본에서 seed 평균 예측의 AVT가 T보다 높았고, 참가자 재표집 CI도 0보다 컸다. 외부 데이터에서의 우위나 원 논문 성능 재현을 뜻하지는 않는다.'
    elif upper < 0:
        interpretation = '이 표본에서 seed 평균 예측의 AVT가 T보다 낮았고, 참가자 재표집 CI도 0보다 작았다. 현재 설정에서 세 모달리티 결합의 이점은 확인되지 않았다.'
    else:
        interpretation = '차이의 CI가 0을 포함하므로, 이 실험만으로 세 모달리티 결합이 텍스트 단독보다 우수하다고 결론 내릴 수 없다.'
    lines.extend([f"5개 seed의 예측 확률을 참가자별로 먼저 평균한 후 평가한 Macro-F1 차이: **{difference['delta_macro_f1']:+.4f}**.", '',
                  f"참가자 paired bootstrap {difference['repeats']}회 95% CI: **[{difference['ci95'][0]:+.4f}, {difference['ci95'][1]:+.4f}]**.", '',
                  interpretation, '',
                  '이 비교는 위 표의 seed별 F1 평균 차이와 다른 계산이다. 다른 21쌍 비교는 탐색 분석으로 test_results.json에 보관한다. 이 CI는 학습된 모델을 고정한 작은 test cohort의 재표집 불확실성으로, 훈련 데이터·seed의 모든 불확실성까지 포함하지 않는다. 다른 병원·언어·임상 진단으로 일반화를 보장하지 않는다.', '',
                  '## 원 논문과의 차이·코드 수정', '',
                  '- 2026-09-28 대조에서 확인: build_slots는 고정 85개 질문 ID에 응답을 합치며, 논문 §4.3의 선행 주질문/후속 질문별 parent topic 관계를 별도 입력하지 않는다. PositionEmbedding도 고정 슬롯 위치만 사용한다. 따라서 핵심 hierarchical question embedding을 충실히 재현한 결과로 해석할 수 없다. 이 차이의 성능 영향은 분리 실험으로 검증하지 않았다.',
                  '- 공개 source의 TransformerBlock이 (tensor, attention score)를 반환해 다음 layer에 tuple이 들어가는 오류를 tensor 출력으로 수정했다. get_config의 없는 속성과 optimizer 호환성도 고쳤다. AVT는 수정된 원 hique()를 직접 호출한다.',
                  '- 단일·이중 조건은 같은 AVT backbone에서 branch를 제거한 통제 ablation이다. 공식 저장소의 서로 다른 named baseline을 그대로 혼합하지 않았다.',
                  '- 원 공개 loader는 참가자별 feature 누적과 시각 결측 차원에 오류가 있어 새 검증 loader를 사용했다. 원 class_report 호출 순서 대신 truth/probability를 명확히 구분한 지표를 계산했다.',
                  '- 공개 visual 추출기의 VGG/fixed1–4초 경로 대신 논문에 명시된 CLNF 얼굴 좌표 통계를 구현했다. 공개 text 경로의 BERT/TensorFlow/PyTorch 혼용 대신 논문 RoBERTa CLS를 사용했다.',
                  '- 제공 수동 전사를 사용했다. 질문은 논문 Appendix의 85개에 normalized exact 및 frozen RoBERTa-base token matching으로 대응시켰다. 원 README의 ASR/BERTScore 경로와 다른 mapping adaptation이다.',
                  f"- 질문 mapping: {prep['unique_question_mapping']}. 참가자 전사에 대한 통계이며 라벨로 매핑을 조정하지 않았다.",
                  '- Participant-only 구간, scrubbed/overlap 제거, 품질 임계값, train-only standardization, 공통 slot mask는 명시한 재현 보강 조건이다. 원 논문의 동일 입력·189명 전체 결과라고 주장하지 않는다.',
                  '- 440 손상, 451/458/480 질문 전사 누락, 300 사전 사례 노출을 제외한다. 402는 가용 시각 구간만 사용한다. 문헌의 공개 사례에 대한 노출 이력도 protocol.json에 기록했다.',
                  '- Attention 점수는 설명의 정당성이나 실제 은폐를 검증하지 않는다. 이 실험의 정답은 PHQ 자기보고 기반 선별 label이다.', '',
                  '## 검증·재현 자료', '',
                  '- protocol.json: 특징·분할·seed·학습 정책. frozen_config.json: 실제 feature/코드/라벨/프로토콜 hash.',
                  '- model_patch_manifest.json: 원 코드 SHA와 수정 목록. environment_tf.txt / environment_features.txt: 환경 버전.',
                  '- runs/{mode}_seed{seed}/: 100epoch history, devlossbestweights, dev/testprediction, augmentation log.',
                  '- evaluation_manifest.json: 모든 학습 산출물 검증 뒤 test 결과를 읽은 시점·완료 hash. test 결과를 보고 설정을 바꾸지 않았다.',
                  '- verification.json: 35개 run의 hash·100epoch history·최저 dev loss checkpoint·test ID 및 별도 계산한 Macro-F1 검증 통과.',
                  '- upstream/, av/, text.npz, features.npz 및 참가자별 자료는 Git 제외되는 Data/hique_reproduction 내부에 보관한다.', '',
                  '실행 방법은 experiments/HIQUE_REPRODUCTION.md를 참고한다.'])
    result = '\n'.join(lines) + '\n'
    (base / 'results_ko.md').write_text(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('Data/hique_reproduction'))
    args = parser.parse_args()
    render(args.data_dir)


if __name__ == '__main__':
    main()
