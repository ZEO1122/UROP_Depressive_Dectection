"""Aggregate the excluding-440 study without claiming exact author reproduction."""
import csv
import json
from pathlib import Path

from .hique_train import digest


def main():
    base = Path('Data/hique_reproduction_excluding440')
    run = base / 'runs'
    results = json.loads((run / 'test_results.json').read_text())
    exposure = json.loads((run / 'evaluation_manifest.json').read_text())
    assert exposure['status'] == 'complete' and exposure['test_results_sha256'] == digest(run / 'test_results.json')
    config = json.loads((run / 'frozen_config.json').read_text())
    prep = json.loads((base / 'preparation_summary.json').read_text())
    eligibility = json.loads((base / 'eligibility.json').read_text())
    audit = json.loads((base / 'speaker_audit.json').read_text())
    text = json.loads((base / 'text_summary.json').read_text())
    asr = json.loads((base / 'asr/observational_summary.json').read_text())
    names = {'H_AVT': '계층 위치 A+V+T', 'F_AVT': '고정 위치 A+V+T', 'H_T': '계층 위치 T'}
    rows = []
    for condition in config['conditions']:
        result = results[condition]
        primary = next(s for s in result['per_seed'] if s['seed'] == 42)
        row = {'condition': condition, 'seed42_macro_f1': primary['macro_f1']}
        for key in ['macro_precision', 'macro_recall', 'macro_f1', 'weighted_precision', 'weighted_recall',
                    'weighted_f1', 'gmean', 'positive_f1', 'sensitivity', 'specificity', 'auroc']:
            summary = result['per_seed_summary'].get(key, {})
            row[key + '_mean'] = summary.get('mean')
            row[key + '_sd'] = summary.get('std')
        rows.append(row)
    with (base / 'modality_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    count = {s:len(ids) for s, ids in config['participants'].items()}
    lines = ['# 440 제외 HiQuE 공개 코드·논문 기반 재구현 결과', '',
             '**저자의 미공개 계층 구현까지 검증한 동일 조건 재현은 아니다. 아래 결과는 공개 전사 규칙과 명시적 부모 위치 가정을 사용한 실행 결과다.**', '',
             '## 실제 실행', '',
             f"- Train {count['train']} / dev {count['dev']} / test {count['test']}명. 공식 분할, 440 제외. 그 외 입력을 확보하지 못해 제외한 참가자: {eligibility['excluded_input_unavailable']}.",
             '- 공개 코드 commit `24c553bf2666b442ae5b0e3490b998a5d4493559`. Whisper base와 RoBERTa/BERTScore는 로컬 GPU에서 실행, 외부 LLM API는 사용하지 않았다.',
             '- 세 조건 × 시드 42/13/23/37/79 = 15회, 각 100 epochs. Batch size 8, Adam 학습률 0.0002, dropout 0.5를 사용했다. 검증 손실이 가장 낮은 checkpoint를 선택하고 softmax argmax로 분류했다(동률이면 class 0).',
             '- 양성 표본만 원본과 복사본 2개로 증강했다. 각 복사본에서 질문 10개의 특징을 모든 모달리티에 걸쳐 함께 마스킹하고 위치 ID는 유지했다. H/F의 특징 입력과 증강은 동일하다.',
             '- 결측 특징은 모달리티별로 0을 채웠으며, 공통 A/V/T 마스크와 표준화는 적용하지 않았다. 마지막 작은 배치도 포함했다(공개 코드의 floor steps와 다름).',
             '- H/F는 파라미터 12,554개의 동일한 네트워크에서 위치 ID만 달라진다. H_T는 같은 텍스트 backbone을 사용하는 통제 ablation이며, 원 저자의 별도 텍스트 baseline과 동일하다고 단정하지 않는다.', '',
             '## 입력과 계층 가정', '',
             '- Whisper base로 전사한 뒤 문장 끝 물음표로 Ellie/Participant를 추정하고 같은 화자를 합치는 공개 규칙을 적용했다. 이는 화자분리 모델이나 검증된 면담자 구간 정답이 아니다.',
             '- BERTScore 0.3.13의 영어 기본 설정(roberta-large, 17번째 층, IDF 및 rescaling 미사용)으로 85개 질문 중 F1이 가장 높은 질문에 대응시켰다. 기존 자체 임계값 0.75는 제거했다.',
             '- 후속 질문의 위치는 가장 최근 주질문(primary topic)의 위치를 상속한다. 같은 표준 질문(canonical question)이 반복되면 마지막 응답과 부모 질문을 남긴다. 이런 위치 연산과 충돌 정책은 명시적 재구현 가정이다.',
             f"- 이벤트/충돌 통계: {prep['audit']}.",
             f"- 텍스트 응답 {text['encoded_slots']}개 중 {text['truncated_slots']}개를 512 tokens에서 잘랐다. RoBERTa-base의 가중치를 고정하고 마지막 층의 CLS 표현을 사용했다.",
             '- 음성(A): eGeMAPSv02 88차원. 영상(V): CLNF 68개 점의 x/y 좌표를 1 fps로 샘플링한 평균·모분산 272차원. success=1이고 좌표가 유한한 프레임을 사용하며, 한 프레임만 있어도 허용했다. Confidence/coverage에 임의 임계값을 적용하지 않았다.', '',
             '## 공개 화자 추정 규칙의 한계', '',
             f"사전 선택한 첫 10명의 train 참가자에서 질문으로 추정한 시간 중 제공 전사의 Ellie 시간과 겹친 비율은 {audit['fraction_overlapping_manual_ellie']:.1%}, Participant 시간과 겹친 비율은 {audit['fraction_overlapping_manual_participant']:.1%}였다.", '',
             '이는 시간 경계 차이와 겹침이 있는 기술 통계이며 정확한 화자분리 precision/recall은 아니다. 질문 복원이 검증됐다고 해석할 수 없으며, 188명의 전사 생성과 최종 적격 참가자의 모델 입력 생성도 실제 질문 복원 성공과 구분해야 한다. 이 감사에서는 예측 대상 라벨을 사용하지 않았고, 감사 결과에 따라 질문 규칙을 변경하지 않았다.', '',
             '## Test 결과', '',
             '±는 5개 시드 간 표준편차(ddof=0)다. 시드 42는 공개 CLI의 기본 시드에 근거한 기준 실행이며, 원 논문 Table 1에 사용된 시드가 확인된 것은 아니다.', '',
             '| 조건 | 시드 42 Macro-F1 | 평균 Macro-F1 ± SD | 평균 양성 F1 | 평균 AUROC |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        auroc = 'N/A' if row['auroc_mean'] is None else f"{row['auroc_mean']:.4f}"
        lines.append(f"| {names[row['condition']]} | {row['seed42_macro_f1']:.4f} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_sd']:.4f} | {row['positive_f1_mean']:.4f} | {auroc} |")
    lines += ['', f"Train의 다수 클래스를 예측하는 대조군 Macro-F1: {results['majority_control']['macro_f1']:.4f}.",
              f"질문 등장 여부만 사용하는 로지스틱 회귀 대조군 Macro-F1: {results['question_presence_control']['macro_f1']:.4f}.", '',
              'Macro/weighted P/R/F1, G-mean, 민감도·특이도 및 전체 시드 결과는 modality_results.csv와 runs/test_results.json에 있다. G-mean은 sqrt(민감도×특이도)로 정의했으며 원문 계산식과 동일함까지 확인된 것은 아니다.', '',
              '## 사전 지정 비교', '']
    for key, title in [('primary_H_AVT_minus_F_AVT', '주요: 계층 위치 AVT − 고정 위치 AVT'),
                       ('secondary_H_AVT_minus_H_T', '보조: 계층 위치 AVT − 계층 위치 T')]:
        d = results[key]
        lo, hi = d['ci95']
        verdict = '신뢰구간이 0을 포함하여 우위를 확인하지 못함' if lo <= 0 <= hi else '고정된 학습 모델과 이 표본에서는 신뢰구간이 0을 포함하지 않음. 외부 일반화나 원 논문 기법의 검증을 의미하지는 않음'
        lines += [f"- {title}: 시드별 Macro-F1 차이의 평균 {d['delta_macro_f1']:+.4f}, 참가자 단위 paired bootstrap 2,000회의 95% 신뢰구간 [{lo:+.4f}, {hi:+.4f}]. {verdict}."]
    lines += ['', '신뢰구간은 모든 시드·조건에서 같은 참가자를 함께 재표집하고 시드별 F1 차이를 평균해 계산했다. 확률 앙상블 간 차이가 아니며, 학습된 모델을 고정한 상태에서 test 표본의 불확실성만 반영한다.', '',
              '## 원 논문과 비교 가능한 범위', '',
              '- 논문 Macro-F1 0.79와 수치상 비교는 가능하지만, 440 제외, 전사·질문 매핑·위치 표현·반복 응답 집계·결측 처리·특징 버전·시드 등 편차가 있으므로 같은 실험 조건에서의 성능 차이라고 단정하지 않는다.',
              '- 이전 실험의 test 45명에 대한 결과와 공개 사례를 이미 확인했다. 이번 test를 새로운 미노출 평가라고 하지 않으며, test 결과에 따라 설정이나 시드를 선택하지 않았다.',
              '- 결과가 좋거나 나쁘더라도 이 실험만으로 원 논문 계층 메커니즘의 재현성이나 우위를 확증하거나 반박하지 않는다. 실제 화자 구간 및 저자의 중간 산출물과 일치하는지 추가 검증이 필요하다.',
              '- PHQ 자기보고 선별 라벨을 예측한 결과이며 임상 진단 성능이 아니다. 기존 AI 주석은 학습 대상으로 사용하지 않았다.', '',
              '## 산출물과 검증', '',
              '- protocol.json/hierarchy_spec.md: 사전에 고정한 가정. runs/frozen_config.json: 실제 참가자 ID, 소스 및 입력 해시.',
              '- ASR 설정·개별 전사, 질문 매핑·이벤트, 모달리티 캐시, features.npz 및 예측은 Git에서 제외된 Data 디렉터리에 보관한다.',
              '- runs/{condition}_seed{seed}: 100 epochs 학습 기록, 검증 손실이 가장 낮은 가중치, 증강 기록, dev/test 예측.',
              '- evaluation_manifest.json: 15회 학습 검증 후의 test 평가 시각과 결과 해시.',
              '- 독립 검증 기록은 verification.json, 실행명령은 run_train.sh/run_evaluate.sh에 보관한다.']
    lines += ['', '## 추가 ASR 감사', '',
              f"- 질문 후보가 없는 참가자: {asr['zero_question_ids']}. 원본에 질문이 없다는 뜻이 아니라 이번 공개 규칙으로 찾지 못했다는 뜻이다.",
              f"- 영어 외 언어로 자동 감지된 참가자: {asr['non_english_ids']}. 365번은 nn(Norwegian Nynorsk)으로 감지됐다. 기본 자동 감지 경로를 유지했으며 오류 가능성으로 기록했다.",
              f"- 부모 위치가 고정 위치와 달라진 참가자: 입력 적격성 필터 이전 {prep['participants_with_changed_positions']}/{prep['participants']}, 변경 위치 총 {prep['changed_positions_total']}개.",
              '- 이러한 전처리 오류 가능성은 원 논문의 성능 차이에 영향을 줄 수 있지만, 이 실행만으로 원인별 영향 크기를 분리하지는 못한다.']
    (base / 'results_ko.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
