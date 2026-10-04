# AI 잠정 주석을 이용한 A 탐색 실험

공식 train의 20명·199 QA에 대해 참가자 1명씩 제외하는 교차검증을 실행했다. 각 참가자의 예측은 그 사람의 자료를 학습하지 않은 모델에서만 얻었다.

**아래는 AI 초안과의 일치도이며 임상 정확도·독립 전문가 평가가 아니다.** 공식 dev/test는 이번 실행에 사용하지 않았다. 모든 특징은 원문에서만 만들고 주석 설명·근거 span·환자 ID는 입력하지 않았다.

## 수면 문제

Macro-F1은 각 목표의 실제 등장 라벨 평균이다. 없는 class의 능력은 평가할 수 없다. 목표별 represented_labels·untested_labels 및 고정 3-class F1은 report.json에 별도 기록했다.

| 입력·목표 | 항상 근거 없음 | Q+A 텍스트 | 관계 특징 추가 | 지지 recall(텍스트/관계) | 부인 recall(텍스트/관계) |
|---|---:|---:|---:|---|---|
| 해당 QA / 보고 근거 상태 | 0.3119 | 0.6432 | 0.6500 | 0.6842 / 0.6842 | 0.2000 / 0.2000 |
| 3-QA / 보고 근거 상태 | 0.3042 | 0.5284 | 0.4598 | 0.6800 / 0.2800 | 0.0000 / 0.1429 |
| 3-QA / 현재 본인 상태 | 0.3052 | 0.5187 | 0.4684 | 0.6250 / 0.2917 | 0.0000 / 0.1429 |

같은 행 안에서만 모델을 비교한다. 해당 QA와 3-QA의 정답이 달라 행 사이 F1 차이를 맥락 효과로 해석하지 않는다.

| 목표 | 관계 특징−텍스트 Macro-F1 | 95% 기술적 CI |
|---|---:|---|
| anchor_report | +0.0068 | [-0.0173, +0.0302] |
| window_report | -0.0686 | [-0.1562, +0.0191] |
| window_current_self | -0.0503 | [-0.1367, +0.0464] |

CI는 OOF 예측을 고정한 참가자 bootstrap 2,000회의 기술적 구간이다. 모델 재학습을 포함한 전체 불확실성이나 임상적 유의성을 보장하지 않는다.

## 흥미 저하

| 목표 | 지지 QA / 지지 참가자 | Q+A 지지 recall | 관계 특징 지지 recall | 단일 class 학습으로 대체한 fold |
|---|---|---:|---:|---:|
| anchor_report | 1 / 1 | 0.0000 | 0.0000 | 1 |
| window_report | 2 / 1 | 0.0000 | 0.0000 | 1 |
| window_current_self | 2 / 1 | 0.0000 | 0.0000 | 1 |

지지 사례가 특정 참가자에게 집중되면 그 사람을 제외한 fold에서 해당 class를 학습할 수 없다. 학습 class가 하나뿐일 때만 다수 class로 대체한다. 위 표의 실제 분포와 대체 fold 수를 확인하고, 높은 전체 정답률을 흥미 저하 탐지 성공으로 해석하지 않는다.

## 문맥을 추가했을 때 기대되는 판단 변화

| 영역 / 모델 | 라벨이 달라진 쌍 수 | 변경쌍의 양쪽 모두 정답 | 유지쌍의 양쪽 모두 정답 |
|---|---:|---:|---:|
| sleep / qa_text | 8 | 0.3750 | 0.8953 |
| sleep / qa_relation | 8 | 0.1250 | 0.8586 |
| interest / qa_text | 1 | 0.0000 | 0.9949 |
| interest / qa_relation | 1 | 0.0000 | 0.9949 |

## B·C의 현재 준비 상태

- B: 기존 B 유효 특징 후보와 새 주석의 교집합은 4명·10 QA다. 전체 새 주석에 대해 특징을 다시 추출한 것이 아니다. 라벨 분포: {'sleep': {'support': 3, 'no_evidence': 7}, 'interest': {'no_evidence': 10}}.
- C: 이 참가자들의 전체 면담에는 1236 QA가 있고 주석 anchor는 199개다. 제공된 주변 문맥이 있더라도 전체 면담 gold의 완전성을 인증할 수 없으므로 coverage 본평가는 수행하지 않았다.

## 범위와 다음 단계

- AI investigators also designed the methods; agreement is not independent clinical accuracy.
- Small pilot sample; repeated/overlapping windows can share evidence within a participant.
- Report class-participant counts and missing/rare classes; high accuracy need not imply positive-case detection.
- Anchor and window targets are different; direct F1 subtraction across scopes is not a context effect.
- Do not compare with earlier weak-label train/dev F1: labels, participants, and validation protocol differ.
- No answer-only gold or independently validated question mutations: neither is evaluated here.
- B needs a new reference-window feasibility protocol; C needs full-interview annotations. Neither is relabeled as complete by this run.

B의 참조 구간 표본 부족과 C의 전체 면담 gold 부족은 이번 주석 일부를 연결하는 것으로 해결되지 않는다. 본 실행은 A의 자연 QA 상태 판별 단계만 수행했으며 질문 변형의 기대 반응은 별도 검증 전이다.

## 재현

```sh
PYTHONPATH=src python -m urop legacy run experiments.run_a_annotated -- --annotation-dir Data/experiments/2026-09-27/ai_annotation_draft --zip-dir Data/DAIC-WOZ --output-dir Data/experiments/2026-09-27/ai_annotated_a
```

설정·코드/주석 hash는 run_config.json, 전체 지표는 report.json, 참가자별 예측·분할은 private_*.jsonl에 있다.
