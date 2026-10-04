# 세 실험의 로컬 탐색 실행

실행 명령과 소스 경로는 [구조 정리 후 재현 안내](../../docs/reproducibility.md)를 따른다. 아래 Data의 실행 스크립트와 옛 테스트 명령은 당시 기록으로 보존하며 직접 재실행 전에 legacy 경로를 사용한다.

2026-10-03 [V3 입력 보강 비교 실험](../hique/v3_input_study/README.md)은 동일train/dev에서25회 실행했다. [결과](../hique/v3_input_study/dev_results.md)에 기존 토크나이저 로딩 오류의 수정과 dev평가 한계를 명시했다.

2026-09-29 [440 제외 후속 재구현](../hique/v2/README.md)은 별도 구현과 출력 경로를 사용한다. [15회 실행 결과](../hique/v2/results.md)에서 실제 cohort와 공개 전처리의 한계를 확인할 수 있다.

후속으로 실행한 **PHQ 라벨 기반 HiQuE 공개 코드 수정 재현**은 [별도 실행 문서](../hique/v1/README.md)와 [결과](../hique/v1/results.md)를 참고한다. 아래 세 실험의 대리 라벨/AI 주석 평가와 구분한다.

2026-09-27. 사용자 선택 범위는 **탐색 실행과 독립 주석 준비까지**다. 독립 평가자·임상 gold가 없으므로 주평가 성능은 아직 산출하지 않는다.

후속으로 사용자가 승인한 **AI 잠정 주석 기반 A 탐색**은 아래 별도 명령을 사용한다. 원래 weak-label 실행 결과를 덮어쓰지 않는다.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m urop legacy run experiments.run_a_annotated --
```

`ai_annotation_draft/ai_annotations.jsonl`과 검증된 출처 hash를 확인하고 공식 train의 주석된 참가자만 1명씩 제외해 평가한다. anchor 보고 상태, 3-QA 보고 상태, 3-QA 현재 본인 상태를 각각 맞는 입력으로 비교한다. 특징 추출기는 각 fold의 학습 참가자에만 fit하며 주석 설명·근거 위치·ID를 특징으로 사용하지 않는다. 결과는 `Data/experiments/2026-09-27/ai_annotated_a/`에 저장한다. [결과와 한계](../ai_annotation_pilot/results.md)를 참고한다. 독립 사람/전문가 gold용 importer나 본평가 runner와는 다르다.

- A: 질문-only / 응답-only / Q+A / 관계 특징 / 규칙 / 다수 class 비교와 합성 질문 변형 점검.
- B: 질문·길이·위치 / raw A/V / 개인 참조 차이 / raw+baseline 비교 및 참조 구간 확보율.
- C: 시간 순서 / 무작위 / BM25 / MMR / 키워드 항목 확보량÷비용 선택과 분량 제한 검증.

## 재현

저장소 루트에서:

```sh
PYTHONPATH=src python -m urop legacy run experiments.run_suite -- \
  --zip-dir Data/DAIC-WOZ \
  --output-dir Data/experiments/2026-09-27 \
  --workers 4
```

사용한 기존 환경: Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, scipy 1.15.3. 실행은 설치·다운로드·API 호출을 하지 않는다. 별도 환경에는 해당 라이브러리가 필요하다.

실행 순서: `prepare.py` → `run_a.py` → `run_b.py`(특징 cache 재사용 가능) → `run_c.py` → `report.py`.

출력 디렉터리는 이 저장소의 `Data/` 아래만 허용한다. QA·특징·참가자별 예측·주석 원문을 공개 경로로 쓰지 않는다. ZIP은 정확한 이름의 member만 읽고 원본 및 중첩 ZIP을 수정/압축 해제하지 않는다.

## 데이터와 평가 계약

- 현재 감사 시점의 440 손상, 451·458·480 질문 전사 누락, 300 사전 노출을 명시적으로 처리한다. 데이터가 보완되면 자동으로 cohort를 바꾸지 말고 새 manifest와 분석계획을 기록한다.
- **Train 107 / Dev 32명**만 읽는다. 공식 test의 ID는 분할 봉인 확인에만 쓰며 전사·음성·시각은 열지 않는다. 세 실행 모듈은 test 행이 들어오면 오류를 낸다.
- weak label `unknown`은 자동 규칙 미적중/범위 모호함도 포함한다. 임상 정보 부족의 정답이 아니다.
- A의 저자 구성 합성 질문은 인간 검증 전이며, 같은 저자가 작성한 규칙의 정답률은 구성상 sanity check다.
- B는 참조 구간 30초와 모달리티별 유효 시간 비율 80%를 사용한다. overlap/scrubbed QA 제외, VUV=1의 F0만 사용, 시각 success/confidence/sentinel과 실제 timestamp를 확인한다. 기준선·원본·차이 모델을 같은 cohort에서 비교한다. 학습 class 누락 또는 극소 dev에서는 추론용 CI를 생략한다.
- C의 8종 키워드 slot은 임상적인 fact나 충분성 gold가 아니다. 특히 slot 선택기와 evaluator가 같은 검출기를 사용하므로 높은 점수는 순환적일 수 있다. 사람 주석 전에는 효용·우월성·정확도 주장을 하지 않는다.
- 저차원 LR와 규칙의 탐색 실행은 원래 문헌의 정확한 재현이나 생성형 LLM 평가가 아니다.

## 결과와 주석 파일

`Data/experiments/2026-09-27/` 아래:

| 파일 | 용도 |
|---|---|
| `results_summary_ko.md` | 자동 집계 결과와 해석 한계 |
| `execution.json` | 실행 명령·종료 상태·코드 hash |
| `preparation_summary.json`, `manifest.json` | 환경 버전·QA hash·참가자별 포함/제외 |
| `qa.jsonl` | train/dev 원문 QA 및 자동 대리 라벨, **평가자에게 제공하지 않음** |
| `results_a/a_report.json` | A 대리 일치도·합성 점검 |
| `results_b/summary.json` | B 표본 확보율·진단용 출력 |
| `results_c/c_summary.json` | C 분량·키워드 대리 coverage |
| `pilot_reviewer_1.csv`, `pilot_reviewer_2.csv` | 서로 다른 순서의 **빈 인간 주석 파일**; train 20명 199 QA |
| `annotation_readme.md` | 주석 범위 및 비공개 자료 취급 |
| `results_a/a_review.csv` | A dev QA의 빈 추가 검토 양식 |
| `results_c/c_human_review_blank_private.csv` | C 전체 면담 근거 주석을 위한 빈 양식 |

평가자에게는 필요한 원문과 해당 빈 주석 파일만 제공한다. 자동 라벨·모델 결과·PHQ를 같이 보여주지 않는다. C의 전체 근거 확보율은 199개 anchor 표집만으로 검증할 수 없고, 정한 면담 cohort의 전체 근거 주석이 필요하다.

재실행 시 작성된 pilot 주석은 보존하며 A/C 검토 양식도 기존 파일을 덮어쓰지 않는다. 새 데이터/지침 버전에는 새 output directory를 사용해야 한다. 현재는 human-gold importer와 본평가 실행기를 제공하지 않으며, 독립 주석 schema 및 불일치 조정 절차가 확정되면 별도로 연결한다.

## 당시 검사 기록

아래는 정리 이전 경로에서 실행한 명령이다. 현재 과거 산출물 확인은 `python -m urop legacy verify`와 [재현 안내](../../docs/reproducibility.md)를 따른다.

```text
python3 -m pytest -q
python3 -m ruff check experiments tests --select E9,F63,F7,F82,F401,F841
python3 -m mypy experiments --ignore-missing-imports --follow-imports=skip
python3 -m compileall -q experiments tests
```

테스트는 가상 자료를 사용한다. 참가자 분할, test 봉인, 주석 보존, 시간 마스크·조기 종료, 예산 준수, 대리 지표의 빈 분모 등을 확인한다. 모든 함수에 엄격한 타입을 붙인 프로젝트는 아니므로 mypy 결과를 완전한 타입 증명으로 해석하지 않는다.

설계: [후보 세 개](../../archive/plans/three_experiment_candidates_2026-09-27_ko.md). 결과: [실행 요약](results.md).
