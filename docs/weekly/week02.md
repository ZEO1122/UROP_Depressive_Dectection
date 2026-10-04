# Week2 이후 연구 진행 기록

현재 진입점은 [실험 목록](../../experiments/README.md)이다. 아래는 주차 폴더에 누적됐던 진행 기록을 보존한 것으로 각 항목의 “새 학습 전”, “test 재평가 없음”은 해당 시점의 상태다.

- [V3 고정 모델 test 평가](../../experiments/hique/v3_input_study/test_results.md) — 같은45명·25개 가중치, 재학습 없음

- [V3 입력 보강25회 실행 결과](../../experiments/hique/v3_input_study/dev_results.md) — 토크나이저 오류 수정, 동일train107/dev32, test재평가 없음

- [HiQuE 보강 실험 계획](../../archive/plans/hique_strengthening_plan_2026-10-03_ko.md) — 제공 전사·반복 응답·길이 처리의 단계별 비교, 새 학습 전

- [HiQuE 성능 격차 원인 진단](../hique_diagnosis.md) — 논문·전체 공개 코드 대조, train107명 입력 감사, 정규화 단일변수 진단

- [440 제외 후속 재구현 결과](../../experiments/hique/v2/results.md) — 입력 적격 186명, 15회 학습·test 평가 완료; 동일 조건 재현 아님

- [440 제외 HiQuE 재현 실험 설계](../../archive/plans/hique_experiment_design_excluding440_ko.md) — 목표188명, 계층 입력 검증 후 본학습

- [현재 연구 방향: Assessment·DSM·LLM](../research_direction.md)
- [선행연구 목록과 조사 기록](../literature/README.md)
- [AI 잠정 주석을 연결한 A 교차검증](../../experiments/ai_annotation_pilot/results.md)
- [HiQuE 공개 코드 기반 수정 재현 결과](../../experiments/hique/v1/results.md) — PHQ 라벨, 35회 학습·test 45명 평가 완료
- [HiQuE 실행 방법과 원 논문 대비 변경점](../../experiments/hique/v1/README.md)

우울 증상의 근거와 평가 정보 부족을 탐지하는 LLM 연구를 검토합니다. 초기 선별 라벨 예측 중심 제안에서 Assessment 지원으로 목표를 수정했습니다.

논문 본문, 검색 응답 원본, 폴더 통합 검증 기록은 로컬 보관 자료이며 GitHub에는 포함하지 않습니다. 2026-09-27에는 [세 후보](../../archive/plans/three_experiment_candidates_2026-09-27_ko.md)의 API 없는 기술 파일럿을 실행했습니다. [결과와 한계](../../experiments/evidence_pilot/results.md)를 확인하세요. 독립 사람 주석 기반 본평가와 임상 검증은 아직 수행하지 않았습니다.
