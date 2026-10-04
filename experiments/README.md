# 실험 목록

현재 기준은 **[HiQuE V3 입력 보강 연구](hique/v3_input_study/README.md)**다. 제공 PHQ 선별 라벨을 예측한 연구이며 임상 진단 성능이 아니다. 논문의 정확 재현을 달성했다는 주장도 하지 않는다.

| 실험 | 목적·평가 범위 | 상태 | 결과·한계 |
|---|---|---|---|
| [Evidence pilot](evidence_pilot/README.md) | 질문 맥락·개인 기준선·근거 선택, train107/dev32 | 탐색 완료 | [결과](evidence_pilot/results.md). 자동 대리 라벨/키워드 지표이며 독립 정답 없음 |
| [AI annotation pilot](ai_annotation_pilot/README.md) | 공식 train20명·199 QA, 참가자 제외 교차검증 | 탐색 완료 | [결과](ai_annotation_pilot/results.md). AI 초안과의 일치도이며 사람/임상 gold 아님 |
| [HiQuE V1](hique/v1/README.md) | 7모달리티 조합×5시드,107/32/45명 | **토크나이저 오류 확인·과거 기록** | [결과](hique/v1/results.md). 계층 누락·자체 전처리·잘못된 T 입력 |
| [HiQuE V2](hique/v2/README.md) | H_AVT/F_AVT/H_T×5시드,107/33/46명 | **토크나이저 오류 확인·과거 기록** | [결과](hique/v2/results.md). ASR 화자 추정·계층 가정·잘못된 T 입력 |
| [HiQuE V3](hique/v3_input_study/README.md) | E0–E4×5시드,107/32/45명 | **현재 기준·평가 완료** | [Dev](hique/v3_input_study/dev_results.md), [Test](hique/v3_input_study/test_results.md). 공식 tokenizer 검증, test 사전 노출 이력 |

V3에서 dev로 사전 선택한 E2의 test Macro-F1은 **0.6541 ± 0.0344**(5시드, SD ddof=0)다. E1−E0의 평균 상승은 +0.0570이나 참가자 재표집 95% 구간은 [-0.0642,+0.1817]로 0을 포함한다. 최고 시드나 test에서 새로 선택한 조건으로 대표 성능을 바꾸지 않는다.

각 폴더의 `config.json`은 **사람과 도구가 읽는 실험 등록 정보**다. 실행기가 자동 로드하는 설정 파일이 아니며, 실제 동결 설정·hash·참가자 목록은 비공개 `Data/` 실행 기록을 우선한다.

새 코드의 진입점은 `python -m urop`이다. 과거 코드는 [정확한 소스 스냅샷](../archive/snapshots/pre-reorg-2026-10-04/)과 legacy runner로 보존한다. [실행·재현 안내](../docs/reproducibility.md), [데이터 안내](../docs/dataset.md), [격차 진단](../docs/hique_diagnosis.md)을 함께 읽는다.
