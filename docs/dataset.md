# 데이터와 비공개 산출물

원본은 `Data/DAIC-WOZ/`의 DAIC-WOZ ZIP과 split/PHQ CSV다. 자료 접근·이용 조건은 [공식 배포처](https://dcapswoz.ict.usc.edu/)를 따른다. 원자료와 참가자별 전사·특징·예측·가중치는 공개 문서로 이동하지 않았다.

- 정상 ZIP 188개, 440은 압축 스트림이 중간에 끊겨 제외했다. 단순한 ZIP 인덱스 손상으로 취급하지 않는다. [다운로드 재검사](data/download_recheck_2026-09-27.md)와 로컬 `Data/audits/440_zip_stream_check_2026-09-28.json`을 참고한다.
- 공식 split은 train107/dev35/test47이다. 440은 dev 소속이다. 실제 사용 인원은 실험마다 다르므로 [실험 목록](../experiments/README.md)과 해당 manifest를 확인한다.
- 451·458·480에는 면담자 전사가 없고 402 영상은 대화보다 일찍 끝난다. 없는 시간 구간을 임의로 연장하지 않는다.
- 409의 PHQ 점수/이진 라벨 불일치는 원 제공 이진 라벨을 유지했다. `full_test_split.csv`는 사용자 제공본이며 원 배포본과의 별도 외부 인증을 의미하지 않는다.
- 300과 기존 test 결과를 검토한 이력이 있다. 후속 test는 새로운 미노출 평가라고 주장하지 않는다.

| 로컬 경로 | 역할 |
|---|---|
| `Data/DAIC-WOZ/` | 원본 ZIP·split·라벨 |
| `Data/experiments/2026-09-27/` | Evidence/AI 주석 파일럿 |
| `Data/hique_reproduction/` | V1 산출물·공개 upstream 고정 소스 |
| `Data/hique_reproduction_excluding440/` | V2 ASR·특징·15개 모델 |
| `Data/hique_diagnosis_2026-10-03/` | 입력 감사·진단 자료 |
| `Data/hique_input_study_v3/` | V3 입력·매핑·25개 모델·dev 결과 |
| `Data/hique_input_study_v3/test_evaluation/` | V3 고정 모델의 test 입력·평가 |
| `.tmp/` | 가상환경·다운로드 모델 캐시 |
| `output/` | PDF·발표자료 등 로컬 생성물 |

현재 정리는 공개 코드·문서의 구조 정리다. 이 데이터 경로와 동결 메타데이터는 그대로 유지하며, 새로운 실험도 기존 결과를 덮어쓰지 않고 별도 출력·프로토콜을 사용한다.

이전 [초기 검사](data/download_audit_2026-09-27.md)는 파일 교체 이전 상태다. 현재 보유 상태와 혼동하지 않는다.
