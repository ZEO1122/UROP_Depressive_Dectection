# HiQuE V3: 입력 보강 연구

[파이프라인 노트북](pipeline.ipynb): 단계별 기초 설명, 논문·GitHub 코드 매핑과 실행을 함께 확인한다.

**현재 기준:** 25회 학습과 공통 test45명 평가 완료. E2 test Macro-F1 **0.6541 ± 0.0344**이며 사전 test 노출 이력이 있다. [Dev 결과](dev_results.md), [Test 결과](test_results.md), [실행·과거 소스 보존 안내](../../../docs/reproducibility.md)를 함께 읽는다.

아래 CLI는 `src/urop`를 사용한다. 패키지를 해당 환경에 설치하거나 `PYTHONPATH=src`를 설정한다. Data의 `run_initial.sh`/`run_extend.sh`/검증 스크립트는 과거 소스 경로·hash를 참조하는 역사 기록으로, 직접 실행하지 말고 legacy runner 사용 여부를 먼저 확인한다. 새 실행은 별도 출력 경로를 사용한다.

현재 test 실행 진입점은 `python -m urop test-features`와 `python -m urop evaluate`다. 당시 사용한 `hique3_test_features.py`와 `hique3_test.py`는 과거 소스 스냅샷에 보존했으며, [test 결과](test_results.md)는 별도 문서다.

설계: [보강 계획](../../../archive/plans/hique_strengthening_plan_2026-10-03_ko.md). 이 경로는 고정 위치 F_AVT에서 입력만 바꾸는 비교이며, 원 논문의 계층 구현이나 미노출 test 성능을 검증하는 경로가 아니다.

## 이번 실행의 중요한 수정

학습 전에 기존 명시적 `RobertaTokenizerFast` 로딩이 설치된 Transformers5.9에서 BPE merges를 읽지 못하고 글자 단위로 처리되는 오류를 발견했다. `AutoTokenizer`와 독립적으로 로드한 공식 `tokenizer.json`의 ID 일치를 검사하고, **E0를 포함한 모든 조건의 텍스트 특징을 다시 계산**했다.

E0는 V2의 동일 ASR 구간·질문 매핑·음성·영상을 재사용하지만, 잘못된 텍스트 특징은 재사용하지 않는다. 오류 수정은 `protocol_amendment_tokenizer.json`에 기록했고 모든 V3 학습 전에 적용했다. V1/V2 원본 코드·가중치·수치는 보존하되 해당 공개 결과 문서에는 오류 정정을 덧붙였다.

## 조건과 표본

Train107명, dev32명(공식dev 중440/451/458 제외)을 모든 조건에 동일하게 사용한다. Train/dev 단계에서는 test 원본·라벨을 읽거나 test 예측을 생성하지 않으며, 후속 test 평가는 별도 경로에서 수행했다. 기존 혼합 feature 캐시는 train/dev 부분만 재사용하며, 이전 산출물의 무변경 확인에는 파일 hash만 사용한다.

| 조건 | 입력·응답 선택 | 텍스트 처리 |
|---|---|---|
| E0 | 기존 ASR, 마지막 응답 | 수정된 토크나이저, 처음512토큰 |
| E1 | 제공 전사·고정 매핑, 마지막 응답 | 처음512토큰 |
| E2 | E1의 첫 응답 선택 | 처음512토큰 |
| E3 | E1의 모든 매핑된 응답을 시간순 연결 | 처음512토큰 |
| E4 | E3와 같은 사건·A/V·텍스트 | 겹치지 않는510개 내용토큰 블록, CLS 길이 가중 평균 |

문자열이 짧으면 E3/E4는 비트 단위로 같은 T를 쓴다. 매핑하지 못한 unknown 응답은 원문 사건 표에는 남지만 모델 입력에 포함하지 않는다. 매핑은 공개85개 질문과 train 발화에 대한 두 AI 검토를 사용했으며, 독립 사람 gold가 아니다.

고정 F_AVT,100epochs,batch8,Adam0.0002,dropout0.5,devloss최저 checkpoint, 모달리티별 결측0, 표준화 없음. Train양성은 복사본2개를 추가하고10질문을 같은 seed·조건에서 동기 마스킹한다. 유효 슬롯이 몇 개 지워졌는지는 별도 기록한다.

## 실행 순서

저장소 루트에서 실행한다. 과거 산출물은 `Data/hique_input_study_v3/`에 보존돼 있다. 현재 CLI의 기본 새 작업 경로는 `Data/runs/hique_input_study/`이며, 새 실험에는 별도 `--base`와 출력 경로를 사용한다. 완료 산출물을 설정 변경 후 덮어쓰지 않는다.

1. `python -m urop prepare --stage inventory`: train Ellie 정규형·출처만 수집한다.
2. train에서 `mapping.json`을 검토하고 고정한다. dev 성능으로 매핑을 수정하지 않는다.
3. `python -m urop prepare --stage prepare`: 사건 표, 세 응답 선택, 검토패킷200개를 만든다.
4. 기존 V2 `segments.jsonl`에서 고정139명의 train/dev 기록만 `segments_E0.jsonl`로 선택한다. 질문 매핑·ASR 문장은 변경하지 않는다.
5. A/V와 텍스트 특징을 추출하고 조립·검증한다.
6. 최초15회 검증 후 모든 조건에 두 시드를 추가해 총25회 실행한다.

```sh
python3 -m urop prepare --stage inventory
python3 -m urop prepare --stage prepare --mapping Data/hique_input_study_v3/mapping.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .tmp/hique-features/bin/python -m urop features --stage av --workers 4
.tmp/hique-features/bin/python -m urop features --stage text --device cuda
.tmp/hique-features/bin/python -m urop features --stage assemble
# 당시 실행 기록: .tmp/hique-features/bin/python Data/hique_input_study_v3/verify_study.py --stage inputs
# 당시 실행 기록: bash Data/hique_input_study_v3/run_initial.sh
# 당시 실행 기록: .tmp/hique-features/bin/python Data/hique_input_study_v3/verify_study.py --stage initial
# 당시 실행 기록: bash Data/hique_input_study_v3/run_extend.sh
# 당시 실행 기록: .tmp/hique-features/bin/python Data/hique_input_study_v3/verify_study.py --stage extend
python3 -m urop train --help  # initial/extend 인수 및 새 output 경로 확인
python3 -m urop report
```

GPU는 RTX4070TiSUPER16GB를 사용한다. 현재 호스트의 sandbox 내부에서는 GPU가 보이지 않아 GPU 접근이 허용된 실행 경로가 필요하다. `.tmp/hique-features`의 로컬 RoBERTa를 사용하며, 작은 classifier는 `.tmp/hique-tf`의 TensorFlow CPU2.15.1로 학습한다. 외부 LLM API 호출은 없다.

`run_initial.sh`/`run_extend.sh`에는 실제 고정한 provenance 파일을 모두 포함한다. 최초 stage의 조건·소스·입력·가중치·예측이 바뀌면 확장을 거부한다. 모델을 구성하는 옛 adapter의 manifest 쓰기 부작용은 해당 경로에 한해서 차단한다.

## 검증과 해석

- 화자·원문 행·질문·답변·시간 구간을 전수 추적한다. Unknown은 경계를 끊고 확인된 맞장구는 이어간다.
- E0의 A/V는 기존 train/dev 배열과 동일한지 확인한다. T는 공식 토크나이저로 재계산한다.
- 모든 실험 문자열의 토큰 ID를 독립 `tokenizer.json`으로 비교한다. BPE merges와 단어 사이 공백 처리도 검사한다.
- E3/E4의 A/V·마스크는 같고, 짧은 T도 같다. 긴 응답의 모든 내용 토큰은 정확히 한 블록에 들어간다.
- 같은 seed의 증강 인덱스는 모든 조건에서 같다. 100epoch 완료·최저devloss 선택·예측에서 재계산한 지표·hash를 확인한다.
- 주 비교 E1−E0는 화자·시간·텍스트·매핑이 함께 바뀐 입력 경로의 효과다. 화자 인식만의 효과로 해석하지 않는다.
- E2−E1, E3−E1, E4−E3을 모두 보고한다. Bootstrap 구간은 checkpoint 선택에 사용한 같은 dev를 재표집한 탐색적 결과다.

텍스트 검증의 NPZ 반복 압축 해제 문제를 수정한 기록과, 잘못된 토크나이저로 생성된 학습 전 캐시는 `code_snapshots/`에 보존했다. 전자는 배열이 완전히 같은 것을 확인한 성능 수정이고, 후자는 실제 입력을 바꾸는 오류 수정으로 구분한다.

주요 결과: `results_ko.md`, `results.csv`, `runs/extend_results.json`. 참가자별 예측·원문 검토패킷·가중치는 Git에 포함하지 않는다.

## 저장된 모델의 test 평가

사용자 요청으로 공통45명(공식test47명 중411/480 제외,300 포함)을 별도 평가했다. 매핑·전처리·모델·threshold를 고정했고, 새 학습이나 대조군 재적합은 하지 않았다. E2는 결과를 읽기 전에 dev 기준 참고 조건으로 기록했다.

`Data/hique_input_study_v3/test_evaluation/`에 test프로토콜,별도특징,동결평가설정,25개 예측 및 집계 결과를 보관한다. `features/training_text_numerical_parity.json`에는15개 학습 특징과 test 전용 추출 경로의 수치 일치 검사가 있다. 실제 평가 명령은 같은 폴더의 `run_evaluate.sh`에 모든 provenance 인수와 함께 저장했다. 완료한 평가는 기본적으로 덮어쓰기를 거부한다.

```sh
# 당시 실행 기록: python3 Data/hique_input_study_v3/test_evaluation/verify_test.py
python3 -m urop test-features --help
python3 -m urop evaluate --help
python3 -m urop test-report
```

기존 V1/V2 test 노출 이력은 유지한다. 이 결과를 새로운 미노출 평가나 원 논문과 같은189명 조건의 정확 재현으로 표현하지 않는다.
