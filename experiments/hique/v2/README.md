# 440 제외 HiQuE 실행 경로

실행 명령과 소스 경로는 [구조 정리 후 재현 안내](../../../docs/reproducibility.md)를 따른다. 아래 Data의 실행 스크립트와 옛 테스트 명령은 당시 기록으로 보존하며 직접 재실행 전에 legacy 경로를 사용한다.

> **상태: 토크나이저 오류가 확인된 과거 실험.** 아래 수치는 당시 실행 기록이며 정상 RoBERTa 입력의 재현 성능으로 인용하지 않는다. 수정 후 현재 기준은 [V3 실험](../v3_input_study/README.md)이다.

후속 분석: [2026-10-03 성능 격차 진단](../../../docs/hique_diagnosis.md). 기존 실행 산출물은 보존했으며, 입력 감사와 train/dev 정규화 대조를 별도 폴더에서 수행했다.

계획: [실험 설계](../../../archive/plans/hique_experiment_design_excluding440_ko.md).
이 경로는 공개 코드의 전사 규칙과 논문의 특징에, 명시적으로 정의한 부모 위치 ID 연산을 결합한다. 저자의 비공개 hierarchy 중간 자료가 없으므로 원 논문과 동일한 구현으로 인증하지 않는다.

## 구성

- `hique2_asr.py`: 공식 Whisper base, GPU, 물음표 기반 의사 화자 분류와 연속 발화 병합. 모든 원본 ZIP은 읽기만 한다.
- `hique2_audit.py`: 사전에 고정한 첫 10개 train ID에서 제공 전사와 추정 화자의 시간 겹침을 검사한다. 이는 화자 인식 정확도나 임상 정답이 아니다.
- `hique2_data.py`: 공식 BERTScore 영어 기본 설정에 따른 질문 대응, 마지막 응답 보존, 부모 관계와 특징 결합.
- `hique2_av.py`: 독립적인 모달리티별 결측 처리, eGeMAPSv02와 CLNF 평균·모분산.
- `hique2_network.py`: 같은 크기의 H/F 네트워크에서 입력 position IDs만 변경한다. H_T는 동일 text backbone을 사용한다.
- `hique2_train.py`: 세 조건 × 다섯 시드, 100 epochs, 증강 기록과 입력 hash 동결, 학습 완료 후 test 평가.
- `hique2_report.py`: 집계 결과와 한계 보고서.

H_AVT와 F_AVT는 동일 A/V/T 특징·표본·초기 가중치·증강·학습 예산을 사용한다. H는 후속 질문의 최근 주질문 topic ID를 위치 조회에 쓰고, F는 고정 질문 ID를 쓴다. 반복 질문은 마지막 응답과 그 응답의 부모 관계를 함께 보존한다. 관측되지 않은 슬롯은 고정 위치 ID를 유지한다. 증강은 특징만 0으로 만들고 위치 ID는 유지하므로 완전한 attention masking은 아니다.

## 환경과 실행

GPU: RTX 4070 Ti SUPER 16GB. 이 환경에서는 sandbox 내부에서 장치가 보이지 않아 GPU 작업은 GPU 접근이 허용된 실행 경로에서 수행한다.

- `.tmp/hique-asr`: 공식 openai-whisper, 기존 PyTorch CUDA 사용.
- `.tmp/hique-features`: openSMILE 2.6.0, bert-score 0.3.13, PyTorch/Transformers. BERTScore와 텍스트 특징은 GPU.
- `.tmp/hique-tf`: TensorFlow CPU 2.15.1. 작은 공개 네트워크 학습용. CPU 실행은 GPU가 없는 시스템이라는 의미가 아니다.

모든 participant 데이터·모델 가중치·중간 산출물은 Git 제외되는 `Data/`와 `.tmp/`에 둔다. 외부 LLM API는 호출하지 않는다. 공개 가중치 다운로드 후 모델은 로컬에서 실행한다.

저장소 루트에서 단계별 실행한다. 기존 결과를 재실행으로 덮어쓰지 않으며, 변경 실험은 별도 출력 경로로 관리한다.

```sh
PYTHONPATH=src python -m urop legacy run --python .tmp/hique-asr/bin/python experiments.hique2_asr --
PYTHONPATH=src python -m urop legacy run experiments.hique2_audit --
MPLCONFIGDIR=/tmp/hique-mpl PYTHONPATH=src python -m urop legacy run --python .tmp/hique-features/bin/python experiments.hique2_data -- --stage prepare --device cuda
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m urop legacy run --python .tmp/hique-features/bin/python experiments.hique2_av -- --segments Data/hique_reproduction_excluding440/segments.jsonl --zip-dir Data/DAIC-WOZ --output-dir Data/hique_reproduction_excluding440/av --workers 4
PYTHONPATH=src python -m urop legacy run --python .tmp/hique-features/bin/python experiments.hique2_data -- --stage text --device cuda
PYTHONPATH=src python -m urop legacy run --python .tmp/hique-features/bin/python experiments.hique2_data -- --stage assemble
```

학습과 평가의 실제 명령은 추가 provenance 인수까지 포함해 `Data/hique_reproduction_excluding440/run_train.sh`, `run_evaluate.sh`에 기록한다. 동일 provenance 목록을 두 단계에 사용해야 한다. 설정·소스·입력 또는 완료 가중치가 달라지면 평가를 거부한다.

```sh
# 당시 실행: bash Data/hique_reproduction_excluding440/run_train.sh
# 당시 실행: bash Data/hique_reproduction_excluding440/run_evaluate.sh
PYTHONPATH=src python -m urop legacy run experiments.hique2_report --
```

## 검증과 해석

원래 목표는 440만 제외한 188명(107/34/47)이다. 실제 사용 인원은 `eligibility.json`과 `runs/frozen_config.json`에서 확인한다. 입력 생성에 성공했다는 사실은 실제 Ellie 질문을 복원했다는 뜻이 아니다. 공개 전사의 화자 정보와 비교한 `speaker_audit.json`을 결과와 함께 읽어야 한다.

주요 비교는 시드별 Macro-F1 차이의 평균 H_AVT−F_AVT다. 참가자 paired bootstrap 2,000회에서 동일 참가자 인덱스를 모든 시드·조건에 적용한다. 확률을 먼저 평균한 ensemble 비교와 구분한다. G-mean은 sqrt(민감도×특이도)로 정의한다.

기존 test45 결과와 공개 사례를 이미 확인했으므로 새 미노출 test라고 주장하지 않는다. 현재 실행은 기존 AI 잠정 주석을 정답으로 사용하지 않는다. 예측 대상은 제공 PHQ 선별 라벨이며 임상 진단 성능이 아니다.

당시 검증 명령 기록(이 경로들은 스냅샷 복원 환경 기준이다. 현재 검증 진입점은 `python -m urop legacy verify`):

```text
python3 -m pytest -q
.tmp/hique-tf/bin/python -m pytest tests/test_hique2_network.py tests/test_hique2_train.py -q
.tmp/hique-features/bin/python -m pytest tests/test_hique2_av.py -q
python3 -m ruff check experiments tests --select E9,F63,F7,F82
python3 -m mypy experiments --ignore-missing-imports
```

원 가중치·epoch 기록·증강·참가자별 예측은 `runs/`에, 집계 결과는 `results_ko.md`와 `modality_results.csv`에 보관한다. 이전 `Data/hique_reproduction/` 실행 결과와 구분한다.
