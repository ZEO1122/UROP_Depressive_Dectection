# HiQuE 공개 코드 기반 수정 재현

실행 명령과 소스 경로는 [구조 정리 후 재현 안내](../../../docs/reproducibility.md)를 따른다. 아래 Data의 실행 스크립트와 옛 테스트 명령은 당시 기록으로 보존하며 직접 재실행 전에 legacy 경로를 사용한다.

> **상태: 토크나이저 오류가 확인된 과거 실험.** 아래 수치는 당시 실행 기록이며 정상 RoBERTa 입력의 재현 성능으로 인용하지 않는다. 수정 후 현재 기준은 [V3 실험](../v3_input_study/README.md)이다.

논문: [HiQuE, CIKM 2024](https://arxiv.org/abs/2408.03648). 공개 구현: [JuHo-Jung/HiQuE](https://github.com/JuHo-Jung/HiQuE), commit `24c553bf2666b442ae5b0e3490b998a5d4493559`.

이 실행은 **adapted reproduction**이다. 공개 코드의 실행 오류를 고치고 논문에 서술된 특징을 구현했다. 원 논문의 동일 전처리·동일 189명·동일 성능을 재현했다는 뜻은 아니다. 기존 AI 주석 실험과 별개로 제공된 PHQ 이진 라벨을 사용한다. PHQ 선별 라벨 예측을 임상 진단으로 해석하지 않는다.

## 데이터와 고정 조건

- 원본은 `Data/DAIC-WOZ/{id}_P.zip`이며 ZIP을 변경하지 않는다. 중첩 ZIP 대신 정확한 최상위 member를 읽는다.
- 440 손상, 451/458/480 interviewer 전사 부재, 300 이전 사례 검토 노출을 제외해 train 107 / dev 32 / test 45명이다. 402는 실제 영상 제공 구간만 사용한다.
- 논문 Appendix A의 85개 질문을 슬롯으로 사용한다. 정규화 exact match, 이어서 frozen RoBERTa token-max F1 ≥ 0.75로 질문을 대응시킨다. 기본 BERTScore와 동일하다고 주장하지 않는다.
- 참가자 응답 구간에서 interviewer와 겹치는 시간 및 비식별화된 발화를 제외한다. 같은 질문의 여러 응답은 합친다. 제공 수동 전사를 사용하며 ASR을 실행하지 않는다.
- T: frozen RoBERTa-base 마지막 layer CLS 768차원, 최대 512 tokens. A: openSMILE 2.6.0 eGeMAPSv02 88차원. V: 제공 CLNF 68개 얼굴 좌표 x/y를 1fps로 표본화한 평균·모분산 272차원.
- 오디오: ≥ 0.5초, coverage ≥ 0.8. 영상: success 및 confidence ≥ 0.8, coverage ≥ 0.5, 최소 2프레임. 모든 조건에 같은 A/V/T 공통 슬롯 mask를 쓴다.
- 모든 참가자에게 공통 슬롯이 있어 추가 제외는 없었다. 공통 슬롯 수는 train 3,628 / dev 1,085 / test 1,545이다.
- 특징별 평균·표준편차는 train의 관측 슬롯에서만 계산하고, 결측은 변환 뒤 0으로 복원한다. 모델은 원 코드처럼 별도 attention mask를 사용하지 않는다.
- A, V, T, AV, AT, VT, AVT × seed 13/23/37/42/79. 각 100 epochs, batch 8, Adam 0.0002, dropout 0.5. Dev cross-entropy가 최소인 checkpoint, threshold 0.5.
- Train 양성에만 원본 외 복사본 2개를 추가한다. 각 복사본은 85개 중 같은 10개 질문을 모든 모달리티에서 가린다.
- 주요 비교는 **seed별 확률을 평균한 AVT와 T의 Macro-F1 차이**와 참가자 paired bootstrap 2,000회 CI다. seed별 F1의 평균±SD도 별도로 보고한다. 질문 등장 여부만 입력하는 logistic regression(C=1) 대조군을 포함한다.
- 공식 binary를 유지한다. 409의 score와 binary 불일치를 수정하지 않는다. Test outcome은 사용자 제공 `full_test_split.csv`에서 읽으며 외부 원본 여부까지 인증한 것은 아니다.

## 공개 코드와의 차이

**2026-09-28 추가 확인:** 현재 `build_slots`는 고정 질문 ID별로 응답을 합치고, `PositionEmbedding`은 0~84 위치를 사용한다. 논문 §4.3에 서술된 후속 질문과 선행 주질문의 parent topic 관계를 별도로 입력하지 않는다. 따라서 논문의 핵심 hierarchical question embedding을 충실히 재현했다고 볼 수 없다. 이는 실행 오류 수정과 별개의 방법론 차이이며, 성능에 미친 크기는 아직 분리 실험으로 검증하지 않았다.

`hique_network.py`는 upstream `fusionmodel.py`의 SHA256을 검사한 뒤 필요한 정의만 AST로 읽는다. 원본 파일 자체는 수정하지 않는다.

1. `TransformerBlock`의 tuple 반환을 tensor 반환으로 고쳐 다음 layer 입력 오류를 해결한다.
2. 존재하지 않는 속성을 참조하는 `get_config`, training 인수 기본값, optimizer 인수의 호환성을 고친다.
3. AVT는 수정된 원 `hique()`를 직접 호출한다. 단일·이중 모델은 동일 backbone의 branch 제거 ablation이다. 저장소의 서로 다른 named baseline을 혼합하지 않는다.
4. 잘못 누적되는 공개 feature loader를 대체한다. 공개 VGG 영상 추출 및 혼합 프레임워크 text 경로 대신 논문 서술에 맞는 CLNF/RoBERTa 특징을 구현한다.
5. 참가자 분리, 품질 기준, train-only scaling, 공통 슬롯 비교 및 올바른 truth/prediction 지표 계산을 추가한다. 이 전처리 변경도 원 논문과의 차이로 보고한다.

수정 목록은 `Data/hique_reproduction/model_patch_manifest.json`에 기록한다. 원 코드와 wrapper의 AVT 출력 일치, 저장/복원, 7개 조건의 입력 영향 및 유한 gradient를 테스트한다.

## 환경과 준비

이번 실행은 GPU 없이 CPU에서 수행했다. 시스템 환경을 변경하지 않고 두 환경을 사용했다.

- `.tmp/hique-features`: Python 3.13, PyTorch 2.10, Transformers 5.9, openSMILE 2.6.0, NumPy 2.1.3, pandas 2.2.3.
- `.tmp/hique-tf`: Python 3.10, TensorFlow CPU 2.15.1, NumPy 1.26.4, scikit-learn 1.6.1.
- 전체 버전은 `Data/hique_reproduction/environment_{features,tf}.txt`에 보관한다.
- upstream checkout은 `Data/hique_reproduction/upstream`, 논문 HTML은 같은 디렉터리의 `paper.html`에 필요하다.
- `FacebookAI/roberta-base` revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`의 tokenizer/config/model.safetensors를 `.tmp/hique_weights/roberta-base`에 보관한다. 모든 추출은 `local_files_only=True`로 실행한다.
- `protocol.json`은 특징 추출·학습 이전에 고정했다. 공개 자료 다운로드 외에 참가자 데이터 외부 전송이나 LLM API 호출은 없다.

저장소 루트에서 실행한다. 기존 산출물은 아래 명령을 무작정 재실행해 덮어쓰지 않는다. 다른 설정의 탐색은 별도 디렉터리와 별도 프로토콜로 구분한다.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m urop legacy run --python .tmp/hique-features/bin/python experiments.hique_data -- --stage prepare --threads 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m urop legacy run --python .tmp/hique-features/bin/python experiments.hique_data -- --stage text --threads 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m urop legacy run --python .tmp/hique-features/bin/python experiments.hique_av -- --segments Data/hique_reproduction/segments.jsonl --zip-dir Data/DAIC-WOZ --output-dir Data/hique_reproduction/av --workers 4
PYTHONPATH=src python -m urop legacy run --python .tmp/hique-features/bin/python experiments.hique_data -- --stage assemble
PYTHONPATH=src python -m urop legacy run --python .tmp/hique-tf/bin/python experiments.hique_train -- --features Data/hique_reproduction/features.npz --zip-dir Data/DAIC-WOZ --output-dir Data/hique_reproduction/runs --phase train
PYTHONPATH=src python -m urop legacy run --python .tmp/hique-tf/bin/python experiments.hique_train -- --features Data/hique_reproduction/features.npz --zip-dir Data/DAIC-WOZ --output-dir Data/hique_reproduction/runs --phase evaluate
PYTHONPATH=src python -m urop legacy run experiments.hique_report --
```

텍스트는 길이순 batch(최대 32개/4,096 padded tokens)로 추출 후 원래 슬롯 순서를 복원한다. 7,007개 슬롯 중 424개는 512 tokens에서 자른다. 최초 느린 batch 실행을 중단 요청하고 이 방식으로 전체 재실행했다. 최초 프로세스가 나중에 종료되며 text cache를 덮어쓴 사실을 발견했다. 학습에 사용 중인 조립 완료 `features.npz`는 hash가 유지됐으며, 이 파일의 원래 batched T 값으로 cache를 복원했다. 두 추출값의 최대 절대 차이는 4.77e-6이었다. `text_cache_recovery.json`에 복구 기록과 hash를 보관한다. 실제 실행한 소스는 `code_snapshots/hique_data_text_executed.py`로 보관했으며 이후 변경은 타입 주석뿐이다.

## 동결과 평가

`runs/frozen_config.json`은 feature 파일, trainer/network 코드, 프로토콜, manifest 및 split/label 파일의 hash와 실제 참가자 목록을 저장한다. Test label 파일은 이 단계에서 바이트 hash만 계산하며 outcome을 파싱하지 않는다. Train 재개 시 다른 설정이면 중단한다.

35개 run이 끝나고 모든 가중치와 입력 hash가 일치한 후에만 evaluate가 test outcome을 읽는다. `evaluation_manifest.json`에 시작·완료 시각과 결과 hash를 기록한다. 완료한 test 결과에 대한 재평가는 기본적으로 거부한다. 문헌의 공개 사례(381/470) 노출은 프로토콜에 명시했으며 사례별 설정 조정에 사용하지 않았다.

## 산출물

- `Data/hique_reproduction/results_ko.md`, `modality_results.csv`: 집계 결과.
- `runs/test_results.json`: seed별 지표, 평균·SD, seed 평균 확률 지표, bootstrap.
- `runs/{mode}_seed{seed}`: 100-epoch history, 최저 dev loss 가중치, dev/test 예측, 증강 기록, 완료 hash.
- `av/`, `text.npz`, `features.npz`: 재사용할 특징과 품질 기록.
- `segments.jsonl`, 질문 mapping 및 참가자별 예측은 비공개 자료다. 모든 데이터·가중치·upstream은 Git 제외되는 `Data/`/`.tmp/` 아래에 둔다.

당시 검증 명령 기록(이 경로들은 스냅샷 복원 환경 기준이다. 현재 검증 진입점은 `python -m urop legacy verify`):

```text
python3 -m pytest -q
.tmp/hique-tf/bin/python -m pytest tests/test_hique_network.py tests/test_hique_train.py -q
.tmp/hique-features/bin/python -m pytest tests/test_hique_av.py -q
python3 -m ruff check experiments tests --select E9,F63,F7,F82
python3 -m mypy experiments --ignore-missing-imports
```
