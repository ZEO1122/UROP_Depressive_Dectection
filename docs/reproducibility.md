# 실행과 기록 보존

현재 구현은 `src/urop/`, 실험 설명·집계 결과는 `experiments/`, 정확한 과거 실행 소스는 `archive/snapshots/pre-reorg-2026-10-04/`에 있다. 먼저 [실험 목록](../experiments/README.md)에서 상태와 사용할 경로를 선택한다.

## 현재 V3 구현

[단일 파이프라인 노트북](../experiments/hique/v3_input_study/pipeline.ipynb)은 현재 V3를 설명 셀과 실행 셀로 나눈 진입점이다. 저장소 또는 실험 폴더에서 열고 Python 3 커널을 사용한다. 기본 `RUN_PIPELINE=False`는 기존 결과 읽기·입력 검사·모델 구성만 수행한다. 새 실행은 새 `RUN_NAME`과 `RUN_PIPELINE=True`를 설정한다. 후속 test는 25회 학습 완료 후 `RUN_TEST=True`로 켠다. 특징과 학습은 기존 두 Python 환경을 별도 프로세스로 호출하므로 커널에 두 ML 환경을 합칠 필요가 없다. 노트북만 복사해서 실행하는 독립 배포물은 아니며, 기존 데이터·ASR/A/V 기준선·질문 매핑·로컬 RoBERTa가 필요하다. 논문/공개 코드와 현재 고정 위치 V3의 차이는 각 단계에서 설명한다.

저장소 루트에서 패키지를 설치한 뒤 같은 인터프리터로 CLI를 실행한다. 이미 준비된 환경에 새 의존성을 설치하는 명령이 아니다.

```sh
python -m pip install --no-deps --no-build-isolation -e .
python -m urop --help
python -m urop prepare --help
python -m urop features --help
python -m urop train --help
python -m urop test-features --help
python -m urop evaluate --help
python -m urop report --help
python -m urop test-report --help
```

특징 추출에는 기존 `.tmp/hique-features` 환경, 작은 TensorFlow 모델 학습에는 `.tmp/hique-tf` 환경을 사용했다. 해당 환경에서도 위처럼 패키지를 설치하거나 `PYTHONPATH=src`를 명시한다. GPU 추출은 실제 GPU 접근 가능 여부를 확인한다. 과거 CPU classifier 실행은 GPU가 없는 기기라는 뜻이 아니다.

현재 사용 중인 직접 의존성 버전은 [특징 환경](../requirements/features.txt)(Python3.13)과 [학습 환경](../requirements/training.txt)(Python3.10)에 나누어 기록했다. 이번 정리에서 외부 라이브러리를 추가 설치하거나 업그레이드하지 않았고, 두 기존 환경에 로컬 프로젝트만 `--no-deps --offline`으로 설치했다. 이 목록은 전체 전이 의존성 lockfile은 아니며, 과거 전체 버전 목록과 가중치 hash는 기존 Data 실행 폴더에 있다. 서로 다른 NumPy/TensorFlow 버전을 한 환경으로 합치지 않는다.

V3 입력·모델·평가 의미는 [V3 설명](../experiments/hique/v3_input_study/README.md)을 따른다. 등록용 `config.json`은 자동 실행 설정이 아니다. 실제 명령의 모든 provenance 인수와 출력 경로를 확인하고, **새 실행에는 새 출력 경로와 동결 설정을 사용한다.** 리팩터링된 소스 hash를 과거 frozen config에 억지로 맞추거나 기록을 수정하지 않는다.

## 과거 실행 소스

legacy runner는 스냅샷 manifest의 hash를 확인하고 과거 `experiments/*.py` 경로에 정확한 실행 소스를 잠시 복원한다. 테스트는 스냅샷의 `tests/` 위치에서 실행한다. 충돌하는 파일을 덮어쓰지 않으며 실행 후 자신이 만든 변경 없는 파일만 정리한다.

```sh
PYTHONPATH=src python -m urop legacy run experiments.run_suite -- --help
PYTHONPATH=src python -m urop legacy run --python .tmp/hique-tf/bin/python experiments.hique3_train -- --help
PYTHONPATH=src python -m urop legacy verify
PYTHONPATH=src python -m urop legacy run pytest archive/snapshots/pre-reorg-2026-10-04/tests -q
```

`--python`은 실행할 **module 이름 앞**에 둔다. `--` 뒤의 인수는 해당 과거 모듈로 전달한다. `verify`는 기존 자료의 hash를 읽어 검사하며 재학습이나 새 test 예측을 수행하는 명령이 아니다. 기존 Data의 `run_*.sh`, `verify_*.py`는 당시 명령의 증거로 그대로 보존했다. 이 스크립트가 옛 모듈 경로를 사용한다고 현재 작업 트리에서 직접 실행할 수 있다고 가정하지 않는다.

V1/V2에는 토크나이저 오류가 있다. legacy는 잘못된 과거 결과를 고치는 기능이 아니라 **당시 실행 소스의 보존**이다. 정상 tokenizer를 검증한 현재 결과는 V3이며, 과거 source line 참조도 스냅샷 파일의 줄 번호로 읽는다.

## 검증 원칙

코드 정리는 특징·예측의 의미를 바꾸지 않아야 한다. 가상 자료 회귀 검사, tokenizer ID 검사, 사건 경계·모달리티 mask·장문 pooling 검사, 저장 checkpoint와의 수치 일치 검사를 구분한다. 실행 환경마다 사용할 수 있는 라이브러리가 다르므로 TensorFlow 전용 검사는 해당 환경에서 수행한다.

공개 문서의 E2 test Macro-F1 0.6541은 저장된 평가 기록이다. 문서 생성이나 구조 정리를 새 실험으로 보고하지 않는다. Test 사전 노출, AI 주석의 비독립성, V1/V2 오류 같은 기존 한계도 유지한다.

2026-10-04 이동 전 백업·계획·원본 hash와 문서 경로 대응표는 로컬 `Data/repo_reorg_2026-10-04/`에 보관한다. 이전 대화의 파일 링크가 과거 경로를 가리키면 `doc_moves.json`에서 새 경로를 찾는다.
