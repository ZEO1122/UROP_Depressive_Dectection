# UROP · 우울증 평가 지원 연구

우울증 평가 지원을 위한 문헌 조사와 DAIC-WOZ 멀티모달 실험 저장소입니다. 연구 방향, 실행 코드, 실험 결과와 과거 기록을 역할별로 관리합니다.

## 먼저 볼 문서

- [HiQuE 단일 파이프라인 노트북](experiments/hique/v3_input_study/pipeline.ipynb): 단계별 한국어 설명, 논문·공개 코드 대응, 기존 결과 확인과 새 실행.
- [연구 방향](docs/research_direction.md)
- [실험 목록과 상태](experiments/README.md)
- [현재 V3 실험](experiments/hique/v3_input_study/README.md) · [dev 결과](experiments/hique/v3_input_study/dev_results.md) · [test 결과](experiments/hique/v3_input_study/test_results.md)
- [데이터 구성·알려진 문제](docs/dataset.md)
- [실행 방법·과거 기록 검증](docs/reproducibility.md)
- [문헌 조사](docs/literature/README.md) · [성능 격차 진단](docs/hique_diagnosis.md)

## 현재 상태

V3는 다섯 조건을 각각 다섯 시드로 학습하고, 저장된 25개 모델을 공통 test 45명에서 평가했습니다. Dev에서 사전에 선택한 **제공 전사·첫 답변(E2)의 test Macro-F1은 0.6541 ± 0.0344**입니다. 주 비교 E1−E0의 95% 재표집 구간은 0을 포함합니다.

이 결과는 원 논문의 0.79를 동일 조건에서 재현한 값이 아닙니다. 기존 test 노출과 전처리·계층 구현 차이가 있으며, PHQ 선별 라벨 예측을 임상 진단으로 해석하지 않습니다. V1/V2에는 토크나이저 로딩 오류가 확인돼 과거 기록으로 보존합니다. 자세한 제한은 각 실험 결과에 있습니다.

## 저장소 구조

```text
src/urop/                   현재 공통 코드와 CLI
  data/                     전사·질문·구간 처리
  features/                 음성·영상·텍스트 특징
  models/                   모델·공개 코드 어댑터
  training/                 학습·증강·dev 보고
  evaluation/               지표·test 평가·보고
experiments/                실험별 설명·등록 설정·집계 결과
  evidence_pilot/           초기 세 탐색 실험
  ai_annotation_pilot/      AI 잠정 주석 실험
  hique/v1/, v2/            오류·한계를 포함한 과거 결과
  hique/v3_input_study/      현재 입력 보강 실험·실행 노트북
  README.md                 실험 목록
docs/                       연구 방향·문헌·데이터·주차별 기록
archive/                    이전 설계와 변경하지 않는 실행 소스
tests/                      현재 코드의 회귀검사
requirements/               기존 특징·학습 환경의 직접 의존성 버전
Data/                       비공개 원본·캐시·가중치·참가자별 결과
output/                     생성 PDF·발표자료
```

`Data/`, `.tmp/`, `output/`은 로컬 자료이며 Git에 포함하지 않습니다. 2026-10-04 구조 정리는 새로운 학습이나 test 평가를 수행하지 않았습니다.

## 실행 시작

기존 라이브러리가 설치된 환경에서 다음 명령으로 진입할 수 있습니다.

```sh
PYTHONPATH=src python3 -m urop --help
PYTHONPATH=src .tmp/hique-features/bin/python -m urop features --help
PYTHONPATH=src .tmp/hique-tf/bin/python -m urop train --help
```

패키지는 기존 두 격리 환경에 의존성 추가 없이 설치했습니다. 새 환경의 설치와 실제 파이프라인 인수는 [재현 안내](docs/reproducibility.md)를 참고하세요. 실험의 `config.json`은 설명용 등록 정보이며 자동 실행 설정으로 읽히는 파일은 아닙니다.

## 과거 실행 감사

과거 소스는 `archive/snapshots/pre-reorg-2026-10-04/`에 바이트 그대로 보존합니다. 과거 절대 경로를 요구하는 검증은 호환 실행기가 소스를 잠시 복원해서 수행합니다.

```sh
PYTHONPATH=src python3 -m urop legacy verify
PYTHONPATH=src python3 -m urop legacy run --python .tmp/hique-tf/bin/python experiments.hique3_test -- --help
```

소스·입력·가중치 hash와 기존 기록은 유지합니다. 과거 frozen JSON을 새 소스 hash로 덮어쓰지 않습니다. [보관 소스 안내](archive/snapshots/pre-reorg-2026-10-04/README.md)를 참고하세요.

## 검증

```sh
python3 -m pytest -q
PYTHONPATH=src .tmp/hique-tf/bin/python -m pytest tests/models tests/training tests/evaluation -q
python3 -m ruff check src tests
python3 -m mypy src/urop
```

현재 테스트와 보관된 원본 회귀검사는 분리합니다. 데이터·가중치·환경이 필요한 검사는 해당 로컬 환경에서 실행해야 합니다. 원문 전사·오디오·얼굴 특징·개인별 예측과 자격 증명은 공유 대상에서 제외합니다.
