# 재현 대상으로 선정한 실험: HiQuE 기반 모달리티 추가 효과

2026-09-27 · 추천·실행안이며 학습을 실행한 결과가 아니다.

## 선정

**Jung et al., HiQuE: Hierarchical Question Embedding Network for Multimodal Depression Detection, CIKM 2024**의 모달리티 비교 실험을 추천한다. [논문](https://arxiv.org/html/2408.03648v1), [공식 코드](https://github.com/JuHo-Jung/HiQuE).

질문: **동일 참가자·질문 구조·학습 조건에서 텍스트에 음성과 시각을 추가하면 PHQ-8 기반 선별 예측이 안정적으로 좋아지는가?**

기존 A의 AI 잠정 주석 일치도와 다른 과제다. 이번 target은 데이터셋의 PHQ-8 선별 라벨이다. 새 임상 주석 없이 실험할 수 있으나 자기보고 라벨이며 독립 임상 진단·증상 은폐의 정답은 아니다.

## 논문에서 확인한 사실

§4.3–4.4는 질문 계층을 구성하고 각 질문 응답의 T/A/V를 처리한다. T는 RoBERTa 표현, A는 openSMILE eGeMAPS 88 functionals, V는 제공된 CLNF 2D 얼굴점의 평균·분산 272차원이다. 원본 얼굴 영상이 없는 현재 데이터 형식과 맞는다. §6.1은 단일·두 모달리티·세 모달리티를 비교한다. Table 1의 저자 보고 macro-F1은 0.79, weighted-F1은 0.82이며 우리 재현 결과나 보장 목표가 아니다. [원문 방법·평가](https://arxiv.org/html/2408.03648v1)

## 추천 이유

- 실제 보유한 세 모달리티를 모두 사용하고, 질문 맥락이라는 기존 관심과 이어진다.
- PHQ-8 라벨을 사용해 199개 AI 주석의 작은 표본과 순환 평가 문제를 주학습에서 피할 수 있다.
- 방법론 전체의 우월성을 주장하기보다 ‘추가 모달리티가 무엇을 더하는가’를 직접 비교하기 좋다.
- 생성형 LLM API는 핵심 경로에 필요하지 않다. 로컬 텍스트 encoder·특징 추출·분류기를 사용할 수 있다.
- 정식 학회 논문과 저자 코드가 있어 출발 근거가 구체적이다. 다만 아래 코드 결함 때문에 즉시 실행 가능한 패키지라고 보지는 않는다.

다른 비교 후보인 [Zhang & Poellabauer, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.650/)도 의미 있지만 LLM 질문 기능 주석·합성 자료 등 재현 범위가 넓다. 첫 T/A/V 기여도 실험에는 HiQuE 쪽을 선택한다. [Phenomics 2024의 삼중 모달리티 앙상블](https://pmc.ncbi.nlm.nih.gov/articles/PMC11467147/)도 후보이나, 이번 확인에서는 HiQuE처럼 직접 연결된 저자 구현을 확인하지 못했다. 코드가 없다고 확정한 것은 아니다.

## 실행안: 하나의 실험, 일곱 조건

| 조건 | 입력 |
|---|---|
| T | 참가자 답변 텍스트 |
| A | 참가자 음성 |
| V | 시각 특징 |
| T+A | 텍스트·음성 |
| T+V | 텍스트·시각 |
| A+V | 음성·시각 |
| T+A+V | 세 모달리티 |

모달리티마다 별도 유리한 cohort를 사용하지 않고, T/A/V 공통 품질 조건을 통과한 같은 참가자로 비교한다. 먼저 train/dev에서 파서·학습을 검증한 뒤 모델·threshold·seed 목록을 고정하고 test를 평가한다. 참가자의 segment와 증강본은 항상 같은 split에 둔다. 시간·화자·scrubbed·visual success mask를 적용하며 원본 ZIP은 수정하지 않는다.

추가 대조군은 **질문 등장 여부 mask만 사용하는 모델** 하나다. 질문 slot의 존재 자체가 라벨을 예측한다면, 결합 모델이 참가자의 행동보다 면담 질문 경로를 이용하는지 해석해야 한다. 이 대조군은 우리의 재현 보강안이며 원 논문의 일곱 조건과 구분한다. 질문 shortcut의 근거는 [Burdisso et al., ClinicalNLP 2024](https://aclanthology.org/2024.clinicalnlp-1.8/)다.

주 지표는 macro-F1, 보조로 양성 F1·민감도·특이도·AUROC·혼동행렬을 보고한다. 고정 seed 5개의 결과와 참가자 단위 paired 차이·CI를 구분한다. 세 모달리티와 각 비교군의 차이를 보고하며 ‘좋은 seed’나 시험 결과로 선택한 최선 비교군을 주 근거로 삼지 않는다.

성공은 논문 수치에 도달하는 것만이 아니다. 같은 조건의 단일·결합 모델을 재현 가능하게 비교하고 추가 modality의 이득·부담·오류를 설명하는 것이다. Attention 값만으로 임상적 설명의 정당성을 주장하지 않는다.

## 실행 전에 해결할 재현 문제

1. **공개 코드 로더 결함:** 확인한 [hique.py의 get_features](https://github.com/JuHo-Jung/HiQuE/blob/main/code/hique.py)는 참가자 loop 안에서 modality list를 다시 초기화하고 참가자별 누적을 하지 않는다. 또한 시각 누락값을 4096차원으로 넣는데 [hique 모델 입력](https://github.com/JuHo-Jung/HiQuE/blob/main/code/fusionmodel.py)은 272차원이다. 정적 확인이므로 실제 실행 증명과 구분한다. 로더·shape를 검사하고 필요한 수정을 기록해야 한다.
2. **논문/코드 실행 설정 대조:** 논문과 코드 기본 epoch/batch 설정 등이 일치하지 않는 부분이 있다. 특정 commit과 사용 설정을 고정하고 임의 보완을 원 논문 그대로라고 부르지 않는다.
3. **현재 환경:** 로컬 PyTorch·Transformers가 있지만 CUDA 장치는 보이지 않고 openSMILE Python 패키지는 없다. 특징 추출 라이브러리·encoder 가중치·실행 자원 준비가 필요하다. CPU 실행 시간은 아직 측정하지 않았다. 원 구현은 TensorFlow 계열이므로 PyTorch 이식은 별도 차이로 기록한다.
4. **전처리 판본:** 공식 README의 ASR/질문 매핑 경로와 제공 수동 전사 사용을 분리한다. 제공 전사로 먼저 진행하면 그 변경을 명시한다. COVAREP나 AU 몇 개로 대체하는 것은 원 특징의 정확 재현이 아니다.
5. **파일·라벨:** 440 손상, 질문 전사 없는 451·458·480, 402의 시각 범위, 사전 사례 노출을 기존 정책과 맞춰 처리한다. 현재 공통 cohort는 확정 전이다. 원 논문의 107/35/47 전체와 다르면 숫자를 직접 순위 비교하지 않는다.
6. **PHQ 원본 유지:** released binary를 주 target으로 사용할지 score-derived binary를 사용할지 사전 고정한다. 409의 불일치를 조용히 덮어쓰지 않는다. test label 출처·버전도 확인한다.

## 확인 수준

논문 관련 방법·모달리티 ablation·학습 설정 및 저자 README, `code/hique.py`, `code/fusionmodel.py`를 웹으로 확인했다. 저장소 전체 감사·설치·학습·성능 재현은 수행하지 않았다. 일부 특징 추출 파일의 직접 조회는 실패했다. **추천 대상은 논문에 기반한 통제 재현 실험이며, 공식 코드를 그대로 실행하면 결과가 재현된다는 보장은 아니다.**
