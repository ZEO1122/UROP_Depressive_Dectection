# 우울증 탐지 선행연구와 세부 연구주제 제안

> 방향 변경: 이 문서는 초기 조사 기록이다. 현재 연구 목표와 추천은 [Assessment·DSM·LLM 기반 수정안](../research_direction.md)을 따른다.

조사일: 2026-09-14 · 도구: OpenResearch CLI 0.2.1 · 목적: UROP Week1 이후의 첫 재현과 연구주제 선정

## 1. 현재 프로젝트에서 출발할 지점

레포는 현재 모델 구현보다는 Week1 발표 자료와 제작·검증 자료 중심이다. 최신 수정본인 `output/daicwoz_format_locked/UROP_Week1_Depression_DAICWOZ_FormatLocked_Revised_Highlighted.pptx`의 36개 슬라이드와 발표 노트를 확인했다. 핵심은 MDD와 선별의 차이, PHQ-8 target, DAIC-WOZ의 text/audio/visual 구조, 화자·시간 정렬, 라벨 예외다. 마지막 계획은 선행연구 재현, 추가 데이터셋 확보, DAIC-WOZ 밖의 방법론 조사다.

따라서 연구의 target은 우선 **인터뷰로부터 PHQ-8 점수 또는 선별 라벨을 예측하는 것**으로 정의한다. 임상 진단 성능, 우울증의 인과적 바이오마커 발견과는 구분한다. 이 문서의 구현 제안과 난도 평가는 조사자의 판단이며 실험 결과가 아니다.

**추천 출발점은 ‘질문 맥락을 통제했을 때 대화 시간 특징이 참가자 텍스트에 추가 정보를 제공하는가?’다.** 시간 특징은 현 데이터의 timestamp로 구현을 시작할 수 있고, 질문 편향·멀티모달 결합·일반화라는 문제를 하나의 실험으로 연결한다. 다만 해당 문제의 일부는 이미 연구되어 있으므로, 아래 제안은 신규성이 확정된 논문 주제가 아닌 검증할 후보다.

## 2. 조사 범위와 증거 수준

OpenResearch의 alphaXiv keyword/embedding과 OpenAlex를 사용했다. 검색 상한은 2026-09-14, 하한은 두지 않았다. 최초 세 검색과 baseline 보완 검색 한 번에서 후보를 얻어 중복·비관련 항목을 제외하고 15편을 골랐다. 체계적 문헌고찰이나 전체 분야의 누락 없는 목록은 아니다. 특히 전통적인 초기 baseline보다 최근 인터뷰 기반 연구가 많이 검색됐다.

핵심 5편은 본문을 확인했다. 나머지 10편은 **검색 메타데이터·초록에서 선정한 후속 읽기 후보**다. 후속 후보의 성능과 세부 프로토콜을 검증된 결과로 사용하지 않았다. arXiv로 확인한 2026년 논문은 이 조사에서 출판 여부를 별도로 확정하지 않았으므로 preprint로 취급한다. 검색 응답의 날짜는 버전·색인 날짜일 수 있다.

## 3. 우선 읽을 핵심 5편

### P01. 질문 편향의 출발점 — Burdisso et al., 2024

[DAIC-WOZ: On the Validity of Using the Therapist’s prompts in Automatic Depression Detection from Clinical Interviews](https://www.alphaxiv.org/abs/2404.14463) · [ClinicalNLP 출판본](https://aclanthology.org/2024.clinicalnlp-1.8/)

- **질문:** 질문을 추가해서 좋아진 성능이 참가자의 언어를 더 잘 이해한 결과인가, 질문 선택 패턴을 이용한 것인가?
- **방법:** Longformer 계열과 GCN에서 참가자만(P), Ellie만(E)을 각각 학습하고 두 예측의 AND 앙상블을 비교한다. 단어의 판별 기여가 인터뷰 어느 구간에 집중되는지도 분석한다.
- **평가:** train 107명, evaluation 35명. 이 evaluation은 공식 development set이다. Table 2의 macro-F1은 P-GCN 0.85, E-GCN 0.88, AND 앙상블 0.90이다. 0.90을 official test 성능이나 depressed-class F1로 인용하면 안 된다.
- **해석:** 일부 후속 질문의 등장 자체가 예측 단서가 될 수 있음을 보여준다. 질문 맥락의 모든 사용이 부당하다는 결론이나, 모든 모델에서 질문이 항상 더 강하다는 결론은 아니다.
- **첫 구현 가치:** 같은 모델에서 입력만 P/E/P+E로 바꿔볼 수 있다. 저비용 TF-IDF+선형 모델은 이 논문의 정확 재현은 아니지만 문제를 확인하는 출발점이다.

근거: 논문 §§3–5, Table 2. [본문](https://www.alphaxiv.org/abs/2404.14463)

[저자 코드](https://github.com/idiap/bias_in_daic-woz)는 논문 대응을 확인했고 평가 스크립트, GCN 재학습 절차, 모델·vectorizer 설명을 제공한다. Longformer까지 동일 수준으로 재현된다고 가정하지 않는다. README의 좋은 점수가 나올 때까지 seed를 바꾸는 제안은 새 연구의 평가 절차로 채택하지 말고 seed 목록을 사전 고정해야 한다. 코드를 실행하거나 checkpoint를 검증한 것은 아니다.

### P02. 질문 효과의 확장과 예외 — Watawana et al., 2026

[When Consistency Becomes Bias: Interviewer Effects in Semi-Structured Clinical Interviews](https://www.alphaxiv.org/abs/2603.24651)

- **질문:** 면접자 질문 효과가 DAIC-WOZ 밖에서도 나타나는가?
- **방법:** DAIC-WOZ, E-DAIC, ANDROIDS에서 participant-only와 interviewer-only를 Longformer/GCN으로 비교한다. E-DAIC와 ANDROIDS에는 ASR 전처리가 포함된다. E-DAIC는 두 화자를 맞춘 ASR 조건이므로 gold transcript 결과와 직접 비교할 수 없다.
- **평가:** Table 1의 ANDROIDS는 5-fold 평균, DAIC-WOZ/E-DAIC는 dev 결과다. Table 2는 test 결과를 별도로 제시한다.
- **중요한 예외:** DAIC-WOZ test의 Longformer는 P 0.68, I 0.53 macro-F1로 참가자 쪽이 높다. GCN은 P 0.59, I 0.62다. 즉 질문 효과의 크기와 방향은 모델·분할에 따라 달라진다.
- **주제 선정 의미:** ‘다른 데이터셋에서도 질문 편향이 있는가?’까지 이미 조사됐다. 질문 제거 이후 무엇이 유지되는지, 어떤 조건에서 무너지는지를 연구해야 차별성이 생긴다.

근거: Data Preparation, Tables 1–2, Results/Discussion. [본문](https://www.alphaxiv.org/abs/2603.24651) · 이 조사에서 별도 저자 코드 저장소는 확인하지 못했다.

### P03. 평가 설계 자체를 점검 — Ishikawa & Duke, 2026

[A Multi-Probe Audit of Clinical-Interview Depression Detection Benchmarks](https://www.alphaxiv.org/abs/2605.23977)

- **질문:** 공식 split의 순위가 안정적인가? 다른 corpus에도 전달되는가? 텍스트 예측은 증상을 직접 말하는 구간에 얼마나 민감한가?
- **방법:** E-DAIC subject-disjoint LOSO, 96개 설정의 CV–test 순위 비교, CMDC/ANDROIDS baseline의 외부 평가, 같은 참가자의 symptom-dense/light 구간 비교를 수행한다.
- **결과:** E-DAIC LOSO에서 text+LLM score의 macro-F1 0.723을 보고한다. 공식 test 최고 모델은 참가자 bootstrap의 32.3%에서만 1위다. 이 수치는 임의의 논문이나 DAIC-WOZ 전체에 적용할 상수가 아니다.
- **한계:** topic slice는 SRDS 기반 자동 정의에 의존한다. 언어와 corpus가 동시에 바뀌는 외부 실험도 있어 성능 저하의 원인을 하나로 분리하기 어렵다. LLM pretraining 오염의 부재도 인증하지 않는다.
- **가져올 요소:** 참가자 단위 paired bootstrap, 같은 데이터 조건에서 비교, 전체 pipeline을 fold 내부에서 학습하는 평가 방식이다. symptom-dense/light 비교만 새로 제안하면 이 논문과 겹친다.

근거: §§2–5, Tables 3–7. [본문](https://www.alphaxiv.org/abs/2605.23977)

논문은 Probe A의 T+L/L-only 코드를 [Zenodo v2](https://doi.org/10.5281/zenodo.19813142)에 공개했다고 명시한다. 이 조사에서는 해당 페이지 접근이 실패했으므로 파일·실행 가능성은 미확인이다. 네 probe 전체 코드가 공개되었다고 확대하지 않는다.

### P04. 가장 작은 구현 출발점 — 대화 시간 특징, 2026

[Can Conversational Temporal Dynamics Improve Depression Detection in Dyads? A Preliminary Investigation in Multi-Modality Perspectives](https://www.alphaxiv.org/abs/2607.03744)

- **질문:** 무엇을 말했는지·음성이 어떤지 외에, 질문과 응답의 시간 구조가 도움이 되는가?
- **방법:** 질문·응답 길이, 길이 비율, 응답 지연, 발화·침묵 비율 등 24개 CTD 특징을 turn pair마다 계산해 세션 평균으로 만든다. train-fit scaling과 L2 logistic regression을 사용하며, frozen RoBERTa/WavLM의 예측과 late fusion한다.
- **평가:** 보고된 cleaned split은 102/33/45다. CTD 단독 dev/test macro-F1은 0.746/0.631, 텍스트+CTD는 0.804/0.669다. test fusion의 95% CI는 [0.509, 0.806]이며 paired 차이의 CI는 0을 포함하거나 접한다. 우월성이 확정됐다고 해석할 수 없다.
- **한계:** fusion 가중치·threshold가 dev에서 선택됐다. 이 조건에서는 음성 가중치가 0이지만 음성 modality가 일반적으로 무용하다는 뜻은 아니다. 면접자 timing에도 질문 선택 과정의 영향이 남을 수 있다고 논문이 명시한다.
- **재현 전 확인:** 본문은 10개 세션 제외와 180개 최종 세션을 함께 적는다. 189−10은 179이므로 제외 목록·원래 cohort·공통 manifest를 코드와 대조해야 한다. 원 논문의 숫자를 임의로 고쳐 쓰지 않는다.
- **특징 해석:** 여기서 ‘voiced time’은 transcript utterance 길이 합에 따른 정의다. 실제 VAD로 측정한 순수 유성음 시간과 같다고 가정하지 않는다.

근거: §§III–VIII, Tables I–III. [본문](https://www.alphaxiv.org/abs/2607.03744)

[저자 저장소](https://github.com/dndbsl/depression-detection-ctd)에 CTD와 fusion의 CPU 실행 절차가 있다. 전체 신경망 경로는 GPU를 사용한다. README는 score≥10으로 binary를 구성하므로 Week1의 released binary 유지 정책과 별도 조건으로 구분해야 한다. 공개 코드가 있다는 사실을 확인했으며 실행 재현은 하지 않았다.

### P05. 음성 딥러닝 비교군 — Zhang et al., 2024

[Improving speech depression detection using transfer learning with wav2vec 2.0 in low-resource environments](https://doi.org/10.1038/s41598-024-60278-1) · Scientific Reports

- **방법:** 참가자 음성을 7초 단위로 나누고 wav2vec 2.0, 1D-CNN/attention pooling, LSTM/self-attention을 결합한다.
- **조건:** DAIC-WOZ는 train/dev를 사용해 F1 0.79를 보고한다. CMDC의 0.9053은 별도 데이터셋 실험이며 DAIC에서 학습한 모델의 zero-shot transfer 결과가 아니다. F1 정의를 macro-F1과 동일시하지 않는다.
- **구현 판단:** frozen encoder+작은 head를 먼저 만들고 그다음 fine-tuning 비교군으로 고려한다. 본문에 독립된 코드 공개 항목은 확인되지 않았다. 정확 재현에는 split, pooling, checkpoint, threshold를 더 확인해야 한다.

근거: 출판사 본문 Materials and methods, Datasets description, Results. [본문](https://www.nature.com/articles/s41598-024-60278-1)

## 4. 추가로 읽을 10편: 초록 기반 후보

아래 표의 설명은 검색 초록에 제시된 연구 방향이다. 정량 결과, 저자 코드, 세부 평가의 타당성은 아직 검증하지 않았다. 후보 제목의 ‘diagnosis’나 초록의 ‘clinical utility’는 이 문서가 검증한 임상적 효용을 뜻하지 않는다.

| ID | 문헌 | 초록에서 확인한 방향 | 다음에 확인할 사항 |
|---|---|---|---|
| P06 | [A Hierarchical Attention Network-Based Approach…](https://doi.org/10.21437/interspeech.2019-2036), 2019 | GloVe 기반 hierarchical attention으로 DAIC-WOZ transcript 분류 | 문장 분할, participant-only 여부, UAR와 F1 구분 |
| P07 | [A Step Towards Preserving Speakers’ Identity…](https://doi.org/10.21437/interspeech.2022-10798), 2022 | DAIC-WOZ/CONVERGE 음성의 speaker adversarial disentanglement | domain label, fold 구성, identity 제거와 성능의 관계 |
| P08 | [HiQuE: Hierarchical Question Embedding Network…](https://doi.org/10.1145/3627673.3679797), 2024 | 주질문·후속질문의 계층을 이용한 multimodal 결합 | 질문-only 대조군, 이용 가능한 visual feature, test 평가 |
| P09 | [Harnessing multimodal approaches…](https://doi.org/10.1038/s44184-024-00112-8), 2024 | E-DAIC에서 LLM 기반 텍스트 정보와 얼굴 특징을 비교 | 최선 모델이 실제로 어떤 modality를 쓰는지, 회귀 split |
| P10 | [D-vlog: Multimodal Vlog Dataset for Depression Detection](https://doi.org/10.1609/aaai.v36i11.21483), 2022 | 실험실 밖 vlog의 audio/visual 데이터와 cross-attention | label 생성 근거, 화자 중복, 최신 배포 가능성 |
| P11 | [Clinically Inspired Symptom-Guided Depression Detection…](https://www.alphaxiv.org/abs/2602.15578), 2026 preprint | PHQ-8 문항과 emotion-aware speech representation을 연결 | item supervision 유무, attention 해석 검증, 코드 |
| P12 | [DepressionAgent: Reading, Listening, Seeing, and Deliberating…](https://www.alphaxiv.org/abs/2608.13891), 2026 preprint | modality별 증거와 상충 정보를 agent 방식으로 종합 | local 실행 가능성, 시각 입력 요구, 반복 비용, explanation 평가 |
| P13 | [Who is Speaking or Who is Depressed?](https://www.alphaxiv.org/abs/2604.14354), 2026 preprint | 훈련 크기를 맞추고 speaker overlap 영향을 비교 | session/segment 분할 방식, identity와 질병 정보 분리 |
| P14 | [Do Depressive Facial Patterns Transfer Across Cultures and Contexts?](https://www.alphaxiv.org/abs/2609.05543), 2026 preprint | 독일 RCT와 E-DAIC의 얼굴 특징 전이 | 라벨 동등성, 인터뷰 상황, 공개 feature 접근 |
| P15 | [TAMFN: Time-Aware Attention Multimodal Fusion Network…](https://doi.org/10.1109/tnsre.2022.3224135), 2022 | D-Vlog에서 시간 정보와 audio/visual fusion | subject-disjoint 여부, P10과 동일 평가인지 |

이 후보들은 text attention → speech representation → question-aware fusion → LLM/증상 단위 모델 → 외부 일반화라는 읽기 경로를 제공한다. 목록이 최신 성능 순위나 exhaustive survey는 아니다.

## 5. 보고된 수치를 비교할 때 필요한 조건

| 논문/모델 | 데이터·평가 | 지표 | 보고값 | 직접 비교를 막는 차이 |
|---|---|---|---|---|
| P01 GCN P / E / ensemble | DAIC-WOZ dev 35명 | macro-F1 | 0.85 / 0.88 / 0.90 | test 아님; 모델 선택 포함 |
| P02 Longformer P / I | DAIC-WOZ test | macro-F1 | 0.68 / 0.53 | P01과 모델·선택 절차 다름 |
| P03 T+L | E-DAIC LOSO | macro-F1 | 0.723 | 다른 데이터·OOF 평가 |
| P04 CTD / text+CTD | cleaned DAIC-WOZ test 45명 | macro-F1 | 0.631 / 0.669 | 제외 세션·라벨 구성·refit 조건 |
| P05 wav2vec 계열 | DAIC-WOZ dev | 논문 F1 | 0.79 | macro-F1 여부 동일시 금지 |

수치의 직접 근거는 각 핵심 논문의 위 링크와 표 위치다. 이 표는 조건의 차이를 드러내기 위한 것이며 모델 순위를 매기는 표가 아니다.

## 6. 세부 연구주제 후보

### A. 질문 맥락을 통제한 대화 시간 특징의 추가 가치 — 1순위

**연구 질문:** 동일한 질문 유형과 유사한 응답 길이를 비교했을 때, 참가자의 응답 시간 특징이 텍스트 모델의 PHQ-8 선별 예측을 개선하는가?

P01/P02가 질문의 언어적 편향을 분석했고 P04는 시간 특징의 추가 가치를 보고하면서 질문 통제를 후속 과제로 남겼다. 따라서 단순 P/E 비교나 CTD 추가만으로는 부족하다. **같은 질문 범주·응답 길이 조건에서의 비교와 response-only 시간 특징**이 후보 차별점이다. 아직 관련 전체 문헌을 검토하지 않았으므로 신규성을 확정하지 않는다. [P01](https://www.alphaxiv.org/abs/2404.14463), [P02](https://www.alphaxiv.org/abs/2603.24651), [P04](https://www.alphaxiv.org/abs/2607.03744)

제안하는 최소 실험은 다음과 같다.

1. 텍스트 대조군: P-only, E-only, P+E의 TF-IDF+logistic regression. 같은 참가자 split과 같은 탐색 예산을 사용한다.
2. 시간 대조군: 총 세션 길이·응답 수만 사용한 모델, CTD 24개 전체, participant-side duration/pause만 사용한 모델을 분리한다. 질문 종료를 기준으로 한 response latency는 완전한 participant-only 특징이 아니므로 별도 열로 둔다.
3. 질문 통제: train에서 질문 범주를 정하고 공통 질문에서만 특징을 구성한다. 표본 수가 허용하면 응답 길이와 질문 빈도에 대한 matching/stratification을 추가한다. 선택 전후의 참가자 수·양성 비율을 공개한다.
4. 결합: P-only text와 각 timing 모델의 late fusion. dev 최적화와 무튜닝 평균 결합을 모두 비교한다.
5. 평가: 전체 session과 공통 질문 subset을 같은 참가자 cohort에서 평가하고 paired 차이·신뢰구간을 보고한다. 질문 subset 선택은 test label을 보지 않고 고정한다.

통제 후 효과가 유지되면 단순 질문 빈도 이상의 정보를 지지한다. 효과가 사라져도 ‘시간 특징은 쓸모없다’로 일반화하지 않고 이 데이터의 질문 정책 의존성을 보고할 수 있다. 관찰 데이터의 통제 비교만으로 우울증의 인과적 시간 바이오마커를 입증하지는 못한다.

**구현 난도 추정:** 낮음–중간. timestamp 기반 최소 모델은 CPU로 시작 가능하다. 주요 난점은 모델 규모보다 전처리·공통 cohort·질문 범주의 정의다.

### B. 증상 진술량에 따른 텍스트·음성의 상호 보완성 — 2순위

**연구 질문:** 우울 증상을 직접 말하는 내용이 적을 때, 음성이 텍스트 예측에 더 큰 추가 가치를 제공하는가?

P03은 이미 symptom-dense/light text–audio stress test를 수행했다. 새 기여를 만들려면 질문·길이를 맞춘 조건, 독립적인 증상 구간 정의, 실제 fusion 이득, 외부 corpus 중 하나 이상이 더 필요하다. P05는 음성 비교군 설계에 참고할 수 있다. [P03](https://www.alphaxiv.org/abs/2605.23977), [P05](https://doi.org/10.1038/s41598-024-60278-1)

고정한 규칙 또는 별도 annotation으로 symptom-heavy/light를 나누고, text-only/audio-only/fusion을 비교한다. 구간을 지웠을 때 성능 저하가 단순 입력 길이 감소 때문인지 확인하도록 random-removal 대조군을 둔다. 증상 문장 자체는 target과 관련된 유효한 정보일 수 있으므로 ‘증상 진술=데이터 누수’라고 명명하지 않는다.

**구현 난도 추정:** 중간. wav2vec/WavLM feature cache와 구간 정의 검증이 필요하다. 로컬 encoder와 작은 head로 시작할 수 있다.

### C. PHQ-8 문항 단위 예측의 이득과 불확실성 — 3순위

**연구 질문:** 총점 직접 회귀와 문항별 ordinal 예측 후 합산 중 어떤 방식이 안정적이며, 어떤 문항에서 오차가 큰가?

Week1의 문항 구조를 활용할 수 있고 P11이 직접적인 충돌 문헌 후보다. P11은 아직 초록만 확인했으므로 구현 전에 supervision과 실험을 정독해야 한다. ‘증상 기반’이라는 이유만으로 해석 가능성이 입증되지는 않는다. [P11](https://www.alphaxiv.org/abs/2602.15578)

공유 encoder+8개 ordinal head, total-score regression, multi-task 모델을 같은 split에서 비교한다. 문항 정답이 없는 test에서는 item MAE를 계산하지 않는다. Week1 기록의 participant 319 missing item은 loss mask로 처리하고 total만 있는 표본과 구분한다. 희소한 문항 점수와 작은 표본 수 때문에 복잡한 모델보다 단순 baseline의 불확실성을 먼저 측정한다.

**구현 난도 추정:** 중간–높음. 문항 라벨 접근 범위와 표본 크기가 제한 요소다.

## 7. 추가 데이터셋을 고르는 순서

| 후보 | 현재 근거 | 이 프로젝트에서의 역할 | 확보 전 확인 |
|---|---|---|---|
| E-DAIC | P02/P03 본문, [USC 배포처](https://dcapswoz.ict.usc.edu/) | DAIC 계열에서 인터뷰 조건 변화 평가 | DAIC-WOZ와 참가자 중복·모드 차이; 완전히 독립된 외부 검증으로 가정하지 않기 |
| ANDROIDS | P02 본문 | 질문 효과의 타 corpus 재현 | 이탈리아어·ASR·임상 라벨·신청 조건; 언어 변화와 프로토콜 변화 구분 |
| CMDC | P03/P05 본문 | 중국어 인터뷰에서 외부 검증 | 라벨 체계와 고정 질문, language-specific encoder, 배포 조건 |
| D-Vlog | P10/P15 초록만 확인 | 임상 인터뷰 밖으로 확장하는 후속 후보 | 라벨 신뢰성·화자 중복·원본 확보; PHQ-8 회귀 데이터로 가정하지 않기 |

추천은 먼저 E-DAIC의 접근 및 중복 여부를 확인하고, 더 강한 외부 검증에는 별도 corpus를 추가하는 순서다. 질문 통제 연구에는 질문·화자·timestamp가 필요하므로 이름만 보고 선택하지 않는다. 이번 작업에서 데이터 신청, 다운로드, 원자료 외부 전송은 수행하지 않았다.

## 8. 첫 구현의 실행 계획

다음은 향후 실행안이며 이번에 완료한 실험이 아니다.

| 단계 | 구현/확인 | 완료 기준 |
|---|---|---|
| 1. 데이터 계약 | participant manifest, 공식 split, label source, 제외 사유, timestamp 오류 | split overlap 0, 라벨 예외 기록, 모든 입력의 동일 cohort |
| 2. 최소 baseline | majority, train prevalence, P/E/P+E TF-IDF+LR, 단순 시간 통계 | 참가자별 예측·confusion matrix·설정 저장 |
| 3. 첫 논문 재현 | P04 CTD 단독 또는 P01 GCN | 논문과 다른 cohort·label·환경을 명시하고 차이를 설명 |
| 4. 후보 A 검증 | response-side timing, 질문·길이 통제, late fusion | 통제 전후 paired 차이와 표본 변화 보고 |
| 5. 신경망 확장 | frozen text/audio encoder 후 필요한 fine-tuning | 동일 cohort에서만 증가분 비교 |
| 6. 외부 검증 | 새 corpus에서 threshold 고정 평가 | in-domain과 transfer 분리, label·언어 차이 명시 |

공통 평가 규칙:

- 분할은 participant 먼저, segment는 그다음이다. 긴 인터뷰의 segment가 많다고 평가 표본 수가 늘어난 것은 아니다.
- TF-IDF vocabulary, scaler, imputer, 특징 선택, 증강은 train 또는 CV 내부에서만 학습한다.
- 공식 dev를 반복적으로 쓰는 정확 재현과 새로운 방법의 독립 검증을 구분한다. 새 방법의 주요 선택은 train 내부 CV로 줄이고 test는 마지막에 한 번 평가한다.
- threshold·fusion weights·seed 목록·제외 규칙은 test 전에 고정한다. seed 평균과 참가자 bootstrap은 서로 다른 불확실성을 나타낸다.
- 분류는 macro-F1, positive-class F1, sensitivity/specificity, balanced accuracy, AUROC와 혼동행렬을 함께 보고한다. 회귀는 원래 PHQ-8 척도의 MAE/RMSE를 사용한다.
- Week1의 409 binary 예외는 released-label 조건과 score-derived 조건을 구분한다. test 성능을 본 뒤 라벨을 고치거나 세션을 제외하지 않는다.
- baseline 비교는 입력이 있는 공통 cohort로 맞춘다. 원자료 전체와 정제 subset 결과는 별도 표로 둔다.
- 기존 COVAREP의 74번째 열 의미가 미해결인 상태에서는 이름을 추측해 해석하지 않는다. raw audio encoder나 의미가 확인된 특징으로 시작할 수 있다.

**읽기 순서:** P01 → P02 → P04 → P03 → P05. 먼저 평가 함정을 이해하고 CPU baseline을 재현한 다음, 필요성이 확인된 음성·멀티모달 모델을 추가하는 흐름이다. 주제는 A를 우선 검증하되, 질문 통제 후에도 의미 있는 이득이 있는지에 따라 B 또는 C로 좁혀가는 것이 적절하다.
