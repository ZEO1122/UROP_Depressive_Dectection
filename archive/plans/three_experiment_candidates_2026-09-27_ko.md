# API 없는 UROP 실험 후보 3개와 선행연구 대조

> 보존된 과거 계획이다. 실행 여부와 실제 조건은 [실험 목록](../../experiments/README.md)의 결과 기록을 우선한다.

2026-09-27 · 후보 비교안. 최종 주제 선정·모델 실행·주석 완료를 뜻하지 않는다. 기존 세 관심 방향과 LLM API 없는 조건을 유지한다.

## 결론과 선택 기준

**세 후보 모두 관련 선행연구가 있다.** ‘이 아이디어가 처음’이라고 제안하지 않는다. 기본 방법의 중복과 이번에 별도로 검증할 질문을 구분했다. 실험 단위는 각각 **질문의 의미 / 개인 기준선 / 선택한 근거의 검토량**으로 달라, 이전 정보 삭제 실험을 세 이름으로 나눈 것이 아니다.

| 후보 | 핵심 질문 | 새로 필요한 gold | 중복에 대한 판단 | 추천 용도 |
|---|---|---|---|---|
| A. 질문 의미와 응답의 관계 검증 | 모델이 질문을 해석하는가, 질문 패턴으로 답을 추정하는가? | QA의 증상 상태·시점, 사람 검증 변형쌍 | 증상 추출·질문 편향·행동검사 각각 기존 연구. 통합된 좁은 DAIC audit는 이번에 동일 프로토콜 미확인 | 가장 작은 첫 파일럿 |
| B. 개인 기준선 보정의 조건부 효과 | 평소 발화 대비 변화의 이득이 질문·길이·진행 순서 통제 후에도 남는가? | 보고 상태 및 기준선 질문 범주 | 초반 질문 기준선 정규화가 2011년부터 존재. 기본 아이디어의 중복 높음 | 음성·시각 분석을 우선할 때 |
| C. 제한된 검토량에서 근거 확보 | 적게 읽으면서 기간·빈도·기능의 근거와 반대 근거를 빠짐없이 찾을 수 있는가? | 전체 면담 내 정보 항목·근거 위치 | DAIC 요약과 budgeted selection 모두 기존 연구. 항목별 coverage·맥락 손실 평가가 적용 차이 후보 | Assessment 지원으로 확장할 때 |

현실적 순서는 **A의 작은 주석·평가 파일럿 → 결과와 주석 자원이 허용하면 C 확장**을 권한다. B는 연구 관심이 비언어 행동에 있을 때 선택할 별도 후보다. 신규성이 아니라 초기 구현 부담과 평가의 명확성을 기준으로 한 권고다.

## 공통 조건

- 최신 [재검사](../../docs/data/download_recheck_2026-09-27.md) 기준 정상 ZIP은 **188명: train 107 / dev 34 / test 47**, 440은 제외한다.
- 질문 전사가 없는 451·458·480, 사전 사례 노출 300을 기존 정책대로 제외하면 질문 맥락 기반 후보 상한은 **107/32/45 = 184명**이다. 실제 적격 QA·품질·주석 범위를 적용한 최종 N은 미정이다.
- 원본 영상 대신 제공된 시각 특징을 사용한다. 402는 유효 시각 구간별로 처리하고 모든 비교는 같은 관측 cohort를 맞춘다.
- API 없이 규칙·TF-IDF·BM25·선형 모델·기존 음성/시각 특징으로 시작한다. API 없음 자체를 신규 기여로 삼지 않는다.
- Train 20명으로 주석 지침과 적격률을 확인한 뒤 확장한다. 두 평가자의 독립 주석은 PHQ·모델 예측을 가리고 진행한다. 임상 전문가 미참여 시 연구자 주석의 기술적 평가로 표현한다.
- 참가자 단위 공식 split을 유지하고 vocabulary/scaler/규칙 조정/모델 선택은 test 밖에서 수행한다. 변형쌍은 원 참가자와 같은 split이다.
- 참가자 단위 paired bootstrap CI를 사용한다. QA 개수나 변형 수를 독립 참가자 수로 세지 않는다.
- 세 후보 모두 DAIC만으로 실제 거짓말·은폐 의도를 판별하지 못한다. 그 주제에는 관측된 보고와 숨겨진 사실을 구분한다는 간접 연결만 있다.

## A. 질문을 이해하는 모델과 질문 패턴을 이용하는 모델 구분

### 질문과 가설

**같은 짧은 답변이라도 질문의 시점·경험 주체·긍정/부정 구조가 달라지면 증상 해석은 달라진다. 모델이 그 관계를 따라가는가?**

예를 들어 가상의 ‘Yes’라는 답변은 ‘최근 잠들기 어려운가?’와 ‘작년에 잠들기 어려웠는가?’에 붙을 때 의미가 다르다. 후자는 과거 경험의 근거이며 현재 증상의 지지·부재를 확정하지 않는다. 이 예는 실제 참가자의 상태를 바꿨다는 주장이 아니라 언어 해석용 합성 검사다.

H-A: 질문·응답 관계를 명시한 규칙 특징은 단순 단어 통계보다 의미 보존 변화에는 안정적이고 의미 변경에는 올바르게 반응한다. 질문-only와 비슷하게 동작하면 질문 형식 의존을 의심하되 그것만으로 임상적 편향을 확정하지 않는다.

### 최소 실험

1. 수면·흥미 저하에서 짧은 응답과 자기완결적인 응답을 모두 뽑아 증상 보고 상태 및 현재/과거·경험 주체를 주석한다.
2. 사람 검토를 거쳐 네 입력 뷰를 만든다: 원래 QA / 의미 보존 질문 바꿔쓰기 / 시점·경험 주체·극성을 바꾼 질문 / 무관 질문을 붙여 원래 해석이 성립하지 않는 경우.
3. 문법·자연스러움이 유지되는 변형만 쓴다. 응답 자체에 충분한 증상 정보가 있으면 무관 질문으로 바꿔도 unknown을 강요하지 않는다. 각 뷰의 정답은 실제 주어진 근거로 정한다.
4. 응답-only, 질문-only, 단순 Q+A TF-IDF/LR, 규칙-only, Q/A 관계 특징+LR를 비교한다. LR에는 질문 특성×응답 극성의 상호작용 특징을 명시해 단순 bag-of-words의 표현 제약만 공격하는 비교를 피한다.
5. 자연 원본 성능과 합성 변형 성능을 분리한다. 질문 유형마다 Yes/No/모호 응답과 상태 분포를 점검해 질문 종류 자체가 정답이 되지 않게 한다. 변형 문구·template family의 학습/시험 중복도 관리한다.

### 평가

- 자연 원본의 보고 상태 macro-F1 및 현재/과거·경험 주체별 성능.
- 의미 보존쌍에서 **양쪽 모두 정답**인 비율.
- 의미 변경쌍에서 **원본과 변경본을 모두 정답 방향으로 판정**한 비율.
- 과거·타인 경험을 현재 본인 증상으로 오인한 비율.
- 짧은 답변/자기완결 응답별 결과와 질문-only 대조. PHQ 예측 성능은 주평가하지 않는다.

### 선행연구 확인

| 문헌 | 확인한 범위와 중복 |
|---|---|
| [Du et al., Extracting Symptoms and their Status from Clinical Conversations, ACL 2019](https://aclanthology.org/P19-1087/) | 본문 §3–4: 임상 대화 약 3천 건에서 증상과 experienced/not experienced/other를 주석하고 여러 turn을 함께 해석한다. 대화에서 증상 상태를 추출하는 과제 자체는 이미 있다. |
| [Burdisso et al., DAIC-WOZ: On the Validity of Using the Therapist’s prompts…, ClinicalNLP 2024](https://aclanthology.org/2024.clinicalnlp-1.8/) | 참가자/질문 입력과 질문 선택 shortcut을 분석한다. ‘질문이 편향을 만든다’는 발견을 재발견한다고 주장할 수 없다. |
| [Mitigating Interviewer Bias in Multimodal Depression Detection…, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.650/) | 본문 D-CoPE·adversarial branch: 질문 맥락을 쓰면서 질문 기능 예측을 통한 adversarial 학습으로 편향을 줄인다. 질문 유익성/편향의 양면성도 이미 다룬다. |
| [Ribeiro et al., CheckList, ACL 2020](https://aclanthology.org/2020.acl-main.442/) | 의미 보존 invariance 및 기대 방향 변화 검증. 질문 바꿔쓰기·의미 변경이라는 평가 틀 자체는 기존 접근이다. |
| [Candidate Generation and Definition-Guided Verification…, arXiv:2609.01833v1, 2026 preprint](https://arxiv.org/html/2609.01833v1) | 문장·주변 문맥·증상 정의에 기반한 증상 유무 검증. 문맥+정의의 단순 조합을 새로운 방법으로 쓰기 어렵다. |

**판정:** 기반 아이디어는 있음. 이번 조사에서는 DAIC에서 질문 의미 변화에 따른 증상 assertion의 기대 전환을 독립 gold로 평가하는 동일한 전체 실험을 확인하지 못했다. 이는 신규성 확정이 아니다. 차이 후보는 기존 PHQ shortcut 분석을 **QA 의미 해석 오류의 종류별 검사**로 바꾸는 데 있다.

## B. 개인의 평소 발화 대비 비언어 변화가 실제로 유용한가

### 질문과 가설

**원래 목소리가 작고 표정 변화가 적은 사람과, 특정 질문에서만 행동이 바뀐 사람을 개인 기준선 보정이 더 잘 구분하는가? 그 이득은 질문 종류·길이·면담 순서를 통제해도 남는가?**

초반 rapport 구간을 **개인 참조 구간**으로 삼는다. 이를 건강하거나 정서적으로 중립인 상태라고 가정하지 않는다. 주 과제는 비언어 특징과 명시적 증상 보고의 통계적 대응이며 숨겨진 실제 증상·거짓말이 아니다.

### 최소 실험

1. PHQ와 A/V를 가린 주석자가 수면·흥미 저하 QA를 명시적 지지 / 명시적 부인 / 모호로 주석한다.
2. Train에서 고정한 일상·배경 질문 범주로 참가자별 참조 구간을 정하고, 최소 유효 음성·얼굴 길이를 고정한다. 부족한 참조 구간은 결측으로 관리한다.
3. 각 QA의 F0·RMS·gap·AU·pose·gaze 요약 `x`와 개인 참조 요약 `b`, 차이 `x-b`를 계산한다.
4. 같은 QA에서 다음을 비교한다: 질문 종류·답변 길이·경과 시간만 / 여기에 raw 특징 추가 / delta 특징 추가 / raw와 baseline을 모두 추가. 질문·답변 텍스트 특징을 포함한 민감도 비교로 어휘와 보고 내용의 혼동도 점검한다.
5. 같은 질문 범주 안의 결과를 별도로 보고하고 가능하면 후반 비증상 질문으로 초반/후반 효과를 점검한다. 개인 참조를 빼는 것만으로 질문 의미·피로·시간 효과가 제거됐다고 가정하지 않는다.

선형 모델에서 `x-b`는 raw와 baseline의 계수에 반대 부호 제약을 주는 것과 같다. 따라서 `raw+baseline` 비교를 생략하면 delta 표현이 독창적이거나 더 풍부한 정보를 만든 것처럼 오해할 수 있다.

### 평가

- 독립 주석한 보고 상태 macro-F1, class별 recall, log loss.
- raw 대비 delta의 paired 성능 차이와 질문 범주별 결과.
- 효과의 개인 간/개인 내 분산 및 길이·진행 순서 통제 후 차이.
- 혼합모형은 보조 연관 분석으로만 사용하고 시험 참가자의 정답으로 random effect를 추정하지 않는다.

### 선행연구 확인

| 문헌 | 확인한 범위와 중복 |
|---|---|
| [Sanchez et al., Using Prosodic and Spectral Features in Detecting Depression in Elderly Males, Interspeech 2011](https://www.sri.com/wp-content/uploads/2021/12/using_prosodic_and_spectral_features_i.pdf) | §3.1, p.3002: 첫 다섯 단순 질문을 기준으로 F0 mean subtraction/z-normalization과 energy 보정. **초반 질문 개인 기준선이라는 발상에 직접 선행연구가 있다.** |
| [Cummins et al., An Investigation of Depressed Speech Detection: Features and Normalization, Interspeech 2011](https://www.isca-archive.org/interspeech_2011/cummins11_interspeech.pdf) | §2.3–3.3: 화자별 정규화와 독립 평가. 정규화가 우울 관련 정보도 지울 수 있으므로 개선을 전제하지 않는다. |
| [Shan et al., What reveals about depression level?…, Information & Management 2020](https://www.sciencedirect.com/science/article/pii/S0378720620302871) | 공개 초록·방법 발췌: 질문별 멀티모달 특징과 9개 질문 의미 범주. 전체 유료 본문 정독으로 표현하지 않는다. |
| [Ravi et al., A Step Towards Preserving Speakers’ Identity…, Interspeech 2022](https://www.isca-archive.org/interspeech_2022/ravi22_interspeech.pdf) | §2–3: adversarial 학습으로 화자 정보와 우울 정보를 분리. 목표는 관련되지만 단순 기준선 차감과 동일 기법은 아니다. |
| [Hidalgo Julia et al., Identifying Vocal and Facial Biomarkers of Depression in Large-Scale Remote Recordings…, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/hidalgojulia25_interspeech.pdf) | §2.3: 반복 기록에서 개인 평균과 편차를 분리하는 mixed-effects 분석. DAIC 단일 면담 내 변화와는 다르지만 개인 중심화 자체는 기존 연구다. |

**판정:** 기본 아이디어의 직접 중복이 높다. 새로운 정규화 기법으로는 추천하지 않는다. ‘보정의 효과가 질문·길이·순서를 통제하면 남는가?’라는 **재현·반증 연구**로는 가치가 있다. 음성·시각 전처리와 적격 baseline의 확보가 A보다 어렵다.

## C. 짧게 읽어도 필요한 임상 근거를 보존하는가

### 질문과 가설

**전체 면담의 20%만 검토할 때, 관련도 높은 구간만 고르는 방법보다 기간·빈도·기능 및 반대 근거를 고르게 보존하는 선택법이 유리한가?**

예컨대 수면 불편을 반복하는 다섯 구간보다 수면 호소·시작 시점·일상 영향이 각각 담긴 세 QA 묶음이 더 유용할 수 있다. 반대로 부인이나 과거 시점을 제외하면 선택된 근거의 의미가 왜곡될 수 있다.

이는 **이미 끝난 면담의 오프라인 근거 선택**이다. 알고리즘은 전체 전사를 검색할 수 있고, budget은 사람에게 보여주는 선택 분량이다. 환자에게 새 질문을 했을 때의 답을 예측하거나, 면담 시간을 줄이는 실험이 아니다.

### 최소 실험

1. 두 증상에 대해 면담 전체에 실제로 있는 증상 보고·기간·빈도·기능·시점·부인/상충 정보를 사람 두 명이 주석한다. 같은 정보의 반복 근거는 하나의 정보 항목으로 묶는다.
2. 질문과 응답, 필요한 지시어 맥락을 포함한 QA 묶음을 선택 단위로 정한다.
3. 원래 순서 / 무작위 / BM25 관련도 / BM25+MMR(중복 억제) / 정보 항목별 예상 추가 확보량을 고려한 선택을 비교한다.
4. 마지막 방법의 항목 예측은 train에서 학습한 작은 분류기 또는 고정 규칙으로 계산한다. test gold를 보고 다음 QA를 선택하는 것은 금지한다. gold 선택은 비실행적 상한선으로만 별도 표시한다.
5. 전체 단어 수의 10·20·30·50% budget에서 선택한다. 필요한 질문·주변 맥락도 비용에 포함한다. QA 개수만 쓰면 긴 QA가 유리하므로 단어 수를 주 budget으로 둔다.

### 평가

- **정보 항목 coverage@budget:** 선택된 근거로 올바르게 확인 가능한 정보 항목 / 전체 면담에서 확인 가능한 정보 항목.
- **Coverage–Budget AUC:** 여러 budget에서의 확보율을 고정 구간에 적분한 성능. 축·적분 범위를 사전에 고정한다.
- 80% coverage까지 필요한 단어 수, 미도달 비율. 임계치는 pilot 후 test 전에 고정한다.
- 부인·시점·경험 주체·상충 근거 손실률, 근거 precision.
- 정보가 원래 없는 slot은 coverage 분모에서 제외하며, 관련 정보가 전혀 없는 면담은 별도 결과로 보고한다. 비언급을 없다는 사실로 바꾸지 않는다.

순수 추출기는 환자에 대한 단정을 생성하지 않으므로 hallucination/과잉확정 지표를 붙이지 않는다. 항목 판정기를 추가한다면 별도 단계에서 평가한다. 단어 수 감소를 실제 임상가 시간 절약이라고 부르려면 추가 사용자 연구가 필요하다.

### 선행연구 확인

| 문헌 | 확인한 범위와 중복 |
|---|---|
| [Knowledge-Infused Abstractive Summarization of Clinical Diagnostic Interviews, JMIR Mental Health 2021](https://mental.jmir.org/2021/5/e20865/) | **DAIC-WOZ 자체**에서 PHQ-9 어휘·ConceptNet·ILP 등을 활용한 요약과 전문가 평가. 생성형 LLM API 없이 임상 면담 요약을 하는 아이디어도 이미 있다. |
| [Qureshi et al., Budget-Aware Routing for Long Clinical Text, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.2114/) | 본문 §3–4: 길이 예산 아래 relevance·coverage·diversity와 여러 선택법을 비교한다. 추출 직접평가와 LLM 생성평가를 구분한다. **budget-aware clinical selection 자체는 신규하지 않다.** |
| [Lin & Bilmes, Multi-document Summarization via Budgeted Maximization of Submodular Functions, NAACL 2010](https://aclanthology.org/N10-1134/) | 제한된 길이의 coverage·중복 억제 최적화. greedy/MMR/coverage라는 알고리즘 개념의 기반 연구다. |
| [CAD-MDD, J Clin Psychiatry 2013](https://pubmed.ncbi.nlm.nih.gov/23945443/) | 적은 질문으로 적응형 우울증 선별. 새 질문 선택도 오래된 과제이나 후보 C의 기록 검토와는 구분한다. |
| [MAGI, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.1278/) | MINI 분기에 기반한 능동 면담. C는 이런 온라인 면담 시스템의 재현이나 신규 버전이라고 주장하지 않는다. |

**판정:** 핵심 방법의 중복은 높지만, 임상 면담의 **항목별 실제 근거 확보와 의미 왜곡을 직접 평가하는 annotation·분석**으로 좁힐 수 있다. 새로운 selection 알고리즘보다 Assessment 지원에 가까운 산출물을 만들기 좋다. 다만 coverage 분모를 위해 전체 면담을 읽고 주석해야 하므로 A보다 주석 부담이 크다.

## 세 관심 주제와의 연결

| 관심 방향 | A | B | C |
|---|---|---|---|
| 은폐·거짓 반응 | 모호/과거/부인을 실제 은폐로 오인하지 않는 전제 작업 | 비언어 차이를 숨김의 증거로 읽을 수 있는지에 앞선 연관성·교란 점검 | 기록에 없는 사실을 찾았다고 주장하지 않는 정보 검토 |
| 임상 지식·면담 맥락 | **주 연결:** 질문-응답 의미와 시점·경험 주체 | 질문 유형 통제가 행동 해석에 미치는 영향 | **주 연결:** 필요한 임상 정보와 근거 맥락 보존 |
| 멀티모달 충분성·과잉추론 | 첫 단계는 텍스트. 이후 잘못된 A/V 확정과 연결 가능 | **주 연결:** 언어 보고와 비언어 관측의 대응 한계 | 첫 단계는 텍스트. A/V는 별도 후속으로 한정 |

세 주제를 모든 후보에 같은 비중으로 억지로 넣지 않았다. 실제 은폐 검증이 최우선이라면 의도와 사실의 독립 reference를 새로 수집하는 별도 설계가 필요하다.

## 검색 범위와 검토의 한계

주요 조회: ACL Anthology·ISCA 원문, 출판사/기관 원문, PMC/PubMed, arXiv. 핵심 방법은 본문 관련 절을 확인했으며 유료/접근 제한 자료는 표에 확인 수준을 구분했다. JMIR 일부 직접 요청은 로봇 확인 화면이 반환되어 병렬 조사에서 확인한 논문 본문·검색 제공 내용과 출처를 종합했다. 체계적 문헌고찰이나 모든 후보의 최초성 인증은 아니다.

대표 검색어:

- `depression symptom evidence classification context window question answer clinical interview negation temporal`
- `DAIC-WOZ question swapping`, `depression question shuffling interview`
- `clinical conversations question negation symptom status`
- `DAIC WOZ within subject neutral baseline question matched acoustic visual normalization depression`
- `depression first questions baseline normalization`, `speaker normalization depression`
- `depression interview adaptive question selection information gain symptom assessment`
- `DAIC clinical diagnostic interview summarization`, `budgeted clinical evidence retrieval coverage`

‘찾지 못한 세부 조합’과 ‘존재하지 않는 아이디어’를 구분한다. 제안한 후속 차이는 이번 문헌 대조에서 도출한 연구자의 판단이며 선행연구 저자가 인증한 연구 공백이 아니다. 이번에는 후보 문서만 추가했으며 실험·데이터 가공·Notion 변경은 수행하지 않았다.
