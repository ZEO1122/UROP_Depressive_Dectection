# API 없는 근거 충분성 실험의 선행연구 중복 검토

2026-09-27 · 실행 전 표적 문헌 검색. 체계적 문헌고찰·최초성 인증이 아니다. 모델 실행과 Notion 수정은 하지 않았다.

## 판단

**설계의 핵심 구성요소는 상당 부분 이미 연구됐다.** 특히 원본/O, 필수 근거 삭제/E, 무관 정보 삭제/R, 중복 근거 보존/S의 의미 구조는 TACL 2022의 SufficientFacts와 매우 가깝다. 지식 특징·불충분 판단·멀티모달 결합·사람의 근거 평가·판단 보류를 각각 새로운 기법으로 내세울 수 없다.

확인한 문헌에서는 DAIC-WOZ의 증상별 기간·빈도·기능 정보를 독립 주석하고, K/D와 비언어 관측을 분리하여 모든 조건을 동일하게 비교한 **동일한 전체 프로토콜**은 확인하지 못했다. 하지만 조합이 정확히 일치하지 않는다는 사실은 충분한 연구 기여나 신규성의 증명이 아니다.

현재 안의 적절한 위치는 **기존 근거 충분성 평가의 임상 면담 적용·검증 연구**다. UROP 파일럿으로서 가치는 있지만 새로운 알고리즘 논문으로 바로 명명할 근거는 약하다.

## 가장 가까운 논문

| 문헌·확인 범위 | 이미 한 일 | 우리 설계와의 차이·영향 |
|---|---|---|
| **Atanasova et al., Fact Checking with Insufficient Evidence**, TACL 2022. §4.2, pp.751–752, Table 3. [공식 논문](https://aclanthology.org/2022.tacl-1.43/) | 근거를 삭제하고 남은 정보가 충분한지 사람에게 주석시킨다. 충분성이 유지되는 이유로 무관 정보 삭제와 잔존 중복 근거를 구분한다. | **E/R/S의 핵심 의미 구조와 직접 중복.** 임상 면담·비언어 과제는 아니지만 삭제 조건 자체의 최초성 주장은 불가. 우리 길이·시간 매칭은 세부 통제이며 그것만으로 신규성 확정 불가. |
| **Lenz et al., Benchmarking large language models against practicing clinicians on psychopathological assessment**, npj Digital Medicine 2026. Methods/Results. [출판사](https://www.nature.com/articles/s41746-026-02852-7) | 세 모의 면담에서 AMDP 정의 제공 유무, absent/present/not assessable, 임상가와 모델의 오류 양상을 비교한다. | 지식 제공과 평가 불가능·과도한 보류는 이미 정신건강 평가에서 연구됨. 우리는 DAIC의 세부 정보 slot 및 통제된 삭제 조건. 원 연구의 모델/임상가 입력은 전사/시청각으로 다르다. |
| **Chen & Peng, HiMA-MDD**, arXiv:2608.21868v1, 2026 preprint. Method Layer 2–3, Experiments/Evaluation, Tables 1–4 및 부록. [본문](https://arxiv.org/html/2608.21868v1) | QA 맥락과 임상 rubric, 음성 단서, 문항 근거·충분성, 빠진 기간·빈도, 충돌을 구조화한다. 주 정량 평가는 PHQ 점수와 선별이다. | 단순한 증상별 근거/기간/빈도 기록은 기존 연구와 겹침. 확인한 평가에서는 우리의 독립 slot gold 및 O/E/R/S 전환 benchmark가 주평가로 제시되지 않음. |
| **Zhu et al., DepressionAgent**, arXiv:2608.13891v1, 2026 preprint. §V-A, Table VII, §V-B–C. [본문](https://arxiv.org/html/2608.13891v1) | D-Vlog에서 우울 관련 단어·문장을 가리고 분류를 평가한다. 이때 A/V는 유지한다. 임상가 5명이 30개 사례의 근거·추론·효용 등을 평가한다. | **멀티모달+masking+사람의 근거 평가도 이미 존재.** 해당 text-only masking은 모든 경로의 언어 근거 삭제와 다르다. 우리는 모델 출력을 보기 전에 만든 slot gold와 정보별 기대 전환을 목표로 한다. |
| **Ishikawa & Duke, A Multi-Probe Audit of Clinical-Interview Depression Detection Benchmarks**, arXiv:2605.23977, 2026 preprint. Probe D, §2.5·3.5, Table 7. [본문 PDF](https://arxiv.org/pdf/2605.23977) | 같은 E-DAIC 참가자의 증상 내용이 많은/적은 구간에서 text/audio 반응을 비교한다. 구간은 SRDS 기반 자동 정의다. | 같은 사람의 내용 조작·구간 비교와 modality 민감도 역시 기존 접근. 우리의 목표는 점수 반응보다 독립적으로 주석한 특정 정보의 확인 가능성. |
| **Afrasyab, Judge-dependent safety gains and model-specific helpfulness costs of evidence-sufficiency prompting in clinical LLMs**, arXiv:2607.18086, 2026 preprint. Methods/Perturbations, Outcomes, Clinician Review, Discussion. [서지](https://arxiv.org/abs/2607.18086), [PDF](https://arxiv.org/pdf/2607.18086) | 임상 맥락·검사 정보를 제거한 paired 자료로 근거 부족 시 과도한 단정과 답할 수 있을 때의 보류 비용을 평가한다. judge 차이와 임상가 검토도 다룬다. | **정보 삭제+과잉단정+불필요한 보류+사람 검토**라는 더 넓은 임상 평가 조합도 존재. DAIC 멀티모달 slot 판별은 다른 과제지만 기본 평가 철학은 신규하지 않음. 자동 judge 및 제한된 사람 검토의 한계를 논문도 명시한다. |

2026 arXiv 문헌은 이번에 확인한 공개본 기준 preprint로 표기했다. 모든 후속 출판 상태를 전수 조사한 것은 아니다. 정식 게재 여부와 내용의 중복 여부는 별도다.

## 기반 방법론과 지표의 출처

| 선행연구 | 현재 설계에 주는 의미 |
|---|---|
| **DeYoung et al., ERASER**, ACL 2020, §4.2. [논문](https://aclanthology.org/2020.acl-main.408/) | 근거 삭제 및 근거만 보존했을 때의 예측 변화로 설명 충실성을 평가한다. 이 논문의 sufficiency/comprehensiveness는 모델 반응 지표이며 임상 정보 충분성 gold와 동일하지 않다. |
| **Ribeiro et al., CheckList**, ACL 2020, §2.2. [논문](https://aclanthology.org/2020.acl-main.442/) | 무관 변화에 대한 invariance와 의미 변화에 대한 directional expectation이라는 행동 검증 틀이 존재한다. E의 전환과 R/S의 유지는 이 틀의 적용이다. |
| **Chapman et al., ConText**, BioNLP 2007, §3·5. [논문](https://aclanthology.org/W07-1011/) | 부정·과거/가정·경험 주체를 규칙으로 판단하는 전통적 임상 NLP 접근. K 특징에 직접 관련되므로 규칙-only baseline이 필요하다. 원 논문의 과제별 시간 규칙을 우리 지침에 무비판적으로 복사하지 않는다. |
| **Uzuner et al., 2010 i2b2/VA challenge**, JAMIA 2011. [논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC3168320/), [공식 주석 지침](https://www.i2b2.org/NLP/Relations/assets/Assertion%20Annotation%20Guideline.pdf) | present/absent/possible/conditional/hypothetical/other experiencer의 assertion 분류가 존재한다. possible은 미언급과 같지 않으며, assertion과 현재성은 분리해야 한다. |
| **Geifman & El-Yaniv, SelectiveNet**, ICML 2019, §2·5. [논문](https://proceedings.mlr.press/v97/geifman19a.html) | risk–coverage와 선택적 기권은 기존 방법. 모델 확신이 낮아서 보류하는 것과 문서에 정보가 없다는 정답은 다른 축이다. |

Macro-F1 및 이번에 정의한 조건부 오류율은 **표준 분류 평가의 과제별 적용**이다. 새로운 이름을 붙였다고 새 metric이 되는 것은 아니다. 삭제 전후 쌍 성공률도 행동검사 지표로 정의해 쓰되 최초 제안이라고 하지 않는다.

## 이전 자료에서 보완해야 할 해석

1. **독립 사람 평가 자체가 비어 있는 연구 공백이라는 표현은 사용하지 않는다.** DepressionAgent에도 임상가 평가가 있다. 우리가 제안하는 차이는 모델 출력의 사후 평정과 구별되는, 입력 근거에 대한 사전 slot annotation 및 통제된 변화의 정답이다.
2. **삭제·무관 삭제·중복 보존은 SufficientFacts를 명시적으로 인용한다.** 임상 특화는 새 annotation 문제를 해결하거나 유의미한 실패를 발견해야 기여가 된다.
3. **API를 쓰지 않는 것은 실행 조건이다.** 비용·재현성의 장점이 있지만 그 자체가 신규성은 아니다. LR 결과로 생성형 LLM의 환각이나 임상 reasoning을 입증하지 않는다.
4. **MentalBench의 기존 요약에 정정 필요가 있다.** 이번에 연 [v1 §4.1.1](https://arxiv.org/html/2602.12871v1)은 Type 2를 핵심 증상은 포함하지만 DSM 진단 개수 기준 아래인 불완전 profile로 설명한다. 이전 Notion의 ‘최소 진단 근거 유지’라는 단정과 맞지 않는다. 이 검토에서는 현재 원문을 우선한다. 다만 선택지 기반 합성 진단 과제라 우리 slot별 판단 보류 gold와 그대로 같지는 않다. 버전 비교 없이 변경 원인을 추정하지 않는다.

## 실행 전 권고

### 연구 질문을 더 좁혀 표현

추천 작업 제목: **우울 증상 면담에서 정보별 근거 충분성과 비언어 단서에 의한 과잉확정 평가: SufficientFacts 접근의 DAIC-WOZ 적용**.

연구 질문 후보: **기간·빈도·기능 근거가 사라졌을 때, 비언어 통계가 잔존 언어 근거의 부족을 잘못 대체하는가? 어떤 담화 유형과 임상 정보에서 이 오류가 발생하는가?**

이는 검증할 응용 질문이다. 오디오·시각이 없는 시간 사실을 직접 제공하지 않는 것은 과제 정의상 명확하므로, 단순히 ‘복원할 수 없다’를 보이는 데 머물면 약하다. 독립 주석으로 실제 오류 패턴을 측정하고, 근거가 충분한 조건의 성능을 유지하면서 오류를 줄이는지 비교해야 한다.

### 첫 단계는 재현 가능한 작은 annotation pilot

- Train 20명에서 두 증상 영역을 독립 주석하여 정보별 class 수, 유일 근거·중복 근거, 길이 대조 적격률 및 평가자 일치도를 확인한다.
- 근거가 없는 자연 원본 사례도 포함한다. 인위적 삭제만으로 학습·평가하면 삭제 흔적을 탐지하는 과제가 될 수 있다.
- 필수 근거를 지울 때 대명사·질문응답 연결이 무너지는지 사람이 검토한다. 의미 결측과 부자연스러운 문장 생성 효과를 분리한다.
- 규칙-only, TF-IDF+LR, K/D 추가, A/V 추가를 같은 gold에서 비교한다. 규칙의 출력을 gold로 재사용하지 않는다.
- 같은 사례의 A/V-only 판정과 임상 정보별 출처를 점검한다. 모델 간 성능 향상만으로 임상 효과나 거짓말 탐지 성능을 주장하지 않는다.
- 충분한 사례·일치도·비자명한 실패가 없다면 범위를 조정하거나 UROP 재현·적용 연구로 보고한다. 대규모 주석부터 시작하지 않는다.

## 검색 범위·한계

공식 학회·저널/PMC 및 arXiv를 중심으로 정신건강·임상 NLP·일반 NLP까지 확장했다. 핵심 표는 본문 관련 절에 기반한다. i2b2 논문은 접근 제한으로 검색 반환 본문과 공식 주석 지침을 대조했다.

대표 검색어:

- `depression clinical interview evidence sufficiency missing information masking abstention multimodal benchmark`
- `clinical NLP evidence removal counterfactual insufficient information symptom assessment`
- `depression symptom evidence missing information benchmark duration frequency functional impairment masking`
- `clinical evidence sufficiency benchmark`
- `DAIC-WOZ insufficient evidence`
- `depression SufficientFacts`
- `clinical interview evidence removal duration`
- `Fact Checking with Insufficient Evidence`, `ERASER`, `CheckList`, `ConText`, `SelectiveNet`

추가 확인 후보로 MIDAS와 AAAI의 *Unveiling the Landscape of Clinical Depression Assessment*도 살폈다. 이번 판단의 핵심 근거는 위 표의 직접 중복 문헌이다. MIDAS의 October 2026 권호 표기를 현재 온라인 게재일과 혼용하지 않았다. 접근에 실패한 medRxiv *Evidence-Graded Decision Authorization for Safe Clinical AI*는 검색 후보로만 남기고 본문 확인 문헌으로 세지 않았다.

이 검색은 선행연구 충돌을 확인하기 위한 표적 조사이며 모든 언어·DB·비색인 논문을 망라하지 않는다. ‘같은 논문을 못 찾음’을 ‘없음’으로 바꾸지 않는다.
