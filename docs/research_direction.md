# Assessment 중심으로 수정한 연구 방향

2026-09-14 · 사용자 피드백 반영. 기존 문헌 목록은 배경 자료로 유지하되, 이전의 CTD 기반 선별 예측 추천을 이 문서로 대체한다.

## 연구 목표의 수정

목표는 임상 면담에서 우울증 관련 증상과 평가에 필요한 맥락을 탐지·구조화하고, 판단에 부족한 정보를 식별하여 임상의의 Assessment를 지원하는 것이다. PHQ-8 binary/score 예측을 주된 성공 기준으로 삼지 않는다.

Screening과 Assessment를 도구만으로 구분하지는 않는다. 설문도 Assessment에서 증상 정도와 경과를 파악하는 데 사용된다. APA의 DSM-5-TR assessment resources에는 PHQ-9도 포함되며, 단독 진단 근거로 사용하지 않도록 설명한다. 따라서 이번 연구의 차이는 설문을 배제하는 것이 아니라, 임상 면담의 증거와 맥락을 다루는 업무를 목표로 삼는 데 있다. [APA assessment measures](https://www.psychiatry.org/psychiatrists/practice/dsm/educational-resources/assessment-measures)

NICE NG222의 1.2.6–1.2.7은 증상 개수뿐 아니라 심각도, 기간·경과, 과거력, 기능 손상 등을 함께 평가하도록 권고한다. 이는 증상 단어 출현 여부만으로 Assessment를 구성하기 어려운 이유다. [NICE 권고](https://www.nice.org.uk/guidance/ng222/chapter/Recommendations)

## DSM과 심리학 지식의 역할

DSM 기반 진단 기준, 임상 평가 지침, 심리학 연구에서 보고된 연관성은 서로 다른 종류의 근거다. DSM은 평가할 개념과 조건을 정하는 기준틀로, 임상 지침은 확인할 영역과 업무 흐름으로, 심리학의 관찰·실험 연구는 검증할 가설과 특징의 근거로 사용한다. 낮은 음성 에너지나 긴 침묵과 우울 증상의 연관성을 개인의 진단 규칙으로 바꾸지 않는다.

이번 조사에서 DSM-5-TR 유료 매뉴얼 전체를 직접 검토하지는 않았다. APA 공식 자료와 임상 지침, DSM-5 기반 선행연구를 확인했다. 구현용 기준표는 접근 가능한 정식 판본을 기준으로 임상 전문가와 검토하고 판본·출처를 기록해야 한다. DSM-5 기반 MentalKG를 그대로 DSM-5-TR 기준표라고 이름 붙이지 않는다. [DSM-5-TR 안내](https://www.psychiatry.org/psychiatrists/practice/dsm/about-dsm)

제안하는 평가 정보 구조는 다음과 같다. 이는 공식 DSM 척도의 재현이 아니라 연구용 schema 초안이다.

| 영역 | 저장할 정보 | 잘못된 단순화 |
|---|---|---|
| 증상 근거 | 참가자 발화 span, 화자, 부정 여부, 경험 주체 | 면접자의 질문을 참가자 증상으로 계산 |
| 시간 | 현재/과거, 시작 시점, 지속 기간, 빈도 | 과거 episode를 현재 증상으로 판정 |
| 개인 내 변화 | 평소 상태와의 변화 | 개인차를 증상으로 확정 |
| 기능·고통 | 일·학업·관계·일상에 미친 영향, 주관적 고통 | 직장이 있다는 사실만으로 증상 배제 |
| 다른 설명·감별 정보 | 신체 상태, 약물·물질, 수면 환경, 기분 상승 과거력 등 확인된 정보 | 가능한 설명을 확정 원인으로 간주 |
| 근거 상태 | 지지 / 명시적 부인 / 불충분 / 상충 | 미언급을 없음으로 처리 |

우선은 흥미 저하, 수면 관련 호소, 에너지 저하처럼 맥락에 따라 해석이 달라지는 2–3개 영역으로 annotation과 오류 분석을 시작한다. 이 작은 범위로 전체 MDD 진단을 완료했다고 주장하지 않는다.

## ‘외부 맥락’의 조작적 정의

**주 분석 대상 발화만으로는 주어지지 않지만, 해당 발화의 증상 의미나 평가의 충분성을 판단하는 데 필요한 정보**로 정의한다. 출처에 따라 세 층으로 나눈다.

| 구분 | 예 | 확보 방식 | LLM의 역할 |
|---|---|---|---|
| K: 외부 임상 지식 | 증상 정의, 시간·기능 조건, 감별 시 확인할 사항 | 판본이 고정된 문헌·검토된 기준표 | 검색된 지식을 현재 근거와 연결 |
| D: 면담 내 담화 맥락 | 앞선 질문, 다른 시점의 답, 부정·대명사·과거력 | 같은 transcript의 관측 발화 | 국소 발화와 전체 면담의 관계 복원 |
| P: 별도 환자 맥락 | 실제 제공된 병력, 근무 형태, 약물, 종단 기록 | 동의·접근 권한이 있는 기록 또는 추가 면담 | 제공된 사실을 통합하고 출처 표시 |

D는 발화 기준으로는 외부지만 데이터셋 밖 지식은 아니다. 논문에서 ‘external context’라고 묶어 부를 경우 K/D/P별 효과를 따로 보고해야 한다. LLM의 pretrained knowledge는 검증된 환자 정보가 아니며, 파라미터에 든 지식만 사용하는 조건과 문헌을 제공하는 조건도 구분한다.

**LLM이 보완할 수 있는 것은 지식과 해석이다. 관측되지 않은 환자 과거력이나 생활환경을 알아내는 것은 아니다.** P가 없으면 unknown으로 유지하고 필요한 확인 사항을 제안한다. 문화·직업·성별의 집단적 경향으로 개인의 사실을 채워 넣지 않는다.

연구용 가상 예시: “요즘 잠을 잘 못 자요.”만 주어지면 수면 관련 호소의 근거는 있지만 기간·빈도·평소 대비 변화·기능 영향은 불충분할 수 있다. “야간 근무를 시작했다”가 실제로 추가되면 수면 일정 변화도 함께 고려한다. 이를 근거로 우울증을 자동 배제하지도, 야간 근무가 원인이라고 확정하지도 않는다. 제공되지 않은 야간 근무 사실을 모델이 생성하면 오류다.

## 새로 검토한 핵심 선행연구

OpenResearch semantic retrieval에서 15편을 확인하고, 관련 후보 6편을 추린 뒤 아래 4편의 추출 본문에서 방법·결과·한계의 관련 부분을 검토했다. 나머지 우선 후보는 Dep-LLM (2606.10796), Interpretable Symptom Vectors (2609.01832)이며 이번에는 초록만 확인했다.

| 문헌 | 확인한 내용 | 우리 주제와 관계·한계 |
|---|---|---|
| [MentalBench](https://www.alphaxiv.org/abs/2602.12871), 2026 preprint | DSM-5 기반 MentalKG, 23개 질환, 24,750개 합성 사례. 정보 부족·감별 모호성을 평가 | DSM 지식 주입과 정보 부족 평가 자체는 이미 연구됨. 합성 사례 정확도를 실제 임상 유용성으로 등치하면 안 됨 |
| [Learning Evidence of Depression Symptoms via Prompt Induction](https://www.alphaxiv.org/abs/2604.24376), 2026, 본문에 SIGIR 서지 기재 | BDI-II 21개 증상에 대한 문장별 evidence classification. 학습 예시에서 guideline을 유도하고 다른 LLM에 재사용 | 증상 정의만 제공하는 baseline보다 구체적인 evidence 기준을 비교할 수 있음. BDI-II·소셜 텍스트 결과를 DSM 면담 평가와 동일시할 수 없음 |
| [The MADRS Pipeline](https://www.alphaxiv.org/abs/2607.28190), 2026 preprint | 실제 임상시험 면담의 전사·분할·MADRS item 평가·rating quality 검토. 본문은 총점 Spearman 0.867 보고 | Assessment 업무와 직접 연결됨. 다단계 시스템이며 LLM 하나가 모든 점수를 내는 구조로 단순화하지 않기. 비공개 데이터로 정확 재현 제약. 상관계수는 진단 정확도가 아님 |
| [When Symptoms Are Not Enough](https://www.alphaxiv.org/abs/2605.23148), 2026 preprint | SCID reference가 있는 555개 면담으로 LLM의 증상·기능·보호 맥락 가중을 분석 | 저자 과제는 screening으로 명명됨. 기능·보호 맥락이 무조건 유익하지 않음을 점검하는 근거. 주요 false-negative 분석은 anxiety/PTSD이므로 같은 효과가 우울증에 입증됐다고 확대하지 않기 |

[MentalBench 코드](https://github.com/HoyunS/MentalBench)와 [Symptom Induction 코드](https://github.com/IRLab-UDC/depression-prompt-induction)의 논문 대응을 확인했다. 실행·가중치·데이터 라이선스의 전체 감사는 하지 않았다. MADRS의 비공개 데이터는 확보하지 않았다.

## 권장 주제

**임상 지식과 면담 맥락을 활용한 LLM의 우울 증상 근거 판별 및 평가 정보 부족 탐지**

영문 작업 제목: *Clinical-Knowledge-Grounded LLMs for Contextual Depression Symptom Assessment and Missing-Information Detection*.

검증할 중심 가설은 “임상 기준을 명시하고 관측된 담화 맥락을 연결하면, 단일 발화 또는 일반 LLM보다 증상 근거 판별과 정보 부족 식별이 좋아지며 근거 없는 단정이 줄어드는가?”다. RAG를 썼다는 사실 자체를 기여로 삼지 않는다. 짧은 기준표면 전체 지식을 넣는 baseline이 충분할 수 있고 retrieval은 오히려 관련 정보를 누락할 수도 있다.

출력은 증상별 근거 span, 시간·기능 맥락, 지지/부인/불충분/상충 상태, 참조 지식, 확인해야 할 정보로 구성한다. 최종 확진 label을 강제로 생성하지 않는다. 이 구조에서 실제 탐지 대상은 **우울 증상에 대한 임상적 근거와 평가 정보의 부족**이다.

후속 연구로는 가장 필요한 다음 질문의 선택, 임상가의 증상 심각도 평가 보조가 가능하다. 이들은 별도의 gold annotation과 데이터가 필요하므로 첫 단계에 함께 묶지 않는다. 특히 추가 질문에 대한 답은 기존 데이터에서 관측되지 않으면 평가할 수 없다.

## 실험 설계

같은 LLM·같은 출력 schema를 사용하고 다음 요인을 분리한다.

| 조건 | 입력 | 목적 |
|---|---|---|
| B0 | 국소 발화만 | 암묵적 pretrained knowledge 기준선 |
| B1 | 국소 발화 + 고정 임상 기준표 K | 명시적 지식의 추가 가치 |
| B2 | 발화 + 관측 면담 맥락 D | 담화 복원의 가치 |
| B3 | 발화 + K + D | 지식·맥락의 상호작용 |
| B4 | B3 + 근거 상태·출처 강제 및 unknown 규칙 | 근거 없는 환자 정보·단정을 줄이는 효과 |

K 제공은 full knowledge와 retrieval을 분리해 비교한다. P는 실제 보조 기록이 확보된 데이터에서만 추가한다. 증상명+정의의 단순 prompting, Symptom Induction 계열을 함께 baseline으로 고려한다. 전문가가 제공하는 평가는 모델 조건을 가린 상태로 수행한다.

주요 평가는 (1) 증상 근거 판별 macro-F1, (2) evidence span precision/recall, (3) 시간·부정·경험 주체 해석 정확도, (4) 불충분 상태 탐지 precision/recall, (5) 근거 없는 주장 비율이다. 질문 추천은 전문가의 필요성·중복성 평가로 별도 검증한다. 실제 임상 workflow 효용은 추후 임상의의 검토 시간·오류·누락 변화 등을 비교해야 주장할 수 있다.

정보 삭제 실험에서는 제거한 span에만 있던 기간·기능 정보가 출력에서 unknown으로 바뀌는지 본다. 다른 발화에 같은 정보가 남아 있으면 ‘완전한 정보 제거’로 처리하지 않는다. 무관한 맥락 추가와 의미를 보존한 표현 변경도 대조한다. 이는 Assessment 정보 처리의 robustness 시험이며 실제 환자 경과를 조작하는 실험이 아니다.

split은 참가자 단위, 합성 데이터는 원래 case family/template 단위로 분리한다. test 예시·정답·전문가 test annotation이 검색 지식이나 prompt induction에 들어가지 않게 한다. 같은 LLM이 만든 라벨로 같은 모델을 평가하는 것을 주 근거로 쓰지 않는다.

## DAIC-WOZ의 역할과 현실적인 시작 순서

현재 확인한 DAIC-WOZ label은 PHQ-8 기반이다. DSM 진단, 충분한 감별 과거력, criterion별 evidence gold가 모두 있다고 가정할 수 없다. 따라서 DAIC-WOZ는 **실제 면담 언어에서 evidence와 missingness를 시험할 초기 corpus**로 사용한다. PHQ-8은 보조적인 연관성 확인에만 사용하고 Assessment gold를 대체하지 않는다.

1. 연구 target을 증상 근거·기간·기능·불충분 상태로 고정한다. 2–3개 증상 영역의 schema와 예시를 먼저 작성한다.
2. 임상 전문가 검토로 annotation 지침을 고정한다. 제안 규모는 pilot 20–30개 면담이며 검정력 계산을 완료한 최종 표본 수가 아니다. 두 평가자의 독립 annotation과 불일치 조정을 계획한다.
3. 공개 MentalBench로 임상 조건·모호성 평가 코드를 익힌다. DAIC-WOZ pilot annotation으로 실제 담화 성능을 별도로 검증한다. 합성 benchmark 결과를 임상 성능이라고 명명하지 않는다.
4. B0–B4를 비교하고 어떤 종류의 맥락이 오류를 줄이거나 늘리는지 분석한다.
5. 진단 판별로 확장하려면 독립된 임상 진단 reference가, 심각도 평가로 확장하려면 clinician-rated symptom label이 있는 데이터를 확보한다.

임상 협력이 없다면 기술적 prototype과 합성 자료 평가는 가능하지만 임상적으로 타당한 Assessment 시스템이라는 주장은 제한된다. 이 경우 LLM judge만으로 임상 gold를 대체하지 않는다.

이전 CTD/음성 특징은 이후 보조 관측으로 남길 수 있다. 예를 들어 발화 속도와 침묵을 evidence 카드에 제공할 수 있지만, 이것을 정신운동성 변화나 집중력 저하의 확정 근거로 바로 변환하지 않는다. 첫 구현의 우선순위는 이제 CTD 성능 재현보다 **annotation schema, MentalBench·증상 evidence baseline, K/D 분리 실험**이다.

## 검색 이력과 현재 상태

OpenResearch 0.2.1, `--no-telemetry`, `discover embedding`, 상한 2026-09-14, 기본 priority, 하한 없음. 검색문: `Large language models supporting clinical depression assessment DSM symptom evidence duration functional impairment differential diagnosis missing context`. 15건 반환, follow-up 검색 없음. 난도 6/10, 최초 결과로 방향 검토에 충분해 추가 round를 사용하지 않았다.

원 응답: `search/assessment_embedding.json`. 위 네 논문에 `orx paper <id>`를 실행한 응답: `papers/assessment_<id>.txt`. 논문 관련 절을 검토했으며 모든 부록이나 임상 기준의 정확성을 독립 검증한 것은 아니다. APA 공식 웹페이지를 확인했고 NICE는 검색에 반환된 권고문을 사용했다(페이지 직접 열기는 403). 연구 주제의 신규성은 예비 판단이다. 모델 구현·학습·환자 평가·원자료 외부 전송은 수행하지 않았다.
