# HiQuE 논문·공개 코드·실행 코드의 성능 격차 진단

이 문서의 `experiments/*.py` 파일명과 줄 번호는 [2026-10-04 이전 소스 스냅샷](../archive/snapshots/pre-reorg-2026-10-04/experiments/) 기준이다. 진단 수치·토크나이저 오류 정정은 당시 기록으로 보존하며 최신 결과는 [V3](../experiments/hique/v3_input_study/README.md)를 따른다.

**같은 날 V3 실행 준비 중 추가로 확인한 직접 오류:** `RobertaTokenizerFast.from_pretrained`를 명시한 기존 로딩 경로가 설치된 Transformers5.9에서 BPE merges 0개·pre-tokenizer 없음으로 구성됐다. 예를 들어 `Hello world`를 정상 단어 토큰 대신 글자로 나눴다. AutoTokenizer는 같은 로컬 파일에서 merges50,000개와 ByteLevel 전처리를 읽고, 공식 tokenizer.json의 ID와 일치했다. 기존 V1/V2 텍스트 특징과 아래 토큰 절단 통계는 이 잘못된 경로의 결과다. 기존 토큰 수를 정상 RoBERTa subword 수로 해석하면 안 된다. 이 직접 구현 오류를 **입력 의미/전처리 차이보다 먼저 수정해야 하는 항목**으로 추가한다. V3는 모든 조건의 텍스트를 동일하게 수정하며, 이 오류가 F1 격차 중 얼마를 설명하는지는 별도의 동일 조건 대조 전에는 확정하지 않는다. 증거는 `Data/hique_input_study_v3/tokenizer_loading_diagnosis.json`이며 기존 수치·원본 파일은 보존한다.

2026-10-03. 분석 대상은 2026-09-29의 H_AVT/F_AVT/H_T 15회 실행이다. 기존 학습·특징·예측을 보존하고 새 감사 산출물만 `Data/hique_diagnosis_2026-10-03/`에 저장했다.

## 결론과 증거 수준

논문의 Macro-F1 0.79와 현재 H_AVT 평균 0.5528은 같은 지표지만, 같은 실험 조건에서 얻은 값이라고 볼 근거가 부족하다. 공개 저장소의 전처리 경로가 논문과 일관되게 연결되지 않고, 이를 보완하며 우리가 결정한 전사·응답 선택·계층 위치·정규화 정책이 저자의 실제 입력과 일치하는지 검증되지 않았다.

가장 근거가 강한 문제는 **화자·질문·답변 구간의 불확실성과 실제 정보 폐기**다. 추가로 **입력 크기 차이, 첫 attention의 극단적 집중, 학습과 검증 성능의 큰 차이**가 관측됐다. 다만 전체 F1 차이 0.2372를 원인별로 몇 점씩 분해할 수는 없다. 아래에서 코드상 확정된 차이, 실제 관측, 아직 검증되지 않은 인과 가설을 구분한다.

## 1. 세 자료의 대응 관계

| 항목 | 논문 | 공개 저장소 | 실제 V2 실행 | 판단 |
|---|---|---|---|---|
| 전사·화자 | §3/4.4.3 인터뷰 전사; Whisper/BERTScore는 §5.4 외부 데이터셋 일반화에서 명시 | README는 일반 Method로 Whisper 소개; 코드에서는 물음표로 화자 분류 | 모든188명 Whisper base 재전사 및 동일 물음표 규칙 | DAIC 본실험의 원 입력과 동치인지 미확인 |
| 질문 코드북 | 66개 주질문+19개 후속 질문 | `300_TRANSCRIPT_FINAL.csv`의 topic/type 컬럼에 의존 | 논문 부록 순서로 복원, 모든 추정 질문을 BERTScore argmax로 배정 | 실제 저자 중간 CSV와 대조 불가 |
| 반복 답변 | 최종 집계 정책 불명확 | `embedding/roberta.py`는 같은 topic의 첫 answer; 별도 BERT 경로는 덮어쓰기 | 마지막 응답만 보존 | 우리가 어느 공개 경로를 따랐는지 구분해야 함 |
| 계층 | §4.3 선행 질문 관계와 계층 위치 | 모델은 고정 `range(85)` 위치; 관계 생성의 완결된 경로 미확인 | 최근 주질문 ID로 위치 조회, 깊이·직전 질문 ID는 모델 입력에 없음 | 논문에 근거한 가정이며 원 구현 복원은 아님 |
| 텍스트 | RoBERTa 마지막 hidden CLS,768차원 | RoBERTa 파일 존재; 다만 TF tensor/PyTorch 모델 혼용, `[1]` pooler | frozen RoBERTa-base 마지막 CLS,512토큰 제한 | CLS는 논문에 맞췄지만 freeze/길이/집계 동치 미확인 |
| 영상 | CLNF 68개 x/y 평균·분산272차원 | VGG 파일을 읽고 항상1–4초 사용; loader 결측4096 | CLNF272, 실제 응답 구간 사용 | 공개 오류를 고친 것이며 현재 VGG 버그가 남은 것은 아님 |
| backbone | §4.5 convolution 출력85×4 | width4,1head,모달리티별 self block2, cross branch3 | 동일 backbone, 위치 조회만 변경 | hidden dimension4를 오류로 단정할 근거 없음 |
| 증강 | train양성3배,85개 중10개 질문 masking | 제공 fit 경로에 증강 생성 단계 없음 | 양성 복사본2개,동기 masking,위치 ID 유지 | 원 저자 증강 tensor와 일치 미확인 |
| 학습 | 100epochs,batch8,lr0.0002,dropout0.5 | CLI 기본50epochs; devloss최저 checkpoint | 논문100epochs,devloss최저,마지막 불완전 batch 포함 | 주요 학습값은 논문 일치, batch 처리 등 차이 |
| 표본 | 107/35/47 | split CSV 사용 | 107/33/46 | train동일,dev440/458제외,test411제외 |

근거: [논문](https://arxiv.org/html/2408.03648v1), [공개 저장소](https://github.com/JuHo-Jung/HiQuE), 실행 코드 `experiments/hique2_{asr,data,av,network,train}.py`.

## 2. 우선순위 1: 잘못된 화자 구분이 모든 모달리티의 구간에 전파된다

공개 `whisper_segment.py`와 우리 `hique2_asr.py`는 `text.endswith('?')`이면 Ellie, 아니면 Participant로 분류한다. 이는 실제 화자 인식이 아니다. 면담자의 “Tell me about …” 같은 요청이 물음표 없이 전사되면 참가자로, 참가자가 되묻는 문장이면 면담자로 취급될 수 있다. 그 다음 추정 Ellie 행을 반드시85개 질문 중 하나에 배정하고 다음 Participant 행을 답변으로 쓰므로 오류가 다음 단계로 전달된다.

**기존 첫10명 감사보다 범위를 넓혀 train107명 전체를 재검사했다.** PHQ 라벨은 읽지 않고 제공 전사의 화자·시간과 비교했다.

| 관측 | 결과 |
|---|---:|
| 추정 질문 행 | 3,017개 |
| 시간 겹침이 Ellie보다 Participant에 더 큰 추정 질문 행 | 534개(17.7%) |
| 추정 질문 시간 중 제공 Ellie 구간과 겹친 비율 | 40.1% |
| 제공 Ellie 전체 발화 시간 중 추정 질문 구간에 포함된 비율 | 34.0% |
| 보존한 답변 구간 시간 중 Ellie 구간과 겹친 비율 | 10.1% |

이 수치를 화자분리 accuracy/precision/recall로 부르면 안 된다. 제공 Ellie 행에는 질문 외 맞장구도 있고, ASR 경계·무음·발화 겹침 및 일부 전사의 시각 오차가 영향을 준다. 다만 우리가 사용하는 질문과 답변이 검증된 면담자/참가자 구간이라고 보기 어렵다는 직접 증거다.

411·458에서는 이 규칙으로 질문 후보가0개였고,365는 Whisper가 영어가 아닌 `nn`으로 자동 감지했다. 원본에 질문이 없거나 실제 노르웨이어라는 뜻이 아니다.

논문 §5.4의 ASR/BERTScore는 E-DAIC/MIT 일반화 절차로 서술되어 있다. 그러나 공개 README는 이 경로를 일반적인 Method로 소개한다. 따라서 “논문 DAIC 본실험은 반드시 수동 전사만 사용했다”라고 확정하지는 않는다. **우리가 검증 없이 이 ASR 경로를 DAIC 본실험 전체에 적용한 선택은 우선 재검토 대상**이다.

결과 근거: `Data/hique_diagnosis_2026-10-03/input_audit.json`, 재검사 코드 `experiments/hique_diagnose_inputs.py`. 구현 위치: `hique2_asr.py:24`, `hique2_data.py:52`, `hique2_data.py:119`.

## 3. 우선순위 2: 반복 질문 덮어쓰기와 긴 응답 절단

현재 `build_record`는 동일한 질문 ID가 다시 나오면 앞선 응답을 버린다. 전체 입력에서 응답1,134개가 덮어써졌고, 서로 다른 부모 사이의 충돌이500개였다. 특히 일반 후속 질문이 여러 주제에서 반복될 때 앞선 문맥이 사라질 수 있다.

Train107명에서는:

- 답변이 연결된 사건3,014개 중 최종2,368개만 보존: **이벤트21.4% 제외**.
- 해당 단계의 단어 수137,789→116,444: **단어15.5% 제외**.
- 보존된2,368개 중217개는512토큰 초과. 내용 subword485,964→374,692: **내용 토큰22.9% 절단**.
- 가장 긴 슬롯은5,423토큰이었다. 물음표 기반 연속 화자 병합이 긴 응답 생성에 영향을 줄 수 있지만, 각 슬롯이 길어진 원인을 전수 판정한 것은 아니다.

이벤트·단어·토큰은 분모와 단위가 다르므로 손실 비율을 더하면 안 된다. 삭제된 정보가 모두 우울증 예측에 유용했다는 의미도 아니다. 하지만 보존해야 할 입력이 상당량 변경됐다는 사실은 확인된다.

**이전 설명 정정:** “마지막 응답을 쓰는 것은 공개 코드와 같다”는 설명은 별도 BERT 파일 덮어쓰기 경로에만 해당한다. 실제 공개 `embedding/roberta.py:33`은 `.values[0]`으로 첫 답변을 고른다. 다만 비공개 `_TRANSCRIPT_FINAL.csv`가 이미 답변을 합친 파일일 수 있으므로, 저자의 전체 반복 처리 정책까지 첫 답변이라고 단정할 수도 없다.

추가 근거: `Data/hique_diagnosis_2026-10-03/token_truncation.json`, `experiments/hique2_data.py:64`, `:154`, 공개 `embedding/roberta.py:33`.

## 4. 우선순위 3: 수치 크기·attention 동작과 일반화 실패

### 기존 체크포인트에서 확인한 현상

- H_AVT 다섯 시드 모두 영상 branch의 첫 self-attention에서 유효 train query의 **83.7~99.8%**가 특정 key 하나에0.999보다 큰 가중치를 줬다.
- 영상 feature embedding RMS는 약335~984, 같은 단계의 위치 embedding RMS는 약0.030~0.035다. 음성도 약19~168 대0.028~0.034로 크기 차이가 크다.
- 텍스트 첫 attention의 정규화 entropy는 거의1로,85개 위치에 거의 균등했다. 이는 첫 층의 관측이며 전체 모델이 모든 질문을 무조건 동일하게 취급한다는 뜻은 아니다.
- 공개 모델의 LayerNorm은 attention 이후에 있다. 따라서 그 뒤 LayerNorm이 있다고 해서 첫 attention에 들어가는 숫자 크기가 이미 맞춰졌다고 볼 수 없다.

특징을 표준화하지 않은 선택과 이 수치 현상이 연관됐을 가능성이 있다. 그러나 원 논문은 정규화 세부를 충분히 명시하지 않았고, 원시 크기 차이만으로 F1 손실량을 확정할 수는 없다.

### 학습이 덜 된 것인가?

H_AVT 다섯 시드의 마지막 epoch 학습 정확도는94.6~100%, 평균98.0%다. 반면 devloss는 최소값 평균0.6505에서 마지막 epoch 평균1.0551로 증가했다. 실제 평가는 마지막 epoch가 아니라 **devloss가 가장 낮은 checkpoint**를 사용했다.

그 checkpoint도 증강 전 train Macro-F1 평균0.8279 대비 dev0.5362로 차이가 크다. 현재 로그는 단순 학습 부족보다 **학습 표본에 잘 맞지만 다른 참가자에게 일반화하지 못하는 양상**을 보여준다. 이것만으로 ASR·특징·정규화 중 무엇이 원인인지는 분리되지 않는다.

근거: `Data/hique_diagnosis_2026-10-03/training/summary.json`, `training_diagnosis.json`. 원본 소스·가중치·manifest는 변경하지 않고 train forward만 추가 검사했다.

### 단일 변수 진단: 표준화만 바꾸면 회복되는가?

이 가설을 추측으로 남기지 않기 위해, 시드42/13/23과100epochs를 사전 기록한 뒤 **train/dev에서만** 추가3회 실행했다. 모델·증강·초기화·checkpoint 선택은 유지하고, train의 관측 슬롯으로만 모달리티별 특징 평균·표준편차를 계산했다. Test 예측은 새로 실행하지 않았다.

| Seed | 원본 dev Macro-F1 | 표준화 dev Macro-F1 | 차이 |
|---|---:|---:|---:|
| 42 | 0.4392 | 0.5417 | +0.1025 |
| 13 | 0.5417 | 0.4762 | −0.0655 |
| 23 | 0.6857 | 0.6857 | 0.0000 |
| 평균 | 0.5555 | 0.5679 | +0.0123 |

영상 첫 attention의0.999초과 집중률은 이3시드에서97.4~99.5%→0~0.127%로 크게 줄었다. 그러나 dev Macro-F1 개선은 일관되지 않았고, 최저 dev cross-entropy 평균은0.6515→0.7119로 오히려 악화됐다.

따라서 **수치 포화는 실제 현상이지만, 표준화 부족만으로 논문과의 큰 성능 격차를 설명할 수 있다는 가설은 이번 소규모 진단에서 지지되지 않았다.** 세 시드와 작은 dev 표본의 탐색 결과이며 표준화의 일반적인 유효성까지 부정하지 않는다. 결과가 좋아 보이는 seed42만 선택하거나 이를 test 개선으로 보고하지 않는다.

근거: `Data/hique_diagnosis_2026-10-03/training/standardized_probe/predeclared.json`, `summary.json`. 각 시드의 학습 기록·가중치는 별도 진단 폴더에 보관했다.

## 5. 계층 정보는 존재하지만, 저자의 계층 모델과 동일하지는 않다

현재 V2를 “계층이 아예 없는 모델”이라고 설명하면 틀린다. 부모 위치를 실제 embedding 조회에 사용한다. 하지만 최근 주질문 ID를 상속하는 연산은 우리가 명시한 해석이다.

논문의 “previous question’s Topic id”를 직전 canonical 질문 ID로 읽으면 현재 구현과 달라지며, 이미 상속된 위치를 뜻한다고 읽으면 root 상속이 가능하다. Fig.3도 정확한 tensor 변환을 유일하게 결정해 주지는 않는다. 전체 보존 후속1,171개 중410개가 깊이2 이상이고,361개에서 현재 부모 ID와 직전 canonical ID가 다르지만, 이를 곧바로361개 오류라고 부를 수는 없다.

더 중요한 점은 현재 네트워크에 실제로 들어가는 것이 `position_ids`이고 `previous_question_id`, `followup_depth`는 감사용 메타데이터라는 점이다. 따라서 H−F 비교가 검증한 것은 **이 특정 위치 ID 개입**이며 원 저자의 모든 계층 처리 효과가 아니다.

논문 Table4에서는 계층을 제외하고 질문 embedding을 사용한 조건도 Macro-F1 0.75다. 우리 고정 위치 AVT는0.5429이므로, 계층 하나만 바꾸면 격차가 모두 사라진다고 보는 근거는 약하다. 두 ablation도 전처리가 같다고 확인된 직접 비교는 아니다.

## 6. 공개 저장소의 완결성 문제와 환경 정보

2026-10-03 GitHub API 확인 결과 최신 커밋은 우리가 고정한 `24c553bf2666b442ae5b0e3490b998a5d4493559`와 같았다. 최신 수정 코드를 놓쳐 생긴 문제라는 증거는 없다.

공개 코드에는 다음 문제가 실제로 존재한다.

- `hique.py`의 feature 목록이 참가자 루프 안에서 초기화돼 마지막 참가자 특징만 반환되는 문제.
- 질문 segment ID와 topic ID의 연결이 완결되지 않은 경로.
- CLNF 모델입력272와 VGG/결측4096 불일치, 시각 특징 추출 구간1–4초 고정.
- RoBERTa 파일의 TF/PyTorch 혼용 및 pooler/CLS 불일치.
- TransformerBlock의 tuple 반환을 다음 block에 그대로 넣는 오류.
- `_TRANSCRIPT_FINAL.csv`와 질문·답변·계층 중간 자료의 생성 경로 미공개.

이 중 실행 오류 상당수는 우리 코드에서 이미 수정했다. 따라서 이를 **현재 실행에 남아 있는 직접 버그**라고 다시 계산하면 안 된다. 핵심은 공개 저장소만 실행해 저자의 Table1 입력과 설정을 복원할 수 없다는 점이다.

외부 사용자의 공개 이슈에도 [중간 CSV 누락 #4](https://github.com/JuHo-Jung/HiQuE/issues/4), [영상 구간 및 ID 문제 #5](https://github.com/JuHo-Jung/HiQuE/issues/5), [문장 ID/topic ID 및 관계 사용 문제 #6](https://github.com/JuHo-Jung/HiQuE/issues/6)가 있다. 이는 사용자 제보이며 저자가 모두 확인한 원인이라는 뜻은 아니다. 해당 지점은 이번에 코드로 직접 대조했다.

[환경 이슈 #2](https://github.com/JuHo-Jung/HiQuE/issues/2)에는 저자가 첨부한 requirements 파일이 있다. TensorFlow2.6.0/NumPy1.19.5/PyTorch1.8.1이 기재돼 있으며 우리 환경과 다르다. 다만 이 목록에는 Whisper/Transformers/BERTScore가 없고 stable/nightly 패키지가 함께 있어, 그대로 완결된 실험 lockfile이라고 보기 어렵다. 환경 차이의 F1 영향은 별도 검증되지 않았다. “환경 정보가 전혀 공개되지 않았다”는 주장도 정확하지 않다.

## 7. 주요 설명에서 배제하거나 제한해야 할 항목

| 설명 후보 | 이번 확인 결과 |
|---|---|
| Macro-F1과 weighted-F1을 혼동했다 | 논문0.79는Macro-F1,우리0.5528도Macro-F1. 지표 이름 차이로 설명되지 않음 |
| 평균5seed라서 낮아 보인다 | H_AVT의 가장 높은 관측 seed도0.5865. 현재 시드 범위만으로0.79를 설명할 수 없음 |
| test에서 한 명 빠져서 그렇다 | 기존46명 예측 고정 후47번째의 가능한 정답/예측4경우를 모두 넣어도 평균Macro-F1 최대0.5735. 단,dev누락이 모델 선택에 미치는 영향까지 배제한 것은 아님 |
| epoch이 부족하다 | 실제15회×100epoch 완료,학습 정확도는 높고devloss가 악화. 단순 epoch추가는 우선 해결책이 아님 |
| 차원을85가 아닌4로 줄였다 | 논문§4.5도 conv출력85×4. 85×85는 attention점수와 표현을 혼용한 서술 가능성이 있어 hidden4를 오류로 단정할 수 없음 |
| 공개 평가의 truth/pred 역순호출 때문 | P/R과weighted지표는 왜곡할 수 있지만 Macro-F1은 교환에 대칭. 해당 오류가0.237의Macro-F1격차를 직접 설명하지 못함 |

텍스트 단독은 예외적으로 시드 변동이 크다. 평균0.5497이지만 seed13은0.7319다. 이는 사후 진단 정보이며, 그 시드를 골라 원 논문 텍스트 성능을 재현했다고 주장하거나 모델을 선택하지 않는다. 또한 우리 H_T 구조와 공개 별도 `text_model()`도 동일하지 않다.

## 8. 우선 수행할 후속 검증

1. 동일한 train/dev 참가자 집합에서 제공 전사의 화자·시간을 사용한 입력과 현 ASR 입력을 비교한다. 질문·답변 연결의 정확성을 먼저 검사하고, 추가 누락 사례에 임의 질문을 만들지 않는다.
2. 저자의 `_TRANSCRIPT_FINAL.csv` 비식별 예시와 생성 코드, 질문순서, 반복답변·parent tensor를 확보해 대응한다. 첫/마지막/집계 중 한 가지를 성능에 맞춰 골라 원 방법이라 부르지 않는다.
3. 원 질문별 A/V/T 특징이나 checkpoint를 받을 수 있다면 소수 사례의 tensor를 직접 비교한다. 정규화·구간·pooling·position·padding의 동치 검증이 우선이다.
4. 제공된 전사와 원 중간 표현을 맞춘 뒤 계층 제거·모달리티 제거를 다시 비교한다. 현재처럼 여러 전처리를 동시에 바꾼 V1/V2 점수 차이를 특정 개선 효과로 해석하지 않는다.

기존 test 결과를 이미 봤으므로 후속 원인 분리와 선택은 train/dev에 제한한다. 새로운 일반화 주장을 하려면 별도의 미노출 평가가 필요하다. 논문 점수에 맞추기 위한 test 튜닝은 수행하지 않는다.

## 감사 산출물

- `experiments/hique_diagnose_inputs.py`:107명 train 시간·응답 보존 감사.
- `Data/hique_diagnosis_2026-10-03/input_audit.json`, `token_truncation.json`:집계와 참가자별 비공개 진단.
- `training/inspect_training.py`, `training/training_diagnosis.json`, `training/summary.json`:체크포인트·학습 곡선·특징 규모·attention 진단.
- `training/standardized_probe/`:사전 지정3개 시드의 train-only 표준화 대조, test 재평가 없음.
- `github_latest_commit.json`, `github_issues.json`, `github_issue*_comments.json`, `author_requirements.txt`:2026-10-03 공개 자료 확인 기록.

이 보고서의 원인 우선순위는 확인된 차이와 관측을 토대로 한 판단이다. 개별 요인이 논문 대비 성능 차이를 얼마나 만들었는지는 저자 입력 복원 및 통제 실험 없이 확정할 수 없다.
