# HiQuE 공개 코드 기반 수정 재현 결과

> **상태: 토크나이저 오류가 확인된 과거 실험.** 아래 수치는 당시 실행 기록이며 정상 RoBERTa 입력의 재현 성능으로 인용하지 않는다. 수정 후 현재 기준은 [V3 실험](../v3_input_study/README.md)이다.

**2026-10-03 추가 정정:** 기존 명시적 RobertaTokenizerFast 로딩이 Transformers5.9에서 BPE merges 없이 글자 단위로 토큰화되는 오류를 발견했다. 이 실행의 RoBERTa 텍스트 특징 및 같은 loader를 쓰는 질문 유사도 처리에 영향을 준다. 아래 수치는 당시 실행 기록이며 정상 RoBERTa 입력의 성능으로 해석하지 않는다. 기존 산출물은 보존하고 V3에서 올바른 토크나이저로 재계산한다.

**실제 PHQ-8 라벨로 학습·평가한 결과다. 기존 AI 주석 일치도 실험과 다르며, 임상 진단 성능은 아니다.**

정확히 동일한 원 실험의 숫자 재현이 아니라, 공개 모델 구조를 수정하고 논문의 특징을 맞춘 adapted reproduction이다.

## 실행 조건

- 원 코드: https://github.com/JuHo-Jung/HiQuE, commit `24c553bf2666b442ae5b0e3490b998a5d4493559`.
- 동일 공통 cohort: train 107 / dev 32 / test 45명. 각 참가자는 한 split에만 존재한다.
- 7개 모달리티 구성 × 5개 seed = 35회 학습. seed [13, 23, 37, 42, 79].
- 각 100 epochs, batch 8, Adam lr 0.0002, head dropout 0.5. Dev loss 최소 checkpoint, 분류 threshold 0.5.
- PHQ8_Binary 원본 유지, 409의 score/binary 불일치를 자동 수정하지 않음. Test label은 사용자 제공 full_test_split.csv이며 외부 배포본 진위까지 인증하지는 않았다.
- Train에서만 scaler를 fit하고 양성 복사본 두 개를 추가해 3배로 증강한다. 각 복사본의 질문 10개를 모든 modality에서 동일하게 가린다.
- Text: frozen RoBERTa-base CLS, 7007개 응답 slot; 512-token 초과 424개를 사전 고정된 방식으로 자름.
- Audio: openSMILE eGeMAPSv02 functionals 88. Visual: CLNF 2D 68개 좌표의 1fps 평균·분산 272.
- 모든 조건에 동일한 T/A/V 공통 관측 slot mask를 사용한다. 질문 등장 여부만 쓰는 LR 대조군은 별도로 평가한다.
- 원본 데이터·AI 주석은 변경하지 않았다. 새로운 모델은 AI 주석을 학습 target으로 사용하지 않았다.

## Test 결과: seed별 점수의 평균

아래 ±는 사전 고정 seed 간 표준편차(ddof=0)이며 참가자 신뢰구간이 아니다. 좋은 seed만 선택하지 않았다.

| 입력 | Macro-F1 평균 ± SD | 양성 F1 평균 | 민감도 평균 | 특이도 평균 | AUROC 평균 |
|---|---:|---:|---:|---:|---:|
| A | 0.5828 ± 0.0450 | 0.4468 | 0.5143 | 0.6903 | 0.5866 |
| V | 0.6026 ± 0.0648 | 0.4752 | 0.5286 | 0.6968 | 0.6253 |
| T | 0.4888 ± 0.0553 | 0.2911 | 0.3143 | 0.6903 | 0.5157 |
| AV | 0.6093 ± 0.0732 | 0.4755 | 0.5143 | 0.7226 | 0.6134 |
| AT | 0.4326 ± 0.0333 | 0.1331 | 0.1143 | 0.8194 | 0.4922 |
| VT | 0.4257 ± 0.0382 | 0.1293 | 0.1286 | 0.8000 | 0.4995 |
| AVT | 0.4951 ± 0.0853 | 0.2870 | 0.2714 | 0.7226 | 0.6120 |

관측된 평균 Macro-F1은 **AV (0.6093)**에서 가장 높았다. 이는 test 결과의 기술적 요약이며 test에서 새 모델을 선택하거나 그 우위를 확증한 분석은 아니다.

질문 등장 여부만 사용하는 대조군: Macro-F1 **0.5342**, 양성 F1 0.2857. 실제 참가자 발화·음성·얼굴 특징을 입력하지 않은 진단용 대조이며, 질문 경로의 영향을 점검한다.

## 사전 지정한 주요 비교: AVT − T

5개 seed의 예측 확률을 참가자별로 먼저 평균한 후 평가한 Macro-F1 차이: **+0.0602**.

참가자 paired bootstrap 2000회 95% CI: **[-0.1583, +0.2630]**.

차이의 CI가 0을 포함하므로, 이 실험만으로 세 모달리티 결합이 텍스트 단독보다 우수하다고 결론 내릴 수 없다.

이 비교는 위 표의 seed별 F1 평균 차이와 다른 계산이다. 다른 21쌍 비교는 탐색 분석으로 test_results.json에 보관한다. 이 CI는 학습된 모델을 고정한 작은 test cohort의 재표집 불확실성으로, 훈련 데이터·seed의 모든 불확실성까지 포함하지 않는다. 다른 병원·언어·임상 진단으로 일반화를 보장하지 않는다.

## 원 논문과의 차이·코드 수정

- 2026-09-28 대조에서 확인: build_slots는 고정 85개 질문 ID에 응답을 합치며, 논문 §4.3의 선행 주질문/후속 질문별 parent topic 관계를 별도 입력하지 않는다. PositionEmbedding도 고정 슬롯 위치만 사용한다. 따라서 핵심 hierarchical question embedding을 충실히 재현한 결과로 해석할 수 없다. 이 차이의 성능 영향은 분리 실험으로 검증하지 않았다.
- 공개 source의 TransformerBlock이 (tensor, attention score)를 반환해 다음 layer에 tuple이 들어가는 오류를 tensor 출력으로 수정했다. get_config의 없는 속성과 optimizer 호환성도 고쳤다. AVT는 수정된 원 hique()를 직접 호출한다.
- 단일·이중 조건은 같은 AVT backbone에서 branch를 제거한 통제 ablation이다. 공식 저장소의 서로 다른 named baseline을 그대로 혼합하지 않았다.
- 원 공개 loader는 참가자별 feature 누적과 시각 결측 차원에 오류가 있어 새 검증 loader를 사용했다. 원 class_report 호출 순서 대신 truth/probability를 명확히 구분한 지표를 계산했다.
- 공개 visual 추출기의 VGG/fixed1–4초 경로 대신 논문에 명시된 CLNF 얼굴 좌표 통계를 구현했다. 공개 text 경로의 BERT/TensorFlow/PyTorch 혼용 대신 논문 RoBERTa CLS를 사용했다.
- 제공 수동 전사를 사용했다. 질문은 논문 Appendix의 85개에 normalized exact 및 frozen RoBERTa-base token matching으로 대응시켰다. 원 README의 ASR/BERTScore 경로와 다른 mapping adaptation이다.
- 질문 mapping: {'non_question': 83, 'exact': 98, 'roberta_token_f1': 63}. 참가자 전사에 대한 통계이며 라벨로 매핑을 조정하지 않았다.
- Participant-only 구간, scrubbed/overlap 제거, 품질 임계값, train-only standardization, 공통 slot mask는 명시한 재현 보강 조건이다. 원 논문의 동일 입력·189명 전체 결과라고 주장하지 않는다.
- 440 손상, 451/458/480 질문 전사 누락, 300 사전 사례 노출을 제외한다. 402는 가용 시각 구간만 사용한다. 문헌의 공개 사례에 대한 노출 이력도 protocol.json에 기록했다.
- Attention 점수는 설명의 정당성이나 실제 은폐를 검증하지 않는다. 이 실험의 정답은 PHQ 자기보고 기반 선별 label이다.

## 검증·재현 자료

- protocol.json: 특징·분할·seed·학습 정책. frozen_config.json: 실제 feature/코드/라벨/프로토콜 hash.
- model_patch_manifest.json: 원 코드 SHA와 수정 목록. environment_tf.txt / environment_features.txt: 환경 버전.
- runs/{mode}_seed{seed}/: 100epoch history, devlossbestweights, dev/testprediction, augmentation log.
- evaluation_manifest.json: 모든 학습 산출물 검증 뒤 test 결과를 읽은 시점·완료 hash. test 결과를 보고 설정을 바꾸지 않았다.
- verification.json: 35개 run의 hash·100epoch history·최저 dev loss checkpoint·test ID 및 별도 계산한 Macro-F1 검증 통과.
- upstream/, av/, text.npz, features.npz 및 참가자별 자료는 Git 제외되는 Data/hique_reproduction 내부에 보관한다.

실행 방법은 experiments/hique/v1/README.md를 참고한다.
