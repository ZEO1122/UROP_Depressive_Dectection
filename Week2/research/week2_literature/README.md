# Week2 우울증 탐지 선행연구 조사

조사일: 2026-09-14. Week1의 DAIC-WOZ 이해를 출발점으로 세부 연구주제와 첫 재현 대상을 선정하기 위한 조사다.

**현재 방향:** 사용자 피드백을 반영해 PHQ-8 선별 예측에서 Assessment 지원으로 변경했다. [Assessment·DSM·LLM 기반 수정안](assessment_reframing_ko.md)을 먼저 읽는다. 아래 초기 정리본의 CTD 1순위 추천은 수정안으로 대체되었다.

- [선행연구 정리와 주제 제안](literature_review_ko.md): 핵심 5편 분석, 추가 10편 후보, 평가 조건 비교, 연구주제 3개, 구현 순서.
- [선정 문헌 데이터](selected_papers.json): OpenResearch 검색에서 선정한 15편의 ID·제목·링크·검토 수준.
- [검색 및 검증 기록](search_log.md): 실제 실행 명령, 검색 범위, 한계.
- `search/` (로컬 전용, Git 제외): OpenResearch 검색 응답 원본.
- `papers/` (로컬 전용, Git 제외): OpenResearch가 반환한 핵심 논문의 텍스트. 1–4는 추출 본문, 5는 메타데이터·초록이며 출판사 본문으로 보완했다.

이번 작업은 문헌조사와 실험 설계까지다. 우울증 탐지 모델의 학습이나 논문 성능 재현은 아직 실행하지 않았다. 기존 Week1 PPT와 DAIC-WOZ 원자료는 수정하지 않았다.
