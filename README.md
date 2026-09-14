# UROP · 우울증 Assessment 지원 연구

임상 지식과 면담 맥락을 활용해 우울 증상의 근거를 식별하고, 평가에 필요한 정보가 무엇인지 탐지하는 LLM 연구를 준비하는 저장소입니다.

현재는 **선행연구 조사와 연구 설계 단계**입니다. 모델 학습·성능 재현·임상 검증은 아직 수행하지 않았습니다.

## 연구 방향

연구 질문은 **“임상 기준과 관측된 면담 맥락이 LLM의 증상 근거 판별 및 정보 부족 탐지를 개선하는가?”**입니다.

- 증상 근거가 있는 발화와 시간·기능 맥락을 연결합니다.
- 근거의 지지·부인·불충분·상충 상태를 구분합니다.
- 문헌의 임상 지식, 면담 내 맥락, 실제 제공된 환자 정보를 구분합니다.
- 관측되지 않은 환자 정보는 추측해서 채우지 않고 추가 확인 사항으로 남깁니다.

DSM 기반 지식은 평가의 기준틀로 활용합니다. PHQ-8 예측을 임상 진단으로 해석하지 않으며, 현재 주제의 신규성과 임상적 타당성은 추가 검토가 필요합니다.

## 읽기 순서

1. [현재 연구 방향: Assessment·DSM·LLM](Week2/research/week2_literature/assessment_reframing_ko.md)
2. [선행연구 정리와 초기 주제 검토](Week2/research/week2_literature/literature_review_ko.md)
3. [선정 문헌 15편과 검토 수준](Week2/research/week2_literature/selected_papers.json)
4. [문헌 검색 방법과 검증 범위](Week2/research/week2_literature/search_log.md)

초기 문서의 PHQ-8/대화 시간 특징 중심 제안은 Assessment 중심 수정안으로 대체되었습니다. 문헌에서 보고한 성능과 이 프로젝트가 직접 재현한 성능을 구분합니다.

## 폴더 구조

```text
UROP/
├── README.md
├── .gitignore
├── Week1/
│   └── README.md                 # 우울증·PHQ-8·DAIC-WOZ 이해
└── Week2/
    ├── README.md
    └── research/
        └── week2_literature/     # 문헌 정리, 연구 방향, 서지 목록
```

로컬에는 `DAIC_WOZ/`, 발표 PPT, `output/`, `.tmp/` 등이 추가로 존재할 수 있습니다. 이들은 GitHub 공유 대상에서 제외됩니다. 일부 조사 기록의 로컬 경로는 자료를 검토했던 당시 위치이며 다른 환경에서 실행할 경로가 아닙니다.

## 데이터 및 공유 범위

DAIC-WOZ는 [USC 공식 배포처](https://dcapswoz.ict.usc.edu/)를 통해 해당 이용 조건에 따라 별도로 확보해야 합니다. 데이터셋, 참가자 전사·녹음·특징·라벨과 파생 자료를 이 저장소에 배포하지 않습니다.

현재 발표 자료와 `output/`에는 데이터셋에서 추출한 예시가 포함되어 있어 로컬에 보관합니다. 공개용 자료가 필요하면 참가자 정보와 제3자 자료의 배포 범위를 확인한 별도 버전을 준비합니다.

내려받은 논문 전문과 검색 원본은 제외하고 서지 링크·요약·검색 방법을 공유합니다. API 키, `.env`, 모델 가중치, feature cache, 실행 로그, 가상환경도 `.gitignore`로 제외합니다. 제외 규칙은 파일을 삭제하지 않습니다.

## 문헌조사 도구

[OpenResearch](https://github.com/alphaXiv/OpenResearch) CLI 0.2.1의 alphaXiv/OpenAlex 검색을 사용했습니다. 조사일은 2026-09-14이며, 체계적 문헌고찰이나 전체 분야를 빠짐없이 수집한 목록은 아닙니다. CLI 바이너리는 포함하지 않으며 설치는 공식 문서를 참고합니다.

## 다음 단계

- 증상 근거·기간·기능·정보 부족에 대한 annotation 지침 작성
- 임상 전문가와 소규모 면담 annotation 검토
- 동일 LLM에서 임상 지식과 면담 맥락 제공 조건 비교
- 근거 판별, 정보 부족 탐지, 근거 없는 주장 비율 평가

## GitHub 업로드 전 확인

```sh
git status --short
git ls-files --others --exclude-standard
git add --dry-run .
```

목록에 공개하려는 파일만 나타나는지 확인합니다. `.gitignore`는 이미 추적 중인 파일이나 과거 Git 기록을 제거하지 않습니다. 이 로컬 저장소에는 도구용 `refs/codex/` 참조가 있으므로 저장소 전체를 보내는 `git push --mirror` 대신 공개할 브랜치만 push합니다.

이 저장소에 제3자 데이터·논문의 재배포 권한을 부여하는 라이선스는 포함하지 않았습니다. 각 자료는 원 배포처의 조건을 따릅니다.
