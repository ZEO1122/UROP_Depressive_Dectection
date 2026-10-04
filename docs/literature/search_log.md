# OpenResearch 검색·검증 기록

- 실행일: 2026-09-14 (Asia/Seoul)
- 목적: Week1 이후 우울증 탐지 세부 주제와 재현 후보 선정. 체계적 문헌고찰은 아님.
- 공식 도구: https://github.com/alphaXiv/OpenResearch
- 버전: `orx 0.2.1`
- 로컬 실행 파일: `../../.tmp/openresearch/bin/orx` (이 문서 기준)
- 공식 release의 macOS arm64 archive SHA-256: `80acc7442df785a2a0dbf047255306d93fb3e9045165c845cc42edda2e660f20`; 공식 설치 스크립트의 값과 일치.
- CLI는 작업 폴더에만 준비했으며 시스템 PATH나 쉘 시작 파일을 변경하지 않았다.
- `--no-telemetry` 옵션을 매번 사용했다. 논문 검색은 로그인 없이 실행됐다.
- 일반 sandbox에서는 네트워크/CLI 실행이 제한돼 승인된 escalated 실행으로 공개 검색 endpoint에 접근했다.
- `SKILL.md`, `orx skill orx-lit-review`, `orx skill orx-reports`를 읽었다. 문헌 검색 loop는 주 에이전트가 직접 수행했다.
- `orx up` 프로젝트나 실험 트리는 생성하지 않았다. 문헌 검색 CLI를 실제 사용했으며 결과는 기존 레포의 `research/week2_literature/`에 기록했다.

## 검색 설정

난도 6/10, 기본 priority 유지, 날짜 하한 없음, 상한 2026-09-14. alphaXiv keyword에는 DAIC-WOZ 단일 키워드를 사용했다. 초기 결과가 최신 연구에 치우쳐 baseline gap을 OpenAlex 한 번으로 보완한 뒤 검색을 종료했다. biology 중심 질문이 아니므로 bioRxiv는 사용하지 않았다.

```sh
ORX=.tmp/openresearch/bin/orx
"$ORX" --no-telemetry discover keyword 'DAIC-WOZ' --published-before 2026-09-14 --limit 15
"$ORX" --no-telemetry discover embedding 'Depression detection from clinical interviews with text speech and visual modalities reproducible models and evaluation bias' --published-before 2026-09-14 --limit 15
"$ORX" --no-telemetry discover openalex 'depression detection multimodal datasets generalization' --published-before 2026-09-14 --limit 15
"$ORX" --no-telemetry discover openalex 'DAIC-WOZ depression detection baseline' --published-before 2026-09-14 --limit 15
```

검색 응답 60건, ID 기준 56개. 동일 논문의 arXiv/DOI 버전 등은 제목을 대조해 선정 시 합쳤다. 비관련 통증·알츠하이머·일반 로봇 대화 연구를 제외했다. 15편 선정 ID 모두 성공한 검색 응답에서 추적 가능하다. 관련성이 높은 논문을 목적별로 골랐으며 인용 수·투표 수로 순위를 정하지 않았다.

## 본문 읽기

```sh
"$ORX" --no-telemetry paper 2404.14463
"$ORX" --no-telemetry paper 2603.24651
"$ORX" --no-telemetry paper 2605.23977
"$ORX" --no-telemetry paper 2607.03744
"$ORX" --no-telemetry paper 10.1038/s41598-024-60278-1
```

앞의 네 명령은 논문 추출 텍스트를 반환했다. 마지막 DOI는 메타데이터·초록만 반환해 출판사 원문 https://www.nature.com/articles/s41598-024-60278-1 을 웹으로 확인했다. OpenAlex 저자 목록에 중복된 이름이 있어 서지 저자 목록을 그대로 복사하지 않았다.

## 외부 확인

- P01: ACL Anthology에서 ClinicalNLP 2024 출판 확인. 저자 GitHub README에서 논문 대응·평가·GCN 학습 절차 확인.
- P04: 저자 GitHub에서 논문 대응·CPU CTD 경로·cleaned split·score-derived binary 확인. 코드 실행 및 내부 구현 감사는 미수행.
- P03: 논문에 명시된 Zenodo DOI 페이지 접근 실패. 코드 공개는 저자의 명시로만 기록하며 가용성을 보증하지 않음.
- HiQuE와 D-vlog DOI 페이지 직접 열기 실패. 이 둘은 성공한 OpenAlex 초록에 기반한 읽기 후보로만 사용.
- USC 공식 페이지에서 DAIC-WOZ/E-DAIC 배포 진입점 확인. 신청 및 추가 데이터 확보는 미수행.

## 검증 범위와 남은 작업

모델 학습, 성능 재현, 데이터 split 재계산을 실행하지 않았다. Week1의 로컬 데이터 사실은 PPT·발표 노트·기존 QA 자료에서 확인한 이력으로 취급했다. 연구주제의 신규성은 예비 판단이며, 구체 주제 확정 전 P08/P11 및 후보 주제의 인접 문헌 정독이 필요하다. P04의 세션 수 산술 불일치는 해결되지 않아 정리본에 명시했다.
