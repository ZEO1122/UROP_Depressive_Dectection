# AI 잠정 주석 탐색

공식 train 20명·199 QA의 AI 초안을 이용해 참가자 1명씩 제외하는 교차검증을 수행했다. **임상 전문가·독립 사람 gold가 아니다.** Dev/test는 이 실행에 사용하지 않았다.

- [주석 범위와 작성 기록](annotation_notes.md)
- [결과와 한계](results.md)
- [원래 대리 라벨 파일럿](../evidence_pilot/README.md)

원문·근거 span·초안·OOF 예측은 `Data/experiments/2026-09-27/ai_annotation_draft/`와 `ai_annotated_a/`에만 보관한다. 특징은 원문에서만 계산하고 주석 설명·ID를 입력에 넣지 않는다.

과거 실행 코드는 [스냅샷](../../archive/snapshots/pre-reorg-2026-10-04/experiments/)에 보존한다. 저장소 루트에서 다음 명령으로 실행할 수 있지만, 기존 출력 대신 별도 출력 경로를 지정해야 한다.

```sh
python -m urop legacy run experiments.run_a_annotated -- --help
```

실행 환경과 복원 방식은 [재현 안내](../../docs/reproducibility.md)를 따른다.
