# 정리 이전 실행 소스

`experiments/`와 `tests/`는 2026-10-04 정리 전 파일을 바이트 그대로 보존한 스냅샷이다. `manifest.json`에 원래 상대 경로와 SHA256이 있다. 수정하지 않는다.

현재 개발은 `src/urop/`와 루트 `tests/`에서 진행한다. 이 스냅샷은 과거 실행을 감사·복원하기 위한 것이며, V1/V2의 잘못된 토크나이저도 기록 보존을 위해 원래대로 남아 있다.

```sh
PYTHONPATH=src python3 -m urop legacy verify
PYTHONPATH=src python3 -m urop legacy run --python .tmp/hique-tf/bin/python experiments.hique3_test -- --help
PYTHONPATH=src python3 -m urop legacy run pytest archive/snapshots/pre-reorg-2026-10-04/tests -q
```

호환 실행기는 원래 `experiments/*.py` 위치에 검증된 소스를 잠시 복사하고 명령이 종료되면 자신이 만든 파일만 제거한다. 다른 내용의 파일이나 심볼릭 링크가 있으면 덮어쓰지 않는다. 실행 도중 파일 내용이 바뀌면 삭제하지 않고 남긴다. 기존 동결 JSON이나 Data 산출물을 이동·갱신하지 않는다.

과거 명령의 원래 동작은 유지된다. 따라서 학습·보고서 생성 명령이 가리키는 출력 경로를 확인해야 하며, 감사만 필요하면 `legacy verify`를 사용한다. 동결된 절대 경로는 현재 저장소 위치를 전제로 한다. 다른 위치로 저장소를 옮길 경우 별도의 경로 이관 검증이 필요하다.

프로세스가 강제 종료돼 `.urop-legacy.lock`이 남으면, 그 안의 PID가 종료됐는지 확인하고 임시 복원 파일을 검토한 뒤 잠금을 제거한다. 자동으로 다른 실행의 잠금을 없애지 않는다.
