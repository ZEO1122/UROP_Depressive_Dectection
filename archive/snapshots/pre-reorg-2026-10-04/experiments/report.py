"""Aggregate measured pilot results without calling proxy scores clinical performance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .prepare import require_private_output


def summarize(directory: Path) -> str:
    directory = require_private_output(directory)
    def read(name: str) -> dict:
        return json.loads((directory / name).read_text())
    prep = read("preparation_summary.json")
    a, b, c = read("results_a/a_report.json"), read("results_b/summary.json"), read("results_c/c_summary.json")
    digest = hashlib.sha256((directory / "qa.jsonl").read_bytes()).hexdigest()
    assert prep["qa_sha256"] == a["data_sha256"] == b["data_sha256"] == c["input_sha256"] == digest
    assert prep["test_transcripts_read"] == 0 and not b["test_used"] and c["sealed_test_rows_excluded"] == 0
    selections = [json.loads(line) for line in (directory / "results_c/c_selections_private.jsonl").read_text().splitlines()]
    overflows = sum(r["selected_words"] > r["word_budget"] for r in selections)
    assert overflows == 0
    reviewer_status = {}
    for number in (1, 2):
        with (directory / f"pilot_reviewer_{number}.csv").open() as stream:
            annotations = list(csv.DictReader(stream))
        reviewer_status[number] = {
            "rows": len(annotations),
            "people": len({row["participant_id"] for row in annotations}),
            "filled_report_labels": sum(bool(row.get("human_report_state", "").strip()) for row in annotations),
        }
    synthetic = [json.loads(line) for line in (directory / "results_a/a_predictions.jsonl").read_text().splitlines()
                 if json.loads(line)["dataset"] == "synthetic_construction"]
    trained_names = ("answer_only", "question_only", "qa_flat", "qa_relation")
    synthetic_unknown = {name: sum(row["predictions"][name] == "unknown" for row in synthetic)
                         for name in trained_names}
    rule_correct = sum(row["predictions"]["relation_rules"] == row["intended_label"] for row in synthetic)
    b_missing = b.get("feasibility", {}).get("missing_training_classes", [])
    b_inference = b.get("paired_bootstrap", {}).get("status", "computed_or_not_applicable")
    c_missing = {split: next((row["n_without_proxy_information"] for row in c["summary"] if row["split"] == split), 0)
                 for split in ("train", "dev")}
    def format_number(value) -> str:
        return "계산 불가" if value is None else f"{value:.4f}"
    comparison = a["natural_paired_bootstrap"]["differences_macro_f1_fixed_3"]["qa_relation"]
    model_names = {"answer_only": "응답만", "question_only": "질문만", "qa_flat": "질문+응답",
                   "qa_relation": "질문+응답+관계 특징", "relation_rules": "관계 규칙", "train_majority": "train 다수 class",
                   "always_unknown": "항상 unknown"}
    text = [
        "# 세 실험 실행 결과 — 독립 임상평가 아님",
        "",
        "**구현과 train/dev 탐색 실행은 완료했다. 독립 사람 주석을 사용하는 본평가는 아직 수행하지 못했다.**",
        "이 문서의 F1은 자동 규칙과의 일치도, coverage는 키워드 검출기의 대리 지표다. 임상 정확도·은폐 탐지율·새 방법의 우월성으로 인용하지 않는다.",
        "",
        "## 사용 자료와 보호",
        "",
        f"- Train {prep['splits']['train']['participants']}명 / Dev {prep['splits']['dev']['participants']}명, 전체 QA {sum(x['qa'] for x in prep['splits'].values()):,}개.",
        "- 공식 test의 전사·특징은 읽지 않았고 test 점수를 계산하지 않았다. 440 손상, 질문 전사 없는 세션 및 사전 사례 노출은 manifest에 기록했다.",
        "- 참가자별 원문·특징·예측은 Git 제외되는 Data 아래에만 저장했다. 외부 LLM API 및 신규 패키지 설치 없음.",
        "- unknown 약한 라벨은 보수적 규칙의 미적중·판단 유보도 포함한다. 실제 면담 정보 부족 비율이 아니다.",
        "",
        "## A — 질문 맥락 해석 기술 파일럿",
        "",
        f"학습 {a['counts']['train']['qa']} QA / 개발평가 {a['counts']['dev']['qa']} QA. 아래는 **독립 정답이 아닌 약한 라벨 일치도**다.",
        "",
        "| 모델 | Dev weak-label Macro-F1 |", "|---|---:|",
    ]
    for model, value in a["natural_dev_WEAK_LABEL_AGREEMENT"].items():
        text.append(f"| {model_names.get(model, model)} | {value['macro_f1_fixed_3']:.4f} |")
    text.extend([
        "",
        f"관계 특징−단순 Q+A 차이 {comparison['point']:+.4f}, 참가자 bootstrap 95% CI [{comparison['ci95'][0]:+.4f}, {comparison['ci95'][1]:+.4f}]. 독립 임상 정답에 대한 효과 추정은 아니다.",
        f"저자가 구성한 합성 짧은 응답 {len(synthetic)}개에서 모델별 unknown 출력 수는 {synthetic_unknown}다. 규칙은 {rule_correct}/{len(synthetic)}을 맞혔으나 기대 label과 함께 설계된 구성상 점검이지 독립 검증이 아니다.",
        "증상별 class 수를 확인하고 두 증상을 구별해 사람 주석으로 재평가해야 한다.",
        "",
        "## B — 개인 기준선 비교의 실행 가능성",
        "",
        "| 분할 | 원래 대상 QA | 공통 유효 참가자 | 공통 유효 QA | 자동 라벨 분포 |",
        "|---|---:|---:|---:|---|",
    ])
    for split, row in b["cohort"].items():
        text.append(f"| {split} | {row['symptom_qa']} | {row['matched_participants']} | {row['matched_qa']} | {row['weak_label_counts']} |")
    text.extend([
        "",
        "증상 질문 전 rapport 참조 30초, overlap·scrubbed 제외, 음성/COVAREP/각 시각 특징의 유효 시간 비율 80% 조건을 적용했다.",
        f"학습 누락 class: {b_missing or '없음'}. Dev 유효 참가자는 {b['cohort']['dev']['matched_participants']}명이고 실행된 비교 모델은 {len(b['models'])}개다. 추론용 CI 상태: {b_inference}. 모델 실행 불가 사유: {b.get('model_blocker', '없음')}.",
        "표본·class가 부족한 경우 이것은 참조 구간 규칙의 실행 가능성 결과이며 유효한 우월성 검증이 아니다. 독립 주석 없이 수치상 우열을 보고 후보를 선택하지 않는다. 사후에 임계값을 낮춰 좋은 수치를 고르지 않았다.",
        "",
        "## C — 분량 제한 근거 선택의 기술 파일럿",
        "",
        f"5개 선택법·4개 budget·무작위 5 seeds로 {len(selections):,}개 선택 결과를 만들었으며 예산 초과는 {overflows}건이다.",
        "전체 자료에서 사용 가능한 대리 slot은 sleep/interest × report/duration/frequency/function의 8종이다. 단어 출현이 해당 임상 정보의 충분한 근거라는 뜻은 아니다.",
        "",
        "| 선택법 | Dev proxy coverage@20% | Dev proxy AUC(10–50%) |",
        "|---|---:|---:|",
    ])
    for row in c["summary"]:
        if row["split"] == "dev":
            text.append(f"| {row['method']} | {format_number(row['proxy_coverage_at_20'])} | {format_number(row['normalized_auc_10_to_50'])} |")
    text.extend([
        "",
        "**slot_gain_per_word는 평가와 같은 검출기를 최적화하므로 높은 점수가 순환적으로 발생한다.** 나머지 선택법도 키워드 기반 대리 평가이며 독립 효용이나 우월성의 근거가 아니다. 실제 근거 precision과 부정·시점 손실은 미측정으로 남겼다.",
        f"전사 전체에 대리 정보가 없는 train {c_missing['train']}명·dev {c_missing['dev']}명은 coverage 분모에서 제외하고 보고했다. 추가 맥락이 필요한 짧은 응답은 독립 주석에서 확인해야 한다.",
        "",
        "## 본평가 전 남은 필수 작업",
        "",
        f"- pilot_reviewer_1.csv: {reviewer_status[1]['people']}명, {reviewer_status[1]['rows']} QA, 입력된 human_report_state {reviewer_status[1]['filled_report_labels']}개. pilot_reviewer_2.csv: {reviewer_status[2]['rows']} QA, 입력된 human_report_state {reviewer_status[2]['filled_report_labels']}개. 입력 유무와 무관하게 현재 실행은 이 주석을 학습/평가 정답으로 사용하지 않는다.",
        "- 독립 평가 후 불일치 조정, 평가자 일치도 확인 및 지침 동결이 필요하다. 자동 라벨이나 AI의 검토를 사람 주석으로 이름 바꾸지 않는다.",
        "- A 합성 변형의 자연스러움·기대 정답 확인, B 참조 구간 확보율 점검, C 전체 면담의 정보 항목별 근거 주석이 필요하다.",
        "- 현재 human-gold importer/주평가 runner는 아직 제공하지 않는다. 각 주석 schema와 조정 절차가 고정된 뒤 연결해야 하며, 현재 proxy 결과를 본평가로 자동 승격하지 않는다.",
        "- test는 그 이후에만 최종 고정한 프로토콜로 평가한다. 이번 작업으로 임상 효용·진단·은폐 탐지 효과를 입증하지 않았다.",
        "",
        "## 재현",
        "",
        "```sh",
        "python3 -m experiments.run_suite --zip-dir Data/DAIC-WOZ --output-dir Data/experiments/2026-09-27 --workers 4",
        "```",
        "execution.json에 코드 hash·명령·종료 상태, preparation_summary.json에 라이브러리 버전과 QA hash를 기록했다. 이미 작성된 주석 파일은 재실행으로 덮어쓰지 않는다.",
    ])
    result = "\n".join(text) + "\n"
    (directory / "results_summary_ko.md").write_text(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.data_dir)


if __name__ == "__main__":
    main()
