import csv
import hashlib
import json

from experiments import report


def test_report_derives_findings_and_annotation_status(tmp_path, monkeypatch):
    """Changing measured results must not retain conclusions from the first run."""
    monkeypatch.setattr(report, "require_private_output", lambda path: path)
    (tmp_path / "qa.jsonl").write_text("fixture\n")
    digest = hashlib.sha256((tmp_path / "qa.jsonl").read_bytes()).hexdigest()

    def save(name, data):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    save("preparation_summary.json", {
        "qa_sha256": digest, "test_transcripts_read": 0,
        "splits": {"train": {"participants": 5, "qa": 10}, "dev": {"participants": 3, "qa": 4}},
    })
    save("results_a/a_report.json", {
        "data_sha256": digest, "counts": {"train": {"qa": 10}, "dev": {"qa": 4}},
        "natural_paired_bootstrap": {"differences_macro_f1_fixed_3": {
            "qa_relation": {"point": 0.4, "ci95": [0.2, 0.6]}}},
        "natural_dev_WEAK_LABEL_AGREEMENT": {"qa_flat": {"macro_f1_fixed_3": 0.25}},
    })
    prediction = {name: "support" for name in ("answer_only", "question_only", "qa_flat", "qa_relation", "relation_rules")}
    (tmp_path / "results_a/a_predictions.jsonl").write_text(json.dumps({
        "dataset": "synthetic_construction", "intended_label": "unknown", "predictions": prediction,
    }) + "\n")
    cohort = {"symptom_qa": 4, "matched_participants": 3, "matched_qa": 4, "weak_label_counts": {"support": 4}}
    save("results_b/summary.json", {
        "data_sha256": digest, "test_used": False, "cohort": {"train": cohort, "dev": cohort},
        "models": {"raw": {}}, "feasibility": {"missing_training_classes": []},
        "paired_bootstrap": {"status": "fixture_status"},
    })
    save("results_c/c_summary.json", {
        "input_sha256": digest, "sealed_test_rows_excluded": 0,
        "summary": [{"split": split, "method": "chronological", "n_without_proxy_information": 7,
                     "proxy_coverage_at_20": None, "normalized_auc_10_to_50": None} for split in ("train", "dev")],
    })
    (tmp_path / "results_c/c_selections_private.jsonl").write_text(json.dumps({"selected_words": 5, "word_budget": 10}) + "\n")
    for reviewer in (1, 2):
        with (tmp_path / f"pilot_reviewer_{reviewer}.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["participant_id", "human_report_state"])
            writer.writerow([1, "support" if reviewer == 1 else ""])
    text = report.summarize(tmp_path)
    assert "합성 짧은 응답 1개" in text
    assert "규칙은 0/1" in text
    assert "Dev 유효 참가자는 3명" in text
    assert "비교 모델은 1개" in text
    assert "train 7명·dev 7명" in text
    assert "입력된 human_report_state 1개" in text
    assert "계산 불가" in text
    assert "CI [+0.2000, +0.6000]" in text
    assert "명확한 개선을 확인하지 못했다" not in text
