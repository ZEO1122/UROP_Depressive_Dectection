from copy import deepcopy
import hashlib
import json

import pytest

from experiments.run_a_annotated import (
    ScopedModel,
    grouped_oof,
    load_annotations,
    paired_intervals,
    scores,
    transition_scores,
    view_relations,
    view_text,
)


def sample(pid, qa, label="no_evidence"):
    source = {
        "previous_question": "previouscanary question", "previous_answer": "previouscanary answer",
        "anchor_question": "Do you have trouble sleeping?", "anchor_answer": "Yes.",
        "next_question": "nextcanary question", "next_answer": "nextcanary answer",
    }
    return {"participant_id": str(pid), "qa_id": str(qa), "source": source,
            "rationale": "forbiddenannotationcanary", "sleep": {"anchor_state": label, "state": label, "current_self_state": label},
            "interest": {"anchor_state": "no_evidence", "state": "no_evidence", "current_self_state": "no_evidence"}}


def test_feature_scope_and_no_annotation_input():
    row = sample(1, 1)
    anchor = view_text(row["source"], "anchor")
    window = view_text(row["source"], "window")
    assert "previouscanary" not in anchor and "nextcanary" not in anchor
    assert "previouscanary" in window and "nextcanary" in window
    assert "forbiddenannotationcanary" not in anchor + window
    assert all(key.startswith("anchor:") for key in view_relations(row["source"], "anchor"))


def test_vocabulary_fits_only_training_sources():
    train = [sample(1, 1, "support"), sample(2, 2, "no_evidence")]
    train[1]["source"]["anchor_answer"] = "I like sports."
    model = ScopedModel("anchor", True).fit(train, ["support", "no_evidence"])
    held = sample(3, 3)
    held["source"]["anchor_answer"] = "heldoutuniquecanary"
    model.predict([held])
    assert "heldoutuniquecanary" not in model.text.vocabulary_
    assert "forbiddenannotationcanary" not in model.text.vocabulary_


def test_participant_holdout_keeps_context_family_together():
    rows = [sample(1, "1a", "support"), sample(1, "1b", "support"),
            sample(2, "2a"), sample(3, "3a")]
    predictions, folds = grouped_oof(rows, "sleep", "anchor_report")
    assert len(folds) == 3 and all(f["held_participant"] not in f["train_participants"] for f in folds)
    only_positive_held = next(f for f in folds if f["held_participant"] == "1")
    assert only_positive_held["constant_fallback"]
    assert predictions["qa_text"][:2] == ["no_evidence", "no_evidence"]


def test_missing_class_and_zero_denominators_are_explicit():
    result = scores(["no_evidence", "no_evidence"], ["no_evidence", "support"])
    assert result["recall"]["support"] is None
    assert result["unnecessary_no_evidence_rate"] is None
    assert result["unsupported_assertion_rate"] == 0.5
    assert "conflict" in result["untested_labels"]


def test_context_change_requires_both_predictions_correct():
    row = sample(1, 1)
    row["sleep"]["state"] = "support"
    result = transition_scores([row], "sleep", ["no_evidence"], ["no_evidence"])
    assert result["changed_label_pairs"] == 1
    assert result["changed_pairs_both_correct"] == 0
    assert result["unchanged_pairs_both_correct"] is None


def test_paired_bootstrap_preserves_identical_predictions():
    rows = [sample(1, 1, "support"), sample(1, 2), sample(2, 3, "deny")]
    prediction = ["support", "no_evidence", "deny"]
    intervals = paired_intervals(rows, prediction, {"qa_text": prediction, "qa_relation": deepcopy(prediction)}, repeats=25)
    assert intervals["point"] == 0
    assert intervals["ci95_descriptive"] == [0.0, 0.0]


def test_annotation_hash_and_official_train_boundary(tmp_path, monkeypatch):
    import experiments.run_a_annotated as module
    monkeypatch.setattr(module, "split_ids", lambda root: {"train": [310, 312], "dev": [320], "test": [300]})
    rows = [sample(310, "310_00"), sample(312, "312_00")]
    for row in rows:
        row.update(annotation_origin="AI_PROVISIONAL_DRAFT", scope="anchor_plus_previous_and_next_QA")
        row["source"].update(participant_id=row["participant_id"], previous_qa_id="", next_qa_id="")
    rows[0]["participant_id"] = 310

    def save():
        raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (tmp_path / "ai_annotations.jsonl").write_bytes(raw)
        (tmp_path / "provenance.json").write_text(json.dumps({"n_anchors": len(rows), "outputs_sha256": {"ai_annotations.jsonl": hashlib.sha256(raw).hexdigest()}}))
    save()
    loaded, _ = load_annotations(tmp_path, tmp_path)
    assert [row["participant_id"] for row in loaded] == ["310", "312"]
    rows[0]["participant_id"] = "300"
    rows[0]["qa_id"] = "300_00"
    rows[0]["source"]["participant_id"] = "300"
    save()
    with pytest.raises(ValueError, match="official train"):
        load_annotations(tmp_path, tmp_path)
    (tmp_path / "ai_annotations.jsonl").write_text("modified")
    with pytest.raises(ValueError, match="changed since"):
        load_annotations(tmp_path, tmp_path)
