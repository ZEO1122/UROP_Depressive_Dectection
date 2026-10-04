from copy import deepcopy

import pytest

from experiments.assemble_ai_annotations import no_evidence, validate_record


def fixture():
    source = {
        "qa_id": "fixture_01", "participant_id": "fixture",
        "anchor_question": "Do you have trouble sleeping?", "anchor_answer": "Yes, sometimes.",
        "previous_qa_id": "fixture_00", "previous_question": "Where are you from?", "previous_answer": "Somewhere.",
        "next_qa_id": "fixture_02", "next_question": "When?", "next_answer": "Last year.",
    }
    record = {"qa_id": "fixture_01", "sleep": None, "interest": None,
              "review_priority": "low", "review_reasons": [], "rationale_ko": "가상 검증 예제"}
    return record, source


def supported_record():
    record, source = fixture()
    value = no_evidence()
    value.update(anchor_state="support", state="support", current_self_state="support",
                 time="current", experiencer="self", note_ko="가상 예제의 명시적 답변")
    value["evidence"] = [
        {"source": "anchor_question", "quote": "Do you have trouble sleeping?", "purpose": "context"},
        {"source": "anchor_answer", "quote": "Yes", "purpose": "report"},
    ]
    record["sleep"] = value
    return record, source


def test_irrelevant_means_no_evidence_not_denial():
    record, source = fixture()
    value, warnings = validate_record(record, source)
    assert value["sleep"]["state"] == "no_evidence"
    assert value["interest"]["current_self_state"] == "no_evidence"
    assert not value["human_reviewed"] and not value["expert_reviewed"]
    assert warnings == []


def test_quote_offsets_and_source_ids_are_exact():
    record, source = supported_record()
    value, _ = validate_record(record, source)
    for evidence in value["sleep"]["evidence"]:
        assert source[evidence["source"]][evidence["start_char"]:evidence["end_char"]] == evidence["quote"]
        assert evidence["source_qa_id"] == source["qa_id"]


def test_invented_quote_rejected():
    record, source = supported_record()
    record["sleep"]["evidence"][1]["quote"] = "I have insomnia for two weeks"
    with pytest.raises(ValueError, match="not present"):
        validate_record(record, source)


def test_question_alone_not_patient_evidence():
    record, source = supported_record()
    record["sleep"]["evidence"] = record["sleep"]["evidence"][:1]
    with pytest.raises(ValueError, match="participant evidence"):
        validate_record(record, source)


def test_anchor_cannot_borrow_neighbour_evidence():
    record, source = supported_record()
    record["sleep"]["evidence"][1] = {"source": "previous_answer", "quote": "Somewhere", "purpose": "report"}
    with pytest.raises(ValueError, match="anchor-answer"):
        validate_record(record, source)


def test_past_only_cannot_be_current_and_slot_needs_own_evidence():
    record, source = supported_record()
    record["sleep"]["time"] = "past"
    with pytest.raises(ValueError, match="current-self"):
        validate_record(record, source)
    record, source = supported_record()
    original = deepcopy(record)
    record["sleep"]["duration"] = {"status": "observed", "text": "2주"}
    with pytest.raises(ValueError, match="duration lacks"):
        validate_record(record, source)
    assert original["sleep"]["duration"]["status"] == "insufficient"
