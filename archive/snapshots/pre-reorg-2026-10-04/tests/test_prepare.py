import csv
import json
import zipfile

import pytest

from experiments.prepare import (
    build_qa,
    normalize_question,
    prepare,
    require_private_output,
    weak_report_label,
)


def row(speaker, start, stop, value, number):
    return dict(speaker=speaker, start_time=str(start), stop_time=str(stop),
                value=value, source_row=number)


def test_grouping_preserves_all_response_intervals():
    qa = build_qa([
        row("Ellie", 0, 1, "sleep1 (how is your sleep)", 2),
        row("Participant", 2, 3, "i have trouble", 3),
        row("Participant", 4, 5, "sleeping", 4),
        row("Ellie", 6, 7, "where are you from", 5),
        row("Participant", 8, 9, "somewhere", 6),
    ], 303, "train")
    assert len(qa) == 2
    assert qa[0]["participant_spans"] == [[2., 3.], [4., 5.]]
    assert qa[0]["duration"] == 2
    assert qa[0]["source_rows"] == [3, 4]
    assert qa[0]["question"] == "how is your sleep"
    assert qa[1]["question_category"] == "rapport"


def test_question_text_alone_never_implies_symptom():
    assert weak_report_label("Do you have insomnia?", "", "sleep")[0] == "unknown"
    assert weak_report_label("what happens when you don't sleep well", "i have trouble sleeping", "sleep")[0] == "unknown"
    assert weak_report_label("how is your sleep", "my mother has insomnia", "sleep")[0] == "unknown"
    assert weak_report_label("how is your sleep", "i sleep well", "sleep")[0] == "deny"


def test_scrubbing_and_overlap_flags():
    qa = build_qa([
        row("Ellie", 0, 3, "how is your sleep", 2),
        row("Participant", 2, 4, "scrubbed_entry", 3),
    ], 303, "train")[0]
    assert qa["scrubbed"] and qa["quality"]["overlap"]
    assert qa["weak_label"] == "unknown"


def test_wrapper_and_private_output():
    assert normalize_question("i wonder (about that)") == "i wonder (about that)"
    assert normalize_question("sleep2 (sleep well)") == "sleep well"
    with pytest.raises(ValueError, match="Data"):
        require_private_output(__import__("pathlib").Path("/tmp/public-results"))


def test_prepare_never_opens_sealed_test_and_exports_blank_labels(tmp_path, monkeypatch):
    import experiments.prepare as module
    monkeypatch.setattr(module, "require_private_output", lambda p: p)
    root = tmp_path / "data"
    root.mkdir()
    all_ids = sorted(set(range(300, 493)) - {342, 394, 398, 460})
    splits = {"train": all_ids[:107], "dev": all_ids[107:142], "test": all_ids[142:]}
    for split, ids in splits.items():
        with (root / f"{split}_split_Depression_AVEC2017.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Participant_ID"])
            writer.writerows([[pid] for pid in ids])
        for pid in ids:
            path = root / f"{pid}_P.zip"
            if split == "test" or pid == 440:
                path.write_bytes(b"not a zip - must never be opened")
            else:
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr(f"{pid}_TRANSCRIPT.csv", "start_time\tstop_time\tspeaker\tvalue\n0\t1\tEllie\thow is your sleep\n2\t3\tParticipant\ti sleep well\n")
    output = tmp_path / "out"
    result = prepare(root, output)
    assert result["test_transcripts_read"] == 0
    qa = [json.loads(line) for line in (output / "qa.jsonl").read_text().splitlines()]
    assert not {r["participant_id"] for r in qa} & set(splits["test"])
    with (output / "pilot_reviewer_1.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows and "weak_label" not in rows[0]
    assert all(not r["human_report_state"] for r in rows)
    rows[0]["human_report_state"] = "support"
    with (output / "pilot_reviewer_1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    prepare(root, output)
    with (output / "pilot_reviewer_1.csv").open() as stream:
        preserved = list(csv.DictReader(stream))
    assert preserved[0]["human_report_state"] == "support"
