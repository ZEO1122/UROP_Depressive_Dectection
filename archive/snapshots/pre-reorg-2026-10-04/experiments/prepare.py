"""Prepare private QA records without reading the sealed official test transcripts.

The weak-label instrument is deliberately conservative and is NOT clinical truth.
Run with ``python -m experiments.prepare``. No participant data leave this machine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import random
import re
import zipfile
from pathlib import Path

SEED = 42
NO_INTERVIEWER = {451, 458, 480}
PRIOR_EXPOSURE = {300}
INVALID_ARCHIVES = {440}


def normalize_question(text: str) -> str:
    """Remove only the documented machine utterance-ID wrapper."""
    match = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s+\((.*)\)", text.strip())
    return (match.group(1) if match else text).strip()


def tokens(text: str) -> list[str]:
    return re.findall(r"\b[\w]+(?:['’][\w]+)?\b", text.lower())


def question_category(question: str) -> str:
    q = question.lower()
    if re.search(r"\bsleep|\binsomnia|\basleep", q):
        return "sleep"
    if re.search(r"\benjoy|\binterest\b|\bfun\b|\bhobb", q):
        return "interest"
    if re.search(
        r"where.*(?:from|grow|live)|how long.*(?:la|los angeles)|"
        r"(?:what|where).*(?:work|job|school|study)|"
        r"(?:tell|what|how).*(?:family|hometown)|how old", q
    ):
        return "rapport"
    return "other"


def weak_report_label(question: str, answer: str, domain: str) -> tuple[str, str]:
    """Lexical instrument for feasibility, never used to populate human labels.

    A support label means a narrow phrase match, NOT a DSM symptom or diagnosis.
    The question alone can never create a positive or negative report label.
    """
    a, q = answer.lower().replace("’", "'"), question.lower()
    if domain not in {"sleep", "interest"}:
        return "unknown", "outside_target"
    if re.search(r"scrubbed|\bxxx\b", a):
        return "unknown", "unreliable_transcript"
    if re.search(r"\b(?:my mother|my father|my son|my daughter|my friend|he|she)\b", a):
        return "unknown", "experiencer_ambiguous"
    if re.search(r"\b(?:last year|years ago|used to|when i was)\b", a + " " + q):
        return "unknown", "time_ambiguous"
    if re.search(r"what are you like when|what happens (?:if|when)|would you|suppose", q):
        return "unknown", "hypothetical_question"
    if domain == "sleep":
        support = bool(re.search(
            r"(?:trouble|hard|difficult|poor|bad|can't|cannot|don't|not).{0,25}"
            r"(?:sleep|asleep)|(?:sleep|asleep).{0,25}(?:problem|difficult|terrible|poorly|badly)|"
            r"\binsomnia\b", a
        ))
        deny = bool(re.search(
            r"(?:no|don't have any).{0,12}(?:problem|trouble).{0,15}(?:sleep|asleep)|"
            r"\bsleep (?:very |pretty |really )?(?:well|good|fine|great)|"
            r"(?:easy|easily).{0,15}(?:sleep|asleep)", a
        ))
        if re.search(r"how easy.*sleep", q) and re.fullmatch(
            r"(?:it'?s |it's |very |pretty |really |quite )*(?:easy|hard|difficult)[ .!]*", a
        ):
            support = bool(re.search(r"hard|difficult", a))
            deny = "easy" in a
    else:
        support = bool(re.search(
            r"(?:no|lost|losing).{0,12}interest|"
            r"(?:don't|do not|can't|cannot|no longer).{0,12}enjoy|"
            r"nothing.{0,12}(?:fun|enjoyable)", a
        ))
        deny = bool(re.search(
            r"still enjoy|haven't lost interest|have not lost interest", a
        ))
    if support and deny:
        return "unknown", "conflicting_lexical_matches"
    if support:
        return "support", "explicit_support_phrase"
    if deny:
        return "deny", "explicit_denial_phrase"
    return "unknown", "no_high_precision_rule"


def build_qa(rows: list[dict], pid: int, split: str) -> list[dict]:
    """Group consecutive interviewer and participant rows, preserving time spans."""
    units: list[dict] = []
    prompts: list[dict] = []
    responses: list[dict] = []

    def flush() -> None:
        if not responses:
            return
        q = " ".join(normalize_question(r["value"]) for r in prompts)
        a = " ".join(r["value"].strip() for r in responses)
        spans = [[float(r["start_time"]), float(r["stop_time"])] for r in responses]
        for start, stop in spans:
            if start < 0 or stop < start:
                raise ValueError(f"Invalid transcript interval for participant {pid}")
        category = question_category(q)
        domain = category if category in {"sleep", "interest"} else "none"
        if domain == "none":
            if re.search(r"\bsleep|\basleep|\binsomnia", a.lower()):
                domain = "sleep"
            elif re.search(r"\binterest\b|\benjoy", a.lower()):
                domain = "interest"
        label, reason = weak_report_label(q, a, domain)
        scrubbed = bool(re.search(r"scrubbed_entry|\bxxx\b", a, re.I))
        overlap = any(b[0] < x[1] for x, b in zip(spans, spans[1:]))
        prompt_overlap = bool(prompts and spans[0][0] < float(prompts[-1]["stop_time"]))
        units.append({
            "participant_id": pid, "split": split, "qa_id": f"{pid}_{len(units):04d}",
            "question": q, "answer": a, "question_category": category, "domain": domain,
            "weak_label": label, "weak_label_reason": reason,
            "label_provenance": "automated_lexical_instrument_not_independent_gold",
            "start": spans[0][0], "stop": max(s[1] for s in spans),
            "duration": sum(stop - start for start, stop in spans),
            "word_count": len(tokens(a)), "participant_spans": spans,
            "q_start": float(prompts[0]["start_time"]) if prompts else None,
            "q_stop": float(prompts[-1]["stop_time"]) if prompts else None,
            "source_rows": [r["source_row"] for r in responses],
            "scrubbed": scrubbed,
            "quality": {"scrubbed": scrubbed, "overlap": overlap or prompt_overlap},
        })

    for row in rows:
        if row["speaker"] == "Ellie":
            if responses:
                flush()
                responses, prompts = [], []
            prompts.append(row)
        elif row["speaker"] == "Participant":
            responses.append(row)
    flush()
    for i, unit in enumerate(units):
        unit["position"] = i / max(1, len(units) - 1)
    return units


def split_ids(root: Path) -> dict[str, list[int]]:
    result = {}
    for split in ("train", "dev", "test"):
        path = root / f"{split}_split_Depression_AVEC2017.csv"
        with path.open() as stream:
            reader = csv.DictReader(stream)
            key = next(k for k in (reader.fieldnames or []) if k.lower() == "participant_id")
            ids = [int(row[key]) for row in reader if row.get(key, "").strip()]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate participant in {split}")
        result[split] = sorted(ids)
    if any(set(result[a]) & set(result[b]) for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("Participant overlap between official splits")
    return result


def require_private_output(path: Path) -> Path:
    private = Path(__file__).resolve().parents[1] / "Data"
    resolved = path.resolve()
    if not resolved.is_relative_to(private):
        raise ValueError("Participant outputs must stay under the repository Data directory")
    return resolved


def prepare(root: Path, output: Path) -> dict:
    output = require_private_output(output)
    output.mkdir(parents=True, exist_ok=True)
    ids = split_ids(root)
    expected = set(range(300, 493)) - {342, 394, 398, 460}
    if set().union(*map(set, ids.values())) != expected:
        raise ValueError("Official participant manifest differs from documented release")
    records: list[dict] = []
    manifest: list[dict] = []
    for split, participants in ids.items():
        for pid in participants:
            path = root / f"{pid}_P.zip"
            reason = (
                "known_invalid_zip" if pid in INVALID_ARCHIVES else
                "missing_interviewer" if pid in NO_INTERVIEWER else
                "prior_example_exposure" if pid in PRIOR_EXPOSURE else
                "sealed_test" if split == "test" else "included"
            )
            if not path.exists():
                reason = "missing_zip"
            entry = {"participant_id": pid, "split": split, "status": reason}
            if path.exists():
                stat = path.stat()
                entry.update(zip_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            # Crucially, do not even open any test archive here.
            if reason == "included":
                with zipfile.ZipFile(path) as archive:
                    member = f"{pid}_TRANSCRIPT.csv"
                    if member not in archive.namelist():
                        raise ValueError(f"Missing exact transcript member: {member}")
                    raw = archive.read(member)
                reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t")
                required = {"start_time", "stop_time", "speaker", "value"}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f"Unexpected transcript schema: {pid}")
                rows = [dict(row, source_row=i + 2) for i, row in enumerate(reader)]
                qa = build_qa(rows, pid, split)
                if not any(row["speaker"] == "Ellie" for row in rows):
                    raise ValueError(f"Unrecorded missing interviewer: {pid}")
                entry.update(qa_count=len(qa), transcript_sha256=hashlib.sha256(raw).hexdigest())
                records.extend(qa)
            manifest.append(entry)
    qa_path = output / "qa.jsonl"
    qa_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pilot_ids = sorted(random.Random(SEED).sample(
        [r["participant_id"] for r in manifest if r["split"] == "train" and r["status"] == "included"], 20
    ))
    (output / "pilot_participants.json").write_text(json.dumps(pilot_ids))
    by_split = {}
    for split in ("train", "dev"):
        subset = [r for r in records if r["split"] == split]
        by_split[split] = {
            "participants": len({r["participant_id"] for r in subset}), "qa": len(subset),
            "target_qa": sum(r["domain"] != "none" for r in subset),
            "weak_label_counts": {label: sum(r["weak_label"] == label and r["domain"] != "none" for r in subset)
                                  for label in ("support", "deny", "unknown")},
        }
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")}
    summary = {
        "scope": "train_dev_feasibility_only", "test_transcripts_read": 0,
        "independent_gold_available": False, "seed": SEED, "splits": by_split,
        "pilot_participants": len(pilot_ids), "python": platform.python_version(),
        "packages": versions, "qa_sha256": hashlib.sha256(qa_path.read_bytes()).hexdigest(),
        "warning": "Weak labels are lexical instrument outputs, not clinical truth; test is sealed.",
    }
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2))
    export_annotation(records, pilot_ids, output)
    return summary


def export_annotation(records: list[dict], pilot_ids: list[int], output: Path) -> None:
    """Two independently ordered BLANK worksheets; no weak labels or PHQ supplied."""
    selected = []
    rng = random.Random(SEED)
    for pid in pilot_ids:
        rows = [r for r in records if r["participant_id"] == pid and not r["scrubbed"]]
        targets = [r for r in rows if r["domain"] != "none"]
        controls = [r for r in rows if r["domain"] == "none"]
        selected.extend(rng.sample(targets, min(6, len(targets))))
        selected.extend(rng.sample(controls, min(6, len(controls))))
    fields = ["qa_id", "participant_id", "question", "answer", "previous_qa", "next_qa",
              "human_symptom", "human_report_state", "human_time", "human_experiencer",
              "human_duration", "human_frequency", "human_function", "human_evidence_spans", "notes"]
    by_pid = {pid: [r for r in records if r["participant_id"] == pid] for pid in pilot_ids}
    for reviewer, seed in (("reviewer_1", 101), ("reviewer_2", 202)):
        review_path = output / f"pilot_{reviewer}.csv"
        # A repeat run must not erase annotation entered by a real reviewer.
        if review_path.exists():
            with review_path.open() as existing:
                if any(any(value.strip() for key, value in row.items()
                           if (key.startswith("human_") or key == "notes") and value)
                       for row in csv.DictReader(existing)):
                    continue
        ordered = selected[:]
        random.Random(seed).shuffle(ordered)
        with review_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for r in ordered:
                row = {key: r[key] for key in ("qa_id", "participant_id", "question", "answer")}
                corpus = by_pid[r["participant_id"]]
                index = next(i for i, x in enumerate(corpus) if x["qa_id"] == r["qa_id"])
                def context(x: dict) -> str:
                    return x["qa_id"] + " Q: " + x["question"] + " A: " + x["answer"]
                row["previous_qa"] = context(corpus[index - 1]) if index else ""
                row["next_qa"] = context(corpus[index + 1]) if index + 1 < len(corpus) else ""
                writer.writerow(row)
    (output / "annotation_readme.md").write_text(
        "# 독립 주석 파일럿\n\n두 평가자는 서로의 파일·자동 라벨·PHQ·모델 예측을 보지 않고 독립 작성합니다.\n"
        "support / deny / no_evidence / conflict; 현재성·경험 주체는 별도 필드입니다.\n"
        "기간·빈도·기능: observed / insufficient / conflict 및 실제 근거 span을 기록합니다.\n"
        "질문 자체를 참가자의 사실로 취급하지 말고, 미언급과 명시적 부인을 구분하세요.\n"
        "파일의 앞뒤 맥락은 평가자용 참고입니다. 평가할 입력 범위를 고정한 후 그 범위의 gold를 작성해야 합니다.\n"
        "C의 전체 coverage gold는 이 anchor 표집만으로 만들 수 없습니다. 전체 면담 주석이 추가로 필요합니다.\n"
        "원본 CSV는 로컬 보호 자료이며 공개 저장소나 외부 API로 보내지 않습니다.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", type=Path, default=Path("Data/DAIC-WOZ"))
    parser.add_argument("--output-dir", type=Path, default=Path("Data/experiments/2026-09-27"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.zip_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
