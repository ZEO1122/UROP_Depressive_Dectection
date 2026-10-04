"""Validate and render AI-authored provisional annotations. Never creates human gold.

This module only assembles hand-authored semantic decisions supplied by Codex.
It does not infer annotations with keywords, models, or existing weak labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

from .prepare import require_private_output

DOMAINS = ("sleep", "interest")
STATES = {"support", "deny", "no_evidence", "conflict"}
TIMES = {"current", "past", "mixed", "unspecified", "conditional", "not_applicable"}
EXPERIENCERS = {"self", "other", "mixed", "unspecified", "not_applicable"}
SLOTS = ("duration", "frequency", "function")
SOURCES = tuple(f"{view}_{speaker}" for view in ("anchor", "previous", "next")
                for speaker in ("question", "answer"))


def no_evidence() -> dict:
    return {
        "anchor_state": "no_evidence", "state": "no_evidence", "current_self_state": "no_evidence",
        "time": "not_applicable", "experiencer": "not_applicable",
        **{slot: {"status": "insufficient", "text": "제공된 범위에서 확인되지 않음"} for slot in SLOTS},
        "evidence": [],
        "note_ko": "이 증상 영역의 지지·명시적 부인 근거를 확인하지 못함. 증상 부재를 뜻하지 않음.",
    }


def validate_record(record: dict, source: dict) -> tuple[dict, list[str]]:
    if record.get("qa_id") != source["qa_id"]:
        raise ValueError("Annotation/source QA mismatch")
    if record.get("review_priority") not in {"low", "medium", "high"}:
        raise ValueError(f"{source['qa_id']}: invalid review priority")
    if not isinstance(record.get("review_reasons"), list) or not str(record.get("rationale_ko", "")).strip():
        raise ValueError(f"{source['qa_id']}: missing rationale/review reasons")
    output = dict(record)
    warnings = []
    for domain in DOMAINS:
        if domain not in record:
            raise ValueError(f"{source['qa_id']}: missing {domain} annotation")
        value = no_evidence() if record[domain] is None else dict(record[domain])
        for field in ("anchor_state", "state", "current_self_state"):
            if value.get(field) not in STATES:
                raise ValueError(f"{source['qa_id']} {domain}: invalid {field}")
        if value.get("time") not in TIMES or value.get("experiencer") not in EXPERIENCERS:
            raise ValueError(f"{source['qa_id']} {domain}: invalid time/experiencer")
        for slot in SLOTS:
            info = value.get(slot, {})
            if info.get("status") not in {"observed", "insufficient", "conflict"} or not isinstance(info.get("text"), str):
                raise ValueError(f"{source['qa_id']} {domain}: invalid {slot}")
        if not isinstance(value.get("evidence"), list) or not value.get("note_ko"):
            raise ValueError(f"{source['qa_id']} {domain}: missing evidence/note")
        evidence = []
        for item in value["evidence"]:
            field, quote = item.get("source"), item.get("quote")
            purpose = item.get("purpose")
            if field not in SOURCES or not isinstance(quote, str) or not quote:
                raise ValueError(f"{source['qa_id']} {domain}: invalid quote source/text")
            if purpose not in {"report", "duration", "frequency", "function", "context"}:
                raise ValueError(f"{source['qa_id']} {domain}: invalid evidence purpose")
            text = source[field]
            if quote not in text:
                raise ValueError(f"{source['qa_id']} {domain}: quote not present in {field}: {quote!r}")
            start = text.index(quote)
            if text.count(quote) > 1:
                warnings.append(f"{source['qa_id']} {domain}: repeated quote in {field}; first occurrence offset stored")
            source_qa_id = source["qa_id"] if field.startswith("anchor") else source[field.split("_")[0] + "_qa_id"]
            evidence.append({**item, "source_qa_id": source_qa_id, "start_char": start,
                             "end_char": start + len(quote), "occurrences": text.count(quote)})
        participant = [item for item in evidence if item["source"].endswith("_answer")]
        if value["state"] != "no_evidence" and not participant:
            raise ValueError(f"{source['qa_id']} {domain}: affirmative/negative state lacks participant evidence")
        if value["anchor_state"] != "no_evidence" and not any(e["source"] == "anchor_answer" for e in participant):
            raise ValueError(f"{source['qa_id']} {domain}: anchor state lacks anchor-answer evidence")
        for slot in SLOTS:
            if value[slot]["status"] in {"observed", "conflict"} and not any(e["purpose"] == slot for e in participant):
                raise ValueError(f"{source['qa_id']} {domain}: {slot} lacks corresponding participant quote")
        if value["current_self_state"] != "no_evidence":
            if not participant or value["time"] in {"past", "conditional", "not_applicable"} or value["experiencer"] in {"other", "not_applicable"}:
                raise ValueError(f"{source['qa_id']} {domain}: current-self label contradicts scope metadata")
        value["evidence"] = evidence
        output[domain] = value
    output.update(participant_id=source["participant_id"], annotation_origin="AI_PROVISIONAL_DRAFT",
                  human_reviewed=False, expert_reviewed=False,
                  scope="anchor_plus_previous_and_next_QA", source=source)
    return output, warnings


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    # BOM permits common desktop spreadsheet programs to read Korean correctly.
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def assemble(directory: Path) -> dict:
    directory = require_private_output(directory)
    source_rows = read_jsonl(directory / "inputs/all_blind.jsonl")
    sources = {row["qa_id"]: row for row in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError("Duplicate input QA")
    records = []
    for shard in sorted((directory / "shards").glob("batch_*.jsonl")):
        records.extend(read_jsonl(shard))
    observed = [row["qa_id"] for row in records]
    if len(set(observed)) != len(observed) or set(observed) != set(sources):
        raise ValueError(f"Missing/extra/duplicate annotations: missing={set(sources)-set(observed)}, extra={set(observed)-set(sources)}")
    provenance = json.loads((directory / "provenance.json").read_text())
    if hashlib.sha256((directory / "annotation_guide.md").read_bytes()).hexdigest() != provenance["guide_sha256"]:
        raise ValueError("Annotation guide changed during the draft pass")
    original = directory.parent / provenance["source"]
    if hashlib.sha256(original.read_bytes()).hexdigest() != provenance["source_sha256"]:
        raise ValueError("Original human worksheet changed; inspect before assembling")
    validated, warnings = [], []
    for record in sorted(records, key=lambda row: row["qa_id"]):
        value, issues = validate_record(record, sources[record["qa_id"]])
        validated.append(value)
        warnings.extend(issues)
    (directory / "ai_annotations.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in validated))
    long_rows, wide_rows, review_rows = [], [], []
    wide_fields = ["qa_id", "participant_id", "anchor_question", "anchor_answer", "previous_qa_id", "next_qa_id"]
    domain_fields = ["anchor_state", "state", "current_self_state", "time", "experiencer"]
    for domain in DOMAINS:
        wide_fields.extend(f"ai_{domain}_{field}" for field in domain_fields)
        wide_fields.extend(f"ai_{domain}_{slot}_{field}" for slot in SLOTS for field in ("status", "text"))
        wide_fields.extend([f"ai_{domain}_evidence_json", f"ai_{domain}_note_ko"])
    wide_fields.extend(["ai_review_priority", "ai_review_reasons", "ai_rationale_ko", "annotation_origin", "human_reviewed"])
    long_fields = ["qa_id", "participant_id", "domain", "anchor_question", "anchor_answer", "previous_question", "previous_answer", "next_question", "next_answer"]
    long_fields.extend("ai_" + field for field in domain_fields)
    long_fields.extend("ai_" + slot + "_" + field for slot in SLOTS for field in ("status", "text"))
    long_fields.extend(["ai_evidence_json", "ai_note_ko", "ai_review_priority", "ai_review_reasons", "annotation_origin", "human_reviewed"])
    for row in validated:
        source = row["source"]
        common = {"qa_id": row["qa_id"], "participant_id": row["participant_id"],
                  "annotation_origin": row["annotation_origin"], "human_reviewed": False,
                  "ai_review_priority": row["review_priority"],
                  "ai_review_reasons": "; ".join(row["review_reasons"])}
        wide = {**common, **{key: source[key] for key in wide_fields[:6]}, "ai_rationale_ko": row["rationale_ko"]}
        for domain in DOMAINS:
            value = row[domain]
            item = {**common, **{key: source[key] for key in SOURCES}, "domain": domain}
            for field in domain_fields:
                item["ai_" + field] = value[field]
                wide[f"ai_{domain}_{field}"] = value[field]
            for slot in SLOTS:
                for field in ("status", "text"):
                    item[f"ai_{slot}_{field}"] = value[slot][field]
                    wide[f"ai_{domain}_{slot}_{field}"] = value[slot][field]
            item["ai_evidence_json"] = json.dumps(value["evidence"], ensure_ascii=False)
            item["ai_note_ko"] = value["note_ko"]
            wide[f"ai_{domain}_evidence_json"] = item["ai_evidence_json"]
            wide[f"ai_{domain}_note_ko"] = value["note_ko"]
            long_rows.append(item)
        wide_rows.append(wide)
        if row["review_priority"] != "low":
            review_rows.append(wide)
    write_csv(directory / "ai_annotations_199_rows.csv", wide_fields, wide_rows)
    write_csv(directory / "ai_annotations_by_domain.csv", long_fields, long_rows)
    write_csv(directory / "needs_review.csv", wide_fields, review_rows)
    summary = {
        "status": "completed_AI_provisional_draft_not_gold", "anchors": len(validated),
        "domain_rows": len(long_rows), "participants": len({r["participant_id"] for r in validated}),
        "priorities": dict(Counter(r["review_priority"] for r in validated)),
        "states": {domain: dict(Counter(r[domain]["state"] for r in validated)) for domain in DOMAINS},
        "anchor_states": {domain: dict(Counter(r[domain]["anchor_state"] for r in validated)) for domain in DOMAINS},
        "current_self_states": {domain: dict(Counter(r[domain]["current_self_state"] for r in validated)) for domain in DOMAINS},
        "slot_states": {domain: {slot: dict(Counter(r[domain][slot]["status"] for r in validated)) for slot in SLOTS} for domain in DOMAINS},
        "evidence_quotes": sum(len(r[d]["evidence"]) for r in validated for d in DOMAINS),
        "all_quotes_verified_exact": True, "warnings": warnings,
        "human_reviewed": False, "expert_reviewed": False,
        "metrics_or_model_training_performed": False,
        "ai_semantic_review_cases": provenance.get("ai_peer_review", {}).get("reviewed_cases", 0),
        "ai_review_corrections": provenance.get("ai_peer_review", {}).get("corrections", 0),
    }
    (directory / "annotation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    provenance.update(status=summary["status"], completed_utc=datetime.now(timezone.utc).isoformat(),
                      annotator_pass="four_disjoint_AI_shards_one_draft_pass",
                      assembler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      outputs_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (directory / "ai_annotations.jsonl", directory / "ai_annotations_199_rows.csv", directory / "ai_annotations_by_domain.csv")})
    (directory / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    render_readme(directory, summary)
    render_cases(directory, validated)
    return summary


def render_readme(directory: Path, summary: dict) -> None:
    lines = ["# AI 주석 초안 — 사람·전문가 정답 아님", "",
             f"Train {summary['participants']}명, {summary['anchors']}개 QA를 읽고 수면 문제와 흥미 저하를 각각 기록했습니다. 영역별 CSV는 {summary['domain_rows']}행입니다.", "",
             "- 먼저 [읽기용 전체 초안](annotated_cases_ko.md)을 보세요.",
             "- [199행 CSV](ai_annotations_199_rows.csv): 원래 표본 하나에 한 행.",
             "- [영역별 CSV](ai_annotations_by_domain.csv): 같은 QA를 수면/흥미 두 영역으로 분리, 문맥과 근거 포함.",
             "- [우선 검토 목록](needs_review.csv): medium/high 항목. 이 우선순위는 주관적 검토 순서이며 오류 확률이 아닙니다.",
             "- [구조화 JSONL](ai_annotations.jsonl): 정확한 인용문·문자 위치·출처 QA ID.", "",
             "## 라벨 읽는 법", "",
             "`state`는 앞뒤 하나씩을 포함한 3-QA 문맥, `anchor_state`는 해당 Q+A만의 판단입니다. `current_self_state`는 그 문맥에서 현재 본인에 대해 확인되는 보고 상태입니다.",
             "`support`는 보고된 호소의 근거이지 진단이 아닙니다. `deny`는 명시적 부인입니다. `no_evidence`는 자료로 확인되지 않음이며 증상 없음과 다릅니다. `conflict`는 같은 시점·같은 사람에 관한 양립 불가능한 보고입니다.",
             "기간·빈도·기능의 observed는 특정 정보를 말한 근거가 있다는 뜻이지 진단에 충분하다는 뜻이 아닙니다. 조건부 영향은 조건부로만 읽고 실제 현재 손상으로 옮기지 않습니다.", "",
             "## 검토 범위와 한계", "",
             "분할 작업자는 행별 자동 라벨·모델 예측·PHQ를 제공받지 않고 원문과 고정 지침으로 작성했습니다. 설계자인 주 에이전트는 이전 집계 결과를 이미 알고 있으므로 완전 맹검이라고 주장하지 않습니다.",
             "여러 AI가 서로 다른 묶음을 작성한 단일 초안입니다. 독립 인간 평가자 2명의 일치도나 임상 gold가 아닙니다. 모든 필드는 ai_ 또는 AI_PROVISIONAL_DRAFT로 표시하며 원래 사람용 빈 CSV는 수정하지 않았습니다.",
             "문자열 검증은 인용이 원문에 있다는 것만 보장합니다. 그 인용의 해석이 맞는지는 별도 검토가 필요합니다. 이 초안으로 모델을 재학습하거나 test를 평가하지 않았습니다.",
             f"AI 의미 검토는 {summary['ai_semantic_review_cases']}개 문맥을 대상으로 했고 {summary['ai_review_corrections']}건을 수정했습니다. 첫 초안과 수정 이력은 first_pass_shards/ 및 review_changes.json에 보존했습니다.",
             "전문가가 아닌 사용자도 발화 누락, 질문과 응답 연결, 과거/현재, 자기/타인, 직접 말하지 않은 사실의 추가 여부를 확인할 수 있습니다. 임상 의미와 충분성의 타당성은 전문가 검토가 별도로 필요합니다.", "",
             "## 집계", "",
             "아래는 겹치는 문맥 창의 라벨 건수이며 참가자 유병률이 아닙니다.", "",
             "```json", json.dumps({k: summary[k] for k in ("priorities", "states", "current_self_states", "slot_states")}, ensure_ascii=False, indent=2), "```", "",
             "이 폴더는 참가자 원문이 포함된 로컬 보호 자료입니다. 외부 공개나 노션 업로드를 하지 않습니다."]
    (directory / "README_ko.md").write_text("\n".join(lines) + "\n")


def render_cases(directory: Path, rows: list[dict]) -> None:
    names = {"sleep": "수면 문제", "interest": "흥미·즐거움 저하"}
    def literal(text: str) -> str:
        # Transcript event tags such as <laughter> must remain visible in Markdown.
        return html.escape(text, quote=False)
    lines = [f"# AI 작성 잠정 주석 — {len(rows)} QA", "", "[사용법과 한계](README_ko.md). 원문은 영어, 해석 메모는 한국어입니다. 정답·진단으로 확정한 자료가 아닙니다.", ""]
    for row in rows:
        source = row["source"]
        lines.extend([f"## {row['qa_id']} · 검토 우선순위 {row['review_priority']}", "",
                      f"**질문:** {literal(source['anchor_question'])}", "", f"**응답:** {literal(source['anchor_answer'])}", "",
                      "<details><summary>앞뒤 문맥</summary>", "",
                      f"이전 {source['previous_qa_id']} Q: {literal(source['previous_question'])}", "",
                      f"A: {literal(source['previous_answer'])}", "",
                      f"다음 {source['next_qa_id']} Q: {literal(source['next_question'])}", "",
                      f"A: {literal(source['next_answer'])}", "", "</details>", "",
                      "| 영역 | 해당 QA만 | 문맥 포함 | 현재 본인 | 시점 / 주체 | 기간 / 빈도 / 기능 |",
                      "|---|---|---|---|---|---|"])
        for domain in DOMAINS:
            value = row[domain]
            slots = " / ".join(value[s]["status"] for s in SLOTS)
            lines.append(f"| {names[domain]} | {value['anchor_state']} | {value['state']} | {value['current_self_state']} | {value['time']} / {value['experiencer']} | {slots} |")
        lines.extend(["", row["rationale_ko"], ""])
        for domain in DOMAINS:
            value = row[domain]
            if value["evidence"] or value["state"] != "no_evidence":
                lines.extend([f"**{names[domain]} 메모:** {value['note_ko']}", ""])
                for slot in SLOTS:
                    if value[slot]["text"]:
                        lines.append(f"- {slot}: {value[slot]['text']}")
                for evidence in value["evidence"]:
                    lines.append(f"- 근거 ({evidence['source_qa_id']} / {evidence['source']} / {evidence['purpose']}): “{literal(evidence['quote'])}”")
                lines.append("")
        if row["review_reasons"]:
            lines.extend(["**추가 확인:** " + "; ".join(row["review_reasons"]), ""])
    (directory / "annotated_cases_ko.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("Data/experiments/2026-09-27/ai_annotation_draft"))
    args = parser.parse_args()
    print(json.dumps(assemble(args.directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
