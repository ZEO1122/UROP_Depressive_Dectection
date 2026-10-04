"""Paper-feature preparation for the pinned, repaired public HiQuE model.

No PHQ labels are used here. Manual transcripts replace the upstream ASR route.
All outputs, weights, cached source and participant text remain under local Data/ or .tmp/.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from .prepare import normalize_question, require_private_output, split_ids

EXCLUDE = {300: "prior_case_exposure", 440: "invalid_zip", 451: "missing_interviewer",
           458: "missing_interviewer", 480: "missing_interviewer"}


def normalize(text: str) -> str:
    text = normalize_question(text).lower().replace("’", "'").replace("l_a", "la")
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return " ".join(text.split())


def strip_prefix(text: str) -> str:
    text = normalize(text)
    match = re.search(r"\b(how|what|where|when|why|who|do|did|does|are|is|can|could|have|has|tell|would|were|describe)\b", text)
    return text[match.start():] if match else text


def is_question(text: str, canonical: dict) -> bool:
    text = strip_prefix(text)
    return text in canonical or bool(re.match(r"^(?:how|what|where|when|why|who|do|did|does|are|is|can|could|have|has|tell|would|were|describe)\b", text))


def subtract_intervals(span: list[float], removed: list[list[float]]) -> list[list[float]]:
    kept = [span]
    for left, right in removed:
        parts = []
        for start, end in kept:
            if right <= start or left >= end:
                parts.append([start, end])
            else:
                if start < left:
                    parts.append([start, left])
                if right < end:
                    parts.append([right, end])
        kept = parts
    return [[a, b] for a, b in kept if b > a]


def question_table(html_path: Path) -> list[dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_path.read_text(), "html.parser")
    candidates = [table for table in soup.find_all("table")
                  if "how has seeing a therapist affected you" in table.get_text(" ", strip=True).lower()]
    if len(candidates) != 1:
        raise ValueError("Expected one 85-question table from paper Appendix A")
    result = []
    for row in candidates[0].find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        if len(cells) == 3 and re.fullmatch(r"\(\d+\)", cells[0]):
            result.append({"slot": int(cells[0][1:-1]) - 1, "question": cells[1], "type": cells[2]})
    if len(result) != 85 or sorted(r["slot"] for r in result) != list(range(85)):
        raise ValueError("Paper question list is not exactly 85 slots")
    return result


def load_encoder(model_dir: Path, threads: int = 8):
    import torch
    from transformers import RobertaModel, RobertaTokenizerFast  # type: ignore[attr-defined]
    torch.set_num_threads(threads)
    tokenizer = RobertaTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)
    model = RobertaModel.from_pretrained(str(model_dir), local_files_only=True, add_pooling_layer=False)
    model.eval()
    return tokenizer, model


def encode_tokens(texts: list[str], tokenizer, model, batch_size: int = 32):
    import torch
    result = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=96, return_tensors="pt")
            outputs = model(**batch).last_hidden_state
            outputs = torch.nn.functional.normalize(outputs, dim=-1)
            for i, mask in enumerate(batch["attention_mask"]):
                n = int(mask.sum())
                result.append(outputs[i, 1:max(2, n - 1)].cpu())
    return result


def map_questions(questions: list[str], references: list[dict], model_dir: Path, threads: int) -> dict:
    """Exact normalized match, otherwise frozen RoBERTa token-max F1 matching.

    This is explicitly a controlled mapping adaptation, not the undocumented
    upstream default BERTScore checkpoint/layer configuration.
    """
    canonical = {normalize(r["question"]): r["slot"] for r in references}
    mapped, pending = {}, []
    for question in questions:
        clean = strip_prefix(question)
        if clean in canonical:
            mapped[question] = {"slot": canonical[clean], "method": "exact", "similarity": 1.0}
        elif not is_question(question, canonical):
            mapped[question] = {"slot": None, "method": "non_question", "similarity": None}
        else:
            pending.append(question)
    if pending:
        import torch
        tokenizer, model = load_encoder(model_dir, threads)
        ref_tokens = encode_tokens([r["question"] for r in references], tokenizer, model)
        query_tokens = encode_tokens([strip_prefix(q) for q in pending], tokenizer, model)
        for question, query in zip(pending, query_tokens):
            similarities = []
            for ref in ref_tokens:
                dots = query @ ref.T
                precision, recall = float(dots.max(dim=1).values.mean()), float(dots.max(dim=0).values.mean())
                similarities.append(2 * precision * recall / max(precision + recall, 1e-12))
            index = int(np.argmax(similarities))
            mapped[question] = {"slot": index if similarities[index] >= 0.75 else None,
                                "method": "roberta_token_f1" if similarities[index] >= 0.75 else "unmapped",
                                "similarity": similarities[index]}
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return mapped


def build_slots(pid: int, split: str, rows: list[dict], mapping: dict) -> dict:
    slots: dict[int, dict] = {}
    current = None
    counts: Counter[str] = Counter()
    interviewer = [[float(r["start_time"]), float(r["stop_time"])] for r in rows if r["speaker"] == "Ellie"]
    for row in rows:
        start, stop = float(row["start_time"]), float(row["stop_time"])
        if start < 0 or stop <= start:
            counts["invalid_interval"] += 1
            continue
        if row["speaker"] == "Ellie":
            item = mapping[normalize_question(row["value"])]
            counts[item["method"]] += 1
            if item["method"] != "non_question":
                current = item["slot"]
                if current is not None:
                    slots.setdefault(current, {"slot": current, "text": "", "spans": [], "question_count": 0})
                    slots[current]["question_count"] += 1
        elif row["speaker"] == "Participant" and current is not None:
            if re.search(r"scrubbed_entry|\bxxx\b", row["value"], flags=re.I):
                counts["scrubbed_response_rows"] += 1
                continue
            spans = subtract_intervals([start, stop], interviewer)
            if not spans:
                counts["fully_overlapped_response_rows"] += 1
                continue
            slots[current]["text"] += " " + row["value"].strip()
            slots[current]["spans"].extend(spans)
            counts["assigned_response_rows"] += 1
    presence = [False] * 85
    for slot, item in slots.items():
        presence[slot] = item["question_count"] > 0
        item["text"] = item["text"].strip()
        # Transcript response rows can overlap each other; merge time intervals.
        merged: list[list[float]] = []
        for a, b in sorted(item["spans"]):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        item["spans"] = merged
    return {"participant_id": pid, "split": split, "slots": [slots[i] for i in sorted(slots)],
            "question_presence": presence, "mapping_counts": dict(counts)}


def prepare(zip_dir: Path, output: Path, model_dir: Path, threads: int) -> None:
    output = require_private_output(output)
    output.mkdir(parents=True, exist_ok=True)
    refs = question_table(output / "paper.html")
    (output / "questions.json").write_text(json.dumps(refs, ensure_ascii=False, indent=2))
    corpus, manifests = [], []
    questions: set[str] = set()
    # This reads only IDs from split files; PHQ targets are not used or cached here.
    for split, ids in split_ids(zip_dir).items():
        for pid in ids:
            if pid in EXCLUDE:
                manifests.append({"participant_id": pid, "split": split, "excluded": EXCLUDE[pid]})
                continue
            path = zip_dir / f"{pid}_P.zip"
            with zipfile.ZipFile(path) as archive:
                raw = archive.read(f"{pid}_TRANSCRIPT.csv")
            rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t"))
            if not any(r["speaker"] == "Ellie" for r in rows):
                raise ValueError(f"Unexpected missing interviewer {pid}")
            questions.update(normalize_question(r["value"]) for r in rows if r["speaker"] == "Ellie")
            corpus.append((pid, split, rows))
            stat = path.stat()
            manifests.append({"participant_id": pid, "split": split, "excluded": None,
                              "zip_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                              "transcript_sha256": hashlib.sha256(raw).hexdigest()})
    print(f"Preparing {len(corpus)} participant records; {len(questions)} distinct interviewer strings", flush=True)
    mapping = map_questions(sorted(questions), refs, model_dir, threads)
    (output / "question_mapping_private.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    prepared = [build_slots(pid, split, rows, mapping) for pid, split, rows in corpus]
    (output / "segments.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared))
    (output / "manifest.json").write_text(json.dumps(manifests, indent=2))
    summary = {"participants": dict(Counter(r["split"] for r in prepared)),
               "unique_question_mapping": dict(Counter(m["method"] for m in mapping.values())),
               "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "test_transcript_access": "label_free_fixed_feature_preparation_only_after_protocol_freeze",
               "mapping_adaptation": "exact then RoBERTa-base final-layer token-max F1>=0.75; no PHQ labels used",
               "aggregation": "Concatenate repeated answers per canonical question; retain source times; no truncation until encoder stage",
               "feature_window": "participant-only speech after scrubbed and interviewer-overlap removal",
               "hierarchy": "85 paper question slots; public model learned slot positions. No unavailable author hierarchy annotation fabricated."}
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def text_features(output: Path, model_dir: Path, threads: int) -> None:
    import torch
    output = require_private_output(output)
    rows = [json.loads(line) for line in (output / "segments.jsonl").read_text().splitlines()]
    tokenizer, model = load_encoder(model_dir, threads)
    indices, texts = [], []
    for i, row in enumerate(rows):
        for slot in row["slots"]:
            if slot["text"]:
                indices.append((i, slot["slot"]))
                texts.append(slot["text"])
    features = np.zeros((len(rows), 85, 768), dtype=np.float32)
    mask = np.zeros((len(rows), 85), dtype=bool)
    full_tokens = tokenizer(texts, truncation=False, add_special_tokens=True, verbose=False)
    truncations = sum(len(ids) > 512 for ids in full_tokens["input_ids"])
    tokenized = tokenizer(texts, padding=False, truncation=True, max_length=512)
    lengths = [len(ids) for ids in tokenized["input_ids"]]
    batches = length_batches(lengths)
    completed = 0
    with torch.inference_mode():
        for batch_index, chosen in enumerate(batches):
            encoded = tokenizer.pad([{key: values[i] for key, values in tokenized.items()} for i in chosen],
                                    padding=True, return_tensors="pt")
            values = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite frozen RoBERTa CLS feature")
            for original_index, value in zip(chosen, values):
                pid_index, slot = indices[original_index]
                features[pid_index, slot] = value
                mask[pid_index, slot] = True
            completed += len(chosen)
            if batch_index % 10 == 0:
                print(f"RoBERTa slots {completed}/{len(texts)}; padded length {encoded['input_ids'].shape[1]}", flush=True)
    np.savez_compressed(output / "text.npz", T=features, mask_T=mask,
                        participant_ids=np.array([r["participant_id"] for r in rows]))
    (output / "text_summary.json").write_text(json.dumps({"encoded_slots": len(texts), "truncated_at_512": truncations,
                                                        "model": json.loads((output / "roberta_provenance.json").read_text()),
                                                        "pooling": "last_hidden_state[:,0,:], no randomly initialized pooler",
                                                        "batching": "length-sorted, at most32sequences/4096paddedtokens; original slot order restored",
                                                        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                        "frozen_eval_mode": True, "feature_shape": list(features.shape)}, indent=2))


def length_batches(lengths: list[int], token_budget: int = 4096, max_batch: int = 32) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    for index in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        if current and (len(current) >= max_batch or lengths[index] * (len(current) + 1) > token_budget):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


def assemble(output: Path) -> None:
    output = require_private_output(output)
    rows = [json.loads(line) for line in (output / "segments.jsonl").read_text().splitlines()]
    text = np.load(output / "text.npz", allow_pickle=False)
    ids = np.array([r["participant_id"] for r in rows])
    if not np.array_equal(ids, text["participant_ids"]):
        raise ValueError("Text participant order differs from manifest")
    audio, visual, ma, mv = [], [], [], []
    for pid in ids:
        av = np.load(output / "av" / f"{pid}.npz", allow_pickle=False)
        audio.append(av["A"]); visual.append(av["V"])
        ma.append(av["mask_A"]); mv.append(av["mask_V"])
    np.savez_compressed(output / "features.npz", A=np.array(audio), V=np.array(visual), T=text["T"],
                        mask_A=np.array(ma), mask_V=np.array(mv), mask_T=text["mask_T"], participant_ids=ids,
                        split=np.array([r["split"] for r in rows]), question_presence=np.array([r["question_presence"] for r in rows]))
    print(f"Assembled {len(rows)} participants into features.npz", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "text", "assemble"], required=True)
    parser.add_argument("--zip-dir", type=Path, default=Path("Data/DAIC-WOZ"))
    parser.add_argument("--output-dir", type=Path, default=Path("Data/hique_reproduction"))
    parser.add_argument("--model-dir", type=Path, default=Path(".tmp/hique_weights/roberta-base"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.zip_dir, args.output_dir, args.model_dir, args.threads)
    elif args.stage == "text":
        text_features(args.output_dir, args.model_dir, args.threads)
    else:
        assemble(args.output_dir)


if __name__ == "__main__":
    main()
