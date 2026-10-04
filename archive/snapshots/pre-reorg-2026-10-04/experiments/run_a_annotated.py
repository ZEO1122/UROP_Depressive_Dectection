"""Participant-held-out Experiment A using AI provisional annotations, not clinical gold."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from .prepare import require_private_output, split_ids
from .run_a import relation_features

LABELS = ("support", "deny", "no_evidence", "conflict")
DOMAINS = ("sleep", "interest")
TASKS = {
    "anchor_report": ("anchor", "anchor_state"),
    "window_report": ("window", "state"),
    "window_current_self": ("window", "current_self_state"),
}
MODELS = ("always_no_evidence", "train_majority", "qa_text", "qa_relation")
SEED = 20260927


def view_text(source: dict, scope: str) -> str:
    parts = ("anchor",) if scope == "anchor" else ("previous", "anchor", "next")
    # Only raw text enters features. No participant ID, rationale, evidence or labels.
    return " ".join(f"{part} question {source[part + '_question']} answer {source[part + '_answer']}"
                    for part in parts)


def view_relations(source: dict, scope: str) -> dict:
    parts = ("anchor",) if scope == "anchor" else ("previous", "anchor", "next")
    result = {}
    for part in parts:
        values = relation_features([{"question": source[part + "_question"], "answer": source[part + "_answer"]}])[0]
        result.update({part + ":" + key: value for key, value in values.items()})
    return result


class ScopedModel:
    def __init__(self, scope: str, relationships: bool):
        self.scope = scope
        self.relationships = relationships
        self.text = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=1)
        self.relations = DictVectorizer()
        self.model = LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=SEED)

    def fit(self, rows: list[dict], targets: list[str]):
        features = self.text.fit_transform([view_text(r["source"], self.scope) for r in rows])
        if self.relationships:
            features = hstack([features, self.relations.fit_transform([view_relations(r["source"], self.scope) for r in rows])])
        self.model.fit(features, targets)
        return self

    def predict(self, rows: list[dict]) -> list[str]:
        features = self.text.transform([view_text(r["source"], self.scope) for r in rows])
        if self.relationships:
            features = hstack([features, self.relations.transform([view_relations(r["source"], self.scope) for r in rows])])
        return self.model.predict(features).tolist()


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def scores(truth: list[str], prediction: list[str]) -> dict:
    cm = confusion_matrix(truth, prediction, labels=LABELS)
    support = cm.sum(axis=1)
    den = support + cm.sum(axis=0)
    f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(len(LABELS)), where=den > 0)
    represented = support > 0
    target, predicted = np.asarray(truth), np.asarray(prediction)
    no_evidence = target == "no_evidence"
    explicit = np.isin(target, ["support", "deny"])
    return {
        "n": len(truth), "accuracy": float((target == predicted).mean()),
        "macro_f1_represented_labels": float(f1[represented].mean()),
        "represented_labels": [label for label, present in zip(LABELS, represented) if present],
        "untested_labels": [label for label, present in zip(LABELS, represented) if not present],
        "macro_f1_fixed_3_without_conflict": float(f1[:3].mean()),
        "confusion_labels": list(LABELS), "confusion_matrix": cm.tolist(),
        "recall": {label: safe_ratio(int(cm[i, i]), int(support[i])) for i, label in enumerate(LABELS)},
        "unsupported_assertion_rate": safe_ratio(int(np.sum(no_evidence & np.isin(predicted, ["support", "deny", "conflict"]))), int(no_evidence.sum())),
        "unnecessary_no_evidence_rate": safe_ratio(int(np.sum(explicit & (predicted == "no_evidence"))), int(explicit.sum())),
        "target_counts": dict(Counter(truth)),
    }


def load_annotations(directory: Path, zip_dir: Path) -> tuple[list[dict], dict]:
    raw = (directory / "ai_annotations.jsonl").read_bytes()
    provenance = json.loads((directory / "provenance.json").read_text())
    if hashlib.sha256(raw).hexdigest() != provenance["outputs_sha256"]["ai_annotations.jsonl"]:
        raise ValueError("Annotations changed since validated draft; revalidate before experiment")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    ids = split_ids(zip_dir)
    train_ids = set(ids["train"])
    if len(rows) != provenance["n_anchors"] or len({r["qa_id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate annotation anchors")
    for row in rows:
        pid = int(row["participant_id"])
        row["participant_id"] = str(pid)
        if int(row["qa_id"].split("_")[0]) != pid or int(row["source"]["participant_id"]) != pid:
            raise ValueError("Anchor/source participant mismatch")
        if pid not in train_ids:
            raise ValueError("Only annotated official train participants may enter this exploratory CV")
        if row["annotation_origin"] != "AI_PROVISIONAL_DRAFT" or row["scope"] != "anchor_plus_previous_and_next_QA":
            raise ValueError("Unexpected annotation provenance/scope")
        for part in ("previous", "next"):
            context_id = row["source"][part + "_qa_id"]
            if context_id and int(context_id.split("_")[0]) != pid:
                raise ValueError("Context crosses participant boundary")
        for domain in DOMAINS:
            for _, label_field in TASKS.values():
                if row[domain][label_field] not in LABELS:
                    raise ValueError("Unknown annotation class")
    if len({r["participant_id"] for r in rows}) < 2:
        raise ValueError("Participant-held-out evaluation requires at least two participants")
    return rows, provenance


def grouped_oof(rows: list[dict], domain: str, task: str) -> tuple[dict, list[dict]]:
    scope, field = TASKS[task]
    people = sorted({r["participant_id"] for r in rows}, key=int)
    outputs: dict[str, list] = {name: [None] * len(rows) for name in MODELS}
    fold_log = []
    for person in people:
        fit_indices = [i for i, row in enumerate(rows) if row["participant_id"] != person]
        hold_indices = [i for i, row in enumerate(rows) if row["participant_id"] == person]
        fit_rows, held_rows = [rows[i] for i in fit_indices], [rows[i] for i in hold_indices]
        targets = [row[domain][field] for row in fit_rows]
        counts = Counter(targets)
        majority = sorted(counts, key=lambda label: (-counts[label], LABELS.index(label)))[0]
        fold = {
            "held_participant": person, "train_participants": sorted({r["participant_id"] for r in fit_rows}, key=int),
            "train_qa": len(fit_rows), "held_qa": len(held_rows), "train_class_counts": dict(counts),
            "missing_training_classes": sorted(set(LABELS) - set(targets)),
            "constant_fallback": len(counts) < 2,
        }
        assert person not in fold["train_participants"]
        for name in MODELS:
            if name == "always_no_evidence":
                predictions = ["no_evidence"] * len(held_rows)
            elif name == "train_majority" or len(counts) < 2:
                predictions = [majority] * len(held_rows)
            else:
                model = ScopedModel(scope, name == "qa_relation").fit(fit_rows, targets)
                predictions = model.predict(held_rows)
            for index, predicted in zip(hold_indices, predictions):
                outputs[name][index] = predicted
        fold_log.append(fold)
    assert all(all(value in LABELS for value in pred) for pred in outputs.values())
    return outputs, fold_log


def paired_intervals(rows: list[dict], truth: list[str], predictions: dict, repeats: int = 2000) -> dict:
    """Descriptive participant bootstrap on fixed OOF predictions; no model refitting."""
    groups = sorted({r["participant_id"] for r in rows}, key=int)
    mask = np.array([label in set(truth) for label in LABELS])
    matrices = {}
    for name in ("qa_text", "qa_relation"):
        matrices[name] = np.array([
            confusion_matrix([truth[i] for i, row in enumerate(rows) if row["participant_id"] == group],
                             [predictions[name][i] for i, row in enumerate(rows) if row["participant_id"] == group], labels=LABELS)
            for group in groups
        ])
    def macro(cm):
        den = cm.sum(axis=0) + cm.sum(axis=1)
        return np.divide(2 * np.diag(cm), den, out=np.zeros(len(LABELS)), where=den > 0)[mask].mean()
    rng = np.random.default_rng(SEED)
    differences, missing_class_resamples = [], 0
    for _ in range(repeats):
        selected = rng.integers(0, len(groups), size=len(groups))
        base, extra = matrices["qa_text"][selected].sum(axis=0), matrices["qa_relation"][selected].sum(axis=0)
        missing_class_resamples += int(np.any((base.sum(axis=1) == 0) & mask))
        differences.append(macro(extra) - macro(base))
    return {
        "comparison": "qa_relation minus qa_text", "point": float(macro(matrices["qa_relation"].sum(axis=0)) - macro(matrices["qa_text"].sum(axis=0))),
        "ci95_descriptive": np.quantile(differences, [.025, .975]).tolist(), "repetitions": repeats,
        "resamples_missing_a_represented_class": missing_class_resamples,
        "class_set_fixed_to_full_OOF_truth": True,
        "interpretation": "Conditional on fitted OOF predictions and AI labels; not nested-refit uncertainty or clinical significance",
    }


def transition_scores(rows: list[dict], domain: str, anchor_prediction: list[str], window_prediction: list[str]) -> dict:
    before = np.array([r[domain]["anchor_state"] for r in rows])
    after = np.array([r[domain]["state"] for r in rows])
    correct = (before == np.asarray(anchor_prediction)) & (after == np.asarray(window_prediction))
    changed = before != after
    return {
        "changed_label_pairs": int(changed.sum()), "unchanged_label_pairs": int((~changed).sum()),
        "changed_pairs_both_correct": safe_ratio(int((correct & changed).sum()), int(changed.sum())),
        "unchanged_pairs_both_correct": safe_ratio(int((correct & ~changed).sum()), int((~changed).sum())),
        "interpretation": "Paired input-scope expansion; not a causal patient change or proof of context improvement from F1 differences",
    }


def followup_readiness(rows: list[dict], annotation_dir: Path) -> dict:
    """Audit B/C prerequisites only; do not fit them using incomplete new labels."""
    base = annotation_dir.parent
    annotated = {row["qa_id"]: row for row in rows}
    result: dict = {}
    coverage_path = base / "results_b/private_coverage.csv"
    if coverage_path.exists():
        with coverage_path.open() as stream:
            available = [r for r in csv.DictReader(stream) if r["qa_id"] in annotated and r["matched"] == "True"]
        result["B"] = {
            "scope": "intersection_with_previously_extracted_B_candidate_pool_only_not_new_extraction",
            "matched_annotated_qa": len(available),
            "matched_participants": len({r["participant_id"] for r in available}),
            "anchor_label_counts": {domain: dict(Counter(annotated[r["qa_id"]][domain]["anchor_state"] for r in available)) for domain in DOMAINS},
            "status": "not_refit_reference_protocol_and_sampling_need_review",
            "input_sha256": hashlib.sha256(coverage_path.read_bytes()).hexdigest(),
        }
    else:
        result["B"] = {"status": "previous_feature_coverage_not_available"}
    manifest_path = base / "manifest.json"
    if manifest_path.exists():
        people = {str(row["participant_id"]) for row in rows}
        manifest = json.loads(manifest_path.read_text())
        relevant = [r for r in manifest if str(r["participant_id"]) in people]
        result["C"] = {
            "participants": len(people), "full_interview_qa": sum(r.get("qa_count", 0) for r in relevant),
            "annotated_anchor_qa": len(rows),
            "status": "not_evaluated_no_exhaustive_full_interview_evidence_annotation",
            "note": "Context windows supply additional text but do not certify complete interview-level gold coverage.",
            "input_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
    else:
        result["C"] = {"status": "full_interview_manifest_not_available"}
    return result


def run(annotation_dir: Path, zip_dir: Path, output_dir: Path) -> dict:
    output_dir = require_private_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, provenance = load_annotations(annotation_dir, zip_dir)
    config = {
        "scope": "A_only_AI_draft_exploration", "evaluation": "leave_one_participant_out_on_official_train_only",
        "hyperparameter_search": False, "C": 1, "class_weight": "balanced", "seed": SEED,
        "models": MODELS, "tasks": TASKS, "labels": LABELS,
        "python_version": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
        "tfidf": {"ngram_range": [1, 2], "max_features": 30000, "fit_scope": "training_participants_in_each_fold_only"},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "relation_features_source_sha256": hashlib.sha256(Path(__file__).with_name("run_a.py").read_bytes()).hexdigest(),
        "annotation_sha256": provenance["outputs_sha256"]["ai_annotations.jsonl"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Analysis configuration saved before fitting, after inspecting annotation class counts; not a preregistration.",
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    report = {
        "status": "completed_AI_draft_agreement_not_independent_gold", "official_test_opened": False,
        "official_dev_used": False, "participants": len({r["participant_id"] for r in rows}), "anchors": len(rows),
        "annotation_sha256": config["annotation_sha256"], "tasks": {}, "context_transitions": {},
        "limitations": [
            "AI investigators also designed the methods; agreement is not independent clinical accuracy.",
            "Small pilot sample; repeated/overlapping windows can share evidence within a participant.",
            "Report class-participant counts and missing/rare classes; high accuracy need not imply positive-case detection.",
            "Anchor and window targets are different; direct F1 subtraction across scopes is not a context effect.",
            "Do not compare with earlier weak-label train/dev F1: labels, participants, and validation protocol differ.",
            "No answer-only gold or independently validated question mutations: neither is evaluated here.",
            "B needs a new reference-window feasibility protocol; C needs full-interview annotations. Neither is relabeled as complete by this run.",
        ],
    }
    all_predictions: list[dict] = []
    all_folds: list[dict] = []
    predictions_lookup: dict = {}
    for domain in DOMAINS:
        for task, (_, label_field) in TASKS.items():
            print(f"A AI draft: {domain} / {task} / participant-held-out fitting", flush=True)
            truth = [row[domain][label_field] for row in rows]
            predictions, folds = grouped_oof(rows, domain, task)
            key = domain + "/" + task
            predictions_lookup[key] = predictions
            class_people = {label: len({r["participant_id"] for r in rows if r[domain][label_field] == label}) for label in LABELS}
            report["tasks"][key] = {
                "target_field": label_field, "class_participants": class_people,
                "models": {name: scores(truth, pred) for name, pred in predictions.items()},
                "relation_difference": paired_intervals(rows, truth, predictions),
                "constant_fallback_folds": sum(f["constant_fallback"] for f in folds),
                "reliable_efficacy_claim": False,
            }
            all_folds.extend({**fold, "domain": domain, "task": task} for fold in folds)
            for i, row in enumerate(rows):
                all_predictions.append({"qa_id": row["qa_id"], "participant_id": row["participant_id"],
                                        "domain": domain, "task": task, "ai_draft_target": truth[i],
                                        "predictions": {name: pred[i] for name, pred in predictions.items()}})
        report["context_transitions"][domain] = {
            name: transition_scores(rows, domain, predictions_lookup[domain + "/anchor_report"][name], predictions_lookup[domain + "/window_report"][name])
            for name in MODELS
        }
    for filename, entries in (("private_oof_predictions.jsonl", all_predictions), ("private_folds.jsonl", all_folds)):
        (output_dir / filename).write_text("".join(json.dumps(r) + "\n" for r in entries))
    report["followup_readiness"] = followup_readiness(rows, annotation_dir)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    (output_dir / "results_ko.md").write_text(render_report(report))
    return report


def render_report(report: dict) -> str:
    def value(number):
        return "계산 불가" if number is None else f"{number:.4f}"
    lines = ["# AI 잠정 주석을 이용한 A 탐색 실험", "",
             f"공식 train의 {report['participants']}명·{report['anchors']} QA에 대해 참가자 1명씩 제외하는 교차검증을 실행했다. 각 참가자의 예측은 그 사람의 자료를 학습하지 않은 모델에서만 얻었다.", "",
             "**아래는 AI 초안과의 일치도이며 임상 정확도·독립 전문가 평가가 아니다.** 공식 dev/test는 이번 실행에 사용하지 않았다. 모든 특징은 원문에서만 만들고 주석 설명·근거 span·환자 ID는 입력하지 않았다.", "",
             "## 수면 문제", "",
             "Macro-F1은 각 목표의 실제 등장 라벨 평균이다. 없는 class의 능력은 평가할 수 없다. 목표별 represented_labels·untested_labels 및 고정 3-class F1은 report.json에 별도 기록했다.", "",
             "| 입력·목표 | 항상 근거 없음 | Q+A 텍스트 | 관계 특징 추가 | 지지 recall(텍스트/관계) | 부인 recall(텍스트/관계) |",
             "|---|---:|---:|---:|---|---|"]
    titles = {"anchor_report": "해당 QA / 보고 근거 상태", "window_report": "3-QA / 보고 근거 상태", "window_current_self": "3-QA / 현재 본인 상태"}
    for task in TASKS:
        data = report["tasks"]["sleep/" + task]["models"]
        lines.append(f"| {titles[task]} | {value(data['always_no_evidence']['macro_f1_represented_labels'])} | {value(data['qa_text']['macro_f1_represented_labels'])} | {value(data['qa_relation']['macro_f1_represented_labels'])} | {value(data['qa_text']['recall']['support'])} / {value(data['qa_relation']['recall']['support'])} | {value(data['qa_text']['recall']['deny'])} / {value(data['qa_relation']['recall']['deny'])} |")
    lines.extend(["", "같은 행 안에서만 모델을 비교한다. 해당 QA와 3-QA의 정답이 달라 행 사이 F1 차이를 맥락 효과로 해석하지 않는다.", "",
                  "| 목표 | 관계 특징−텍스트 Macro-F1 | 95% 기술적 CI |",
                  "|---|---:|---|"])
    for task in TASKS:
        diff = report["tasks"]["sleep/" + task]["relation_difference"]
        lines.append(f"| {task} | {diff['point']:+.4f} | [{diff['ci95_descriptive'][0]:+.4f}, {diff['ci95_descriptive'][1]:+.4f}] |")
    lines.extend(["", "CI는 OOF 예측을 고정한 참가자 bootstrap 2,000회의 기술적 구간이다. 모델 재학습을 포함한 전체 불확실성이나 임상적 유의성을 보장하지 않는다.", "",
                  "## 흥미 저하", "",
                  "| 목표 | 지지 QA / 지지 참가자 | Q+A 지지 recall | 관계 특징 지지 recall | 단일 class 학습으로 대체한 fold |",
                  "|---|---|---:|---:|---:|"])
    for task in TASKS:
        data = report["tasks"]["interest/" + task]
        count = data["models"]["qa_text"]["target_counts"].get("support", 0)
        lines.append(f"| {task} | {count} / {data['class_participants']['support']} | {value(data['models']['qa_text']['recall']['support'])} | {value(data['models']['qa_relation']['recall']['support'])} | {data['constant_fallback_folds']} |")
    lines.extend(["", "지지 사례가 특정 참가자에게 집중되면 그 사람을 제외한 fold에서 해당 class를 학습할 수 없다. 학습 class가 하나뿐일 때만 다수 class로 대체한다. 위 표의 실제 분포와 대체 fold 수를 확인하고, 높은 전체 정답률을 흥미 저하 탐지 성공으로 해석하지 않는다.", "",
                  "## 문맥을 추가했을 때 기대되는 판단 변화", "",
                  "| 영역 / 모델 | 라벨이 달라진 쌍 수 | 변경쌍의 양쪽 모두 정답 | 유지쌍의 양쪽 모두 정답 |",
                  "|---|---:|---:|---:|"])
    for domain, models in report["context_transitions"].items():
        for name in ("qa_text", "qa_relation"):
            data = models[name]
            lines.append(f"| {domain} / {name} | {data['changed_label_pairs']} | {value(data['changed_pairs_both_correct'])} | {value(data['unchanged_pairs_both_correct'])} |")
    lines.extend(["", "## B·C의 현재 준비 상태", ""])
    readiness = report.get("followup_readiness", {})
    b_info, c_info = readiness.get("B", {}), readiness.get("C", {})
    if "matched_participants" in b_info:
        lines.append(f"- B: 기존 B 유효 특징 후보와 새 주석의 교집합은 {b_info['matched_participants']}명·{b_info['matched_annotated_qa']} QA다. 전체 새 주석에 대해 특징을 다시 추출한 것이 아니다. 라벨 분포: {b_info['anchor_label_counts']}.")
    if "full_interview_qa" in c_info:
        lines.append(f"- C: 이 참가자들의 전체 면담에는 {c_info['full_interview_qa']} QA가 있고 주석 anchor는 {c_info['annotated_anchor_qa']}개다. 제공된 주변 문맥이 있더라도 전체 면담 gold의 완전성을 인증할 수 없으므로 coverage 본평가는 수행하지 않았다.")
    lines.extend(["", "## 범위와 다음 단계", ""])
    lines.extend("- " + item for item in report["limitations"])
    lines.extend(["", "B의 참조 구간 표본 부족과 C의 전체 면담 gold 부족은 이번 주석 일부를 연결하는 것으로 해결되지 않는다. 본 실행은 A의 자연 QA 상태 판별 단계만 수행했으며 질문 변형의 기대 반응은 별도 검증 전이다.", "",
                  "## 재현", "", "```sh",
                  "python3 -m experiments.run_a_annotated --annotation-dir Data/experiments/2026-09-27/ai_annotation_draft --zip-dir Data/DAIC-WOZ --output-dir Data/experiments/2026-09-27/ai_annotated_a",
                  "```", "", "설정·코드/주석 hash는 run_config.json, 전체 지표는 report.json, 참가자별 예측·분할은 private_*.jsonl에 있다."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, default=Path("Data/experiments/2026-09-27/ai_annotation_draft"))
    parser.add_argument("--zip-dir", type=Path, default=Path("Data/DAIC-WOZ"))
    parser.add_argument("--output-dir", type=Path, default=Path("Data/experiments/2026-09-27/ai_annotated_a"))
    args = parser.parse_args()
    result = run(args.annotation_dir, args.zip_dir, args.output_dir)
    print(json.dumps({"status": result["status"], "participants": result["participants"], "anchors": result["anchors"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
