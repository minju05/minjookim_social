from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def summarize_predictions(records, predictions) -> dict[str, object]:
    by_type_total = defaultdict(int)
    by_type_correct = defaultdict(int)

    for record, prediction in zip(records, predictions):
        by_type_total[record.question_type] += 1
        if prediction.correct:
            by_type_correct[record.question_type] += 1

    summary = {
        "num_questions": len(predictions),
        "overall_accuracy": (sum(p.correct for p in predictions) / len(predictions)) if predictions else 0.0,
        "accuracy_by_type": {},
    }
    for question_type, total in sorted(by_type_total.items()):
        summary["accuracy_by_type"][question_type] = by_type_correct[question_type] / total
    return summary


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
