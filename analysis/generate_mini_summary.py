#!/usr/bin/env python3
"""Generate summary JSONs from mini full results."""

import json
from collections import defaultdict
from pathlib import Path

# Read full results
output_dir = Path("outputs")
full_jsonl = output_dir / "muma_gpt4o_mini_minju_full.jsonl"

results = []
with open(full_jsonl) as f:
    for line in f:
        results.append(json.loads(line))

# Generate predictions.json (predictions vs ground truth)
predictions = []
for r in results:
    predictions.append({
        "episode_id": r.get("episode_id"),
        "question_id": r.get("question_id"),
        "question_type": r.get("question_type"),  # belief, belief_of_goal, social_goal
        "q_variant": r.get("q_variant"),  # Q1, Q2, Q3, Q4
        "prediction": r.get("prediction", {}).get("choice_letter"),
        "ground_truth": r.get("ground_truth"),
        "correct": r.get("correct", False),
        "reasoning": r.get("prediction", {}).get("reasoning")[:100] + "..." 
                     if r.get("prediction", {}).get("reasoning") else None
    })

# Save predictions.json
with open(output_dir / "muma_gpt4o_mini_predictions.json", "w") as f:
    json.dump(predictions, f, indent=2, ensure_ascii=False)

# Generate summary.json (statistics)
total = len(results)
correct = sum(1 for r in results if r.get("correct", False))
overall_accuracy = correct / total if total > 0 else 0

# By task type
by_task = defaultdict(lambda: {"total": 0, "correct": 0})
for r in results:
    task = r.get("question_type")
    if task:
        by_task[task]["total"] += 1
        if r.get("correct"):
            by_task[task]["correct"] += 1

# By Q variant
by_q = defaultdict(lambda: {"total": 0, "correct": 0})
for r in results:
    q = r.get("q_variant")
    if q:
        by_q[q]["total"] += 1
        if r.get("correct"):
            by_q[q]["correct"] += 1

# By task + Q variant
by_task_q = defaultdict(lambda: {"total": 0, "correct": 0})
for r in results:
    task = r.get("question_type")
    q = r.get("q_variant")
    if task and q:
        key = f"{task}_{q}"
        by_task_q[key]["total"] += 1
        if r.get("correct"):
            by_task_q[key]["correct"] += 1

summary = {
    "model": "gpt-4o-mini",
    "dataset": "MuMA-ToM",
    "total_questions": total,
    "correct_predictions": correct,
    "overall_accuracy": round(overall_accuracy, 4),
    "accuracy_by_task": {
        task: round(data["correct"] / data["total"], 4) if data["total"] > 0 else 0
        for task, data in sorted(by_task.items())
    },
    "accuracy_by_q_variant": {
        q: round(data["correct"] / data["total"], 4) if data["total"] > 0 else 0
        for q, data in sorted(by_q.items())
    },
    "breakdown_by_task_q": {
        key: {
            "count": data["total"],
            "correct": data["correct"],
            "accuracy": round(data["correct"] / data["total"], 4) if data["total"] > 0 else 0
        }
        for key, data in sorted(by_task_q.items())
    }
}

# Save summary.json
with open(output_dir / "muma_gpt4o_mini_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"✓ Generated predictions.json: {len(predictions)} records")
print(f"✓ Generated summary.json: overall accuracy {summary['overall_accuracy']:.1%}")
print(f"\n📊 Summary:")
print(f"  Task breakdown: {summary['accuracy_by_task']}")
print(f"  Q breakdown: {summary['accuracy_by_q_variant']}")
