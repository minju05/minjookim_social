#!/usr/bin/env python3
"""Analyze QA patterns where BOTH models were wrong."""

import json
from pathlib import Path
from collections import defaultdict

# Load both results
minju_results = {}
ours_results = {}

minju_file = Path("outputs/muma_gpt4o_paper_minju.jsonl")
ours_file = Path("outputs/muma_gpt4o_low_detail.jsonl")

print("Loading results...")
with open(minju_file) as f:
    for line in f:
        data = json.loads(line)
        key = (data["episode_id"], data["question_id"])
        minju_results[key] = data

with open(ours_file) as f:
    for line in f:
        data = json.loads(line)
        key = (data["episode_id"], data["question_id"])
        ours_results[key] = data

# Find cases where BOTH are wrong
both_wrong = []
for key in minju_results:
    if key in ours_results:
        minju = minju_results[key]
        ours = ours_results[key]
        
        minju_correct = minju.get("correct", False)
        ours_correct = ours.get("correct", False)
        
        if not minju_correct and not ours_correct:
            q_num = key[1].split("_")[-1] if "_" in key[1] else "?"
            both_wrong.append({
                "episode_id": key[0],
                "question_id": key[1],
                "question_type": minju.get("question_type"),
                "q_variant": f"Q{q_num}",
                "ground_truth": minju.get("gold_answer"),
                "minju_pred": minju.get("predicted_letter"),
                "ours_pred": ours.get("predicted_letter"),
                "same_wrong": minju.get("predicted_letter") == ours.get("predicted_letter"),
                "minju_reasoning": minju.get("reasoning"),
                "ours_reasoning": ours.get("reasoning"),
            })

print(f"\n✓ Both wrong cases: {len(both_wrong)}")
print(f"  Same wrong answer: {sum(1 for x in both_wrong if x['same_wrong'])}")
print(f"  Different wrong answer: {sum(1 for x in both_wrong if not x['same_wrong'])}")

# Analyze by type
by_type = defaultdict(lambda: {"count": 0, "same": 0})
by_q = defaultdict(lambda: {"count": 0, "same": 0})
by_type_q = defaultdict(lambda: {"count": 0, "same": 0})

for item in both_wrong:
    t = item["question_type"]
    q = item["q_variant"]
    by_type[t]["count"] += 1
    if item["same_wrong"]:
        by_type[t]["same"] += 1
    
    by_q[q]["count"] += 1
    if item["same_wrong"]:
        by_q[q]["same"] += 1
        
    key = f"{t}_{q}"
    by_type_q[key]["count"] += 1
    if item["same_wrong"]:
        by_type_q[key]["same"] += 1

print("\n📊 Analysis by Type:")
for t in sorted(by_type.keys()):
    total = by_type[t]["count"]
    same = by_type[t]["same"]
    print(f"  {t}: {total} cases ({same} same, {total-same} different)")

print("\n📊 Analysis by Q variant:")
for q in sorted(by_q.keys()):
    total = by_q[q]["count"]
    same = by_q[q]["same"]
    print(f"  {q}: {total} cases ({same} same, {total-same} different)")

print("\n📊 Analysis by Type × Q:")
for key in sorted(by_type_q.keys()):
    total = by_type_q[key]["count"]
    same = by_type_q[key]["same"]
    if total > 0:
        print(f"  {key}: {total} ({same} same)")

# Load actual QA data
print("\n\n========== SAME WRONG ANSWER EXAMPLES ==========")
with open("/home/mindrium-admin3/datasets/muma_tom_videos/questions.json") as f:
    qa_data = json.load(f)

same_wrong_cases = [x for x in both_wrong if x["same_wrong"]]
print(f"\nShowing 10 examples from {len(same_wrong_cases)} same-wrong cases:\n")

for i, case in enumerate(same_wrong_cases[:10]):
    ep_id = case["episode_id"]
    
    if ep_id not in qa_data:
        continue
        
    ep_data = qa_data[ep_id]
    
    print(f"[{i+1}] EP{ep_id} {case['q_variant']} ({case['question_type']})")
    print(f"    Context: {ep_data.get('description', '')[:100]}...")
    print(f"    Ground Truth: {case['ground_truth'][:100]}...")
    print(f"    Both predicted: {case['minju_pred']}")
    print(f"    Minju reasoning: {case['minju_reasoning'][:80] if case['minju_reasoning'] else 'N/A'}...")
    print()

print("\n\n========== DIFFERENT WRONG ANSWER EXAMPLES ==========")
diff_wrong_cases = [x for x in both_wrong if not x["same_wrong"]]
print(f"\nShowing 10 examples from {len(diff_wrong_cases)} different-wrong cases:\n")

for i, case in enumerate(diff_wrong_cases[:10]):
    ep_id = case["episode_id"]
    
    if ep_id not in qa_data:
        continue
        
    ep_data = qa_data[ep_id]
    
    print(f"[{i+1}] EP{ep_id} {case['q_variant']} ({case['question_type']})")
    print(f"    Context: {ep_data.get('description', '')[:100]}...")
    print(f"    Ground Truth: {case['ground_truth'][:100]}...")
    print(f"    Minju predicted: {case['minju_pred']}")
    print(f"    Ours predicted: {case['ours_pred']}")
    print()

# Save detailed analysis
analysis = {
    "total_both_wrong": len(both_wrong),
    "same_wrong": sum(1 for x in both_wrong if x["same_wrong"]),
    "different_wrong": sum(1 for x in both_wrong if not x["same_wrong"]),
    "by_type": {t: dict(v) for t, v in by_type.items()},
    "by_q": {q: dict(v) for q, v in by_q.items()},
    "by_type_q": {k: dict(v) for k, v in by_type_q.items()},
}

with open("both_wrong_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

print("✓ Saved to both_wrong_analysis.json")
