"""
CECR 900q Full Analysis
- 표 1: Overall Accuracy (900q)
- 표 2: Accuracy by question_type (900q)
- 표 3: Modality Ablation (900q)
- 표 4: Benevolence Bias / Help-Prevent Confusion (900q)
- 표 5: Ablation Study (30q pilot)
- Objectivity check: 주장 vs 실제 수치 점검
- outputs/cecr_900q_tables.md
"""
import json
import csv
from collections import defaultdict
from pathlib import Path

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
def load_jsonl(path):
    return {json.loads(l)["question_id"]: json.loads(l)
            for l in Path(path).read_text(encoding="utf-8").strip().splitlines()}

baseline_raw  = load_jsonl("outputs/muma_gpt4o_paper_minju.jsonl")    # v1 (베이스라인)
cecr_vid_raw  = load_jsonl("outputs/cecr_gpt4o_video_900.jsonl")       # CECR Video
cecr_txt_raw  = load_jsonl("outputs/cecr_gpt4o_textonly_900.jsonl")    # CECR TextOnly

# ablation (30q pilot)
wo1_raw = load_jsonl("outputs/cecr_wo_step1_30.jsonl")
wo2_raw = load_jsonl("outputs/cecr_wo_step2_30.jsonl")
cecr_vid_30_raw = load_jsonl("outputs/cecr_gpt4o_video_30.jsonl")

sampled_30 = json.load(open("sampled_30_questions.json", encoding="utf-8"))
pilot_ids = [r["question_id"] for r in sampled_30]
pilot_meta = {r["question_id"]: r for r in sampled_30}

QUESTION_TYPES = ["belief", "belief_of_goal", "social_goal"]
N_TOTAL = {"belief": 202, "belief_of_goal": 496, "social_goal": 202}

# ── 집계 헬퍼 ─────────────────────────────────────────────────────────────────
def stats(raw, ids=None):
    """ids가 None이면 전체"""
    recs = list(raw.values()) if ids is None else [raw[i] for i in ids if i in raw]
    n = len(recs)
    c = sum(r["correct"] for r in recs)
    by = {}
    for qt in QUESTION_TYPES:
        s = [r for r in recs if r["question_type"] == qt]
        by[qt] = (sum(r["correct"] for r in s), len(s))
    return {"n": n, "c": c, "by": by}

def fmt(c, t):
    return f"{c}/{t} ({c/t*100:.1f}%)" if t > 0 else "N/A"

def sign(n):
    return f"+{n}" if n >= 0 else str(n)

def sign_pp(pp):
    return f"+{pp:.1f}" if pp >= 0 else f"{pp:.1f}"

def classify_social(text):
    t = text.lower()
    for kw in ["prevent", "hinder", "obstruct", "mislead"]:
        if kw in t: return "prevent"
    for kw in ["indifferent", "neutral"]:
        if kw in t: return "neutral"
    for kw in ["help", "helping", "helped", "locate"]:
        if kw in t: return "help"
    return "other"

def hp_confusion(gold_cat, pred_cat, correct):
    if correct: return False
    if "neutral" in (gold_cat, pred_cat) or "other" in (gold_cat, pred_cat): return False
    return gold_cat != pred_cat

# ── 전체 집계 ─────────────────────────────────────────────────────────────────
bl  = stats(baseline_raw)
cv  = stats(cecr_vid_raw)
ct  = stats(cecr_txt_raw)

# social_goal help/prevent confusion (900q)
def sg_bias_stats(raw):
    sg_recs = [r for r in raw.values() if r["question_type"] == "social_goal"]
    total   = len(sg_recs)
    correct = sum(r["correct"] for r in sg_recs)
    errors  = total - correct
    conf = 0
    for r in sg_recs:
        gold_cat = classify_social(r.get("gold_answer", ""))
        pred_cat = classify_social(r.get("predicted_answer", ""))
        if hp_confusion(gold_cat, pred_cat, r["correct"]):
            conf += 1
    return {"total": total, "correct": correct, "errors": errors, "conf": conf}

bl_sg  = sg_bias_stats(baseline_raw)
cv_sg  = sg_bias_stats(cecr_vid_raw)
ct_sg  = sg_bias_stats(cecr_txt_raw)

# 30q pilot ablation
bl_30  = stats(baseline_raw, pilot_ids)
cv_30  = stats(cecr_vid_30_raw, pilot_ids)
ct_30  = stats(cecr_txt_raw, pilot_ids)
wo1_30 = stats(wo1_raw, pilot_ids)
wo2_30 = stats(wo2_raw, pilot_ids)

# ── Markdown 빌드 ──────────────────────────────────────────────────────────────
lines = []

lines.append("# CECR Full Analysis — MuMA-ToM 900 Questions\n")
lines.append("*Baseline = GPT-4o v1 (paper_minju) | CECR = v9_cecr prompt | seed=2025*\n")
lines.append("---\n")

# ── Table 1: Overall Accuracy ─────────────────────────────────────────────────
lines.append("## Table 1. Overall Accuracy (900q)\n")
lines.append("| Method | Modality | Correct | Accuracy |")
lines.append("|---|---|---:|---:|")
lines.append(f"| Baseline (v1) | Text+Video | {bl['c']}/{bl['n']} | {bl['c']/bl['n']*100:.1f}% |")
lines.append(f"| CECR          | Text Only  | {ct['c']}/{ct['n']} | {ct['c']/ct['n']*100:.1f}% |")
lines.append(f"| CECR          | Text+Video | {cv['c']}/{cv['n']} | {cv['c']/cv['n']*100:.1f}% |")
lines.append("")

cv_vs_bl = cv['c'] - bl['c']
ct_vs_bl = ct['c'] - bl['c']
cv_vs_ct = cv['c'] - ct['c']

lines.append("**Findings:**")
lines.append(f"- CECR Text+Video vs Baseline: {sign(cv_vs_bl)} ({sign_pp(cv_vs_bl/bl['n']*100)} pp)")
lines.append(f"- CECR Text Only  vs Baseline: {sign(ct_vs_bl)} ({sign_pp(ct_vs_bl/bl['n']*100)} pp)")
lines.append(f"- CECR Text+Video vs TextOnly: {sign(cv_vs_ct)} ({sign_pp(cv_vs_ct/ct['n']*100)} pp)")

if cv['c'] > bl['c'] and cv['c'] > ct['c']:
    lines.append("- ✅ CECR Text+Video is the best variant overall.")
elif ct['c'] > bl['c']:
    lines.append("- ⚠️  CECR Text Only outperforms Baseline, but Video does not consistently add value.")
else:
    lines.append("- ⚠️  Neither CECR variant outperforms Baseline v1 on 900q. Interpret claims carefully.")
lines.append("")

# ── Table 2: By Question Type ────────────────────────────────────────────────
lines.append("## Table 2. Accuracy by Question Type (900q)\n")
lines.append("| Method | belief (202) | belief_of_goal (496) | social_goal (202) | Total |")
lines.append("|---|---:|---:|---:|---:|")
for label, s in [("Baseline (v1)", bl), ("CECR Text+Video", cv), ("CECR Text Only", ct)]:
    b  = s["by"]["belief"]
    bg = s["by"]["belief_of_goal"]
    sg = s["by"]["social_goal"]
    lines.append(f"| {label} | {fmt(*b)} | {fmt(*bg)} | {fmt(*sg)} | {fmt(s['c'], s['n'])} |")

# Delta rows
lines.append(f"| Δ (CECR Vid − BL) | "
             f"{sign(cv['by']['belief'][0]-bl['by']['belief'][0])} | "
             f"{sign(cv['by']['belief_of_goal'][0]-bl['by']['belief_of_goal'][0])} | "
             f"{sign(cv['by']['social_goal'][0]-bl['by']['social_goal'][0])} | "
             f"{sign(cv_vs_bl)} |")
lines.append("")

lines.append("**Findings:**")
for qt in QUESTION_TYPES:
    cv_c, cv_t = cv["by"][qt]
    bl_c, bl_t = bl["by"][qt]
    ct_c, ct_t = ct["by"][qt]
    delta_vid = cv_c - bl_c
    delta_txt = ct_c - bl_c
    best = "Video" if cv_c >= ct_c else "TextOnly"
    lines.append(f"- **{qt}**: Baseline {fmt(bl_c,bl_t)} → CECR Video {fmt(cv_c,cv_t)} ({sign(delta_vid)}) / TextOnly {fmt(ct_c,ct_t)} ({sign(delta_txt)}) → best: {best}")
lines.append("")

# ── Table 3: Modality Ablation ───────────────────────────────────────────────
lines.append("## Table 3. Modality Ablation — CECR (900q)\n")
lines.append("| Setting | Correct | Accuracy | belief | belief_of_goal | social_goal |")
lines.append("|---|---:|---:|---:|---:|---:|")
for label, s in [("CECR Text Only", ct), ("CECR Text+Video", cv)]:
    b  = s["by"]["belief"]
    bg = s["by"]["belief_of_goal"]
    sg = s["by"]["social_goal"]
    lines.append(f"| {label} | {s['c']}/{s['n']} | {s['c']/s['n']*100:.1f}% | "
                 f"{fmt(*b)} | {fmt(*bg)} | {fmt(*sg)} |")

d_total = cv['c'] - ct['c']
d_pp = d_total / ct['n'] * 100
lines.append(f"| Δ (Video − TextOnly) | {sign(d_total)} | {sign_pp(d_pp)} pp | "
             f"{sign(cv['by']['belief'][0]-ct['by']['belief'][0])} | "
             f"{sign(cv['by']['belief_of_goal'][0]-ct['by']['belief_of_goal'][0])} | "
             f"{sign(cv['by']['social_goal'][0]-ct['by']['social_goal'][0])} |")
lines.append("")

lines.append("**Findings:**")
if d_total > 0:
    lines.append(f"- Adding video frames improves CECR by {sign(d_total)} cases ({sign_pp(d_pp)} pp) overall.")
else:
    lines.append(f"- Adding video frames does **not** improve CECR overall ({sign(d_total)} cases, {sign_pp(d_pp)} pp).")

sg_delta = cv['by']['social_goal'][0] - ct['by']['social_goal'][0]
bog_delta = cv['by']['belief_of_goal'][0] - ct['by']['belief_of_goal'][0]
if sg_delta > 0:
    lines.append(f"- social_goal benefits from frames ({sign(sg_delta)} cases), suggesting visual grounding helps social intent reasoning.")
else:
    lines.append(f"- social_goal does not benefit from frames ({sign(sg_delta)} cases).")
if bog_delta > 0:
    lines.append(f"- belief_of_goal benefits from frames ({sign(bog_delta)} cases).")
else:
    lines.append(f"- belief_of_goal does not benefit from frames ({sign(bog_delta)} cases); text context may be sufficient.")
lines.append("")

# ── Table 4: Benevolence Bias ───────────────────────────────────────────────
lines.append("## Table 4. Benevolence Bias / Help-Prevent Confusion — social_goal (900q)\n")
lines.append("| Method | Correct | Errors | Help/Prevent Confusion | Confusion Rate (of errors) |")
lines.append("|---|---:|---:|---:|---:|")
for label, sg in [("Baseline (v1)", bl_sg), ("CECR Text+Video", cv_sg), ("CECR Text Only", ct_sg)]:
    conf_rate = f"{sg['conf']/sg['errors']*100:.1f}%" if sg['errors'] > 0 else "N/A"
    lines.append(f"| {label} | {sg['correct']}/{sg['total']} ({sg['correct']/sg['total']*100:.1f}%) | "
                 f"{sg['errors']} | {sg['conf']} | {conf_rate} |")
lines.append("")

lines.append("**Findings:**")
bl_conf_rate  = bl_sg['conf']/bl_sg['errors']*100  if bl_sg['errors']  > 0 else 0
cv_conf_rate  = cv_sg['conf']/cv_sg['errors']*100  if cv_sg['errors']  > 0 else 0
ct_conf_rate  = ct_sg['conf']/ct_sg['errors']*100  if ct_sg['errors']  > 0 else 0
lines.append(f"- Baseline help/prevent confusion rate: {bl_conf_rate:.1f}% of errors")
lines.append(f"- CECR Text+Video confusion rate: {cv_conf_rate:.1f}% of errors")
lines.append(f"- CECR Text Only  confusion rate: {ct_conf_rate:.1f}% of errors")
if cv_conf_rate < bl_conf_rate:
    lines.append("- ✅ CECR reduces help/prevent confusion, supporting the Benevolence Bias mitigation claim.")
else:
    lines.append("- ⚠️  CECR does not reduce help/prevent confusion vs Baseline. Benevolence Bias claim needs qualification.")
lines.append("")

# ── Table 5: Ablation Study (30q pilot) ──────────────────────────────────────
lines.append("## Table 5. Ablation Study — 30q Pilot (seed=2025)\n")
lines.append("*Step 1 = Character State Identification | Step 2 = Consistency Check*\n")
lines.append("| Method | Step1 | Step2 | belief | belief_of_goal | social_goal | Total |")
lines.append("|---|:---:|:---:|---:|---:|---:|---:|")

ablation_rows = [
    ("Baseline (v1)", "✗", "✗", bl_30),
    ("CECR Full",     "✓", "✓", cv_30),
    ("w/o Step 1",    "✗", "✓", wo1_30),
    ("w/o Step 2",    "✓", "✗", wo2_30),
]
for name, s1, s2, s in ablation_rows:
    b  = s["by"]["belief"]
    bg = s["by"]["belief_of_goal"]
    sg = s["by"]["social_goal"]
    lines.append(f"| {name} | {s1} | {s2} | {fmt(*b)} | {fmt(*bg)} | {fmt(*sg)} | {fmt(s['c'],s['n'])} |")
lines.append("")

lines.append("**Findings:**")
lines.append(f"- w/o Step 1 ({fmt(wo1_30['c'],30)}) vs CECR Full ({fmt(cv_30['c'],30)}): "
             f"{sign(wo1_30['c']-cv_30['c'])} → Step 1 contribution is {'minimal' if abs(wo1_30['c']-cv_30['c']) <= 1 else 'notable'}.")
lines.append(f"- w/o Step 2 ({fmt(wo2_30['c'],30)}) vs CECR Full ({fmt(cv_30['c'],30)}): "
             f"{sign(wo2_30['c']-cv_30['c'])} → Step 2 (Consistency Check) is {'the key driver' if cv_30['c']-wo2_30['c'] >= 2 else 'marginally helpful'}.")
lines.append("")

# ── Objectivity Check ─────────────────────────────────────────────────────────
lines.append("---\n")
lines.append("## ⚖️ Objectivity Check\n")
lines.append("*Potential issues to address before paper submission:*\n")

issues = []

# 1. 900q 전체에서 CECR < Baseline v1?
if cv['c'] <= bl['c'] and ct['c'] <= bl['c']:
    issues.append("🔴 **CECR does not outperform Baseline v1 on 900q overall.** "
                  f"(BL={bl['c']/bl['n']*100:.1f}%, CECR-Vid={cv['c']/cv['n']*100:.1f}%, CECR-Txt={ct['c']/ct['n']*100:.1f}%) "
                  "— Claims of overall improvement cannot be made. Focus on specific question types where CECR wins.")
elif cv['c'] <= bl['c']:
    issues.append("🟡 **CECR Text+Video does not outperform Baseline v1 overall.** "
                  f"CECR-Vid={cv['c']/cv['n']*100:.1f}% vs BL={bl['c']/bl['n']*100:.1f}%. "
                  "Only CECR Text Only shows marginal gain. Video frames may add noise.")

# 2. Video < TextOnly?
if cv['c'] < ct['c']:
    issues.append(f"🟡 **Video underperforms TextOnly** ({cv['c']/cv['n']*100:.1f}% vs {ct['c']/ct['n']*100:.1f}%). "
                  "Multimodal claim needs to be scoped — video helps only specific question types (e.g., social_goal).")

# 3. social_goal: CECR이 도움되는지
cv_sg_acc = cv['by']['social_goal'][0] / cv['by']['social_goal'][1]
bl_sg_acc = bl['by']['social_goal'][0] / bl['by']['social_goal'][1]
if cv_sg_acc > bl_sg_acc:
    issues.append(f"✅ **social_goal**: CECR Video ({cv_sg_acc*100:.1f}%) > Baseline ({bl_sg_acc*100:.1f}%). "
                  "This is the strongest claim for CECR's contribution.")
else:
    issues.append(f"🔴 **social_goal**: CECR Video ({cv_sg_acc*100:.1f}%) ≤ Baseline ({bl_sg_acc*100:.1f}%). "
                  "Benevolence Bias mitigation claim is weakened.")

# 4. Ablation: Step1이 사실상 불필요?
if wo1_30['c'] >= cv_30['c']:
    issues.append(f"🟡 **Ablation**: w/o Step 1 ({wo1_30['c']}/30) ≥ CECR Full ({cv_30['c']}/30). "
                  "Step 1 may be redundant or even harmful. Reconsider framing.")

# 5. 30q pilot vs 900q 방향이 다른가?
pilot_cv_acc = cv_30['c'] / 30
full_cv_acc  = cv['c'] / cv['n']
if abs(pilot_cv_acc - full_cv_acc) > 0.08:
    issues.append(f"🟡 **Pilot vs Full gap**: Pilot CECR={pilot_cv_acc*100:.1f}% vs Full={full_cv_acc*100:.1f}% "
                  f"(gap={abs(pilot_cv_acc-full_cv_acc)*100:.1f}pp). "
                  "Pilot results may not generalize — present both honestly.")

for issue in issues:
    lines.append(f"- {issue}")

lines.append("")
lines.append("---\n")
lines.append("*Generated by analyze_cecr_900q.py*")

# ── 저장 ─────────────────────────────────────────────────────────────────────
out_md = Path("outputs/cecr_900q_tables.md")
out_md.write_text("\n".join(lines), encoding="utf-8")
print(f"✅ Saved: {out_md}")

# ── 콘솔 요약 ──────────────────────────────────────────────────────────────────
print(f"""
==================================================
  900q 결과 요약
==================================================
  Baseline v1  : {bl['c']}/{bl['n']} ({bl['c']/bl['n']*100:.1f}%)
  CECR TextOnly: {ct['c']}/{ct['n']} ({ct['c']/ct['n']*100:.1f}%)
  CECR Video   : {cv['c']}/{cv['n']} ({cv['c']/cv['n']*100:.1f}%)

  belief      : BL={fmt(*bl['by']['belief'])} | TextOnly={fmt(*ct['by']['belief'])} | Video={fmt(*cv['by']['belief'])}
  bog         : BL={fmt(*bl['by']['belief_of_goal'])} | TextOnly={fmt(*ct['by']['belief_of_goal'])} | Video={fmt(*cv['by']['belief_of_goal'])}
  social_goal : BL={fmt(*bl['by']['social_goal'])} | TextOnly={fmt(*ct['by']['social_goal'])} | Video={fmt(*cv['by']['social_goal'])}

  Objectivity issues: {len(issues)}개
==================================================
""")
