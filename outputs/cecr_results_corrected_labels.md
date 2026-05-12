# CECR Results — GitHub Labels vs HF Labels 비교

## 라벨 오류 요약

| | belief | bog | social_goal | 총합 |
|---|---:|---:|---:|---:|
| **HF 버전 (실험 당시)** | 202 | 496 | 202 | 900 |
| **GitHub 원본 (정확)** | 300 | 300 | 300 | 900 |

- 49개 에피소드 × 4문항 = **196개** 잘못 분류: belief(98) + sg(98) → bog
- **overall accuracy는 변하지 않음** (라벨만 달라지고, 정답/오답 판정은 동일)

---

## Table 1. Overall Accuracy (변화 없음)

| Method | Modality | Correct/Total | Accuracy | Δ vs Baseline |
|--------|----------|:---:|:---:|:---:|
| Baseline | Text+Video | 515/900 | 57.2% | — |
| CECR | Text Only | 622/900 | 69.1% | **+11.9pp** |
| CECR | Text+Video | 609/900 | 67.7% | **+10.5pp** |

---

## Table 2. By Question Type — HF Labels (구버전, 잘못된 분포)

| Method | belief (n=202) | bog (n=496) | sg (n=202) | Total |
|--------|:---:|:---:|:---:|:---:|
| Baseline | 136/202 (67.3%) | 273/496 (55.0%) | 106/202 (52.5%) | 515/900 (57.2%) |
| CECR TextOnly | 168/202 (83.2%) | 317/496 (63.9%) | 137/202 (67.8%) | 622/900 (69.1%) |
| CECR Video | 156/202 (77.2%) | 310/496 (62.5%) | 143/202 (70.8%) | 609/900 (67.7%) |

---

## Table 3. By Question Type — GitHub Labels (수정, 올바른 분포 300×3)

| Method | belief (n=300) | bog (n=300) | sg (n=300) | Total |
|--------|:---:|:---:|:---:|:---:|
| Baseline | 204/300 (68.0%) | 154/300 (51.3%) | 157/300 (52.3%) | 515/900 (57.2%) |
| CECR TextOnly | 247/300 (82.3%) | 180/300 (60.0%) | 195/300 (65.0%) | 622/900 (69.1%) |
| CECR Video | 236/300 (78.7%) | 165/300 (55.0%) | 208/300 (69.3%) | 609/900 (67.7%) |

---

## Table 4. Δ (CECR - Baseline) 비교

### HF Labels 기준 (구버전)
| | belief | bog | sg | Total |
|---|:---:|:---:|:---:|:---:|
| TextOnly − Baseline | +15.9pp | +8.9pp | +15.3pp | +11.9pp |
| Video − Baseline | +9.9pp | +7.5pp | +18.3pp | +10.5pp |

### GitHub Labels 기준 (수정)
| | belief | bog | sg | Total |
|---|:---:|:---:|:---:|:---:|
| TextOnly − Baseline | +14.3pp | +8.7pp | +12.7pp | +11.9pp |
| Video − Baseline | +10.7pp | +3.7pp | +17.0pp | +10.5pp |

---

## Table 5. Modality 비교 (TextOnly vs Video)

### GitHub Labels 기준
| | belief | bog | sg | Total |
|---|:---:|:---:|:---:|:---:|
| Video − TextOnly | 236-247 = **-11** | 165-180 = **-15** | 208-195 = **+13** | -13 (-1.4pp) |

---

## Table 6. Ablation (30q pilot) — GitHub Labels 기준

> ⚠️ 30q 파일럿은 HF labels 기준 10/10/10으로 샘플링 → GitHub labels로 재매핑 시 **14/2/14** 분포
> (bog로 잘못 분류된 질문 중 일부가 포함됨)

| Method | belief (n=14) | bog (n=2) | sg (n=14) | Total (n=30) |
|--------|:---:|:---:|:---:|:---:|
| Baseline | — | — | — | 19/30 (63.3%) |
| CECR Full | 12/14 (85.7%) | 2/2 (100%) | 10/14 (71.4%) | 24/30 (80.0%) |
| w/o Step 1 | 11/14 (78.6%) | 2/2 (100%) | 11/14 (78.6%) | 24/30 (80.0%) |
| w/o Step 2 | 10/14 (71.4%) | 1/2 (50.0%) | 10/14 (71.4%) | 21/30 (70.0%) |

> baseline 30q는 별도 JSONL 없어 재계산 불가 (이전 결과 19/30 그대로 사용)

---

## 요약: 어떤 숫자가 바뀌었나

| 항목 | 변화 |
|------|------|
| Overall accuracy | **변화 없음** |
| per-type 수치 | 분모 변경 (202→300, 496→300) |
| bog 정확도 | Baseline 55.0% → **51.3%** (더 낮아짐, 어려운 문제) |
| sg CECR Video | 70.8% → **69.3%** (소폭 하락) |
| 주요 결론 | **동일** (CECR이 모든 type에서 향상) |
