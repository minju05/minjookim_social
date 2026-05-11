# MuMA-ToM 확장 구현 가이드

> 이 문서는 두 트랙(KDD Workshop: PIRG / ARR: LIMP++)의 구현 흐름을 정리한 것입니다.
> LIMP 베이스라인 재현이 선행되어야 합니다.

---

## 0. 사전 준비: LIMP 베이스라인 재현

### 0.1 환경 설정

```
MuMA-ToM 데이터셋 다운로드: https://scai.cs.jhu.edu/projects/MuMA-ToM/
  - 테스트셋: 225 시나리오, 900 QA (belief/social goal/belief-of-goal 각 300)
  - 트레이닝셋: 1,030 videos (action + goal 주석 포함)
  - 부가 데이터: depth images, instance segmentation, camera data

API 키:
  - Gemini API (VLM): gemini-2.5-flash  (원 논문: gemini-1.5-pro)
  - OpenAI API (LLM): gpt-4o-mini       (원 논문: gpt-4o)

⚠️  모델 변경 전략:
  - 1차: 전체 파이프라인을 gpt-4o-mini로 실행
  - 성능 모니터링 기준: social goal accuracy가 LIMP 재현값(~67.7%) 대비 -10%p 이상 하락 시
    → inverse planning(Step 5)만 gpt-4o로 교체하는 하이브리드 전환 고려
  - belief / social_goal / belief-of-goal 세 지표를 각각 기록해서 어느 스텝이 병목인지 추적
```

### 0.2 LIMP 파이프라인 요약 (교체 대상 파악용)

```
[Step 1] Visual Perception
  입력: RGB 비디오
  모델: Gemini 1.5 Pro (VLM)
  출력: 두 에이전트의 raw action/utterance 텍스트

[Step 2] Text Parsing
  입력: 텍스트 (대화 또는 행동 서술)
  모델: gpt-4o-mini  (원 논문: gpt-4o)
  출력: 에이전트별 action/utterance 리스트

[Step 3] Multimodal Fusion
  입력: Step 1 + Step 2 출력
  모델: gpt-4o-mini
  출력: fused action/utterance table + initial state

[Step 4] Hypothesis Parsing
  입력: fused info + 질문
  모델: gpt-4o-mini
  출력: 각 선택지별 (belief, social_goal, believed_goal) 트리플

[Step 5] Inverse Multi-Agent Planning (IMP)
  입력: fused info + hypotheses
  모델: gpt-4o-mini  ← 성능 하락 시 gpt-4o로 교체 우선 검토
  출력: 각 타임스텝별 action/utterance likelihood → 최종 선택지 확률
  핵심: log prob of token 'A' (Likely) 추출
```

### 0.3 재현 체크리스트

- [ ] Gemini API로 비디오에서 raw action 추출 확인
- [ ] GPT-4o로 text parsing → fused table 생성 확인
- [ ] IMP의 log probability 추출 동작 확인 (GPT-4o top-5 logprobs 사용)
- [ ] 전체 정확도 ~76.6% 근방 재현 확인
  - Belief: ~93.4%, Social Goal: ~67.7%, Belief-of-Goal: ~68.7%

---

## Track 1: KDD Workshop — PIRG

> **Perspective-Isolated Rationale Grounding**
> 포지셔닝: "왜 기존 방법들이 adversarial social intent에서 실패하는가"
> Contribution: training-free 방법 비교 + failure mode 분석

### 1.1 실험 구조

```
Condition 0 (Base):     GPT-4o / Gemini zero-shot         ← 논문 재현
Condition 1 (Method A): Single-Turn Structured CoT         ← 새로 구현
Condition 2 (Method B): Multi-Agent 분리형 (Judge 통합)    ← 새로 구현
```

평가 단위: 3 질문 유형 × 3 조건 = 9-cell breakdown

---

### 1.2 Method A: Single-Turn Structured CoT

**개념**: 하나의 프롬프트 안에서 "사실 추출 → A 추론 → B 추론 → 결론" 순서를 강제

**구현**:

```python
STRUCTURED_COT_PROMPT = """
아래 영상/텍스트 정보를 단계적으로 분석하세요.

[Step 1 - 사실 추출]
- 각 에이전트가 한 행동을 시간순으로 나열하세요.
- 대화가 있다면 누가 무엇을 말했는지 정리하세요.

[Step 2 - Agent A 관점 추론]
- Agent A는 무엇을 알고 있는가?
- Agent A의 행동 의도는 무엇인가?

[Step 3 - Agent B 관점 추론]
- Agent B는 무엇을 알고 있는가?
- Agent B의 행동 의도는 무엇인가?

[Step 4 - 결론]
위 분석을 바탕으로 다음 질문에 답하세요.
질문: {question}
선택지: {options}
답: (A/B/C 중 하나)
"""
```

**주의**: 단계 순서가 바뀌거나 생략되지 않도록 파싱 로직 필요
- GPT-4o가 Step을 건너뛰는 경우가 있으므로, 각 Step 헤더 존재 여부 검증 추가

---

### 1.3 Method B: Multi-Agent 분리형

**개념**: 에이전트별 독립 관찰 → Judge LLM이 통합 결정
**구현 난이도**: 낮음 (객체 추적 없이 프롬프트 레벨 역할 분리만)

**구현 흐름**:

```
[Sub-step B1] Agent i 관점 프롬프트
  "이 시나리오에서 Agent i의 시점에서만 관찰된 것을 서술하세요.
   Agent j의 행동은 Agent i가 직접 목격한 경우에만 포함하세요."

[Sub-step B2] Agent j 관점 프롬프트
  "이 시나리오에서 Agent j의 시점에서만 관찰된 것을 서술하세요.
   Agent i의 행동은 Agent j가 직접 목격한 경우에만 포함하세요."

[Sub-step B3] Judge 프롬프트
  입력: B1 결과 + B2 결과 + 질문/선택지
  "다음은 두 에이전트의 독립적인 관찰 결과입니다.
   이를 종합해서 질문에 답하세요.
   [Agent i 관찰]: {obs_i}
   [Agent j 관찰]: {obs_j}
   질문: {question}
   답: (A/B/C)"
```

**API 호출 횟수**: 시나리오당 3회 (B1 + B2 + Judge)
- 비용 추정: 900 문항 × 3 = 2,700 calls

---

### 1.4 에러 분석 설계

```python
# 결과 저장 구조
result = {
    "scenario_id": str,
    "question_type": "belief" | "social_goal" | "belief_of_goal",
    "social_intent": "help" | "hinder" | "independent",  # ground truth
    "condition": "base" | "method_a" | "method_b",
    "pred": "A" | "B" | "C",
    "gold": "A" | "B" | "C",
    "correct": bool,
    "error_type": str  # 실패 시 분류 (선택)
}
```

**에러 타입 분류 기준** (adversarial 케이스 중심):
- `adversarial_misread`: hindering을 helping으로 오해
- `visual_failure`: 객체/행동 인식 실패로 인한 오답
- `belief_confusion`: 에이전트 간 믿음 경계 혼동

**분석 출력 테이블**:

|  | Belief | Social Goal | Belief-of-Goal |
|--|--------|-------------|----------------|
| Base (재현) | | | |
| Method A (CoT) | | | |
| Method B (Multi-Agent) | | | |
| Human | 98.9 | 94.4 | 87.1 |

---

## Track 2: ARR — LIMP++

> **Recursive Belief Modeling with Intent-Sensitive Scoring**
> 포지셔닝: LIMP의 두 구조적 한계를 inverse planning 단계 증강으로 해결
> Contribution: Module A (belief-of-goal↑) + Module C (social goal↑) 각각 ablation 가능

### 2.1 전체 구조

```
LIMP 기본 파이프라인 유지
    ↓ Step 1~3 동일 (Multimodal Fusion까지)
    ↓ Step 4 동일 (Hypothesis Parsing까지)
    ↓
[Step 5'] LIMP++ Inverse Planning  ← 이 부분만 교체/증강
    ├── Module A: Explicit Recursive Belief Estimation
    └── Module C: Intent-Sensitive Consistency Scoring
```

**Ablation 조건**:
```
LIMP        : 기존 IMP 그대로
LIMP+A      : Module A만 추가
LIMP+C      : Module C만 추가
LIMP++ (A+C): 둘 다 추가
```

---

### 2.2 Module A: Explicit Recursive Belief Estimation

**타겟**: belief-of-goal inference 문항 (현재 68.7%)
**핵심 아이디어**: Agent j의 belief `b_j(s)`를 조건에 명시적으로 포함

**구현 흐름**:

```
[A-Step 1] Agent j의 관찰 분리
  fused table에서 Agent j가 직접 관찰 가능한 행동만 필터링
  → Agent j가 같은 방에 있었던 시점의 Agent i 행동만 포함

[A-Step 2] Belief Estimator 호출
  입력: s_0 + Agent j의 행동 시퀀스 + Agent j가 목격한 Agent i 행동
  출력: b_j(s^t) — Agent j가 t 시점에 갖는 믿음 서술

[A-Step 3] Belief-Aware Scoring
  기존 IMP 프롬프트에 b_j 추가:
  "agent i의 belief of agent j's belief: {b_j_hat}"
```

**Belief Estimator 프롬프트**:

```python
BELIEF_ESTIMATOR_PROMPT = """
주어진 정보만을 바탕으로, Agent {j}가 현재 시점에서 갖고 있을
환경 상태에 대한 믿음(belief)을 추정하세요.

초기 상태: {s_0}
Agent {j}의 행동 시퀀스: {actions_j}
Agent {j}가 직접 목격한 Agent {i}의 행동: {visible_actions_i}

주의: Agent {j}가 보지 못한 정보는 포함하지 마세요.

출력 형식:
Agent {j}의 현재 믿음: [물체]는 [위치]에 있다 (확신도: high/medium/low)
"""
```

**수정된 IMP 프롬프트**:

```python
# 기존 LIMP IMP 프롬프트에서 H 정의 부분 확장
BELIEF_AWARE_IMP_PROMPT = """
Decide if agent's action is likely with the information provided.

agent's social goal: {social_goal}
agent's belief: {belief}
agent's belief of other agent's goal: {belief_of_goal}
agent's belief of other agent's belief: {b_j_hat}   # ← 추가된 부분
other agent's utterance: {utterance}
Initial state: {initial_state}
Previous Actions: {prev_actions}

Respond with only either A or B:
Agent's Action or Utterance: {action_or_utterance}
A) Likely
B) Unlikely
"""
```

---

### 2.3 Module C: Intent-Sensitive Consistency Scoring

**타겟**: social goal inference 문항 (현재 67.7%)
**핵심 아이디어**: 말-행동 불일치가 hindering의 positive signal임을 명시화

**구현 흐름**:

```
[C-Step 1] 발화-행동 쌍 추출
  각 타임스텝 t에서 utterance u_i^t와 이후 k 스텝 행동 a_i^{t+1:t+k} 추출
  (k=2~3 권장, 너무 길면 noisy)

[C-Step 2] Consistency Scorer 호출
  입력: u_i^t, a_i^{t+1:t+k}, s_0
  출력: "Consistent" 또는 "Inconsistent" + 근거 한 문장

[C-Step 3] Social Goal에 따른 Score 반전 적용
  hypothesis H의 social goal이 "help"이면: consistent → positive
  hypothesis H의 social goal이 "hinder"이면: inconsistent → positive
  hypothesis H의 social goal이 "independent"이면: neutral (0)

[C-Step 4] 최종 Score 통합
  P(H) ∝ [기존 LIMP score] × exp(λ × Σ S_C^t(H))
  λ는 validation set으로 탐색 (권장 범위: 0.1~1.0, grid search)
```

**Consistency Scorer 프롬프트**:

```python
CONSISTENCY_SCORER_PROMPT = """
Agent {i}의 발화와 그 이후 행동을 보고,
말과 행동이 일치하는지 판단하세요.

초기 상태: {s_0}
Agent {i}의 발화: {utterance}
Agent {i}의 이후 행동: {subsequent_actions}

출력 형식:
판단: Consistent / Inconsistent
근거: (한 문장)
"""
```

**Score 반전 로직**:

```python
def compute_consistency_score(consistency: str, social_goal: str) -> float:
    is_consistent = (consistency == "Consistent")
    if social_goal == "help":
        return 1.0 if is_consistent else 0.0
    elif social_goal == "hinder":
        return 1.0 if not is_consistent else 0.0  # 반전
    else:  # independent
        return 0.5  # neutral

def compute_limp_plus_plus_score(limp_log_score: float,
                                  consistency_scores: list[float],
                                  lam: float) -> float:
    consistency_term = lam * sum(consistency_scores)
    return limp_log_score + consistency_term
```

---

### 2.4 λ 하이퍼파라미터 탐색

```python
# validation split: 테스트셋 900문항 중 일부 (예: social_goal 300문항의 20%)를 validation으로 사용
# 또는 training set의 주석 기반 small validation 구성

lambda_grid = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

for lam in lambda_grid:
    scores = evaluate_limp_plus_c(validation_set, lam=lam)
    print(f"λ={lam}: social_goal_acc={scores['social_goal']:.3f}")

# 선택된 λ로 전체 테스트셋 평가
```

---

### 2.5 예상 결과 테이블 (ARR 논문용)

| Method | Belief | Social Goal | Belief-of-Goal | All |
|--------|--------|-------------|----------------|-----|
| LIMP (재현) | 93.4 | 67.7 | 68.7 | 76.6 |
| LIMP+A | 93.4 | 67.7 | **?** | ? |
| LIMP+C | 93.4 | **?** | 68.7 | ? |
| LIMP++ | 93.4 | **?** | **?** | ? |
| Human | 98.9 | 94.4 | 87.1 | 93.5 |

---

## 공통: 디렉토리 구조 제안

```
project/
├── data/
│   ├── test/          # 225 시나리오, 900 QA
│   └── train/         # 1,030 videos (LIMP++ LoRA용, 선택)
│
├── limp_baseline/
│   ├── visual_perception.py     # Gemini VLM 호출
│   ├── text_parsing.py          # GPT-4o 텍스트 파싱
│   ├── multimodal_fusion.py     # fusion + initial state 복원
│   ├── hypothesis_parsing.py    # mental variable 추출
│   └── inverse_planning.py      # IMP scoring (log prob)
│
├── pirg/                        # KDD Track
│   ├── structured_cot.py        # Method A
│   ├── multi_agent_judge.py     # Method B
│   └── error_analysis.py        # breakdown 분석
│
├── limp_plus_plus/              # ARR Track
│   ├── belief_estimator.py      # Module A
│   ├── consistency_scorer.py    # Module C
│   ├── scoring.py               # 통합 score 계산 + λ 탐색
│   └── ablation.py              # LIMP / +A / +C / ++ 비교
│
├── eval/
│   ├── metrics.py               # accuracy by question type
│   └── results/                 # JSON 결과 저장
│
└── configs/
    ├── limp_config.yaml
    ├── pirg_config.yaml
    └── limp_plus_plus_config.yaml
```

---

## 구현 순서 권장

```
Phase 1 (1~2일): LIMP 재현
  → visual_perception.py + text_parsing.py + multimodal_fusion.py 구현
  → hypothesis_parsing.py + inverse_planning.py 구현
  → 전체 정확도 확인 (목표: 76.6% ± 2%)

Phase 2 (1일): KDD 실험 (PIRG)
  → structured_cot.py (Method A) 구현
  → multi_agent_judge.py (Method B) 구현
  → error_analysis.py로 3×3 breakdown 생성

Phase 3 (2~3일): ARR 실험 (LIMP++)
  → belief_estimator.py (Module A) 구현
  → consistency_scorer.py (Module C) 구현
  → λ grid search + ablation 실험

Phase 4 (병렬): 논문 작성
  → KDD: failure mode 분석 중심
  → ARR: Module A/C contribution 분리 검증
```

---

## 공통: API 비용 추정

| 실험 | 호출 수 | gpt-4o-mini 기준 | gpt-4o 기준 (참고) |
|------|---------|-----------------|-------------------|
| LIMP 재현 | 900 × ~5 = 4,500 | ~**$0.3~0.6** | ~$5~10 |
| PIRG Method A | 900 × 1 = 900 | ~**$0.05~0.1** | ~$1~2 |
| PIRG Method B | 900 × 3 = 2,700 | ~**$0.2~0.3** | ~$3~5 |
| LIMP+A | 900 × ~7 = 6,300 | ~**$0.4~0.8** | ~$8~12 |
| LIMP+C | 900 × ~7 = 6,300 | ~**$0.4~0.8** | ~$8~12 |
| **총합** | | ~**$1.4~2.6** | ~$25~40 |

> VLM: Gemini 2.5 Flash 사용 시 visual perception 비용 추가 (미미한 수준)
>
> ⚠️ 하이브리드 전환 시 (Step 5만 gpt-4o로 교체): 총합 ~$8~12 예상
