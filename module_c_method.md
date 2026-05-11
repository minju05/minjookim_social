# Module C: Intent-Sensitive Action-Utterance Consistency Scoring

## 0. 한 줄 요약

> GPT-4o-mini 프레임 샘플링 파이프라인에, **발화(utterance)와 실제 행동의 시각적 일관성**을 추가 신호로 활용해 intent 추론의 정확도를 높이는 모듈.

---

## 1. 왜 만들었나

MuMA-ToM 벤치마크의 세 질문 유형 중 `social_goal`과 `belief_of_goal`은 에이전트의 **의도(intent)**를 추론해야 한다.

```
Q: Was Agent A trying to help or hinder Agent B?
선택지:
  A) Help — Agent A가 B를 도우려 했다
  B) Hinder — Agent A가 B를 방해하려 했다
  C) Indifferent — Agent A는 B에 관심 없었다
```

기존 GPT-4o-mini 파이프라인은 비디오 프레임을 보고 확률 P(A), P(B), P(C)를 직접 산출하는데, 이때 놓치는 신호가 있다:

- 에이전트가 **말한 것**(utterance)과 **실제로 한 것**(비디오 행동)이 **일치하는가 / 불일치하는가**
- 일치 → help 가능성 상승
- 불일치 → hinder 가능성 상승

---

## 2. 전체 파이프라인

```
[비디오 + 질문]
      │
      ▼
Step 1: Base logprobs 추출
      GPT-4o-mini + 프레임 샘플링
      → P(A), P(B), P(C)
      │
      ▼
Step 2: Utterance 추출 (text_context에서)
      GPT-4o-mini 텍스트 콜
      → "X는 Y에 있다" 같은 발화 1개
      │
      ▼
Step 3: Visual Consistency 계산
      GPT-4o-mini + 프레임 샘플링
      → P(consistent) ∈ [0, 1]
      │
      ▼
Step 4: Intent-Sensitive 확률 조정
      선택지 분류 (help / hinder / independent)
      + 조정 공식 적용
      → Adjusted P(A), P(B), P(C)
      │
      ▼
   최종 예측: argmax(adjusted_probs)
```

---

## 3. Step별 상세 구현

### 3.1 Step 1: Base Logprobs

**우리 버전 프롬프트** 사용 (민주 버전보다 성능 70.1% vs 57.2%):

```python
SYSTEM_PROMPT = (
    "You are an expert in theory of mind and social reasoning. "
    "Answer the following multiple-choice question about a video clip. "
    "Respond with only a single letter: A, B, or C."
)

# user content 구조:
# [question_type + text_context + question + A/B/C]
# [Frame 0.0s 이미지]
# [Frame 1.0s 이미지]
# ...
```

`logprobs=True, top_logprobs=5, max_tokens=1` 로 호출해서 A/B/C 각각의 확률 추출.

**⚠ 버그 수정 (2025-05-02)**: `top_logprobs`에 `'B'`와 `' B'`(앞에 공백) 모두 포함될 수 있는데, 둘 다 `.strip().upper()` 하면 `'B'`가 됨. 루프가 두 번 매칭되면서 후순위 `' B'`의 near-zero 값이 정답 `'B'`의 1.0을 덮어쓰는 문제. → **첫 번째 매치만 저장**하도록 수정.

### 3.2 Step 2: Utterance 추출

```python
prompt = (
    "Read the following text describing a social interaction.\n"
    "Text: {text_context}\n\n"
    "Extract any direct utterance (spoken words) made by one of the agents. "
    "If there is no utterance, reply with exactly 'NONE'. "
    "Otherwise reply with only the utterance text, nothing else."
)
```

- `text_context`가 없으면 Step 3/4 스킵
- 모델이 `NONE` 반환하면 역시 스킵
- utterance가 추출된 경우에만 Module C 적용

### 3.3 Step 3: Visual Consistency Score

```python
# user content:
"An agent made the following statement: \"{utterance}\"\n"
"Watch the video frames and decide if the agent's actual behavior is consistent with this statement.\n"
"A) Consistent — their actions match what they said\n"
"B) Inconsistent — their actions contradict what they said\n"
"Respond with only A or B."
```

logprobs에서 `P('A')` 추출 → **P(consistent)** ∈ [0, 1]

- 1.0에 가까울수록 행동이 말과 일치 (help 신호)
- 0.0에 가까울수록 행동이 말과 불일치 (hinder 신호)

### 3.4 Step 4: Intent-Sensitive 확률 조정

선택지 텍스트 키워드 기반으로 분류:

| 분류 | 키워드 | 공식 |
|------|--------|------|
| `help` | help, assist, cooperat, support, aid | $s_c = P(\text{consistent})$ |
| `hinder` | hinder, prevent, obstruct, mislead, deceiv, lie | $s_c = 1 - P(\text{consistent})$ |
| `independent` | 해당 없음 | 변경 없음 |

조정 공식:

$$P'(l) = P(l) \cdot s_c^\lambda$$

이후 정규화:

$$\hat{P}(l) = \frac{P'(l)}{\sum_k P'(k)}$$

**직관**: 
- utterance와 행동이 일치할 때(P(consistent)↑) → help 선택지의 확률을 올리고, hinder는 내림
- 불일치할 때 → 반대로
- λ (lambda_weight)로 Module C의 영향 강도 조절 (기본값 1.0)

---

## 4. 실행 예시

```bash
# Dry run (1개)
python run_muma_gpt4o_c.py --limit 1

# 50개 실험
python run_muma_gpt4o_c.py --limit 50 --output outputs/muma_gpt4o_c_predictions.jsonl

# λ 조정 실험
python run_muma_gpt4o_c.py --limit 50 --lambda-weight 0.5

# 재실행 시 자동 resume (이미 완료된 question_id 스킵)
python run_muma_gpt4o_c.py --limit 50 --offset 50
```

출력 예시 (`outputs/dry_run_test.jsonl`):
```json
{
  "question_id": "4005_1",
  "question_type": "belief",
  "predicted_letter": "B",
  "correct": true,
  "base_probs": {"A": 1e-09, "B": 1.0, "C": 1.6e-09},
  "adjusted_probs": {"A": 1e-09, "B": 0.999, "C": 1.6e-09},
  "consistency_score": 0.182
}
```

---

## 5. API 콜 수

질문 1개당:

| 콜 | 목적 | 프레임 포함 |
|----|------|------------|
| 1회 | Base logprobs | ✓ (최대 24프레임) |
| 1회 | Utterance 추출 | ✗ (텍스트만) |
| 1회 | Visual consistency | ✓ (최대 24프레임) |

→ 질문당 **최대 3콜** (utterance 없으면 2콜, text_context 없으면 1콜)

50개 기준: **최대 150 API 콜**

---

## 6. 신규 파일 목록

| 파일 | 설명 |
|------|------|
| `muma_gpt4o_c.py` | Module C 핵심 구현. utterance 추출, visual consistency 계산, intent-sensitive 확률 조정 |
| `run_muma_gpt4o_c.py` | 실험 러너. resume 지원, jsonl 저장, accuracy 요약 자동 출력 |
| `muma_limp_c.py` | Module C의 LIMP 버전 (action text 기반). Gemini action 추출 완료 시 사용 가능 |
| `run_muma_limp_c.py` | LIMP+C 러너 (현재 Gemini action 미추출로 미실행) |

---

## 7. 한계 및 향후 과제

- **utterance 없는 질문**: text_context에 직접 발화가 없으면 Module C 미적용 → base_probs만 사용
- **hinder/help 이외 선택지**: `independent`로 분류된 선택지는 조정 없음 → edge case 존재
- **λ 튜닝 미완료**: 기본값 1.0으로만 실험. 최적 λ는 validation set으로 탐색 필요
- **LIMP+C**: Gemini 기반 action 추출이 완료되면 action text를 utterance 대신 사용하는 더 정교한 버전 실행 가능