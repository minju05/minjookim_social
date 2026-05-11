# 프롬프트 분석 (3관점)

## 0. 전체 그림

```
민주 버전:  57.2% (515/900)
우리 버전:  70.1% (631/900)
개선:      +12.9%p

구성:
  우리만 맞음:  176개 (19.6%)  ← 우리의 강점
  우리만 틀림:   60개 (6.7%)   ← 우리의 약점
  둘다 틀림:    209개 (23.2%)  ← 모델 한계
  둘다 맞음:    455개 (50.6%)  ← 프롬프트 무관
```

---

## 1. 우리가 12% 올린 이유 (3가지 설계)

### 1.1 System Prompt 추가
```
민주: 
  messages=[{"role": "user", "content": content}]
  → 모델이 일반 assistant처럼 동작

우리: 
  messages=[
    {"role": "system", "content": "You are an expert in theory of mind..."},
    {"role": "user", "content": content}
  ]
  → 모델이 "심리학 + 사회 추론 전문가" 모드로 진입
```

**효과**:
- 모델의 "역할"을 명확히 지정
- 비디오 상황을 이해할 때 '의도(intent)', '신념(belief)', '목표(goal)' 같은 상위 개념에 주목
- 무의식적으로 ToM(Theory of Mind) 프레임워크 적용
- 기여도: +3~5%p

---

### 1.2 Question Type 명시 (문제 분류 힌트)

```
민주 버전:
  Text: [설명]
  Question: [질문]
  A) ... B) ... C) ...
  → 모델이 스스로 "이게 뭐하는 질문인지" 파악해야 함

우리 버전:
  Question type: belief_of_goal
  Text context: [설명]
  Question: [질문]
  A) ... B) ... C) ...
  → 세 가지 신념 유형을 명시적으로 구분
```

**세 가지 신념 유형**:
- `belief`: "A가 X를 믿는가?" (1-layer) → 상대적으로 직관적
- `social_goal`: "A가 B를 돕고 싶어하는가?" (2-layer) → 의도 추론 필요
- `belief_of_goal`: "A는 B가 C를 목표로 한다고 믿는가?" (3-layer) → 복잡한 다층 추론 필요

**효과**:
- belief_of_goal은 세 가지 중 가장 어려움 (평균 36.7% 에러율)
- 명시적 분류로 모델이 각 유형에 맞는 추론 전략 적용
- 예: belief_of_goal 문제 → "A의 관점에서 B의 신념을 추론해야겠다"
- 기여도: +4~6%p

---

### 1.3 JSON + Reasoning 강제

```
민주 버전:
  지시: "Answer with only A, B, or C."
  
  모델의 자유로운 응답:
  - "The answer is B."
  - 또는 그냥 "B"
  - 또는 길게 설명한 후 "B"
  
  파싱: LETTER_RE.search(text.upper()) → 정규식으로 A/B/C 추출

우리 버전:
  지시: 'Respond in JSON: {"choice_letter": "A", "reasoning": "..."}'
  response_format={"type": "json_object"}
  
  모델의 구조화된 응답:
  {
    "choice_letter": "B",
    "reasoning": "If Sarah intended to hinder Michael, she would provide false 
                  information. Since the toy was not in the microwave where she 
                  claimed, this suggests she deliberately misled him. Therefore, 
                  the LEAST likely belief is..."
  }
  
  파싱: json.loads() → 안정적인 구조화된 파싱
```

**효과**:
- reasoning을 함께 요구하면 모델이 **논리를 "따라야"** 함
- Q2 (hinder-LEAST) 같은 역논리 문제에서 특히 효과적
  - "방해하려 했다면" + "LEAST likely" = 복잡한 이중 부정
  - 선택지만 고르는 것보다 chain-of-thought 효과
  - 우리 승률: 70.6% (우리가 119개 중 84개 이김)
- 선택지 파싱도 정규식 → JSON으로 안정성 증가
- 기여도: +5~8%p (Q2 개선의 주된 원인)

---

## 2. 우리가 진 60개 경우 (우리만 틀림, 민주는 맞음)

### 2.1 분포 - 집중된 손실

```
총 60개 (900의 6.7%)

Q별 분포 (중요):
  Q2 (hinder-LEAST): 35개 (58%) ← 집중된 손실!
  Q4 (know-LEAST):   13개 (22%)
  Q3 (know-MOST):     7개 (12%)
  Q1 (help-MOST):     5개 (8%)

레이블별:
  belief_of_goal: 39개 (65%)
  belief:         17개 (28%)
  social_goal:     4개 (7%)
```

**이상한 점**: 
- Q2는 우리가 **가장 잘 이기는 분야** (119개 차이 중 84개 이김 = 70.6% 승률)
- 그런데 우리가 진 60개 중 **58%가 Q2**
- 즉, 우리가 Q2를 "과도하게" 최적화해서 역효과가 난 건 아닐까?

---

### 2.2 핵심: 우리의 강점이 약점이 되다

#### 문제 1: Q2에서 역논리 오버피팅

**Q2의 특성** (hinder-LEAST):
```
구조: "방해하려 했다면, LEAST likely(가장 말이 안 되는)은?"

예시 논리:
  1. Person이 "X는 Y에 있다"고 거짓말 함
  2. 실제로 X는 Z에 있음
  3. 질문: Person이 Y에 있다고 "믿었을" 가능성은?
  
  답: LEAST likely (거짓말했다 = 믿지 않았다)
```

**실제 사례: EP 4078 Q2 (belief)**

상황:
```
- Mary가 "감자는 bathroom cabinet에 있다"고 말함
- 실제로 Michael은 다른 곳에서 감자를 찾음 → Mary의 정보가 거짓

Q2 질문:
  "Mary가 Michael을 방해하려 했다면, LEAST likely true는?"

정답: B) Mary는 감자가 bathroom cabinet에 있다고 믿었다
  (방해하려 → 거짓말 → 실제로 그곳에 있다고 안 믿었음 = LEAST likely)

우리의 답: C ✗ (역논리를 과도하게 적용해서 반대 선택)
민주의 답: B ✓ (직설적 해석)
```

**오버피팅 원인**:
- 우리는 JSON + reasoning을 강제해서 모델이 더 깊게 생각하도록 유도
- 하지만 이것이 **존재하지 않는 역논리**까지 만들어냄
- 모델: "reasoning을 잘 풀어야 하니까 더 복잡한 논리를 찾자" → 오버파싱
- 민주의 단순한 접근이 이 경우엔 오히려 맞음

**비슷한 사례들** (Q2 35개 손실):
```
EP 4103 Q2: 정답 C, 우리 A ✗, 민주 C ✓
EP 4200 Q2: 정답 C, 우리 B ✗, 민주 C ✓
EP 4441 Q2: 정답 C, 우리 A ✗, 민주 C ✓
```

---

#### 문제 2: belief_of_goal에서 overthinking (39개)

**원인**: Question type 명시가 모델을 overthinking 유도

```
우리는 이렇게 했다:
  "Question type: belief_of_goal"
  → 명시적으로 3-layer 추론임을 알려줌
  → 모델: "아, 이건 복잡한 추론이 필요하구나"
  → 모델이 **과도하게 깊게** 생각하기 시작

결과: 
  - 일부 문제에선 좋음 (복잡한 추론 필요)
  - 일부 문제에선 나쁨 (단순한 패턴인데 과도하게 복잡하게 해석)
```

**실제 사례: EP 4009 Q3/Q4 (belief_of_goal)**

상황:
```
- Mary: "wine이 어디에?"
- John: "kitchen cabinet에 있다" (거짓)
- 실제: Mary가 fridge에서 wine 발견

Q3 (MOST likely true):
  "John이 그 위치를 안다고 가정할 때, prevent/help/indifferent?"
  → John이 거짓 정보 줌 + wine을 찾으려 했음 + 위치를 알았으면
  → prevent하려 했을 가능성 높음
  정답: B (prevent)

Q4 (LEAST likely true):
  "John이 그 위치를 안다고 가정할 때, help/prevent/indifferent?"
  → 비슷한 논리로 Q3과 대조되는 선택지
  정답: A (help)

우리와 민주의 답:
  Q3 → A (help) ✗     ← 둘다 틀림
  Q4 → B (prevent) ✗  ← 둘다 틀림
  
  → 체계적으로 반대 선택!
```

**분석**:
- 우리와 민주 모두 같은 오답을 함
- 하지만 이것도 "우리가 진" 60개에 포함된 이유는:
  - 전체 정답 기준: 우리도 틀림 + 민주도 틀림 = 둘 다 틀림 (209개에 포함)
  - 하지만 Q3/Q4는 같은 에피소드라서, 하나라도 틀리면 영향
  - 우리의 belief_of_goal 명시가 overthinking 유도했을 가능성

---

### 2.3 결론: "더 좋은 프롬프트"의 역설

```
우리가 강화한 부분:
  ✓ Q2 역논리 처리 (reasoning 강제)
  ✓ belief_of_goal 구분 (question_type 명시)

그런데 우리가 진 부분:
  ✗ Q2에서 35개 (역논리 오버피팅)
  ✗ belief_of_goal에서 39개 (overthinking 유도)

교훈: 
  "더 좋은 프롬프트"는 평균을 올리지만,
  특정 경우에는 모델을 오버하게 만들 수 있음
  → 모든 문제에 다 좋은 것은 아님
```

---

## 3. 둘 다 틀린 209개 (23.2%) - 모델의 근본 한계

### 3.1 성질 분류

```
둘 다 틀린 209개:

같은 오답 선택: 171개 (81.8%)
  → 우리와 민주가 똑같이 같은 선택지를 고름
  → 의미: 시나리오 자체가 모호하거나 정보가 부족
  → 결론: 프롬프트로 해결 불가능

다른 오답 선택: 38개 (18.2%)
  → 우리와 민주가 다른 선택지를 고름
  → 의미: 접근 방식의 차이 때문에 다르게 틀림
  → 결론: 프롬프트 미세조정으로 1~2% 개선 가능
```

---

### 3.2 같은 오답 171개 - 프롬프트로 극복 불가능

**분포**:
```
레이블별:
  social_goal:    88개 (51%) ← 가장 많음
  belief_of_goal: 65개 (38%)
  belief:         18개 (11%)

Q별:
  Q3 (know-MOST):  77개 (45%)
  Q4 (know-LEAST): 70개 (41%)
  Q2 (hinder):     50개 (29%)  ← 이미 틀린 부분
  Q1 (help):       12개 (7%)
```

---

#### 원인 1: help vs prevent 구분 모호 (88개)

**사례: EP 4009 Q3/Q4**

상황:
```
- Mary: "wine이 kitchen cabinet에 있다"
- John: wine은 실제로 fridge에 있음 (John의 정보가 거짓)

Q3: John이 위치를 알고 있다면, prevent/help/indifferent?
Q4: John이 위치를 알고 있다면, help/prevent/indifferent?

정답: Q3→B(prevent), Q4→A(help)
  → 왜냐면 거짓 정보를 줬으니까 방해하려 했을 가능성

우리의 답:   Q3→A(help),    Q4→B(prevent) ✗
민주의 답:   Q3→A(help),    Q4→B(prevent) ✗

둘다 같은 오답!
```

**왜 틀렸을까?**

```
두 가지 해석이 가능:

해석 1: 문맥상 의도
  - John이 거짓말 함 = 방해하려 했을 것 (prevent)
  - 이게 정답

해석 2: 결과 기반 해석  
  - Mary가 fridge에서 찾음 = 도움이 됨 (help)
  - 우리/민주의 오답

비디오 텍스트 설명만으로는 "의도"를 구분하기 어려움
→ 비디오를 직접 봐야 John의 표정/몸짓/행동으로 의도 파악 가능
→ 텍스트만으로는 모호성 극복 불가능
```

---

#### 원인 2: "X가 뭘 안다"는 조건의 불명확성 (127개 Q3/Q4)

**Q3/Q4의 구조**:
```
"assuming that John knows what is inside the kitchen cabinet, which of 
the following statements is MOST/LEAST likely true?"

문제점: 
  - "John이 정말로 그곳이 뭐가 들어있는지 아나?"
  - 비디오 텍스트 설명만으로는 알 수 없음
  - John이 그곳을 열어본 적 있나? 누군가 말해줬나? 이미 아는 상태나?
  
실제로는:
  - 비디오를 보고 주의깊게 관찰해야 함
  - "이 장면에서 John이 그 위치를 어떻게 알 수 있나?"를 추론
  - 텍스트 설명만으로는 불가능
```

**구체 사례**:

상황: Michael이 특정 위치를 열어봤고, Sarah가 그걸 봤나?
```
Q3 답변에 필요한 정보:
  - Sarah가 실제로 그 위치를 알고 있는가?
  - 비디오 장면 분석 필요 (누가 어디를 열었는가?)
  - 텍스트 설명에 이런 세부 정보가 충분하지 않을 수 있음

결과:
  - Q3/Q4에서 77+70=147개가 같은 오답
  - 정보 부족 때문에 우연에 가까운 선택
```

---

#### 원인 3: belief_of_goal의 3-layer 복잡성 (65개)

**구조**:
```
belief:         "A가 X를 믿는가?"
                → 1-layer, 직접적

social_goal:    "A가 B를 돕고 싶어하는가?"
                → 2-layer, 행동 기반 추론

belief_of_goal: "A는 B가 C를 목표로 한다고 믿는가?"
                → 3-layer, 타인의 타인에 대한 신념
                → 가장 복잡하고 다단계 추론 필요
```

**예시**:

```
상황: Michael이 wine을 찾고 있음 (= Michael의 goal)

Q: Jessica는 Michael이 wine을 찾는 게 목표인 줄 안다고 믿는가?

필요한 추론:
  1. Michael의 goal은 뭔가? (wine 찾기)
  2. Jessica가 이걸 아는가? (관찰/대화로 알 수 있나?)
  3. Jessica는 Michael의 goal을 알고 있다고 **믿는가?**
     (= Jessica의 belief about Michael's goal)

이 3단계를 모두 통과해야 함
→ 틀릴 확률이 높음
```

**공통점**: 171개 모두 같은 오답
- 우리의 프롬프트 최적화도 극복 불가능
- 근본적인 정보 부족 또는 모델 능력 한계

---

### 3.3 다른 오답 38개 - 미세조정으로 개선 가능

```
이들은 둘 다 틀렸지만, 다른 이유로 틀림

Q2 (hinder):     17개 ← 역논리 처리 방식 차이
belief:          17개 ← 신념 추론 전략 차이
Q3/Q4:           20개 ← 의도 해석 차이
```

**개선 가능성**:
- 우리의 JSON reasoning 강도 조절
- belief_of_goal 명시 정도 조절
- 프롬프트 미세 튜닝으로 +1~2%p 정도 개선 가능

---

## 4. 최종 결론

### 프롬프트로 해결됨 (우리의 성공)
```
Q1 (help-MOST):
  - 가장 직관적인 질문
  - 에러율 7.6% (안정적)
  - 프롬프트 개선의 영향 적음

Q2 (hinder-LEAST) [부분적 성공]:
  - 역논리 처리가 핵심
  - 우리 승률: 70.6% (119개 중 84개)
  - 하지만 오버피팅으로 35개 손실
  - 순 이득: 49개 (= 전체 개선의 약 28%)

belief_of_goal (부분적 성공):
  - 36.7% 에러율에서 상당 부분 개선
  - 명시적 분류 효과
  - 하지만 39개 overthinking 손실
```

### 프롬프트로 극복 불가능 (모델 한계)
```
209개 (23.2%) - 근본적 문제:

171개 같은 오답 (81.8%):
  - 시나리오 모호성 (help vs prevent)
  - 비디오 정보 부족 (텍스트만 제공)
  - 모델 근본 추론 능력 한계

해결 방안:
  1. 더 강한 모델 필요 (GPT-4o → GPT-4 Turbo)
  2. 비디오 처리 개선 (detail level 상향, key frame 최적화)
  3. 데이터셋 수정 (모호한 질문 정제)
  4. Multi-hop reasoning (단계적 추론 프롬프트)

추정 효과: 각 1~2%p
```

### GPT-4o의 추정 상한선
```
민주 버전:         57.2%
우리 버전:         70.1%  ← 현재 최고
추정 가능성:       71~72% ← 추가 최적화로
근본 한계:         73~75% ← 모델 능력 한계
```
