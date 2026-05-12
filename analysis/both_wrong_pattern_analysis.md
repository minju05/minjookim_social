# 공통 오답 패턴 분석 (둘 다 틀린 209개 사례)

## 📊 개요

| 분류 | 개수 | 설명 |
|------|------|------|
| **같은 오답** | 171 | 두 모델이 동일한 잘못된 답 선택 |
| **다른 오답** | 38 | 두 모델이 서로 다른 잘못된 답 선택 |
| **총계** | 209 | 민주 vs 우리 모두 틀린 경우 |

---

## 🔴 패턴 1: 같은 오답 (171개) - Help/Prevent 혼동

### 가장 심각한 패턴: Help ↔ Prevent 역전

두 모델 모두 **help(돕다)와 prevent(막다)의 개념을 정반대로 혼동**합니다.

#### 패턴 특징
- **주로 발생 위치**: social_goal 질문 (Q3, Q4)
- **문제 유형**: belief_of_goal (3층 추론)
- **혼동 패턴**:
  - Ground Truth: "prevent X from finding Y" → 모델 예측: "help X locate Y"
  - Ground Truth: "help X locate Y" → 모델 예측: "prevent X from finding Y"

#### 구체적 사례들

**사례 1: EP4009 Q3-Q4 (social_goal)**
```
Context: Mary walked into kitchen → bathroom
         John entered kitchen → cabinet (found wine)

Q3 Ground Truth: John has been trying to PREVENT Mary from finding the wine
   Model Prediction: John has been trying to HELP Mary locate the wine  ❌
   Model Answer: A

Q4 Ground Truth: John has been trying to HELP Mary locate the wine
   Model Prediction: John has been trying to PREVENT Mary from finding the wine  ❌
   Model Answer: B
```
→ **정확히 반대로 맞춤** (도와주려는 의도를 방해로, 방해하려는 의도를 도움으로)

**사례 2: EP4017 Q3-Q4**
```
Q3 Ground Truth: Sarah preventing Michael from toy → Model: "helping"  ❌
Q4 Ground Truth: Sarah helping Michael with toy → Model: "preventing"  ❌
```

**사례 3: EP4043 Q3-Q4**
```
Q3 Ground Truth: Sarah preventing Michael from juice → Model: "helping"  ❌
Q4 Ground Truth: Sarah helping Michael with juice → Model: "preventing" (A 선택)  ❌
```

**사례 4: EP4077 Q3-Q4, EP4078 Q3-Q4**
- 동일한 help/prevent 패턴 반복 발생
- **Q3와 Q4가 대칭 구조이지만 두 모델 모두 일관되게 혼동**

### 분석: 왜 이런 일이?

1. **비디오 정보 부족**
   - 시각적으로 "help" vs "prevent" 행동의 결과가 명확하지 않음
   - 텍스트 설명만으로는 의도 파악 어려움

2. **Question Type: social_goal의 복잡성**
   - 단순 관찰 (belief)이 아닌 **의도 추론** 필요
   - 3층 구조: 캐릭터 행동 → 캐릭터의 의도 → 타인에게 주는 영향

3. **Q3/Q4 구조상 문제**
   - Q3: "know_MOST" (가장 많이 알 사람을 묻는 질문)
   - Q4: "know_LEAST" (가장 적게 알 사람)
   - 이 구조가 help/prevent와 상충할 수 있음

---

## 🟡 패턴 2: 다른 오답 (38개) - 각각 다르게 틀림

### 특징
- **주로 발생**: belief 질문의 Q2 (14/24 cases)
- **특징**: 두 모델이 **전혀 다른 오답을 선택**
- **의미**: 각 모델이 다른 전략/오류를 사용하고 있음을 시사

#### 구체적 사례들

**사례 1: EP4009 Q2 (belief)**
```
Ground Truth: John believed wine was inside kitchen cabinet
Minju Predicted: C
Ours Predicted: B
→ 두 모델이 모두 틀렸지만 다른 보기 선택
```

**사례 2: EP4034 Q2 (belief)**
```
Ground Truth: Jessica believed beer was inside stove
Minju Predicted: C
Ours Predicted: A
```

**사례 3: EP4057 Q2 (belief)**
```
Ground Truth: Sarah believed juice inside microwave
Minju Predicted: A
Ours Predicted: C
```

### 이 패턴의 의미

1. **두 모델의 전략 차이**
   - 민주 버전: 특정 편향 (특정 보기로 쏠림)
   - 우리 버전: JSON 강제 + reasoning으로 다른 편향 발생
   - → "더 나은 시스템 프롬프트"도 **다른 종류의 오류 도입**

2. **belief 질문의 어려움**
   - Q2는 "역논리" 처리 필요
   - "X가 Y 위치에 가서, Z를 만들었으므로 X는 Y가 안 Y라고 생각"
   - 긴 추론 체인에서 **각 모델이 다르게 중단**

---

## 📈 질문 타입별 분석

### belief (24 cases)
| 분류 | Q1 | Q2 | Q3 | Q4 |
|------|----|----|----|----|
| 같은 오답 | 0 | 7 | 3 | 0 |
| 다른 오답 | 1 | 10 | 2 | 1 |
| **총계** | 1 | 17 | 5 | 1 |

→ **Q2에서 집중됨** (17/24 = 70.8%)
→ **Q2 (역논리)가 둘 다 풀기 어려움**

### belief_of_goal (121 cases)
| 분류 | Q1 | Q2 | Q3 | Q4 |
|------|----|----|----|----|
| 같은 오답 | 11 | 23 | 38 | 31 |
| 다른 오답 | 1 | 3 | 7 | 7 |
| **총계** | 12 | 26 | 45 | 38 |

→ **Q3, Q4에서 가장 많음** (83/121 = 68.6%)
→ **3층 추론 문제 (belief_of_goal)가 가장 어려움**

### social_goal (64 cases)
| 분류 | Q1 | Q2 | Q3 | Q4 |
|------|----|----|----|----|
| 같은 오답 | 0 | 3 | 27 | 28 |
| 다른 오답 | 0 | 0 | 5 | 1 |
| **총계** | 0 | 3 | 32 | 29 |

→ **Q3, Q4의 help/prevent 혼동** (59/64 = 92%)
→ **의도 추론 과제에서 극명한 편향**

---

## 🎯 핵심 인사이트

### 1. 세 가지 주요 난제 (공통)

| 난제 | 사례 수 | 원인 | 해결 방법 |
|------|--------|------|---------|
| **Help/Prevent 혼동** | ~171 | 비디오 불충분, 의도 모호 | 더 명확한 인물 행동 시각화 |
| **3층 추론 (belief_of_goal)** | 121 | 연쇄 추론 누적 오류 | CoT + 중간 검증 필요 |
| **Q2 역논리** | 17 (belief) | 역으로 가는 추론 복잡 | 명시적 역논리 프롬프트 |

### 2. 두 모델의 오류 패턴

**민주 버전 (57.2%)**
- 더 단순하고 일관된 오류
- 같은 오답에 집중 (시스템 오류)

**우리 버전 (70.1%)**
- 더 복잡한 reasoning 시도
- 다른 오답도 증가 (전략 분산)
- **그럼에도 같은 오답(help/prevent)에서 여전히 실패**
  → 프롬프트 개선으로 해결 안 됨, **데이터/시각화 문제**

### 3. 결론

**공통으로 틀린 209개는 모두:**

1. **시각 정보 부족** (비디오에서 의도를 읽기 어려움)
2. **다층 추론 복잡성** (특히 belief_of_goal)
3. **데이터셋 특성** (help vs prevent를 비디오만으로 판단 불가)

→ **프롬프트 최적화의 한계에 도달**
→ **다음 단계**: 액션 추출 + 명시적 의도 모델링 필요

---

## 📋 결론: 도움이 될 만한 개선안

1. **Help/Prevent 명확화**
   - 프롬프트에 "help는 X가 Y를 찾는 것을 용이하게" 등 정의 추가
   - 또는 액션 수준에서 명시적 구분

2. **3층 추론 분해**
   - belief_of_goal은 중간 단계 검증 필요
   - "First, what did Person1 see?", "Then, what does Person2 know about Person1?"

3. **Q2 특수 처리**
   - 역논리 문제 인식
   - "NOT", "prevent", "don't know" 등에 특수 마스킹

4. **시각적 강화**
   - 더 많은 frame 또는 scene description
   - 행동 결과까지 명확히 표현
