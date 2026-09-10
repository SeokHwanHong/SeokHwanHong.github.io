---
layout: single
title: "반도체 8대 공정은 어떤 흐름으로 진행될까"
categories:
  - "Semiconductor Industry"
author_profile: true
toc: true
toc_sticky: true
---

반도체 8대 공정을 처음 공부할 때는 각 장비나 화학반응부터 외우기보다 **웨이퍼 위에 회로를 만들고, 전기적 성질을 부여하고, 연결하고, 검사한 뒤 제품으로 완성하는 전체 흐름**을 먼저 이해하는 것이 좋다.

또한 ‘8대 공정’은 실제 수백 개의 세부 공정을 이해하기 위한 대표적인 분류다. 실제 Fab에서는 산화, 포토, 식각, 증착, 세정, 평탄화, 검사 등 많은 공정이 반복된다.

## 1. 전체 흐름부터 보기

반도체 8대 공정은 크게 다음과 같이 정리할 수 있다.

```text
① 웨이퍼 제조
       ↓
② 산화
       ↓
③ 포토
       ↓
④ 식각
       ↓
⑤ 증착·이온주입
       ↓
   위 공정들을 반복
       ↓
⑥ 금속배선
       ↓
⑦ EDS
       ↓
⑧ 패키징
```

중요한 점은 ②~⑤가 한 번씩만 진행되는 것이 아니라는 것이다.

```text
막을 만든다
↓
가공할 위치를 정한다
↓
깎는다
↓
필요한 물질을 쌓거나 전기적 성질을 바꾼다
↓
다시 반복한다
```

이 과정을 여러 번 반복하면서 웨이퍼 위에 트랜지스터와 다양한 회로 구조를 만든다.

## 2. 웨이퍼 제조 — 반도체를 만들 바닥을 준비한다

모든 공정의 시작은 **Wafer**다.

고순도 실리콘으로 단결정 Ingot을 만든 뒤 얇게 절단하고 표면을 연마해 웨이퍼를 만든다.

```text
고순도 Silicon
↓
Ingot
↓
절단
↓
연마
↓
Wafer
```

가장 쉽게 생각하면

> **Wafer = 반도체 회로를 만들 도화지**

라고 볼 수 있다.

## 3. 산화 — 보호막과 절연막을 만든다

실리콘 웨이퍼 표면에 **SiO₂ 산화막**을 형성한다.

산화막은

- 전기적 절연
- 표면 보호
- 이온주입이나 식각 시 특정 영역 보호

등의 역할을 한다.

```text
SiO₂
████████
Si Wafer
────────
```

즉 산화공정은

> **웨이퍼 위에 필요한 보호·절연막을 만드는 공정**

이다.

## 4. 포토 — 어디를 가공할 것인지 표시한다

웨이퍼 위에 빛에 반응하는 **Photoresist(PR)**를 바르고, 회로 패턴이 들어 있는 Mask를 이용해 빛을 조사한다.

노광과 현상을 거치면 원하는 위치에만 PR Pattern이 남는다.

포토공정은 실제로 웨이퍼를 깎는 과정이 아니다.

> **“어디를 남기고 어디를 가공할 것인가”를 표시하는 과정**

이다.

반도체 회로가 작아질수록 더 작고 정확한 패턴을 만들어야 하기 때문에 EUV와 같은 첨단 노광기술이 중요해진다.

## 5. 식각 — 표시한 위치를 실제로 깎는다

포토공정에서 만든 PR Pattern을 기준으로 필요 없는 부분을 제거한다.

```text
Photo
→ 가공할 위치를 결정

Etch
→ 실제 재료를 제거
```

식각은 크게 Wet Etch와 Dry Etch로 나눌 수 있다.

- **Wet Etch**: 액체 화학물질을 이용
- **Dry Etch**: Plasma와 반응성 Gas를 이용

미세공정에서는 원하는 방향으로 정밀하게 깎는 것이 중요하기 때문에 Dry Etch가 특히 중요하다.

## 6. 증착과 이온주입 — 쌓고 전기적 특성을 만든다

### 증착(Deposition)

웨이퍼 위에 필요한 물질을 매우 얇게 형성하는 공정이다.

```text
Thin Film
████████
Wafer
────────
```

대표적인 방식은

- CVD
- PVD
- ALD

등이다.

반도체 구조가 3차원화되고 미세해질수록 얇은 막을 얼마나 균일하게 형성하는지가 중요해진다.

### 이온주입(Ion Implantation)

순수 실리콘에 Boron, Phosphorus, Arsenic 등의 불순물을 주입해 전기적 특성을 바꾼다.

이를 통해 p-type과 n-type 영역을 만들고 트랜지스터가 동작할 수 있는 전기적 특성을 형성한다.

즉

> **증착 = 필요한 물질을 쌓는다**  
> **이온주입 = 실리콘의 전기적 성질을 바꾼다**

라고 이해하면 된다.

## 7. 가장 중요한 반복 구조

전공정의 핵심은 다음 과정이 반복된다는 것이다.

```text
증착
→ 재료를 쌓는다

포토
→ 위치를 정한다

식각
→ 원하는 구조로 깎는다

이온주입
→ 전기적 특성을 만든다
```

필요에 따라 다시 산화, 증착, 포토, 식각, 세정, 평탄화 등의 공정을 반복한다.

그 결과 웨이퍼 위에

- Transistor
- Gate
- Capacitor
- Contact

등의 구조가 형성된다.

이처럼 **소자 자체를 만드는 단계**를 주로 FEOL(Front End Of Line)이라고 한다.

## 8. 금속배선 — 만들어진 소자들을 연결한다

트랜지스터를 많이 만드는 것만으로는 회로가 동작하지 않는다.

각 소자를 전기적으로 연결해야 한다.

```text
Transistor   Transistor   Transistor
     │           │           │
─────┴───────────┴───────────┴──── Metal
```

이처럼 금속으로 신호가 이동할 통로를 만드는 것이 **Metal Interconnect**다.

소자를 만든 뒤 배선을 형성하는 과정은 일반적으로 BEOL(Back End Of Line)이라고 한다.

```text
FEOL
→ Transistor 등 소자 자체를 만든다

BEOL
→ 만들어진 소자들을 배선으로 연결한다
```

## 9. EDS — 웨이퍼에서 정상 칩을 골라낸다

웨이퍼 한 장에는 수많은 Die가 만들어지지만 모든 Die가 정상인 것은 아니다.

```text
Wafer

○ ○ ○ X ○
○ X ○ ○ ○
○ ○ ○ ○ X
```

EDS(Electrical Die Sorting)는 Probe Card 등을 이용해 각 Die에 전기적 신호를 넣고 정상 동작 여부를 검사하는 과정이다.

주요 목적은

- 정상·불량 Die 분류
- 수리 가능한 Die 확인
- 불필요한 후공정 비용 감소
- 공정·설계 문제 확인

등이다.

이 과정은 **수율(Yield)**과 직접 연결된다.

```text
Yield
=
정상 Chip 수 / 전체 Chip 수
```

정도로 우선 이해하면 된다.

## 10. 패키징 — Die를 실제 제품으로 만든다

검사가 끝난 웨이퍼를 개별 Die로 자르고, 외부 기기와 연결할 수 있도록 Package로 만든다.

```text
Wafer
↓
Dicing
↓
Die
↓
기판 부착
↓
외부 전기 연결
↓
보호
↓
Final Test
↓
Package
```

과거에는 Packaging을 단순한 ‘포장’ 과정으로 생각했지만, AI 시대에는 성능을 결정하는 핵심 기술이 됐다.

HBM을 예로 들면 여러 DRAM Die를 TSV로 연결하고 Base Die 및 GPU와 함께 패키징해야 하기 때문에 적층, 열관리, 배선, 수율이 모두 중요하다.

## 11. 8대 공정 밖에도 중요한 공정이 많다

실제 Fab에서는 8대 공정 외에도 다음 공정들이 매우 중요하다.

### Cleaning

공정 중 발생한 Particle, 유기물, 금속 오염 등을 제거한다.

### CMP

층을 반복해서 쌓으면 표면이 울퉁불퉁해지므로 Chemical Mechanical Polishing을 통해 표면을 평탄화한다.

### Annealing / Diffusion

열처리를 통해 이온주입된 불순물을 활성화하거나 재료의 구조와 특성을 조정한다.

### Inspection / Metrology

공정 중 패턴, 두께, 결함, Overlay 등을 측정해 공정이 정상적으로 진행되는지 확인한다.

즉

> **8대 공정은 실제 수백 개의 세부 Step을 이해하기 위한 큰 지도**

라고 보는 것이 가장 정확하다.

## 12. 한눈에 정리하면

| 공정 | 핵심 역할 |
| --- | --- |
| 웨이퍼 | 반도체를 만들 실리콘 기판 준비 |
| 산화 | 보호·절연막 형성 |
| 포토 | 가공할 위치와 패턴 정의 |
| 식각 | 패턴에 따라 재료 제거 |
| 증착·이온주입 | 물질 적층·전기적 특성 부여 |
| 금속배선 | 소자들을 전기적으로 연결 |
| EDS | 웨이퍼 상태에서 정상 Die 선별 |
| 패키징 | Die를 연결·보호해 실제 제품화 |

## 13. 앞으로의 공부 순서

이제부터는 각 공정을 하나씩 깊게 살펴본다.

```text
1. 웨이퍼·산화
→ 출발 재료와 절연막

2. 포토
→ 미세한 패턴을 어떻게 만드는가

3. 식각
→ 패턴대로 어떻게 정밀하게 깎는가

4. 증착
→ 원자 수준의 박막을 어떻게 형성하는가

5. 이온주입
→ 반도체의 전기적 성질을 어떻게 만드는가

6. 금속배선
→ 수많은 Transistor를 어떻게 연결하는가

7. EDS·수율
→ 불량을 어떻게 찾고 공정을 평가하는가

8. Packaging
→ Die를 실제 제품으로 어떻게 만드는가
```

그리고 각 공정을 공부할 때 단순히 원리만 보는 것이 아니라

> **공정 원리 → 주요 장비 → 공정변수 → Defect → 수율 → 데이터**

순서로 연결해보면 반도체 제조현장에서 어떤 문제가 발생하고 어떤 데이터가 사용되는지도 함께 이해할 수 있다.

## 참고자료

- [Samsung Semiconductor - Eight Essential Semiconductor Fabrication Processes](https://semiconductor.samsung.com/kr/support/tools-resources/fabrication-process/)
- [SK hynix Newsroom - Semiconductor Front-end Process](https://news.skhynix.com/en/semiconductor-front-end-process-episode-2/)
- [SK hynix Newsroom - Understanding Semiconductor Manufacturing Technologies](https://news.skhynix.com/en/sk-hynix-publishes-technical-book-understanding-semiconductor-manufacturing-technologies-to-stimulate-the-semiconductor-ecosystem/)
