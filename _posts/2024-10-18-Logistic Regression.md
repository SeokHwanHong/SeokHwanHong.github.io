---
layout: single        # 문서 형식
title: Logistic Regression # 제목
categories: Machine Learning    # 카테고리
tag: [ML, Statistics, Classification]
author_profile: false # 홈페이지 프로필이 다른 페이지에도 뜨는지 여부
sidebar:              # 페이지 왼쪽에 카테고리 지정
    nav: "counts"       # sidebar의 주소 지정
#search: false # 블로그 내 검색 비활성화
use_math: true
---
# Keywords
Machine Learning, Statistics, Classification, Supervised Learning



# 1. Definition
Logistic Regression 은 회귀 계수(Weight Coefficients) 를 학습해 주어진 입력 데이터 $X$ 가 특정 클래스에 속할 확률을 추정한다. 이를 위해 로지스틱 함수 또는 시그모이드 함수를 사용한다. 이는 입력값을 0과 1사이의 확률값으로 변환한다.



# 2. Logistic Regression
## 2.1. Logistic Function
로지스틱 함수 (또는 시그모이드 함수) 는 로지스틱 회귀에서 예측값을 확률로 변환하기 위해 사용하는 함수다. 이는 다음과 같이 정의된다.

$$
\begin{split}
    \sigma(z) = \frac{1}{1 + e^{-z}}
\end{split}
$$

이는 입력값 $z$ 의 크기에 상관없이 항상 0과 1사이의 값을 반환한다. 이 때, 임의의 임계값을 지정해 임계값보다 크면 클래스 1로, 작다면 0으로 분류한다. 즉, 어떤 데이터가 어느 한 클래스에 속할 확률을 반환한다.


## 2.2. Statistical Representation
#### - Notation
$x$ : 입력 데이터

$y$ : 주어진 입력 데이터의 클래스

$$\mathbb{P} (y = 1 \vert x)$$ : 입력 데이터 $$x$$ 가 클래스 1에 속할 확률

$w$ : 가중치

$b$ : 편향


#### - Logistic Regression
로지스틱 회귀는 다음과 같이 표현된다.

$$
\begin{split}
    \mathbb{P} (y = 1 \vert x) = \frac{1}{\exp[-(w \cdot x + b)]}
\end{split}
$$


## 2.3. Log Likelihood Function
로지스틱 회귀는 모형의 $w$ 와 $b$ 를 학습하기 위해 로그 가능도 함수를 최대화하는 방향으로 작동한다. 즉, 모형이 실제 데이터에 대해 올바른 확률을 반환하도록 가중치를 조정한다. 로지스틱 회귀에서 로그 가능도 함수는 다음과 같다.

$$
\begin{split}
    l(w,b) = \sum_{i=1}^n \left[ y_i \log \mathbb{P}(y_i = 1 \vert x_i) + (1 - y_i) \log (1 - \mathbb{P}(y_i = 1 \vert x_i)) \right]
\end{split}
$$

로그 가능도 함수를 최대화하는 것은 경사 하강법과 같은 최적화 알고리즘을 사용해 $w$ 와 $b$ 를 학습하는 것과 유사하다. 



# 3. Advantages & Disadvantages of Logistic Regression
#### - Advantages
1. 예측 결과가 확률로 해석되기 때문에 결과에 대한 직관적 이해가 가능

2. 계산이 빠르고 비용이 작아 효율적

3. 다중 클래스 문제로 확장 가능

#### - Disadvantages
1. 선형적인 결정 경계 사용으로 인해 데이터에 따라 불안정

2. 이상치에 민감

3. 다중 공선성에 취약



# 4. Multi-Class Logistic Regression
다중 클래스 로지스틱 회귀는 클래스가 3개 이상인 경우 각 클래스에 속할 확률을 예측하는 모형이다. 이진 로지스틱 회귀에서는 시그모이드 함수를 사용하는 반면, 여기서는 소프트맥스 함수를 이용해 확률을 계산한다. 


## 4.1. Softmax Function
소프트맥스 함수는 각 클래스의 예측 함수를 확률로 변환해 입력 데이터가 각 클래스에 속할 확률을 반환한다. 이는 다음과 같다.

$$
\begin{split}
    \mathbb{P} (y = k \vert x) = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)} \quad \text{for} \quad k = 1, \dots , K
\end{split}
$$

여기서 $$z_k = w_k \cdot x + b_k$$ 는 클래스 k 에 대한 점수(logits)다. 

## 4.2. Cross-Entropy Loss
모형 학습 시 cross-entropy 손실 함수를 사용해 가중치를 학습한다. 이는 모델이 예측한 확률과 실제 클래스 간 차이를 최소화하는 방향으로 가중치를 갱신한다. 이는 다음과 같이 정의된다.

$$
\begin{split}
    L(w, b) = -\sum_{i=1}^n \sum_{k=1}^K y_{ik} \log \mathbb{P} (y = k \vert x_i)
\end{split}
$$

여기서 $y_{ik}$ 는 $i$ 번째 데이터가 클래스 k 에 속하는지 여부를 나타내는 이진값(0 또는 1) 이다. 다중 클래스 로지스틱 회귀에서의 목표는 이 손실함수를 최소화해 모형의 가중치와 절편을 학습하는 것이다.

## 4.3. OvR Training
OvR (One versus Rest 또는 One versus All) 학습 방식은 각 클래스에 이진 로지스틱 회귀를 여러 번 적용해 분류기를 학습하는 방식이다. 소프트맥스 함수보다 구현이 쉽지만 개별적인 이진 분류기를 학습해야하기 때문에 더 많은 모형을 학습해야하는 단점이 있다.

