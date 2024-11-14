---
layout: single        # 문서 형식
title: Support Vector Machine # 제목
categories: Machine Learning    # 카테고리
tag: [ML, Statistics, Classification]
toc: true             # 글 목차
toc_sticky : true
toc_label: 목차
author_profile: false # 홈페이지 프로필이 다른 페이지에도 뜨는지 여부
sidebar:              # 페이지 왼쪽에 카테고리 지정
    nav: "docs"       # sidebar의 주소 지정
#search: false # 블로그 내 검색 비활성화
use_math: true
---
# Keywords
Machine Learning, Statistics, Classification, Supervised Learning



# 1. Definition
Support Vector Machine 은 데이터가 선형적으로 분리 가능할 때, 각 클래스 간 최대 마진을 찾는 알고리즘이다. 여기서 마진(margin)은 두 클래스 간 결정 경계(Decision Boundary) 와 가장 가까운 데이터 간의 거리를 의미한다. 이 때 Support Vector 는 결정 경계에 가장 가까이 있는 데이터로, 결정 경계를 정의하는 가장 중요한 역할을 한다.



# 2. Hyperplane Searching
SVM 은 두 클래스 간 데이터를 선형으로 분리할 수 있는 초평면(hyperplane)을 찾아내는 것을 목표로 한다.

## 2.1. Linear Decision Boundary
#### - Notation

$\omega$ : 초평면의 법선 벡터
$x$ : 입력 데이터
$b$ : 절편(또는 편향)
$y_i$ : 데이터 $x_i$ 의 실제 레이블(1 또는 -1) 

#### - Hyperplane
초평면은 두 클래스 간 데이터를 분리하며, SVM의 목적은 초평면에 대해 마진이 최대가 되도록 설정하는 것이다. 이는 다음과 같이 정의된다.

$$
\omega \cdot x + b = 0
$$

#### - Margin

$$
Margin = \frac{2}{||\omega||}
$$

마진이 최대화되면 모형의 일반화 성능이 향상될 가능성이 커진다. 즉, SVM 의 문제는 $\omega$ 의 크기를 최소화해 마진을 최대로 하는 최적화 문제로 변환된다.

#### - Optimization
각 데이터 $x_i$ 가 올바르게 분류되어야 하는 제약조건 하에 최적화가 진행되어야 한다. 이는 다음과 같이 표현할 수 있다.

$$
y_i (\omega \cdot x_i + b) \ge 1 \quad \forall \: i \in \mathbb{N}
$$

따라서 SVM 의 최적화 문제는 다음과 같이 표현할 수 있다.

$$
\min \: \frac{1}{2}||\omega||^2 \quad subject \:\: to \quad y_i(\omega \cdot x_i + b) \ge 1 \quad \forall \: i \in \mathbb{N}
$$

이 문제는 이차 계획법(Quadratic Programming)을 사용해 풀 수 있다.



# 3. Slack Variables & Soft Marign
실제 데이터가 완벽하게 선형적으로 분리되지 않는 경우, SVM 은 이를 처리하기 위해 Slack Variables 을 도입해 일부 데이터가 결정 경계를 넘어가도록 허용하는 Soft Margin 방법을 사용한다. Slack Variables $\xi_i$ 를 도입한 최적화 문제를 다음과 같이 표현할 수 있다.

$$
\min \: \left[ \frac{1}{2} ||\omega||^2 + C \sum_{i=1}^n \xi_i \quad subject \: to \quad y_i(\omega \cdot x_i + b) \ge 1 - \xi_i \right], \quad \xi \ge 0 \:\: \& \:\: \forall \: i \in \mathbb{N}
$$

여기서 $C$ 는 규제 파라미터로 마진과 오차 간의 균형을 조정한다. 그리고 $\xi_i$ 는 데이터가 결정 경계로부터 벗어난 정도를 표현한다. 만약 $\xi_i = 0$ 이면 데이터가 마진을 완벽하게 만족하고, $\xi_i > 0$ 이면 데이터가 마진을 넘어 결정 경계에 가깝거나 넘어가 있다는 것을 의미한다.



# 4. Kernel Trick
Slack Variabels 와 Soft Margin 이외에도, 커널 트릭을 사용해 비선형 데이터를 선형적으로 분리가능한 고차원 공간으로 변환한다. 

1. Linear Kernel
 $K(x_i, x_j) = x_i \cdot x_j $
2. Polynomial Kernel 
$K(x_i, x_j) = (x_i \cdot x_j + 1)^d$
3. Gaussian Radial Basis Function Kernel 
$K(x_i, x_j) = \exp (-\gamma \: ||x_i - x_j ||^2 ) $
4. Sigmoid Kernel 
$K(x_i, x_j) = \tanh (\alpha x_i \cdot x_j + c) $

$K(x_i, x_j)$ 은 고차원 공간에서의 내적을 원래의 저차원 공간에서 계산하게 하며, 이로 인해 복잡한 비선형 데이터를 효과적으로 처리할 수 있다.



# 5. Advantages & Disadvantages of SVM
#### - Advantages
1. 커널 트릭을 이용한 비선형 데이터 처리 
2. 고차원 데이터 처리
3. 일반화 성능

#### - Disadvantages
1. 서포트 벡터만을 기준으로 학습하기 때문에, 데이터가 많을수록 속도 감소
2. 파라미터 선택이 복잡
3. 해석이 어려움


