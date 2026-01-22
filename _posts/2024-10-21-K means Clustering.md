---
layout: single        # 문서 형식
title: K-means Clustering # 제목
categories: Machine Learning    # 카테고리
tag: [ML, Statistics, Clustering]
author_profile: false # 홈페이지 프로필이 다른 페이지에도 뜨는지 여부
sidebar:              # 페이지 왼쪽에 카테고리 지정
    nav: "counts"       # sidebar의 주소 지정
#search: false # 블로그 내 검색 비활성화
use_math: true
---
# Keywords
Machine Learning, Statistics, Clustering, Unsupervised Learning



# 1. Definition
K-means Clustering 은 비지도 학습 기반의 군집화 알고리즘으로 주어진 데이터셋을 사전에 지정한 K 개의 군집으로 분할한다. 각 군집은 중심정을 기준으로 형성되며 데이터는 가장 가까운 중심점에 할당된다. 



# 2. Optimization
## 2.1. Notation
$n$ : 데이터의 수

$K$ : 군집의 수

$x_i$ : i 번째 데이터

$\mu_k$ : k 번째 군집의 중심점(centroid)

$$r_{ik} \in \{0, 1\}$$ : $$x_i$$ 가 k 번째 군집에 속하는지 여부를 나타내는 변수. $$r_{ik} = 1$$ 이면 $$x_i$$ 가 k 번째 군집에 속하는 것을 의미.

## 2.2. Optimization
K-means Clustering 은 군집 내 거리 제곱합을 최소화하는 최적화 문제로 간주할 수 있다. 즉, 군집 내 데이터와 중심점 간의 거리 제곱합을 최소화하는 것이 목표다.

$$
\begin{split}
    \min \sum_{i=1}^n \sum_{k=1}^K r_{ik} \Vert x_i - \mu_k \Vert^2
\end{split}
$$

# 3. Algorithm
**1. 초기화**

군집 중심점 K 개를 임의로 선택한다. 무작위나 특정 알고리즘으로 선택할 수 있다.


**2. 할당**

각 데이터 $x_i$ 을 유클리드 거리를 바탕으로 가장 가까운 중심점에 할당한다. 

$$
\begin{split}
    r_{ik} = \begin{cases} 1 & \text{if} \quad k = argmin_j \Vert x_i - \mu_j \Vert^2 \\ 
    0 & \text{otherwise} \end{cases} 
\end{split}
$$


**3. 중심점 갱신**

각 군집에 할당된 데이터들의 평균을 계산해 중심점을 갱신한다. 새로운 중심점은 군집 내 모든 데이터의 평균으로 계산한다.

$$
\begin{split}
    \mu_k = \frac{\sum_{i=1}^n r_{ik} x_i}{\sum_{i=1}^n r_{ik}} 
\end{split}
$$


**4. 반복**

2 ~ 3 을 반복해 중심점이 더 이상 변하지 않거나 지정된 최대 반복 횟수에 도달할 때까지 진행한다.


# 4. Advantages & Disadvantages of K-means Clustering
#### - Advantages
1. 이해가 직관적이고 구현이 간단하며 대규모 데이터 셋에서도 빠르게 작동

2. 시간 복잡도가 $$\mathcal{O} (n \cdot K \cdot t)$$ 로 각 요소에 대해 선형적으로 비례하기때문에 빠르게 처리 가능

3. 각 데이터가 하나의 군집에 명확하게 속하기 때문에 결과를 직관적으로 확인 가능


#### - Disadvantages
1. 사전에 적절한 K 값 지정 필요

2. 비선형 구조 처리 한계

3. 초기 중심점 선택에 따라 결과 변동성 존재

4. 노이즈와 이상치에 민감