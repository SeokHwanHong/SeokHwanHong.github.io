---
layout: single        # 문서 형식
title: Hierarchical Clustering # 제목
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
Hierarchical Clustering (계층적 군집 분석) 은 데이터를 계층적으로 나누거나 병합해 집단을 형성하는 비지도 학습 기반 알고리즘이다. 군집의 상호 관계를 트리 형태로 나타내는 덴드로그램을 그려 시각적으로 탐색도 가능하다.



# 2. Distance
두 군집 간 유사성을 측정하는 방법으로 거리함수를 사용한다. 주로 유클리디안 거리로 계산을 한다.

$$
\begin{split}   
    d(x_i, x_j) = \sqrt{\sum_{k=1}^n (x_{ik} - x_{jk})^2} 
\end{split}
$$

## 2.1. Distance(Dissimilarity) Measure between Clusters
군집 간 거리(비유사성) 를 측정하는 방법에 따라 덴드로그램의 결과가 달라지기 때문에 방법을 선택하는 것은 굉장히 중요하다.


**1. Single Linkage**

두 군집 간 가장 가까운 점들의 거리를 사용한다.

$$
\begin{split}
    d(C_i, C_j) = \min \{ d(x_a, x_b) \vert x_a \in C_i, x_b \in C_j  \}
\end{split}
$$


**2. Complete Linkage**

두 군집 간 가장 먼 점들의 거리를 사용한다.

$$
\begin{split}
    d(C_i, C_j) = \max \{ d(x_a, x_b) \vert x_a \in C_i, x_b \in C_j  \}
\end{split}
$$


**3. Average Linkage**

두 군집 내 모든 데이터 간 평균 거리를 사용한다.

$$
\begin{split}
    d(C_i, C_j) = \frac{1}{\vert C_i \vert \cdot \vert C_j \vert} \sum_{x_a \in C_i} \sum_{x_b \in C_j} d(x_a, x_b)
\end{split}
$$


# 3. Algorithm
## 3.1. Agglomerative
1. 데이터 n 개에 대해 군집 n 개로 설정. 즉, 각 데이터 별로 군집이 있다고 가정

2. 유사도 계산

3. 가장 가까운 군집으로 병합

4. 새롭게 형성된 군집 간 거리를 다시 계산해 갱신

5. 모든 군집이 하나로 병합될 때까지 3 ~ 4 를 반복

6. 덴드로그램 확인


## 3.2. Divisive
1. 모든 데이터가 동일한 군집에서 시작

2. 가장 이질적인 데이터를 분할해 두 개의 군집으로 분할

3. 각 군집을 반복적으로 분할

4. 더 이상 분할할 수 없을 때까지 분할 반복


# 4. Advantages & Disadvantages of Hierarchical Clustering
#### - Advantages
1. 덴드로그램을 통해 데이터 구조 및 군집화 과정을 시각적으로 이해 가능 

2. 다양한 거리 측도를 사용할 수 있어 여러 데이터 유형에 적용 가능

3. 군집 수를 사전에 설정할 필요가 없음

#### - Disadvantages
1. 시간 복잡도가 $$\mathcal{O}(n^2)$$ 또는 그 이상으로 연산량이 많음

2. 한 번 병합된 군집은 되돌릴 수 없어, 최종 결과에 영향을 미칠 수 있음

3. 노이즈 및 이상치에 민감



