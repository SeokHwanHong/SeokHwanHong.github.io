---
layout: single        # 문서 형식
title: Spectral Clustering # 제목
categories: Machine Learning    # 카테고리
tag: [ML, Statistics, Clustering]
author_profile: false # 홈페이지 프로필이 다른 페이지에도 뜨는지 여부
sidebar:              # 페이지 왼쪽에 카테고리 지정
    nav: "docs"       # sidebar의 주소 지정
#search: false # 블로그 내 검색 비활성화
use_math: true
---
# Keywords
Machine Learning, Statistics, Clustering, Unsupervised Learning



# 1. Definition
Spectral Clustering 은 그래프 이론을 기반으로 군집을 형성하는 알고리즘이다. 이는 데이터 간 유사도를 바탕으로 그래프를 구성 후 Laplacian matrix 의 고유벡터를 활용해 저차원 공간으로 변환해 군집을 수행한다. 고차원 데이터나 비선형적 구조의 데이터에서 성능이 뛰어나다는 특징이 있다.


# 2. Graph Strcuture
유사도 행렬과 그래프 이론을 사용해 군집을 수행한다.

## 2.1. Similarity Matrix $W$
Similarity Matrix (유사도 행렬) $W$ 는 데이터 간 유사성을 나타내는 대칭 행렬로, 각 요소 $W_{ij}$ 는 데이터 $x_i$ 와 $x_j$ 간 유사도를 표현한다. 이는 다음과 같이 정의한다.

$$
W_{ij} = \begin{cases} \exp (-\frac{||x_i - x_j||^2}{2 \sigma^2}) &  \text{if} \quad ||x_i - x_j|| < r, \\ 
 0 & \text{else} \end{cases}
$$

여기서 $\sigma$ 는 유사도를 조절하는 스케일 파라미터, $r$ 은 특정 거리 임계값이다.

## 2.2. Graph Laplacian $L$
유사도 행렬 $W$ 를 기반으로 그래프 라플라시안 $L$ 을 게산한다. 이는 그래프의 구조를 나타내는 중요한 역할을 하며, 다음과 같이 정의한다.

$$
L = D - W
$$

여기서 $D$ 는 대각행렬로, 각 대각 원소 $D_{ii}$ 는 각 데이터 $x_i$ 와 연결된 모든 유사도의 합이다.
이에 대해 정규화된 그래프 라플라시안은 다음과 같이 정의한다.

$$
\begin{split}
L_{sym} &= I - D^{-\frac{1}{2}} W D^{-\frac{1}{2}} \\
L_{rw} &= D^{-1} W 
\end{split}
$$

# 3. Algorithm
1. $W$ 계산
데이터들 간 유사도를 계산해 $n \times n$ 크기의 유사도 행렬 $W$ 를 생성한다. 주로 가우시간 커널이나 KNN 으로 유사도를 계산한다.

2. $L$ 계산
$W$ 를 바탕으로 $L$ 을 계산한다. 이는 그래프의 구조 정보를 요약한다.

3. 고유벡터 계산
고유값 분해나 SVD 를 $L$ 에 적용해 고유벡터를 계산한다. 이 때, 가장 작은 고유벡터 K 개를 사용한다.

4. 고유벡터로 새로운 저차원 표현
고유벡터를 새로운 특징 공간에서 데이터의 좌표로 사용해 데이터를 저차원 공간으로 투영한다.

5. K-means 군집화
저차원 공간에서 K-means 와 같은 군집화 알고리즘을 수행한다.


# 4. Advantages & Disadvantages of Spectral Clustering
#### - Advantages
1. 복잡한 군집 구조(비선형 등) 처리 가능
2. 저차원 공간에서 군집화를 수행하기 때문에, 데이터 차원의 저주 문제 완화
3. 커널 방식과 그래프 구조 활용덕분에 다양한 유형의 데이터에 적용 가능

#### - Disadvantages
1. 유사도 행렬 및 라플라시안 고유벡터 계산 비용이 많음
2. 군집 수 설정 필요

