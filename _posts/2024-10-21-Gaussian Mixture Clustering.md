---
layout: single        # 문서 형식
title: Gaussian Mixture Clustering # 제목
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
Gaussian Mixture Model (GMM) Clustering 은 데이터를 여러 개의 정규분포의 혼합으로 표현하는 군집화 알고리즘이다. GMM 은 K-means 처럼 데이터들을 군집화하지만, 선형적인 경계가 아닌 확률 분포를 사용해 각 군집이 특정 정규분포를 따른다고 가정한다. 




# 2. Gaussian Mixture Model
## 2.1. Notation

$x$ : 데이터

$d$ : 데이터의 차원

$\mu_k$ : k 번째 군집의 평균

$\Sigma_k$ : k 번째 군집의 공분산

$p(x)$ : 데이터 $x$ 의 확률밀도함수

$K$ : 군집의 수 (정규분포의 수)

$$\pi_k$$ : 각 정규분포의 가중치, $$\sum_{k=1}^K = 1$$


## 2.2. Gaussian Distribution
Gaussian Distribution 은 정규분포라고도 하며 다음과 같이 표현할 수 있다.

$$
\begin{split}
    \mathcal{N} (x \vert \mu_k, \Sigma_k) = (2 \pi)^{-\frac{d}{2}} \vert \Sigma_k \vert^{-\frac{1}{2}} \exp \left[ -\frac{1}{2} (x-\mu_k)^T \Sigma_k^{-1} (x-\mu_k) \right]
\end{split}
$$

## 2.3. Gaussian Mixture Model
GMM 은 여러 정규분포의 가중합(Mixture)으로 표현되며 각 정규분포에 대한 가중치 $\pi_k$ 도 학습한다. 이를 다음과 같이 표현할 수 있다.

$$
\begin{split}
    p(x) = \sum_{k=1}^K \pi_k \cdot \mathcal{N}(x \vert \mu_k, \Sigma_k)
\end{split}
$$




# 3. EM Algorithm
## 3.1. Definition
EM 알고리즘은 데이터가 불완전하거나 누락된 경우, 또는 잠재변수가 포함된 모형의 파라미터를 최대가능도추정(Maximum Likelihood Estimation, MLE) 으로 학습하기 위한 반복적 최적화 알고리즘이다. E-step 과 M-step 두 단계로 이루어져 있으며, 각 반복에서 파라미터를 점진적으로 최적화한다. 

## 3.2. Likelihood Function
**1. 불완전한 데이터의 로그 가능도 함수**

불완전한 데이터가 포함된 모형의 로그 가능도 함수는 다음과 같다.

$$
\begin{split}
    \log p(X \vert \theta) = \sum_{i=1}^n \log p(x_i \vert \theta)
\end{split}
$$ 

여기서 $$X$$ 는 관측 데이터, $$\theta$$ 는 모형의 파라미터, $$p(x_i \vert \theta)$$ 는 파라미터 $$\theta$$ 에 대한 데이터 $$x_i$$ 의 확률 밀도 함수이다.


**2. 완전한 데이터의 로그 가능도 함수**

데이터가 불완전하거나 잠재 변수가 포함된 경우, 직접적으로 로그 가능도 함수 $$\log p(X \vert \theta)$$ 를 최적화하는 것은 난해하다. 따라서 잠재 변수 $$Z$$ 를 도입해 완전한 데이터의 가능도를 계산 후 EM 알고리즘으로 최적화한다.

$$
\begin{split}
    \log p(X, Z \vert \theta) = \sum_{i=1}^n \log p(x_i, z_i \vert \theta)
\end{split}
$$

## 3.3. Algorithm
**1. 초기화**

파라미터 $\theta^{(i)}$ 를 임의로 설정한다. 초기값 설정에 따라 EM 알고리즘의 수렴 속도와 결과에 영향을 줄 수 있다.

**2. E-step (Expectation step)**

현재 파라미터 $$\theta^{(t)}$$ 에 대해 각 잠재 변수 $$Z$$ 의 기댓값을 계산한다. 즉, $Z$ 에 대한 책임도를 계산한다. 그리고 다음의 함수 Q에 대해 $$\theta^{(t)}$$ 로 완전한 데이터의 로그 가능도의 기댓값을 계산한다. 

$$
\begin{split}
    Q(\theta \vert \theta^{(t)}) = \mathbb{E}_{Z \vert X,\theta^{(t)}} [\log p(X,Z \vert \theta) ]
\end{split}
$$

**3. M-step (Maximization Step)**

2번에서 구한 기댓값을 기반으로 Q 함수를 최대화해 파라미터 $\theta$ 를 갱신한다. 이는 다음과 같이 표현된다.

$$
\begin{split}
    \theta^{(t+1)} = argmax_{\theta} Q(\theta \vert \theta^{(t)})
\end{split}
$$

이 단계에서 현재의 책임도(기댓값) 을 바탕으로 MLE를 계산해 파라미터를 갱신한다. 


**4. 반복**
2 ~ 3 을 반복해 로그 가능도 함수가 수렴하거나 파라미터의 변화가 매우 작아질 때까지 반복한다.




# 4. Algorithm
GMM Clustering 은 EM 알고리즘을 사용해 데이터에 가장 잘 맞는 정규분포의 매개변수 (평균, 공분산, 가중치) 를 학습한다.

**1. 초기화**

각 정규분포의 매개변수인 $\mu_k, \Sigma_k, \pi_k$ 의 초기값을 설정한다. 주로 K-means를 이용해 초기값을 설정한다.

**2. E-Step**

각 데이터 $x_i$ 가 각 정규분포에 속할 기댓값(책임도) $r_{ik}$ 를 계산한다. 이는 데이터 $x_i$ 가 k 번째 군집에 속할 확률을 나타낸다. 그리고 다음과 같이 정의된다.

$$
\begin{split}
    r_{ik} = \frac{\pi_k \cdot \mathcal{N} (x_i \vert \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \cdot \mathcal{N} (x_i \vert \mu_j, \Sigma_j)}
\end{split}
$$

3. M-step
책임도 $r_{ik}$ 를 기반으로 매개변수들을 갱신한다.

$$
\begin{split}
    \pi_k &= \frac{1}{n} \sum_{i=1}^n r_{ik} \\
    \mu_k &= \frac{\sum_{i=1}^n r_{ik} \cdot x_i}{\sum_{i=1}^n r_{ik}}   \\
    \Sigma_k &= \frac{\sum_{i=1}^n r_{ik}(x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^n r_{ik}} 
\end{split}
$$




# 5. Advantages & Disadvantages of Gaussian Mixture Clustering
#### - Advantages
1. 타원형 경계나 비대칭적인 군집도 처리 가능

2. 군집 경계의 불확실성 반영

3. 데이터의 분포가 여러 정규분포로 혼합되어있을 때 효과적


#### - Disadvantages
1. 복잡한 공분산 행렬 계산으로 인해 연산량이 많음

2. 군집의 수 K 를 사전에 지정

3. 초기값 선택에 따라 결과 변동성 존재

4. 이상치에 민감