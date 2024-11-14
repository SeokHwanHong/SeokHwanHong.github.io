---
layout: single        # 문서 형식
title: Transformation for Classification in Statistics # 제목
categories: Statistics    # 카테고리
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
Statistics, Classification, Machine Learning

# 1. Classification
#### - Notation
$k$ : class 수
$n$ : 데이터 수
$X = (X_1, X_2, ... , X_p) \in \mathbb{R}^p$ : Predictors
$Y \in y = \{1, 2, ... , k \} $ : Class Label (Categorical)
$\mathcal{D} = \{(x_i,y_i) | i = 1, 2, ... , n \}$ : 학습 데이터
$\phi : \mathbb{R}^p \rightarrow y$ : 결정 규칙
$\delta_k(x)$ : 판별식
$\pi_k(x) = \mathbb{P}(Y = k)$ : 전체에서 $k$ 번째 class 가 차지하는 비율


#### - Classification Rule
주어진 데이터 쌍 $\mathcal{D}$ 에 대해 최적의 결정 규칙 $\phi $ 를 계산하는 것을 목표로 한다.

$$
\phi(x) = \argmax_{x} \: \delta_k(x)
$$

#### - Optimal Classification

1. 0-1 손실함수
예측값 $\hat{y}$ 에 대해 올바르게 예측했다면 0을, 잘못 예측했다면 1을 부여하는 손실함수이다.

$$
\mathcal{L} (y, \phi(x)) = I(y \neq \phi(x)) = \begin{cases} 0 & \text{if} \quad \hat{y} = y \\ 
1 & \text{if} \quad \hat{y} \neq y \end{cases}, 
$$


2. 위험함수
결정 규칙 $\phi$ 의 오류율을 확인하는 함수이다.

$$
\begin{split}
R(\phi) &= \mathbb{E}_{X,Y} [\mathcal{L}(Y, \phi(x))] = \mathbb{E}_{X,Y}[I(y \neq \phi(x))] = \mathbb{P}(Y \neq \phi(x)) \\
&\rightarrow \mathbb{E}_{X,Y} [\mathcal{L}(Y, \phi(x)) | X = x] = \mathbb{P}(Y \neq \phi(x) | X = x) = 1 - \mathbb{P}[Y = \phi(x) | X = x]
\end{split}
$$

3. 베이지안 분류기
베이즈 결정 규칙 $\phi^*$ 와 오류율 $R^*$ 에 대해 다음과 같이 표현 가능하다.

$$
\phi^*(x) = \argmax_k \: \mathbb{P} (Y = k | X = x)
$$

이에 대해 사후확률 $P_k(x) = \mathbb{P} (Y = k | X = x)$ 와 $X | Y = k$ 의 확률밀도함수 $f_k(x)$ 를 이용해 다음과 같이 표현할 수 있다.

$$
P_k(x) = \mathbb{P} (Y = k | X = x) \propto \pi_k(x) \cdot f_k(x)
$$

또한, 최적화 결정 규칙 $\hat{\phi} = \argmax_k \hat{P}_k(x)$ 를 대입하면 다음과 같이 표현 가능하다.

$$
P_k(x) = \mathbb{P} (Y = k | X = x) = \frac{\mathbb{P}(Y = k, X = x)}{\mathbb{P}(X = x)} = \frac{\mathbb{P}(X = x | Y = k) \mathbb{P}(Y = k)}{\mathbb{P}(X = x)} 
$$


# 2. Lineaer Discriminant Analysis (LDA)
## 2.1. Definition
선형판별분석 (Linear Discriminant Analysis, LDA) 이란 분류모델과 차원축소까지 동시에 사용하는 알고리즘이다. 이는 입력 데이터 세트를 저차원 공간으로 투영해 차원을 축소하며 지도학습에서 사용한다. 또한, 클래스 간 분리를 최대화하는 축을 탐색하기 위해 클래스 간 분산과 클래스 내부 분산의 비율을 최대화 하는 방식으로 차원을 축소하며 축이 선형적인 특성을 띈다. 

## 2.2. Notation
1. $X | Y = k \sim \mathcal{N}(\mu_k, \Sigma_k)$
2. $\Sigma_1 = \Sigma_2 = ... = \Sigma_k$
3. $f_k(X) = (2 \pi)^{-\frac{p}{2}} |\Sigma|^{-\frac{1}{2}} \exp[-\frac{1}{2}(X-\mu_k)^{T} \Sigma^{-1} (X-\mu_k)] $
4. $p_k(x) = \frac{\pi_k f_k(x)}{\Sigma \pi_k f_k(x)}$

## 2.3. Decision Rule
$\Sigma \pi_k f_k(x)$ 는 상수이기 때문에 결정 규칙은 다음과 같이 표현할 수 있다.

$$
\begin{split}
\phi^*(x) &= \argmax_k \: p_k(x) \propto \argmax_k \: \pi_k f_k(x) \propto \argmax_k \: \log (\pi_k(x) f_k(x)) \\
&\rightarrow \log (\pi_k(x) f_k(x)) = \log \pi_{k} + \log f_k(x) \\
&\propto \log \pi_{k} - \frac{1}{2}(X-\mu_k)^T \Sigma^{-1} (X-\mu_k) \\
&\propto \log \pi_k -\frac{1}{2} (-2 \mu_k^T \Sigma^{-1}x + \mu_k^T \Sigma^{-1}\mu_k) \\
&= \log \pi_k + \mu_k^T \Sigma^{-1} x - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k = \delta_k(x)
\end{split}
$$

## 2.4. Binary Case
가정 : 각 class의 비율이 동일 ( $ \pi_1 = \pi_2 = \frac{1}{2}$ )
$\quad \rightarrow \: \delta_k(x) = (\Sigma^{-1} \mu_k)^T (x -\frac{1}{2} \mu_k)$

#### - Bayes Decision Boundary

$$
\begin{split}
&(\Sigma^{-1} (\mu_1 - \mu_2))^T x - \frac{1}{2} (\mu_1^T \Sigma^{-1}\mu_1 - \mu_2^T \Sigma^{-1}\mu_2) \\
&= (\Sigma^{-1}(\mu_1 - \mu_2))^T x - \frac{1}{2} ((\mu_1 - \mu_2)^T \Sigma^{-1} (\mu_1 - \mu_2) ) \\
&= (\Sigma^{-1} (\mu_1 - \mu_2))^T (x - \frac{1}{2}(\mu_1 + \mu_2)) = 0
\end{split}
$$

#### - Optimal Boundary
위 조건들을 이용해 다음과 같이 표현 가능하다.

$$
\begin{split}
&\delta_1 (x) = \delta_2(x) \\
&\rightarrow (\Sigma^{-1} \mu_1)^T (x - \frac{1}{2}\mu_1) = (\Sigma^{-1} \mu_2)^T (x - \frac{1}{2}\mu_2) \\
&\rightarrow ((\Sigma^{-1} \mu_1)^T - (\Sigma^{-1} \mu_2)^T)x = \frac{1}{2} (\Sigma^{-1} \mu_1)^T \mu_1 - \frac{1}{2} (\Sigma^{-1} \mu_2)^T \mu_2 \\
\end{split}
$$

#### - Plug-in Estimator

1. Class Proportion : $\pi_k = \mathbb{P} (Y = k) \rightarrow \hat{\pi}_k = \frac{n_k}{n}$
2. Class Mean : $\hat{\mu}_k = \frac{1}{n} \sum_{i=1}^{n_k} x_{ik} \in \mathbb{R}^p $
3. Common Covariance matrix : $\hat{\Sigma} = \frac{1}{n-k} \sum_{k=1}{K} \sum_{i=1}^{n_k} (x_{ik} - \hat{\mu}_k) (x_{ik} - \hat{\mu}_k)^T $

# 3. Quadratic Discriminant Analysis (QDA)
## 3.1. Definition
선형판별분석과 동일한 과정을 갖지만, 클래스별로 다른 공분산 구조를 갖는 판별 분석 알고리즘이다.

## 3.2. Notation

1. $X | Y = k \sim \mathcal{N}(\mu_k, \Sigma_k) $ 및 등분산성 가정
2. $\hat{\Sigma}_k = \frac{1}{n_k-1} \sum_{i=1}^{n_k}(x_{ik} - \hat{\mu}_k)(x_{ik} - \hat{\mu}_k)^T $
3. $\delta_k(x) = \log \pi_k - \frac{1}{2} \log |\pi_k| - \frac{1}{2} (x - \mu_k)^{T} \Sigma^{-1}_k (x - \mu_k)$
4. $f_k(x) = (2 \pi)^{-\frac{p}{2}} |\Sigma|^{-\frac{1}{2}} \exp[-\frac{1}{2}(x - \mu_k)^{T} \Sigma^{-1} (x - \mu_k)] $
5. Plug-in Estimators : $\hat{\pi}_k, \hat{\mu}_k, \hat{\Sigma}_k $

## 3.3. LDA vs QDA
1. $\pi_k = \frac{n_k}{n}$
2. $\hat{mu}_k = \frac{1}{n_k} \sum_{i=1}^{n_k} x_{ik} $
3. LDA 공분산

$$
\hat{\Sigma} = \frac{1}{n-k} \sum_{k=1}^K \sum_{i=1}^{n_k}(x_{ik} - \hat{\mu}_k) (x_{ik} - \hat{\mu}_k)^T
$$

4. QDA 공분산

$$
\hat{\Sigma}_k = \frac{1}{n_k-1} \sum_{i=1}^{n_k}(x_{ik} - \hat{\mu}_k) (x_{ik} - \hat{\mu}_k)^T
$$


# 참고
https://velog.io/@swan9405/LDA-Linear-Discriminant-Analysis