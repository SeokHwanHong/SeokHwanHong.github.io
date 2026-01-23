---
layout: single        # 문서 형식
title: A Principal Component Analysis Approach to Noise Removal for Speech Denosing (2018) # 제목
categories: Electronics    # 카테고리
tag: [CV, Mathematics, Electronics, Statistics]
author_profile: false # 홈페이지 프로필이 다른 페이지에도 뜨는지 여부
sidebar:              # 페이지 왼쪽에 카테고리 지정
    nav: "counts"       # sidebar의 주소 지정
#search: false # 블로그 내 검색 비활성화
use_math: true
---
# Keywords
PCA, Dynamic Embedding, Noise Removal



# 1. Principal Component Analysis
## 1.1. Dynamical Embedding Technology
#### - Data : 1-dimensional speech signal
1차원 음성 신호란 시간에 따라 변화하는 음향 신호이다. 시간 t 를 기준으로 진폭을 나타내는 신호로 표현된다. 보통의 경우 PCA 를 이용해 데이터의 특징을 추출하는 것이 어렵다. 

#### - Dynamical Embedding
1차원 음성 신호를 정제하고자 Dynamical Embedding 라는 방법을 사용하는데, 이는 1 차원 데이터를 다차원으로 변환하거나 원본 데이터의 정보를 유지하면서 저차원의 데이터를 고차원으로 변환하는 것이다. 일반적으로 음성 신호의 분포는 시간에 의해 결정되기 때문에 데이터를 1차원에서 다차원으로 시간에 따라 재구성해야한다. 즉, 시간에 따라 결정되는 embedded dynamic matrix $V_{\tau}(t)$ 는 다음과 같이 구성된다.

$$
\begin{split}
    V_{\tau}(t) = \left[ \begin{matrix} \nu_t & \nu_{t+\tau} & \cdots & \nu_{t+R \tau} \\ 
    \nu_{t+\tau} & \nu_{t+2\tau} & \cdots & \nu_{t+(R+1)\tau} \\  
    \vdots & \vdots & \ddots & \vdots \\
    \nu_{t+(g-1)\tau} & \nu_{t+g\tau} & \cdots & \nu_{t+(g+R+1)\tau} 
    \end{matrix} \right]
\end{split}
$$

여기서 $\tau$ 는 시간 증가량, $g$ 는 임베딩 차원, $R > g$ 는 delay vector 이다. $R$ 의 크기에 따라 처리 가능한 데이터의 양이 결정된다. 실제로 $g$ 와$\tau$ 는 데이터 내 가장 주파수가 낮은 부분에 기반해 설정되고 다음과 같이 표현 가능하다.

$$
\begin{split}
    g \ge \frac{f_0}{f_c}, \quad \tau = 1
\end{split}
$$

여기서 $f_0$ 는 데이터의 샘플링 주파수, $f_c$ 는 측정한 신호의 가장 낮은 주파수를 나타낸다.



## 1.2. Principal Component Analysis
#### - Definition
주성분 분석(Principal Component Analysis, PCA) 는 통계학에서 데이터 집합을 단순화하기 위해 회전행렬을 사용하는 비지도 학습 기법이다. 데이터 간 선형 결합을 이용해 새로운 좌표축인 주성분을 설정하는데, 이는 원래 변수들과 관련이 없으며 각 주성분은 데이터에 대해 분산이 최대가 되도록 구성된다.


#### - Notation
$$\nu_i (t_j) \text{ for } i = 1, 2, ... , n, \; j,k = 1,2, \dots ,m $$ : 데이터 집합

$t_j$ : 샘플링 시간

$$z_{jk} = \sum_{i=1}^{n} \nu_i (t_j) \nu_i(t_k) $$ : 자기상관 행렬 $$Z$$ 의 (j,k) 번째 원소

$$Z = Q \Lambda Q^{T}$$ : 자기상관 행렬에 대한 고유값 분해

$Q$ : 주성분의 방향을 나타내는 고유벡터 행렬

$$Q^{T} : Q$$ 에 대한 회전 행렬(rotation matrix)

$$\Lambda$$ : 내림차순으로 정렬된 고유값 대각행렬

#### - Principal Component
주성분은 데이터를 회전행렬 $Q^T$ 에 대해 선형 매핑 변환을 적용함으로써 얻을 수 있다. 데이터의 분산이 가장 큰 방향이 첫 번째 좌표축으로 설정하며 이를 첫 번째 주성분이라고 한다. 첫 번째 주성분 다음으로 분산이 큰 좌표축을 두 번째 주성분으로 명명하며, 이러한 방식으로 주성분을 순차적으로 정렬한다. 이를 수식으로 표현하면 다음과 같다.

$$
\begin{split}
    &P = Q^T V \\
    &p_{ki} = \sum_{j=1}^m q_{jk} \nu_i(t_k) \quad \text{for } i=1,2,...,n; \; j,k = 1,2,...,m 
\end{split}
$$

여기서 $P$ 는 주성분 행렬이고 $p_{ki}$ 는 i 번째 데이터의 k 번째 주성분이다. 

#### - Reconstruction
데이터 $V$ 는 $P$ 와 $Q$ 를 이용해 다음과 같이 복원할 수 있다.

$$
\begin{split}
    &V = QP \\
    &\nu_i (t_j) = \sum_{j=1}^m q_{jk} p_{ki} \quad \text{for } i=1,2,...,n; \; j,k = 1,2,...,m 
\end{split}
$$

또한 이 과정에서 저차수의 주성분은 데이터의 유용성을 나타내고, 반대로 고차수의 주성분은 데이터의 노이즈를 나타낸다. 따라서, 노이즈를 제거하기 위해 일부의 저차수 주성분만을 이용해 원본 데이터로 복원한다. 이는 다음과 같이 표현할 수 있다.

$$
\begin{split}
    \hat{\nu_i} (t_j) = \sum_{k=1}^L q_{jk} p_{ki} \quad \text{for } i=1,2,...,n; \; j,k = 1,2,...,L 
\end{split}
$$

여기서 $\hat{\nu_i} (t_j)$ 는 저차수의 주성분들로만 복원한 데이터이다. 



#### - Evaluation Metric
**1. Mean Square Relative Error**

원본 데이터와 재구성된 데이터 간 차이를 정량적으로 평가하기 위해 평균 제곱 상대 오차를 계산할 수 있다. 이는 다음과 표현할 수 있다. 

$$
\begin{split}
    K = \sqrt{\sum_{j=1}^m \left(\frac{\nu_0(t_j) - \hat{\nu}(t_j)}{\nu_0(t_j)} \right) / m }
\end{split}
$$

이를 통해 데이터가 얼마나 정확하게 재구성되어있는지, 노이즈가 잘 제거되었는지 등을 확인할 수 있다.


**2. Signal Noise Ratio**

신호 대비 잡음 비율을 확인해 재구성된 데이터의 정확도를 확인할 수 있다. 이는 다음과 같다.

$$
\begin{split}
    SNR = 10 \cdot \log \left(  \frac{\sum_n s^2(n)/n}{\sum_n [s(n) - \hat{s}(n)] / n} \right)
\end{split}
$$

여기서 $$s(n)$$ 은 원본 데이터, $$\hat{s}(n)$$ 은 노이즈를 제거한 데이터이다. 따라서 $$\sum_n s^2(n)/n$$ 은 음성 데이터의 정도를, $$\sum_n [s(n) - \hat{s}(n)] / n$$ 은 잡음 데이터의 정도를 나타낸다.