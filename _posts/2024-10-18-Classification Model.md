---
layout: single        # 문서 형식
title: Classification Models in Statistics # 제목
categories: Machine Learning    # 카테고리
tag: [ML, Statistics, Classification]
author_profile: false # 홈페이지 프로필이 다른 페이지에도 뜨는지 여부
sidebar:              # 페이지 왼쪽에 카테고리 지정
    nav: "docs"       # sidebar의 주소 지정
#search: false # 블로그 내 검색 비활성화
use_math: true
---
# Keywords
Machine Learning, Statistics, Classification, Bagging, Boosting

# 1. Decision Tree
## 1.1. Definition
의사결정나무(Decision Tree)란 일련의 분류 규칙을 통해 데이터를 분류 및 회귀하는 지도 학습 모델 중 하나이다. 결과 모델이 Tree 구조를 갖기 때문에 Decision Tree 라고 한다. 

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/classification model/figure8.3.jpg" height = 400>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 1 : Decision Tree Example ]</figcaption>
</figure>

figure 1에서 확인할 수 있듯이, 한 영역을 2개로 분리하는데 이를 반복적으로 수행하는 알고리즘이다. 

## 1.2. Regression Tree 
#### - Notation 
$X_j$ : a splitting variable 
$s$ : split pount
$R_1(j,s) = \{ x | x_j \le s \} $ : $X_k \le s$ 인 점들의 영역
$R_2(j,s) = \{ x | x_j \ge s \} $ : $X_k > s$ 인 점들의 영역 
$C_1 = \frac{1}{n_1} \sum_{i : x_i \in \mathbb{R}_1(j,s)} $ : $R_1$ 에 속하는 점들에 대한 label (y) 값
$C_2 = \frac{1}{n_1} \sum_{i : x_i \in \mathbb{R}_2(j,s)} $ : $R_2$ 에 속하는 점들에 대한 label (y) 값

#### - Goals
다음과 같이 정의되는 RSS 를 최소화하는 것이 목표이다.

$$
RSS = \sum_{m=1}^M \sum_{i \in \mathbb{R_m}} (y_i - C_m)^2
$$

즉, RSS를 최소화하는 것은 최적의 분리점 $s$ 를 찾는 것과 동일하다.


## 1.3. Classification Tree
#### - Notation 

$x_i \in \mathbb{R}_m$ : 데이터
$\hat{c}_m$ : 어느 한 노드에 있는 데이터의 가장 주된 클래스
$K$ : 클래스의 수
$\hat{p}_{mk} = \frac{1}{n_m} \sum_{x_i \in \mathbb{R}_m} I(y_i = k) $ : m 번째 노드에서 클래스 'k' 에 속하는 관측치의 비율

#### - Objective Functions
1. Misclassification Error
클래스가 잘못 분류된 관측치들의 수에 대한 목적함수이다.

$$
\begin{split}
E &= \frac{1}{n_m} \sum_{x_i \in \mathbb{R}_m} I(y_i \neq \hat{C}_m) \\
&= 1 - \hat{P}_m \hat{C}_m = 1- \max_k (\hat{P}_{mk})
\end{split}
$$

2. Gini Index
불순도를 측정하는 지표로 데이터의 통계적 분산 정도를 정량화해 표현한 값이다.

$$
G = \sum_{k=1}^K \hat{P}_{mk} (1 - \hat{P}_{mk})
$$

3. Cross-Entropy or Deviance
Cross Entropy 역시 Gini Index와 마찬가지로 불순도를 측정하는 지표로 정보량의 기댓값을 표현한 값이다.

$$
D = -\sum_{k=1}^K \hat{P}_{mk} \log \hat{P}_{mk}
$$

## 1.4. Advantages & Disadvantages of Tree
#### - Advantages
1. 트리 구조가 시각적으로 직관적이어서 해석이 용이
2. 데이터를 비선형적으로 분할해 비선형적 관계 처리가 용이
3. 전처리가 크게 필요하지 않음
4. 다양한 문제에 적용 가능
5. 앙상블 기법에 유리

#### - Disadvantages
1. 한 번에 변수 한 개만 최적화 가능
2. 과적합 가능성이 존재
3. 모형 전체에 대한 특징(global structure)을 잡아내기 어려움 
4. 모형 자체에 불안정성으로 인해 분산이 높음
5. 결과가 robust 하지않음

## 1.5. Applications
1. 범주형 데이터에 대해 분류 문제를 해결하기 위해 사용
2. 해석이 중요한 경우 사용
3. 변수 간 상호작용이 있는 경우 robust한 특성을 살려 사용




# 2. Bagging
## 2.1. Definition
Bagging 이란 Bootstrap Aggregating 의 약자로 부트스트랩 샘플에서 생성된 여러 버전의 예측 규칙을 종합하는 메타 알고리즘이다. 


## 2.2. Notation
$n$ : 표본 수
$B$ : 부트스트랩 표본 수
$Z = \{(x_1, y_1), (x_2, y_2), ... ,(x_n, y_n)  \} $ : 주어진 Data
$P_n = \sum_{i=1}^n \delta(x_i, y_i) $ : $Z$ 에 대한 경험적 분포 (empirical distribution)
$Z^*$ : 주어진 Data 로 생성한 크기가 $n$ 인 부트스트랩 표본
$\mathcal{D}$ : 확률분포 $P$ 에서 추출한 크기가 $n$ 인 무작위 훈련 표본

## 2.3. Working in Regression
#### - Goals
부트스트랩 표본 $Z^*$ 에 대한 적합된 예측 규칙 $\hat{f}^{*b}$ 들을 $\hat{f}_{bag}$ 로 결합한다.

$$
\hat{f}_{bag}(x) = \frac{1}{B} \sum_{b=1}^B \hat{f}^{*b} (x)
$$

#### - Ideal Aggregated Prediction Rule
가장 이상적인 예측 규칙에 대한 결합은 다음과 같다.

$$
f_{ag}^* (x, P) = \mathbb{E}(\hat{f}(x,\mathcal{D}))
$$

이에 평균 예측 오차는 각각 다음과 같다.

$$
\begin{split}
R(\hat{f}) = \mathbb{E}_{X,Y} [\mathbb{E}_\mathcal{D} (Y - \hat{f}(X, \mathcal{D}))^2] \\
R(\hat{f}_{ag}^*) = \mathbb{E}_{X,Y} [ (Y - \hat{f}_{ag}^*(X, P))^2] \\
\end{split}
$$

이 때 $P_n$ 이 원본 훈련 데이터의 경험적 분포일 때 $\hat{f}_{bag}(x)$ 는 $\hat{f}_{ag}^*(x, P)$ 가 아닌 $\hat{f}_{ag}^*(x, P_n)$ 이다.


## 2.4. Working in Classification
#### - Goals
주어진 데이터 $x$ 에 대해 클래스 $k$ 를 예측하는 예측 규칙 $\hat{\phi}^{*b}$ 의 비율 $\hat{p}_k(x)$ 들 중 이를 최대로 하는 규칙을 탐색한다.

$$
\hat{\phi}_{bag}(x) = \argmax_{k = 1, ... , K} \: \hat{p}_k (x)
$$

#### - Assumptions
이진 분류 문제의 분류기 $\hat{\phi}^{*b}(x)$ 들이 독립이고 베이지안 결정 규칙이 class 1로 예측하는 주어진 데이터 $x$ 에 대해 각각의 오차 확률은 0.5 보다 작다.

#### - Bagging Effect for Classification
클래스 1로 분류할 확률 $S_1$ 은 다음과 같다.

$$
S_1(x) = \sum_{b=1}^B I(\hat{\phi}^{*b}(x) = 1) 
$$ 

$S_1(x)$ 는 $n=B$ 이고 $p = 1 - err$ 인 이항분포이고, $B$ 가 커지면 $\mathbb{P} = (S_1(x) > B / 2) \rightarrow 1$ 이다.


## 2.5. Advantages & Disadvantages of Bagging
#### - Advantages
1. 노이즈에 대해 robust
2. 고차원 데이터에 적합
3. 병렬 처리 가능

#### - Disadvantages
1. 복잡성이 증가
2. 모델 해석이 난해




# 3. Random Forest
## 3.1. Definition
랜덤 포레스트란 서로 연관이 없는 의사 결정 나무들을 학습해 그 결과를 결합해 최종적으로 예측하는 모형이다. 

## 3.2. Random Forest Algorithm
1. 부트스트랩 표본 $\mathbf{Z}^{*b} $ 을 B개 생성한다.

2. 최소 노드 크기가 될 때까지 다음의 단계를 반복함으로써 $\mathbf{Z}^{*b}$ 으로 의사 결정 나무 $T_b$ 를 학습한다. 
2.1. 변수 $p$ 개를 선택해 그 중 변수 $m$ 개를 선택한다. ($m < p$ 로 가정, $m = p$ 인 경우 bagging)
2.2. 변수 $m$ 개 중 최적의 변수와 분할점을 선택한다.
2.3. 노드를 2개로 분리한다.

3. $\{ T_n \}_{b=1}^B$ 들로 투표를 진행해 결과를 예측한다.

## 3.3. Advantages & Disadvantages of Random Forest
#### - Advantages
1. 과적합 방지
2. 높은 예측 성능
3. 노이즈에 대해 robust
4. 병렬처리 가능

#### - Disadvantages
1. 모델이 복잡
2. 시간 및 메모리 사용량이 많이 필요
3. 특성 중요도 계산이 복잡 및 불안정



# 4. AdaBoost
## 4.1. Definition
AdaBoost (Adaptive Boosting) 는 약한 학습기들을 결합해 강한 학습기로 만드는 부스팅 기법 중 하나이다. 반복적으로 모델을 학습해 이전 단계에서 잘못 예측된 데이터에 가중치를 높여 후속 모델이 이를 더 잘 학습하도록 유도한다. 이를 통해 점진적으로 성능을 향상시키고, 최종적으로 더 강력한 분류 또는 회귀 모형을 구성하는 방식이다.

## 4.2. Notation
$x_i$ : 입력 데이터
$y_i$ : 입력 데이터 $x_i$ 의 정답 레이블
$\mathcal{D} = \{ (x_1, y_1), (x_2, y_2), ... ,(x_n, y_n) \}$ : 주어진 학습 데이터
$W_1(i)$ : $x_i$ 에 대한 가중치, 초기값 $= \frac{1}{n}$
$h(\cdot)$ : 약한 학습기
$t$ : 약한 학습기의 가중치를 학습시키는 주기

## 4.3. Weak Learner
약한 학습기란 단독으로는 성능이 뛰어나지 않지만, 부스팅과 같은 앙상블 기법을 통해 결합되면 강력한 성능을 발휘할 수 있는 모형을 의미한다. 특정 작업에서 무작위 추측보다는 더 나은 성능을 내는 예측 모형이다. 주로 단순한 모형들을 사용해 예측 성능과 복잡도 모두 낮고 학습 속도가 빠르다는 특징이 있다. 이는 복잡한 패턴이 아닌 간단한 규칙에 따라 데이터를 분류하거나 예측한다. 

#### - Training Algorithm
1. 약한 학습기 학습 
각 반복 $t$ 번째마다 약한 학습기 $h(t)$ 를 학습 후, 데이터 포인트 $x_i$ 에 대한 예측값 출력한다. 이 때 $h(t)$ 는 각 데이터 표본에 대한 가중치 $W_i(t)$ 를 기반으로 학습한다.

2. 오류율 계산
다음과 같이 정의되는 오류율 $\mathbf{\epsilon}_t$ 를 각 반복마다 계산한다.

$$
\mathbf{\epsilon}_t = \sum_{i=1}^{n} W_t(i) \cdot I(h_t(x) \neq y_i)
$$

즉, 예측 클래스와 실제 클래스가 맞지 않은 경우에 대한 식이다.

3. 가중치 계산
각 약한 학습기의 중요도를 나타내는 가중치 $\alpha_t$ 를 다음과 같이 정의한다.

$$
\alpha_t = \frac{1}{2} \ln \frac{1 - \mathbf{\epsilon}_t}{\mathbf{\epsilon}_t}
$$

오류율 $\epsilon_t$ 에 따라 $\alpha_t$ 를 계산한다. 오류율이 낮은 학습기는 더 높은 가중치를 부여받고, 반대로 오류율이 높으면 더 낮은 가중치를 부여받는다.
 
4. 가중치 갱신
다음 반복에서 사용할 데이터 표본의 가중치 $W_{t+1} (i)$ 를 다음과 같이 갱신한다.

$$
W_{t+1} (i) = \frac{D_t(i) \cdot \exp(-\alpha_t y_i h_t(x_i))}{Z_t}
$$

여기서 $Z_t = \sum_{i=1}^n W_t(i) \cdot \exp(-\alpha y_i h_t(x_i)) $ 는 가중치 정규화 상수이다. $y_i \cdot h_t(x_i)$ 는 데이터 포인트 $x_i$ 에 대한 예측의 정확한지 여부를 나타낸다. 정확히 예측하면 $y_i \cdot h_t(x_i) = 1$, 잘못 예측하면 $y_i \cdot h_t(x_i) = -1$ 이다.

5. 최종 예측
각 반복에서 학습한 약한 학습기들의 가중합을 사용해 최종 예측을 만든다. 각 학습기의 가중치 $\alpha_t$ 에 따라 예측값에 영향을 미친다. 최종 예측은 다음과 같이 정의한다.

$$
H(x) = sign(\sum_{t=1}^T \alpha_t \cdot h_t(x)) 
$$

각 약한 학습기의 예측에 가중치를 부여한 결과를 모두 더한 값에 대해 부호를 기준으로 예측을 결정한다.

## 4.3. Advantages & Disadvantages of AdaBoost
#### - Advantages
1. 과적합 방지
2. 높은 예측 성능
3. 해석이 용이
4. 다양한 모형에 적용 가능

#### - Disadvantages
1. 이상치에 민감
2. 시간 및 메모리 사용량이 많이 필요
3. 선형 분리 가능성에 의존



# 5. Gradient Boosting
## 5.1. Definition
Gradient Boosting 은 주로 결정 트리와 같은 약한 학습기를 여러개 결합해 강한 예측 모형을 만드는 부스팅 기법 중 하나이다. 어차를 점진적으로 줄이기 위해 각 반복 단계에서 새로운 모형을 추가해 주어진 데이터에 대해 학습하는 과정을 계속해 개선하는 방식으로 작동한다.



## 5.2. Notation
$x_i$ : 입력 값
$y_i$ : $x_i$ 에 해당하는 실제 값
$F(x_i)$ : $x_i$ 에 대한 모형 $F$ 의 예측 값



## 5.3. Loss Function
문제의 종류에 따라 다른 손실함수를 사용하지만, 일반적인 표현은 다음과 같다.

$$
L(y_i, F(x_i)) = \frac{1}{2} (y_i - F(x_i))^2
$$

회귀 문제에서는 주로 MSE 를, 분류 문제에서는 Cross Entropy 등을 사용한다.



## 5.4. Training Algorithm
1. 초기 모형 설정
초기 모형 $F_0(x)$ 를 종속 변수의 평균 등과 같이 매우 간단한 값으로 설정한다.

2. 잔차 계산
각 반복 단계에서 잔차를 계산한다. 이를 바탕으로 새로운 약한 학습기는 이 잔차를 예측한다. $m$ 번째 모형 $F_m(x)$ 에 대한 잔차는 다음과 같다.

$$
r_i^{(m)} = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}
$$

3. 약한 학습기 학습
새로운 약한 학습기 $h_m(x)$ 는 이전 모델이 예측하지 못한 잔차를 예측하도록 학습된다. 이 학습기는 잔차의 기울기(gradient)를 학습해 오차를 줄이는 방향으로 갱신된다.

$$
h_m(x) = \argmin_h \sum_{i=1}^n (r_i^{(m)} - h(x_i))^2
$$

여기서 $r_i^{(m)}$ 는 이전 모델이 잘못 예측한 부분을 포함하며, 이 부분을 다음 학습시가 보정하는 방식이다.

4. 모형 갱신
새롭게 학습된 약한 학습기 $h_m(x)$ 는 기존 모형에 추가되어 모형이 점점 더 정확한 예측을 할 수 있도록 갱신된다. 이는 다음과 같이 표현할 수 있다.

$$
F_{m+1}(x) = F_m(x) + \nu \cdot h_m(x)
$$

여기서 $\nu$ 는 학습률로 새로운 학습기의 기여도를 조정하는 파라미터이다.

5. 반복
1~4 과정을 반복해 모델을 점진적으로 개선한다. 각 학습기가 예측하지 못한 부분을 다음 학습기가 보정하며 여러 단계에 걸쳐 점차적으로 성능을 향상시킨다.



## 5.5. Advantages & Disadvantages of Gradient Boosting
#### - Advantages
1. 유연성
2. 높은 예측 성능
3. 편향과 분산 절충

#### - Disadvantages
1. 병렬처리가 아닌 이전 모델의 잔차를 기반으로 학습을 진행하기 때문에 많은 시간과 자원 소모
2. 복잡한 모형을 만들 수 있기 때문에 과적합 가능성 존재
3. 해석이 난해




# 6. XGBoost
## 6.1. Definition
XGBoost (Extreme Gradient Boosting) 는 Gradient Boosting 에 추가적인 최적화와 기능을 제공해 효율성과 성능을 극대화한 알고리즘이다.

## 6.2. Loss Function
#### - Notation
$T$ : 트리 내 리프 노드의 수
$\omega_j$ : 트리 리프의 가중치
$\gamma$ : 리프 노드에 대한 패널티
$\lambda$ : L2 정규화 항이다. 
$f_k$ : 학습기
$h_m(x)$ : 약한 학습기
$F_m(x)$ : 학습에 사용하는 모형
$F(x)$ : 최종 예측 모형
$\eta$ : 학습률

#### - Objective Function
모형 복잡도를 제어하기 위해 각 학습기 $f_k$ 에 대해 정규화 항을 도입한다.

$$
\Omega (f) = \gamma T + \frac{1}{2} \lambda \sum_j \omega_j^2
$$

이 정규화 항을 이용해 목적 함수를 다음과 같이 표현할 수 있다.

$$
Objective = \sum_{i=1}^n L(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)
$$

여기서 $L(y_i, \hat{y}_i)$ 는 손실 함수를 의미한다.


## 6.3. Taylor Series
XGBoost 는 일반적인 Gradient Boosting 과는 다르게, 2차 테일러 근사를 사용해 손실 함수를 근사해 최적화를 한다. 이를 통해 더 정확하게 손실을 최소화할 수 있으며 갱신 과정에서 좀 더 정교하게 조정이 가능하다. 이는 다음과 같이 표현할 수 있다.

$$
L(y_i, \hat{y}_i) \approx L(y_i, \hat{y}_i^{(t)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)
$$

여기서 Notation 은 다음과 같다.
$ g_i = \frac{\partial L(y_i, \hat{y}_i) }{\partial \hat{y}_i}$ : 손실 함수의 1차 기울기 (Gradient)
$ h_i = \frac{\partial^2 L(y_i, \hat{y}_i) }{\partial \hat{y}_i^2}$ : 손실 함수의 2차 기울기 (Hessian)
$f_t(x_i)$ : 새로운 학습기의 예측값

## 6.4. Optimization of Tree Structure
XGB 는 각 학습기의 구조를 최적화할 때, 분할 기준을 $g_i$ 와 $h_i$ 를 사용해 계산한다. 이를 이용해 각 트리의 노드에서 데이터 분할 기준을 결정한다.

1. 노드의 최적 가중치 계산
트리의 리프 노드에 대한 최적의 가중치 $\omega^*$ 를 다음과 같이 정의한다.

$$
\omega^* = -\frac{\sum_i g_i}{\sum_i h_i + \lambda}
$$

2. 분할 이득 계산
노드 분할의 이득을 다음과 같이 정의한다.

$$
Gain = \frac{1}{2} \left[\frac{(\sum_{i \in L} g_i)^2}{\sum_{i \in L} h_i + \lambda} + \frac{(\sum_{i \in R} g_i)^2}{\sum_{i \in L} h_i + \lambda} - \frac{(\sum_{i} g_i)^2}{\sum_{i} h_i + \lambda}\right] - \gamma 
$$

여기서 $L$ 과 $R$ 은 각각 왼쪽과 오른쪽 자식 노드이고 $\gamma$ 는 노드 분할에 대한 패널티이다. 즉, 이 분할 이득이 양수일 경우에만 해당 노드를 분할한다. 이 과정을 반복해 트리가 성장하고 최적화된 트리 구조를 얻는다.


## 6.5. Advantages & Disadvantages of XGB
#### - Advantages
1. 병렬 처리와 다양한 최적 기술로 인한 빠른 학습 속도
2. 높은 예측 성능
3. 유연성
4. 정규화, 가지치기, 학습률 조정 등을 이용한 과적합 방지

#### - Disadvantages
1. 복잡성이 높음
2. 해석이 난해



# 7. CatBoost
## 7.1. Definition
CatBoost(Categorical Boosting) 은 범주형 데이터를 처리하는데 특화된 Gradient Boosting 알고리즘이다. 기본적으로 다른 부스팅 알고리즘들과 유사하며 범주형 데이터에 효과적이고 강력한 성능을 보인다. 

## 7.2. Target Encoding
대부분의 기계학습 알고리즘은 범주형 데이터를 처리할 때 원-핫 인코딩이나 라벨 인코딩과 같이 데이터의 차원을 증가시키는 기법을 적용하기 때문에 계산 비용이 높아지고 모형 성능이 감소하는 현상이 발생한다. 이에 반해 CatBoost 는 범주형 데이터에 대한 타겟 인코딩을 적용함으로써 각 범주를 평균값이나 통계적 수치로 변환해 더 효율적으로 연산을 진행한다. 이는 다음과 같이 표현할 수 있다.

$$
TE (x_i) = \frac{\sum_{j \neq i} y_j \cdot I(x_j = x_i)}{\sum_{j \neq i} I(x_j = x_i)} 
$$

과적합을 방지하기 위해 $i$ 번째 데이터를 제외한 나머지 데이터의 정보를 사용해 평균값을 계산한다. 또한, 순서를 무작위로 진행해 더 안정적인 인코딩 값을 얻는다.

## 7.3. Advantages & Disadvantages of XGB
#### - Advantages
1. 범주형 데이터에 특화 
2. 높은 예측 성능
3. 순서 무작위화와 타켓 인코딩을 결합해 과적합 방지
4. 병렬 처리와 효율적인 알고리즘으로 빠르게 학습

#### - Disadvantages
1. 하이퍼 파라미터 튜닝이 필요해 최초 설정이 복잡
2. 타켓 인코딩으로 인한 메모리 사용량이 많음


# 참고
https://wooono.tistory.com/104
https://leedakyeong.tistory.com/entry/%EC%9D%98%EC%82%AC%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4Decision-Tree-CART-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%A7%80%EB%8B%88%EA%B3%84%EC%88%98Gini-Index%EB%9E%80
https://leedakyeong.tistory.com/entry/Decision-Tree%EB%9E%80-ID3-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98