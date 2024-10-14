---
layout: single        # 문서 형식
title: Auto Encoder Basics (2021) # 제목
categories: Neural Network    # 카테고리
toc: true             # 글 목차
author_profile: false # 홈페이지 프로필이 다른 페이지에도 뜨는지 여부
sidebar:              # 페이지 왼쪽에 카테고리 지정
    nav: "docs"       # sidebar의 주소 지정
#search: false # 블로그 내 검색 비활성화
use_math: true
---
# Keywords
Unsupervised manner, latent space, encoder & decoder, clustering


# 1. Introduction
#### - Purpose 
비지도 학습 기반의 군집화와 같은 다양한 작업에 사용

#### - Notation
$A : \mathbb{R}^n \rightarrow \mathbb{R}^p$ (encoder)
$B :  \mathbb{R}^p \rightarrow \mathbb{R}^n$ (decoder)
$\Delta $ : reconstruction loss function, decoder의 결과물과 input 간 거리 비교, L-2 정규화 사용
$\argmin_{A,B} \mathbb{E}[\Delta(\mathbf{x}, B \circ A (\mathbf{x}))]$ 를 만족하는 $A,B$ 를 학습하는 것이 목표

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/AE/figure1.jpg" height = 200>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 1 : AE Example ]</figcaption>
</figure>


#### - Types
1. $A,B$ : nerual networks
2. $A,B$ : linear operations $\rightarrow$ linear autoencoder 
3. $A,B$ : drop non-linear operations in LAE(Linear AutoEncoder) $\rightarrow$ PCA
$\rightarrow$ Autoencoder는 PCA의 일반화 버전
4. $A,B$ : 층대로 점진적으로 학습 -> 'stacked' 버전



# 2. Regularized AutoEncoders
#### - 정규화가 필요한 이유
$A, B$ 를 identity operator로 설정하면 input과 동일한 output이 나오므로, 다른 결과를 내기 위해 추가적인 정규화가 필요하다. 또한 은닉층의 크기가 입력값의 크기와 동일하거나 더 큰 경우, 인코더가 identity 함수를 학습할 가능성이 존재한다. 이 경우 역시 정규화가 필요하다. 

#### - Bottleneck
representation의 차원을 input보다 축소해 병목(bottleneck) 현상을 유도하는 것을 의미한다. 이는 데이터 압축, 특징 추출 등에 사용한다. 병목 구간이 작으면 데이터가 많이 압축가능한 것처럼 보이지만, 실제로는 인코더와 디코더의 복잡성이 충분하다면 모델이 각 데이터를 개별적으로 학습해 과적합이 발생할 수 있다.


## 2.1. Sparse AE
편향-분산 tradeoff 를 다루는 방법 중 하나로 은닉층의 활성화에 대한 희소성을 강화하는 것이다. 

1. $L_1$ regularization
희소성을 추가하기 위해 $L_1$ 정규를 추가한다. 이를 추가한 AE 최적화 목적함수는 다음과 같다.

$$
\argmin_{A,B} \mathbb{E} [\Delta (\mathbf{x}, B \circ A (\mathbf{x}))] + \lambda \sum_i |a_i|
$$

여기서 $a_i$ 는 i번째 은닉층의 활성화이고 모든 은닉층의 활성화를 i번 반복한다. 

2. Kullback & Leibler Divergence
$\lambda$ 를 사용하는 대신 KL 발산을 이용한다. 각 뉴런의 활성확률을 $Bernoulli(p)$ 로 가정하여 이 확률을 조정한다. 각 뉴런 $j$ 에 대해 경험적 확률 $\hat{p}_j = \frac{1}{m} \sum_{i} a_i(x)$ 과 실제 확률간 차이를 측정해 정규화 계수로 사용하는데 이 차이가 0이 되도록 함수를 학습한다. 전체 손실함수는 다음과 같다.

$$
\argmin_{A,B} \mathbb{E} [\Delta (\mathbf{x}, B \circ A (\mathbf{x}))] + \sum_j KL(p || \hat{p}_j)
$$

## 2.2. Denoising AE
<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/AE/figure2.jpg" height = 300>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 2 : Denosing AE Example ]</figcaption>
</figure>

Denosing AE는 input에 여러 잡음(noise)들을 추가해 AE가 깨끗한 input을 내도록 한다. 

$\tilde{x} \sim C(\tilde{\mathbf{x}}|\mathbf{x})$, element-wise product $\odot$ 에 대해
1. $ C_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}(\mathbf{x}, \sigma^2 \mathcal{I})$
여기서 분산 $\sigma$ 를 잡음의 정도로 설정한다.

2. $ C_{p}(\tilde{\mathbf{x}}|\mathbf{x}) = \beta \odot \mathbf{x}, \quad \beta \sim Ber(p)$
여기서 $p$ 는 $\mathbf{x}$ 가 0이나 무효화되지 않을 확률을 설정한다. 


## 2.3. Contractive AE
Contractive AE 은 인코더의 특징 추출이 작은 변동에도 덜 민감하도록 구성한다. 이를 위해 인코더가 디코더에 의한 복원에 중요하지않은 입력의 변화를 무시하도록 한다. 은닉층 $h$ 에 대한 Jacobian 행렬 $J_{ji} = \nabla_{x_i} h_j (x_i)$ 에 대한 전체 최적화 손실 함수는 다음과 같다.

$$
\mathcal{L} = \argmin_{A,B} \mathbb{E}[\Delta (\mathbf{x}, B \circ A(\mathbf{x}))] + \lambda ||J_{A}(\mathbf{x})||_2^2
$$

실제로는 복원 손실함수와 정규화 손실함수는 반대방향으로 결과를 유도한다. 정규화 손실은 잠재 공간에서 불필요한 변동을 억제해 더 단순한 표현을 만들고 복원 손실은 입력값을 더 정확하게 복원하려고 한다. 따라서 두 손실 함수는 상충되는 목표를 가지고 있어서 정규화 손실이 잠재 표현의 변화를 줄이면 복원이 어려워지고, 중요한 정보만 남도록 만들어 과적합을 방지하는 동시에 유의미한 특징을 추출하는 것을 목표로 한다. 


# 3. Variational AE

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/AE/figure3.jpg" height = 300>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 3 : Graphical Representation of VAE ]</figcaption>
</figure>

VAE는 확률 분포를 이용해 데이터 생성을 다룬 생성형 모형이다. 

#### - 가정
관측한 데이터셋 $\mathbf{X} = \{\mathbf{x}\}_{i=1}^N \overset{iid}{\sim} V$ 에 대해 관측하지 않은 잠재 변수 $\mathbf{z}_i$ 가 조건부로 주어지는 생성 모형을 가정

#### - Notation
$\theta$ : 생성 분포를 결정하는 모수, 미지(unknown) $\rightarrow \: p_{\theta}$ : 확률적 디코더 (생성 모형)
$\phi$ : 입력값에 대한 모수, 미지 $\rightarrow \: q_{\phi}$ : 확률적 인코더 
$p_{\theta}(\mathbf{z}_i)$ : 잠재변수 $\mathbf{z}_i$ 에 대한 사전분포

#### - Likelihood
1. marginal log-likelihood 
$\log p_{\theta} (\mathbf{x}_1, \mathbf{x}_2, \dots \mathbf{x}_N)$ 에 대해 각 점에서의 marginal log-likelihood 는 다음과 같다.

$$
\log p_{\theta}(\mathbf{x}_i) = D_{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}_i) || p_{\theta}(\mathbf{z} | \mathbf{x}_i)) + \mathcal{L}(\theta, \phi; \mathbf{x}_i)
$$

2. variational lower bound 

$$
\mathcal{L}(\theta, \phi; \mathbf{x}_i) \overset{\Delta}{=} \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x}_i)} [-\log q_{\phi}(\mathbf{z} | \mathbf{x}) + \log p_{\theta}(\mathbf{x}, \mathbf{z})] \\
$$

3. lower bound with KL divergence
KL divergence는 항상 0보다 크기때문에 다음과 같이 표현할 수 있다.

$$
\mathcal{L}(\theta, \phi; \mathbf{x}_i) = -D_{KL} (q_{\phi} (\mathbf{z} | \mathbf{x}_i) || p_{\theta}(\mathbf{z})) + \mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x}_i)} [\log p_{\theta}(\mathbf{x}_i|\mathbf{z})]
$$

따라서 $\theta$ 와 $\phi$ 에 대해 모든 점에서의 $\mathcal{L}(\theta, \phi; \mathbf{x}_i)$ 을 최대화함으로써 하한에 대해 추론할 수 있다.

4. mini-batch
mini-batch $\mathbf{X}^M = \{ \mathbf{x}_i \}_{i=1}^M$ 을 이용해 다음과 같이 표현할 수 있다.

$$
\mathcal{L} (\theta, \phi ; \mathbf{X}) \approx \tilde{\mathcal{L}}^M (\theta, \phi ; \mathbf{X}^M) = \frac{N}{M} \sum_{i=1}^{M} \mathcal{L}(\theta, \phi ; \mathbf{x}_i)
$$

기존 베이지안 방법의 분해법과 다르게 reparameterization 과 확률적 경사 하강법을 이용해 $\tilde{\mathcal{L}}^M (\theta, \phi ; \mathbf{X}^M)$ 의 기울기를 근사한다. 따라서 사후 분포의 구조적인 가정에 덜 의존하고, 더 유연하게 데이터를 학습할 수 있다. 

## 3.1. The Reparameterization Trick
#### - Notation

$\epsilon \sim p(\epsilon)$ : 보조 잡음 변수 (auxiliary noise variable)
$ g_{\phi} (\epsilon, \mathbf{x})$ : 미분 가능한 함수로의 변환
$\tilde{\mathbf{z}} \sim q_{\phi} (\mathbf{z} | \mathbf{x})$ : $ g_{\phi} (\epsilon, \mathbf{x})$ 를 이용한 확률변수

#### - Reparameterization Trick

$$
\mathcal{L}(\theta, \phi ; \mathbf{x}_i) \approx \tilde{\mathcal{L}} (\theta, \phi ; \mathbf{x}_i) = \frac{1}{L} \sum_{l=1}^L \log p_{\theta}
 (\mathbf{x}_i, \mathbf{z}_{(i, l)}) - \log q_{\phi}
 (\mathbf{z}_{(i, l)} | \mathbf{x}_i)  
$$

#### - Reparameterization Trick using Mini-Batch

$$
\hat{\mathcal{L}}^M (\theta, \phi ; \mathbf{X}) = \frac{N}{M} \sum_{i=1}^M \tilde{\mathcal{L}} (\theta, \phi ; \mathbf{x}_i)
$$

#### - Pseudo-code for VAE

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/AE/algorithm1.jpg" height = 200>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 3 : Pseudo-Code for VAE ]</figcaption>
</figure>

Algorithm1 에서 주로 $M = 100$, $L = 1$ 으로 설정한다. 

#### - 가중합으로 변환

$$
\mathcal{L} (\theta, \phi ; \mathbf{x}_i) = \frac{1}{L} \sum_{l=1}^L \log \frac{1}{k} \sum_{j=1}^k \frac{p_{\theta} (\mathbf{x}_i, \mathbf{z}_{(j,l)})}{q_{\phi} (\mathbf{z}_{z,l} | \mathbf{x}_i)}
$$

이는 가능도 함수 $q_{\phi} (\mathbf{z}_{z,l} | \mathbf{x}_i)$ 를 가중치로 표현해 생성 신경망의 기울기를 근사한 사후분포의 표본들에서 뽑은 가중합으로 학습한다. 


## 3.2. Disentangled AE
lower bound with KL divergence 에서 KL divergence에 $\beta$ 를 추가함으로써 feature들간의 상관관계를 낮추고자 한다. 이는 다음과 같이 표현할 수 있다. 

$$
\mathcal{L} (\theta, \phi ; \mathbf{x}^{(i)}) = -\beta D_{KL}(q_{\phi}(z | \mathbf{x^{(i)}}) || p_{\theta}(z)) + \mathbb{E}_{q_{\phi}(z|\mathbf{x}^{(i)})} [\log p_{\theta}(\mathbf{x}^{(i)} | z) ]
$$

주로 사전분포 $p_{\theta}(z) \sim \mathcal{N}(0, \mathcal{I})$ 로 설정한다. 이 경우 모든 feature들은 상관관계가 없다. 그리고 KL divergence는 잠재 특성 분포 $q_{\phi}(z|\mathbf{x}^{(i)})$ 가 상관관계가 낮아지도록 규제한다.


# 4. Advanced AE Techniques
## 4.1. Wasserstein AE
#### - Wasserstein Distance
Optimal Trasport 거리의 특별한 경우이며, 서로 다른 두 분포간의 거리를 계산한다. 이는 다음과 같이 표현할 수 있다.

$$
W_c(P_X, P_G) = \inf_{\Gamma \in P(X \sim P_X, Y \sim P_G)} \mathbb{E}_{(X,Y) \sim \Gamma} [c(X,Y)]
$$

여기서 $c(x,y)$ 는 비용함수이다. 이 때 $c(x,y) = d^P(x,y)$ 로 평가값일 경우, $W^{\frac{1}{p}}$ 를 p-Wasserstein 거리라고 한다. 그리고 $p=1$ 인 경우 "Earth Moving Distance" 라고 하며 다음과 같이 정의된다.

$$
W_1(P_X, P_G) = \sup_{f \in \mathfrak{F}} \mathbb{E}_{X \sim P_X} [f(X)] - \mathbb{E}_{Y \sim P_G} [f(Y)]
$$

이는 X의 분포를 G의 분포와 최단거리로 일치시키려는 것을 목표로 한다.

#### - Wasserstein AE (WAE)
lower bound with KL divergence 에서는 표본들이 지나치게 유사해져 잠재 공간을 충분히 활용하지 못한다는 단점이 발생한다. GAN에서는 OT 거리를 사용해 실제 이미지의 분포와 가짜 이미지의 분포를 구분한다. 그리고 WAE에서는 AE의 손실 함수를 수정해 다음과 같은 목적 함수를 사용한다.

$$
D_{WAE} (P_X, P_G) = \inf_{Q(Z|X) \in \mathfrak{Q}} [c(X, G(Z))] + \lambda \cdot D_Z(Q_Z, P_Z)  
$$

여기서 $Q$ 는 인코더, $G$ 는 디코더이다. 여기서 inf 부분에서 디코더의 매핑을 통해 transportation plan이 고려되기 때문에 결과 분포와 표본 분포에 대해 패널티가 부과된다. 그리고 뒤쪽 $\lambda$ 부분에서는 잠재 공간 분포와 사전 분포 간의 거리를 페널티로 부과된다. 



# 참고
https://roytravel.tistory.com/133
https://link.springer.com/article/10.1007/s11263-014-0697-5
https://ddongwon.tistory.com/126