---
layout: single        # 문서 형식
title: DeblurGAN # 제목
categories: Deblurring    # 카테고리
tag: [DL, Image, Generative, Lecture]
author_profile: false # 홈페이지 프로필이 다른 페이지에도 뜨는지 여부
sidebar:              # 페이지 왼쪽에 카테고리 지정
    nav: "docs"       # sidebar의 주소 지정
#search: false # 블로그 내 검색 비활성화
use_math: true
---
# Keywords
Blur, GAN, Critic Network


# 1. Introduction
## 1.1. Image Deblurring
#### - Notation
일정하지않은 blur model의 일반적인 식은 다음과 같다.
$$
I_{B} = k(M) * I_{S} + N
$$
$I_{B}$ : blur 이미지
$k_{M}$ : motion field $M$ 으로 결정되는 알 수 없는 blur kernel
$I_{S}$ : 깔끔한 잠재적인 이미지
$ * $ : 합성곱 연산
$ N $ : 추가적인 노이즈

이 경우 deblurring 알고리즘으로 잠재적인 깔끔한 이미지인 $I_S$ 와 blur kernel $k(M)$ 을 추정한다. 이 때 각 픽셀별 blur kernel을 찾는 것은 해가 굉장히 많은 문제(ill-posed problem)이고 휴리스틱한 방법, 이미지 통계, blur에 대한 가정에 의존한다.

#### - Solutions
1. blur kernel 추정
이미지 전반에 걸쳐 blur가 균일하게 발생한다는 가정 하에 카메라 흔들림을 이용해 blur를 처리한다. 유도된 blur kernel 을 기준으로 카메라의 움직임을 추정해 deconvolution 작업을 수행함으로써 원본 이미지를 탐색한다. 또는 blur kernel의 지역 선형성과 간단한 휴리스틱을 가정해 미지의 kernel을 빠르게 추정한다. 이 방법은 빠르지만 제한된 이미지에서만 잘 작동한다.

2. deep learning 
CNN을 기반으로 한 모형들을 이용해 미지의 blur kernel을 추정한다.

## 1.2. GANs
### 1.2.1. Vanilla GANs
Vanilla GANs의 식은 다음과 같다.
$$
\min_{G} \max_{D} \underset{x \sim \mathbb{P}_r}{\mathbb{E}} [\log(D(x))] + \underset{\bar{x} \sim \mathbb{P}_g}{\mathbb{E}} [1-\log(D(\bar{x}))] 
$$
$\mathbb{P}_r$ : 실제 데이터의 분포
$\mathbb{P}_g$ : 모형의 분포
$z \sim P(z)$ : 간단한 잡음 분포의 표본, 입력값
$\bar{x} = G(z)$ : 실제 데이터와 유사한 데이터를 생성하는 생성자

### 1.2.3. WGAN

#### - Wasserstein Distance
Wasserstein 거리는 두 확률 분포 간 거리를 다음과 같이 정의한다.
$$
W(p,q) = \inf_{\gamma \in \Gamma(P,Q)} \underset{(x,y) \sim \gamma}{\mathbb{E}}[||x - y||]
$$
이는 두 분포간 차이가 얼마나 연속적으로 발생하는지 반영하며 겹치지 않는 경우에도 유의미한 값을 제시한다. 주로 이미지에서 스타일 변화나 왜곡을 측정하는데 사용되며 딥러닝에서는 비용함수로 많이 사용된다.

#### - Loss Function

Vanilla GANs을 학습시키는 과정에서 mode collapse, gradient 소실 등 여러 문제들이 발생한다. 이를 해결해고자 다음과 같은 WGAN(Wasserstein GANs) 을 제안한다.
$$
\min_{G} \max_{D \in \mathcal{D}} \underset{x \sim \mathbb{P}_r}{\mathbb{E}} [D(x)] - \underset{\bar{x} \sim \mathbb{P}_g}{\mathbb{E}} [D(\bar{x})]
$$
$\mathcal{D}$ : 1 - Lipschitz 함수의 집합 
$\mathbb{P}_g$ : 모형의 분포

여기서 중요한 아이디어는 Lipschitz 상수 $K$ 와 Wasserstein 거리 $W(P_r, P_{\theta})$ 에 대해 critic value가 $K \cdot W(P_r, P_{\theta})$ 로 근사하는 것이다. WGAN에 Lipschitz 제약을 강제하기 위해 여러 방법을 추가하는데, 그 중 gradient 제약 항을 추가하는 식은 다음과 같다.
$$
\lambda \underset{\bar{x} \sim P_{\bar{x}}}{\mathbb{E}} [(|| \nabla_{\bar{x}} D(\bar{x}) ||_2 -1)^2]
$$
이는 생성자 모형 선택에 있어 강건(robust)하고 hyperparameter 튜닝이 필요없다는 장점이 있다.

### 1.2.3. Conditional Adversarial Networks
Vanilla GANs 과 다르게 cGAN 은 y 라는 조건을 추가해 자신이 원하는 label 의 데이터를 생성하는 모형이다. 이를 그림으로 표현하면 다음과 같다.

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/DeblurGAN/cgan architecture.jpg" height = 400>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 1 : Conditional Adversarial Network ]</figcaption>
</figure>

손실함수 역시 Vanilla GANs 의 그것에서 y를 조건부 확률로 추가한 형태이다. 이는 다음과 같다.

$$
\min_{G} \max_{D} \: V(D,G) =  \underset{x \sim p_{data}(\mathbf{x})}{\mathbb{E}} [\log D(\mathbb{x}|\mathbb{y})] - \underset{z \sim p_z(z)}{\mathbb{E}} [\log(1-D(G(z|y)))]
$$

# 2. Motion Blur Generation
대부분의 상황에서 깔끔한 이미지와 blur 이미지 짝을 찾는 것은 쉽지 않다. 그래서 본 논문에서는 깔끔한 이미지에 Markov Process를 이용해 blur 이미지를 생성해 데이터셋을 구성한다. 이에 관한 알고리즘은 다음과 같다.

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/DeblurGAN/algorithm1.jpg" height = 500>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Algorithm 1 : Motion Blur Kernel Generation ]</figcaption>
</figure>

1. random trajectories generation 적용
2. sub-pixel 보간법을 trajectory vector에 적용해 blur kernel 생성
$\rightarrow$ trajectory vector : 이미지 상에서 물체의 움직임을 표현하는 벡터
3. 이 때 Trajectory 생성은 Markov Process를 적용
4. trajectory의 다음 위치는 다양한 요소를 근거로 무작위로 생성

# 3. The Proposed Method
#### - 목표
입력값으로 오직 blurred image $I_B$ 만 주어졌을 때 깔끔한 이미지 $I_S$ 복원

#### - 학습
생성자로 사전학습된 CNN $G_{\theta_G}$ 를 이용해 Deblurring을 진행한다. 그리고 판별자 $D_{\theta_D}$ 를 추가해 기존 GANs과 같이 두 신경망을 동시에 학습한다. 대략적인 학습 구조는 다음과 같다.

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/DeblurGAN/training.jpg" height = 500>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 2 : DeblurGAN Training ]</figcaption>
</figure>


## 3.1. Loss Function
손실함수를 다음과 같이 정의한다.

$$
\mathcal{L} = \mathcal{L}_{GAN} + \lambda \cdot \mathcal{L}_{X}
$$

$\mathcal{L}_{GAN}$ : 적대적 신경망(adverasiral net)의 손실
$\mathcal{L}_{X}$ : 생성 신경망(content net)의 손실
$\lambda$ : 규제항, 본 논문의 실험에서는 모두 100으로 설정

이 때, 입력값과 출력값 사이의 불일치에 대한 규제가 필요없기 때문에 판별자에 대한 조건을 부여하지 않는다.

#### - Advesarial Loss
본 논문에서는 WGAN-GP를 critic function 으로 사용하는데, 이는 생성자 모형 선택에 있어 더욱 강건하다는 특징을 갖는다.

$$
\mathcal{L}_{GAN} = \sum_{n=1}^{N} D_{\theta_{D}}(G_{\theta_G}(I^{B}))
$$

GANs의 요소가 없이 훈련된 DeBlurGAN은 수렴하지만, 부드럽고 blur 가 있는 이미지를 생성한다. 이는 주로 부분적인 세부사항을 복원하는데 집중한다.

#### - Content Loss
Content 손실로 L2-손실인 인지적 손실을 사용하는데, 이는 생성된 이미지와 목표 이미지의 CNN 특징 맵 간의 차이에 기반한다. 그리고 다음과 같이 정의한다.

$$
\mathcal{L} = \frac{1}{W_{i,j} H_{i,j}} \sum_{x=1}^{W_{i,j}} \sum_{y=1}^{H_{i,j}}(\phi_{i,j}(I^{S})_{x,y} - \phi_{i,j}(G_{\theta_G}(I^{B}))_{x,y})
$$

$\phi_{i,j}$ : VGG19 에서 ImagNet으로 사전 학습된, i번째 맥스풀링 층 이전의 j번째 합성곱으로 얻은 특징 맵
$W_{i,j}, H_{i,j}$ : 특징 맵의 차원(너비, 높이)

원본 이미지의 특징맵과 blur 이미지를 깔끔하게 생성한 특징맵의 차이를 비교(L2 norm) 후 전체를 특징맵의 크기만큼으로 나누어 scaling을 진행하는 구조다. 본 논문에서는 i=3, j=3 인 $VGG_{3,3}$ 을 사용했다. 이는 주로 전체적인 이미지를 복원하는데 집중한다. 

## 3.2. Network Architecture
#### - Generator
<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/DeblurGAN/generator.jpg" height = 100>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 3 : Generator Architecture ]</figcaption>
</figure>

생성자는 stride 가 2인 residual block 9개와 transposed convolutional block 2개로 구성된다. 

#### - Critic Networks
gradient 규제를 추가한 Wasserstein GAN 의 critic network 인 $D_{\theta_{D}}$ 를 사용한다. 이는 PatchGAN 의 그것과 동일하다.


# 참고
https://roytravel.tistory.com/133
https://link.springer.com/article/10.1007/s11263-014-0697-5
https://ddongwon.tistory.com/126