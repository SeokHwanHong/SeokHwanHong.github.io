---
layout: single        # 문서 형식
title: Total Variation # 제목
categories: Electronics    # 카테고리
tag: [CV, Mathematics, Electronics]
author_profile: false # 홈페이지 프로필이 다른 페이지에도 뜨는지 여부
sidebar:              # 페이지 왼쪽에 카테고리 지정
    nav: "counts"       # sidebar의 주소 지정
#search: false # 블로그 내 검색 비활성화
use_math: true
---
# Keywords
Bayes's Thoerem, MAP, Regularization, ADMM


# 1. Inverse Problem
## 1.1. Image Deconvolution
#### - Notation
$b$ : 실제 데이터 (blur 가 존재하는 이미지)

$x$ : 원본 이미지 (blur 가 존재하지 않는 이미지) 

$c$ : blur kernel (카메라 렌즈 등에 의한 blur)

$$\eta$$ : 노이즈

#### - Image Formation Model
**1. real world**

$$
\begin{split}
    b = c * x + \eta
\end{split}
$$


**2. frequency domain by Fourier Trasform**

$$
\begin{split}
    b = \mathcal{F}^{-1} \{\mathcal{F} \{c\} \cdot \mathcal{F} \{x\} \} + \eta
\end{split}
$$


**3. inverse filtering**

inverse filtering 을 이용해 noise 가 없이 계산이 가능하다.

$$
\begin{split}
    \tilde{x}_{IF} = \mathcal{F}^{-1} \left\{ \frac{\mathcal{F} \{b\}}{\mathcal{F} \{c\}}   \right\}
\end{split}
$$


**4. Wiener filtering**

$$
\begin{split}
    \tilde{x}_{WF} = \mathcal{F}^{-1} \left\{ \frac{ \vert \mathcal{F} \{c\} \vert^2}{\vert \mathcal{F} \{c\} \vert^2 + 1/SNR} \cdot \frac{\mathcal{F}\{b\}}{\mathcal{F}\{c\}} \right\}
\end{split}
$$


## 1.2. Bayesian Perspective
#### - Ill-Posed Problem
역문제 풀이에서 measurement를 만족하는 무수히 많은 해가 발생하는 경우가 있다. 이를 가장 적절히 처리하기 위해 이미지를 베이지안 관점의 식으로 해석한다.

#### - Notation
$$\mathbf{A} \in \mathbb{R}^{M \times N}$$ : blur kernel 

$$\mathbf{b} \in \mathbb{R}^{M},\: \mathbf{b}_i \sim \mathcal{N}((Ax)_i, \sigma^2)$$ : 실제로 얻은 이미지

$$\mathbf{x} \in \mathbb{R}^{M},\: \mathbf{x}_i \sim \mathcal{N}(\mathbf{x}_i, 0) $$ : 원본 이미지

$$\eta_i \overset{iid}{\sim} \mathcal{N} (0, \sigma^2)$$ : 각 노이즈는 픽셀에 대해 iid(identically independently distributed)


#### - Mathematical Expression of Images

**1. Image formation model**

$$
\begin{split}
    \mathbf{b} = \mathbf{Ax} + \eta
\end{split}
$$


**2. Probability ($$\mathbb{P}$$) of observation $$i$$**

$$
\begin{split}
    p(\mathbf{b}_i \vert \mathbf{x}_i, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left[ -\frac{(\mathbf{b}_i - (\mathbf{Ax}_i))^2}{2\sigma^2}\right]
\end{split}
$$


**3. Joint $$\mathbb{P}$$ of all observations**

$$
\begin{split}
    p(\mathbf{b} \vert \mathbf{x}, \sigma) &= \prod_{i=1}^M p(\mathbf{b}_i \vert \mathbf{x}_i, \sigma) \\
    &= \prod_{i=1}^M\frac{1}{\sqrt{2\pi\sigma^2}} \exp \left[ -\frac{(\mathbf{b}_i - (\mathbf{Ax}_i))^2}{2\sigma^2}\right] \\
    &\propto \exp \left[ -\frac{(\mathbf{b} - (\mathbf{Ax}))^2}{2\sigma^2}\right]
\end{split}
$$


#### - Bayesian perspective
**1. Bayes'rule**

베이즈룰을 이용해 사후분포를 분해하면 다음과 같이 표현할 수 있다.

$$
\begin{split}
    \underbrace{p(\mathbf{x} \vert \mathbf{b}, \sigma)}_{posterior} = \frac{p(\mathbf{b} \vert \mathbf{x}, \sigma) \cdot \mathbf{p}(\mathbf{x})}{p(\mathbf{b})} \propto \underbrace{p(\mathbf{b} \vert \mathbf{x}, \sigma)}_{\text{image formation model}} \cdot \underbrace{\mathbf{p}(\mathbf{x})}_{prior}
\end{split}
$$

이 때, $$\mathbf{p}$$ 에 대한 분포는 주로 주어져 있으므로, 상수로 취급해 식을 전개하는 경우 무시할 수 있다.


**2. Maximum-a-posterior(MAP) solution**

1번에서 분해한 식에 대해 이미지의 노이즈를 최소로 하는 $x$ 를 다음과 같은 최적화문제로 표현할 수 있다.

$$
\begin{split}
    \mathbf{x}_{MAP} &= argmin_x - \log (p(\mathbf{x} \vert \mathbf{b}, \sigma)) \\ 
    &\propto argmin_x - \log (p(\mathbf{b} \vert \mathbf{x}, \sigma)) - \log (p(\mathbf{x})) \\
    &= \frac{1}{2\sigma^2} \Vert \mathbf{b} - \mathbf{Ax} \Vert^2_2 + \Psi(\mathbf{x})
\end{split}
$$

이 때 사전분포 $$\log (p(\mathbf{x}))$$ 는 $$\mathbf{x}$$ 의 분포를 반영하는데, 이를 regularizer $$\Psi(\mathbf{x})$$ 로 표현할 수 있다. 이는 분야마다 함수, 알고리즘, 딥러닝 모형 등으로 다양하게 표현이 가능하며 반영하는 정보에 따라 원하는 최적화 값으로 구성이 가능하다. 


## 1.3. Examples of Priors / Regularizers

<p align="center">
  <a href="#">
    <img src="/images/TV/image1.jpg" height="175" />
  </a>
  <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>[ Figure 1 : Types of Images ]</b> 
</p>

Figure 1 의 왼쪽 사진은 blur 가 심한 이미지이다. 이미지 내 객체들의 edge들을 명확히 구분하기 위해서는 $$\Psi(\mathbf{x}) = \Vert \Delta \mathbf{x} \Vert_{2} $$ 과 같은 Laplace 연산자를 사용한다.  가운데 사진은 객체가 드문드문 존재하는데, 이를 이미지 상에서는 0과 1로만 표현이 가능하다. 따라서 $$\Psi(\mathbf{x}) = \Vert \mathbf{x} \Vert_{1}$$ 과 같은  연산자를 사용한다. 마지막으로 오른쪽 사진은 일반적인 이미지로, 이미지 내 픽셀들의 기울기 들이 거의 0으로 수렴한다. 따라서 대체로 급격한 pixel 변화가 없다는 정보를 반영하기 위해 Total Variation filter 를 사용한다.



# 2. Total Variation
## 2.1. Definition
Total Variation (TV) 은 신호나 이미지의 픽셀 간 변화의 총합을 측정하는 방법이다. edge 에서는 변화량이 높은 값을 갖기 때문에 이를 최소화하면 이미지 내 급격한 변화는 유지하며 부드러운 영역은 매끄럽게 처리가 가능하다. 1차원 신호 $f$ 가 주어졌을 때, TV 는 인접한 값 간의 차이의 절대값을 합산하는 식으로 다음과 같이 표현 가능하다.

$$
\begin{split}
    TV(f) = \sum_{i=1}^{N-1} \vert f_{i+1} - f_i \vert
\end{split}
$$

## 2.2. 2-D TV
#### - Pixel Varaiation Operators

2차원 신호는 행렬 연산자를 이용하면 다음과 같다.

$$\mathbf{x}$$ : 입력 이미지 

$$d_x = \begin{bmatrix} 0 & 0 & 0 \\ 0 & -1 & 1 \\ 0 & 0 & 0 \end{bmatrix}$$ : x 축 이동방향에 대한 픽셀 변화 연산자

$$d_x = \begin{bmatrix} 0 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 1 & 0 \end{bmatrix}$$ : y 축 이동방향에 대한 픽셀 변화 연산자

따라서 각 축에 대한 연산자를 이용해 픽셀 변화를 감지해 edge를 표현할 수 있다. 

<p align="center">
  <a href="#">
    <img src="/images/TV/image2.jpg" height="175" />
  </a>
  <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>[ Figure 2 : Images Using 1-d TV Regularization ]</b> 
</p>

왼쪽부터 각각 grey scale 이 적용된 원본이미지 $$(\mathbf{x}), \; d_x$$ 가 적용된 이미지 $$(\mathbf{D}_x(\mathbf{x}))$$, 그리고 $$d_y$$ 가 적용된 이미지 $$(\mathbf{D}_y(\mathbf{x}))$$ 이다. 각 축에 대해 연산자를 따로 적용가능하며, 연산자를 한번에 적용할 수도 있다. 


#### - TV pseudo-norm
TV 가 0으로 수렴하면 이미지 상에서는 검정색으로 나타나고, 1로 수렴하면 하얀색으로 edge가 두드러진다. 따라서 TV 를 조정하기 위해 pseududo-norm 을 이용해 표현하는데, 이는 다음과 같다.

$$
\begin{split}
    TV_{anisotropic} (\mathbf{x}) &= \Vert \mathbf{D}_x(\mathbf{x})\Vert_{1} + \Vert \mathbf{D}_y(\mathbf{x}) \Vert_{1} \\ 
    &= \sum_{i=1}^N  \vert \mathbf{D}_x(\mathbf{x})_i \vert + \vert \mathbf{D}_y(\mathbf{x})_i \vert \\ 
    &= \sqrt{(\mathbf{D}_x(\mathbf{x}))_i^2} + \sqrt{(\mathbf{D}_y(\mathbf{x}))_i^2} \\
\end{split}
$$

$$
\begin{split}
    TV_{isotropic} (\mathbf{x}) &= \Vert \mathbf{D}(\mathbf{x}) \Vert_{2,1}\\ 
    &= \sum_{i=1}^N \left \Vert \begin{bmatrix} \mathbf{D}_x(\mathbf{x})_i \\
     \mathbf{D}_x(\mathbf{y})_i \end{bmatrix} \right \Vert_{2} \\
      &= \sum_{i=1}^N \sqrt{(\mathbf{D}_x(\mathbf{x}))_i^2 + (\mathbf{D}_y(\mathbf{x}))_i^2} \\
\end{split}
$$

이를 실제 이미지에 적용하면 다음 그림과 같다.

<p align="center">
  <a href="#">
    <img src="/images/TV/image3.jpg" height="175" />
  </a>
  <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>[ Figure 3 : Images Using 2-d TV Regularization ]</b> 
</p>

가운데 사진은 연산자 조합이 isotropic TV 가 적용된 것이고, 오른쪽 사진은 연산자 조합이 anisotropic TV 가 적용된 것이다. 위 예시들에서 알 수 있듯이, TV pseudo-norm 은 자연 이미지에서 가장 인기있는 정규화 방법 중 하나이다. 



# 3. Regularized Inverse Problem
## 3.1. Solving Regularized Inverse Problem
#### - Notation
$$\mathbf{A}$$ : 필터

$$\mathbf{b}$$ : 필터를 거쳐 얻은 이미지

$$\mathbf{x}$$ : 원본 이미지

$$\Psi$$ : 정규화 함수

$$\lambda$$ : 정규화 가중치


#### - Objective Function
정규화 방법을 추가한 역문제는 다음과 같이 표현할 수 있다.

$$
\begin{split}
    Objective = \min_{\mathbf{x}} \frac{1}{2} \Vert \mathbf{b} - \mathbf{Ax} \Vert_{2}^2 + \lambda \Psi(\mathbf{x})
\end{split}
$$

이를 해결하기 위해 다양한 방법을 적용할 수 있는데, 일반적으로 알고리즘을 이용해 반복적으로 parameter 값들을 갱신한다. 이외에도 PyTorch 내 Adam solver를 사용하는 방법도 존재한다. 


## 3.2. Regularized Image Reconstruction
정규화된 이미지 역문제를 해결하기 위해 slack variable $z$ 을 도입 후 Lagrange multiplier 를 적용하는 방법이 존재한다. 이는 다음과 같이 표현할 수 있다.

**1. Slack Variable 도입**

$$
\begin{split}
    Objective &= \min_{\mathbf{x}} \frac{1}{2} \Vert \mathbf{b} - \mathbf{Ax} \Vert_{2}^2 + \lambda \Psi(\mathbf{x}) \\ 
    &= \min_{\mathbf{x}} \frac{1}{2} \Vert \mathbf{b} - \mathbf{Ax} \Vert_{2}^2 + \lambda \Psi(\mathbf{z}) \\
    &\quad \text{subject to} \quad Kx-z = 0
\end{split}
$$

여기서 $K$ 는 slack 변수를 도입하기 위한 임의의 제약조건이다.


**2. Lagrange Multiplier 적용**

$$f(x) = \min_{\mathbf{x}}\frac{1}{2} \Vert \mathbf{b} - \mathbf{Ax} \Vert_{2}^2, \;\; g(z) = \lambda \Psi(z)$$ 에 대해 Lagrange Multiplier 을 적용한 식을 다음과 같이 표현할 수 있다.

$$
\begin{split}
    L(x,z,y) = f(x) + g(z) + y^{T} (Kx - z)
\end{split}
$$

그리고 우리의 목표는 다음과 같다.

$$
\begin{split}
    \nabla_{x,z,y} L = 0 \Longleftrightarrow \nabla_{x,z}f(x) + g(z) = y^T \nabla_{x,z} (Kx-z) \quad \text{subject to} \quad Kx-z=0
\end{split}
$$

2개의 조건에 맞춰 최적의 $x,z,y$ 를 계산해야하는데, 이는 수학적으로 쉽게 풀 수는 없다. 


## 3.3. Augmented Lagrange
따라서 3.2.2 의 문제를 해결하기 위해 항을 추가해 수렴속도와 안정성을 보완하고자 한다. penalty parameter $\rho$ 에 대해 다음과 같이 표현할 수 있다. 

$$
\begin{split}
    L_{\rho} (x,z,y) = f(x) + g(z) + y^T (Kx-z) + \frac{\rho}{2} \Vert Kx-z \Vert_{2}^2
\end{split}
$$

여기서 추가적으로 scaled dual variable $$u = \frac{y}{\rho}$$ 를 추가해 다음과 같이 표현할 수 있다.

$$
\begin{split}
    L_{\rho} (x,z,y) = f(x) + g(z) + \frac{\rho}{2} \Vert Kx-z+u \Vert^2 + \frac{\rho}{2} \Vert u \Vert^2
\end{split}
$$

## 3.4. Alternating Direction Method of Multipliers
Alternating Direction Method of Multipliers (ADMM) 은 최적화 문제를 해결하기 위한 반복적 알고리즘으로, 복잡한 최적화 문제를 더 단순한 하위 문제로 분할해 이를 반복적으로 해결함으로써 전체 최적화 문제를 해결하는 방법이다. proximal operator $$prox_{\lambda, f}(v) = argmin_{x} \left(f(x) + \frac{1}{2 \lambda} \Vert x-v \Vert_{2}^2\right)$$ 에 대해 scaled dual variable 가 적용된 Augmented Lagrange 문제에 ADMM 을 도입하는 알고리즘은 다음과 같다.

$$
\begin{split}
    &\text{repeat until converged}\\
    &x \leftarrow prox_{\Vert \cdot \Vert_{2}, \rho} (\nu) = argmin_{x} L_{\rho} (x,z,y) = argmin_{x} \frac{1}{2} \Vert Ax-b \Vert_{2}^2 + \frac{\rho}{2} \Vert Kx - \nu \Vert, \; \nu = z - u \\
    &z \leftarrow prox_{\Psi, \rho} (\nu) = argmin_{z} L_{\rho} (x,z,y) = argmin_{z} \lambda \Psi(z) + \frac{\rho}{2} \Vert \nu-z \Vert, \; \nu = Kx + u \\
    &u \leftarrow u + Kx - z \\
\end{split}
$$

이 때, $$x$$ 는 closed form 으로 계산이 가능하므로 $$\tilde{A}x = \tilde{b}$$ 를 이용해 $$x$$ 를 갱신할 수 있으므로 다음과 같이 표현할 수 있다.

$$
\begin{split}
    prox_{\Vert \cdot \Vert_{2}, \rho} (\nu) = \left( \underbrace{A^T A + \rho K^T K}_{\tilde{A}} \right)^{-1} \left( \underbrace{A^T b + \rho K^T \nu }_{\tilde{b}} \right)
\end{split}
$$

여기서 $$\tilde{A}$$ 가 대칭이고 positive definite 행렬이라면 conjugate gradient 방법을 적용할 수 있다.



# 4. ADMM for Image Deconvolution with TV
#### - Notation
$$\mathbf{x} \in \mathbb{R}^N $$ : 미지의 깔끔한 이미지

$$\mathbf{C} \in \mathbb{R}^{N\times N} $$ : 합성곱 필터

$$\mathbf{z}, \mathbf{u}  \in \mathbb{R}^{2N} $$ : slack, dual 변수

$$\mathbf{D} = \begin{bmatrix} \mathbf{D}_x \\ \mathbf{D}_y\end{bmatrix} \in \mathbb{R}^{2N \times N} $$ : 유한한 TV 정규화 연산자


## 4.1. Deconvolution Problem
ADMM 을 적용해 Deconvolution 을 수행하기 전 Agumented Lagrange 에서 TV 를 이용한 정규화를 표현하면 다음과 같다.

$$
\begin{split}
    L_{\rho} (\mathbf{x},&\mathbf{z},\mathbf{u}) = f(\mathbf{x}) + g(\mathbf{z}) + \frac{\rho}{2} \Vert \mathbf{Kx-z+u} \Vert^2 + \frac{\rho}{2} \Vert \mathbf{u} \Vert^2 \\
    &= \underbrace{\frac{1}{2} \Vert \mathbf{Cx} - \mathbf{b} \Vert_{2}^2}_{f(\mathbf{x})} + \underbrace{\lambda \Vert \mathbf{z} \Vert_{1}}_{g(\mathbf{z})} + \frac{\rho}{2} \Vert \underbrace{\mathbf{Dx-z+u}}_{TV \; regularization} \Vert^2 + \frac{\rho}{2} \Vert \mathbf{u} \Vert^2
\end{split}
$$


## 4.2. Update $\mathbf{x}$
따라서 위 문제에 ADMM 을 적용하면 다음과 같다.

$$
\begin{split}
    &\text{while not converged}\\
    &x \leftarrow prox_{\Vert \cdot \Vert_{2}, \rho} (\nu) =  argmin_{x} \frac{1}{2} \Vert \mathbf{Cx-b} \Vert_{2}^2 + \frac{\rho}{2} \Vert \mathbf{Dx - \nu} \Vert_{2}^2, \; \mathbf{\nu = z - u} \\
    &z \leftarrow prox_{ \Vert \cdot \Vert_{1}, \rho} (\nu) = argmin_{z} \lambda \Vert \mathbf{z} \Vert_{1} + \frac{\rho}{2} \Vert \mathbf{\nu-z} \Vert, \; \mathbf{\nu = Dx + u} 
\end{split}
$$

여기서 $x$ 를 갱신하기 위해 식을 정리하면 다음과 같다.

$$
\begin{split}
    &\frac{1}{2} \Vert \mathbf{Cx-b} \Vert_{2}^2 + \frac{\rho}{2} \Vert \mathbf{Dx - \nu} \Vert_{2}^2 \\
    &= \frac{1}{2}(\mathbf{Cx-b})^T (\mathbf{Cx-b}) + \frac{\rho}{2}(\mathbf{Dx-\nu})^T (\mathbf{Dx-\nu}) \\
    &= \frac{1}{2} (\mathbf{x}^T \mathbf{C}^T \mathbf{Cx} - 2 \mathbf{x}^T \mathbf{C}^T \mathbf{b} + \mathbf{b}^T \mathbf{b}) + \frac{\rho}{2} (\mathbf{x}^T \mathbf{D}^T \mathbf{Dx} - 2 \mathbf{x}^T \mathbf{C}^T \nu + \nu^T \nu)  \\
\end{split}
$$

이제 기울기를 0으로 설정해 최적의 값을 탐색하고, 이는 다음과 같이 표현할 수 있다.

$$
\begin{split}
    0 = \nabla_x f(\mathbf{x}) = \mathbf{C}^T \mathbf{Cx} - \mathbf{C}^T \mathbf{b} + \rho \mathbf{D}^T \mathbf{Dx} - \rho \mathbf{D}^T \nu
\end{split}
$$

따라서 이를 x에 대해 정리하면 closed-form 의 해를 구할 수 있고, 다음과 같이 표현할 수 있다.

$$
\begin{split}
    \mathbf{x} \leftarrow (\mathbf{C}^T \mathbf{C} + \rho \mathbf{D}^T \mathbf{D})^{-1} (\mathbf{C}^T \mathbf{b} + \rho \mathbf{D}^T \nu)
\end{split}
$$


## 4.3. Fourier Transform
그러나 $x$ 를 갱신하는 과정에서 합성곱 연산으로 인해 연산량이 증가한다는 문제점이 발생한다. 이를 해결하기 위해 Fourier 변환을 적용해 주파수 영역에서 연산을 진행한다. 따라서 각 요소를 다음과 같이 변환할 수 있다.

$$
\begin{split}
    &\mathbf{C}^T \mathbf{C} \Longleftrightarrow \mathcal{F}^{-1} \{ \mathcal{F}\{c\}^* \cdot \mathcal{F}\{c\} \} \\
    &\mathbf{C}^T \mathbf{b} \Longleftrightarrow \mathcal{F}^{-1} \{ \mathcal{F}\{c\}^* \cdot \mathcal{F}\{b\} \} \\
    &\mathbf{D}^T \mathbf{D} \Longleftrightarrow \mathcal{F}^{-1} \{ \mathcal{F}\{d_x\}^* \cdot \mathcal{F}\{d_x\} + \mathcal{F}\{d_y\}^* \cdot \mathcal{F}\{d_y\} \} \\
    &\mathbf{D}^T \mathbf{z} \Longleftrightarrow \mathcal{F}^{-1} \{ \mathcal{F}\{d_x\}^* \cdot \mathcal{F}\{\nu_1\} + \mathcal{F}\{d_y\}^* \cdot \mathcal{F}\{\nu_2\} \} \\
\end{split}
$$ 

여기서 $$\nu_1 = \nu(1:N), \: \nu_2 = \nu(N+1:2N)$$ 이다. 이를 정리해 proximal operator 에 대입하면 다음과 같이 표현할 수 있다.

$$
\begin{split}
    prox_{||\cdot||_2, \rho} (\mathbf{x}) = \mathcal{F}^{-1} \left\{ \frac{\mathcal{F}\{c\}^* \cdot \mathcal{F}\{b\} + \rho(\mathcal{F}\{d_x\}^* \cdot \mathcal{F}\{\nu_1\} + \mathcal{F}\{d_y\}^* \cdot \mathcal{F}\{\nu_2\})}{\mathcal{F}\{c\}^* \cdot \mathcal{F}\{c\} + \rho(\mathcal{F}\{d_x\}^* \cdot \mathcal{F}\{d_x\} + \mathcal{F}\{d_y\}^* \cdot \mathcal{F}\{d_y\})} \right\}
\end{split}
$$

여기서 $$\mathcal{F}^{-1} \left\{ \frac{\mathcal{F}\{c\}^* \cdot \mathcal{F}\{b\}}{\mathcal{F}\{c\}^* \cdot \mathcal{F}\{c\} + \rho(\mathcal{F}\{d_x\}^* \cdot \mathcal{F}\{d_x\} + \mathcal{F}\{d_y\}^* \cdot \mathcal{F}\{d_y\})} \right\}$$ 은 고정된 값이기 때문에 알고리즘 진행 중 최초 1번만 연산하면 되므로 연산량이 현저히 줄어든다.


## 4.4. Update $\mathbf{z}$
$$\mathbf{z}$$ 를 갱신하기 위해 원소별 soft thresholding 연산자 $$\mathcal{S}_K (\cdot)$$ 를 도입하는데, 이는 다음과 같다.

$$
\begin{split}
    prox_{\Vert \cdot \Vert_{1}, \rho} (\nu) &= \mathcal{S}_K (\nu) \\
    &= \begin{cases} \nu - k & \text{if} \quad \nu > k \\ 
    0 & \text{if} \quad \vert \nu \vert \le k \\
    \nu + k & \text{if} \quad \nu < -k \\ \end{cases} \\
    &= (\nu - k)_+ - (-\nu - k)_+ 
\end{split}
$$

여기서의 soft thresholding 은 anisotropic TV 에 대한 proximal operator 이며 $$k = \lambda / \rho, \; \nu = \mathbf{Dx + u}$$ 이다.



# 5. ADMM for Image Deconvolution with Denoiser
## 5.1. Deconvolution Problem

ADMM 을 사용하는 것과 유사하게 식을 구성하며, 이는 다음과 같다.

$$
\begin{split}
    L_{\rho} (\mathbf{x},\mathbf{z},\mathbf{u}) = f(\mathbf{x}) + g(\mathbf{z}) + \frac{\rho}{2} \Vert \mathbf{Kx-z+u} \Vert^2 + \frac{\rho}{2} \Vert \mathbf{u} \Vert^2
\end{split}
$$

이 때, $$\mathbf{K} = \mathbf{I}, \; \nu \in \mathbb{R}^{N}$$ 로 설정한다.


## 5.2. Update $\mathbf{x}$
TV 를 사용하는 경우와 유사하지만 더 간단한 형태로 $$\mathbf{x}$$ 를 갱신한다.

$$
\begin{split}
    \mathbf{x} \leftarrow (\mathbf{C}^T \mathbf{C} + \rho \mathbf{I})^{-1} (\mathbf{C}^T \mathbf{b} + \rho \nu)
\end{split}
$$

따라서 FFT 를 이용한 갱신은 다음과 같다.

$$
\begin{split}
    prox_{\Vert \cdot \Vert_{2}, \rho} (\mathbf{x}) = \mathcal{F}^{-1} \left\{ \frac{\mathcal{F}\{c\}^* \cdot \mathcal{F}\{b\} + \rho \mathcal{F} \{\nu\}}{\mathcal{F}\{c\}^* \cdot \mathcal{F}\{c\} + \rho} \right\}
\end{split}
$$


## 5.3. Update $\mathbf{z}$

$$\mathbf{z}$$ 를 갱신하는 방법은 다음과 같다.

$$
\begin{split}
    \mathbf{z} &\leftarrow prox_{\mathcal{D}, \rho} (\nu) \\
    &= argmin_{\mathbf{z}} \lambda \Psi(\mathbf{z}) + \frac{\rho}{2} \Vert \nu-\mathbf{z} \Vert_{2}^2, \nu = \mathbf{x + u} \\
    &= argmin_{\mathbf{z}} \Psi(\mathbf{z}) + \frac{\rho}{2 \lambda} \Vert \nu-\mathbf{z} \Vert_{2}^2
\end{split}
$$

이 형태는 Gaussian noise 가 추가된 이미지의 최적화 문제를 해결하는 과정, 즉, denosing 과정과 동일한 것이다. 따라서 $$\mathbf{z}$$ 를 갱신하는 과정 중 denoiser $$\Psi(\mathbf{z})$$ 를 상황에 맞게 선택하여 사용이 가능하다. 그리고 다양한 상황은 고전적인 알고리즘뿐만이 아니라 CNN, BM3 와 같은 딥러닝 모델들을 사용 가능하다는 것을 의미한다.
