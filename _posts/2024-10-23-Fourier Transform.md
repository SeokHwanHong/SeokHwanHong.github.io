---
layout: single        # 문서 형식
title: Fourier Transform # 제목
categories: Electronics    # 카테고리
tag: [CV, Mathematics, Electronics]
author_profile: false # 홈페이지 프로필이 다른 페이지에도 뜨는지 여부
sidebar:              # 페이지 왼쪽에 카테고리 지정
    nav: "docs"       # sidebar의 주소 지정
#search: false # 블로그 내 검색 비활성화
use_math: true
---
# Keywords
Euler's Formula, Fourier Series, Kernel, Filter


# 1. Definition
Fourier 변환은 시간 또는 공간에 따른 연속적인 신호를 주파수 영역으로 변환을 의미한다. 이를 통해 신포에 포함된 다양한 주파수 성분의 크기와 위상을 알 수 있다.



# 2. Fourier Transform
## 2.1. Euler's Formula
Fourier 변환에 앞서 삼각함수를 분해하기 위해 Euler 공식을 사용한다

$$
e^{ix} = \cos{x} + i \sin{x}
$$

여기서 $e$ 는 자연로그의 밑, $i$ 는 허수 단위를 의미한다. 

## 2.2. Fourier Transform
Fourier 변환은 연속이고 미분 가능한 함수 $f$ 에 대해 다음이 성립한다.

$$
f(x) = \int_{-\infin}^{\infin} \hat{f(\xi)} e^{2 \pi i \xi x} d\xi \quad \Longleftrightarrow \quad \hat{f}(x) = \int_{-\infin}^{\infin} f(\xi) e^{2 \pi i \xi x} d\xi
$$

$f(x)$ 를 2차원공간(이미지 공간)으로 확장하면 다음과 같이 표현할 수 있다.

$$
f(x,y) = \int_{-\infin}^{\infin} F(k_x, k_y) e^{2 \pi i (k_x x + k_y y)} dk_x dk_y
$$

여기서 Euler 공식을 이용하면 다음과 같이 표현할 수 있다.

$$
\begin{split}
F(k_x, k_y) &= A e^{j \phi} \\
e^{2 \pi j (k_x x + k_y y)} &= \cos(2\pi [k_x x + k_y y]) + j \sin(2\pi [k_x x + k_y y]) \\
\end{split}
$$

따라서 $f(x,y)$ 를 다음과 같이 정리할 수 있다.

$$
f(x,y) = \int_{-\infin}^{\infin} A\cos(2\pi [k_x x + k_y y] + \phi) + jA \sin(2\pi [k_x x + k_y y] + \phi)
$$

즉, 실제 신호의 Fourier 계수들은 켤례 대칭(conjugate symmetric)이다. 

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/FT/image1.jpg" height = 250>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 1 : Fourier Transform Image & Original Image ]</figcaption>
</figure>

변환한 이미지와 원본 이미지의 예시는 Figure 1 에서 확인할 수 있다.


## 2.3. Convoltuion Theorem
임의의 커널 $g$ 에 대해 다음이 성립한다.

$$
x * g = F^{-1} \{F\{x\} \times F\{g\}\}
$$

즉, 2차원 공간에서의 합성곱이 Fourier 변환을 적용한 후 주파수 공간에서는 곱셈으로 변환되는 것을 알 수 있다.

## Proof of 2.3

1. 2차원 합성곱

$$
(f*g)(x,y) = \int_{-\infin}^{\infin} \int_{-\infin}^{\infin} f(x^{\prime}, y^{\prime}) g(x - x^{\prime}, y - y^{\prime}) dx^{\prime} dy^{\prime}
$$

2. 2차원 푸리에 변환

$$
\begin{split}
F(u,v) &= \mathcal{F}\{f(x,y)\} = \int_{-\infin}^{\infin} \int_{-\infin}^{\infin} f(x,y) \exp[-j \cdot 2 \pi(ux+vy)] dx dy \\
G(u,v) &= \mathcal{F}\{f(x,y)\} = \int_{-\infin}^{\infin} \int_{-\infin}^{\infin} g(x,y) \exp[-j \cdot 2 \pi(ux+vy)] dx dy \\ 
\end{split}
$$

3. 합성곱의 푸리에 변환

합성곱의 푸리에 변환은 다음과 같다.

$$
\mathcal{F} \{ (f*g)(x,y) \} = \int_{-\infin}^{\infin} \int_{-\infin}^{\infin} (f*g)(x,y) \exp[-j \cdot 2 \pi(ux+vy)] dx dy
$$

여기에서 $(f*g)(x,y)$ 을 합성곱의 정의에 따라 대입하면 다음과 같다.

$$
\mathcal{F} \{ (f*g)(x,y) \} = \int_{-\infin}^{\infin} \int_{-\infin}^{\infin} \left[\int_{-\infin}^{\infin} \int_{-\infin}^{\infin} f(x^{\prime}, y^{\prime}) g(x - x^{\prime}, y - y^{\prime}) dx^{\prime} dy^{\prime} \right] \exp[-j \cdot 2 \pi(ux+vy)] dx dy
$$

$x^{\prime}$ 과 $y^{\prime}$ 에 대해 적분 후 $x$ 와 $y$ 에 대해 적분을 수행하기 위해 적분 순서를 변경하면 다음과 같다.

$$
= \int_{-\infin}^{\infin} \int_{-\infin}^{\infin} f(x^{\prime}, y^{\prime})\left[\int_{-\infin}^{\infin} \int_{-\infin}^{\infin}  g(x - x^{\prime}, y - y^{\prime}) \exp [-j \cdot 2 \pi(ux+vy) ] dx dy \right]  dx^{\prime} dy^{\prime}
$$

내부 적분에서 $u$ 와 $v$ 에 대한 항을 정리하기 위해 다음과 같이 치환해 적분을 변환할 수 있다.

$$
x^{\prime \prime} = x - x^{\prime}, y^{\prime \prime} = y - y^{\prime}
$$

$$
= \int_{-\infin}^{\infin} \int_{-\infin}^{\infin} f(x^{\prime}, y^{\prime}) \left[ \int_{-\infin}^{\infin} \int_{-\infin}^{\infin} g(x^{\prime \prime}, y^{\prime \prime}) \exp[-j \cdot 2 \pi(u(x^{\prime} + x^{\prime\prime}) + v(x^{\prime} + x^{\prime\prime}))] dx^{\prime \prime}  dy^{\prime \prime} \right] dx^{\prime} dy^{\prime}
$$

여기서 푸리에 변환의 성질을 적용해 외부 적분을 $f(x^{\prime}, y^{\prime})$ 에 대한 푸리에 변환으로 표현하면, 각각의 푸리에 변환의 곱으로 표현된다. 즉,

$$
= F(u,v) \cdot G(u,v) \qquad \Box
$$




# 3. Discrete FT
## 3.1. Why Discrete?
1. 컴퓨터에서 작동하기 위해서는 값을 이산형으로 표현해야함
2. 주기함수를 입력할 경우 FT를 적용한 결과는 이산적인 값을 가지는 분포
3. 주기함수와 특정 신호값의 합성곱을 입력할 경우 경우 결과값은 분포가 겹치는 주기함수
4. 따라서 Fourier 공간에서 이산값을 표현하기 위해서는 주기를 가지는 이산값을 입력해야함

## 3.2. Discrete FT
#### - Notation
$\hat{x}[k]$ : 주파수 영역에서의 k 번째 주파수 성분
$x[n]$ : 시간 영역 (실제 영역)에서의 이산 신호
$X$ : 시간 영역의 이미지 (원본 이미지)
$N$ : 샘플 총 개수

#### - Algorithm
이산 신호를 이용해 이미지를 처리하는 알고리즘은 다음과 같다.

$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} \hat{x}[k] \exp[2i \pi n / N] \Longleftrightarrow \hat{x}[k] = \frac{1}{N} \sum_{k=0}^{N-1} x[n] \exp[2i \pi n / N]
$$

실제 영역과 주파수 영역을 번갈아가며 거치며 알고리즘이 진행된다. 다만, 이는 시간복잡도가 $\mathcal{O}(N^2)$ 으로 연산량이 많아 실제 이미지에 적용하기에는 난해하다.

## 3.3. Fast FT
그래서 DFT의 단점을 상쇄하기 위해 Cooley & Tukey (1965) 는 시간복잡도를 $\mathcal{O}(N \log N)$ 줄인 Fast Fourier Transform 을 발표한다.

#### - Algorithm
시간 복잡도를 줄이기 위한 트릭은 다음과 같다.

$$
\begin{split}
X_{N+k} &= \sum_{n=0}^{N-1} x_n \cdot \exp [-2i\pi (N+k) n / N] \\
&= \sum_{n=0}^{N-1} x_n \cdot \exp [-2i\pi n -2i\pi k n / N] \\
&= \sum_{n=0}^{N-1} x_n \cdot \exp [-2i\pi k n / N] \\
(&\because \sum_{n=0}^{N-1} x_n \cdot \exp [-2i\pi n] \: are\: Twiddle \: Factor) 
\end{split}
$$

주파수를 짝수와 홀수에 따라 나누고 트릭을 $X_k$ 에 적용하면 다음과 같다.

$$
\begin{split}
X_k &= \sum_{n=0}^{N-1} x_n \cdot \exp [-2i\pi k n / N] \\
&= \sum_{m=0}^{N/2-1} x_{2m} \cdot \exp [-2i\pi k (2m) / N] + \sum_{m=0}^{N/2-1} x_{2m+1} \cdot \exp [-2i\pi k (2m+1) / N] \\
&= \sum_{m=0}^{N/2-1} x_{2m} \cdot \exp \left[-2i\pi k \frac{m}{N/2} \right] + \exp [-2i\pi k/N] \sum_{m=0}^{N/2-1} x_{2m+1} \cdot \exp \left[-2i\pi k \frac{m}{N/2} \right] \\
\end{split}
$$

따라서 DFT 의 연산량을 감소함으로써 더 빠르게 Fourier 변환을 할 수 있다.





# 4. Filtering
## 4.1. Definition
Filter 란 신호에서 잡음과 같이 원치 않은 성분을 제거하거나 특정 주파수 성분을 강조하는데 사용된다. 원본영역(시간영역)에서 필터를 설계할 수 이씨만, Fourier 변환을 이용하면 주파수영역(신호영역)에서 간단한 곱셉으로 필터링을 적용할 수 있다. 주파수영역에서 필터링을 다음과 같이 표현할 수 있다.

$$
Y(f) = X(f) \cdot H(f)
$$

$X(f)$ : 입력 신호의 Fourier 변환
$H(f)$ : 필터의 주파수 응답 함수 
$Y(f)$ : 필터가 적용된 후의 신호

이 식은 필터의 주파수 응답 $H(f)$ 을 입력 신호의 FT $X(f)$에 곱해 주파수 성분을 제어하는 방식으로 동작한다.
또한, 원본영역에서의 LPF 를 다


## 4.2. Low-pass Filter
Low-pass Filter(LPF, 저역 통과 필터) 는 낮은 주파수 성분을 통과시키고 높은 주파수 성분을 억제하는 필터다. 주로 신호에서 잡음을 제거하거나 부드럽게 처리하는데 사용된다. 원본영역에서 합성곱 필터링은 다음과 같이 표현할 수 있다.

$$
b = x * c
$$

$x$ : 원본 신호
$c$ : 합성곱 필터 e.g. Point Spread Function (PSF) 
$b$ : 필터가 적용된 후의 이미지 (blur 이미지)

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/FT/image2.jpg" height = 250>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 2 : Low-pass Filter in Primal Domain ]</figcaption>
</figure>

#### - Gaussian Filter
Gaussian filter 는 가우시안 함수를 기반으로 필터링이며 이를 주파수영역에서 확인하면 다음과 같다.

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/FT/image3.jpg" height = 250>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 3 : Gaussian Filter ]</figcaption>
</figure>

위 그림에서 확인 가능하듯이, 저주파 성분은 통과시키고 고주파 성분은 급격하게 차단되지 않고 점진적으로 감소한다. 따라서 이미지가 부드럽고 자연스러운 흐름을 유지한다. 하지만 파라미터 값 조정에 따라 이미지가 지나치게 부드러워져 중요한 디테일들이 사라질 수 있다는 단점이 있다.

#### - Hard Cutoff Filter
Hard Cutoff Filter (HCF) 는 주파수 도메인에서 임계 주파수를 기준으로 모든 고주파 성분을 완전히 차단하고, 저주파 성분만 통과시키는 필터이다. 이를 주파수 영역에서 확인하면 다음과 같다.

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/FT/image4.jpg" height = 250>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 4 : Hard Cutoff Filter ]</figcaption>
</figure>

위 그림에서 확인 가능하듯이, 정확히 설정된 임계 주파수에서 신호를 필터링하기 때문에 불필요한 주파수 성분을 확실하게 제거할 수 있다. 하지만 이로 인해 부자연스러운 경계가 형성되거나 이미지가 과도하게 처리된 것처럼 보일 수 있다는 단점도 있다.


## 4.3. High-pass Filter

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/FT/image5.jpg" height = 250>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 5 : Hard Cutoff Filter in Primal Domain ]</figcaption>
</figure>

High-pass Filter (HCF, 고역 통과 필터) 는 주파수영역에서 고주파 성분을 갑자기 차단하면서 이미지 가장자리에 불필요한 진동 패턴이 발생하는 링잉 현상이 발생한다. 이를 해결하기 위해 고주파만 처리하는 High-Pass Filter 를 적용한다.

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/FT/image6.jpg" height = 250>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 6 : Sharpening with High-pass Filter ]</figcaption>
</figure>

이는 주로 엣지 검출이나 이미지 샤프팅 등 이미지의 선명도를 높이기 위한 작업에 사용한다. 


## 4.4. Edge Detection Filter
#### - Band-pass Filter
Band-pass Filter (BPS, 대역 통과 필터) 는 LPF 와 HPF 에서 아이디어를 얻어, 특정 주파수만 검출해 원본영역에서는 엣지만 검출하도록 구성한 필터이다. 이를 그림에서 확인하면 다음과 같다.

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/FT/image7.jpg" height = 250>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 7 : Band-pass Filter ]</figcaption>
</figure>

#### - Oriented Band-pass Filter
Oriented Band-pass Filter (OBPS, 방향성 대역 통과 필터) 는 Band-pass Filter에서 특정 방향의 성분까지 고려해 이들만 통과하는 필터다. 이를 그림에서 확인하면 다음과 같다.

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/FT/image8.jpg" height = 250>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 8 : Oriented Band-pass Filter ]</figcaption>
</figure>




# 5. Anti-Aliasing
## 5.1. Aliasing
Aliasing 은 샘플링 속도가 신호에서 가장 높은 주파수 성분(최대 주파수) 의 두 배 이상이어야 고주파 성분이 표현되지만, 그렇지 못하는 경우 고주파 성분이 저주파 성분으로 왜곡되어 나타나는 현상을 의미한다. 주로 샘플링이 불충분하거나 고주파 성분이 충분히 억제되지 않은 것이 원인인데, 특히 이미지에서는 선명한 경계나 디테일한 패턴이 낮은 해상도로 렌더링될 때 불필요한 왜곡이 발생한다. 

## 5.2. Anti-Aliasing
Anti-Aliasing (AA) 은 이러한 왜곡들을 줄이거나 없애기 위한 기법이다. 이에 관해서는 다양한 기법들이 존재한다.

#### - Sampling Based
이미지에서 가장 기본적인 방법은 샘플링 해상도를 증가하는 것이다. 이를 super-sampling AA (SSAA) 라고 하며, 픽셀보다 더 많은 샘플을 사용해 경계에서 발생하는 왜곡을 감소시킨다.

#### - Multi-sampling Based
Multi-sampling AA (MSAA) 는 계산 비용이 매우 높은 SSAA 의 단점을 보완하며, 경계 영역에서만 추가 샘플링을 수행하는 방식이다. 

#### - Post Processing Based
Fast Approximate AA (FXAA) 와 같은 기술은 렌더링 후 이미지를 후처리해 AA 를 적용하는 방식이다. 경계를 부드럽게 처리해 Aliasing 을 감소하지만, 세밀한 부분에서는 효과가 떨어질 수 있다.
 



# 6. Devonvolution
일반적으로 이미지는 다음과 같은 과정으로 생성된다.

$$
B = I * K
$$

$I$ : 원본 이미지
$K$ : kernel, filter e.g. 카메라 렌즈
$B$ : Blur 된 이미지, 실제로 촬영된 이미지

위 식은 다음과 같은 FT 로 표현이 가능하다.

$$
F(B) = F(I) \cdot F(K) 
$$

이 때, deconvolution 을 적용해 원본 이미지를 복원하는 식은 다음과 같다.

$$
F(\hat{I}) = F(B) \ F(K)
$$

따라서 위 식에 역 FT를 적용하면 원본 이미지를 복원할 수 있다.

$$
\hat{I} = F^{-1} (F(B) / F(K))
$$

## 6.1. Naïve Deconvolution
이상적인 상황에서는 $\hat{I}$ 를 쉽게 복원이 가능하지만, 실제로는 이미지를 얻는 과정에서 다양한 노이즈가 추가된다. 

$$
B = K \ast I + N
$$

여기서 $N$ 은 노이즈를 의미한다. 이러한 노이즈들을 처리하기 위해 역 가우시안 필터 등을 적용한 Naïve Deconvolution 은 다음과 같다.

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/FT/image9.jpg" height = 250>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 9 : Naïve Deconvolution ]</figcaption>
</figure>


## 6.2. Wiener Deconvolution
Wiener Deconvolution 은 블러링이나 왜곡이 발생한 신호 또는 이미지에서 원래의 신호를 추정한다. 이는 노이즈와 신호의 통계적 특성을 고려해 보다 정확한 복원 결과를 나타낸다. 수식으로 표현하면 다음과 같다.

$$
\hat{I} = F^{-1} \left(\frac{|F(K)|^2}{|F(K)|^2 + 1/SNR(\omega)} \cdot \frac{F(B)}{F(K)} \right)
$$

여기서 $SNR(\omega) = \frac{Var_{signal}(\omega)}{Var_{noise}(\omega)}$ 로 신호의 분산과 노이즈의 분산 간 비율을 의미한다.

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/FT/image10.jpg" height = 250>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 10 : Naïve & Wiener Deconvolution ]</figcaption>
</figure>

#### - Derivation of 6.2.
1. Sensing Model : $b = k * i + n$
여기서 n은 평균이고 신호 $i$ 와 독립

2. Fourier Transform : $B = K \cdot I + N$

3. 목표 : 다음을 만족하는 함수 $H(\omega)$ 를 탐색

$$
\min_H \mathbb{E}[||I - HB||^2]
$$

즉, Fourier 영역에서 오차의 기댓값을 최소로 하는 함수를 탐색

4. 2를 3의 식에 대입
$$
\begin{split}
&\min_H \mathbb{E} [||I - H(K \cdot I + N)||^2] \\
&= \min_H \mathbb{E} [||(1 - HK)I - HN||^2] \\
&= \min_H ||1 - HK||^2 \: \mathbb{E} [||I||^2] - 2H(1-HK) \: \mathbb{E}[IN] + ||H||^2 \: \mathbb{E} [||N||^2]\\
\end{split}
$$

이 때, I와 N은 독립이므로 $\mathbb{E}[IN] = \mathbb{E}[I] \mathbb{E}[N]$ 인데, $\mathbb{E}[N] = 0 $ 이므로  $\mathbb{E}[IN] = 0$ 이다. 따라서 다음과 같이 정리할 수 있다.

$$
\begin{split}
&= \min_H ||1 - HK||^2 \: \mathbb{E} [||I||^2] + ||H||^2 \: \mathbb{E} [||N||^2] \\
& = L
\end{split}
$$

$L$ 을 $H$ 에 대해 미분한 값을 0으로 설정해 확인하면,

$$
\frac{\partial L}{\partial H} = 0 \\
\Longrightarrow -2K(1-HK)\mathbb{E}[||I||^2] + 2H\mathbb{E}[||N||^2] = 0 \\
\Longrightarrow H = \frac{K \mathbb{E}[||I||^2]}{K^2 \mathbb{E}[||I||^2] + \mathbb{E}[||N||^2]} = \frac{1}{K + \frac{\mathbb{E}[||N||^2]}{\mathbb{E}[||I||^2]}} \qquad \Box
$$
