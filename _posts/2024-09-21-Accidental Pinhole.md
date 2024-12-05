---
layout: single        # 문서 형식
title: Accidental pinhole and Pinspeck Cameras # 제목
categories: Computer Vision # 카테고리
tag: [CV, Lecture]
author_profile: false # 홈페이지 프로필이 다른 페이지에도 뜨는지 여부
sidebar:              # 페이지 왼쪽에 카테고리 지정
    nav: "docs"       # sidebar의 주소 지정
#search: false # 블로그 내 검색 비활성화
use_math: true
---
# Keywords
pinhole, anti-pinhole(inverse pinhole), pinspeck camera


# 1. Introduction
## 1.1. Pinhole
<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/pinhole/pinhole.jpg" height = 250>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 1 : Pinhole ]</figcaption>
</figure>

핀홀이란 종이에 핀을 뚫은 구멍처럼 매우 작은 구멍을 뜻한다. 핀홀 카메라는 렌즈를 사용하지않고 작은 구멍을 통해 빛을 받아들여 촬영하는 사진기이다. 맺히는 상은 기존 물체에 대해 거꾸로 되어있으며 상이 맺히기까지 많은 빛을 요구하므로 많은 시간이 소요된다.

## 1.2. Anti-Pinhole
<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/pinhole/anti.jpg" height = 250>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 2 : Anti-Pinhole ]</figcaption>
</figure>
anti-pinhole은 핀홀과 정반대로 빛이 진행하는 방향에 대해 특정 위치에만 그림자가 지도록 물체를 두는 방법이다. 본 논문에서는 이 두 개념을 이용해 자연 상황에서 핀홀카메라와 유사하게 이미지를 만들어내는 실험을 하고자 한다.

# 2. Pinhole and Pinspeck Cameras 
## 2.1. Pinhole Camera

#### - Notations

$S(x)$ : 핀홀카메라로 촬영된 이미지
$T(x)$ : 특정 지점 $x$ 를 통과하는 빛의 양을 나타내는 함수(apeture function), $0 \le T(x) \le 1$
$I(x) = T(x) * S(x)$ : 식을 변환해 만든 이미지

#### - Room-size Example

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/pinhole/figure3.jpg" height = 500>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 3 : Example of Pinhole Cameras ]</figcaption>
</figure>

figure 3-a 의 환경과 같이 방에 창문을 열어 일정량의 빛이 들어오도록 구성한다. 그리고 빛의 양을 조절하기 위해 각각 낮(b,c)과 밤(d,e)에 실험을 진행한다. 밤에 핀홀카메라로 촬영된 이미지 $S(x)$ 는 작은 delta 함수(적은 양의 빛)로도 만들어낼 수 있다는 것이다. 즉, $I(x)$ 는 창문에 비치는 풍경에 어느정도 중첩되어 상이 나타나는 것처럼 보인다. 반대로 낮에 촬영된 이미지는 뿌옇게 blurring이 발생한 것을 알 수 있다. 이 예제에서 우리는 깔끔한 $I(x)$ 를 얻기 위해서는 $T(x)$ 가 적당히 작아야하는 것을 알 수 있다.

## 2.2. Pinspeck Camera
<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/pinhole/figure4.jpg" height = 150>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 4 : Pinspeck Camera ]</figcaption>
</figure>

1.2 의 anit-pinhole 의 원리를 이용한 카메라다. figure 4-a 와 b 에서 중첩되는 부분을 제외한 부분이 c인데, 이 c에 투영되는 이미지를 촬영하는 카메라이다. 이를 식으로 표현하면 다음과 같다.

$$
I_{window}(x) - I_{occludedwindow}(x) = T_{hole}(x) * S(x)
$$


#### - Room-size Example

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/pinhole/figure5.jpg" height = 350>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 5 : Example of Pinspeck Camera ]</figcaption>
</figure>

2.1 의 예제와 유사하게 방을 구성 후 사람을 추가해 anti-pinhole 역할을 하도록 한다. 

<figure style="text-align: center; display: inline-block; width: 100%;">
    <img src = "/images/pinhole/figure6.jpg" height = 350>    
    <figcaption style="display: block; width: 100%; text-align: center;">[ Figure 6 : result of Example ]</figcaption>
</figure>

실제 촬영된 이미지는 figure 6-a이고, 이를 뒤집으면 6-b 를 확인할 수 있다. 6-b의 가운데를 확인하면 실제 풍경과 유사한 이미지를 얻을 수 있다는 것을 확인할 수 있다.

## 2.4. Limitations
1. 원하는 이미지를 추출하기 위해서는 최소 2장의 이미지나 비디오가 필요
2. 핀홀 카메라들은 충분한 빛을 모으기 위해서는 일정 시간 동안 빛에 노출시켜야 하는데, anti-pinhole을 이용한 카메라는 가리개로 인해 빛이 막히기 때문에 노이즈 비율에 대한 신호(signal Nosie ratio, SNR)가 추가적으로 필요

#### - Possion Noise
Possion 잡음이란 빛의 강도의 제곱근값을 취한 것을 의미한다.
구멍(핀홀)의 크기(영역) $A = \int T(x) \mathrm{d}x$ 에 대해 가려지지 않은 사진의 SNR $\propto \sqrt{A_{window}}$, 가려진 사진의 SNR $\propto A_{occluder}$ 이며 이에 대한 잡음은 $\sqrt{A_{window}}$ 와 비례한다. 따라서 우연히 찍힌 이미지의 SNR은 원본 이미지의 $\frac{A_{occluder}}{A_{window}}$ 비율로 감소한다. 여기서 잡음의 특정 부분들은 이전에 언급했던 한계점들로 인해 SNR이 감소하므로, 실험환경에서 작은 창문을 사용하거나 방에 작은 빛만 들어오도록 구성하면 된다.



# 참고
https://roytravel.tistory.com/133
https://link.springer.com/article/10.1007/s11263-014-0697-5