---
layout: single
title: "Up & Down Sampling"
categories:
  - "Deep Learning"
author_profile: true
use_math: true
toc: true
toc_sticky: true
---

## 1. Definition : Down Sampling
인코딩 진행 시 데이터의 수를 줄이는 과정, 또는 고해상도 데이터를 저해상도로 변환하는 과정(이미지 처리)이다.

## 2. Effects of Down Sampling
1. 연산량 감소
2. 노이즈 제거

## 3. Type of Down Sampling
**Decimation**

균일한 간격으로 픽셀을 제거해 해상도를 감소하는 방법이다. 간단하지만 Aliasing 문제 발생 가능이 존재한다.

**Gaussian blur and subsampling **

Gaussian blur를 적용 후 일정 간격으로 픽셀을 선택해 요약하는 방법이다. 노이즈 변형 대처에 효과적이다.

**Pooling**

한 영역 내에서 평균, 최대값 등을 계산해 데이터를 요약하는 방법이다.

**Atrous(Dilated) Convolution**

커널의 크기를 유지하면서 합성곱 연산을 수행하는 기법이다. 기존 합성곱과 동일한 파라미터 수와 계산량을 유지하면서도 receptive field는 증가하므로 semgentation 성능 역시 증가한다.

## 4. Definition : Up Sampling
Down Sampling과 반대로 디코딩 진행 시 데이터를 복원하기 위해 데이터 차원을 늘리거나 해상도를 증가시키는 과정이다.







## 참고

https://dacon.io/forum/408203
https://wikidocs.net/147019
