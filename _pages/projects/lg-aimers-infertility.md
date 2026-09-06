---
title: "LG Aimers 6기: 난임 환자 임신 성공 예측"
layout: single
permalink: /projects/lg-aimers-infertility/
author_profile: false
toc: true
toc_sticky: true
---

## 문제

난임 시술 데이터에서 환자의 임신 성공 여부를 예측하는 프로젝트다. 수치형·범주형·결측치가 함께 있는 정형 데이터를 다뤘다.

## 접근

- 변수의 의미와 값 범위를 먼저 확인해 수치형과 범주형 데이터를 재구성했다.
- CatBoost와 LightGBM을 중심으로 모델을 비교했다.
- 각 모델의 예측을 앙상블해 일반화 성능을 높였다.

## 결과와 배운 점

최종 앙상블로 상위 3% 성과를 기록했다. 모델을 복잡하게 만드는 것보다 데이터의 의미와 변수 특성에 맞는 전처리·모델 선택이 중요하다는 점을 배웠다.
