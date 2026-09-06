---
title: "GCN을 이용한 반지도 노드 분류"
layout: single
permalink: /projects/gcn-semi-supervised-classification/
author_profile: false
toc: true
toc_sticky: true
---

## 문제

라벨이 일부만 주어진 GEMSEC Facebook 그래프 데이터에서 노드의 클래스를 예측하는 반지도 분류 문제를 다뤘다.

## 접근

- 노드 간 연결 정보를 edge index 형태로 구성했다.
- Graph Convolution Network로 이웃 노드의 정보를 함께 학습했다.
- 학습·추론 노트북을 분리해 실험 과정과 결과 확인 흐름을 정리했다.

## 배운 점

표 형태 데이터와 달리 그래프 데이터에서는 개별 노드의 속성뿐 아니라 연결 구조가 예측에 직접 활용된다는 점을 이해했다. 라벨이 적은 상황에서 그래프 구조를 활용하는 반지도 학습의 장점도 함께 확인했다.
