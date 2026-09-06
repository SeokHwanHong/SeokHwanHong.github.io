---
title: "2025 Inha AI Challenge: 색채화·초해상도"
layout: single
permalink: /projects/inha-colorization-super-resolution/
author_profile: false
toc: true
toc_sticky: true
---

## 문제

저해상도 흑백 이미지를 자연스러운 색상과 높은 해상도로 복원하는 이미지 생성 프로젝트다.

## 접근

- L-CAD로 512×512 이미지를 256×256 색채화 결과로 생성했다.
- SwinIR을 연결해 256×256 결과를 512×512로 업스케일링하고 세부 화질을 보완했다.
- 색채화와 초해상도를 분리한 2단계 파이프라인으로 결과를 생성했다.

## 구현

재현 가능한 추론 순서를 정리했다.

1. L-CAD로 색채화
2. SwinIR로 초해상도 복원
3. 복원 이미지를 제출 형식으로 변환

## 배운 점

생성 품질은 단일 모델의 결과만으로 판단하기보다, 전체 파이프라인에서 단계별 출력이 다음 단계에 어떤 영향을 주는지 함께 봐야 한다는 점을 확인했다.
