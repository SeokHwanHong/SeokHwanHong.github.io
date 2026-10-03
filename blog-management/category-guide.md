# 카테고리 운영 기준

## 기본 원칙

- 게시글마다 대표 Category를 하나만 지정합니다.
- Tag는 사용하지 않습니다.
- 세부 기술명은 제목과 본문에서 설명하고, Category를 지나치게 세분화하지 않습니다.
- Category의 이름, 설명과 표시 순서는 `_data/categories.yml`에서 관리합니다.

## 상위 Category

| Category | 기준 |
|---|---|
| Data Science | 통계, 확률, 수학적 도구와 데이터 분석 기반 |
| Machine Learning | 분류, 회귀, 군집화 등 전통적인 머신러닝 |
| Deep Learning | 신경망 기초, 생성 모델과 범용 딥러닝 구조 |
| Computer Vision | 이미지 인식, 복원, 초해상도 등 영상 분야 |
| Programming | 언어, 자료구조, 알고리즘과 구현 기록 |
| Data Engineering | 데이터베이스, 파이프라인, 분산 처리와 운영 |
| SK Encore DE 2기 | 부트캠프 수업, 회고와 프로젝트 기록 |

## SK Encore DE 2기 하위 분류

- `회고록`: 학습 과정과 경험에 대한 회고
- `수업 내용`: 수업에서 다룬 개념과 실습 정리
- `프로젝트`: 프로젝트의 문제, 설계, 구현과 결과

```yaml
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
```

다른 상위 Category에는 `subcategory`를 사용하지 않습니다.
