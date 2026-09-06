---
title: "GCN 구현: 희소 라벨 환경의 반지도 노드 분류"
layout: single
permalink: /projects/gcn-semi-supervised-classification/
author_profile: false
toc: true
toc_sticky: true
---

Kipf와 Welling의 *Semi-Supervised Classification with Graph Convolutional Networks*를 읽고, GEMSEC Facebook 페이지 네트워크에 적용해 본 프로젝트다. 논문 식을 재현하는 데서 멈추지 않고, 노드 속성이 없는 대규모 그래프에서 특징을 만들고 학습 곡선과 임베딩까지 확인했다.

## 프로젝트 한눈에 보기

| 항목 | 내용 |
| --- | --- |
| 문제 | 일부 라벨만 있는 그래프에서 Facebook 페이지의 8개 카테고리 분류 |
| 데이터 | GEMSEC Facebook 네트워크, 134,833개 노드 |
| 특징 | 노드 차수와 이웃 평균 차수로 만든 Local Degree Profile → 128차원 투영 |
| 모델 | 2-layer GCN (`128 → 32 → 8`) |
| 학습 | Adam, lr=0.001, weight decay=5e-4, dropout=0.5, 300 epoch |
| 확인 결과 | 학습·검증 곡선 수렴, t-SNE로 클래스별 임베딩 분리 확인 |

## 1. 라벨이 부족할 때, 연결 자체를 정보로 쓰기

일반적인 표 형태 분류에서는 각 행의 속성으로 클래스를 예측한다. 이 프로젝트의 노드는 Facebook 페이지이고, 간선은 상호 좋아요 관계다. 라벨이 없는 노드가 많지만 연결 구조 안에는 "비슷한 페이지가 연결될 가능성"이라는 정보가 남아 있다.

데이터는 Artist, Athletes, Company, Government, New Sites, Politician, Public Figure, TV Show의 8개 카테고리로 이뤄졌다. 특히 Artist가 50,515개 노드인 반면 TV Show는 3,892개로 클래스 크기 차이가 컸다. 단순 무작위 분할은 다수 클래스에 유리한 모델을 만들 가능성이 있었다.

## 2. 그래프 합성곱을 실제 코드로 옮기기

GCN의 핵심은 자기 자신과 이웃의 정보를 함께 평균내는 정규화 인접행렬이다.

$$
\hat{A} = \tilde{D}^{-\frac{1}{2}}(A + I)\tilde{D}^{-\frac{1}{2}},
\qquad
H^{(l+1)} = \sigma(\hat{A}H^{(l)}W^{(l)})
$$

코드에서는 모든 노드에 self-loop를 추가한 뒤 sparse COO tensor로 `\hat{A}`를 만들었다. 그래프 전체를 dense 행렬로 바꾸지 않고 `torch.spmm`으로 이웃 정보를 모으도록 구현했다.

이 데이터에는 바로 쓸 수 있는 노드 속성이 없었다. 그래서 각 노드의 차수와 정규화 인접행렬을 이용해 얻은 **이웃 평균 차수**를 Local Degree Profile로 만들고, 이를 선형층으로 128차원 특징으로 확장했다. 그 위에 ReLU와 dropout을 포함한 2-layer GCN을 쌓아 8개 클래스를 예측했다.

## 3. 샘플링에서 만난 과적합 문제

처음에는 클래스별 표본 수 차이를 반영한 stratified sampling을 생각했다. 그러나 발표 자료와 실험 메모에서 이 방식은 다수 클래스 편향과 과적합을 만들 수 있음을 확인했다. 그래서 각 클래스에서 학습용 500개, 검증용 500개를 동일하게 뽑아 총 4,000개 학습·4,000개 검증 노드를 구성했다. 나머지 노드는 테스트에 사용했다.

| 구성 | 설정 |
| --- | --- |
| 인접행렬 | self-loop 추가 후 대칭 정규화 |
| 특징 | 차수 + 이웃 평균 차수 → 128차원 |
| 은닉층 | 32차원, ReLU, dropout 0.5 |
| 최적화 | Adam, learning rate 0.001, weight decay 5×10⁻⁴ |
| 학습 | 최대 300 epoch, 시드 42 고정 |

<figure>
  <img src="{{ '/images/projects/gcn/training-loss.png' | relative_url }}" alt="GCN 학습과 검증 손실 곡선" loading="lazy">
  <figcaption>학습·검증 손실은 초기 약 20 epoch에서 빠르게 감소하고, 약 50 epoch 이후 수렴 경향을 보였다.</figcaption>
</figure>

<figure>
  <img src="{{ '/images/projects/gcn/training-accuracy.png' | relative_url }}" alt="GCN 학습과 검증 정확도 곡선" loading="lazy">
  <figcaption>정확도는 학습과 함께 상승했으며, 검증 정확도는 약 0.67 수준에서 안정됐다.</figcaption>
</figure>

## 4. 정확도만으로는 부족했던 검증

수치가 오르더라도 각 카테고리가 실제로 분리되는지는 별도 확인이 필요했다. 저장한 180 epoch 체크포인트의 출력(logit)을 사용해 테스트 노드 중 5,000개를 표본으로 뽑고, t-SNE(perplexity 30, 1,000 iteration)로 2차원에 투영했다.

<figure>
  <img src="{{ '/images/projects/gcn/tsne-epoch-150.png' | relative_url }}" alt="GCN 노드 임베딩 t-SNE 시각화" loading="lazy">
  <figcaption>학습이 진행된 뒤 클래스별 임베딩이 이전 epoch보다 더 분리되는 양상을 확인했다.</figcaption>
</figure>

초기 epoch와 비교하면 학습이 진행될수록 클래스별 군집이 더 분리됐다. 다만 150 epoch 이후에는 그래프 모양의 변화가 크지 않은데도 정확도는 조금씩 변할 수 있었다. 이 경험을 통해 단일 정확도뿐 아니라 손실 곡선, 검증 성능, 임베딩 구조를 함께 봐야 한다는 점을 확인했다.

## 5. 한계와 다음 단계

이 구현은 full-batch gradient descent를 사용한다. 따라서 그래프가 더 크고 조밀해지면 메모리 요구량이 노드·간선 수에 따라 빠르게 커진다. 또한 이웃과 self-loop를 같은 비중으로 두는 기본 가정이 모든 그래프에 맞는 것도 아니다.

다음 단계에서는 GraphSAGE나 neighbor sampling으로 대규모 그래프에 확장하고, 노드 차수만이 아니라 텍스트·메타데이터 같은 실제 속성을 추가해 구조 정보와 속성 정보가 각각 얼마나 기여하는지 비교해 보고 싶다.

> 본문은 GCN 논문 리뷰 발표 자료와 직접 작성한 학습·추론 노트북을 기준으로 정리했다.
