---
title: "GCN 구현: 희소 라벨 환경의 반지도 노드 분류"
layout: single
permalink: /projects/gcn-semi-supervised-classification/
author_profile: false
toc: true
toc_sticky: true
use_math: true
---

[GitHub 코드와 실행 안내](https://github.com/SeokHwanHong/Projects/tree/main/gcn-semi-supervised-classification)

GCN 논문을 리뷰하고 전파 규칙을 PyTorch로 구현해 Facebook 그래프에 적용했다. 페이지의 속성 대신 연결 구조에서 입력 특징을 만들고, 일부 라벨로 모델을 학습했다. **구조 특징 구성부터 GCN 구현, 학습 결과 확인까지 직접 수행한 과정**을 정리한다.

## 1. 연결 구조를 분류에 활용하기

GEMSEC Facebook 데이터에서 노드는 페이지, 간선은 상호 “좋아요” 관계다. Artist, Athletes, Company, Government, New Sites, Politician, Public Figure, TV Show의 8개 범주별 네트워크로 구성돼 있다.

프로젝트의 목표는 GCN 논문의 전파 규칙을 이해하고 실제 데이터에 적용하는 것이었다. GCN은 노드의 특징과 이웃의 정보를 함께 사용한다. 일부 노드에만 학습 라벨을 부여한 상태에서도 연결을 통해 정보를 전달할 수 있다는 점이 이번 구현의 핵심이었다.

먼저 범주별 CSV의 간선을 결합하고, 각 파일의 카테고리를 해당 파일에 등장하는 노드의 라벨로 부여했다. 제공된 ID와 간선을 그대로 입력에 사용했다.

## 2. 노드 속성 대신 구조 특징 만들기

입력 데이터에는 바로 사용할 페이지 속성이 없었다. 그래서 연결에서 얻을 수 있는 두 값을 특징으로 구성했다.

$$
X_{\mathrm{raw}}=[d,\ \hat{A}d]
$$

첫 번째 값은 입력 간선의 첫 번째 노드 열에서 집계한 차수다. 두 번째 값은 self-loop를 포함한 정규화 인접행렬로 차수를 집계한 값이다. 각 노드의 연결 수와 주변 연결 구조를 두 값으로 표현했다.

두 특징은 `Linear(2, 128)`로 투영했다. 투영 가중치는 초기화한 뒤 고정하고, 이 128차원 표현을 GCN 입력으로 사용했다.

## 3. 전파 규칙을 PyTorch로 구현하기

논문에서 설명하는 층별 전파 규칙은 다음과 같다.

$$
\hat{A}=\tilde{D}^{-\frac12}(A+I)\tilde{D}^{-\frac12},
\qquad
H^{(l+1)}=\sigma(\hat{A}H^{(l)}W^{(l)})
$$

코드에서는 모든 노드에 self-loop를 추가하고, 정규화 계수를 계산해 희소 COO 텐서를 만들었다. 각 층은 특징에 가중치를 곱한 뒤 `torch.spmm`으로 연결에 따른 집계를 수행한다. 희소 행렬 연산을 사용해 전체 인접행렬을 밀집 행렬로 만들지 않고 계산했다.

모델은 `128 → 32 → 8`의 2층 GCN으로 구성했다. 첫 층 뒤에 ReLU와 dropout을 적용하고, 마지막 층은 8개 범주에 대한 log-softmax 값을 출력한다.

| 구성 | 설정 |
| --- | --- |
| 입력 | 구조 특징 2개를 고정 128차원으로 투영 |
| GCN | `128 → 32 → 8` |
| 활성화·정규화 | ReLU, dropout 0.5 |
| 최적화 | Adam, learning rate 0.001, weight decay 5×10⁻⁴ |
| 학습 소스 | 300 epoch, 시드 42 |

## 4. 일부 라벨로 학습하고 결과 살펴보기

클래스별로 학습·검증 표본을 추출해 마스크를 구성했다. 모델은 전체 그래프의 특징과 연결을 사용해 출력을 만들고, 학습 마스크에 속한 노드에서만 NLL 손실을 계산한다. 검증 마스크의 정확도와 손실은 매 epoch 기록했다.

<figure class="project-figure">
  <img src="{{ '/images/projects/gcn/training-loss.png' | relative_url }}" alt="GCN 학습·검증 손실 곡선" loading="lazy">
  <figcaption>학습 이력 CSV로 그린 손실 곡선. 학습이 진행되면서 손실이 감소했다.</figcaption>
</figure>

<figure class="project-figure">
  <img src="{{ '/images/projects/gcn/training-accuracy.png' | relative_url }}" alt="GCN 학습·검증 정확도 곡선" loading="lazy">
  <figcaption>학습·검증 정확도 변화. 학습 후반에는 검증 정확도의 변화 폭이 줄었다.</figcaption>
</figure>

학습 노트북의 마지막 저장 로그는 Train Accuracy 0.6130, Validation Accuracy 0.6815다. 손실과 정확도 곡선을 함께 보며 학습 진행에 따른 변화를 살펴봤다.

학습 중 저장한 체크포인트를 불러와 최종 log-softmax 출력에도 t-SNE를 적용했다. 테스트 마스크에서 최대 5,000개를 추출하고, perplexity 30·반복 수 1,000·random state 42로 2차원에 투영했다.

## 5. 한계와 다음 단계

학습 곡선은 확인했지만 기준 모델 비교까지 진행하지는 않았다. 따라서 같은 구조 특징을 사용하는 단순 모델에 비해 GCN의 추가 집계가 얼마나 도움이 됐는지는 남은 질문이다. 전체 정확도에 더해 카테고리별로 어떤 오류가 발생했는지도 살펴볼 필요가 있다.

다음에는 데이터와 분할을 확인한 뒤, 같은 특징 행렬과 분할에서 MLP와 GCN을 비교하려 한다. Accuracy·Macro-F1과 클래스별 재현율을 함께 보면 전체 성능과 범주별 차이를 확인할 수 있다. 입력 특징에 이미 이웃 정보가 포함돼 있으므로 이 비교는 GCN의 추가 집계 효과를 살펴보는 실험이다. 아직 수행하지 않은 후속 계획이다.

이번 학습은 전체 그래프 연결을 공유하는 transductive 설정이며, 제시한 정확도는 과거 검증 로그다. 학습·검증 정확도는 dropout과 가중치 갱신 시점의 조건이 다르다.

자료를 대조하면서 표본 수의 기록 불일치와 t-SNE 범례의 라벨 순서 오류도 확인했다. 기존 t-SNE 그림은 본문에서 제외했으며, 그래프 ID·간선 구성과 추론 입력 복원에 대한 세부 점검 사항은 [GitHub의 확인 사항](https://github.com/SeokHwanHong/Projects/tree/main/gcn-semi-supervised-classification#한계와-후속-비교)에 모았다.

## 코드와 참고자료

- [학습·시각화 노트북과 실행 안내](https://github.com/SeokHwanHong/Projects/tree/main/gcn-semi-supervised-classification)
- [Kipf·Welling, GCN 논문](https://arxiv.org/abs/1609.02907)
- [GEMSEC Facebook 데이터](https://snap.stanford.edu/data/gemsec-Facebook.html)

발표자료의 Citation Network·NELL 비교표는 원논문 결과이며, 이 글의 적용 과정은 Facebook 실험을 다룬다. 이번 글 수정에서는 학습·추론을 재실행하지 않았다.
