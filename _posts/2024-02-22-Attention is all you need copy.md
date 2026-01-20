---
layout: single        # 문서 형식
title: 'Attention is all you need 리뷰'         # 제목
categories: Natural Language Process    # 카테고리
tag: [DL, NLP, Attention]
author_profiel: false # 홈페이지 프로필이 다른 페이지에도 뜨는지 여부
sidebar:              # 페이지 왼쪽에 카테고리 지정
    nav: "nav"       # sidebar의 주소 지정
#search: false # 블로그 내 검색 비활성화
use_math: true
---

# 1. Introduction

#### - Sequence Modeling
순서를 가지는 데이터(sequential data)로부터 또 다른 순서를 가지는 데이터를 생성하는 작업(task)을 수행하는 모델을 의미한다. 기존 순환모델들은 모든 데이터를 한번에 처리하기보단 순서 위치(sequence position)에 따라 순차적으로 입력해야한다.

#### - Trasduction (Transductive Problem) 
학습 시 사전에 미리 train dataset 뿐만 아니라 test dataset도 관측한 상태를 의미한다. test dataset의 label은 미관측 상태지만 학습이 진행되는 동안 train dataset 내 labeled data의 특징이나 데이터 간 연관성, 패턴 등을 공유하거나 전파하는 등 추가적인 정보를 활용함으로써 test dataset의 label을 추론한다. 



# 2. Background
#### - 기존 NLP 연구들에서 CNN 기반 모델들의 특징
CNN 기반 모델들(Extended Neural GPU, ByteNet, ConvS2S 등)은 sequential 계산을 줄이기 위해 블록을 쌓으며 모든 입출력 위치에 대한 숨겨진 표현(hidden representations)을 병렬적으로 계산한다. 이 모델들은 임의의 입력 또는 출력 위치 간 신호를 연결할 때 연산량이 증가해 먼 위치 간의 의존성 학습이 어려워진다. 이런 단점을 보완하기 위해 Transformer를 도입하는데, attention의 가중치가 적용된 position의 평균을 이용하기 때문에 유효 해상도가 낮아지는 단점이 존재한다.

#### - Attention

![Figure1 : Self Attention](/images/attentionisallyouneed/selfattention.jpg){: .align-center width="300"}

Attention Mechanism은 input과 output sequence에서 거리와 관계없이 의존성을 모델링 가능하기 때문에 다양한 작업에서 강력한 sequence modeling 및 transductive model의 필수적인 부분이 되었다. 그래서 본 논문에서 순환과정(recurrence) 대신 input과 output 사이의 전체 구조 내 의존성(global denpendency)를 찾는 attention mechanism만 사용한다. 그리고 이는 더 많은 병렬처리가 가능해 동일 시간 동안 더 많은 연산이 가능하다. 전체 구조를 요약하면 다음과 같다.

1. Position-wise Feed-Forward Networks
2. Embeddings and Softmax
3. Positional Encoding

#### - 장점
1. 층당 전체 연산 수 감소
2. 병렬화 가능 계산
3. 신경망 내 장거리 의존성 간 경로 길이가 짧아져 학습이 용이

#### - Self-Attention
Self-Attention은 input sequence 내에서 서로 관련된 부분들을 찾아 집중하는 방식으로 작동하는 메커니즘이다. 기존 RNN 모델처럼 sequence를 순차적으로 처리하지 않고, 모든 위치 간의 관계를 동시에 고려해 학습하도록 작동한다. Query, Key, Value 의 시작값이 동일하고 자기 자신과의 내적을 통해 (각각에 대한 weight matrix를 곱) 고유의 값을 계산한다.


# 3. Model Architecture

- overall architecture

![Figure2 : Self Attention](/images/attentionisallyouneed/model%20architecture.jpg){: .align-center width="450"}

## 3.1. Attention

#### - Scaled Dot-Product Attention

![Figure3 : Self Attention](/images/attentionisallyouneed/sdpa.jpg){: .align-center width="150"}

#### - Attention sequence
$Attention(Q,K,V) = softmax({Q{K^{T}}/\sqrt{d_v}}) * V$ 

input : queries and keys of dimensions $d_{k}$ (= $d_{q}$), values of $d_{v}$

1. Q와 K의 내적
2. 1번의 결과값을 $\sqrt{d_v}$로 나눠줌으로써 scaling
3. Masking(opt.)
4. 3번의 결과값에 SoftMax 함수 적용
5. 4번의 결과값에 V를 곱함


#### - Multi-Head Attention

![Figure4 : Self Attention](/images/attentionisallyouneed/mha.jpg){: .align-center}


Multi-head Attention(MHA) 는 서로 다른 공간에 있는 정보들을 하나로 병합한다.

\[
MultiHead(Q, K, V) = Concat(head_1,\: ... \:, head_h) W^O \\
\: where \quad head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
\]

여기서 모수 행렬은 각각 $ W_i^Q \in \mathbb{R}^{d_{model} \times d_k}, W_i^K \in \mathbb{R}^{d_{model} \times d_k}, W_i^V \in \mathbb{R}^{d_{model} \times d_v}, W_i^O \in \mathbb{R}^{hd_{v} \times d_model}$ 이다.
$\:$ 본 논문에서는 평행한 attention layer를 $h = 8$, 차원들은 $d_k = d_v = d_{model} / h = 64$ 로 설정한다. 각 head마다 차원량이 감소하기때문에 전체 계산량은 full dimensionality 인 단일 head attention과 유사하다.


## 3.2. Encoder & Decoder Stacks
#### - Notation

$(x_1, x_2, ... , x_n)$ : an input sequence of symbol representations

$ \mathbf{z} = (z_1, z_2, ... , z_n)$ : a sequence of continuous representations

$(y_1, y_2, ... , y_n)$ : an output sequence 

### 3.2.1. overall architecture
#### - Encoder
$N = 6 $의 동일한 레이어로 이루어진 스택을 구성하며, $(x_1, x_2, ... , x_n)$을 $(z_1, z_2, ... , z_n)$로 mapping 한다.
 
#### - Decoder
$\mathbf{z}$가 주어졌을 때, 한번에 하나씩 $(y_1, y_2, ... , y_n)$을 생성한다. 각 단계에서 모델은 autoregressive이며, 이전에 생성한 심볼은 다음 심볼을 생성할 때 추가 입력으로 사용한다.

## 3.3. Position-wise Feed-Forward Networks
각 encoder와 decoder에는 완전 연결 feed-forward network가 있는데 각 위치가 분리되고 동일하도록 구성되어있다. 이는 ReLU 를 포함하는 2개의 선형 결합으로 구성되어있고 다음과 같다.


[\
    FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
]\

이 선형변환은 여러 위치에서 동일하지만 층에서 층으로 이동할 때 다른 모수를 사용한다. 본 논문에서 입력값과 출력값의 차원을 $d_{model} = 512$, 내부 층에서는 $d_{ff} = 2048$ 을 사용한다.

## 3.4. Embddings and Softmax
입력 및 출력 token을 차원이 $d_{model}$ 인 벡터로 변환하기 위해 학습된 embedding을 사용한다. 또한 decoder의 출력값을 next-token 예측값으로 변환하기 위해 학습된 선형변환과 softmax 함수를 사용한다. 본 논문에서는 두 embedding 층 사이에서 동일한 가중치 행렬과 pre-softmax 선형변환을 공유한다. embedding 층에서는 이 가중치들과 $\sqrt{d_{model}}$ 을 곱한다.

## 3.5. Positional Encoding
recurrence와 convolution이 없고 모형이 순서를 인식하기 위해 positional encoding을 추가한다. 이는 enocder와 decoder의 stack의 처음에 추가하는 입력 embedding이고 차원은 다른 embedding들과 동일하게 $d_{model}$ 이다. 본 논문에서는 sine과 cosine 함수를 이용해 다음과 같이 사용한다.
\[ 
    PE_{(pos, 2i)} = \sin (pos / 10000^{2i/d_{model}}) \\
    PE_{(pos, 2i+1)} = \cos (pos / 10000^{2i/d_{model}}) 
\]

여기서 $pos$ 는 위치를, $i$ 는 차원을 의미한다. 
삼각함수를 사용한 이유는 모델이 관련 위치들을 더 잘 학습할 것이라고 가정하기 때문이다. 


# 참고
- attention is all you need

https://brave-greenfrog.tistory.com/19

- sequence model

https://wooono.tistory.com/241
https://dos-tacos.github.io/translation/transductive-learning/
https://jadon.tistory.com/29

- inductive problem

https://velog.io/@kimdyun/Inductive-Transductive-Learning-%EC%B0%A8%EC%9D%B4%EC%A0%90

- self-attention

https://codingopera.tistory.com/43

- BERT

- NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE

- Vision Transformer
