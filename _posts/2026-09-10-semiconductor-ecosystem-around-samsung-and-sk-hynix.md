---
layout: single
title: "삼성전자·SK하이닉스를 중심으로 보는 반도체 산업 생태계"
categories:
  - "Semiconductor Industry"
author_profile: true
toc: true
toc_sticky: true
---

지금까지 삼성전자와 SK하이닉스의 역사, 기술력, 방향성을 살펴봤다. 두 회사는 한국을 대표하는 반도체 기업이지만, 실제 반도체 산업은 한 기업이 모든 것을 혼자 수행하는 구조가 아니다.

반도체 하나가 시장에 나오기까지는 설계, 제조, 장비, 소재, 패키징, 테스트, 최종 수요기업이 서로 연결된 거대한 생태계가 필요하다.

이 글에서는 삼성전자와 SK하이닉스를 중심에 두고 **반도체 산업이 어떤 기업과 역할로 연결되어 있는지**를 간단히 정리한다.

## 1. 반도체 산업은 하나의 긴 가치사슬이다

전체 흐름을 단순화하면 다음과 같다.

```text
칩 설계
↓
Wafer 제조
↓
반도체 공정
↓
Packaging / Test
↓
Server·Smartphone·Automotive 등 최종 제품
```

하지만 실제 산업에서는 이 과정마다 전문기업들이 존재한다.

```text
EDA / IP
↓
Fabless
↓
Foundry 또는 IDM
↓
장비·소재 기업
↓
Packaging / Test
↓
Cloud·Mobile·Automotive 등 고객
```

삼성전자와 SK하이닉스는 이 가운데 **반도체 제조를 직접 수행하는 IDM(Integrated Device Manufacturer)**에 해당한다.

## 2. 삼성전자는 종합 반도체 IDM에 가깝다

삼성전자는 반도체 생태계의 여러 단계를 내부에 가지고 있다.

```text
Samsung Electronics

├ Memory
│  ├ DRAM
│  ├ HBM
│  └ NAND
│
├ System LSI
│  └ Logic 설계
│
├ Foundry
│  └ 타사 반도체 위탁생산
│
└ Advanced Packaging
```

즉 삼성은 메모리를 직접 설계·생산하면서 동시에 Logic 반도체도 설계하고, 다른 회사가 설계한 칩을 Foundry에서 생산할 수도 있다.

AI 시대에는 HBM, Logic, Chiplet, Advanced Packaging의 연결이 중요해지고 있기 때문에 삼성은 이 폭넓은 사업구조를 하나의 통합 솔루션으로 묶으려 하고 있다.

## 3. SK하이닉스는 메모리에 깊게 특화된 IDM이다

SK하이닉스는 삼성보다 사업범위가 좁지만 메모리 분야에 훨씬 집중되어 있다.

```text
SK hynix

├ DRAM
│  ├ HBM
│  ├ DDR
│  ├ LPDDR
│  └ SOCAMM
│
├ NAND
│
├ Solidigm
│  └ Enterprise SSD
│
└ Advanced Memory
   ├ HBF
   └ Custom HBM
```

자체 Foundry나 CPU·GPU 사업은 없지만, 대신 AI 시스템에서 발생하는 Memory Wall을 해결하기 위해 HBM부터 DRAM, SSD, 새로운 Memory Tier까지 영역을 확장하고 있다.

즉 삼성전자가 반도체 기술의 **폭**을 넓히는 구조라면, SK하이닉스는 메모리 기술의 **깊이**를 늘리는 구조에 가깝다.

## 4. 두 회사 밖에도 반드시 필요한 기업들이 있다

삼성과 SK하이닉스가 반도체를 직접 제조한다고 해도 모든 장비와 기술을 자체적으로 만들 수는 없다.

### 반도체 장비

반도체 공정에는 다양한 장비가 필요하다.

대표적으로

- 노광
- 식각
- 증착
- 세정
- 이온주입
- CMP
- 검사·계측

등의 공정마다 전문 장비기업이 존재한다.

예를 들어 첨단 노광공정에서는 ASML의 EUV 장비가 중요하고, 식각·증착·검사 분야에서도 여러 글로벌 장비업체가 삼성전자와 SK하이닉스의 생산라인에 장비를 공급한다.

### 소재

공정에는

- Silicon Wafer
- Photoresist
- Gas
- Chemical
- Metal
- Packaging Material

등 수많은 소재가 필요하다.

공정이 미세해질수록 소재의 순도와 균일성이 수율에 직접 영향을 미친다.

### EDA와 IP

Logic 반도체를 설계할 때는 EDA 소프트웨어와 반도체 IP가 필요하다.

특히 삼성 Foundry처럼 외부 고객의 Logic Chip을 생산하는 사업에서는 설계 생태계와 Foundry 공정이 긴밀하게 연결돼야 한다.

## 5. AI 시대에는 고객과 메모리 회사의 관계도 바뀐다

과거 메모리 시장에서는 비교적 표준화된 DRAM이나 NAND를 제조해 여러 고객에게 공급하는 방식이 일반적이었다.

하지만 HBM과 AI Accelerator가 중요해지면서 관계가 바뀌고 있다.

```text
과거
Memory 업체
↓
표준 제품
↓
고객

현재·미래
AI Accelerator 업체
       ↕
Memory / Foundry / Packaging 업체
       ↓
공동설계
```

GPU와 AI ASIC의 구조에 따라 필요한 HBM 용량, 대역폭, 전력, Base Die, Package 구성이 달라지기 때문에 고객과 반도체 기업의 공동개발이 점점 중요해지고 있다.

삼성전자와 SK하이닉스 모두 결국 **부품 공급업체에서 시스템 공동설계 파트너로 이동하려는 이유**가 여기에 있다.

## 6. 반도체 산업의 경쟁은 결국 공정 경쟁이다

기업 전략을 이해하면 마지막에는 결국 같은 질문으로 돌아온다.

> **실제로 반도체는 어떻게 만들어지는가?**

DRAM, NAND, HBM, Logic Chip 모두 결국 Wafer 위에서 수많은 공정을 반복해 만들어진다.

대표적으로

```text
Wafer
↓
산화
↓
Photo
↓
Etch
↓
Deposition
↓
Ion Implantation
↓
Metal / Interconnect
↓
Test / Packaging
```

와 같은 공정들이 반복된다.

그리고 기업 간 경쟁력도 실제 생산단계에서는

- 공정 미세화
- 공정 균일성
- Defect 감소
- 수율
- 생산성
- 원가
- 장비 안정성

으로 나타난다.

따라서 삼성전자와 SK하이닉스의 기술력을 제대로 이해하려면 이제 기업 분석에서 한 단계 더 내려가 **반도체 제조공정 자체를 이해해야 한다.**

## 7. 지금까지의 흐름과 다음 단계

지금까지의 공부 흐름은 다음과 같다.

```text
반도체 산업의 역사
↓
반도체 산업의 최근 트렌드
↓
SK하이닉스의 역사·기술·전략
↓
삼성전자의 역사·기술·전략
↓
삼성전자 vs SK하이닉스
↓
반도체 산업 생태계
↓
반도체 제조공정
```

기업을 통해 **누가 무엇을 만드는지**를 이해했다면, 이제부터는

> **그 제품이 실제 Fab 안에서 어떤 과정을 거쳐 만들어지는지**

를 공부할 차례다.

다음 글부터는 반도체의 **8대 공정**을 중심으로 각 공정의 역할과 연결관계를 정리한다.
