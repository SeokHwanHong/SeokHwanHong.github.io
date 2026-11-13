---
layout: single
title: "Kafka 수업 정리 ③: 복제·커밋 순서·멱등성으로 장애 이해하기"
date: 2026-11-11 19:00:00 +0900
permalink: /sk-encore-de-2/kafka/kafka-replication-delivery/
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

## 저장됐다는 말의 범위를 나누기

STEP 10~11은 브로커 복제와 컨슈머 처리 순서에 의도적으로 장애를 넣는다. 같은 ‘성공’이라도 Kafka에 기록된 것, DB 처리가 끝난 것, 소비 위치가 커밋된 것은 다르다. 11월 11일은 해당 자료의 복습 정리 날짜다.

이 파트의 장애 수치와 트랜잭션 출력은 **제공된 정답 자료의 예시**를 기준으로 설명한다.

## 1. 복제 실습은 세 브로커에서

PowerShell의 `C:\kafkacamp`에서 단일 브로커를 멈춘 뒤 복제용 구성을 켠다. 둘은 호스트 9092를 함께 쓰므로 동시에 켜지 않는다.

```powershell
docker compose -f kafka1.yml stop
docker compose -f kafka3.yml up -d
docker ps
docker exec -it kafka-1 bash
```

세 컨테이너는 `kafka-1`, `kafka-2`, `kafka-3`이며 호스트 접속 포트는 각각 9092·9093·9094다. 컨테이너 간 명령은 **내부 listener인 `kafka-1:19092`**를 쓴다. 다른 브로커를 localhost의 호스트 공개 포트로 찾게 하면 컨테이너 자신의 주소를 보게 될 수 있다.

```bash
./kafka-topics.sh --bootstrap-server kafka-1:19092 --create --topic tags_rf3 --partitions 3 --replication-factor 3 --config min.insync.replicas=2
./kafka-topics.sh --bootstrap-server kafka-1:19092 --create --topic strict3 --partitions 1 --replication-factor 3 --config min.insync.replicas=3
./kafka-topics.sh --bootstrap-server kafka-1:19092 --describe --topic tags_rf3
```

| 항목 | 의미 |
| --- | --- |
| Leader | 해당 파티션의 읽기·쓰기 요청을 담당하는 리더 |
| Replicas | 복제본이 배치된 브로커 목록 |
| ISR | 현재 충분히 따라오고 있는 복제본 집합 |
| replication factor | 배치할 복제본 수 |
| min.insync.replicas | acks=all 쓰기의 최소 ISR 조건 |

파티션은 작업·로그를 나누는 것이고 복제본은 같은 파티션을 복사하는 것이다. 서로 다른 두 축이다.

## 2. 브로커 하나를 멈추면

복제 실험에서는 `acks=all`을 명시해 확인 조건을 분명히 한다. 아래는 kafka-1의 Bash다.

```bash
./kafka-console-producer.sh --bootstrap-server kafka-1:19092 --topic tags_rf3 --command-property acks=all
```

몇 줄을 보낸 뒤 별도 PowerShell에서 두 번째 브로커를 멈춘다.

```powershell
docker stop kafka-2
```

자료의 예시는 ISR이 세 개에서 두 개로 줄고 필요하면 리더가 바뀌는 모습이다. 복제본 세 개를 설계했다고 항상 세 개가 정상 동기화돼 있는 것은 아니다.

ISR이 두 개면 min ISR 2인 `tags_rf3`는 조건을 만족할 수 있다. 반면 min ISR 3인 `strict3`의 acks=all 쓰기는 조건을 충족하지 못해 `NOT_ENOUGH_REPLICAS` 등의 오류·재시도를 보일 수 있다. acks=all은 고정된 모든 복제본이 항상 살아 있다는 뜻이 아니라 현재 ISR과 최소 조건을 함께 보는 설정이다.

`acks=1`로 조건을 우회해 쓰는 것과 장애 상황에서도 같은 내구성을 확보하는 것은 다르다. 다음 PowerShell 명령으로 복구한 후 ISR과 쓰기 성공 여부를 다시 확인한다.

```powershell
docker start kafka-2
```

세 브로커 실습을 마쳤다면 컨테이너 셸을 종료하고, 자료의 절차대로 복제용 컨테이너를 정리한 뒤 단일 브로커를 시작한다. 이 정리에서 복제 실습 데이터는 사라질 수 있다.

## 3. 처리와 커밋 중 무엇이 먼저인가

STEP 11은 단일 브로커와 `tags` 토픽으로 돌아온다. tags에는 하루 파일 2,008행을 한 번 보냈다는 전제다. 리스트를 가짜 DB로 두고 550개 처리 뒤 장애를 발생시킨다.

| 순서 | 장애가 끼어드는 지점 | 재시작 시 위험 |
| --- | --- | --- |
| 커밋 → 처리 | 커밋했지만 처리하기 전 | 처리하지 않은 레코드를 건너뜀 |
| 처리 → 커밋 | 처리했지만 커밋하기 전 | 같은 레코드를 다시 처리 |
| 처리 → 커밋 + 키 기반 upsert | 같은 레코드의 재처리 | 같은 논리적 결과를 유지하도록 구성 |

아래는 처리 후 커밋의 핵심을 보여주는 축약 코드다. `c`는 `enable.auto.commit=False`인 Consumer, `m`은 오류 없이 받은 메시지, `db`는 딕셔너리라는 전제다.

```python
import json

rec = json.loads(m.value())
db[rec["event_id"]] = rec
c.commit(message=m, asynchronous=False)
```

`commit(message=m)`은 해당 파티션에서 m의 다음 위치를 커밋한다. 커밋 성공은 DB 트랜잭션 성공과 자동으로 묶이지 않는다. 여러 파티션을 처리할 때는 파티션별로 처리 완료 범위를 관리해야 한다.

## 4. 장애 전후의 누락·중복 해석하기

자료의 출력 예시는 다음과 같다. 정확한 재처리 개수는 소비 순서·커밋 진행 등에 따라 달라질 수 있어 검증할 때는 키별 내용도 확인한다.

| 자료의 실험 | 예시 결과 | 읽을 점 |
| --- | --- | --- |
| 먼저 커밋 후 처리 중 장애 | 리스트 2,007행 | 한 레코드의 처리 누락 |
| 처리 후 100개마다 커밋, 장애·재시작 | 리스트 2,058행 | 재처리 50행 + 원래 재전송 포함 |
| event_id 딕셔너리로 처리 | 고유 이벤트 1,988개 | 같은 키를 다시 반영해도 항목 수 유지 |

업무 이벤트가 이미 20행 중복된 입력이므로 2,008행을 전부 처리했다고 2,008개의 고유 결과를 만들어야 하는 것은 아니다. 반대로 결과 행 수가 같다고 모든 값이 정확하다고 결론 내릴 수도 없다.

딕셔너리는 멱등 결과를 설명하는 메모리 모형이다. 실제 서비스에서는 영속 저장소의 고유 키·upsert, 처리와 커밋의 실패 정책을 함께 구현해야 한다. 같은 event_id의 내용이 충돌한다면 무조건 마지막으로 받은 값을 덮는 규칙도 따로 검토한다.

## 5. 멱등 프로듀서와 트랜잭션

```python
from confluent_kafka import Producer

p = Producer({"bootstrap.servers": "localhost:9092", "enable.idempotence": True})
p.produce("hello", value="sent once, even if retried")
p.flush()
```

멱등 프로듀서는 전송 재시도에서 같은 요청이 중복 기록되는 문제를 줄이는 장치다. 사용자가 같은 이벤트를 새 produce 요청으로 두 번 보내는 것까지 업무 키로 제거하지는 않는다. 컨슈머가 외부 DB에 같은 값을 두 번 적용하는 문제도 별도다.

자료의 `tx_demo`는 `transactional.id="tx-11"`을 설정하고 세 레코드를 전송·flush한 뒤 abort한다. `read_committed` 소비자는 취소된 트랜잭션 레코드를 읽지 않고, `read_uncommitted`는 브로커에 기록된 취소 레코드를 볼 수 있다. 예시 출력은 각각 0개·3개다.

이때 격리 수준은 클라이언트 기본값에 기대지 않고 `isolation.level`로 명시해서 비교한다. Kafka 트랜잭션을 배웠다고 Spark의 Kafka sink나 외부 DB 쓰기까지 자동으로 하나의 트랜잭션이 된다고 해석하지 않는다.

## 정리

복제는 브로커 장애에 대한 저장·가용성 문제, 소비 커밋은 읽기 진행 문제, 멱등 쓰기는 중복 적용 문제를 다룬다. ‘정확히 한 번’을 말하려면 어느 구간과 결과에 대한 설명인지부터 정해야 한다. 마지막 파트에서는 이 구분을 Kafka–Spark–Delta 파이프라인으로 연결한다.

작성 근거: `kafka3.yml`, STEP 10~11의 lab·answer. 실패·재시작 출력은 자료의 예시이며 별도 기동 결과가 아니다.

참고: [confluent-kafka Python API](https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html), [librdkafka 설정](https://github.com/confluentinc/librdkafka/blob/master/CONFIGURATION.md).


## 함께 읽기

- [Kafka 수업 정리 ①: 토픽·파티션·컨슈머 그룹으로 로그 읽기](/sk-encore-de-2/kafka/kafka-topics-partitions-groups/)
- [Kafka 수업 정리 ②: Python 송수신·키 분배와 이벤트 재생](/sk-encore-de-2/kafka/kafka-python-keys-replay/)
- [Kafka 수업 정리 ④: Spark 시간 창·Delta 파이프라인과 튜닝](/sk-encore-de-2/kafka/kafka-spark-delta-pipeline/)
