---
layout: single
title: "Kafka 수업 정리 ②: Python 송수신·키 분배와 이벤트 재생"
date: 2026-11-10 19:00:00 +0900
permalink: /sk-encore-de-2/kafka/kafka-python-keys-replay/
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

## 파일의 이벤트를 메시지로 전달하기

STEP 06~09는 콘솔에서 하던 송수신을 Python으로 바꾸고, 같은 로그를 다른 키로 분배하거나 시간 간격을 재현하는 과정이다. 11월 10일은 자료를 재구성한 복습 정리 날짜다.

실행 전제는 `C:\kafkacamp`, `kafka1.yml`의 단일 브로커, 제공된 가상환경과 데이터다. Python 코드는 Windows VS Code Interactive 환경을 기준으로 한다. `lab`의 작성 내용과 `answer`를 비교해 설명용 코드의 오타·누락을 보완했다.

## 1. 토픽 준비와 비동기 전송

새 실습 상태에서 `mini` 토픽을 만든다. 이미 존재한다면 자동 삭제하지 말고 토픽 상태와 이전 전송 여부를 먼저 확인한다.

```python
import os
import json
from confluent_kafka import Producer, Consumer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic

os.chdir(r"C:\kafkacamp")
BOOT = "localhost:9092"
admin = AdminClient({"bootstrap.servers": BOOT})
admin.create_topics([NewTopic("mini", num_partitions=3, replication_factor=1)])["mini"].result()
```

관리 API의 반환 future에서 `.result()`로 생성 완료나 오류를 확인한다. 이후 프로듀서로 파일을 보낸다.

```python
errors = []
def report(err, msg):
    if err is not None:
        errors.append(str(err))
    else:
        print(msg.partition(), msg.offset())

p = Producer({"bootstrap.servers": BOOT})
with open("data/tags_mini.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        p.produce("mini", key=rec["card_id"], value=line.strip(), on_delivery=report)
        p.poll(0)
remaining = p.flush(10)
if remaining or errors:
    raise RuntimeError({"remaining": remaining, "errors": errors})
```

`produce()`는 로컬 전송 대기열에 요청을 넣는 비동기 API다. `poll()`·`flush()`에서 전달 콜백을 처리하며, 성공·실패와 제한 시간 뒤 남은 요청을 확인한다. ‘동기는 자동, 비동기는 수동’이라는 뜻이 아니다. 요청 완료를 기다리는 방식의 차이다.

자료는 첫 행을 먼저 보낸 뒤 나머지 11행을 보낸다. 이 글은 처음부터 12행을 한 번 보내는 형태로 축약했다. 둘을 연달아 실행하면 첫 이벤트가 추가 전송된다. 파일 자체에도 M07 재전송이 있어 12행이지만 고유 이벤트는 11개다.

## 2. poll·처리·commit 구분하기

같은 `mini`를 새 실습 그룹으로 읽는다. 아래 `g07m`에 기존 커밋이 있으면 그 위치부터 재개한다.

```python
c = Consumer({
    "bootstrap.servers": BOOT,
    "group.id": "g07m",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
})
c.subscribe(["mini"])
rows, empty = [], 0
try:
    while empty < 5:
        m = c.poll(1.0)
        if m is None:
            empty += 1
            continue
        if m.error():
            raise RuntimeError(m.error())
        empty = 0
        rec = json.loads(m.value())
        rows.append(rec)
        # 이 예제의 처리 결과는 rows에 보관한다.
        c.commit(message=m, asynchronous=False)
finally:
    c.close()
print("읽은 행:", len(rows), "고유 이벤트:", len({r["event_id"] for r in rows}))
```

`poll()`은 메시지 또는 None·오류를 반환할 수 있다. 가져온 후 처리하고, 성공한 범위의 다음 오프셋을 커밋한다. 파티션의 오프셋 4를 처리했다면 커밋 값은 보통 5다. 이 예제의 리스트는 프로세스 메모리이므로 실제 DB의 내구성을 구현한 것은 아니다.

빈 poll 5회를 종료 조건으로 쓴 것은 작은 실습을 끝내기 위한 방법이다. 토픽의 영구적인 끝이나 앞으로 새 데이터가 없음을 증명하는 조건은 아니다.

정답 예시의 12행·11개는 새 그룹이 파일을 한 번 보낸 토픽을 전부 읽을 때의 기준이다. 같은 그룹으로 재개했을 때 0행이 나왔다고 데이터가 삭제됐다는 뜻은 아니다. 커밋하지 않았을 때도 기존 처리 결과 자체가 휘발되는 것이 아니라, 다음 읽기에서 같은 레코드를 다시 받을 수 있다.

## 3. 키를 바꾸면 분배도 달라진다

자료는 `by_card`, `by_station`에 각각 카드 번호와 역 코드를 키로 2,008행을 보내 비교한다. 토픽은 세 파티션으로 생성하고, 다시 여섯 파티션의 별도 토픽과 비교한다.

```python
p = Producer({"bootstrap.servers": BOOT, "partitioner": "murmur2_random"})
# by_card와 by_station은 STEP 08에서 3개 파티션으로 준비한 토픽이다.
with open("data/tags_day.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        p.produce("by_card", key=rec["card_id"], value=line.strip())
        p.produce("by_station", key=rec["station_id"], value=line.strip())
p.flush()
```

`murmur2_random`은 키가 있을 때 Java 계열과 맞는 Murmur2 기반 분배를 사용하기 위한 자료의 설정이다. 서로 다른 클라이언트에서 같은 키를 보내도 파티셔너가 다르면 결과가 달라질 수 있다. JSON 안의 card_id가 같더라도 Kafka key를 따로 넣지 않으면 키 분배 조건이 성립하지 않는다.

| 키 선택 | 유지하려는 관계 | 살펴볼 점 |
| --- | --- | --- |
| 카드 번호 | 같은 카드의 이벤트 흐름 | 카드별 활동량과 파티션별 분포 |
| 역 코드 | 같은 역의 이벤트 흐름 | 적은 종류의 키·쏠림·빈 파티션 |

파티션 수를 늘렸다고 키 종류와 빈도 차이가 사라지지는 않는다. lab에는 여섯 파티션 비교에서 역 기준은 빈 공간이 생긴다는 메모가 남아 있다. 개별 측정 수치는 저장돼 있지 않아, 분포를 관찰하려 한 기록으로 활용한다.

## 4. 오프셋 범위로 입력 분포 확인하기

```python
c = Consumer({"bootstrap.servers": BOOT, "group.id": "counter"})
try:
    for part in range(3):
        lo, hi = c.get_watermark_offsets(TopicPartition("by_card", part))
        print(part, lo, hi, hi - lo)
finally:
    c.close()
```

이번처럼 삭제·압축·트랜잭션이 없는 새 실습 토픽에서는 `hi-lo`를 레코드 수 검산에 사용할 수 있다. 일반적인 Kafka 로그에서는 오프셋 구멍·제어 레코드 등으로 실제 소비 가능한 메시지 수와 다를 수 있다. 이를 항상 정확한 메시지 count라고 부르지 않는다.

실습 코드에 있는 옵션 이름은 `group.id`다. `group_id`로 바꾸면 confluent-kafka의 설정 키와 맞지 않는다.

## 5. 수신 시간 간격을 줄여 재생하기

완성된 `tools/replay.py`는 JSONL 한 줄을 한 메시지로 보내며 키를 card_id로 지정한다. `speed`가 있으면 이전 행과 현재 행의 recv_time 차이를 나눈 만큼 기다린다.

```python
# 작업 디렉터리 C:\kafkacamp가 Python 모듈 검색 경로에 있어야 한다.
from tools.replay import send_file, send_part, counts

# tags는 새로 만든 3개 파티션 토픽, tags_live도 별도로 준비한다.
send_file("data/tags_day.jsonl", "tags")
send_part(1, topic="tags_live", speed=600)
print(counts("tags_live"))
```

600초 간격은 speed 600에서 1초의 대기가 된다. 하지만 flush·직렬화·네트워크 비용도 있으므로 전체 실행 시간이 원래 시간의 정확히 1/600인 것은 아니다. 같은 조각을 다시 보내면 새로운 Kafka 오프셋이 생기며 내용은 반복된다.

| 시각 | 의미 |
| --- | --- |
| `tag_time` | 합성 이벤트가 발생한 시각 |
| `recv_time` | 합성 서버에 도착한 시각, 재생 간격의 기준 |
| Kafka `timestamp` | Kafka 레코드의 시각. 브로커 설정에 따라 의미가 달라짐 |

과거 날짜의 이벤트를 지금 재생하면 tag_time과 Kafka timestamp는 다르다. 이벤트 지연과 이번 재생 처리의 지연도 같은 값으로 계산하면 안 된다.

## 정리

송신 요청, 전달 확인, 메시지 소비, 업무 처리와 커밋은 각각 다른 단계다. 키는 같은 흐름을 묶는 기준이고, 오프셋은 저장 위치다. 다음 파트에서는 이 단계 사이에서 장애가 발생하면 어떤 누락·중복이 생기는지 다룬다.

작성 근거: STEP 06~09의 lab·answer, `tools/replay.py`, 첨부 JSONL. 설명용 코드는 오류 처리와 종료 조건을 보완했다. 원본 파일은 변경하지 않았다.

참고: [confluent-kafka Python API](https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html).


## 함께 읽기

- [Kafka 수업 정리 ①: 토픽·파티션·컨슈머 그룹으로 로그 읽기](/sk-encore-de-2/kafka/kafka-topics-partitions-groups/)
- [Kafka 수업 정리 ③: 복제·커밋 순서·멱등성으로 장애 이해하기](/sk-encore-de-2/kafka/kafka-replication-delivery/)
- [Kafka 수업 정리 ④: Spark 시간 창·Delta 파이프라인과 튜닝](/sk-encore-de-2/kafka/kafka-spark-delta-pipeline/)
