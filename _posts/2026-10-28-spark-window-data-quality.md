---
layout: single
title: "Spark 수업 정리 ③: 윈도 함수와 데이터 정제·이동 단위 구성"
date: 2026-10-28 19:00:00 +0900
permalink: /sk-encore-de-2/spark/spark-window-data-quality/
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

## 이벤트를 연결해 이동 한 건 만들기

STEP 12~13과 최종 문제를 바탕으로 시간 창, 순위, 중복 제거와 격리를 정리한다. 10월 28일은 복습 시리즈의 정리 날짜다.

승차·하차 이벤트를 읽는 것과 이동 시간을 계산하는 것은 다르다. 먼저 이벤트 식별자와 정상 기준을 정한 뒤 같은 이동의 기록을 연결해야 한다.

## 1. 두 종류의 윈도 구분하기

| 구분 | 하는 일 | 예 |
| --- | --- | --- |
| 분석 윈도 `Window` | 행을 유지하며 그룹·순서에 따른 값을 붙임 | 순위, 이전 태그 시각, 누적합 |
| 시간 창 `F.window` | 시각을 구간으로 묶어 집계 | 15분별 승차 수 |

```python
from pyspark.sql import SparkSession, functions as F, Window

spark = SparkSession.builder.master("local[2]").appName("spark-day3").getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "Asia/Seoul")
SCHEMA = """
event_id STRING, card_id STRING, trip_id STRING, tag_type STRING,
station_id STRING, tag_time TIMESTAMP, fare INT, recv_time TIMESTAMP
"""
raw = spark.read.schema(SCHEMA).json("data/tags_day.jsonl")
stations = spark.read.option("header", True).csv("data/stations.csv")

w = Window.partitionBy("card_id").orderBy("tag_time", "event_id")
(raw.withColumn("seq", F.row_number().over(w))
    .withColumn("prev_time", F.lag("tag_time").over(w))
    .select("card_id", "event_id", "seq", "tag_time", "prev_time").show(5))
```

`row_number`는 순번, `rank`는 동점 뒤 순위 건너뛰기, `dense_rank`는 동점 뒤 연속 순위에 사용한다. 결과를 재현하려면 정렬 기준이 동점에서 어떻게 결정되는지도 정한다.

시간 창은 `F.window("tag_time", "15 minutes")`처럼 쓴다. 길이와 이동 간격이 같은 tumbling window는 구간이 겹치지 않는다. 길이 1시간, 간격 30분인 sliding window는 겹치므로 하나의 이벤트가 여러 창에 포함될 수 있다. 이때 창별 건수를 더해 전체 이벤트 수라고 해석하면 안 된다.

## 2. 중복 키는 event_id

```python
dedup = raw.dropDuplicates(["event_id"])
```

`card_id`는 여러 이동에서 재사용되고, `trip_id`는 승차와 하차가 함께 쓰는 키다. 어느 쪽도 태그 이벤트의 중복 제거 키로 적절하지 않다. 첨부 데이터의 재전송은 같은 `event_id`를 반복하므로 해당 키를 사용한다.

다만 같은 ID에 서로 다른 내용이 충돌하는 운영 데이터라면 임의 한 행을 남기는 것으로 충분하지 않다. 수신 시각·버전·원천 우선순위 등 별도의 선택 규칙이 필요하다. 이번 데이터에서 중복을 제거하면 2,008행에서 1,988행이 된다.

## 3. 격리와 정상 데이터를 명시적으로 나누기

작은 역 기준표는 드라이버로 가져와 유효 코드 목록으로 사용한다. 대규모 기준표는 collect 대신 조인 방식으로 검증한다.

```python
valid_ids = [r.station_id for r in stations.select("station_id").collect()]
is_bad = (F.col("station_id").isNull()
          | ~F.col("station_id").isin(valid_ids)
          | F.col("tag_time").isNull())
quarantine = dedup.filter(is_bad).withColumn(
    "reason",
    F.when(F.col("station_id").isNull()
           | ~F.col("station_id").isin(valid_ids), "UNKNOWN_STATION")
     .otherwise("BAD_TIME"))
clean = dedup.filter(~is_bad)
```

SQL 조건식에는 TRUE·FALSE뿐 아니라 NULL도 있다. `~isin(...)`만 사용하면 역 코드가 null인 행은 정상·격리 어느 쪽에도 남지 않을 수 있다. `isNull()`을 명시하는 이유다. 위 reason은 역 오류를 우선하는 단일 사유 분류다. 여러 오류를 모두 기록하려면 별도 플래그나 사유 목록이 필요하다.

첨부 데이터 원문을 event_id 기준으로 대조한 결과는 다음과 같다.

| 단계 | 건수 |
| --- | ---: |
| 원본 | 2,008 |
| 재전송 중복 제외 | 20 |
| 미등록 역 격리 | 6 |
| 잘못된 시각 격리 | 4 |
| 정상 이벤트 | 1,978 |

**2,008 = 20 + 10 + 1,978**로 대사한다. 여기서 정상은 이번 두 검증을 통과했다는 뜻이다. 음수 요금, 태그 종류, ID 충돌 등 모든 업무 규칙을 검사한 결과는 아니다.

## 4. 승차를 기준으로 하차 붙이기

```python
board = clean.filter(F.col("tag_type") == "IN").select(
    "trip_id", "card_id",
    F.col("station_id").alias("board_st"),
    F.col("tag_time").alias("board_time"))
alight = clean.filter(F.col("tag_type") == "OUT").select(
    "trip_id", F.col("station_id").alias("alight_st"),
    F.col("tag_time").alias("alight_time"), "fare")
trips = board.join(alight, "trip_id", "left")
trips = trips.withColumn("travel_min",
    (F.unix_timestamp("alight_time") - F.unix_timestamp("board_time")) / 60)
```

정상 이벤트는 승차 992건과 하차 986건이다. 이 데이터에서는 이동별 승·하차가 각각 최대 한 건이므로, 승차 기준 left join 결과는 992행이고 하차가 없는 이동은 14건이다. 하차는 있으나 정상 승차가 없는 8건은 이 표에 포함되지 않는다.

```python
orphan_alight = alight.join(board.select("trip_id"), "trip_id", "left_anti")
```

따라서 **992행은 모든 이동 수가 아니라 정상 승차를 기준으로 한 이동 표의 크기**다. 정상 이벤트 전체에 나타난 고유 trip_id는 1,000개다. 하차 누락을 모두 실제 미하차로 단정할 수도 없다. 원래 없던 태그 외에 정제로 제외된 태그도 영향을 준다.

## 5. 정제 전후 같은 질문을 비교하기

역별 승차 집계는 raw와 clean에서 모두 계산할 수 있지만 모집단이 다르다. 정제 전 큰 숫자가 더 정확한 이용량은 아니다. 정상 이벤트의 fare 합계는 **1,647,800**이며, 이것도 실습 로그의 기록 요금 합이지 실제 교통 수입이 아니다.

최종 연습은 정제 데이터를 이용해 노선별 승차, 30분 초과 이동, 가장 붐빈 15분 창, 두 번 이상 이용한 카드, 출발–도착 역 조합을 구한다. 질문마다 이벤트·이동·카드 중 집계 단위를 먼저 적으면 잘못된 중복 제거와 조인을 줄일 수 있다.

## 정리와 다음 내용

정제는 행을 줄이는 일이 아니라 **남긴 이유와 제외한 이유를 설명하는 과정**이다. 이동 표를 만들 때도 어느 쪽을 기준으로 연결했는지에 따라 결과가 달라진다. 다음 글에서는 이 결과를 파티션·Parquet·Delta Lake로 저장하고 관리한다.

작성 근거: STEP 12~13·20의 lab·answer 및 첨부 원문 데이터. 건수는 Python으로 별도 대조했으며 Spark 전체 실습을 재실행한 결과는 아니다.


## 함께 읽기

- [Spark 수업 정리 ①: 실행 환경과 DataFrame의 처리 방식](/sk-encore-de-2/spark/spark-basics-dataframe/)
- [Spark 수업 정리 ②: 스키마·시간 처리와 집계·조인](/sk-encore-de-2/spark/spark-schema-aggregation-join/)
- [Spark 수업 정리 ④: 파티션·Parquet·Delta Lake로 결과 관리하기](/sk-encore-de-2/spark/spark-partitions-parquet-delta/)
- [Spark 수업 정리 ⑤: Structured Streaming과 메달리온 파이프라인](/sk-encore-de-2/spark/spark-streaming-medallion/)
