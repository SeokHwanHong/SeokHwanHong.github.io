---
layout: single
title: "Spark 수업 정리 ②: 스키마·시간 처리와 집계·조인"
date: 2026-10-27 19:00:00 +0900
permalink: /sk-encore-de-2/spark/spark-schema-aggregation-join/
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

## 문자열 로그를 분석 가능한 표로 바꾸기

이번 글은 STEP 07~11을 중심으로 스키마, 집계, 시간 처리, 조인과 Spark SQL을 정리한다. 날짜는 10월 26~30일 복습 시리즈의 두 번째 정리일이다. 실제 수업 일자를 재현한 기록은 아니다.

핵심 질문은 “읽혔는가?”에서 “어떤 자료형과 기준으로 읽혔는가?”로 바뀐다. 시간처럼 보이는 문자열도 실제 timestamp가 아니면 오류나 비교 방식이 달라진다.

## 1. 스키마를 명시하는 이유

```python
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.master("local[2]").appName("spark-day2").getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "Asia/Seoul")
SCHEMA = """
event_id STRING, card_id STRING, trip_id STRING, tag_type STRING,
station_id STRING, tag_time TIMESTAMP, fare INT, recv_time TIMESTAMP
"""
day = spark.read.schema(SCHEMA).json("data/tags_day.jsonl")
day.printSchema()
```

자동 추론은 빠른 탐색에 편리하지만, 운영 처리에서는 입력마다 자료형이 달라지지 않도록 계약을 정하는 것이 중요하다. `StructType`·`StructField`로 작성하거나 위처럼 DDL 문자열로 표현할 수 있다. JSON은 필드 이름이 대응해야 하므로 `tag_time`을 `tagtime`으로 잘못 적으면 원래 필드를 읽지 못한다.

## 2. 파싱 실패를 어떻게 볼 것인가

자료에는 분이 61·63·67·75처럼 잘못된 시각이 들어 있다. 문자열로 읽으면 값은 존재하지만, timestamp로 해석할 수는 없다. 원문과 파싱 결과를 나란히 남겨 원인을 확인한다.

```python
RAW_SCHEMA = SCHEMA.replace("tag_time TIMESTAMP", "tag_time STRING")
raw_text = spark.read.schema(RAW_SCHEMA).json("data/tags_day.jsonl")
parsed = raw_text.withColumn("tag_ts", F.try_to_timestamp("tag_time"))
parsed.filter(F.col("tag_ts").isNull()).select("event_id", "tag_time").show()
```

`try_to_timestamp`는 변환할 수 없는 값을 null로 돌려준다. 일반 `to_timestamp`의 실패 동작은 ANSI 설정 등에 영향을 받으므로 같은 함수로 혼동하지 않는다.

JSON 읽기 모드도 구분한다.

| 모드 | 목적 |
| --- | --- |
| `PERMISSIVE` | 파싱 실패를 허용하며 corrupt-record 열 등으로 확인 |
| `DROPMALFORMED` | 파싱할 수 없는 레코드 제외 |
| `FAILFAST` | 잘못된 레코드에서 실패시키기 |

`_corrupt_record STRING`을 스키마에 포함하면 손상 레코드 확인에 활용할 수 있다. 원본 보존은 이 열에만 의존하지 않고 원천 파일과 읽기 정책을 함께 관리한다.

## 3. 무엇을 세는지부터 정하기

```python
day.groupBy("tag_type").agg(
    F.count("*").alias("rows"),
    F.countDistinct("event_id").alias("events")
).show()
```

첨부 원문 데이터의 승차는 1,012행, 하차는 996행으로 총 2,008행이다. 고유 이벤트는 1,988개다. 차이 20행은 재전송 중복이다. 따라서 정제 전 `count(*)`를 고유 승객 수나 고유 이동 수로 부르면 안 된다.

| 표현 | 세는 대상 |
| --- | --- |
| `count("*")` | 행 수 |
| `count("fare")` | fare가 null이 아닌 행 수 |
| `countDistinct("event_id")` | 고유 이벤트 수 |
| `countDistinct("card_id")` | 고유 카드 수 |

요금 합계는 `sum`, 평균은 `avg`, 조건별 열 펼치기는 `pivot`으로 표현한다. 중복 이벤트가 남아 있으면 요금도 중복 합산될 수 있으므로, 탐색 집계와 최종 정제 집계를 구분한다.

## 4. 발생 시각과 수신 시각

`tag_time`은 실제 태그 시각, `recv_time`은 서버 수신 시각이다. 데이터 생성 코드에는 일부 하차 태그가 20~40분 늦게 도착하도록 한 설정이 있다. 두 시간을 바꾸어 집계하면 다른 질문에 답하게 된다.

```python
(day.filter((F.col("tag_type") == "IN") & F.col("tag_time").isNotNull())
    .groupBy(F.hour("tag_time").alias("hour"))
    .count().orderBy("hour").show())

(day.withColumn("delay_sec",
                F.unix_timestamp("recv_time") - F.unix_timestamp("tag_time"))
    .orderBy(F.col("delay_sec").desc())
    .select("event_id", "tag_time", "recv_time", "delay_sec").show(5))
```

일부 잘못된 시각은 null이 되므로 집계에서 포함·제외 기준을 정해야 한다. 특정 시간대를 고를 때는 `07:00 이상, 08:00 미만`과 같은 반개구간을 쓰면 경계가 겹치지 않는다. 입력에 시간대 정보가 없으므로 이 복습 예제에서는 세션 시간대를 서울로 명시한다.

## 5. 역 기준표와 연결하기

```python
stations = spark.read.option("header", True).csv("data/stations.csv")
joined = day.join(stations, "station_id", "left")
joined.select("event_id", "station_id", "station_nm", "line").show(5)

unknown = day.join(stations.select("station_id"), "station_id", "left_anti")
unknown.select("event_id", "station_id").show()
```

`inner`는 연결되는 행만, `left`는 왼쪽 이벤트를 남긴다. `left_anti`는 기준표에 없는 이벤트를 찾는 데 유용하다. 자료에 있는 `S99`는 역 기준표에 없는 코드다.

기준표의 키가 중복되면 한 이벤트가 여러 행으로 늘어난다. 조인 전에 역 코드의 고유성을 확인하고, 조인 후 행 수가 예상과 같은지 확인한다. 알려지지 않은 역이 inner join에서 사라졌다고 정제가 완료된 것은 아니다. 누락 원인을 별도로 남겨야 한다.

## 6. Spark SQL과 DataFrame API

```python
day.createOrReplaceTempView("tags")
stations.createOrReplaceTempView("stations")
result = spark.sql("""
    SELECT s.station_nm, COUNT(*) AS tag_cnt
    FROM tags t JOIN stations s ON t.station_id = s.station_id
    WHERE t.tag_type = 'IN'
    GROUP BY s.station_nm
    ORDER BY tag_cnt DESC, s.station_nm
""")
result.show()
```

임시 뷰는 세션에서 SQL로 조회할 이름을 붙이는 것이다. 별도 영구 테이블을 저장하지는 않는다. SQL과 DataFrame API는 같은 Spark SQL 엔진에서 계획·최적화되며, 같은 의미의 질의를 두 방식으로 표현할 수 있다. MySQL의 모든 문법이 그대로 지원된다는 뜻은 아니다.

자료의 CTE 예제는 역별 승차 수에서 평균 초과 역을 찾는다. 원본 승차를 역 코드별로 묶으면 S99를 포함해 9개 그룹이므로 평균은 1,012/9 ≈ 112.44다. 이 기준보다 큰 8개 등록 역이 남고 S99는 제외된다. 평균을 계산한 모집단에 미등록 역도 포함돼 있으므로, 정제 후 평균과 같은 지표로 해석하지 않는다.

## 정리와 다음 내용

스키마는 읽기 계약이고, 집계는 질문의 단위를 정하는 일이며, 조인은 그 단위를 유지하면서 정보를 붙이는 작업이다. 다음 글에서는 윈도 함수와 정제 규칙을 이용해 이벤트를 이동 단위로 바꾼다.

작성 근거: STEP 07~11의 lab·answer, 첨부 JSONL·역 기준표·데이터 생성 코드. 수치는 첨부 데이터의 정적 집계와 대조했다.


## 함께 읽기

- [Spark 수업 정리 ①: 실행 환경과 DataFrame의 처리 방식](/sk-encore-de-2/spark/spark-basics-dataframe/)
- [Spark 수업 정리 ③: 윈도 함수와 데이터 정제·이동 단위 구성](/sk-encore-de-2/spark/spark-window-data-quality/)
- [Spark 수업 정리 ④: 파티션·Parquet·Delta Lake로 결과 관리하기](/sk-encore-de-2/spark/spark-partitions-parquet-delta/)
- [Spark 수업 정리 ⑤: Structured Streaming과 메달리온 파이프라인](/sk-encore-de-2/spark/spark-streaming-medallion/)
