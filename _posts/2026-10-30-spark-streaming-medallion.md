---
layout: single
title: "Spark 수업 정리 ⑤: Structured Streaming과 메달리온 파이프라인"
date: 2026-10-30 19:00:00 +0900
permalink: /sk-encore-de-2/spark/spark-streaming-medallion/
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

## 배치에서 새 데이터의 반복 처리로

10월 30일 복습은 STEP 18~20이다. 고정 파일 전체를 읽던 배치 처리에서 출발해, 새 파일이 도착하면 처리하는 Structured Streaming과 Bronze–Silver–Gold 구조로 이어간다.

실습 입력은 Kafka가 아니라 **파일 소스**다. 실제 외부 교통 API를 연결한 것도 아니다. 같은 수업용 합성 로그를 열 개 파일로 나누어 데이터의 도착을 흉내 낸다.

## 1. 스트리밍 DataFrame 만들기

앞 글의 SparkSession 또는 Delta 기능을 설정한 세션을 사용한다.

```python
from pyspark.sql import functions as F

RAW_SCHEMA = """
event_id STRING, card_id STRING, trip_id STRING, tag_type STRING,
station_id STRING, tag_time STRING, fare INT, recv_time STRING
"""
sdf = spark.readStream.schema(RAW_SCHEMA).json("data/stream_in")
print(sdf.isStreaming)
```

읽는 경로는 먼저 준비한다. 스트리밍 파일 입력에는 스키마를 명시하고, 완성된 파일을 입력 폴더로 이동해 읽다 만 파일이 보이지 않도록 한다. 자료는 숨김 임시 이름으로 복사한 뒤 `os.replace`로 최종 이름을 노출한다.

배치의 `show()`·`count()`로 스트리밍 전체 결과를 바로 얻으려 하지 않고 `writeStream.start()`로 쿼리를 시작한다. 종료할 때는 해당 쿼리의 `stop()`을 호출한다.

## 2. 출력 모드와 메모리 전광판

| 모드 | 출력 대상 |
| --- | --- |
| `append` | 결과 테이블에 새로 추가되는 확정 행 |
| `update` | 이번 처리로 바뀐 결과 행 |
| `complete` | 결과 테이블 전체 |

자료의 역별 누적 집계는 `groupBy("station_id").count()`를 memory sink의 complete 모드로 보여준다. 이는 작은 결과를 SQL로 확인하기 위한 학습용 구성이다. 메모리 테이블을 영속 운영 저장소로 사용하지 않는다.

역별 누적 수는 계속 바뀌기 때문에 워터마크 없는 해당 집계에 append 모드를 적용할 수 없다. 단순히 워터마크 한 줄을 추가하면 모든 누적 집계가 append 가능해지는 것도 아니다. 이벤트 시간 창 집계처럼 결과를 확정할 조건이 맞아야 한다. 이번 자료에는 워터마크를 활용한 완성된 운영 예제가 없다.

## 3. foreachBatch와 availableNow

```python
def save_batch(batch_df, batch_id):
    batch_df.write.mode("append").parquet("out/review/stream_raw")

q = (spark.readStream.schema(RAW_SCHEMA).json("data/stream_in")
     .writeStream.foreachBatch(save_batch)
     .option("checkpointLocation", "chk/review/raw")
     .trigger(availableNow=True).start())
q.awaitTermination()
```

`foreachBatch`는 마이크로배치 결과를 일반 배치 DataFrame으로 받아 저장·MERGE 등을 수행한다. `availableNow`는 실행 시 이용 가능한 입력을 처리하고 종료하므로 영구 실행 프로세스와는 다르다. 여러 마이크로배치로 나뉠 수 있고, 정확한 배치 경계는 입력과 설정에 따라 달라진다.

자료의 10개 입력 파일은 첫 9개가 201행, 마지막이 199행으로 총 2,008행이다. 3개 파일을 먼저 놓는 예제에서는 603행, 네 번째까지는 804행이다. 파일 수와 입력 행 수를 먼저 확인하면 새 데이터 처리 여부를 검산하기 쉽다.

## 4. 체크포인트가 기억하는 것

체크포인트에는 입력 진행 위치, 완료한 배치 등의 복구 정보가 남는다. 같은 쿼리가 같은 체크포인트로 다시 시작하면 이미 처리한 파일을 보통 다시 처리하지 않는다.

자료의 실험은 기존 Parquet 출력을 남겨 둔 채 체크포인트를 삭제한다. 804행 입력을 다시 읽어 append하면 1,608행으로 늘어나는 장면이다. 체크포인트와 출력 결과는 함께 관리해야 한다.

다만 체크포인트가 있다는 사실만으로 모든 외부 쓰기가 exactly-once가 되지는 않는다. `foreachBatch`의 쓰기는 재시도될 수 있으므로, 출력 쪽도 중복 적용에 견디도록 설계해야 한다. `batch_id`를 활용한 중복 적용 방지나 키 기반 멱등 upsert가 그 방법이다.

## 5. Bronze–Silver–Gold 역할 나누기

| 계층 | 행 단위 | 자료의 처리 |
| --- | --- | --- |
| Bronze | 수신한 이벤트 | 문자열 시각 보존, 수집 시각 추가, Delta 저장 |
| Silver | trip_id별 이동 | 중복 제거·시각 변환·유효성 확인 후 MERGE |
| Gold | 역별 집계 | Silver의 승차 역을 집계하고 역 이름 연결 |

메달리온은 각 계층이 책임질 데이터의 의미를 나누는 구조다. 이 실습은 `run_bronze()`와 `run_silver()`를 순서대로 호출하고 Gold를 배치로 다시 만드는 방식이다. 독립 스케줄러가 전 단계를 자동 운영하는 구성은 아니다.

Bronze는 입력 원문 중 시각 표현을 문자열로 남기며 Delta sink와 전용 체크포인트로 수집한다. 다만 고정 스키마로 JSON을 읽은 결과이므로 원본 JSON의 모든 알 수 없는 필드·바이트를 완전히 보존하는 아카이브와는 구분한다.

## 6. Silver에서 이동별 정보를 보완하기

자료는 `foreachBatch` 내부에서 event_id를 중복 제거하고, `try_to_timestamp`로 시각을 해석하며 미등록 역·잘못된 시각을 제외한다. 이후 trip_id별로 승차·하차 열을 만들고 기존 Silver에 MERGE한다.

```python
def to_trips(ev):
    return ev.groupBy("trip_id").agg(
        F.first("card_id").alias("card_id"),
        F.max(F.when(F.col("tag_type") == "IN", F.col("station_id"))).alias("board_st"),
        F.max(F.when(F.col("tag_type") == "IN", F.col("tag_time"))).alias("board_time"),
        F.max(F.when(F.col("tag_type") == "OUT", F.col("station_id"))).alias("alight_st"),
        F.max(F.when(F.col("tag_type") == "OUT", F.col("tag_time"))).alias("alight_time"),
        F.max("fare").alias("fare"))
```

이 함수는 전달된 이벤트의 tag_time이 timestamp로 변환됐다는 전제다. 같은 이동의 승·하차가 다른 배치로 들어와도 기존 열을 null로 지우지 않고 보완하도록 MERGE 규칙을 둔다.

배치 내부 `dropDuplicates`는 서로 다른 배치 사이의 모든 중복을 기억하는 연산이 아니다. 이번 예제는 trip_id별 Silver의 보완 규칙으로 동일 이벤트 재도착에도 같은 논리적 결과를 유지하도록 구성한다. 임의 정정 데이터까지 항상 순서와 무관하게 처리한다는 보장은 별도 설계가 필요하다.

## 7. 배치 결과와 대조하고 Gold 만들기

전체 입력이 도착한 뒤 Silver에서 확인할 기준은 다음과 같다. 아래 값은 수업자료의 기대 결과이며, 첨부 원문 데이터의 별도 Python 집계와 일치한다.

| 확인 항목 | 기대값 |
| --- | ---: |
| Bronze 입력 행 | 2,008 |
| 정상 이벤트의 고유 이동 | 1,000 |
| 하차 정보 없는 이동 | 14 |
| 승차 정보 없는 이동 | 8 |
| 기록 요금 합계 | 1,647,800 |

승차 기준 left join으로 만든 992행과 전체 trip_id를 통합한 1,000행을 같은 지표로 비교하지 않는다. 후자는 승차를 잃은 하차 8건도 남긴다.

```python
stations = spark.read.option("header", True).csv("data/stations.csv")
silver = spark.read.format("delta").load("lake/s19/silver")
gold = (silver.filter(F.col("board_st").isNotNull())
        .groupBy("board_st").agg(F.count("*").alias("board_cnt"))
        .join(stations, F.col("board_st") == F.col("station_id"))
        .select("station_nm", "line", "board_cnt"))
gold.write.format("delta").mode("overwrite").save("lake/review/gold/board_by_station")
```

Gold는 이동 테이블에서 승차 역이 존재하는 이동을 센다. 원본 이벤트의 중복 행 수를 다시 세는 것과 다르다. 실제 장애 검증에서는 행 수뿐 아니라 trip_id별 값과 합계도 비교한다.

## 8. 이번 실습에서 더 보완할 부분

Silver 예제는 잘못된 이벤트를 필터링하지만 별도 격리 테이블에 쓰지는 않는다. STEP 13의 사유별 격리를 스트리밍에도 연결하면 누락 원인을 추적할 수 있다. 같은 ID의 상충 내용, 요금 정정 우선순위, 지연 데이터 허용 범위와 상태 보존 기간도 후속 과제다.

체크포인트를 지우는 실험은 복구 동작을 배우기 위한 전용 실습 경로에서 수행한다. 재실행 후에도 키별 내용이 같은지 검증하는 것까지 포함해야 멱등성을 설명할 수 있다. 자료의 일부 Delta 예제는 `execute()` 반환값에 결과 출력 함수를 이어 쓰므로, 실행 환경의 API 반환형을 확인하고 MERGE 이후 테이블을 다시 읽는 방식으로도 검증한다.

## 다섯 편을 마치며

이번 흐름은 데이터를 읽는 문법에서 시작해, 행의 의미와 정제 기준을 정하고, 저장 방식과 재처리 결과를 관리하는 과정으로 이어졌다. 배치와 스트리밍에서 공통으로 중요한 것은 **새 데이터가 도착하거나 같은 입력을 다시 처리해도 결과의 의미가 유지되는가**다.

작성 근거: STEP 18~20, STEP 17의 MERGE 규칙, 첨부 입력 파일. Python으로 원문 건수를 검산했으며 Spark·Delta 파이프라인 전체를 이 환경에서 재실행하지 않았다.

참고: [Spark 4.1.3 Structured Streaming](https://spark.apache.org/docs/4.1.3/streaming/apis-on-dataframes-and-datasets.html), [Delta MERGE와 foreachBatch](https://docs.delta.io/delta-update/).


## 함께 읽기

- [Spark 수업 정리 ①: 실행 환경과 DataFrame의 처리 방식](/sk-encore-de-2/spark/spark-basics-dataframe/)
- [Spark 수업 정리 ②: 스키마·시간 처리와 집계·조인](/sk-encore-de-2/spark/spark-schema-aggregation-join/)
- [Spark 수업 정리 ③: 윈도 함수와 데이터 정제·이동 단위 구성](/sk-encore-de-2/spark/spark-window-data-quality/)
- [Spark 수업 정리 ④: 파티션·Parquet·Delta Lake로 결과 관리하기](/sk-encore-de-2/spark/spark-partitions-parquet-delta/)
