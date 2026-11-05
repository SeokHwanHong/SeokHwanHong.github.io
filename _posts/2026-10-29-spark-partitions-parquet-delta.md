---
layout: single
title: "Spark 수업 정리 ④: 파티션·Parquet·Delta Lake로 결과 관리하기"
date: 2026-10-29 19:00:00 +0900
permalink: /sk-encore-de-2/spark/spark-partitions-parquet-delta/
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

## 계산 결과를 어떻게 나누고 저장할까

10월 29일 복습은 STEP 14~17이다. DataFrame의 결과가 맞는지 확인했다면, 반복 계산 비용과 저장 형식·변경 이력을 함께 살펴본다.

아래 코드는 앞 글에서 만든 `spark`, `raw`, `clean`과 `stations`를 이어서 사용하는 예제다. Delta 예제를 실행하려면 별도 세션 설정도 필요하다.

## 1. 실행 파티션과 저장 파티션

파티션이라는 말은 두 맥락에서 등장한다.

| 구분 | 의미 | API 예 |
| --- | --- | --- |
| 실행 파티션 | 작업을 나눠 처리하는 데이터 조각 | `repartition`, `coalesce` |
| 저장 파티션 | 열 값별 디렉터리로 파일 배치 | `write.partitionBy` |

`repartition(4)`는 셔플을 거쳐 네 파티션으로 재분배한다. `repartition(4, "station_id")`는 역 코드 해시를 기준으로 나눈다. 키의 종류와 빈도가 한쪽으로 몰리면 데이터 양도 불균형해질 수 있다. `IN`·`OUT` 두 종류뿐인 열로 네 파티션을 만든다고 네 조각에 골고루 나뉘는 것은 아니다.

`coalesce(1)`은 파티션을 줄일 때 보통 전체 셔플 없이 합치는 방식이다. 한 파티션으로 몰아 쓴다는 비용이 있으므로 작은 외부 내보내기와 대규모 저장을 구분한다.

## 2. 셔플과 실행 계획 보기

```python
from pyspark.sql import functions as F

agg = clean.groupBy("station_id").count()
agg.explain()
print(spark.conf.get("spark.sql.shuffle.partitions"))
print(spark.conf.get("spark.sql.adaptive.enabled"))
```

groupBy·join 같은 연산에서는 같은 키를 모으기 위해 데이터 이동이 필요할 수 있다. 실행 계획의 `Exchange`는 이를 살펴보는 단서다. `spark.sql.shuffle.partitions`는 셔플 파티션 수에 영향을 주지만 AQE가 실행 중 조정할 수 있어 최종 파티션 수를 무조건 고정하는 값으로 보면 안 된다.

자료는 작은 데이터에서 200개와 8개 셔플 파티션을 비교하도록 한다. 적은 데이터에 너무 많은 작업을 만들면 스케줄링 비용이 상대적으로 커질 수 있다. 반대로 큰 데이터에 너무 적은 파티션을 쓰면 개별 작업이 무거워진다. 8개가 언제나 최적이라는 뜻은 아니다.

## 3. cache는 반복 계산을 줄이기 위한 선택

```python
clean.cache()
clean.count()       # 액션으로 캐시를 채움
clean.groupBy("station_id").count().show()
clean.unpersist()
```

같은 정제 결과를 반복 조회한다면 캐시가 유용하다. 캐시를 선언한 시점과 실제로 채우는 시점은 구분한다. 필요 없는 데이터를 계속 캐시하면 메모리를 소비하므로 작업 후 해제한다. 속도를 비교할 때는 첫 실행 비용, 캐시 유무, AQE와 반복 횟수를 맞추고 Spark UI도 함께 본다.

## 4. CSV와 Parquet

```python
clean.write.mode("overwrite").option("header", True).csv("out/review/clean_csv")
clean.write.mode("overwrite").parquet("out/review/clean_parquet")
spark.read.parquet("out/review/clean_parquet").printSchema()
```

CSV는 사람이 열어 보기 쉽지만 자료형을 별도로 관리해야 한다. Parquet은 열 기반 형식으로 스키마와 압축을 활용하고 필요한 열을 읽는 분석에 적합하다. 자료에서는 같은 데이터를 두 형식으로 저장해 크기와 읽기 결과를 비교한다. 실제 크기 비율은 파일 수와 압축 등 조건에 따라 달라지므로 고정 성능 수치로 옮기지 않는다.

| 저장 모드 | 의미 |
| --- | --- |
| 기본 `errorifexists` | 경로가 있으면 오류 |
| `append` | 기존 결과에 추가 |
| `overwrite` | 해당 쓰기 범위의 결과 교체 |
| `ignore` | 경로가 있으면 쓰기 생략 |

정상 1,978행을 Parquet에 append로 두 번 넣으면 중복을 자동 제거하지 않고 3,956행이 된다. 저장 성공과 중복 없는 결과는 다르다.

```python
(clean.write.mode("overwrite").partitionBy("tag_type")
      .parquet("out/review/by_type"))
outs = spark.read.parquet("out/review/by_type").filter("tag_type = 'OUT'")
outs.explain()
```

`tag_type=IN`·`tag_type=OUT` 같은 디렉터리가 생긴다. 필터 조건에 맞는 저장 파티션만 읽는지 `PartitionFilters`로 확인한다. 고유값이 너무 많은 키로 디렉터리를 나누면 작은 파일과 폴더가 늘어날 수 있다.

## 5. Delta Lake: 파일과 변경 이력을 함께 관리

Delta Lake는 데이터 파일에 트랜잭션 로그를 더해 테이블 버전을 관리한다. 폴더의 모든 Parquet을 임의로 합치는 대신 로그가 정의하는 현재 스냅샷을 읽는다.

새 Interactive 세션에서 호환되는 Delta 패키지를 설치한 후 설정한다. 기존 일반 Spark 세션이 있으면 `getOrCreate()`만으로 모든 설정이 새로 적용된다고 가정하지 않는다.

```python
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable

builder = (SparkSession.builder.master("local[2]").appName("spark-delta-review")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"))
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "Asia/Seoul")
```

Bronze에서는 잘못된 시각도 다시 확인할 수 있도록 문자열로 읽은 자료를 저장한다. 파싱한 열만 보존하면 실패 원문이 사라질 수 있다.

```python
RAW_SCHEMA = """
event_id STRING, card_id STRING, trip_id STRING, tag_type STRING,
station_id STRING, tag_time STRING, fare INT, recv_time STRING
"""
raw_text = spark.read.schema(RAW_SCHEMA).json("data/tags_day.jsonl")
path = "lake/review/bronze"
raw_text.write.format("delta").mode("overwrite").save(path)
dt = DeltaTable.forPath(spark, path)
dt.history().select("version", "operation").show()
```

실습은 작은 로그 12행을 잘못 append하고, `versionAsOf`로 이전 버전과 비교한 뒤 삭제하는 흐름이다. 새로 만든 테이블 기준 2,008 → 2,020 → 2,008행으로 변한다. 오래된 버전 읽기는 필요한 로그·데이터 파일이 보존돼 있을 때 가능하며 무기한 보장되는 기능은 아니다.

## 6. MERGE로 뒤늦은 하차를 보완하기

STEP 17에서는 수신 시각 08:30 이전과 이후로 이벤트를 나눈다. trip_id별로 승·하차 열을 접으면 뒤 묶음에 같은 이동의 추가 정보가 들어온다.

| 항목 | 자료의 기대 건수 |
| --- | ---: |
| 앞 묶음의 이동 | 527 |
| 뒤 묶음의 이동 | 661 |
| 두 묶음에서 겹치는 이동 | 188 |
| 단순 연결 결과 | 1,188 |
| trip_id 기준 통합 결과 | 1,000 |

`MERGE`는 같은 trip_id면 열을 보완하고, 없으면 추가한다. 자료는 승·하차 열에서 새 값이 null이면 기존 값을 유지하는 `coalesce` 규칙을 쓴다. 같은 입력을 다시 적용해도 같은 논리적 결과가 남도록 하는 멱등성의 예다. 논리적 결과가 같아도 새 트랜잭션과 파일 작업이 발생할 수 있다.

자료의 `max`·`first`·`greatest`는 이 실습 데이터의 구성에 맞춘 집계·보완 규칙이다. 같은 이동에 상충하는 역·카드·요금 정정이 여러 개 들어오는 문제까지 해결하지는 않는다. `MERGE` 대상 키로 source를 한 행으로 만들고 어떤 값이 최신인지 정의하는 것이 우선이다.

작은 파일은 OPTIMIZE로 합치고, ZORDER는 특정 열의 값이 가까운 파일 배치를 돕는 데 활용할 수 있다. 활성 파일 수가 줄어도 과거 버전 파일이 즉시 물리 삭제되는 것은 아니다.

## 정리와 다음 내용

실행 파티션은 계산 비용, 저장 파티션과 파일 형식은 읽기·쓰기 방식, Delta 로그와 MERGE는 변경·재처리의 기준에 연결된다. 다음 글에서는 이 구조를 파일 스트리밍과 Bronze–Silver–Gold 파이프라인에 적용한다.

작성 근거: STEP 14~17의 lab·answer. Delta 전후 건수는 자료의 기대 결과이며 이 작성 환경에서 재실행하지 않았다.

참고: [Spark 성능 튜닝](https://spark.apache.org/docs/4.1.3/sql-performance-tuning.html), [Delta MERGE](https://docs.delta.io/delta-update/), [Spark–Delta 호환성](https://docs.delta.io/releases/).


## 함께 읽기

- [Spark 수업 정리 ①: 실행 환경과 DataFrame의 처리 방식](/sk-encore-de-2/spark/spark-basics-dataframe/)
- [Spark 수업 정리 ②: 스키마·시간 처리와 집계·조인](/sk-encore-de-2/spark/spark-schema-aggregation-join/)
- [Spark 수업 정리 ③: 윈도 함수와 데이터 정제·이동 단위 구성](/sk-encore-de-2/spark/spark-window-data-quality/)
- [Spark 수업 정리 ⑤: Structured Streaming과 메달리온 파이프라인](/sk-encore-de-2/spark/spark-streaming-medallion/)
