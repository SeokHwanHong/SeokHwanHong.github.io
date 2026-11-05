---
layout: single
title: "Spark 수업 정리 ①: 실행 환경과 DataFrame의 처리 방식"
date: 2026-10-26 19:00:00 +0900
permalink: /sk-encore-de-2/spark/spark-basics-dataframe/
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

## Spark로 데이터를 처리하는 첫 단계

교통카드 승하차 로그를 예제로 Spark의 실행 방식부터 정리한다. 전체 흐름은 파일 읽기 → 조건 선택 → 집계·조인 → 데이터 정제 → 저장 → 스트리밍 처리다. 작은 데이터로 결과를 손으로 확인한 뒤 같은 연산을 하루 로그에 적용한다.

이 글부터 이어지는 다섯 편은 제공된 `sparkcamp` 수업자료를 **10월 26~30일 복습 시리즈**로 재구성한 기록이다. 자료의 메모와 파일 기록에는 10월 20~22일이 포함돼 있으며, 게시 날짜가 실제 수업 진행일을 뜻하지는 않는다.

## 1. 실습 파일과 한 행의 의미

| 파일 | 역할 | 첨부 데이터의 규모 |
| --- | --- | ---: |
| `stations.csv` | 역 코드·역 이름·노선·자치구 기준표 | 8행 |
| `tags_mini.jsonl` | 직접 확인하는 작은 태그 로그 | 12행 |
| `tags_day.jsonl` | 하루 승하차 이벤트 | 2,008행 |
| `stream_src/part_*.jsonl` | 순차 도착을 흉내 내는 입력 조각 | 10개 파일 |

데이터는 `make_data.py`로 만든 **실습용 합성 데이터**다. 실제 교통 이용 통계가 아니다. 하루 로그의 한 행은 이동 한 건이 아니라 **승차 또는 하차 태그 한 번**이다. 같은 `trip_id`에 승차와 하차가 있고, 같은 `event_id`가 재전송될 수도 있다.

| 열 | 의미 |
| --- | --- |
| `event_id` | 개별 태그 이벤트의 식별자 |
| `card_id` | 카드 식별자. 한 카드에 여러 이동이 가능 |
| `trip_id` | 승차와 하차를 연결할 이동 식별자 |
| `tag_type` | 승차 `IN` / 하차 `OUT` |
| `station_id` | 태그가 발생한 역 코드 |
| `tag_time` | 태그가 발생한 시각 |
| `fare` | 해당 이벤트에 기록된 요금 |
| `recv_time` | 서버에 도착한 시각 |

행 수, 고유 이벤트 수, 이동 수, 카드 수는 다른 질문에 답한다. 이 구분이 이후 중복 제거와 집계의 기준이 된다.

## 2. 실행 환경부터 확인하기

수업자료는 Windows, Java 17, Python 3.13, VS Code의 Python·Jupyter 확장을 전제로 한다. 프로젝트 폴더에서 가상환경을 만들고 제공된 의존성을 설치한다.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe make_data.py
```

VS Code 인터프리터도 이 가상환경으로 선택한다. 설치한 Python과 실행 중인 Python이 다르면 패키지가 없다는 오류를 만날 수 있다.

```python
import os
import sys

print(sys.version)
print(sys.executable)
print(os.environ.get("JAVA_HOME"))
print(os.environ.get("HADOOP_HOME"))
```

`JAVA_HOME`은 JDK 경로를, 자료의 Windows 실습 구성에서 `HADOOP_HOME`은 제공된 Hadoop 보조 도구 경로를 가리킨다. 환경변수를 바꿨다면 VS Code와 터미널을 다시 열어 새 값을 읽게 한다. 모든 설치 과정에서 기존 Java를 삭제하거나 PC를 반복 재부팅해야 하는 것은 아니다.

자료의 패키지 고정값은 PySpark 4.1.3, delta-spark 4.4.1, pandas 2.3.3, pyarrow 26.0.0, ipykernel 7.4.0이다. 이는 **첨부 requirements의 기록**이다. 새 환경을 구성할 때는 패키지 배포 여부와 Spark–Delta 호환 조합을 확인한다. 이 글 작성 과정에서 해당 조합을 설치·실행하지는 않았다.

## 3. SparkSession과 로컬 실행

아래 예제들은 작업 디렉터리가 `sparkcamp`라는 전제다.

```python
import os
import sys
from pyspark.sql import SparkSession, functions as F

os.environ["PYSPARK_PYTHON"] = sys.executable
spark = (SparkSession.builder
         .master("local[2]")
         .appName("sparkcamp-review")
         .getOrCreate())
spark.conf.set("spark.sql.session.timeZone", "Asia/Seoul")
print(spark.version)
print(spark.sparkContext.uiWebUrl)
```

`local[2]`는 로컬에서 두 개의 작업 스레드를 사용하는 설정이다. 별도 컴퓨터 두 대나 Executor 두 개를 만든다는 뜻은 아니다. 데이터는 파티션으로 나뉘고, 작업은 사용 가능한 실행 자원에 배정된다.

Spark UI에서는 Jobs·Stages·Storage 등을 보며 실행을 확인한다. `spark.range(1000000).count()`처럼 작은 확인 작업부터 실행해 연결 상태를 살펴볼 수 있다.

## 4. DataFrame은 처리 계획을 가지고 있다

```python
rows = [("T001", "강남", "홍대입구", 1650),
        ("T002", "잠실", "강남", 1650),
        ("T003", "홍대입구", "시청", 1550)]
df = spark.createDataFrame(rows, ["trip_id", "board", "alight", "fare"])
selected = df.filter(F.col("fare") > 1600).select("trip_id", "fare")
selected.explain()
selected.show()
```

`filter`와 `select`는 어떤 데이터를 만들지 표현하는 변환이다. `show`, `count`, 저장 등의 액션이 실제 계산을 유발한다. 이를 지연 실행이라고 한다. 다만 파일 읽기의 스키마 추론처럼 데이터를 살펴봐야 하는 준비 작업도 있어, 모든 API 호출이 전혀 실행되지 않는다고 이해해서는 안 된다.

DataFrame은 pandas처럼 보이지만, 전체 결과를 처음부터 Python 메모리에 보관하는 표로 생각하면 안 된다. `collect()`와 `toPandas()`는 결과를 드라이버로 가져오므로, 집계나 필터로 충분히 줄인 결과에 사용한다.

## 5. 파일 읽기와 조건식

```python
stations = spark.read.option("header", True).csv("data/stations.csv")
mini = spark.read.json("data/tags_mini.jsonl")
mini.printSchema()

(mini.filter((F.col("tag_type") == "OUT") & (F.col("fare") >= 1600))
     .select("event_id", "trip_id", "station_id", "fare")
     .orderBy(F.col("fare").desc(), "event_id")
     .show())
```

CSV의 `header=True`는 첫 줄을 열 이름으로 읽는 옵션이다. 숫자·시간 자료형까지 지정하는 옵션은 아니다. JSONL은 한 줄에 JSON 객체 하나를 담으며, 폴더를 읽으면 여러 파일을 함께 처리할 수 있다.

복합 조건에는 Python의 `and`·`or` 대신 `&`·`|`를 사용하고 각 비교를 괄호로 감싼다. 새 열은 `withColumn`, 조건 분기는 `when`, 이름 변경은 `alias`로 표현한다. 원본에 `IN`·`OUT` 외 값이 들어올 수 있다면 `otherwise("하차")`로 모두 덮지 말고 미등록 값을 별도로 처리한다.

## 6. 저장 결과가 폴더인 이유

```python
spark.range(10).write.mode("overwrite").csv("out/review/check")
spark.read.csv("out/review/check").count()
```

Spark의 파일 저장 경로는 보통 단일 파일명이 아니라 디렉터리다. 내부에 `part-*` 데이터 파일과 완료 표시 등이 생긴다. 출력 파일 수는 쓰기 시점의 파티션과 실행·저장 설정에 영향을 받는다. `local[2]`라고 모든 작업에서 파일이 항상 두 개 생기는 것은 아니다.

작은 내보내기를 한 파일로 줄일 때는 `coalesce(1)`을 쓸 수 있지만, 큰 결과 전체를 한 작업으로 몰면 병목이 생긴다. 파일을 하나로 만드는 것과 분산 처리를 효율적으로 하는 것은 별개의 목적이다.

## 정리와 다음 내용

첫 단계에서는 코드 문법보다 **데이터의 행 단위, 실행 환경, 변환과 액션**을 구분하는 것이 중요하다. 다음 글에서는 스키마를 직접 정하고, 시간·집계·조인으로 로그를 해석한다.

작성 근거: `시작하기.txt`, `requirements.txt`, `etc/261020.md`, `make_data.py`, STEP 03~06의 lab·answer. 예제는 설명에 맞춰 축약했다.


## 함께 읽기

- [Spark 수업 정리 ②: 스키마·시간 처리와 집계·조인](/sk-encore-de-2/spark/spark-schema-aggregation-join/)
- [Spark 수업 정리 ③: 윈도 함수와 데이터 정제·이동 단위 구성](/sk-encore-de-2/spark/spark-window-data-quality/)
- [Spark 수업 정리 ④: 파티션·Parquet·Delta Lake로 결과 관리하기](/sk-encore-de-2/spark/spark-partitions-parquet-delta/)
- [Spark 수업 정리 ⑤: Structured Streaming과 메달리온 파이프라인](/sk-encore-de-2/spark/spark-streaming-medallion/)
