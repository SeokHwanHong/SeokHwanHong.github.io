---
layout: single
title: "JSON·pandas 실습 2: 오류 데이터 격리와 검증 가능한 매출 파이프라인"
date: 2026-10-07 17:00:00 +0900
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

## 10월 7일 수업 정리

이번 수업은 주문 JSON을 상품별 행으로 펼친 다음, 잘못된 값을 격리하고 상품 정보와 결합해 카테고리별 매출을 저장·시각화하는 흐름을 다뤘다. 첨부된 `jsonpd_part2.zip`의 h00~h10 실습 파일과 데이터·출력 파일을 확인해 정리했다.

[10월 6일 pandas 기초·처리 흐름 정리]({% post_url 2026-10-06-python-pandas-pipeline %})와 같은 주문 예제를 사용한다. 이 글에서는 **행의 단위를 지키는 정제, 결합 관계 검증, 집계 결과를 다시 확인하는 과정**에 초점을 둔다. 첨부의 `261006.md`는 빈 파일이므로 날짜는 사용자가 알려 준 10월 7일을 기준으로 했다.

원본에는 오류를 경험하는 셀과 아직 작성하지 않은 과제가 있다. 아래 코드는 설명에 필요한 오타를 바로잡은 정리 예제다. 마지막 종합 과제는 **복습용 보완 예제**이며 원본에서 완료된 결과로 소개하지 않는다.

## 1. 주문 한 건과 상품 한 행은 다르다

| 자료 | 크기 | 한 행 또는 객체의 의미 |
| --- | ---: | --- |
| `orders.json` | 주문 객체 5개 | 주문 한 건, 상품 목록 포함 |
| `order_lines.csv` | 7행 | 주문에 포함된 상품 항목 하나 |
| `products.csv` | 3행 | 상품 한 종류의 이름·분류·단가 |
| `orders_dirty.json` | 주문 객체 8개, 펼친 표 9행 | 중복·수량 오류·날짜 오류·도시 누락을 포함한 연습 자료 |

한 주문에 여러 상품이 있으므로 `order_id`는 상품 표에서 반복될 수 있다. 주문 번호가 같다는 이유로 중복을 제거하면 다른 상품까지 사라진다. `status="paid"`인 상품 행은 6행이지만 결제된 주문은 4건이다. 상품 항목 수는 행 수로, 주문 건수는 `order_id.nunique()`로 구분한다.

```python
import json
from pathlib import Path
import pandas as pd

meta = ["order_id", "ordered_at", "status",
        ["customer", "id"], ["customer", "city"]]
with open("data/orders.json", encoding="utf-8") as file:
    orders = json.load(file)
lines = pd.json_normalize(orders, record_path="items", meta=meta)
paid = lines.loc[lines["status"] == "paid"]
print(len(lines), len(paid), paid["order_id"].nunique())  # 7, 6, 4
```

`record_path`는 반복할 상품 목록을, `meta`는 각 상품 행에 붙일 주문·고객 정보를 지정한다. 중첩 JSON이 표로 바뀌는 순간부터 어떤 단위의 행을 세고 있는지 확인해야 한다.

## 2. 오류를 지우기보다 사유를 남겨 격리하기

### 없는 도시 키와 잘못된 수량은 별개의 문제

도시 키가 없는 주문은 `json_normalize(..., errors="ignore")`로 펼치면 해당 메타데이터가 결측으로 남는다. 이 옵션이 수량의 문자열 오류나 잘못된 날짜까지 해결해 주는 것은 아니다.

```python
with open("data/orders_dirty.json", encoding="utf-8") as file:
    dirty = json.load(file)
raw = pd.json_normalize(dirty, record_path="items", meta=meta, errors="ignore")
step = raw.drop_duplicates().copy()
step["qty_num"] = pd.to_numeric(step["qty"], errors="coerce")
step["date"] = pd.to_datetime(step["ordered_at"], errors="coerce")

bad_mask = step["qty_num"].isna() | step["date"].isna()
bad = step.loc[bad_mask].copy()
good = step.loc[~bad_mask].copy()
good["customer.city"] = good["customer.city"].fillna("미상")
good["qty"] = good["qty_num"].astype("int64")
```

숫자로 변환하지 못한 값은 NaN, 유효하지 않은 날짜는 NaT가 된다. 원래 열을 남겨 두어야 변환 실패 사유를 확인할 수 있다.

| 주문 | 원래 값 | 격리 사유 |
| --- | --- | --- |
| O004 | 수량 `null` | 수량 누락 |
| O005 | 수량 `"두개"` | 숫자 변환 실패 |
| O006 | 날짜 `2026-02-30` | 존재하지 않는 날짜 |

모르는 수량을 0으로 바꾸면 평균과 매출의 의미가 바뀐다. 도시의 "미상" 표시는 계산값을 임의로 만들지 않으면서 누락을 드러내는 방식이다.

### 같은 행 단위로 건수를 맞추기

```python
n_dup = len(raw) - len(step)
assert len(raw) == n_dup + len(bad) + len(good)
print(len(raw), "=", n_dup, "+", len(bad), "+", len(good))  # 9 = 1 + 3 + 5
print(good["qty"].sum())  # 12
Path("output").mkdir(exist_ok=True)
bad.to_csv("output/quarantine.csv", index=False, encoding="utf-8-sig")
```

원본 9행을 중복 1행, 격리 3행, 정상 5행으로 설명할 수 있다. 원본 주문 객체 8개와 상품 행 9개를 섞어 비교하면 건수 검증이 맞지 않는다. 완전 동일 행을 중복으로 처리하는 것은 이 실습의 규칙이며, 실제 데이터에서는 상품 항목 식별자와 업무 정의부터 확인해야 한다.

## 3. merge는 결합 관계와 누락을 함께 검증한다

주문 상품 표에서는 같은 상품 ID가 여러 번 등장하고, 상품 기준표에서는 ID가 고유해야 한다. 이 관계를 `validate="many_to_one"`으로 명시한다.

```python
products = pd.read_csv("data/products.csv")
joined = lines.merge(products, on="product_id", how="left",
                     validate="many_to_one", indicator=True)
unmatched = joined.loc[joined["_merge"] != "both",
                       ["order_id", "product_id", "_merge"]]
if not unmatched.empty:
    raise ValueError("상품 기준표에 없는 상품이 있습니다.")
assert len(joined) == len(lines)
```

상품 기준표에 같은 키가 두 개 있으면 주문 한 행이 여러 행으로 늘어날 수 있다. many_to_one 검증은 이를 예외로 알려 준다. 기준표에 없는 P009는 inner 결합에서 사라지지만, left 결합과 `indicator=True`를 사용하면 `left_only`로 확인할 수 있다.

원본 h09의 두 표현은 다음과 같이 바로잡아야 한다.

| 원본 표현 | 정리한 표현 | 이유 |
| --- | --- | --- |
| `products.iloc[0]`을 concat에 전달 | `products.iloc[[0]]` | Series가 아닌 한 행 DataFrame으로 중복 예제를 구성 |
| `check.loc["_merge"]` | `check.loc[check["_merge"] != "both", ...]` | 행 이름 선택이 아니라 열의 값으로 행을 필터링 |

일부 셀에서 의도적으로 MergeError를 발생시키므로 파일 전체를 연속 실행하는 방식과 셀별 오류 체험을 구분해야 한다.

## 4. 집계할 열과 결제 상태를 명시하기

```python
joined["amount"] = joined["qty"] * joined["unit_price"]
paid = joined.loc[joined["status"] == "paid"].copy()
summary = paid.groupby("category", as_index=False).agg(
    quantity=("qty", "sum"), revenue=("amount", "sum")
)
print(summary)
assert summary["revenue"].sum() == paid["amount"].sum()
```

열을 선택하지 않고 전체 표를 sum하면 문자열까지 이어 붙일 수 있다. 집계 대상과 방법을 명시하면 결과의 의미가 분명해진다.

| 카테고리 | 결제 수량 | 매출 |
| --- | ---: | ---: |
| 문구 | 6 | 21,000원 |
| 생활 | 4 | 48,000원 |
| 합계 | 10 | **69,000원** |

전체 상품 금액은 74,000원이며 그중 환불 상태의 금액은 5,000원이다. 여기서 매출은 paid 행의 합계로 정의했다. 결제와 환불의 상태 기준을 섞지 않아야 같은 데이터를 집계해도 수치가 일관된다.

## 5. 저장한 표와 그림을 다시 확인하기

```python
summary.to_csv("output/category_revenue.csv", index=False, encoding="utf-8-sig")
saved = pd.read_csv("output/category_revenue.csv")
assert saved.columns.tolist() == ["category", "quantity", "revenue"]
assert saved["revenue"].sum() == 69000
```

`index=False`를 빠뜨린 비교 출력에는 불필요한 인덱스 열이 추가된다. 파일이 만들어졌다는 사실뿐 아니라 다시 읽었을 때 열과 값이 의도대로 유지되는지도 확인한다.

![수업 실습에서 저장한 카테고리별 매출 그래프](/images/bootcamp/261007/revenue_font.png)

*첨부 output/revenue_font.png 원본. 문구 21,000원, 생활 48,000원으로 집계한 결과다.*

원본은 Windows의 `Malgun Gothic`을 지정했다. 다른 운영체제에서는 설치된 한글 글꼴을 선택해야 한다. 제목·축 이름·단위·금액 표기를 함께 넣으면 막대 높이가 무엇을 의미하는지 쉽게 읽을 수 있다. 원본 주석의 figsize는 실제로 `(너비, 높이)` 순서이며, 축 이름의 `cateogry`는 `category`로 고치면 된다.

## 6. 복습용 보완: 지저분한 주문을 같은 파이프라인에 연결하기

원본 h10의 마지막 종합 과제는 지시문만 남아 있다. 아래는 앞에서 만든 good와 products를 연결하는 보완 예제다. 원본 과제가 완료됐다고 해석하지 않는다.

```python
clean_lines = good.drop(columns="qty_num").copy()
clean_joined = clean_lines.merge(products, on="product_id", how="left",
                                validate="many_to_one", indicator=True)
if (clean_joined["_merge"] != "both").any():
    raise ValueError("미등록 상품을 확인해야 합니다.")
clean_joined["amount"] = clean_joined["qty"] * clean_joined["unit_price"]
clean_paid = clean_joined.loc[clean_joined["status"] == "paid"]
dirty_summary = clean_paid.groupby("category", as_index=False).agg(
    quantity=("qty", "sum"), revenue=("amount", "sum")
)
dirty_summary.to_csv("output/dirty_category_revenue.csv",
                     index=False, encoding="utf-8-sig")
print(len(bad), len(good), dirty_summary["revenue"].sum())  # 3, 5, 60000
```

정상 행의 문구 수량은 9개·매출 24,000원, 생활 수량은 3개·매출 36,000원으로 합계는 60,000원이다. 정상 자료의 69,000원과 다른 입력이므로 두 결과를 같게 맞추는 것이 목적은 아니다. 정제 이후 남은 행과 금액의 근거를 설명할 수 있어야 한다.

## 정리

이번 실습에서 연결한 흐름은 **읽기 → 펼치기 → 정제·격리 → 결합 검증 → 집계 → 저장 → 시각화**다. 단순히 오류 없이 실행하는 것에 더해 행의 단위, 누락·중복 처리 기준, 상품 기준표의 고유성, 집계 전후 금액을 확인하는 것이 핵심이었다.

첨부 requirements.txt는 pandas 3.0.5, Matplotlib 3.10.8, requests 2.33.1, ipykernel 7.3.0을 지정한다. 본문 수치는 pandas 2.2.3의 별도 실행 환경에서 첨부 데이터로 재집계했으며, 원본 가상환경 전체를 실행한 결과는 아니다. 선택 API 부록은 실습 지시문만 남아 있어 호출 성공 결과는 포함하지 않았다.
