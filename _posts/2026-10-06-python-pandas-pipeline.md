---
layout: single
title: "pandas 데이터 처리: 선택·정제·결합에서 매출 집계와 시각화까지"
date: 2026-10-06 17:00:00 +0900
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

## 10월 6일 학습 흐름

이전 수업에서는 중첩 JSON을 주문 상품 단위로 펼쳐 CSV로 저장했다. 이번에는 같은 데이터를 pandas로 읽고, 조건에 맞는 행 선택부터 오류 데이터 격리, 상품 정보 결합, 매출 집계와 그래프 저장까지 연결했다.

이 글은 h05~h10 실습 파일과 make_data.py를 바탕으로 정리했다. 원본의 오류 체험 셀은 원인을 설명하고, 실행을 막는 오타는 아래 예제에서 바로잡았다. 비어 있는 종합 과제는 **복습용 보완 예제**로 구분했다. API 부록은 실습 지시문만 있어 실제 호출 결과를 제시하지 않는다.

## 1. 실습 데이터와 한 행의 의미

먼저 make_data.py를 실행하면 data 폴더에 주문 JSON 두 종류와 상품 CSV가 만들어진다.

```bash
python make_data.py
```

| 파일 | 구성 | 용도 |
| --- | --- | --- |
| orders.json | 주문 객체 5개, 주문 상품 7행 | 선택·결합·매출 집계 |
| products.csv | 상품 3종 | 상품명·카테고리·단가 연결 |
| orders_dirty.json | 중복을 포함한 주문 객체 8개, 펼친 상품 9행 | 중복·결측·형식 오류 처리 |

주문 한 건에 상품이 여러 개 있을 수 있다. 따라서 주문 개수와 펼친 표의 행 수는 다르다. 이 표에서 한 행은 주문 자체가 아니라 **주문에 포함된 상품 항목 하나**다.

첨부 requirements(1).txt에는 pandas 3.0.5, Matplotlib 3.10.8, requests 2.33.1, ipykernel 7.3.0이 지정되어 있다. 아래 결과를 재검증한 환경은 pandas 2.2.3·Matplotlib 3.10.8이며 지정 환경 전체를 그대로 재현한 결과는 아니다.

## 2. CSV와 pandas의 자료형 차이

csv.DictReader는 일반적으로 CSV의 값을 문자열로 읽는다. 수량 "2"와 "1"을 그대로 더하면 "21"이 되므로 계산 전에 int로 변환해야 한다. pandas.read_csv는 열의 자료형을 추론하지만, 모든 숫자 모양 문자열을 반드시 정수로 바꾸는 것은 아니다. 결측이나 문자 혼합 여부에 따라 결과가 달라지므로 dtypes를 확인한다.

```python
import pandas as pd

df = pd.read_csv("data/order_lines.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head(3))
print(df.dtypes)
print(df.describe())
print(df["qty"].sum(), df["qty"].mean())
```

order_lines.csv는 h05의 펼치기·CSV 저장 단계에서 만든다. 원본의 파일 확인 경로 `ordeR_lines.csv`는 `order_lines.csv`로 고쳐야 대소문자를 구분하는 환경에서도 열린다.

| 표현 | 결과 |
| --- | --- |
| `df["qty"]` | 한 열을 나타내는 Series |
| `df[["order_id", "qty"]]` | 여러 열을 가진 DataFrame |
| `df["qty"].to_numpy()` | NumPy 배열 |

리스트에 2를 곱하면 원소가 반복되지만 NumPy 배열이나 숫자 Series에 2를 곱하면 각 값에 연산이 적용된다. 조건식도 원소별 True·False 결과를 만든다.

```python
import numpy as np

print([2, 1, 3] * 2)             # [2, 1, 3, 2, 1, 3]
print(np.array([2, 1, 3]) * 2)   # [4 2 6]
print(df["qty"] >= 2)
```

## 3. json_normalize로 중첩 JSON 펼치기

json_normalize에 주문 목록만 전달하면 customer의 키는 펼쳐지지만 items의 리스트는 한 칸에 남는다. 상품별 행을 얻으려면 record_path로 반복 대상을, meta로 각 행에 붙일 주문 정보를 지정한다.

```python
import json

with open("data/orders.json", encoding="utf-8") as file:
    orders = json.load(file)

meta = ["order_id", "ordered_at", "status",
        ["customer", "id"], ["customer", "city"]]
lines = pd.json_normalize(orders, record_path="items", meta=meta)
lines = lines.rename(columns={"customer.id": "customer_id", "customer.city": "city"})
lines.to_csv("data/order_lines.csv", index=False, encoding="utf-8")
print(lines.shape)  # (7, 7)
```

## 4. 조건 선택과 새 열 만들기

조건식으로 만든 불리언 마스크를 대괄호에 넣으면 True인 행만 남는다. loc는 행 조건과 열 목록을 함께 지정할 때 사용한다.

```python
df = pd.read_csv("data/order_lines.csv")
mask = df["status"] == "paid"
paid = df.loc[mask].copy()
print(df.loc[mask, ["order_id", "product_id", "qty"]])
print(df.loc[(df["status"] == "paid") & (df["qty"] >= 2)])
print(df.loc[df["city"].isin(["부산", "대전"])])

df["is_bulk"] = df["qty"] >= 2
price = {"P001": 5000, "P002": 2000, "P003": 12000}
df["unit_price"] = df["product_id"].map(price)
df["amount"] = df["qty"] * df["unit_price"]
```

여러 조건을 결합할 때는 `and`·`or` 대신 원소별 논리 연산자를 사용하고 각 비교식을 괄호로 감싼다. AND 연산자는 `&`, OR 연산자는 <code>&#124;</code>다. `map`은 상품 ID에 대응하는 가격을 붙이며, 사전에 없는 ID는 결측으로 남으므로 별도 확인이 필요하다.

```python
print(df["status"].value_counts())
print(df.loc[(df["city"] == "서울") & (df["status"] == "paid")]
        .sort_values("qty", ascending=False))
print((df["amount"] >= 10000).sum())                 # 3행
print(df.loc[df["status"] == "refunded", "amount"].sum())  # 5,000원
```

value_counts는 여기서 주문 상품 행을 센다. paid는 6행이지만 결제 주문은 4건이다. 주문 건수를 구하려면 order_id의 고유값 수를 확인해야 한다.

## 5. 오류 데이터는 삭제 전에 구분하기

### 없는 메타데이터와 형식 오류

dirty 자료에는 도시 키가 빠진 주문이 있어 기본 json_normalize 호출에서 KeyError가 발생한다. `errors="ignore"`는 없는 meta 키를 결측으로 남기지만, 모든 JSON 구조 오류를 무시하거나 잘못된 수량을 고쳐 주는 옵션은 아니다.

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

주문번호만으로 중복을 제거하면 O001에 포함된 서로 다른 상품 중 하나가 사라진다. 이 실습에서는 완전히 같은 행만 중복으로 처리한다. 실제 업무에서는 같은 상품을 여러 항목으로 기록할 수도 있으므로 명세와 항목 식별자를 확인한 뒤 중복 기준을 정해야 한다.

to_numeric의 coerce는 변환할 수 없는 값을 NaN으로, to_datetime은 잘못된 날짜를 NaT로 만든다. 원래 열을 남겨 두면 격리 사유를 추적하기 쉽다.

| 격리 주문 | 원본 값 | 사유 |
| --- | --- | --- |
| O004 | `qty=None` | 수량 누락 |
| O005 | `qty="두개"` | 숫자 변환 실패 |
| O006 | `ordered_at="2026-02-30"` | 존재하지 않는 날짜 |

모르는 수량을 0으로 채우면 실제 0과 구분되지 않고 평균도 바뀐다. 반면 도시에 "미상"을 넣는 것은 수량·매출을 임의로 만들어 내지 않는 표시 방식이다.

### 행 수 보존 확인

```python
from pathlib import Path

n_dup = len(raw) - len(step)
assert len(raw) == n_dup + len(bad) + len(good)
print(len(raw), "=", n_dup, "+", len(bad), "+", len(good))
Path("output").mkdir(exist_ok=True)
bad.to_csv("output/quarantine.csv", index=False, encoding="utf-8-sig")
```

실제 결과는 **펼친 원본 9행 = 중복 1행 + 격리 3행 + 정상 5행**이다. 원본 주문 객체 8개와 비교하는 식이 아니라 동일한 상품 행 단위로 비교한다. 정상 행의 수량 합계는 12개다.

## 6. merge에서 행 증가와 미등록 상품 점검하기

가격뿐 아니라 상품명·카테고리도 필요하다면 상품 표를 merge한다. 주문 상품에는 같은 product_id가 반복될 수 있지만 상품 표에서는 고유해야 하므로 many_to_one 관계를 검증한다.

```python
products = pd.read_csv("data/products.csv")
joined = lines.merge(products, on="product_id", how="left",
                     validate="many_to_one", indicator=True)
unmatched = joined.loc[joined["_merge"] != "both",
                       ["order_id", "product_id", "_merge"]]
if not unmatched.empty:
    raise ValueError("상품 표에 등록되지 않은 상품이 있습니다.")
```

상품 표의 키가 중복되면 결합 행 수와 매출이 부풀려질 수 있다. 실습에서 의도한 중복 행은 `products.iloc[[0]]`처럼 DataFrame으로 추가해야 한다. `products.iloc[0]`은 Series이므로 원하는 한 행 결합과 다르다.

```python
bad_products = pd.concat([products, products.iloc[[0]]], ignore_index=True)
try:
    lines.merge(bad_products, on="product_id", how="left", validate="many_to_one")
except pd.errors.MergeError:
    print("상품 표의 product_id 중복을 확인해야 합니다.")
```

inner 결합은 상품 표에 없는 P009 주문을 결과에서 제외한다. left 결합과 indicator를 사용하면 left_only로 남아 누락을 찾을 수 있다. 원본의 `check.loc["_merge"]`는 열이 아니라 행 이름을 조회하므로 `check.loc[check["_merge"] != "both", 열목록]`으로 고쳐야 한다. 관계 검증과 미매칭 검사는 서로 다른 문제를 확인한다.

## 7. 집계 대상과 집계 열 명시하기

```python
joined["amount"] = joined["qty"] * joined["unit_price"]
paid = joined.loc[joined["status"] == "paid"].copy()
summary = paid.groupby("category", as_index=False).agg(
    quantity=("qty", "sum"), revenue=("amount", "sum")
)
by_city = paid.groupby("city", as_index=False)["amount"].sum()
assert summary["revenue"].sum() == by_city["amount"].sum()
summary.to_csv("output/category_revenue.csv", index=False, encoding="utf-8-sig")
```

groupby 뒤 표 전체에 sum을 적용하면 문자열 열까지 결합되거나 자료형에 따라 오류가 발생할 수 있다. 집계 열과 함수를 명시하면 결과의 의미가 분명해진다.

| 카테고리 | 결제 수량 | 결제 매출 |
| --- | ---: | ---: |
| 문구 | 6 | 21,000원 |
| 생활 | 4 | 48,000원 |
| 합계 | 10 | 69,000원 |

전체 상태의 금액 합계는 74,000원, refunded 금액은 5,000원이다. 여기서는 paid만 매출로 집계했다. 다른 상태가 추가될 수 있으므로 환불 금액은 전체에서 paid를 빼기보다 refunded 조건으로 직접 구하는 편이 명확하다.

CSV 저장 시 `index=False`를 생략하면 행 인덱스도 저장된다. 이 파일을 다시 읽으면 Unnamed: 0 같은 불필요한 열이 생길 수 있다. utf-8-sig는 UTF-8 BOM을 포함하며 Excel 등에서 한글 인식에 도움이 될 수 있지만, 자료형을 보존하는 옵션은 아니다.

## 8. 집계 결과를 그래프로 확인하기

```python
import matplotlib.pyplot as plt

# Windows 예시. 다른 환경에서는 설치된 한글 글꼴을 지정한다.
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(7, 4))  # 너비, 높이(인치)
bars = ax.bar(summary["category"], summary["revenue"])
ax.bar_label(bars, labels=[f"{v:,.0f}원" for v in summary["revenue"]], padding=4)
ax.set(title="카테고리별 결제 매출", xlabel="카테고리", ylabel="매출(원)")
ax.set_ylim(0, summary["revenue"].max() * 1.2)
fig.tight_layout()
fig.savefig("output/category_revenue.png", dpi=160)
plt.close(fig)
```

figsize의 순서는 너비·높이다. 원본 주석의 높이·너비 표기를 바로잡았고, 데이터가 바뀌어도 막대가 잘리지 않도록 y축 상한을 매출 최댓값 기준으로 설정했다.

![실습 데이터의 카테고리별 결제 매출: 문구 21000원, 생활 48000원](/images/bootcamp/2026-10-06/category-revenue.png)

위 그림은 첨부 실습 데이터를 재실행해 만든 결과다. 검증 환경에는 한글 글꼴이 없어 Stationery는 문구, Household는 생활로 표기했다.

## 9. 복습용 보완: dirty 자료를 집계까지 연결하기

원본 h10 마지막 종합 과제는 비어 있다. 앞에서 만든 good을 이용해 결합과 집계를 이어가는 예제는 다음과 같다.

```python
clean = good.rename(columns={"customer.id": "customer_id", "customer.city": "city"})
if clean["order_id"].isna().any() or (clean["qty"] <= 0).any():
    raise ValueError("주문번호와 양수 수량을 확인해야 합니다.")
result = clean.merge(products, on="product_id", how="left",
                     validate="many_to_one", indicator=True)
if result["_merge"].ne("both").any():
    raise ValueError("미등록 상품이 있습니다.")
result["amount"] = result["qty"] * result["unit_price"]
dirty_summary = result.loc[result["status"].eq("paid")].groupby(
    "category", as_index=False
).agg(quantity=("qty", "sum"), revenue=("amount", "sum"))
dirty_summary.to_csv("output/dirty_category_revenue.csv", index=False, encoding="utf-8-sig")
print(dirty_summary)
```

| 카테고리 | 수량 | 매출 |
| --- | ---: | ---: |
| 문구 | 9 | 24,000원 |
| 생활 | 3 | 36,000원 |
| 합계 | 12 | 60,000원 |

dirty 자료는 원래 주문에 오류만 추가한 완전히 동일한 거래 집합이 아니다. 상태·수량·주문 구성도 달라서 69,000원과의 차이를 정제 손실로 해석하면 안 된다.

또한 이 실습의 정상 분류는 수량·날짜 변환 가능 여부를 기준으로 한다. 일반 데이터에서는 정수 수량 여부, 양수 범위, 상품 단가와 상태값까지 추가 검증해야 한다. 원본의 `qty <= 0` 검사만으로는 결측이나 모든 잘못된 자료형을 잡을 수 없다.

## 10. 선택 API 실습과 핵심 정리

hA_api_optional.py는 요청, 상태 확인, JSON 변환과 저장을 연습하는 선택 과제다. 구현된 호출이 없으므로 API 연동을 완료한 것으로 정리하지 않았다. 요청 실패 여부를 확인한 뒤 JSON으로 해석하고 원본 응답을 저장한다는 흐름이 기존 파일 처리와 연결된다.

이번 실습의 핵심은 **한 행의 의미를 정하고, 변환·결합 단계에서 데이터가 늘거나 사라지는 이유를 확인한 뒤 집계하는 것**이었다. 숫자가 계산된다는 사실만으로 결과가 올바른 것은 아니다. 중복·격리 건수, 상품 키의 고유성, 미매칭 행과 집계 합계가 맞는지 함께 확인해야 한다.
