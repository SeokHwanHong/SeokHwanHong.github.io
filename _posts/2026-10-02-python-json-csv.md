---
layout: single
title: "Python 데이터 분석 기초: JSON에서 CSV와 매출 집계까지"
date: 2026-10-02 17:00:00 +0900
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

## 10월 2일 학습 흐름

가상환경을 설정하고 Python 딕셔너리와 JSON 문자열의 차이를 확인했다. 이후 JSON 파일을 읽어 중첩 구조를 탐색하고, 주문 상품 단위의 표로 펼쳐 CSV 저장과 매출 계산까지 연결했다.

이 글은 수업 메모와 h00~h05 실습 파일을 바탕으로 정리했다. 원본의 빈 실습 구간은 완료한 것으로 간주하지 않았다. requirements.txt, make_data.py, 주문 JSON과 products.csv는 첨부되지 않아 실제 수업 데이터의 건수·집계 결과는 제시하지 않는다. 아래 주문과 가격은 실행 흐름을 설명하기 위해 추가한 예제다.

## 1. 가상환경과 셀 실행

Windows에서 Python 3.13 가상환경을 생성한다.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

두 번째 명령은 requirements.txt를 별도로 받은 경우에 사용한다. 수업의 환경 확인 코드에서는 pandas, NumPy, Matplotlib을 불러오지만, 아래 JSON·CSV 핵심 예제는 표준 라이브러리만 사용한다.

```python
# %% 환경 확인
import sys
from pathlib import Path

print(sys.version.split()[0])
print(Path.cwd())

# 별도로 설치한 패키지의 버전 확인
import pandas as pd
import numpy as np
import matplotlib

print(pd.__version__, np.__version__, matplotlib.__version__)
```

VS Code에서 가상환경 인터프리터를 선택한다. Python·Jupyter 확장이 준비된 환경에서는 .py 파일의 `# %%`로 셀을 나누고 Run Cell로 실행할 수 있다. Shift+Enter의 동작은 실행 위치와 설정에 따라 달라질 수 있다.

파일을 수정했다고 이미 실행된 변수까지 자동으로 바뀌지는 않는다. 변경한 셀과 그 값에 의존하는 후속 셀을 다시 실행해야 한다. 이것은 객체지향을 절차지향으로 바꾸는 것이 아니라 실행 상태를 갱신하는 문제다.

## 2. 딕셔너리와 JSON 문자열

```python
import json

order = {
    "order_id": "O001",
    "city": "서울",
    "paid": True,
    "memo": None,
    "items": ["노트", "펜"],
}
print(order["city"])
print(order["items"][1])

python_text = str(order)
json_text = json.dumps(order, ensure_ascii=False, indent=2)
print(type(python_text))  # str
print(json_text)

restored = json.loads(json_text)
assert restored == order
```

`str(order)`는 Python 표현을 문자열로 만든다. JSON 규칙에 맞는 직렬화에는 `json.dumps`를 사용한다.

| Python 값 | JSON 표현 |
| --- | --- |
| True / False | true / false |
| None | null |
| 딕셔너리 | 객체 |
| 리스트 | 배열 |

`ensure_ascii=False`는 한글을 이스케이프하지 않고 표시하도록 한다. 기본값에서 보이는 유니코드 이스케이프는 한글이 손상된 것이 아니다. `indent=1`과 `indent=2`는 각각 들여쓰기 폭을 지정한다.

문자열은 딕셔너리처럼 키로 접근할 수 없다. `json_text[1:11]`은 문자열 슬라이싱이며 중첩 데이터를 펼치는 작업과는 다르다.

### 의도적인 오류 확인

```python
try:
    json.loads(python_text)
except json.JSONDecodeError as error:
    print("Python 표현 문자열은 유효한 JSON이 아닙니다:", error.msg)
```

수업의 오류 실습을 그대로 전체 실행하면 이후 코드가 멈추므로, 글에서는 예외를 잡아 원인을 확인하도록 구성했다.

## 3. JSON 파일 저장과 읽기

| 함수 | 대상 |
| --- | --- |
| json.dumps / json.loads | 문자열로 변환 / 문자열에서 읽기 |
| json.dump / json.load | 파일 객체에 저장 / 파일 객체에서 읽기 |

```python
from pathlib import Path

data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)
order_path = data_dir / "my_order.json"

with order_path.open("w", encoding="utf-8") as file:
    json.dump(order, file, ensure_ascii=False, indent=2)

with order_path.open("r", encoding="utf-8") as file:
    loaded = json.load(file)

assert loaded == order
print(file.closed)  # True
```

상대 경로는 현재 작업 디렉터리를 기준으로 해석한다. `with` 블록을 벗어나면 파일이 닫히며, file 또는 f는 파일 객체를 가리키는 변수명이다. `print(f.close)`는 닫힘 여부가 아니라 메서드 객체를 출력한다.

저장과 읽기의 인코딩을 맞추는 것도 중요하다. UTF-8 파일을 CP949로 읽거나 그 반대로 읽으면 오류 또는 잘못된 문자 해석이 발생할 수 있다.

## 4. 중첩 구조 탐색

다음은 설명용 주문 두 건이다. 이후 코드도 이 데이터를 이어서 사용한다.

```python
orders = [
    {
        "order_id": "O001", "status": "paid",
        "customer": {"id": "C001", "city": "서울"},
        "items": [
            {"product_id": "P001", "qty": 2},
            {"product_id": "P002", "qty": 1},
        ],
    },
    {
        "order_id": "O002", "status": "pending",
        "customer": {"id": "C002", "city": "부산"},
        "items": [{"product_id": "P001", "qty": 3}],
    },
]
first = orders[0]
print(first["customer"]["city"])
print(first["items"][0]["product_id"])
print(first["customer"].get("phone", "없음"))

for order in orders:
    print(order["order_id"], order["status"], len(order["items"]))
```

바깥이 리스트이면 인덱스로 주문을 선택한 뒤 딕셔너리의 키로 내려간다. `len(dict)`는 키 개수이고, `len(list)`는 원소 개수이므로 파일을 읽은 직후에는 자료형부터 확인해야 한다.

원본에는 order.json과 orders.json이 혼재한다. 실제 파일을 사용할 때는 이름을 통일하고, 최상위 구조가 주문 리스트인지 확인한다. `dict.get(key, default)`의 기본값은 키가 없을 때 사용된다. 키가 존재하고 값이 None이면 None이 그대로 반환된다.

## 5. 중첩 데이터를 표로 펼치기

먼저 ‘한 행이 무엇인가’를 정한다. 이번 표에서는 한 행이 주문 전체가 아니라 **주문 안의 상품 항목 한 줄**이다.

```python
rows = []
for order in orders:
    for item in order["items"]:
        rows.append({
            "order_id": order["order_id"],
            "status": order["status"],
            "customer_id": order["customer"]["id"],
            "city": order["customer"]["city"],
            "product_id": item["product_id"],
            "qty": item["qty"],
        })

order_ids = set()
total_qty = 0
paid_qty = 0
paid_by_product = {}

for row in rows:
    order_ids.add(row["order_id"])
    total_qty += row["qty"]
    if row["status"] == "paid":
        paid_qty += row["qty"]
        product_id = row["product_id"]
        paid_by_product[product_id] = (
            paid_by_product.get(product_id, 0) + row["qty"]
        )

print(len(rows), len(order_ids))  # 상품 항목 3행, 주문 2건
print(total_qty, paid_qty)       # 전체 수량 6, 결제 완료 수량 3
print(paid_by_product)           # {'P001': 2, 'P002': 1}
```

상품 항목 수, 주문 수, 상품 수량 합계는 서로 다르다. 주문 ID가 주문별로 고유하다는 전제에서 고유 ID 수로 주문 건수를 센다. 리스트를 items 열에 그대로 넣는 방식도 목적에 따라 가능하지만, 상품별 집계가 목표라면 안쪽 반복문으로 펼치는 편이 적합하다.

원본 주석에는 7개 열이라고 되어 있지만 실제 구성은 6개 열이다. 이 글에서는 ordered_at을 포함하지 않고 저장 컬럼도 6개로 맞췄다. 또한 내장 함수 `sum`을 변수명으로 덮어쓰지 않도록 합계 변수명을 구분했다.

## 6. CSV 저장과 읽기

```python
import csv

csv_path = data_dir / "order_lines.csv"
fieldnames = [
    "order_id", "status", "customer_id", "city", "product_id", "qty"
]
with csv_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with csv_path.open("r", encoding="utf-8", newline="") as file:
    table = list(csv.reader(file))
print(table[0])  # 헤더

with csv_path.open("r", encoding="utf-8", newline="") as file:
    lines = list(csv.DictReader(file))
print(type(lines[0]["qty"]))  # str

restored_qty = 0
for line in lines:
    restored_qty += int(line["qty"])
assert restored_qty == total_qty
```

`csv.reader`는 각 행을 리스트로 읽으며 헤더도 첫 행에 포함한다. `DictReader`는 헤더를 키로 사용한다. 기본 설정에서는 숫자도 문자열로 읽으므로 수량 계산 전에 형 변환이 필요하다.

`type(print(lines[0]))`는 행의 자료형이 아니라 print의 반환값인 None의 자료형을 확인한다. 원하는 값에 직접 type을 적용해야 한다.

`newline=""`는 csv 모듈이 줄바꿈을 처리하도록 하기 위한 설정이다. Excel에서 한글 인식이 필요한 경우 UTF-8 BOM을 추가하는 `utf-8-sig`를 사용할 수 있다. BOM의 첫 세 바이트는 `EF BB BF`이며, 일반 UTF-8 저장에는 붙지 않는다.

## 7. 상품 가격을 연결해 매출 계산

실제 수업에서는 products.csv를 읽어 가격 딕셔너리를 만든다. 해당 파일이 없어 여기서는 설명용 가격을 사용한다.

```python
products = [
    {"product_id": "P001", "unit_price": "1000"},
    {"product_id": "P002", "unit_price": "2000"},
]
price_by_product = {}
for product in products:
    price_by_product[product["product_id"]] = int(product["unit_price"])

revenue = 0
for line in lines:
    if line["status"] == "paid":
        product_id = line["product_id"]
        revenue += int(line["qty"]) * price_by_product[product_id]

assert revenue == 4000
print(revenue)
```

이 예제의 4,000원은 실제 수업 데이터의 결과가 아니다. 상품별 가격이 하나이고, 결제 완료 주문에 그 가격을 적용한다는 단순한 가정이다. 실제 매출 데이터에서는 주문 당시 가격, 할인, 환불과 가격 정보 누락도 확인해야 한다.

## 정리

JSON을 표로 바꾸기 전에 자료형과 중첩 구조를 확인하고, 분석할 행의 단위를 먼저 정해야 한다. 상품 단위로 펼친 뒤에는 행 수를 주문 건수로 오해하지 않도록 주의한다. CSV로 저장했다 다시 읽을 때는 숫자의 자료형이 달라질 수 있으므로, 변환 후 원래 합계와 비교하는 검증까지 이어간다.
