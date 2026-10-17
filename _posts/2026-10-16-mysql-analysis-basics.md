---
layout: single
title: "MySQL 수업 정리: 테이블 설계에서 조인·집계·검증까지"
date: 2026-10-16 17:00:00 +0900
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

## 수업 자료의 범위

`sql.zip`의 SQL 39개 파일을 바탕으로 MySQL 실습을 정리했다. 카페의 지점·메뉴·회원·주문·주문 상세를 이용해 스키마 생성, 데이터 변경, 조회·집계, 관계 검증과 운영 기초를 학습하는 구성이다. 파일별 수업 날짜가 없어 **10월 16일 통합 정리 글**로 기록하며 각 STEP을 특정 날짜의 수업으로 배정하지 않았다.

| 구간 | 자료 | 학습 목표 |
| --- | --- | --- |
| 접속·설계 | 03~05 | 연결 확인, DB·테이블과 자료형 |
| 입력·변경 | 06~08 | INSERT, UPDATE·DELETE, 트랜잭션, 연습 데이터 적재 |
| 조회·분석 | 09~14 | SELECT·WHERE·함수·GROUP BY·JOIN·서브쿼리 |
| 무결성·운영 | 15~20 | 제약·ALTER·인덱스·뷰·CSV·권한·백업 |
| 연습 | test 10개, 연습 7개, 마무리 과제 | 조건별 조회와 분석 질문 복습 |
| 확장 | 윈도 함수·CTE 부록 | 순위·누적합·단계별 질의 구성 |

마무리 과제는 질문만 남아 있다. 제출·완료한 답안으로 소개하지 않으며 본문의 질의는 학습 내용을 설명하는 예제다.

## 1. 접속 확인과 테이블의 역할

```sql
SELECT VERSION(), NOW(), DATABASE();
SHOW DATABASES;
USE starcoffee;
SHOW TABLES;
SHOW CREATE TABLE orders;
```

서버 버전·시각과 현재 DB를 먼저 확인한다. 자료는 `utf8mb4_0900_ai_ci`와 MySQL 8 계열 문법을 사용한다. 다른 DBMS에서 그대로 실행된다고 가정하지 않는다.

| 테이블 | 한 행의 의미 | 식별 기준 |
| --- | --- | --- |
| store | 지점 한 곳 | store_id |
| menu | 메뉴 한 종류 | menu_id |
| member | 회원 한 명 | member_id |
| orders | 주문 한 건 | order_id |
| order_item | 주문에 포함된 메뉴 항목 | order_id + seq |

금액에는 고정 소수점인 DECIMAL, 수량·식별 번호에는 INT, 날짜에는 DATE·DATETIME을 사용한다. 주문 상세는 한 주문에 여러 행이 연결되므로 주문과 같은 단위가 아니다.

첨부 INSERT 구문을 SQLite의 별도 메모리 DB에 읽어 행 수와 기본 합계를 대조한 결과 회원 **120명**, 주문 **800건**, 주문 상세 **1,516행**이었다. 자료 주석의 ‘약 1,900행’보다 실제 INSERT 내용을 기준으로 한다. MySQL 서버 전체 실습이나 실행계획을 재현한 검증은 아니다.

## 2. 변경하기 전에 같은 조건으로 조회하기

```sql
SELECT menu_id, menu_nm, price FROM menu WHERE menu_id = 303;
START TRANSACTION;
UPDATE menu SET price = 3500 WHERE menu_id = 303;
SELECT menu_id, menu_nm, price FROM menu WHERE menu_id = 303;
ROLLBACK;
```

UPDATE·DELETE는 대상 조건을 먼저 SELECT로 확인한다. ROLLBACK 예제는 트랜잭션을 지원하는 엔진에서 커밋 전 변경을 되돌리는 경우다. 이미 커밋된 변경이나 MySQL의 암묵적 커밋을 일으키는 DDL까지 되돌린다는 뜻은 아니다.

DELETE는 조건에 맞는 행을 삭제하고, TRUNCATE는 테이블을 비우며, DROP은 테이블 자체를 제거한다. 속도 차이만으로 선택하지 않고 삭제 범위·참조 관계·트랜잭션 특성을 함께 본다. 첨부 08은 기존 연습 행을 비우고 다시 넣으며 99는 DB를 삭제하는 초기화 자료이므로 조회 예제와 구분한다.

## 3. WHERE와 NULL을 정확히 다루기

```sql
SELECT menu_nm, price
FROM menu
WHERE category IN ('커피', '디저트') AND price >= 5000
ORDER BY price DESC, menu_id;

SELECT member_id, member_nm FROM member WHERE email IS NULL;
SELECT COUNT(*) AS 전체, COUNT(member_id) AS 회원주문 FROM orders;
```

NULL은 일반 값처럼 등호로 비교하지 않는다. `email = NULL`은 참·거짓의 비교가 아니라 UNKNOWN을 만들며, WHERE는 TRUE인 행만 남긴다. 원본 주석의 ‘항상 거짓’은 SQL의 삼치 논리로 바로잡아 이해해야 한다. COUNT(*)는 행 수, COUNT(열)는 해당 열이 NULL이 아닌 행 수다.

날짜·시각 범위는 종료 경계가 다음 기간의 시작인 반개구간으로 작성하면 해당 월의 마지막 시각을 따로 계산하지 않아도 된다.

```sql
SELECT order_id, order_dt, total_amt
FROM orders
WHERE order_dt >= '2026-06-01' AND order_dt < '2026-07-01';
```

## 4. 집계 기준과 매출의 정의

WHERE는 묶기 전 행을 고르고, HAVING은 묶은 결과를 고른다. 매출이라고 이름 붙이기 전에 취소 여부와 기간을 명시한다.

```sql
SELECT store_id, COUNT(*) AS 주문수, SUM(total_amt) AS 매출
FROM orders
WHERE status = 'PAID'
GROUP BY store_id
HAVING SUM(total_amt) >= 2500000
ORDER BY 매출 DESC;
```

첨부 주문 INSERT 데이터의 PAID는 **752건**, 금액 합은 **9,662,500원**이다. 전체 상태 합계는 10,234,500원으로 다르다. 이는 초기 적재 구문의 점검 결과이며 뒤 실습에서 UPDATE·DELETE를 실행한 DB의 현재 값과는 다를 수 있다.

문자열 함수 LENGTH는 바이트 수, CHAR_LENGTH는 문자 수다. IFNULL·COALESCE는 결측 표시를 보완할 수 있지만 모르는 금액을 임의의 0으로 바꾸는 정책과는 구분한다.

## 5. JOIN 후 금액이 부풀려지는 이유

```sql
SELECT SUM(total_amt) AS 주문기준금액 FROM orders;

SELECT SUM(o.total_amt) AS 상세결합후금액
FROM orders o JOIN order_item i ON i.order_id = o.order_id;
```

같은 주문 금액이 상세 항목 수만큼 반복된다. 첨부 자료를 별도 집계하면 두 값은 **10,234,500원 대 24,050,500원**이다. 더 큰 값이 새로운 매출을 의미하지 않는다.

주문 건수는 주문 테이블이나 COUNT(DISTINCT order_id)로, 상세별 판매량과 금액은 상세 테이블에서 계산한다. 여러 상세·결제 표를 함께 연결한다면 업무 키별로 먼저 집계한 후 조인하는 것이 필요하다. SUM(DISTINCT 금액)은 서로 다른 주문의 같은 금액까지 합쳐 버려 일반적인 해결책이 아니다.

```sql
SELECT mn.menu_id, mn.menu_nm, SUM(i.qty) AS 판매수량, SUM(i.amt) AS 매출
FROM order_item i
JOIN orders o ON o.order_id = i.order_id
JOIN menu mn ON mn.menu_id = i.menu_id
WHERE o.status = 'PAID'
GROUP BY mn.menu_id, mn.menu_nm;
```

INNER JOIN은 대응하지 않는 행을 제외한다. LEFT JOIN은 왼쪽 행을 남기지만 오른쪽 키가 여러 개면 행이 늘 수 있다. LEFT JOIN 뒤 오른쪽 표 조건을 WHERE에 넣으면 미매칭 행을 제거할 수 있다는 점도 확인한다.

## 6. 서브쿼리에서 NOT IN과 NOT EXISTS 구분하기

```sql
SELECT m.member_id, m.member_nm
FROM member m
WHERE NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.member_id = m.member_id
);
```

비회원 주문 때문에 orders.member_id에는 NULL이 있다. 이를 제거하지 않은 NOT IN은 예상과 달리 결과를 제외할 수 있다. NOT EXISTS는 현재 회원과 연결되는 주문이 존재하는지 확인한다. `SELECT 1`의 1은 반환 결과를 0·1로 바꾸는 연산이 아니라, 존재 확인에 필요한 상수 표현이다.

## 7. 제약조건과 업무 규칙은 서로 보완한다

PRIMARY KEY는 식별자의 고유성과 비결측을, FOREIGN KEY는 참조 대상의 존재를 검증한다. 선택적 외래키의 NULL은 허용될 수 있으므로 NOT NULL 여부도 별도 확인한다. UNIQUE, DEFAULT, AUTO_INCREMENT 역시 다른 목적의 제약이다.

외래키가 모두 맞아도 진료 시작보다 종료가 빠르거나 금액 합계가 맞지 않을 수 있다. 구조적 참조 무결성과 업무 의미의 정합성을 나누어 검사해야 한다. 이것이 [외래 EMR 프로젝트]({{ '/projects/medallion-emr/' | relative_url }})의 시간·수가·청구 대사와 연결되는 부분이다.

## 8. 인덱스·뷰·CTE·윈도 함수

```sql
EXPLAIN SELECT * FROM orders
WHERE store_id = 1
  AND order_dt >= '2026-06-01' AND order_dt < '2026-07-01';
```

인덱스는 조회를 돕지만 쓰기와 저장 비용도 발생한다. `(store_id, order_dt)`처럼 조건에 맞는 복합 인덱스를 검토하고 실제 EXPLAIN으로 선택 여부를 확인한다. 함수로 감싼 날짜 조건은 일반 컬럼 인덱스의 범위 탐색을 어렵게 할 수 있으나, 인덱스가 있다고 항상 사용되는 것도 아니다. 본문에서 실행계획 변화나 속도 향상을 실측 성과로 제시하지 않는다.

일반 VIEW는 SELECT 정의에 이름을 붙인 것으로 결과를 별도로 저장하는 물리 테이블과 다르다. CTE는 복잡한 질의를 단계로 나누며, 윈도 함수는 GROUP BY처럼 행을 접는 대신 순위·누적합을 각 행에 붙인다.

```sql
WITH daily AS (
    SELECT DATE(order_dt) AS sales_date, SUM(total_amt) AS revenue
    FROM orders WHERE status = 'PAID'
    GROUP BY DATE(order_dt)
)
SELECT sales_date, revenue,
       SUM(revenue) OVER (ORDER BY sales_date) AS cumulative_revenue
FROM daily ORDER BY sales_date;
```

## 9. CSV·권한·백업까지 확인하기

CSV는 적재 성공 메시지뿐 아니라 전후 행 수·한글 인코딩·NULL 표기·열 대응을 확인한다. LOCAL INFILE은 서버 설정과 클라이언트 허용이 모두 필요하다. 원본의 외부 CSV 경로는 첨부에 없는 파일이므로 가져오기를 완료했다고 기록하지 않는다.

조회 작업에는 필요한 SELECT 권한을 부여하고 관리자 계정으로 모든 작업을 수행하는 방식을 피한다. 원본의 예시 비밀번호는 본문에 재사용하지 않는다. 백업 파일을 만드는 것과 복원 검증을 마치는 것은 다르므로 별도 DB에서 테이블·행 수·제약을 확인한다. 원본의 `< 파일.sql` 복원 예시는 PowerShell에서 그대로 작동하는 명령으로 간주하지 않고 CMD 리다이렉션 또는 mysql 클라이언트의 SOURCE 등 실행 환경에 맞는 방식을 사용한다.

## 다음 복습

마무리 과제의 판매 메뉴 조회, 월별·지점별 매출, 미주문 회원과 인기 메뉴를 직접 작성하고 원본 주문 기준 집계와 상세 집계를 대조한다. SQL을 작성하는 목적은 문법을 통과하는 것이 아니라 **질문에 맞는 단위와 분모로 재현 가능한 결과를 만드는 것**이다.
