---
layout: single
title: "Scraping & Crawling"
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

## 260908: Scraping & Crawling

### 인터넷 상에서 데이터 가져오기
Crawling: HTML 구조를 이용해 인터넷 페이지 상의 데이터를 가져오는 것
Selenium: 실제 사람인 것처럼 브라우저를 제어해 웹 페이지의 데이터를 가져오는 오픈소스 프레임워크

데이터를 짧은 시간에 과도하게 요청하면 로봇으로 인식되어 차단될 수 있으므로, 요청 간격과 사이트 정책을 지켜야 한다.


### 라이브러리 설치

```bash
pip install requests beautifulsoup4 lxml pandas openpyxl
```

pip: 패키지
requests: https 요청 보내기
beautifulsoup4: 파싱
lxml: xml과 html 처리
pandas: dataframe 구성
openpyxl: excel 파일 읽기 및 저장

### Crawling 기본 코드

```python
### 라이브러리 불러오기
import requests
import bs4 # beautiful soup4
import pandas as pd

### 버전 확인
print("requests: ", requests.__version__)
print("beautiful soup4: ", bs4.__version__)
print("pandas: ", pd.__version__)

### 사이트에서 데이터 가져오기
# 파이썬이 아닌 브라우저라고 인식시키는 코드
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
}

url = "https://quotes.toscrape.com/page/2/" # 특정 버전에 맞추기 위해 변수에 저장, 속성을 타입에 저장
res = requests.get(url, headers=headers, timeout=10)

# res = requests.get("https://quotes.toscrape.com/") 로 가능, 하지만 버전에 따라 안될 수도 있음
print(res.status_code) # 200은 요청이 정상 처리되었음을 의미
# 200: 정상 / 301&302: 진행 중 / 403: 차단 / 404: 오타, 삭제됨
# 429: 너무 빨리 긁음, time.sleep()으로 딜레이 추가하면 정상 작동 / 500: 서버 오류 / 503: 사이트 점검 중
print(res.text[:600])
```

### Crawling 완성형 코드

```python
import requests
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

def fetch(url, params=None, retries=3, delay=1.0):
    """URL을 요청해서 HTML 문자열을 돌려준다. 실패하면 재시도."""
    for attempt in range(1, retries + 1):
        try:
            res = requests.get(url, params=params, headers=HEADERS, timeout=10)
            res.raise_for_status()            # 4xx, 5xx면 예외 발생
            res.encoding = res.apparent_encoding
            return res.text
        except requests.exceptions.HTTPError as e:
            print("  [%d/%d] HTTP 에러: %s" % (attempt, retries, e))
        except requests.exceptions.Timeout:
            print("  [%d/%d] 시간 초과" % (attempt, retries))
        except requests.exceptions.RequestException as e:
            print("  [%d/%d] 요청 실패: %s" % (attempt, retries, e))
        time.sleep(delay * attempt)           # 재시도할수록 더 오래 쉼
    return None


# 사용
html = fetch("https://quotes.toscrape.com/")
if html:
    print("성공! 길이:", len(html))
else:
    print("최종 실패")
```

### requests & BeautifulSoup으로 데이터 파이프라인 구축

```python
import requests
from bs4 import BeautifulSoup

response = requests.get("https://quotes.toscrape.com/")
html = response.text
soup = BeautifulSoup(html, "lxml")

# html.parser: 파이썬 내장 -> 보통 속도
# lxml: 라이브러리 -> 빠른 속도
# html5lib: 가장 호환성이 좋지만 속도가 제일 느림

print(soup.prettify()[:100]) # 들여쓰기 문법
print(soup.title)            # title 태그
print(soup.title.text)       # title 태그 내 문자
print(soup.h1)               # 첫번째 header 태그
print(soup.h1.text)          # 첫번째 header 태그 내 문자
print(soup.a['href'])        # 첫번째 a 태그 내 href 속성
print(soup.a['style'])       # 첫번재 a 태그 내 style 속성

# find(): 태그 한 개를 가져오되, 조건에 맞는 첫 번째를 기준으로 가져옴
# find_all(): 리스트를 가져오되, 조건에 맞는 전부를 가져옴
# 리스트로 가져오는 이유: 가변형 & 데이터를 가져와서 원하는 데로 다루기 위해
# select(): css 선택자, 선택자에 맞는 전부
# select_one(): css 선택자, 선택자에 맞는 첫 번째

print(soup.find("h1"))
print("------------------------------")
print(soup.find_all("a"))
print("------------------------------")
print(soup.find("div", class_="quote")) # class: python 내장 함수 / class_ 로 써야 작동함
print("------------------------------")
print(soup.find("div", id="main"))
print("------------------------------")
print(soup.find_all(["h1", "h2", "h3"]))
print("------------------------------")
print(soup.find_all("a", limit=5))      # limit=n: 개수 제한
print("------------------------------")
print(soup.find_all(string="GoodReads.com"))  # 글자 기준으로 탐색


### 정규표현식
import re
print("------------------------------")
print(soup.find_all("a", href=re.compile(r"/author/")))

### lambda 함수: 익명함수, 동작
print("------------------------------")
print(soup.find_all(lambda tag:tag.name == "a" and len(tag.text)>10))
```

### 크롤링 예문 1

```python

import requests
from bs4 import BeautifulSoup

response = requests.get("https://quotes.toscrape.com/")
html = response.text
soup = BeautifulSoup(html, "lxml")

# 한 개만 출력
q = soup.select_one("div.quote")
print("본문: ", q.select_one("span.text").get_text(strip=True)) # get_text() : 해당하는 모든 텍스트 불러옴 / strip=True: "" 내 글자만 가져옴
print("저자: ", q.select_one("small.author").get_text(strip=True))
print("태그: ", [t.get_text(strip=True) for t in q.select("a.tag")])


# 여러 개 출력
quotes = soup.select("div.quote")
for q in quotes:
    text = q.select_one("span.text").get_text(strip=True)
    author = q.select_one("small.author").get_text(strip=True)
    tag = [t.get_text(strip=True) for t in q.select("a.tag")]

    print("본문: ", text)
    print("저자: ", author)
    print("태그: ", tag)
    print("--------------------------")


# 모든 페이지 출력

import time, requests
from bs4 import BeautifulSoup

BASE = "https://quotes.toscrape.com/page/{}/"
all_data = []

for page in range(1,100): # 총 페이지(상수형)
    url = BASE.format(page)
    res = requests.get(url, timeout=10)
    soup = BeautifulSoup(res.text, "lxml")
    quotes = soup.select("div.quote")

    if not quotes:
        print(f"total page: {page}")
        break

    for q in quotes:
        all_data.append({
            "text": q.select_one("span.text").get_text(strip=True),
            "author": q.select_one("small.author").get_text(strip=True),
            "tag": ",".join(t.get_text(strip=True) for t in q.select("a.tag"))
        })
    time.sleep(0.5) # 딜레이 추가

print(all_data)
```


### 크롤링 예문 2

```python

import requests, time
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

FEEDS = {
    "연합뉴스-전체": "https://www.yna.co.kr/rss/news.xml",
    "한겨례-사회":   "https://hani.co.kr/rss/society/",
    "경향-전체":     "https://www.khan.co.kr/rssdata/total_news.xml"
}

rows=[]

for press, url in FEEDS.items(): # items(): key와 값을 모두 출력
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.encoding = res.apparent_encoding # 터미널 출력 시 방지
        soup = BeautifulSoup(res.text, "xml")

        items = soup.find_all("item")
        for item in items:
            rows.append({
                "언론사":press,
                "제목":item.find("title").get_text(strip=True),
                "링크":item.find("link").get_text(strip=True),
            })
        print("OK", press, len(items), "건")
    except Exception as e: # 모든 예외상황인 경우
        print("Crawling Failed", press, e)
    time.sleep(0.5)

df = pd.DataFrame(rows)
print(df.head(10))
df.to_csv("news_rss", index=False, encoding="utf-8-sig")
```
