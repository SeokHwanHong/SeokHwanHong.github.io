---
layout: single
title: "Python 기초 문법과 자료구조"
date: 2026-09-03 09:00:00 +0900
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

> 수업 날짜: 2026-09-03

## 학습 주제

- 출력·형 변환·기본 연산
- 내장 함수와 문자열 포맷
- list·tuple·set·dictionary와 인덱싱

## 수업 기록

### Python 특징
1. 확장성이 용이, Python 을 이용해 여러 작업 진행
2. 직관적, 다른 언어에 비해 쉬움
3. 객체지향언어

### 
visual studio code 의 우측 인터페이스 agent는 gemini 기반
터미널: 코드를 실행시키 위한 포트
콘솔: 뷰 포트
powershell 은 window 기반이 아닌데 11부터 들어옴
window: linux & dos 섞어서 씀
확장에서 code runner 설치하면 여러 언어 및 에이전트 연동 용이
파이썬 디버그는 F5

### 예문

```
print("Hello, World!") # 강제 개행
print('ok bye~~')
print("가", end="") # 줄넘김 없이 출력
print("나다")
print(5); print(3.0); print(5+3.0) # 자동으로 자료형 변환
# print('1'+1) # 문자열과 수치형은 연산 불가
print('1'+'1') # 문자열+문자열 -> 문자열+문자열 자체로 출력, 공백없음(11)
print('1', '1') # 문자열+문자열 -> 문자열+문자열 자체로 출력, 공백발생(1 1)
print('안녕\n하세요') # 문자 내 이스케이프 문자 삽입 가능
```

### 기본 연산
```
a = 5; b = '5'
print(a + int(b)) # 정수로 변환
print(str(a) + b) # 문자로 변환

print(3+2)      # 합
print(3-2)      # 차
print(3*2)      # 곱
print(3/2)      # 나누기(실수형)
print(3//2)     # 몫(정수형)
print(3%2)      # 나머지(정수형)
print(3 ** 2)   # 제곱
print(3/2 == 1) # False

print(True==1) # T/F: C는 0&1, java는 boolen, python은 0&1
print((3//2==1) + 2) # 3으로 반환
```

### 기본 연산 함수
```
print("최댓값: ", max(1,2,3,4))
print("최솟값: "+ str(min(1,2,3,4)))
print("절댓값: ", abs(-20))
print("제곱: ", pow(2,4))
print("반올림: ", round(3.141592, -1))
print("반올림: ", round(3.141592, 5))
print("몫&나머지: ", divmod(10,7))

from math import * 
print("올림: ", ceil(3.14))
print("내림: ", floor(3.14))
print("제곱근: ", sqrt(16))  # 반환값 실수형

from random import * 
print(random())         # 0.0~1.0 사이의 실수
print(random()*10)      # 0.0~10 사이의 실수
print(int(random()*10)) # 0.0~10 사이의 정수
print(randrange(1,10))  # 1~10(미만) 사이의 정수
print(randint(1,10))    # 1~10(이하) 사이의 정수
```

### print 형식
```
# 1. c언어 형식
print("올해는 %d년 입니다." %2025)
print("저의 장래희망은 %s 입니다." %"지구정복")
print("올해 %d년은 %s의 해 입니다." %(2026, "병오년"))
print("%.2f" %3.1234566)


# 2. 포맷 형식: jQuery 기반
print("좋아하는 동물은 {}와 {}입니다.".format('강아지', "고양이"))
print("좋아하는 동물은 {1}와 {0}입니다.".format('강아지', "고양이")) # format argument 주소 지정 가능

동물1 = '사자'; 동물2 = '호랑이' # 한글 변수 가능
print(f"동물원에는 {동물1}과 {동물2}가 있습니다.")
```

### Python 자료형
1. list: [], 배열과 유사, 순서가 존재, 가변형(mutable)
2. tuple: (), 순서가 존재, 불변형(immutable)
3. set(집합): {}, 순서 없음, 중복 허용 안됨 
4. dictionary(사전): {키:값}

### list 예문
```
hobby = ["야구", "축구", 1234, True, print("Hello")]
# print(hobby)
# - 출력 결과
# Hello
# ['야구', '축구', 1234, True, None]
# list보다 print의 우선순위가 높음

print(hobby[1])
# - 출력 결과
# Hello
# 축구
# 우선순위에 의해 hobby[-1] 자동적으로 실행, 이후 hobby[1] 출력

print(type(hobby))

# hobby.append(["농구"]) # 이중 리스트는 그대로 들어감
hobby.append("농구")   # 적재방식: 알 수 없음, 나오는 방식에 따라 결정
print(hobby)

hobby.pop() # pop() 이 존재는 함, 하지만 무조건 stack은 아님
print(hobby)

# python 은 배열의 크기가 가변, 다른 언어는 불변

hobby.remove("축구") # remove() 는 객체(원소)를 지정해야함
print(hobby)
```

### set, dict 예문
```
s1 = (1,3,2,5,4)
print(s1) # 주소 번호 호출

s1 = s1 + (10,11)
print(s1) # 튜플 형태로는 삽입 가능

# set: 순서 x, 줄복하용x
s2 = {4.,5,5,5,5,5,6,6,7,7,7,8,8,8,8,8}
# 숫자만 장렬
print(s2)

# 출력할 때마다 순서 변경
s3 = {"베트남", "홍콩", "베트남", "서울", "일본"}
print(s3)

# 추가 시에도 위치 무작위
s3.add("스페인") 
print(s3)

# 보통 remove를 이용해 원소 제거
s3.remove("스페인")
print(s3)
```

### 형질변환
```
# 형질 변환
a = 5
print(type(a))

str(a)
print(type(a))
# a의 타입을 저장하지 않아서 int 반환

a1 = [1,1,1,1,2,2,2,2,3,3,4,4,5,5,5,6,3,2,6,7,7,8,8,8]
print(a1)
a1 = set(a1)
print(a1) # list -> dict 로 변경 -> 순서도 정렬

a1 = list(a1)
print(a1)

b1 = {1,10,2,3,4,100,1000}
print(b1)
# 결과: {1, 2, 3, 100, 4, 1000, 10} -> 정렬 불가, 100 넘어가면 깨짐 

s1 = (10,20,30)
print(s1[1])    # list, tuple, set, dict 모두 주소 호출 방식은 동일
```

### dictionary 예문
```
pororo = {"주인공":["뽀로로", "펭귄"], "조연1":"크롱", "조연2":"에디", "악역":"루피"}

print(pororo)
print(pororo.keys())
print(pororo.values())
print(pororo.items())

# print(pororo[0]) # 해당 dict 는 key:value 쌍이기 때문에 호출 방식이 다름
print(pororo["주인공"])
print(pororo["주인공"][1])

print(pororo['악역'])

pokemon = {"A01":"피카츄", "B01":"파이리"}
name = "B01"

print(pokemon.get(name)) # key를 기준으로 즉시 value 추출 -> 변수명 충돌 방지
print(pokemon[name])     # key를 기준으로 직접 value 추출

pokemon.update({"D01":"뮤츠"})

print(pokemon)
```

### Python 인덱싱
```
str = "가나다라마바사"
print(str)
print(str[3])
print(len(str))
# 주소도 0부터 시작

# 인덱싱 기준은 ~부터 ~미만까지 
birthday = "20260828"
print("년도: "+birthday[0:4]+"년") 
print("월: "+birthday[4:6]+"월") 
print("일: "+birthday[-2:8]+"일") 
print(birthday[::2]) # 2칸씩 출력
print(birthday[3:8:-1]) # 3부터 8까지 -1만큼씩 갈 수 없으므로 아무것도 반환하지 않음 
print(birthday[8:3:-1]) 
```

### 자료 구조
1. Stack: 선입후출(First In Last Out, FILO), 데이터 적재(Push) & 데아터 반출(Pop)
2. Queue: 선입선출(First In First Out, FIFO), 데이터 적재(Queue) & 데이터 반출(Dequeue)
