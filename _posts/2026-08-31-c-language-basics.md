---
layout: single
title: "C 언어 기초와 반복문 실습"
date: 2026-08-31 09:00:00 +0900
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

> 수업 날짜: 2026-08-31

## 학습 주제

- 6개월 목표 설정
- C 자료형·조건문·반복문
- 배열과 전광판 출력 실습

## 수업 기록

## 6개월 내의 목표 설정
1. 커리어
2. 자격증
3. 취업
4. 포폴
5. 대회(수상경력)
6. 코테 준비
7. 기타 목표 등

## C 언어
비주류 언어, 하지만 기본
C -> 원시 프로그램 -> 번역 (compiler) -> 목적(.obj) -> 링커 -> 모듈 -> 로더
java는 complier 랑 interpreter 사이 -> 실행 -> 대표적 예시: 파이썬

### C 언어의 자료형
int: 정수형, %d
float: 실수형, %f
char: 문자형(정수형 포함), %c
변수: 변할 수 있는 값
함수: 이미 기능을 정의
상수: 숫자 or 데이터


### 예문
#include<stdio.h>

main() {
	int a = 10;
	printf("a=%d \n", a);
	int b = 12;
	printf("b=%d \n", b);

	printf("a+b=%d \n", a + b);

}

ASCII 코드 - C언어 기반, 128비트 언어 체계
영&숫자: 1바이트 / 한글&특수문자: 2바이트

```
#include<stdio.h>

main() {
	char ch = 'a'; // '': "" 내부에서 사용
	printf("ch=%c\n", ch);
	printf("ch=%d\n", ch);   // 문자형을 정수형으로 변환하여 출력 시 ASCII 코드 상 번호 출력
	printf("ch=%c\n", ch+3); // ASCII 코드 상에서 3 증가한 문자 출력
}

// 제어문: if

#include<stdio.h>

main() {
	int su = 100;

	if (su > 10) { // True 영역
		printf("True");
	}
	else { // False 영역
		printf("False");
	}
}

### 예제
// 두 수를 입력하여 차이를 구하시오. 단, 음수는 허용하지 않음.
// scanf 를 사용하기 위해 파일-속성-sdl검사-아니요 로 변경

#include<stdio.h>

main() {
	int one, two;
	
	printf("one : ");
	scanf("%d", &one); // &: 연결연산자
	printf("two : ");
	scanf("%d", &two);
	

	if (one < two) {
		int t = one;
		one = two;
		two = t;
	}
	printf("answer: %d", one - two);
}


### 구구단
#include<stdio.h>

main() {
	int i, j;
	for (i = 1; i <= 9; i++) {
		for (j = 2; j < 10; j++) {
			printf("%dX%d=%2d ", j, i, i * j);
		}
		printf("\n");
	}
}
```

### 비교 연산자
> : 초과
>= : 이상
< : 미만
<= : 이하
== : 같다
!= : 다르다

### 논리비교연산자
&&: and
||: or


### 반복문 2개를 이용해서 별 찍기
```
#include<stdio.h>

// for문 1개로 작성
main() {
	int i;
	for (i = 1; i <= 25; i++) {
		printf("*");
		if (i % 5 == 0) {
			printf("\n");
		}
	}
}

// for 문 2개로 작성
main() {
	int i, j;
	//int su = 1;
	for (i = 0;i < 5;i++) {
		for (j = 0;j <= i;j++) { // i 대신 새로운 변수 su 를 넣어서도 가능은 함
			printf("*");
		}
		printf("\n");
		//su++;
	}
//}

// 역행1
main() {
	int i, j;
	int su = 5;

	for (i = 0;i < 5;i++) {
		for (j = 0;j < su;j++) {
			printf("*");
		}
		printf("\n");
		su--;
	}
}

// 역행2
main() {
	int i, j;

	for (i = 0;i < 5;i++) {
		for (j = 0;j < 5-i ;j++) {
			printf("*");
		}
		printf("\n");
	}
}
```

### 전광판 만들기

```
#include<stdio.h>
#include<windows.h> // 라이브러리: 사전에 설정된 함수 모음집

main() {
	// 배열 생성: 0부터 시작
	char ar[5] = {'H', 'E', 'L', 'L', 'O'};
	int i, j;

	int su = 0;

	for (i = 0; i < 5; i++) {
		for (j = 0; j < 5; j++) {
			printf("%c ", ar[su]);
			su++;
		} // su가 5 이상으로 넘어가므로 ?로 출력됨
		printf("\n");
	}
}

main() {
	// 배열 생성: 0부터 시작
	char ar[5] = { 'H', 'E', 'L', 'L', 'O' };
	int i, j, x;
	int su = 0;

	for (x = 0;;x++) { // while문 대신 무한대로 출력하도록 설정 (x의 범위 미설정)
		for (i = 0; i < 5; i++) {
			su = i;
			for (j = 0; j < 5; j++) {
				printf("%c ", ar[su]);
				su++;
				if (su == 5) {
					su = 0;
				}
			}
			printf("\n");

			Sleep(1000);   // 기준 밀리초 -> 1000ms = 1s
			system("cls"); // 화면지우기
		}
	}
}

// 그러면 회전방향을 반대로 하는건?
// 아니면 글자를 수평에서 수직으로 회전하는건?
```

### 코드리뷰 및 변경 팁
1. 외부 변수 추가
2. 코드 안에서 변경가능한 부분 탐색
3. 패턴화

### 기타
- 현업가면 새 코드 작성보다 기존 코드 유지보수 -> 코드 리뷰가 필수
- C는 메모리 주소를 저장해서 값을 불러옴
