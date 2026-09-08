---
layout: single
title: "Java 배열·정렬·객체지향 기초"
date: 2026-09-02 09:00:00 +0900
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

> 수업 날짜: 2026-09-02

## 학습 주제

- 배열 순회와 최댓값·최솟값 탐색
- 버블·선택 정렬
- 완전수·로또 예제와 상속·오버로딩

## 수업 기록

코드는 파트를 기억해두고 필요할 때 가져오기
상황에 따라 유동적으로 변경

파이썬 문법: 자동으로 출력
자바 문법: 하나하나 다 지정


### 가장 큰 값 찾기

```
package test;

public class test10 {
	public static void main(String[] args) {
		int ar[] = {33,44,22,11,99,88,77,66,55}; 
		
		// System.out.printlr(ar); // 출력 불가 -> 배열의 원소들을 하나씩 출력은 가능 / 배열 자체는 불가능
		
		// 기존 for문 -> 메모리 주소를 직접 지정
		for (int i = 0; i < 8; i++) {
			System.out.print(ar[i]+" ");
		}
		System.out.println();
		
		// 향상된 for 문 -> 메모리 주소를 간접 지정
		for(int x:ar) { // ar 내 원소들을 처음부터 끝까지 x라는 변수에 입력
			System.out.print(x+" "); // x의 값들을 출력
		}
		System.out.println();
		
		//
		for (int i=0; i<ar.length; i++) {
			System.out.print(ar[i]+" ");
		}
		System.out.println();
		
		
		// 최댓값 탐색
		// 몇번째에 최댓값이 있었는지 확인이 불가능
		int max = 0;
		
		for (int x:ar) {
			if(max < x) {
				max = x;
			}
		}
		System.out.println("최댓값: "+max);
		
		// 바뀐 순서 확인1: 변수 추가
		int max1 = 0;
		int n = 0, juso1 = 0; // n: 위치값, juso: 주소값 -> 표현 방식 차이
		
		for (int x:ar) {
			n++;
			if(max < x) {
				max1 = x;
				juso1 = n;
			}
		}
		System.out.println("최댓값: "+max1);
		System.out.println("위치값: "+juso1);
		
		// 바뀐 순서 확인2: 내부 구조 변경
		int max2 = 0;
		int juso2 = 0;
		
		for (int i = 0; i < ar.length; i++) {
			if(max2 < ar[i]) {
				max2 = ar[i];
				juso2 = i;
			}
		}
		System.out.println("최댓값: "+max2);
		System.out.println("주소값: "+juso2);
		System.out.println("위치값: "+(juso2+1)+"번째");
		
		// 두 번째로 큰 수: 가장 큰 수를 제외
		ar[juso2] = 0;
		int max3 = 0; 
		int juso3 = 0;
		
		for (int i = 0; i < ar.length; i++) {
			if(max3 < ar[i]) {
				max3 = ar[i];
				juso3 = i;
			}
		}
		System.out.println("최댓값: "+max3);
		System.out.println("주소값: "+juso3);
		System.out.println("위치값: "+(juso3+1)+"번째");
		
		// 최솟값
		int min = 100;
		for (int i = 0; i < ar.length; i++) {
			if (min > ar[i]) {
				min = ar[i];
			}
		}
		System.out.println("최솟값: "+min);
	} 
}
```

### 정렬
1. 버블
가장 전통적, 앞뒤로 비교해 정렬, n^2, 가장 큰 값이 가장 맨 뒤로 이동

```
// 오름차순으로 정렬

package test;

public class test11 {
	public static void main(String[] args) {
		int ar[] = {42,36,49,27,15};
		
		// 정렬 전
		System.out.println("- 정렬 전");
		for(int x:ar) {
			System.out.print(x + " ");
		}
		System.out.println();
		
		// 버블 정렬
		System.out.println("\n == 버블 정렬 ==");
		System.out.println("- 버블 정렬 순서");
		
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j< 4; j++) {
				if (ar[j] > ar[j+1]) {
					// 위치 변경
					int tmp = ar[j];
					ar[j] = ar[j+1];
					ar[j+1] = tmp;
				}
			}
			for(int x:ar) {
				System.out.print(x + " ");
			}
			System.out.println();
		}
		
//		// 입력 배열 길이에 관련없는 코드
//		for (int i = 0; i < ar.length; i++) {
//			for (int j = 0; j< ar.length-1; j++) {
//				if (ar[j] > ar[j+1]) {
//					// 위치 변경
//					int tmp = ar[j];
//					ar[j] = ar[j+1];
//					ar[j+1] = tmp;
//				}
//			}
		
		// 정렬 결과
		System.out.println("- 버블 정렬 결과");
		for(int x:ar) {
			System.out.print(x + " ");
		}
		System.out.println();		
	}
}

```

2. 선택 정렬
현재 인덱스를 기준으로 값들을 정렬, n^2

```
package test;

public class test12 {

	public static void main(String[] args) {
		int ar[] = {42,36,49,27,15};
		
		// 정렬 전
		System.out.println("- 정렬 전");
		for(int x:ar) {
			System.out.print(x + " ");
		}
		System.out.println();
		
		System.out.println("\n == 선택 정렬 ==");
		System.out.println("- 선택 정렬 과정");
	
		for (int i = 0; i < ar.length; i++) {
			for (int j = 0; j< ar.length; j++) {
				if (ar[i] > ar[j]) {
					// 위치 변경
					int tmp = ar[i];
					ar[i] = ar[j];
					ar[j] = tmp;
				}
				for(int x:ar) {
					System.out.print(x + " ");
				}
				System.out.println();
			}				
		}
		// 정렬 결과
		System.out.println("- 선택 정렬 결과");
		for(int x:ar) {
			System.out.print(x + " ");
		}
		System.out.println();
	}
}
```

### 완전수 구하기
완전수(perfect number): 자기 자신을 제외한 약수의 합이 자기 자신
약수를 계산할 때, 해당 수의 절반까지만 진행하면 됨


```
package test;

public class test13 {

	public static void main(String[] args) {
		int su; 

		for (su=2; su <= 100; su++) {
			int cnt = 0;
			int sum = 0;
			
			System.out.print(su+": {");	
			for (int i = 1; i <= su/2; i++) {
				if(su % i == 0) {
					System.out.print(i+", ");
					cnt++;
					sum = sum + i;
				}
			}
			
			System.out.print(su+"}");
			
			// Prime Number
			if (cnt == 1) {
				System.out.println(" Prime number!!");
			}
			else {
				System.out.println();
			}
			
			// Perfect Number
			if (sum == su) {
				System.out.println("Perfect number!!");
			}
			else {
				System.out.println();
			}
		}

	}

}

```


### 예문: 로또 번호 맞추기

```
package test;

public class test14 {

	public static void main(String[] args) {
		int ar[] = new int[6];
		int lotto[] = {1,2,3,4,5,6};
		int cnt = 0;
		int num = 1;
		
		for (int z=0;;z++) {
			cnt = 0;
			for (int i=0; i < ar.length; i++) {
				ar[i] = (int) (Math.random() * 45) + 1; // Math.random: (0,1) 사이의 실수 
				
				// 중복 검증 로직
				for (int j=0; j < i; j++) {
					if (ar[j] == ar[i]) {
						i--;  // ar[i] 가 중복이기 때문에 뒤로 한칸 이동
						break;
					}
				}
			}
			
			for (int i = 0; i < ar.length; i++) {
				for (int j = 0; j < ar.length; j++) {
					if(lotto[i] == ar[j]) {
						cnt++;
					}
				}

			}
			num++;
			if (cnt == 6) {
				break;
			}
		}
		System.out.println("실제 결과");
		for (int x:lotto) {
			System.out.print(x+" ");
		}
		System.out.println("\n내 결과");
		for (int x:ar) {
			System.out.print(x+" ");
		}
		System.out.println("\n맞은 갯수: "+ cnt + "\n총 " + num + " 번");
	} 
}
```


### 상속 예문1

```
package test;

class Animal {
	void sound() {
		System.out.println("동물 소리");	
	}
}

class Dog extends Animal { // extend: 상속받음
	void sound() {
		System.out.println("멍멍");
		super.sound(); // 위치와 상관없이 가장 상위 클래스를 상속받음, 최상위 클래스는 상속받을 수 없음
		super.sound();
	}
}
		
class Cat extends Animal {
	void sound() {
		System.out.println("야옹");
	}
}

public class Main{
	public static void main(String[] args){
		Animal a = new Animal();
		a.sound();
		a = new Dog();
		a.sound();
	}
}

// 순서 지향 -> 부모 class에 함수가 없으면 상속받는 class에서는 사용 불가
// 그런데 파이썬에서는 오류가 발생하지않음

```


### 상속 예문2

```
package test;

class Parent{
	String name = "부모";
	
	void print() {
		System.out.println("부모 메서드");
	}
}

class Child extends Parent{
	String name = "자식";
	
	void print() {
		System.out.println("자식 메서드");
	}
}

public class Main{
	public static void main(String[] args){
		Parent p = new Child(); 
		// Child 메서드에 Parent 선언 -> Parent 로 상속
		// 부모는 자식으로부터 상속받을 수 없음 -> Child p = new Parent(); 는 불가능
		// Child 를 먼저 선언 -> Parent 값을 상속 -> Child 값은 덮어 씌워짐 -> return 이 부모
		
		System.out.println(p.name);
	}
}
```


### 상속 예문3

```
package test;

class TT{
	void print(int x) {    // 정수형
		System.out.println("정수: "+x);
	}
	void print(double x) { // 실수형
		System.out.println("실수: "+x);
	}
	void print(char x) {   // 문자형
		System.out.println("문자: "+x);
	}
}

public class Main{
	public static void main(String[] args){
		TT t = new TT();
		t.print(10);
		t.print(3.14);
		t.print('A'); // UNI코드 기반 -> A는 문자형으로만 등록되어있음
	}
}
```


### 기술면접 대표 질문

1. 상속
2. 추상화
3. 인터페이스
