---
layout: single
title: "Python 제어문·함수와 HTML 입문"
date: 2026-09-04 09:00:00 +0900
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

> 수업 날짜: 2026-09-04

## 학습 주제

- for 문과 리스트 컴프리헨션
- 함수·재귀·map
- HTML·CSS와 가위바위보·키오스크 실습

## 수업 기록

> 가위·바위·보 이미지 3장은 함께 보관했습니다. 키오스크 예제에서 쓰는 음료 이미지 4종은 원본 압축파일에 포함되어 있지 않아, 해당 경로는 자리표시자로 남겼습니다.


### for 문
```
# c언어 for문: for(from;to;step){}
# Python for문: for 변수 in 인:
 
for i in range(0,10,1): # range(from,to(미만),step), step의 default는 1
    print(i, end=' ')
print()

for i in range(10): 
    print(i, end=' ')
print()

ls = [1,2,3,4,5,6,7,8,9]
for i in ls:
    print(i, end=" ")
print()

for i in range(8,-1,-1):
    print(ls[i], end=" ")
print()
```

### list comprehension
```
# for 문이 자료구조 내에서 구성

ls = []
for i in range(5):
    ls.append(i)
print(ls)

ls2 = [x for x in range(5)]
print(ls2)

# 이중 for문으로도 구성 가능
ls3 = [(i,j) for i in range(3) for j in range(5)]
print(ls3)

ls4 = [i for i in range(10) if i%2 == 0]
print(ls4)


ls5 = [i for i in range(10) if i%2 == 0 if i%4 == 0]
print(ls5)

"""
1. C언어 문법

for(i=0; i<10;i++){
    if(i%2==0){
        if(i%4==0){
            ls5.append(i)
        }
    }
}
# -------------------------------------
2. Python 문법

ls5 = []
for i in range(10):
    if i%2 == 0:
        if i%4 == 0:
            ls5.append(i)
"""
# 이해가 안된다면 절차적 언어로 작성 -> 디버깅
```

### 함수
```
#1. 매개값(인수, 인자) 등이 없는 경우
def pp():
    print("hello")
pp()    

# 2. 매개값(인수, 인자) 등이 있는 경우
#  2.1. 매개값(인수, 인자) 등이 있고 반환값이 있는 경우
def add(a,b):
    return a+b
print(add(3,5))

#  2.2. 매개값(인수, 인자) 등이 있고 반환값이 없는 경우
def add(a,b):
    print(a+b)
add(3,5)
```

### 재귀 함수
```
def f(n):
    if n == 1:
        return 1
    return n * f(n-1)

print(f(5))
```

### map 함수
```
def po(x):
    return x ** 2

num = [1,2,3,4,5]
num_po = map(po, num)
print(num_po) # <map object at 0x0000021842104640>: 연산은 성공, 보이지는 않음
print(list(num_po))

p_num = [x**2 for x in range(1,6,1)]
print(p_num)
```


### html
정적인 언어, 움직이는게 없음 -> 계산, 연산 등이 없음
java - html: Spring / Python - html: Django
java script: java 와 다른 script 언어, python 과 유사
주로 다른 언어들과 같이 사용

### html 예문1
```
<!DOCTYPE html>
<html lang="en"> <!-- 언어 = 영어(en)--> 
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
Hello!
<body> <!-- 주로 페이지의 내용을 기입, 근데 밖에 써도 상관은 없음 -->
    Hello~    
</body>
</html>
<!-- html 연결은 edge가 좋음. y? 가벼워서 -->
```

html 연결은 edge가 좋음. y? 가벼워서
웹페이지 상에서 F12를 눌러서 console 창을 눌러 실제 진행상황 확인 가능 
쿠키 or 캐시: 웹페이지 상의 로그 등, 기존 캐시나 쿠키가 남아 업데이트가 안되는 경우가 존재
새로고침 버튼 우클릭 -> 캐시 비우기 및 강한 새로고침으로 확인 가능 
jQuery가 아닌 이상 개행 등이 난해 -> script 내에서 "<br>" 등으로 이용


### html 변수 종류
1. var: 가변형
2. let: 부분 가변형 
3. const: 불변형


### html 예문2
```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <script> 
        a = 100
        b = 'a'
        c = 'hello'
        d = '1000'

        document.write(a + "<br>") // 함수 내에 + "<br>" 로 개행 입력  
        console.log(b)
        document.write(a+d + "<br>")
        document.write(a+Number(d) + "<br>") // Number(): 정수형

    </script>
</body>
</html>
```


### html로 가위바위보 페이지 만들기1
```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <!-- 개별 설정 -->
    <!-- width와 height를 지정해 다른 이미지들과 강제로 동일한 해상도로 설정 -->
    <!-- 주먹 -->
    <button type="button" onclick="game(1)"><img src = "/images/bootcamp/2026-09-04/rps-rock.png" width="200" height="200">  <!-- onclick: 일종의 트리거 -->
    </button>
    <!-- 가위 -->
    <button type="button" onclick="game(2)"><img src = "/images/bootcamp/2026-09-04/rps-scissors.png" width="200" height="200">
    </button>
    <!-- 보자기 -->
    <button type="button" onclick="game(3)"><img src = "/images/bootcamp/2026-09-04/rps-paper.png" width="200" height="200">
    </button>

    <script>
        com = Math.floor(Math.random() * 3) + 1 // 1~3 사이의 난수
        console.log(com)

        function game(s){
            // 비긴 경우
            if (s == com){
                document.write("Draw")
            }
            // 사람이 이긴 경우
            else if((s==1&&com==2) || (s==2&&com==3) || (s==3&&com==1)){
                document.write("You Win")
            }
            // 사람이 진 경우
            else{
                document.write("Computer wins."+"<br>")
                // document.write("<img src=cc.png>")
            }
        }
    </script>
</body>
</html>
```

### css
Cascading Style Sheet, 웹페이지의 디자인과 레이아웃을 꾸미는 스타일 시트 언어

### html로 가위바위보 페이지 만들기2

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<style>
/* 
함수 // .함수  
이미지가 버튼보다 위에 있거나 컬러가 없는 경우 적용 불가
color: 또는 background-color: 등으로 사용 
태그 - 태그 전체 서식 지정 가능
-> .class - class 지정을 한 번에 가능 
*/ 
    button{
        background-color: red;
    }
    .ig{
        width: 200px;
        height: 200px;
    }
</style>
<body>
    <button type="button" id="com1"><img src="/images/bootcamp/2026-09-04/rps-rock.png" class="ig">
    </button>
    <button type="button" id="com2"><img src="/images/bootcamp/2026-09-04/rps-scissors.png" class="ig">
    </button>
    <button type="button" id="com3"><img src="/images/bootcamp/2026-09-04/rps-paper.png" class="ig">
    </button>
</body>
<script>
    document.getElementById("com1").addEventListener("click", () => game(1)) // com1 이라는 이벤트 발생 시 game(1) 을 실행
    document.getElementById("com2").addEventListener("click", () => game(2)) // => : 보낸다 
    document.getElementById("com3").addEventListener("click", () => game(3))

    com = Math.floor(Math.random() * 3) + 1 // 1~3 사이의 난수
        console.log(com)

        function game(s){
            // 비긴 경우
            if (s == com){
                document.write("Draw")
            }
            // 사람이 이긴 경우
            else if((s==1&&com==2) || (s==2&&com==3) || (s==3&&com==1)){
                document.write("You Win")
            }
            // 사람이 진 경우
            else{
                document.write("Computer wins."+"<br>")
                // document.write("<img src=cc.png>")
            }
        }
</script>
</html>
```

### 키오스크 메뉴판 만들기

```
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>스타벅스 메뉴판</title>
</head>
<style>
    *{ box-sizing: border-box; }
    body{
        margin: 0;
        padding: 40px 20px;
        font-family: "Pretendard", "맑은 고딕", sans-serif;
        background-color: #1E3932;
        color: #1E3932;
    }
    .board{
        max-width: 1000px;
        margin: auto;
        background-color: #F7F5F0;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 20px 50px rgba(0,0,0,.35);
    }

    /* 상단 헤더 */
    .header{
        background-color: #00704A;
        color: white;
        text-align: center;
        padding: 30px 20px 26px;
        border-bottom: 4px solid #C9A961;
    }
    .header .brand{
        margin: 0;
        font-size: 14px;
        letter-spacing: 8px;
        color: #C9A961;
    }
    .header h1{
        margin: 8px 0 0;
        font-size: 40px;
        letter-spacing: 12px;
        font-weight: 700;
    }

    /* 메뉴 2단 */
    .menu{
        display: flex;
        flex-wrap: wrap;
    }
    .item{
        flex: 1 1 400px;
        padding: 34px 30px;
    }
    .item:first-child{
        border-right: 1px solid #E3DED4;
    }

    /* 사진 */
    .photo{
        width: 210px;
        height: 250px;
        margin: 0 auto 18px;
        background-color: #EFEBE3;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .coffee_img{
        width: 190px;
        height: 230px;
        object-fit: contain;
    }

    /* 이름 / 영문명 */
    .name{
        text-align: center;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
    }
    .eng{
        text-align: center;
        font-size: 12px;
        letter-spacing: 3px;
        color: #8A8578;
        margin: 6px 0 22px;
    }

    /* HOT / ICE 토글 */
    .temp{
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
    }
    .temp input{ display: none; }
    .temp label{
        flex: 1;
        text-align: center;
        padding: 11px 0;
        border: 1.5px solid #D9D3C7;
        border-radius: 30px;
        font-weight: 700;
        letter-spacing: 2px;
        color: #9A9488;
        cursor: pointer;
    }
    .temp .hot:checked + label{
        background-color: #C0392B;
        border-color: #C0392B;
        color: white;
    }
    .temp .ice:checked + label{
        background-color: #2C7BB6;
        border-color: #2C7BB6;
        color: white;
    }

    /* 옵션 칩 */
    .opts{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 22px;
    }
    .opt input{ display: none; }
    .opt span{
        display: inline-block;
        padding: 9px 14px;
        border: 1.5px solid #D9D3C7;
        border-radius: 30px;
        font-size: 14px;
        color: #6B665C;
        cursor: pointer;
    }
    .opt span b{
        color: #B0A996;
        font-weight: 500;
    }
    .opt input:checked + span{
        background-color: #00704A;
        border-color: #00704A;
        color: white;
    }
    .opt input:checked + span b{ color: #BFE3D3; }

    /* 단가 / 수량 */
    .line{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-top: 1px solid #E3DED4;
        font-size: 15px;
        color: #6B665C;
    }
    .stepper{
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .btncnt{
        width: 34px;
        height: 34px;
        border: none;
        border-radius: 50%;
        background-color: #00704A;
        color: white;
        font-size: 20px;
        line-height: 1;
        cursor: pointer;
    }
    .cnt{
        min-width: 24px;
        text-align: center;
        font-size: 18px;
        font-weight: 700;
        color: #1E3932;
    }

    /* 금액 */
    .cash{
        border-top: 2px solid #1E3932;
        padding-top: 14px;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        font-weight: 700;
    }
    .cash .big{
        font-size: 28px;
        color: #00704A;
    }

    /* 하단 합계 */
    .total {
        background-color: #1E3932;
        color: white;
        display: flex;
        justify-content: space-between; /* ◀ 좌측 글자와 우측 금액을 양끝으로 밀어줌 */
        align-items: center;            /* ◀ 위아래 높이를 가운데로 똑바로 정렬 */
        padding: 24px 34px;
        letter-spacing: 2px;
        width: 100%;                    /* ◀ 박스 크기를 가득 채우도록 고정 */
        box-sizing: border-box;
    }

    /* '총 주문금액' 글자 스타일 추가 */
    .total > span:first-child {
        font-size: 16px;
        font-weight: 700;
        white-space: nowrap;            /* ◀ 글자가 절대 아래로 줄바꿈되지 않게 방지 */
    }

    /* 우측 숫자 금액 스타일 */
    .total .sum {
        font-size: 32px;
        font-weight: 700;
        color: #C9A961;
        white-space: nowrap;            /* ◀ 금액 숫자도 한 줄로 깨끗하게 고정 */
        margin-left: auto;              /* ◀ 숫자를 무조건 오른쪽 끝으로 밀어주는 안전장치 */
    }


    /* 아메리카노 옵션 대화상자 전체 */
    .opts-container {
        display: flex;
        flex-direction: column;
        gap: 16px;
        margin: 20px 0;
        text-align: left;
    }

    /* 각 옵션 한 줄 */
    .opt-row {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    /* [샷 추가] 등 타이틀 스타일 */
    .opt-row h4 {
        margin: 0;
        font-size: 15px;
        color: #1E3932;
        font-weight: 700;
    }

    /* 버튼들을 가로로 정렬하는 컨테이너 */
    .opt-inputs {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }

    /* 기존 동그라미 라디오 버튼은 완벽하게 숨김 */
    .opt-inputs .opt input[type="radio"] {
        display: none !important;
    }

    /* 💡 핵심: HOT/ICE와 동일한 타원형 버튼 스타일 (기본: 흰색 계열) */
    .opt-inputs .opt span {
        display: inline-block;
        padding: 7px 12px;       /* ◀ 내부 위아래/좌우 여백을 줄여 버튼을 날씬하게 만듦 */
        border: 1.5px solid #D9D3C7;
        border-radius: 30px;
        font-size: 12px;          /* ◀ 기존 14px에서 12px로 글자 크기 축소 */
        font-weight: 700;
        color: #9A9488;
        background-color: white;
        cursor: pointer;
        text-align: center;
        white-space: nowrap;      /* ◀ 버튼 안에서 글자가 절대 강제로 줄바꿈되지 않도록 방지 */
        transition: all 0.2s ease;
    }

    /* 금액을 강조하는 <b> 태그 */
    .opt-inputs .opt span b {
        font-size: 11px;          /* ◀ 가격 부분 글자 크기도 미세하게 조절 */
        font-weight: 500;
        color: #B0A996;
        margin-left: 2px;
    }

    /* 💡 핵심: 선택(checked)되었을 때 스타벅스 초록색으로 대변신 */
    .opt-inputs .opt input[type="radio"]:checked + span {
        background-color: #00704A !important;
        border-color: #00704A !important;
        color: white !important;
    }

    /* 선택되었을 때 내부 금액 글자 색상 보정 */
    .opt-inputs .opt input[type="radio"]:checked + span b {
        color: #BFE3D3 !important;
    }


    /* 접이식 컨테이너 디자인 */
    details.opts-container {
        background: #F4F2EC;
        border: 1px solid #D9D3C7;
        border-radius: 12px;
        margin: 15px 0 25px; /* ◀ 아래쪽 여백(25px)을 넉넉히 주어 단가 라인과의 간격을 확보 */
        overflow: hidden;
        transition: all 0.3s ease;
    }

    /* 클릭하는 타이틀 바 디자인 */
    details.opts-container summary {
        padding: 12px 16px;
        font-size: 14px;
        font-weight: 700;
        color: #1E3932;
        cursor: pointer;
        background-color: #EFEBE3;
        list-style: none; /* 기본 삼각형 화살표 제거 (크롬/사파리) */
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* 기본 삼각형 화살표 제거 (파이어폭스 등 기타 브라우저) */
    details.opts-container summary::-webkit-details-marker {
        display: none;
    }

    /* 펼쳐졌을 때 안쪽 옵션들의 여백 조절 */
    .opts-inner-content {
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 14px;
    }
</style>
<body>

<div class="board">

    <div class="header">
        <p class="brand">S T A R B U C K S</p>
        <h1>MENU</h1>
    </div>

    <div class="menu">

        <!-- 아메리카노 -->
        <div class="item">
            <div class="photo"><img src="[원본-미포함] 아메리카노-HOT" class="coffee_img" id="aa_img"></div>
            <p class="name">아메리카노</p>
            <p class="eng">CAFFE AMERICANO</p>

            <!-- HOT/ICE 토글 (항상 노출) -->
            <div class="temp">
                <input type="radio" name="aa_type" id="aa_hot" class="hot" checked onclick="aaimg('hot'); aa_total();">
                <label for="aa_hot">HOT</label>
                <input type="radio" name="aa_type" id="aa_ice" class="ice" onclick="aaimg('ice'); aa_total();">
                <label for="aa_ice">ICE</label> 
            </div>

            <!-- 💡 [오직 세부 옵션 4개만] 접고 펼치는 상자 -->
            <details class="opts-container">
                <summary>⚙️ 세부 옵션 조절 (클릭)</summary>
                
                <div class="opts-inner-content">
                    <!-- 1. 샷 추가 -->
                    <div class="opt-row">
                        <h4>[샷 추가]</h4>
                        <div class="opt-inputs">
                            <label class="opt"><input type="radio" name="aa_shot" value="0" checked onclick="aa_total()"><span>없음</span></label>
                            <label class="opt"><input type="radio" name="aa_shot" value="300" onclick="aa_total()"><span>1샷 <b>+300원</b></span></label>
                            <label class="opt"><input type="radio" name="aa_shot" value="600" onclick="aa_total()"><span>2샷 <b>+600원</b></span></label>
                            <label class="opt"><input type="radio" name="aa_shot" value="900" onclick="aa_total()"><span>3샷 <b>+900원</b></span></label>
                        </div>
                    </div>

                    <!-- 2. 시럽 추가 -->
                    <div class="opt-row">
                        <h4>[시럽 추가]</h4>
                        <div class="opt-inputs">
                            <label class="opt"><input type="radio" name="aa_syrup" value="0" checked onclick="aa_total()"><span>없음</span></label>
                            <label class="opt"><input type="radio" name="aa_syrup" value="500" onclick="aa_total()"><span>바닐라 시럽 <b>+500원</b></span></label>
                            <label class="opt"><input type="radio" name="aa_syrup" value="500" onclick="aa_total()"><span>헤이즐넛 시럽 <b>+500원</b></span></label>
                        </div>
                    </div>

                    <!-- 3. 우유 변경 -->
                    <div class="opt-row">
                        <h4>[우유 변경]</h4>
                        <div class="opt-inputs">
                            <label class="opt"><input type="radio" name="aa_milk" value="0" checked onclick="aa_total()"><span>없음</span></label>
                        </div>
                    </div>

                    <!-- 4. 휘핑 크림 -->
                    <div class="opt-row">
                        <h4>[휘핑 크림]</h4>
                        <div class="opt-inputs">
                            <label class="opt"><input type="radio" name="aa_whipped" value="0" checked onclick="aa_total()"><span>없음</span></label>
                            <label class="opt"><input type="radio" name="aa_whipped" value="0" onclick="aa_total()"><span>적게</span></label>
                            <label class="opt"><input type="radio" name="aa_whipped" value="500" onclick="aa_total()"><span>보통 <b>+500원</b></span></label>
                            <label class="opt"><input type="radio" name="aa_whipped" value="800" onclick="aa_total()"><span>많이 <b>+800원</b></span></label>
                        </div>
                    </div>
                </div>
            </details> <!-- 💡 여기서 디테일 박스를 닫아줍니다! -->

            <!-- 💡 [단가, 수량, 금액]은 박스 바깥에 있으므로 항상 화면에 노출됩니다 -->
            <div class="line">
                <span>단가</span>
                <span id="aa_price">5,000원</span>
            </div>
            <div class="line">
                <span>수량</span>
                <div class="stepper">
                    <button type="button" class="btncnt" onclick="aacnt('minus')">-</button>
                    <span class="cnt" id="acnt">1</span>
                    <button type="button" class="btncnt" onclick="aacnt('plus')">+</button>
                </div>
            </div>
            <div class="cash">
                <span>금액</span>
                <span class="big" id="aa_cash">5,000원</span>
            </div>
        </div>

        <!-- 돌체라떼 -->
        <div class="item">
            <div class="photo"><img src="[원본-미포함] 돌체라떼-HOT" class="coffee_img" id="dol_img"></div>
            <p class="name">돌체라떼</p>
            <p class="eng">DOLCE LATTE</p>

            <div class="temp">
                <!-- onclick 뒤에 dol_total()을 함께 실행하도록 추가했습니다 -->
                <input type="radio" name="dol_type" id="dol_hot" class="hot" checked onclick="dolimg('hot'); dol_total();">
                <label for="dol_hot">HOT</label>
                <input type="radio" name="dol_type" id="dol_ice" class="ice" onclick="dolimg('ice'); dol_total();">
                <label for="dol_ice">ICE</label>
            </div>


        <!-- 💡 [오직 세부 옵션 4개만] 접고 펼치는 상자 -->
            <details class="opts-container">
                <summary>⚙️ 세부 옵션 조절 (클릭)</summary>
                
                <div class="opts-inner-content">
                    <!-- 1. 샷 추가 -->
                    <div class="opt-row">
                        <h4>[샷 추가]</h4>
                        <div class="opt-inputs">
                            <label class="opt"><input type="radio" name="aa_shot" value="0" checked onclick="aa_total()"><span>없음</span></label>
                            <label class="opt"><input type="radio" name="aa_shot" value="300" onclick="aa_total()"><span>1샷 <b>+300원</b></span></label>
                            <label class="opt"><input type="radio" name="aa_shot" value="600" onclick="aa_total()"><span>2샷 <b>+600원</b></span></label>
                            <label class="opt"><input type="radio" name="aa_shot" value="900" onclick="aa_total()"><span>3샷 <b>+900원</b></span></label>
                        </div>
                    </div>

                    <!-- 2. 시럽 추가 -->
                    <div class="opt-row">
                        <h4>[시럽 추가]</h4>
                        <div class="opt-inputs">
                            <label class="opt"><input type="radio" name="aa_syrup" value="0" checked onclick="aa_total()"><span>없음</span></label>
                            <label class="opt"><input type="radio" name="aa_syrup" value="500" onclick="aa_total()"><span>바닐라 시럽 <b>+500원</b></span></label>
                            <label class="opt"><input type="radio" name="aa_syrup" value="500" onclick="aa_total()"><span>헤이즐넛 시럽 <b>+500원</b></span></label>
                        </div>
                    </div>

                    <!-- 3. 우유 변경 -->
                    <div class="opt-row">
                        <h4>[우유 변경]</h4>
                        <div class="opt-inputs">
                            <label class="opt"><input type="radio" name="aa_milk" value="0" checked onclick="aa_total()"><span>없음</span></label>
                        </div>
                    </div>

                    <!-- 4. 휘핑 크림 -->
                    <div class="opt-row">
                        <h4>[휘핑 크림]</h4>
                        <div class="opt-inputs">
                            <label class="opt"><input type="radio" name="aa_whipped" value="0" checked onclick="aa_total()"><span>없음</span></label>
                            <label class="opt"><input type="radio" name="aa_whipped" value="0" onclick="aa_total()"><span>적게</span></label>
                            <label class="opt"><input type="radio" name="aa_whipped" value="500" onclick="aa_total()"><span>보통 <b>+500원</b></span></label>
                            <label class="opt"><input type="radio" name="aa_whipped" value="800" onclick="aa_total()"><span>많이 <b>+800원</b></span></label>
                        </div>
                    </div>
                </div>
            </details> <!-- 💡 여기서 디테일 박스를 닫아줍니다! -->

            <!-- 단가/수량/금액 라인은 기존 유지 -->
            <div class="line">
                <span>단가</span>
                <span id="dol_price">6,000원</span>
            </div>
            <div class="line">
                <span>수량</span>
                <div class="stepper">
                    <button type="button" class="btncnt" onclick="dolcnt('minus')">-</button>
                    <span class="cnt" id="dcnt">1</span>
                    <button type="button" class="btncnt" onclick="dolcnt('plus')">+</button>
                </div>
            </div>
            <div class="cash">
                <span>금액</span>
                <span class="big" id="dol_cash">6,000원</span>
            </div>
        </div>

    <div class="total">
        <span>총 주문금액</span>
        <span class="sum" id="all_cash">11,000원</span>
    </div>

</div>

</body>
<script>
    // 1. 각 메뉴별 기본 데이터 초기화
    let aa_base = 5000;    // 아메리카노 기본 단가 (HOT)
    let aa_qty = 1;        // 아메리카노 기본 수량
    let aaSum = 5000;      // 아메리카노 최종 금액 저장용

    let dol_base = 6000;   // 돌체라떼 기본 단가 (HOT)
    let dol_qty = 1;       // 돌체라떼 기본 수량
    let dolSum = 6000;     // 돌체라떼 최종 금액 저장용

    // 숫자를 1,000원 형태로 바꿔주는 편리한 함수
    function won(n){
        return n.toLocaleString() + "원";
    }

    // ==========================================
    // 2. 아메리카노 연산 및 제어 로직
    // ==========================================
    function aaimg(type) {
        const img = document.getElementById('aa_img');
        if(type === 'hot') {
            img.src = '[원본-미포함] 아메리카노-HOT';
        } else {
            img.src = '[원본-미포함] 아메리카노-ICE';
        }
        aa_total(); // 💡 HOT/ICE 바뀔 때마다 즉시 가격 재계산
    }

    function aacnt(action) {
        if(action === 'plus') aa_qty++;
        else if(action === 'minus' && aa_qty > 1) aa_qty--;
        
        document.getElementById('acnt').innerText = aa_qty;
        aa_total(); // 수량이 바뀌면 합계 재계산
    }

    function aa_total() {
        // 💡 ICE가 선택되었다면 500원 추가 연산
        let icePrice = 0;
        if (document.getElementById('aa_ice').checked) {
            icePrice = 500;
        }

        // 라디오 버튼 그룹에서 사용자가 선택한 라디오 버튼의 가치(value) 수집
        const shot = parseInt(document.querySelector('input[name="aa_shot"]:checked').value);
        const syrup = parseInt(document.querySelector('input[name="aa_syrup"]:checked').value);
        const milk = parseInt(document.querySelector('input[name="aa_milk"]:checked').value);
        const whipped = parseInt(document.querySelector('input[name="aa_whipped"]:checked').value);

        // 옵션과 ICE 금액이 포함된 최종 '단가' 계산
        let singlePrice = aa_base + icePrice + shot + syrup + milk + whipped;
        // 단가 * 수량 = '최종 금액'
        aaSum = singlePrice * aa_qty;

        // 화면 갱신 (won 함수 활용)
        document.getElementById('aa_price').innerText = won(singlePrice);
        document.getElementById('aa_cash').innerText = won(aaSum);
        
        order_total(); // 전체 결제 금액 갱신
    }

    // ==========================================
    // 3. 돌체라떼 연산 및 제어 로직
    // ==========================================
    function dolimg(type) {
        const img = document.getElementById('dol_img');
        if(type === 'hot') {
            img.src = '[원본-미포함] 돌체라떼-HOT';
        } else {
            img.src = '[원본-미포함] 돌체라떼-ICE';
        }
        dol_total(); // 💡 HOT/ICE 바뀔 때마다 즉시 가격 재계산
    }

    function dolcnt(action) {
        if(action === 'plus') dol_qty++;
        else if(action === 'minus' && dol_qty > 1) dol_qty--;
        
        document.getElementById('dcnt').innerText = dol_qty;
        dol_total(); // 수량이 바뀌면 합계 재계산
    }

    function dol_total() {
        // 💡 ICE가 선택되었다면 500원 추가 연산
        let icePrice = 0;
        if (document.getElementById('dol_ice').checked) {
            icePrice = 500;
        }

        // 돌체라떼 라디오 버튼 그룹에서 선택된 가치(value) 수집
        const shot = parseInt(document.querySelector('input[name="dol_shot"]:checked').value);
        const syrup = parseInt(document.querySelector('input[name="dol_syrup"]:checked').value);
        const milk = parseInt(document.querySelector('input[name="dol_milk"]:checked').value);
        const whipped = parseInt(document.querySelector('input[name="dol_whipped"]:checked').value);

        // 옵션과 ICE 금액이 포함된 최종 '단가' 계산
        let singlePrice = dol_base + icePrice + shot + syrup + milk + whipped;
        // 단가 * 수량 = '최종 금액'
        dolSum = singlePrice * dol_qty;

        // 화면 갱신 (won 함수 활용)
        document.getElementById('dol_price').innerText = won(singlePrice);
        document.getElementById('dol_cash').innerText = won(dolSum);
        
        order_total(); // 전체 결제 금액 갱신
    }

    // ==========================================
    // 4. 하단 [총 주문금액] 계산기
    // ==========================================
    function order_total() {
        document.getElementById("all_cash").innerText = won(aaSum + dolSum);
    }

    // 💡 키오스크 웹페이지가 처음 켜졌을 때 자동으로 기본 세팅 금액을 연산 및 연동해 둡니다.
    window.onload = function() {
        aa_total();
        dol_total();
    }
</script>
</html>
```
