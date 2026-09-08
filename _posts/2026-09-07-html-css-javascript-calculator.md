---
layout: single
title: "HTML·CSS·JavaScript 실습: 계산기 구현"
date: 2026-09-07 09:00:00 +0900
categories:
  - "SK Encore DE 2기"
subcategory: "수업 내용"
author_profile: true
toc: true
toc_sticky: true
---

> 수업 날짜: 2026-09-07

## 학습 주제

- prompt와 DOM 출력
- HTML 테이블 생성과 CSS 스타일링
- 일반·공학용 계산기 구현과 수식 파싱

## 수업 기록

### html 상에서 java script의 prompt 입력

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
        // prompt가 연산자 우선순위, but 가려져서 확인 난해 
        UserName = prompt("Enter name: ", "") // 표시내용, 초기값(string)
        console.log(typeof(UserName))
        document.write("Your name: "+UserName+"<br>")

        Age = prompt("Age: ", "0")
        console.log(typeof(Age))
        document.write("Age: "+Age+"<br>")
        document.write("Age after 10 years: "+(Number(Age)+10)+"<br>")
    </script>
</body>
</html>
```

### 하드코딩 vs 자동화

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <!-- 하드코딩 방식 -->
    <table border="1"> <!-- border="1": 표의 각 셀에 대해 테두리 생성-->
        <tr bgcolor="blue"> <!-- 행 전체에 배경색상 변경 -->
            <td bgcolor="red">1</td> <!-- 셀 하나의 배경색상만 변경 -->
            <td bgcolor="#00592d"   >2</td> <!-- RGB 코드도 가능 -->
            <td>3</td>
            <td rowspan="2"> <!-- 행 병합 -->
                5
            </td>
        </tr>
        <tr>
            <td>4</td>
            <td colspan="2" align="center"> <!-- 열 병합-->
                6
            </td>
        </tr>
    </table>
    <br><br><br>

    <!-- script를 이용한 자동 생성 -->
    <script>
        col_ar = ['red', 'blue', 'skyblue', '#f0cddb', 'f4aeb1', 'e6a513']
        t = "<table border=1>"
        num = 1
        for(i=0;i<5;i++){
            t+="<tr>"
            for(j=0;j<5;j++){
                x = Math.floor(Math.random() * 6)
                t+="<td align=center bgcolor="+col_ar[x]+">"+ num++ +"</td>"
            }
            t+="</tr>"
        }
        t+="</table>"
        document.write(t)
    </script>
</body>
</html>
```

### 계산기 예문1: JAVA
```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<style>
    .inbox{
        text-align: right;
        height: 40px;
        width: 170px;
        font-size: 15px;

    }
</style>
<body>
    <table border="1">
        <tr>
            <td colspan="4"><input type="text" id="inputbox" class="inbox"></td>
        </tr>
        <tr>
            <td algin="center"><button type="button" id="su7">7</button></td>
            <td algin="center"><button type="button" id="su8">8</button></td>
            <td algin="center"><button type="button" id="su9">9</button></td>
            <td algin="center"><button type="button" id="suplus">+</button></td>
        </tr>
        <tr>
            <td algin="center"><button type="button" id="su4">4</button></td>
            <td algin="center"><button type="button" id="su5">5</button></td>
            <td algin="center"><button type="button" id="su6">6</button></td>
            <td algin="center"><button type="button" id="suminus">-</button></td>
        </tr>
        <tr>
            <td algin="center"><button type="button" id="su1">1</button></td>
            <td algin="center"><button type="button" id="su2">2</button></td>
            <td algin="center"><button type="button" id="su3">3</button></td>
            <td algin="center"><button type="button" id="sumul">*</button></td>
        </tr>
        <tr>
            <td algin="center"><button type="button" id="su0">0</button></td>
            <td algin="center"><button type="button" id="susum">계산</button></td>
            <td algin="center"><button type="button" id="suback">B</button></td>
            <td algin="center"><button type="button" id="sudiv">/</button></td>
        </tr>
    </table>
</body>
<script>
    ar = [] // 디스플레이 상 표시
    input = document.getElementById("inputbox")
    document.getElementById("su7").addEventListener('click', on7)
    document.getElementById("suplus").addEventListener('click', onplus)
    document.getElementById("susum").addEventListener('click', onsum)
    document.getElementById("suback").addEventListener('click', onback)

    // 상황에 따라 각 버튼에 대한 함수 생성 or 모든 버튼에 대해 통일된 함수 생성
    // 디스플레이에 숫자 표시 함수
    function on7(){
        ar.push(7)
        input.value = ar.join("") // 숫자 사이의 "," 삭제
        // 디스플레이 상에서 우측->좌측 입력  
    }
    function onplus(){
        ar.push("+")
        input.value = ar.join("") // 숫자 사이의 "," 삭제
        // 디스플레이 상에서 우측->좌측 입력  
    }
    function onsum(){
        t = input.value
        ar=[]
        input.value = eval(t) // eval(): 문자형을 입력해 수식들을 포함되어 있는 경우를 계산
    }
    function onback(){
        ar.pop()
        input.value = ar.join("")
    }
</script>
</html>
```

### 계산기 예문2: JAVA Script
```
<!-- JS로 계산기 만들기 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="test.css">
</head>
<body>
    <table border="1">
        <tr>
            <td colspan="4"><input type="text" id="input"></td>
        </tr>
        <tbody id="buttons"></tbody> <!-- 버튼이 들어갈 구역을 지정 -->
    </table>
    <script>
        tt = '' // 깨짐방지용
        input = document.getElementById("input")
        buttons = document.getElementById("buttons")
        btnLayout = [
            [7,8,9,'+'],
            [4,5,6,'-'],
            [1,2,3,'*'],
            [0,'=','C','/'] // C: clear
        ]

        // forEach 를 이용해 button 구성
        btnLayout.forEach(row => { // rowby로 실행, 관계연산식
            tr = document.createElement('tr')
            row.forEach(text=>{
                td = document.createElement('td')
                btn = document.createElement('button')
                btn.textContent = text
                btn.onclick = () => on(text) // on에서 각 text를 가지고 작업
                td.appendChild(btn)
                tr.appendChild(td)
            });
            buttons.appendChild(tr)
        });

        on = (val) => { // function 이 생략됨, 보내기 함수
            if (val == '='){
                tt = eval(tt) // 사칙연산도 알아서 우선순위에 맞게 계산
            }
            else if(val == 'C'){
                tt = ''
            }
            else{
                tt += val
                input.value = tt
            }
        }
    </script>
</body>
</html>

/* css 파일에서 F5 누르면 안됨 -> 인터넷 연결이 아니라 로컬로 연결됨*/ 

button{
    width: 50px;
    height: 50px;
    font-size: 20px;
}

button:hover{ /* hover: 커서를 버튼에 올렸을 때 */
    background-color: red;
}
```

### 계산기 예문3
```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <table border="1">
        <tr>
            <td colspan="4"><input type="text" id="input"></td>
        </tr>
        <tr>
            <td><button type="button" onclick="on(7)">7</button></td>
            <td><button type="button" onclick="on(8)">8</button></td>
            <td><button type="button" onclick="on(9)">9</button></td>
            <td><button type="button" onclick="on('+')">+</button></td>
        </tr>
        <tr>
            <td><button type="button" onclick="on(4)">4</button></td>
            <td><button type="button" onclick="on(5)">5</button></td>
            <td><button type="button" onclick="on(6)">6</button></td>
            <td><button type="button" onclick="on('-')">-</button></td>
        </tr>
        <tr>
            <td><button type="button" onclick="on(1)">1</button></td>
            <td><button type="button" onclick="on(2)">2</button></td>
            <td><button type="button" onclick="on(3)">3</button></td>
            <td><button type="button" onclick="on('*')">*</button></td>
        </tr>
        <tr>
            <td><button type="button" onclick="on(0)">0</button></td>
            <td><button type="button" onclick="calc()">=</button></td>
            <td><button type="button" onclick="back()">B</button></td>
            <td><button type="button" onclick="on('/')">/</button></td>
        </tr>
    </table>
</body>
<script>
    expr = ""
    input = document.getElementById("input")
    function on(ch){
        expr += ch
        input.value = expr
    }
    function back(){
        expr = expr.slice(0,-1)
        input.value = expr
    }
    function tokenize(s){
        tokens = []
        num = ""

        for(i = 0 ; i < s.length ; i++){
            ch = s[i]
            if(isDigit(ch)){
                num += ch
            }
            else{
                if(num !== ""){
                    tokens.push(Number(num))
                    num=""
                }
                tokens.push(ch)
            }
        }
        if(num !== ""){
            tokens.push(Number(num))
        }
        return tokens
    }
    function isDigit(ch){
        return(ch>='0' && ch<='9') || ch ==='.'
    }
    function evaluate(tokens){
        stage = []
        for (i=0;i<tokens.length;i++){
            t = tokens[i]
            if(t === '*' || t === '/'){
                left = stage.pop()
                right = tokens[++i]
                stage.push(t==='*'?left*right:left/right)
            }
            else{
                stage.push(t)
            }

        }
        result = stage[0]
        for(i=1;i<stage.length;i+=2){
            op = stage[i]
            value = stage[i+1]
            if(op==='+'){
                result += value
            }
            else{
                result -= value
            }
        }
        return result
    }
    function calc(){
        tokens = tokenize(expr) // 문자열 분리
        result = evaluate(tokens) // 사칙연산 구현
        expr = String(result)
        input.value = expr
    }
</script>
</html>
```

### 계산기 예문4: 직접 구성

```
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="test.css">
</head>
<body class="calculator-page">
    <main class="calculator-frame" data-mode="normal" aria-label="계산기">
    <div class="calculator-tabs" role="tablist" aria-label="계산기 종류">
        <button type="button" id="tab-normal" role="tab" data-mode="normal" aria-selected="true" aria-controls="calculator-panel">일반 계산기</button>
        <button type="button" id="tab-scientific" role="tab" data-mode="scientific" aria-selected="false" aria-controls="calculator-panel" tabindex="-1">공학용 계산기</button>
    </div>
    <div id="calculator-panel" role="tabpanel" aria-labelledby="tab-normal">
    <div id="scientific-options" hidden>
        <label for="angle-unit">각도 단위</label>
        <select id="angle-unit">
            <option value="DEG">DEG (도)</option>
            <option value="RAD">RAD (라디안)</option>
        </select>
    </div>
    <table class="calculator">
        <tr>
            <td id="display" colspan="6">
                <div class="display-box">
                    <div id="expression" aria-label="계산식"></div>
                    <input type="text" id="input" value="0" aria-label="입력 및 결과">
                </div>
            </td>
        </tr>
        <tbody id="buttons"></tbody> <!-- 버튼이 들어갈 구역을 지정 -->
    </table>
    </div>
    </main>
    <script>
        const input = document.getElementById("input")
        const expressionDisplay = document.getElementById('expression')
        let expression = '0' // 계산에는 쉼표 없는 원본을 사용한다.
        const buttons = document.getElementById("buttons")
        const tabs = document.querySelectorAll('.calculator-tabs [role="tab"]')
        let currentMode = 'normal'
        const angleUnit = document.getElementById('angle-unit')
        let isResult = false // 계산 결과에는 CE를 적용하지 않는다.
        let isInitial = true // 기본 0은 새 숫자를 입력하면 교체한다.
        const layouts = {
            normal: [
            [1,2,3,'+','-','('],
            [4,5,6,'*','/',')'],
            [7,8,9,'%','√','^2'],
            [0,'.','±','CE','AC','='] // CE: 마지막 문자 삭제, AC: 전체 삭제
            ],
            scientific: [
                [1,2,3,'+','-','(','sin','cos','tan'],
                [4,5,6,'*','/',')','ln','log','eˣ'],
                [7,8,9,'%','√','^2','π','e','xʸ'],
                [0,'.','±','CE','AC','=','abs','∛','1/x']
            ]
        }

        // 모드에 맞는 버튼만 다시 만들고 공통 입력창은 유지한다.
        function renderButtons() {
        const btnLayout = layouts[currentMode]
        document.querySelector('.calculator-frame').dataset.mode = currentMode
        document.getElementById('scientific-options').hidden = currentMode !== 'scientific'
        const columns = btnLayout[0].length
        document.getElementById('display').colSpan = columns
        buttons.replaceChildren()
        btnLayout.forEach(row => {
            const tr = document.createElement('tr')
            row.forEach((text, column)=>{
                const td = document.createElement('td')
                if (text === '=') td.colSpan = columns - row.length + 1
                const btn = document.createElement('button')
                const labels = { '*': '×', '/': '÷', '^2': 'x²' }
                btn.textContent = labels[text] || text
                if (text === '=') btn.className = 'key-equals'
                else if (currentMode === 'scientific' && column >= 6) btn.className = 'key-scientific'
                else if (typeof text === 'number' || text === '.' || text === '±') btn.className = 'key-number'
                else if (text === 'CE') btn.className = 'key-clear'
                else if (text === 'AC') btn.className = 'key-clear'
                else btn.className = 'key-operator'
                if (currentMode === 'scientific' && column === 6) btn.classList.add('science-start')
                btn.onclick = () => on(text) // on에서 각 text를 가지고 작업
                td.appendChild(btn)
                tr.appendChild(td)
            });
            buttons.appendChild(tr)
        });
        }
        tabs.forEach(tab => {
            tab.addEventListener('keydown', event => {
                const index = Array.from(tabs).indexOf(tab)
                let target
                if (event.key === 'ArrowRight') target = tabs[(index + 1) % tabs.length]
                if (event.key === 'ArrowLeft') target = tabs[(index + tabs.length - 1) % tabs.length]
                if (event.key === 'Home') target = tabs[0]
                if (event.key === 'End') target = tabs[tabs.length - 1]
                if (target) {
                    event.preventDefault()
                    target.focus()
                    target.click()
                }
            })
            tab.addEventListener('click', () => {
                if (tab.disabled || !layouts[tab.dataset.mode]) return
                currentMode = tab.dataset.mode
                tabs.forEach(item => {
                    const selected = item === tab
                    item.setAttribute('aria-selected', String(selected))
                    item.tabIndex = selected ? 0 : -1
                })
                document.getElementById('calculator-panel').setAttribute('aria-labelledby', tab.id)
                renderButtons()
            })
        })
        renderButtons()

        // 연속된 사칙연산 기호는 마지막 기호만 남긴다. 예: 2+*3 → 2*3
        function normalizeOperators(expression) {
            return expression.replace(/[+*/-](?:\s*[+*/-])+/g, operators => operators.trim().slice(-1))
        }

        // 문자열을 실행하지 않고 숫자와 연산자만 해석하는 계산 함수
        function calculate(expression, unit = angleUnit.value) {
            expression = normalizeOperators(expression)
            let position = 0

            function skipSpaces() {
                while (/\s/.test(expression[position] || '') && position < expression.length) {
                    position++
                }
            }

            function consume(symbol) {
                skipSpaces()
                if (expression[position] !== symbol) return false
                position++
                return true
            }

            // 덧셈·뺄셈보다 곱셈·나눗셈을 먼저 계산한다.
            function parseExpression() {
                let value = parseTerm()
                while (true) {
                    if (consume('+')) value += parseTerm()
                    else if (consume('-')) value -= parseTerm()
                    else return value
                }
            }

            function parseTerm() {
                let value = parseUnary()
                while (true) {
                    if (consume('*')) value *= parseUnary()
                    else if (consume('/')) {
                        const divisor = parseUnary()
                        if (divisor === 0) throw new Error('0으로 나눌 수 없습니다.')
                        value /= divisor
                    } else if (consume('%')) { // %는 나머지 연산
                        const divisor = parseUnary()
                        if (divisor === 0) throw new Error('0으로 나눌 수 없습니다.')
                        value %= divisor
                    } else return value
                }
            }

            // 음수와 양수 부호, 제곱근 처리
            function parseUnary() {
                if (consume('+')) return parseUnary()
                if (consume('-')) return -parseUnary()
                if (consume('√')) {
                    const value = parseUnary()
                    if (value < 0) throw new Error('음수의 제곱근은 계산할 수 없습니다.')
                    return Math.sqrt(value)
                }
                return parsePower()
            }

            function parsePower() {
                const value = parsePrimary()
                // 거듭제곱은 오른쪽부터 계산: 2^3^2 → 2^(3^2)
                return consume('^') ? value ** parseUnary() : value
            }

            function parsePrimary() {
                if (consume('(')) {
                    const value = parseExpression()
                    if (!consume(')')) throw new Error('닫는 괄호가 필요합니다.')
                    return value
                }
                skipSpaces()
                if (consume('π')) return Math.PI
                const name = expression.slice(position).match(/^[a-z]+/)
                if (name) {
                    position += name[0].length
                    if (name[0] === 'e') return Math.E
                    if (!['sin','cos','tan','ln','log','exp','abs','cbrt','inv'].includes(name[0])) {
                        throw new Error('지원하지 않는 함수입니다.')
                    }
                    if (!consume('(')) throw new Error('함수 뒤에 여는 괄호가 필요합니다.')
                    const argument = parseExpression()
                    if (!consume(')')) throw new Error('닫는 괄호가 필요합니다.')
                    if (name[0] === 'abs') return Math.abs(argument)
                    if (name[0] === 'cbrt') return Math.cbrt(argument)
                    if (name[0] === 'inv') {
                        if (argument === 0) throw new Error('0의 역수는 계산할 수 없습니다.')
                        return 1 / argument
                    }
                    if (name[0] === 'ln' || name[0] === 'log') {
                        if (argument <= 0) throw new Error('로그에는 양수를 입력해야 합니다.')
                        return name[0] === 'ln' ? Math.log(argument) : Math.log10(argument)
                    }
                    if (name[0] === 'exp') return Math.exp(argument)
                    const radians = unit === 'DEG' ? (argument % 360) * Math.PI / 180 : argument
                    if (name[0] === 'tan' && Math.abs(Math.cos(radians)) < 1e-14) {
                        throw new Error('이 각도에서는 tan을 계산할 수 없습니다.')
                    }
                    return Math[name[0]](radians)
                }
                // 소수와 지수 표기(계산 결과의 1e+21 등)도 숫자로 읽는다.
                const number = expression.slice(position).match(/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/)
                if (!number) throw new Error('숫자 또는 괄호가 필요합니다.')
                position += number[0].length
                return Number(number[0])
            }

            const result = parseExpression()
            skipSpaces()
            if (position !== expression.length) throw new Error('잘못된 수식입니다.')
            if (!Number.isFinite(result)) throw new Error('계산 가능한 범위를 벗어났습니다.')
            return result
        }

        function formatNumbers(value) {
            return value.replace(/\d+(?:\.\d*)?(?:[eE][+-]?\d+)?/g, number => {
                const [mantissa, exponent] = number.split(/(?=[eE])/)
                const [integer, fraction] = mantissa.split('.')
                return integer.replace(/\B(?=(\d{3})+(?!\d))/g, ',')
                    + (fraction === undefined ? '' : '.' + fraction) + (exponent || '')
            })
        }

        function toggleSign(value) {
            // 음수를 괄호로 감싸 연속 연산자 교체 규칙과 충돌하지 않게 한다.
            const number = '(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?'
            const negative = value.match(new RegExp('\\(-(' + number + ')\\)$'))
            if (negative) return value.slice(0, negative.index) + negative[1]
            if (new RegExp('^-' + number + '$').test(value)) return value.slice(1)
            const last = value.match(new RegExp(number + '$'))
            if (last) return value.slice(0, last.index) + '(-' + last[0] + ')'
            // 닫힌 괄호 식에도 부호 전환을 적용한다.
            if (value.endsWith(')')) {
                let depth = 0
                for (let i = value.length - 1; i >= 0; i--) {
                    if (value[i] === ')') depth++
                    if (value[i] === '(') depth--
                    if (depth === 0) return value.slice(0, i) + '(-' + value.slice(i) + ')'
                }
            }
            return value
        }

        function on(val) {
            input.setCustomValidity('')
            if (val === '=') {
                try {
                    const result = calculate(expression)
                    const unitLabel = /(?:sin|cos|tan)\(/.test(expression) ? ' [' + angleUnit.value + ']' : ''
                    expressionDisplay.textContent = formatNumbers(expression) + unitLabel + ' ='
                    expression = String(result)
                    isResult = true
                    isInitial = false
                } catch (error) {
                    input.setCustomValidity(error.message)
                    input.reportValidity()
                }
            } else if (val === 'CE') {
                if (isResult) return
                expression = expression.slice(0, -1) || '0'
                isInitial = expression === '0'
                expressionDisplay.textContent = ''
            } else if (val === 'AC') {
                expression = '0'
                expressionDisplay.textContent = ''
                isResult = false
                isInitial = true
            } else if (val === '±') {
                expression = isResult ? String(-Number(expression)) : toggleSign(expression)
                expressionDisplay.textContent = ''
                isInitial = expression === '0'
            } else {
                const functionNames = { sin: 'sin', cos: 'cos', tan: 'tan', ln: 'ln', log: 'log', 'eˣ': 'exp', abs: 'abs', '∛': 'cbrt', '1/x': 'inv' }
                if (Object.hasOwn(functionNames, val)) {
                    const name = functionNames[val]
                    expression = isResult ? name + '(' + expression + ')' :
                        (isInitial ? '' : expression) + name + '('
                    expressionDisplay.textContent = ''
                    isInitial = false
                    isResult = false
                    input.value = formatNumbers(expression)
                    return
                }
                if (val === 'xʸ') val = '^'
                // 새 숫자는 새 계산을 시작하고, 연산자는 결과에 이어 붙인다.
                if ((isInitial || isResult) && /^[\d.(√πe]/.test(String(val))) {
                    expression = val === '.' ? '0' : ''
                }
                expression = normalizeOperators(expression + val)
                expressionDisplay.textContent = ''
                isResult = false
                isInitial = false
            }
            input.value = formatNumbers(expression)
        }
        input.addEventListener('input', () => {
            const cursor = input.selectionStart
            const normalizedPrefix = normalizeOperators(input.value.slice(0, cursor).replace(/,/g, ''))
            expression = normalizeOperators(input.value.replace(/,/g, '')) || '0'
            input.value = formatNumbers(expression)
            expressionDisplay.textContent = ''
            isResult = false
            isInitial = expression === '0'
            const nextCursor = formatNumbers(normalizedPrefix).length
            input.setSelectionRange(nextCursor, nextCursor)
            input.setCustomValidity('')
        })
    </script>
</body>
</html>
```
