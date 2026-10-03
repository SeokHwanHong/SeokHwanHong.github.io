# 게시글 작성 기준

## 파일과 기본 정보

- 게시글은 `_posts/`에 `YYYY-MM-DD-title.md` 형식으로 저장합니다.
- 파일명에는 공백 대신 하이픈을 사용하는 것을 권장합니다.
- 제목은 front matter의 `title`에서 관리하고 본문에 같은 H1 제목을 반복하지 않습니다.
- Category는 하나만 지정하며 `tags`는 작성하지 않습니다.

```yaml
---
title: "게시글 제목"
date: 2026-09-04
categories:
  - "Data Engineering"
author_profile: true
---
```

`SK Encore DE 2기` 글에는 다음 중 하나의 `subcategory`를 추가합니다.

```yaml
subcategory: "회고록"
```

## Markdown

- 제목은 `##`부터 시작해 순서대로 계층을 구성합니다.
- 코드 블록에는 가능한 한 언어를 지정합니다.
- 표, 목록과 코드 블록 앞뒤에는 빈 줄을 둡니다.
- Liquid 문법 예시인 `{{ ... }}`와 `{% ... %}`를 그대로 보여줄 때는 `raw` 블록으로 감쌉니다.
- 수식은 인라인 `$...$`, 블록 `$$...$$` 형식을 사용하고 구분자를 닫았는지 확인합니다.

## 이미지와 링크

- 이미지는 `images/주제명/` 아래에 모아 관리합니다.
- 이미지 경로는 사이트 루트 기준으로 작성합니다.
- 모든 이미지에 내용을 설명하는 대체 텍스트를 작성합니다.

```markdown
![이미지 설명](/images/topic/image.png)
```

## 발행 전 확인

- front matter가 정상적인 YAML인지 확인
- 제목 계층, 코드 블록과 수식 구분자 확인
- 이미지와 내부 링크가 실제 파일을 가리키는지 확인
- PC와 모바일에서 표, 코드와 수식이 본문 밖으로 넘치지 않는지 확인
- Jekyll 빌드 성공 여부 확인

공통 오류는 다음 명령으로 검사합니다.

```bash
python blog-management/check_blog.py
```
