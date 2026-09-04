# Seokhwan Hong — Data & Engineering Notes

데이터와 AI를 공부하고, 데이터 엔지니어링과 서비스 구현으로 관심 영역을 확장해 가는 과정을 기록하는 기술 블로그입니다.

- 블로그: <https://seokhwanhong.github.io>
- 운영 기준: [`blog-management/`](blog-management/)

## 주요 디렉터리

| 경로 | 용도 |
|---|---|
| `_posts/` | 발행된 Markdown 게시글 |
| `images/` | 게시글에서 사용하는 이미지 |
| `_pages/` | 카테고리, 프로젝트, 소개 페이지 |
| `_data/` | 카테고리, 프로젝트, 메뉴 데이터 |
| `_layouts/`, `_includes/` | 페이지 구조와 공통 UI |
| `_sass/`, `assets/` | 블로그 스타일과 정적 자원 |
| `blog-management/` | 작성 규칙과 주요 변경 기록 |

## 로컬 실행

Ruby와 Bundler를 설치한 환경에서 다음 명령으로 확인할 수 있습니다.

```bash
bundle install
bundle exec jekyll serve
```

게시글을 추가하거나 분류를 변경할 때는 [`게시글 작성 기준`](blog-management/post-guide.md)과 [`카테고리 기준`](blog-management/category-guide.md)을 먼저 확인합니다.
