# Draft PR #1 — 블로그 콘텐츠 분류와 홈 화면 개편

- PR: [#1 블로그 콘텐츠 분류와 홈 화면 개편](https://github.com/SeokHwanHong/SeokHwanHong.github.io/pull/1)
- 브랜치: `blog-renewal`
- 상태: Draft

## 목적

공부 기록을 중심으로 기존 게시글을 정리하고, 프로젝트와 소개 페이지가 자연스럽게 포트폴리오 역할을 하도록 블로그 구조를 개편합니다.

## 주요 결정

- Minimal Mistakes 기반 구조 유지
- 게시글마다 대표 Category 하나만 사용
- Tag 기능 제거
- `SK Encore DE 2기`에 회고록, 수업 내용, 프로젝트 하위 분류 사용
- 최근 공부 기록을 프로젝트보다 먼저 보여주는 홈 구성
- 기존 게시글의 의미는 바꾸지 않고 Markdown 표시 오류만 교정

## 변경 범위

- 기존 게시글 35개 재분류 및 front matter 정리
- Markdown 제목 계층과 이미지 경로 교정
- 중복된 VQ-VAE 본문을 가진 DEiT 게시글 제거
- 홈, 카테고리, 프로젝트와 소개 페이지 개편
- 한국어 메뉴, 검색, 소개 문구와 프로필 링크 정리
- 블로그 관리 문서와 저장소 README 추가
- 게시글 공통 오류 자동 검사 추가

## 검증 항목

- 게시글 YAML 형식
- Category와 하위 분류 값
- 로컬 이미지 참조
- Markdown 코드 블록, 수식과 Liquid 문법
- Jekyll 빌드
