# 블로그 관리

이 디렉터리는 블로그에 노출되는 콘텐츠가 아니라, 저장소를 일관되게 관리하기 위한 기준과 주요 변경 기록을 보관합니다.

## 문서 구성

| 문서 | 내용 |
|---|---|
| [`category-guide.md`](category-guide.md) | Category와 하위 분류 운영 기준 |
| [`post-guide.md`](post-guide.md) | Markdown 게시글 작성 및 점검 기준 |
| [`changes/`](changes/) | 규모가 큰 변경 작업과 관련 PR 기록 |
| [`check_blog.py`](check_blog.py) | 게시글 메타데이터, 문법과 이미지 경로 자동 검사 |

## 변경 기록 원칙

- GitHub Pull Request는 커밋, 검토 의견과 빌드 결과를 확인하는 기준입니다.
- `changes/` 문서는 나중에 다시 참고할 가치가 있는 변경의 목적과 결정 사항만 요약합니다.
- 단순 오탈자나 게시글 한 건 추가처럼 작은 작업은 별도 문서를 만들지 않습니다.
- 파일명은 `PR번호-작업명.md` 형식을 사용합니다. 예: `001-blog-renewal.md`

## 기본 작업 흐름

1. `master`에서 작업 브랜치를 만듭니다.
2. 게시글과 설정을 수정합니다.
3. Markdown, 이미지 경로와 Jekyll 빌드를 확인합니다.
4. Draft PR에서 화면과 변경 범위를 검토합니다.
5. 검토가 끝나면 `master`에 병합합니다.

로컬에서 공통 오류를 확인하려면 저장소 루트에서 다음 명령을 실행합니다.

```bash
python blog-management/check_blog.py
```
