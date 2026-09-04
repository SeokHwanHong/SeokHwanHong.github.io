#!/usr/bin/env python3
"""Validate blog post metadata and common Markdown rendering problems."""

from pathlib import Path
from urllib.parse import unquote
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
POSTS = ROOT / "_posts"
CATEGORY_DATA = ROOT / "_data" / "categories.yml"
POST_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}-.+\.md$")


def front_matter(text: str, path: Path) -> tuple[list[str], str]:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"{path}: front matter가 없습니다.")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"{path}: front matter가 닫히지 않았습니다.") from exc
    return lines[1:end], "\n".join(lines[end + 1 :])


def scalar(lines: list[str], key: str) -> str | None:
    prefix = f"{key}:"
    for line in lines:
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            return value.strip('"\'') or None
    return None


def list_value(lines: list[str], key: str) -> list[str]:
    values = []
    collecting = False
    for line in lines:
        if line.startswith(f"{key}:"):
            collecting = True
            inline = line.split(":", 1)[1].strip()
            if inline:
                values.append(inline.strip("[] ").strip('"\''))
            continue
        if collecting and re.match(r"^\s+-\s+", line):
            values.append(re.sub(r"^\s+-\s+", "", line).strip().strip('"\''))
        elif collecting and line and not line.startswith(" "):
            break
    return values


def configured_categories() -> tuple[set[str], set[str]]:
    text = CATEGORY_DATA.read_text(encoding="utf-8")
    parents = set(re.findall(r'^- name: "([^"]+)"', text, flags=re.MULTILINE))
    children = set(re.findall(r'^\s{4}- name: "([^"]+)"', text, flags=re.MULTILINE))
    return parents, children


def local_image_targets(body: str) -> list[str]:
    markdown = re.findall(r"!\[[^]]*\]\(([^)\s]+)", body)
    html = re.findall(r'<img\b[^>]*\bsrc=["\']([^"\']+)', body, flags=re.IGNORECASE)
    return markdown + html


def validate_post(path: Path, parents: set[str], children: set[str]) -> list[str]:
    errors = []
    if not POST_NAME.match(path.name):
        errors.append(f"{path}: 파일명은 YYYY-MM-DD-title.md 형식이어야 합니다.")

    text = path.read_text(encoding="utf-8")
    try:
        metadata, body = front_matter(text, path)
    except ValueError as exc:
        return [str(exc)]

    if not scalar(metadata, "title"):
        errors.append(f"{path}: title이 없습니다.")
    categories = list_value(metadata, "categories")
    if len(categories) != 1:
        errors.append(f"{path}: Category는 하나만 지정해야 합니다: {categories}")
    elif categories[0] not in parents:
        errors.append(f"{path}: 등록되지 않은 Category입니다: {categories[0]}")

    if any(line.startswith(("tag:", "tags:")) for line in metadata):
        errors.append(f"{path}: Tag는 사용하지 않습니다.")
    if any(line.startswith("author_profiel:") for line in metadata):
        errors.append(f"{path}: author_profile 오타가 있습니다.")

    subcategory = scalar(metadata, "subcategory")
    primary = categories[0] if len(categories) == 1 else None
    if primary == "SK Encore DE 2기" and subcategory not in children:
        errors.append(f"{path}: SK Encore 글에는 올바른 subcategory가 필요합니다.")
    elif primary != "SK Encore DE 2기" and subcategory:
        errors.append(f"{path}: subcategory는 SK Encore 글에만 사용합니다.")

    in_code = False
    previous_heading = 1
    for number, line in enumerate(body.splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading:
            level = len(heading.group(1))
            if level == 1:
                errors.append(f"{path}:{number}: 본문 H1은 사용하지 않습니다.")
            if level > previous_heading + 1:
                errors.append(f"{path}:{number}: 제목 단계가 H{previous_heading}에서 H{level}로 건너뜁니다.")
            previous_heading = level
        if re.search(r"\*\*[^*\n]*\s\*\*", line):
            errors.append(f"{path}:{number}: 굵은 글씨 닫기 앞의 공백을 제거해야 합니다.")
    if in_code:
        errors.append(f"{path}: 코드 블록이 닫히지 않았습니다.")
    if body.count("$$") % 2:
        errors.append(f"{path}: 수식 구분자 $$의 개수가 맞지 않습니다.")

    for target in local_image_targets(body):
        if target.startswith(("http://", "https://", "data:")):
            continue
        clean = unquote(target.split("#", 1)[0].split("?", 1)[0])
        resolved = ROOT / clean.lstrip("/") if clean.startswith("/") else path.parent / clean
        if not resolved.exists():
            errors.append(f"{path}: 이미지 파일을 찾을 수 없습니다: {target}")
    return errors


def main() -> int:
    parents, children = configured_categories()
    posts = sorted(POSTS.rglob("*.md"))
    errors = [error for post in posts for error in validate_post(post, parents, children)]
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Blog check passed: {len(posts)} posts, {len(parents)} categories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
