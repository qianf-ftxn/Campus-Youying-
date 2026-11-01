"""Hierarchy-aware policy chunks. Source text is retained; headings are navigation."""
from __future__ import annotations

import re
from dataclasses import dataclass

CHUNKING_VERSION = "hierarchical-v2"
NUM = r"[一二三四五六七八九十百千万零〇两\d]+"
CN_NUM = r"[一二三四五六七八九十百千万零〇两]+"
ARTICLE = re.compile(rf"^\s*(第{NUM}条)(?:\s|[、：:]|(?=[^一二三四五六七八九十百千万零〇两\d]))")
PAGE = re.compile(r"^\s*<!-- page:(\d+) -->\s*$")


@dataclass
class Unit:
    text: str
    page: int | None


def chunking_signature(parent_size: int, child_size: int, overlap: int) -> str:
    return f"{CHUNKING_VERSION}:{parent_size}:{child_size}:{overlap}"


def topic_heading(line: str) -> tuple[int, str] | None:
    """Articles are leaves; chapters/sections and short numbered captions are topics."""
    text = line.strip()
    match = re.match(r"^(#{1,6})\s+(.+)$", text)
    if match:
        return len(match[1]), match[2].strip()
    match = re.match(rf"^(第{NUM}([章节]))\s*(.*)$", text)
    if match:
        return (2 if match[2] == "章" else 3), text
    # Do not mistake an enumerated substantive sentence for a section heading.
    if len(text) <= 48 and not re.search(r"[，。！？；;!?]", text):
        if re.match(rf"^{CN_NUM}[、．.]\s*\S", text):
            return 4, text
        if re.match(rf"^[（(]{NUM}[）)]\s*\S", text):
            return 5, text
    return None


def _is_table(line: str) -> bool:
    return line.strip().startswith("|") and line.count("|") >= 2


def atomic_units(lines: list[tuple[str, int | None]]) -> list[Unit]:
    """Keep each article, numbered step, paragraph and complete table together."""
    result, pending = [], []
    in_article, in_table = False, False

    def flush():
        nonlocal pending
        text = "\n".join(line for line, _ in pending).strip()
        if text:
            page = next((page for line, page in pending if line.strip()), None)
            result.append(Unit(text, page))
        pending = []

    for line, page in lines:
        stripped = line.strip()
        article = bool(ARTICLE.match(line))
        table = _is_table(line)
        subheading = bool(re.match(r"^#{3,6}\s+", stripped))
        step = bool(re.match(rf"^(?:\d+[.、．]|[（(]{NUM}[）)])\s*\S", stripped))
        if article or subheading or (step and not in_article):
            flush()
            in_article = article
        if table != in_table and (table or in_table):
            flush()
        if not stripped and not in_article and not in_table:
            flush()
        else:
            pending.append((line, page))
        in_table = table
    flush()
    return result


def _split_prose(text: str, size: int, overlap: int = 0) -> list[str]:
    """Split long prose at sentence boundaries; only long sentences use hard windows."""
    sentences = [part for part in re.split(r"(?<=[。！？；])", text) if part.strip()]
    result, pending = [], []
    has_new_sentence = False

    def flush():
        nonlocal pending, has_new_sentence
        if not pending or not has_new_sentence:
            return
        result.append("".join(pending).strip())
        tail = []
        for sentence in reversed(pending):
            if sum(map(len, tail)) + len(sentence) > overlap:
                break
            tail.insert(0, sentence)
        pending = tail
        has_new_sentence = False

    for sentence in sentences:
        if len(sentence) > size:
            flush()
            pending = []
            result.extend(chunk_text(sentence, size, overlap))
        else:
            if pending and sum(map(len, pending)) + len(sentence) > size:
                flush()
                if sum(map(len, pending)) + len(sentence) > size:
                    pending = []
            pending.append(sentence)
            has_new_sentence = True
    if pending and has_new_sentence:
        text = "".join(pending).strip()
        if text:
            result.append(text)
    return result


def _pack(units: list[Unit], target: int, *, split_long: bool, overlap: int = 0) -> list[Unit]:
    expanded = []
    for unit in units:
        if len(unit.text) > target and split_long and not any(_is_table(line) for line in unit.text.splitlines()):
            expanded.extend(Unit(part, unit.page) for part in _split_prose(unit.text, target, overlap))
        else:
            expanded.append(unit)
    result, pending = [], []
    for unit in expanded:
        if pending and sum(len(part.text) for part in pending) + len(unit.text) + 2 * len(pending) > target:
            result.append(Unit("\n\n".join(part.text for part in pending), pending[0].page))
            pending = []
        pending.append(unit)
    if pending:
        result.append(Unit("\n\n".join(part.text for part in pending), pending[0].page))
    return result


def _article_path(body: str) -> str:
    articles = [match[1] for line in body.splitlines() if (match := ARTICLE.match(line))]
    if not articles:
        return ""
    return articles[0] if articles[0] == articles[-1] else f"{articles[0]}—{articles[-1]}"


def knowledge_heading(title: str, section: str) -> str:
    parts = []
    for part in [title, *section.split(" / ")]:
        if part and part not in parts:
            parts.append(part)
    return " / ".join(parts)[:512]


def child_heading(parent: str, body: str) -> str:
    article = _article_path(body)
    return knowledge_heading(parent, article) if article else parent


def parent_sections(content: str, target: int) -> list[tuple[str, str, int | None]]:
    """Parents follow topic boundaries; adjacent articles stay within their topic."""
    sections, lines, stack = [], [], []
    page = None
    within_article = False

    def flush():
        nonlocal lines
        if any(line.strip() and topic_heading(line) is None for line, _ in lines):
            sections.append((" / ".join(name for _, name in stack) or "正文", atomic_units(lines)))
        lines = []

    for line in content.replace("\r\n", "\n").splitlines():
        if marker := PAGE.match(line):
            page = int(marker[1])
            continue
        heading = topic_heading(line)
        if ARTICLE.match(line):
            within_article = True
        if heading and heading[0] >= 4 and within_article:
            # Conditions enumerated inside an article belong to that article.
            heading = None
        # Markdown h3+ stay inside h1/h2 topic; Chinese subsections are explicit topics.
        if heading and not (line.strip().startswith("#") and heading[0] > 2):
            flush()
            within_article = False
            level, title = heading
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            # Numbered captions remain in the body for exact source attribution.
            if not line.strip().startswith("#"):
                lines.append((line, page))
        else:
            lines.append((line, page))
    flush()
    result = []
    for heading, units in sections:
        for unit in _pack(units, max(100, target), split_long=True):
            article = _article_path(unit.text)
            path = f"{heading} / {article}" if article else heading
            result.append((path, unit.text, unit.page))
    return result


def child_sections(content: str, size: int, overlap: int) -> list[str]:
    size = max(40, size)
    overlap = min(max(0, overlap), size // 3)
    units = atomic_units([(line, None) for line in content.splitlines()])
    children = []
    for unit in units:
        # Each article/step is its own retrieval unit; do not coalesce sibling articles.
        if len(unit.text) <= size or any(_is_table(line) for line in unit.text.splitlines()):
            children.append(unit.text)
        else:
            children.extend(_split_prose(unit.text, size, overlap))
    return children or ([content] if content.strip() else [])


def chunk_text(content: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", content or "").strip()
    step = max(1, size - overlap)
    return [text[start:start + size] for start in range(0, len(text), step)] if text else []
