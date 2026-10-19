#!/usr/bin/env python3
"""Refresh the curated public Xidian snapshot; never used by normal app startup.

Extra collector dependency: pip install -r scripts/requirements-knowledge.txt
The app imports committed normalized text without network access or this parser.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse
import zipfile
import xml.etree.ElementTree as ET

import httpx
from bs4 import BeautifulSoup, NavigableString
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = ("抱歉，您是在校外访问", "统一身份认证平台", "您没有访问权限", "请输入验证码")


def official(url):
    parsed = urlparse(url)
    return parsed.scheme in {"https", "http"} and (parsed.hostname == "xidian.edu.cn" or (parsed.hostname or "").endswith(".xidian.edu.cn"))


def fetch(url):
    # Redirect destinations are checked before requesting, not just afterwards.
    with httpx.Client(timeout=35, headers={"User-Agent": "CampusYouying-public-policy-snapshot/1.0"}) as client:
        for _ in range(6):
            if not official(url):
                raise ValueError("只收集 xidian.edu.cn 官方公开资源")
            with client.stream("GET", url) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers["location"])
                    continue
                response.raise_for_status()
                data = bytearray()
                for part in response.iter_bytes():
                    data.extend(part)
                    if len(data) > 25 * 1024 * 1024:
                        raise ValueError("附件超过 25MB，请单独核对")
                return bytes(data), str(response.url), response.headers.get("content-type", "")
    raise ValueError("重定向过多")


def clean(text):
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def markdown(node):
    if isinstance(node, NavigableString):
        return str(node)
    if node.name in {"script", "style", "noscript", "iframe"}:
        return ""
    if node.name == "table":
        # 展开 rowspan/colspan，不能把“硕士/三年级”移到错误列。
        grid = {}
        trs = [tr for tr in node.find_all("tr") if tr.find_parent("table") is node]
        for r, tr in enumerate(trs):
            c = 0
            for cell in tr.find_all(["th", "td"], recursive=False):
                while (r, c) in grid:
                    c += 1
                text = clean(cell.get_text("", strip=False)).replace("\n", " ").replace("|", "\\|")
                height, width = min(100, int(cell.get("rowspan", 1))), min(100, int(cell.get("colspan", 1)))
                for dr in range(height):
                    for dc in range(width):
                        grid[r + dr, c + dc] = text
                c += width
        if not grid:
            return ""
        rows = [[grid.get((r, c), "") for c in range(max(c for _, c in grid) + 1)]
                for r in range(max(r for r, _ in grid) + 1)]
        n = len(rows[0])
        lines = ["| " + " | ".join(r) + " |" for r in rows]
        lines.insert(1, "| " + " | ".join(["---"] * n) + " |")
        return "\n\n" + "\n".join(lines) + "\n\n"
    if node.name == "br":
        return "\n"
    if node.name in {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6"} and not node.find(["div", "table", "p"]):
        text = clean(node.get_text("", strip=False))
        return "\n\n" + text + "\n\n" if text else ""
    return "".join(markdown(child) for child in node.children)


def attachment_text(data, extension):
    if extension == ".pdf":
        pages = [(p.extract_text() or "").strip() for p in PdfReader(io.BytesIO(data)).pages]
        if any(len(p) < 30 for p in pages):
            raise ValueError("PDF 存在无可提取文字的页面，需人工 OCR 核对")
        return "\n\n".join(f"<!-- page:{i+1} -->\n{text}" for i, text in enumerate(pages))
    if extension == ".docx":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            info = z.getinfo("word/document.xml")
            if info.file_size > 10 * 1024 * 1024:
                raise ValueError("Word XML 过大")
            root = ET.fromstring(z.read(info))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        return "\n\n".join(
            "".join(t.text or "" for t in p.findall(".//w:t", ns)) for p in root.findall(".//w:p", ns))
    raise ValueError("当前收集器只提取 PDF/DOCX；旧 DOC 需人工转换")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def collect_pdf(item, destination, checked_on):
    raw, final_url, _ = fetch(item["url"])
    if not raw.startswith(b"%PDF"):
        raise ValueError("官方附件未返回 PDF 正文")
    text = attachment_text(raw, ".pdf")
    identity, category = item["id"], item["category"]
    raw_path = f"originals/{category}/{identity}.pdf"
    text_path = f"texts/{category}/{identity}.md"
    for relative in (raw_path, text_path):
        (destination / relative).parent.mkdir(parents=True, exist_ok=True)
    (destination / raw_path).write_bytes(raw)
    content = f"# {item['title']}\n\n资料适用范围：{item['scope']}。\n\n版本说明：{item['freshness']}。核查日期：{checked_on}。\n\n" + text + "\n"
    (destination / text_path).write_text(content, encoding="utf-8")
    return dict(id=identity, category=category, title=item["title"], school="西安电子科技大学", domain="policy",
        department=item["department"], source_url=item["url"], resolved_url=final_url,
        published_on=None, effective_on=None, expires_on=None, checked_on=checked_on,
        version=f"snapshot-{checked_on.replace('-', '')}-{digest(content.encode())[:12]}", status=item["status"],
        scope=item["scope"], freshness=item["freshness"], content_path=text_path, content_sha256=digest(content.encode()),
        original_path=raw_path, original_sha256=digest(raw), attachments=[], extraction_warnings=[])


def collect(item, destination, checked_on="2026-10-19"):
    if item.get("format") == "pdf":
        return collect_pdf(item, destination, checked_on)
    url = item["url"]
    identity = re.sub(r"[^a-zA-Z0-9]+", "-", urlparse(url).netloc.split(".")[0] + "-" + urlparse(url).path.removeprefix("/info/").removesuffix(".htm")).strip("-")
    category = item["category"]
    raw, final_url, mime = fetch(url)
    soup = BeautifulSoup(raw, "html.parser")
    plain = soup.get_text(" ", strip=True)
    if any(word in plain for word in BLOCKED):
        raise ValueError("官方网页要求校内访问或登录，未抓取正文")
    title = clean(soup.title.get_text()).split("-研究生院")[0].split("-西安电子科技大学")[0].strip() if soup.title else item["title"]
    title = title or item["title"]
    body = soup.select_one(".v_news_content") or soup.select_one("#vsb_content")
    if body is None:
        raise ValueError("找不到官方正文区域；拒绝把导航页当政策")
    date_match = re.search(r"(?:发布时间|发布日期|发表时间)[^0-9]{0,12}(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})", plain)
    published = "-".join([date_match[1], date_match[2].zfill(2), date_match[3].zfill(2)]) if date_match else None
    if not published:
        published = next((text for text in soup.stripped_strings if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text)), None)
    if published and published > checked_on:
        raise ValueError("发布日期晚于本次核查日期")
    raw_path = f"originals/{category}/{identity}.html"
    (destination / raw_path).parent.mkdir(parents=True, exist_ok=True)
    (destination / raw_path).write_bytes(raw)
    body_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", markdown(body)).strip()
    attachments = []
    for a in soup.select("a[href]"):
        href = urljoin(final_url, a["href"])
        text = clean(a.get_text(" ", strip=True))
        if ("download" in href or re.search(r"\.(pdf|docx?)(?:$|\?)", href, re.I)) and text:
            if not any(x["url"] == href for x in attachments):
                attachments.append({"title": text, "url": href})
    warnings = []
    if body.find(["img", "object", "embed"]):
        warnings.append("网页含图像或嵌入内容；正文提取不含图像文字，见原网页/附件")
    if body.select("td[rowspan], td[colspan], th[rowspan], th[colspan]"):
        warnings.append("网页表格含合并单元格，已按原始行列跨度展开并重复相应表头/适用对象")
    extracted = []
    # 短网页往往只挂附件，逐个保存可公开访问的制度附件；不抓取名单或个人申请材料。
    if len(re.findall(r"[\u4e00-\u9fff]", body_text)) < 250:
        for i, att in enumerate(attachments):
            if item.get("attachment_title_contains") and not any(term in att["title"] for term in item["attachment_title_contains"]):
                continue
            if any(w in att["title"] for w in ["名单", "申请表", "名额", "统计表", "信息表", "承诺书", "模板", "汇总表", "推荐表"]):
                continue
            ext_match = re.search(r"\.(pdf|docx?)", att["title"], re.I) or re.search(r"\.(pdf|docx?)", att["url"], re.I)
            if not ext_match:
                continue
            ext = "." + ext_match[1].lower()
            if ext not in {".pdf", ".docx"}:
                warnings.append("旧 DOC 附件未提取：" + att["title"])
                continue
            try:
                data, att_url, _ = fetch(att["url"])
                if not (data.startswith(b"%PDF") if ext == ".pdf" else data.startswith(b"PK")):
                    raise ValueError("附件返回非预期文件（可能需认证）")
                path = f"originals/{category}/{identity}-attachment-{i+1}{ext}"
                (destination / path).write_bytes(data)
                att.update(path=path, sha256=digest(data), resolved_url=att_url)
                text = attachment_text(data, ext)
                if len(re.findall(r"[\u4e00-\u9fff]", text)) < 100:
                    raise ValueError("附件可检索正文过少")
                extracted.append("## 附件：" + att["title"] + "\n\n" + text)
            except Exception as exc:
                warnings.append(att["title"] + ": " + str(exc))
    if extracted:
        body_text += "\n\n" + "\n\n".join(extracted)
    status = item["status"]
    if len(re.findall(r"[\u4e00-\u9fff]", body_text)) < 200:
        status = "DRAFT"
        warnings.append("未取得充分正文，保留原始资料但不默认参与检索")
    prefix = f"# {title}\n\n资料适用范围：{item['scope']}。\n\n网页发布日期：{published or '官网正文未标明'}；资料核查日期：{checked_on}。发布日期不等同于制度生效日期。\n\n"
    text = prefix + body_text + "\n"
    text_path = f"texts/{category}/{identity}.md"
    (destination / text_path).parent.mkdir(parents=True, exist_ok=True)
    (destination / text_path).write_text(text, encoding="utf-8")
    return dict(id=identity, category=category, title=title, school="西安电子科技大学", domain="policy",
        department=item["department"], source_url=url, resolved_url=final_url, index_url=item.get("index_url"),
        published_on=published, effective_on=item.get("effective_on"), expires_on=item.get("expires_on"),
        checked_on=checked_on, version=f"snapshot-{checked_on.replace('-', '')}-{digest(text.encode())[:12]}", status=status,
        scope=item["scope"], freshness=item["freshness"], content_path=text_path, content_sha256=digest(text.encode()),
        original_path=raw_path, original_sha256=digest(raw), attachments=attachments, extraction_warnings=warnings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "knowledge/xidian")
    parser.add_argument("--checked-on", help="核查日期 YYYY-MM-DD；默认沿用目录中的日期")
    args = parser.parse_args()
    catalog = json.loads((args.output / "catalog.json").read_text())
    checked_on = args.checked_on or catalog["checked_on"]
    datetime.strptime(checked_on, "%Y-%m-%d")
    entries, failures = [], []
    def run(item):
        try:
            entry = collect(item, args.output, checked_on)
            print(entry["id"], entry["status"], entry["title"], flush=True)
            return entry, None
        except Exception as exc:
            print("UNAVAILABLE", item["url"], str(exc), flush=True)
            return None, {**item, "reason": str(exc)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for entry, failure in pool.map(run, catalog["documents"]):
            if entry:
                entries.append(entry)
            if failure:
                failures.append(failure)
    manifest = {"schema_version": 1, "school": catalog["school"], "checked_on": checked_on,
                "documents": sorted(entries, key=lambda d: (d["category"], d["id"])),
                "unavailable": [*catalog.get("known_gaps", []), *failures], "excluded": catalog.get("excluded", [])}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("SAVED", len(entries), "UNAVAILABLE", len(failures), flush=True)

if __name__ == "__main__":
    main()
