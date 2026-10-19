"""Versioned, offline official documents distributed with the application."""
from __future__ import annotations

from datetime import date
import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def bundle_root(settings) -> Path:
    root = Path(settings.bundled_knowledge_dir)
    return root if root.is_absolute() else PROJECT_ROOT / root


def _checked_file(root: Path, relative: str, expected_hash: str) -> bytes:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"资料文件缺失或路径不合法：{relative}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"资料校验失败，请恢复仓库原文件或重新收集：{relative}")
    return data


def load_bundle(root: Path) -> tuple[dict, list[tuple[dict, str]]]:
    """Validate the entire package before any database writes; no HTTP calls."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("school"):
        raise ValueError("资料清单版本或学校无效")
    rows, identities = [], set()
    for item in manifest["documents"]:
        identity = item["id"]
        if identity in identities or not identity or "/" in identity or "\\" in identity:
            raise ValueError(f"资料 ID 重复或无效：{identity}")
        identities.add(identity)
        if item["school"] != manifest["school"] or item["domain"] != "policy":
            raise ValueError(f"资料学校或领域不匹配：{identity}")
        parsed = urlparse(item["source_url"])
        host = parsed.hostname or ""
        if parsed.scheme not in {"https", "http"} or not (host == "xidian.edu.cn" or host.endswith(".xidian.edu.cn")):
            raise ValueError(f"资料不是西电官方来源：{identity}")
        if item["status"] not in {"ACTIVE", "DRAFT", "ARCHIVED"}:
            raise ValueError(f"资料状态无效：{identity}")
        for field in ("published_on", "effective_on", "expires_on"):
            if item.get(field):
                date.fromisoformat(item[field])
        text = _checked_file(root, item["content_path"], item["content_sha256"]).decode("utf-8")
        if not text.strip() or len(text.encode()) > 10 * 1024 * 1024:
            raise ValueError(f"资料正文为空或过大：{identity}")
        _checked_file(root, item["original_path"], item["original_sha256"])
        for attachment in item.get("attachments", []):
            if attachment.get("path"):
                _checked_file(root, attachment["path"], attachment["sha256"])
        rows.append((item, text))
    return manifest, rows


def import_bundle(service, settings, root: Path | None = None) -> dict:
    if not settings.bundled_knowledge_enabled:
        return {"enabled": False, "imported": 0, "reason": "资料包导入已关闭"}
    root = root or bundle_root(settings)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if settings.campus_name != manifest["school"]:
        return {"enabled": True, "imported": 0, "reason": "学校配置不同，未将西电资料混入其他学校", "school": manifest["school"]}
    manifest, rows = load_bundle(root)
    from app.models.entities import KnowledgeDocument

    imported, existing = 0, 0
    for position, (item, content) in enumerate(rows, 1):
        source = "bundled/xidian/" + item["id"] + ".md"
        found = service.db.query(KnowledgeDocument).filter_by(source=source, version=item["version"]).first()
        service.ensure_source(source, content, title=item["title"], version=item["version"],
            domain="policy", school=item["school"], department=item["department"],
            source_url=item["source_url"], status=item["status"], is_demo=False,
            **{field: date.fromisoformat(item[field]) if item.get(field) else None
               for field in ("published_on", "effective_on", "expires_on")})
        if found:
            existing += 1
        else:
            imported += 1
        logging.getLogger(__name__).info("Official corpus %d/%d: %s", position, len(rows), item["title"])
    return {"enabled": True, "imported": imported, "existing": existing, "documents": len(rows),
            "school": manifest["school"], "checkedOn": manifest["checked_on"]}


def bundle_status(settings) -> dict:
    if not settings.bundled_knowledge_enabled:
        return {"enabled": False}
    try:
        manifest = json.loads((bundle_root(settings) / "manifest.json").read_text(encoding="utf-8"))
        missing = []
        for item in manifest["documents"]:
            for path in [item["content_path"], item["original_path"],
                         *[a["path"] for a in item.get("attachments", []) if a.get("path")]]:
                candidate = (bundle_root(settings) / path).resolve()
                if not candidate.is_relative_to(bundle_root(settings).resolve()) or not candidate.is_file():
                    missing.append(path)
        return {"enabled": True, "available": not missing, "school": manifest["school"],
                "matchesSchool": manifest["school"] == settings.campus_name,
                "documents": len(manifest["documents"]), "checkedOn": manifest["checked_on"],
                "missingFiles": missing}
    except (OSError, ValueError, KeyError, TypeError):
        return {"enabled": True, "available": False, "error": "随项目分发的官方资料清单缺失或无效，请恢复 knowledge/xidian 或手动上传文件。"}
