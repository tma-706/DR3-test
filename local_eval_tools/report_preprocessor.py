"""Faithful report normalization for local DR3 evaluation."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pymupdf
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from evaluators.utils.base import EvalConfig
from evaluators.utils.llm_client import extract_json_from_text


class PreprocessError(RuntimeError):
    """Raised when a report cannot be normalized faithfully."""


@dataclass
class PreprocessResult:
    text: str
    metadata: Dict[str, Any]
    reused: bool = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _page_text(page: pymupdf.Page) -> str:
    """Extract every text block in reading order without a character cap."""
    blocks = page.get_text("blocks", sort=True)
    text_blocks = []
    for block in blocks:
        if len(block) < 5 or not isinstance(block[4], str):
            continue
        value = block[4].strip()
        if value:
            text_blocks.append(value)
    return "\n\n".join(text_blocks)


def _visual_candidates(document: pymupdf.Document) -> Dict[int, List[Dict[str, Any]]]:
    """Find large, central raster/vector visuals and reject small decorations."""
    xref_frequency: Dict[int, int] = {}
    page_images: Dict[int, List[Dict[str, Any]]] = {}
    for page_index, page in enumerate(document):
        infos = page.get_image_info(xrefs=True)
        page_images[page_index] = infos
        for info in infos:
            xref = int(info.get("xref", 0))
            if xref:
                xref_frequency[xref] = xref_frequency.get(xref, 0) + 1

    candidates: Dict[int, List[Dict[str, Any]]] = {}
    for page_index, page in enumerate(document):
        page_area = max(page.rect.get_area(), 1.0)
        accepted: List[Dict[str, Any]] = []
        for info in page_images[page_index]:
            bbox = pymupdf.Rect(info["bbox"])
            area_ratio = bbox.get_area() / page_area
            center_y = (bbox.y0 + bbox.y1) / 2 / max(page.rect.height, 1.0)
            xref = int(info.get("xref", 0))
            repeated_decoration = xref and xref_frequency.get(xref, 0) >= max(3, len(document) // 2)
            if area_ratio < 0.08 or center_y < 0.10 or center_y > 0.90 or repeated_decoration:
                continue
            accepted.append(
                {
                    "kind": "raster",
                    "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                    "area_ratio": round(area_ratio, 4),
                    "xref": xref,
                }
            )

        drawing_count = len(page.get_drawings())
        text_chars = len(page.get_text("text"))
        if not accepted and drawing_count >= 20 and text_chars < 1500:
            accepted.append(
                {
                    "kind": "vector_page",
                    "bbox": [page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1],
                    "area_ratio": 1.0,
                    "drawing_count": drawing_count,
                }
            )
        if accepted:
            candidates[page_index] = accepted
    return candidates


def _render_clip(page: pymupdf.Page, bbox: List[float]) -> bytes:
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=pymupdf.Rect(bbox), alpha=False)
    return pixmap.tobytes("png")


def _describe_visual(
    png_bytes: bytes,
    page_number: int,
    config: EvalConfig,
    model: str,
) -> Dict[str, Any]:
    if not config.api_key or not config.base_url or not model:
        raise PreprocessError(
            f"Page {page_number} contains a substantive visual but OpenRouter vision config is missing"
        )

    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        max_retries=0,
        timeout=120.0,
    )
    image_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    prompt = (
        "Transcribe this report visual faithfully for evaluation. Preserve every visible title, "
        "label, value, legend, relationship, flow step, and table cell that carries meaning. "
        "Do not infer facts that are not visible. Return JSON only with keys: substantive "
        "(boolean), markdown (string), reason (short string). If it is only a logo, border, "
        "background, or decoration, set substantive=false and markdown=''."
    )

    transient = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                temperature=config.temperature,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            parsed = extract_json_from_text(content)
            if not isinstance(parsed, dict):
                raise PreprocessError(f"Vision response for page {page_number} was not valid JSON")
            return {
                "substantive": bool(parsed.get("substantive")),
                "markdown": str(parsed.get("markdown", "")).strip(),
                "reason": str(parsed.get("reason", "")).strip(),
                "model": model,
            }
        except (AuthenticationError, PermissionDeniedError) as exc:
            raise PreprocessError(f"OpenRouter authorization failed: {exc}") from exc
        except transient as exc:
            if attempt == 2:
                raise PreprocessError(f"Vision request failed after retries: {exc}") from exc
            time.sleep(2 ** attempt)
        except PreprocessError:
            raise
        except Exception as exc:
            raise PreprocessError(f"Vision request failed: {exc}") from exc
    raise PreprocessError("Vision request failed")


def inspect_report(path: Path) -> Dict[str, Any]:
    """Read-only report inspection used by --dry-run."""
    if path.suffix.lower() in {".md", ".txt"}:
        return {
            "format": path.suffix.lower().lstrip("."),
            "characters": len(path.read_text(encoding="utf-8")),
            "pages": None,
            "visual_candidates": [],
        }
    if path.suffix.lower() != ".pdf":
        raise PreprocessError(f"Unsupported report format: {path.suffix}")
    document = pymupdf.open(path)
    try:
        candidates = _visual_candidates(document)
        return {
            "format": "pdf",
            "pages": len(document),
            "characters": sum(len(page.get_text("text")) for page in document),
            "visual_candidates": [
                {"page": index + 1, "count": len(items)}
                for index, items in candidates.items()
            ],
        }
    finally:
        document.close()


def preprocess_report(
    report_path: Path,
    output_dir: Path,
    config: EvalConfig,
    overwrite: bool = False,
) -> PreprocessResult:
    output_path = output_dir / "report_for_eval.md"
    metadata_path = output_dir / "preprocess_metadata.json"
    fingerprint = sha256_file(report_path)

    if not overwrite and output_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("status") == "success"
                and metadata.get("source_sha256") == fingerprint
                and metadata.get("vision_model", "") == os.getenv("OPENROUTER_VISION_MODEL", "")
            ):
                return PreprocessResult(
                    text=output_path.read_text(encoding="utf-8"),
                    metadata=metadata,
                    reused=True,
                )
        except (OSError, json.JSONDecodeError):
            pass

    suffix = report_path.suffix.lower()
    metadata: Dict[str, Any] = {
        "status": "success",
        "source_report": str(report_path),
        "source_sha256": fingerprint,
        "source_format": suffix.lstrip("."),
        "vision_model": os.getenv("OPENROUTER_VISION_MODEL", ""),
        "vision_candidates": [],
        "vision_used": [],
        "report_truncated": False,
    }

    if suffix in {".md", ".txt"}:
        text = report_path.read_text(encoding="utf-8")
        metadata.update({"pages": None, "extracted_characters": len(text)})
    elif suffix == ".pdf":
        document = pymupdf.open(report_path)
        try:
            candidates = _visual_candidates(document)
            chunks: List[str] = []
            for page_index, page in enumerate(document):
                page_number = page_index + 1
                chunks.append(f"<!-- REPORT_PAGE: {page_number} -->\n\n{_page_text(page)}")
                for visual_index, candidate in enumerate(candidates.get(page_index, []), start=1):
                    metadata["vision_candidates"].append(
                        {"page": page_number, "index": visual_index, **candidate}
                    )
                    visual = _describe_visual(
                        _render_clip(page, candidate["bbox"]),
                        page_number,
                        config,
                        metadata["vision_model"],
                    )
                    metadata["vision_used"].append(
                        {
                            "page": page_number,
                            "index": visual_index,
                            "substantive": visual["substantive"],
                            "reason": visual["reason"],
                            "model": visual["model"],
                        }
                    )
                    if visual["substantive"] and visual["markdown"]:
                        chunks.append(
                            f"<!-- REPORT_VISUAL: page={page_number}, index={visual_index} -->"
                            f"\n\n{visual['markdown']}"
                        )
            text = "\n\n".join(chunks).strip() + "\n"
            metadata.update(
                {
                    "pages": len(document),
                    "extracted_characters": len(text),
                }
            )
        finally:
            document.close()
    else:
        raise PreprocessError(f"Unsupported report format: {suffix}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return PreprocessResult(text=text, metadata=metadata, reused=False)
