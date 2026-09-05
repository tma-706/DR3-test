"""Local adapters that keep the official evaluator prompts and rubrics intact."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pymupdf
from openai import OpenAI

import evaluators.factual_accuracy as factual_accuracy_module
from evaluators.citation_coverage import CitationCoverageEvaluator
from evaluators.factual_accuracy import FactualAccuracyAgentEvaluator
from evaluators.utils.base import EvalConfig, EvalResult
from local_eval_tools.citation_resolver import (
    CitationResolution,
    resolve_report_citations,
)
from local_eval_tools.report_preprocessor import (
    _describe_visual,
    _render_clip,
    _visual_candidates,
)


USER_FILE_EXTENSIONS = {
    ".pdf", ".md", ".txt", ".doc", ".docx", ".ppt", ".pptx",
    ".csv", ".tsv", ".xls", ".xlsx", ".json", ".jsonl",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
}

DR3_CONTROL_FILENAMES = {
    "useful_search.jsonl",
    "useful_search.json",
    "task.json",
    "task.md",
}

_GENERIC_FILE_RE = re.compile(
    r"([^\[\],]+\.(?:pdf|md|txt|docx?|pptx?|csv|tsv|xlsx?|jsonl?|"
    r"png|jpe?g|gif|webp|bmp|tiff?))",
    re.IGNORECASE,
)

_LATEX_OPTION_RE = re.compile(r"(\\(?:[A-Za-z@]+|\\))\s*\[[^\[\]\r\n]*\]")
_MARKDOWN_INLINE_LINK_RE = re.compile(
    r"!?\[([^\[\]\r\n]*)\]\((?:\\.|[^()\r\n]|\([^()\r\n]*\))*\)"
)


def _without_latex_optional_arguments(text: str) -> str:
    """Prevent TeX command options from being parsed as general [title] citations."""
    return _LATEX_OPTION_RE.sub(r"\1", text)


def _without_markdown_inline_links(text: str) -> str:
    """Keep link labels as prose while hiding Markdown brackets from citation parsing."""
    return _MARKDOWN_INLINE_LINK_RE.sub(r"\1", text)


def list_user_files(dataset_dir: Path) -> List[Path]:
    """Enumerate only the official files placed directly in a task directory."""
    return sorted(
        (
            path
            for path in dataset_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in USER_FILE_EXTENSIONS
                and path.name.casefold() not in DR3_CONTROL_FILENAMES
            )
        ),
        key=lambda path: path.name.casefold(),
    )


class LocalCitationCoverageEvaluator(CitationCoverageEvaluator):
    """Official DR3 citation scoring scoped to local user files only."""

    USER_DOC_EXTENSIONS = USER_FILE_EXTENSIONS

    def _extract_explicit_citations(self, content: str) -> List[str]:
        parser_input = _without_markdown_inline_links(content)
        parser_input = _without_latex_optional_arguments(parser_input)
        return super()._extract_explicit_citations(
            parser_input
        )

    def evaluate(
        self,
        result_text: str,
        dataset_dir: Optional[Path] = None,
        required_titles: Optional[List[str]] = None,
        **kwargs,
    ) -> EvalResult:
        if required_titles is not None:
            titles = list(required_titles)
        elif dataset_dir and dataset_dir.is_dir():
            titles = [path.name for path in list_user_files(dataset_dir)]
        else:
            return EvalResult(
                metric_name=self.metric_name,
                score=-1,
                details={"status": "failed", "error": "Missing user-file dataset directory"},
                weight=self.weight,
            )

        if not titles:
            return EvalResult(
                metric_name=self.metric_name,
                score=-1,
                details={"status": "failed", "error": "No official user files found"},
                weight=self.weight,
            )

        explicit_citations = [
            citation
            for citation in self._extract_explicit_citations(result_text)
            if not citation.startswith(("REPORT_PAGE:", "REPORT_VISUAL:"))
        ]
        explicitly_cited: List[str] = []
        for title in titles:
            for citation in explicit_citations:
                matched, _ = self._match_title_to_citation(title, citation)
                if matched:
                    explicitly_cited.append(title)
                    break

        citation_resolution = resolve_report_citations(result_text, titles)
        for title in citation_resolution.resolved_inline_user_files:
            if title not in explicitly_cited:
                explicitly_cited.append(title)

        result = super().evaluate(
            result_text=result_text,
            required_titles=list(titles),
        )
        result.details.update(
            {
                "status": "success" if result.score >= 0 else "failed",
                "scope": "official_user_files_only",
                "required_sources": titles,
                "scoring_mechanism": "official_dr3",
                "explicitly_cited": explicitly_cited,
                "citation_resolution": citation_resolution.to_dict(),
                "explicitly_missing": [
                    title for title in titles if title not in explicitly_cited
                ],
            }
        )
        return result


class LocalFactualAccuracyEvaluator(FactualAccuracyAgentEvaluator):
    """OpenRouter/user-file adapter around the official FA prompts and scoring."""

    def __init__(
        self,
        config: Optional[EvalConfig] = None,
        citation_resolution: Optional[CitationResolution] = None,
    ):
        config = config or EvalConfig()
        factual_accuracy_module.API_KEY = config.api_key
        factual_accuracy_module.API_BASE_URL = config.base_url
        factual_accuracy_module.MODEL_NAME = config.model_name
        factual_accuracy_module.GEMINI_MULTIMODAL_MODEL = config.model_name
        factual_accuracy_module.GPT5_MODEL = config.model_name
        super().__init__(config)
        self.local_config = config
        self.citation_resolution = citation_resolution
        self.source_visual_audit: List[Dict] = []
        self.api_client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=0,
            timeout=120.0,
        )
        self.gemini_client = self.api_client

    def evaluate(self, result_text: str, **kwargs) -> EvalResult:
        self.source_visual_audit = []
        effective_text = result_text
        if self.citation_resolution is not None:
            effective_text = self.citation_resolution.enrich_for_factual_accuracy(
                result_text
            )
        result = super().evaluate(result_text=effective_text, **kwargs)
        if self.citation_resolution is not None:
            result.details["citation_resolution"] = self.citation_resolution.to_dict()
        if self.source_visual_audit:
            result.details["source_visual_preprocessing"] = self.source_visual_audit
        return result

    def _get_source_key(self, citations: List[str], source_type: str) -> str:
        source_key = super()._get_source_key(citations, source_type)
        if not source_key.startswith("unknown:") or not citations:
            return source_key
        match = _GENERIC_FILE_RE.search(citations[0])
        if not match:
            return source_key
        filename = match.group(1).strip()
        suffix = Path(filename).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}:
            return f"image:{filename}"
        return f"doc:{filename}"

    def _verify_batch_with_api_thread(
        self,
        claims: List[Dict],
        source_info: Dict,
        client: OpenAI,
    ) -> List[Dict]:
        file_path = source_info.get("file_path")
        if source_info.get("type") == "file" and file_path and file_path.suffix.lower() == ".pdf":
            document = pymupdf.open(file_path)
            try:
                pages = [
                    f"--- Page {index + 1} ---\n{page.get_text('text', sort=True)}"
                    for index, page in enumerate(document)
                ]
                candidates = _visual_candidates(document)
                cited_pages = set(factual_accuracy_module.get_page_numbers_from_claims(claims))
                selected_pages = cited_pages or {index + 1 for index in candidates}
                vision_model = os.getenv("OPENROUTER_VISION_MODEL", "")
                for page_index, items in candidates.items():
                    page_number = page_index + 1
                    if page_number not in selected_pages:
                        continue
                    for visual_index, candidate in enumerate(items, start=1):
                        visual = _describe_visual(
                            _render_clip(document[page_index], candidate["bbox"]),
                            page_number,
                            self.local_config,
                            vision_model,
                        )
                        self.source_visual_audit.append(
                            {
                                "source": file_path.name,
                                "page": page_number,
                                "index": visual_index,
                                "substantive": visual["substantive"],
                                "reason": visual["reason"],
                                "model": visual["model"],
                            }
                        )
                        if visual["substantive"] and visual["markdown"]:
                            pages[page_index] += (
                                f"\n\n--- Visual {visual_index} on page {page_number} ---"
                                f"\n{visual['markdown']}"
                            )
            finally:
                document.close()
            source_info = {
                "found": True,
                "type": "text",
                "content": "\n\n".join(pages),
                "name": file_path.name,
            }
        return super()._verify_batch_with_api_thread(claims, source_info, client)


def explicit_citations(result_text: str) -> List[str]:
    """Expose the official citation parser for the FA zero-policy gate."""
    evaluator = LocalCitationCoverageEvaluator()
    return [
        citation
        for citation in evaluator._extract_explicit_citations(result_text)
        if not citation.startswith(("REPORT_PAGE:", "REPORT_VISUAL:"))
    ]


def all_explanations(details: Dict) -> Iterable[str]:
    for verification in (
        details.get("verification_result", {}).get("verifications", [])
    ):
        yield str(verification.get("explanation", ""))
        for result in verification.get("citation_results", []):
            yield str(result.get("explanation", ""))
