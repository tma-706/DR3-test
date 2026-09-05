"""Resolve indirect report citations to exact official user-file names.

This module is intentionally local to the evaluation wrapper. It does not
change DR3's citation parser or prompts. It only supplies an equivalent direct
file citation to factual-accuracy evaluation when an indirect reference can be
resolved without guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_URL_RE = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
_BRACKET_ENTRY_RE = re.compile(
    r"^\s*\[(?P<label>\^?[A-Za-z0-9_.:-]+)\]\s*(?::|[-–—])?\s*(?P<body>.*)$"
)
_NUMBERED_ENTRY_RE = re.compile(r"^\s*(?P<label>\d+)\s*[.)]\s+(?P<body>.*)$")
_BIBITEM_ENTRY_RE = re.compile(
    r"^\s*\\bibitem(?:\[[^]]*\])?\{(?P<label>[^}]+)\}\s*(?P<body>.*)$"
)
_INLINE_BRACKET_RE = re.compile(r"\[(?P<content>[^]\n]+)\]")
_LATEX_CITE_RE = re.compile(
    r"\\cite[a-zA-Z*]*\s*(?:\[[^]]*\]\s*)*\{(?P<labels>[^}]+)\}"
)


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _normalize_label(label: str) -> str:
    return label.strip().lstrip("^")


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:!?)>]}\"")


@dataclass(frozen=True)
class ReferenceEntry:
    label: str
    raw_text: str
    kind: str
    user_files: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "raw_text": self.raw_text,
            "kind": self.kind,
            "user_files": list(self.user_files),
            "urls": list(self.urls),
        }


@dataclass(frozen=True)
class InlineCitation:
    marker: str
    labels: tuple[str, ...]
    start: int
    end: int
    user_files: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()
    unresolved_labels: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "marker": self.marker,
            "labels": list(self.labels),
            "user_files": list(self.user_files),
            "urls": list(self.urls),
            "unresolved_labels": list(self.unresolved_labels),
        }


@dataclass
class CitationResolution:
    reference_entries: dict[str, ReferenceEntry]
    inline_citations: list[InlineCitation]

    @property
    def resolved_inline_user_files(self) -> list[str]:
        return _unique(
            source
            for citation in self.inline_citations
            for source in citation.user_files
        )

    @property
    def external_inline_urls(self) -> list[str]:
        return _unique(
            url for citation in self.inline_citations for url in citation.urls
        )

    @property
    def unresolved_inline_labels(self) -> list[str]:
        return _unique(
            label
            for citation in self.inline_citations
            for label in citation.unresolved_labels
        )

    def enrich_for_factual_accuracy(self, report_text: str) -> str:
        """Replace resolved markers with DR3-compatible direct citations."""
        enriched = report_text
        for citation in reversed(self.inline_citations):
            if not citation.user_files:
                continue
            direct = " ".join(f"[Doc: {name}]" for name in citation.user_files)
            enriched = enriched[: citation.start] + direct + enriched[citation.end :]
        return enriched

    def to_dict(self) -> dict:
        return {
            "reference_entries": {
                label: entry.to_dict()
                for label, entry in self.reference_entries.items()
            },
            "inline_citations": [item.to_dict() for item in self.inline_citations],
            "resolved_inline_user_files": self.resolved_inline_user_files,
            "external_inline_urls": self.external_inline_urls,
            "unresolved_inline_labels": self.unresolved_inline_labels,
        }


def _entry_start(line: str) -> tuple[str, str, tuple[int, int]] | None:
    for pattern in (_BIBITEM_ENTRY_RE, _BRACKET_ENTRY_RE, _NUMBERED_ENTRY_RE):
        match = pattern.match(line)
        if not match:
            continue
        label = _normalize_label(match.group("label"))
        body = match.group("body")
        if pattern is _BRACKET_ENTRY_RE:
            label_span = (match.start("label") - 1, match.end("label") + 1)
        else:
            label_span = match.span()
        return label, body, label_span
    return None


def _labels_from_bracket(content: str, known_labels: set[str]) -> list[str]:
    value = content.strip().lstrip("^")
    labels: list[str] = []
    for part in re.split(r"\s*[,;]\s*", value):
        normalized = _normalize_label(part)
        if normalized in known_labels:
            labels.append(normalized)
            continue
        range_match = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", normalized)
        if not range_match:
            continue
        start, end = (int(item) for item in range_match.groups())
        if start <= end and end - start <= 100:
            labels.extend(
                str(number)
                for number in range(start, end + 1)
                if str(number) in known_labels
            )
    return _unique(labels)


def _classify_entry(
    body: str, official_names: Sequence[str]
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    urls = tuple(_unique(_clean_url(match.group()) for match in _URL_RE.finditer(body)))
    text_without_urls = _URL_RE.sub(" ", body)
    user_files = tuple(
        name
        for name in official_names
        if re.search(
            rf"(?<![\w.-]){re.escape(name)}(?![\w.-])",
            text_without_urls,
            re.IGNORECASE,
        )
    )
    if user_files and urls:
        kind = "mixed"
    elif user_files:
        kind = "user_file"
    elif urls:
        kind = "external_url"
    else:
        kind = "unresolved"
    return kind, user_files, urls


def resolve_report_citations(
    report_text: str, official_user_files: Sequence[str | Path]
) -> CitationResolution:
    """Resolve citations without relying on a particular section heading.

    Only an exact official basename outside a URL is accepted as a user-file
    mapping. Title-only references remain unresolved by design.
    """
    official_names = _unique(Path(value).name for value in official_user_files)
    lines = report_text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    candidates: list[tuple[int, int, str, str, tuple[int, int]]] = []
    for index, line in enumerate(lines):
        offsets.append(offset)
        entry = _entry_start(line.rstrip("\r\n"))
        if entry:
            candidates.append((index, offset, *entry))
        offset += len(line)

    entries: dict[str, ReferenceEntry] = {}
    definition_spans: list[tuple[int, int]] = []
    candidate_lines = {candidate[0] for candidate in candidates}
    for index, absolute_offset, label, first_body, label_span in candidates:
        body_parts = [first_body]
        for next_index in range(index + 1, min(len(lines), index + 8)):
            if next_index in candidate_lines or not lines[next_index].strip():
                break
            body_parts.append(lines[next_index].strip())
        body = " ".join(part for part in body_parts if part).strip()
        kind, user_files, urls = _classify_entry(body, official_names)
        if kind == "unresolved":
            continue
        entries[label] = ReferenceEntry(label, body, kind, user_files, urls)
        definition_spans.append(
            (absolute_offset + label_span[0], absolute_offset + label_span[1])
        )

    def is_definition(start: int, end: int) -> bool:
        return any(
            start < span_end and end > span_start
            for span_start, span_end in definition_spans
        )

    inline: list[InlineCitation] = []
    known_labels = set(entries)
    for match in _INLINE_BRACKET_RE.finditer(report_text):
        if is_definition(match.start(), match.end()):
            continue
        labels = _labels_from_bracket(match.group("content"), known_labels)
        if labels:
            inline.append(
                _resolve_inline(match.group(), labels, match.start(), match.end(), entries)
            )

    for match in _LATEX_CITE_RE.finditer(report_text):
        labels = [
            _normalize_label(item)
            for item in match.group("labels").split(",")
            if item.strip()
        ]
        inline.append(
            _resolve_inline(match.group(), labels, match.start(), match.end(), entries)
        )

    inline.sort(key=lambda item: item.start)
    return CitationResolution(entries, inline)


def _resolve_inline(
    marker: str,
    labels: Sequence[str],
    start: int,
    end: int,
    entries: dict[str, ReferenceEntry],
) -> InlineCitation:
    user_files: list[str] = []
    urls: list[str] = []
    unresolved: list[str] = []
    for label in labels:
        entry = entries.get(label)
        if entry is None:
            unresolved.append(label)
            continue
        user_files.extend(entry.user_files)
        urls.extend(entry.urls)
    return InlineCitation(
        marker=marker,
        labels=tuple(labels),
        start=start,
        end=end,
        user_files=tuple(_unique(user_files)),
        urls=tuple(_unique(urls)),
        unresolved_labels=tuple(_unique(unresolved)),
    )
