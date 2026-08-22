"""CLI orchestration for standalone, user-files-only DR3 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from openai import AuthenticationError, PermissionDeniedError
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from evaluators.depth_quality import OverallQualityEvaluator
from evaluators.format_compliance import FormatComplianceEvaluator
from evaluators.information_recall import InformationRecallEvaluator
from evaluators.utils.base import EvalConfig, EvalResult
from local_eval_tools.adapters import (
    LocalCitationCoverageEvaluator,
    LocalFactualAccuracyEvaluator,
    all_explanations,
    explicit_citations,
    list_user_files,
)
from local_eval_tools.report_preprocessor import (
    SUPPORTED_REPORT_EXTENSIONS,
    inspect_report,
    preprocess_report,
    sha256_file,
)


LOGGER = logging.getLogger(__name__)
CONSOLE = Console()
METRIC_FILES = {
    "IR": "eval_information_recall.json",
    "CC": "eval_citation_coverage.json",
    "FA": "eval_factual_accuracy.json",
    "IF": "eval_format_compliance.json",
    "DQ": "eval_depth_quality.json",
}
TECHNICAL_MARKERS = (
    "api error",
    "authentication",
    "authorization",
    "connection error",
    "timeout",
    "parse error",
    "empty response",
    "max retries",
    "not initialized",
)


class LocalEvalError(RuntimeError):
    """A setup/data error that should not become a score of zero."""


@dataclass(frozen=True)
class TaskInputs:
    task: str
    query: str
    query_data: Dict[str, Any]
    report_path: Path
    dataset_dir: Path
    checklist_path: Path
    insights_path: Path


def normalize_task_id(value: Any) -> str:
    text = str(value).strip()
    if not text.isdigit():
        raise LocalEvalError(f"Task ID must be numeric, got {value!r}")
    return text.zfill(3)


def load_queries(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        raise LocalEvalError(f"Query file not found: {path}")
    queries: Dict[str, Dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            task = normalize_task_id(item.get("task", item.get("number")))
        except Exception as exc:
            raise LocalEvalError(f"Invalid query.jsonl line {line_number}: {exc}") from exc
        if not str(item.get("query", "")).strip():
            raise LocalEvalError(f"Task {task} has an empty query")
        queries[task] = item
    return queries


def find_task_dir(root: Path, task: str) -> Path:
    direct = root / task
    if direct.is_dir():
        return direct
    if root.is_dir():
        for candidate in root.iterdir():
            if (
                candidate.is_dir()
                and candidate.name.isdigit()
                and normalize_task_id(candidate.name) == task
            ):
                return candidate
    return direct


def find_report(results_root: Path, task: str) -> Path:
    task_dir = find_task_dir(results_root, task)
    if not task_dir.is_dir():
        raise LocalEvalError(f"Report directory not found: {task_dir}")
    candidates = sorted(
        path for path in task_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_REPORT_EXTENSIONS
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise LocalEvalError(f"No supported report found in {task_dir}")
    raise LocalEvalError(
        f"Ambiguous reports for task {task}: {', '.join(path.name for path in candidates)}"
    )


def discover_tasks(results_root: Path) -> List[str]:
    tasks = []
    if not results_root.is_dir():
        return tasks
    for path in results_root.iterdir():
        if path.is_dir() and path.name.isdigit():
            task = normalize_task_id(path.name)
            try:
                find_report(results_root, task)
            except LocalEvalError:
                continue
            tasks.append(task)
    return sorted(set(tasks))


def resolve_inputs(
    task: str,
    datasets_root: Path,
    ground_truth_root: Path,
    results_root: Path,
    queries: Dict[str, Dict[str, Any]],
    report_path: Optional[Path] = None,
) -> TaskInputs:
    task = normalize_task_id(task)
    query_data = queries.get(task)
    if query_data is None:
        raise LocalEvalError(f"Query not found for task {task}")
    dataset_dir = find_task_dir(datasets_root, task)
    ground_truth_dir = find_task_dir(ground_truth_root, task)
    checklist_path = ground_truth_dir / "checklist.json"
    insights_path = ground_truth_dir / "gold_insights_from_source.json"
    if not dataset_dir.is_dir():
        raise LocalEvalError(f"Dataset directory not found: {dataset_dir}")
    if not list_user_files(dataset_dir):
        raise LocalEvalError(f"No official user files found in {dataset_dir}")
    for required in (checklist_path, insights_path):
        if not required.is_file():
            raise LocalEvalError(f"Ground-truth file not found: {required}")
    return TaskInputs(
        task=task,
        query=str(query_data["query"]),
        query_data=query_data,
        report_path=report_path if report_path is not None else find_report(results_root, task),
        dataset_dir=dataset_dir,
        checklist_path=checklist_path,
        insights_path=insights_path,
    )


def read_gold_insights(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    insights = data.get("gold_insights")
    if not isinstance(insights, list) or not insights:
        raise LocalEvalError(f"No gold_insights in {path}")
    return insights


def result_payload(task: str, code: str, result: EvalResult) -> Dict[str, Any]:
    if result.score < 0:
        return {
            "task": task,
            "metric": code,
            "metric_name": result.metric_name,
            "status": "error",
            "score": None,
            "error": result.details.get("error", "Evaluator returned an error score"),
            "details": result.details,
        }
    return {
        "task": task,
        "metric": code,
        "metric_name": result.metric_name,
        "status": "success",
        "score": float(result.score),
        "details": result.details,
    }


def error_payload(task: str, code: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "task": task,
        "metric": code,
        "metric_name": METRIC_FILES[code][5:-5],
        "status": "error",
        "score": None,
        "error": str(exc),
        "error_type": type(exc).__name__,
    }


def _cached_metric(path: Path, signature: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") == "success" and payload.get("input_signature") == signature:
        return payload
    return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _input_signature(
    code: str,
    report_text: str,
    inputs: TaskInputs,
    config: EvalConfig,
) -> str:
    source_hashes = [
        {"name": path.name, "sha256": sha256_file(path)}
        for path in list_user_files(inputs.dataset_dir)
    ]
    material = {
        "metric": code,
        "report_sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
        "query": inputs.query,
        "checklist_sha256": sha256_file(inputs.checklist_path),
        "insights_sha256": sha256_file(inputs.insights_path),
        "source_files": source_hashes,
        "model": config.model_name,
        "temperature": config.temperature,
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _factual_accuracy(
    report_text: str,
    inputs: TaskInputs,
    config: EvalConfig,
    citation_coverage: EvalResult,
) -> EvalResult:
    matched_citations = citation_coverage.details.get("cited", [])
    if not matched_citations:
        return EvalResult(
            metric_name="factual_accuracy",
            score=0.0,
            details={
                "status": "success",
                "policy": "legitimate_zero",
                "reason": "No explicit citation matched an official user file",
                "extracted_citations": citation_coverage.details.get(
                    "extracted_citations", []
                ),
                "matched_citations": [],
                "claim_count": 0,
            },
        )
    result = LocalFactualAccuracyEvaluator(config).evaluate(
        result_text=report_text,
        source_folder=inputs.dataset_dir,
        case_id=inputs.task,
    )
    explanations = [text.casefold() for text in all_explanations(result.details) if text]
    if explanations and all(
        any(marker in explanation for marker in TECHNICAL_MARKERS)
        for explanation in explanations
    ):
        raise LocalEvalError("All factual-accuracy verifications failed technically")
    return result


def report_output_dir(output_root: Path, inputs: TaskInputs) -> Path:
    """Keep evaluations for different report formats separate within a task."""
    report_format = inputs.report_path.suffix.lower().lstrip(".")
    if not report_format:
        raise LocalEvalError(f"Report has no file extension: {inputs.report_path}")
    return output_root / inputs.task / report_format


def evaluate_task(
    inputs: TaskInputs,
    output_root: Path,
    config: EvalConfig,
    overwrite: bool,
) -> Dict[str, Any]:
    task_output = report_output_dir(output_root, inputs)
    task_output.mkdir(parents=True, exist_ok=True)
    preprocessed = preprocess_report(
        report_path=inputs.report_path,
        output_dir=task_output,
        config=config,
        overwrite=overwrite,
    )
    report_text = preprocessed.text
    gold_insights = read_gold_insights(inputs.insights_path)
    citation_coverage = LocalCitationCoverageEvaluator(config).evaluate(
        result_text=report_text,
        dataset_dir=inputs.dataset_dir,
    )
    report_has_matched_citations = bool(citation_coverage.details.get("cited", []))

    factories: Dict[str, Callable[[], EvalResult]] = {
        "IR": lambda: InformationRecallEvaluator(config).evaluate(
            result_text=report_text,
            source_gold_insights=gold_insights,
            query=inputs.query,
            auto_extract=False,
            evaluate_only="source_documents",
        ),
        "CC": lambda: citation_coverage,
        "FA": lambda: _factual_accuracy(
            report_text, inputs, config, citation_coverage
        ),
        "IF": lambda: FormatComplianceEvaluator(config).evaluate(
            result_text=report_text,
            checklist_path=inputs.checklist_path,
            query=inputs.query,
            auto_generate=False,
        ),
        "DQ": lambda: OverallQualityEvaluator(config).evaluate(
            result_text=report_text,
            query_data=inputs.query_data,
        ),
    }

    metrics: Dict[str, Dict[str, Any]] = {}
    fatal_auth_error: Optional[str] = None
    for code, factory in factories.items():
        path = task_output / METRIC_FILES[code]
        signature = _input_signature(code, report_text, inputs, config)
        cached = None if overwrite else _cached_metric(path, signature)
        if cached is not None:
            cached["cached"] = True
            metrics[code] = cached
            continue
        requires_api = code in {"IR", "IF", "DQ"} or (
            code == "FA" and report_has_matched_citations
        )
        if fatal_auth_error and requires_api:
            payload = error_payload(inputs.task, code, LocalEvalError(fatal_auth_error))
            payload["input_signature"] = signature
            payload["cached"] = False
            _write_json(path, payload)
            metrics[code] = payload
            continue
        try:
            payload = result_payload(inputs.task, code, factory())
        except (AuthenticationError, PermissionDeniedError) as exc:
            fatal_auth_error = f"OpenRouter authorization failed; later API metrics were not retried: {exc}"
            payload = error_payload(inputs.task, code, exc)
        except Exception as exc:
            LOGGER.exception("Task %s metric %s failed", inputs.task, code)
            payload = error_payload(inputs.task, code, exc)
        payload["input_signature"] = signature
        payload["cached"] = False
        _write_json(path, payload)
        metrics[code] = payload

    scores = {
        "task": inputs.task,
        "scores": {code: item.get("score") for code, item in metrics.items()},
        "status": {code: item["status"] for code, item in metrics.items()},
    }
    _write_json(task_output / "scores.json", scores)
    errors = {
        code: item.get("error", "Unknown error")
        for code, item in metrics.items()
        if item["status"] == "error"
    }
    summary = {
        "task": inputs.task,
        "status": "success" if not errors else ("partial_error" if len(errors) < 5 else "error"),
        "query": inputs.query,
        "report": str(inputs.report_path),
        "report_format": inputs.report_path.suffix.lower().lstrip("."),
        "dataset_dir": str(inputs.dataset_dir),
        "official_user_files": [path.name for path in list_user_files(inputs.dataset_dir)],
        "preprocess_reused": preprocessed.reused,
        "scores": scores["scores"],
        "metric_status": scores["status"],
        "errors": errors,
    }
    _write_json(task_output / "summary.json", summary)
    return summary


def dry_run(
    tasks: Sequence[str],
    datasets_root: Path,
    ground_truth_root: Path,
    results_root: Path,
    queries: Dict[str, Dict[str, Any]],
    report_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    summaries = []
    for task in tasks:
        try:
            inputs = resolve_inputs(
                task, datasets_root, ground_truth_root, results_root, queries, report_path
            )
            inspection = inspect_report(inputs.report_path)
            summaries.append(
                {
                    "task": task,
                    "status": "ready",
                    "report": str(inputs.report_path),
                    "files": len(list_user_files(inputs.dataset_dir)),
                    **inspection,
                }
            )
        except Exception as exc:
            summaries.append({"task": task, "status": "error", "error": str(exc)})
    return summaries


def print_table(rows: Sequence[Dict[str, Any]], dry: bool = False) -> None:
    table = Table(title="DR3 Local Evaluation" + (" — dry run" if dry else ""))
    table.add_column("Task")
    table.add_column("Status")
    if dry:
        table.add_column("Report")
        table.add_column("Pages")
        table.add_column("Chars")
        table.add_column("Vision pages")
        for row in rows:
            visuals = ",".join(str(item["page"]) for item in row.get("visual_candidates", [])) or "-"
            table.add_row(
                row["task"], row["status"], Path(row.get("report", "-")).name,
                str(row.get("pages", "-")), str(row.get("characters", "-")), visuals,
            )
    else:
        for code in METRIC_FILES:
            table.add_column(code, justify="right")
        for row in rows:
            values = []
            for code in METRIC_FILES:
                score = row.get("scores", {}).get(code)
                values.append("ERR" if score is None else f"{score:.1f}")
            table.add_row(row["task"], row["status"], *values)
    CONSOLE.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DR3 metrics on local user files only")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--task", action="append", help="Task ID; repeat for multiple tasks")
    choice.add_argument("--all", action="store_true", help="Evaluate every task that has a report")
    parser.add_argument("--workers", type=int, default=1, help="Parallel task workers (default: 1)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute successful cached outputs")
    parser.add_argument("--dry-run", action="store_true", help="Validate and inspect without API calls or writes")
    parser.add_argument("--datasets-root", type=Path, default=PROJECT_ROOT / "local_eval" / "datasets")
    parser.add_argument("--ground-truth-root", type=Path, default=PROJECT_ROOT / "local_eval" / "ground_truth")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "local_eval" / "results")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "eval_result")
    parser.add_argument(
        "--report",
        type=Path,
        help="Explicit report path; valid only with exactly one --task",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.report and (args.all or len(args.task or []) != 1):
        raise SystemExit("--report requires exactly one --task")
    report_path = args.report.resolve() if args.report else None
    if report_path and (
        not report_path.is_file()
        or report_path.suffix.lower() not in SUPPORTED_REPORT_EXTENSIONS
    ):
        raise SystemExit(f"Unsupported or missing report: {report_path}")
    queries = load_queries(args.datasets_root / "query.jsonl")
    tasks = discover_tasks(args.results_root) if args.all else [
        normalize_task_id(task) for task in args.task
    ]
    if not tasks:
        CONSOLE.print("[red]No report tasks found.[/red]")
        return 1

    if args.dry_run:
        rows = dry_run(
            tasks,
            args.datasets_root,
            args.ground_truth_root,
            args.results_root,
            queries,
            report_path,
        )
        print_table(rows, dry=True)
        return 1 if any(row["status"] == "error" for row in rows) else 0

    config = EvalConfig()
    if not config.api_key or not config.base_url or not config.model_name:
        CONSOLE.print("[red]Missing OPENROUTER_API_KEY, OPENROUTER_BASE_URL, or OPENROUTER_MODEL.[/red]")
        return 1

    resolved: List[TaskInputs] = []
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        try:
            resolved.append(
                resolve_inputs(
                    task,
                    args.datasets_root,
                    args.ground_truth_root,
                    args.results_root,
                    queries,
                    report_path,
                )
            )
        except Exception as exc:
            rows.append({"task": task, "status": "error", "scores": {}, "error": str(exc)})

    def run_one(item: TaskInputs) -> Dict[str, Any]:
        try:
            return evaluate_task(item, args.output_root, config, args.overwrite)
        except Exception as exc:
            LOGGER.error("Task %s failed: %s", item.task, traceback.format_exc())
            task_output = report_output_dir(args.output_root, item)
            task_output.mkdir(parents=True, exist_ok=True)
            summary = {
                "task": item.task,
                "status": "error",
                "scores": {},
                "errors": {"task": str(exc)},
            }
            _write_json(task_output / "summary.json", summary)
            return summary

    if args.workers == 1:
        rows.extend(run_one(item) for item in resolved)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, item): item.task for item in resolved}
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: row["task"])
    print_table(rows)
    return 1 if any(row["status"] != "success" for row in rows) else 0
