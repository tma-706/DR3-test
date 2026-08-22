from pathlib import Path

from evaluators.utils.base import EvalConfig, EvalResult
from local_eval_tools.adapters import LocalCitationCoverageEvaluator, explicit_citations
from local_eval_tools.report_preprocessor import preprocess_report
from local_eval_tools import runner
from local_eval_tools.runner import (
    TaskInputs,
    _factual_accuracy,
    evaluate_task,
    normalize_task_id,
    result_payload,
)


def test_normalize_task_id():
    assert normalize_task_id("8") == "008"
    assert normalize_task_id(12) == "012"
    assert normalize_task_id("050") == "050"


def test_cc_requires_explicit_citation():
    evaluator = LocalCitationCoverageEvaluator()
    result = evaluator.evaluate(
        "The prose mentions alpha.pdf but has no citation marker.",
        required_titles=["alpha.pdf"],
    )
    assert result.score == 0
    assert result.details["status"] == "success"
    assert result.details["extracted_citations"] == []


def test_cc_matches_only_official_filenames():
    evaluator = LocalCitationCoverageEvaluator()
    result = evaluator.evaluate(
        "Supported claim [Doc: alpha.pdf]. Irrelevant [Doc: outside.pdf].",
        required_titles=["alpha.pdf", "table.csv"],
    )
    assert result.score == 50
    assert result.details["cited"] == ["alpha.pdf"]
    assert result.details["missing"] == ["table.csv"]
    assert "text_contains" in result.details["fallbacks_disabled"]


def test_internal_page_markers_are_not_citations():
    text = "<!-- REPORT_PAGE: 1 -->\n<!-- REPORT_VISUAL: page=2, index=1 -->"
    assert explicit_citations(text) == []


def test_latex_options_are_not_citations():
    text = (
        r"\documentclass[11pt,a4paper]{article} "
        r"\includegraphics[width=0.92\textwidth]{flowchart.png} "
        r"[HUD-sec8-FY25.pdf]"
    )
    assert explicit_citations(text) == ["HUD-sec8-FY25.pdf"]


def test_fa_skips_unmatched_bracket_candidates(tmp_path: Path, monkeypatch):
    coverage = EvalResult(
        "citation_coverage",
        0.0,
        {"extracted_citations": ["Diamond"], "cited": []},
    )
    inputs = TaskInputs(
        task="012",
        query="Question",
        query_data={"task": "012", "query": "Question"},
        report_path=tmp_path / "final_report.pdf",
        dataset_dir=tmp_path,
        checklist_path=tmp_path / "checklist.json",
        insights_path=tmp_path / "gold.json",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("FA evaluator must not run without a matched official citation")

    monkeypatch.setattr(runner, "LocalFactualAccuracyEvaluator", fail_if_called)
    result = _factual_accuracy("[Diamond]", inputs, EvalConfig(), coverage)

    assert result.score == 0.0
    assert result.details["matched_citations"] == []
    assert result.details["extracted_citations"] == ["Diamond"]


def test_markdown_preprocess_is_complete_and_cacheable(tmp_path: Path):
    source = tmp_path / "final_report.md"
    output = tmp_path / "out"
    report = "# Report\n\n" + ("complete content\n" * 6000)
    source.write_text(report, encoding="utf-8")
    config = EvalConfig(api_key="", base_url="", model_name="", temperature=0)

    first = preprocess_report(source, output, config)
    second = preprocess_report(source, output, config)

    assert first.text == report
    assert first.metadata["report_truncated"] is False
    assert first.reused is False
    assert second.reused is True
    assert second.text == report


def test_latex_preprocess_reads_source_text_without_rendering(tmp_path: Path):
    source = tmp_path / "final_report.tex"
    output = tmp_path / "out"
    report = r"\section{Finding} HUD uses 50\% of area median income."
    source.write_text(report, encoding="utf-8")
    config = EvalConfig(api_key="", base_url="", model_name="", temperature=0)

    result = preprocess_report(source, output, config)

    assert result.text == report
    assert result.metadata["source_format"] == "tex"
    assert result.metadata["pages"] is None
    assert result.metadata["vision_used"] == []


def test_zero_is_success_but_negative_score_is_error():
    zero = result_payload("008", "CC", EvalResult("citation_coverage", 0, {}))
    failed = result_payload(
        "008",
        "IR",
        EvalResult("information_recall", -1, {"error": "API failed"}),
    )
    assert zero["status"] == "success"
    assert zero["score"] == 0
    assert failed["status"] == "error"
    assert failed["score"] is None


def test_full_task_output_schema_without_network(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "datasets" / "008"
    ground_truth = tmp_path / "ground_truth" / "008"
    results = tmp_path / "results" / "008"
    output = tmp_path / "eval_result"
    dataset.mkdir(parents=True)
    ground_truth.mkdir(parents=True)
    results.mkdir(parents=True)
    (dataset / "source.txt").write_text("official data", encoding="utf-8")
    (ground_truth / "checklist.json").write_text(
        '{"checklist":[{"id":1,"requirement":"Answer clearly"}]}',
        encoding="utf-8",
    )
    (ground_truth / "gold_insights_from_source.json").write_text(
        '{"gold_insights":[{"insight":"official data","source":"source.txt"}]}',
        encoding="utf-8",
    )
    report = results / "final_report.md"
    report.write_text("# Answer\n\nNo citation marker.", encoding="utf-8")

    class FakeEvaluator:
        def __init__(self, metric_name, score):
            self.metric_name = metric_name
            self.score = score

        def evaluate(self, **kwargs):
            return EvalResult(self.metric_name, self.score, {"fake": True})

    monkeypatch.setattr(
        runner,
        "InformationRecallEvaluator",
        lambda config: FakeEvaluator("information_recall", 80),
    )
    monkeypatch.setattr(
        runner,
        "FormatComplianceEvaluator",
        lambda config: FakeEvaluator("format_compliance", 90),
    )
    monkeypatch.setattr(
        runner,
        "OverallQualityEvaluator",
        lambda config: FakeEvaluator("overall_quality", 70),
    )
    inputs = TaskInputs(
        task="008",
        query="Question",
        query_data={"task": "008", "query": "Question"},
        report_path=report,
        dataset_dir=dataset,
        checklist_path=ground_truth / "checklist.json",
        insights_path=ground_truth / "gold_insights_from_source.json",
    )
    summary = evaluate_task(
        inputs,
        output,
        EvalConfig(api_key="test", base_url="https://example.invalid/v1", model_name="test"),
        overwrite=True,
    )

    assert summary["status"] == "success"
    assert summary["scores"] == {"IR": 80.0, "CC": 0.0, "FA": 0.0, "IF": 90.0, "DQ": 70.0}
    task_output = output / "008" / "md"
    for name in (
        "report_for_eval.md",
        "preprocess_metadata.json",
        "eval_information_recall.json",
        "eval_citation_coverage.json",
        "eval_factual_accuracy.json",
        "eval_format_compliance.json",
        "eval_depth_quality.json",
        "scores.json",
        "summary.json",
    ):
        assert (task_output / name).is_file()
