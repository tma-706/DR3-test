from pathlib import Path

import pymupdf

from evaluators.utils.base import EvalConfig, EvalResult
from local_eval_tools.adapters import (
    LocalCitationCoverageEvaluator,
    explicit_citations,
    list_user_files,
)
from local_eval_tools.citation_resolver import resolve_report_citations
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


def test_cc_keeps_official_dr3_full_text_fallback():
    evaluator = LocalCitationCoverageEvaluator()
    result = evaluator.evaluate(
        "The prose mentions alpha.pdf but has no citation marker.",
        required_titles=["alpha.pdf"],
    )
    assert result.score == 100
    assert result.details["status"] == "success"
    assert result.details["extracted_citations"] == []
    assert result.details["explicitly_cited"] == []
    assert result.details["match_details"]["alpha.pdf"]["match_type"] == "text_contains"
    assert result.details["scoring_mechanism"] == "official_dr3"


def test_cc_matches_only_official_filenames():
    evaluator = LocalCitationCoverageEvaluator()
    result = evaluator.evaluate(
        "Supported claim [Doc: alpha.pdf]. Irrelevant [Doc: outside.pdf].",
        required_titles=["alpha.pdf", "table.csv"],
    )
    assert result.score == 50
    assert result.details["cited"] == ["alpha.pdf"]
    assert result.details["missing"] == ["table.csv"]
    assert result.details["explicitly_cited"] == ["alpha.pdf"]
    assert result.details["scope"] == "official_user_files_only"


def test_user_file_scope_excludes_dr3_control_files(tmp_path: Path):
    (tmp_path / "source.txt").write_text("data", encoding="utf-8")
    (tmp_path / "useful_search.json").write_text("[]", encoding="utf-8")
    (tmp_path / "task.md").write_text("task", encoding="utf-8")

    assert [path.name for path in list_user_files(tmp_path)] == ["source.txt"]


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


def test_markdown_links_are_not_user_file_citations():
    evaluator = LocalCitationCoverageEvaluator()
    result = evaluator.evaluate(
        "Official website [hud.gov](https://www.hud.gov).",
        required_titles=["HUD-sec8-FY25.pdf"],
    )

    assert result.score == 0
    assert result.details["extracted_citations"] == []
    assert result.details["cited"] == []


def test_file_citation_still_matches_beside_markdown_link():
    evaluator = LocalCitationCoverageEvaluator()
    result = evaluator.evaluate(
        "Income limits use ACS data [HUD-sec8-FY25.pdf, Page 2]. "
        "Official website [hud.gov](https://www.hud.gov).",
        required_titles=["HUD-sec8-FY25.pdf"],
    )

    assert result.score == 100
    assert result.details["cited"] == ["HUD-sec8-FY25.pdf"]
    assert result.details["extracted_citations"] == [
        "HUD-sec8-FY25.pdf, Page 2"
    ]


def test_numeric_reference_resolves_exact_official_file():
    report = "Claim grounded in the notice [1].\n\nArbitrary source list\n[1]: alpha.pdf"
    resolution = resolve_report_citations(report, ["alpha.pdf"])

    assert resolution.resolved_inline_user_files == ["alpha.pdf"]
    assert resolution.reference_entries["1"].kind == "user_file"
    assert "notice [Doc: alpha.pdf]" in resolution.enrich_for_factual_accuracy(report)


def test_url_reference_is_not_confused_with_local_pdf():
    report = "External claim [1].\n\n[1]: https://example.com/files/alpha.pdf"
    resolution = resolve_report_citations(report, ["alpha.pdf"])

    assert resolution.resolved_inline_user_files == []
    assert resolution.reference_entries["1"].kind == "external_url"
    assert resolution.external_inline_urls == ["https://example.com/files/alpha.pdf"]


def test_mixed_reference_keeps_file_and_url_separate():
    report = "User-file claim [1].\n\n[1]: alpha.pdf; https://example.com/context"
    resolution = resolve_report_citations(report, ["alpha.pdf"])

    assert resolution.resolved_inline_user_files == ["alpha.pdf"]
    assert resolution.reference_entries["1"].kind == "mixed"
    assert resolution.external_inline_urls == ["https://example.com/context"]


def test_bibliography_file_is_not_explicit_until_label_is_used():
    report = "Uncited prose.\n\nWhatever this section is called\n[1]: alpha.pdf"
    coverage = LocalCitationCoverageEvaluator().evaluate(
        report,
        required_titles=["alpha.pdf"],
    )

    assert coverage.score == 100  # Preserve official DR3 full-text fallback.
    assert coverage.details["explicitly_cited"] == []


def test_numeric_groups_ranges_footnotes_and_latex_cite_resolve():
    report = (
        "Combined claims [1, 2] and [1-2]. Footnote claim [^3]. "
        r"LaTeX claim \cite{hud}." "\n\n"
        "References may have any heading\n"
        "[1]: alpha.pdf\n"
        "2. beta.csv\n"
        "[^3]: gamma.txt\n"
        r"\bibitem{hud} delta.docx"
    )
    resolution = resolve_report_citations(
        report,
        ["alpha.pdf", "beta.csv", "gamma.txt", "delta.docx"],
    )

    assert resolution.resolved_inline_user_files == [
        "alpha.pdf",
        "beta.csv",
        "gamma.txt",
        "delta.docx",
    ]


def test_pdf_normalized_text_uses_same_numeric_resolution():
    report = (
        "<!-- REPORT_PAGE: 1 -->\nGrounded PDF claim [1].\n\n"
        "Documents consulted\n[1]: alpha.pdf"
    )
    coverage = LocalCitationCoverageEvaluator().evaluate(
        report,
        required_titles=["alpha.pdf"],
    )

    assert coverage.details["explicitly_cited"] == ["alpha.pdf"]
    assert coverage.details["citation_resolution"][
        "resolved_inline_user_files"
    ] == ["alpha.pdf"]


def test_fa_receives_resolved_numeric_citation(tmp_path: Path, monkeypatch):
    (tmp_path / "alpha.pdf").write_bytes(b"official")
    report = "Grounded claim [1].\n\n[1]: alpha.pdf"
    coverage = LocalCitationCoverageEvaluator().evaluate(
        report,
        required_titles=["alpha.pdf"],
    )
    inputs = TaskInputs(
        task="012",
        query="Question",
        query_data={"task": "012", "query": "Question"},
        report_path=tmp_path / "final_report.md",
        dataset_dir=tmp_path,
        checklist_path=tmp_path / "checklist.json",
        insights_path=tmp_path / "gold.json",
    )
    observed = {}

    class FakeFA:
        def __init__(self, config, citation_resolution=None):
            observed["resolution"] = citation_resolution

        def evaluate(self, result_text, **kwargs):
            observed["text"] = observed["resolution"].enrich_for_factual_accuracy(
                result_text
            )
            return EvalResult("factual_accuracy", 100.0, {"ok": True})

    monkeypatch.setattr(runner, "LocalFactualAccuracyEvaluator", FakeFA)
    result = _factual_accuracy(report, inputs, EvalConfig(), coverage)

    assert result.score == 100
    assert "[Doc: alpha.pdf]" in observed["text"]


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


def test_fa_keeps_zero_for_filename_mention_without_explicit_citation(
    tmp_path: Path, monkeypatch
):
    coverage = LocalCitationCoverageEvaluator().evaluate(
        "The prose mentions source.pdf without a citation marker.",
        required_titles=["source.pdf"],
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
        raise AssertionError("FA must not run without an explicit user-file citation")

    monkeypatch.setattr(runner, "LocalFactualAccuracyEvaluator", fail_if_called)
    result = _factual_accuracy("source.pdf", inputs, EvalConfig(), coverage)

    assert coverage.score == 100
    assert coverage.details["explicitly_cited"] == []
    assert result.score == 0
    assert result.details["policy"] == "legitimate_zero"


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


def test_pdf_md_and_tex_reports_use_the_same_user_file_cc_flow(tmp_path: Path):
    report_text = "Supported finding [source.txt]."
    md_path = tmp_path / "final_report.md"
    tex_path = tmp_path / "final_report.tex"
    pdf_path = tmp_path / "final_report.pdf"
    md_path.write_text(report_text, encoding="utf-8")
    tex_path.write_text(report_text, encoding="utf-8")

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), report_text)
    document.save(pdf_path)
    document.close()

    config = EvalConfig(api_key="", base_url="", model_name="", temperature=0)
    evaluator = LocalCitationCoverageEvaluator()
    for report_path in (md_path, tex_path, pdf_path):
        preprocessed = preprocess_report(
            report_path,
            tmp_path / report_path.suffix.lstrip("."),
            config,
        )
        result = evaluator.evaluate(
            preprocessed.text,
            required_titles=["source.txt"],
        )
        assert result.score == 100
        assert result.details["explicitly_cited"] == ["source.txt"]


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
