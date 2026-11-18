"""Unit and integration tests for Internal Recall@k Evaluator"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

import pytest

from src.indexing.schema import InternalChunk
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)
from src.utils.evaluation import (
    CODE_RECALL_AT_5_TARGET,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_K,
    DEFAULT_K_VALUES,
    DEFAULT_MAX_CONTEXT_LENGTH,
    DOCS_RECALL_AT_5_TARGET,
    RecallEvaluator,
    classify_question_category,
    compute_iou,
    compute_question_recall_at_k,
    compute_span_iou,
    evaluate_search_results,
    is_source_match,
    recall_at_k,
    validate_student_search_results,
)


class TestConstants:
    """Verify moulinette threshold constants."""

    def test_constants_match_specification(self) -> None:
        assert DEFAULT_IOU_THRESHOLD == pytest.approx(0.05)
        assert DEFAULT_MAX_CONTEXT_LENGTH == 2000
        assert DEFAULT_K == 5
        assert DEFAULT_K_VALUES == (1, 3, 5, 10)
        assert DOCS_RECALL_AT_5_TARGET == pytest.approx(0.80)
        assert CODE_RECALL_AT_5_TARGET == pytest.approx(0.50)


class TestCharacterSpanIoU:
    """Verify exact IoU = max(0, min(E1, E2) - max(S1, S2)) / (max(E1, E2) - min(S1, S2))."""

    def test_identical_spans_yield_one(self) -> None:
        assert compute_span_iou(100, 500, 100, 500) == pytest.approx(1.0)
        assert compute_iou((100, 500), (100, 500)) == pytest.approx(1.0)
        assert compute_iou(100, 500, 100, 500) == pytest.approx(1.0)

    def test_partial_overlap_matches_exact_formula(self) -> None:
        # S1=100, E1=200, S2=150, E2=250
        # Numerator: max(0, min(200, 250) - max(100, 150)) = 200 - 150 = 50
        # Denominator: max(200, 250) - min(100, 150) = 250 - 100 = 150
        expected = 50.0 / 150.0
        assert compute_span_iou(100, 200, 150, 250) == pytest.approx(expected)
        assert compute_span_iou(150, 250, 100, 200) == pytest.approx(expected)

    def test_disjoint_and_adjacent_spans_yield_zero(self) -> None:
        # Disjoint with gap: [0, 100] and [200, 300]
        assert compute_span_iou(0, 100, 200, 300) == pytest.approx(0.0)
        # Touching boundary: [100, 200] and [200, 300] -> min(E1,E2)-max(S1,S2) = 0
        assert compute_span_iou(100, 200, 200, 300) == pytest.approx(0.0)
        # Degenerate zero-width span
        assert compute_span_iou(100, 100, 100, 200) == pytest.approx(0.0)

    def test_contained_subspan_iou(self) -> None:
        # [200, 300] inside [100, 500] -> intersection = 100, union_span = 400 -> 0.25
        assert compute_span_iou(200, 300, 100, 500) == pytest.approx(0.25)

    def test_exact_005_threshold_boundary(self) -> None:
        # [0, 50] and [0, 1000] -> intersection = 50, denominator = 1000 -> IoU = 0.05
        assert compute_span_iou(0, 50, 0, 1000) == pytest.approx(0.05)
        # [0, 49] and [0, 1000] -> IoU = 0.049 < 0.05
        assert compute_span_iou(0, 49, 0, 1000) == pytest.approx(0.049)

    def test_compute_iou_on_minimal_source_and_internal_chunk(self) -> None:
        src1 = MinimalSource(
            file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
            first_character_index=100,
            last_character_index=300,
        )
        chunk2 = InternalChunk(
            chunk_id=1,
            file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
            first_character_index=200,
            last_character_index=400,
            chunk_type="markdown_section",
            token_count=20,
            content="sample",
        )
        # Intersection: [200, 300] = 100; Denominator: 400 - 100 = 300 -> 1/3
        assert compute_iou(src1, chunk2) == pytest.approx(100.0 / 300.0)


class TestSourceMatchingAndRecallAtK:
    """Verify verbatim file_path matching and IoU >= 0.05 threshold."""

    def test_matches_when_file_path_verbatim_and_iou_at_least_005(self) -> None:
        gt = MinimalSource(
            file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
            first_character_index=0,
            last_character_index=1000,
        )
        cand_exact_005 = MinimalSource(
            file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
            first_character_index=0,
            last_character_index=50,  # IoU = 50 / 1000 = 0.05
        )
        cand_below_005 = MinimalSource(
            file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
            first_character_index=0,
            last_character_index=49,  # IoU = 49 / 1000 = 0.049
        )

        assert is_source_match(cand_exact_005, gt) is True
        assert is_source_match(cand_below_005, gt) is False

    def test_rejects_different_file_path_even_with_identical_span(self) -> None:
        gt = MinimalSource(
            file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
            first_character_index=100,
            last_character_index=500,
        )
        wrong_file = MinimalSource(
            file_path="data/raw/vllm-0.10.1/docs/features/quantization.md",
            first_character_index=100,
            last_character_index=500,
        )
        assert is_source_match(wrong_file, gt) is False

        # Raw dict with unnormalized "./" prefix fails verbatim comparison
        unnormalized_dict = {
            "file_path": "./data/raw/vllm-0.10.1/docs/features/lora.md",
            "first_character_index": 100,
            "last_character_index": 500,
        }
        assert is_source_match(unnormalized_dict, gt) is False

    def test_question_recall_at_k_single_and_multi_source(self) -> None:
        gt1 = MinimalSource(
            file_path="data/raw/vllm-0.10.1/vllm/entrypoints/llm.py",
            first_character_index=1000,
            last_character_index=1500,
        )
        gt2 = MinimalSource(
            file_path="data/raw/vllm-0.10.1/vllm/engine/llm_engine.py",
            first_character_index=2000,
            last_character_index=2500,
        )
        miss = MinimalSource(
            file_path="data/raw/vllm-0.10.1/vllm/config.py",
            first_character_index=0,
            last_character_index=300,
        )
        hit1 = MinimalSource(
            file_path="data/raw/vllm-0.10.1/vllm/entrypoints/llm.py",
            first_character_index=1100,
            last_character_index=1400,
        )
        hit2 = MinimalSource(
            file_path="data/raw/vllm-0.10.1/vllm/engine/llm_engine.py",
            first_character_index=2100,
            last_character_index=2400,
        )

        retrieved = [miss, hit1, miss, miss, hit2]
        # At k=1: only [miss] -> 0/2 = 0.0
        assert compute_question_recall_at_k(retrieved, [gt1, gt2], k=1) == pytest.approx(0.0)
        # At k=2: [miss, hit1] -> 1/2 = 0.5
        assert compute_question_recall_at_k(retrieved, [gt1, gt2], k=2) == pytest.approx(0.5)
        # At k=5: [miss, hit1, miss, miss, hit2] -> 2/2 = 1.0
        assert compute_question_recall_at_k(retrieved, [gt1, gt2], k=5) == pytest.approx(1.0)

    def test_validates_student_results_max_context_length_2000(self) -> None:
        valid_res = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="q1",
                    question="Valid question?",
                    retrieved_sources=[
                        MinimalSource(
                            file_path="data/raw/vllm-0.10.1/docs/a.md",
                            first_character_index=0,
                            last_character_index=2000,
                        )
                    ],
                )
            ],
            k=5,
        )
        ok, errs = validate_student_search_results(valid_res)
        assert ok is True
        assert errs == []

        invalid_res = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="q1",
                    question="Invalid oversized chunk?",
                    retrieved_sources=[
                        MinimalSource(
                            file_path="data/raw/vllm-0.10.1/docs/a.md",
                            first_character_index=0,
                            last_character_index=2001,  # 2001 > 2000 limit
                        )
                    ],
                )
            ],
            k=5,
        )
        ok_bad, errs_bad = validate_student_search_results(invalid_res)
        assert ok_bad is False
        assert len(errs_bad) == 1
        assert "exceeds max_context_length" in errs_bad[0]


class TestDetailedSummaryTableAndDatasetEvaluation:
    """Verify dataset evaluation and Recall@5 summary table for Docs and Code."""

    def test_evaluates_docs_and_code_and_formats_summary_table(
        self, tmp_path: Path
    ) -> None:
        # 5 Docs questions (4/5 hits -> 80% Recall@5 == PASS)
        docs_gt_questions: List[Union[AnsweredQuestion, UnansweredQuestion]] = []
        docs_student_items: List[MinimalSearchResults] = []
        for i in range(5):
            gt_src = MinimalSource(
                file_path=f"data/raw/vllm-0.10.1/docs/feature_{i}.md",
                first_character_index=100,
                last_character_index=500,
            )
            docs_gt_questions.append(
                AnsweredQuestion(
                    question_id=f"doc-q-{i}",
                    question=f"How does doc feature {i} work?",
                    sources=[gt_src],
                    answer=f"Answer {i}",
                )
            )
            cand_src = (
                gt_src
                if i < 4
                else MinimalSource(
                    file_path="data/raw/vllm-0.10.1/docs/unrelated.md",
                    first_character_index=0,
                    last_character_index=50,
                )
            )
            docs_student_items.append(
                MinimalSearchResults(
                    question_id=f"doc-q-{i}",
                    question=f"How does doc feature {i} work?",
                    retrieved_sources=[cand_src],
                )
            )

        # 4 Code questions (3/4 hits -> 75% Recall@5 >= 50% == PASS)
        code_gt_questions: List[Union[AnsweredQuestion, UnansweredQuestion]] = []
        code_student_items: List[MinimalSearchResults] = []
        for i in range(4):
            gt_src = MinimalSource(
                file_path=f"data/raw/vllm-0.10.1/vllm/module_{i}.py",
                first_character_index=200,
                last_character_index=800,
            )
            code_gt_questions.append(
                AnsweredQuestion(
                    question_id=f"code-q-{i}",
                    question=f"Where is module {i} implemented?",
                    sources=[gt_src],
                    answer=f"Code answer {i}",
                )
            )
            cand_src = (
                gt_src
                if i < 3
                else MinimalSource(
                    file_path="data/raw/vllm-0.10.1/vllm/other.py",
                    first_character_index=0,
                    last_character_index=50,
                )
            )
            code_student_items.append(
                MinimalSearchResults(
                    question_id=f"code-q-{i}",
                    question=f"Where is module {i} implemented?",
                    retrieved_sources=[cand_src],
                )
            )

        evaluator = RecallEvaluator()
        combined_report = evaluator.evaluate_docs_and_code(
            docs_student_results=StudentSearchResults(
                search_results=docs_student_items, k=5
            ),
            docs_dataset=RagDataset(rag_questions=docs_gt_questions),
            code_student_results=StudentSearchResults(
                search_results=code_student_items, k=5
            ),
            code_dataset=RagDataset(rag_questions=code_gt_questions),
            k=5,
        )

        assert combined_report.is_valid is True
        assert combined_report.num_docs_questions == 5
        assert combined_report.num_code_questions == 4
        assert combined_report.num_questions == 9
        assert combined_report.docs_recall_at_5 == pytest.approx(0.80)
        assert combined_report.code_recall_at_5 == pytest.approx(0.75)
        assert combined_report.docs_passed is True
        assert combined_report.code_passed is True

        # Dict-style access on EvaluationResult
        assert combined_report["Recall@5"] == pytest.approx(7.0 / 9.0)
        assert combined_report["docs_recall@5"] == pytest.approx(0.80)
        assert combined_report["code_recall@5"] == pytest.approx(0.75)
        assert combined_report[5] == pytest.approx(7.0 / 9.0)

        table = str(combined_report)
        assert "Student data is valid: True" in table
        assert "Evaluation Results" in table
        assert "Recall@5:" in table
        assert "Docs (Documentation)" in table
        assert "Code (Python Source)" in table
        assert "0.800 (80.0%)" in table
        assert "0.750 (75.0%)" in table
        assert ">= 80.0% (0.800)" in table
        assert ">= 50.0% (0.500)" in table
        assert "PASS" in table

    def test_evaluates_from_json_files_on_disk(self, tmp_path: Path) -> None:
        gt_src = MinimalSource(
            file_path="data/raw/vllm-0.10.1/docs/serving/openai.md",
            first_character_index=500,
            last_character_index=900,
        )
        dataset = RagDataset(
            rag_questions=[
                AnsweredQuestion(
                    question_id="q-disk-1",
                    question="How to start OpenAI server?",
                    sources=[gt_src],
                    answer="Use vllm serve.",
                )
            ]
        )
        student = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="q-disk-1",
                    question="How to start OpenAI server?",
                    retrieved_sources=[gt_src],
                )
            ],
            k=5,
        )

        dataset_file = tmp_path / "dataset_docs_public.json"
        student_file = tmp_path / "student_docs.json"
        dataset_file.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
        student_file.write_text(student.model_dump_json(indent=2), encoding="utf-8")

        report = evaluate_search_results(
            student_search_results_path=student_file,
            dataset_path=dataset_file,
            k=5,
        )
        assert report.is_valid is True
        assert report.recall_at_5 == pytest.approx(1.0)
        assert report.docs_recall_at_5 == pytest.approx(1.0)
        assert recall_at_k(student_file, dataset_file, k=5) == pytest.approx(1.0)

    def test_real_public_datasets_perfect_oracle_evaluation_if_present(self) -> None:
        docs_path = Path("data/datasets/AnsweredQuestions/dataset_docs_public.json")
        code_path = Path("data/datasets/AnsweredQuestions/dataset_code_public.json")
        if not docs_path.is_file() or not code_path.is_file():
            pytest.skip("Public ground-truth datasets not present on disk")

        docs_raw = json.loads(docs_path.read_text(encoding="utf-8"))
        docs_oracle = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id=q["question_id"],
                    question=q["question"],
                    retrieved_sources=[
                        MinimalSource(
                            file_path=s["file_path"],
                            first_character_index=s["first_character_index"],
                            last_character_index=min(
                                s["last_character_index"],
                                s["first_character_index"] + 2000,
                            ),
                        )
                        for s in q["sources"]
                    ],
                )
                for q in docs_raw["rag_questions"]
            ],
            k=5,
        )

        report = evaluate_search_results(docs_oracle, docs_path, k=5)
        assert report.is_valid is True
        assert report.recall_at_5 == pytest.approx(1.0)
        assert report.docs_passed is True

    def test_invalid_parameters_and_missing_files_raise_clean_errors(self) -> None:
        with pytest.raises(ValueError, match="iou_threshold"):
            RecallEvaluator(iou_threshold=1.5)

        with pytest.raises(ValueError, match="max_context_length"):
            RecallEvaluator(max_context_length=0)

        with pytest.raises(ValueError, match="default_k"):
            RecallEvaluator(default_k=0)

        with pytest.raises(FileNotFoundError):
            evaluate_search_results("nonexistent_student.json", "nonexistent_dataset.json")

        assert classify_question_category([], dataset_hint="dataset_docs_public.json") == "docs"
        assert classify_question_category([], dataset_hint="dataset_code_public.json") == "code"
