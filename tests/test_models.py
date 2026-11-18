"""Tests for data interchange schemas in src/models.py."""

from pathlib import Path
import json
import pytest
from pydantic import ValidationError

from src.models import (
    MinimalSource,
    UnansweredQuestion,
    AnsweredQuestion,
    RagDataset,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    normalize_file_path,
)


class TestFilePathNormalization:
    """Test path normalization logic."""

    def test_strip_leading_relative_prefix(self) -> None:
        p = normalize_file_path("./data/raw/vllm-0.10.1/docs/README.md")
        assert p == "data/raw/vllm-0.10.1/docs/README.md"

    def test_clean_forward_slashes(self) -> None:
        p = normalize_file_path("data\\raw\\vllm-0.10.1\\vllm\\model.py")
        assert p == "data/raw/vllm-0.10.1/vllm/model.py"

    def test_path_instance_input(self) -> None:
        p = normalize_file_path(Path("./data/raw/vllm-0.10.1/vllm/entrypoints/llm.py"))
        assert p == "data/raw/vllm-0.10.1/vllm/entrypoints/llm.py"

    def test_absolute_path_with_data_raw(self) -> None:
        p = normalize_file_path("/home/user/workspace/data/raw/vllm-0.10.1/vllm/engine.py")
        assert p == "data/raw/vllm-0.10.1/vllm/engine.py"

    def test_relative_path_within_cwd(self) -> None:
        cwd_abs = Path.cwd() / "src/models.py"
        p = normalize_file_path(cwd_abs)
        assert p == "src/models.py"

    def test_redundant_internal_slashes_and_dots(self) -> None:
        p = normalize_file_path("./data/raw//vllm-0.10.1/sub/../docs/index.md")
        assert p == "data/raw/vllm-0.10.1/docs/index.md"

    def test_empty_path_raises_error(self) -> None:
        with pytest.raises(ValueError):
            normalize_file_path("")

    def test_invalid_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            normalize_file_path(123)  # type: ignore[arg-type]


class TestMinimalSource:
    """Test MinimalSource model validation and behavior."""

    def test_valid_minimal_source(self) -> None:
        src = MinimalSource(
            file_path="./data/raw/vllm-0.10.1/vllm/model.py",
            first_character_index=100,
            last_character_index=350,
        )
        assert src.file_path == "data/raw/vllm-0.10.1/vllm/model.py"
        assert src.first_character_index == 100
        assert src.last_character_index == 350
        assert src.span_length == 250

    def test_zero_start_index(self) -> None:
        src = MinimalSource(
            file_path="data/raw/vllm-0.10.1/docs/index.md",
            first_character_index=0,
            last_character_index=150,
        )
        assert src.first_character_index == 0
        assert src.span_length == 150

    def test_negative_start_index_fails(self) -> None:
        with pytest.raises(ValidationError):
            MinimalSource(
                file_path="data/raw/vllm-0.10.1/vllm/model.py",
                first_character_index=-1,
                last_character_index=10,
            )

    def test_last_index_not_greater_than_first_fails(self) -> None:
        # Equal indices
        with pytest.raises(ValidationError) as exc_info:
            MinimalSource(
                file_path="data/raw/vllm-0.10.1/vllm/model.py",
                first_character_index=50,
                last_character_index=50,
            )
        assert "must be greater than" in str(exc_info.value)

        # Inverted indices
        with pytest.raises(ValidationError) as exc_info:
            MinimalSource(
                file_path="data/raw/vllm-0.10.1/vllm/model.py",
                first_character_index=100,
                last_character_index=40,
            )
        assert "must be greater than" in str(exc_info.value)

    def test_immutability(self) -> None:
        src = MinimalSource(
            file_path="data/raw/vllm-0.10.1/vllm/model.py",
            first_character_index=10,
            last_character_index=100,
        )
        with pytest.raises(ValidationError):
            setattr(src, "first_character_index", 20)

    def test_hashability(self) -> None:
        src1 = MinimalSource(
            file_path="data/raw/vllm-0.10.1/vllm/model.py",
            first_character_index=10,
            last_character_index=100,
        )
        src2 = MinimalSource(
            file_path="./data/raw/vllm-0.10.1/vllm/model.py",
            first_character_index=10,
            last_character_index=100,
        )
        s = {src1, src2}
        assert len(s) == 1


class TestQuestionsAndDatasets:
    """Test UnansweredQuestion, AnsweredQuestion, and RagDataset."""

    def test_unanswered_question_defaults(self) -> None:
        uq = UnansweredQuestion(question="How does batching work?")
        assert len(uq.question_id) > 0
        assert uq.question == "How does batching work?"

    def test_custom_question_id(self) -> None:
        uq = UnansweredQuestion(question_id="custom-id-42", question="Test question")
        assert uq.question_id == "custom-id-42"

    def test_empty_question_fails(self) -> None:
        with pytest.raises(ValidationError):
            UnansweredQuestion(question="")

    def test_whitespace_and_tabs_question_fails(self) -> None:
        with pytest.raises(ValidationError):
            UnansweredQuestion(question="   ")
        with pytest.raises(ValidationError):
            UnansweredQuestion(question="\t\t")
        with pytest.raises(ValidationError):
            UnansweredQuestion(question=" \t \n ")

    def test_question_strips_surrounding_whitespace_and_tabs(self) -> None:
        uq = UnansweredQuestion(question=" \t How does batching work? \t\n")
        assert uq.question == "How does batching work?"

    def test_answered_question(self) -> None:
        source = MinimalSource(
            file_path="data/raw/vllm-0.10.1/vllm/entrypoints/llm.py",
            first_character_index=500,
            last_character_index=900,
        )
        aq = AnsweredQuestion(
            question_id="q-1",
            question="What is LLM?",
            sources=[source],
            answer="LLM is the main entrypoint class.",
        )
        assert aq.question_id == "q-1"
        assert len(aq.sources) == 1
        assert aq.answer == "LLM is the main entrypoint class."

    def test_rag_dataset_mixed(self) -> None:
        uq = UnansweredQuestion(question="Q1")
        source = MinimalSource(
            file_path="data/raw/vllm-0.10.1/docs/README.md",
            first_character_index=0,
            last_character_index=50,
        )
        aq = AnsweredQuestion(
            question="Q2",
            sources=[source],
            answer="Ans2",
        )
        dataset = RagDataset(rag_questions=[uq, aq])
        assert len(dataset.rag_questions) == 2
        assert isinstance(dataset.rag_questions[0], UnansweredQuestion)
        assert isinstance(dataset.rag_questions[1], AnsweredQuestion)

    def test_load_real_public_code_dataset(self) -> None:
        path = Path("data/datasets/AnsweredQuestions/dataset_code_public.json")
        if not path.exists():
            pytest.skip("Dataset file not found")
        content = path.read_text(encoding="utf-8")
        dataset = RagDataset.model_validate_json(content)
        assert len(dataset.rag_questions) > 0
        first_q = dataset.rag_questions[0]
        assert isinstance(first_q, AnsweredQuestion)
        assert len(first_q.sources) > 0
        assert first_q.sources[0].first_character_index >= 0
        assert (
            first_q.sources[0].last_character_index
            > first_q.sources[0].first_character_index
        )

    def test_load_real_public_unanswered_dataset(self) -> None:
        path = Path("data/datasets/UnansweredQuestions/dataset_code_public.json")
        if not path.exists():
            pytest.skip("Dataset file not found")
        content = path.read_text(encoding="utf-8")
        dataset = RagDataset.model_validate_json(content)
        assert len(dataset.rag_questions) > 0
        first_q = dataset.rag_questions[0]
        assert isinstance(first_q, UnansweredQuestion)
        assert not isinstance(first_q, AnsweredQuestion)


class TestSearchResultsAndAnswers:
    """Test search result models and student outputs."""

    def test_minimal_search_results(self) -> None:
        source = MinimalSource(
            file_path="data/raw/vllm-0.10.1/vllm/model.py",
            first_character_index=10,
            last_character_index=20,
        )
        res = MinimalSearchResults(
            question_id="q-10",
            question=" \t Explain caching \t\n",
            retrieved_sources=[source],
        )
        assert res.question_id == "q-10"
        assert res.question == "Explain caching"
        assert len(res.retrieved_sources) == 1

        with pytest.raises(ValidationError):
            MinimalSearchResults(
                question_id="q-10",
                question="   \t   ",
                retrieved_sources=[source],
            )

    def test_minimal_answer(self) -> None:
        source = MinimalSource(
            file_path="data/raw/vllm-0.10.1/vllm/model.py",
            first_character_index=10,
            last_character_index=20,
        )
        ans = MinimalAnswer(
            question_id="q-10",
            question="Explain caching",
            retrieved_sources=[source],
            answer="Caching stores computed states.",
        )
        assert ans.question_id == "q-10"
        assert ans.answer == "Caching stores computed states."
        assert len(ans.retrieved_sources) == 1

    def test_student_search_results_batch(self) -> None:
        res = MinimalSearchResults(
            question_id="q-1",
            question="Test",
            retrieved_sources=[],
        )
        student_res = StudentSearchResults(search_results=[res], k=5)
        assert student_res.k == 5
        assert len(student_res.search_results) == 1

        # Check serialization matches expected format
        dumped = json.loads(student_res.model_dump_json())
        assert "search_results" in dumped
        assert dumped["k"] == 5

    def test_student_search_results_and_answer_batch(self) -> None:
        ans = MinimalAnswer(
            question_id="q-1",
            question="Test",
            retrieved_sources=[],
            answer="Answer text",
        )
        student_ans = StudentSearchResultsAndAnswer(search_results=[ans], k=5)
        assert student_ans.k == 5
        assert len(student_ans.search_results) == 1

        dumped = json.loads(student_ans.model_dump_json())
        assert "search_results" in dumped
        assert dumped["search_results"][0]["answer"] == "Answer text"
        assert dumped["k"] == 5
