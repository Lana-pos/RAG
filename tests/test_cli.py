"""Unit and integration tests for CLI Interface Router (RAG-702)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
from unittest.mock import MagicMock

import fire
import numpy as np
from numpy.typing import NDArray
import pytest
import torch
import torch.nn as nn

from src.__main__ import main as entrypoint_main
from src.cli import (
    CLIAnswerDatasetResult,
    CLIAnswerResult,
    CLISearchDatasetResult,
    CLISearchResult,
    DEFAULT_ANSWER_K,
    DEFAULT_CORPUS_DIR,
    DEFAULT_DATASET_SEARCH_K,
    DEFAULT_EVALUATE_K,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_SEARCH_K,
    RAGCLI,
    main,
    run_cli,
)
from src.indexing.indexer import IndexingResult
from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from src.retrieval.dense import EMBEDDING_DIM
from src.utils.evaluation import EvaluationResult, validate_student_search_results


def _make_mock_st_model(seed: int = 42) -> MagicMock:
    """Return a deterministic mock SentenceTransformer model for offline testing."""
    rng = np.random.default_rng(seed)
    mock = MagicMock()

    def _encode(sentences: List[str], **kwargs: Any) -> NDArray[np.float32]:
        n = len(sentences)
        return rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)

    mock.encode.side_effect = _encode
    return mock


class DummyTokenizer:
    """Lightweight deterministic tokenizer for offline Qwen3-0.6B CLI testing."""

    def __init__(self) -> None:
        self.pad_token: str = "<|endoftext|>"
        self.eos_token: str = "<|im_end|>"
        self.pad_token_id: int = 0
        self.eos_token_id: int = 2
        self._id_to_text: Dict[int, str] = {
            101: "Use ",
            102: "`/v1/load_lora_adapter` ",
            103: "or ",
            104: "`vllm serve --port 8000` ",
            105: "to configure the server.",
        }

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 3000,
    ) -> Dict[str, torch.Tensor]:
        word_count = max(1, min(len(text.split()), max_length))
        input_ids = torch.arange(1, word_count + 1, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        word_count = max(1, len(text.split()))
        return list(range(1, word_count + 1))

    def decode(
        self,
        token_ids: Union[torch.Tensor, Sequence[int]],
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            ids_list = token_ids.detach().cpu().view(-1).tolist()
        else:
            ids_list = list(token_ids)

        pieces: List[str] = [
            self._id_to_text[int(tid)]
            for tid in ids_list
            if int(tid) in self._id_to_text
        ]
        return "".join(pieces) if pieces else "Grounded answer from context."


class DummyCausalLM(nn.Module):
    """Lightweight PyTorch causal LM mock for deterministic greedy decoding."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros((8, 8), dtype=torch.bfloat16),
            requires_grad=False,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        gen_tokens = torch.tensor(
            [[101, 102, 103, 104, 105]],
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.cat([input_ids, gen_tokens], dim=-1)


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """Create a sample vLLM corpus with Python and Markdown files."""
    repo_dir = tmp_path / "data" / "raw" / "vllm-0.10.1"
    py_dir = repo_dir / "vllm" / "entrypoints" / "openai"
    md_dir = repo_dir / "docs" / "features"

    py_dir.mkdir(parents=True)
    md_dir.mkdir(parents=True)

    py_file = py_dir / "api_server.py"
    py_file.write_text(
        '"""OpenAI compatible HTTP server entrypoint."""\n\n'
        "class OpenAIServingChat:\n"
        '    """Handles chat completions on the OpenAI compatible server."""\n\n'
        "    def __init__(self, engine_client: str, lora_modules: list[str]) -> None:\n"
        "        self.engine_client = engine_client\n"
        "        self.lora_modules = lora_modules\n\n"
        "    async def create_chat_completion(self, request: dict) -> dict:\n"
        '        """Create a chat completion response."""\n'
        '        return {"status": "ok"}\n\n\n'
        "def run_server(host: str = '0.0.0.0', port: int = 8000) -> None:\n"
        '    """Start the vLLM OpenAI API server on the specified host and port."""\n'
        "    pass\n",
        encoding="utf-8",
    )

    md_file = md_dir / "lora.md"
    md_file.write_text(
        "# LoRA Adapters\n\n"
        "vLLM supports dynamic loading of LoRA adapters at runtime.\n\n"
        "## Dynamically Loading a LoRA Adapter\n\n"
        "Send a POST request to the `/v1/load_lora_adapter` HTTP endpoint "
        "to dynamically load a LoRA adapter in vLLM.\n\n"
        "## Configuring the OpenAI Server\n\n"
        "Use `vllm serve Qwen/Qwen3-0.6B --enable-lora --port 8000` to launch.\n",
        encoding="utf-8",
    )

    return repo_dir


@pytest.fixture
def indexed_cli(sample_corpus: Path, tmp_path: Path) -> RAGCLI:
    """Return an initialized and indexed RAGCLI instance backed by offline mocks."""
    processed_dir = tmp_path / "data" / "processed"
    cli = RAGCLI(
        corpus_dir=sample_corpus,
        output_dir=processed_dir,
        show_progress=False,
        _st_model=_make_mock_st_model(),
        _llm_model=DummyCausalLM(),
        _llm_tokenizer=DummyTokenizer(),
    )
    cli.index(max_chunk_size=2000)
    return cli


class TestCLIConstantsAndAliases:
    """Verify CLI defaults, aliases, and __main__ exports."""

    def test_defaults_and_aliases(self) -> None:
        assert DEFAULT_CORPUS_DIR == Path("data/raw/vllm-0.10.1")
        assert DEFAULT_PROCESSED_DIR == Path("data/processed")
        assert DEFAULT_MAX_CHUNK_SIZE == 2000
        assert DEFAULT_SEARCH_K == 5
        assert DEFAULT_DATASET_SEARCH_K == 10
        assert DEFAULT_ANSWER_K == 5
        assert DEFAULT_EVALUATE_K == 5
        assert DEFAULT_MODEL_NAME == "Qwen/Qwen3-0.6B"

        assert entrypoint_main is main

    def test_cli_exposes_all_six_required_commands(self) -> None:
        cli = RAGCLI()
        for cmd_name in (
            "index",
            "search",
            "search_dataset",
            "answer",
            "answer_dataset",
            "evaluate",
        ):
            assert hasattr(cli, cmd_name), f"Missing CLI command: {cmd_name}"
            assert callable(getattr(cli, cmd_name))


class TestCLIIndexCommand:
    """Verify `index [--max_chunk_size 2000]` builds all index artifacts."""

    def test_index_command_builds_chunks_bm25_and_embeddings(
        self,
        sample_corpus: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        processed_dir = tmp_path / "processed"
        cli = RAGCLI(
            corpus_dir=sample_corpus,
            output_dir=processed_dir,
            show_progress=False,
            _st_model=_make_mock_st_model(),
        )

        result = cli.index(max_chunk_size=2000)
        captured = capsys.readouterr()

        assert isinstance(result, IndexingResult)
        assert result.num_files == 2
        assert result.num_chunks > 0
        assert (processed_dir / "chunks.json").is_file()
        assert (processed_dir / "bm25_index.pkl").is_file()
        assert (processed_dir / "embeddings.npy").is_file()
        assert "Ingestion complete! Indices saved under" in captured.out
        assert all(c.span_length <= 2000 for c in result.chunks)

    def test_index_via_python_fire_router(
        self,
        sample_corpus: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        processed_dir = tmp_path / "fire_processed"
        cli = RAGCLI(
            corpus_dir=sample_corpus,
            output_dir=processed_dir,
            show_progress=False,
            _st_model=_make_mock_st_model(),
        )

        res = run_cli(command=["index", "--max_chunk_size", "1500"], cli=cli)
        captured = capsys.readouterr()

        assert isinstance(res, IndexingResult)
        assert all(c.span_length <= 1500 for c in res.chunks)
        assert "Ingestion complete! Indices saved under" in captured.out


class TestCLISearchCommand:
    """Verify `search <query> [--k 5]` and Pydantic schema conformance."""

    def test_search_single_query_conforms_to_pydantic_schemas(
        self,
        indexed_cli: RAGCLI,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        capsys.readouterr()  # Clear indexing output
        res = indexed_cli.search(
            "What HTTP endpoint is used to dynamically load a LoRA adapter?",
            k=5,
        )
        captured = capsys.readouterr()

        assert isinstance(res, MinimalSearchResults)
        assert isinstance(res, CLISearchResult)
        assert res.k == 5
        assert 1 <= len(res.retrieved_sources) <= 5
        assert all(isinstance(s, MinimalSource) for s in res.retrieved_sources)
        assert all(s.span_length <= 2000 for s in res.retrieved_sources)
        assert all(not s.file_path.startswith("./") for s in res.retrieved_sources)

        # Stdout must be valid JSON conforming to both MinimalSearchResults and StudentSearchResults
        single_validated = MinimalSearchResults.model_validate_json(captured.out)
        batch_validated = StudentSearchResults.model_validate_json(captured.out)
        assert single_validated.question == res.question
        assert batch_validated.k == 5
        assert len(batch_validated.search_results) == 1

        # Dict-style access on result
        assert res["k"] == 5
        assert len(res["search_results"]) == 1
        assert "retrieved_sources" in res

    def test_search_via_python_fire_with_default_k(
        self,
        indexed_cli: RAGCLI,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        capsys.readouterr()
        res = fire.Fire(
            indexed_cli,
            command=["search", "How to configure OpenAI server?", "--k", "3"],
        )
        captured = capsys.readouterr()

        assert isinstance(res, MinimalSearchResults)
        assert isinstance(res, CLISearchResult)
        assert res.k == 3
        assert len(res.retrieved_sources) <= 3
        parsed = json.loads(captured.out)
        assert parsed["k"] == 3
        assert parsed["question"] == "How to configure OpenAI server?"


class TestCLISearchDatasetCommand:
    """Verify `search_dataset` writes 100% valid StudentSearchResults JSON files."""

    def test_search_dataset_writes_valid_student_search_results_file(
        self,
        indexed_cli: RAGCLI,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dataset = RagDataset(
            rag_questions=[
                UnansweredQuestion(
                    question_id="q-lora-1",
                    question="What HTTP endpoint dynamically loads a LoRA adapter?",
                ),
                UnansweredQuestion(
                    question_id="q-openai-2",
                    question="How do I start the OpenAI compatible server?",
                ),
            ]
        )
        dataset_path = (
            tmp_path / "datasets" / "UnansweredQuestions" / "dataset_docs_public.json"
        )
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")

        save_dir = tmp_path / "output" / "search_results" / "UnansweredQuestions"
        capsys.readouterr()

        res = run_cli(
            command=[
                "search_dataset",
                "--dataset_path",
                str(dataset_path),
                "--save_directory",
                str(save_dir),
                "--k",
                "10",
            ],
            cli=indexed_cli,
        )
        captured = capsys.readouterr()

        expected_file = save_dir / "dataset_docs_public.json"
        assert expected_file.is_file()
        assert f"Saved student_search_results to {expected_file}" in captured.out

        # 100% Pydantic StudentSearchResults schema validation on written file
        raw_content = expected_file.read_text(encoding="utf-8")
        raw_dict = json.loads(raw_content)
        assert set(raw_dict.keys()) == {"search_results", "k"}

        validated = StudentSearchResults.model_validate_json(raw_content)
        assert validated.k == 10
        assert len(validated.search_results) == 2
        assert validated.search_results[0].question_id == "q-lora-1"
        assert validated.search_results[1].question_id == "q-openai-2"

        is_valid, errors = validate_student_search_results(
            validated, max_context_length=2000
        )
        assert is_valid is True, f"Validation errors: {errors}"
        assert isinstance(res, StudentSearchResults)
        assert isinstance(res, CLISearchDatasetResult)
        assert res.output_path == expected_file


class TestCLIAnswerCommand:
    """Verify `answer <query> [--k 5]` produces grounded MinimalAnswer JSON."""

    def test_answer_single_query_conforms_to_pydantic_schemas(
        self,
        indexed_cli: RAGCLI,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        capsys.readouterr()
        res = run_cli(
            command=[
                "answer",
                "What HTTP endpoint is used to dynamically load a LoRA adapter?",
                "--k",
                "5",
            ],
            cli=indexed_cli,
        )
        captured = capsys.readouterr()

        assert isinstance(res, MinimalAnswer)
        assert isinstance(res, CLIAnswerResult)
        assert res.k == 5
        assert len(res.answer) > 0
        assert "/v1/load_lora_adapter" in res.answer

        # Stdout must validate against both MinimalAnswer and StudentSearchResultsAndAnswer
        single_ans = MinimalAnswer.model_validate_json(captured.out)
        batch_ans = StudentSearchResultsAndAnswer.model_validate_json(captured.out)
        assert single_ans.answer == res.answer
        assert batch_ans.k == 5
        assert len(batch_ans.search_results) == 1
        assert batch_ans.search_results[0].answer == res.answer


class TestCLIAnswerDatasetCommand:
    """Verify `answer_dataset` writes 100% valid StudentSearchResultsAndAnswer JSON."""

    def test_answer_dataset_writes_valid_student_search_results_and_answer_file(
        self,
        indexed_cli: RAGCLI,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # 1. Run search_dataset first
        dataset = RagDataset(
            rag_questions=[
                UnansweredQuestion(
                    question_id="q-ans-1",
                    question="What endpoint loads a LoRA adapter?",
                ),
                UnansweredQuestion(
                    question_id="q-ans-2",
                    question="How do I start the OpenAI server?",
                ),
            ]
        )
        dataset_path = tmp_path / "dataset_docs_public.json"
        dataset_path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")

        search_save_dir = tmp_path / "search_results" / "UnansweredQuestions"
        indexed_cli.search_dataset(
            dataset_path=dataset_path,
            save_directory=search_save_dir,
            k=5,
        )
        student_sr_path = search_save_dir / "dataset_docs_public.json"
        assert student_sr_path.is_file()

        # 2. Run answer_dataset via Fire
        answer_save_dir = (
            tmp_path / "search_results_and_answer" / "UnansweredQuestions"
        )
        capsys.readouterr()

        res = run_cli(
            command=[
                "answer_dataset",
                "--student_search_results_path",
                str(student_sr_path),
                "--save_directory",
                str(answer_save_dir),
            ],
            cli=indexed_cli,
        )
        captured = capsys.readouterr()

        expected_ans_file = answer_save_dir / "dataset_docs_public.json"
        assert expected_ans_file.is_file()
        assert "Loaded 2 questions ... Processed 2 of 2 questions" in captured.out
        assert (
            f"Saved student_search_results_and_answer to {expected_ans_file}"
            in captured.out
        )

        # 100% Pydantic StudentSearchResultsAndAnswer validation on written file
        raw_content = expected_ans_file.read_text(encoding="utf-8")
        raw_dict = json.loads(raw_content)
        assert set(raw_dict.keys()) == {"search_results", "k"}

        validated = StudentSearchResultsAndAnswer.model_validate_json(raw_content)
        assert validated.k == 5
        assert len(validated.search_results) == 2
        assert all(isinstance(item, MinimalAnswer) for item in validated.search_results)
        assert all(len(item.answer.strip()) > 0 for item in validated.search_results)

        is_valid, errors = validate_student_search_results(
            validated, max_context_length=2000
        )
        assert is_valid is True, f"Validation errors: {errors}"
        assert isinstance(res, StudentSearchResultsAndAnswer)
        assert isinstance(res, CLIAnswerDatasetResult)
        assert res.output_path == expected_ans_file


class TestCLIEvaluateCommand:
    """Verify `evaluate` computes Recall@k and outputs the summary table."""

    def test_evaluate_command_computes_recall_and_prints_table(
        self,
        indexed_cli: RAGCLI,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Build ground-truth dataset matching our sample corpus chunks
        lora_chunk = next(
            c for c in (indexed_cli._chunks or []) if "lora.md" in c.file_path
        )
        gt_dataset = RagDataset(
            rag_questions=[
                AnsweredQuestion(
                    question_id="q-eval-1",
                    question="What HTTP endpoint dynamically loads a LoRA adapter?",
                    sources=[lora_chunk.to_source()],
                    answer="The /v1/load_lora_adapter endpoint.",
                )
            ]
        )
        gt_path = tmp_path / "AnsweredQuestions" / "dataset_docs_public.json"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_text(gt_dataset.model_dump_json(indent=2), encoding="utf-8")

        search_dir = tmp_path / "search_results"
        indexed_cli.search_dataset(
            dataset_path=gt_path,
            save_directory=search_dir,
            k=5,
        )
        student_sr_path = search_dir / "dataset_docs_public.json"

        capsys.readouterr()
        eval_json_out = tmp_path / "eval_report.json"
        report = run_cli(
            command=[
                "evaluate",
                "--student_search_results_path",
                str(student_sr_path),
                "--dataset_path",
                str(gt_path),
                "--k",
                "5",
                "--output_path",
                str(eval_json_out),
            ],
            cli=indexed_cli,
        )
        captured = capsys.readouterr()

        assert isinstance(report, EvaluationResult)
        assert report.is_valid is True
        assert report.recall_at_5 == pytest.approx(1.0)
        assert "Student data is valid: True" in captured.out
        assert "Evaluation Results" in captured.out
        assert "Recall@5: 1.000" in captured.out
        assert eval_json_out.is_file()


class TestCLIDegenerateInputs:
    """Verify CLI handles degenerate inputs gracefully without unhandled tracebacks."""

    def test_empty_and_whitespace_and_numeric_queries(
        self,
        indexed_cli: RAGCLI,
    ) -> None:
        # 1. Empty string query
        empty_search = indexed_cli.search("", k=5)
        assert isinstance(empty_search, MinimalSearchResults)
        assert empty_search.retrieved_sources == []
        MinimalSearchResults.model_validate_json(str(empty_search))
        StudentSearchResults.model_validate_json(str(empty_search))

        # 2. Whitespace-only query
        ws_search = indexed_cli.search("   \t\n  ", k=5)
        assert isinstance(ws_search, CLISearchResult)
        assert ws_search.retrieved_sources == []

        # 3. Numeric query (as coerced by Python Fire)
        num_search = indexed_cli.search(12345, k=5)
        assert isinstance(num_search, MinimalSearchResults)

        # 4. Empty query to answer
        empty_ans = indexed_cli.answer("   ", k=5)
        assert isinstance(empty_ans, MinimalAnswer)
        assert empty_ans.retrieved_sources == []
        MinimalAnswer.model_validate_json(str(empty_ans))
        StudentSearchResultsAndAnswer.model_validate_json(str(empty_ans))

    def test_k_zero_and_negative_k_handled_gracefully(
        self,
        indexed_cli: RAGCLI,
        tmp_path: Path,
    ) -> None:
        # k=0 and k=-5 in single search
        s_zero = indexed_cli.search("LoRA adapter", k=0)
        assert isinstance(s_zero, CLISearchResult)
        assert s_zero.retrieved_sources == []
        assert s_zero.k == 1
        StudentSearchResults.model_validate_json(str(s_zero))

        s_neg = indexed_cli.search("LoRA adapter", k=-5)
        assert isinstance(s_neg, CLISearchResult)
        assert s_neg.retrieved_sources == []
        assert s_neg.k == 1

        # k=0 in search_dataset still produces valid StudentSearchResults (k >= 1)
        ds = RagDataset(
            rag_questions=[UnansweredQuestion(question_id="q1", question="LoRA?")]
        )
        ds_path = tmp_path / "ds.json"
        ds_path.write_text(ds.model_dump_json(), encoding="utf-8")
        out_dir = tmp_path / "out_k0"

        ds_res = indexed_cli.search_dataset(
            dataset_path=ds_path,
            save_directory=out_dir,
            k=0,
        )
        assert isinstance(ds_res, CLISearchDatasetResult)
        assert ds_res.k == 1
        assert ds_res.search_results[0].retrieved_sources == []
        saved_file = out_dir / "ds.json"
        StudentSearchResults.model_validate_json(saved_file.read_text(encoding="utf-8"))

        # k=0 in answer
        ans_zero = indexed_cli.answer("LoRA adapter", k=0)
        assert isinstance(ans_zero, CLIAnswerResult)
        assert ans_zero.retrieved_sources == []
        assert ans_zero.k == 1

    def test_max_chunk_size_clamping_and_invalid_values(
        self,
        sample_corpus: Path,
        tmp_path: Path,
    ) -> None:
        cli = RAGCLI(
            corpus_dir=sample_corpus,
            output_dir=tmp_path / "clamped_out",
            show_progress=False,
            _st_model=_make_mock_st_model(),
        )
        # Oversized max_chunk_size=5000 must be clamped to 2000
        res_over = cli.index(max_chunk_size=5000)
        assert all(c.span_length <= 2000 for c in res_over.chunks)

        # Non-positive max_chunk_size=0 must be clamped to 2000 without crashing
        res_zero = cli.index(max_chunk_size=0)
        assert res_zero.num_chunks > 0
        assert all(c.span_length <= 2000 for c in res_zero.chunks)

    def test_missing_files_and_malformed_json_do_not_crash(
        self,
        indexed_cli: RAGCLI,
        tmp_path: Path,
    ) -> None:
        # 1. Missing dataset file in search_dataset
        res_missing = indexed_cli.search_dataset(
            dataset_path=tmp_path / "does_not_exist.json",
            save_directory=tmp_path / "out",
        )
        assert isinstance(res_missing, StudentSearchResults)
        assert res_missing.search_results == []

        # 2. Malformed JSON in search_dataset
        bad_json = tmp_path / "corrupt.json"
        bad_json.write_text("{corrupt_json: [1, 2", encoding="utf-8")
        res_bad = indexed_cli.search_dataset(
            dataset_path=bad_json,
            save_directory=tmp_path / "out",
        )
        assert isinstance(res_bad, StudentSearchResults)
        assert res_bad.search_results == []

        # 3. Missing file in answer_dataset
        ans_missing = indexed_cli.answer_dataset(
            student_search_results_path=tmp_path / "missing_sr.json",
            save_directory=tmp_path / "out_ans",
        )
        assert isinstance(ans_missing, StudentSearchResultsAndAnswer)
        assert ans_missing.search_results == []

        # 4. Malformed JSON in evaluate
        eval_bad = indexed_cli.evaluate(
            student_search_results_path=bad_json,
            dataset_path=bad_json,
            k=5,
        )
        assert isinstance(eval_bad, EvaluationResult)
        assert eval_bad.is_valid is False

        # 5. Missing corpus_dir in index
        bad_cli = RAGCLI(
            corpus_dir=tmp_path / "missing_corpus",
            output_dir=tmp_path / "missing_out",
            enable_dense=False,
            show_progress=False,
        )
        idx_res = bad_cli.index()
        assert isinstance(idx_res, IndexingResult)
        assert idx_res.num_files == 0

    def test_main_entrypoint_returns_clean_exit_codes(
        self,
        tmp_path: Path,
    ) -> None:
        # Help flag exits cleanly with code 0
        assert main(["--help"]) == 0
        assert main(["search", "--help"]) == 0

        # Missing file via main() does not raise unhandled exception
        code = main(
            [
                "search_dataset",
                "--dataset_path",
                str(tmp_path / "nonexistent.json"),
                "--save_directory",
                str(tmp_path / "out"),
            ]
        )
        assert code == 0
