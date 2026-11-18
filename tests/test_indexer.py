"""Unit and integration tests for Ingestion Coordinator & Corpus Indexer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.typing import NDArray
import pytest

from src.indexing.indexer import (
    DEFAULT_BM25_INDEX_PATH,
    DEFAULT_CHUNKS_PATH,
    DEFAULT_CORPUS_DIR,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_PROCESSED_DIR,
    CorpusIndexer,
    IndexingResult,
    build_index,
    load_chunks,
    save_chunks,
)
from src.indexing.schema import InternalChunk
from src.models import MinimalSource
from src.retrieval.dense import EMBEDDING_DIM


def _make_mock_st_model(seed: int = 42) -> MagicMock:
    """Return a deterministic mock SentenceTransformer model."""
    rng = np.random.default_rng(seed)
    mock = MagicMock()

    def _encode(sentences: List[str], **kwargs: Any) -> NDArray[np.float32]:
        n = len(sentences)
        return rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)

    mock.encode.side_effect = _encode
    return mock


@pytest.fixture
def sample_corpus_dir(tmp_path: Path) -> Path:
    """Create a realistic miniature repository with Python and Markdown files."""
    repo_dir = tmp_path / "data" / "raw" / "vllm-0.10.1"
    py_dir = repo_dir / "vllm" / "engine"
    md_dir = repo_dir / "docs" / "serving"
    git_dir = repo_dir / ".git"

    py_dir.mkdir(parents=True)
    md_dir.mkdir(parents=True)
    git_dir.mkdir(parents=True)

    # 1. Python file with class and function
    py_file = py_dir / "async_llm_engine.py"
    py_file.write_text(
        '"""Async LLM Engine module."""\n\n'
        "class AsyncLLMEngine:\n"
        '    """Asynchronous inference engine for vLLM."""\n\n'
        "    def __init__(self, model: str) -> None:\n"
        "        self.model = model\n\n"
        "    async def generate(self, prompt: str) -> str:\n"
        "        return prompt\n\n\n"
        "def create_engine(model_name: str) -> AsyncLLMEngine:\n"
        '    """Factory helper to create an AsyncLLMEngine."""\n'
        "    return AsyncLLMEngine(model_name)\n",
        encoding="utf-8",
    )

    # 2. Markdown file with multiple sections and a code fence
    md_file = md_dir / "openai_compatible_server.md"
    md_file.write_text(
        "# OpenAI Compatible Server\n\n"
        "vLLM provides an HTTP server that implements the OpenAI API protocol.\n\n"
        "## Quickstart\n\n"
        "Start the server with the following command:\n\n"
        "```bash\n"
        "vllm serve Qwen/Qwen3-0.6B --port 8000\n"
        "```\n\n"
        "## Configuration Parameters\n\n"
        "Use `--max-model-len` and `--tensor-parallel-size` to tune serving.\n",
        encoding="utf-8",
    )

    # 3. Ignored file inside .git and binary file (should be skipped)
    (git_dir / "config.py").write_text("x = 1\n", encoding="utf-8")
    (py_dir / "compiled.so").write_bytes(b"\x00\x01\x02\x03binary")

    return repo_dir


class TestConstantsAndAliases:
    """Verify default output path constants and class/function aliases."""

    def test_default_paths_match_acceptance_criteria(self) -> None:
        assert DEFAULT_CORPUS_DIR == Path("data/raw/vllm-0.10.1")
        assert DEFAULT_PROCESSED_DIR == Path("data/processed")
        assert DEFAULT_CHUNKS_PATH == Path("data/processed/chunks.json")
        assert DEFAULT_BM25_INDEX_PATH == Path("data/processed/bm25_index.pkl")
        assert DEFAULT_EMBEDDINGS_PATH == Path("data/processed/embeddings.npy")
        assert DEFAULT_MAX_CHUNK_SIZE == 2000


class TestEndToEndIndexingPipeline:
    """Verify full indexing pipeline produces chunks.json, bm25_index.pkl, and embeddings.npy."""

    def test_full_pipeline_creates_all_three_artifacts(
        self, sample_corpus_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "processed"
        mock_model = _make_mock_st_model()

        indexer = CorpusIndexer(
            corpus_dir=sample_corpus_dir,
            output_dir=output_dir,
            max_chunk_size=2000,
            show_progress=False,
            _model=mock_model,
        )
        result = indexer.index()

        assert isinstance(result, IndexingResult)
        assert result.num_files == 2
        assert result.num_chunks > 0
        assert len(result) == result.num_chunks
        assert result.elapsed_seconds >= 0.0

        # Check dict-style access on IndexingResult
        assert result["num_files"] == 2
        assert result["num_chunks"] == result.num_chunks
        assert "chunks_path" in result
        assert "nonexistent_key" not in result
        assert "num_chunks" in list(result.keys())

        # Verify all 3 required files exist on disk
        expected_chunks_path = output_dir / "chunks.json"
        expected_bm25_path = output_dir / "bm25_index.pkl"
        expected_embeddings_path = output_dir / "embeddings.npy"

        assert result.chunks_path == expected_chunks_path
        assert result.bm25_index_path == expected_bm25_path
        assert result.embeddings_path == expected_embeddings_path

        assert expected_chunks_path.exists()
        assert expected_bm25_path.exists()
        assert expected_embeddings_path.exists()

        # Verify sequential chunk_ids (0 .. N-1) and max_chunk_size compliance
        for idx, chunk in enumerate(result.chunks):
            assert chunk.chunk_id == idx
            assert chunk.span_length <= 2000
            assert len(chunk.content) <= 2000
            assert chunk.file_path.startswith("data/raw/vllm-0.10.1/")

        # Verify embeddings.npy shape and L2 normalization
        loaded_emb = np.load(expected_embeddings_path)
        assert loaded_emb.shape == (result.num_chunks, EMBEDDING_DIM)
        assert loaded_emb.dtype == np.float32
        norms = np.linalg.norm(loaded_emb, axis=1)
        np.testing.assert_allclose(norms, np.ones(result.num_chunks), atol=1e-5)

    def test_load_roundtrip_and_hybrid_search(
        self, sample_corpus_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "processed"
        mock_model = _make_mock_st_model()

        build_index(
            corpus_dir=sample_corpus_dir,
            output_dir=output_dir,
            max_chunk_size=2000,
            show_progress=False,
            _model=mock_model,
        )

        # Load persisted artifacts back
        chunks, bm25_index, dense_encoder = CorpusIndexer.load(
            output_dir=output_dir,
            _model=_make_mock_st_model(seed=99),
        )
        assert len(chunks) > 0
        assert bm25_index.num_docs == len(chunks)
        assert dense_encoder is not None
        assert dense_encoder.embeddings is not None
        assert dense_encoder.embeddings.shape == (len(chunks), EMBEDDING_DIM)

        # Load hybrid ranker directly and run query
        ranker = CorpusIndexer.load_hybrid_ranker(
            output_dir=output_dir,
            _model=_make_mock_st_model(seed=99),
        )
        sources = ranker.search("How to start OpenAI compatible server?", k=3)
        assert len(sources) > 0
        assert all(isinstance(src, MinimalSource) for src in sources)

    def test_enforces_small_max_chunk_size_ceiling(
        self, sample_corpus_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "processed_small"
        mock_model = _make_mock_st_model()

        result = build_index(
            corpus_dir=sample_corpus_dir,
            output_dir=output_dir,
            max_chunk_size=120,
            show_progress=False,
            _model=mock_model,
        )
        assert result.num_chunks > 0
        for idx, chunk in enumerate(result.chunks):
            assert chunk.chunk_id == idx
            assert chunk.span_length <= 120
            assert len(chunk.content) <= 120


class TestTqdmIntegration:
    """Verify tqdm progress tracking is integrated into file indexing and tokenization."""

    def test_tqdm_is_called_for_indexing_and_tokenizing(
        self, sample_corpus_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "processed_tqdm"
        mock_model = _make_mock_st_model()

        with patch("src.indexing.indexer.tqdm") as mock_tqdm:
            # Make mock_tqdm pass through the iterable unchanged
            mock_tqdm.side_effect = lambda iterable, **kwargs: iterable

            indexer = CorpusIndexer(
                corpus_dir=sample_corpus_dir,
                output_dir=output_dir,
                show_progress=True,
                _model=mock_model,
            )
            indexer.index()

            assert mock_tqdm.call_count == 2
            descriptions = [
                call.kwargs.get("desc") for call in mock_tqdm.call_args_list
            ]
            assert "Indexing" in descriptions
            assert "Tokenizing" in descriptions


class TestEdgeCasesAndErrorHandling:
    """Verify defensive handling of empty directories, invalid sizes, and corrupt JSON."""

    def test_empty_corpus_directory_creates_valid_empty_artifacts(
        self, tmp_path: Path
    ) -> None:
        empty_corpus = tmp_path / "empty_corpus"
        empty_corpus.mkdir()
        output_dir = tmp_path / "processed_empty"

        indexer = CorpusIndexer(
            corpus_dir=empty_corpus,
            output_dir=output_dir,
            show_progress=False,
            _model=_make_mock_st_model(),
        )
        result = indexer.index()

        assert result.num_files == 0
        assert result.num_chunks == 0
        assert result.chunks == []
        assert result.chunks_path.exists()
        assert result.bm25_index_path.exists()
        assert result.embeddings_path is not None and result.embeddings_path.exists()

        loaded_chunks = load_chunks(result.chunks_path)
        assert loaded_chunks == []
        loaded_emb = np.load(result.embeddings_path)
        assert loaded_emb.shape == (0, EMBEDDING_DIM)

    def test_enable_dense_false_skips_embeddings(
        self, sample_corpus_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "processed_no_dense"
        indexer = CorpusIndexer(
            corpus_dir=sample_corpus_dir,
            output_dir=output_dir,
            enable_dense=False,
            show_progress=False,
        )
        result = indexer.index()

        assert result.chunks_path.exists()
        assert result.bm25_index_path.exists()
        assert result.embeddings_path is None
        assert result.dense_encoder is None
        assert result.embeddings is None
        assert not (output_dir / "embeddings.npy").exists()

    def test_invalid_max_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_chunk_size must be positive"):
            CorpusIndexer(max_chunk_size=0)
        with pytest.raises(ValueError, match="max_chunk_size must be positive"):
            CorpusIndexer(max_chunk_size=-100)

    def test_missing_corpus_dir_raises(self, tmp_path: Path) -> None:
        indexer = CorpusIndexer(
            corpus_dir=tmp_path / "does_not_exist",
            output_dir=tmp_path / "out",
            enable_dense=False,
            show_progress=False,
        )
        with pytest.raises(FileNotFoundError):
            indexer.index()

    def test_save_and_load_chunks_errors(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_chunks(tmp_path / "missing_chunks.json")

        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{corrupt json", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_chunks(bad_json)

        wrong_type_json = tmp_path / "wrong.json"
        wrong_type_json.write_text(json.dumps({"not_chunks": 123}), encoding="utf-8")
        with pytest.raises(ValueError, match="Expected JSON array"):
            load_chunks(wrong_type_json)

        # Also support {"chunks": [...]} dict wrapper format
        wrapped_json = tmp_path / "wrapped.json"
        sample_chunk = InternalChunk(
            chunk_id=0,
            file_path="data/raw/vllm-0.10.1/a.py",
            first_character_index=0,
            last_character_index=10,
            chunk_type="python_module",
            token_count=2,
            content="x = 1\n",
        )
        direct_path = save_chunks([sample_chunk], tmp_path / "direct.json")
        assert load_chunks(direct_path) == [sample_chunk]

        wrapped_json.write_text(
            json.dumps({"chunks": [sample_chunk.model_dump()]}),
            encoding="utf-8",
        )
        loaded = load_chunks(wrapped_json)
        assert len(loaded) == 1
        assert loaded[0] == sample_chunk


@pytest.mark.skipif(
    not DEFAULT_CORPUS_DIR.exists(),
    reason="vllm-0.10.1 corpus not present on disk",
)
class TestVLLMCorpusIngestion:
    """Verify file count, chunk count, and 2000-character ceiling on vllm-0.10.1."""

    def test_vllm_corpus_chunking_and_bm25_within_sla(self, tmp_path: Path) -> None:
        indexer = CorpusIndexer(
            corpus_dir=DEFAULT_CORPUS_DIR,
            output_dir=tmp_path / "vllm_processed",
            max_chunk_size=2000,
            enable_dense=False,
            show_progress=False,
        )
        files, chunks = indexer.chunk_corpus()

        # Acceptance criteria: ~1,965 files, ~14,000 chunks
        assert 1500 <= len(files) <= 2500
        assert 10000 <= len(chunks) <= 35000

        # Zero over-limit chunks & sequential chunk_ids
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_id == idx
            assert chunk.span_length <= 2000
            assert len(chunk.content) <= 2000
            assert chunk.file_path.startswith("data/raw/vllm-0.10.1/")
