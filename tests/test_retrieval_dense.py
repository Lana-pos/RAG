"""Unit and performance tests for DenseEncoder (src/retrieval/dense.py).

All tests use a lightweight mock SentenceTransformer so the suite runs
fully offline without downloading the real model.  The mock produces
deterministic random embeddings of the correct dimensionality (384-d),
allowing every public API contract to be verified without network access.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.typing import NDArray
import pytest

from src.indexing.schema import InternalChunk
from src.models import MinimalSource
from src.retrieval.dense import (
    DEFAULT_EMBEDDINGS_PATH,
    EMBEDDING_DIM,
    MODEL_NAME,
    DenseEncoder,
    build_dense_index,
    load_dense_index,
)


def _make_mock_model(n_texts: int | None = None, seed: int = 42) -> MagicMock:
    """Return a MagicMock whose `.encode()` returns deterministic float32 vectors.

    If *n_texts* is ``None``, the mock infers the vector count from the length
    of the *sentences* argument passed to ``.encode()``.
    """
    rng = np.random.default_rng(seed)
    cache: dict[str, NDArray[np.float32]] = {}
    mock = MagicMock()

    def _encode(sentences: List[str], **kwargs: Any) -> NDArray[np.float32]:
        rows: list[NDArray[np.float32]] = []
        for s in sentences:
            if s not in cache:
                cache[s] = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
            rows.append(cache[s])
        if not rows:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        return np.stack(rows, axis=0)

    mock.encode.side_effect = _encode
    return mock


def _make_chunks(n: int = 4) -> List[InternalChunk]:
    """Return a list of *n* synthetic InternalChunk fixtures."""
    chunks = []
    for i in range(n):
        chunks.append(
            InternalChunk(
                chunk_id=i,
                file_path=f"data/raw/vllm-0.10.1/vllm/module_{i}.py",
                first_character_index=i * 100,
                last_character_index=i * 100 + 50,
                chunk_type="python_func",
                token_count=10,
                content=f"def function_{i}(): pass  # chunk {i}",
            )
        )
    return chunks


@pytest.fixture
def mock_model() -> MagicMock:
    """Fresh mock SentenceTransformer per test."""
    return _make_mock_model()


@pytest.fixture
def sample_chunks() -> List[InternalChunk]:
    return _make_chunks(4)


@pytest.fixture
def encoder(mock_model: MagicMock, sample_chunks: List[InternalChunk]) -> DenseEncoder:
    """DenseEncoder already fitted on sample_chunks, using mock model."""
    enc = DenseEncoder(_model=mock_model)
    enc.encode_corpus(sample_chunks, save_path=None)
    return enc


class TestDenseEncoderConstruction:
    """Tests for __init__ and model injection."""

    def test_default_attributes(self, mock_model: MagicMock) -> None:
        enc = DenseEncoder(_model=mock_model)
        assert enc.model_name == MODEL_NAME
        assert enc.batch_size == 64
        assert enc.embeddings is None
        assert enc.chunks is None

    def test_custom_model_name_and_batch_size(self, mock_model: MagicMock) -> None:
        enc = DenseEncoder(model_name="custom/model", batch_size=32, _model=mock_model)
        assert enc.model_name == "custom/model"
        assert enc.batch_size == 32

    def test_inject_model_skips_load(self) -> None:
        """When _model is provided, no SentenceTransformer import is attempted."""
        with patch("src.retrieval.dense.DenseEncoder._load_model") as mock_load:
            DenseEncoder(_model=MagicMock())
        mock_load.assert_not_called()

    def test_load_model_raises_import_error_when_missing(self) -> None:
        """_load_model propagates ImportError if sentence-transformers absent."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            enc = DenseEncoder.__new__(DenseEncoder)
            enc.model_name = MODEL_NAME
            enc.device = None
            with pytest.raises(ImportError, match="sentence-transformers"):
                enc._load_model()


class TestEncodeCorpus:
    """Tests for DenseEncoder.encode_corpus."""

    def test_returns_float32_array(
        self, mock_model: MagicMock, sample_chunks: List[InternalChunk]
    ) -> None:
        enc = DenseEncoder(_model=mock_model)
        matrix = enc.encode_corpus(sample_chunks, save_path=None)
        assert matrix.dtype == np.float32

    def test_shape_is_n_by_384(
        self, mock_model: MagicMock, sample_chunks: List[InternalChunk]
    ) -> None:
        enc = DenseEncoder(_model=mock_model)
        matrix = enc.encode_corpus(sample_chunks, save_path=None)
        assert matrix.shape == (len(sample_chunks), EMBEDDING_DIM)

    def test_rows_are_unit_vectors(
        self, mock_model: MagicMock, sample_chunks: List[InternalChunk]
    ) -> None:
        """Every row of the matrix must have L2 norm ≈ 1."""
        enc = DenseEncoder(_model=mock_model)
        matrix = enc.encode_corpus(sample_chunks, save_path=None)
        norms = np.linalg.norm(matrix, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(sample_chunks)), atol=1e-5)

    def test_embeddings_and_chunks_attached(
        self, mock_model: MagicMock, sample_chunks: List[InternalChunk]
    ) -> None:
        enc = DenseEncoder(_model=mock_model)
        enc.encode_corpus(sample_chunks, save_path=None)
        assert enc.embeddings is not None
        assert enc.chunks is not None
        assert len(enc.chunks) == len(sample_chunks)

    def test_encode_corpus_saves_npy_file(
        self,
        mock_model: MagicMock,
        sample_chunks: List[InternalChunk],
        tmp_path: Path,
    ) -> None:
        save_path = tmp_path / "sub" / "embeddings.npy"
        enc = DenseEncoder(_model=mock_model)
        enc.encode_corpus(sample_chunks, save_path=save_path)
        assert save_path.exists()
        loaded = np.load(save_path)
        assert loaded.shape == (len(sample_chunks), EMBEDDING_DIM)

    def test_encode_corpus_no_save_when_none(
        self,
        mock_model: MagicMock,
        sample_chunks: List[InternalChunk],
        tmp_path: Path,
    ) -> None:
        enc = DenseEncoder(_model=mock_model)
        enc.encode_corpus(sample_chunks, save_path=None)
        # Confirm default path was not written
        assert not (tmp_path / "embeddings.npy").exists()

    def test_empty_chunks_raises(self, mock_model: MagicMock) -> None:
        enc = DenseEncoder(_model=mock_model)
        with pytest.raises(ValueError, match="non-empty"):
            enc.encode_corpus([], save_path=None)

    def test_plain_strings_accepted(self, mock_model: MagicMock) -> None:
        texts = ["hello world", "dense retrieval rocks"]
        enc = DenseEncoder(_model=mock_model)
        matrix = enc.encode_corpus(texts, save_path=None)
        assert matrix.shape == (2, EMBEDDING_DIM)
        assert enc.chunks is not None

    def test_encode_calls_model_encode(
        self,
        mock_model: MagicMock,
        sample_chunks: List[InternalChunk],
    ) -> None:
        enc = DenseEncoder(_model=mock_model)
        enc.encode_corpus(sample_chunks, save_path=None)
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        texts_arg = call_args[0][0]
        assert len(texts_arg) == len(sample_chunks)


class TestL2Normalize:
    """Tests for DenseEncoder._l2_normalize static method."""

    def test_unit_norm_after_normalization(self) -> None:
        rng = np.random.default_rng(0)
        mat = rng.standard_normal((10, EMBEDDING_DIM)).astype(np.float32)
        normed = DenseEncoder._l2_normalize(mat)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, np.ones(10), atol=1e-5)

    def test_already_normalized_unchanged(self) -> None:
        rng = np.random.default_rng(1)
        mat = rng.standard_normal((5, EMBEDDING_DIM)).astype(np.float32)
        once = DenseEncoder._l2_normalize(mat)
        twice = DenseEncoder._l2_normalize(once)
        np.testing.assert_allclose(once, twice, atol=1e-5)

    def test_zero_row_stays_zero(self) -> None:
        mat = np.zeros((3, EMBEDDING_DIM), dtype=np.float32)
        normed = DenseEncoder._l2_normalize(mat)
        np.testing.assert_array_equal(normed, mat)

    def test_output_dtype_is_float32(self) -> None:
        mat = np.ones((2, EMBEDDING_DIM), dtype=np.float64)
        normed = DenseEncoder._l2_normalize(mat.astype(np.float32))
        assert normed.dtype == np.float32


class TestEncodeQuery:
    """Tests for DenseEncoder.encode_query."""

    def test_returns_1d_unit_vector(self, encoder: DenseEncoder) -> None:
        q_vec = encoder.encode_query("vLLM async engine")
        assert q_vec.ndim == 1
        assert q_vec.shape == (EMBEDDING_DIM,)
        assert abs(float(np.linalg.norm(q_vec)) - 1.0) < 1e-5

    def test_dtype_is_float32(self, encoder: DenseEncoder) -> None:
        q_vec = encoder.encode_query("some query")
        assert q_vec.dtype == np.float32


class TestGetScores:
    """Tests for DenseEncoder.get_scores."""

    def test_scores_shape_matches_corpus(self, encoder: DenseEncoder) -> None:
        scores = encoder.get_scores("test query")
        assert scores.shape == (4,)
        assert scores.dtype == np.float32

    def test_scores_in_valid_cosine_range(self, encoder: DenseEncoder) -> None:
        scores = encoder.get_scores("test query")
        assert float(scores.min()) >= -1.01
        assert float(scores.max()) <= 1.01

    def test_raises_before_encode(self, mock_model: MagicMock) -> None:
        enc = DenseEncoder(_model=mock_model)
        with pytest.raises(RuntimeError, match="encode_corpus"):
            enc.get_scores("query")

    def test_cosine_is_dot_product_of_unit_vectors(
        self, encoder: DenseEncoder
    ) -> None:
        """Verify cosine(q, d_i) == q · d_i when both are unit vectors."""
        q_vec = encoder.encode_query("some question")
        expected = encoder.embeddings @ q_vec  # type: ignore[operator]
        actual = encoder.get_scores("some question")
        np.testing.assert_allclose(actual, expected.astype(np.float32), atol=1e-5)


class TestSearch:
    """Tests for DenseEncoder.search."""

    def test_returns_k_results(self, encoder: DenseEncoder) -> None:
        results = encoder.search("test", k=2)
        assert len(results) == 2

    def test_results_sorted_descending(self, encoder: DenseEncoder) -> None:
        results = encoder.search("query", k=4)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_zero_returns_empty(self, encoder: DenseEncoder) -> None:
        assert encoder.search("query", k=0) == []

    def test_k_negative_returns_empty(self, encoder: DenseEncoder) -> None:
        assert encoder.search("query", k=-1) == []

    def test_k_larger_than_corpus(self, encoder: DenseEncoder) -> None:
        results = encoder.search("query", k=100)
        assert len(results) == 4  # corpus has 4 chunks

    def test_result_indices_in_valid_range(self, encoder: DenseEncoder) -> None:
        results = encoder.search("test", k=4)
        for idx, _ in results:
            assert 0 <= idx < 4

    def test_raises_before_encode(self, mock_model: MagicMock) -> None:
        enc = DenseEncoder(_model=mock_model)
        with pytest.raises(RuntimeError, match="encode_corpus"):
            enc.search("query")


class TestGetTopK:
    """Tests for DenseEncoder.get_top_k."""

    def test_returns_chunk_ids(
        self, encoder: DenseEncoder, sample_chunks: List[InternalChunk]
    ) -> None:
        top_k = encoder.get_top_k("vllm engine", k=3)
        assert len(top_k) == 3
        all_ids = {c.chunk_id for c in sample_chunks}
        for cid in top_k:
            assert cid in all_ids

    def test_k_zero_returns_empty(self, encoder: DenseEncoder) -> None:
        assert encoder.get_top_k("query", k=0) == []

    def test_fallback_without_chunks(self, mock_model: MagicMock) -> None:
        """When no chunk metadata, returns raw doc indices."""
        enc = DenseEncoder(_model=mock_model)
        # Manually inject an embedding matrix without chunk metadata
        enc.embeddings = DenseEncoder._l2_normalize(
            np.random.default_rng(99).standard_normal((3, EMBEDDING_DIM)).astype(np.float32)
        )
        enc.chunks = None
        result = enc.get_top_k("test", k=2)
        assert len(result) == 2
        for idx in result:
            assert 0 <= idx < 3


class TestSearchSources:
    """Tests for DenseEncoder.search_sources."""

    def test_returns_minimal_source_objects(self, encoder: DenseEncoder) -> None:
        sources = encoder.search_sources("async llm engine", k=2)
        assert len(sources) == 2
        for src in sources:
            assert isinstance(src, MinimalSource)

    def test_source_file_paths_are_valid(
        self, encoder: DenseEncoder, sample_chunks: List[InternalChunk]
    ) -> None:
        sources = encoder.search_sources("function", k=4)
        valid_paths = {c.file_path for c in sample_chunks}
        for src in sources:
            assert src.file_path in valid_paths

    def test_raises_without_chunks(self, mock_model: MagicMock) -> None:
        enc = DenseEncoder(_model=mock_model)
        enc.embeddings = DenseEncoder._l2_normalize(
            np.ones((2, EMBEDDING_DIM), dtype=np.float32)
        )
        enc.chunks = None
        with pytest.raises(ValueError, match="chunk metadata"):
            enc.search_sources("query")


class TestPersistence:
    """Tests for DenseEncoder.save and DenseEncoder.load."""

    def test_save_creates_npy_file(
        self,
        encoder: DenseEncoder,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "emb.npy"
        saved = encoder.save(path)
        assert saved.exists()
        assert saved == path

    def test_save_before_encode_raises(self, mock_model: MagicMock) -> None:
        enc = DenseEncoder(_model=mock_model)
        with pytest.raises(RuntimeError, match="encode_corpus"):
            enc.save(Path("/tmp/should_not_exist.npy"))

    def test_load_roundtrip(
        self,
        encoder: DenseEncoder,
        sample_chunks: List[InternalChunk],
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "emb.npy"
        encoder.save(path)

        fresh_mock = _make_mock_model(seed=99)
        loaded_enc = DenseEncoder.load(
            embeddings_path=path,
            chunks=sample_chunks,
            _model=fresh_mock,
        )
        assert loaded_enc.embeddings is not None
        assert loaded_enc.embeddings.shape == encoder.embeddings.shape  # type: ignore[union-attr]
        np.testing.assert_array_equal(loaded_enc.embeddings, encoder.embeddings)
        assert loaded_enc.chunks is not None
        assert len(loaded_enc.chunks) == len(sample_chunks)

    def test_load_missing_file_raises(self, mock_model: MagicMock, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            DenseEncoder.load(
                embeddings_path=tmp_path / "nonexistent.npy",
                _model=mock_model,
            )

    def test_load_wrong_dim_raises(self, mock_model: MagicMock, tmp_path: Path) -> None:
        bad_matrix = np.ones((5, 128), dtype=np.float32)
        path = tmp_path / "bad.npy"
        np.save(path, bad_matrix)
        with pytest.raises(ValueError, match="384"):
            DenseEncoder.load(embeddings_path=path, _model=mock_model)

    def test_load_chunk_count_mismatch_raises(
        self,
        encoder: DenseEncoder,
        sample_chunks: List[InternalChunk],
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "emb.npy"
        encoder.save(path)
        too_few_chunks = sample_chunks[:2]  # only 2 of 4
        with pytest.raises(ValueError, match="does not match"):
            DenseEncoder.load(
                embeddings_path=path,
                chunks=too_few_chunks,
                _model=_make_mock_model(),
            )

    def test_default_embeddings_path_constant(self) -> None:
        assert str(DEFAULT_EMBEDDINGS_PATH) == "data/processed/embeddings.npy"


class TestConstants:
    """Sanity-check public constants."""

    def test_embedding_dim_is_384(self) -> None:
        assert EMBEDDING_DIM == 384

    def test_model_name(self) -> None:
        assert MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"


class TestConvenienceFunctions:
    """Tests for build_dense_index and load_dense_index."""

    def test_build_dense_index(
        self,
        sample_chunks: List[InternalChunk],
        tmp_path: Path,
    ) -> None:
        mock = _make_mock_model()
        with patch("src.retrieval.dense.DenseEncoder._load_model", return_value=mock):
            enc = build_dense_index(sample_chunks, save_path=tmp_path / "e.npy")
        assert enc.embeddings is not None
        assert (tmp_path / "e.npy").exists()

    def test_load_dense_index(
        self,
        encoder: DenseEncoder,
        sample_chunks: List[InternalChunk],
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "emb.npy"
        encoder.save(path)
        fresh_mock = _make_mock_model(seed=7)
        loaded = load_dense_index(
            embeddings_path=path,
            chunks=sample_chunks,
            _model=fresh_mock,
        )
        assert loaded.embeddings is not None
        assert loaded.embeddings.shape == (len(sample_chunks), EMBEDDING_DIM)


class TestPerformanceSLA:
    """Validates sub-50 ms per-query retrieval latency acceptance criterion.

    Uses a synthetic 50 k-chunk corpus to stress-test the vectorized
    cosine similarity path.  The corpus embeddings are pre-generated without
    model inference, isolating the retrieval kernel latency only.
    """

    def test_query_latency_under_50ms(self) -> None:
        """get_scores + argpartition top-k must complete < 50 ms (mean over 50 trials)."""
        n_docs = 50_000
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((n_docs, EMBEDDING_DIM)).astype(np.float32)
        embeddings = DenseEncoder._l2_normalize(raw)

        mock = _make_mock_model()
        enc = DenseEncoder(_model=mock)
        enc.embeddings = embeddings
        enc.chunks = None  # not needed for latency test

        n_trials = 50
        latencies: list[float] = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            _ = enc.search("vLLM async engine attention prefill", k=5)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        mean_ms = float(np.mean(latencies)) * 1000.0
        p95_ms = float(np.percentile(latencies, 95)) * 1000.0

        assert mean_ms < 50.0, (
            f"Mean query latency {mean_ms:.2f} ms exceeds 50 ms SLA"
        )
        assert p95_ms < 100.0, (
            f"P95 query latency {p95_ms:.2f} ms exceeds 100 ms threshold"
        )

    def test_encode_corpus_shape_and_norm_at_scale(self) -> None:
        """encode_corpus must return (N, 384) normalized matrix for large N."""
        n = 1000
        rng = np.random.default_rng(7)
        # Mock model that returns correctly-shaped random embeddings
        mock = MagicMock()
        mock.encode.return_value = rng.standard_normal(
            (n, EMBEDDING_DIM)
        ).astype(np.float32)
        chunks = [f"chunk content {i}" for i in range(n)]
        enc = DenseEncoder(_model=mock)
        matrix = enc.encode_corpus(chunks, save_path=None)
        assert matrix.shape == (n, EMBEDDING_DIM)
        norms = np.linalg.norm(matrix, axis=1)
        np.testing.assert_allclose(norms, np.ones(n), atol=1e-5)
