"""Unit tests for Vectorized BM25Okapi Inverted Index (src/retrieval/bm25.py)."""

from pathlib import Path
import time
import numpy as np
import pytest

from src.indexing.schema import InternalChunk
from src.models import MinimalSource
from src.retrieval.bm25 import (
    BM25Index,
    DEFAULT_INDEX_PATH,
    PostingArray,
    PureNumPyCSRMatrix,
    load_bm25_index,
    save_bm25_index,
)


@pytest.fixture
def sample_corpus() -> list[list[str]]:
    """Sample tokenized corpus for testing BM25 retrieval."""
    return [
        ["async", "engine", "args", "asyncengineargs"],
        ["openai", "compatible", "server", "engine"],
        ["token", "tokenizer", "subword", "model"],
        ["engine", "model", "cache", "kv"],
    ]


@pytest.fixture
def sample_chunks() -> list[InternalChunk]:
    """Sample InternalChunk objects for testing."""
    return [
        InternalChunk(
            chunk_id=0,
            file_path="data/raw/vllm-0.10.1/vllm/engine/async_llm_engine.py",
            first_character_index=100,
            last_character_index=350,
            chunk_type="python_class",
            token_count=4,
            content="class AsyncLLMEngine: ...",
        ),
        InternalChunk(
            chunk_id=1,
            file_path="data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
            first_character_index=20,
            last_character_index=400,
            chunk_type="python_func",
            token_count=4,
            content="def run_openai_server(): ...",
        ),
        InternalChunk(
            chunk_id=2,
            file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
            first_character_index=0,
            last_character_index=500,
            chunk_type="markdown_section",
            token_count=4,
            content="# LoRA Adapters Documentation",
        ),
    ]


class TestBM25IndexBuild:
    """Tests for index construction and internal data structures."""

    def test_empty_corpus(self) -> None:
        index = BM25Index([])
        assert index.num_docs == 0
        assert index.avgdl == 0.0
        assert len(index.doc_lengths) == 0
        assert len(index.inverted_index) == 0
        assert index.get_scores("engine").shape == (0,)
        assert index.search("engine") == []
        assert index.get_top_k("engine") == []

    def test_builds_expected_attributes(self, sample_corpus: list[list[str]]) -> None:
        index = BM25Index(sample_corpus, k1=1.5, b=0.75)
        assert index.num_docs == 4
        assert np.array_equal(index.doc_lengths, np.array([4, 4, 4, 4]))
        assert index.avgdl == 4.0
        assert "engine" in index.vocabulary
        assert "engine" in index.inverted_index

        # Check postings for "engine" (occurs in doc 0, 1, 3)
        posting = index.inverted_index["engine"]
        assert isinstance(posting, np.ndarray)
        assert isinstance(posting, PostingArray)
        assert list(posting.doc_ids) == [0, 1, 3]
        assert list(posting.term_frequencies) == [1.0, 1.0, 1.0]

        # Check postings accessor
        postings_dict = index.postings
        assert "engine" in postings_dict
        assert list(postings_dict["engine"][0]) == [0, 1, 3]

        get_p = index.get_postings("engine")
        assert get_p is not None
        assert list(get_p[0]) == [0, 1, 3]
        assert index.get_postings("nonexistent") is None

    def test_term_frequency_matrix_dense(
        self, sample_corpus: list[list[str]]
    ) -> None:
        index = BM25Index(sample_corpus, store_dense=True)
        tf_mat = index.term_frequency_matrix
        assert isinstance(tf_mat, np.ndarray)
        assert tf_mat.shape == (4, len(index.vocabulary))

        # Check engine occurrences
        engine_col = index.vocabulary["engine"]
        assert tf_mat[0, engine_col] == 1
        assert tf_mat[1, engine_col] == 1
        assert tf_mat[2, engine_col] == 0
        assert tf_mat[3, engine_col] == 1

        assert index.get_term_frequency_matrix(dense=True).shape == tf_mat.shape

    def test_term_frequency_matrix_sparse(
        self, sample_corpus: list[list[str]]
    ) -> None:
        index = BM25Index(sample_corpus, store_dense=False)
        tf_mat = index.term_frequency_matrix
        assert tf_mat.shape == (4, len(index.vocabulary))
        dense_converted = index.get_term_frequency_matrix(dense=True)
        assert isinstance(dense_converted, np.ndarray)
        assert dense_converted.shape == (4, len(index.vocabulary))

        engine_col = index.vocabulary["engine"]
        assert dense_converted[0, engine_col] == 1
        assert dense_converted[2, engine_col] == 0


class TestBM25ScoringAndRanking:
    """Tests for BM25Okapi formula execution and rank ordering."""

    def test_exact_formula_calculation(self) -> None:
        # Two documents:
        # doc 0: ["cat", "dog"] -> len 2
        # doc 1: ["cat", "cat", "fish"] -> len 3
        # avgdl = 2.5
        # k1 = 1.5, b = 0.75
        corpus = [["cat", "dog"], ["cat", "cat", "fish"]]
        index = BM25Index(corpus, k1=1.5, b=0.75)

        # N = 2, df(cat) = 2
        # idf(cat) = ln(1 + (2 - 2 + 0.5) / (2 + 0.5)) = ln(1 + 0.5 / 2.5) = ln(1.2)
        import math
        expected_idf_cat = math.log(1.0 + (2.0 - 2.0 + 0.5) / (2.0 + 0.5))
        assert pytest.approx(index.idf["cat"], rel=1e-5) == expected_idf_cat

        # doc 0: dl=2, len_ratio = 2 / 2.5 = 0.8
        # K_0 = 1.5 * (1 - 0.75 + 0.75 * 0.8) = 1.5 * (0.25 + 0.6) = 1.5 * 0.85 = 1.275
        # tf = 1: term_score = idf * (1 * 2.5) / (1 + 1.275) = idf * 2.5 / 2.275
        expected_score_doc0 = expected_idf_cat * 2.5 / 2.275

        # doc 1: dl=3, len_ratio = 3 / 2.5 = 1.2
        # K_1 = 1.5 * (1 - 0.75 + 0.75 * 1.2) = 1.5 * (0.25 + 0.9) = 1.5 * 1.15 = 1.725
        # tf = 2: term_score = idf * (2 * 2.5) / (2 + 1.725) = idf * 5.0 / 3.725
        expected_score_doc1 = expected_idf_cat * 5.0 / 3.725

        scores = index.get_scores(["cat"])
        assert pytest.approx(scores[0], rel=1e-5) == expected_score_doc0
        assert pytest.approx(scores[1], rel=1e-5) == expected_score_doc1

    def test_ranking_relevance(self, sample_corpus: list[list[str]]) -> None:
        index = BM25Index(sample_corpus)
        # "async engine args" matches doc 0 with 3 matching terms, and doc 1, 3 with 1 term
        results = index.search("async engine args", k=5)
        assert len(results) > 0
        top_doc_idx, top_score = results[0]
        assert top_doc_idx == 0  # Doc 0 is by far the best match
        assert top_score > 0

    def test_document_length_normalization(self) -> None:
        # doc 0 has 2 words (short), doc 1 has 10 words (long).
        # Both contain "query_term" once.
        corpus = [
            ["query_term", "noise"],
            ["query_term"] + ["noise"] * 9,
        ]
        index = BM25Index(corpus, k1=1.5, b=0.75)
        scores = index.get_scores("query_term")
        # Short document should have a higher score due to length normalization penalty on long doc
        assert scores[0] > scores[1]

    def test_query_term_weighting(self) -> None:
        corpus = [
            ["foo", "bar"],
            ["foo", "baz"],
        ]
        index = BM25Index(corpus)
        # Repeated token in query should boost the score
        score_single = index.get_scores(["foo"])[0]
        score_double = index.get_scores(["foo", "foo"])[0]
        assert pytest.approx(score_double, rel=1e-4) == score_single * 2.0


class TestBM25FromChunksAndSources:
    """Tests for integration with InternalChunk and MinimalSource."""

    def test_build_inverted_index_from_internal_chunks(
        self, sample_chunks: list[InternalChunk]
    ) -> None:
        index = BM25Index.build_inverted_index(sample_chunks)
        assert index.num_docs == 3
        assert index.chunks is not None
        assert index.chunk_ids == [0, 1, 2]

        # Search for "openai server"
        top_k = index.get_top_k("openai server", k=1)
        assert top_k == [1]

        # Search sources returns MinimalSource
        sources = index.search_sources("openai server", k=1)
        assert len(sources) == 1
        assert isinstance(sources[0], MinimalSource)
        assert sources[0].file_path == (
            "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py"
        )
        assert sources[0].first_character_index == 20
        assert sources[0].last_character_index == 400


class TestBM25DegenerateAndEdgeCases:
    """Tests for defensive handling of degenerate inputs."""

    def test_empty_query(self, sample_corpus: list[list[str]]) -> None:
        index = BM25Index(sample_corpus)
        scores = index.get_scores("")
        assert np.all(scores == 0)
        assert index.search("") == []
        assert index.get_top_k("") == []

    def test_unseen_query_terms(self, sample_corpus: list[list[str]]) -> None:
        index = BM25Index(sample_corpus)
        scores = index.get_scores("completely_unseen_term_12345")
        assert np.all(scores == 0)
        assert index.search("completely_unseen_term_12345", filter_zero=True) == []
        # filter_zero=False returns entries with 0.0
        results_unfiltered = index.search(
            "completely_unseen_term_12345", k=2, filter_zero=False
        )
        assert len(results_unfiltered) == 2
        assert results_unfiltered[0][1] == 0.0

    def test_k_bounds(self, sample_corpus: list[list[str]]) -> None:
        index = BM25Index(sample_corpus)
        assert index.search("engine", k=0) == []
        assert index.search("engine", k=-5) == []
        assert index.get_top_k("engine", k=0) == []

        # k larger than matching documents
        results = index.search("engine", k=100)
        assert len(results) == 3  # exactly 3 docs contain "engine"

    def test_documents_with_zero_length(self) -> None:
        corpus = [[], ["hello", "world"]]
        index = BM25Index(corpus)
        assert index.num_docs == 2
        assert index.doc_lengths[0] == 0
        scores = index.get_scores("hello")
        assert scores[0] == 0.0
        assert scores[1] > 0.0


class TestBM25Serialization:
    """Tests for saving and loading the binary BM25 index."""

    def test_save_and_load_roundtrip(
        self, sample_corpus: list[list[str]], tmp_path: Path
    ) -> None:
        index_file = tmp_path / "index_dir" / "test_index.pkl"
        orig_index = BM25Index(sample_corpus)
        saved_path = orig_index.save(index_file)
        assert saved_path.exists()

        loaded_index = BM25Index.load(index_file)
        assert loaded_index.num_docs == orig_index.num_docs
        assert loaded_index.avgdl == orig_index.avgdl
        assert np.array_equal(loaded_index.doc_lengths, orig_index.doc_lengths)
        assert loaded_index.vocabulary == orig_index.vocabulary

        # Verify query scoring produces identical outputs
        query = "async engine"
        orig_scores = orig_index.get_scores(query)
        loaded_scores = loaded_index.get_scores(query)
        assert np.allclose(orig_scores, loaded_scores)

    def test_helper_functions_save_load(
        self, sample_corpus: list[list[str]], tmp_path: Path
    ) -> None:
        index_file = tmp_path / "helper_index.pkl"
        orig = BM25Index(sample_corpus)
        save_bm25_index(orig, index_file)
        loaded = load_bm25_index(index_file)
        assert loaded.num_docs == 4

    def test_load_non_existent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            BM25Index.load(tmp_path / "non_existent.pkl")

    def test_default_index_path_constant(self) -> None:
        assert str(DEFAULT_INDEX_PATH) == "data/processed/bm25_index.pkl"


class TestPureNumPyCSRMatrix:
    """Tests for fallback PureNumPyCSRMatrix functionality."""

    def test_csr_matrix_operations(self) -> None:
        # 3x4 matrix:
        # row 0: col 1 -> 5.0
        # row 1: col 0 -> 2.0, col 3 -> 7.0
        # row 2: empty
        data = [5.0, 2.0, 7.0]
        indices = [1, 0, 3]
        indptr = [0, 1, 3, 3]
        shape = (3, 4)

        csr = PureNumPyCSRMatrix(data, indices, indptr, shape)
        assert csr.shape == shape
        assert csr.nnz == 3
        assert csr[0, 1] == 5.0
        assert csr[0, 0] == 0.0
        assert csr[1, 0] == 2.0
        assert csr[1, 3] == 7.0
        assert csr[2, 0] == 0.0

        dense = csr.toarray()
        assert dense.shape == (3, 4)
        assert dense[0, 1] == 5.0
        assert dense[1, 3] == 7.0
        assert dense[2, 2] == 0.0


class TestBM25PerformanceSLA:
    """Validates sub-millisecond per-query retrieval latency SLA."""

    def test_sub_millisecond_query_latency(self) -> None:
        # Synthetic corpus with 14,000 documents (~size of vLLM codebase)
        num_docs = 14000
        vocab_size = 5000

        # Deterministic synthetic corpus
        rng = np.random.default_rng(42)
        corpus = [
            [f"token_{tid}" for tid in rng.integers(0, vocab_size, size=50)]
            for _ in range(num_docs)
        ]

        index = BM25Index(corpus, store_dense=False)
        assert index.num_docs == 14000

        # Benchmark 50 realistic queries (e.g., 5 tokens each)
        queries = [
            [f"token_{tid}" for tid in rng.integers(0, vocab_size, size=5)]
            for _ in range(50)
        ]

        latencies: list[float] = []
        for q in queries:
            t0 = time.perf_counter()
            top_k = index.get_top_k(q, k=5)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            assert len(top_k) <= 5

        mean_latency_ms = float(np.mean(latencies)) * 1000.0
        p95_latency_ms = float(np.percentile(latencies, 95)) * 1000.0

        # Sub-millisecond acceptance criterion: mean < 1.0 ms
        assert mean_latency_ms < 1.0, f"Mean latency {mean_latency_ms:.3f} ms >= 1.0 ms"
        assert p95_latency_ms < 2.0, f"P95 latency {p95_latency_ms:.3f} ms >= 2.0 ms"
