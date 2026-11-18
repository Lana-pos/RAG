"""Unit tests for Reciprocal Rank Fusion (RRF) Hybrid Ranker (src/retrieval/hybrid.py)."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import numpy as np
from numpy.typing import NDArray
import pytest

from src.indexing.schema import InternalChunk
from src.models import MinimalSource
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import EMBEDDING_DIM, DenseEncoder
from src.retrieval.hybrid import (
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_RRF_K,
    RRF_K,
    HybridRanker,
    clamp_source,
    deduplicate_sources,
    fuse_sources,
    reciprocal_rank_fusion,
    rrf_score,
    to_minimal_source,
)


@pytest.fixture
def sample_chunks() -> List[InternalChunk]:
    """Sample InternalChunk objects representing code and documentation."""
    return [
        InternalChunk(
            chunk_id=0,
            file_path="data/raw/vllm-0.10.1/vllm/engine/async_llm_engine.py",
            first_character_index=100,
            last_character_index=450,
            chunk_type="python_class",
            token_count=12,
            content="class AsyncLLMEngine: asynchronous inference engine for vLLM",
        ),
        InternalChunk(
            chunk_id=1,
            file_path="data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
            first_character_index=20,
            last_character_index=380,
            chunk_type="python_func",
            token_count=10,
            content="def run_openai_server(): launch openai compatible http server",
        ),
        InternalChunk(
            chunk_id=2,
            file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
            first_character_index=0,
            last_character_index=600,
            chunk_type="markdown_section",
            token_count=11,
            content="# LoRA Adapters\nHow to configure dynamic LoRA adapters in vLLM",
        ),
        InternalChunk(
            chunk_id=3,
            file_path="data/raw/vllm-0.10.1/vllm/core/block_manager.py",
            first_character_index=500,
            last_character_index=1200,
            chunk_type="python_class",
            token_count=14,
            content="class BlockSpaceManager: manages PagedAttention KV cache blocks",
        ),
    ]


def _make_deterministic_dense_encoder(
    chunks: List[InternalChunk],
    corpus_vectors: NDArray[np.float32],
    query_vector: NDArray[np.float32],
) -> DenseEncoder:
    """Create a DenseEncoder with controlled corpus and query embeddings."""
    mock_model = MagicMock()

    def _encode(sentences: List[str], **kwargs: Any) -> NDArray[np.float32]:
        if len(sentences) == len(chunks):
            return corpus_vectors.astype(np.float32)
        return np.tile(query_vector, (len(sentences), 1)).astype(np.float32)

    mock_model.encode.side_effect = _encode
    encoder = DenseEncoder(_model=mock_model)
    encoder.encode_corpus(chunks, save_path=None)
    return encoder


class TestRRFFormula:
    """Verify exact adherence to RRF(d) = sum_{m in {bm25, dense}} 1 / (60 + Rank_m(d))."""

    def test_default_rrf_k_constant_is_60(self) -> None:
        assert DEFAULT_RRF_K == 60
        assert RRF_K == 60

    def test_rrf_score_single_rank(self) -> None:
        assert pytest.approx(rrf_score(1, k=60)) == 1.0 / 61.0
        assert pytest.approx(rrf_score(2, k=60)) == 1.0 / 62.0
        assert pytest.approx(rrf_score(10, k=60)) == 1.0 / 70.0
        assert pytest.approx(rrf_score(1, k=60, weight=2.0)) == 2.0 / 61.0

    def test_rrf_score_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="rank must be a positive"):
            rrf_score(0, k=60)
        with pytest.raises(ValueError, match="rank must be a positive"):
            rrf_score(-3, k=60)
        with pytest.raises(ValueError, match="k must be non-negative"):
            rrf_score(1, k=-1)

    def test_exact_rrf_fusion_scores_with_scored_tuples(self) -> None:
        # BM25 ranking: doc 0 (rank 1), doc 1 (rank 2), doc 2 (rank 3)
        bm25_results = [(0, 12.5), (1, 8.3), (2, 4.1)]
        # Dense ranking: doc 2 (rank 1), doc 1 (rank 2), doc 3 (rank 3)
        dense_results = [(2, 0.92), (1, 0.85), (3, 0.71)]

        fused = reciprocal_rank_fusion(bm25_results, dense_results, k=60)
        fused_dict = dict(fused)

        # Expected RRF scores:
        # doc 1: 1/(60+2) + 1/(60+2) = 1/62 + 1/62 = 2/62
        # doc 2: 1/(60+3) + 1/(60+1) = 1/63 + 1/61 = 124/3843
        # doc 0: 1/(60+1) = 1/61
        # doc 3: 1/(60+3) = 1/63
        expected_doc1 = (1.0 / 62.0) + (1.0 / 62.0)
        expected_doc2 = (1.0 / 63.0) + (1.0 / 61.0)
        expected_doc0 = 1.0 / 61.0
        expected_doc3 = 1.0 / 63.0

        assert pytest.approx(fused_dict[1], rel=1e-7) == expected_doc1
        assert pytest.approx(fused_dict[2], rel=1e-7) == expected_doc2
        assert pytest.approx(fused_dict[0], rel=1e-7) == expected_doc0
        assert pytest.approx(fused_dict[3], rel=1e-7) == expected_doc3

        # Verify order:
        # 2/62 (0.03225806) vs 1/63 + 1/61 (0.03226646) -> doc 2 > doc 1 > doc 0 > doc 3
        ranked_ids = [doc_id for doc_id, _ in fused]
        assert ranked_ids == [2, 1, 0, 3]

    def test_consensus_candidate_beats_single_modality_leader(self) -> None:
        # Doc 10 is rank 2 in both BM25 and Dense: 1/62 + 1/62 = 0.032258
        # Doc 1 is rank 1 in BM25 only: 1/61 = 0.016393
        # Doc 2 is rank 1 in Dense only: 1/61 = 0.016393
        bm25_ids = [1, 10, 3]
        dense_ids = [2, 10, 4]

        fused = reciprocal_rank_fusion(bm25_ids, dense_ids, k=60)
        assert fused[0][0] == 10
        assert pytest.approx(fused[0][1]) == 2.0 / 62.0

    def test_supports_sequence_of_rankings_invocation(self) -> None:
        r1 = [0, 1]
        r2 = [1, 2]
        fused_two_args = reciprocal_rank_fusion(r1, r2, k=60)
        fused_seq_arg = reciprocal_rank_fusion([r1, r2], k=60)
        assert fused_two_args == fused_seq_arg

    def test_top_k_slicing_and_negative_k(self) -> None:
        bm25_ids = [0, 1, 2, 3]
        dense_ids = [3, 2, 1, 0]
        top_2 = reciprocal_rank_fusion(bm25_ids, dense_ids, k=60, top_k=2)
        assert len(top_2) == 2
        assert reciprocal_rank_fusion(bm25_ids, dense_ids, k=60, top_k=0) == []
        assert reciprocal_rank_fusion(bm25_ids, dense_ids, k=60, top_k=-5) == []

        with pytest.raises(ValueError, match="non-negative"):
            reciprocal_rank_fusion(bm25_ids, dense_ids, k=-1)


class TestDeduplicationAndMinimalSources:
    """Verify deduplication and character span accuracy of returned MinimalSource objects."""

    def test_deduplicates_identical_sources_across_bm25_and_dense(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        # Both BM25 and Dense return overlapping chunks
        bm25_ranked = [sample_chunks[0], sample_chunks[1], sample_chunks[2]]
        dense_ranked = [sample_chunks[2], sample_chunks[0], sample_chunks[3]]

        sources = fuse_sources(bm25_ranked, dense_ranked, rrf_k=60)
        assert len(sources) == 4
        assert all(isinstance(s, MinimalSource) for s in sources)

        # Ensure all returned MinimalSource objects are unique
        assert len(set(sources)) == len(sources)

        # Verify rank order:
        # chunk 0: rank 1 in BM25 (1/61) + rank 2 in Dense (1/62) = 0.032522
        # chunk 2: rank 3 in BM25 (1/63) + rank 1 in Dense (1/61) = 0.032266
        # chunk 1: rank 2 in BM25 (1/62) = 0.016129
        # chunk 3: rank 3 in Dense (1/63) = 0.015873
        assert sources[0] == sample_chunks[0].to_source()
        assert sources[1] == sample_chunks[2].to_source()
        assert sources[2] == sample_chunks[1].to_source()
        assert sources[3] == sample_chunks[3].to_source()

    def test_deduplicates_duplicate_entries_within_same_modality(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        src_0 = sample_chunks[0].to_source()
        src_1 = sample_chunks[1].to_source()

        # Duplicate src_0 inside bm25 list
        bm25_sources = [src_0, src_0, src_1]
        dense_sources = [src_1]

        fused = reciprocal_rank_fusion(bm25_sources, dense_sources, k=60)
        scores = dict(fused)

        # src_0 should only receive rank 1 from BM25 (1/61), NOT 1/61 + 1/62
        assert pytest.approx(scores[src_0]) == 1.0 / 61.0
        # src_1 is at index 2 (rank 3) in BM25 (1/63) + rank 1 in Dense (1/61)
        assert pytest.approx(scores[src_1]) == (1.0 / 63.0) + (1.0 / 61.0)

    def test_deduplicates_different_chunks_with_identical_character_spans(self) -> None:
        chunk_a = InternalChunk(
            chunk_id=10,
            file_path="./data/raw/vllm-0.10.1/vllm/model.py",
            first_character_index=50,
            last_character_index=250,
            chunk_type="python_func",
            token_count=8,
            content="def forward(self): pass",
        )
        chunk_b_duplicate_span = InternalChunk(
            chunk_id=99,
            file_path="data/raw/vllm-0.10.1/vllm/model.py",
            first_character_index=50,
            last_character_index=250,
            chunk_type="python_func",
            token_count=8,
            content="def forward(self): pass",
        )

        sources = fuse_sources([chunk_a], [chunk_b_duplicate_span], rrf_k=60)
        assert len(sources) == 1
        assert sources[0].file_path == "data/raw/vllm-0.10.1/vllm/model.py"
        assert sources[0].first_character_index == 50
        assert sources[0].last_character_index == 250

    def test_clamps_oversized_character_spans_to_max_chunk_size(self) -> None:
        oversized = MinimalSource(
            file_path="data/raw/vllm-0.10.1/docs/large.md",
            first_character_index=1000,
            last_character_index=4500,  # span_length = 3500 > 2000
        )
        clamped = clamp_source(oversized, max_chunk_size=2000)
        assert clamped.file_path == "data/raw/vllm-0.10.1/docs/large.md"
        assert clamped.first_character_index == 1000
        assert clamped.last_character_index == 3000
        assert clamped.span_length == DEFAULT_MAX_CHUNK_SIZE

    def test_deduplicate_sources_helper(self) -> None:
        s1 = MinimalSource(
            file_path="data/raw/vllm-0.10.1/a.py",
            first_character_index=0,
            last_character_index=100,
        )
        s1_dup = MinimalSource(
            file_path="./data/raw/vllm-0.10.1/a.py",
            first_character_index=0,
            last_character_index=100,
        )
        s2 = MinimalSource(
            file_path="data/raw/vllm-0.10.1/b.py",
            first_character_index=10,
            last_character_index=200,
        )
        deduped = deduplicate_sources([s1, s1_dup, s2], k=5)
        assert deduped == [s1, s2]
        assert deduplicate_sources([s1, s2], k=1) == [s1]
        assert deduplicate_sources([s1, s2], k=0) == []


class TestHybridRankerIntegration:
    """Integration tests combining BM25Index and DenseEncoder inside HybridRanker."""

    def test_hybrid_ranker_combines_bm25_and_dense(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        bm25 = BM25Index.build_inverted_index(sample_chunks)

        # Construct embeddings so that chunk 2 ("LoRA Adapters") is #1 in Dense,
        # while query "openai lora" matches chunk 1 ("openai") and chunk 2 ("lora") in BM25
        corpus_vecs = np.eye(len(sample_chunks), EMBEDDING_DIM, dtype=np.float32)
        # Query vector aligns strongest with chunk 2, then chunk 1, then chunk 0, then chunk 3
        query_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        query_vec[2] = 0.9
        query_vec[1] = 0.4
        query_vec[0] = 0.1

        dense = _make_deterministic_dense_encoder(sample_chunks, corpus_vecs, query_vec)
        ranker = HybridRanker(bm25_index=bm25, dense_encoder=dense, rrf_k=60)

        assert ranker.k == 60
        assert ranker.rrf_k == 60

        results = ranker.search("lora", k=3)
        assert len(results) == 3
        assert all(isinstance(src, MinimalSource) for src in results)

        # Chunk 2 is #1 in both BM25 (only doc containing "lora") and #1 in Dense
        assert results[0].file_path == "data/raw/vllm-0.10.1/docs/features/lora.md"
        assert results[0].first_character_index == 0
        assert results[0].last_character_index == 600

        # Verify callable interface and search_sources alias return identical results
        assert ranker("lora", k=3) == results
        assert ranker.search_sources("lora", k=3) == results

    def test_search_with_scores_and_get_top_k(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        bm25 = BM25Index.build_inverted_index(sample_chunks)
        corpus_vecs = np.eye(len(sample_chunks), EMBEDDING_DIM, dtype=np.float32)
        query_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        query_vec[0] = 0.95
        query_vec[3] = 0.30
        dense = _make_deterministic_dense_encoder(sample_chunks, corpus_vecs, query_vec)

        ranker = HybridRanker(bm25_index=bm25, dense_encoder=dense)

        scored = ranker.search_with_scores("AsyncLLMEngine", k=2)
        assert len(scored) == 2
        top_doc_idx, top_rrf_score = scored[0]
        assert top_doc_idx == 0
        # Rank 1 in BM25 (1/61) + Rank 1 in Dense (1/61) = 2/61
        assert pytest.approx(top_rrf_score) == 2.0 / 61.0

        top_ids = ranker.get_top_k("AsyncLLMEngine", k=2)
        assert top_ids[0] == 0

        all_scores = ranker.get_scores("AsyncLLMEngine")
        assert 0 in all_scores
        assert pytest.approx(all_scores[0]) == 2.0 / 61.0

    def test_direct_fuse_and_rank_methods(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        ranker = HybridRanker(chunks=sample_chunks, k=60)
        bm25_res = [(0, 5.0), (1, 3.0)]
        dense_res = [(1, 0.9), (2, 0.8)]

        fused_sources = ranker.fuse(bm25_res, dense_res, k=2)
        assert len(fused_sources) == 2
        assert fused_sources[0] == sample_chunks[1].to_source()
        assert fused_sources[1] == sample_chunks[0].to_source()

        assert ranker.rank(bm25_res, dense_res, k=2) == fused_sources

        fused_tuples = ranker.fuse_rankings(bm25_res, dense_res, top_k=2)
        assert fused_tuples[0][0] == 1
        assert pytest.approx(fused_tuples[0][1]) == (1.0 / 62.0) + (1.0 / 61.0)


class TestProgressiveDegradationAndEdgeCases:
    """Verify graceful degradation and defensive input handling."""

    def test_empty_and_whitespace_queries_return_empty_list(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        bm25 = BM25Index.build_inverted_index(sample_chunks)
        ranker = HybridRanker(bm25_index=bm25)

        assert ranker.search("", k=5) == []
        assert ranker.search("   \t\n  ", k=5) == []
        assert ranker.search_with_scores("", k=5) == []
        assert ranker.get_top_k("", k=5) == []
        assert ranker.get_scores("") == {}

    def test_non_positive_k_returns_empty_list(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        bm25 = BM25Index.build_inverted_index(sample_chunks)
        ranker = HybridRanker(bm25_index=bm25)

        assert ranker.search("vLLM", k=0) == []
        assert ranker.search("vLLM", k=-3) == []
        assert ranker.search_with_scores("vLLM", k=0) == []

    def test_degrades_gracefully_when_dense_encoder_is_none(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        bm25 = BM25Index.build_inverted_index(sample_chunks)
        ranker = HybridRanker(bm25_index=bm25, dense_encoder=None)

        sources = ranker.search("openai server", k=2)
        assert len(sources) == 1
        assert sources[0] == sample_chunks[1].to_source()

    def test_degrades_gracefully_when_bm25_is_none(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        corpus_vecs = np.eye(len(sample_chunks), EMBEDDING_DIM, dtype=np.float32)
        query_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        query_vec[3] = 1.0
        dense = _make_deterministic_dense_encoder(sample_chunks, corpus_vecs, query_vec)

        ranker = HybridRanker(bm25_index=None, dense_encoder=dense)
        sources = ranker.search("kv cache block manager", k=2)
        assert len(sources) == 2
        assert sources[0] == sample_chunks[3].to_source()

    def test_raises_runtime_error_when_neither_retriever_provided(self) -> None:
        ranker = HybridRanker()
        with pytest.raises(RuntimeError, match="at least one retriever"):
            ranker.search("query")

    def test_raises_value_error_when_chunk_metadata_missing(self) -> None:
        bm25 = BM25Index([["async", "engine"], ["openai", "server"]])
        ranker = HybridRanker(bm25_index=bm25)
        with pytest.raises(ValueError, match="chunk metadata"):
            ranker.search("async engine", k=2)

    def test_to_minimal_source_error_cases(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        with pytest.raises(ValueError, match="chunk metadata"):
            to_minimal_source(0, chunks=None)
        with pytest.raises(IndexError, match="out of bounds"):
            to_minimal_source(99, chunks=sample_chunks)
        with pytest.raises(ValueError, match="chunk metadata"):
            to_minimal_source("invalid_string_item")
