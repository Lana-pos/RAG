"""Reciprocal Rank Fusion (RRF) hybrid ranker combining BM25 and dense retrieval.

Fuses lexical BM25Okapi rankings and semantic MiniLM cosine similarity rankings
using Reciprocal Rank Fusion (default k=60):
    RRF(d) = sum_{m in {bm25, dense}} 1 / (60 + Rank_m(d))

Returns deduplicated, ranked MinimalSource objects with validated relative file
paths and character spans strictly bounded within the character ceiling.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from src.indexing.schema import InternalChunk
from src.models import MinimalSource
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseEncoder

logger = logging.getLogger(__name__)

DEFAULT_RRF_K: int = 60
RRF_K: int = DEFAULT_RRF_K
DEFAULT_MAX_CHUNK_SIZE: int = 2000
DEFAULT_CANDIDATE_MULTIPLIER: int = 4
DEFAULT_MIN_CANDIDATES: int = 50


def rrf_score(
    rank: int,
    k: int = DEFAULT_RRF_K,
    weight: float = 1.0,
) -> float:
    """Compute Reciprocal Rank Fusion score contribution for a single 1-based rank.
    Calculates how much a search result contributes to the final ranking.
    A document gets a higher contribution when it appears near the top of a search result list.
    rank: 1-based rank position of the document in a retriever's ranking (rank >= 1).
    k: RRF smoothing constant (default: 60).
    weight: Optional modality weight multiplier (default: 1.0).
    """
    if rank < 1:
        raise ValueError(f"rank must be a positive 1-based integer (>= 1), got {rank}")
    if k < 0:
        raise ValueError(f"k must be non-negative (>= 0), got {k}")
    return float(weight) / float(k + rank)


def clamp_source(
    source: MinimalSource,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> MinimalSource:
    """Enforce hard character span ceiling on a MinimalSource.
    Guarantees that ``last_character_index - first_character_index <= max_chunk_size``
    so moulinette evaluation never rejects an oversized span.
    Checks whether a source's character range is too large.
    If it exceeds the maximum allowed size, it shortens the range.
    source: MinimalSource instance to validate and clamp.
    max_chunk_size: Maximum permitted character span length (default: 2000).
    """
    if max_chunk_size <= 0:
        raise ValueError(f"max_chunk_size must be positive, got {max_chunk_size}")

    if source.span_length <= max_chunk_size:
        return source

    return MinimalSource(
        file_path=source.file_path,
        first_character_index=source.first_character_index,
        last_character_index=source.first_character_index + max_chunk_size,
    )


def _is_scored_tuple(item: Any) -> bool:
    """Checks whether an item is a tuple
    containing a document and a numerical score, such as (document_id, score)."""
    if not (isinstance(item, tuple) and len(item) == 2):
        return False
    first, second = item
    if isinstance(first, (MinimalSource, InternalChunk, str, int, np.integer)):
        return isinstance(second, (int, float, np.number)) and not isinstance(
            second, bool
        )
    return False


def _unwrap_ranked_item(item: Any) -> Any:
    """Extracts the document or chunk from a search result.
    If the result contains both a document and a score, it returns only the document."""
    if _is_scored_tuple(item):
        return item[0]
    return item


def _canonical_item_key(payload: Any) -> Any:
    """Creates a unique identifier for a document or chunk.
    This helps the program recognize when the same result appears more than once."""
    if isinstance(payload, (int, np.integer)):
        return int(payload)
    if isinstance(payload, MinimalSource):
        return (
            payload.file_path,
            payload.first_character_index,
            payload.last_character_index,
        )
    if isinstance(payload, InternalChunk):
        return (
            "chunk",
            payload.chunk_id,
            payload.file_path,
            payload.first_character_index,
            payload.last_character_index,
        )
    if hasattr(payload, "file_path") and hasattr(payload, "first_character_index"):
        return (
            str(payload.file_path),
            int(payload.first_character_index),
            int(payload.last_character_index),
        )
    return payload


def _normalize_payload(payload: Any) -> Any:
    """NConverts NumPy integer values into normal
    Python integers so they are easier to work with."""
    if isinstance(payload, np.integer):
        return int(payload)
    return payload


def _has_source_metadata(chunks: Optional[Sequence[Any]]) -> bool:
    """Checks whether a list contains chunk information,
    such as file paths and character positions."""
    if not chunks:
        return False
    first = chunks[0]
    return isinstance(first, (InternalChunk, MinimalSource)) or (
        hasattr(first, "file_path")
        and hasattr(first, "first_character_index")
        and hasattr(first, "last_character_index")
    )


def to_minimal_source(
    item: Any,
    chunks: Optional[Sequence[Any]] = None,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> MinimalSource:
    """Converts a search result into a MinimalSource object.
    It can handle chunk objects, source objects, or document indexes.
    It also checks the character range.
    item: Candidate item (MinimalSource, InternalChunk, int doc index, or
        ``(payload, score)`` tuple).
    chunks: Optional sequence of InternalChunk or MinimalSource objects used
        to resolve integer document indices.
    max_chunk_size: Maximum character span ceiling (default: 2000).
    """
    payload = _unwrap_ranked_item(item)

    if isinstance(payload, MinimalSource):
        return clamp_source(payload, max_chunk_size=max_chunk_size)

    if isinstance(payload, InternalChunk):
        return clamp_source(payload.to_source(), max_chunk_size=max_chunk_size)

    if isinstance(payload, (int, np.integer)):
        idx = int(payload)
        if chunks is None or not _has_source_metadata(chunks):
            raise ValueError(
                "Cannot return MinimalSource objects: no chunk metadata available."
            )
        if idx < 0 or idx >= len(chunks):
            raise IndexError(
                f"Document index {idx} out of bounds for chunks of length {len(chunks)}"
            )
        chunk_obj = chunks[idx]
        return to_minimal_source(chunk_obj, chunks=None, max_chunk_size=max_chunk_size)

    if hasattr(payload, "to_source") and callable(payload.to_source):
        src = payload.to_source()
        if isinstance(src, MinimalSource):
            return clamp_source(src, max_chunk_size=max_chunk_size)

    if (
        hasattr(payload, "file_path")
        and hasattr(payload, "first_character_index")
        and hasattr(payload, "last_character_index")
    ):
        src = MinimalSource(
            file_path=payload.file_path,
            first_character_index=payload.first_character_index,
            last_character_index=payload.last_character_index,
        )
        return clamp_source(src, max_chunk_size=max_chunk_size)

    raise ValueError(
        f"Cannot convert item of type {type(payload).__name__} to MinimalSource: "
        "no chunk metadata available."
    )


def deduplicate_sources(
    sources: Sequence[MinimalSource],
    k: Optional[int] = None,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> List[MinimalSource]:
    """Removes duplicate sources from a list while keeping their original order.
    It can also limit how many sources are returned.
    sources: Sequence of MinimalSource objects in ranked order.
    k: Optional maximum number of deduplicated sources to return.
    max_chunk_size: Maximum character span length ceiling (default: 2000).
    """
    if k is not None and k <= 0:
        return []

    seen: Set[Tuple[str, int, int]] = set()
    deduped: List[MinimalSource] = []

    for src in sources:
        clamped = clamp_source(src, max_chunk_size=max_chunk_size)
        key = (
            clamped.file_path,
            clamped.first_character_index,
            clamped.last_character_index,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clamped)
        if k is not None and len(deduped) >= k:
            break

    return deduped


def _is_sequence_of_rankings(arg: Sequence[Any]) -> bool:
    """Checks whether the input contains several separate search result lists or just one list."""
    if len(arg) == 0:
        return False
    first = arg[0]
    if isinstance(first, list):
        return True
    if isinstance(first, tuple) and not _is_scored_tuple(first):
        return True
    return False


def reciprocal_rank_fusion(
    bm25_results: Sequence[Any],
    dense_results: Optional[Sequence[Any]] = None,
    k: int = DEFAULT_RRF_K,
    *,
    rrf_k: Optional[int] = None,
    top_k: Optional[int] = None,
    weights: Optional[Sequence[float]] = None,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> List[Tuple[Any, float]]:
    """Combines two or more ranked search result lists into one.
    It gives points to documents based on their positions in each list.
    Documents that rank highly in multiple lists usually receive a higher final score.
    For example, if a document appears near the top in both BM25 and Dense
    results, it gets a strong combined ranking
    bm25_results: Ranked lexical results, or a sequence of ranked result lists
        when *dense_results* is omitted.
    dense_results: Ranked dense semantic results.
    k: RRF smoothing constant (default: 60).
    rrf_k: Optional keyword alias for *k*.
    top_k: Optional limit on the number of fused results to return.
    weights: Optional per-ranking weight sequence.
    bm25_weight: Weight for BM25 ranking when fusing two lists (default: 1.0).
    dense_weight: Weight for dense ranking when fusing two lists (default: 1.0).
    """
    effective_rrf_k = rrf_k if rrf_k is not None else k
    if effective_rrf_k < 0:
        raise ValueError(f"RRF k must be non-negative, got {effective_rrf_k}")

    if top_k is not None and top_k <= 0:
        return []

    rankings: List[Sequence[Any]]
    modality_weights: List[float]

    if dense_results is not None:
        rankings = [bm25_results, dense_results]
        if weights is not None:
            if len(weights) != 2:
                raise ValueError(
                    f"Expected 2 weights for (bm25, dense), got {len(weights)}"
                )
            modality_weights = [float(w) for w in weights]
        else:
            modality_weights = [float(bm25_weight), float(dense_weight)]
    elif _is_sequence_of_rankings(bm25_results):
        rankings = list(bm25_results)
        if weights is not None:
            if len(weights) != len(rankings):
                raise ValueError(
                    f"weights length ({len(weights)}) must match rankings count ({len(rankings)})"
                )
            modality_weights = [float(w) for w in weights]
        elif len(rankings) == 2:
            modality_weights = [float(bm25_weight), float(dense_weight)]
        else:
            modality_weights = [1.0] * len(rankings)
    else:
        rankings = [bm25_results]
        if weights is not None:
            if len(weights) != 1:
                raise ValueError(f"Expected 1 weight, got {len(weights)}")
            modality_weights = [float(weights[0])]
        else:
            modality_weights = [float(bm25_weight)]

    scores_by_key: Dict[Any, float] = {}
    payload_by_key: Dict[Any, Any] = {}
    best_rank_by_key: Dict[Any, int] = {}
    first_seen_order: Dict[Any, int] = {}
    order_counter = 0

    for ranking, weight in zip(rankings, modality_weights):
        if weight == 0.0:
            continue
        seen_in_modality: Set[Any] = set()
        for idx, raw_item in enumerate(ranking):
            rank = idx + 1  # 1-based rank
            payload = _normalize_payload(_unwrap_ranked_item(raw_item))
            key = _canonical_item_key(payload)

            if key in seen_in_modality:
                continue
            seen_in_modality.add(key)

            contribution = rrf_score(rank=rank, k=effective_rrf_k, weight=weight)
            scores_by_key[key] = scores_by_key.get(key, 0.0) + contribution

            if key not in payload_by_key:
                payload_by_key[key] = payload
                first_seen_order[key] = order_counter
                order_counter += 1

            if key not in best_rank_by_key or rank < best_rank_by_key[key]:
                best_rank_by_key[key] = rank

    sorted_keys = sorted(
        scores_by_key.keys(),
        key=lambda key: (
            -scores_by_key[key],
            best_rank_by_key[key],
            first_seen_order[key],
        ),
    )

    if top_k is not None:
        sorted_keys = sorted_keys[:top_k]

    return [(payload_by_key[key], scores_by_key[key]) for key in sorted_keys]


def fuse_sources(
    bm25_results: Sequence[Any],
    dense_results: Optional[Sequence[Any]] = None,
    k: Optional[int] = None,
    *,
    rrf_k: int = DEFAULT_RRF_K,
    chunks: Optional[Sequence[Any]] = None,
    bm25_chunks: Optional[Sequence[Any]] = None,
    dense_chunks: Optional[Sequence[Any]] = None,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> List[MinimalSource]:
    """Combines BM25 and Dense results using RRF, converts the results
    into MinimalSource objects, removes duplicates, and returns the best sources.
    bm25_results: Ranked BM25 items (MinimalSource, InternalChunk, doc indices,
        or ``(doc, score)`` tuples).
    dense_results: Ranked Dense items (MinimalSource, InternalChunk, doc indices,
        or ``(doc, score)`` tuples).
    k: Optional number of top MinimalSource objects to return. If None, returns
        all unique fused sources.
    rrf_k: RRF smoothing constant (default: 60).
    chunks: Shared chunk metadata list for resolving integer doc indices.
    bm25_chunks: Optional BM25-specific chunk metadata list.
    dense_chunks: Optional Dense-specific chunk metadata list.
    bm25_weight: Weight multiplier for BM25 ranks (default: 1.0).
    dense_weight: Weight multiplier for Dense ranks (default: 1.0).
    max_chunk_size: Maximum character span length ceiling (default: 2000).
    """
    if k is not None and k <= 0:
        return []

    lexical_lookup = bm25_chunks if bm25_chunks is not None else chunks
    semantic_lookup = dense_chunks if dense_chunks is not None else chunks

    bm25_sources: List[MinimalSource] = [
        to_minimal_source(item, chunks=lexical_lookup, max_chunk_size=max_chunk_size)
        for item in bm25_results
    ]

    dense_sources: Optional[List[MinimalSource]] = None
    if dense_results is not None:
        dense_sources = [
            to_minimal_source(
                item, chunks=semantic_lookup, max_chunk_size=max_chunk_size
            )
            for item in dense_results
        ]

    fused = reciprocal_rank_fusion(
        bm25_results=bm25_sources,
        dense_results=dense_sources,
        k=rrf_k,
        top_k=k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )

    return [src for src, _ in fused]


class HybridRanker:
    """This class manages the complete hybrid search process.
    It connects BM25 and Dense retrieval and combines their results using RRF.
    """

    def __init__(
        self,
        bm25_index: Optional[BM25Index] = None,
        dense_encoder: Optional[DenseEncoder] = None,
        chunks: Optional[Sequence[InternalChunk]] = None,
        rrf_k: int = DEFAULT_RRF_K,
        k: Optional[int] = None,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
        min_candidates: int = DEFAULT_MIN_CANDIDATES,
    ) -> None:
        """Initialize the HybridRanker.
        bm25_index: Optional initialized BM25Index for lexical retrieval.
        dense_encoder: Optional initialized DenseEncoder for semantic retrieval.
        chunks: Optional sequence of InternalChunk metadata records. If omitted,
            automatically resolved from *dense_encoder* or *bm25_index*.
        rrf_k: Reciprocal Rank Fusion smoothing constant (default: 60).
        k: Optional alias for *rrf_k* if passed during initialization.
        bm25_weight: Weight applied to lexical BM25 RRF contribution (default: 1.0).
        dense_weight: Weight applied to semantic Dense RRF contribution (default: 1.0).
        max_chunk_size: Hard character span ceiling for returned sources (default: 2000).
        candidate_multiplier: Factor multiplied by target *k* to determine per-engine
            candidate pool size before RRF fusion (default: 4).
        min_candidates: Minimum candidate pool size retrieved per engine (default: 50).
        """
        effective_rrf_k = k if k is not None else rrf_k
        if effective_rrf_k < 0:
            raise ValueError(f"rrf_k must be non-negative, got {effective_rrf_k}")
        if max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be positive, got {max_chunk_size}")

        self.bm25_index = bm25_index
        self.dense_encoder = dense_encoder
        self.rrf_k = int(effective_rrf_k)
        self.bm25_weight = float(bm25_weight)
        self.dense_weight = float(dense_weight)
        self.max_chunk_size = int(max_chunk_size)
        self.candidate_multiplier = max(1, int(candidate_multiplier))
        self.min_candidates = max(1, int(min_candidates))

        self.chunks: Optional[List[InternalChunk]] = (
            list(chunks) if chunks is not None else self._discover_chunks()
        )

    @property
    def k(self) -> int:
        """Return the Reciprocal Rank Fusion constant k (default: 60)."""
        return self.rrf_k

    def _discover_chunks(self) -> Optional[List[InternalChunk]]:
        """Discover InternalChunk metadata from attached dense or BM25 retrievers."""
        if (
            self.dense_encoder is not None
            and self.dense_encoder.chunks is not None
            and _has_source_metadata(self.dense_encoder.chunks)
        ):
            return list(self.dense_encoder.chunks)

        if (
            self.bm25_index is not None
            and self.bm25_index.chunks is not None
            and _has_source_metadata(self.bm25_index.chunks)
        ):
            return list(self.bm25_index.chunks)

        return None

    def _resolve_chunks_lookup(self) -> Optional[Sequence[Any]]:
        """Return the active chunk metadata collection, refreshing from engines if needed.
        Gets the available chunk information.
        If it is not already stored, it tries to find it from the connected retrievers."""
        if self.chunks is not None and _has_source_metadata(self.chunks):
            return self.chunks
        discovered = self._discover_chunks()
        if discovered is not None:
            self.chunks = discovered
        return discovered

    def _compute_candidate_pool_size(
        self,
        k: int,
        candidate_k: Optional[int] = None,
    ) -> int:
        """Decides how many search results BM25 and Dense should retrieve before combining them.
        It usually retrieves more results than the final number needed."""
        if candidate_k is not None:
            return max(k, int(candidate_k))
        return max(k, k * self.candidate_multiplier, self.min_candidates)

    def _retrieve_bm25_candidates(
        self,
        query: str,
        pool_size: int,
    ) -> List[Tuple[int, float]]:
        """Searches the BM25 index and returns the matching documents with their scores.
        If BM25 search fails, it logs a warning and returns an empty list."""
        if self.bm25_index is None or self.bm25_index.num_docs == 0:
            return []
        try:
            return self.bm25_index.search(query, k=pool_size, filter_zero=True)
        except Exception as exc:
            logger.warning("BM25 retrieval failed for query '%s': %s", query, exc)
            return []

    def _retrieve_dense_candidates(
        self,
        query: str,
        pool_size: int,
    ) -> List[Tuple[int, float]]:
        """Searches the DenseEncoder and returns the matching documents with their scores.
        If Dense search fails, it logs a warning and returns an empty list."""
        if self.dense_encoder is None or self.dense_encoder.embeddings is None:
            return []
        try:
            return self.dense_encoder.search(query, k=pool_size)
        except Exception as exc:
            logger.warning("Dense retrieval failed for query '%s': %s", query, exc)
            return []

    def _ensure_retrievers_configured(self) -> None:
        """Raise RuntimeError if neither BM25 nor Dense retriever is attached."""
        if self.bm25_index is None and self.dense_encoder is None:
            raise RuntimeError(
                "HybridRanker requires at least one retriever (bm25_index or dense_encoder) "
                "to execute search queries."
            )

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        candidate_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Searches with BM25 and Dense, combines their results using RRF,
        and returns the document IDs together with their final RRF scores..
            query: Search query string.
            k: Number of top results to return (default: 5).
            candidate_k: Optional candidate pool size per retriever before RRF fusion.
        """
        self._ensure_retrievers_configured()

        if k <= 0 or not isinstance(query, str) or not query.strip():
            return []

        pool_size = self._compute_candidate_pool_size(k=k, candidate_k=candidate_k)
        bm25_candidates = self._retrieve_bm25_candidates(query, pool_size=pool_size)
        dense_candidates = self._retrieve_dense_candidates(query, pool_size=pool_size)

        if not bm25_candidates and not dense_candidates:
            return []

        fused = reciprocal_rank_fusion(
            bm25_results=bm25_candidates,
            dense_results=dense_candidates,
            k=self.rrf_k,
            top_k=k,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
        )
        return [(int(doc_idx), float(score)) for doc_idx, score in fused]

    def get_scores(
        self,
        query: str,
        candidate_k: Optional[int] = None,
    ) -> Dict[int, float]:
        """Returns the RRF score for every candidate document found for a query.
        The results are stored in a dictionary where the document
        ID is the key and its score is the value.
        """
        self._ensure_retrievers_configured()

        if not isinstance(query, str) or not query.strip():
            return {}

        pool_size = (
            candidate_k
            if candidate_k is not None
            else max(
                self.min_candidates,
                self.bm25_index.num_docs if self.bm25_index is not None else 0,
                len(self.dense_encoder.embeddings)
                if (
                    self.dense_encoder is not None
                    and self.dense_encoder.embeddings is not None
                )
                else 0,
            )
        )
        if pool_size <= 0:
            return {}

        ranked = self.search_with_scores(query, k=pool_size, candidate_k=pool_size)
        return {doc_idx: score for doc_idx, score in ranked}

    def get_top_k(
        self,
        query: str,
        k: int = 5,
        candidate_k: Optional[int] = None,
    ) -> List[int]:
        """Returns the IDs of the best matching chunks or documents for a query.
        It does not return the full source information.
            query: Search query string.
            k: Number of top chunk IDs to return (default: 5).
            candidate_k: Optional candidate pool size per retriever.
        """
        scored = self.search_with_scores(query, k=k, candidate_k=candidate_k)
        chunks_lookup = self._resolve_chunks_lookup()

        if chunks_lookup is not None:
            result_ids: List[int] = []
            for doc_idx, _ in scored:
                chunk_obj = chunks_lookup[doc_idx]
                if hasattr(chunk_obj, "chunk_id"):
                    result_ids.append(int(chunk_obj.chunk_id))
                else:
                    result_ids.append(int(doc_idx))
            return result_ids

        if self.bm25_index is not None and self.bm25_index.chunk_ids is not None:
            return [int(self.bm25_index.chunk_ids[doc_idx]) for doc_idx, _ in scored]

        return [doc_idx for doc_idx, _ in scored]

    def search_sources(
        self,
        query: str,
        k: int = 5,
        candidate_k: Optional[int] = None,
    ) -> List[MinimalSource]:
        """Performs a hybrid search and returns the best results as MinimalSource objects.
        It also removes duplicate sources and checks their character ranges.
            query: Search query string.
            k: Maximum number of deduplicated MinimalSource objects to return (default: 5).
            candidate_k: Optional candidate pool size per retriever before RRF fusion.
        """
        self._ensure_retrievers_configured()

        if k <= 0 or not isinstance(query, str) or not query.strip():
            return []

        chunks_lookup = self._resolve_chunks_lookup()
        bm25_chunks = (
            self.bm25_index.chunks
            if (
                self.bm25_index is not None
                and _has_source_metadata(self.bm25_index.chunks)
            )
            else chunks_lookup
        )
        dense_chunks = (
            self.dense_encoder.chunks
            if (
                self.dense_encoder is not None
                and _has_source_metadata(self.dense_encoder.chunks)
            )
            else chunks_lookup
        )

        if bm25_chunks is None and dense_chunks is None and chunks_lookup is None:
            raise ValueError(
                "Cannot return MinimalSource objects: no chunk metadata available."
            )

        pool_size = self._compute_candidate_pool_size(k=k, candidate_k=candidate_k)
        bm25_candidates = self._retrieve_bm25_candidates(query, pool_size=pool_size)
        dense_candidates = self._retrieve_dense_candidates(query, pool_size=pool_size)

        if not bm25_candidates and not dense_candidates:
            return []

        return fuse_sources(
            bm25_results=bm25_candidates,
            dense_results=dense_candidates,
            k=k,
            rrf_k=self.rrf_k,
            chunks=chunks_lookup,
            bm25_chunks=bm25_chunks,
            dense_chunks=dense_chunks,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
            max_chunk_size=self.max_chunk_size,
        )

    def search(
        self,
        query: str,
        k: int = 5,
        candidate_k: Optional[int] = None,
    ) -> List[MinimalSource]:
        """The main search method.
        It calls search_sources() and returns the best matching sources for the user's query.
            query: Search query string.
            k: Maximum number of MinimalSource objects to return (default: 5).
            candidate_k: Optional candidate pool size per retriever before RRF fusion.
        """
        return self.search_sources(query=query, k=k, candidate_k=candidate_k)

    def fuse_rankings(
        self,
        bm25_results: Sequence[Any],
        dense_results: Optional[Sequence[Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """Combines BM25 and Dense result lists using RRF.
        It returns the combined results together with their RRF scores.
            bm25_results: Ranked lexical items or ``(doc, score)`` tuples.
            dense_results: Ranked semantic items or ``(doc, score)`` tuples.
            top_k: Optional maximum number of results to return.
        """
        return reciprocal_rank_fusion(
            bm25_results=bm25_results,
            dense_results=dense_results,
            k=self.rrf_k,
            top_k=top_k,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
        )

    def fuse(
        self,
        bm25_results: Sequence[Any],
        dense_results: Optional[Sequence[Any]] = None,
        k: Optional[int] = None,
        chunks: Optional[Sequence[Any]] = None,
    ) -> List[MinimalSource]:
        """Combines existing BM25 and Dense results,
        converts them into MinimalSource objects, and removes duplicates.
            bm25_results: Ranked BM25 results (MinimalSource, InternalChunk, doc indices,
                or ``(doc, score)`` tuples).
            dense_results: Ranked Dense results (MinimalSource, InternalChunk, doc indices,
                or ``(doc, score)`` tuples).
            k: Optional maximum number of deduplicated MinimalSource objects to return.
            chunks: Optional chunk metadata sequence used to resolve integer doc indices.
        """
        active_chunks = chunks if chunks is not None else self._resolve_chunks_lookup()
        return fuse_sources(
            bm25_results=bm25_results,
            dense_results=dense_results,
            k=k,
            rrf_k=self.rrf_k,
            chunks=active_chunks,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
            max_chunk_size=self.max_chunk_size,
        )

    def rank(
        self,
        bm25_results: Sequence[Any],
        dense_results: Optional[Sequence[Any]] = None,
        k: Optional[int] = None,
        chunks: Optional[Sequence[Any]] = None,
    ) -> List[MinimalSource]:
        """An alternative name for fuse().
        It performs the same task: combines rankings and returns ordered, unique sources."""
        return self.fuse(
            bm25_results=bm25_results,
            dense_results=dense_results,
            k=k,
            chunks=chunks,
        )

    def __call__(
        self,
        query: str,
        k: int = 5,
        candidate_k: Optional[int] = None,
    ) -> List[MinimalSource]:
        """Allows the HybridRanker object to be used like a function.
        When called, it automatically runs the search() method."""
        return self.search(query=query, k=k, candidate_k=candidate_k)
