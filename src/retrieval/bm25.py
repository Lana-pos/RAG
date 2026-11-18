"""Vectorized BM25Okapi Inverted Index implementation for lexical retrieval.

Implements pure Python and vectorized NumPy BM25Okapi scoring (k1=1.5, b=0.75)
with sparse inverted index lookups, document length normalization, and binary
pickle persistence with zero external database daemon dependencies.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import pickle
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    _PostingArrayBase = np.ndarray[Any, Any]
else:
    _PostingArrayBase = np.ndarray

try:
    from scipy.sparse import csr_matrix, spmatrix
    _HAS_SCIPY = True
except ImportError:
    csr_matrix = None
    spmatrix = None
    _HAS_SCIPY = False

from src.indexing.schema import InternalChunk
from src.models import MinimalSource
from src.retrieval.tokenizer import tokenize_code

DEFAULT_INDEX_PATH = Path("data/processed/bm25_index.pkl")


class PostingArray(_PostingArrayBase):
    """NumPy 1D array representing document IDs containing a term,
    with attached term frequency array for vectorized BM25 scoring.
    It behaves like a NumPy array.
    It stores document IDs.
    It also stores how many times the token appears in those documents.
    """

    doc_ids: NDArray[np.int32]
    term_frequencies: NDArray[np.float32]
    # A NumPy array containing document IDs.
    # A NumPy array containing term frequencies corresponding to the document IDs.

    def __new__(
        cls,
        doc_ids: Union[Sequence[int], NDArray[Any]],
        term_frequencies: Union[Sequence[Union[float, int]], NDArray[Any]],
    ) -> PostingArray:
        """Create a new PostingArray instance with document IDs and term frequencies."""
        arr = np.asarray(doc_ids, dtype=np.int32)
        obj = arr.view(cls)
        obj.doc_ids = arr
        obj.term_frequencies = np.asarray(term_frequencies, dtype=np.float32)
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        """Finalize the PostingArray instance, ensuring doc_ids and term_frequencies are set."""
        if obj is None:
            return
        doc_ids = getattr(obj, "doc_ids", None)
        self.doc_ids = (
            doc_ids if doc_ids is not None else np.array([], dtype=np.int32)
        )
        term_freqs = getattr(obj, "term_frequencies", None)
        self.term_frequencies = (
            term_freqs if term_freqs is not None else np.array([], dtype=np.float32)
        )

    def __reduce__(self) -> Any:
        return (self.__class__, (self.doc_ids, self.term_frequencies))


class PureNumPyCSRMatrix:
    """Lightweight pure NumPy Compressed Sparse Row matrix implementation.

    Used as fallback when scipy is unavailable, providing identical interface
    for `.shape`, `[row, col]`, `.toarray()`, `.data`, `.indices`, `.indptr`.
    """

    def __init__(
        self,
        data: Sequence[Union[int, float]] | NDArray[Any],
        indices: Sequence[int] | NDArray[Any],
        indptr: Sequence[int] | NDArray[Any],
        shape: Tuple[int, int],
    ) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.indptr = np.asarray(indptr, dtype=np.int32)
        self.shape = shape

    @property
    def nnz(self) -> int:
        """Number of stored non-zero values."""
        return int(len(self.data))

    def toarray(self) -> NDArray[np.float32]:
        """Convert sparse representation to dense 2D NumPy array."""
        dense = np.zeros(self.shape, dtype=np.float32)
        for row in range(self.shape[0]):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            cols = self.indices[start:end]
            dense[row, cols] = self.data[start:end]
        return dense

    def __getitem__(self, key: Tuple[int, int]) -> float:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError("Index must be a (row, col) tuple")
        row, col = key
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            raise IndexError("Matrix index out of bounds")
        start = self.indptr[row]
        end = self.indptr[row + 1]
        row_cols = self.indices[start:end]
        match_idx = np.where(row_cols == col)[0]
        if len(match_idx) > 0:
            return float(self.data[start + match_idx[0]])
        return 0.0


class BM25Index:
    """Vectorized BM25Okapi inverted index with pure NumPy retrieval.
    Calculates Okapi BM25 scores:
        Score(D, Q) = sum_{q in Q} IDF(q) * (tf * (k1 + 1))
                      / (tf + k1 * (1 - b + b * (|D| / avgdl)))
    with sub-millisecond per-query retrieval latency.
    Receive documents or chunks.
    Count words in each document.
    Build the term frequency matrix.
    Build the inverted index.
    Calculate BM25 scores when you search.
    """

    def __init__(
        self,
        corpus: Optional[Sequence[Sequence[str]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        store_dense: Optional[bool] = None,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer if tokenizer is not None else tokenize_code

        self.num_docs: int = 0
        self.doc_lengths: NDArray[np.int32] = np.array([], dtype=np.int32)
        self.avgdl: float = 0.0
        self.vocabulary: Dict[str, int] = {}
        self.terms: List[str] = []
        self.idf: Dict[str, float] = {}
        self.inverted_index: Dict[str, PostingArray] = {}
        self._term_frequency_matrix: Any = np.zeros((0, 0), dtype=np.int32)
        self._K_d: NDArray[np.float32] = np.array([], dtype=np.float32)

        self.chunks: Optional[List[Any]] = None
        self.chunk_ids: Optional[List[int]] = None

        if corpus is not None:
            self.fit(corpus, store_dense=store_dense)

    @property
    def term_frequency_matrix(self) -> Any:
        """Return the corpus term frequency matrix (dense or sparse)."""
        return self._term_frequency_matrix

    @property
    def postings(self) -> Dict[str, Tuple[NDArray[np.int32], NDArray[np.float32]]]:
        """Return dictionary mapping term -> (doc_ids, term_frequencies)."""
        return {
            term: (posting.doc_ids, posting.term_frequencies)
            for term, posting in self.inverted_index.items()
        }

    def get_postings(
        self, term: str
    ) -> Optional[Tuple[NDArray[np.int32], NDArray[np.float32]]]:
        """Return (doc_ids, term_frequencies) tuple for a given term, or None."""
        norm_term = term.lower()
        if norm_term in self.inverted_index:
            posting = self.inverted_index[norm_term]
            return (posting.doc_ids, posting.term_frequencies)
        return None

    def get_term_frequency_matrix(self, dense: bool = False) -> Any:
        """Return term frequency matrix, converting to dense np.ndarray if requested."""
        if dense and hasattr(self._term_frequency_matrix, "toarray"):
            return self._term_frequency_matrix.toarray()
        return self._term_frequency_matrix

    def fit(
        self,
        corpus_tokens: Sequence[Sequence[str]],
        store_dense: Optional[bool] = None,
    ) -> BM25Index:
        """Build document lengths, inverted index, and term frequency matrix.
        The function will build:
        1. The number of documents.
        2. The length of each document.
        3. The average document length.
        4. The inverted index.
        5. The IDF score for each word.
        6. The term frequency matrix.
        corpus_tokens: Sequence of token lists representing documents.
        store_dense: Whether to force dense NumPy term frequency matrix.
            If None, auto-selects dense if num_docs * vocab_size <= 10_000_000.
        self for method chaining.
        """
        self.num_docs = len(corpus_tokens)
        self.doc_lengths = np.array(
            [len(doc) for doc in corpus_tokens], dtype=np.int32
        )
        self.avgdl = float(np.mean(self.doc_lengths)) if self.num_docs > 0 else 0.0

        if self.avgdl > 0:
            len_ratio = self.doc_lengths.astype(np.float32) / self.avgdl
        else:
            len_ratio = np.zeros(self.num_docs, dtype=np.float32)
        self._K_d = (self.k1 * (1.0 - self.b + self.b * len_ratio)).astype(np.float32)

        # 1. First pass: count frequencies and accumulate postings
        postings_builder: Dict[str, Tuple[List[int], List[int]]] = {}
        for doc_id, doc in enumerate(corpus_tokens):
            counts = Counter(doc)
            for term, freq in counts.items():
                if term not in postings_builder:
                    postings_builder[term] = ([], [])
                postings_builder[term][0].append(doc_id)
                postings_builder[term][1].append(freq)

        # 2. Sort vocabulary and build inverted index + IDFs
        self.terms = sorted(postings_builder.keys())
        self.vocabulary = {term: idx for idx, term in enumerate(self.terms)}
        self.inverted_index = {}
        self.idf = {}

        n_docs = self.num_docs
        for term, (doc_ids, freqs) in postings_builder.items():
            self.inverted_index[term] = PostingArray(doc_ids, freqs)
            df = len(doc_ids)
            # Okapi BM25 smoothed non-negative IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
            idf_val = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            self.idf[term] = float(idf_val)

        # 3. Build term frequency matrix
        vocab_size = len(self.terms)
        dense_threshold = 10_000_000
        should_store_dense = (
            store_dense
            if store_dense is not None
            else (self.num_docs * vocab_size <= dense_threshold)
        )

        if should_store_dense:
            dense_tf = np.zeros((self.num_docs, vocab_size), dtype=np.int32)
            for term, posting in self.inverted_index.items():
                col_idx = self.vocabulary[term]
                dense_tf[posting.doc_ids, col_idx] = posting.term_frequencies.astype(
                    np.int32
                )
            self._term_frequency_matrix = dense_tf
        else:
            # Build CSR sparse matrix representation
            indptr: List[int] = [0]
            indices: List[int] = []
            data: List[float] = []
            for doc in corpus_tokens:
                counts = Counter(doc)
                sorted_entries = sorted(
                    [(self.vocabulary[t], float(c)) for t, c in counts.items()]
                )
                for col_idx, count_val in sorted_entries:
                    indices.append(col_idx)
                    data.append(count_val)
                indptr.append(len(indices))

            matrix_shape = (self.num_docs, vocab_size)
            if _HAS_SCIPY and csr_matrix is not None:
                self._term_frequency_matrix = csr_matrix(
                    (data, indices, indptr), shape=matrix_shape, dtype=np.float32
                )
            else:
                self._term_frequency_matrix = PureNumPyCSRMatrix(
                    data=data, indices=indices, indptr=indptr, shape=matrix_shape
                )

        return self

    @classmethod
    def build(
        cls,
        corpus_tokens: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        store_dense: Optional[bool] = None,
    ) -> BM25Index:
        """Factory method to construct and fit a BM25Index from tokenized documents.
        this involves calculating information such as:
        - How many documents exist.
        - How frequently each token appears.
        - How long each document is.
        - Information needed to calculate BM25 scores."""
        index = cls(k1=k1, b=b, tokenizer=tokenizer)
        index.fit(corpus_tokens, store_dense=store_dense)
        return index

    @classmethod
    def build_inverted_index(
        cls,
        chunks: Sequence[Union[InternalChunk, str, Sequence[str]]],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        store_dense: Optional[bool] = None,
    ) -> BM25Index:
        """Construct a BM25Index from InternalChunk records, strings, or token lists.

        Conforms to architectural lifecycle contract:
        Lexical index builder called during ingestion coordinator pipeline.
        """
        tok = tokenizer if tokenizer is not None else tokenize_code
        tokenized_corpus: List[List[str]] = []
        chunk_objects: List[Any] = []
        chunk_ids: List[int] = []

        for idx, item in enumerate(chunks):
            if isinstance(item, InternalChunk):
                tokens = tok(item.content)
                tokenized_corpus.append(tokens)
                chunk_objects.append(item)
                chunk_ids.append(item.chunk_id)
            elif isinstance(item, str):
                tokens = tok(item)
                tokenized_corpus.append(tokens)
                chunk_objects.append(item)
                chunk_ids.append(idx)
            else:
                # Already a sequence of tokens
                tokens = [t.lower() for t in item]
                tokenized_corpus.append(tokens)
                chunk_objects.append(item)
                chunk_ids.append(idx)

        index = cls.build(
            corpus_tokens=tokenized_corpus,
            k1=k1,
            b=b,
            tokenizer=tok,
            store_dense=store_dense,
        )
        index.chunks = chunk_objects
        index.chunk_ids = chunk_ids
        return index

    def get_scores(self, query: Union[str, Sequence[str]]) -> NDArray[np.float32]:
        """Compute vectorized BM25 scores across all corpus documents.
        query: Raw query string or pre-tokenized sequence of strings.
        1D NumPy array of length num_docs with BM25 relevance scores.
        """
        if self.num_docs == 0:
            return np.array([], dtype=np.float32)

        if isinstance(query, str):
            query_tokens = self.tokenizer(query)
        else:
            query_tokens = [t.lower() for t in query]

        if not query_tokens:
            return np.zeros(self.num_docs, dtype=np.float32)

        scores = np.zeros(self.num_docs, dtype=np.float32)
        query_counts = Counter(query_tokens)

        k1 = self.k1
        k_d = self._K_d

        for term, qtf in query_counts.items():
            if term not in self.inverted_index:
                continue
            posting = self.inverted_index[term]
            doc_ids = posting.doc_ids
            tfs = posting.term_frequencies
            idf_val = self.idf[term]

            # Vectorized Okapi BM25 candidate scoring
            denom = tfs + k_d[doc_ids]
            term_score = (idf_val * qtf) * (tfs * (k1 + 1.0)) / denom
            scores[doc_ids] += term_score

        return scores

    def search(
        self,
        query: Union[str, Sequence[str]],
        k: int = 5,
        filter_zero: bool = True,
    ) -> List[Tuple[int, float]]:
        """Retrieve top-k documents matching the query sorted by relevance.
        query: Query text or sequence of tokens.
        k: Maximum number of candidate documents to return.
        filter_zero: If True, only returns documents with BM25 score > 0.
        List of (doc_index, score) tuples in descending score order.
        """
        if k <= 0 or self.num_docs == 0:
            return []

        scores = self.get_scores(query)

        if filter_zero:
            matching_indices = np.flatnonzero(scores > 0)
            if len(matching_indices) == 0:
                return []
            matching_scores = scores[matching_indices]
            if len(matching_indices) <= k:
                sorted_order = np.argsort(-matching_scores)
                return [
                    (int(matching_indices[i]), float(matching_scores[i]))
                    for i in sorted_order
                ]
            top_sub_idx = np.argpartition(-matching_scores, k)[:k]
            top_sub_idx = top_sub_idx[np.argsort(-matching_scores[top_sub_idx])]
            return [
                (int(matching_indices[i]), float(matching_scores[i]))
                for i in top_sub_idx
            ]

        if self.num_docs <= k:
            sorted_indices = np.argsort(-scores)
            return [(int(i), float(scores[i])) for i in sorted_indices]

        top_indices = np.argpartition(-scores, k)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(int(i), float(scores[i])) for i in top_indices]

    def get_top_k(
        self,
        query: Union[str, Sequence[str]],
        k: int = 5,
        filter_zero: bool = True,
    ) -> List[int]:
        """Return top-k document indices matching the query.
        query: Query text or sequence of tokens.
        k: Maximum number of chunk indices to return.
        ilter_zero: If True, only returns chunks with BM25 score > 0.
        Returns list of integer document/chunk indices in descending relevance order.
        """
        results = self.search(query=query, k=k, filter_zero=filter_zero)
        if self.chunk_ids is not None:
            return [self.chunk_ids[doc_idx] for doc_idx, _ in results]
        return [doc_idx for doc_idx, _ in results]

    def search_sources(
        self,
        query: Union[str, Sequence[str]],
        k: int = 5,
        filter_zero: bool = True,
    ) -> List[MinimalSource]:
        """Return top-k results mapped to MinimalSource schema objects.
        Requires index to have been built with InternalChunk objects.
        """
        if self.chunks is None:
            raise ValueError(
                "Cannot return MinimalSource objects: index was not built with chunk metadata"
            )
        results = self.search(query=query, k=k, filter_zero=filter_zero)
        sources: List[MinimalSource] = []
        for doc_idx, _ in results:
            chunk = self.chunks[doc_idx]
            if hasattr(chunk, "to_source"):
                sources.append(chunk.to_source())
            else:
                sources.append(
                    MinimalSource(
                        file_path=chunk.file_path,
                        first_character_index=chunk.first_character_index,
                        last_character_index=chunk.last_character_index,
                    )
                )
        return sources

    def save(self, file_path: Union[str, Path] = DEFAULT_INDEX_PATH) -> Path:
        """Persist binary index payload to disk via pickle.
        file_path: Destination path for binary pickle file.
        Returns resolved Path where index was serialized.
        """
        target_path = Path(file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return target_path

    @classmethod
    def load(cls, file_path: Union[str, Path] = DEFAULT_INDEX_PATH) -> BM25Index:
        """Deserialize binary index payload from disk.
        file_path: Location of serialized pickle payload.
        Returns deserialized BM25Index instance.
        """
        target_path = Path(file_path)
        if not target_path.exists():
            raise FileNotFoundError(f"BM25 index binary not found at: {target_path}")
        with open(target_path, "rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Loaded object is of type {type(loaded).__name__}, expected {cls.__name__}"
            )
        return loaded


def save_bm25_index(
    index: BM25Index, file_path: Union[str, Path] = DEFAULT_INDEX_PATH
) -> Path:
    """Convenience function to save BM25Index to disk."""
    return index.save(file_path)


def load_bm25_index(file_path: Union[str, Path] = DEFAULT_INDEX_PATH) -> BM25Index:
    """Convenience function to load BM25Index from disk."""
    return BM25Index.load(file_path)
