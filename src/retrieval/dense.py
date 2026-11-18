"""Dense semantic retrieval using sentence-transformers/all-MiniLM-L6-v2.

Encodes corpus chunks into 384-dimensional L2-normalized embeddings,
persists the embedding matrix to disk, and performs vectorized cosine
similarity search with sub-50 ms CPU latency.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from src.indexing.schema import InternalChunk
from src.models import MinimalSource

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")

# Batch size for encoding: balances memory and throughput on CPU
DEFAULT_BATCH_SIZE = 64


class DenseEncoder:
    """Dense vector encoder for semantic similarity retrieval.
    Uses ``sentence-transformers/all-MiniLM-L6-v2`` to encode text chunks
    into 384-dimensional L2-normalized embeddings and retrieves top-k
    candidates via vectorized cosine similarity (dot-product on unit vectors).
    model_name: Identifier of the sentence-transformers model in use.
    embedding_dim: Dimensionality of the produced embeddings (384).
    embeddings: Float32 NumPy array of shape ``(N, 384)`` after
        :meth:`encode_corpus` has been called; ``None`` before.
    chunks: List of :class:`~src.indexing.schema.InternalChunk` objects
        that correspond row-wise to :attr:`embeddings`.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
        _model: Any = None,
    ) -> None:
        """Initialise the encoder, optionally loading the sentence-transformer.
            model_name: HuggingFace model identifier (default:
            sentence-transformers/all-MiniLM-L6-v2).
            batch_size: Number of texts encoded per forward pass.
            device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
            `None` lets sentence-transformers auto-select.
            _model: Pre-instantiated SentenceTransformer to inject (used in
                tests to avoid network calls).
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        self.embeddings: Optional[NDArray[np.float32]] = None
        self.chunks: Optional[List[InternalChunk]] = None

        if _model is not None:
            self._model = _model
        else:
            self._model = self._load_model()

    def _load_model(self) -> Any:
        """Lazy-load the SentenceTransformer model.
        Loaded :class:`sentence_transformers.SentenceTransformer` instance:
        loads the model that converts text into numbers.
        ImportError: If ``sentence-transformers`` is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for dense retrieval. "
                "Install it with: pip install sentence-transformers>=3.0.0"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.device is not None:
            kwargs["device"] = self.device

        logger.info("Loading model '%s'…", self.model_name)
        return SentenceTransformer(self.model_name, **kwargs)

    @staticmethod
    def _l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return L2-row-normalized copy of *matrix*.
        Rows with zero norm are left as zero vectors (no division by zero).
        matrix: 2-D float32 array of shape ``(N, D)``.
        L2-normalized array of the same shape.
        """
        norms: NDArray[np.float32] = np.linalg.norm(matrix, axis=1, keepdims=True).astype(
            np.float32
        )
        # Avoid division by zero for all-zero rows
        safe_norms: NDArray[np.float32] = np.where(norms == 0.0, 1.0, norms).astype(np.float32)
        return (matrix / safe_norms).astype(np.float32)

    def encode_corpus(
        self,
        chunks: Sequence[Union[InternalChunk, str]],
        save_path: Optional[Union[str, Path]] = DEFAULT_EMBEDDINGS_PATH,
    ) -> NDArray[np.float32]:
        """Converts the document chunks into embeddings.
        Receives document chunks.
        Extracts the text from each chunk.
        Sends the text to the embedding model.
        Converts each text into 384 numbers.
        Normalizes the embeddings.
        Stores the embeddings inside the encoder.
        Optionally saves them to a .npy file.
        """
        if len(chunks) == 0:
            raise ValueError("chunks must be non-empty to encode a corpus")

        # Extract texts and store chunk metadata
        texts: List[str] = []
        internal_chunks: List[InternalChunk] = []
        for idx, chunk in enumerate(chunks):
            if isinstance(chunk, InternalChunk):
                texts.append(chunk.content)
                internal_chunks.append(chunk)
            else:
                texts.append(str(chunk))
                # Build a synthetic InternalChunk so search_sources works
                internal_chunks.append(
                    InternalChunk(
                        chunk_id=idx,
                        file_path="data/raw/unknown",
                        first_character_index=0,
                        last_character_index=max(1, len(str(chunk))),
                        chunk_type="markdown_section",
                        token_count=len(str(chunk).split()),
                        content=str(chunk),
                    )
                )

        logger.info("Encoding %d chunks with model '%s'…", len(texts), self.model_name)
        t0 = time.perf_counter()

        raw: NDArray[np.float32] = cast(
            NDArray[np.float32],
            self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=False,  # we normalize ourselves for transparency
                convert_to_numpy=True,
            ).astype(np.float32),
        )

        # Guarantee L2-normalized unit vectors (cosine ≡ dot-product)
        matrix = self._l2_normalize(raw)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Encoded %d texts in %.2f s (%.1f texts/s); shape=%s",
            len(texts),
            elapsed,
            len(texts) / max(elapsed, 1e-9),
            matrix.shape,
        )

        self.embeddings = matrix
        self.chunks = internal_chunks

        if save_path is not None:
            target = Path(save_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, matrix)
            logger.info("Embedding matrix saved to '%s'", target)

        return matrix

    def encode_query(self, query: str) -> NDArray[np.float32]:
        """Converts the user's question into an embedding.
        """
        raw: NDArray[np.float32] = cast(
            NDArray[np.float32],
            self._model.encode(
                [query],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=False,
                convert_to_numpy=True,
            ).astype(np.float32),
        )
        vec: NDArray[np.float32] = cast(
            NDArray[np.float32], self._l2_normalize(raw)[0]
        )
        return vec

    def get_scores(self, query: str) -> NDArray[np.float32]:
        """Calculates how similar the user's question is to every document chunk.
        Checks whether document embeddings exist.
        Converts the query into an embedding.
        Compares the query embedding against all document embeddings.
        Returns similarity scores.
        """
        if self.embeddings is None:
            raise RuntimeError(
                "Call encode_corpus() before searching, or load() an existing matrix."
            )
        q_vec = self.encode_query(query)
        # (N, D) @ (D,) → (N,)  — single BLAS dot-product
        scores: NDArray[np.float32] = (self.embeddings @ q_vec).astype(np.float32)
        return scores

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List[tuple[int, float]]:
        """Finds the top-k most relevant chunks.
        This returns the k=x chunks with the highest similarity scores.
        The function uses NumPy to select the best results efficiently.
        """
        if k <= 0:
            return []
        if self.embeddings is None:
            raise RuntimeError(
                "Call encode_corpus() before searching, or load() an existing matrix."
            )

        scores = self.get_scores(query)
        n = len(scores)

        if n == 0:
            return []

        effective_k = min(k, n)
        if effective_k == n:
            # All docs fit — just sort
            sorted_indices = np.argsort(-scores)
            return [(int(i), float(scores[i])) for i in sorted_indices]

        # O(N) partition + sort the shortlist
        top_indices = np.argpartition(-scores, effective_k)[:effective_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(int(i), float(scores[i])) for i in top_indices]

    def get_top_k(self, query: str, k: int = 5) -> List[int]:
        """Returns only the IDs of the top matching chunks.
        """
        results = self.search(query, k=k)
        if self.chunks is not None:
            return [self.chunks[doc_idx].chunk_id for doc_idx, _ in results]
        return [doc_idx for doc_idx, _ in results]

    def search_sources(
        self,
        query: str,
        k: int = 5,
    ) -> List[MinimalSource]:
        """Returns the matching chunks as MinimalSource objects /
        Return the actual source information for the relevant chunks.
        """
        if self.chunks is None:
            raise ValueError(
                "No chunk metadata available. "
                "Build the encoder with encode_corpus(chunks) using InternalChunk objects."
            )
        results = self.search(query, k=k)
        return [self.chunks[doc_idx].to_source() for doc_idx, _ in results]

    def save(
        self,
        embeddings_path: Union[str, Path] = DEFAULT_EMBEDDINGS_PATH,
    ) -> Path:
        """Saves the embedding matrix to disk.
        """
        if self.embeddings is None:
            raise RuntimeError("No embeddings to save. Call encode_corpus() first.")
        target = Path(embeddings_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, self.embeddings)
        logger.info("Saved %d embeddings to '%s'", len(self.embeddings), target)
        return target

    @classmethod
    def load(
        cls,
        embeddings_path: Union[str, Path] = DEFAULT_EMBEDDINGS_PATH,
        chunks: Optional[List[InternalChunk]] = None,
        model_name: str = MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
        _model: Any = None,
    ) -> DenseEncoder:
        """Loads previously saved embeddings.
        Checks whether the file exists.
        Loads the embeddings.
        Checks that they have 384 dimensions.
        Optionally attaches chunk metadata.
        Creates a DenseEncoder ready for searching.
        """
        target = Path(embeddings_path)
        if not target.exists():
            raise FileNotFoundError(
                f"Embedding matrix not found at: {target}"
            )

        matrix: NDArray[np.float32] = cast(
            NDArray[np.float32],
            np.load(target).astype(np.float32),
        )

        if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Expected embedding matrix of shape (N, {EMBEDDING_DIM}), "
                f"got {matrix.shape}"
            )

        if chunks is not None and len(chunks) != matrix.shape[0]:
            raise ValueError(
                f"chunks length ({len(chunks)}) does not match "
                f"embedding rows ({matrix.shape[0]})"
            )

        encoder = cls(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            _model=_model,
        )
        encoder.embeddings = matrix
        encoder.chunks = list(chunks) if chunks is not None else None
        logger.info(
            "Loaded embedding matrix %s from '%s'", matrix.shape, target
        )
        return encoder


def build_dense_index(
    chunks: Sequence[Union[InternalChunk, str]],
    save_path: Union[str, Path] = DEFAULT_EMBEDDINGS_PATH,
    model_name: str = MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
) -> DenseEncoder:
    """Creates a DenseEncoder.
    Calls encode_corpus().
    Saves the embeddings.
    Returns the encoder.
    Creates and saves the index
    """
    encoder = DenseEncoder(model_name=model_name, batch_size=batch_size, device=device)
    encoder.encode_corpus(chunks, save_path=save_path)
    return encoder


def load_dense_index(
    embeddings_path: Union[str, Path] = DEFAULT_EMBEDDINGS_PATH,
    chunks: Optional[List[InternalChunk]] = None,
    model_name: str = MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
    _model: Any = None,
) -> DenseEncoder:
    """This is a shortcut for loading an existing dense index.
    """
    return DenseEncoder.load(
        embeddings_path=embeddings_path,
        chunks=chunks,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        _model=_model,
    )
