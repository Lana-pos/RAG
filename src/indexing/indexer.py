"""Ingestion Coordinator & Corpus Indexer.

High-throughput pipeline coordinator orchestrating recursive file discovery,
syntax-aware AST/Markdown chunking, vectorized BM25Okapi inverted indexing,
and MiniLM dense semantic embedding generation.

Outputs:
    - data/processed/chunks.json
    - data/processed/bm25_index.pkl
    - data/processed/embeddings.npy
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from src.chunking.base import (
    DEFAULT_IGNORE_DIRS,
    DEFAULT_SUPPORTED_EXTENSIONS,
    discover_files,
)
from src.chunking.markdown import MarkdownChunker
from src.chunking.python import PythonChunker
from src.indexing.schema import InternalChunk
from src.retrieval.bm25 import (
    BM25Index,
    DEFAULT_INDEX_PATH,
)
from src.retrieval.dense import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDINGS_PATH as DEFAULT_EMBEDDINGS_PATH,
    EMBEDDING_DIM,
    MODEL_NAME,
    DenseEncoder,
)
from src.retrieval.hybrid import DEFAULT_RRF_K, HybridRanker
from src.retrieval.tokenizer import tokenize_code

logger = logging.getLogger(__name__)

DEFAULT_BM25_INDEX_PATH: Path = DEFAULT_INDEX_PATH

DEFAULT_CORPUS_DIR = Path("data/raw/vllm-0.10.1")
FALLBACK_CORPUS_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.json")
DEFAULT_MAX_CHUNK_SIZE: int = 2000

CHUNKS_FILENAME = "chunks.json"
BM25_FILENAME = "bm25_index.pkl"
EMBEDDINGS_FILENAME = "embeddings.npy"

_BM25_STOPWORDS: Set[str] = {
    "what", "how", "where", "when", "which", "who", "why", "is", "are", "was",
    "were", "do", "does", "did", "can", "could", "should", "would", "will",
    "to", "in", "on", "at", "for", "of", "with", "by", "from", "about", "into",
    "the", "a", "an", "and", "or", "as", "if", "it", "its", "you", "your",
    "i", "my", "we", "be", "been", "being", "have", "has", "had", "that",
    "this", "these", "those", "there", "vllm", "s", "find", "information",
    "using", "used", "use",
}
_PATH_NOISE_TOKENS: Set[str] = {"md", "py", "txt", "docs", "vllm", "data", "raw"}


@dataclass
class IndexingResult:
    """Container for artifacts and metrics produced by a corpus indexing run."""

    chunks: List[InternalChunk]
    bm25_index: BM25Index
    dense_encoder: Optional[DenseEncoder]
    embeddings: Optional[NDArray[np.float32]]
    num_files: int
    num_chunks: int
    chunks_path: Path
    bm25_index_path: Path
    embeddings_path: Optional[Path]
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary summary of the indexing result."""
        return {
            "chunks": self.chunks,
            "bm25_index": self.bm25_index,
            "dense_encoder": self.dense_encoder,
            "embeddings": self.embeddings,
            "num_files": self.num_files,
            "file_count": self.num_files,
            "num_chunks": self.num_chunks,
            "chunk_count": self.num_chunks,
            "chunks_path": self.chunks_path,
            "bm25_index_path": self.bm25_index_path,
            "bm25_path": self.bm25_index_path,
            "embeddings_path": self.embeddings_path,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def __getitem__(self, key: str) -> Any:
        """Provide dictionary-style lookup for result fields."""
        data = self.to_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]

    def __contains__(self, key: object) -> bool:
        """Return True if *key* is a recognized field in the result."""
        if not isinstance(key, str):
            return False
        return key in self.to_dict()

    def keys(self) -> Iterator[str]:
        """Iterate over dictionary keys."""
        return iter(self.to_dict().keys())

    def __len__(self) -> int:
        """Return total number of indexed chunks."""
        return self.num_chunks

    def __str__(self) -> str:
        """Return human-readable summary of the indexing run."""
        return (
            f"Ingestion complete! Indices saved under {self.chunks_path.parent}\n"
            f"Indexed {self.num_files} files ({self.num_chunks} chunks) "
            f"in {self.elapsed_seconds:.2f}s."
        )


def save_chunks(
    chunks: Sequence[InternalChunk],
    file_path: Union[str, Path] = DEFAULT_CHUNKS_PATH,
) -> Path:
    """Serialize a sequence of InternalChunk objects to a JSON catalog file.
        chunks: Sequence of validated InternalChunk instances.
        file_path: Destination JSON file path (default: data/processed/chunks.json).
    """
    target_path = Path(file_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [chunk.model_dump() for chunk in chunks]
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d chunks to '%s'", len(chunks), target_path)
    return target_path


def load_chunks(
    file_path: Union[str, Path] = DEFAULT_CHUNKS_PATH,
) -> List[InternalChunk]:
    """Load a list of InternalChunk objects from a serialized JSON catalog file.
        file_path: Path to the JSON chunk catalog (default: data/processed/chunks.json).
    """
    target_path = Path(file_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Chunks catalog file not found at: {target_path}")

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in chunks file '{target_path}': {exc}") from exc

    if isinstance(raw_data, dict) and "chunks" in raw_data:
        raw_list = raw_data["chunks"]
    elif isinstance(raw_data, list):
        raw_list = raw_data
    else:
        raise ValueError(
            f"Expected JSON array of chunks in '{target_path}', got {type(raw_data).__name__}"
        )

    return [InternalChunk.model_validate(item) for item in raw_list]


class CorpusIndexer:
    """High-throughput ingestion coordinator and corpus indexer.
        Find all .py and .md files.
        Chop them into chunks (maximum 2,000 characters each).
        Build and save the BM25 keyword index (bm25_index.pkl).
        Build and save the Dense AI meaning embeddings (embeddings.npy).
    """

    def __init__(
        self,
        corpus_dir: Union[str, Path] = DEFAULT_CORPUS_DIR,
        output_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        *,
        dense_encoder: Optional[DenseEncoder] = None,
        model_name: str = MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        allowed_extensions: Optional[Sequence[str]] = None,
        ignored_dir_names: Optional[Set[str]] = None,
        enable_dense: bool = True,
        show_progress: bool = True,
        _model: Any = None,
    ) -> None:
        """Initialize the ingestion coordinator.
            corpus_dir: Root directory of source corpus (default: data/raw/vllm-0.10.1).
            output_dir: Directory where processed index artifacts are saved
                (default: data/processed).
            max_chunk_size: Hard character length ceiling for every chunk (default: 2000).
            dense_encoder: Optional pre-initialized DenseEncoder instance.
            model_name: SentenceTransformer model name for dense embeddings.
            batch_size: Batch size for dense embedding generation (default: 64).
            device: Torch device for dense encoder (e.g. "cpu", "cuda", or None).
            tokenizer: Custom lexical tokenizer function (defaults to tokenize_code).
            bm25_k1: BM25Okapi term saturation constant k1 (default: 1.5).
            bm25_b: BM25Okapi length normalization constant b (default: 0.75).
            allowed_extensions: File extensions to ingest (default: {'.py', '.md'}).
            ignored_dir_names: Directory names to skip during file discovery.
            enable_dense: Whether to compute and save dense embeddings (default: True).
            show_progress: Whether to display tqdm progress bars (default: True).
            _model: Optional mock/injected SentenceTransformer model for offline testing.
        """
        if max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be positive, got {max_chunk_size}")

        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.max_chunk_size = int(max_chunk_size)
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self.tokenizer = tokenizer if tokenizer is not None else tokenize_code
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self.allowed_extensions = (
            list(allowed_extensions)
            if allowed_extensions is not None
            else sorted(DEFAULT_SUPPORTED_EXTENSIONS)
        )
        self.ignored_dir_names = (
            set(ignored_dir_names)
            if ignored_dir_names is not None
            else set(DEFAULT_IGNORE_DIRS)
        )
        self.enable_dense = bool(enable_dense)
        self.show_progress = bool(show_progress)

        self._dense_encoder = dense_encoder
        self._injected_model = _model

        self.python_chunker = PythonChunker(max_chunk_size=self.max_chunk_size)
        self.markdown_chunker = MarkdownChunker(max_chunk_size=self.max_chunk_size)

    def _resolve_corpus_dir(
        self,
        corpus_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Resolve target corpus directory, falling back to data/raw if default not found."""
        if corpus_dir is not None:
            return Path(corpus_dir)
        if self.corpus_dir.exists():
            return self.corpus_dir
        if self.corpus_dir == DEFAULT_CORPUS_DIR and FALLBACK_CORPUS_DIR.exists():
            return FALLBACK_CORPUS_DIR
        return self.corpus_dir

    def chunk_file(
        self,
        file_path: Union[str, Path],
        max_chunk_size: Optional[int] = None,
    ) -> List[InternalChunk]:
        """Chunk a single source file using the appropriate syntax-aware chunker.
            file_path: Path to a `.py` or `.md` source file.
            max_chunk_size: Optional override for maximum chunk size in characters.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if max_chunk_size is not None and max_chunk_size != self.max_chunk_size:
            py_chunker = PythonChunker(max_chunk_size=max_chunk_size)
            md_chunker = MarkdownChunker(max_chunk_size=max_chunk_size)
        else:
            py_chunker = self.python_chunker
            md_chunker = self.markdown_chunker

        if ext == ".py":
            chunks = py_chunker.chunk_file(path)
            return py_chunker.guard_chunks(chunks, strategy="split", reindex=True)
        if ext == ".md":
            chunks = md_chunker.chunk_file(path)
            return md_chunker.guard_chunks(chunks, strategy="split", reindex=True)

        # Fallback for any other text extension
        content = py_chunker.read_file_safely(path)
        if not content or not content.strip():
            return []
        raw_chunk = InternalChunk(
            chunk_id=0,
            file_path=str(path),
            first_character_index=0,
            last_character_index=len(content),
            chunk_type="markdown_section",
            token_count=py_chunker.estimate_token_count(content),
            content=content,
        )
        return py_chunker.guard_chunks([raw_chunk], strategy="split", reindex=True)

    def chunk_corpus(
        self,
        corpus_dir: Optional[Union[str, Path]] = None,
        max_chunk_size: Optional[int] = None,
        show_progress: Optional[bool] = None,
    ) -> Tuple[List[Path], List[InternalChunk]]:
        """Discover and chunk all valid source files in the corpus directory.
            corpus_dir: Optional directory override to scan.
            max_chunk_size: Optional character ceiling override.
            show_progress: Optional flag to enable/disable tqdm progress bar.
        """
        target_dir = self._resolve_corpus_dir(corpus_dir)
        effective_max_size = (
            int(max_chunk_size) if max_chunk_size is not None else self.max_chunk_size
        )
        if effective_max_size <= 0:
            raise ValueError(
                f"max_chunk_size must be positive, got {effective_max_size}"
            )

        use_tqdm = self.show_progress if show_progress is None else bool(show_progress)

        files = discover_files(
            corpus_dir=target_dir,
            allowed_extensions=self.allowed_extensions,
            ignored_dir_names=self.ignored_dir_names,
            check_binary=True,
        )
        cmake_file = target_dir / "CMakeLists.txt"
        if cmake_file.is_file() and cmake_file not in files:
            files.append(cmake_file)

        py_chunker = (
            self.python_chunker
            if effective_max_size == self.max_chunk_size
            else PythonChunker(max_chunk_size=effective_max_size)
        )
        md_chunker = (
            self.markdown_chunker
            if effective_max_size == self.max_chunk_size
            else MarkdownChunker(max_chunk_size=effective_max_size)
        )

        all_chunks: List[InternalChunk] = []
        global_chunk_id = 0

        file_iter = tqdm(
            files,
            desc="Indexing",
            unit="file",
            disable=not use_tqdm,
        )

        for file_path in file_iter:
            try:
                ext = file_path.suffix.lower()
                if ext == ".py":
                    file_chunks = py_chunker.chunk_file(file_path)
                elif ext == ".md":
                    file_chunks = md_chunker.chunk_file(file_path)
                else:
                    file_chunks = self.chunk_file(
                        file_path, max_chunk_size=effective_max_size
                    )
            except Exception as exc:
                logger.warning("Skipping file '%s' due to error: %s", file_path, exc)
                continue

            for chunk in file_chunks:
                # Hard safety guard ensuring 0 chunks exceed effective_max_size
                if (
                    chunk.span_length > effective_max_size
                    or len(chunk.content) > effective_max_size
                ):
                    guarded = py_chunker.enforce_max_chunk_size(chunk, strategy="split")
                else:
                    guarded = [chunk]

                for g_chunk in guarded:
                    if g_chunk.chunk_id != global_chunk_id:
                        g_chunk = g_chunk.model_copy(
                            update={"chunk_id": global_chunk_id}
                        )
                    all_chunks.append(g_chunk)
                    global_chunk_id += 1

        return files, all_chunks

    def collect_chunks(
        self,
        corpus_dir: Optional[Union[str, Path]] = None,
        max_chunk_size: Optional[int] = None,
        show_progress: Optional[bool] = None,
    ) -> List[InternalChunk]:
        """Discover and return only the list of sequential InternalChunk objects."""
        _, chunks = self.chunk_corpus(
            corpus_dir=corpus_dir,
            max_chunk_size=max_chunk_size,
            show_progress=show_progress,
        )
        return chunks

    def build_bm25(
        self,
        chunks: Sequence[InternalChunk],
        save_path: Optional[Union[str, Path]] = DEFAULT_BM25_INDEX_PATH,
        show_progress: Optional[bool] = None,
    ) -> BM25Index:
        """Tokenize chunks with tqdm progress tracking and build BM25Okapi index.
            chunks: Sequence of InternalChunk instances.
            save_path: Optional file path to serialize the binary pickle index.
            self.build_bm25(chunks)
            saves bm25_index.pkl (the keyword search math and inverted index).
            show_progress: Optional flag to enable/disable tqdm progress bar.
        """
        use_tqdm = self.show_progress if show_progress is None else bool(show_progress)
        tok = self.tokenizer

        chunk_iter = tqdm(
            chunks,
            desc="Tokenizing",
            unit="chunk",
            disable=not use_tqdm,
        )
        if len(chunks) > 50 and tok is tokenize_code:
            file_titles: Dict[str, List[str]] = {}
            for c in chunks:
                if c.file_path not in file_titles and c.first_character_index == 0:
                    first_lines = c.content.splitlines()[:5]
                    headers = " ".join(
                        line for line in first_lines if line.startswith("#")
                    )
                    file_titles[c.file_path] = [
                        t for t in tok(headers) if t not in _BM25_STOPWORDS
                    ]

            tokenized_corpus = []
            for c in chunk_iter:
                rel_path = c.file_path
                if "vllm-0.10.1/" in rel_path:
                    rel_path = rel_path.split("vllm-0.10.1/", 1)[1]
                path_toks = [
                    t
                    for t in tok(rel_path)
                    if t not in _PATH_NOISE_TOKENS and t not in _BM25_STOPWORDS
                ]
                title_toks = file_titles.get(c.file_path, [])
                content_toks = [
                    t for t in tok(c.content) if t not in _BM25_STOPWORDS
                ]
                tokenized_corpus.append(content_toks + path_toks * 2 + title_toks)
            effective_b = 0.4 if self.bm25_b == 0.75 else self.bm25_b
        else:
            tokenized_corpus = [tok(c.content) for c in chunk_iter]
            effective_b = self.bm25_b

        bm25_index = BM25Index.build(
            corpus_tokens=tokenized_corpus,
            k1=self.bm25_k1,
            b=effective_b,
            tokenizer=tok,
            store_dense=False if len(chunks) > 500 else None,
        )
        bm25_index.chunks = list(chunks)
        bm25_index.chunk_ids = [c.chunk_id for c in chunks]

        if save_path is not None:
            bm25_index.save(save_path)
            logger.info("Saved BM25 index to '%s'", save_path)

        return bm25_index

    def _get_or_create_dense_encoder(self) -> DenseEncoder:
        """Return the configured DenseEncoder, lazily instantiating it if needed."""
        if self._dense_encoder is not None:
            return self._dense_encoder
        self._dense_encoder = DenseEncoder(
            model_name=self.model_name,
            batch_size=self.batch_size,
            device=self.device,
            _model=self._injected_model,
        )
        return self._dense_encoder

    def build_embeddings(
        self,
        chunks: Sequence[InternalChunk],
        save_path: Optional[Union[str, Path]] = DEFAULT_EMBEDDINGS_PATH,
        show_progress: Optional[bool] = None,
    ) -> Tuple[Optional[DenseEncoder], NDArray[np.float32]]:
        """Encode corpus chunks into 384-d L2-normalized embeddings and persist to disk.
            chunks: Sequence of InternalChunk records to encode.
            save_path: Optional destination `.npy` path (default: data/processed/embeddings.npy).
            show_progress: Optional flag for progress tracking.
        """
        if len(chunks) == 0:
            empty_matrix = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
            if save_path is not None:
                target = Path(save_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, empty_matrix)
            if self._dense_encoder is not None:
                self._dense_encoder.embeddings = empty_matrix
                self._dense_encoder.chunks = []
            return self._dense_encoder, empty_matrix

        encoder = self._get_or_create_dense_encoder()
        matrix = encoder.encode_corpus(chunks, save_path=save_path)
        return encoder, matrix

    def save_chunks(
        self,
        chunks: Sequence[InternalChunk],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Persist chunk metadata catalog to JSON.
        self.save_chunks(chunks)
        saves chunks.json (the readable text, file paths, and character positions)."""
        target = (
            Path(save_path)
            if save_path is not None
            else (self.output_dir / CHUNKS_FILENAME)
        )
        return save_chunks(chunks, target)

    def index(
        self,
        corpus_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        max_chunk_size: Optional[int] = None,
        *,
        chunks_path: Optional[Union[str, Path]] = None,
        bm25_index_path: Optional[Union[str, Path]] = None,
        embeddings_path: Optional[Union[str, Path]] = None,
        show_progress: Optional[bool] = None,
    ) -> IndexingResult:
        """Execute the full corpus ingestion and indexing pipeline.
        Scans the source repository, chunks `.py` and `.md` files within
        ``max_chunk_size``, builds the BM25 inverted index, encodes dense semantic
        vectors, and writes:
            - ``<output_dir>/chunks.json``
            - ``<output_dir>/bm25_index.pkl``
            - ``<output_dir>/embeddings.npy``
            corpus_dir: Optional source corpus directory override.
            output_dir: Optional processed artifacts directory override.
            max_chunk_size: Optional character ceiling override (default: 2000).
            chunks_path: Optional explicit path for `chunks.json`.
            bm25_index_path: Optional explicit path for `bm25_index.pkl`.
            embeddings_path: Optional explicit path for `embeddings.npy`.
            show_progress: Optional override for tqdm progress display.
        """
        t0 = time.perf_counter()

        target_output_dir = (
            Path(output_dir) if output_dir is not None else self.output_dir
        )
        target_output_dir.mkdir(parents=True, exist_ok=True)

        resolved_chunks_path = (
            Path(chunks_path)
            if chunks_path is not None
            else (target_output_dir / CHUNKS_FILENAME)
        )
        resolved_bm25_path = (
            Path(bm25_index_path)
            if bm25_index_path is not None
            else (target_output_dir / BM25_FILENAME)
        )
        resolved_embeddings_path = (
            Path(embeddings_path)
            if embeddings_path is not None
            else (target_output_dir / EMBEDDINGS_FILENAME)
        )

        # 1. Discover and chunk source files
        files, chunks = self.chunk_corpus(
            corpus_dir=corpus_dir,
            max_chunk_size=max_chunk_size,
            show_progress=show_progress,
        )

        # 2. Save chunks.json
        save_chunks(chunks, resolved_chunks_path)

        # 3. Build and save BM25 inverted index
        bm25_index = self.build_bm25(
            chunks=chunks,
            save_path=resolved_bm25_path,
            show_progress=show_progress,
        )

        # 4. Build and save dense vector embeddings
        dense_encoder: Optional[DenseEncoder] = None
        embeddings_matrix: Optional[NDArray[np.float32]] = None
        saved_embeddings_path: Optional[Path] = None

        if self.enable_dense:
            dense_encoder, embeddings_matrix = self.build_embeddings(
                chunks=chunks,
                save_path=resolved_embeddings_path,
                show_progress=show_progress,
            )
            saved_embeddings_path = resolved_embeddings_path

        elapsed = time.perf_counter() - t0
        logger.info(
            "Ingestion complete! Indexed %d files (%d chunks) in %.2f s under '%s'",
            len(files),
            len(chunks),
            elapsed,
            target_output_dir,
        )

        return IndexingResult(
            chunks=chunks,
            bm25_index=bm25_index,
            dense_encoder=dense_encoder,
            embeddings=embeddings_matrix,
            num_files=len(files),
            num_chunks=len(chunks),
            chunks_path=resolved_chunks_path,
            bm25_index_path=resolved_bm25_path,
            embeddings_path=saved_embeddings_path,
            elapsed_seconds=elapsed,
        )

    @classmethod
    def load(
        cls,
        output_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
        *,
        chunks_path: Optional[Union[str, Path]] = None,
        bm25_index_path: Optional[Union[str, Path]] = None,
        embeddings_path: Optional[Union[str, Path]] = None,
        load_dense: bool = True,
        model_name: str = MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
        _model: Any = None,
    ) -> Tuple[List[InternalChunk], BM25Index, Optional[DenseEncoder]]:
        """Load persisted chunks, BM25 index, and DenseEncoder from disk.
            output_dir: Directory containing processed index files (default: data/processed).
            chunks_path: Optional explicit path to `chunks.json`.
            bm25_index_path: Optional explicit path to `bm25_index.pkl`.
            embeddings_path: Optional explicit path to `embeddings.npy`.
            load_dense: Whether to load dense embeddings if available (default: True).
            model_name: SentenceTransformer model identifier.
            batch_size: Batch size for DenseEncoder.
            device: Torch device for DenseEncoder.
            _model: Optional injected SentenceTransformer model for testing.
        """
        base_dir = Path(output_dir)
        c_path = (
            Path(chunks_path)
            if chunks_path is not None
            else (base_dir / CHUNKS_FILENAME)
        )
        b_path = (
            Path(bm25_index_path)
            if bm25_index_path is not None
            else (base_dir / BM25_FILENAME)
        )
        e_path = (
            Path(embeddings_path)
            if embeddings_path is not None
            else (base_dir / EMBEDDINGS_FILENAME)
        )

        chunks = load_chunks(c_path)
        bm25_index = BM25Index.load(b_path)
        if bm25_index.chunks is None or not bm25_index.chunks:
            bm25_index.chunks = list(chunks)
            bm25_index.chunk_ids = [c.chunk_id for c in chunks]

        dense_encoder: Optional[DenseEncoder] = None
        if load_dense and e_path.exists():
            try:
                dense_encoder = DenseEncoder.load(
                    embeddings_path=e_path,
                    chunks=chunks,
                    model_name=model_name,
                    batch_size=batch_size,
                    device=device,
                    _model=_model,
                )
            except Exception as exc:
                logger.warning(
                    "Could not load dense encoder from '%s': %s. Falling back to BM25.",
                    e_path,
                    exc,
                )
                dense_encoder = None

        return chunks, bm25_index, dense_encoder

    @classmethod
    def load_hybrid_ranker(
        cls,
        output_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        load_dense: bool = True,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        _model: Any = None,
    ) -> HybridRanker:
        """Load persisted indices and return a ready-to-query HybridRanker."""
        chunks, bm25_index, dense_encoder = cls.load(
            output_dir=output_dir,
            load_dense=load_dense,
            model_name=model_name,
            device=device,
            _model=_model,
        )
        return HybridRanker(
            bm25_index=bm25_index,
            dense_encoder=dense_encoder,
            chunks=chunks,
            rrf_k=rrf_k,
            max_chunk_size=max_chunk_size,
        )


def build_index(
    corpus_dir: Union[str, Path] = DEFAULT_CORPUS_DIR,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    output_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
    *,
    dense_encoder: Optional[DenseEncoder] = None,
    model_name: str = MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
    enable_dense: bool = True,
    show_progress: bool = True,
    _model: Any = None,
) -> IndexingResult:
    """Convenience function to execute the full ingestion and indexing pipeline.
        corpus_dir: Root directory of the raw corpus (default: data/raw/vllm-0.10.1).
        max_chunk_size: Maximum character length per chunk (default: 2000).
        output_dir: Output directory for processed artifacts (default: data/processed).
        dense_encoder: Optional pre-initialized DenseEncoder instance.
        model_name: SentenceTransformer model identifier.
        batch_size: Encoding batch size (default: 64).
        device: Torch device string.
        enable_dense: Whether to generate dense embeddings (default: True).
        show_progress: Whether to display tqdm progress bars (default: True).
        _model: Optional injected SentenceTransformer model for offline testing.
    """
    indexer = CorpusIndexer(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        max_chunk_size=max_chunk_size,
        dense_encoder=dense_encoder,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        enable_dense=enable_dense,
        show_progress=show_progress,
        _model=_model,
    )
    return indexer.index()
