"""CLI Interface Router built with Python Fire.
Exposes the complete command suite matching the subject specification:
    - index [--max_chunk_size 2000]
    - search <query> [--k 5]
    - search_dataset --dataset_path ... --save_directory ... [--k 10]
    - answer <query> [--k 5]
    - answer_dataset --student_search_results_path ... --save_directory ...
    - evaluate --student_search_results_path ... --dataset_path ... [--k 5]
All output files are strictly validated against Pydantic v2 schemas
(:class:`~src.models.StudentSearchResults` and
:class:`~src.models.StudentSearchResultsAndAnswer`) with hard character-span
ceiling enforcement (``<= 2000`` characters) and normalized relative POSIX
file paths. Degenerate inputs (empty queries, ``k <= 0``, missing files,
malformed JSON) are handled gracefully without unhandled tracebacks.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
from types import FrameType
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import fire
from pydantic import BaseModel, PrivateAttr
from tqdm import tqdm

from src.generation.llm import (
    DEFAULT_DEVICE,
    DEFAULT_FALLBACK_ANSWER,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME as DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    EMPTY_QUERY_ANSWER,
    LLMInferenceEngine,
    normalize_retrieved_sources,
)
from src.indexing.indexer import (
    BM25_FILENAME,
    CHUNKS_FILENAME,
    DEFAULT_CORPUS_DIR as DEFAULT_CORPUS_DIR,
    DEFAULT_MAX_CHUNK_SIZE as DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_PROCESSED_DIR as DEFAULT_PROCESSED_DIR,
    CorpusIndexer,
    IndexingResult,
)
from src.indexing.schema import InternalChunk
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
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import DEFAULT_RRF_K, HybridRanker, clamp_source
from src.utils.evaluation import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_CONTEXT_LENGTH,
    EvaluationResult,
    RecallEvaluator,
)

logger = logging.getLogger(__name__)

# Ensure Python Fire never blocks on an interactive terminal pager (e.g. `less`)
os.environ.setdefault("PAGER", "cat")

DEFAULT_SEARCH_K: int = 5
DEFAULT_DATASET_SEARCH_K: int = 10
DEFAULT_ANSWER_K: int = 5
DEFAULT_EVALUATE_K: int = 5
DEFAULT_K: int = 5


def _is_invoked_by_fire() -> bool:
    """Return True if the current method call is being dispatched by Python Fire."""
    frame: Optional[FrameType] = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == "_CallAndUpdateTrace":
            mod_name = str(frame.f_globals.get("__name__", ""))
            if mod_name.startswith("fire"):
                return True
        frame = frame.f_back
    return False


def _emit_output(text: str) -> None:
    """Print *text* to stdout when called directly in Python (outside Fire)."""
    if not _is_invoked_by_fire():
        print(text)


def _resolve_output_file_path(
    save_directory: Optional[Union[str, Path]],
    output_path: Optional[Union[str, Path]],
    source_path: Optional[Union[str, Path]],
    default_filename: str,
) -> Optional[Path]:
    """Resolve destination JSON file path from ``--save_directory`` or ``--output_path``."""
    if output_path is not None and str(output_path).strip():
        return Path(str(output_path).strip())

    if save_directory is not None and str(save_directory).strip():
        save_dir_path = Path(str(save_directory).strip())
        if save_dir_path.suffix.lower() == ".json" and not save_dir_path.is_dir():
            return save_dir_path

        filename = default_filename
        if source_path is not None and str(source_path).strip():
            src_name = Path(str(source_path).strip()).name
            if src_name:
                filename = src_name
        return save_dir_path / filename

    return None


class _CLIBaseResultMixin:
    """Base mixin providing common output_path, dictionary access, and JSON serialization."""

    _output_path: Optional[Path]

    @property
    def output_path(self) -> Optional[Path]:
        """Return file path where results were written, if saved to disk."""
        return getattr(self, "_output_path", None)

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation of the result."""
        model_self = cast(BaseModel, self)
        data: Dict[str, Any] = json.loads(model_self.model_dump_json())
        out_path = self.output_path
        if out_path is not None:
            data["output_path"] = str(out_path)
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serialize to formatted JSON."""
        payload = {k: v for k, v in self.to_dict().items() if k != "output_path"}
        return json.dumps(payload, ensure_ascii=False, indent=indent)

    def __getitem__(self, key: str) -> Any:
        data = self.to_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self.to_dict()

    def keys(self) -> Iterator[str]:
        """Iterate over dictionary keys."""
        return iter(self.to_dict().keys())

    def __str__(self) -> str:
        return self.to_json(indent=2)

    def __repr__(self) -> str:
        return self.to_json(indent=2)


class _CLISingleResultMixin(_CLIBaseResultMixin):
    """Mixin for single-query CLI results providing dual single/batch representation."""

    _k: int

    @property
    def k(self) -> int:
        """Return requested top-k count (clamped to >= 1)."""
        return getattr(self, "_k", 1)

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation supporting both single and batch keys."""
        model_self = cast(BaseModel, self)
        base: Dict[str, Any] = json.loads(model_self.model_dump_json())
        data: Dict[str, Any] = {
            **base,
            "search_results": [base],
            "k": self.k,
        }
        out_path = self.output_path
        if out_path is not None:
            data["output_path"] = str(out_path)
        return data


class CLISearchResult(_CLISingleResultMixin, MinimalSearchResults):
    """Single-query search result model with dual Minimal/Student JSON serialization."""

    _k: int = PrivateAttr(default=DEFAULT_SEARCH_K)
    _output_path: Optional[Path] = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        k: int = DEFAULT_SEARCH_K,
        output_path: Optional[Path] = None,
        **data: Any,
    ) -> None:
        super().__init__(**data)
        self._k = max(1, int(k))
        self._output_path = output_path

    @property
    def search_results(self) -> List[MinimalSearchResults]:
        """Return single-element list of MinimalSearchResults for batch compatibility."""
        return [
            MinimalSearchResults(
                question_id=self.question_id,
                question=self.question,
                retrieved_sources=list(self.retrieved_sources),
            )
        ]

    def to_batch(self) -> StudentSearchResults:
        """Convert single-query search result into a validated StudentSearchResults model."""
        return StudentSearchResults(
            search_results=self.search_results,
            k=self._k,
        )


class CLISearchDatasetResult(_CLIBaseResultMixin, StudentSearchResults):
    """Batch dataset search result model conforming to StudentSearchResults."""

    _output_path: Optional[Path] = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        output_path: Optional[Path] = None,
        **data: Any,
    ) -> None:
        super().__init__(**data)
        self._output_path = output_path

    def __str__(self) -> str:
        if self._output_path is not None:
            return f"Saved student_search_results to {self._output_path}"
        return self.to_json(indent=2)


class CLIAnswerResult(_CLISingleResultMixin, MinimalAnswer):
    """Single-query answer result model with dual MinimalAnswer/Student JSON serialization."""

    _k: int = PrivateAttr(default=DEFAULT_ANSWER_K)
    _output_path: Optional[Path] = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        k: int = DEFAULT_ANSWER_K,
        output_path: Optional[Path] = None,
        **data: Any,
    ) -> None:
        super().__init__(**data)
        self._k = max(1, int(k))
        self._output_path = output_path

    @property
    def search_results(self) -> List[MinimalAnswer]:
        """Return single-element list of MinimalAnswer for batch compatibility."""
        return [
            MinimalAnswer(
                question_id=self.question_id,
                question=self.question,
                retrieved_sources=list(self.retrieved_sources),
                answer=self.answer,
            )
        ]

    def to_batch(self) -> StudentSearchResultsAndAnswer:
        """Convert single-query answer into a validated StudentSearchResultsAndAnswer model."""
        return StudentSearchResultsAndAnswer(
            search_results=self.search_results,
            k=self._k,
        )


class CLIAnswerDatasetResult(_CLIBaseResultMixin, StudentSearchResultsAndAnswer):
    """Batch dataset answer result model conforming to StudentSearchResultsAndAnswer."""

    _output_path: Optional[Path] = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        output_path: Optional[Path] = None,
        **data: Any,
    ) -> None:
        super().__init__(**data)
        self._output_path = output_path

    def __str__(self) -> str:
        if self._output_path is not None:
            n = len(self.search_results)
            return (
                f"Loaded {n} questions ... Processed {n} of {n} questions\n"
                f"Saved student_search_results_and_answer to {self._output_path}"
            )
        return self.to_json(indent=2)


def _parse_dataset_questions(
    dataset_path: Union[str, Path],
) -> List[Tuple[str, str]]:
    """Load questions from *dataset_path* as ``[(question_id, question_text), ...]``.
    Supports :class:`RagDataset` JSON (``{"rag_questions": [...]}``),
    :class:`StudentSearchResults` JSON (``{"search_results": [...]}``), or a
    JSON list of question objects/strings. Graceful against empty question
    strings inside degenerate test datasets.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in dataset file '{path}': {exc}") from exc

    raw_items: Sequence[Any]
    if isinstance(raw_data, Mapping):
        if "rag_questions" in raw_data:
            try:
                validated = RagDataset.model_validate(raw_data)
                return [(q.question_id, q.question) for q in validated.rag_questions]
            except Exception:
                seq = raw_data["rag_questions"]
                if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
                    raise ValueError(
                        f"Expected 'rag_questions' list in '{path}', got {type(seq).__name__}"
                    )
                raw_items = seq
        elif "search_results" in raw_data:
            seq = raw_data["search_results"]
            if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
                raise ValueError(
                    f"Expected 'search_results' list in '{path}', got {type(seq).__name__}"
                )
            raw_items = seq
        else:
            raise ValueError(
                f"Expected 'rag_questions' key in dataset file '{path}', "
                f"got keys {sorted(raw_data.keys())}"
            )
    elif isinstance(raw_data, list):
        raw_items = raw_data
    else:
        raise ValueError(
            f"Expected JSON object or array in '{path}', got {type(raw_data).__name__}"
        )

    parsed_questions: List[Tuple[str, str]] = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, (UnansweredQuestion, AnsweredQuestion, MinimalSearchResults)):
            parsed_questions.append((item.question_id, item.question))
        elif isinstance(item, Mapping):
            raw_qid = item.get("question_id")
            q_id = str(raw_qid).strip() if raw_qid is not None else f"q_{idx + 1}"
            if not q_id:
                q_id = f"q_{idx + 1}"
            raw_q = item.get("question", "")
            q_text = str(raw_q) if raw_q is not None else ""
            parsed_questions.append((q_id, q_text))
        elif isinstance(item, str):
            parsed_questions.append((f"q_{idx + 1}", item))
        else:
            raise ValueError(
                f"Unsupported question item at index {idx} in '{path}': {type(item).__name__}"
            )

    return parsed_questions


class RAGCLI:
    """Python Fire Command-Line Interface router for RAG Against the Machine.
    Exposes the 6 mandatory commands required by the subject specification:
        1. ``index [--max_chunk_size 2000]``
        2. ``search <query> [--k 5]``
        3. ``search_dataset --dataset_path ... --save_directory ... [--k 10]``
        4. ``answer <query> [--k 5]``
        5. ``answer_dataset --student_search_results_path ... --save_directory ...``
        6. ``evaluate --student_search_results_path ... --dataset_path ... [--k 5]``
    """

    def __init__(
        self,
        *,
        corpus_dir: Union[str, Path] = DEFAULT_CORPUS_DIR,
        output_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        enable_dense: Optional[bool] = None,
        show_progress: bool = True,
        raise_on_error: bool = False,
        indexer: Optional[CorpusIndexer] = None,
        ranker: Optional[HybridRanker] = None,
        llm_engine: Optional[LLMInferenceEngine] = None,
        evaluator: Optional[RecallEvaluator] = None,
        _st_model: Any = None,
        _llm_model: Any = None,
        _llm_tokenizer: Any = None,
    ) -> None:
        """Initialize the CLI router with configurable paths and optional injected engines.
        All parameters are keyword-only so Python Fire never consumes positional
        subcommand names (`index`, `search`, `answer`, etc.) during class instantiation.

            corpus_dir: Root directory of source corpus (default: ``data/raw/vllm-0.10.1``).
            output_dir: Directory where index artifacts are stored (default: ``data/processed``).
            model_name: Hugging Face causal LM identifier (default: ``Qwen/Qwen3-0.6B``).
            device: Execution device string (default: ``"cpu"``).
            enable_dense: Whether to enable dense MiniLM embeddings (default: ``True``).
            show_progress: Whether to display ``tqdm`` progress bars (default: ``True``).
            raise_on_error: If True, re-raises exceptions instead of returning graceful
                error fallbacks. Defaults to ``False`` per subject CLI robustness rules.
            indexer: Optional pre-configured :class:`CorpusIndexer` instance.
            ranker: Optional pre-configured :class:`HybridRanker` instance.
            llm_engine: Optional pre-configured :class:`LLMInferenceEngine` instance.
            evaluator: Optional pre-configured :class:`RecallEvaluator` instance.
            _st_model: Optional mock SentenceTransformer model for offline testing.
            _llm_model: Optional mock causal LM for offline testing.
            _llm_tokenizer: Optional mock tokenizer for offline testing.
        """
        self.corpus_dir: Path = Path(corpus_dir)
        self.output_dir: Path = Path(output_dir)
        self.model_name: str = str(model_name).strip() or DEFAULT_MODEL_NAME
        self.device: str = str(device).strip() or DEFAULT_DEVICE
        self.enable_dense: bool = (
            bool(enable_dense) if enable_dense is not None else (_st_model is not None)
        )
        self.show_progress: bool = bool(show_progress)
        self.raise_on_error: bool = bool(raise_on_error)

        self._indexer: Optional[CorpusIndexer] = indexer
        self._ranker: Optional[HybridRanker] = ranker
        self._ranker_output_dir: Optional[Path] = None
        self._chunks: Optional[List[InternalChunk]] = (
            list(ranker.chunks)
            if (ranker is not None and ranker.chunks is not None)
            else None
        )
        self._llm_engine: Optional[LLMInferenceEngine] = llm_engine
        self._evaluator: Optional[RecallEvaluator] = evaluator

        self._st_model: Any = _st_model
        self._llm_model: Any = _llm_model
        self._llm_tokenizer: Any = _llm_tokenizer

    def _sanitize_max_chunk_size(self, max_chunk_size: Any) -> int:
        """Clamp *max_chunk_size* into ``[1, 2000]`` to guarantee moulinette validity."""
        try:
            size = int(max_chunk_size)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid max_chunk_size=%r; defaulting to %d.",
                max_chunk_size,
                DEFAULT_MAX_CHUNK_SIZE,
            )
            return DEFAULT_MAX_CHUNK_SIZE

        if size <= 0:
            logger.warning(
                "max_chunk_size=%d must be positive; clamping to %d.",
                size,
                DEFAULT_MAX_CHUNK_SIZE,
            )
            return DEFAULT_MAX_CHUNK_SIZE

        if size > DEFAULT_MAX_CHUNK_SIZE:
            logger.warning(
                "max_chunk_size=%d exceeds moulinette limit (%d); clamping to %d.",
                size,
                DEFAULT_MAX_CHUNK_SIZE,
                DEFAULT_MAX_CHUNK_SIZE,
            )
            return DEFAULT_MAX_CHUNK_SIZE

        return size

    def _parse_k(self, k: Any, default_k: int) -> Tuple[int, int]:
        """Return ``(effective_retrieval_k, schema_k)`` where ``schema_k >= 1``."""
        try:
            parsed = int(k)
        except (TypeError, ValueError):
            logger.warning("Invalid k=%r; defaulting to k=%d.", k, default_k)
            return default_k, max(1, default_k)

        if parsed <= 0:
            logger.warning(
                "Non-positive k=%d requested; returning 0 sources with schema k=1.",
                parsed,
            )
            return 0, 1

        return parsed, parsed

    def _get_or_load_ranker(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        enable_dense: Optional[bool] = None,
    ) -> Tuple[Optional[HybridRanker], Optional[List[InternalChunk]]]:
        """Return active :class:`HybridRanker` and chunk catalog, loading from disk if needed."""
        target_dir = Path(output_dir) if output_dir is not None else self.output_dir
        if enable_dense is not None:
            use_dense = bool(enable_dense)
        elif self.enable_dense:
            use_dense = True
        else:
            use_dense = (target_dir / "embeddings.npy").is_file()
        clamped_max_size = self._sanitize_max_chunk_size(max_chunk_size)

        if self._ranker is not None and (
            output_dir is None or self._ranker_output_dir == target_dir
        ):
            self._ranker.max_chunk_size = clamped_max_size
            return self._ranker, self._chunks

        chunks, bm25_index, dense_encoder = CorpusIndexer.load(
            output_dir=target_dir,
            load_dense=use_dense,
            device=self.device,
            _model=self._st_model,
        )
        ranker = HybridRanker(
            bm25_index=bm25_index,
            dense_encoder=dense_encoder,
            chunks=chunks,
            rrf_k=DEFAULT_RRF_K,
            max_chunk_size=clamped_max_size,
        )
        self._ranker = ranker
        self._ranker_output_dir = target_dir
        self._chunks = chunks
        return ranker, chunks

    def _get_or_create_llm_engine(
        self,
        model_name: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        chunks: Optional[Sequence[InternalChunk]] = None,
    ) -> LLMInferenceEngine:
        """Return active :class:`LLMInferenceEngine`, instantiating lazily if needed."""
        resolved_model = (
            str(model_name).strip()
            if (model_name is not None and str(model_name).strip())
            else self.model_name
        )
        active_chunks = list(chunks) if chunks is not None else self._chunks

        if self._llm_engine is not None:
            if (
                active_chunks is not None
                and self._llm_engine.prompt_builder.context_loader is not None
            ):
                self._llm_engine.prompt_builder.context_loader.register_chunks(
                    active_chunks
                )
            return self._llm_engine

        self._llm_engine = LLMInferenceEngine(
            model_name=resolved_model,
            max_new_tokens=max(1, int(max_new_tokens)),
            temperature=max(0.0, float(temperature)),
            device=self.device,
            chunks=active_chunks,
            _model=self._llm_model,
            _tokenizer=self._llm_tokenizer,
        )
        return self._llm_engine

    def index(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        corpus_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        *,
        enable_dense: Optional[bool] = None,
        show_progress: Optional[bool] = None,
    ) -> IndexingResult:
        """Ingest source corpus and build BM25 and dense indices under ``data/processed/``.
        CLI Usage:
            ``uv run python -m src index [--max_chunk_size 2000]``
            max_chunk_size: Maximum allowed character span per chunk (default: 2000).
            corpus_dir: Optional source directory override (default: ``data/raw/vllm-0.10.1``).
            output_dir: Optional destination directory override (default: ``data/processed``).
            enable_dense: Optional toggle for dense MiniLM embeddings (default: ``True``).
            show_progress: Optional toggle for ``tqdm`` progress bars (default: ``True``).
        """
        effective_max_size = self._sanitize_max_chunk_size(max_chunk_size)
        target_corpus_dir = (
            Path(corpus_dir) if corpus_dir is not None else self.corpus_dir
        )
        target_output_dir = (
            Path(output_dir) if output_dir is not None else self.output_dir
        )
        use_dense = self.enable_dense if enable_dense is None else bool(enable_dense)
        use_progress = (
            self.show_progress if show_progress is None else bool(show_progress)
        )

        try:
            indexer = (
                self._indexer
                if self._indexer is not None
                else CorpusIndexer(
                    corpus_dir=target_corpus_dir,
                    output_dir=target_output_dir,
                    max_chunk_size=effective_max_size,
                    device=self.device,
                    enable_dense=use_dense,
                    show_progress=use_progress,
                    _model=self._st_model,
                )
            )
            result = indexer.index(
                corpus_dir=target_corpus_dir,
                output_dir=target_output_dir,
                max_chunk_size=effective_max_size,
                show_progress=use_progress,
            )

            # Cache the freshly built ranker and chunks in memory
            self._chunks = list(result.chunks)
            self._ranker = HybridRanker(
                bm25_index=result.bm25_index,
                dense_encoder=result.dense_encoder,
                chunks=result.chunks,
                rrf_k=DEFAULT_RRF_K,
                max_chunk_size=effective_max_size,
            )
            self._ranker_output_dir = target_output_dir

            _emit_output(str(result))
            return result

        except Exception as exc:
            logger.error("Indexing failed: %s", exc)
            print(f"ERROR: Indexing failed: {exc}", file=sys.stderr)
            if self.raise_on_error:
                raise

            empty_bm25 = BM25Index.build([])
            fallback_result = IndexingResult(
                chunks=[],
                bm25_index=empty_bm25,
                dense_encoder=None,
                embeddings=None,
                num_files=0,
                num_chunks=0,
                chunks_path=target_output_dir / CHUNKS_FILENAME,
                bm25_index_path=target_output_dir / BM25_FILENAME,
                embeddings_path=None,
                elapsed_seconds=0.0,
            )
            return fallback_result

    def search(
        self,
        query: Any = "",
        k: int = DEFAULT_SEARCH_K,
        *,
        output_dir: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        save_directory: Optional[Union[str, Path]] = None,
        question_id: Optional[str] = None,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        enable_dense: Optional[bool] = None,
        return_json: bool = False,
    ) -> Union[CLISearchResult, str]:
        """Retrieve top-k candidate sources for a single query.
        CLI Usage:
            ``uv run python -m src search "<query>" [--k 5]``
            query: Search query string.
            k: Number of top candidate sources to retrieve (default: 5).
            output_dir: Optional directory containing built index artifacts.
            output_path: Optional file path to save ``StudentSearchResults`` JSON.
            save_directory: Optional directory to save ``StudentSearchResults`` JSON.
            question_id: Optional explicit question UUID/identifier.
            max_chunk_size: Maximum character span ceiling (default: 2000).
            enable_dense: Optional toggle for dense semantic retrieval.
            return_json: If True, returns the formatted JSON string directly.
        """
        raw_query_str = "" if query is None else str(query)
        clean_query = raw_query_str.strip()
        effective_k, schema_k = self._parse_k(k, default_k=DEFAULT_SEARCH_K)
        clamped_max_size = self._sanitize_max_chunk_size(max_chunk_size)

        retrieved_sources: List[MinimalSource] = []

        if not clean_query:
            logger.warning("Empty query provided to 'search'; returning 0 sources.")
        elif effective_k > 0:
            try:
                ranker, _ = self._get_or_load_ranker(
                    output_dir=output_dir,
                    max_chunk_size=clamped_max_size,
                    enable_dense=enable_dense,
                )
                if ranker is not None:
                    raw_sources = ranker.search(clean_query, k=effective_k)
                    retrieved_sources = [
                        clamp_source(src, max_chunk_size=clamped_max_size)
                        for src in raw_sources[:effective_k]
                    ]
            except Exception as exc:
                logger.error("Search failed for query %r: %s", clean_query, exc)
                print(f"ERROR: Search failed: {exc}", file=sys.stderr)
                if self.raise_on_error:
                    raise

        target_out_path = _resolve_output_file_path(
            save_directory=save_directory,
            output_path=output_path,
            source_path=None,
            default_filename="student_search_results.json",
        )

        result_kwargs: Dict[str, Any] = {
            "question": clean_query if clean_query else "[Empty question]",
            "retrieved_sources": retrieved_sources,
            "k": schema_k,
            "output_path": target_out_path,
        }
        if question_id is not None and str(question_id).strip():
            result_kwargs["question_id"] = str(question_id).strip()

        search_result = CLISearchResult(**result_kwargs)

        if target_out_path is not None:
            try:
                target_out_path.parent.mkdir(parents=True, exist_ok=True)
                batch_payload = json.loads(search_result.to_batch().model_dump_json())
                with open(target_out_path, "w", encoding="utf-8") as f:
                    json.dump(batch_payload, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.error("Failed to save search output to '%s': %s", target_out_path, exc)
                print(f"ERROR: Failed to save search output: {exc}", file=sys.stderr)
                if self.raise_on_error:
                    raise

        json_output = search_result.to_json(indent=2)
        _emit_output(json_output)

        if return_json:
            return json_output
        return search_result

    def search_dataset(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        save_directory: Optional[Union[str, Path]] = None,
        k: int = DEFAULT_DATASET_SEARCH_K,
        *,
        output_dir: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        enable_dense: Optional[bool] = None,
        show_progress: Optional[bool] = None,
        return_json: bool = False,
    ) -> Union[CLISearchDatasetResult, str]:
        """Run batch search over a dataset and write a ``StudentSearchResults`` JSON file.
        CLI Usage:
            ``uv run python -m src search_dataset --dataset_path <path> --save_directory <dir>``
            dataset_path: Path to input ``RagDataset`` JSON file.
            save_directory: Directory where the ``StudentSearchResults`` JSON file will be saved.
            k: Number of candidate sources retrieved per question (default: 10).
            output_dir: Optional directory containing built index artifacts
                (default: ``data/processed``).
            output_path: Optional explicit output JSON file path override.
            max_chunk_size: Hard character span ceiling (default: 2000).
            enable_dense: Optional toggle for dense semantic retrieval.
            show_progress: Optional toggle for ``tqdm`` progress bar.
            return_json: If True, returns the formatted JSON string.
        """
        effective_k, schema_k = self._parse_k(k, default_k=DEFAULT_DATASET_SEARCH_K)
        clamped_max_size = self._sanitize_max_chunk_size(max_chunk_size)
        use_progress = (
            self.show_progress if show_progress is None else bool(show_progress)
        )

        if dataset_path is None or not str(dataset_path).strip():
            msg = "ERROR: --dataset_path is required for 'search_dataset'."
            logger.error(msg)
            print(msg, file=sys.stderr)
            if self.raise_on_error:
                raise ValueError(msg)
            empty_res = CLISearchDatasetResult(search_results=[], k=schema_k)
            return empty_res.to_json(indent=2) if return_json else empty_res

        try:
            questions = _parse_dataset_questions(dataset_path)
        except Exception as exc:
            logger.error("Failed to load dataset '%s': %s", dataset_path, exc)
            print(f"ERROR: {exc}", file=sys.stderr)
            if self.raise_on_error:
                raise
            empty_res = CLISearchDatasetResult(search_results=[], k=schema_k)
            return empty_res.to_json(indent=2) if return_json else empty_res

        ranker: Optional[HybridRanker] = None
        if effective_k > 0 and len(questions) > 0:
            try:
                ranker, _ = self._get_or_load_ranker(
                    output_dir=output_dir,
                    max_chunk_size=clamped_max_size,
                    enable_dense=enable_dense,
                )
            except Exception as exc:
                logger.error("Failed to load index for search_dataset: %s", exc)
                print(f"ERROR: Failed to load index: {exc}", file=sys.stderr)
                if self.raise_on_error:
                    raise

        search_results_list: List[MinimalSearchResults] = []
        q_iter = tqdm(
            questions,
            desc="Searching dataset",
            unit="question",
            disable=not use_progress,
        )

        for q_id, q_text in q_iter:
            clean_q = q_text.strip()
            sources: List[MinimalSource] = []
            if clean_q and effective_k > 0 and ranker is not None:
                try:
                    raw_srcs = ranker.search(clean_q, k=effective_k)
                    sources = [
                        clamp_source(src, max_chunk_size=clamped_max_size)
                        for src in raw_srcs[:effective_k]
                    ]
                except Exception as exc:
                    logger.warning(
                        "Search failed for question '%s': %s", q_id, exc
                    )
                    sources = []

            search_results_list.append(
                MinimalSearchResults(
                    question_id=q_id,
                    question=clean_q if clean_q else "[Empty question]",
                    retrieved_sources=sources,
                )
            )

        target_out_path = _resolve_output_file_path(
            save_directory=save_directory,
            output_path=output_path,
            source_path=dataset_path,
            default_filename="student_search_results.json",
        )

        batch_result = CLISearchDatasetResult(
            search_results=search_results_list,
            k=schema_k,
            output_path=target_out_path,
        )

        if target_out_path is not None:
            try:
                target_out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_out_path, "w", encoding="utf-8") as f:
                    f.write(batch_result.to_json(indent=2))
                logger.info(
                    "Saved student_search_results (%d questions) to '%s'",
                    len(search_results_list),
                    target_out_path,
                )
            except Exception as exc:
                logger.error(
                    "Failed to write StudentSearchResults to '%s': %s",
                    target_out_path,
                    exc,
                )
                print(f"ERROR: Failed to write output file: {exc}", file=sys.stderr)
                if self.raise_on_error:
                    raise

        _emit_output(str(batch_result))

        if return_json:
            return batch_result.to_json(indent=2)
        return batch_result

    def answer(
        self,
        query: Any = "",
        k: int = DEFAULT_ANSWER_K,
        *,
        output_dir: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        save_directory: Optional[Union[str, Path]] = None,
        question_id: Optional[str] = None,
        model_name: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        enable_dense: Optional[bool] = None,
        return_json: bool = False,
    ) -> Union[CLIAnswerResult, str]:
        """Retrieve top-k sources and generate a grounded answer for a single query.
        CLI Usage:
            ``uv run python -m src answer "<query>" [--k 5]``
            query: Technical question string.
            k: Number of candidate sources to retrieve and ground (default: 5).
            output_dir: Optional directory containing built index artifacts.
            output_path: Optional file path to save ``StudentSearchResultsAndAnswer`` JSON.
            save_directory: Optional directory to save ``StudentSearchResultsAndAnswer`` JSON.
            question_id: Optional explicit question UUID/identifier.
            model_name: Optional Hugging Face model override (default: ``Qwen/Qwen3-0.6B``).
            max_new_tokens: Maximum generated tokens (default: 128).
            temperature: Decoding temperature (default: 0.0 for greedy decoding).
            max_chunk_size: Hard character span ceiling (default: 2000).
            enable_dense: Optional toggle for dense semantic retrieval.
            return_json: If True, returns the formatted JSON string directly.
        """
        raw_query_str = "" if query is None else str(query)
        clean_query = raw_query_str.strip()
        effective_k, schema_k = self._parse_k(k, default_k=DEFAULT_ANSWER_K)
        clamped_max_size = self._sanitize_max_chunk_size(max_chunk_size)

        retrieved_sources: List[MinimalSource] = []
        loaded_chunks: Optional[List[InternalChunk]] = self._chunks

        if not clean_query:
            logger.warning("Empty query provided to 'answer'; returning fallback answer.")
            generated_answer_text = EMPTY_QUERY_ANSWER
        else:
            if effective_k > 0:
                try:
                    ranker, loaded_chunks = self._get_or_load_ranker(
                        output_dir=output_dir,
                        max_chunk_size=clamped_max_size,
                        enable_dense=enable_dense,
                    )
                    if ranker is not None:
                        raw_sources = ranker.search(clean_query, k=effective_k)
                        retrieved_sources = [
                            clamp_source(src, max_chunk_size=clamped_max_size)
                            for src in raw_sources[:effective_k]
                        ]
                except Exception as exc:
                    logger.warning(
                        "Retrieval step in 'answer' failed (%s); continuing with empty context.",
                        exc,
                    )
                    if self.raise_on_error:
                        raise

            try:
                engine = self._get_or_create_llm_engine(
                    model_name=model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    chunks=loaded_chunks,
                )
                ans_model = engine.generate_answer(
                    question=clean_query,
                    retrieved_sources=retrieved_sources,
                    question_id=question_id,
                    k=effective_k if effective_k > 0 else None,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    return_json=False,
                )
                if isinstance(ans_model, MinimalAnswer):
                    generated_answer_text = ans_model.answer
                    retrieved_sources = list(ans_model.retrieved_sources)
                    if question_id is None:
                        question_id = ans_model.question_id
                else:
                    generated_answer_text = str(ans_model)
            except Exception as exc:
                logger.error("LLM answer generation failed: %s", exc)
                print(f"ERROR: Answer generation failed: {exc}", file=sys.stderr)
                if self.raise_on_error:
                    raise
                generated_answer_text = DEFAULT_FALLBACK_ANSWER

        target_out_path = _resolve_output_file_path(
            save_directory=save_directory,
            output_path=output_path,
            source_path=None,
            default_filename="student_search_results_and_answer.json",
        )

        ans_kwargs: Dict[str, Any] = {
            "question": clean_query if clean_query else "[Empty question]",
            "retrieved_sources": normalize_retrieved_sources(
                retrieved_sources,
                k=effective_k if effective_k > 0 else 0,
                max_chunk_chars=clamped_max_size,
            ),
            "answer": generated_answer_text,
            "k": schema_k,
            "output_path": target_out_path,
        }
        if question_id is not None and str(question_id).strip():
            ans_kwargs["question_id"] = str(question_id).strip()

        answer_result = CLIAnswerResult(**ans_kwargs)

        if target_out_path is not None:
            try:
                target_out_path.parent.mkdir(parents=True, exist_ok=True)
                batch_payload = json.loads(answer_result.to_batch().model_dump_json())
                with open(target_out_path, "w", encoding="utf-8") as f:
                    json.dump(batch_payload, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.error("Failed to save answer output to '%s': %s", target_out_path, exc)
                print(f"ERROR: Failed to save answer output: {exc}", file=sys.stderr)
                if self.raise_on_error:
                    raise

        json_output = answer_result.to_json(indent=2)
        _emit_output(json_output)

        if return_json:
            return json_output
        return answer_result

    def answer_dataset(
        self,
        student_search_results_path: Optional[Union[str, Path]] = None,
        save_directory: Optional[Union[str, Path]] = None,
        *,
        k: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        show_progress: Optional[bool] = None,
        return_json: bool = False,
    ) -> Union[CLIAnswerDatasetResult, str]:
        """Generate grounded answers for a dataset of search results.
        CLI Usage:
            ``uv run python -m src answer_dataset --student_search_results_path <p>``
            ``--save_directory <dir>``
            student_search_results_path: Path to input ``StudentSearchResults`` JSON file.
            save_directory: Directory where ``StudentSearchResultsAndAnswer`` JSON will be written.
            k: Optional top-k override. If omitted, inherits ``k`` from the input search results.
            output_dir: Optional directory containing ``chunks.json`` for fast span hydration.
            output_path: Optional explicit output JSON file path override.
            model_name: Optional Hugging Face model override (default: ``Qwen/Qwen3-0.6B``).
            max_new_tokens: Maximum generated tokens per answer (default: 128).
            temperature: Decoding temperature (default: 0.0).
            show_progress: Optional toggle for ``tqdm`` progress bar.
            return_json: If True, returns the formatted JSON string directly.
        """
        use_progress = (
            self.show_progress if show_progress is None else bool(show_progress)
        )
        fallback_k = max(1, int(k)) if (k is not None and int(k) >= 1) else DEFAULT_ANSWER_K

        if (
            student_search_results_path is None
            or not str(student_search_results_path).strip()
        ):
            msg = "ERROR: --student_search_results_path is required for 'answer_dataset'."
            logger.error(msg)
            print(msg, file=sys.stderr)
            if self.raise_on_error:
                raise ValueError(msg)
            empty_res = CLIAnswerDatasetResult(search_results=[], k=fallback_k)
            return empty_res.to_json(indent=2) if return_json else empty_res

        target_out_path = _resolve_output_file_path(
            save_directory=save_directory,
            output_path=output_path,
            source_path=student_search_results_path,
            default_filename="student_search_results_and_answer.json",
        )

        # Load chunks catalog if available so PromptBuilder can hydrate spans from memory
        loaded_chunks = self._chunks
        if loaded_chunks is None:
            target_proc_dir = (
                Path(output_dir) if output_dir is not None else self.output_dir
            )
            chunks_file = target_proc_dir / CHUNKS_FILENAME
            if chunks_file.is_file():
                try:
                    from src.indexing.indexer import load_chunks

                    loaded_chunks = load_chunks(chunks_file)
                    self._chunks = loaded_chunks
                except Exception:
                    loaded_chunks = None

        try:
            engine = self._get_or_create_llm_engine(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                chunks=loaded_chunks,
            )

            batch_model = engine.answer_dataset(
                student_search_results_path=student_search_results_path,
                output_path=target_out_path,
                k=k if (k is not None and int(k) >= 1) else None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                show_progress=use_progress,
                return_json=False,
            )
            assert isinstance(batch_model, StudentSearchResultsAndAnswer)

            cli_batch = CLIAnswerDatasetResult(
                search_results=list(batch_model.search_results),
                k=batch_model.k,
                output_path=target_out_path,
            )
            _emit_output(str(cli_batch))

            if return_json:
                return cli_batch.to_json(indent=2)
            return cli_batch

        except Exception as exc:
            logger.error("answer_dataset failed: %s", exc)
            print(f"ERROR: {exc}", file=sys.stderr)
            if self.raise_on_error:
                raise
            empty_res = CLIAnswerDatasetResult(search_results=[], k=fallback_k)
            return empty_res.to_json(indent=2) if return_json else empty_res

    def evaluate(
        self,
        student_search_results_path: Optional[Union[str, Path]] = None,
        dataset_path: Optional[Union[str, Path]] = None,
        k: int = DEFAULT_EVALUATE_K,
        *,
        max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        output_path: Optional[Union[str, Path]] = None,
    ) -> EvaluationResult:
        """Evaluate student search results against a ground-truth dataset (IoU >= 0.05).
        CLI Usage:
            ``uv run python -m src evaluate --student_search_results_path <p>``
            ``--dataset_path <p> [--k 5]``
            student_search_results_path: Path to student ``StudentSearchResults`` JSON file.
            dataset_path: Path to ground-truth ``AnsweredQuestions`` ``RagDataset`` JSON file.
            k: Primary Recall@k cutoff to report (default: 5).
            max_context_length: Maximum allowed character span length (default: 2000).
            iou_threshold: Minimum character-span IoU required for a match (default: 0.05).
            output_path: Optional file path to save the evaluation report JSON.
        """
        _, effective_k = self._parse_k(k, default_k=DEFAULT_EVALUATE_K)
        clamped_max_len = self._sanitize_max_chunk_size(max_context_length)

        try:
            effective_iou = float(iou_threshold)
            if effective_iou < 0.0 or effective_iou > 1.0:
                logger.warning(
                    "Invalid iou_threshold=%f; clamping to %f.",
                    effective_iou,
                    DEFAULT_IOU_THRESHOLD,
                )
                effective_iou = DEFAULT_IOU_THRESHOLD
        except (TypeError, ValueError):
            effective_iou = DEFAULT_IOU_THRESHOLD

        if (
            student_search_results_path is None
            or not str(student_search_results_path).strip()
        ):
            msg = "ERROR: --student_search_results_path is required for 'evaluate'."
            logger.error(msg)
            print(msg, file=sys.stderr)
            if self.raise_on_error:
                raise ValueError(msg)
            return EvaluationResult(
                k=effective_k,
                is_valid=False,
                validation_errors=[msg],
                summary_table=msg,
            )

        if dataset_path is None or not str(dataset_path).strip():
            msg = "ERROR: --dataset_path is required for 'evaluate'."
            logger.error(msg)
            print(msg, file=sys.stderr)
            if self.raise_on_error:
                raise ValueError(msg)
            return EvaluationResult(
                k=effective_k,
                is_valid=False,
                validation_errors=[msg],
                summary_table=msg,
            )

        try:
            evaluator = (
                self._evaluator
                if self._evaluator is not None
                else RecallEvaluator(
                    iou_threshold=effective_iou,
                    max_context_length=clamped_max_len,
                    default_k=effective_k,
                )
            )
            report = evaluator.evaluate(
                student_search_results_path=student_search_results_path,
                dataset_path=dataset_path,
                k=effective_k,
                max_context_length=clamped_max_len,
                iou_threshold=effective_iou,
            )

            if output_path is not None and str(output_path).strip():
                out_file = Path(str(output_path).strip())
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

            _emit_output(str(report))
            return report

        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            print(f"ERROR: Evaluation failed: {exc}", file=sys.stderr)
            if self.raise_on_error:
                raise
            return EvaluationResult(
                k=effective_k,
                is_valid=False,
                validation_errors=[str(exc)],
                summary_table=f"Student data is valid: False\nERROR: {exc}",
            )


def run_cli(
    command: Optional[Union[Sequence[str], str]] = None,
    cli: Optional[RAGCLI] = None,
) -> Any:
    """Dispatch CLI arguments through Python Fire on :class:`RAGCLI`.
        command: Optional argument list or command string. If ``None``, reads
            ``sys.argv[1:]``.
        cli: Optional pre-configured :class:`RAGCLI` instance.
    """
    component: Any = cli if cli is not None else RAGCLI()
    if isinstance(command, Sequence) and not isinstance(command, str):
        cmd_arg: Optional[Union[List[str], str]] = [str(arg) for arg in command]
    else:
        cmd_arg = command
    return fire.Fire(component, command=cmd_arg, name="python -m src")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute the primary CLI entrypoint for ``uv run python -m src <command>``.
    Catches ``FireExit``, ``SystemExit``, and unexpected exceptions to ensure the
    CLI never crashes with an unhandled traceback on degenerate inputs.
    argv: Optional list of CLI arguments (defaults to ``sys.argv[1:]``).
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    try:
        run_cli(command=args)
        return 0
    except SystemExit as exc:
        code = exc.code
        if code is None or code == 0:
            return 0
        return int(code) if isinstance(code, int) else 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.error("Unhandled CLI error: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
