"""Context Grounding & Prompt Formatting.
Formats retrieved source chunks into a grounded instruction prompt for
Qwen/Qwen3-0.6B that binds generated statements to verifiable file paths
and character spans while constraining prompt token length within Qwen3's
sub-second CPU budget (2k-4k tokens).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import re
import tempfile
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from src.indexing.schema import InternalChunk
from src.models import MinimalSearchResults, MinimalSource, normalize_file_path

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME: str = "Qwen/Qwen3-0.6B"

# Qwen3 sub-second CPU token budget bounds (2k-4k tokens)
MIN_TOKEN_BUDGET: int = 2000
MAX_TOKEN_BUDGET: int = 4000
DEFAULT_MAX_PROMPT_TOKENS: int = 3000
DEFAULT_TOKEN_BUDGET: int = DEFAULT_MAX_PROMPT_TOKENS
DEFAULT_MAX_CONTEXT_TOKENS: int = 2600
DEFAULT_RESERVED_PROMPT_TOKENS: int = 350

# Source & chunk limits
DEFAULT_MAX_CHUNK_CHARS: int = 2000
DEFAULT_TOP_K: int = 5
DEFAULT_CHARS_PER_TOKEN: float = 4.0

# Default paths
DEFAULT_CORPUS_ROOT: Path = Path("data/raw")
DEFAULT_CHUNKS_PATH: Path = Path("data/processed/chunks.json")

DEFAULT_SYSTEM_PROMPT: str = (
    "You are a precise technical assistant for the vLLM codebase and documentation.\n"
    "Answer the user's question strictly using ONLY the retrieved source chunks "
    "provided in the context below.\n"
    "Grounding Rules:\n"
    "1. Base every statement directly on the provided [Source N] blocks.\n"
    "2. Do NOT hallucinate, guess, or invent APIs, classes, functions, CLI flags, "
    "or behavior not supported by the retrieved sources.\n"
    "3. Reference exact identifiers, parameters, or file paths from the sources "
    "when relevant.\n"
    "4. If the retrieved sources do not contain sufficient information to answer "
    "the question, state clearly that the answer cannot be determined from the "
    "provided context.\n"
    "5. Provide a concise, direct, and technically accurate answer."
)

SYSTEM_PROMPT: str = DEFAULT_SYSTEM_PROMPT

NO_CONTEXT_PLACEHOLDER: str = (
    "[No relevant source chunks were retrieved from the corpus.]"
)

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(
    text: str,
    tokenizer: Optional[Any] = None,
) -> int:
    """Estimate or compute the token count of *text*.
    If a Hugging Face tokenizer (with ``.encode``) or a callable tokenizer
    is supplied, uses it directly. Otherwise, computes a fast, deterministic
    upper-bound token estimate combining code-aware regex splitting and
    character-to-token ratio (1 token per ~4 characters) suitable for Qwen3's
    BPE vocabulary on CPU.
        text: Input string to measure.
        tokenizer: Optional Hugging Face tokenizer or callable ``str -> Sequence``
            or ``str -> int``.
    """
    if not text or not isinstance(text, str):
        return 0

    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
                try:
                    encoded = tokenizer.encode(text, add_special_tokens=False)
                except TypeError:
                    encoded = tokenizer.encode(text)
                if isinstance(encoded, Sequence):
                    return len(encoded)
            if callable(tokenizer):
                result = tokenizer(text)
                if isinstance(result, int):
                    return max(0, result)
                if isinstance(result, Mapping) and "input_ids" in result:
                    input_ids = result["input_ids"]
                    if isinstance(input_ids, Sequence):
                        if len(input_ids) > 0 and isinstance(
                            input_ids[0], Sequence
                        ):
                            return len(input_ids[0])
                        return len(input_ids)
                if isinstance(result, Sequence):
                    return len(result)
        except Exception as exc:
            logger.debug(
                "Custom tokenizer failed, falling back to heuristic: %s", exc
            )

    if not text.strip():
        return 0

    regex_count = len(_TOKEN_PATTERN.findall(text))
    char_count = math.ceil(len(text) / DEFAULT_CHARS_PER_TOKEN)
    return max(1, regex_count, char_count)


def truncate_to_token_budget(
    text: str,
    max_tokens: int,
    tokenizer: Optional[Any] = None,
    suffix: str = "",
) -> str:
    """Truncate *text* so its token count is strictly ``<= max_tokens``.
    Preserves line or word boundaries when possible while guaranteeing that
    ``estimate_tokens(result, tokenizer) <= max_tokens``.
        text: Input string to constrain.
        max_tokens: Maximum allowed token count (must be >= 0).
        tokenizer: Optional custom tokenizer or callable.
        suffix: Optional truncation indicator (e.g. ``"..."``) appended if budget allows.
    """
    if max_tokens <= 0 or not text:
        return ""

    if estimate_tokens(text, tokenizer=tokenizer) <= max_tokens:
        return text

    suffix_tokens = (
        estimate_tokens(suffix, tokenizer=tokenizer) if suffix else 0
    )
    use_suffix = bool(suffix) and max_tokens > suffix_tokens + 2
    target_tokens = max_tokens - suffix_tokens if use_suffix else max_tokens

    # If a Hugging Face tokenizer with encode/decode is provided
    if (
        tokenizer is not None
        and hasattr(tokenizer, "encode")
        and hasattr(tokenizer, "decode")
        and callable(tokenizer.encode)
        and callable(tokenizer.decode)
    ):
        try:
            try:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                token_ids = tokenizer.encode(text)
            if isinstance(token_ids, list) and len(token_ids) > target_tokens:
                truncated_ids = token_ids[:target_tokens]
                decoded = str(
                    tokenizer.decode(truncated_ids, skip_special_tokens=True)
                )
                candidate = (
                    (decoded.rstrip() + suffix) if use_suffix else decoded
                )
                if (
                    estimate_tokens(candidate, tokenizer=tokenizer)
                    <= max_tokens
                ):
                    return candidate
        except Exception as exc:
            logger.debug("Tokenizer decode truncation fallback: %s", exc)

    # Step 1: Enforce hard character ceiling corresponding to target_tokens
    max_chars = int(target_tokens * int(DEFAULT_CHARS_PER_TOKEN))
    working = text[:max_chars] if len(text) > max_chars else text

    # Step 2: Enforce regex token count ceiling
    matches = list(_TOKEN_PATTERN.finditer(working))
    if len(matches) > target_tokens:
        cut_pos = matches[target_tokens - 1].end()
        working = working[:cut_pos]
    elif len(working) < len(text) and len(working) > 16:
        # Cleanly break at last newline or space if near the end of the slice
        min_keep = int(len(working) * 0.8)
        nl_pos = working.rfind("\n")
        if nl_pos >= min_keep:
            working = working[:nl_pos]
        else:
            sp_pos = working.rfind(" ")
            if sp_pos >= min_keep:
                working = working[:sp_pos]

    working = working.rstrip()
    result = f"{working}{suffix}" if (use_suffix and working) else working

    # Final safety loop for custom callable tokenizers
    while result and estimate_tokens(result, tokenizer=tokenizer) > max_tokens:
        if use_suffix and result.endswith(suffix):
            result = result[: -len(suffix)].rstrip()
            use_suffix = False
            continue
        cut = max(1, int(len(result) * 0.85))
        if cut >= len(result):
            result = result[:-1]
        else:
            result = result[:cut].rstrip()

    return result


@dataclass(frozen=True)
class GroundedSource:
    """Represents a retrieved source chunk hydrated with content and header metadata."""

    index: int
    file_path: str
    first_character_index: int
    last_character_index: int
    content: str
    token_count: int = 0
    truncated: bool = False

    @property
    def span_length(self) -> int:
        """Return character span length."""
        return max(0, self.last_character_index - self.first_character_index)

    @property
    def header(self) -> str:
        """Return explicit file path and character span header."""
        return format_source_header(
            file_path=self.file_path,
            first_character_index=self.first_character_index,
            last_character_index=self.last_character_index,
            index=self.index,
        )

    @property
    def formatted(self) -> str:
        """Return formatted source block with header and grounded content."""
        if self.content:
            return f"{self.header}\n{self.content}"
        return self.header

    def format_header(self) -> str:
        """Return formatted source header string."""
        return self.header

    def format_block(self) -> str:
        """Return formatted source block string."""
        return self.formatted

    def to_minimal_source(self) -> MinimalSource:
        """Convert to a validated MinimalSource instance."""
        last_idx = max(
            self.first_character_index + 1, self.last_character_index
        )
        return MinimalSource(
            file_path=self.file_path,
            first_character_index=max(0, self.first_character_index),
            last_character_index=last_idx,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize grounded source metadata to a dictionary."""
        return {
            "index": self.index,
            "file_path": self.file_path,
            "first_character_index": self.first_character_index,
            "last_character_index": self.last_character_index,
            "content": self.content,
            "token_count": self.token_count,
            "truncated": self.truncated,
            "header": self.header,
        }


def _unwrap_source_item(item: Any) -> Tuple[Any, Optional[str]]:
    """Unwrap scored or content-paired tuples ``(source, score_or_content)``."""
    if isinstance(item, tuple) and len(item) == 2:
        first, second = item
        if isinstance(second, str) and not isinstance(first, str):
            return first, second
        if isinstance(second, (int, float)) and not isinstance(second, bool):
            return first, None
    return item, None


def extract_source_metadata(
    source: Any,
    default_index: int = 1,
    fallback_content: Optional[str] = None,
) -> Tuple[str, int, int, Optional[str]]:
    """Extract ``(file_path, first_char_idx, last_char_idx, content_or_none)`` from *source*.
    Supports :class:`MinimalSource`, :class:`InternalChunk`, :class:`GroundedSource`,
    dictionaries, ``(source, score)`` tuples, and plain strings.
        source: Source object to inspect.
        default_index: 1-based source number used when synthesizing fallback paths.
        fallback_content: Optional explicit text snippet paired with the source.
    """
    unwrapped, paired_text = _unwrap_source_item(source)
    explicit_text = (
        fallback_content if fallback_content is not None else paired_text
    )

    if isinstance(unwrapped, GroundedSource):
        text = (
            explicit_text if explicit_text is not None else unwrapped.content
        )
        return (
            unwrapped.file_path,
            unwrapped.first_character_index,
            unwrapped.last_character_index,
            text,
        )

    if isinstance(unwrapped, InternalChunk):
        text = (
            explicit_text if explicit_text is not None else unwrapped.content
        )
        return (
            unwrapped.file_path,
            unwrapped.first_character_index,
            unwrapped.last_character_index,
            text,
        )

    if isinstance(unwrapped, MinimalSource):
        return (
            unwrapped.file_path,
            unwrapped.first_character_index,
            unwrapped.last_character_index,
            explicit_text,
        )

    if isinstance(unwrapped, Mapping):
        raw_path = str(
            unwrapped.get("file_path")
            or unwrapped.get("path")
            or unwrapped.get("source")
            or f"context_snippet_{default_index}.txt"
        )
        try:
            norm_path = normalize_file_path(raw_path)
        except ValueError:
            norm_path = f"context_snippet_{default_index}.txt"

        raw_content: Optional[str] = explicit_text
        if raw_content is None:
            for key in ("content", "text", "snippet", "chunk"):
                val = unwrapped.get(key)
                if isinstance(val, str):
                    raw_content = val
                    break

        first_idx = int(
            unwrapped.get(
                "first_character_index",
                unwrapped.get("start_char", unwrapped.get("start", 0)),
            )
        )
        first_idx = max(0, first_idx)

        default_last = first_idx + (len(raw_content) if raw_content else 1)
        last_idx = int(
            unwrapped.get(
                "last_character_index",
                unwrapped.get("end_char", unwrapped.get("end", default_last)),
            )
        )
        if last_idx <= first_idx:
            last_idx = first_idx + max(
                1, len(raw_content) if raw_content else 1
            )

        return norm_path, first_idx, last_idx, raw_content

    if isinstance(unwrapped, str):
        text_val = explicit_text if explicit_text is not None else unwrapped
        end_idx = max(1, len(text_val))
        return f"context_snippet_{default_index}.txt", 0, end_idx, text_val

    if (
        hasattr(unwrapped, "file_path")
        and hasattr(unwrapped, "first_character_index")
        and hasattr(unwrapped, "last_character_index")
    ):
        raw_path = str(getattr(unwrapped, "file_path"))
        try:
            norm_path = normalize_file_path(raw_path)
        except ValueError:
            norm_path = raw_path
        first_idx = max(0, int(getattr(unwrapped, "first_character_index")))
        last_idx = max(
            first_idx + 1, int(getattr(unwrapped, "last_character_index"))
        )
        obj_content = getattr(unwrapped, "content", None)
        text_out = (
            explicit_text
            if explicit_text is not None
            else (str(obj_content) if obj_content is not None else None)
        )
        return norm_path, first_idx, last_idx, text_out

    text_str = explicit_text if explicit_text is not None else str(unwrapped)
    return (
        f"context_snippet_{default_index}.txt",
        0,
        max(1, len(text_str)),
        text_str,
    )


def format_source_header(
    source: Optional[Any] = None,
    index: int = 1,
    *,
    file_path: Optional[str] = None,
    first_character_index: Optional[int] = None,
    last_character_index: Optional[int] = None,
) -> str:
    """Format an explicit grounding header with source rank, file path, and character span.
    Produces a header of the form:
        ``[Source 1] file_path: data/raw/vllm-0.10.1/... | character_span: 100-250 (chars 100:250)``
        source: Optional MinimalSource, InternalChunk, GroundedSource, or dict.
        index: 1-based source rank number (default: 1).
        file_path: Explicit file path override if *source* is omitted.
        first_character_index: Explicit start character index override.
        last_character_index: Explicit end character index override.

    """
    if source is not None and not isinstance(source, str):
        extracted_path, extracted_first, extracted_last, _ = (
            extract_source_metadata(source, default_index=index)
        )
        resolved_path = file_path if file_path is not None else extracted_path
        resolved_first = (
            first_character_index
            if first_character_index is not None
            else extracted_first
        )
        resolved_last = (
            last_character_index
            if last_character_index is not None
            else extracted_last
        )
    else:
        raw_path = (
            file_path
            if file_path is not None
            else (
                source
                if isinstance(source, str)
                else f"context_snippet_{index}.txt"
            )
        )
        try:
            resolved_path = normalize_file_path(raw_path)
        except ValueError:
            resolved_path = (
                str(raw_path).strip() or f"context_snippet_{index}.txt"
            )
        resolved_first = (
            max(0, int(first_character_index))
            if first_character_index is not None
            else 0
        )
        resolved_last = (
            max(resolved_first + 1, int(last_character_index))
            if last_character_index is not None
            else resolved_first + 1
        )

    return (
        f"[Source {index}] "
        f"file_path: {resolved_path} | "
        f"character_span: {resolved_first}-{resolved_last} "
        f"(chars {resolved_first}:{resolved_last})"
    )


def format_source_block(
    source: Any,
    content: Optional[str] = None,
    index: int = 1,
    max_chars: Optional[int] = DEFAULT_MAX_CHUNK_CHARS,
) -> str:
    """Format a single retrieved source with its explicit header and text content.
    source: Source metadata object (MinimalSource, InternalChunk, dict, or str).
    content: Optional explicit text snippet override.
    index: 1-based source rank number (default: 1).
    max_chars: Optional character ceiling for the snippet body (default: 2000).
    """
    file_path, first_idx, last_idx, extracted_content = (
        extract_source_metadata(
            source, default_index=index, fallback_content=content
        )
    )
    body = content if content is not None else (extracted_content or "")
    if max_chars is not None and max_chars > 0 and len(body) > max_chars:
        body = body[:max_chars]

    header = format_source_header(
        file_path=file_path,
        first_character_index=first_idx,
        last_character_index=last_idx,
        index=index,
    )
    body_cleaned = body.strip("\r\n")
    if body_cleaned:
        return f"{header}\n{body_cleaned}"
    return header


def resolve_safe_path(
    file_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
    allowed_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Sanitize and resolve *file_path* while enforcing path traversal mitigation.
    Security (Blocking Hackers/Bad Inputs): It blocks anyone from using ".." or outside paths
    (like "/etc/passwd") to escape your project folder and read private files on your computer.
    file_path: Relative or candidate file path to resolve.
    Smart File Finder (Fixing Messy Paths): Even with safe files, different
    scripts and pytest tests look for files from different starting folders.
    This function automatically fixes common path mix-ups—like accidentally doubling
    base_dir: Base directory from which relative paths are resolved
        (defaults to current working directory).
    allowed_root: Explicit root boundary that the resolved file must stay
        within. Defaults to *base_dir* (or current working directory).
    """
    raw_str = str(file_path).strip()
    if not raw_str:
        raise ValueError("file_path cannot be empty")
    if "\x00" in raw_str:
        raise ValueError("Invalid null byte in file_path")

    clean_rel = raw_str.replace("\\", "/")
    parts = [p for p in clean_rel.split("/") if p]
    if ".." in parts:
        raise ValueError(
            f"Path traversal detected in file_path '{file_path}': '..' is not allowed"
        )

    effective_base = (
        Path(base_dir).resolve()
        if base_dir is not None
        else Path.cwd().resolve()
    )
    effective_root = (
        Path(allowed_root).resolve()
        if allowed_root is not None
        else effective_base
    )
    temp_root = Path(tempfile.gettempdir()).resolve()

    if clean_rel.startswith("/"):
        abs_explicit = Path(clean_rel).resolve()
        in_allowed_root = False
        try:
            if os.path.commonpath(
                [str(effective_root), str(abs_explicit)]
            ) == str(effective_root):
                in_allowed_root = True
        except ValueError:
            pass
        if not in_allowed_root and allowed_root is None and base_dir is None:
            try:
                if os.path.commonpath(
                    [str(temp_root), str(abs_explicit)]
                ) == str(temp_root):
                    in_allowed_root = True
            except ValueError:
                pass
        if not in_allowed_root:
            raise ValueError(
                f"Path traversal detected: '{file_path}' escapes allowed root "
                f"'{effective_root}'"
            )

    candidates: List[Path] = []
    rel_path = Path(clean_rel.lstrip("/"))

    # 1. Direct resolution under effective_base
    candidates.append((effective_base / rel_path).resolve())

    # 2. If base_dir already points inside data/raw/vllm-0.10.1 and file_path
    #    starts with data/raw/vllm-0.10.1/, resolve relative to base_dir's suffix match
    base_posix = effective_base.as_posix()
    rel_posix = rel_path.as_posix()
    for marker in ("data/raw/vllm-0.10.1/", "data/raw/"):
        if rel_posix.startswith(marker) and base_posix.endswith(
            marker.rstrip("/")
        ):
            stripped = rel_posix[len(marker) :]
            if stripped:
                candidates.append((effective_base / stripped).resolve())

    # 3. Project cwd fallback when base_dir was not explicitly restricted
    cwd_resolved = Path.cwd().resolve()
    if effective_base != cwd_resolved:
        candidates.append((cwd_resolved / rel_path).resolve())

    # 4. Handle pytest tmp_path paths whose leading '/' was stripped by normalize_file_path
    abs_candidate = Path("/" + clean_rel.lstrip("/")).resolve()
    if allowed_root is None and base_dir is None:
        try:
            if os.path.commonpath([str(temp_root), str(abs_candidate)]) == str(
                temp_root
            ):
                candidates.append(abs_candidate)
        except ValueError:
            pass
    elif base_dir is not None:
        try:
            if os.path.commonpath(
                [str(effective_base), str(abs_candidate)]
            ) == str(effective_base):
                candidates.append(abs_candidate)
        except ValueError:
            pass

    allowed_roots: List[str] = [str(effective_root)]
    if allowed_root is None and base_dir is None:
        allowed_roots.append(str(temp_root))

    verified_candidates: List[Path] = []
    for candidate in candidates:
        candidate_str = str(candidate)
        for root_str in allowed_roots:
            try:
                if os.path.commonpath([root_str, candidate_str]) == root_str:
                    verified_candidates.append(candidate)
                    break
            except ValueError:
                continue

    if not verified_candidates:
        raise ValueError(
            f"Path traversal blocked: '{file_path}' resolves outside "
            f"allowed root '{effective_root}'"
        )

    for candidate in verified_candidates:
        if candidate.is_file():
            return candidate

    return verified_candidates[0]


class ContextLoader:
    """Hydrates retrieved MinimalSource locations with exact character spans from disk or catalog.
    Implements in-memory file caching, optional :class:`InternalChunk` catalog lookup,
    ``os.path.commonpath`` path traversal protection, and character span clamping.
    """

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        *,
        corpus_root: Optional[Union[str, Path]] = None,
        chunks: Optional[Sequence[InternalChunk]] = None,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_CHARS,
    ) -> None:
        """Initialize the ContextLoader.
        base_dir: Optional root directory for resolving relative source paths.
        corpus_root: Optional explicit security boundary directory.
        chunks: Optional pre-loaded sequence of InternalChunk records for fast
            in-memory span lookup without disk I/O.
        max_chunk_size: Hard ceiling on extracted character span length (default: 2000).
        """
        if max_chunk_size <= 0:
            raise ValueError(
                f"max_chunk_size must be positive, got {max_chunk_size}"
            )

        self.base_dir: Optional[Path] = (
            Path(base_dir) if base_dir is not None else None
        )
        self.corpus_root: Optional[Path] = (
            Path(corpus_root) if corpus_root is not None else None
        )
        self.max_chunk_size: int = int(max_chunk_size)

        self._exact_chunk_map: Dict[Tuple[str, int, int], str] = {}
        self._file_chunks_map: Dict[str, List[InternalChunk]] = {}
        self._file_cache: Dict[Path, str] = {}

        if chunks is not None:
            self.register_chunks(chunks)

    def register_chunks(self, chunks: Sequence[InternalChunk]) -> None:
        """Index a collection of InternalChunk objects for O(1) in-memory span lookup."""
        for chunk in chunks:
            key = (
                chunk.file_path,
                chunk.first_character_index,
                chunk.last_character_index,
            )
            self._exact_chunk_map[key] = chunk.content
            self._file_chunks_map.setdefault(chunk.file_path, []).append(chunk)

    def _lookup_in_catalog(
        self,
        file_path: str,
        first_idx: int,
        last_idx: int,
    ) -> Optional[str]:
        """Look up character span content in the registered InternalChunk catalog."""
        exact = self._exact_chunk_map.get((file_path, first_idx, last_idx))
        if exact is not None:
            return exact[: self.max_chunk_size]

        file_chunks = self._file_chunks_map.get(file_path)
        if not file_chunks:
            return None

        for chunk in file_chunks:
            if (
                chunk.first_character_index <= first_idx
                and chunk.last_character_index >= last_idx
            ):
                rel_start = first_idx - chunk.first_character_index
                rel_end = last_idx - chunk.first_character_index
                if rel_end <= len(chunk.content):
                    return chunk.content[rel_start:rel_end][
                        : self.max_chunk_size
                    ]

        return None

    def _read_file_cached(self, resolved_path: Path) -> str:
        """Read a UTF-8 source file using a context manager and cache its contents."""
        if resolved_path in self._file_cache:
            return self._file_cache[resolved_path]

        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(
                resolved_path, "r", encoding="utf-8", errors="replace"
            ) as f:
                text = f.read()

        self._file_cache[resolved_path] = text
        return text

    def load_span(
        self,
        source: Any,
        *,
        fallback_content: Optional[str] = None,
        strict: bool = False,
    ) -> str:
        """Extract the exact character span ``[first_character_index:last_character_index]``.
        Checks in order:
            1. Inline content attached to *source* (e.g. ``InternalChunk.content``)
            2. Registered in-memory ``chunks`` catalog
            3. Raw file on disk under ``[first_character_index:last_character_index]``
            source: MinimalSource, InternalChunk, GroundedSource, dict, or str.
            fallback_content: Optional explicit content string.
            strict: If True, raises FileNotFoundError/ValueError when disk file is missing.
        """
        file_path, first_idx, last_idx, inline_content = (
            extract_source_metadata(source, fallback_content=fallback_content)
        )
        effective_last = min(last_idx, first_idx + self.max_chunk_size)

        if inline_content is not None and inline_content != "":
            return inline_content[: self.max_chunk_size]

        catalog_hit = self._lookup_in_catalog(
            file_path, first_idx, effective_last
        )
        if catalog_hit is not None:
            return catalog_hit

        try:
            resolved = resolve_safe_path(
                file_path,
                base_dir=self.base_dir,
                allowed_root=self.corpus_root,
            )
            if not resolved.is_file():
                if strict:
                    raise FileNotFoundError(
                        f"Source file not found: {file_path}"
                    )
                logger.debug(
                    "Source file '%s' not found on disk; using empty span.",
                    file_path,
                )
                return ""
            file_text = self._read_file_cached(resolved)
            snippet = file_text[first_idx:effective_last]
            return snippet[: self.max_chunk_size]
        except Exception as exc:
            if strict:
                raise
            logger.warning("Could not load span from '%s': %s", file_path, exc)
            return ""

    def load_sources(
        self,
        sources: Sequence[Any],
        *,
        context_snippets: Optional[Sequence[str]] = None,
        k: Optional[int] = None,
    ) -> List[GroundedSource]:
        """Hydrate top-*k* sources into :class:`GroundedSource` instances."""
        if k is not None and k <= 0:
            return []

        selected = list(sources[:k]) if k is not None else list(sources)
        grounded_list: List[GroundedSource] = []

        for idx, item in enumerate(selected):
            rank = idx + 1
            paired_snippet: Optional[str] = None
            if context_snippets is not None and idx < len(context_snippets):
                paired_snippet = context_snippets[idx]

            file_path, first_idx, last_idx, inline_content = (
                extract_source_metadata(
                    item, default_index=rank, fallback_content=paired_snippet
                )
            )
            content = (
                inline_content
                if inline_content is not None
                else self.load_span(item, fallback_content=paired_snippet)
            )
            clamped_content = content[: self.max_chunk_size].strip("\r\n")
            clamped_last = min(last_idx, first_idx + self.max_chunk_size)
            if clamped_last <= first_idx:
                clamped_last = first_idx + max(1, len(clamped_content))

            grounded_list.append(
                GroundedSource(
                    index=rank,
                    file_path=file_path,
                    first_character_index=first_idx,
                    last_character_index=clamped_last,
                    content=clamped_content,
                    token_count=estimate_tokens(clamped_content),
                    truncated=len(content) > self.max_chunk_size,
                )
            )

        return grounded_list


def load_source_span(
    source: Any,
    base_dir: Optional[Union[str, Path]] = None,
    chunks: Optional[Sequence[InternalChunk]] = None,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_CHARS,
    strict: bool = False,
) -> str:
    """Convenience function to load the character span text for a single *source*."""
    loader = ContextLoader(
        base_dir=base_dir,
        chunks=chunks,
        max_chunk_size=max_chunk_size,
    )
    return loader.load_span(source, strict=strict)


class PromptBuilder:
    """Formats retrieved source chunks into a grounded instruction prompt for Qwen3-0.6B.
    Responsibilities:
        1. Injects top-*k* retrieved sources with explicit ``file_path`` and
           ``first_character_index`` / ``last_character_index`` span headers.
        2. Hydrates ``MinimalSource`` spans from disk or ``InternalChunk`` catalog.
        3. Constrains total prompt token length strictly within Qwen3's sub-second
           CPU token budget (2k-4k tokens, default 3000 tokens).
        4. Enforces strict grounding instructions to prevent model hallucinations.
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
        *,
        max_prompt_tokens: Optional[int] = None,
        token_budget: Optional[int] = None,
        max_context_tokens: Optional[int] = None,
        top_k: int = DEFAULT_TOP_K,
        k: Optional[int] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        tokenizer: Optional[Any] = None,
        context_loader: Optional[ContextLoader] = None,
        chunks: Optional[Sequence[InternalChunk]] = None,
        base_dir: Optional[Union[str, Path]] = None,
        use_chat_template: bool = False,
        enable_thinking: bool = False,
    ) -> None:
        """Initialize the PromptBuilder.
        max_tokens: Maximum total prompt token budget (default: 3000, within 2k-4k SLA).
        max_prompt_tokens: Optional keyword alias for *max_tokens*.
        token_budget: Optional keyword alias for *max_tokens*.
        max_context_tokens: Optional explicit cap on tokens allocated to context chunks.
        top_k: Default maximum number of retrieved sources to inject (default: 5).
        k: Optional keyword alias for *top_k*.
        system_prompt: Grounding system instruction text.
        max_chunk_chars: Hard character ceiling per source chunk (default: 2000).
        tokenizer: Optional Hugging Face tokenizer or callable for token counting.
        context_loader: Optional pre-configured ContextLoader instance.
        chunks: Optional InternalChunk catalog for in-memory source hydration.
        base_dir: Optional base directory for resolving source file paths on disk.
        use_chat_template: If True, formats prompts using Qwen ChatML markers by default.
        enable_thinking: Whether to allow Qwen3 ``<think>`` blocks in ChatML mode
            (default: False for sub-second CPU latency).
        """
        resolved_budget = (
            max_prompt_tokens
            if max_prompt_tokens is not None
            else (token_budget if token_budget is not None else max_tokens)
        )
        if resolved_budget <= 0:
            raise ValueError(
                f"max_tokens must be positive, got {resolved_budget}"
            )

        resolved_k = k if k is not None else top_k
        if resolved_k <= 0:
            raise ValueError(f"top_k must be positive, got {resolved_k}")

        if max_chunk_chars <= 0:
            raise ValueError(
                f"max_chunk_chars must be positive, got {max_chunk_chars}"
            )

        self.max_tokens: int = int(resolved_budget)
        self.max_prompt_tokens: int = self.max_tokens
        self.token_budget: int = self.max_tokens
        self.max_context_tokens: Optional[int] = (
            int(max_context_tokens) if max_context_tokens is not None else None
        )
        self.top_k: int = int(resolved_k)
        self.system_prompt: str = system_prompt.strip()
        self.max_chunk_chars: int = int(max_chunk_chars)
        self.tokenizer: Optional[Any] = tokenizer
        self.use_chat_template: bool = bool(use_chat_template)
        self.enable_thinking: bool = bool(enable_thinking)

        self.context_loader: ContextLoader = (
            context_loader
            if context_loader is not None
            else ContextLoader(
                base_dir=base_dir,
                chunks=chunks,
                max_chunk_size=self.max_chunk_chars,
            )
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate or compute token count using the builder's configured tokenizer."""
        return estimate_tokens(text, tokenizer=self.tokenizer)

    def _normalize_inputs(
        self,
        question: Union[str, MinimalSearchResults, Sequence[Any]],
        sources: Optional[Union[Sequence[Any], str]] = None,
        *,
        query: Optional[str] = None,
        chunks: Optional[Sequence[Any]] = None,
        retrieved_sources: Optional[Sequence[Any]] = None,
        context_snippets: Optional[Sequence[str]] = None,
        contexts: Optional[Sequence[Any]] = None,
    ) -> Tuple[str, List[Any], Optional[Sequence[str]]]:
        """Normalize positional/keyword args into ``(question_str, sources_list, snippets)``."""
        resolved_question: str = ""
        resolved_sources: Optional[Sequence[Any]] = None

        # Handle MinimalSearchResults passed as first argument
        if isinstance(question, MinimalSearchResults):
            resolved_question = question.question
            resolved_sources = question.retrieved_sources
        # Handle reversed positional args: (sources_list, question_str)
        elif (
            not isinstance(question, str)
            and isinstance(question, Sequence)
            and isinstance(sources, str)
        ):
            resolved_question = sources
            resolved_sources = question
        elif isinstance(question, str):
            resolved_question = question
            if sources is not None and not isinstance(sources, str):
                resolved_sources = sources
            elif isinstance(sources, str):
                resolved_sources = [sources]
        else:
            resolved_question = str(query or "")
            if isinstance(question, Sequence):
                resolved_sources = question

        if query is not None and not resolved_question.strip():
            resolved_question = query

        if resolved_sources is None:
            for candidate in (
                retrieved_sources,
                chunks,
                contexts,
                context_snippets,
            ):
                if candidate is not None:
                    resolved_sources = candidate
                    break

        clean_question = resolved_question.strip()
        if not clean_question:
            clean_question = "[No question provided]"

        final_sources = (
            list(resolved_sources) if resolved_sources is not None else []
        )
        return clean_question, final_sources, context_snippets

    def select_and_ground_sources(
        self,
        sources: Sequence[Any],
        *,
        context_snippets: Optional[Sequence[str]] = None,
        k: Optional[int] = None,
        max_context_tokens: Optional[int] = None,
    ) -> List[GroundedSource]:
        """Hydrate top-*k* sources and constrain their combined token count within budget.
        - checks if code chunks fit inside Qwen's strict token budget
        -If they are too big, it automatically shrinks them using one of two master strategies
        so that no source is completely thrown away if it can possibly fit.
        sources: Sequence of candidate sources (MinimalSource, InternalChunk, dict, str).
        context_snippets: Optional explicit text snippets corresponding to *sources*.
        k: Optional override for number of top sources to include.
        max_context_tokens: Maximum token budget available for the context section.

        """
        effective_k = k if k is not None else self.top_k
        if effective_k <= 0 or not sources:
            return []

        hydrated = self.context_loader.load_sources(
            sources,
            context_snippets=context_snippets,
            k=effective_k,
        )
        if not hydrated:
            return []

        budget = (
            max_context_tokens
            if max_context_tokens is not None
            else (
                self.max_context_tokens
                if self.max_context_tokens is not None
                else max(64, self.max_tokens - DEFAULT_RESERVED_PROMPT_TOKENS)
            )
        )
        if budget <= 0:
            return []

        # Measure each hydrated source's header and content token cost
        header_costs: List[int] = []
        content_costs: List[int] = []
        block_costs: List[int] = []

        for src in hydrated:
            h_tokens = self.estimate_tokens(src.header) + 1
            c_tokens = self.estimate_tokens(src.content)
            b_tokens = self.estimate_tokens(src.formatted) + 1
            header_costs.append(h_tokens)
            content_costs.append(c_tokens)
            block_costs.append(b_tokens)

        total_full_tokens = sum(block_costs)
        if total_full_tokens <= budget:
            return [
                GroundedSource(
                    index=src.index,
                    file_path=src.file_path,
                    first_character_index=src.first_character_index,
                    last_character_index=src.last_character_index,
                    content=src.content,
                    token_count=b_cost,
                    truncated=src.truncated,
                )
                for src, b_cost in zip(hydrated, block_costs)
            ]

        n = len(hydrated)
        total_headers_cost = sum(header_costs)

        # Strategy A: If all top-k headers + at least 6 content tokens per source fit in budget,
        # keep all top-k sources and allocate the remaining content token budget across them.
        if total_headers_cost + (n * 6) <= budget:
            available_for_content = max(n, budget - total_headers_cost)

            # Rank-decayed weights: top sources get a slightly larger share
            raw_weights = [1.0 / (1.0 + 0.15 * i) for i in range(n)]
            weight_sum = sum(raw_weights)
            allocations = [
                max(4, int(available_for_content * (w / weight_sum)))
                for w in raw_weights
            ]

            # Adjust allocations so sum(allocations) <= available_for_content
            while sum(allocations) > available_for_content:
                for i in range(n - 1, -1, -1):
                    if sum(allocations) <= available_for_content:
                        break
                    if allocations[i] > 1:
                        allocations[i] -= 1

            # Redistribute surplus tokens from short chunks to longer chunks
            surplus = 0
            needs_more: List[int] = []
            for i in range(n):
                if content_costs[i] <= allocations[i]:
                    surplus += allocations[i] - content_costs[i]
                    allocations[i] = content_costs[i]
                else:
                    needs_more.append(i)

            for i in needs_more:
                if surplus <= 0:
                    break
                needed = content_costs[i] - allocations[i]
                extra = min(surplus, needed)
                allocations[i] += extra
                surplus -= extra

            constrained_all: List[GroundedSource] = []
            for src, c_cost, allowed_c_tokens in zip(
                hydrated, content_costs, allocations
            ):
                if c_cost <= allowed_c_tokens:
                    new_content = src.content
                    was_truncated = src.truncated
                else:
                    new_content = truncate_to_token_budget(
                        src.content,
                        max_tokens=allowed_c_tokens,
                        tokenizer=self.tokenizer,
                        suffix="...",
                    )
                    was_truncated = True

                final_src = GroundedSource(
                    index=src.index,
                    file_path=src.file_path,
                    first_character_index=src.first_character_index,
                    last_character_index=src.last_character_index,
                    content=new_content,
                    token_count=self.estimate_tokens(
                        f"{src.header}\n{new_content}"
                        if new_content
                        else src.header
                    ),
                    truncated=was_truncated,
                )
                constrained_all.append(final_src)

            return constrained_all

        # Strategy B: Ultra-tight token budget where not all k headers can fit.
        # Pack sources greedily from rank 1 upwards until budget is exhausted.
        packed: List[GroundedSource] = []
        remaining = budget

        for idx, (src, h_cost, b_cost) in enumerate(
            zip(hydrated, header_costs, block_costs)
        ):
            if b_cost <= remaining:
                packed.append(
                    GroundedSource(
                        index=src.index,
                        file_path=src.file_path,
                        first_character_index=src.first_character_index,
                        last_character_index=src.last_character_index,
                        content=src.content,
                        token_count=b_cost,
                        truncated=src.truncated,
                    )
                )
                remaining -= b_cost
            elif remaining > h_cost:
                content_budget = remaining - h_cost
                clipped = truncate_to_token_budget(
                    src.content,
                    max_tokens=content_budget,
                    tokenizer=self.tokenizer,
                    suffix="...",
                )
                new_src = GroundedSource(
                    index=src.index,
                    file_path=src.file_path,
                    first_character_index=src.first_character_index,
                    last_character_index=src.last_character_index,
                    content=clipped,
                    token_count=self.estimate_tokens(
                        f"{src.header}\n{clipped}" if clipped else src.header
                    ),
                    truncated=True,
                )
                packed.append(new_src)
                break
            elif idx == 0 and remaining >= h_cost:
                packed.append(
                    GroundedSource(
                        index=src.index,
                        file_path=src.file_path,
                        first_character_index=src.first_character_index,
                        last_character_index=src.last_character_index,
                        content="",
                        token_count=h_cost,
                        truncated=True,
                    )
                )
                break
            else:
                break

        return packed

    def format_context(
        self,
        sources: Sequence[Any],
        *,
        context_snippets: Optional[Sequence[str]] = None,
        k: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Format retrieved sources into a grounded context string with headers.
        sources: Sequence of MinimalSource, InternalChunk, dict, or str items.
        context_snippets: Optional explicit text snippets paired with *sources*.
        k: Optional maximum number of top sources to format.
        max_tokens: Optional token budget for the formatted context section.
        """
        effective_budget = (
            max_tokens
            if max_tokens is not None
            else (
                self.max_context_tokens
                if self.max_context_tokens is not None
                else max(64, self.max_tokens - DEFAULT_RESERVED_PROMPT_TOKENS)
            )
        )
        grounded = self.select_and_ground_sources(
            sources,
            context_snippets=context_snippets,
            k=k,
            max_context_tokens=effective_budget,
        )
        if not grounded:
            return NO_CONTEXT_PLACEHOLDER

        context_text = "\n\n".join(src.formatted for src in grounded)
        if (
            max_tokens is not None
            and self.estimate_tokens(context_text) > max_tokens
        ):
            context_text = truncate_to_token_budget(
                context_text, max_tokens=max_tokens, tokenizer=self.tokenizer
            )
        return context_text

    def build_prompt(
        self,
        question: Union[str, MinimalSearchResults, Sequence[Any]] = "",
        sources: Optional[Union[Sequence[Any], str]] = None,
        *,
        query: Optional[str] = None,
        chunks: Optional[Sequence[Any]] = None,
        retrieved_sources: Optional[Sequence[Any]] = None,
        context_snippets: Optional[Sequence[str]] = None,
        contexts: Optional[Sequence[Any]] = None,
        k: Optional[int] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_prompt_tokens: Optional[int] = None,
        token_budget: Optional[int] = None,
        system_prompt: Optional[str] = None,
        use_chat_template: Optional[bool] = None,
    ) -> str:
        """Build a grounded instruction prompt injecting top-*k* sources within token budget.
        question: User question string or :class:`MinimalSearchResults` object.
        sources: Retrieved sources (MinimalSource, InternalChunk, dict, or str snippets).
        query: Optional keyword alias for *question*.
        chunks: Optional keyword alias for *sources*.
        retrieved_sources: Optional keyword alias for *sources*.
        context_snippets: Optional explicit text snippets corresponding to *sources*.
        contexts: Optional keyword alias for *sources*.
        k: Optional number of top sources to inject (defaults to ``self.top_k``).
        top_k: Optional keyword alias for *k*.
        max_tokens: Optional override for total prompt token budget.
        max_prompt_tokens: Optional keyword alias for *max_tokens*.
        token_budget: Optional keyword alias for *max_tokens*.
        system_prompt: Optional override for grounding system instructions.
        use_chat_template: Optional override to render Qwen ChatML markers.
        """
        effective_budget = (
            max_prompt_tokens
            if max_prompt_tokens is not None
            else (
                token_budget
                if token_budget is not None
                else (
                    max_tokens if max_tokens is not None else self.max_tokens
                )
            )
        )
        if effective_budget <= 0:
            raise ValueError(
                f"max_tokens must be positive, got {effective_budget}"
            )

        effective_k = (
            k
            if k is not None
            else (top_k if top_k is not None else self.top_k)
        )
        render_chatml = (
            use_chat_template
            if use_chat_template is not None
            else self.use_chat_template
        )
        if render_chatml:
            return self.format_chatml(
                question=question,
                sources=sources,
                query=query,
                chunks=chunks,
                retrieved_sources=retrieved_sources,
                context_snippets=context_snippets,
                contexts=contexts,
                k=effective_k,
                max_tokens=effective_budget,
                system_prompt=system_prompt,
            )

        clean_question, source_list, snippets = self._normalize_inputs(
            question=question,
            sources=sources,
            query=query,
            chunks=chunks,
            retrieved_sources=retrieved_sources,
            context_snippets=context_snippets,
            contexts=contexts,
        )

        active_sys_prompt = (
            system_prompt.strip()
            if system_prompt is not None
            else self.system_prompt
        )

        # Ensure system prompt and question fit within effective_budget with room for context
        max_question_tokens = max(16, effective_budget // 4)
        if self.estimate_tokens(clean_question) > max_question_tokens:
            clean_question = truncate_to_token_budget(
                clean_question,
                max_tokens=max_question_tokens,
                tokenizer=self.tokenizer,
                suffix="...",
            )

        max_sys_tokens = max(24, effective_budget // 3)
        if self.estimate_tokens(active_sys_prompt) > max_sys_tokens:
            active_sys_prompt = truncate_to_token_budget(
                active_sys_prompt,
                max_tokens=max_sys_tokens,
                tokenizer=self.tokenizer,
            )

        # Calculate scaffolding overhead
        scaffolding_sample = (
            f"System Instructions:\n{active_sys_prompt}\n\n"
            f"Retrieved Context (Top-{max(0, effective_k)} Sources):\n\n\n"
            f"Question:\n{clean_question}\n\n"
            f"Answer:\n"
        )
        scaffolding_tokens = self.estimate_tokens(scaffolding_sample) + 8
        available_context_tokens = max(
            0, effective_budget - scaffolding_tokens
        )
        if self.max_context_tokens is not None:
            available_context_tokens = min(
                available_context_tokens, self.max_context_tokens
            )

        grounded_sources = self.select_and_ground_sources(
            source_list,
            context_snippets=snippets,
            k=effective_k,
            max_context_tokens=available_context_tokens,
        )

        if grounded_sources:
            context_block = "\n\n".join(
                src.formatted for src in grounded_sources
            )
            context_title = (
                f"Retrieved Context (Top-{len(grounded_sources)} Sources):"
            )
        else:
            context_block = NO_CONTEXT_PLACEHOLDER
            context_title = "Retrieved Context (Top-0 Sources):"

        prompt = (
            f"System Instructions:\n{active_sys_prompt}\n\n"
            f"{context_title}\n"
            f"{context_block}\n\n"
            f"Question:\n{clean_question}\n\n"
            f"Answer:\n"
        )

        # Final hard token budget guard
        if self.estimate_tokens(prompt) > effective_budget:
            suffix_part = f"\n\nQuestion:\n{clean_question}\n\nAnswer:\n"
            suffix_tokens = self.estimate_tokens(suffix_part)
            if suffix_tokens + 8 < effective_budget:
                prefix_part = (
                    f"System Instructions:\n{active_sys_prompt}\n\n"
                    f"{context_title}\n"
                    f"{context_block}"
                )
                clipped_prefix = truncate_to_token_budget(
                    prefix_part,
                    max_tokens=effective_budget - suffix_tokens - 2,
                    tokenizer=self.tokenizer,
                )
                prompt = f"{clipped_prefix}{suffix_part}"

            if self.estimate_tokens(prompt) > effective_budget:
                prompt = truncate_to_token_budget(
                    prompt,
                    max_tokens=effective_budget,
                    tokenizer=self.tokenizer,
                )

        return prompt

    def __call__(
        self,
        question: Union[str, MinimalSearchResults, Sequence[Any]] = "",
        sources: Optional[Union[Sequence[Any], str]] = None,
        **kwargs: Any,
    ) -> str:
        """Callable interface forwarding to :meth:`build_prompt`."""
        return self.build_prompt(question=question, sources=sources, **kwargs)

    def build_messages(
        self,
        question: Union[str, MinimalSearchResults, Sequence[Any]] = "",
        sources: Optional[Union[Sequence[Any], str]] = None,
        *,
        query: Optional[str] = None,
        chunks: Optional[Sequence[Any]] = None,
        retrieved_sources: Optional[Sequence[Any]] = None,
        context_snippets: Optional[Sequence[str]] = None,
        contexts: Optional[Sequence[Any]] = None,
        k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build OpenAI / Hugging Face chat messages ``[system_msg, user_msg]``.
        Constrains combined token length across system and user messages within *max_tokens*.
        """
        effective_budget = (
            max_tokens if max_tokens is not None else self.max_tokens
        )
        if effective_budget <= 0:
            raise ValueError(
                f"max_tokens must be positive, got {effective_budget}"
            )

        effective_k = k if k is not None else self.top_k
        clean_question, source_list, snippets = self._normalize_inputs(
            question=question,
            sources=sources,
            query=query,
            chunks=chunks,
            retrieved_sources=retrieved_sources,
            context_snippets=context_snippets,
            contexts=contexts,
        )

        active_sys_prompt = (
            system_prompt.strip()
            if system_prompt is not None
            else self.system_prompt
        )
        max_sys_tokens = max(24, effective_budget // 3)
        if self.estimate_tokens(active_sys_prompt) > max_sys_tokens:
            active_sys_prompt = truncate_to_token_budget(
                active_sys_prompt,
                max_tokens=max_sys_tokens,
                tokenizer=self.tokenizer,
            )

        max_question_tokens = max(16, effective_budget // 4)
        if self.estimate_tokens(clean_question) > max_question_tokens:
            clean_question = truncate_to_token_budget(
                clean_question,
                max_tokens=max_question_tokens,
                tokenizer=self.tokenizer,
                suffix="...",
            )

        overhead_tokens = (
            self.estimate_tokens(active_sys_prompt)
            + self.estimate_tokens(clean_question)
            + 32
        )
        available_context_tokens = max(0, effective_budget - overhead_tokens)
        if self.max_context_tokens is not None:
            available_context_tokens = min(
                available_context_tokens, self.max_context_tokens
            )

        grounded_sources = self.select_and_ground_sources(
            source_list,
            context_snippets=snippets,
            k=effective_k,
            max_context_tokens=available_context_tokens,
        )

        if grounded_sources:
            context_block = "\n\n".join(
                src.formatted for src in grounded_sources
            )
            context_title = (
                f"Retrieved Context (Top-{len(grounded_sources)} Sources):"
            )
        else:
            context_block = NO_CONTEXT_PLACEHOLDER
            context_title = "Retrieved Context (Top-0 Sources):"

        user_content = (
            f"{context_title}\n"
            f"{context_block}\n\n"
            f"Question:\n{clean_question}"
        )

        sys_tokens = self.estimate_tokens(active_sys_prompt)
        max_user_tokens = max(1, effective_budget - sys_tokens - 12)
        if self.estimate_tokens(user_content) > max_user_tokens:
            user_content = truncate_to_token_budget(
                user_content,
                max_tokens=max_user_tokens,
                tokenizer=self.tokenizer,
            )

        return [
            {"role": "system", "content": active_sys_prompt},
            {"role": "user", "content": user_content},
        ]

    def format_chatml(
        self,
        question: Union[str, MinimalSearchResults, Sequence[Any]] = "",
        sources: Optional[Union[Sequence[Any], str]] = None,
        *,
        query: Optional[str] = None,
        chunks: Optional[Sequence[Any]] = None,
        retrieved_sources: Optional[Sequence[Any]] = None,
        context_snippets: Optional[Sequence[str]] = None,
        contexts: Optional[Sequence[Any]] = None,
        k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        add_generation_prompt: bool = True,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """Render the grounded prompt using Qwen3's ChatML template within token budget."""
        effective_budget = (
            max_tokens if max_tokens is not None else self.max_tokens
        )
        if effective_budget <= 0:
            raise ValueError(
                f"max_tokens must be positive, got {effective_budget}"
            )

        # Reserve tokens for ChatML special tags
        inner_budget = max(16, effective_budget - 24)
        messages = self.build_messages(
            question=question,
            sources=sources,
            query=query,
            chunks=chunks,
            retrieved_sources=retrieved_sources,
            context_snippets=context_snippets,
            contexts=contexts,
            k=k,
            max_tokens=inner_budget,
            system_prompt=system_prompt,
        )

        think_flag = (
            enable_thinking
            if enable_thinking is not None
            else self.enable_thinking
        )
        parts: List[str] = []
        for msg in messages:
            parts.append(
                f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>"
            )

        if add_generation_prompt:
            if think_flag:
                parts.append("<|im_start|>assistant\n")
            else:
                parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")

        rendered = "\n".join(parts)
        if self.estimate_tokens(rendered) > effective_budget:
            rendered = truncate_to_token_budget(
                rendered,
                max_tokens=effective_budget,
                tokenizer=self.tokenizer,
            )
        return rendered


def format_context(
    sources: Sequence[Any],
    *,
    context_snippets: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    tokenizer: Optional[Any] = None,
    chunks: Optional[Sequence[InternalChunk]] = None,
    base_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Format retrieved sources into a grounded context block with explicit headers."""
    builder = PromptBuilder(
        max_tokens=max(
            max_tokens + DEFAULT_RESERVED_PROMPT_TOKENS, max_tokens
        ),
        max_context_tokens=max_tokens,
        top_k=k if k is not None and k > 0 else DEFAULT_TOP_K,
        max_chunk_chars=max_chunk_chars,
        tokenizer=tokenizer,
        chunks=chunks,
        base_dir=base_dir,
    )
    return builder.format_context(
        sources=sources,
        context_snippets=context_snippets,
        k=k,
        max_tokens=max_tokens,
    )


def build_prompt(
    question: Union[str, MinimalSearchResults, Sequence[Any]] = "",
    sources: Optional[Union[Sequence[Any], str]] = None,
    *,
    query: Optional[str] = None,
    chunks: Optional[Sequence[Any]] = None,
    retrieved_sources: Optional[Sequence[Any]] = None,
    context_snippets: Optional[Sequence[str]] = None,
    contexts: Optional[Sequence[Any]] = None,
    k: Optional[int] = None,
    top_k: Optional[int] = None,
    max_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    max_prompt_tokens: Optional[int] = None,
    token_budget: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    tokenizer: Optional[Any] = None,
    catalog_chunks: Optional[Sequence[InternalChunk]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    use_chat_template: bool = False,
) -> str:
    """Build a grounded instruction prompt from *question* and retrieved *sources*.
    Injects top-*k* sources with explicit ``file_path`` and character span headers
    while constraining prompt token length within Qwen3's sub-second CPU budget
    (2k-4k tokens, default 3000).
    """
    effective_budget = (
        max_prompt_tokens
        if max_prompt_tokens is not None
        else (token_budget if token_budget is not None else max_tokens)
    )
    effective_k = (
        k if k is not None else (top_k if top_k is not None else DEFAULT_TOP_K)
    )

    # If `chunks` was passed as a catalog of InternalChunk while `sources` is also present
    in_memory_catalog: Optional[Sequence[InternalChunk]] = catalog_chunks
    if (
        in_memory_catalog is None
        and sources is not None
        and chunks is not None
        and len(chunks) > 0
        and isinstance(chunks[0], InternalChunk)
    ):
        in_memory_catalog = chunks
        chunks = None

    builder = PromptBuilder(
        max_tokens=(
            effective_budget
            if effective_budget > 0
            else DEFAULT_MAX_PROMPT_TOKENS
        ),
        top_k=effective_k if effective_k > 0 else DEFAULT_TOP_K,
        system_prompt=system_prompt,
        max_chunk_chars=max_chunk_chars,
        tokenizer=tokenizer,
        chunks=in_memory_catalog,
        base_dir=base_dir,
        use_chat_template=use_chat_template,
    )
    if effective_budget <= 0:
        raise ValueError(
            f"max_tokens must be positive, got {effective_budget}"
        )

    return builder.build_prompt(
        question=question,
        sources=sources,
        query=query,
        chunks=chunks,
        retrieved_sources=retrieved_sources,
        context_snippets=context_snippets,
        contexts=contexts,
        k=effective_k,
        max_tokens=effective_budget,
        use_chat_template=use_chat_template,
    )


def build_messages(
    question: Union[str, MinimalSearchResults, Sequence[Any]] = "",
    sources: Optional[Union[Sequence[Any], str]] = None,
    *,
    query: Optional[str] = None,
    chunks: Optional[Sequence[Any]] = None,
    retrieved_sources: Optional[Sequence[Any]] = None,
    context_snippets: Optional[Sequence[str]] = None,
    contexts: Optional[Sequence[Any]] = None,
    k: Optional[int] = None,
    max_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tokenizer: Optional[Any] = None,
    base_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, str]]:
    """Convenience function to build grounded chat messages within token budget."""
    builder = PromptBuilder(
        max_tokens=max_tokens,
        top_k=k if k is not None and k > 0 else DEFAULT_TOP_K,
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        base_dir=base_dir,
    )
    return builder.build_messages(
        question=question,
        sources=sources,
        query=query,
        chunks=chunks,
        retrieved_sources=retrieved_sources,
        context_snippets=context_snippets,
        contexts=contexts,
        k=k,
        max_tokens=max_tokens,
    )


def format_chatml(
    question: Union[str, MinimalSearchResults, Sequence[Any]] = "",
    sources: Optional[Union[Sequence[Any], str]] = None,
    *,
    query: Optional[str] = None,
    chunks: Optional[Sequence[Any]] = None,
    retrieved_sources: Optional[Sequence[Any]] = None,
    context_snippets: Optional[Sequence[str]] = None,
    contexts: Optional[Sequence[Any]] = None,
    k: Optional[int] = None,
    max_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tokenizer: Optional[Any] = None,
    base_dir: Optional[Union[str, Path]] = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> str:
    """Convenience function to render a Qwen3 ChatML grounded prompt within token budget."""
    builder = PromptBuilder(
        max_tokens=max_tokens,
        top_k=k if k is not None and k > 0 else DEFAULT_TOP_K,
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        base_dir=base_dir,
        enable_thinking=enable_thinking,
    )
    return builder.format_chatml(
        question=question,
        sources=sources,
        query=query,
        chunks=chunks,
        retrieved_sources=retrieved_sources,
        context_snippets=context_snippets,
        contexts=contexts,
        k=k,
        max_tokens=max_tokens,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
