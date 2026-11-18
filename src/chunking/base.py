"""Abstract Base Chunker & File Traversal Engine.

Provides recursive source file discovery and abstract base contract for all
language chunkers, enforcing strict max_chunk_size guards.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Union,
)

from src.indexing.schema import InternalChunk

DEFAULT_SUPPORTED_EXTENSIONS: Set[str] = {".py", ".md"}

DEFAULT_IGNORE_DIRS: Set[str] = {
    ".git",
    ".github",
    ".gitlab",
    ".gemini",
    ".buildkite",
    ".venv",
    "venv",
    "env",
    ".env",
    "virtualenv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    ".vscode",
    ".idea",
    "build",
    "dist",
    "site-packages",
    "node_modules",
    ".eggs",
}

DEFAULT_IGNORE_PATTERNS: Set[str] = {
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.exe",
    "*.bin",
    "*.pkl",
    "*.pt",
    "*.pth",
    "*.tar",
    "*.gz",
    "*.zip",
    "*.7z",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.pdf",
}


def is_binary_file(file_path: Union[str, Path], sample_size: int = 8192) -> bool:
    """Check if a file appears to be binary.
    Reads an initial sample of bytes from the file. If null bytes (b'\\x00')
    are found, or if the sample cannot be decoded as UTF-8, the file is
    treated as binary.
        file_path: Path to target file.
        sample_size: Number of bytes to read for heuristic inspection.
    """
    path = Path(file_path)
    if not path.is_file():
        return False

    try:
        # rb means read binary
        with path.open("rb") as f:
            chunk = f.read(sample_size)
            if b"\x00" in chunk:
                return True
            try:
                chunk.decode("utf-8")
            except UnicodeDecodeError as e:
                # If decode error occurs only at the very boundary (cut-off multi-byte character)
                if e.start >= len(chunk) - 4:
                    try:
                        chunk[: e.start].decode("utf-8")
                        return False
                    except UnicodeDecodeError:
                        return True
                return True
            return False
    except (OSError, PermissionError):
        return True


class FileTraversalEngine:
    """Recursive file discovery engine with filtering for binary, git, and virtualenv files."""

    def __init__(
        self,
        allowed_extensions: Optional[Sequence[str]] = None,
        ignored_dir_names: Optional[Set[str]] = None,
        check_binary: bool = True,
    ) -> None:
        """Initialize the traversal engine.
            allowed_extensions: Sequence of file extensions to include (e.g. ['.py', '.md']).
                Defaults to {'.py', '.md'}.
            ignored_dir_names: Set of directory names to ignore. Defaults to standard
                git, virtualenv, and cache directory names.
            check_binary: Whether to inspect files for binary content before inclusion.
        """
        if allowed_extensions is None:
            self.allowed_extensions = {ext.lower() for ext in DEFAULT_SUPPORTED_EXTENSIONS}
        else:
            self.allowed_extensions = {
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in allowed_extensions
            }

        self.ignored_dir_names = (
            set(ignored_dir_names)
            if ignored_dir_names is not None
            else set(DEFAULT_IGNORE_DIRS)
        )
        self.check_binary = check_binary

    def is_ignored_directory(self, dir_name: str) -> bool:
        """Check if a directory name should be ignored during traversal."""
        if dir_name.startswith("."):
            return True
        lower_name = dir_name.lower()
        if lower_name in self.ignored_dir_names:
            return True
        if lower_name.endswith(".egg-info"):
            return True
        return False

    def is_valid_file(self, file_path: Path) -> bool:
        """Check if a file matches extension criteria and is not hidden or binary."""
        file_name = file_path.name
        if file_name.startswith("."):
            return False

        ext = file_path.suffix.lower()
        if ext not in self.allowed_extensions:
            return False

        if self.check_binary and is_binary_file(file_path):
            return False

        return True

    def discover(self, corpus_dir: Union[str, Path]) -> List[Path]:
        """Discover valid source files within target directory.
        Recursively discovers all matching files (by default .py and .md),
        ignoring hidden directories, git trees, virtualenvs, cache folders,
        and binaries. Ensures path traversal mitigation.
            corpus_dir: Path to directory to search.
        """
        root_path = Path(corpus_dir)
        if not root_path.exists():
            raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")
        if not root_path.is_dir():
            raise NotADirectoryError(f"Corpus path is not a directory: {corpus_dir}")

        resolved_root = root_path.resolve()
        discovered: List[Path] = []

        for root, dirs, files in os.walk(root_path, topdown=True):
            # Prune ignored directory trees in-place
            dirs[:] = [d for d in dirs if not self.is_ignored_directory(d)]

            for file_name in files:
                file_path = Path(root) / file_name

                # Path traversal mitigation: ensure file resolves within corpus root
                try:
                    resolved_file = file_path.resolve()
                    if not resolved_file.is_relative_to(resolved_root):
                        continue
                except (ValueError, RuntimeError):
                    continue

                if self.is_valid_file(file_path):
                    discovered.append(file_path)

        # Deterministic sorting
        discovered.sort(key=lambda p: p.as_posix())
        return discovered


def discover_files(
    corpus_dir: Union[str, Path],
    allowed_extensions: Optional[Sequence[str]] = None,
    ignored_dir_names: Optional[Set[str]] = None,
    check_binary: bool = True,
) -> List[Path]:
    """Discover .py and .md files in the corpus directory, ignoring git/venvs/binaries.
        corpus_dir: Root directory of corpus to traverse.
        allowed_extensions: File extensions to include. Defaults to {'.py', '.md'}.
        ignored_dir_names: Directories to ignore. Defaults to DEFAULT_IGNORE_DIRS.
        check_binary: Whether to filter out binary files.
    """
    engine = FileTraversalEngine(
        allowed_extensions=allowed_extensions,
        ignored_dir_names=ignored_dir_names,
        check_binary=check_binary,
    )
    return engine.discover(corpus_dir)


class BaseChunker(ABC):
    """Abstract base contract for language-specific document chunkers."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize chunker with character ceiling guard.
            max_chunk_size: Maximum allowed character length for any chunk.
                Defaults to 2000 characters.
        """
        if max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be positive, got {max_chunk_size}")
        self._max_chunk_size = max_chunk_size

    @property
    def max_chunk_size(self) -> int:
        """Return maximum chunk size in characters."""
        return self._max_chunk_size

    @abstractmethod
    def chunk_file(self, file_path: Path) -> List[InternalChunk]:
        """Parse source file into discrete InternalChunk units.
            file_path: Path to target source file.
        """
        raise NotImplementedError

    def __call__(self, file_path: Path) -> List[InternalChunk]:
        """Callable interface forwarding to chunk_file."""
        return self.chunk_file(file_path)

    @classmethod
    def discover_files(
        cls,
        corpus_dir: Union[str, Path],
        allowed_extensions: Optional[Sequence[str]] = None,
        ignored_dir_names: Optional[Set[str]] = None,
        check_binary: bool = True,
    ) -> List[Path]:
        """Convenience method to discover source files in corpus directory."""
        return discover_files(
            corpus_dir=corpus_dir,
            allowed_extensions=allowed_extensions,
            ignored_dir_names=ignored_dir_names,
            check_binary=check_binary,
        )

    @staticmethod
    def estimate_token_count(text: str) -> int:
        """Estimate token count for text using word and punctuation splitting."""
        if not text:
            return 0
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return max(1, len(tokens)) if text.strip() else 0

    @staticmethod
    def read_file_safely(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read a text file safely with fallback error handling."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="replace")

    def clamp_chunk(self, chunk: InternalChunk) -> InternalChunk:
        """Clamp oversized chunk to max_chunk_size characters.
        Adjusts last_character_index and truncates content so neither
        exceeds max_chunk_size.
            chunk: InternalChunk to clamp.

        """
        if (
            chunk.span_length <= self._max_chunk_size
            and len(chunk.content) <= self._max_chunk_size
        ):
            return chunk

        new_content = chunk.content[: self._max_chunk_size]
        new_last_index = chunk.first_character_index + len(new_content)
        # Ensure last_character_index > first_character_index
        if new_last_index <= chunk.first_character_index:
            new_last_index = chunk.first_character_index + 1
            if not new_content:
                new_content = " "

        token_count = self.estimate_token_count(new_content)

        return InternalChunk(
            chunk_id=chunk.chunk_id,
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=new_last_index,
            chunk_type=chunk.chunk_type,
            token_count=token_count,
            content=new_content,
        )

    def split_chunk(
        self,
        chunk: InternalChunk,
        start_chunk_id: Optional[int] = None,
    ) -> List[InternalChunk]:
        """Split an oversized chunk into multiple chunks within max_chunk_size.
        Subdivides along paragraph breaks (\\n\\n), line breaks (\\n), or spaces
        where possible, maintaining exact relative character offsets mapping to disk.
            chunk: InternalChunk to split.
            start_chunk_id: Optional starting chunk_id for generated sub-chunks.
                If None, uses chunk.chunk_id as base.
        """
        if (
            chunk.span_length <= self._max_chunk_size
            and len(chunk.content) <= self._max_chunk_size
        ):
            return [chunk]

        content = chunk.content
        total_len = len(content)
        base_id = chunk.chunk_id if start_chunk_id is None else start_chunk_id

        # If content matches span length, we can do exact 1-to-1 character slicing
        if total_len == chunk.span_length:
            sub_chunks: List[InternalChunk] = []
            cursor = 0
            idx = 0

            while cursor < total_len:
                remaining = total_len - cursor
                if remaining <= self._max_chunk_size:
                    chunk_slice_len = remaining
                else:
                    window = content[cursor : cursor + self._max_chunk_size]
                    min_chunk_len = max(1, self._max_chunk_size // 4)

                    # 1. Try double newline (paragraph boundary)
                    break_pos = window.rfind("\n\n")
                    if break_pos != -1 and break_pos >= min_chunk_len:
                        chunk_slice_len = break_pos + 2
                    else:
                        # 2. Try single newline
                        break_pos = window.rfind("\n")
                        if break_pos != -1 and break_pos >= min_chunk_len:
                            chunk_slice_len = break_pos + 1
                        else:
                            # 3. Try word boundary
                            break_pos = window.rfind(" ")
                            if break_pos != -1 and break_pos >= min_chunk_len:
                                chunk_slice_len = break_pos + 1
                            else:
                                # 4. Hard cut at max_chunk_size
                                chunk_slice_len = self._max_chunk_size

                sub_content = content[cursor : cursor + chunk_slice_len]
                sub_first = chunk.first_character_index + cursor
                sub_last = sub_first + len(sub_content)

                if sub_last <= sub_first:
                    sub_last = sub_first + 1

                token_count = self.estimate_token_count(sub_content)

                sub_chunks.append(
                    InternalChunk(
                        chunk_id=base_id + idx,
                        file_path=chunk.file_path,
                        first_character_index=sub_first,
                        last_character_index=sub_last,
                        chunk_type=chunk.chunk_type,
                        token_count=token_count,
                        content=sub_content,
                    )
                )

                cursor += chunk_slice_len
                idx += 1

            return sub_chunks

        # Fallback if content length != span_length (e.g. synthetic or transformed content)
        # We split content by character slices, and scale/clamp character spans safely
        fallback_sub_chunks: List[InternalChunk] = []
        cursor = 0
        idx = 0
        span_len = chunk.span_length

        while cursor < total_len:
            remaining = total_len - cursor
            chunk_slice_len = min(remaining, self._max_chunk_size)
            if remaining > self._max_chunk_size:
                window = content[cursor : cursor + self._max_chunk_size]
                min_chunk_len = max(1, self._max_chunk_size // 4)
                break_pos = window.rfind("\n\n")
                if break_pos != -1 and break_pos >= min_chunk_len:
                    chunk_slice_len = break_pos + 2
                else:
                    break_pos = window.rfind("\n")
                    if break_pos != -1 and break_pos >= min_chunk_len:
                        chunk_slice_len = break_pos + 1
                    else:
                        break_pos = window.rfind(" ")
                        if break_pos != -1 and break_pos >= min_chunk_len:
                            chunk_slice_len = break_pos + 1

            sub_content = content[cursor : cursor + chunk_slice_len]

            # Compute proportional span bounded by max_chunk_size
            rel_start = cursor / total_len if total_len > 0 else 0.0
            rel_end = (cursor + chunk_slice_len) / total_len if total_len > 0 else 1.0

            sub_first = chunk.first_character_index + int(rel_start * span_len)
            sub_last = chunk.first_character_index + int(rel_end * span_len)

            if sub_last <= sub_first:
                sub_last = sub_first + min(len(sub_content), self._max_chunk_size, 1)

            # Ensure span length does not exceed max_chunk_size
            if sub_last - sub_first > self._max_chunk_size:
                sub_last = sub_first + self._max_chunk_size

            token_count = self.estimate_token_count(sub_content)

            fallback_sub_chunks.append(
                InternalChunk(
                    chunk_id=base_id + idx,
                    file_path=chunk.file_path,
                    first_character_index=sub_first,
                    last_character_index=sub_last,
                    chunk_type=chunk.chunk_type,
                    token_count=token_count,
                    content=sub_content,
                )
            )

            cursor += chunk_slice_len
            idx += 1

        return fallback_sub_chunks

    def enforce_max_chunk_size(
            self,
            chunk: InternalChunk,
            strategy: Literal["split", "clamp"] = "split",
    ) -> List[InternalChunk]:
        """Enforce character ceiling guard on an individual chunk.
        If chunk exceeds max_chunk_size, applies the chosen strategy:
        - 'split': Subdivides into multiple valid chunks <= max_chunk_size.
        - 'clamp': Clamps/truncates to a single chunk <= max_chunk_size.
            chunk: InternalChunk to inspect and guard.
            strategy: Either 'split' or 'clamp'. Defaults to 'split'.
        """
        if (
            chunk.span_length <= self._max_chunk_size
            and len(chunk.content) <= self._max_chunk_size
        ):
            return [chunk]

        if strategy == "clamp":
            return [self.clamp_chunk(chunk)]
        elif strategy == "split":
            return self.split_chunk(chunk)
        else:
            raise ValueError(f"Unknown guard strategy: '{strategy}', expected 'split' or 'clamp'")

    def guard_chunks(
        self,
        chunks: Sequence[InternalChunk],
        strategy: Literal["split", "clamp"] = "split",
        reindex: bool = True,
    ) -> List[InternalChunk]:
        """Apply hard size guard to a collection of chunks.
        Guarantees that no returned chunk exceeds max_chunk_size.
            chunks: Sequence of InternalChunk instances to validate.
            strategy: Guard strategy ('split' or 'clamp').
            reindex: If True, re-assigns sequential chunk_ids starting from 0.
        """
        guarded: List[InternalChunk] = []
        for chunk in chunks:
            guarded.extend(self.enforce_max_chunk_size(chunk, strategy=strategy))

        if reindex:
            return [
                InternalChunk(
                    chunk_id=idx,
                    file_path=c.file_path,
                    first_character_index=c.first_character_index,
                    last_character_index=c.last_character_index,
                    chunk_type=c.chunk_type,
                    token_count=c.token_count,
                    content=c.content,
                )
                for idx, c in enumerate(guarded)
            ]
        return guarded
