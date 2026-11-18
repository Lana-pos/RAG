"""Heading-Aware Markdown Document Chunker.

Parses Markdown documentation into syntactically and semantically coherent units
along heading boundaries (# through ######), preserving associated text paragraphs
and fenced code blocks (``` and ~~~), enforcing a character ceiling <= max_chunk_size,
gracefully dividing oversized sections at paragraph boundaries, and recording exact
0-based character offsets mapping to disk.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

from src.chunking.base import BaseChunker
from src.indexing.schema import InternalChunk

_FENCE_PATTERN = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")
_ATX_HEADING_PATTERN = re.compile(r"^[ ]{0,3}(#{1,6})(?:[ \t]+.*|[ \t]*)$")


def find_code_fence_intervals(text: str) -> List[Tuple[int, int]]:
    """Identify character offset intervals of all fenced code blocks.
    Recognizes backtick (```) and tilde (~~~) code fences, respecting CommonMark
    rules (fence character matching, fence length matching, info string semantics).
        text: Target Markdown text.
    """
    intervals: List[Tuple[int, int]] = []
    lines = text.splitlines(keepends=True)

    in_fence = False
    fence_char = ""
    fence_len = 0
    fence_start = 0

    cursor = 0
    for line in lines:
        line_len = len(line)
        line_stripped = line.rstrip("\r\n")

        if not in_fence:
            match = _FENCE_PATTERN.match(line_stripped)
            if match:
                marker = match.group(1)
                fence_char = marker[0]
                fence_len = len(marker)
                in_fence = True
                fence_start = cursor
        else:
            # Check for closing fence: same character, at least same length, no info string
            match = _FENCE_PATTERN.match(line_stripped)
            if match:
                marker = match.group(1)
                rest = match.group(2).strip()
                if marker[0] == fence_char and len(marker) >= fence_len and not rest:
                    intervals.append((fence_start, cursor + line_len))
                    in_fence = False

        cursor += line_len

    if in_fence:
        # Unclosed fence runs to the end of the text
        intervals.append((fence_start, len(text)))

    return intervals


class MarkdownChunker(BaseChunker):
    """Heading-aware Markdown chunker segmenting text along heading hierarchy."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize Markdown chunker with character ceiling guard.
            max_chunk_size: Maximum allowed character length for any chunk.
                Defaults to 2000 characters.
        """
        super().__init__(max_chunk_size=max_chunk_size)

    def chunk_file(self, file_path: Path) -> List[InternalChunk]:
        """Parse Markdown source file into discrete InternalChunk units.
            file_path: Path to target source file.
        """
        source = self.read_file_safely(file_path)
        return self.chunk_text(source, file_path=file_path)

    def chunk_text(
        self,
        source: str,
        file_path: Union[str, Path] = "document.md",
    ) -> List[InternalChunk]:
        """Parse in-memory Markdown source text into discrete InternalChunk units.
        Identifies top-level sections along ATX headers (# through ######), preserving
        associated text paragraphs and code fences within each respective section.
        Sections exceeding max_chunk_size are gracefully subdivided along paragraph
        boundaries without fracturing code blocks.
            source: Markdown source text.
            file_path: Relative or absolute path representation for metadata.
        """
        if not source or not source.strip():
            return []

        clean_path = str(file_path)
        source_len = len(source)

        # 1. Identify all code fences so internal hashes are not mistaken for headings
        code_intervals = find_code_fence_intervals(source)

        def is_inside_fence(offset: int) -> bool:
            for f_start, f_end in code_intervals:
                if f_start <= offset < f_end:
                    return True
            return False

        # 2. Scan lines and identify heading boundaries
        lines = source.splitlines(keepends=True)
        heading_offsets: List[int] = []

        cursor = 0
        for line in lines:
            line_stripped = line.rstrip("\r\n")
            if not is_inside_fence(cursor) and _ATX_HEADING_PATTERN.match(line_stripped):
                heading_offsets.append(cursor)
            cursor += len(line)

        # 3. Partition into top-level sections
        # Sections cover [start, end) ranges contiguously
        section_spans: List[Tuple[int, int]] = []

        if not heading_offsets:
            # No headings found: whole text is a single section
            section_spans.append((0, source_len))
        else:
            # Handle preamble before first heading if present
            if heading_offsets[0] > 0:
                preamble_content = source[: heading_offsets[0]]
                if preamble_content.strip():
                    section_spans.append((0, heading_offsets[0]))
                else:
                    # If preamble is just leading whitespace, include it in first section
                    heading_offsets[0] = 0

            for i in range(len(heading_offsets)):
                sec_start = heading_offsets[i]
                sec_end = (
                    heading_offsets[i + 1]
                    if i + 1 < len(heading_offsets)
                    else source_len
                )
                if sec_end > sec_start:
                    section_spans.append((sec_start, sec_end))

        # 4. Generate chunks, subdividing oversized sections at paragraph boundaries
        chunks: List[InternalChunk] = []

        for sec_start, sec_end in section_spans:
            sec_len = sec_end - sec_start
            sec_content = source[sec_start:sec_end]

            if sec_len <= self._max_chunk_size:
                # Section fits within max_chunk_size
                chunks.append(
                    InternalChunk(
                        chunk_id=len(chunks),
                        file_path=clean_path,
                        first_character_index=sec_start,
                        last_character_index=sec_end,
                        chunk_type="markdown_section",
                        token_count=self.estimate_token_count(sec_content),
                        content=sec_content,
                    )
                )
            else:
                # Oversized section: gracefully subdivide along paragraph boundaries
                sub_chunks = self._subdivide_oversized_section(
                    sec_content=sec_content,
                    sec_start=sec_start,
                    clean_path=clean_path,
                    start_chunk_id=len(chunks),
                )
                chunks.extend(sub_chunks)

        # 5. Fallback safety guard and sequential re-indexing
        return self.guard_chunks(chunks, strategy="split", reindex=True)

    def _subdivide_oversized_section(
        self,
        sec_content: str,
        sec_start: int,
        clean_path: str,
        start_chunk_id: int,
    ) -> List[InternalChunk]:
        """Subdivide oversized section along paragraph boundaries preserving code fences.
            sec_content: Text content of the oversized section.
            sec_start: 0-based character offset where section begins in source file.
            clean_path: File path string for InternalChunk metadata.
            start_chunk_id: Initial integer ID for generated sub-chunks.
        """
        sec_len = len(sec_content)
        fences = find_code_fence_intervals(sec_content)

        def is_offset_in_fence(pos: int) -> Optional[Tuple[int, int]]:
            """Return fence interval if pos is strictly inside a code fence."""
            for f_s, f_e in fences:
                if f_s < pos < f_e:
                    return (f_s, f_e)
            return None

        sub_chunks: List[InternalChunk] = []
        cursor = 0
        min_chunk_len = max(1, self._max_chunk_size // 4)

        while cursor < sec_len:
            remaining = sec_len - cursor
            if remaining <= self._max_chunk_size:
                slice_len = remaining
            else:
                window_end = cursor + self._max_chunk_size
                fence_overlap = is_offset_in_fence(window_end)

                chosen_split: Optional[int] = None

                # Case 1: Window boundary slices through a code fence
                if fence_overlap is not None:
                    f_start, f_end = fence_overlap
                    # If the code fence can fit on its own, break before it
                    if f_start > cursor and (f_end - f_start) <= self._max_chunk_size:
                        # Find paragraph break before code fence
                        pre_fence = sec_content[cursor:f_start]
                        last_p = pre_fence.rfind("\n\n")
                        if last_p != -1 and (cursor + last_p + 2) >= (cursor + min_chunk_len):
                            chosen_split = cursor + last_p + 2
                        else:
                            # Split cleanly right at fence start
                            chosen_split = f_start
                    elif f_start <= cursor:
                        # Fence itself is oversized and starts at/before cursor:
                        # Must split inside fence along newline
                        fence_window = sec_content[cursor:window_end]
                        last_nl = fence_window.rfind("\n")
                        if last_nl != -1 and last_nl >= min_chunk_len:
                            chosen_split = cursor + last_nl + 1
                        else:
                            chosen_split = window_end

                # Case 2: Window boundary does not cut a code fence
                if chosen_split is None:
                    window = sec_content[cursor:window_end]

                    # 1. Look for double newline (paragraph boundary) outside code fences
                    candidate_p = [
                        m.end()
                        for m in re.finditer(r"\n[ \t]*\n+", window)
                        if m.end() >= min_chunk_len
                        and is_offset_in_fence(cursor + m.end()) is None
                    ]
                    if candidate_p:
                        chosen_split = cursor + candidate_p[-1]
                    else:
                        # 2. Look for single newline outside code fences
                        candidate_nl = [
                            m.end()
                            for m in re.finditer(r"\n", window)
                            if m.end() >= min_chunk_len
                            and is_offset_in_fence(cursor + m.end()) is None
                        ]
                        if candidate_nl:
                            chosen_split = cursor + candidate_nl[-1]
                        else:
                            # 3. Look for space outside code fences
                            candidate_sp = [
                                m.end()
                                for m in re.finditer(r" ", window)
                                if m.end() >= min_chunk_len
                                and is_offset_in_fence(cursor + m.end()) is None
                            ]
                            if candidate_sp:
                                chosen_split = cursor + candidate_sp[-1]
                            else:
                                # 4. Hard cut at max_chunk_size
                                chosen_split = window_end

                # Guard against non-advancing cursor
                slice_len = max(1, min(chosen_split - cursor, self._max_chunk_size))

            chunk_content = sec_content[cursor : cursor + slice_len]
            c_first = sec_start + cursor
            c_last = c_first + len(chunk_content)

            sub_chunks.append(
                InternalChunk(
                    chunk_id=start_chunk_id + len(sub_chunks),
                    file_path=clean_path,
                    first_character_index=c_first,
                    last_character_index=c_last,
                    chunk_type="markdown_section",
                    token_count=self.estimate_token_count(chunk_content),
                    content=chunk_content,
                )
            )

            cursor += slice_len

        return sub_chunks


def chunk_markdown(
    file_path: Union[str, Path],
    content: Optional[str] = None,
    max_size: int = 2000,
) -> List[InternalChunk]:
    """Convenience function to chunk Markdown content or file.
        file_path: Source file path representation.
        content: Optional raw Markdown text. If omitted, reads from file_path.
        max_size: Maximum allowed character ceiling per chunk. Defaults to 2000.
    """
    chunker = MarkdownChunker(max_chunk_size=max_size)
    if content is not None:
        return chunker.chunk_text(content, file_path=file_path)
    return chunker.chunk_file(Path(file_path))
