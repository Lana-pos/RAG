"""AST-Based Python Code Chunker.

Parses Python source files into syntactically integral units (classes, methods,
standalone async/sync functions, and module docstrings/statements) using the standard
`ast` module, maintaining exact UTF-8 character offsets mapping to disk.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import (
    List,
    Optional,
    Sequence,
    Union,
)

from src.chunking.base import BaseChunker
from src.indexing.schema import ChunkType, InternalChunk


def is_valid_python(code: str) -> bool:
    """Check if a code snippet represents syntactically valid Python.
    Validates raw code, dedented code, signatures, and statement blocks
    using standard `ast.parse`.
        code: Python source code snippet to validate.
    """
    if not code or not code.strip():
        return True
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        pass

    dedented = textwrap.dedent(code)
    try:
        ast.parse(dedented)
        return True
    except SyntaxError:
        pass

    # Handle signatures (e.g. def foo(): or class Bar:)
    try:
        ast.parse(dedented.rstrip() + "\n    pass")
        return True
    except SyntaxError:
        pass

    # Handle triple-quote docstrings or strings
    for quote in ('"""', "'''"):
        try:
            ast.parse(dedented.rstrip() + "\n" + quote)
            return True
        except SyntaxError:
            pass
        try:
            ast.parse(quote + "\n" + dedented.lstrip())
            return True
        except SyntaxError:
            pass

    # Handle closing unclosed containers (lists, tuples, dicts, parameter lists)
    clean_ded = dedented.rstrip(" ,\n")
    for suffix in (
        "\n): pass",
        "\n):\n    pass",
        "\n]",
        "\n)",
        "\n}",
        "\n]\n    pass",
        "\n)\n    pass",
        "\n}\n    pass",
        ":\n    pass",
        "]",
        ")",
        "}",
    ):
        try:
            ast.parse(clean_ded + suffix)
            return True
        except SyntaxError:
            pass

    # Handle opening missing container prefixes and parameter lists
    clean_l_ded = dedented.lstrip(" ,\n")
    for prefix in (
        "def f(\n",
        "[\n",
        "(\n",
        "{\n",
    ):
        try:
            ast.parse(prefix + clean_l_ded)
            return True
        except SyntaxError:
            pass
        try:
            ast.parse(prefix + clean_l_ded + "    pass")
            return True
        except SyntaxError:
            pass

    return False


class PythonChunker(BaseChunker):
    """AST-based Python code chunker parsing files into syntactically integral units."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize Python chunker with character ceiling guard.
            max_chunk_size: Maximum allowed character length for any chunk.
                Defaults to 2000 characters.
        """
        super().__init__(max_chunk_size=max_chunk_size)

    @staticmethod
    def _compute_line_starts(lines: Sequence[str]) -> List[int]:
        """Compute cumulative starting character positions for each line."""
        line_starts = [0]
        for line in lines:
            line_starts.append(line_starts[-1] + len(line))
        return line_starts

    @staticmethod
    def _get_char_offset(
        lineno: int,
        col_offset: int,
        line_starts: Sequence[int],
        lines: Sequence[str],
        source_len: int,
    ) -> int:
        """Map 1-based line number and 0-based UTF-8 byte col_offset to character offset.
        CPython's AST reports col_offset in UTF-8 bytes within the line. This method
        accurately decodes up to col_offset bytes to compute the 0-based character index.
            lineno: 1-based line number from AST node.
            col_offset:column offset, it tells you how far across a line of
                Python code a particular piece of code starts or ends.
            line_starts: Precomputed line starting character indices.
            lines: List of line strings preserving line endings.
            source_len: Total character length of source file.
        """
        line_idx = lineno - 1
        if line_idx < 0:
            return 0
        if line_idx >= len(lines):
            return source_len

        line_str = lines[line_idx]
        line_start = line_starts[line_idx]
        line_bytes = line_str.encode("utf-8")

        # Clamp col_offset to line length in bytes
        safe_col = max(0, min(col_offset, len(line_bytes)))
        char_col = len(line_bytes[:safe_col].decode("utf-8", errors="replace"))
        return min(source_len, line_start + char_col)

    def chunk_file(self, file_path: Path) -> List[InternalChunk]:
        """Parse Python source file into discrete InternalChunk units using AST.
            file_path: Path to target source file.
        """
        source = self.read_file_safely(file_path)
        return self.chunk_text(source, file_path=file_path)

    def chunk_text(
        self,
        source: str,
        file_path: Union[str, Path] = "module.py",
    ) -> List[InternalChunk]:
        """Parse in-memory Python source text into discrete InternalChunk units.
            source: Python source code string.
            file_path: Relative or absolute path representation for metadata.
        """
        if not source or not source.strip():
            return []

        clean_path = str(file_path)

        # Attempt to parse AST; fall back to module-level character splitting if invalid syntax
        try:
            tree = ast.parse(source)
        except SyntaxError:
            fallback_chunk = InternalChunk(
                chunk_id=0,
                file_path=clean_path,
                first_character_index=0,
                last_character_index=len(source),
                chunk_type="python_module",
                token_count=self.estimate_token_count(source),
                content=source,
            )
            return self.guard_chunks([fallback_chunk], strategy="split", reindex=True)

        lines = source.splitlines(keepends=True)
        line_starts = self._compute_line_starts(lines)
        source_len = len(source)

        raw_chunks: List[tuple[ChunkType, int, int]] = []

        # 1. Detect and preserve standalone module docstring if present
        docstring_end_offset = 0
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            doc_node = tree.body[0]
            d_start = self._get_char_offset(
                doc_node.lineno,
                doc_node.col_offset,
                line_starts,
                lines,
                source_len,
            )
            d_end = self._get_char_offset(
                doc_node.end_lineno,  # type: ignore[arg-type]
                doc_node.end_col_offset,  # type: ignore[arg-type]
                line_starts,
                lines,
                source_len,
            )
            if d_end > d_start:
                raw_chunks.append(("python_module", d_start, d_end))
                docstring_end_offset = d_end

        # 2. Iterate through top-level nodes in tree.body
        # Group contiguous non-(class, func) statements as python_module chunks
        pending_module_stmts: List[ast.stmt] = []

        def flush_module_stmts(stmts: Sequence[ast.stmt]) -> None:
            if not stmts:
                return
            group_chunks = self._group_statements(
                stmts=stmts,
                chunk_type="python_module",
                source=source,
                line_starts=line_starts,
                lines=lines,
                source_len=source_len,
            )
            raw_chunks.extend(group_chunks)

        for stmt in tree.body:
            # Skip the already-emitted module docstring
            if docstring_end_offset > 0 and stmt is tree.body[0]:
                continue

            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                flush_module_stmts(pending_module_stmts)
                pending_module_stmts = []
                func_chunks = self._process_function_node(
                    node=stmt,
                    source=source,
                    line_starts=line_starts,
                    lines=lines,
                    source_len=source_len,
                )
                raw_chunks.extend(func_chunks)

            elif isinstance(stmt, ast.ClassDef):
                flush_module_stmts(pending_module_stmts)
                pending_module_stmts = []
                class_chunks = self._process_class_node(
                    node=stmt,
                    source=source,
                    line_starts=line_starts,
                    lines=lines,
                    source_len=source_len,
                )
                raw_chunks.extend(class_chunks)

            else:
                pending_module_stmts.append(stmt)

        flush_module_stmts(pending_module_stmts)

        # 3. Convert raw spans to validated InternalChunk instances
        chunks: List[InternalChunk] = []
        for ctype, start_idx, end_idx in raw_chunks:
            # Clamp bounds
            start_idx = max(0, min(start_idx, source_len))
            end_idx = max(start_idx + 1, min(end_idx, source_len))
            content = source[start_idx:end_idx]
            if not content.strip():
                continue
            chunks.append(
                InternalChunk(
                    chunk_id=len(chunks),
                    file_path=clean_path,
                    first_character_index=start_idx,
                    last_character_index=end_idx,
                    chunk_type=ctype,
                    token_count=self.estimate_token_count(content),
                    content=content,
                )
            )

        # If no chunks were created, fallback to single whole-file chunk
        if not chunks:
            chunks.append(
                InternalChunk(
                    chunk_id=0,
                    file_path=clean_path,
                    first_character_index=0,
                    last_character_index=source_len,
                    chunk_type="python_module",
                    token_count=self.estimate_token_count(source),
                    content=source,
                )
            )

        # 4. Enforce strict max_chunk_size ceiling and sequential reindexing
        return self.guard_chunks(chunks, strategy="split", reindex=True)

    def _get_node_span(
        self,
        node: ast.AST,
        line_starts: Sequence[int],
        lines: Sequence[str],
        source_len: int,
    ) -> tuple[int, int]:
        """Compute character span for an AST node using line boundaries for uniform indentation:
        Where does this function or class start?
        Where does it end?"""
        decorator_list = getattr(node, "decorator_list", None)
        if decorator_list:
            first_l = decorator_list[0].lineno
        else:
            first_l = getattr(node, "lineno", 1)

        end_l = getattr(node, "end_lineno", first_l)

        # Start from beginning of first line to preserve uniform base indentation across lines
        start_char = line_starts[first_l - 1] if 0 <= first_l - 1 < len(line_starts) else 0

        # End at end of line to include complete line content
        if end_l is not None and 0 <= end_l < len(line_starts):
            end_char = line_starts[end_l]
        else:
            end_col = getattr(node, "end_col_offset", 0)
            end_char = self._get_char_offset(end_l, end_col, line_starts, lines, source_len)

        return start_char, max(start_char + 1, min(source_len, end_char))

    def _process_function_node(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        source: str,
        line_starts: Sequence[int],
        lines: Sequence[str],
        source_len: int,
    ) -> List[tuple[ChunkType, int, int]]:
        """Process a sync or async function/method AST node.
        This function is responsible for processing a Python function and deciding
        whether it should become one chunk or be divided into smaller chunks."""
        s, e = self._get_node_span(node, line_starts, lines, source_len)
        content_len = e - s

        # Case 1: Function fits within max_chunk_size
        if content_len <= self.max_chunk_size:
            return [("python_func", s, e)]

        # Case 2: Oversized function — subdivide along internal logical blocks
        return self._subdivide_oversized_function(
            node=node,
            start_char=s,
            end_char=e,
            source=source,
            line_starts=line_starts,
            lines=lines,
            source_len=source_len,
        )

    def _subdivide_oversized_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        start_char: int,
        end_char: int,
        source: str,
        line_starts: Sequence[int],
        lines: Sequence[str],
        source_len: int,
    ) -> List[tuple[ChunkType, int, int]]:
        """Subdivide an oversized function along statement boundaries keeping syntax valid."""
        res: List[tuple[ChunkType, int, int]] = []
        body = node.body
        if not body:
            return [("python_func", start_char, min(end_char, start_char + self.max_chunk_size))]

        # Block 0: Function signature + initial statements
        first_stmt = body[0]
        end_l = getattr(first_stmt, "end_lineno", first_stmt.lineno)
        if end_l is not None and 0 <= end_l < len(line_starts):
            hdr_end = line_starts[end_l]
        else:
            hdr_end = self._get_char_offset(
                end_l,
                getattr(first_stmt, "end_col_offset", 0),
                line_starts,
                lines,
                source_len,
            )

        if hdr_end - start_char <= self.max_chunk_size:
            start_idx = 1
            curr_hdr_end = hdr_end
            while start_idx < len(body):
                stmt = body[start_idx]
                st_end_l = getattr(stmt, "end_lineno", stmt.lineno)
                if st_end_l is not None and 0 <= st_end_l < len(line_starts):
                    st_e = line_starts[st_end_l]
                else:
                    st_e = self._get_char_offset(
                        st_end_l,
                        getattr(stmt, "end_col_offset", 0),
                        line_starts,
                        lines,
                        source_len,
                    )
                if st_e - start_char <= self.max_chunk_size:
                    curr_hdr_end = st_e
                    start_idx += 1
                else:
                    break
            res.append(("python_func", start_char, curr_hdr_end))
        else:
            sig_end = (
                line_starts[first_stmt.lineno - 1]
                if 0 <= first_stmt.lineno - 1 < len(line_starts)
                else start_char
            )
            if sig_end > start_char:
                res.append(("python_func", start_char, sig_end))
            start_idx = 0

        # Subdivide remaining statements
        rem_stmts = body[start_idx:]
        if rem_stmts:
            res.extend(
                self._group_statements(
                    stmts=rem_stmts,
                    chunk_type="python_func",
                    source=source,
                    line_starts=line_starts,
                    lines=lines,
                    source_len=source_len,
                )
            )

        return res

    def _process_class_node(
        self,
        node: ast.ClassDef,
        source: str,
        line_starts: Sequence[int],
        lines: Sequence[str],
        source_len: int,
    ) -> List[tuple[ChunkType, int, int]]:
        """Process a ClassDef AST node.
        If the class fits, it creates a whole-class chunk and individual method chunks.
        If the class is too large, it calls _subdivide_oversized_class()."""
        s, e = self._get_node_span(node, line_starts, lines, source_len)
        content_len = e - s

        # Case 1: Class fits within max_chunk_size
        if content_len <= self.max_chunk_size:
            res: List[tuple[ChunkType, int, int]] = [("python_class", s, e)]
            # Also emit individual methods for high-granularity method indexing
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    res.extend(
                        self._process_function_node(
                            node=sub,
                            source=source,
                            line_starts=line_starts,
                            lines=lines,
                            source_len=source_len,
                        )
                    )
            return res

        # Case 2: Oversized class — subdivide along methods and preamble
        return self._subdivide_oversized_class(
            node=node,
            start_char=s,
            end_char=e,
            source=source,
            line_starts=line_starts,
            lines=lines,
            source_len=source_len,
        )

    def _subdivide_oversized_class(
        self,
        node: ast.ClassDef,
        start_char: int,
        end_char: int,
        source: str,
        line_starts: Sequence[int],
        lines: Sequence[str],
        source_len: int,
    ) -> List[tuple[ChunkType, int, int]]:
        """Subdivide an oversized class into preamble, methods, and nested classes.
        res will store all the chunks created by this method.
        body contains everything inside the class."""
        res: List[tuple[ChunkType, int, int]] = []
        body = node.body
        if not body:
            return [("python_class", start_char, min(end_char, start_char + self.max_chunk_size))]

        # Find index of first method or nested class
        first_m_idx = next(
            (
                i
                for i, m in enumerate(body)
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ),
            None,
        )

        if first_m_idx is not None:
            first_method = body[first_m_idx]
            first_m_s, _ = self._get_node_span(first_method, line_starts, lines, source_len)
            pre_stmts = body[:first_m_idx]
            if first_m_s > start_char:
                pre_len = first_m_s - start_char
                if pre_len <= self.max_chunk_size:
                    res.append(("python_class", start_char, first_m_s))
                elif pre_stmts:
                    res.extend(
                        self._group_statements(
                            stmts=pre_stmts,
                            chunk_type="python_class",
                            source=source,
                            line_starts=line_starts,
                            lines=lines,
                            source_len=source_len,
                            initial_start_char=start_char,
                        )
                    )
                else:
                    res.append(
                        ("python_class", start_char, min(first_m_s,
                                                         start_char + self.max_chunk_size))
                    )

            # Process all body items (methods, nested classes, and intervening attributes)
            inter_stmts: List[ast.stmt] = []
            seen_first_method = False
            for item in body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    seen_first_method = True
                    if inter_stmts:
                        res.extend(
                            self._group_statements(
                                stmts=inter_stmts,
                                chunk_type="python_class",
                                source=source,
                                line_starts=line_starts,
                                lines=lines,
                                source_len=source_len,
                            )
                        )
                        inter_stmts = []
                    res.extend(
                        self._process_function_node(
                            node=item,
                            source=source,
                            line_starts=line_starts,
                            lines=lines,
                            source_len=source_len,
                        )
                    )
                elif isinstance(item, ast.ClassDef):
                    seen_first_method = True
                    if inter_stmts:
                        res.extend(
                            self._group_statements(
                                stmts=inter_stmts,
                                chunk_type="python_class",
                                source=source,
                                line_starts=line_starts,
                                lines=lines,
                                source_len=source_len,
                            )
                        )
                        inter_stmts = []
                    res.extend(
                        self._process_class_node(
                            node=item,
                            source=source,
                            line_starts=line_starts,
                            lines=lines,
                            source_len=source_len,
                        )
                    )
                else:
                    if seen_first_method:
                        inter_stmts.append(item)

            if inter_stmts:
                res.extend(
                    self._group_statements(
                        stmts=inter_stmts,
                        chunk_type="python_class",
                        source=source,
                        line_starts=line_starts,
                        lines=lines,
                        source_len=source_len,
                    )
                )
        else:
            # Class without methods: group attributes and statements
            res.extend(
                self._group_statements(
                    stmts=body,
                    chunk_type="python_class",
                    source=source,
                    line_starts=line_starts,
                    lines=lines,
                    source_len=source_len,
                    initial_start_char=start_char,
                )
            )

        return res

    def _group_statements(
        self,
        stmts: Sequence[ast.stmt],
        chunk_type: ChunkType,
        source: str,
        line_starts: Sequence[int],
        lines: Sequence[str],
        source_len: int,
        initial_start_char: Optional[int] = None,
    ) -> List[tuple[ChunkType, int, int]]:
        """Group sequence of statements into syntactically integral chunks within max_chunk_size.
        A. Grouping
        Instead of creating one chunk for every small statement, it combines consecutive statements
        as long as the combined size stays within the limit.
        B. Recursion
        If a compound statement (such as a large if or for block) is too large, it tries to process
        the statements inside it separately."""
        if not stmts:
            return []

        res: List[tuple[ChunkType, int, int]] = []
        idx = 0

        while idx < len(stmts):
            stmt = stmts[idx]
            line_idx = stmt.lineno - 1
            if idx == 0 and initial_start_char is not None:
                st_s = initial_start_char
            else:
                st_s = line_starts[line_idx] if 0 <= line_idx < len(line_starts) else 0

            st_e = self._get_char_offset(
                getattr(stmt, "end_lineno", stmt.lineno),
                getattr(stmt, "end_col_offset", 0),
                line_starts,
                lines,
                source_len,
            )

            # If a single compound statement exceeds max_chunk_size, recurse into its body
            if st_e - st_s > self.max_chunk_size and hasattr(stmt, "body") and stmt.body:
                inner_stmts = [s for s in stmt.body if isinstance(s, ast.stmt)]
                if inner_stmts:
                    res.extend(
                        self._group_statements(
                            stmts=inner_stmts,
                            chunk_type=chunk_type,
                            source=source,
                            line_starts=line_starts,
                            lines=lines,
                            source_len=source_len,
                            initial_start_char=st_s,
                        )
                    )
                    idx += 1
                    continue

            # Accumulate statements while within max_chunk_size
            block_start = st_s
            block_end = st_e
            while idx + 1 < len(stmts):
                next_stmt = stmts[idx + 1]
                next_e = self._get_char_offset(
                    getattr(next_stmt, "end_lineno", next_stmt.lineno),
                    getattr(next_stmt, "end_col_offset", 0),
                    line_starts,
                    lines,
                    source_len,
                )
                if next_e - block_start <= self.max_chunk_size:
                    block_end = next_e
                    idx += 1
                else:
                    break

            res.append((chunk_type, block_start, block_end))
            idx += 1

        return res
