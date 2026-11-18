"""Unit tests for AST-Based Python Code Chunker in src/chunking/python.py."""

from pathlib import Path
from typing import List

from src.chunking.python import PythonChunker, is_valid_python
from src.indexing.schema import InternalChunk


class TestPythonChunker:
    """Tests for PythonChunker detecting classes, functions, docstrings and offsets."""

    def test_chunker_initialization(self) -> None:
        chunker = PythonChunker(max_chunk_size=1500)
        assert chunker.max_chunk_size == 1500

    def test_detect_module_docstring(self, tmp_path: Path) -> None:
        code = '''"""This is a test module docstring.
It describes the module in detail.
"""

import os
'''
        test_file = tmp_path / "module_doc.py"
        test_file.write_text(code, encoding="utf-8")

        chunker = PythonChunker()
        chunks: List[InternalChunk] = chunker.chunk_file(test_file)

        assert len(chunks) >= 1
        doc_chunk = chunks[0]
        expected_doc = (
            '"""This is a test module docstring.\n'
            'It describes the module in detail.\n"""'
        )
        assert doc_chunk.first_character_index == 0
        assert doc_chunk.last_character_index == len(expected_doc)
        span_slice = code[doc_chunk.first_character_index:doc_chunk.last_character_index]
        assert span_slice == doc_chunk.content

    def test_detect_sync_and_async_functions(self, tmp_path: Path) -> None:
        code = '''def sync_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

async def async_fetch(url: str) -> str:
    """Fetch URL asynchronously."""
    return f"data from {url}"
'''
        test_file = tmp_path / "funcs.py"
        test_file.write_text(code, encoding="utf-8")

        chunker = PythonChunker()
        chunks = chunker.chunk_file(test_file)

        func_chunks = [c for c in chunks if c.chunk_type == "python_func"]
        assert len(func_chunks) == 2

        # Verify sync function
        sync_chunk = next(c for c in func_chunks if "sync_add" in c.content)
        assert "def sync_add" in sync_chunk.content
        assert "return a + b" in sync_chunk.content
        sync_slice = code[sync_chunk.first_character_index:sync_chunk.last_character_index]
        assert sync_slice == sync_chunk.content
        assert is_valid_python(sync_chunk.content)

        # Verify async function
        async_chunk = next(c for c in func_chunks if "async_fetch" in c.content)
        assert "async def async_fetch" in async_chunk.content
        async_slice = code[async_chunk.first_character_index:async_chunk.last_character_index]
        assert async_slice == async_chunk.content
        assert is_valid_python(async_chunk.content)

    def test_detect_classes_and_methods(self, tmp_path: Path) -> None:
        code = '''class Calculator:
    """A simple calculator class."""

    def __init__(self, base: int = 0) -> None:
        self.base = base

    def add(self, val: int) -> int:
        return self.base + val
'''
        test_file = tmp_path / "calc.py"
        test_file.write_text(code, encoding="utf-8")

        chunker = PythonChunker()
        chunks = chunker.chunk_file(test_file)

        class_chunks = [c for c in chunks if c.chunk_type == "python_class"]
        assert len(class_chunks) == 1
        assert "class Calculator:" in class_chunks[0].content
        assert is_valid_python(class_chunks[0].content)

        func_chunks = [c for c in chunks if c.chunk_type == "python_func"]
        assert len(func_chunks) == 2
        method_names = [c.content for c in func_chunks]
        assert any("__init__" in m for m in method_names)
        assert any("def add" in m for m in method_names)
        for fc in func_chunks:
            assert is_valid_python(fc.content)
            assert code[fc.first_character_index:fc.last_character_index] == fc.content

    def test_decorators_included_in_spans(self, tmp_path: Path) -> None:
        code = '''@staticmethod
def static_worker() -> None:
    pass

class Service:
    @property
    def is_active(self) -> bool:
        return True
'''
        test_file = tmp_path / "decorated.py"
        test_file.write_text(code, encoding="utf-8")

        chunker = PythonChunker()
        chunks = chunker.chunk_file(test_file)

        worker_chunk = next(c for c in chunks if "static_worker" in c.content)
        assert worker_chunk.content.startswith("@staticmethod")
        w_slice = code[worker_chunk.first_character_index:worker_chunk.last_character_index]
        assert w_slice == worker_chunk.content

        prop_chunk = next(c for c in chunks if "is_active" in c.content)
        assert "@property" in prop_chunk.content
        p_slice = code[prop_chunk.first_character_index:prop_chunk.last_character_index]
        assert p_slice == prop_chunk.content

    def test_exact_utf8_character_offsets_with_multibyte_chars(self, tmp_path: Path) -> None:
        code = '''# Header with emoji 🚀 and accent café
"""Docstring with em-dash — and symbols ✨"""

def greet(name: str) -> str:
    # Say hello with special character 🌟
    return f"Bonjour {name} ☕"
'''
        test_file = tmp_path / "unicode_test.py"
        test_file.write_text(code, encoding="utf-8")

        chunker = PythonChunker()
        chunks = chunker.chunk_file(test_file)

        for chunk in chunks:
            expected_slice = code[chunk.first_character_index:chunk.last_character_index]
            assert chunk.content == expected_slice
            assert chunk.span_length == len(chunk.content)
            assert chunk.last_character_index > chunk.first_character_index
            assert is_valid_python(chunk.content)

    def test_subdivide_oversized_class(self, tmp_path: Path) -> None:
        # Create a class that exceeds max_chunk_size
        methods = []
        for i in range(10):
            methods.append(
                f"    def compute_step_{i:02d}(self, val: int) -> int:\n"
                f"        '''Step {i} documentation.'''\n"
                f"        intermediate = val * {i + 1}\n"
                f"        return intermediate + {i}\n"
            )
        class_body = "\n".join(methods)
        code = f'''class LargePipeline:
    """Pipeline coordinator with multiple step executions."""
    PIPELINE_VERSION = "2.0.0"

{class_body}
'''
        test_file = tmp_path / "large_class.py"
        test_file.write_text(code, encoding="utf-8")

        chunker = PythonChunker(max_chunk_size=500)
        chunks = chunker.chunk_file(test_file)

        # Verify all chunks strictly adhere to max_chunk_size
        for chunk in chunks:
            assert chunk.span_length <= 500
            assert len(chunk.content) <= 500
            assert code[chunk.first_character_index:chunk.last_character_index] == chunk.content
            assert is_valid_python(chunk.content)

        # Verify class preamble was captured
        class_chunks = [c for c in chunks if c.chunk_type == "python_class"]
        assert any("LargePipeline" in c.content for c in class_chunks)

        # Verify methods were captured
        func_chunks = [c for c in chunks if c.chunk_type == "python_func"]
        assert len(func_chunks) >= 10
        for i in range(10):
            assert any(f"compute_step_{i:02d}" in c.content for c in func_chunks)

    def test_subdivide_oversized_function(self, tmp_path: Path) -> None:
        # Create a function that exceeds max_chunk_size
        stmts = []
        for i in range(25):
            stmts.append(
                f"    step_{i:02d} = input_data * {i + 1}\n"
                f"    if step_{i:02d} > 100:\n"
                f"        result_accumulator.append(step_{i:02d})\n"
            )
        func_body = "".join(stmts)
        code = f'''def complex_workflow(input_data: int) -> list:
    """Execute a complex sequential workflow."""
    result_accumulator = []
{func_body}    return result_accumulator
'''
        test_file = tmp_path / "large_func.py"
        test_file.write_text(code, encoding="utf-8")

        chunker = PythonChunker(max_chunk_size=600)
        chunks = chunker.chunk_file(test_file)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.span_length <= 600
            assert len(chunk.content) <= 600
            assert code[chunk.first_character_index:chunk.last_character_index] == chunk.content
            assert is_valid_python(chunk.content)
            assert chunk.chunk_type == "python_func"

        # Verify header block captures function signature
        assert "def complex_workflow" in chunks[0].content
        # Verify subsequent blocks capture inner statements
        assert any("step_20" in c.content for c in chunks)

    def test_empty_and_whitespace_files(self, tmp_path: Path) -> None:
        chunker = PythonChunker()

        empty_file = tmp_path / "empty.py"
        empty_file.write_text("", encoding="utf-8")
        assert chunker.chunk_file(empty_file) == []

        ws_file = tmp_path / "whitespace.py"
        ws_file.write_text("   \n\n\t  \n", encoding="utf-8")
        assert chunker.chunk_file(ws_file) == []

    def test_syntax_error_fallback(self, tmp_path: Path) -> None:
        bad_code = "def incomplete_func(\n    x = 1\n"
        bad_file = tmp_path / "bad.py"
        bad_file.write_text(bad_code, encoding="utf-8")

        chunker = PythonChunker()
        chunks = chunker.chunk_file(bad_file)

        assert len(chunks) == 1
        assert chunks[0].chunk_type == "python_module"
        assert chunks[0].content == bad_code

    def test_module_level_constants_and_imports(self, tmp_path: Path) -> None:
        code = '''import sys
from typing import Dict, List

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 5
CONFIG_REGISTRY: Dict[str, int] = {
    "timeout": DEFAULT_TIMEOUT,
    "retries": MAX_RETRIES,
}
'''
        test_file = tmp_path / "constants.py"
        test_file.write_text(code, encoding="utf-8")

        chunker = PythonChunker()
        chunks = chunker.chunk_file(test_file)

        assert len(chunks) >= 1
        assert all(c.chunk_type == "python_module" for c in chunks)
        assert any("CONFIG_REGISTRY" in c.content for c in chunks)
        for c in chunks:
            assert code[c.first_character_index:c.last_character_index] == c.content

    def test_sequential_chunk_indexing(self, tmp_path: Path) -> None:
        code = '''"""Doc."""
def f1(): pass
def f2(): pass
class C1:
    def m1(self): pass
'''
        test_file = tmp_path / "seq.py"
        test_file.write_text(code, encoding="utf-8")

        chunker = PythonChunker()
        chunks = chunker.chunk_file(test_file)

        # Verify sequential chunk_ids starting from 0
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == i

    def test_callable_interface(self, tmp_path: Path) -> None:
        code = "def run(): pass\n"
        test_file = tmp_path / "run.py"
        test_file.write_text(code, encoding="utf-8")

        chunker = PythonChunker()
        chunks_via_call = chunker(test_file)
        chunks_via_method = chunker.chunk_file(test_file)

        assert chunks_via_call == chunks_via_method
