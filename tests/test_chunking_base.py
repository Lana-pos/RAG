"""Unit tests for BaseChunker and FileTraversalEngine in src/chunking/base.py."""

from pathlib import Path
from typing import List
import pytest

from src.chunking.base import (
    DEFAULT_IGNORE_DIRS,
    DEFAULT_SUPPORTED_EXTENSIONS,
    BaseChunker,
    FileTraversalEngine,
    discover_files,
    is_binary_file,
)
from src.indexing.schema import InternalChunk


class DummyChunker(BaseChunker):
    """Concrete implementation of BaseChunker for testing."""

    def chunk_file(self, file_path: Path) -> List[InternalChunk]:
        content = self.read_file_safely(file_path)
        chunk = InternalChunk(
            chunk_id=0,
            file_path=str(file_path),
            first_character_index=0,
            last_character_index=len(content),
            chunk_type="python_module",
            token_count=self.estimate_token_count(content),
            content=content,
        )
        return self.guard_chunks([chunk])


class TestFileTraversalEngine:
    """Tests for FileTraversalEngine, discover_files, and binary filtering."""

    def test_default_constants(self) -> None:
        assert ".py" in DEFAULT_SUPPORTED_EXTENSIONS
        assert ".md" in DEFAULT_SUPPORTED_EXTENSIONS
        assert ".git" in DEFAULT_IGNORE_DIRS
        assert ".venv" in DEFAULT_IGNORE_DIRS
        assert "venv" in DEFAULT_IGNORE_DIRS
        assert "__pycache__" in DEFAULT_IGNORE_DIRS

    def test_is_binary_file(self, tmp_path: Path) -> None:
        text_file = tmp_path / "hello.py"
        text_file.write_text("print('hello world')", encoding="utf-8")
        assert not is_binary_file(text_file)

        bin_file = tmp_path / "blob.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03binarydata")
        assert is_binary_file(bin_file)

        # Non-existent file returns False safely
        assert not is_binary_file(tmp_path / "missing.txt")

    def test_file_discovery_with_filtering(self, tmp_path: Path) -> None:
        # 1. Valid files
        valid_py = tmp_path / "module.py"
        valid_py.write_text("def test(): pass", encoding="utf-8")

        sub_dir = tmp_path / "subpkg"
        sub_dir.mkdir()
        valid_md = sub_dir / "README.md"
        valid_md.write_text("# Documentation", encoding="utf-8")

        # 2. Ignored files/directories
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config.py").write_text("git config", encoding="utf-8")

        github_dir = tmp_path / ".github"
        github_dir.mkdir()
        (github_dir / "ci.py").write_text("ci script", encoding="utf-8")

        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        (venv_dir / "site.py").write_text("site", encoding="utf-8")

        venv_plain = tmp_path / "venv"
        venv_plain.mkdir()
        (venv_plain / "lib.py").write_text("lib", encoding="utf-8")

        pycache_dir = tmp_path / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "module.cpython-310.py").write_text("cached", encoding="utf-8")

        # 3. Disallowed extensions
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("some notes", encoding="utf-8")

        # 4. Binary file pretending to be .py
        binary_py = tmp_path / "fake.py"
        binary_py.write_bytes(b"\x00\x00\xff\xfeprint('fake')")

        # 5. Hidden file
        hidden_py = tmp_path / ".secret.py"
        hidden_py.write_text("secret", encoding="utf-8")

        discovered = discover_files(tmp_path)
        discovered_names = [p.name for p in discovered]

        assert "module.py" in discovered_names
        assert "README.md" in discovered_names
        assert len(discovered) == 2
        assert "config.py" not in discovered_names
        assert "ci.py" not in discovered_names
        assert "site.py" not in discovered_names
        assert "lib.py" not in discovered_names
        assert "notes.txt" not in discovered_names
        assert "fake.py" not in discovered_names
        assert ".secret.py" not in discovered_names

    def test_custom_extensions_and_ignored_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("print(1)")
        (tmp_path / "doc.md").write_text("# Doc")
        (tmp_path / "data.json").write_text("{}")

        engine = FileTraversalEngine(allowed_extensions=[".json"])
        discovered = engine.discover(tmp_path)
        assert len(discovered) == 1
        assert discovered[0].name == "data.json"

    def test_discover_non_existent_or_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_files(tmp_path / "does_not_exist")

        f = tmp_path / "file.py"
        f.write_text("pass")
        with pytest.raises(NotADirectoryError):
            discover_files(f)

    def test_real_corpus_discovery(self) -> None:
        corpus = Path("data/raw/vllm-0.10.1")
        if not corpus.exists():
            pytest.skip("data/raw/vllm-0.10.1 not present")

        discovered = discover_files(corpus)
        assert len(discovered) > 1000

        # Check all discovered files are .py or .md
        for path in discovered:
            assert path.suffix.lower() in {".py", ".md"}
            assert not path.name.startswith(".")
            parts = [part.lower() for part in path.parts]
            assert ".git" not in parts
            assert ".github" not in parts
            assert ".venv" not in parts
            assert "__pycache__" not in parts


class TestBaseChunker:
    """Tests for BaseChunker abstract class and utilities."""

    def test_cannot_instantiate_base_chunker_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseChunker()  # type: ignore[abstract]

    def test_valid_instantiation_of_subclass(self) -> None:
        chunker = DummyChunker(max_chunk_size=1500)
        assert chunker.max_chunk_size == 1500

    def test_invalid_max_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError):
            DummyChunker(max_chunk_size=0)
        with pytest.raises(ValueError):
            DummyChunker(max_chunk_size=-100)

    def test_token_count_estimation(self) -> None:
        assert DummyChunker.estimate_token_count("") == 0
        assert DummyChunker.estimate_token_count("hello world") == 2
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        assert DummyChunker.estimate_token_count(code) > 10

    def test_read_file_safely(self, tmp_path: Path) -> None:
        test_file = tmp_path / "sample.py"
        test_file.write_text("x = 42\n", encoding="utf-8")
        content = DummyChunker.read_file_safely(test_file)
        assert content == "x = 42\n"

        with pytest.raises(FileNotFoundError):
            DummyChunker.read_file_safely(tmp_path / "not_found.py")

    def test_callable_chunker(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("y = 100\n", encoding="utf-8")
        chunker = DummyChunker()
        chunks = chunker(test_file)
        assert len(chunks) == 1
        assert chunks[0].content == "y = 100\n"


class TestHardGuard:
    """Tests for clamping, splitting, and hard max_chunk_size enforcement."""

    def test_chunk_within_limit_not_modified(self) -> None:
        chunker = DummyChunker(max_chunk_size=2000)
        chunk = InternalChunk(
            chunk_id=1,
            file_path="data/raw/vllm-0.10.1/vllm/test.py",
            first_character_index=0,
            last_character_index=500,
            chunk_type="python_func",
            token_count=50,
            content="a" * 500,
        )
        guarded = chunker.enforce_max_chunk_size(chunk)
        assert len(guarded) == 1
        assert guarded[0] == chunk

    def test_clamp_oversized_chunk(self) -> None:
        chunker = DummyChunker(max_chunk_size=2000)
        chunk = InternalChunk(
            chunk_id=1,
            file_path="data/raw/vllm-0.10.1/vllm/test.py",
            first_character_index=100,
            last_character_index=4100,
            chunk_type="python_func",
            token_count=400,
            content="x" * 4000,
        )
        clamped = chunker.clamp_chunk(chunk)
        assert clamped.span_length == 2000
        assert len(clamped.content) == 2000
        assert clamped.first_character_index == 100
        assert clamped.last_character_index == 2100
        assert clamped.chunk_id == 1

    def test_split_oversized_chunk_with_paragraphs(self) -> None:
        chunker = DummyChunker(max_chunk_size=500)
        # 3 paragraphs of 300 chars each = ~900 chars
        p1 = "First paragraph. " + ("a" * 280) + "\n\n"
        p2 = "Second paragraph. " + ("b" * 280) + "\n\n"
        p3 = "Third paragraph. " + ("c" * 280)
        content = p1 + p2 + p3

        chunk = InternalChunk(
            chunk_id=10,
            file_path="data/raw/vllm-0.10.1/docs/guide.md",
            first_character_index=100,
            last_character_index=100 + len(content),
            chunk_type="markdown_section",
            token_count=100,
            content=content,
        )

        sub_chunks = chunker.split_chunk(chunk)
        assert len(sub_chunks) >= 2

        # Verify all sub-chunks obey ceiling
        for sc in sub_chunks:
            assert sc.span_length <= 500
            assert len(sc.content) <= 500
            assert sc.last_character_index > sc.first_character_index
            assert sc.file_path == chunk.file_path

        # Verify exact contiguous span coverage
        for i in range(len(sub_chunks) - 1):
            assert sub_chunks[i].last_character_index == sub_chunks[i + 1].first_character_index

        assert sub_chunks[0].first_character_index == chunk.first_character_index
        assert sub_chunks[-1].last_character_index == chunk.last_character_index

    def test_split_oversized_chunk_with_lines(self) -> None:
        chunker = DummyChunker(max_chunk_size=400)
        lines = [f"line {i:03d} " + ("d" * 50) + "\n" for i in range(20)]
        content = "".join(lines)

        chunk = InternalChunk(
            chunk_id=5,
            file_path="data/raw/vllm-0.10.1/vllm/lines.py",
            first_character_index=0,
            last_character_index=len(content),
            chunk_type="python_module",
            token_count=150,
            content=content,
        )

        sub_chunks = chunker.split_chunk(chunk)
        assert len(sub_chunks) > 1

        for sc in sub_chunks:
            assert sc.span_length <= 400
            assert len(sc.content) <= 400
            assert sc.last_character_index > sc.first_character_index

    def test_split_oversized_unbroken_string(self) -> None:
        chunker = DummyChunker(max_chunk_size=300)
        content = "Z" * 1000

        chunk = InternalChunk(
            chunk_id=1,
            file_path="data/raw/vllm-0.10.1/vllm/unbroken.py",
            first_character_index=50,
            last_character_index=1050,
            chunk_type="python_module",
            token_count=10,
            content=content,
        )

        sub_chunks = chunker.split_chunk(chunk)
        assert len(sub_chunks) == 4  # 300 + 300 + 300 + 100 = 1000

        for sc in sub_chunks:
            assert sc.span_length <= 300
            assert len(sc.content) <= 300

        # Exact contiguous coverage
        assert sub_chunks[0].first_character_index == 50
        assert sub_chunks[-1].last_character_index == 1050

    def test_guard_chunks_batch_with_reindexing(self) -> None:
        chunker = DummyChunker(max_chunk_size=500)
        chunk_small = InternalChunk(
            chunk_id=99,
            file_path="data/raw/vllm-0.10.1/vllm/small.py",
            first_character_index=0,
            last_character_index=100,
            chunk_type="python_func",
            token_count=20,
            content="small chunk",
        )
        chunk_large = InternalChunk(
            chunk_id=100,
            file_path="data/raw/vllm-0.10.1/vllm/large.py",
            first_character_index=0,
            last_character_index=1200,
            chunk_type="python_class",
            token_count=200,
            content="x" * 1200,
        )

        guarded = chunker.guard_chunks([chunk_small, chunk_large], strategy="split", reindex=True)
        assert len(guarded) > 2
        # Verify sequential chunk IDs
        for expected_id, chunk in enumerate(guarded):
            assert chunk.chunk_id == expected_id
            assert chunk.span_length <= 500
            assert len(chunk.content) <= 500

    def test_enforce_max_chunk_size_invalid_strategy_raises(self) -> None:
        chunker = DummyChunker(max_chunk_size=100)
        chunk = InternalChunk(
            chunk_id=1,
            file_path="data/raw/vllm-0.10.1/vllm/test.py",
            first_character_index=0,
            last_character_index=200,
            chunk_type="python_func",
            token_count=20,
            content="x" * 200,
        )
        with pytest.raises(ValueError):
            chunker.enforce_max_chunk_size(chunk, strategy="unknown")  # type: ignore[arg-type]
