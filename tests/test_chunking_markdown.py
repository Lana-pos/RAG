"""Unit tests for Heading-Aware Markdown Chunker in src/chunking/markdown.py."""

from pathlib import Path
from typing import List
import pytest

from src.chunking.markdown import (
    MarkdownChunker,
    chunk_markdown,
    find_code_fence_intervals,
)
from src.indexing.schema import InternalChunk


class TestMarkdownChunker:
    """Tests for MarkdownChunker detecting headings, code fences, and offsets."""

    def test_chunker_initialization(self) -> None:
        chunker = MarkdownChunker(max_chunk_size=1500)
        assert chunker.max_chunk_size == 1500

        with pytest.raises(ValueError, match="max_chunk_size must be positive"):
            MarkdownChunker(max_chunk_size=0)

        with pytest.raises(ValueError, match="max_chunk_size must be positive"):
            MarkdownChunker(max_chunk_size=-10)

    def test_empty_and_whitespace_markdown(self, tmp_path: Path) -> None:
        chunker = MarkdownChunker()

        empty_file = tmp_path / "empty.md"
        empty_file.write_text("", encoding="utf-8")
        assert chunker.chunk_file(empty_file) == []

        ws_file = tmp_path / "whitespace.md"
        ws_file.write_text("   \n\n\t  \n", encoding="utf-8")
        assert chunker.chunk_file(ws_file) == []
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   \n\t  \n") == []

    def test_single_heading_section(self, tmp_path: Path) -> None:
        doc = (
            "# Getting Started\n\n"
            "Welcome to the project documentation.\n"
            "This guide walks through initial setup.\n"
        )
        test_file = tmp_path / "single.md"
        test_file.write_text(doc, encoding="utf-8")

        chunker = MarkdownChunker()
        chunks: List[InternalChunk] = chunker.chunk_file(test_file)

        assert len(chunks) == 1
        c = chunks[0]
        assert c.chunk_type == "markdown_section"
        assert c.first_character_index == 0
        assert c.last_character_index == len(doc)
        assert c.content == doc
        assert c.span_length == len(doc)
        assert doc[c.first_character_index:c.last_character_index] == c.content

    def test_hierarchical_headings_segmentation(self, tmp_path: Path) -> None:
        doc = (
            "# Main Architecture\n\n"
            "Overview of the system architecture.\n\n"
            "## Storage Layer\n\n"
            "Details about persistent storage and indexes.\n\n"
            "### Inverted Index\n\n"
            "BM25 index representation.\n\n"
            "### Dense Vectors\n\n"
            "MiniLM vector storage.\n\n"
            "## Serving Engine\n\n"
            "FastAPI and Uvicorn orchestration.\n"
        )
        test_file = tmp_path / "hierarchy.md"
        test_file.write_text(doc, encoding="utf-8")

        chunker = MarkdownChunker()
        chunks = chunker.chunk_file(test_file)

        assert len(chunks) == 5

        # Verify chunk boundaries map to headers
        assert chunks[0].content.startswith("# Main Architecture")
        assert chunks[1].content.startswith("## Storage Layer")
        assert chunks[2].content.startswith("### Inverted Index")
        assert chunks[3].content.startswith("### Dense Vectors")
        assert chunks[4].content.startswith("## Serving Engine")

        # Verify contiguous coverage across the whole file
        assert chunks[0].first_character_index == 0
        assert chunks[-1].last_character_index == len(doc)

        for i in range(len(chunks) - 1):
            assert chunks[i].last_character_index == chunks[i + 1].first_character_index

        for c in chunks:
            assert c.chunk_type == "markdown_section"
            assert doc[c.first_character_index:c.last_character_index] == c.content
            assert c.span_length == len(c.content)

    def test_preamble_before_first_heading(self, tmp_path: Path) -> None:
        doc = (
            "---\n"
            "title: Project Guide\n"
            "author: Core Team\n"
            "---\n\n"
            "This is an introductory preamble before any section.\n\n"
            "# Section One\n\n"
            "First section content.\n"
        )
        test_file = tmp_path / "preamble.md"
        test_file.write_text(doc, encoding="utf-8")

        chunker = MarkdownChunker()
        chunks = chunker.chunk_file(test_file)

        assert len(chunks) == 2
        # Preamble chunk
        assert chunks[0].first_character_index == 0
        assert "title: Project Guide" in chunks[0].content
        assert chunks[0].last_character_index == doc.index("# Section One")

        # First section chunk
        assert chunks[1].first_character_index == doc.index("# Section One")
        assert chunks[1].content.startswith("# Section One")
        assert chunks[1].last_character_index == len(doc)

        # Offsets are contiguous and match text
        assert chunks[0].last_character_index == chunks[1].first_character_index
        for c in chunks:
            assert doc[c.first_character_index:c.last_character_index] == c.content

    def test_document_without_headings(self, tmp_path: Path) -> None:
        doc = (
            "This is a markdown document without any explicit headings.\n\n"
            "It has paragraphs of text and formatting like **bold** and *italic*.\n\n"
            "Everything should be captured cleanly in a single chunk.\n"
        )
        test_file = tmp_path / "no_headers.md"
        test_file.write_text(doc, encoding="utf-8")

        chunker = MarkdownChunker()
        chunks = chunker.chunk_file(test_file)

        assert len(chunks) == 1
        assert chunks[0].first_character_index == 0
        assert chunks[0].last_character_index == len(doc)
        assert chunks[0].content == doc
        assert chunks[0].chunk_type == "markdown_section"

    def test_code_fences_with_hash_comments_not_treated_as_headings(
        self,
        tmp_path: Path,
    ) -> None:
        doc = (
            "# Python Examples\n\n"
            "Here is how you define a function:\n\n"
            "```python\n"
            "# This is a python comment, not a markdown heading!\n"
            "## Nor is this an H2 heading!\n"
            "### Nor is this an H3 heading!\n"
            "def calculate(val: int) -> int:\n"
            "    # Another comment\n"
            "    return val * 2\n"
            "```\n\n"
            "And here is a bash script:\n\n"
            "~~~bash\n"
            "# Bash comment\n"
            "## Not a header either\n"
            "echo 'running job'\n"
            "~~~\n\n"
            "## Next True Heading\n\n"
            "This is the actual second section.\n"
        )
        test_file = tmp_path / "code_fence.md"
        test_file.write_text(doc, encoding="utf-8")

        chunker = MarkdownChunker()
        chunks = chunker.chunk_file(test_file)

        # There are only two true headings: "# Python Examples" and "## Next True Heading"
        assert len(chunks) == 2

        section_1 = chunks[0]
        assert section_1.content.startswith("# Python Examples")
        assert "```python" in section_1.content
        assert "# This is a python comment" in section_1.content
        assert "~~~bash" in section_1.content
        assert "# Bash comment" in section_1.content

        section_2 = chunks[1]
        assert section_2.content.startswith("## Next True Heading")
        assert "This is the actual second section." in section_2.content

        for c in chunks:
            assert doc[c.first_character_index:c.last_character_index] == c.content

    def test_nested_backticks_in_code_fence(self, tmp_path: Path) -> None:
        doc = (
            "# Code Fence Levels\n\n"
            "````markdown\n"
            "```python\n"
            "# nested comment\n"
            "print('hello')\n"
            "```\n"
            "````\n\n"
            "## Following Section\n\n"
            "End of doc.\n"
        )
        intervals = find_code_fence_intervals(doc)
        assert len(intervals) == 1
        fence_slice = doc[intervals[0][0]:intervals[0][1]]
        assert fence_slice.startswith("````markdown")
        assert fence_slice.strip().endswith("````")

        chunker = MarkdownChunker()
        chunks = chunker.chunk_text(doc)
        assert len(chunks) == 2
        assert chunks[0].content.startswith("# Code Fence Levels")
        assert chunks[1].content.startswith("## Following Section")

    def test_subdivide_oversized_section_at_paragraph_boundaries(
        self,
        tmp_path: Path,
    ) -> None:
        # Create a section with 3 distinct paragraphs exceeding max_chunk_size
        p1 = "Paragraph 1: " + ("A" * 250) + "\n\n"
        p2 = "Paragraph 2: " + ("B" * 250) + "\n\n"
        p3 = "Paragraph 3: " + ("C" * 250) + "\n"
        doc = "# Large Section\n\n" + p1 + p2 + p3

        test_file = tmp_path / "large_section.md"
        test_file.write_text(doc, encoding="utf-8")

        chunker = MarkdownChunker(max_chunk_size=400)
        chunks = chunker.chunk_file(test_file)

        assert len(chunks) >= 2

        # Verify strict character ceiling
        for c in chunks:
            assert len(c.content) <= 400
            assert c.span_length <= 400
            assert c.span_length == len(c.content)
            assert c.last_character_index > c.first_character_index
            assert doc[c.first_character_index:c.last_character_index] == c.content
            assert c.chunk_type == "markdown_section"

        # Verify contiguous coverage
        assert chunks[0].first_character_index == 0
        assert chunks[-1].last_character_index == len(doc)
        for i in range(len(chunks) - 1):
            assert chunks[i].last_character_index == chunks[i + 1].first_character_index

        # Verify sequential chunk IDs
        for idx, c in enumerate(chunks):
            assert c.chunk_id == idx

    def test_oversized_section_preserves_code_block_intact(
        self,
        tmp_path: Path,
    ) -> None:
        # Preamble text of ~200 chars, code block of ~250 chars, follow-up of ~200 chars
        # Total section ~650 chars. With max_chunk_size=450, code block would be cut
        # if split naively, but must be kept intact in its own chunk.
        pre = "## Feature Overview\n\n" + ("Introductory text. " * 8) + "\n\n"
        code = "```python\n" + ("x = 100\n" * 15) + "```\n\n"
        post = ("Concluding remarks. " * 8) + "\n"
        doc = pre + code + post

        test_file = tmp_path / "code_split.md"
        test_file.write_text(doc, encoding="utf-8")

        chunker = MarkdownChunker(max_chunk_size=450)
        chunks = chunker.chunk_file(test_file)

        assert len(chunks) >= 2

        for c in chunks:
            assert len(c.content) <= 450
            assert c.span_length <= 450
            assert doc[c.first_character_index:c.last_character_index] == c.content

        # Verify the entire code block is preserved intact in one chunk
        code_chunks = [c for c in chunks if "```python" in c.content]
        assert len(code_chunks) == 1
        assert "```\n" in code_chunks[0].content
        assert ("x = 100\n" * 15) in code_chunks[0].content

    def test_oversized_code_fence_split_gracefully(self, tmp_path: Path) -> None:
        # A single code block that by itself exceeds max_chunk_size (e.g. 1000 chars with max 400)
        code_lines = [f"line_{i:03d} = {i * 10}\n" for i in range(50)]
        code_block = "```python\n" + "".join(code_lines) + "```\n"
        doc = "## Massive Code Block\n\n" + code_block

        chunker = MarkdownChunker(max_chunk_size=400)
        chunks = chunker.chunk_text(doc)

        assert len(chunks) > 1

        # Hard ceiling strictly enforced
        for c in chunks:
            assert len(c.content) <= 400
            assert c.span_length <= 400
            assert doc[c.first_character_index:c.last_character_index] == c.content

        # Contiguous coverage
        assert chunks[0].first_character_index == 0
        assert chunks[-1].last_character_index == len(doc)
        for i in range(len(chunks) - 1):
            assert chunks[i].last_character_index == chunks[i + 1].first_character_index

    def test_exact_utf8_character_offsets_with_multibyte_chars(
        self,
        tmp_path: Path,
    ) -> None:
        doc = (
            "# Guide 🚀 with Accents: café & résumé\n\n"
            "Emojis: 🔥 ✨ 🌟 and symbols: — € ¥\n\n"
            "```python\n"
            "# Accented comment: voilà\n"
            "msg = 'Bonjour le monde ☕'\n"
            "```\n\n"
            "## Section 2: Multimodal 视觉模型\n\n"
            "DeepSeek-V3 and Qwen2.5-VL support.\n"
        )
        test_file = tmp_path / "unicode.md"
        test_file.write_text(doc, encoding="utf-8")

        chunker = MarkdownChunker()
        chunks = chunker.chunk_file(test_file)

        assert len(chunks) == 2

        for c in chunks:
            expected_slice = doc[c.first_character_index:c.last_character_index]
            assert c.content == expected_slice
            assert c.span_length == len(c.content)
            assert c.last_character_index > c.first_character_index

        assert "🚀" in chunks[0].content
        assert "café" in chunks[0].content
        assert "☕" in chunks[0].content
        assert "视觉模型" in chunks[1].content

    def test_headings_with_indentation(self) -> None:
        # CommonMark allows up to 3 spaces for ATX headings
        doc = (
            "   # Indented 3 spaces (valid)\n\n"
            "Content 1.\n\n"
            "  ## Indented 2 spaces (valid)\n\n"
            "Content 2.\n\n"
            "    # Indented 4 spaces is an indented code block, not heading\n\n"
            "Content 3.\n"
        )
        chunker = MarkdownChunker()
        chunks = chunker.chunk_text(doc)

        assert len(chunks) == 2
        assert chunks[0].content.startswith("   # Indented 3 spaces")
        assert chunks[1].content.startswith("  ## Indented 2 spaces")
        assert "    # Indented 4 spaces" in chunks[1].content

    def test_chunk_markdown_convenience_function(self, tmp_path: Path) -> None:
        doc = "# API Reference\n\nDetails here.\n"
        test_file = tmp_path / "api.md"
        test_file.write_text(doc, encoding="utf-8")

        # Via content string
        chunks_content = chunk_markdown("virtual.md", content=doc, max_size=1000)
        assert len(chunks_content) == 1
        assert chunks_content[0].content == doc

        # Via file path
        chunks_file = chunk_markdown(test_file, max_size=1000)
        assert len(chunks_file) == 1
        assert chunks_file[0].content == doc

        # Via chunker __call__
        chunker = MarkdownChunker()
        chunks_call = chunker(test_file)
        assert len(chunks_call) == 1
        assert chunks_call[0].content == doc

    def test_real_vllm_documents_adhere_to_ceiling(self) -> None:
        # Test on real markdown documents from the corpus
        sample_files = [
            Path("data/raw/vllm-0.10.1/docs/usage/reproducibility.md"),
            Path("data/raw/vllm-0.10.1/docs/features/lora.md"),
            Path("data/raw/vllm-0.10.1/docs/configuration/tpu.md"),
            Path("data/raw/vllm-0.10.1/README.md"),
        ]

        chunker = MarkdownChunker(max_chunk_size=2000)

        for sample_path in sample_files:
            if not sample_path.exists():
                continue

            raw_text = sample_path.read_text(encoding="utf-8", errors="replace")
            chunks = chunker.chunk_file(sample_path)

            assert len(chunks) > 0

            # Verify every chunk obeys ceiling and is contiguous
            for c in chunks:
                assert len(c.content) <= 2000
                assert c.span_length <= 2000
                assert c.last_character_index > c.first_character_index
                assert c.chunk_type == "markdown_section"
                assert raw_text[c.first_character_index:c.last_character_index] == c.content

            # Contiguous coverage
            assert chunks[0].first_character_index == 0
            assert chunks[-1].last_character_index == len(raw_text)
            for i in range(len(chunks) - 1):
                assert chunks[i].last_character_index == chunks[i + 1].first_character_index
