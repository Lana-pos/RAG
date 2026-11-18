"""Tests for InternalChunk metadata schema in src/indexing/schema.py."""

import pytest
from pydantic import ValidationError

from src.indexing.schema import InternalChunk
from src.models import MinimalSource


class TestInternalChunk:
    """Test InternalChunk validation, normalization, and conversion."""

    def test_valid_internal_chunk(self) -> None:
        chunk = InternalChunk(
            chunk_id=1,
            file_path="./data/raw/vllm-0.10.1/vllm/model.py",
            first_character_index=100,
            last_character_index=500,
            chunk_type="python_class",
            token_count=80,
            content="class TestModel:\n    pass\n",
        )
        assert chunk.chunk_id == 1
        assert chunk.file_path == "data/raw/vllm-0.10.1/vllm/model.py"
        assert chunk.first_character_index == 100
        assert chunk.last_character_index == 500
        assert chunk.span_length == 400
        assert chunk.chunk_type == "python_class"
        assert chunk.token_count == 80
        assert "class TestModel" in chunk.content

    def test_invalid_span_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            InternalChunk(
                chunk_id=2,
                file_path="data/raw/vllm-0.10.1/docs/index.md",
                first_character_index=500,
                last_character_index=100,
                chunk_type="markdown_section",
                token_count=40,
                content="# Intro",
            )

    def test_negative_span_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            InternalChunk(
                chunk_id=3,
                file_path="data/raw/vllm-0.10.1/docs/index.md",
                first_character_index=-5,
                last_character_index=100,
                chunk_type="markdown_section",
                token_count=40,
                content="# Intro",
            )

    def test_negative_chunk_id_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            InternalChunk(
                chunk_id=-1,
                file_path="data/raw/vllm-0.10.1/docs/index.md",
                first_character_index=0,
                last_character_index=100,
                chunk_type="markdown_section",
                token_count=40,
                content="# Intro",
            )

    def test_invalid_chunk_type_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            InternalChunk(
                chunk_id=1,
                file_path="data/raw/vllm-0.10.1/docs/index.md",
                first_character_index=0,
                last_character_index=100,
                chunk_type="invalid_type",  # type: ignore[arg-type]
                token_count=40,
                content="# Intro",
            )

    def test_to_source_conversion(self) -> None:
        chunk = InternalChunk(
            chunk_id=42,
            file_path="./data/raw/vllm-0.10.1/vllm/entrypoints/llm.py",
            first_character_index=120,
            last_character_index=850,
            chunk_type="python_func",
            token_count=150,
            content="def generate():\n    pass\n",
        )
        source = chunk.to_source()
        assert isinstance(source, MinimalSource)
        assert source.file_path == "data/raw/vllm-0.10.1/vllm/entrypoints/llm.py"
        assert source.first_character_index == 120
        assert source.last_character_index == 850
        assert source.span_length == 730

    def test_immutability(self) -> None:
        chunk = InternalChunk(
            chunk_id=1,
            file_path="data/raw/vllm-0.10.1/vllm/model.py",
            first_character_index=10,
            last_character_index=100,
            chunk_type="python_module",
            token_count=20,
            content="import sys",
        )
        with pytest.raises(ValidationError):
            setattr(chunk, "content", "import os")
