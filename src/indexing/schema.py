"""Internal indexing metadata schemas for chunk catalog storage."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from src.models import MinimalSource, normalize_file_path

ChunkType = Literal["python_class", "python_func", "python_module", "markdown_section"]


class InternalChunk(BaseModel):
    """Internal metadata and storage schema for indexed corpus chunks:
        Internal: intended for use inside the indexing system.
        Metadata: information describing the chunk.
        Schema: the structure and validation rules for that information.
        Chunk catalog: a collection or registry of indexed chunks."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    chunk_id: int = Field(
        ...,
        ge=0,
        description="Sequential integer chunk identifier",
    )
    file_path: str = Field(
        ...,
        description="Normalized relative path to source file",
    )
    first_character_index: int = Field(
        ...,
        ge=0,
        description="0-based starting character position within file",
    )
    last_character_index: int = Field(
        ...,
        gt=0,
        description="0-based ending character position within file",
    )
    chunk_type: ChunkType = Field(
        ...,
        description=(
            "Syntactic chunk type (python_class, python_func, python_module, markdown_section)"
        ),
    )
    token_count: int = Field(
        ...,
        ge=0,
        description="Token count for the chunk",
    )
    content: str = Field(
        ...,
        description="Raw text content of the chunk",
    )

    @field_validator("file_path", mode="before")
    @classmethod
    def _validate_file_path(cls, v: Any) -> str:
        return normalize_file_path(v)

    @model_validator(mode="after")
    def _validate_character_spans(self) -> Self:
        if self.last_character_index <= self.first_character_index:
            raise ValueError(
                f"last_character_index ({self.last_character_index}) must be greater than "
                f"first_character_index ({self.first_character_index})"
            )
        return self

    @property
    def span_length(self) -> int:
        """Return length of character span."""
        return self.last_character_index - self.first_character_index

    def to_source(self) -> MinimalSource:
        """Convert internal chunk metadata to an interchange MinimalSource object."""
        return MinimalSource(
            file_path=self.file_path,
            first_character_index=self.first_character_index,
            last_character_index=self.last_character_index,
        )
