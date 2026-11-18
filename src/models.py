"""Data interchange schemas for the RAG codebase question answering system.

Defines immutable Pydantic v2 schemas used across ingestion, indexing,
retrieval, generation, and evaluation harness interfaces.
"""

from __future__ import annotations

import posixpath
import uuid
from pathlib import Path
from typing import Any, List, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self


def normalize_file_path(path: Union[str, Path]) -> str:
    """Normalize file path to a strictly relative string without leading './'.
    Converts backslashes to forward slashes, strips leading redundant './'
    or '/' prefixes, and strips parent project directories if an absolute
    path within the repository or data/raw tree is provided.
        path: Path object or string representing a file location.
    """
    if isinstance(path, Path):
        path = str(path)
    if not isinstance(path, str):
        raise TypeError(f"file_path must be a string or Path, got {type(path).__name__}")

    raw_path = path.strip()
    if not raw_path:
        raise ValueError("file_path cannot be empty")

    clean_path = raw_path.replace("\\", "/")

    # If an absolute path contains 'data/raw/', slice from 'data/raw/'
    raw_marker = "data/raw/"
    if raw_marker in clean_path:
        clean_path = clean_path[clean_path.index(raw_marker):]
    elif clean_path.startswith("/"):
        try:
            rel = Path(clean_path).resolve().relative_to(Path.cwd().resolve())
            clean_path = str(rel).replace("\\", "/")
        except ValueError:
            clean_path = clean_path.lstrip("/")

    # Remove any leading './' or '/'
    while clean_path.startswith("./"):
        clean_path = clean_path[2:]
    clean_path = clean_path.lstrip("/")

    # Resolve redundant internal elements like '//' or 'a/b/../c'
    norm = posixpath.normpath(clean_path)
    while norm.startswith("./"):
        norm = norm[2:]
    norm = norm.lstrip("/")

    if not norm or norm == ".":
        raise ValueError("file_path cannot be empty or resolve to current directory")

    return norm


class MinimalSource(BaseModel):
    """The class stores the minimum information needed to identify a source location
    in the indexed documents or source code.
    Represents a discrete source location in the indexed corpus:
        Source location: where a piece of text came from.
        Indexed: processed and stored so that the system can search it.
        Corpus: the collection of documents or source files your system works with.
        Discrete: one specific, identifiable piece or location."""

    model_config = ConfigDict(frozen=True, extra="ignore")
    """frozen=True means you cannot normally reassign model fields after creation.
    extra="ignore" means unexpected input fields are ignored.
    """

    file_path: str = Field(
        ...,
        description=(
            "Path relative to project root (e.g. data/raw/vllm-0.10.1/docs/features/lora.md)"
        ),
    )
    first_character_index: int = Field(
        ...,
        ge=0,
        description="0-based starting character position within the file",
    )
    last_character_index: int = Field(
        ...,
        gt=0,
        description="0-based ending character position within the file",
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


class UnansweredQuestion(BaseModel):
    """Represents a question awaiting retrieval and answer generation:
        Retrieval means searching for relevant information, such as passages in documents.
        Answer generation means using that information to produce an answer."""

    model_config = ConfigDict(frozen=True, extra="ignore")
    # prevents normal reassignment of fields after the object has been created.
    # ignores unexpected fields that aren't defined in the model.
    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question",
    )
    question: str = Field(
        ...,
        min_length=1,
        description="Question text",
    )

    @field_validator("question")
    @classmethod
    def _validate_question(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("question cannot be empty or contain only whitespace/tabs")
        return cleaned


class AnsweredQuestion(UnansweredQuestion):
    """Represents a ground-truth answered question with reference sources."""

    sources: List[MinimalSource] = Field(
        default_factory=list,
        description="Reference source locations containing ground truth answers",
    )
    answer: str = Field(
        ...,
        description="Ground truth reference answer text",
    )


class RagDataset(BaseModel):
    """Container for question datasets loaded from disk."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]] = Field(
        default_factory=list,
        description="Collection of answered or unanswered RAG questions",
    )


class MinimalSearchResults(BaseModel):
    """Result of search retrieval for a single question."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question",
    )
    question: str = Field(
        ...,
        min_length=1,
        description="Question text",
    )
    retrieved_sources: List[MinimalSource] = Field(
        default_factory=list,
        description="List of candidate sources retrieved by the search engine",
    )

    @field_validator("question")
    @classmethod
    def _validate_question(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("question cannot be empty or contain only whitespace/tabs")
        return cleaned


class MinimalAnswer(MinimalSearchResults):
    """Result of answer generation for a single question."""

    answer: str = Field(
        ...,
        description="Generated natural language answer",
    )


class StudentSearchResults(BaseModel):
    """Official batch search output evaluated by the moulinette."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    search_results: List[MinimalSearchResults] = Field(
        default_factory=list,
        description="Batch retrieval results",
    )
    k: int = Field(
        ...,
        ge=1,
        description="Number of candidate sources retrieved per question",
    )


class StudentSearchResultsAndAnswer(BaseModel):
    """Official batch answer output evaluated by the moulinette."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    search_results: List[MinimalAnswer] = Field(
        default_factory=list,
        description="Batch retrieval and answer generation results",
    )
    k: int = Field(
        ...,
        ge=1,
        description="Number of candidate sources retrieved per question",
    )
