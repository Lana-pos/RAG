"""Tests for code-aware lexical tokenization."""

from src.retrieval.tokenizer import tokenize_code


def test_tokenizes_camel_case_and_retains_composite() -> None:
    assert tokenize_code("AsyncEngineArgs") == [
        "async",
        "engine",
        "args",
        "asyncengineargs",
    ]


def test_splits_snake_case_and_dotted_attribute_paths() -> None:
    assert tokenize_code("openai_compatible_server") == [
        "openai",
        "compatible",
        "server",
        "openai_compatible_server",
    ]
    assert tokenize_code("torch.nn.Linear") == [
        "torch",
        "nn",
        "linear",
        "torch.nn.linear",
    ]


def test_preserves_keywords_and_lowercases_tokens() -> None:
    assert tokenize_code("ASYNC await True") == ["async", "await", "true"]


def test_strips_markdown_formatting_and_retains_identifiers() -> None:
    assert tokenize_code(
        "## **AsyncEngineArgs** uses `openai_compatible_server`."
    ) == [
        "async",
        "engine",
        "args",
        "asyncengineargs",
        "uses",
        "openai",
        "compatible",
        "server",
        "openai_compatible_server",
    ]


def test_empty_input_returns_no_tokens() -> None:
    assert tokenize_code("") == []
