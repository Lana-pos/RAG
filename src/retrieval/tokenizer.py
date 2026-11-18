"""Code-aware tokenization for lexical retrieval."""

from __future__ import annotations

import re

_LEXICAL_TOKEN_PATTERN = re.compile(r"\w+(?:[._]\w+)*", re.UNICODE)
_CAMEL_CASE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)


def _split_camel_case(identifier: str) -> list[str]:
    """Split an identifier at lower-to-upper and acronym-to-word boundaries."""
    return _CAMEL_CASE_BOUNDARY_PATTERN.split(identifier)


def tokenize_code(text: str) -> list[str]:
    """Return lowercase lexical tokens, subwords, and composite code symbols.
    Markdown punctuation is ignored while identifier punctuation (underscores
    and dots) is retained in composite tokens. Programming keywords are
    treated like any other identifier and are not filtered out.
    """
    tokens: list[str] = []

    for match in _LEXICAL_TOKEN_PATTERN.finditer(text):
        identifier = match.group()
        if (
            identifier.startswith("_")
            and identifier.endswith("_")
            and not (identifier.startswith("__") and identifier.endswith("__"))
        ):
            identifier = identifier.strip("_")
            if not identifier:
                continue

        components = identifier.split(".")
        for component in components:
            subwords: list[str] = []
            for part in component.split("_"):
                if part:
                    subwords.extend(_split_camel_case(part))

            tokens.extend(word.lower() for word in subwords)

            composite_component = component.lower()
            if len(subwords) > 1:
                tokens.append(composite_component)

        if len(components) > 1:
            tokens.append(identifier.lower())

    return tokens
