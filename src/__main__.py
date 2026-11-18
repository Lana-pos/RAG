"""CLI entrypoint module for RAG Against the Machine (``uv run python -m src``)."""

from __future__ import annotations

import sys

from src.cli import main as main

if __name__ == "__main__":
    sys.exit(main())
