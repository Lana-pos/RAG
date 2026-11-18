"""Unit and integration tests for Context Grounding & Prompt Formatting"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from src.generation.prompt import (
    ContextLoader,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_TOP_K,
    GroundedSource,
    MAX_TOKEN_BUDGET,
    MIN_TOKEN_BUDGET,
    NO_CONTEXT_PLACEHOLDER,
    PromptBuilder,
    build_messages,
    build_prompt,
    estimate_tokens,
    format_chatml,
    format_context,
    format_source_block,
    format_source_header,
    load_source_span,
    resolve_safe_path,
    truncate_to_token_budget,
)
from src.indexing.schema import InternalChunk
from src.models import MinimalSearchResults, MinimalSource


@pytest.fixture
def sample_chunks() -> List[InternalChunk]:
    """Create a realistic list of InternalChunk instances."""
    return [
        InternalChunk(
            chunk_id=0,
            file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
            first_character_index=4695,
            last_character_index=4825,
            chunk_type="markdown_section",
            token_count=28,
            content=(
                "## Dynamically Serving LoRA Adapters\n"
                "You can dynamically load a LoRA adapter by sending a POST request "
                "to the `/v1/load_lora_adapter` endpoint."
            ),
        ),
        InternalChunk(
            chunk_id=1,
            file_path="data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
            first_character_index=267,
            last_character_index=410,
            chunk_type="python_func",
            token_count=32,
            content=(
                "@router.post('/v1/load_lora_adapter')\n"
                "async def load_lora_adapter(request: LoadLoRAAdapterRequest):\n"
                "    return await engine.add_lora(request)"
            ),
        ),
        InternalChunk(
            chunk_id=2,
            file_path="data/raw/vllm-0.10.1/vllm/lora/request.py",
            first_character_index=100,
            last_character_index=220,
            chunk_type="python_class",
            token_count=24,
            content=(
                "class LoRARequest:\n"
                "    lora_name: str\n"
                "    lora_int_id: int\n"
                "    lora_path: str"
            ),
        ),
    ]


@pytest.fixture
def sample_disk_corpus(tmp_path: Path) -> Path:
    """Create temporary source files on disk to test span hydration."""
    lora_doc = tmp_path / "data" / "raw" / "vllm-0.10.1" / "docs" / "features" / "lora.md"
    lora_doc.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "# LoRA Adapters in vLLM\n\n"
        "vLLM supports serving Low-Rank Adaptation (LoRA) adapters efficiently.\n\n"
        "## Dynamically Serving LoRA Adapters\n"
        "Use `/v1/load_lora_adapter` HTTP POST endpoint to register an adapter at runtime.\n"
    )
    lora_doc.write_text(text, encoding="utf-8")
    return tmp_path


class TestConstantsAndAliases:
    """Verify Qwen3 token budget constants and architectural aliases."""

    def test_qwen3_token_budget_constants(self) -> None:
        assert DEFAULT_MODEL_NAME == "Qwen/Qwen3-0.6B"
        assert MIN_TOKEN_BUDGET == 2000
        assert MAX_TOKEN_BUDGET == 4000
        assert MIN_TOKEN_BUDGET <= DEFAULT_MAX_PROMPT_TOKENS <= MAX_TOKEN_BUDGET
        assert DEFAULT_TOKEN_BUDGET == DEFAULT_MAX_PROMPT_TOKENS
        assert DEFAULT_MAX_CHUNK_CHARS == 2000
        assert DEFAULT_TOP_K == 5


class TestSourceHeadersAndTopKInjection:
    """Verify top-k sources are injected with explicit file path and character span headers."""

    def test_format_source_header_includes_file_path_and_character_spans(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        chunk = sample_chunks[0]
        header = format_source_header(chunk, index=1)

        assert "[Source 1]" in header
        assert "data/raw/vllm-0.10.1/docs/features/lora.md" in header
        assert "4695" in header
        assert "4825" in header
        assert "4695-4825" in header
        assert "4695:4825" in header

    def test_format_source_block_with_minimal_source_and_content(self) -> None:
        source = MinimalSource(
            file_path="data/raw/vllm-0.10.1/vllm/entrypoints/llm.py",
            first_character_index=64000,
            last_character_index=64120,
        )
        snippet = "def __init__(self, model: str, trust_remote_code: bool = False):"
        block = format_source_block(source, content=snippet, index=2)

        assert "[Source 2]" in block
        assert "data/raw/vllm-0.10.1/vllm/entrypoints/llm.py" in block
        assert "64000-64120" in block
        assert snippet in block

    def test_injects_top_k_sources_with_explicit_headers(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        question = "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?"
        builder = PromptBuilder(top_k=2)
        prompt = builder.build_prompt(question, sample_chunks)

        # Top-2 sources must be present with full headers and content
        assert "[Source 1]" in prompt
        assert "data/raw/vllm-0.10.1/docs/features/lora.md" in prompt
        assert "4695-4825" in prompt
        assert "/v1/load_lora_adapter" in prompt

        assert "[Source 2]" in prompt
        assert "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py" in prompt
        assert "267-410" in prompt
        assert "async def load_lora_adapter" in prompt

        # 3rd source must be excluded because top_k=2
        assert "[Source 3]" not in prompt
        assert "data/raw/vllm-0.10.1/vllm/lora/request.py" not in prompt
        assert question in prompt

        # Verify standalone format_context helper as well
        ctx = format_context(sample_chunks, k=2)
        assert "[Source 1]" in ctx
        assert "[Source 2]" in ctx
        assert "[Source 3]" not in ctx

    def test_supports_scored_tuples_dicts_and_string_snippets(self) -> None:
        dict_source = {
            "file_path": "data/raw/vllm-0.10.1/docs/usage.md",
            "first_character_index": 10,
            "last_character_index": 50,
            "content": "Use vllm serve to launch the server.",
        }
        minimal = MinimalSource(
            file_path="data/raw/vllm-0.10.1/vllm/config.py",
            first_character_index=100,
            last_character_index=200,
        )
        scored_tuple = (minimal, "class ModelConfig:\n    pass")
        raw_string = "Standalone context snippet."

        prompt = build_prompt(
            "How to configure and launch vLLM?",
            sources=[dict_source, scored_tuple, raw_string],
            k=3,
        )

        assert "[Source 1]" in prompt
        assert "data/raw/vllm-0.10.1/docs/usage.md" in prompt
        assert "10-50" in prompt
        assert "Use vllm serve to launch the server." in prompt

        assert "[Source 2]" in prompt
        assert "data/raw/vllm-0.10.1/vllm/config.py" in prompt
        assert "100-200" in prompt
        assert "class ModelConfig:" in prompt

        assert "[Source 3]" in prompt
        assert "Standalone context snippet." in prompt


class TestTokenBudgetConstraint:
    """Verify prompt token length is strictly constrained within Qwen3's 2k-4k budget."""

    def test_constrains_oversized_corpus_chunks_within_default_2k_4k_budget(self) -> None:
        # Create 15 chunks each 2000 characters long (~7500+ total tokens)
        heavy_chunks: List[InternalChunk] = []
        for i in range(15):
            body = (
                f"def function_{i}(arg_a: int, arg_b: str) -> bool:\n"
                + ("    # Detailed implementation comment line for testing token budget\n" * 28)
            )[:2000]
            heavy_chunks.append(
                InternalChunk(
                    chunk_id=i,
                    file_path=f"data/raw/vllm-0.10.1/vllm/module_{i}.py",
                    first_character_index=i * 2000,
                    last_character_index=(i + 1) * 2000,
                    chunk_type="python_func",
                    token_count=estimate_tokens(body),
                    content=body,
                )
            )

        question = "How do the module functions handle arguments in vLLM?"
        prompt = build_prompt(question, heavy_chunks, k=10)
        prompt_tokens = estimate_tokens(prompt)

        # Must be strictly within default budget (3000) and <= 4000 (Qwen3 2k-4k SLA)
        assert prompt_tokens <= DEFAULT_MAX_PROMPT_TOKENS
        assert prompt_tokens <= MAX_TOKEN_BUDGET

        # All top-10 headers must still be present via Strategy A allocation
        for idx in range(1, 11):
            assert f"[Source {idx}]" in prompt
            assert f"data/raw/vllm-0.10.1/vllm/module_{idx - 1}.py" in prompt

        # Chunks 11..15 (beyond k=10) must not be in prompt
        assert "[Source 11]" not in prompt

    @pytest.mark.parametrize("budget", [2000, 2048, 3000, 4000])
    def test_enforces_explicit_2k_to_4k_token_budgets(self, budget: int) -> None:
        chunks = [
            InternalChunk(
                chunk_id=i,
                file_path=f"data/raw/vllm-0.10.1/docs/doc_{i}.md",
                first_character_index=0,
                last_character_index=1950,
                chunk_type="markdown_section",
                token_count=500,
                content=(f"Section {i}: " + ("vLLM configuration parameter details. " * 55))[
                    :1950
                ],
            )
            for i in range(12)
        ]

        builder = PromptBuilder(max_tokens=budget, top_k=10)
        prompt = builder.build_prompt("Explain all configuration parameters.", chunks)
        total_tokens = builder.estimate_tokens(prompt)

        assert total_tokens <= budget

    def test_enforces_strict_small_token_budget_strategy_b(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        # With an ultra-small budget (e.g. 160 tokens), Strategy B packs only what fits
        builder = PromptBuilder(max_tokens=160, top_k=3)
        prompt = builder.build_prompt("What is the LoRA endpoint?", sample_chunks)
        assert estimate_tokens(prompt) <= 160
        assert "[Source 1]" in prompt

    def test_handles_unspaced_long_character_sequences(self) -> None:
        # 20,000 continuous 'x' characters without spaces (~5,000 char-based tokens)
        blob = "x" * 20000
        truncated = truncate_to_token_budget(blob, max_tokens=250)
        assert estimate_tokens(truncated) <= 250
        assert len(truncated) <= 250 * 4

    def test_handles_abnormally_long_question_within_budget(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        huge_question = "How does vLLM schedule batches? " * 800
        prompt = build_prompt(huge_question, sample_chunks, max_tokens=2000)
        assert estimate_tokens(prompt) <= 2000
        assert "[Source 1]" in prompt

    def test_custom_hf_tokenizer_integration(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text, **kw: text.split()
        mock_tokenizer.decode.side_effect = lambda ids, **kw: " ".join(ids)

        builder = PromptBuilder(max_tokens=120, tokenizer=mock_tokenizer)
        prompt = builder.build_prompt("How to load LoRA?", sample_chunks)
        assert builder.estimate_tokens(prompt) <= 120
        assert mock_tokenizer.encode.called

        # Also test callable tokenizer returning dict with input_ids
        def dict_tokenizer(text: str) -> Dict[str, List[int]]:
            return {"input_ids": [1] * len(text.split())}

        assert estimate_tokens("one two three four", tokenizer=dict_tokenizer) == 4


class TestContextLoaderAndSecurity:
    """Verify disk span loading, catalog lookup, and os.path.commonpath security."""

    def test_hydrates_exact_character_span_from_disk(
        self, sample_disk_corpus: Path
    ) -> None:
        lora_file = (
            sample_disk_corpus
            / "data"
            / "raw"
            / "vllm-0.10.1"
            / "docs"
            / "features"
            / "lora.md"
        )
        full_text = lora_file.read_text(encoding="utf-8")
        target_phrase = "Use `/v1/load_lora_adapter` HTTP POST endpoint"
        start_idx = full_text.index(target_phrase)
        end_idx = start_idx + len(target_phrase)

        source = MinimalSource(
            file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
            first_character_index=start_idx,
            last_character_index=end_idx,
        )

        loader = ContextLoader(base_dir=sample_disk_corpus)
        extracted = loader.load_span(source, strict=True)
        assert extracted == target_phrase

        # Second call hits in-memory file cache
        extracted_cached = loader.load_span(source, strict=True)
        assert extracted_cached == target_phrase

        # Build full prompt from MinimalSource hydrated from disk
        builder = PromptBuilder(base_dir=sample_disk_corpus)
        prompt = builder.build_prompt("Which endpoint loads a LoRA adapter?", [source])
        assert target_phrase in prompt
        assert f"{start_idx}-{end_idx}" in prompt

    def test_hydrates_from_in_memory_chunk_catalog_without_disk(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        # Create MinimalSource matching sample_chunks[1] and a sub-span of sample_chunks[0]
        exact_source = sample_chunks[1].to_source()
        sub_source = MinimalSource(
            file_path=sample_chunks[0].file_path,
            first_character_index=sample_chunks[0].first_character_index,
            last_character_index=sample_chunks[0].first_character_index + 36,
        )

        loader = ContextLoader(chunks=sample_chunks)
        assert loader.load_span(exact_source) == sample_chunks[1].content
        assert loader.load_span(sub_source) == "## Dynamically Serving LoRA Adapters"

    def test_path_traversal_is_blocked_by_commonpath(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Path traversal"):
            resolve_safe_path("../../etc/passwd", base_dir=tmp_path)

        with pytest.raises(ValueError, match="Path traversal"):
            resolve_safe_path("/etc/passwd", base_dir=tmp_path, allowed_root=tmp_path)

        with pytest.raises(ValueError, match="null byte"):
            resolve_safe_path("data/raw/file\x00.py", base_dir=tmp_path)

        with pytest.raises(ValueError, match="cannot be empty"):
            resolve_safe_path("   ", base_dir=tmp_path)


class TestGroundingAndFormats:
    """Verify grounding rules, ChatML formatting, message building, and edge cases."""

    def test_grounding_instructions_and_empty_sources_placeholder(self) -> None:
        prompt = build_prompt("What is vLLM?", sources=[])
        assert DEFAULT_SYSTEM_PROMPT in prompt
        assert "Do NOT hallucinate" in prompt
        assert NO_CONTEXT_PLACEHOLDER in prompt
        assert "Retrieved Context (Top-0 Sources):" in prompt

    def test_accepts_minimal_search_results_directly(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        search_res = MinimalSearchResults(
            question_id="q-101",
            question="How is LoRA loaded dynamically?",
            retrieved_sources=[c.to_source() for c in sample_chunks[:2]],
        )
        builder = PromptBuilder(chunks=sample_chunks)
        prompt = builder.build_prompt(search_res)

        assert "How is LoRA loaded dynamically?" in prompt
        assert "[Source 1]" in prompt
        assert "[Source 2]" in prompt
        assert "/v1/load_lora_adapter" in prompt

    def test_accepts_reversed_positional_arguments(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        prompt = build_prompt(sample_chunks[:1], "What endpoint loads LoRA?")
        assert "What endpoint loads LoRA?" in prompt
        assert "[Source 1]" in prompt
        assert "/v1/load_lora_adapter" in prompt

    def test_build_messages_and_chatml_formatting(
        self, sample_chunks: List[InternalChunk]
    ) -> None:
        messages = build_messages(
            "What HTTP endpoint loads LoRA?",
            sample_chunks[:2],
            max_tokens=2500,
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "[Source 1]" in messages[1]["content"]
        assert "[Source 2]" in messages[1]["content"]

        chatml = format_chatml(
            "What HTTP endpoint loads LoRA?",
            sample_chunks[:2],
            max_tokens=2500,
            enable_thinking=False,
        )
        assert "<|im_start|>system" in chatml
        assert "<|im_start|>user" in chatml
        assert "<|im_start|>assistant" in chatml
        assert "<think>\n\n</think>" in chatml
        assert estimate_tokens(chatml) <= 2500

    def test_grounded_source_methods_and_serialization(self) -> None:
        gs = GroundedSource(
            index=1,
            file_path="data/raw/vllm-0.10.1/README.md",
            first_character_index=0,
            last_character_index=50,
            content="# vLLM",
            token_count=5,
            truncated=False,
        )
        assert gs.span_length == 50
        assert gs.format_header() == gs.header
        assert gs.format_block() == gs.formatted
        assert gs.to_minimal_source() == MinimalSource(
            file_path="data/raw/vllm-0.10.1/README.md",
            first_character_index=0,
            last_character_index=50,
        )
        data = gs.to_dict()
        assert data["index"] == 1
        assert data["file_path"] == "data/raw/vllm-0.10.1/README.md"

    def test_invalid_configuration_parameters_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            PromptBuilder(max_tokens=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            PromptBuilder(top_k=0)

        with pytest.raises(ValueError, match="max_chunk_chars must be positive"):
            PromptBuilder(max_chunk_chars=0)

        with pytest.raises(ValueError, match="max_chunk_size must be positive"):
            ContextLoader(max_chunk_size=0)

        builder = PromptBuilder()
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            builder.build_prompt("Test?", [], max_tokens=-5)

    def test_real_vllm_corpus_span_loading_if_present(self) -> None:
        lora_path = Path("data/raw/vllm-0.10.1/docs/features/lora.md")
        if not lora_path.exists():
            pytest.skip("vllm-0.10.1 corpus not present")

        source = MinimalSource(
            file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
            first_character_index=4695,
            last_character_index=6098,
        )
        span_text = load_source_span(source)
        assert len(span_text) == 6098 - 4695
        assert "load_lora_adapter" in span_text

        prompt = build_prompt(
            "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?",
            [source],
        )
        assert "data/raw/vllm-0.10.1/docs/features/lora.md" in prompt
        assert "4695-6098" in prompt
        assert estimate_tokens(prompt) <= DEFAULT_MAX_PROMPT_TOKENS
