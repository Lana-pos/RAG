"""Unit and integration tests for Local CPU Inference Engine"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.generation.llm import (
    DEFAULT_DEVICE,
    DEFAULT_DO_SAMPLE,
    DEFAULT_FALLBACK_ANSWER,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    DO_SAMPLE,
    EMPTY_QUERY_ANSWER,
    LLMInferenceEngine,
    MAX_NEW_TOKENS,
    MAX_RAM_BYTES,
    MAX_RAM_GB,
    MAX_RAM_MB,
    MODEL_NAME,
    TEMPERATURE,
    answer_dataset,
    clean_generated_answer,
    estimate_model_memory_bytes,
    estimate_model_memory_gb,
    estimate_model_memory_mb,
    generate_answer,
    generate_answer_json,
    get_process_memory_bytes,
    get_process_memory_gb,
    get_process_memory_mb,
    normalize_retrieved_sources,
    parse_answer_json,
    resolve_torch_dtype,
    to_minimal_source,
)
from src.indexing.schema import InternalChunk
from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


class DummyTokenizer:
    """Lightweight deterministic tokenizer for offline CPU inference testing."""

    def __init__(self) -> None:
        self.pad_token: str = "<|endoftext|>"
        self.eos_token: str = "<|im_end|>"
        self.pad_token_id: int = 0
        self.eos_token_id: int = 2
        self._id_to_text: Dict[int, str] = {
            101: "To ",
            102: "configure ",
            103: "the ",
            104: "OpenAI ",
            105: "server, ",
            106: "use ",
            107: "`vllm serve` ",
            108: "with ",
            109: "`--port 8000`.",
        }
        self.last_prompt: str = ""

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 3000,
    ) -> Dict[str, torch.Tensor]:
        self.last_prompt = text
        word_count = max(1, min(len(text.split()), max_length))
        input_ids = torch.arange(1, word_count + 1, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        word_count = max(1, len(text.split()))
        return list(range(1, word_count + 1))

    def decode(
        self,
        token_ids: Union[torch.Tensor, Sequence[int]],
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            ids_list = token_ids.detach().cpu().view(-1).tolist()
        else:
            ids_list = list(token_ids)

        pieces: List[str] = []
        for tid in ids_list:
            if int(tid) in self._id_to_text:
                pieces.append(self._id_to_text[int(tid)])
        if pieces:
            return "".join(pieces)
        return "You can dynamically load a LoRA adapter via `/v1/load_lora_adapter`."


class DummyCausalLM(nn.Module):
    """Lightweight PyTorch causal LM verifying torch.inference_mode() and greedy kwargs."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 16, dtype=dtype)
        self.inference_mode_recorded: List[bool] = []
        self.generate_calls: List[Dict[str, Any]] = []
        self.eval_called: bool = False
        self.generated_suffix: List[int] = [
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
        ]

    def eval(self) -> DummyCausalLM:
        self.eval_called = True
        return super().eval()

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.inference_mode_recorded.append(bool(torch.is_inference_mode_enabled()))
        self.generate_calls.append(dict(kwargs))

        input_ids: torch.Tensor = kwargs["input_ids"]
        max_new: int = int(kwargs.get("max_new_tokens", 128))
        suffix_ids = self.generated_suffix[:max_new]
        suffix_tensor = torch.tensor(
            [suffix_ids],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, suffix_tensor], dim=-1)


@pytest.fixture
def dummy_tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


@pytest.fixture
def dummy_model() -> DummyCausalLM:
    return DummyCausalLM(dtype=torch.bfloat16)


@pytest.fixture
def sample_chunks() -> List[InternalChunk]:
    return [
        InternalChunk(
            chunk_id=0,
            file_path="data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
            first_character_index=100,
            last_character_index=280,
            chunk_type="markdown_section",
            token_count=32,
            content=(
                "# OpenAI Compatible Server\n"
                "To configure the OpenAI server, run `vllm serve Qwen/Qwen3-0.6B --port 8000`."
            ),
        ),
        InternalChunk(
            chunk_id=1,
            file_path="data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
            first_character_index=267,
            last_character_index=410,
            chunk_type="python_func",
            token_count=28,
            content=(
                "async def run_server(args: Namespace) -> None:\n"
                "    app = build_async_engine_client(args)"
            ),
        ),
    ]


class TestConstantsAndAliases:
    """Verify Qwen3-0.6B constants, decoding defaults, and class aliases."""

    def test_qwen3_constants_match_acceptance_criteria(self) -> None:
        assert DEFAULT_MODEL_NAME == "Qwen/Qwen3-0.6B"
        assert MODEL_NAME == "Qwen/Qwen3-0.6B"
        assert DEFAULT_MAX_NEW_TOKENS == 128
        assert MAX_NEW_TOKENS == 128
        assert DEFAULT_TEMPERATURE == 0.0
        assert TEMPERATURE == 0.0
        assert DEFAULT_DO_SAMPLE is False
        assert DO_SAMPLE is False
        assert DEFAULT_DEVICE == "cpu"
        assert MAX_RAM_GB == 1.5
        assert MAX_RAM_MB == 1536.0
        assert MAX_RAM_BYTES == int(1.5 * 1024 * 1024 * 1024)


class TestInferenceModeAndGreedyDecoding:
    """Verify inference runs inside torch.inference_mode() with greedy decoding."""

    def test_executes_with_torch_inference_mode_and_greedy_parameters(
        self,
        dummy_model: DummyCausalLM,
        dummy_tokenizer: DummyTokenizer,
    ) -> None:
        engine = LLMInferenceEngine(
            _model=dummy_model,
            _tokenizer=dummy_tokenizer,
        )
        assert engine.model_name == "Qwen/Qwen3-0.6B"
        assert engine.temperature == 0.0
        assert engine.max_new_tokens == 128
        assert engine.do_sample is False

        answer = engine.generate("How to configure the OpenAI server in vLLM?")

        # 1. Must execute inside torch.inference_mode()
        assert len(dummy_model.inference_mode_recorded) == 1
        assert dummy_model.inference_mode_recorded[0] is True
        assert engine._last_inference_mode_enabled is True

        # 2. Must pass greedy decoding parameters (temperature=0.0, max_new_tokens=128)
        assert len(dummy_model.generate_calls) == 1
        call_kwargs = dummy_model.generate_calls[0]
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_new_tokens"] == 128
        assert call_kwargs["do_sample"] is False

        # 3. Decoded answer matches expected natural language output without prompt prefix
        assert answer == "To configure the OpenAI server, use `vllm serve` with `--port 8000`."

    def test_retries_cleanly_if_transformers_rejects_zero_temperature_with_do_sample_false(
        self,
        dummy_tokenizer: DummyTokenizer,
    ) -> None:
        mock_model = MagicMock()
        call_count = 0

        def _side_effect(**kwargs: Any) -> torch.Tensor:
            nonlocal call_count
            call_count += 1
            assert torch.is_inference_mode_enabled() is True
            if call_count == 1 and kwargs.get("temperature") == 0.0:
                raise ValueError(
                    "temperature (=0.0) has to be a strictly positive float"
                )
            return torch.tensor([[101, 102, 103, 104, 105, 106, 107, 108, 109]])

        mock_model.generate.side_effect = _side_effect
        engine = LLMInferenceEngine(_model=mock_model, _tokenizer=dummy_tokenizer)

        result = engine.generate("How to configure OpenAI server?")
        assert call_count == 2
        assert "vllm serve" in result

    def test_custom_max_new_tokens_and_temperature_overrides(
        self,
        dummy_model: DummyCausalLM,
        dummy_tokenizer: DummyTokenizer,
    ) -> None:
        engine = LLMInferenceEngine(
            _model=dummy_model,
            _tokenizer=dummy_tokenizer,
        )
        ans = engine.generate("Test prompt", max_new_tokens=5, temperature=0.0)
        assert dummy_model.generate_calls[-1]["max_new_tokens"] == 5
        assert dummy_model.generate_calls[-1]["temperature"] == 0.0
        assert dummy_model.generate_calls[-1]["do_sample"] is False
        assert ans == "To configure the OpenAI server,"


class TestMemoryFootprintUnder15GB:
    """Verify total RAM footprint stays strictly under 1.5 GB."""

    def test_process_and_engine_memory_stay_under_1_5_gb(
        self,
        dummy_model: DummyCausalLM,
        dummy_tokenizer: DummyTokenizer,
    ) -> None:
        engine = LLMInferenceEngine(
            _model=dummy_model,
            _tokenizer=dummy_tokenizer,
        )
        _ = engine.generate("How to configure OpenAI server?")

        rss_bytes = get_process_memory_bytes()
        rss_mb = get_process_memory_mb()
        rss_gb = get_process_memory_gb()

        assert rss_bytes > 0
        assert rss_mb > 0.0
        assert rss_gb < MAX_RAM_GB
        assert rss_bytes < MAX_RAM_BYTES

        assert engine.memory_footprint_gb < 1.5
        assert engine.get_memory_usage_gb() < 1.5
        assert engine.get_memory_usage_mb() < 1536.0
        assert engine.get_memory_usage_bytes() < MAX_RAM_BYTES
        assert engine.check_memory_budget() is True

    def test_bfloat16_loading_and_low_cpu_mem_usage_configuration(
        self,
        dummy_model: DummyCausalLM,
        dummy_tokenizer: DummyTokenizer,
    ) -> None:
        with patch("src.generation.llm.AutoTokenizer") as mock_auto_tok, patch(
            "src.generation.llm.AutoModelForCausalLM"
        ) as mock_auto_model:
            mock_auto_tok.from_pretrained.return_value = dummy_tokenizer
            mock_auto_model.from_pretrained.return_value = dummy_model

            engine = LLMInferenceEngine(lazy_load=False)
            assert engine.is_loaded is True
            assert engine.torch_dtype == torch.bfloat16

            mock_auto_tok.from_pretrained.assert_called_once_with("Qwen/Qwen3-0.6B")
            mock_auto_model.from_pretrained.assert_called_once_with(
                "Qwen/Qwen3-0.6B",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            assert dummy_model.eval_called is True
            for param in dummy_model.parameters():
                assert param.requires_grad is False

            # Unload frees references
            engine.unload_model()
            assert engine.is_loaded is False

    def test_downcasts_oversized_model_to_bfloat16_when_exceeding_budget(
        self,
        dummy_tokenizer: DummyTokenizer,
    ) -> None:
        model_fp32 = DummyCausalLM(dtype=torch.float32)
        with patch("src.generation.llm.AutoTokenizer") as mock_auto_tok, patch(
            "src.generation.llm.AutoModelForCausalLM"
        ) as mock_auto_model, patch(
            "src.generation.llm.estimate_model_memory_gb",
            return_value=2.4,  # Simulate 2.4 GB float32 footprint > 1.5 GB
        ):
            mock_auto_tok.from_pretrained.return_value = dummy_tokenizer
            mock_auto_model.from_pretrained.return_value = model_fp32

            engine = LLMInferenceEngine(lazy_load=False)
            for param in engine.model.parameters():
                assert param.dtype == torch.bfloat16

    def test_estimate_model_memory_utilities(self, dummy_model: DummyCausalLM) -> None:
        assert estimate_model_memory_bytes(None) == 0
        assert estimate_model_memory_bytes(dummy_model) > 0
        assert estimate_model_memory_mb(dummy_model) > 0.0
        assert 0.0 < estimate_model_memory_gb(dummy_model) < 1.5

        assert resolve_torch_dtype(None) == torch.bfloat16
        assert resolve_torch_dtype("bfloat16") == torch.bfloat16
        assert resolve_torch_dtype("bf16") == torch.bfloat16
        assert resolve_torch_dtype("float16") == torch.float16
        assert resolve_torch_dtype("float32") == torch.float32
        assert resolve_torch_dtype(torch.bfloat16) == torch.bfloat16

        with pytest.raises(ValueError, match="Unsupported torch_dtype"):
            resolve_torch_dtype("int4_invalid")


class TestCleanNaturalLanguageJSONOutput:
    """Verify generated natural language answers are cleanly formatted in JSON."""

    def test_generate_answer_returns_minimal_answer_and_clean_json(
        self,
        dummy_model: DummyCausalLM,
        dummy_tokenizer: DummyTokenizer,
        sample_chunks: List[InternalChunk],
    ) -> None:
        engine = LLMInferenceEngine(
            _model=dummy_model,
            _tokenizer=dummy_tokenizer,
            chunks=sample_chunks,
        )

        # 1. Structured MinimalAnswer model
        ans = engine.generate_answer(
            question="How to configure the OpenAI server?",
            retrieved_sources=sample_chunks,
            question_id="q-openai-1",
        )
        assert isinstance(ans, MinimalAnswer)
        assert ans.question_id == "q-openai-1"
        assert ans.question == "How to configure the OpenAI server?"
        assert len(ans.retrieved_sources) == 2
        assert (
            ans.retrieved_sources[0].file_path
            == "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md"
        )
        assert (
            ans.answer
            == "To configure the OpenAI server, use `vllm serve` with `--port 8000`."
        )

        # 2. JSON string output via generate_answer_json / return_json=True
        json_str = engine.generate_answer_json(
            question="How to configure the OpenAI server?",
            retrieved_sources=sample_chunks,
            question_id="q-openai-1",
        )
        parsed = json.loads(json_str)
        assert parsed["question_id"] == "q-openai-1"
        assert parsed["question"] == "How to configure the OpenAI server?"
        assert len(parsed["retrieved_sources"]) == 2
        assert (
            parsed["answer"]
            == "To configure the OpenAI server, use `vllm serve` with `--port 8000`."
        )

        # Validate round-trip parsing back to MinimalAnswer
        roundtrip = parse_answer_json(json_str)
        assert isinstance(roundtrip, MinimalAnswer)
        assert roundtrip == ans

    def test_answer_dataset_produces_valid_student_search_results_and_answer_json(
        self,
        dummy_model: DummyCausalLM,
        dummy_tokenizer: DummyTokenizer,
        sample_chunks: List[InternalChunk],
        tmp_path: Path,
    ) -> None:
        search_results = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="q1",
                    question="How to configure the OpenAI compatible server?",
                    retrieved_sources=[c.to_source() for c in sample_chunks],
                ),
                MinimalSearchResults(
                    question_id="q2",
                    question="Which function builds the async engine client?",
                    retrieved_sources=[sample_chunks[1].to_source()],
                ),
            ],
            k=5,
        )

        input_json_path = tmp_path / "search_results" / "dataset_docs_public.json"
        input_json_path.parent.mkdir(parents=True, exist_ok=True)
        input_json_path.write_text(
            search_results.model_dump_json(indent=2),
            encoding="utf-8",
        )

        save_dir = tmp_path / "search_results_and_answer"
        engine = LLMInferenceEngine(
            _model=dummy_model,
            _tokenizer=dummy_tokenizer,
            chunks=sample_chunks,
        )

        batch_ans = engine.answer_dataset(
            student_search_results_path=input_json_path,
            save_directory=save_dir,
            show_progress=False,
        )

        assert isinstance(batch_ans, StudentSearchResultsAndAnswer)
        assert batch_ans.k == 5
        assert len(batch_ans.search_results) == 2
        assert all(isinstance(item, MinimalAnswer) for item in batch_ans.search_results)

        expected_output_file = save_dir / "dataset_docs_public.json"
        assert expected_output_file.is_file()

        saved_payload = json.loads(expected_output_file.read_text(encoding="utf-8"))
        assert saved_payload["k"] == 5
        assert len(saved_payload["search_results"]) == 2
        assert (
            saved_payload["search_results"][0]["answer"]
            == "To configure the OpenAI server, use `vllm serve` with `--port 8000`."
        )

        parsed_batch = parse_answer_json(expected_output_file.read_text(encoding="utf-8"))
        assert isinstance(parsed_batch, StudentSearchResultsAndAnswer)
        assert parsed_batch == batch_ans

    def test_clean_generated_answer_strips_think_tags_chatml_and_json_wrappers(
        self,
    ) -> None:
        # 1. Qwen3 <think>...</think> block + special tokens
        raw_with_think = (
            "<|im_start|>assistant\n"
            "<think>\nLet me check [Source 1] for the LoRA endpoint.\n</think>\n\n"
            "Answer: Use `/v1/load_lora_adapter` to load adapters dynamically.<|im_end|>"
        )
        assert (
            clean_generated_answer(raw_with_think)
            == "Use `/v1/load_lora_adapter` to load adapters dynamically."
        )

        # 2. Unclosed <think> tag truncated at max_new_tokens=128
        raw_unclosed_think = "<think>\nvLLM uses PagedAttention to manage KV cache blocks."
        assert (
            clean_generated_answer(raw_unclosed_think)
            == "vLLM uses PagedAttention to manage KV cache blocks."
        )

        # 3. Echoed prompt prefix removal
        prompt = "System Instructions:\nAnswer strictly.\n\nQuestion:\nWhat is vLLM?\n\nAnswer:\n"
        echoed = prompt + "vLLM is a high-throughput LLM serving engine."
        assert (
            clean_generated_answer(echoed, prompt=prompt)
            == "vLLM is a high-throughput LLM serving engine."
        )

        # 4. Model wrapped response in ```json {"answer": "..."} ```
        fenced_json = '```json\n{"answer": "Set `--tensor-parallel-size` to 4."}\n```'
        assert (
            clean_generated_answer(fenced_json)
            == "Set `--tensor-parallel-size` to 4."
        )

        # 5. Empty or whitespace output falls back gracefully
        assert clean_generated_answer("   \n\t  ") == DEFAULT_FALLBACK_ANSWER
        assert clean_generated_answer("<think>only thinking</think>") == DEFAULT_FALLBACK_ANSWER


class TestEdgeCasesAndHelpers:
    """Verify source span clamping, degenerate inputs, and convenience helpers."""

    def test_clamps_oversized_minimal_source_spans_to_2000_chars(self) -> None:
        oversized = MinimalSource(
            file_path="data/raw/vllm-0.10.1/vllm/model.py",
            first_character_index=500,
            last_character_index=5000,  # 4500 chars > 2000 limit
        )
        clamped = to_minimal_source(oversized)
        assert clamped.first_character_index == 500
        assert clamped.last_character_index == 2500
        assert clamped.span_length == 2000

        norm_list = normalize_retrieved_sources([oversized], k=1)
        assert len(norm_list) == 1
        assert norm_list[0].span_length <= 2000

    def test_handles_degenerate_empty_queries_gracefully(
        self,
        dummy_model: DummyCausalLM,
        dummy_tokenizer: DummyTokenizer,
    ) -> None:
        engine = LLMInferenceEngine(
            _model=dummy_model,
            _tokenizer=dummy_tokenizer,
        )
        # Empty prompt in generate()
        assert engine.generate("   ") == EMPTY_QUERY_ANSWER
        empty_json = json.loads(engine.generate("", return_json=True))
        assert empty_json["answer"] == EMPTY_QUERY_ANSWER

        # Empty question in generate_answer() does not crash with ValidationError
        ans = engine.generate_answer("   ", retrieved_sources=[])
        assert isinstance(ans, MinimalAnswer)
        assert ans.answer == EMPTY_QUERY_ANSWER

    def test_module_level_convenience_functions_and_callable_interface(
        self,
        dummy_model: DummyCausalLM,
        dummy_tokenizer: DummyTokenizer,
        sample_chunks: List[InternalChunk],
    ) -> None:
        engine = LLMInferenceEngine(
            _model=dummy_model,
            _tokenizer=dummy_tokenizer,
        )
        text = engine.generate("How to configure OpenAI server?")
        assert "vllm serve" in text

        ans = generate_answer(
            "How to configure OpenAI server?",
            sample_chunks,
            _model=dummy_model,
            _tokenizer=dummy_tokenizer,
        )
        assert isinstance(ans, MinimalAnswer)

        ans_json = generate_answer_json(
            "How to configure OpenAI server?",
            sample_chunks,
            _model=dummy_model,
            _tokenizer=dummy_tokenizer,
        )
        parsed = json.loads(ans_json)
        assert parsed["question"] == "How to configure OpenAI server?"

        batch = answer_dataset(
            [
                UnansweredQuestion(
                    question_id="uq-1",
                    question="How to configure OpenAI server?",
                ),
                AnsweredQuestion(
                    question_id="aq-1",
                    question="How does run_server work?",
                    sources=[sample_chunks[1].to_source()],
                    answer="Reference answer",
                ),
            ],
            show_progress=False,
            _model=dummy_model,
            _tokenizer=dummy_tokenizer,
        )
        assert isinstance(batch, StudentSearchResultsAndAnswer)
        assert len(batch.search_results) == 2

        # Test callable engine interface
        engine = LLMInferenceEngine(_model=dummy_model, _tokenizer=dummy_tokenizer)
        assert isinstance(engine("Prompt only"), str)
        assert isinstance(engine("Question?", sample_chunks), MinimalAnswer)

    def test_invalid_init_and_generate_parameters_raise_value_error(
        self,
        dummy_model: DummyCausalLM,
        dummy_tokenizer: DummyTokenizer,
    ) -> None:
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            LLMInferenceEngine(model_name="   ")

        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            LLMInferenceEngine(max_new_tokens=0)

        with pytest.raises(ValueError, match="temperature must be non-negative"):
            LLMInferenceEngine(temperature=-0.1)

        with pytest.raises(ValueError, match="max_prompt_tokens must be positive"):
            LLMInferenceEngine(max_prompt_tokens=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            LLMInferenceEngine(top_k=0)

        with pytest.raises(ValueError, match="max_ram_gb must be positive"):
            LLMInferenceEngine(max_ram_gb=0.0)

        engine = LLMInferenceEngine(_model=dummy_model, _tokenizer=dummy_tokenizer)
        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            engine.generate("test", max_new_tokens=0)

        with pytest.raises(ValueError, match="temperature must be non-negative"):
            engine.generate("test", temperature=-1.0)

        with pytest.raises(FileNotFoundError):
            engine.answer_dataset(student_search_results_path="nonexistent_file.json")

        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_answer_json("{not valid json")
