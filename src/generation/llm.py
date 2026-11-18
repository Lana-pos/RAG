"""Local CPU Inference Engine for Qwen/Qwen3-0.6B.
while prompt.py was the factory that built the text prompt string, llm.py takes that prompt,
loads the Qwen model into your computer's RAM, generates the answer, cleans up the text,
and formats everything into clean JSON.
Executes grounded natural language answer generation on CPU using Hugging Face
``transformers`` and ``torch`` with:
    1. ``torch.inference_mode()`` and greedy decoding
       (``temperature=0.0``, ``max_new_tokens=128``, ``do_sample=False``).
    2. Compact ``bfloat16`` CPU weight loading (``low_cpu_mem_usage=True``)
       keeping total RAM footprint strictly under 1.5 GB.
    3. Cleanly formatted, Pydantic-validated JSON serialization for
       :class:`~src.models.MinimalAnswer` and
       :class:`~src.models.StudentSearchResultsAndAnswer`.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
import re
import resource
import sys
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from tqdm import tqdm

from src.generation.prompt import (
    ContextLoader,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MODEL_NAME as DEFAULT_MODEL_NAME,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TOP_K,
    GroundedSource,
    PromptBuilder,
    extract_source_metadata,
)
from src.indexing.schema import InternalChunk
from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
    normalize_file_path,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


MODEL_NAME: str = DEFAULT_MODEL_NAME
DEFAULT_MAX_NEW_TOKENS: int = 128
MAX_NEW_TOKENS: int = DEFAULT_MAX_NEW_TOKENS
DEFAULT_TEMPERATURE: float = 0.0
TEMPERATURE: float = DEFAULT_TEMPERATURE
DEFAULT_DO_SAMPLE: bool = False
DO_SAMPLE: bool = DEFAULT_DO_SAMPLE
DEFAULT_DEVICE: str = "cpu"
DEFAULT_DTYPE_NAME: str = "bfloat16"

# Memory footprint ceiling: 1.5 GB
MAX_RAM_GB: float = 1.5
MAX_RAM_MB: float = 1536.0
MAX_RAM_BYTES: int = int(1.5 * 1024 * 1024 * 1024)

DEFAULT_FALLBACK_ANSWER: str = (
    "Based on the retrieved context, there is insufficient information "
    "to determine the answer."
)
EMPTY_QUERY_ANSWER: str = (
    "No question was provided. Please provide a valid technical question."
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SPECIAL_TOKENS_RE = re.compile(
    r"<\|im_start\|>\s*(?:assistant|user|system)?\s*"
    r"|<\|im_end\|>"
    r"|<\|endoftext\|>"
    r"|<\|eot_id\|>",
    re.IGNORECASE,
)
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:final\s+answer|answer|assistant|response)\s*:\s*",
    re.IGNORECASE,
)
_CODE_FENCE_JSON_RE = re.compile(
    r"^```(?:json)?\s*(.*?)\s*```$",
    re.DOTALL | re.IGNORECASE,
)


def get_process_memory_bytes() -> int:
    """Return current resident set size (RSS) memory of the process in bytes.
    Checks how much RAM your Python program is currently using on your computer's operating system
    """
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            return int(parts[1]) * 1024
        except OSError:
            pass

    statm_path = Path("/proc/self/statm")
    if statm_path.is_file():
        try:
            with open(statm_path, "r", encoding="utf-8") as f:
                fields = f.read().split()
                if len(fields) >= 2 and fields[1].isdigit():
                    page_size = resource.getpagesize()
                    return int(fields[1]) * page_size
        except OSError:
            pass

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def get_process_memory_mb() -> float:
    """Return current process RSS memory usage in megabytes (MB)."""
    return get_process_memory_bytes() / (1024.0 * 1024.0)


def get_process_memory_gb() -> float:
    """Return current process RSS memory usage in gigabytes (GB / GiB)."""
    return get_process_memory_bytes() / (1024.0**3)


def estimate_model_memory_bytes(model: Any) -> int:
    """Compute memory footprint in bytes of a PyTorch model's parameters and buffers.
    Safely returns ``0`` if *model* is ``None`` or a mock without tensor parameters.
        model: Loaded PyTorch / Hugging Face model or mock object.
    """
    if model is None:
        return 0

    try:
        if hasattr(model, "get_memory_footprint") and callable(
            model.get_memory_footprint
        ):
            footprint = model.get_memory_footprint()
            if isinstance(footprint, (int, float)) and not isinstance(
                footprint, bool
            ):
                return max(0, int(footprint))
    except Exception:
        pass

    total_bytes = 0
    try:
        if hasattr(model, "parameters") and callable(model.parameters):
            for param in model.parameters():
                if isinstance(param, torch.Tensor):
                    total_bytes += int(param.numel()) * int(
                        param.element_size()
                    )
        if hasattr(model, "buffers") and callable(model.buffers):
            for buf in model.buffers():
                if isinstance(buf, torch.Tensor):
                    total_bytes += int(buf.numel()) * int(buf.element_size())
    except Exception:
        return 0

    return total_bytes


def estimate_model_memory_mb(model: Any) -> float:
    """Return estimated model parameter + buffer footprint in megabytes (MB)."""
    return estimate_model_memory_bytes(model) / (1024.0 * 1024.0)


def estimate_model_memory_gb(model: Any) -> float:
    """Return estimated model parameter + buffer footprint in gigabytes (GB / GiB)."""
    return estimate_model_memory_bytes(model) / (1024.0**3)


def resolve_torch_dtype(
    dtype: Optional[Any] = None,
    device: str = DEFAULT_DEVICE,
) -> torch.dtype:
    """Forces PyTorch to load Qwen using bfloat16 instead of float32.
    This cuts the model's memory usage in half
    (~1.19 GB instead of ~2.4 GB) so it fits comfortably on a standard CPU without crashing.
    """
    if isinstance(dtype, torch.dtype):
        return dtype

    if isinstance(dtype, str):
        normalized = dtype.strip().lower()
        mapping: Dict[str, torch.dtype] = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "auto": torch.bfloat16,
        }
        if normalized in mapping:
            return mapping[normalized]
        raise ValueError(
            f"Unsupported torch_dtype '{dtype}'. Expected one of {sorted(mapping.keys())}."
        )

    if dtype is None:
        return torch.bfloat16

    raise TypeError(
        f"torch_dtype must be a torch.dtype, str, or None, got {type(dtype).__name__}"
    )


def clean_generated_answer(
    raw_text: Any,
    prompt: Optional[str] = None,
    fallback: str = DEFAULT_FALLBACK_ANSWER,
) -> str:
    """Clean raw model output into a coherent natural language answer string.
    Removes echoed prompt prefixes, Qwen3 ``<think>...</think>`` reasoning blocks,
    ChatML special tokens (``<|im_start|>``, ``<|im_end|>``, ``<|endoftext|>``),
    and unwraps nested JSON ``{"answer": "..."}`` objects if the LLM emitted JSON.
        raw_text: Raw decoded string from the language model.
        prompt: Optional input prompt to strip if echoed at the start of *raw_text*.
        fallback: Fallback string returned if the cleaned output is empty.
    """
    if raw_text is None:
        return fallback

    text = str(raw_text)
    if not text.strip():
        return fallback

    # 1. Strip echoed prompt prefix if present
    if prompt and text.startswith(prompt):
        text = text[len(prompt) :]
    elif prompt and prompt.strip() and text.strip().startswith(prompt.strip()):
        idx = text.find(prompt.strip())
        if idx != -1:
            text = text[idx + len(prompt.strip()) :]

    # 2. If full ChatML conversation was decoded, extract only the last assistant turn
    assistant_marker = "<|im_start|>assistant"
    if assistant_marker in text:
        text = text.rsplit(assistant_marker, 1)[-1]

    # 3. Remove completed <think>...</think> blocks
    text = _THINK_BLOCK_RE.sub("", text)

    # 4. Handle unclosed <think> tags (e.g. truncated at max_new_tokens=128)
    lower_text = text.lower()
    if "<think>" in lower_text:
        think_pos = lower_text.find("<think>")
        before_think = text[:think_pos].strip()
        after_think = text[think_pos + len("<think>") :].strip()
        text = before_think if before_think else after_think

    # 5. Strip stray closing </think> tags and special tokens
    text = re.sub(r"</think>", "", text, flags=re.IGNORECASE)
    text = _SPECIAL_TOKENS_RE.sub("", text)
    text = text.strip()

    # 6. Unwrap markdown code fences if the entire response is wrapped in json ...
    fence_match = _CODE_FENCE_JSON_RE.match(text)
    if fence_match:
        inner = fence_match.group(1).strip()
        if inner:
            text = inner

    # 7. If the model emitted a JSON object like {"answer": "..."}, extract the string
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, Mapping):
                for key in ("answer", "response", "text", "result", "content"):
                    val = parsed.get(key)
                    if isinstance(val, str) and val.strip():
                        text = val.strip()
                        break
        except (json.JSONDecodeError, TypeError):
            pass

    # 8. Strip redundant leading "Answer:" / "Assistant:" label
    text = _ANSWER_PREFIX_RE.sub("", text).strip()

    # Cut off any repeated "\nAnswer:" or "\nQuestion:" continuation blocks
    text = re.split(
        r"\n+\s*(?:final\s+answer|answer|question|user|assistant)\s*:",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    # 9. Collapse 3+ consecutive blank lines into double newline
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return fallback
    return text


def to_minimal_source(
    source: Any,
    default_index: int = 1,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> MinimalSource:
    """Convert any supported source representation into a clamped :class:`MinimalSource`.
    Guarantees ``last_character_index - first_character_index <= max_chunk_chars``
    (default 2000 characters) as required by the moulinette grader.
        source: MinimalSource, InternalChunk, GroundedSource, dict, tuple, or str.
        default_index: 1-based fallback index for synthetic sources.
        max_chunk_chars: Maximum allowed character span width (default: 2000).
    """
    if isinstance(source, MinimalSource):
        if source.span_length <= max_chunk_chars:
            return source
        return MinimalSource(
            file_path=source.file_path,
            first_character_index=source.first_character_index,
            last_character_index=source.first_character_index
            + max_chunk_chars,
        )

    if isinstance(source, GroundedSource):
        ms = source.to_minimal_source()
        return to_minimal_source(
            ms, default_index=default_index, max_chunk_chars=max_chunk_chars
        )

    if isinstance(source, InternalChunk):
        ms = source.to_source()
        return to_minimal_source(
            ms, default_index=default_index, max_chunk_chars=max_chunk_chars
        )

    file_path, first_idx, last_idx, _ = extract_source_metadata(
        source, default_index=default_index
    )
    first_clamped = max(0, int(first_idx))
    last_clamped = max(first_clamped + 1, int(last_idx))
    if last_clamped - first_clamped > max_chunk_chars:
        last_clamped = first_clamped + max_chunk_chars

    try:
        norm_path = normalize_file_path(file_path)
    except ValueError:
        norm_path = f"context_snippet_{default_index}.txt"

    return MinimalSource(
        file_path=norm_path,
        first_character_index=first_clamped,
        last_character_index=last_clamped,
    )


def normalize_retrieved_sources(
    sources: Optional[Sequence[Any]],
    k: Optional[int] = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> List[MinimalSource]:
    """Normalize a sequence of candidate sources into validated :class:`MinimalSource` items."""
    if not sources:
        return []
    if k is not None and k <= 0:
        return []

    selected = list(sources[:k]) if k is not None else list(sources)
    return [
        to_minimal_source(
            item, default_index=idx + 1, max_chunk_chars=max_chunk_chars
        )
        for idx, item in enumerate(selected)
    ]


def format_answer_json(
    answer: Union[
        MinimalAnswer,
        StudentSearchResultsAndAnswer,
        MinimalSearchResults,
        Mapping[str, Any],
        str,
    ],
    *,
    question: Optional[str] = None,
    question_id: Optional[str] = None,
    retrieved_sources: Optional[Sequence[Any]] = None,
    k: Optional[int] = None,
    indent: int = 2,
) -> str:
    """Format a generated answer or batch of answers into cleanly formatted JSON.
    Validates the payload against :class:`MinimalAnswer` or
    :class:`StudentSearchResultsAndAnswer` before serializing with consistent
    UTF-8 indentation.
        answer: MinimalAnswer, StudentSearchResultsAndAnswer, dict, or raw answer string.
        question: Question text when *answer* is passed as a plain string.
        question_id: Optional question identifier when *answer* is a plain string.
        retrieved_sources: Optional retrieved sources when *answer* is a plain string.
        k: Optional top-k count if formatting a batch payload.
        indent: JSON indentation spaces (default: 2).
    """
    if isinstance(answer, (MinimalAnswer, StudentSearchResultsAndAnswer)):
        payload = json.loads(answer.model_dump_json())
        return json.dumps(payload, ensure_ascii=False, indent=indent)

    if isinstance(answer, MinimalSearchResults):
        minimal_ans = MinimalAnswer(
            question_id=question_id or answer.question_id,
            question=question or answer.question,
            retrieved_sources=normalize_retrieved_sources(
                (
                    retrieved_sources
                    if retrieved_sources is not None
                    else answer.retrieved_sources
                ),
                k=k,
            ),
            answer=DEFAULT_FALLBACK_ANSWER,
        )
        payload = json.loads(minimal_ans.model_dump_json())
        return json.dumps(payload, ensure_ascii=False, indent=indent)

    if isinstance(answer, Mapping):
        if "search_results" in answer:
            batch_data = dict(answer)
            if k is not None and "k" not in batch_data:
                batch_data["k"] = max(1, k)
            elif "k" not in batch_data:
                results_seq = batch_data.get("search_results", [])
                batch_data["k"] = max(
                    1, len(results_seq) if isinstance(results_seq, list) else 5
                )
            validated_batch = StudentSearchResultsAndAnswer.model_validate(
                batch_data
            )
            payload = json.loads(validated_batch.model_dump_json())
            return json.dumps(payload, ensure_ascii=False, indent=indent)

        single_data = dict(answer)
        if question is not None and "question" not in single_data:
            single_data["question"] = question
        if question_id is not None and "question_id" not in single_data:
            single_data["question_id"] = question_id
        if (
            retrieved_sources is not None
            and "retrieved_sources" not in single_data
        ):
            single_data["retrieved_sources"] = [
                s.model_dump()
                for s in normalize_retrieved_sources(retrieved_sources, k=k)
            ]
        if (
            "question" not in single_data
            or not str(single_data["question"]).strip()
        ):
            single_data["question"] = "[Question]"
        if "answer" not in single_data:
            single_data["answer"] = DEFAULT_FALLBACK_ANSWER
        validated_single = MinimalAnswer.model_validate(single_data)
        payload = json.loads(validated_single.model_dump_json())
        return json.dumps(payload, ensure_ascii=False, indent=indent)

    # Plain string answer
    raw_str = str(answer).strip()
    if raw_str.startswith("{") and raw_str.endswith("}"):
        try:
            parsed_dict = json.loads(raw_str)
            if isinstance(parsed_dict, Mapping) and (
                "search_results" in parsed_dict
                or ("question" in parsed_dict and "answer" in parsed_dict)
            ):
                return format_answer_json(
                    parsed_dict,
                    question=question,
                    question_id=question_id,
                    retrieved_sources=retrieved_sources,
                    k=k,
                    indent=indent,
                )
        except json.JSONDecodeError:
            pass

    clean_q = (question or "").strip() or "[Question]"
    norm_sources = normalize_retrieved_sources(retrieved_sources, k=k)
    kwargs: Dict[str, Any] = {
        "question": clean_q,
        "retrieved_sources": norm_sources,
        "answer": clean_generated_answer(raw_str),
    }
    if question_id is not None:
        kwargs["question_id"] = question_id

    validated = MinimalAnswer(**kwargs)
    payload = json.loads(validated.model_dump_json())
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def parse_answer_json(
    json_data: Union[str, bytes, Mapping[str, Any]],
) -> Union[MinimalAnswer, StudentSearchResultsAndAnswer]:
    """Parse and validate a JSON string or dict into Pydantic answer models.
        json_data: JSON string, bytes, or dictionary representing either a
            :class:`MinimalAnswer` or :class:`StudentSearchResultsAndAnswer`.
    """
    try:
        if isinstance(json_data, (str, bytes)):
            parsed = json.loads(json_data)
        elif isinstance(json_data, Mapping):
            parsed = dict(json_data)
        else:
            raise ValueError(
                f"Expected JSON str, bytes, or dict, got {type(json_data).__name__}"
            )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc

    if not isinstance(parsed, Mapping):
        raise ValueError(
            f"Expected JSON object at root, got {type(parsed).__name__}"
        )

    if "search_results" in parsed:
        return StudentSearchResultsAndAnswer.model_validate(parsed)
    return MinimalAnswer.model_validate(parsed)


class LLMInferenceEngine:
    """Local CPU Inference Engine for ``Qwen/Qwen3-0.6B`` using Hugging Face Transformers.
    Acceptance Criteria Compliance:
        1. **Inference Mode & Greedy Decoding:** Executes all forward passes inside
           ``with torch.inference_mode():`` using deterministic greedy decoding
           (``temperature=0.0``, ``max_new_tokens=128``, ``do_sample=False``).
        2. **Memory Footprint < 1.5 GB:** Loads ``Qwen/Qwen3-0.6B`` in ``torch.bfloat16``
           with ``low_cpu_mem_usage=True`` (~1.19 GB weight footprint), disables
           parameter gradients, and releases intermediate tensors via garbage collection.
        3. **Clean JSON Output:** Returns grounded natural language answers validated
           and formatted via :class:`~src.models.MinimalAnswer` and
           :class:`~src.models.StudentSearchResultsAndAnswer`.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = DEFAULT_DO_SAMPLE,
        device: str = DEFAULT_DEVICE,
        torch_dtype: Optional[Any] = None,
        max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
        top_k: int = DEFAULT_TOP_K,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        max_ram_gb: float = MAX_RAM_GB,
        low_cpu_mem_usage: bool = True,
        local_files_only: bool = False,
        use_chat_template: bool = True,
        enable_thinking: bool = False,
        lazy_load: bool = True,
        prompt_builder: Optional[PromptBuilder] = None,
        context_loader: Optional[ContextLoader] = None,
        chunks: Optional[Sequence[InternalChunk]] = None,
        base_dir: Optional[Union[str, Path]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        _model: Optional[Any] = None,
        _tokenizer: Optional[Any] = None,
    ) -> None:
        """Initialize the Qwen3-0.6B Local CPU Inference Engine.
            model_name: Hugging Face model identifier (default: ``"Qwen/Qwen3-0.6B"``).
            temperature: Decoding temperature (default: ``0.0`` for greedy decoding).
            max_new_tokens: Maximum number of generated tokens (default: ``128``).
            do_sample: Sampling flag (default: ``False``; forced ``False`` when
                ``temperature == 0.0``).
            device: Execution device string (default: ``"cpu"``).
            torch_dtype: Weight dtype (default: ``torch.bfloat16`` on CPU for < 1.5 GB RAM).
            max_prompt_tokens: Maximum prompt token budget (default: ``3000``).
            top_k: Default number of top retrieved sources to ground prompts (default: ``5``).
            max_chunk_chars: Hard character ceiling per source chunk (default: ``2000``).
            max_ram_gb: Maximum allowed RAM footprint in GB (default: ``1.5``).
            low_cpu_mem_usage: Load weights without duplicate memory allocation (default: ``True``).
            local_files_only: Restrict Hugging Face hub lookups to local cache (default: ``False``).
            use_chat_template: Format prompts with ChatML markers (default: ``False``).
            enable_thinking: Enable Qwen3 ``<think>`` blocks (default: ``False``).
            lazy_load: Defer weight loading until first inference call (default: ``True``).
            prompt_builder: Optional pre-configured :class:`PromptBuilder` instance.
            context_loader: Optional pre-configured :class:`ContextLoader` instance.
            chunks: Optional in-memory :class:`InternalChunk` catalog for span hydration.
            base_dir: Optional root directory for resolving source files on disk.
            system_prompt: Grounding system instruction text.
            model: Optional pre-instantiated causal LM (or mock for offline testing).
            tokenizer: Optional pre-instantiated tokenizer (or mock for offline testing).
            _model: Alias for *model* (matching ``DenseEncoder(_model=...)`` convention).
            _tokenizer: Alias for *tokenizer*.
        """
        if not model_name or not str(model_name).strip():
            raise ValueError("model_name cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be positive, got {max_new_tokens}"
            )
        if temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {temperature}"
            )
        if max_prompt_tokens <= 0:
            raise ValueError(
                f"max_prompt_tokens must be positive, got {max_prompt_tokens}"
            )
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if max_chunk_chars <= 0:
            raise ValueError(
                f"max_chunk_chars must be positive, got {max_chunk_chars}"
            )
        if max_ram_gb <= 0.0:
            raise ValueError(f"max_ram_gb must be positive, got {max_ram_gb}")

        self.model_name: str = str(model_name).strip()
        self.temperature: float = float(temperature)
        self.max_new_tokens: int = int(max_new_tokens)
        self.do_sample: bool = (
            False if self.temperature == 0.0 else bool(do_sample)
        )
        self.device: str = str(device).strip() or DEFAULT_DEVICE
        self.torch_dtype: torch.dtype = resolve_torch_dtype(
            torch_dtype, device=self.device
        )
        self.max_prompt_tokens: int = int(max_prompt_tokens)
        self.top_k: int = int(top_k)
        self.max_chunk_chars: int = int(max_chunk_chars)
        self.max_ram_gb: float = float(max_ram_gb)
        self.max_ram_bytes: int = int(self.max_ram_gb * (1024**3))
        self.low_cpu_mem_usage: bool = bool(low_cpu_mem_usage)
        self.local_files_only: bool = bool(local_files_only)
        self.use_chat_template: bool = bool(use_chat_template)
        self.enable_thinking: bool = bool(enable_thinking)
        self.lazy_load: bool = bool(lazy_load)

        injected_model = model if model is not None else _model
        injected_tokenizer = tokenizer if tokenizer is not None else _tokenizer

        self._model: Optional[Any] = injected_model
        self._tokenizer: Optional[Any] = injected_tokenizer

        # Telemetry for verifying inference_mode() and decoding parameters in tests
        self._last_inference_mode_enabled: bool = False
        self._last_generation_kwargs: Dict[str, Any] = {}

        self.prompt_builder: PromptBuilder = (
            prompt_builder
            if prompt_builder is not None
            else PromptBuilder(
                max_tokens=self.max_prompt_tokens,
                top_k=self.top_k,
                system_prompt=system_prompt,
                max_chunk_chars=self.max_chunk_chars,
                tokenizer=self._tokenizer,
                context_loader=context_loader,
                chunks=chunks,
                base_dir=base_dir,
                use_chat_template=self.use_chat_template,
                enable_thinking=self.enable_thinking,
            )
        )

        if not self.lazy_load and (
            self._model is None or self._tokenizer is None
        ):
            self._load_model()

    @property
    def is_loaded(self) -> bool:
        """Return True if both tokenizer and model are loaded in memory."""
        return self._model is not None and self._tokenizer is not None

    @property
    def model(self) -> Any:
        """Return the underlying causal language model, loading it on demand if needed."""
        if self._model is None:
            self._load_model()
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        """Set or replace the underlying causal language model."""
        self._model = value

    @property
    def tokenizer(self) -> Any:
        """Return the underlying Hugging Face tokenizer, loading it on demand if needed."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Any) -> None:
        """Set or replace the underlying tokenizer and sync with prompt_builder."""
        self._tokenizer = value
        if hasattr(self, "prompt_builder") and self.prompt_builder is not None:
            self.prompt_builder.tokenizer = value

    @property
    def memory_footprint_bytes(self) -> int:
        """Return current model memory footprint in bytes (or process RSS if greater)."""
        model_bytes = estimate_model_memory_bytes(self._model)
        if model_bytes > 0:
            return model_bytes
        return get_process_memory_bytes()

    @property
    def memory_footprint_mb(self) -> float:
        """Return current memory footprint in megabytes (MB)."""
        return self.memory_footprint_bytes / (1024.0 * 1024.0)

    @property
    def memory_footprint_gb(self) -> float:
        """Return current memory footprint in gigabytes (GB)."""
        return self.memory_footprint_bytes / (1024.0**3)

    def get_memory_usage_bytes(self) -> int:
        """Return current memory footprint in bytes."""
        return self.memory_footprint_bytes

    def get_memory_usage_mb(self) -> float:
        """Return current memory footprint in megabytes (MB)."""
        return self.memory_footprint_mb

    def get_memory_usage_gb(self) -> float:
        """Return current memory footprint in gigabytes (GB)."""
        return self.memory_footprint_gb

    def check_memory_budget(self, max_gb: Optional[float] = None) -> bool:
        """Verify that the memory footprint stays strictly under *max_gb* (default 1.5 GB).
        max_gb: Optional override ceiling in GB (defaults to ``self.max_ram_gb``).
        """
        limit_gb = max_gb if max_gb is not None else self.max_ram_gb
        model_gb = estimate_model_memory_gb(self._model)
        if model_gb > 0:
            return model_gb < limit_gb
        return get_process_memory_gb() < limit_gb

    def _load_tokenizer(self) -> Any:
        """Load the Hugging Face AutoTokenizer for ``self.model_name``."""
        if AutoTokenizer is None:
            raise ImportError(
                "transformers is required for LLM inference. "
                "Install it with: uv sync"
            )

        logger.info("Loading tokenizer '%s'...", self.model_name)
        tok_kwargs: Dict[str, Any] = {}
        if self.local_files_only:
            tok_kwargs["local_files_only"] = True

        try:
            tok = AutoTokenizer.from_pretrained(self.model_name, **tok_kwargs)
        except TypeError:
            tok = AutoTokenizer.from_pretrained(self.model_name)

        if (
            getattr(tok, "pad_token", None) is None
            and getattr(tok, "eos_token", None) is not None
        ):
            try:
                tok.pad_token = tok.eos_token
            except Exception:
                pass

        return tok

    def _load_causal_lm(self) -> Any:
        """Load the Hugging Face AutoModelForCausalLM on CPU within < 1.5 GB RAM."""
        if AutoModelForCausalLM is None:
            raise ImportError(
                "transformers is required for LLM inference. "
                "Install it with: uv sync"
            )

        logger.info(
            "Loading model '%s' on '%s' with dtype=%s...",
            self.model_name,
            self.device,
            self.torch_dtype,
        )

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
        }
        if self.local_files_only:
            load_kwargs["local_files_only"] = True

        loaded_model: Any
        try:
            loaded_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
        except TypeError:
            # Fallback if a newer/older transformers or mock rejects specific kwargs
            fallback_kwargs: Dict[str, Any] = {"torch_dtype": self.torch_dtype}
            try:
                loaded_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **fallback_kwargs,
                )
            except TypeError:
                loaded_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name
                )

        # Ensure evaluation mode and disable parameter gradients to save memory
        if hasattr(loaded_model, "eval") and callable(loaded_model.eval):
            loaded_model.eval()

        if hasattr(loaded_model, "parameters") and callable(
            loaded_model.parameters
        ):
            try:
                for param in loaded_model.parameters():
                    if isinstance(param, torch.Tensor):
                        param.requires_grad_(False)
            except Exception:
                pass

        # Downcast to bfloat16 if model was loaded in float32 and exceeds budget
        model_gb = estimate_model_memory_gb(loaded_model)
        if (
            model_gb >= self.max_ram_gb
            and hasattr(loaded_model, "to")
            and callable(loaded_model.to)
        ):
            logger.info(
                "Model footprint (%.2f GB) exceeds %.2f GB budget; casting to bfloat16.",
                model_gb,
                self.max_ram_gb,
            )
            loaded_model = loaded_model.to(dtype=torch.bfloat16)
            gc.collect()

        # Move to target device if not already there
        if hasattr(loaded_model, "to") and callable(loaded_model.to):
            try:
                param_iter = (
                    iter(loaded_model.parameters())
                    if hasattr(loaded_model, "parameters")
                    and callable(loaded_model.parameters)
                    else iter(())
                )
                first_param = next(param_iter, None)
                if (
                    isinstance(first_param, torch.Tensor)
                    and str(first_param.device) != self.device
                ):
                    loaded_model = loaded_model.to(self.device)
            except Exception:
                pass

        gc.collect()
        return loaded_model

    def _load_model(self) -> Tuple[Any, Any]:
        """Load tokenizer and model if not already initialized."""
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer()
            if self.prompt_builder.tokenizer is None:
                self.prompt_builder.tokenizer = self._tokenizer

        if self._model is None:
            self._model = self._load_causal_lm()

        return self._tokenizer, self._model

    def load_model(self) -> Tuple[Any, Any]:
        """Public method to explicitly load the tokenizer and model into memory."""
        return self._load_model()

    def unload_model(self) -> None:
        """Release model and tokenizer references and reclaim RAM via garbage collection."""
        self._model = None
        self._tokenizer = None
        if hasattr(self, "prompt_builder") and self.prompt_builder is not None:
            self.prompt_builder.tokenizer = None
        gc.collect()

    def _prepare_inputs(
        self,
        prompt_text: str,
        tokenizer: Any,
    ) -> Tuple[Dict[str, Any], int, Optional[torch.Tensor]]:
        """Tokenize *prompt_text* into model input tensors on ``self.device``.
        Supports standard Hugging Face tokenizers, callable mocks, and `.encode` mocks.
        """
        raw_encoded: Any = None
        if callable(tokenizer):
            try:
                raw_encoded = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_prompt_tokens,
                )
            except TypeError:
                try:
                    raw_encoded = tokenizer(prompt_text, return_tensors="pt")
                except TypeError:
                    raw_encoded = tokenizer(prompt_text)

        model_inputs: Dict[str, Any] = {}

        if isinstance(raw_encoded, Mapping) or (
            raw_encoded is not None and hasattr(raw_encoded, "keys")
        ):
            for key in raw_encoded.keys():
                val = raw_encoded[key]
                if isinstance(val, list):
                    tensor_val = torch.tensor(
                        [val] if (val and isinstance(val[0], int)) else val,
                        dtype=torch.long,
                    )
                    model_inputs[key] = tensor_val.to(self.device)
                elif isinstance(val, torch.Tensor):
                    if val.dim() == 1:
                        val = val.unsqueeze(0)
                    model_inputs[key] = val.to(self.device)
                else:
                    model_inputs[key] = val
        elif isinstance(raw_encoded, torch.Tensor):
            tensor_ids = (
                raw_encoded.unsqueeze(0)
                if raw_encoded.dim() == 1
                else raw_encoded
            )
            model_inputs["input_ids"] = tensor_ids.to(self.device)
            model_inputs["attention_mask"] = torch.ones_like(
                model_inputs["input_ids"], device=self.device
            )
        elif isinstance(raw_encoded, list):
            tensor_ids = torch.tensor(
                (
                    [raw_encoded]
                    if (raw_encoded and isinstance(raw_encoded[0], int))
                    else (raw_encoded if raw_encoded else [[1]])
                ),
                dtype=torch.long,
                device=self.device,
            )
            model_inputs["input_ids"] = tensor_ids
            model_inputs["attention_mask"] = torch.ones_like(
                tensor_ids, device=self.device
            )

        if "input_ids" not in model_inputs and hasattr(tokenizer, "encode"):
            try:
                encoded_ids = tokenizer.encode(
                    prompt_text, return_tensors="pt"
                )
            except TypeError:
                encoded_ids = tokenizer.encode(prompt_text)

            if isinstance(encoded_ids, torch.Tensor):
                tensor_ids = (
                    encoded_ids.unsqueeze(0)
                    if encoded_ids.dim() == 1
                    else encoded_ids
                ).to(self.device)
            elif isinstance(encoded_ids, Sequence) and len(encoded_ids) > 0:
                if isinstance(encoded_ids[0], int):
                    tensor_ids = torch.tensor(
                        [list(encoded_ids)],
                        dtype=torch.long,
                        device=self.device,
                    )
                else:
                    tensor_ids = torch.tensor(
                        [list(encoded_ids[0])],
                        dtype=torch.long,
                        device=self.device,
                    )
            else:
                tensor_ids = torch.tensor(
                    [[1]], dtype=torch.long, device=self.device
                )

            model_inputs["input_ids"] = tensor_ids
            model_inputs["attention_mask"] = torch.ones_like(
                tensor_ids, device=self.device
            )

        if "input_ids" not in model_inputs:
            fallback_ids = torch.tensor(
                [[1]], dtype=torch.long, device=self.device
            )
            model_inputs["input_ids"] = fallback_ids
            model_inputs["attention_mask"] = torch.ones_like(
                fallback_ids, device=self.device
            )

        input_ids_obj = model_inputs.get("input_ids")
        input_ids_tensor: Optional[torch.Tensor] = (
            input_ids_obj if isinstance(input_ids_obj, torch.Tensor) else None
        )
        input_seq_len = (
            int(input_ids_tensor.shape[-1])
            if (input_ids_tensor is not None and input_ids_tensor.dim() >= 1)
            else 0
        )

        return model_inputs, input_seq_len, input_ids_tensor

    def _decode_generated_tokens(
        self,
        raw_output: Any,
        tokenizer: Any,
        input_seq_len: int,
        input_ids_tensor: Optional[torch.Tensor],
        prompt_text: str,
    ) -> str:
        """Decode newly generated token IDs into a string using *tokenizer*."""
        if isinstance(raw_output, str):
            return raw_output

        sequences: Any = raw_output
        if hasattr(raw_output, "sequences"):
            sequences = getattr(raw_output, "sequences")

        if isinstance(sequences, torch.Tensor):
            seq_tensor = sequences[0] if sequences.dim() >= 2 else sequences
            new_tokens_tensor = seq_tensor

            if input_seq_len > 0 and seq_tensor.shape[-1] > input_seq_len:
                # Verify if seq_tensor starts with input_ids (standard HF causal LM behavior)
                prefix_matches = True
                if (
                    input_ids_tensor is not None
                    and input_ids_tensor.dim() >= 2
                    and input_ids_tensor.shape[-1] == input_seq_len
                ):
                    try:
                        prefix_matches = bool(
                            torch.equal(
                                seq_tensor[:input_seq_len].cpu(),
                                input_ids_tensor[0].cpu(),
                            )
                        )
                    except Exception:
                        prefix_matches = True

                if prefix_matches:
                    new_tokens_tensor = seq_tensor[input_seq_len:]

            if hasattr(tokenizer, "decode") and callable(tokenizer.decode):
                try:
                    decoded = tokenizer.decode(
                        new_tokens_tensor,
                        skip_special_tokens=True,
                    )
                except TypeError:
                    decoded = tokenizer.decode(new_tokens_tensor)
                if isinstance(decoded, str):
                    return decoded

            if hasattr(tokenizer, "batch_decode") and callable(
                tokenizer.batch_decode
            ):
                batch_decoded = tokenizer.batch_decode(
                    new_tokens_tensor.unsqueeze(0),
                    skip_special_tokens=True,
                )
                if (
                    isinstance(batch_decoded, Sequence)
                    and len(batch_decoded) > 0
                ):
                    return str(batch_decoded[0])

        if isinstance(sequences, Sequence) and not isinstance(
            sequences, (str, bytes)
        ):
            seq_list: Sequence[Any] = (
                sequences[0]
                if (
                    len(sequences) > 0
                    and isinstance(sequences[0], Sequence)
                    and not isinstance(sequences[0], (str, bytes))
                )
                else sequences
            )
            tokens_to_decode = seq_list
            if (
                input_seq_len > 0
                and len(seq_list) > input_seq_len
                and input_ids_tensor is not None
            ):
                try:
                    prompt_list = input_ids_tensor[0].tolist()
                    if list(seq_list[:input_seq_len]) == prompt_list:
                        tokens_to_decode = seq_list[input_seq_len:]
                except Exception:
                    tokens_to_decode = seq_list[input_seq_len:]

            if hasattr(tokenizer, "decode") and callable(tokenizer.decode):
                try:
                    decoded_list = tokenizer.decode(
                        list(tokens_to_decode),
                        skip_special_tokens=True,
                    )
                except TypeError:
                    decoded_list = tokenizer.decode(list(tokens_to_decode))
                if isinstance(decoded_list, str):
                    return decoded_list

        return str(raw_output)

    def generate(
        self,
        prompt: Union[
            str, MinimalSearchResults, Sequence[Dict[str, str]]
        ] = "",
        sources: Optional[Sequence[Any]] = None,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        return_json: bool = False,
        as_json: bool = False,
        clean_output: bool = True,
        **kwargs: Any,
    ) -> str:
        """Execute CPU inference with ``torch.inference_mode()`` and greedy decoding.
        prompt: Grounded prompt string, :class:`MinimalSearchResults`, or chat messages.
        sources: Optional retrieved sources; if provided, formats *prompt* as a
            question with *sources* via ``self.prompt_builder``.
        max_new_tokens: Maximum tokens to generate (default: ``128``).
        temperature: Decoding temperature (default: ``0.0``).
        do_sample: Sampling toggle (default: ``False``; forced ``False`` when
            ``temperature == 0.0``).
        return_json: If True, wraps the generated answer in a JSON object string.
        as_json: Alias for *return_json*.
        clean_output: If True (default), strips ``<think>`` blocks and special tokens.
        **kwargs: Additional keyword arguments forwarded to ``model.generate``
        """
        effective_max_new_tokens = (
            int(max_new_tokens)
            if max_new_tokens is not None
            else self.max_new_tokens
        )
        if effective_max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be positive, got {effective_max_new_tokens}"
            )

        effective_temperature = (
            float(temperature) if temperature is not None else self.temperature
        )
        if effective_temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {effective_temperature}"
            )

        effective_do_sample = (
            False
            if effective_temperature == 0.0
            else (bool(do_sample) if do_sample is not None else self.do_sample)
        )

        # Build prompt string if structured inputs or sources were supplied
        if isinstance(prompt, MinimalSearchResults):
            prompt_text = self.prompt_builder.build_prompt(
                question=prompt.question,
                sources=(
                    sources
                    if sources is not None
                    else prompt.retrieved_sources
                ),
            )
        elif isinstance(prompt, Sequence) and not isinstance(prompt, str):
            # Chat messages list [{"role": ..., "content": ...}]
            parts: List[str] = []
            for msg in prompt:
                if isinstance(msg, Mapping):
                    role = str(msg.get("role", "user"))
                    content = str(msg.get("content", ""))
                    parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
            prompt_text = "\n".join(parts)
        elif sources is not None:
            prompt_text = self.prompt_builder.build_prompt(
                question=str(prompt),
                sources=sources,
            )
        else:
            prompt_text = str(prompt)

        if not prompt_text or not prompt_text.strip():
            empty_ans = EMPTY_QUERY_ANSWER
            if return_json or as_json:
                return json.dumps(
                    {"answer": empty_ans}, ensure_ascii=False, indent=2
                )
            return empty_ans

        tokenizer, model = self._load_model()

        # Execute strictly inside torch.inference_mode()
        with torch.inference_mode():
            self._last_inference_mode_enabled = bool(
                torch.is_inference_mode_enabled()
            )

            model_inputs, input_seq_len, input_ids_tensor = (
                self._prepare_inputs(prompt_text, tokenizer)
            )

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": effective_max_new_tokens,
                "temperature": effective_temperature,
                "do_sample": effective_do_sample,
            }

            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if isinstance(pad_token_id, int):
                gen_kwargs["pad_token_id"] = pad_token_id
            elif isinstance(eos_token_id, int):
                gen_kwargs["pad_token_id"] = eos_token_id
            if isinstance(eos_token_id, int):
                gen_kwargs["eos_token_id"] = eos_token_id

            gen_kwargs.update(kwargs)
            self._last_generation_kwargs = dict(gen_kwargs)

            try:
                raw_output = model.generate(**model_inputs, **gen_kwargs)
            except (ValueError, TypeError) as exc:
                # Some HF transformers releases reject temperature=0.0 when do_sample=False
                # during GenerationConfig validation; retry pure greedy decoding cleanly.
                if (
                    not effective_do_sample
                    and "temperature" in str(exc).lower()
                ):
                    greedy_kwargs = dict(gen_kwargs)
                    greedy_kwargs.pop("temperature", None)
                    greedy_kwargs["do_sample"] = False
                    greedy_kwargs["temperature"] = None
                    greedy_kwargs["top_p"] = None
                    greedy_kwargs["top_k"] = None
                    raw_output = model.generate(
                        **model_inputs, **greedy_kwargs
                    )
                else:
                    raise

            decoded_text = self._decode_generated_tokens(
                raw_output=raw_output,
                tokenizer=tokenizer,
                input_seq_len=input_seq_len,
                input_ids_tensor=input_ids_tensor,
                prompt_text=prompt_text,
            )

            del model_inputs
            del raw_output

        answer_text = (
            clean_generated_answer(decoded_text, prompt=prompt_text)
            if clean_output
            else decoded_text
        )

        if return_json or as_json:
            return json.dumps(
                {"answer": answer_text}, ensure_ascii=False, indent=2
            )
        return answer_text

    def generate_answer(
        self,
        question: Union[str, MinimalSearchResults, UnansweredQuestion] = "",
        retrieved_sources: Optional[Sequence[Any]] = None,
        *,
        sources: Optional[Sequence[Any]] = None,
        chunks: Optional[Sequence[Any]] = None,
        context_snippets: Optional[Sequence[str]] = None,
        question_id: Optional[str] = None,
        k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_prompt_tokens: Optional[int] = None,
        return_json: bool = False,
        as_json: bool = False,
        indent: int = 2,
    ) -> Union[MinimalAnswer, str]:
        """Generate a grounded natural language answer for *question* and *retrieved_sources*.
        question: Question string, :class:`MinimalSearchResults`, or
            :class:`UnansweredQuestion`.
        retrieved_sources: Candidate sources retrieved for the question.
        sources: Optional keyword alias for *retrieved_sources*.
        chunks: Optional keyword alias for *retrieved_sources*.
        context_snippets: Optional explicit text snippets paired with *retrieved_sources*.
        question_id: Optional question identifier override.
        k: Optional top-k source limit.
        max_new_tokens: Optional override for maximum generated tokens (default: 128).
        temperature: Optional override for decoding temperature (default: 0.0).
        max_prompt_tokens: Optional override for prompt token budget (default: 3000).
        return_json: If True, returns a cleanly formatted JSON string instead of
            a :class:`MinimalAnswer` instance.
        as_json: Alias for *return_json*.
        indent: Indentation spaces when serializing to JSON (default: 2).
        """
        resolved_q_id: Optional[str] = question_id
        raw_question_str: str = ""
        candidate_sources: Optional[Sequence[Any]] = (
            retrieved_sources
            if retrieved_sources is not None
            else (sources if sources is not None else chunks)
        )

        if isinstance(question, MinimalSearchResults):
            raw_question_str = question.question
            if resolved_q_id is None:
                resolved_q_id = question.question_id
            if candidate_sources is None:
                candidate_sources = question.retrieved_sources
        elif isinstance(question, UnansweredQuestion):
            raw_question_str = question.question
            if resolved_q_id is None:
                resolved_q_id = question.question_id
            if candidate_sources is None and isinstance(
                question, AnsweredQuestion
            ):
                candidate_sources = question.sources
        else:
            raw_question_str = str(question)

        effective_k = k if (k is not None and k > 0) else None
        raw_source_list = (
            list(candidate_sources[:effective_k])
            if (candidate_sources is not None and effective_k is not None)
            else (
                list(candidate_sources)
                if candidate_sources is not None
                else []
            )
        )
        normalized_sources = normalize_retrieved_sources(
            raw_source_list,
            k=effective_k,
            max_chunk_chars=self.max_chunk_chars,
        )

        clean_q = raw_question_str.strip()
        if not clean_q:
            # Graceful fallback for degenerate empty question inputs (FR-10)
            fallback_kwargs: Dict[str, Any] = {
                "question": "[Empty question]",
                "retrieved_sources": normalized_sources,
                "answer": EMPTY_QUERY_ANSWER,
            }
            if resolved_q_id is not None:
                fallback_kwargs["question_id"] = resolved_q_id
            empty_model = MinimalAnswer(**fallback_kwargs)
            if return_json or as_json:
                return format_answer_json(empty_model, indent=indent)
            return empty_model

        prompt_budget = (
            max_prompt_tokens
            if max_prompt_tokens is not None
            else self.max_prompt_tokens
        )
        grounded_prompt = self.prompt_builder.build_prompt(
            question=clean_q,
            sources=raw_source_list,
            context_snippets=context_snippets,
            k=effective_k if effective_k is not None else self.top_k,
            max_tokens=prompt_budget,
        )

        answer_text = self.generate(
            grounded_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_json=False,
        )

        ans_kwargs: Dict[str, Any] = {
            "question": clean_q,
            "retrieved_sources": normalized_sources,
            "answer": answer_text,
        }
        if resolved_q_id is not None:
            ans_kwargs["question_id"] = resolved_q_id

        minimal_answer = MinimalAnswer(**ans_kwargs)
        if return_json or as_json:
            return format_answer_json(minimal_answer, indent=indent)
        return minimal_answer

    def generate_answer_json(
        self,
        question: Union[str, MinimalSearchResults, UnansweredQuestion] = "",
        retrieved_sources: Optional[Sequence[Any]] = None,
        *,
        sources: Optional[Sequence[Any]] = None,
        chunks: Optional[Sequence[Any]] = None,
        context_snippets: Optional[Sequence[str]] = None,
        question_id: Optional[str] = None,
        k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_prompt_tokens: Optional[int] = None,
        indent: int = 2,
    ) -> str:
        """Generate a grounded answer and return it formatted as a validated JSON string."""
        ans_model = self.generate_answer(
            question=question,
            retrieved_sources=retrieved_sources,
            sources=sources,
            chunks=chunks,
            context_snippets=context_snippets,
            question_id=question_id,
            k=k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_prompt_tokens=max_prompt_tokens,
            return_json=False,
        )
        return format_answer_json(ans_model, indent=indent)

    def answer_dataset(
        self,
        student_search_results: Optional[
            Union[
                StudentSearchResults,
                RagDataset,
                Sequence[
                    Union[
                        MinimalSearchResults,
                        UnansweredQuestion,
                        Mapping[str, Any],
                    ]
                ],
                str,
                Path,
            ]
        ] = None,
        *,
        student_search_results_path: Optional[Union[str, Path]] = None,
        save_directory: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        show_progress: bool = True,
        return_json: bool = False,
        as_json: bool = False,
        indent: int = 2,
    ) -> Union[StudentSearchResultsAndAnswer, str]:
        """Generate grounded answers for an entire dataset of retrieved search results.
        Accepts a :class:`StudentSearchResults` model, a path to a JSON file on disk,
        a :class:`RagDataset`, or a sequence of :class:`MinimalSearchResults`.
        Optionally persists the validated :class:`StudentSearchResultsAndAnswer` JSON
        payload to *save_directory* or *output_path*.
            student_search_results: Input batch search results object or JSON file path.
            student_search_results_path: Optional explicit path to search results JSON file.
            save_directory: Optional directory where output JSON file will be written.
            output_path: Optional explicit file path where output JSON will be written.
            k: Optional top-k value override.
            max_new_tokens: Optional max new tokens override (default: 128).
            temperature: Optional decoding temperature override (default: 0.0).
            show_progress: Whether to display a ``tqdm`` progress bar (default: True).
            return_json: If True, returns the formatted JSON string.
            as_json: Alias for *return_json*.
            indent: JSON indentation spaces (default: 2).
        """
        source_file_path: Optional[Path] = None
        target_input = (
            student_search_results
            if student_search_results is not None
            else student_search_results_path
        )

        if target_input is None:
            raise ValueError(
                "Either student_search_results or student_search_results_path must be provided."
            )

        items: List[
            Union[MinimalSearchResults, UnansweredQuestion, Mapping[str, Any]]
        ] = []
        resolved_k: int = k if (k is not None and k >= 1) else self.top_k

        if isinstance(target_input, (str, Path)):
            candidate_path = Path(target_input)
            if not candidate_path.is_file():
                raise FileNotFoundError(
                    f"Student search results file not found: {candidate_path}"
                )
            source_file_path = candidate_path
            try:
                with open(candidate_path, "r", encoding="utf-8") as f:
                    raw_json = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in '{candidate_path}': {exc}"
                ) from exc

            if isinstance(raw_json, Mapping) and "search_results" in raw_json:
                validated_sr = StudentSearchResults.model_validate(raw_json)
                items = list(validated_sr.search_results)
                if k is None:
                    resolved_k = validated_sr.k
            elif isinstance(raw_json, Mapping) and "rag_questions" in raw_json:
                validated_ds = RagDataset.model_validate(raw_json)
                items = list(validated_ds.rag_questions)
            elif isinstance(raw_json, list):
                items = [
                    (
                        MinimalSearchResults.model_validate(entry)
                        if isinstance(entry, Mapping)
                        and "retrieved_sources" in entry
                        else entry
                    )
                    for entry in raw_json
                ]
            else:
                raise ValueError(
                    f"Unsupported JSON schema in '{candidate_path}'."
                )
        elif isinstance(target_input, StudentSearchResults):
            items = list(target_input.search_results)
            if k is None:
                resolved_k = target_input.k
        elif isinstance(target_input, RagDataset):
            items = list(target_input.rag_questions)
        elif isinstance(target_input, Sequence) and not isinstance(
            target_input, (str, bytes)
        ):
            items = list(target_input)
        else:
            raise TypeError(
                f"Unsupported student_search_results type: {type(target_input).__name__}"
            )

        generated_answers: List[MinimalAnswer] = []
        iterator = tqdm(
            items,
            desc="Generating answers",
            unit="question",
            disable=not show_progress,
        )

        for item in iterator:
            if isinstance(item, MinimalAnswer):
                generated_answers.append(item)
            elif isinstance(item, MinimalSearchResults):
                ans = self.generate_answer(
                    question=item,
                    k=resolved_k,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    return_json=False,
                )
                assert isinstance(ans, MinimalAnswer)
                generated_answers.append(ans)
            elif isinstance(item, UnansweredQuestion):
                srcs = (
                    item.sources if isinstance(item, AnsweredQuestion) else []
                )
                ans = self.generate_answer(
                    question=item.question,
                    retrieved_sources=srcs,
                    question_id=item.question_id,
                    k=resolved_k,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    return_json=False,
                )
                assert isinstance(ans, MinimalAnswer)
                generated_answers.append(ans)
            elif isinstance(item, Mapping):
                q_str = str(item.get("question", ""))
                q_id = item.get("question_id")
                raw_srcs = item.get(
                    "retrieved_sources", item.get("sources", [])
                )
                ans = self.generate_answer(
                    question=q_str,
                    retrieved_sources=(
                        raw_srcs if isinstance(raw_srcs, Sequence) else []
                    ),
                    question_id=str(q_id) if q_id is not None else None,
                    k=resolved_k,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    return_json=False,
                )
                assert isinstance(ans, MinimalAnswer)
                generated_answers.append(ans)
            else:
                raise TypeError(
                    f"Unsupported dataset item type: {type(item).__name__}"
                )

        batch_result = StudentSearchResultsAndAnswer(
            search_results=generated_answers,
            k=max(1, resolved_k),
        )
        formatted_json = format_answer_json(batch_result, indent=indent)

        # Save to disk if output_path or save_directory is specified
        target_out_path: Optional[Path] = None
        if output_path is not None:
            target_out_path = Path(output_path)
        elif save_directory is not None:
            save_dir = Path(save_directory)
            filename = (
                source_file_path.name
                if source_file_path is not None
                else "student_search_results_and_answer.json"
            )
            target_out_path = save_dir / filename

        if target_out_path is not None:
            target_out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_out_path, "w", encoding="utf-8") as f:
                f.write(formatted_json)
            logger.info(
                "Saved StudentSearchResultsAndAnswer (%d answers) to '%s'",
                len(generated_answers),
                target_out_path,
            )

        if return_json or as_json:
            return formatted_json
        return batch_result

    def __call__(
        self,
        question_or_prompt: Union[
            str, MinimalSearchResults, UnansweredQuestion
        ] = "",
        retrieved_sources: Optional[Sequence[Any]] = None,
        *,
        return_json: bool = False,
        as_json: bool = False,
        **kwargs: Any,
    ) -> Union[MinimalAnswer, str]:
        """Callable interface forwarding to :meth:`generate_answer` or :meth:`generate`."""
        if isinstance(
            question_or_prompt, (MinimalSearchResults, UnansweredQuestion)
        ):
            return self.generate_answer(
                question=question_or_prompt,
                retrieved_sources=retrieved_sources,
                return_json=return_json,
                as_json=as_json,
                **kwargs,
            )
        if retrieved_sources is not None:
            return self.generate_answer(
                question=question_or_prompt,
                retrieved_sources=retrieved_sources,
                return_json=return_json,
                as_json=as_json,
                **kwargs,
            )
        return self.generate(
            prompt=question_or_prompt,
            return_json=return_json,
            as_json=as_json,
            **kwargs,
        )


def generate_answer(
    question: Union[str, MinimalSearchResults, UnansweredQuestion],
    retrieved_sources: Optional[Sequence[Any]] = None,
    *,
    question_id: Optional[str] = None,
    k: Optional[int] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    device: str = DEFAULT_DEVICE,
    return_json: bool = False,
    as_json: bool = False,
    indent: int = 2,
    chunks: Optional[Sequence[InternalChunk]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    _model: Optional[Any] = None,
    _tokenizer: Optional[Any] = None,
) -> Union[MinimalAnswer, str]:
    """Convenience function to generate a grounded :class:`MinimalAnswer` (or JSON string)."""
    engine = LLMInferenceEngine(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
        chunks=chunks,
        base_dir=base_dir,
        _model=_model,
        _tokenizer=_tokenizer,
    )
    return engine.generate_answer(
        question=question,
        retrieved_sources=retrieved_sources,
        question_id=question_id,
        k=k,
        return_json=return_json,
        as_json=as_json,
        indent=indent,
    )


def generate_answer_json(
    question: Union[str, MinimalSearchResults, UnansweredQuestion],
    retrieved_sources: Optional[Sequence[Any]] = None,
    *,
    question_id: Optional[str] = None,
    k: Optional[int] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    device: str = DEFAULT_DEVICE,
    indent: int = 2,
    chunks: Optional[Sequence[InternalChunk]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    _model: Optional[Any] = None,
    _tokenizer: Optional[Any] = None,
) -> str:
    """Convenience function to generate a grounded answer formatted as a JSON string."""
    res = generate_answer(
        question=question,
        retrieved_sources=retrieved_sources,
        question_id=question_id,
        k=k,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
        return_json=True,
        indent=indent,
        chunks=chunks,
        base_dir=base_dir,
        _model=_model,
        _tokenizer=_tokenizer,
    )
    return str(res)


def answer_dataset(
    student_search_results: Optional[
        Union[
            StudentSearchResults,
            RagDataset,
            Sequence[
                Union[
                    MinimalSearchResults, UnansweredQuestion, Mapping[str, Any]
                ]
            ],
            str,
            Path,
        ]
    ] = None,
    *,
    student_search_results_path: Optional[Union[str, Path]] = None,
    save_directory: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    k: Optional[int] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    device: str = DEFAULT_DEVICE,
    show_progress: bool = True,
    return_json: bool = False,
    as_json: bool = False,
    indent: int = 2,
    chunks: Optional[Sequence[InternalChunk]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    _model: Optional[Any] = None,
    _tokenizer: Optional[Any] = None,
) -> Union[StudentSearchResultsAndAnswer, str]:
    """Convenience function to generate answers for a batch dataset of search results."""
    engine = LLMInferenceEngine(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
        chunks=chunks,
        base_dir=base_dir,
        _model=_model,
        _tokenizer=_tokenizer,
    )
    return engine.answer_dataset(
        student_search_results=student_search_results,
        student_search_results_path=student_search_results_path,
        save_directory=save_directory,
        output_path=output_path,
        k=k,
        show_progress=show_progress,
        return_json=return_json,
        as_json=as_json,
        indent=indent,
    )
