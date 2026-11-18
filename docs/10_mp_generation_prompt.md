# Mandatory Part: `src/generation/prompt.py`

## 1. Overview & Architectural Role

[`src/generation/prompt.py`](../src/generation/prompt.py) implements **Context Grounding & Prompt Formatting** required by Chapter VI.3:
> *"Pass the retrieved context to the model within its token budget, and produce structured JSON following the provided pydantic models. A satisfactory answer is: Coherent and understandable, Grounded in the retrieved sources, with no major hallucination, On point: it answers the question actually asked."*

`PromptBuilder` bridges the retrieval engine and the local LLM. It extracts the raw text for retrieved character spans, formats explicit source headers, and constrains prompt length within the CPU budget.

---

## 2. Context Hydration: `ContextLoader`

### 2.1 The Need for Hydration
During retrieval, the search engine returns `MinimalSource` objects containing only location coordinates (`file_path`, `first_character_index`, `last_character_index`). The language model needs the actual text content inside those character spans.

### 2.2 Two-Tier Span Resolution
`ContextLoader` extracts the text snippet for each source:
1. **Tier 1 (In-Memory Catalog):** If `chunks.json` is loaded in memory, it queries `_exact_chunk_map` in $O(1)$ without disk I/O.
2. **Tier 2 (Disk File Slicing):** If not found in the catalog:
   - Resolves the safe path using `resolve_safe_path()`.
   - Reads the file using UTF-8 (caching it in `_file_cache`).
   - Slices the exact span: `file_text[first_idx : last_idx]`.

### 2.3 Path Traversal Security (`resolve_safe_path`)
To prevent path traversal exploits (`../../etc/passwd`), `resolve_safe_path` verifies using `os.path.commonpath`:
```python
if os.path.commonpath([str(allowed_root), str(candidate_path)]) != str(allowed_root):
    raise ValueError("Path traversal detected")
```

---

## 3. Explicit Source Header Formatting

The language model must know which source provided which evidence. `format_source_header()` formats each snippet with an explicit header:
```text
[Source 1] file_path: data/raw/vllm-0.10.1/docs/features/lora.md | character_span: 4695-6098 (chars 4695:6098)
The `/v1/load_lora_adapter` endpoint is used to dynamically load a LoRA adapter in vLLM...
```

---

## 4. `PromptBuilder` & Token Budget Management

### 4.1 System Prompt Formulation
Instructs `Qwen/Qwen3-0.6B` to stay grounded and avoid hallucination:
```text
You are a technical code-assistant answering questions about the vLLM codebase.
Answer the user's question using ONLY the provided context sources.
Be direct, concise, and factual. Do not speculate or invent information.
If the answer cannot be found in the sources, state clearly that the information is not available.
```

### 4.2 Token Budget Truncation (`truncate_text_to_token_budget`)
On CPU, generating tokens with an LLM is latency-critical. If the context is too large, inference slows down dramatically.
`PromptBuilder` enforces `max_prompt_tokens` (default: 2048–4096 tokens):
- Evaluates token counts using the model's tokenizer.
- If total sources exceed the budget, it progressively includes the highest-ranked sources first and truncates lower-ranked sources cleanly along sentence boundaries.
- Formats messages using `tokenizer.apply_chat_template` (ChatML format for Qwen models).
