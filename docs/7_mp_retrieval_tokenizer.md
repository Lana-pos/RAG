# Mandatory Part: `src/retrieval/tokenizer.py`

## 1. Overview & Architectural Role

[`src/retrieval/tokenizer.py`](../src/retrieval/tokenizer.py) implements the **Code-Aware Sub-Word Tokenizer** required for lexical BM25 retrieval.

### The Vocabulary Mismatch Problem in Code Search
In codebases, identifiers are formatted with compound naming conventions:
- `camelCase` and `PascalCase`: e.g. `AsyncEngineArgs`, `ModelRunner`, `VLLMContext`
- `snake_case`: e.g. `openai_compatible_server`, `max_model_len`, `fused_moe`
- Dotted attribute paths: e.g. `torch.nn.Linear`, `vllm.engine.llm_engine`

Standard English natural-language tokenizers fail on code:
- If a query asks: *"What arguments does the async engine take?"*, a naive tokenizer searching for `"async"` or `"engine"` will **fail** to match the identifier `AsyncEngineArgs` because the characters are joined together.
- Conversely, if the tokenizer only splits words, a search for `"AsyncEngineArgs"` will not match the exact composite identifier.

---

## 2. Implementation: `tokenize_code`

### 2.1 Pattern Matching
1. **Identifier Matching:**
   ```python
   _LEXICAL_TOKEN_PATTERN = re.compile(r"\w+(?:[._]\w+)*", re.UNICODE)
   ```
   Matches words including internal underscores (`_`) and dots (`.`), while discarding markdown punctuation (`#`, `*`, `` ` ``, `[`, `]`).

2. **CamelCase Boundary Splitting:**
   ```python
   _CAMEL_CASE_BOUNDARY_PATTERN = re.compile(
       r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
   )
   ```
   Splits lower-to-upper and acronym-to-word boundaries:
   - `AsyncEngine` $\rightarrow$ `["Async", "Engine"]`
   - `HTTPServer` $\rightarrow$ `["HTTP", "Server"]`

### 2.2 Dual Token Emission: Sub-words AND Composite Symbols
For every identifier encountered, `tokenize_code`:
1. Splits on dots (`.`) to handle module paths.
2. Splits components on underscores (`_`).
3. Splits components on camelCase boundaries.
4. Emits all lowercase sub-words **plus** the composite term.

#### Example:
```python
tokenize_code("AsyncEngineArgs")
# Emits: ['async', 'engine', 'args', 'asyncengineargs']

tokenize_code("openai_compatible_server")
# Emits: ['openai', 'compatible', 'server', 'openai_compatible_server']

tokenize_code("torch.nn.Linear")
# Emits: ['torch', 'nn', 'linear', 'torch.nn.linear']
```

### 2.3 Preserving Programming Keywords
Common programming keywords (`async`, `await`, `class`, `def`, `return`, `import`) are treated as valid search terms and are not stripped as generic English stopwords.
