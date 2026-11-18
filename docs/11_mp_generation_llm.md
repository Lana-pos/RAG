# Mandatory Part: `src/generation/llm.py`

## 1. Overview & Architectural Role

[`src/generation/llm.py`](../src/generation/llm.py) implements the **Local CPU Inference Engine** using the mandatory model:
> **`Qwen/Qwen3-0.6B`**

Per Chapter V.4 and VI.3 of the subject:
- You must use `Qwen/Qwen3-0.6B` as the default model.
- Execution runs entirely locally on CPU without external API keys or cloud dependencies.
- Output files must be structured JSON conforming 100% to Pydantic models (`MinimalAnswer` and `StudentSearchResultsAndAnswer`).
- The RAM footprint must remain low (comfortably under campus machine limits).

---

## 2. Model Loading & Memory Optimization

### 2.1 Lazy Model Loading
The neural network weights (~1.2 GB) are loaded only when an `answer` or `answer_dataset` command is actually executed. Simple indexing and retrieval operations do not load the model, keeping CLI commands instantaneous.

### 2.2 Low-Memory CPU Footprint
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
)
```
- `low_cpu_mem_usage=True` loads weights directly into model layers without duplicating memory buffers.
- Total resident memory stays strictly under **1.5 GB RAM**.

---

## 3. Deterministic Inference Strategy

LLM answers must be deterministic and factual:
1. **Greedy Decoding:**
   - `temperature = 0.0`
   - `do_sample = False`
   This eliminates stochastic randomness and guarantees reproducible answers.
2. **Token Generation Ceiling:**
   - `max_new_tokens = 128`
   Concise answers prevent CPU timeouts while providing sufficient length for technical explanations.
3. **Inference Context:**
   - Executed under `with torch.inference_mode():` to disable autograd tracking and minimize memory overhead.

---

## 4. Single-Query & Batch Answer Generation

### 4.1 Single Query: `LLMInferenceEngine.answer`
```python
def answer(
    self,
    question: str,
    sources: Sequence[MinimalSource],
    k: int = 5,
    question_id: Optional[str] = None,
) -> MinimalAnswer:
```
1. Feeds question and top-$k$ sources to `PromptBuilder`.
2. Tokenizes the prompt and runs greedy decoding.
3. Strips special prompt tokens (`<|im_end|>`, `<|endoftext|>`).
4. Wraps result in a validated `MinimalAnswer(question_id=..., question=..., retrieved_sources=..., answer=...)`.

### 4.2 Batch Dataset: `LLMInferenceEngine.answer_dataset`
```python
def answer_dataset(
    self,
    student_search_results_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    ...
) -> StudentSearchResultsAndAnswer:
```
1. Loads the `StudentSearchResults` JSON file produced by `search_dataset`.
2. Iterates over all questions with a `tqdm` progress bar (`"Answering dataset"`).
3. Hydrates sources and generates an answer for each question.
4. Serializes the completed `StudentSearchResultsAndAnswer` batch model to disk.
