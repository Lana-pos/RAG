# Mandatory Part: `src/indexing/schema.py`

## 1. Overview & Architectural Role

[`src/indexing/schema.py`](../src/indexing/schema.py) defines the **internal chunking schema** used during the ingestion, indexing, and storage phases.

While [`src/models.py`](../src/models.py) defines the *external interchange models* evaluated by the moulinette, `InternalChunk` stores the *internal payload* (including raw text content, token counts, sequential IDs, and syntactic node types) saved to `data/processed/chunks.json`.

---

## 2. Model Definitions

### 2.1 `ChunkType`
```python
ChunkType = Literal[
    "python_class",
    "python_func",
    "python_module",
    "markdown_section",
]
```
A type alias representing the syntactic origin of the chunk:
- `"python_class"`: An entire Python class or sub-divided class block extracted via AST.
- `"python_func"`: A standalone synchronous or asynchronous function.
- `"python_module"`: Module-level docstrings, imports, or statements.
- `"markdown_section"`: A Markdown section delimited by ATX headings (`#`, `##`, etc.).

### 2.2 `InternalChunk`
```python
class InternalChunk(BaseModel):
    chunk_id: int
    file_path: str
    first_character_index: int
    last_character_index: int
    chunk_type: ChunkType
    token_count: int
    content: str
```

#### Fields and Properties
1. `chunk_id: int`: 0-based sequential integer identifier (`ge=0`), assigned contiguously across the entire corpus (`0` to `N-1`).
2. `file_path: str`: Normalized relative file path (validated via `normalize_file_path`).
3. `first_character_index: int`: Byte/character offset of the start of the chunk in the original file on disk.
4. `last_character_index: int`: Byte/character offset of the end of the chunk in the file (`last > first`).
5. `chunk_type: ChunkType`: Syntactic category.
6. `token_count: int`: Estimated token count in the text snippet.
7. `content: str`: The raw text content of the snippet.

#### Methods
- `span_length -> int`: Computes `last_character_index - first_character_index`.
- `to_source() -> MinimalSource`: Converts the internal chunk into an external `MinimalSource(file_path=..., first_character_index=..., last_character_index=...)`, discarding internal content to match the moulinette schema.
