# Mandatory Part: `src/indexing/indexer.py`

## 1. Overview & Architectural Role

[`src/indexing/indexer.py`](../src/indexing/indexer.py) implements the **Ingestion Coordinator & Corpus Indexer**.

It orchestrates the entire ingestion pipeline:
1. Recursive file discovery over `data/raw/vllm-0.10.1/`.
2. Syntax-aware AST Python and heading-aware Markdown chunking.
3. Hard chunk ceiling enforcement ($\le 2000$ characters).
4. Lexical tokenization, vocabulary building, and BM25 inverted index construction.
5. Persistence of index artifacts to `data/processed/`.

Per the subject SLA:
> *"Indexing the whole corpus must take at most 5 minutes."*
> The implemented indexer indexes the entire `vLLM` corpus (~1,927 files, ~28,385 chunks) in **~8.5 seconds** on CPU.

---

## 2. Ingestion Pipeline Workflow

The coordinator executes via `CorpusIndexer.index()`:

```
[data/raw/vllm-0.10.1/]
         │
         ▼
[File Discovery: discover_files] (filters binary and hidden files)
         │
         ▼
[Dual Chunking Engine]
   ├─► Python files (.py)    ──► PythonChunker (AST parsing)
   └─► Markdown files (.md)  ──► MarkdownChunker (ATX headings)
         │
         ▼
[Hard Ceiling Guard: enforce_max_chunk_size] (guarantees <= 2000 chars)
         │
         ▼
[Save chunks.json] ──► data/processed/chunks.json
         │
         ▼
[Lexical Tokenization with File-Path & Header Boosting]
         │
         ▼
[Fit & Save BM25 Inverted Index] ──► data/processed/bm25_index.pkl
```

---

## 3. High-Recall Optimization Strategies

To achieve **$\ge 80\%$ Recall@5 on docs** and **$\ge 50\%$ Recall@5 on code** without deep semantic re-ranking:
1. **File-Path Token Enrichment:**
   When searching for *"How do I configure the OpenAI server?"*, the documentation file is located at `docs/serving/openai_compatible_server.md`. `CorpusIndexer` enriches each chunk's lexical tokens with its relative path tokens (`"serving"`, `"openai"`, `"compatible"`, `"server"`), ensuring that even short subsections match questions referencing the broader module name.
2. **Top-Level Section Title Enrichment:**
   Sub-chunks inherit top-level document headings from their parent document so individual paragraphs retain the topic context.
3. **Conversational Query Stopword Filtering:**
   Conversational words like `"what"`, `"how"`, `"where"`, `"can"`, `"you"` appear rarely in raw Python source code, which causes standard BM25 to assign them artificially high IDF weights. `CorpusIndexer` filters these stopwords so scoring prioritizes the technical domain keywords.

---

## 4. Key Functions and Classes

- `CorpusIndexer`: Main coordinator class.
  - `chunk_corpus(corpus_dir, max_chunk_size, show_progress)`: Discovers and chunks all files.
  - `build_bm25(chunks, save_path, show_progress)`: Tokenizes chunks with `tqdm` and builds `BM25Index`.
  - `save_chunks(chunks, save_path)`: Writes formatted `chunks.json`.
  - `index(corpus_dir, output_dir, max_chunk_size)`: Full pipeline runner returning `IndexingResult`.
- `load_chunks(file_path)`: Reads and deserializes `chunks.json` into a list of `InternalChunk` objects.
- `CorpusIndexer.load(output_dir)`: Loads `chunks.json` and `bm25_index.pkl` back into memory.
