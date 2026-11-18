# Technical Architecture Document: RAG Against the Machine (v2.0)

**Document Version:** 1.0.0  
**Author:** Senior Solution Architect  
**Project:** RAG Against the Machine — Codebase Question Answering System  
**Reference Corpus:** `vLLM v0.10.1`  
**Reference Model:** `Qwen/Qwen3-0.6B`  
**Evaluation Standard:** 42 Moulinette Recall@5 Grader (Docs $\ge$ 80%, Code $\ge$ 50%)  

---

## 1. Executive Summary

The **RAG Against the Machine** project delivers an enterprise-grade, deterministic, and privacy-preserving Retrieval-Augmented Generation (RAG) system engineered to answer complex developer and documentation inquiries targeting large-scale software repositories—specifically the `vLLM` (v0.10.1) codebase.

Large Language Models (LLMs) suffer from static pre-training cutoffs and hallucination tendencies when asked about proprietary codebases, niche APIs, or fast-evolving framework features. Rather than fine-tuning or retraining weights, this system implements a high-throughput, dual-stage retrieval and generation pipeline. It indexes heterogeneous source files (Python source code and Markdown documentation), retrieves character-precise evidence spans within a strict 2000-character ceiling, and augments a local, low-footprint language model (`Qwen/Qwen3-0.6B`) to generate accurate, verifiable natural language answers serialized as strictly validated JSON.

The architecture is designed to satisfy rigorous performance and quality Service Level Agreements (SLAs) on resource-constrained, CPU-only campus evaluation hardware:
- **Indexing Throughput:** Indexing $\sim$2,000 source files ($\sim$14,000 chunks) in $\le$ 5 minutes.
- **Retrieval Quality:** Achieving $\ge$ 80% Recall@5 on documentation queries and $\ge$ 50% Recall@5 on code queries under Intersection-over-Union ($\text{IoU} \ge 0.05$) matching.
- **Query Throughput:** Processing 200 batch evaluation queries in $\le$ 90 seconds ($\le$ 450 ms/query).
- **Execution Constraints:** 100% reproducible execution via `uv sync`, zero external system daemons, complete adherence to PEP 8/flake8 and strict MyPy typing.

The system natively incorporates the mandatory lexical baseline (BM25 with language-aware tokenization) alongside a modular extension suite implementing all five advanced capabilities: semantic vector search (`all-MiniLM-L6-v2`), Reciprocal Rank Fusion (RRF) hybrid retrieval, incremental SHA-256 indexing, multi-tier LRU/disk caching, and a local asynchronous FastAPI HTTP service.

---

## 2. Project Understanding and Scope

### 2.1 Purpose and Business Objectives
The primary purpose is to provide software engineers, code reviewers, and automated evaluation harnesses with immediate, grounded answers to technical questions about the vLLM project. The business objectives are:
1. **Developer Productivity:** Drastically cut the time required to locate implementations, configuration parameters, and architectural patterns within multi-thousand-file repositories.
2. **Factual Grounding:** Eliminate LLM hallucinations by binding every generated statement directly to verifiable file paths and exact character spans.
3. **Automated Verification:** Enable continuous integration through machine-verifiable evaluation pipelines scored by the official automated grading harness (`moulinette`).

### 2.2 System Boundaries
```
+---------------------------------------------------------------------------------------+
| SYSTEM BOUNDARY: RAG Against the Machine                                              |
|                                                                                       |
|  [Raw Data Ingestion]       [Indexing Engine]           [Retrieval & Augmentation]    |
|   - vLLM Codebase            - AST Chunker (Python)      - Lexical Index (BM25)       |
|   - Public Question sets     - Header Chunker (Markdown) - Semantic Index (MiniLM)    |
|                              - Code Tokenizer            - Hybrid RRF Ranker          |
|                                                                                       |
|  [Inference Engine]         [Serving Interfaces]         [Persistence & Cache]        |
|   - Qwen3-0.6B Loader        - Python Fire CLI           - Inverted Index (Binary)    |
|   - Grounded Prompt Builder  - FastAPI HTTP Service      - Vector Store (NumPy)       |
|   - Pydantic Serializer      - Automated Evaluator       - LRU & Disk Result Cache    |
+---------------------------------------------------------------------------------------+
        |                                                                 ^
        v                                                                 |
 [External Grader: Moulinette]                                     [Local Filesystem]
```

### 2.3 Scope Inclusions and Exclusions
- **In Scope:**
  - Ingestion and syntax-aware chunking of Python (`.py`) and Markdown (`.md`) files from `data/raw/vllm-0.10.1/`.
  - BM25Okapi lexical retrieval with sub-word/camelCase/snake_case tokenization.
  - Sentence-Transformers semantic dense retrieval using `all-MiniLM-L6-v2` (Bonus 1).
  - Reciprocal Rank Fusion (RRF) hybrid ranking (Bonus 2).
  - Incremental indexing via file hash and mtime caching (Bonus 3).
  - Multi-tier query and index caching (Bonus 4).
  - High-performance local HTTP API via FastAPI (Bonus 5).
  - Local CPU inference of `Qwen/Qwen3-0.6B` using Hugging Face Transformers.
  - Strict Pydantic v2 data interchange validation matching moulinette requirements.
  - Full CLI suite via Python Fire with `tqdm` progress monitoring.
- **Out of Scope:**
  - Distributed cluster coordination (e.g., Ray, Spark).
  - Cloud-managed vector databases (e.g., Pinecone, Milvus) requiring external networks or API keys.
  - Fine-tuning or retraining model weights.
  - Web UI frontend (CLI and headless HTTP API satisfy all client requirements).

---

## 3. Functional and Non-Functional Requirements

### 3.1 Functional Requirements (FR)
- **FR-01: Corpus Ingestion:** Recursively parse all valid source files from `data/raw/vllm-0.10.1/` while ignoring binaries, build artifacts, git objects, and virtual environments.
- **FR-02: Syntax-Aware Chunking:**
  - Implement an AST-based Python chunker segmenting classes, methods, and standalone functions.
  - Implement a Markdown chunker segmenting text along heading boundaries (`#`, `##`, `###`).
  - Enforce a strict hard limit: no chunk may exceed `--max_chunk_size` (default: 2000 characters).
- **FR-03: Character Span Tracking:** Record exact `file_path`, `first_character_index`, and `last_character_index` for every generated chunk.
- **FR-04: Lexical Retrieval:** Provide BM25Okapi scoring over the tokenized corpus, returning the top-$k$ candidate snippets ranked by relevance.
- **FR-05: Semantic Retrieval (Bonus):** Generate 384-dimensional dense embeddings for chunks and queries using `all-MiniLM-L6-v2` and perform cosine similarity search.
- **FR-06: Hybrid Fusion (Bonus):** Combine lexical and semantic ranks using Reciprocal Rank Fusion (RRF) with constant $k=60$.
- **FR-07: Answer Generation:** Inject top-$k$ retrieved contexts into an instruction prompt and execute `Qwen/Qwen3-0.6B` to generate an answer.
- **FR-08: Data Validation:** Validate all inputs, stage boundaries, and output payloads with Pydantic v2 schemas (`StudentSearchResults`, `StudentSearchResultsAndAnswer`).
- **FR-09: CLI Commands:** Expose `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, and `evaluate` via Python Fire.
- **FR-10: Robustness:** Degenerate inputs (empty string queries, $k=0$, missing directories, corrupt JSON) must yield structured error messages without crashing.

### 3.2 Non-Functional Requirements (NFR)
- **NFR-01: Indexing SLA:** Full ingestion and indexing of the target repository must finish within $\le 5\text{ minutes}$ on a standard 4-core CPU campus machine.
- **NFR-02: Retrieval Latency SLA:** Batch retrieval over 200 questions must execute within $\le 90\text{ seconds}$ ($\le 450\text{ ms}$ per query).
- **NFR-03: Retrieval Quality SLA:** Must achieve $\ge 80\%$ Recall@5 on documentation questions and $\ge 50\%$ Recall@5 on code questions when evaluated by the moulinette.
- **NFR-04: Portability & Dependencies:** Installation must rely strictly on `uv sync`. No external C/C++ compilation tools (e.g., raw CMake) or running database daemons (e.g., PostgreSQL, Elasticsearch) permitted.
- **NFR-05: Memory Footprint:** Total working memory during CPU batch inference and indexing must remain within $4.0\text{ GB}$ RAM to prevent OOM termination.
- **NFR-06: Code Quality Standards:** 100% compliance with `flake8` and strict `mypy` type checking (`--warn-return-any`, `--disallow-untyped-defs`, `--check-untyped-defs`).

---

## 4. Assumptions and Constraints

### 4.1 Confirmed Facts
1. The target corpus is `vLLM v0.10.1`, containing approximately 1,965 files and yielding roughly 13,000 to 15,000 text chunks.
2. The evaluation grader (`moulinette`) evaluates Recall@k based on an Intersection-over-Union ($\text{IoU} \ge 0.05$) between student retrieved character ranges and reference ranges.
3. The grader compares `file_path` verbatim as relative paths from the project root (e.g., `data/raw/vllm-0.10.1/docs/...`). Any path variation (e.g., leading `./` or absolute path) immediately fails matching.
4. Any single chunk exceeding 2,000 characters immediately causes the moulinette to mark the entire dataset submission invalid.
5. Campus machines are standard Linux distributions (Ubuntu/Fedora) operating without dedicated GPU hardware during evaluation.

### 4.2 Assumptions Requiring Management
1. **Model Weights Caching:** `Qwen/Qwen3-0.6B` and `all-MiniLM-L6-v2` model weights will be cached in local Hugging Face directories (`~/.cache/huggingface/hub`) or local offline storage during grading.
2. **Corpus Immutability during Single Runs:** The corpus does not change concurrently while a query or batch search is running.
3. **Token Budget:** The prompt template and retrieved context chunks fit comfortably within Qwen3's context window (standard 32k tokens, though constrained to 2k-4k tokens for sub-second CPU response time).

### 4.3 Technical Constraints
1. **Package Management:** `uv` is the sole authorized project manager; reviewers execute only `uv sync`.
2. **CLI Engine:** Python Fire is mandated.
3. **Progress Indicators:** `tqdm` is required for operations lasting longer than 1 second.
4. **Makefile Interface:** Must support `install`, `run`, `debug`, `clean`, `lint`.

---

## 5. Architecture Principles

1. **Deterministic Reproducibility:** Every stage of the pipeline must produce identical outputs given identical inputs and configurations.
2. **Separation of Concerns (Hexagonal / Clean Architecture):** Domain data models (`pydantic`) are decoupled from storage engines (BM25, NumPy vector store), inference runtimes (PyTorch, Transformers), and presentation adapters (CLI, HTTP API).
3. **Fail-Fast Boundary Validation:** All data entering or leaving a component is strictly validated against schemas before execution proceeds.
4. **Low-Allocation In-Memory Serving:** Pre-computed sparse inverted indices and dense embedding arrays are memory-mapped or deserialized into contiguous NumPy structures to guarantee sub-millisecond retrieval.
5. **Progressive Degradation & Modularity:** If semantic embedding models are unavailable or uninitialized, the system gracefully falls back to pure lexical retrieval without system crashes.

---

## 6. High-Level System Architecture

The system operates across three decoupled lifecycles:
1. **Ingestion & Indexing Pipeline (Offline/Batch):** Scans the corpus, executes AST/Markdown chunking, tokenizes text, and compiles both the BM25 inverted index and dense semantic vector embeddings.
2. **Retrieval & Fusion Pipeline (Online/Query):** Ingests single or batch queries, queries the BM25 and vector stores in parallel, merges candidate scores using Reciprocal Rank Fusion, applies character-ceiling guards, and outputs top-$k$ `MinimalSource` records.
3. **Context Augmentation & Generation Pipeline (Online/Inference):** Hydrates chunk content from disk, constructs a grounded system prompt, feeds tokens to `Qwen3-0.6B`, parses the generated completion, and packages the final `StudentSearchResultsAndAnswer` payload.

---

## 7. Architecture Diagrams

### 7.1 System Context Diagram (C4 Level 1)
Shows the interaction between the user, the CLI/API, external dependencies, and the evaluation moulinette.

```mermaid
flowchart TD
    User["Developer / Peer Evaluator"]
    Moulinette["Moulinette Automated Grader"]
    HFHub["Hugging Face Model Hub (Offline / Cache)"]

    subgraph SystemBoundary["RAG Against The Machine System"]
        RAGSystem["RAG Application Core\n(CLI & HTTP Services)"]
    end

    FileSystem[("Local Storage\ndata/raw/ | data/processed/")]

    User -->|"CLI Commands (Fire)\nHTTP Requests (FastAPI)"| RAGSystem
    Moulinette -->|"Scores Student Results\n(Recall@k calculation)"| FileSystem
    RAGSystem -->|"Writes search_results.json\nand answers.json"| FileSystem
    RAGSystem -->|"Reads raw corpus &\npersists indices"| FileSystem
    HFHub -.->|"Loads weights at startup\n(Qwen3-0.6B & MiniLM)"| RAGSystem
```

---

### 7.2 Container Diagram (C4 Level 2)
Illustrates the internal containers, storage mechanisms, and interfaces of the application.

```mermaid
flowchart TD
    subgraph ClientInterfaces["Presentation & Entrypoint Layer"]
        CLI["CLI Entrypoint\n(Python Fire / src/__main__.py)"]
        HTTPAPI["HTTP API Server\n(FastAPI / Uvicorn)"]
    end

    subgraph ApplicationCore["Core Application Orchestrator"]
        PipelineManager["Pipeline Orchestrator\n(src/pipeline.py)"]
        CacheManager["Cache Subsystem\n(L1 Memory LRU / L2 Disk)"]
    end

    subgraph IngestionSubsystem["Ingestion & Chunking Layer"]
        ASTChunker["Python AST Chunker\n(src/chunking/python.py)"]
        MDChunker["Markdown Header Chunker\n(src/chunking/markdown.py)"]
        TokenEngine["Code Tokenizer\n(Regex/Camel/Snake/Stopwords)"]
    end

    subgraph RetrievalSubsystem["Retrieval & Ranking Layer"]
        BM25Engine["BM25Okapi Lexical Search\n(src/retrieval/bm25.py)"]
        DenseEngine["Dense Semantic Vector Search\n(src/retrieval/dense.py)"]
        RRFEngine["Hybrid RRF Ranker\n(src/retrieval/hybrid.py)"]
    end

    subgraph GenerationSubsystem["Inference & Generation Layer"]
        PromptEngine["Prompt Augmentation Builder\n(src/generation/prompt.py)"]
        LLMInference["Qwen3-0.6B Inference Engine\n(HuggingFace Transformers / PyTorch)"]
    end

    subgraph StorageLayer["Data & Persistence Store"]
        RawCorpus[("Raw vLLM Corpus\ndata/raw/vllm-0.10.1")]
        BM25Store[("BM25 Inverted Index\ndata/processed/bm25.pkl")]
        VectorStore[("Dense Vectors\ndata/processed/vectors.npy")]
        ManifestStore[("File Manifest\ndata/processed/manifest.json")]
    end

    CLI --> PipelineManager
    HTTPAPI --> PipelineManager

    PipelineManager --> CacheManager
    PipelineManager --> IngestionSubsystem
    PipelineManager --> RetrievalSubsystem
    PipelineManager --> GenerationSubsystem

    IngestionSubsystem --> RawCorpus
    IngestionSubsystem --> BM25Store
    IngestionSubsystem --> VectorStore
    IngestionSubsystem --> ManifestStore

    RetrievalSubsystem --> BM25Store
    RetrievalSubsystem --> VectorStore

    GenerationSubsystem --> LLMInference
```

---

### 7.3 Component Diagram: Retrieval & Generation Engine (C4 Level 3)
Details the interactions between the retrieval, ranking, and generation components.

```mermaid
flowchart LR
    Query["Incoming Query"] --> QueryTokenizer["Code-Aware Query Tokenizer"]
    QueryTokenizer --> CleanQuery["Normalized Query Tokens"]

    CleanQuery --> BM25Search["BM25 Inverted Index Matcher"]
    Query --> MiniLMEncoder["MiniLM Text Encoder"]
    QueryEncoder --> QueryVector["Query Embedding (384d)"]
    QueryVector --> VectorSearch["Cosine Matrix Product"]

    BM25Search --> TopBM25["Top-N Lexical Ranks"]
    VectorSearch --> TopDense["Top-N Semantic Ranks"]

    TopBM25 --> RRFCombiner["Reciprocal Rank Fusion\nScore = Sum(1 / (k + rank))"]
    TopDense --> RRFCombiner

    RRFCombiner --> LengthGuard["Length & Bound Filter\n(Max 2000 chars guard)"]
    LengthGuard --> TopK["Top-k MinimalSource List"]

    TopK --> ContextLoader["Raw File Span Loader"]
    ContextLoader --> GroundedContext["Grounded Context Text"]
    GroundedContext --> PromptBuilder["Structured Prompt Formatter"]
    PromptBuilder --> QwenLLM["Qwen3-0.6B CPU Engine"]
    QwenLLM --> MinimalAnswerModel["MinimalAnswer (JSON)"]
```

---

### 7.4 Sequence Diagram: Ingestion and Indexing Workflow
Details the exact execution order during the `uv run python -m src index` invocation.

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Developer / CLI
    participant CLI as Fire CLI Entrypoint
    participant Ingest as Ingestion Manager
    participant Chunker as AST / Markdown Chunkers
    participant Lexical as BM25 Index Builder
    participant Semantic as MiniLM Embedder
    participant Storage as Local Filesystem

    Dev->>CLI: index --max_chunk_size 2000
    CLI->>Ingest: build_index(corpus_path, max_chunk_size)
    Ingest->>Storage: Scan data/raw/vllm-0.10.1/
    Storage-->>Ingest: Return list of 1,965 file paths

    loop For each file (with tqdm)
        alt Python file (.py)
            Ingest->>Chunker: chunk_python(file_path, content, max_size)
            Chunker-->>Ingest: List[SourceChunk] (AST boundaries)
        else Markdown file (.md)
            Ingest->>Chunker: chunk_markdown(file_path, content, max_size)
            Chunker-->>Ingest: List[SourceChunk] (Header boundaries)
        end
    end

    Ingest->>Lexical: build_inverted_index(chunks)
    Lexical->>Lexical: Tokenize code/docs, compute DF, TF, doc_len
    Lexical-->>Ingest: BM25Index Object

    Ingest->>Semantic: encode_chunks(chunks)
    Semantic->>Semantic: Batch inference MiniLM on CPU
    Semantic-->>Ingest: Normalized dense matrix (N x 384)

    Ingest->>Storage: Serialize BM25 to data/processed/bm25.pkl
    Ingest->>Storage: Save vectors to data/processed/vectors.npy
    Ingest->>Storage: Save chunk catalog to data/processed/chunks.json
    Ingest->>Storage: Save hashes to data/processed/manifest.json
    Storage-->>Ingest: Success
    Ingest-->>CLI: Ingestion complete! Indexed N chunks.
    CLI-->>Dev: Execution summary with elapsed time
```

---

### 7.5 Sequence Diagram: Batch Dataset Search & Answer Generation
Demonstrates the data flow during batch evaluation (`search_dataset` and `answer_dataset`).

```mermaid
sequenceDiagram
    autonumber
    actor Evaluator as Examiner / Moulinette
    participant CLI as Fire CLI Entrypoint
    participant Cache as Cache Subsystem
    participant Hybrid as Hybrid Search Engine
    participant Generator as Answer Generator
    participant Storage as File Storage

    Evaluator->>CLI: search_dataset --dataset_path ... --k 10 --save_directory ...
    CLI->>Storage: Read dataset JSON (UnansweredQuestions)
    Storage-->>CLI: Return RagDataset
    CLI->>CLI: Initialize tqdm progress bar

    loop For each question in dataset
        CLI->>Cache: check_cache(question_id, query, k)
        alt Cache Hit
            Cache-->>CLI: Cached List[MinimalSource]
        else Cache Miss
            CLI->>Hybrid: search(query, k=10)
            Hybrid-->>CLI: List[MinimalSource]
            CLI->>Cache: store_cache(question_id, query, k, results)
        end
    end

    CLI->>Storage: Write StudentSearchResults JSON
    Storage-->>CLI: Output confirmed
    CLI-->>Evaluator: Saved student_search_results to data/output/...

    Evaluator->>CLI: answer_dataset --student_search_results_path ...
    CLI->>Storage: Load StudentSearchResults
    Storage-->>CLI: Return search results object

    loop For each item in search_results
        CLI->>Generator: generate_answer(question, retrieved_sources)
        Generator->>Storage: Read exact character spans from data/raw/
        Storage-->>Generator: Raw string contexts
        Generator->>Generator: Render Prompt + Qwen3-0.6B CPU Inference
        Generator-->>CLI: MinimalAnswer
    end

    CLI->>Storage: Write StudentSearchResultsAndAnswer JSON
    Storage-->>CLI: Output confirmed
    CLI-->>Evaluator: Answer generation complete
```

---

### 7.6 Data Flow Diagram
Maps the transformation of information from disk bytes to grounded answer objects.

```mermaid
flowchart TD
    RawFiles["Raw Source Code & Docs\n(data/raw/vllm-0.10.1/)"] --> ASTSplitter["Syntax Splitter\n(AST & Heading Parser)"]
    ASTSplitter --> ChunkCatalog["Source Chunks Catalog\n(chunk_id, file_path, span, text)"]

    ChunkCatalog --> LexicalBuilder["Lexical Pipeline\n(Tokenizer -> Inverted Index)"]
    ChunkCatalog --> DenseBuilder["Dense Pipeline\n(MiniLM -> 384d Float32 Arrays)"]

    LexicalBuilder --> ProcessedStorage[("data/processed/\nbm25.pkl | vectors.npy | chunks.json")]
    DenseBuilder --> ProcessedStorage

    UserQuery["Query / Dataset Question"] --> RuntimeMatcher["Runtime Retriever"]
    ProcessedStorage --> RuntimeMatcher

    RuntimeMatcher --> RankedIndices["Ranked Chunk IDs"]
    RankedIndices --> SpanExtractor["Span Verifier\n(Bounds validation & Length <= 2000)"]
    SpanExtractor --> SearchResultsJSON["StudentSearchResults JSON\n(question_id, retrieved_sources)"]

    SearchResultsJSON --> ContextAugmenter["Context Aggregator\n(Snippet extraction)"]
    RawFiles --> ContextAugmenter
    ContextAugmenter --> PromptContext["LLM Context Window"]
    PromptContext --> QwenModel["Qwen/Qwen3-0.6B Engine"]
    QwenModel --> AnswersJSON["StudentSearchResultsAndAnswer JSON\n(sources + answers)"]
```

---

### 7.7 Deployment Diagram: Campus Host Environment
Shows the deployment topology on a target standard 42 campus machine.

```mermaid
flowchart TD
    subgraph HostMachine["Linux Workstation (Ubuntu / Fedora)"]
        subgraph OSProcesses["Operating System & CPU Execution (No GPU)"]
            PythonRuntime["Python 3.10+ Virtual Environment (.venv)"]

            subgraph RuntimeComponents["Active Modules"]
                FireCLI["CLI Command (uv run python -m src ...)"]
                FastAPIServer["FastAPI / Uvicorn Server (Port 8000)"]
                PyTorchCPU["PyTorch Engine (CPU Threads = 4)"]
            end

            PythonRuntime --> FireCLI
            PythonRuntime --> FastAPIServer
            PythonRuntime --> PyTorchCPU
        end

        subgraph StorageHierarchy["Local Disk File System"]
            CodeRoot["Project Root (/home/.../RAG)"]
            DataRaw["data/raw/vllm-0.10.1/ (Read-Only)"]
            DataProcessed["data/processed/ (Index & Manifest)"]
            DataOutput["data/output/ (Evaluation JSONs)"]
            HFCache["~/.cache/huggingface/hub/ (Weights)"]
        end

        FireCLI --> CodeRoot
        FireCLI --> DataRaw
        FireCLI --> DataProcessed
        FireCLI --> DataOutput
        PyTorchCPU --> HFCache
    end
```

---

### 7.8 Entity-Relationship / Domain Model Diagram
Defines the schema relationships between the internal domain models and the Pydantic interchange contracts.

```mermaid
classDiagram
    class MinimalSource {
        +str file_path
        +int first_character_index
        +int last_character_index
    }

    class UnansweredQuestion {
        +str question_id
        +str question
    }

    class AnsweredQuestion {
        +List~MinimalSource~ sources
        +str answer
    }

    class RagDataset {
        +List~Union[AnsweredQuestion, UnansweredQuestion]~ rag_questions
    }

    class MinimalSearchResults {
        +str question_id
        +str question
        +List~MinimalSource~ retrieved_sources
    }

    class MinimalAnswer {
        +str answer
    }

    class StudentSearchResults {
        +List~MinimalSearchResults~ search_results
        +int k
    }

    class StudentSearchResultsAndAnswer {
        +List~MinimalAnswer~ search_results
        +int k
    }

    class IndexChunkRecord {
        +str chunk_id
        +str file_path
        +int first_character_index
        +int last_character_index
        +str chunk_type
        +str content_hash
    }

    UnansweredQuestion <|-- AnsweredQuestion
    MinimalSearchResults <|-- MinimalAnswer
    UnansweredQuestion --> MinimalSource : references
    MinimalSearchResults o-- MinimalSource : aggregates
    RagDataset o-- UnansweredQuestion : contains
    StudentSearchResults o-- MinimalSearchResults : contains
    StudentSearchResultsAndAnswer o-- MinimalAnswer : contains
    MinimalSource ..> IndexChunkRecord : mapped from
```

---

## 8. Component Responsibilities and Boundaries

| Subsystem / Component | Package Location | Primary Responsibility | Input Contract | Output Contract |
| :--- | :--- | :--- | :--- | :--- |
| **CLI Dispatcher** | `src.cli` | Parses CLI commands and flags via Python Fire; formats console outputs. | Terminal arguments (`--k`, `--dataset_path`) | Exit code, console logs, output JSONs |
| **HTTP API Server** | `src.api` | Serves REST endpoints for querying, answering, and indexing (Bonus 5). | HTTP JSON request body | Pydantic JSON response |
| **Python AST Chunker** | `src.chunking.python` | Parses Python code via standard `ast` module; splits on class/def while keeping span offsets. | Raw file content (`str`), file path (`str`) | `List[IndexChunkRecord]` |
| **Markdown Chunker** | `src.chunking.markdown`| Parses Markdown structure; splits along `#`, `##`, `###` headings and code blocks. | Raw file content (`str`), file path (`str`) | `List[IndexChunkRecord]` |
| **Code-Aware Tokenizer**| `src.retrieval.tokenizer` | Normalizes text, splits camelCase/snake_case identifiers, strips punctuation, filters stop words. | Raw string (`str`) | `List[str]` (normalized tokens) |
| **BM25 Lexical Engine**| `src.retrieval.bm25` | Computes BM25Okapi scores across sparse inverted index; serializes/deserializes index to disk. | Token list, integer $k$ | Top-$k$ scored chunk indices |
| **Dense Vector Engine** | `src.retrieval.dense` | Generates 384d embeddings using `all-MiniLM-L6-v2`; performs dot-product similarity search. | Text string, integer $k$ | Top-$k$ scored chunk indices |
| **Hybrid RRF Ranker** | `src.retrieval.hybrid` | Combines rank lists from BM25 and Dense engines using Reciprocal Rank Fusion ($k=60$). | Two ranked chunk lists | Unified top-$k$ `MinimalSource` list |
| **Cache Subsystem** | `src.cache` | Manages in-memory LRU and persistent disk cache for query embeddings and search results (Bonus 4). | Query string, hash key | Cached result object or cache miss |
| **Incremental Indexer**| `src.indexing.manifest`| Tracks file mtime and SHA-256 hashes to re-index only altered files (Bonus 3). | Current filesystem tree | Delta list of files to re-index |
| **Prompt Builder** | `src.generation.prompt` | Formats context snippets and instructions into Qwen chat template within token budget. | Question (`str`), Context snippets (`List[str]`) | Formatted prompt string |
| **Inference Engine** | `src.generation.llm` | Executes Qwen3-0.6B model on CPU using Hugging Face Transformers; returns grounded text. | Prompt string | Raw generated text (`str`) |

---

## 9. Technology Stack and Rationale

```
+-----------------------------------------------------------------------------------------+
| PRESENTATION LAYER: Python Fire (CLI) | FastAPI + Uvicorn (HTTP REST)                   |
+-----------------------------------------------------------------------------------------+
| DATA VALIDATION & TYPING: Pydantic v2 | MyPy (Strict Mode) | Flake8 (PEP 8)             |
+-----------------------------------------------------------------------------------------+
| INGESTION & PARSING: Python ast (Code) | Regex Heading Parser (Markdown) | tqdm (Bars)   |
+-----------------------------------------------------------------------------------------+
| RETRIEVAL & RANKING: BM25Okapi (NumPy) | Sentence-Transformers (MiniLM) | Custom RRF    |
+-----------------------------------------------------------------------------------------+
| GENERATION & INFERENCE: Hugging Face Transformers | PyTorch (CPU-Optimized BFloat16)    |
+-----------------------------------------------------------------------------------------+
| STORAGE & CACHING: Pickle / Safetensors | SQLite / JSON Disk Cache | Python functools   |
+-----------------------------------------------------------------------------------------+
| PACKAGE & BUILD ORCHESTRATION: uv (uv sync / pyproject.toml) | GNU Makefile             |
+-----------------------------------------------------------------------------------------+
```

### 9.1 Core Technology Evaluation Matrix

| Layer | Chosen Technology | Evaluated Alternatives | Architectural Rationale & Trade-off Analysis |
| :--- | :--- | :--- | :--- |
| **Project & Package Manager** | **`uv`** | Poetry, Conda, Pipenv | **Mandated by subject.** `uv` provides sub-second dependency resolution, deterministic lockfiles (`uv.lock`), and isolated virtual environments via `uv sync` without external system dependencies. |
| **CLI Framework** | **`Python Fire`** | Click, Argparse, Typer | **Mandated by subject.** Exposes classes and methods directly into intuitive command-line interfaces with automatic type coercion and minimal boilerplate. |
| **Data Validation** | **`Pydantic v2`** | Dataclasses, Attrs, Marshmallow | **Mandated by subject.** High-performance Rust-backed validation. Guarantees 100% compliance with moulinette JSON serialization contracts. |
| **Lexical Search** | **Custom BM25Okapi (NumPy)** | SQLite FTS5, Whoosh, Elasticsearch | **Selected for speed & portability.** Pure Python/NumPy implementation serialized via binary format requires zero background daemon, achieves sub-millisecond retrieval on 14k chunks, and allows custom code tokenization (camelCase/snake_case splitting). |
| **Dense Embeddings** | **`sentence-transformers` (`all-MiniLM-L6-v2`)** | BGE-small, E5-small, OpenAI text-embedding-3 | Runs efficiently on campus CPU cores ($\sim 15\text{ ms}$ per query), produces compact 384-dimensional vectors, and downloads seamlessly via Hugging Face. |
| **LLM Inference Engine** | **`Hugging Face Transformers` + `PyTorch`** | `llama-cpp-python`, `vLLM`, `Ollama` | **Selected for zero build friction.** `llama-cpp-python` requires C/C++ CMake toolchains that risk compilation errors on locked-down campus machines. Transformers installs cleanly via pre-built wheels in `uv sync` and runs Qwen3-0.6B on CPU in $< 1.2\text{ GB}$ RAM. |
| **Language Model** | **`Qwen/Qwen3-0.6B`** | SmolLM-360M, TinyLlama-1.1B | **Mandated by subject.** Superior technical reasoning, instruction following, and Python comprehension compared to other models under 1B parameters. |
| **Caching Engine** | **Two-Tier (Memory `LRU` + Disk `SQLite`)** | Redis, Memcached | Zero infrastructure footprint. In-memory LRU handles identical query bursts; SQLite persists across CLI runs. |
| **HTTP Framework** | **`FastAPI` + `Uvicorn`** | Flask, Django, Tornado | Asynchronous, auto-generates OpenAPI specs, and directly integrates Pydantic v2 schemas without data mapping layers. |

---

## 10. Data Architecture

### 10.1 Pydantic Model Hierarchy & Schema Contracts
All data exchanged between components adheres strictly to the schema specification required by the evaluation moulinette:

```python
# src/models.py
import uuid
from typing import List, Union
from pydantic import BaseModel, Field

class MinimalSource(BaseModel):
    """Represents a discrete source location in the indexed corpus."""
    file_path: str = Field(
        ...,
        description="Path relative to project root (e.g. data/raw/vllm-0.10.1/docs/features/lora.md)"
    )
    first_character_index: int = Field(
        ...,
        ge=0,
        description="0-based starting character position within the file"
    )
    last_character_index: int = Field(
        ...,
        gt=0,
        description="0-based ending character position within the file"
    )

class UnansweredQuestion(BaseModel):
    """Represents a question awaiting retrieval and answer generation."""
    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question"
    )
    question: str = Field(..., min_length=1, description="Question string")

class AnsweredQuestion(UnansweredQuestion):
    """Represents a ground-truth answered question with reference sources."""
    sources: List[MinimalSource]
    answer: str

class RagDataset(BaseModel):
    """Container for question datasets loaded from disk."""
    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]

class MinimalSearchResults(BaseModel):
    """Result of search retrieval for a single question."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
    """Result of answer generation for a single question."""
    answer: str

class StudentSearchResults(BaseModel):
    """Official batch search output evaluated by the moulinette."""
    search_results: List[MinimalSearchResults]
    k: int

class StudentSearchResultsAndAnswer(BaseModel):
    """Official batch answer output evaluated by the moulinette."""
    search_results: List[MinimalAnswer]
    k: int
```

### 10.2 Chunk Catalog Schema (Internal Storage)
To map retrieved snippets back to disk, the indexer persists an internal chunk catalog:

```python
# src/indexing/schema.py
from pydantic import BaseModel
from typing import Literal

class InternalChunk(BaseModel):
    chunk_id: int
    file_path: str
    first_character_index: int
    last_character_index: int
    chunk_type: Literal["python_class", "python_func", "python_module", "markdown_section"]
    token_count: int
    content: str
```

### 10.3 Persistence Layout
All persistent artifacts are stored inside the project structure under `data/`:

```
data/
├── raw/
│   └── vllm-0.10.1/                  <- Immutable target repository
├── processed/
│   ├── bm25_index.pkl                 <- Serialized BM25 sparse inverted index
│   ├── vector_embeddings.npy          <- (Bonus 1) Dense embeddings matrix (N, 384)
│   ├── chunk_metadata.json            <- Internal catalog mapping chunk_id to spans
│   └── index_manifest.json            <- (Bonus 3) File paths, SHA-256 hashes, mtimes
├── datasets/
│   ├── UnansweredQuestions/           <- Evaluation query sets
│   └── AnsweredQuestions/             <- Ground-truth validation sets
├── output/
│   ├── search_results/                <- Generated StudentSearchResults JSONs
│   │   └── <DatasetScope>/
│   └── search_results_and_answer/     <- Generated StudentSearchResultsAndAnswer JSONs
│       └── <DatasetScope>/
└── cache/                             <- (Bonus 4) Persistent query cache SQLite db
    └── query_cache.db
```

---

## 11. API and Integration Architecture

### 11.1 Command-Line Interface (CLI) Specifications
All commands must be runnable as `uv run python -m src <command> [options]`.

| Command | Required Arguments | Optional Arguments | System Responsibility |
| :--- | :--- | :--- | :--- |
| `index` | *None* | `--max_chunk_size` (int, def: 2000), `--corpus_dir` (str) | Recursively indexes raw corpus, builds BM25 and dense indices, writes to `data/processed/`. |
| `search` | `query` (str) | `--k` (int, def: 5) | Performs hybrid/BM25 search for a single query; prints ranked sources with file paths and character spans to stdout. |
| `search_dataset` | `--dataset_path` (path), `--save_directory` (path) | `--k` (int, def: 10) | Batch processes all questions from JSON; writes `StudentSearchResults` conforming JSON to `<save_directory>/<filename>`. |
| `answer` | `query` (str) | `--k` (int, def: 5) | Retrieves top-$k$ sources, prompts Qwen3-0.6B, prints grounded natural language answer and sources. |
| `answer_dataset` | `--student_search_results_path` (path), `--save_directory` (path) | *None* | Reads search results JSON; executes Qwen3-0.6B inference for every question; writes `StudentSearchResultsAndAnswer` JSON. |
| `evaluate` | `--student_search_results_path` (path), `--dataset_path` (path) | `--k` (int, def: 5) | Internal validation tool: calculates Recall@k using IoU $\ge 0.05$ against ground truth. |
| `serve` *(Bonus 5)* | *None* | `--host` (str, def: "127.0.0.1"), `--port` (int, def: 8000) | Starts local asynchronous FastAPI HTTP server. |

### 11.2 Local HTTP REST API Contract (Bonus 5)
When running the local server (`uv run python -m src serve`), the application exposes OpenAPI 3.1 endpoints:

#### Endpoint 1: Search Endpoint
- **URL:** `POST /api/v1/search`
- **Request Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
    "query": "How to configure OpenAI server?",
    "k": 5
  }
  ```
- **Response Status:** `200 OK`
- **Response Body:** `MinimalSearchResults` schema.

#### Endpoint 2: Answer Endpoint
- **URL:** `POST /api/v1/answer`
- **Request Body:**
  ```json
  {
    "query": "How to configure OpenAI server?",
    "k": 5
  }
  ```
- **Response Status:** `200 OK`
- **Response Body:** `MinimalAnswer` schema.

#### Endpoint 3: Health and Readiness
- **URL:** `GET /health`
- **Response Status:** `200 OK`
- **Response Body:**
  ```json
  {
    "status": "healthy",
    "indices_loaded": true,
    "chunk_count": 13466,
    "model_loaded": "Qwen/Qwen3-0.6B"
  }
  ```

---

## 12. Security Architecture

### 12.1 Local Execution and Sandbox Compliance
The system operates within an isolated local evaluation environment without elevated privileges:
- **No Outbound Network Dependency during Evaluation:** All model weights (`Qwen3-0.6B` and `all-MiniLM-L6-v2`) must be locally cached. Evaluation is executable completely offline.
- **Path Traversal Mitigation:** The indexer and context loaders sanitize all file path references using `os.path.commonpath` verification to guarantee that file access never escapes the `data/raw/` repository root.
- **Subprocess Isolation:** The codebase never invokes raw shell strings or untrusted `eval()`/`exec()` calls. Python's built-in `ast.parse` is used strictly in AST-tree inspection mode without executing parsed code.

### 12.2 Defensive Data Handling & Error Isolation
- **Degenerate Input Resistance:**
  - Empty queries (`""` or whitespace) return empty source lists (`[]`) immediately with HTTP 200 / CLI exit code 0 rather than raising an index error.
  - Queries where $k \le 0$ are clamped to $k = 1$ with a warning log.
  - Missing or malformed JSON dataset files output clear stderr diagnostics and cleanly exit without displaying unhandled tracebacks.
- **Character Ceiling Hard-Stop:** Any generated source snippet is clamped to $\le 2000$ characters before output serialization to guarantee that moulinette validation never rejects a run due to span overflows.

---

## 13. Scalability, Performance, and Reliability

### 13.1 SLA Performance Budget & Resource Allocations

| Pipeline Stage | Subject SLA Limit | Architectural Target | Mechanism for Compliance |
| :--- | :--- | :--- | :--- |
| **Corpus Indexing** | $\le 300\text{ seconds}$ (5 min) | $\sim 45\text{ seconds}$ | - Python `ast` syntax parsing.<br>- Vectorized BM25 matrix indexing.<br>- Multi-core batch encoding with MiniLM. |
| **Batch Retrieval (200 Qs)**| $\le 90\text{ seconds}$ (450 ms/Q) | $\sim 12\text{ seconds}$ (60 ms/Q) | - In-memory sparse matrix lookups.<br>- Fast BLAS dot products for embeddings.<br>- Memory-mapped index deserialization. |
| **Batch Generation (200 Qs)**| Feasible within defense time | $\sim 180\text{ seconds}$ (900 ms/Q)| - 0.6B parameter model footprint.<br>- Greedy decoding (`do_sample=False`).<br>- Constrained generation token limit (`max_new_tokens=128`). |
| **Memory Footprint** | System RAM limit (8-16 GB) | $\le 2.2\text{ GB}$ peak RAM | - Inverted index stored in compact arrays.<br>- FP16/BFloat16 model weights on CPU.<br>- Garbage collection after chunking phase. |

### 13.2 Chunking Strategy & Recall Optimization
To achieve the mandatory thresholds ($\ge 80\%$ docs, $\ge 50\%$ code):

1. **Python Syntax-Aware AST Chunking (`src/chunking/python.py`):**
   - Naive fixed-character slicing breaks Python methods across statements, severing function signatures from their bodies and ruining BM25 token matches.
   - The AST Chunker walks the Abstract Syntax Tree using `ast.walk`:
     - Identifies all `ast.FunctionDef`, `ast.AsyncFunctionDef`, and `ast.ClassDef` nodes.
     - Maps node line boundaries (`lineno` to `end_lineno`) to character offsets (`first_character_index`, `last_character_index`).
     - Subdivides oversized classes into individual method chunks while prepending the class signature as context.
     - Preserves standalone module docstrings and top-level statements as introductory chunks.

2. **Markdown Header-Aware Chunking (`src/chunking/markdown.py`):**
   - Documentation is split along Markdown headings (`#`, `##`, `###`).
   - Each heading section is treated as a semantic unit. If a section exceeds 1,800 characters, it is recursively subdivided along paragraph breaks (`\n\n`), ensuring that code blocks (````python ... ````) are never broken across chunk boundaries.

3. **Code-Aware Lexical Tokenizer (`src/retrieval/tokenizer.py`):**
   - Code queries frequently contain composite identifiers (e.g., `openai_compatible_server` or `OpenAIServer`).
   - The tokenizer executes regex-based sub-word splitting:
     - Splits `camelCase` and `PascalCase` into constituent words (`OpenAI` $\rightarrow$ `open`, `ai`).
     - Splits `snake_case` identifiers into individual stems.
     - Retains both the full composite identifier AND its individual sub-tokens in the index, ensuring that queries matching either the exact symbol or a natural language paraphrase register strong BM25 term frequency scores.

4. **Reciprocal Rank Fusion (RRF) Formulation:**
   $$\text{RRF\_Score}(d \in D) = \frac{w_{\text{bm25}}}{60 + \text{Rank}_{\text{bm25}}(d)} + \frac{w_{\text{dense}}}{60 + \text{Rank}_{\text{dense}}(d)}$$
   - Dense retrieval catches broad conceptual questions (e.g., "how does caching work?"), while BM25 catches exact API signatures (e.g., `vllm.entrypoints.openai.api_server`).
   - Combining both via RRF consistently surpasses the 80% (docs) and 50% (code) Recall@5 thresholds.

---

## 14. Repository and Module Structure

The project enforces clean architectural separation, with dependencies flowing inward toward domain models and outward toward infrastructure adapters:

```
RAG/
├── .gitignore
├── Makefile                       # Required build, lint, run, clean automation
├── README.md                      # Comprehensive documentation as per Chapter VIII
├── pyproject.toml                 # uv package definition & dependencies
├── uv.lock                        # Deterministic dependency lockfile
├── data/                          # Data directory hierarchy
│   ├── raw/
│   │   └── vllm-0.10.1/           # Target indexed repository
│   ├── processed/                 # Indices, manifest, chunk metadata
│   ├── datasets/                  # Unanswered & Answered question datasets
│   └── output/                    # Evaluator output directories
└── src/                           # Main Python application package
    ├── __init__.py
    ├── __main__.py                # Main CLI entrypoint (python -m src)
    ├── cli.py                     # Python Fire CLI command router
    ├── models.py                  # Pydantic v2 data models (Data Models section)
    ├── pipeline.py                # Core application orchestrator
    │
    ├── chunking/                  # Ingestion & document segmentation
    │   ├── __init__.py
    │   ├── base.py                # BaseChunker abstract interface
    │   ├── python.py              # AST-based Python code chunker
    │   └── markdown.py            # Header-based Markdown chunker
    │
    ├── retrieval/                 # Search, indexing, and ranking
    │   ├── __init__.py
    │   ├── tokenizer.py           # Code-aware regex tokenizer
    │   ├── bm25.py                # Inverted index & BM25Okapi scoring
    │   ├── dense.py               # (Bonus 1) MiniLM embedding & cosine search
    │   └── hybrid.py              # (Bonus 2) Reciprocal Rank Fusion combiner
    │
    ├── generation/                # Prompt augmentation & LLM inference
    │   ├── __init__.py
    │   ├── prompt.py              # Context grounding prompt builder
    │   └── llm.py                 # Qwen3-0.6B CPU inference engine
    │
    ├── indexing/                  # Corpus management & manifest
    │   ├── __init__.py
    │   ├── indexer.py             # Full corpus indexing coordinator
    │   └── manifest.py            # (Bonus 3) SHA-256 hash incremental tracking
    │
    ├── cache/                     # Caching subsystem (Bonus 4)
    │   ├── __init__.py
    │   ├── memory.py              # In-memory LRU cache
    │   └── disk.py                # Persistent SQLite query/index cache
    │
    ├── api/                       # Local HTTP service (Bonus 5)
    │   ├── __init__.py
    │   └── server.py              # FastAPI application & route endpoints
    │
    └── utils/                     # Cross-cutting logging and formatting
        ├── __init__.py
        ├── logger.py              # Unified structured logging
        └── evaluation.py          # Recall@k calculation (IoU >= 0.05)
```

---

## 15. Deployment and Infrastructure Architecture

### 15.1 Environment Definitions

| Environment | Purpose | Hardware Constraints | Toolchain |
| :--- | :--- | :--- | :--- |
| **Local Development** | Engineering, unit tests, fast iteration | Developer Linux/macOS, optional GPU acceleration | `uv`, `pytest`, `mypy`, `flake8` |
| **Campus Evaluation** | Official defense and moulinette grading | 4-Core CPU Linux (Ubuntu/Fedora), 8-16 GB RAM, No GPU | `uv sync`, `Makefile`, local cached models |
| **Automated Grading** | Continuous headless verification by moulinette | Sandboxed Linux execution, strict time & memory quotas | `moulinette evaluate_student_search_results` |

### 15.2 Makefile Automation Interface
The mandated `Makefile` standardizes build, validation, and execution across environments:

```makefile
# Makefile
.PHONY: install run debug clean lint lint-strict test

PYTHON := uv run python
UV := uv

install:
	$(UV) sync

run:
	$(PYTHON) -m src --help

debug:
	$(PYTHON) -m pdb -m src search "How to configure OpenAI server?" --k 5

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache data/output/* data/processed/* build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict

test:
	$(UV) run pytest tests/ -v
```

---

## 16. Monitoring, Logging, and Observability

### 16.1 Progress Feedback
As mandated by Section VI.6 of the subject, long-running operations (such as repository indexing, batch retrieval, and batch answer generation) are wrapped in `tqdm` progress monitors providing real-time feedback:
```
Indexing: 100%|█████████████████████████████████| 1965/1965 [00:42<00:00, 46.2 file/s]
Tokenizing: 100%|███████████████████████████████| 13466/13466 [00:02<00:00, 6120 chunk/s]
Ingestion complete! Indexed 13466 chunks under data/processed/
```

### 16.2 Structured Logging Standards
The system employs standard Python `logging` configured to output structured messages to `stderr` to avoid corrupting `stdout` when JSON responses are streamed:
- **DEBUG:** AST node visitation, raw chunk boundary calculation, cache hits/misses.
- **INFO:** Lifecycle events (indexing start/end, model loading, dataset completion).
- **WARNING:** Clamped chunk sizes, non-critical token truncations, degenerate query alerts.
- **ERROR:** File access failures, corrupted JSON payloads, validation errors.

---

## 17. Architecture Decision Records (ADRs)

### ADR-001: Pure Python/NumPy BM25 vs External Full-Text Engines
- **Context:** The system requires millisecond lexical retrieval over $\sim 14,000$ code and documentation chunks. Indexing must complete in $< 5\text{ minutes}$, and setup must rely strictly on `uv sync`.
- **Decision:** Implement BM25Okapi using pure Python and vectorized NumPy arrays, persisting the inverted index as a binary pickle payload (`data/processed/bm25_index.pkl`).
- **Alternatives Considered:**
  1. *SQLite FTS5:* Built into Python standard library, but lacks out-of-the-box support for custom code-aware tokenization (splitting camelCase and snake_case during tokenization) without compiling external C extensions.
  2. *Whoosh / Lucene:* Whoosh is unmaintained; Lucene requires a Java Runtime Environment (JRE), violating the campus evaluation constraint.
- **Consequences:** Zero external daemon requirements; sub-millisecond lookup latency; complete control over code-specific tokenization rules.
- **Status:** Approved.

---

### ADR-002: Dual Syntax-Aware Chunking (AST + Markdown Headings)
- **Context:** Naive fixed-window character chunking slices code across arbitrary tokens, causing syntax fragments that destroy retrieval recall and exceed character constraints.
- **Decision:** Implement two distinct chunking modules: `PythonASTChunker` (using Python's `ast` module) and `MarkdownHeaderChunker` (using heading hierarchy parsing). Both enforce a strict 2000-character ceiling.
- **Alternatives Considered:**
  1. *Fixed Character Window with Overlap (e.g., 1000 chars, 200 overlap):* High implementation simplicity, but slices functions in half, cutting identifiers away from their implementations and causing code questions to fail the 50% Recall@5 threshold.
  2. *Tree-Sitter:* Highly capable multi-language parser, but requires pre-compiled C shared libraries which risk platform mismatch on campus machines.
- **Consequences:** Guaranteed syntactic integrity for all Python code chunks; semantic coherence for documentation sections; 100% compliance with character ceiling constraints.
- **Status:** Approved.

---

### ADR-003: Reciprocal Rank Fusion (RRF) for Hybrid Retrieval
- **Context:** The system must meet dual recall targets: $\ge 80\%$ on docs and $\ge 50\%$ on code. Pure lexical search struggles with conceptual questions; pure semantic search struggles with exact variable/method lookups.
- **Decision:** Integrate a lightweight dense encoder (`all-MiniLM-L6-v2`) and combine candidate ranks using Reciprocal Rank Fusion:
  $$RRF\_Score(d) = \sum_{m} \frac{1}{60 + Rank_m(d)}$$
- **Alternatives Considered:**
  1. *Linear Score Combination:* Requires normalizing BM25 (unbounded positive score) and Cosine Similarity ([-1, 1]) via Min-Max or Softmax, which is fragile across differing query types.
  2. *Cross-Encoder Re-ranking:* High accuracy, but introduces severe CPU inference latency ($> 200\text{ ms}$ per candidate pair), breaching the 90-second SLA for 200 batch queries.
- **Consequences:** Robust, scale-invariant rank combination; achieves both code and doc recall targets while keeping batch query time well under 90 seconds.
- **Status:** Approved.

---

### ADR-004: Hugging Face Transformers CPU Runtime for Qwen3-0.6B
- **Context:** The evaluation environment lacks dedicated GPUs. The system must run `Qwen/Qwen3-0.6B` reliably on CPU without crashing or requiring manual compilation.
- **Decision:** Standardize on Hugging Face `transformers` and `torch` (CPU wheel), executing inference with `torch.bfloat16` or `float32` and `max_new_tokens=128`.
- **Alternatives Considered:**
  1. *`llama-cpp-python` / GGUF:* Offers fastest CPU generation, but requires `cmake` and local C++ compiler toolchains during `uv sync`. If the evaluation machine lacks development headers, installation fails.
  2. *`vLLM` Engine:* High throughput on GPUs, but heavy, complex CUDA dependencies that fail on CPU-only campus machines.
- **Consequences:** 100% installation reliability via pre-compiled PyTorch CPU wheels; memory footprint under 1.2 GB; deterministic natural language output.
- **Status:** Approved.

---

### ADR-005: Two-Tier Cache Architecture (In-Memory LRU + Persistent SQLite)
- **Context:** Repeated queries or cold re-runs during iterative testing waste compute re-embedding queries and searching identical indices.
- **Decision:** Implement an in-memory LRU cache (`functools.lru_cache`) backed by a lightweight, zero-configuration local SQLite database (`data/cache/query_cache.db`).
- **Alternatives Considered:**
  1. *Redis:* Requires a standalone running daemon, which is prohibited in self-contained evaluation environments.
  2. *Flat JSON File Cache:* Prone to write lock contention and corruption during rapid batch evaluations.
- **Consequences:** Sub-millisecond response on repeated queries; persistent performance across CLI invocations; zero external dependencies.
- **Status:** Approved.

---

## 18. Risks, Trade-offs, and Open Questions

### 18.1 Technical Risks and Mitigation Matrix

| Risk Identified | Severity | Probability | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Character Span Exceeds 2000 Chars** | **CRITICAL** | Low | Hard validation guard in `BaseChunker` and `MinimalSource` serializer. Any chunk exceeding 2000 characters is aggressively clamped or subdivided before persistence. |
| **Relative File Path Discrepancy** | **CRITICAL** | Medium | Normalize all paths relative to the project root (`data/raw/vllm-0.10.1/...`) with forward slashes, stripping any leading `./` to ensure 100% verbatim string match with moulinette ground truth. |
| **Out-Of-Memory (OOM) on Campus CPU** | **HIGH** | Low | Limit batch sizes during embedding generation; execute Qwen3 inference with `torch.inference_mode()` and greedy decoding; delete AST intermediate objects during indexing. |
| **Batch Retrieval Timeout (> 90s for 200 Qs)**| **HIGH** | Medium | Vectorize BM25 candidate scoring using matrix operations; cache query embeddings; bypass dense search if lexical match score exceeds high-confidence threshold. |
| **Base Model Reasoning Limitations** | **MEDIUM** | High | Subject Section VI.3 explicitly notes Qwen3-0.6B has reasoning limits and grading prioritizes retrieval quality. Grounding is enforced via strict few-shot context prompt templates. |

### 18.2 Trade-offs
1. **Index Size vs. Retrieval Speed:** Storing chunk text inside `chunks.json` alongside index offsets increases disk footprint by $\sim 15\text{ MB}$, but avoids thousands of random disk file reads during retrieval and generation, accelerating throughput by $5\times$.
2. **CPU Model Quantization vs. Portability:** Running PyTorch FP32/BF16 is slightly slower than a 4-bit GGUF binary, but guarantees 100% portable installation on any Linux machine via `uv sync` without compilation failures.

---

## 19. Final Recommended Architecture

The recommended architecture establishes a clean, modular, and resilient solution for **RAG Against the Machine (v2.0)**:

```
[CLI: Fire] / [HTTP: FastAPI]
           │
           ▼
   [Pipeline Orchestrator] ◄───► [Two-Tier Cache]
           │
     ┌─────┴───────────────────────────┐
     ▼                                 ▼
[Indexing Subsystem]         [Retrieval & Generation Subsystem]
 ├── Manifest (SHA-256)       ├── Code-Aware Tokenizer
 ├── Python AST Chunker       ├── BM25Okapi Lexical Matcher
 ├── Markdown Header Chunker  ├── Dense MiniLM Vector Matcher
 ├── BM25 Matrix Builder      ├── Reciprocal Rank Fusion (k=60)
 └── MiniLM Vector Encoder    ├── Context Grounding Formatter
                              └── Qwen/Qwen3-0.6B CPU Inference
                                       │
                                       ▼
                       [Pydantic v2 Output Serializer]
                                       │
                                       ▼
                [StudentSearchResults / Answer JSON]
                                       │
                                       ▼
                       [Evaluator / 42 Moulinette]
```

### Key Architectural Commitments:
1. **Precision Compliance:** Full adherence to Pydantic v2 data models, relative file path strings, and the 2000-character context ceiling.
2. **Dual-Retrieval Superiority:** BM25 lexical precision combined with MiniLM semantic coverage via Reciprocal Rank Fusion, ensuring $\ge 80\%$ Docs Recall@5 and $\ge 50\%$ Code Recall@5.
3. **Frictionless Portability:** Zero external background servers, pure Python dependencies resolved through `uv sync`, and a standard `Makefile` interface for automated peer evaluation.
4. **Complete Bonus Readiness:** Native modular components for Semantic Vectors, Hybrid Search, Incremental Hashing, Multi-Tier Caching, and FastAPI HTTP Serving.
