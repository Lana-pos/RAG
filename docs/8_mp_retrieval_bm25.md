# Mandatory Part: `src/retrieval/bm25.py`

## 1. Overview & Architectural Role

[`src/retrieval/bm25.py`](../src/retrieval/bm25.py) implements the **Vectorized BM25Okapi Inverted Index** required by Chapter VI.1:
> *"For retrieval itself, implement at least one of the two classic lexical methods: TF-IDF, BM25."*

BM25Okapi is the lexical retrieval engine that enables the system to search across ~28,000 chunks in **sub-millisecond latency** without requiring external databases (e.g. Elasticsearch, PostgreSQL) or server daemons.

---

## 2. Okapi BM25 Mathematical Formulation

BM25 scores the relevance of a document $D$ against a search query $Q = \{q_1, q_2, \dots, q_n\}$ using:

$$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot qtf \cdot \frac{tf(q, D) \cdot (k_1 + 1)}{tf(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
1. **$tf(q, D)$ (Term Frequency):** Number of times token $q$ appears in document $D$.
2. **$qtf$ (Query Term Frequency):** Number of times token $q$ appears in the query.
3. **$|D|$ & $\text{avgdl}$:** Length of document $D$ and average document length across the corpus.
4. **$k_1$ (Term Saturation Parameter, default 1.5):** Controls how quickly additional occurrences of a term saturate in score.
5. **$b$ (Length Normalization Parameter, default 0.75; 0.4 calibrated for mixed code/docs):** Penalizes long documents.
6. **$\text{IDF}(q)$ (Inverse Document Frequency):** Computed using the smoothed Robertson-Spärck Jones formula:
   $$\text{IDF}(q) = \ln\left(1 + \frac{N - df(q) + 0.5}{df(q) + 0.5}\right)$$
   Where $N$ is total documents in the corpus and $df(q)$ is the number of documents containing term $q$.

---

## 3. Data Structures & Vectorization

### 3.1 `PostingArray`
A specialized NumPy 1D array subclass storing:
- `doc_ids: NDArray[np.int32]`: Sorted document IDs containing the term.
- `term_frequencies: NDArray[np.float32]`: Occurrence count of the term in each document.

### 3.2 Precomputed Document Length Vector ($K_d$)
To avoid repeated division during search, `BM25Index.fit()` precomputes a 1D vector of length $N$:
```python
len_ratio = doc_lengths / avgdl
self._K_d = (k1 * (1.0 - b + b * len_ratio)).astype(np.float32)
```

### 3.3 Vectorized Scoring: `get_scores`
When a query arrives:
1. `scores = np.zeros(num_docs, dtype=np.float32)` is allocated.
2. For each query term $q$:
   - Fetches posting array in $O(1)$ from `self.inverted_index[q]`.
   - Vectorized computation on candidate documents:
     ```python
     denom = tfs + self._K_d[doc_ids]
     term_score = (idf * qtf) * (tfs * (k1 + 1.0)) / denom
     scores[doc_ids] += term_score
     ```
3. Only documents that contain at least one query term are touched, achieving sub-millisecond search latency.

---

## 4. Key Methods

- `fit(corpus_tokens)`: Analyzes tokenized documents, computes document lengths, vocabulary, IDFs, and builds the inverted index.
- `search(query, k=5, filter_zero=True)`: Uses `np.argpartition` to find top-$k$ document indices with highest BM25 score without sorting the entire corpus.
- `search_sources(query, k=5)`: Maps top-$k$ document indices back to `MinimalSource` objects with exact `file_path`, `first_character_index`, and `last_character_index`.
- `save(path)` / `load(path)`: Serializes/deserializes the binary index payload to `data/processed/bm25_index.pkl` via Python `pickle`.
