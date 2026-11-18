# Mandatory Part: `src/utils/evaluation.py`

## 1. Overview & Architectural Role

[`src/utils/evaluation.py`](../src/utils/evaluation.py) implements the **Internal Recall@k Evaluator** replicating the 42 Moulinette evaluation logic.

The subject states in Chapter VI.6:
> *"The evaluate command is for your own iteration. The official recall@k used during the defense is computed by the provided moulinette executable, not by your code. Your solution must never import or call the moulinette."*

`src/utils/evaluation.py` is an independent, pure-Python implementation of the evaluation algorithm. It allows developers to benchmark and verify retrieval quality locally before running the external grader binary.

---

## 2. Character-Span Intersection-over-Union (IoU)

### 2.1 The IoU Formula
A candidate chunk retrieved by the student and a reference chunk in ground truth are represented as character intervals $[S_1, E_1)$ and $[S_2, E_2)$.
Intersection-over-Union is calculated as:

$$\text{IoU} = \frac{\max(0, \min(E_1, E_2) - \max(S_1, S_2))}{\max(E_1, E_2) - \min(S_1, S_2)}$$

```python
def calculate_character_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return float(intersection / union) if union > 0 else 0.0
```

### 2.2 Match Criteria (`is_source_match`)
A candidate source is judged a valid match if and only if:
1. **Verbatim Path Match:** `candidate.file_path == ground_truth.file_path` exactly.
2. **Span Length Guard:** Candidate span length is $\le 2000$ characters (`max_context_length`). Any source $> 2000$ chars is immediately invalidated.
3. **IoU Threshold:** $\text{IoU} \ge 0.05$ (5% character span overlap).

Because the required overlap threshold is low ($\text{IoU} \ge 0.05$), the student does not need to match reference boundaries character-for-character; identifying the correct section of the correct file counts as a match.

---

## 3. Recall@k Metric Calculation

For a single question with reference sources $G$ and top-$k$ student predictions $P_k$:

$$\text{Recall@}k = \frac{\sum_{g \in G} \mathbb{I}\Big(\exists p \in P_k : \text{is\_source\_match}(p, g)\Big)}{|G|}$$

Across a dataset of $M$ questions:
$$\text{Mean Recall@}k = \frac{1}{M} \sum_{i=1}^{M} \text{Recall@}k_i$$

The evaluator computes Recall at $k \in \{1, 3, 5, 10\}$.

---

## 4. Evaluation Output Table

Running `uv run python -m src evaluate ...` produces the formatted summary report:
```text
🎯 Evaluation Results
========================================
📊 Questions evaluated: 100
📈 Recall@1: 0.640 (64.0%)
📈 Recall@3: 0.800 (80.0%)
📈 Recall@5: 0.870 (87.0%)  <- docs threshold is 80% [PASS]
📈 Recall@10: 0.890 (89.0%)
```
This output format mirrors the official Moulinette display.
