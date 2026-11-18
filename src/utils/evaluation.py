"""Internal Recall@k Evaluator replicating 42 Moulinette logic.
Implements:
    1. Character-span Intersection-over-Union (IoU):
       ``IoU = max(0, min(E1, E2) - max(S1, S2)) / (max(E1, E2) - min(S1, S2))``
    2. Verbatim ``file_path`` matching with ``IoU >= 0.05`` overlap threshold
       and ``max_context_length <= 2000`` validation.
    3. Question-level and dataset-level Recall@k calculation across multiple *k*
       cutoffs (``k in {1, 3, 5, 10}``) with automatic Docs vs. Code breakdown.
    4. Detailed summary table generation benchmarking Recall@5 against project
       SLA targets (Docs >= 80%, Code >= 50%).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from src.indexing.schema import InternalChunk
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)

logger = logging.getLogger(__name__)

DEFAULT_IOU_THRESHOLD: float = 0.05
DEFAULT_MAX_CONTEXT_LENGTH: int = 2000
DEFAULT_K: int = 5
DEFAULT_K_VALUES: Tuple[int, ...] = (1, 3, 5, 10)

DOCS_RECALL_AT_5_TARGET: float = 0.80
CODE_RECALL_AT_5_TARGET: float = 0.50


def _extract_raw_source_fields(source: Any) -> Tuple[str, int, int]:
    """Extract verbatim ``(file_path, first_character_index, last_character_index)``.
    Preserves the exact ``file_path`` string without normalizing when *source*
    is a dict or custom object so verbatim path comparisons accurately reject
    unnormalized prefixes (e.g. ``./data/raw/...``).
    """
    if isinstance(source, tuple) and len(source) == 2:
        first_elem, second_elem = source
        if isinstance(first_elem, int) and isinstance(second_elem, int):
            return "", int(first_elem), int(second_elem)
        source = first_elem

    if isinstance(source, (MinimalSource, InternalChunk)):
        return (
            source.file_path,
            int(source.first_character_index),
            int(source.last_character_index),
        )

    if isinstance(source, Mapping):
        raw_path = str(source.get("file_path", source.get("path", "")))
        s_idx = int(
            source.get(
                "first_character_index",
                source.get("start_char", source.get("start", 0)),
            )
        )
        e_idx = int(
            source.get(
                "last_character_index",
                source.get("end_char", source.get("end", 0)),
            )
        )
        return raw_path, s_idx, e_idx

    if (
        hasattr(source, "file_path")
        and hasattr(source, "first_character_index")
        and hasattr(source, "last_character_index")
    ):
        return (
            str(getattr(source, "file_path")),
            int(getattr(source, "first_character_index")),
            int(getattr(source, "last_character_index")),
        )

    raise TypeError(
        f"Cannot extract character span coordinates from object of type {type(source).__name__}"
    )


def compute_span_iou(
    start_1: int,
    end_1: int,
    start_2: int,
    end_2: int,
) -> float:
    """Compute character-span Intersection-over-Union (IoU).
    Implements the exact acceptance criterion formula:
        ``IoU = max(0, min(E1, E2) - max(S1, S2)) / (max(E1, E2) - min(S1, S2))``
        start_1: Starting character index of first span (``S1``).
        end_1: Ending character index of first span (``E1``).
        start_2: Starting character index of second span (``S2``).
        end_2: Ending character index of second span (``E2``).
    """
    s1, e1 = int(start_1), int(end_1)
    s2, e2 = int(start_2), int(end_2)

    if e1 <= s1 or e2 <= s2:
        return 0.0

    intersection = max(0, min(e1, e2) - max(s1, s2))
    if intersection <= 0:
        return 0.0

    union_span = max(e1, e2) - min(s1, s2)
    if union_span <= 0:
        return 0.0

    return float(intersection) / float(union_span)


def compute_iou(
    first: Any,
    second: Any = None,
    third: Optional[int] = None,
    fourth: Optional[int] = None,
    *,
    check_file_path: bool = False,
) -> float:
    """Compute character-span IoU between two sources, span tuples, or four integer bounds.
    Supports both calling conventions:
        1. ``compute_iou(s1, e1, s2, e2)`` with 4 integer character indices.
        2. ``compute_iou(source_a, source_b)`` with :class:`MinimalSource`,
           :class:`InternalChunk`, dicts, or ``(start, end)`` tuples.
        first: First source object, ``(S1, E1)`` tuple, or ``S1`` integer.
        second: Second source object, ``(S2, E2)`` tuple, or ``E1`` integer.
        third: ``S2`` integer when called with 4 positional arguments.
        fourth: ``E2`` integer when called with 4 positional arguments.
        check_file_path: If True, returns ``0.0`` when ``file_path`` strings
            do not match verbatim.
    """
    if third is not None and fourth is not None:
        return compute_span_iou(int(first), int(second), int(third), int(fourth))

    if second is None:
        raise TypeError(
            "compute_iou requires either 2 source/span arguments or 4 integer indices."
        )

    path_1, s1, e1 = _extract_raw_source_fields(first)
    path_2, s2, e2 = _extract_raw_source_fields(second)

    if check_file_path and path_1 != path_2:
        return 0.0

    return compute_span_iou(s1, e1, s2, e2)


def is_source_match(
    candidate: Any,
    ground_truth: Any,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> bool:
    """Return True if *candidate* matches *ground_truth* under moulinette rules.
    Matching rules (Acceptance Criterion 2):
        1. ``file_path`` must match verbatim (exact string equality).
        2. Character-span ``IoU >= iou_threshold`` (default ``0.05``).
        candidate: Retrieved candidate source.
        ground_truth: Reference ground-truth source.
        iou_threshold: Minimum IoU required for a match (default: ``0.05``).
    """
    if iou_threshold < 0.0 or iou_threshold > 1.0:
        raise ValueError(
            f"iou_threshold must be in [0.0, 1.0], got {iou_threshold}"
        )

    cand_path, s1, e1 = _extract_raw_source_fields(candidate)
    gt_path, s2, e2 = _extract_raw_source_fields(ground_truth)

    if not cand_path or not gt_path:
        return False

    # Verbatim file path check
    if cand_path != gt_path:
        return False

    iou = compute_span_iou(s1, e1, s2, e2)
    return iou >= iou_threshold


def validate_student_search_results(
    student_results: Any,
    max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH,
) -> Tuple[bool, List[str]]:
    """Validate student search results against moulinette structural & length rules.
    Any retrieved source exceeding *max_context_length* (default 2000 characters)
    or having invalid span bounds marks the entire submission invalid.
        student_results: :class:`StudentSearchResults`, :class:`StudentSearchResultsAndAnswer`,
            dict, or sequence of :class:`MinimalSearchResults`.
        max_context_length: Maximum allowed character span length (default: 2000).
    """
    errors: List[str] = []

    if max_context_length <= 0:
        raise ValueError(
            f"max_context_length must be positive, got {max_context_length}"
        )

    results_list: Sequence[Any]
    if isinstance(student_results, (StudentSearchResults, StudentSearchResultsAndAnswer)):
        results_list = student_results.search_results
    elif isinstance(student_results, Mapping):
        raw_seq = student_results.get("search_results")
        if not isinstance(raw_seq, Sequence) or isinstance(raw_seq, (str, bytes)):
            return False, ["Missing or invalid 'search_results' list in student payload."]
        results_list = raw_seq
    elif isinstance(student_results, Sequence) and not isinstance(
        student_results, (str, bytes)
    ):
        results_list = student_results
    else:
        return False, [
            f"Unsupported student_results type: {type(student_results).__name__}"
        ]

    for q_idx, item in enumerate(results_list):
        sources: Sequence[Any]
        if isinstance(item, MinimalSearchResults):
            q_id = item.question_id
            sources = item.retrieved_sources
        elif isinstance(item, Mapping):
            q_id = str(item.get("question_id", f"index_{q_idx}"))
            raw_srcs = item.get("retrieved_sources", [])
            if not isinstance(raw_srcs, Sequence) or isinstance(raw_srcs, (str, bytes)):
                errors.append(f"Question '{q_id}': 'retrieved_sources' is not a list.")
                continue
            sources = raw_srcs
        else:
            errors.append(
                f"Question at index {q_idx} has invalid type {type(item).__name__}."
            )
            continue

        for s_idx, src in enumerate(sources):
            try:
                file_path, start_idx, end_idx = _extract_raw_source_fields(src)
            except Exception as exc:
                errors.append(f"Question '{q_id}' source #{s_idx}: {exc}")
                continue

            if not file_path or not file_path.strip():
                errors.append(f"Question '{q_id}' source #{s_idx}: empty file_path.")
            if file_path.startswith("./") or file_path.startswith("/"):
                errors.append(
                    f"Question '{q_id}' source #{s_idx}: unnormalized file_path '{file_path}'."
                )
            if start_idx < 0:
                errors.append(
                    f"Question '{q_id}' source #{s_idx}: "
                    f"negative first_character_index ({start_idx})."
                )
            if end_idx <= start_idx:
                errors.append(
                    f"Question '{q_id}' source #{s_idx}: last_character_index ({end_idx}) "
                    f"<= first_character_index ({start_idx})."
                )
            span_len = end_idx - start_idx
            if span_len > max_context_length:
                errors.append(
                    f"Question '{q_id}' source #{s_idx} ('{file_path}'): "
                    f"span length {span_len} exceeds max_context_length ({max_context_length})."
                )

    return len(errors) == 0, errors


def classify_question_category(
    ground_truth_sources: Sequence[Any],
    dataset_hint: Optional[str] = None,
) -> Literal["docs", "code", "other"]:
    """Classify a ground-truth question into ``'docs'`` or ``'code'`` category.
    Inspects ground-truth source file extensions/paths first (``.md`` / ``/docs/``
    -> ``'docs'``, ``.py`` -> ``'code'``), falling back to *dataset_hint* if needed.
    """
    for src in ground_truth_sources:
        try:
            file_path, _, _ = _extract_raw_source_fields(src)
            lower_path = file_path.lower().replace("\\", "/")
            if lower_path.endswith((".md", ".rst", ".txt")) or "/docs/" in lower_path:
                return "docs"
            if lower_path.endswith((".py", ".pyx", ".c", ".cpp", ".cu", ".h")):
                return "code"
        except Exception:
            continue

    if dataset_hint:
        hint_lower = dataset_hint.lower()
        if "doc" in hint_lower:
            return "docs"
        if "code" in hint_lower:
            return "code"

    return "other"


def compute_question_recall_at_k(
    retrieved_sources: Sequence[Any],
    ground_truth_sources: Sequence[Any],
    k: int = DEFAULT_K,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> float:
    """Compute Recall@k for a single question.
    For a question with reference sources ``G`` and top-*k* retrieved candidate
    sources ``R_k``, returns the fraction of reference sources ``g in G`` that
    are matched by at least one candidate ``r in R_k`` (verbatim ``file_path``
    and ``IoU >= iou_threshold``).
        retrieved_sources: Ranked list of candidate sources returned by the system.
        ground_truth_sources: Reference sources from the ground-truth dataset.
        k: Top-k cutoff (must be >= 0).
        iou_threshold: Minimum IoU overlap threshold (default: ``0.05``).
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if not ground_truth_sources:
        return 1.0
    if k == 0 or not retrieved_sources:
        return 0.0

    top_k_candidates = list(retrieved_sources[:k])
    matched_count = 0

    for gt_src in ground_truth_sources:
        if any(
            is_source_match(cand_src, gt_src, iou_threshold=iou_threshold)
            for cand_src in top_k_candidates
        ):
            matched_count += 1

    return float(matched_count) / float(len(ground_truth_sources))


def _compute_best_ious_per_gt(
    retrieved_sources: Sequence[Any],
    ground_truth_sources: Sequence[Any],
    k: int = DEFAULT_K,
) -> List[float]:
    """Return the highest IoU achieved in top-*k* for each ground-truth source."""
    top_k_candidates = list(retrieved_sources[: max(0, k)])
    best_ious: List[float] = []
    for gt_src in ground_truth_sources:
        gt_path, s2, e2 = _extract_raw_source_fields(gt_src)
        best = 0.0
        for cand_src in top_k_candidates:
            cand_path, s1, e1 = _extract_raw_source_fields(cand_src)
            if cand_path == gt_path:
                score = compute_span_iou(s1, e1, s2, e2)
                if score > best:
                    best = score
        best_ious.append(best)
    return best_ious


@dataclass
class QuestionEvalResult:
    """Per-question evaluation metrics across multiple Recall@k cutoffs."""

    question_id: str
    question: str
    category: Literal["docs", "code", "other"]
    num_ground_truth: int
    num_retrieved: int
    matched_at_k: Dict[int, int] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    best_ious: List[float] = field(default_factory=list)

    @property
    def recall_at_1(self) -> float:
        """Return Recall@1 for this question."""
        return self.recall_at_k.get(1, 0.0)

    @property
    def recall_at_3(self) -> float:
        """Return Recall@3 for this question."""
        return self.recall_at_k.get(3, 0.0)

    @property
    def recall_at_5(self) -> float:
        """Return Recall@5 for this question."""
        return self.recall_at_k.get(5, 0.0)

    @property
    def recall_at_10(self) -> float:
        """Return Recall@10 for this question."""
        return self.recall_at_k.get(10, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize per-question result to a dictionary."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "num_ground_truth": self.num_ground_truth,
            "num_retrieved": self.num_retrieved,
            "matched_at_k": dict(self.matched_at_k),
            "recall_at_k": dict(self.recall_at_k),
            "recall@5": self.recall_at_5,
            "best_ious": list(self.best_ious),
        }


@dataclass
class EvaluationResult:
    """Aggregate dataset evaluation report replicating 42 moulinette output + SLA table."""

    k: int = DEFAULT_K
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    docs_recall_at_k: Dict[int, float] = field(default_factory=dict)
    code_recall_at_k: Dict[int, float] = field(default_factory=dict)
    num_questions: int = 0
    num_docs_questions: int = 0
    num_code_questions: int = 0
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH
    question_results: List[QuestionEvalResult] = field(default_factory=list)
    summary_table: str = ""

    @property
    def recall_at_1(self) -> float:
        """Return aggregate Recall@1."""
        return self.recall_at_k.get(1, 0.0)

    @property
    def recall_at_3(self) -> float:
        """Return aggregate Recall@3."""
        return self.recall_at_k.get(3, 0.0)

    @property
    def recall_at_5(self) -> float:
        """Return aggregate Recall@5."""
        return self.recall_at_k.get(5, self.recall_at_k.get(self.k, 0.0))

    @property
    def recall_at_10(self) -> float:
        """Return aggregate Recall@10."""
        return self.recall_at_k.get(10, self.recall_at_k.get(self.k, 0.0))

    @property
    def docs_recall_at_5(self) -> float:
        """Return Documentation Recall@5."""
        return self.docs_recall_at_k.get(5, self.docs_recall_at_k.get(self.k, 0.0))

    @property
    def code_recall_at_5(self) -> float:
        """Return Code Recall@5."""
        return self.code_recall_at_k.get(5, self.code_recall_at_k.get(self.k, 0.0))

    @property
    def score(self) -> float:
        """Return Recall at the primary requested *k*."""
        return self.recall_at_k.get(self.k, self.recall_at_5)

    @property
    def docs_passed(self) -> bool:
        """Return True if Documentation Recall@5 meets the >= 80% SLA target."""
        return self.docs_recall_at_5 >= DOCS_RECALL_AT_5_TARGET

    @property
    def code_passed(self) -> bool:
        """Return True if Code Recall@5 meets the >= 50% SLA target."""
        return self.code_recall_at_5 >= CODE_RECALL_AT_5_TARGET

    def to_dict(self) -> Dict[str, Any]:
        """Return a comprehensive dictionary view of evaluation metrics."""
        data: Dict[str, Any] = {
            "k": self.k,
            "is_valid": self.is_valid,
            "validation_errors": list(self.validation_errors),
            "num_questions": self.num_questions,
            "num_docs_questions": self.num_docs_questions,
            "num_code_questions": self.num_code_questions,
            "iou_threshold": self.iou_threshold,
            "max_context_length": self.max_context_length,
            "recall_at_k": dict(self.recall_at_k),
            "docs_recall_at_k": dict(self.docs_recall_at_k),
            "code_recall_at_k": dict(self.code_recall_at_k),
            "recall_at_1": self.recall_at_1,
            "recall_at_3": self.recall_at_3,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "Recall@1": self.recall_at_1,
            "Recall@3": self.recall_at_3,
            "Recall@5": self.recall_at_5,
            "Recall@10": self.recall_at_10,
            "recall@1": self.recall_at_1,
            "recall@3": self.recall_at_3,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "docs_recall_at_5": self.docs_recall_at_5,
            "code_recall_at_5": self.code_recall_at_5,
            "docs_recall@5": self.docs_recall_at_5,
            "code_recall@5": self.code_recall_at_5,
            "docs_passed": self.docs_passed,
            "code_passed": self.code_passed,
            "score": self.score,
            "summary_table": self.summary_table,
        }
        for k_val, score_val in self.recall_at_k.items():
            data[f"Recall@{k_val}"] = score_val
            data[f"recall@{k_val}"] = score_val
            data[f"recall_at_{k_val}"] = score_val
        return data

    def __getitem__(self, key: Union[str, int]) -> Any:
        """Allow dictionary-style lookup by metric name or integer k."""
        if isinstance(key, int):
            if key in self.recall_at_k:
                return self.recall_at_k[key]
            raise KeyError(key)
        mapping = self.to_dict()
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        """Return True if *key* is present in the evaluation report."""
        if isinstance(key, int):
            return key in self.recall_at_k
        if isinstance(key, str):
            return key in self.to_dict()
        return False

    def keys(self) -> Iterator[str]:
        """Iterate over dictionary keys."""
        return iter(self.to_dict().keys())

    def format_table(self) -> str:
        """Return the formatted summary table string."""
        if self.summary_table:
            return self.summary_table
        return format_summary_table(self)

    def __str__(self) -> str:
        """Return human-readable moulinette output and detailed Recall@5 summary table."""
        return self.format_table()


def format_summary_table(
    result: Optional[Union[EvaluationResult, Mapping[str, Any]]] = None,
    *,
    docs_result: Optional[Union[EvaluationResult, Mapping[str, Any], float]] = None,
    code_result: Optional[Union[EvaluationResult, Mapping[str, Any], float]] = None,
    k: int = DEFAULT_K,
) -> str:
    """Generate a detailed summary table of Recall@5 (and Recall@1/3/10) for Docs and Code.
    Includes both the 42 moulinette header format and a structured Markdown/ASCII
    table comparing Docs Recall@5 (target >= 80.0%) and Code Recall@5 (target >= 50.0%).
        result: Optional primary :class:`EvaluationResult` or metrics dict.
        docs_result: Optional explicit documentation evaluation result or Recall@5 float.
        code_result: Optional explicit code evaluation result or Recall@5 float.
        k: Primary evaluation cutoff (default: 5).
    """
    is_valid = True
    overall_r_map: Dict[int, float] = {}
    docs_r_map: Dict[int, float] = {}
    code_r_map: Dict[int, float] = {}
    n_total = 0
    n_docs = 0
    n_code = 0

    if isinstance(result, EvaluationResult):
        is_valid = result.is_valid
        overall_r_map = dict(result.recall_at_k)
        docs_r_map = dict(result.docs_recall_at_k)
        code_r_map = dict(result.code_recall_at_k)
        n_total = result.num_questions
        n_docs = result.num_docs_questions
        n_code = result.num_code_questions
    elif isinstance(result, Mapping):
        is_valid = bool(result.get("is_valid", True))
        if isinstance(result.get("recall_at_k"), Mapping):
            overall_r_map = {
                int(kv): float(vv) for kv, vv in result["recall_at_k"].items()
            }
        else:
            for cutoff in (1, 3, 5, 10):
                for key_variant in (
                    f"Recall@{cutoff}",
                    f"recall@{cutoff}",
                    f"recall_at_{cutoff}",
                ):
                    if key_variant in result:
                        overall_r_map[cutoff] = float(result[key_variant])
        if isinstance(result.get("docs_recall_at_k"), Mapping):
            docs_r_map = {
                int(kv): float(vv) for kv, vv in result["docs_recall_at_k"].items()
            }
        elif "docs_recall_at_5" in result or "docs_recall@5" in result:
            val = float(result.get("docs_recall_at_5", result.get("docs_recall@5", 0.0)))
            docs_r_map = {1: val, 3: val, 5: val, 10: val}

        if isinstance(result.get("code_recall_at_k"), Mapping):
            code_r_map = {
                int(kv): float(vv) for kv, vv in result["code_recall_at_k"].items()
            }
        elif "code_recall_at_5" in result or "code_recall@5" in result:
            val = float(result.get("code_recall_at_5", result.get("code_recall@5", 0.0)))
            code_r_map = {1: val, 3: val, 5: val, 10: val}

        n_total = int(result.get("num_questions", 0))
        n_docs = int(result.get("num_docs_questions", 0))
        n_code = int(result.get("num_code_questions", 0))

    # Override / populate from explicit docs_result if provided
    if docs_result is not None:
        if isinstance(docs_result, EvaluationResult):
            docs_r_map = dict(docs_result.recall_at_k)
            n_docs = docs_result.num_questions
            is_valid = is_valid and docs_result.is_valid
        elif isinstance(docs_result, Mapping):
            r5 = float(
                docs_result.get(
                    "recall_at_5",
                    docs_result.get("Recall@5", docs_result.get("recall@5", 0.0)),
                )
            )
            docs_r_map = {
                1: float(docs_result.get("recall_at_1", docs_result.get("Recall@1", r5))),
                3: float(docs_result.get("recall_at_3", docs_result.get("Recall@3", r5))),
                5: r5,
                10: float(docs_result.get("recall_at_10", docs_result.get("Recall@10", r5))),
            }
            n_docs = int(docs_result.get("num_questions", n_docs))
        elif isinstance(docs_result, (int, float)):
            r5 = float(docs_result)
            docs_r_map = {1: r5, 3: r5, 5: r5, 10: r5}

    # Override / populate from explicit code_result if provided
    if code_result is not None:
        if isinstance(code_result, EvaluationResult):
            code_r_map = dict(code_result.recall_at_k)
            n_code = code_result.num_questions
            is_valid = is_valid and code_result.is_valid
        elif isinstance(code_result, Mapping):
            r5 = float(
                code_result.get(
                    "recall_at_5",
                    code_result.get("Recall@5", code_result.get("recall@5", 0.0)),
                )
            )
            code_r_map = {
                1: float(code_result.get("recall_at_1", code_result.get("Recall@1", r5))),
                3: float(code_result.get("recall_at_3", code_result.get("Recall@3", r5))),
                5: r5,
                10: float(code_result.get("recall_at_10", code_result.get("Recall@10", r5))),
            }
            n_code = int(code_result.get("num_questions", n_code))
        elif isinstance(code_result, (int, float)):
            r5 = float(code_result)
            code_r_map = {1: r5, 3: r5, 5: r5, 10: r5}

    # If overall_r_map is empty but docs/code maps exist, synthesize overall average
    if not overall_r_map and (docs_r_map or code_r_map):
        if n_total == 0:
            n_total = n_docs + n_code
        for cutoff in (1, 3, 5, 10):
            if docs_r_map and code_r_map:
                if n_docs + n_code > 0:
                    overall_r_map[cutoff] = (
                        docs_r_map.get(cutoff, 0.0) * n_docs
                        + code_r_map.get(cutoff, 0.0) * n_code
                    ) / (n_docs + n_code)
                else:
                    overall_r_map[cutoff] = 0.5 * (
                        docs_r_map.get(cutoff, 0.0) + code_r_map.get(cutoff, 0.0)
                    )
            elif docs_r_map:
                overall_r_map[cutoff] = docs_r_map.get(cutoff, 0.0)
            elif code_r_map:
                overall_r_map[cutoff] = code_r_map.get(cutoff, 0.0)

    r1 = overall_r_map.get(1, 0.0)
    r3 = overall_r_map.get(3, 0.0)
    r5 = overall_r_map.get(5, overall_r_map.get(k, 0.0))
    r10 = overall_r_map.get(10, r5)

    docs_r1 = docs_r_map.get(1, 0.0)
    docs_r3 = docs_r_map.get(3, 0.0)
    docs_r5 = docs_r_map.get(5, 0.0)
    docs_r10 = docs_r_map.get(10, docs_r5)
    docs_status = "PASS" if docs_r5 >= DOCS_RECALL_AT_5_TARGET else "FAIL"

    code_r1 = code_r_map.get(1, 0.0)
    code_r3 = code_r_map.get(3, 0.0)
    code_r5 = code_r_map.get(5, 0.0)
    code_r10 = code_r_map.get(10, code_r5)
    code_status = "PASS" if code_r5 >= CODE_RECALL_AT_5_TARGET else "FAIL"

    lines: List[str] = [
        f"Student data is valid: {is_valid}",
        "Evaluation Results",
        "========================================",
        (
            f"Recall@1: {r1:.3f} "
            f"Recall@3: {r3:.3f} "
            f"Recall@5: {r5:.3f} "
            f"Recall@10: {r10:.3f}"
        ),
        "",
        (
            "| Dataset Scope | Questions | Recall@1 | Recall@3 "
            "| Recall@5 | Recall@10 | Target (R@5) | Status |"
        ),
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
        (
            f"| Docs (Documentation) | {n_docs} | {docs_r1:.3f} ({docs_r1 * 100:.1f}%) "
            f"| {docs_r3:.3f} ({docs_r3 * 100:.1f}%) | {docs_r5:.3f} ({docs_r5 * 100:.1f}%) "
            f"| {docs_r10:.3f} ({docs_r10 * 100:.1f}%) | >= 80.0% (0.800) | {docs_status} |"
        ),
        (
            f"| Code (Python Source) | {n_code} | {code_r1:.3f} ({code_r1 * 100:.1f}%) "
            f"| {code_r3:.3f} ({code_r3 * 100:.1f}%) | {code_r5:.3f} ({code_r5 * 100:.1f}%) "
            f"| {code_r10:.3f} ({code_r10 * 100:.1f}%) | >= 50.0% (0.500) | {code_status} |"
        ),
        (
            f"| Overall (Combined) | {n_total} | {r1:.3f} ({r1 * 100:.1f}%) "
            f"| {r3:.3f} ({r3 * 100:.1f}%) | {r5:.3f} ({r5 * 100:.1f}%) "
            f"| {r10:.3f} ({r10 * 100:.1f}%) | IoU >= 0.05 | "
            f"{'VALID' if is_valid else 'INVALID'} |"
        ),
    ]
    return "\n".join(lines)


def _load_student_results_map(
    student_search_results: Union[
        StudentSearchResults,
        StudentSearchResultsAndAnswer,
        Sequence[Union[MinimalSearchResults, Mapping[str, Any]]],
        Mapping[str, Any],
        str,
        Path,
    ],
) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]], Any]:
    """Load student search results into lookup maps by ``question_id`` and ``question`` text.

    Returns:
        Tuple of ``(by_question_id_map, by_question_text_map, raw_container_for_validation)``.
    """
    raw_container: Any = student_search_results
    items: Sequence[Any] = []

    if isinstance(student_search_results, (str, Path)):
        path = Path(student_search_results)
        if not path.is_file():
            raise FileNotFoundError(
                f"Student search results file not found: {path}"
            )
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON in student search results file '{path}': {exc}"
            ) from exc
        raw_container = raw_json
        if isinstance(raw_json, Mapping) and "search_results" in raw_json:
            seq = raw_json["search_results"]
            items = seq if isinstance(seq, Sequence) else []
        elif isinstance(raw_json, list):
            items = raw_json
        else:
            raise ValueError(
                f"Expected 'search_results' key in '{path}', got {type(raw_json).__name__}"
            )
    elif isinstance(
        student_search_results, (StudentSearchResults, StudentSearchResultsAndAnswer)
    ):
        items = student_search_results.search_results
    elif isinstance(student_search_results, Mapping):
        if "search_results" in student_search_results:
            seq = student_search_results["search_results"]
            items = seq if isinstance(seq, Sequence) else []
        else:
            # Map of {question_id: [sources]}
            by_id_direct: Dict[str, List[Any]] = {}
            for q_id, srcs in student_search_results.items():
                if isinstance(srcs, Sequence) and not isinstance(srcs, (str, bytes)):
                    by_id_direct[str(q_id)] = list(srcs)
            return by_id_direct, {}, list(items)
    elif isinstance(student_search_results, Sequence):
        items = student_search_results
    else:
        raise TypeError(
            f"Unsupported student_search_results type: {type(student_search_results).__name__}"
        )

    by_id: Dict[str, List[Any]] = {}
    by_question: Dict[str, List[Any]] = {}

    for item in items:
        if isinstance(item, MinimalSearchResults):
            by_id[item.question_id] = list(item.retrieved_sources)
            by_question[item.question.strip()] = list(item.retrieved_sources)
        elif isinstance(item, Mapping):
            item_q_id = item.get("question_id")
            q_text = item.get("question")
            srcs = item.get("retrieved_sources", item.get("sources", []))
            src_list = (
                list(srcs)
                if (isinstance(srcs, Sequence) and not isinstance(srcs, (str, bytes)))
                else []
            )
            if item_q_id is not None:
                by_id[str(item_q_id)] = src_list
            if isinstance(q_text, str) and q_text.strip():
                by_question[q_text.strip()] = src_list

    return by_id, by_question, raw_container


def _load_ground_truth_questions(
    dataset: Union[
        RagDataset,
        Sequence[Union[AnsweredQuestion, Mapping[str, Any]]],
        Mapping[str, Any],
        str,
        Path,
    ],
) -> Tuple[List[Tuple[str, str, List[Any]]], Optional[str]]:
    """Load ground-truth questions as ``[(question_id, question_text, gt_sources), ...]``."""
    dataset_hint: Optional[str] = None
    raw_questions: Sequence[Any] = []

    if isinstance(dataset, (str, Path)):
        path = Path(dataset)
        dataset_hint = path.name
        if not path.is_file():
            raise FileNotFoundError(f"Ground-truth dataset file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON in ground-truth dataset '{path}': {exc}"
            ) from exc

        if isinstance(raw_json, Mapping) and "rag_questions" in raw_json:
            seq = raw_json["rag_questions"]
            raw_questions = seq if isinstance(seq, Sequence) else []
        elif isinstance(raw_json, list):
            raw_questions = raw_json
        else:
            raise ValueError(
                f"Expected 'rag_questions' key in '{path}', got {type(raw_json).__name__}"
            )
    elif isinstance(dataset, RagDataset):
        raw_questions = dataset.rag_questions
    elif isinstance(dataset, Mapping):
        if "rag_questions" in dataset:
            seq = dataset["rag_questions"]
            raw_questions = seq if isinstance(seq, Sequence) else []
        else:
            # Map of {question_id: [gt_sources]}
            parsed_map: List[Tuple[str, str, List[Any]]] = []
            for q_id, srcs in dataset.items():
                if isinstance(srcs, Sequence) and not isinstance(srcs, (str, bytes)):
                    parsed_map.append((str(q_id), str(q_id), list(srcs)))
            return parsed_map, None
    elif isinstance(dataset, Sequence):
        raw_questions = dataset
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")

    questions_out: List[Tuple[str, str, List[Any]]] = []
    for idx, q_item in enumerate(raw_questions):
        if isinstance(q_item, AnsweredQuestion):
            questions_out.append(
                (q_item.question_id, q_item.question, list(q_item.sources))
            )
        elif isinstance(q_item, UnansweredQuestion):
            srcs = getattr(q_item, "sources", [])
            questions_out.append((q_item.question_id, q_item.question, list(srcs)))
        elif isinstance(q_item, MinimalSearchResults):
            questions_out.append(
                (q_item.question_id, q_item.question, list(q_item.retrieved_sources))
            )
        elif isinstance(q_item, Mapping):
            if q_item.get("is_valid") is False:
                continue
            q_id = str(q_item.get("question_id", f"q_{idx}"))
            q_text = str(q_item.get("question", q_id))
            srcs = q_item.get("sources", q_item.get("retrieved_sources", []))
            src_list = (
                list(srcs)
                if (isinstance(srcs, Sequence) and not isinstance(srcs, (str, bytes)))
                else []
            )
            questions_out.append((q_id, q_text, src_list))

    return questions_out, dataset_hint


class RecallEvaluator:
    """Internal Recall@k Evaluator replicating the 42 moulinette grading harness.
    Responsibilities:
        1. Validates student output structure and enforces ``max_context_length <= 2000``.
        2. Computes character-span IoU:
           ``IoU = max(0, min(E1, E2) - max(S1, S2)) / (max(E1, E2) - min(S1, S2))``
        3. Matches candidate sources against ground-truth sources if ``file_path``
           matches verbatim and ``IoU >= 0.05``.
        4. Computes Recall@k (for ``k in (1, 3, 5, 10)`` and custom ``k``) and
           generates a detailed summary table for Docs and Code Recall@5.
    """

    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH,
        k_values: Sequence[int] = DEFAULT_K_VALUES,
        default_k: int = DEFAULT_K,
    ) -> None:
        """Initialize the RecallEvaluator.

        Args:
            iou_threshold: Minimum IoU required for a span match (default: ``0.05``).
            max_context_length: Maximum allowed character span per source (default: ``2000``).
            k_values: Cutoff ranks to evaluate (default: ``(1, 3, 5, 10)``).
            default_k: Primary Recall@k cutoff (default: ``5``).
        """
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError(
                f"iou_threshold must be in [0.0, 1.0], got {iou_threshold}"
            )
        if max_context_length <= 0:
            raise ValueError(
                f"max_context_length must be positive, got {max_context_length}"
            )
        if default_k <= 0:
            raise ValueError(f"default_k must be positive, got {default_k}")

        self.iou_threshold: float = float(iou_threshold)
        self.max_context_length: int = int(max_context_length)
        self.default_k: int = int(default_k)

        cutoffs = sorted(
            {int(kv) for kv in k_values if int(kv) > 0} | {1, 3, 5, 10, self.default_k}
        )
        self.k_values: Tuple[int, ...] = tuple(cutoffs)

    @staticmethod
    def compute_iou(
        first: Any,
        second: Any = None,
        third: Optional[int] = None,
        fourth: Optional[int] = None,
        *,
        check_file_path: bool = False,
    ) -> float:
        """Compute character-span IoU."""
        return compute_iou(
            first,
            second,
            third,
            fourth,
            check_file_path=check_file_path,
        )

    def is_source_match(
        self,
        candidate: Any,
        ground_truth: Any,
        iou_threshold: Optional[float] = None,
    ) -> bool:
        """Check whether *candidate* matches *ground_truth* (verbatim path & IoU >= threshold)."""
        threshold = (
            float(iou_threshold) if iou_threshold is not None else self.iou_threshold
        )
        return is_source_match(candidate, ground_truth, iou_threshold=threshold)

    def evaluate_question(
        self,
        retrieved_sources: Sequence[Any],
        ground_truth_sources: Sequence[Any],
        *,
        question_id: str = "",
        question: str = "",
        category: Optional[Literal["docs", "code", "other"]] = None,
        dataset_hint: Optional[str] = None,
        k_values: Optional[Sequence[int]] = None,
    ) -> QuestionEvalResult:
        """Evaluate Recall@k for a single question across all configured *k* cutoffs."""
        cutoffs = (
            sorted({int(kv) for kv in k_values if int(kv) > 0} | {1, 3, 5, 10})
            if k_values is not None
            else list(self.k_values)
        )
        resolved_category: Literal["docs", "code", "other"] = (
            category
            if category is not None
            else classify_question_category(ground_truth_sources, dataset_hint=dataset_hint)
        )

        matched_at_k: Dict[int, int] = {}
        recall_map: Dict[int, float] = {}
        n_gt = len(ground_truth_sources)

        for k_cutoff in cutoffs:
            if n_gt == 0:
                matched_at_k[k_cutoff] = 0
                recall_map[k_cutoff] = 1.0
            else:
                top_cands = list(retrieved_sources[:k_cutoff])
                matched = sum(
                    1
                    for gt in ground_truth_sources
                    if any(
                        is_source_match(cand, gt, iou_threshold=self.iou_threshold)
                        for cand in top_cands
                    )
                )
                matched_at_k[k_cutoff] = matched
                recall_map[k_cutoff] = float(matched) / float(n_gt)

        max_k = max(cutoffs) if cutoffs else self.default_k
        best_ious = _compute_best_ious_per_gt(
            retrieved_sources, ground_truth_sources, k=max_k
        )

        return QuestionEvalResult(
            question_id=question_id,
            question=question,
            category=resolved_category,
            num_ground_truth=n_gt,
            num_retrieved=len(retrieved_sources),
            matched_at_k=matched_at_k,
            recall_at_k=recall_map,
            best_ious=best_ious,
        )

    def evaluate(
        self,
        student_search_results: Optional[
            Union[
                StudentSearchResults,
                StudentSearchResultsAndAnswer,
                Sequence[Union[MinimalSearchResults, Mapping[str, Any]]],
                Mapping[str, Any],
                str,
                Path,
            ]
        ] = None,
        dataset: Optional[
            Union[
                RagDataset,
                Sequence[Union[AnsweredQuestion, Mapping[str, Any]]],
                Mapping[str, Any],
                str,
                Path,
            ]
        ] = None,
        *,
        student_search_results_path: Optional[Union[str, Path]] = None,
        dataset_path: Optional[Union[str, Path]] = None,
        k: Optional[int] = None,
        max_context_length: Optional[int] = None,
        iou_threshold: Optional[float] = None,
    ) -> EvaluationResult:
        """Evaluate student search results against a ground-truth dataset.
            student_search_results: Student search results object or JSON path.
            dataset: Ground-truth dataset (:class:`RagDataset` or JSON path).
            student_search_results_path: Keyword alias for *student_search_results*.
            dataset_path: Keyword alias for *dataset*.
            k: Primary Recall@k cutoff (default: ``5``).
            max_context_length: Maximum allowed character span length (default: ``2000``).
            iou_threshold: Minimum IoU overlap threshold (default: ``0.05``).
        """
        resolved_student = (
            student_search_results
            if student_search_results is not None
            else student_search_results_path
        )
        resolved_dataset = dataset if dataset is not None else dataset_path

        if resolved_student is None:
            raise ValueError(
                "student_search_results or student_search_results_path must be provided."
            )
        if resolved_dataset is None:
            raise ValueError("dataset or dataset_path must be provided.")

        effective_k = int(k) if k is not None else self.default_k
        if effective_k <= 0:
            raise ValueError(f"k must be positive, got {effective_k}")

        effective_max_len = (
            int(max_context_length)
            if max_context_length is not None
            else self.max_context_length
        )
        effective_iou = (
            float(iou_threshold) if iou_threshold is not None else self.iou_threshold
        )

        by_id, by_question, raw_container = _load_student_results_map(resolved_student)
        is_valid, validation_errors = validate_student_search_results(
            raw_container, max_context_length=effective_max_len
        )

        gt_questions, dataset_hint = _load_ground_truth_questions(resolved_dataset)
        cutoffs = sorted(set(self.k_values) | {effective_k, 1, 3, 5, 10})

        old_threshold = self.iou_threshold
        self.iou_threshold = effective_iou
        try:
            q_results: List[QuestionEvalResult] = []
            for q_id, q_text, gt_sources in gt_questions:
                if q_id in by_id:
                    cand_sources = by_id[q_id]
                elif q_text.strip() in by_question:
                    cand_sources = by_question[q_text.strip()]
                else:
                    cand_sources = []

                q_eval = self.evaluate_question(
                    retrieved_sources=cand_sources,
                    ground_truth_sources=gt_sources,
                    question_id=q_id,
                    question=q_text,
                    dataset_hint=dataset_hint,
                    k_values=cutoffs,
                )
                q_results.append(q_eval)
        finally:
            self.iou_threshold = old_threshold

        n_total = len(q_results)
        docs_qs = [qr for qr in q_results if qr.category == "docs"]
        code_qs = [qr for qr in q_results if qr.category == "code"]

        overall_recall: Dict[int, float] = {}
        docs_recall: Dict[int, float] = {}
        code_recall: Dict[int, float] = {}

        for cutoff in cutoffs:
            overall_recall[cutoff] = (
                sum(qr.recall_at_k.get(cutoff, 0.0) for qr in q_results) / n_total
                if n_total > 0
                else 0.0
            )
            docs_recall[cutoff] = (
                sum(qr.recall_at_k.get(cutoff, 0.0) for qr in docs_qs) / len(docs_qs)
                if docs_qs
                else 0.0
            )
            code_recall[cutoff] = (
                sum(qr.recall_at_k.get(cutoff, 0.0) for qr in code_qs) / len(code_qs)
                if code_qs
                else 0.0
            )

        report = EvaluationResult(
            k=effective_k,
            recall_at_k=overall_recall,
            docs_recall_at_k=docs_recall,
            code_recall_at_k=code_recall,
            num_questions=n_total,
            num_docs_questions=len(docs_qs),
            num_code_questions=len(code_qs),
            is_valid=is_valid,
            validation_errors=validation_errors,
            iou_threshold=effective_iou,
            max_context_length=effective_max_len,
            question_results=q_results,
        )
        report.summary_table = format_summary_table(report, k=effective_k)
        return report

    def evaluate_docs_and_code(
        self,
        docs_student_results: Any,
        docs_dataset: Any,
        code_student_results: Any,
        code_dataset: Any,
        *,
        k: Optional[int] = None,
    ) -> EvaluationResult:
        """Evaluate both Docs and Code datasets and combine into a unified report and table."""
        effective_k = int(k) if k is not None else self.default_k
        docs_rep = self.evaluate(docs_student_results, docs_dataset, k=effective_k)
        code_rep = self.evaluate(code_student_results, code_dataset, k=effective_k)

        combined_qs = list(docs_rep.question_results) + list(code_rep.question_results)
        n_total = len(combined_qs)
        cutoffs = sorted(set(docs_rep.recall_at_k.keys()) | set(code_rep.recall_at_k.keys()))

        overall_recall: Dict[int, float] = {}
        for cutoff in cutoffs:
            overall_recall[cutoff] = (
                sum(qr.recall_at_k.get(cutoff, 0.0) for qr in combined_qs) / n_total
                if n_total > 0
                else 0.0
            )

        combined = EvaluationResult(
            k=effective_k,
            recall_at_k=overall_recall,
            docs_recall_at_k=dict(docs_rep.recall_at_k),
            code_recall_at_k=dict(code_rep.recall_at_k),
            num_questions=n_total,
            num_docs_questions=docs_rep.num_questions,
            num_code_questions=code_rep.num_questions,
            is_valid=docs_rep.is_valid and code_rep.is_valid,
            validation_errors=docs_rep.validation_errors + code_rep.validation_errors,
            iou_threshold=self.iou_threshold,
            max_context_length=self.max_context_length,
            question_results=combined_qs,
        )
        combined.summary_table = format_summary_table(
            combined,
            docs_result=docs_rep,
            code_result=code_rep,
            k=effective_k,
        )
        return combined


def recall_at_k(
    retrieved_or_student: Any,
    ground_truth: Any,
    k: int = DEFAULT_K,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> float:
    """Compute Recall@k for either a single question's sources or an entire dataset.

    Automatically detects whether *retrieved_or_student* and *ground_truth* represent
    single-question source lists ``(List[MinimalSource], List[MinimalSource])``
    or dataset-level structures ``(StudentSearchResults, RagDataset)``.
    """
    if isinstance(
        retrieved_or_student,
        (StudentSearchResults, StudentSearchResultsAndAnswer, str, Path, Mapping),
    ) or isinstance(ground_truth, (RagDataset, str, Path, Mapping)):
        evaluator = RecallEvaluator(iou_threshold=iou_threshold, default_k=k)
        res = evaluator.evaluate(retrieved_or_student, ground_truth, k=k)
        return res.recall_at_k.get(k, 0.0)

    if (
        isinstance(retrieved_or_student, Sequence)
        and len(retrieved_or_student) > 0
        and isinstance(retrieved_or_student[0], MinimalSearchResults)
    ):
        evaluator = RecallEvaluator(iou_threshold=iou_threshold, default_k=k)
        res = evaluator.evaluate(retrieved_or_student, ground_truth, k=k)
        return res.recall_at_k.get(k, 0.0)

    return compute_question_recall_at_k(
        retrieved_sources=retrieved_or_student,
        ground_truth_sources=ground_truth,
        k=k,
        iou_threshold=iou_threshold,
    )


def evaluate_search_results(
    student_search_results: Optional[Any] = None,
    dataset: Optional[Any] = None,
    *,
    student_search_results_path: Optional[Union[str, Path]] = None,
    dataset_path: Optional[Union[str, Path]] = None,
    k: int = DEFAULT_K,
    max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> EvaluationResult:
    """Evaluate student search results against a ground-truth dataset."""
    evaluator = RecallEvaluator(
        iou_threshold=iou_threshold,
        max_context_length=max_context_length,
        default_k=k,
    )
    return evaluator.evaluate(
        student_search_results=student_search_results,
        dataset=dataset,
        student_search_results_path=student_search_results_path,
        dataset_path=dataset_path,
        k=k,
        max_context_length=max_context_length,
        iou_threshold=iou_threshold,
    )
