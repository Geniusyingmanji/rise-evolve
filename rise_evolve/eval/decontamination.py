from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from rise_evolve.io import iter_jsonl, repo_path


TEXT_FIELDS = {
    "instruction",
    "text",
    "prompt",
    "target",
    "target_description",
    "expected_target",
    "rational_target_description",
    "editor_prompt",
    "question",
    "reference",
    "gt",
}

FORBIDDEN_TRAIN_FIELDS = {
    "benchmark_id",
    "benchmark_source_path",
    "official_answer",
    "official_target",
    "official_score",
}

FORBIDDEN_TRAIN_PATH_PARTS = [
    "data/benchmarks/",
    "outputs/eval/",
]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_ngrams(text: str, n: int = 5) -> Set[str]:
    words = normalize_text(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def iter_text_values(obj: Any, parent_key: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            yield from iter_text_values(value, key_str)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_text_values(value, parent_key)
    elif isinstance(obj, str):
        if parent_key in TEXT_FIELDS or len(obj.split()) >= 5:
            yield parent_key, obj


@dataclass
class BenchmarkText:
    benchmark: str
    sample_id: str
    field: str
    normalized_text: str
    text_hash: str
    ngrams: Set[str]


def load_benchmark_text_index(benchmarks_dir: Path) -> List[BenchmarkText]:
    index_path = benchmarks_dir / "benchmark_text_index.jsonl"
    if not index_path.exists():
        return []
    rows: List[BenchmarkText] = []
    for row in iter_jsonl(index_path):
        norm = row.get("normalized_text") or normalize_text(str(row.get("text", "")))
        rows.append(
            BenchmarkText(
                benchmark=str(row.get("benchmark", "")),
                sample_id=str(row.get("sample_id", "")),
                field=str(row.get("field", "")),
                normalized_text=norm,
                text_hash=str(row.get("text_hash") or sha256_text(norm)),
                ngrams=word_ngrams(norm),
            )
        )
    return rows


def check_row(
    row: Dict[str, Any],
    benchmark_texts: Sequence[BenchmarkText],
    row_id: str,
    ngram_threshold: float = 0.82,
    hash_index: Dict[str, List[BenchmarkText]] | None = None,
    ngram_index: Dict[str, Set[int]] | None = None,
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    hash_index = hash_index or {}
    ngram_index = ngram_index or {}
    for field in FORBIDDEN_TRAIN_FIELDS:
        if field in row:
            issues.append({"severity": "high", "type": "forbidden_field", "row_id": row_id, "field": field})

    for key, value in iter_text_values(row):
        norm = normalize_text(value)
        if not norm:
            continue
        text_hash = sha256_text(norm)
        for needle in FORBIDDEN_TRAIN_PATH_PARTS:
            if needle in value:
                issues.append({"severity": "high", "type": "forbidden_path", "row_id": row_id, "field": key, "path": value})
        for bench in hash_index.get(text_hash, []):
            issues.append(
                {
                    "severity": "high",
                    "type": "exact_benchmark_text",
                    "row_id": row_id,
                    "field": key,
                    "benchmark": bench.benchmark,
                    "sample_id": bench.sample_id,
                }
            )
        row_ngrams = word_ngrams(norm)
        candidate_ids: Set[int] = set()
        for ngram in row_ngrams:
            candidate_ids.update(ngram_index.get(ngram, set()))
        for bench_idx in candidate_ids:
            bench = benchmark_texts[bench_idx]
            if text_hash == bench.text_hash:
                continue
            overlap = jaccard(row_ngrams, bench.ngrams)
            if overlap >= ngram_threshold and len(norm.split()) >= 5:
                issues.append(
                    {
                        "severity": "medium",
                        "type": "ngram_benchmark_text",
                        "row_id": row_id,
                        "field": key,
                        "benchmark": bench.benchmark,
                        "sample_id": bench.sample_id,
                        "overlap": round(overlap, 4),
                    }
                )
    return issues


def check_files(
    train_files: Sequence[Path],
    benchmarks_dir: Path | None = None,
    ngram_threshold: float = 0.82,
    limit: int | None = None,
) -> Dict[str, Any]:
    benchmarks_dir = benchmarks_dir or repo_path("data", "benchmarks")
    benchmark_texts = load_benchmark_text_index(benchmarks_dir)
    hash_index: Dict[str, List[BenchmarkText]] = {}
    ngram_index: Dict[str, Set[int]] = {}
    for idx, bench in enumerate(benchmark_texts):
        hash_index.setdefault(bench.text_hash, []).append(bench)
        for ngram in bench.ngrams:
            ngram_index.setdefault(ngram, set()).add(idx)
    issues: List[Dict[str, Any]] = []
    rows_checked = 0
    for path in train_files:
        for row in iter_jsonl(path):
            row_id = str(row.get("task_id") or row.get("id") or row.get("item_id") or rows_checked)
            issues.extend(
                check_row(
                    row,
                    benchmark_texts,
                    row_id,
                    ngram_threshold=ngram_threshold,
                    hash_index=hash_index,
                    ngram_index=ngram_index,
                )
            )
            rows_checked += 1
            if limit is not None and rows_checked >= limit:
                break
        if limit is not None and rows_checked >= limit:
            break

    severity_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    for issue in issues:
        severity_counts[issue["severity"]] = severity_counts.get(issue["severity"], 0) + 1
        type_counts[issue["type"]] = type_counts.get(issue["type"], 0) + 1
    return {
        "ok": not any(issue["severity"] == "high" for issue in issues),
        "rows_checked": rows_checked,
        "benchmark_text_items": len(benchmark_texts),
        "severity_counts": severity_counts,
        "type_counts": type_counts,
        "issues": issues[:200],
    }
