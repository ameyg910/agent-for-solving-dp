from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for",
    "is", "are", "be", "with", "from", "by", "as", "at", "it", "this",
    "that", "you", "your", "we", "can", "will", "given", "find",
    "determine", "print", "output", "input",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokens(text: str) -> list[str]:
    return [
        token
        for token in normalize_text(text).split()
        if len(token) > 1 and token not in STOPWORDS
    ]


def problem_text(problem: dict[str, Any]) -> str:
    fields = (
        "title",
        "description",
        "input_format",
        "output_format",
        "interaction_format",
        "note",
    )

    return "\n".join(
        str(problem.get(field) or "")
        for field in fields
    )


def short_text(problem: dict[str, Any]) -> str:
    fields = (
        "description",
        "input_format",
        "output_format",
        "interaction_format",
        "note",
    )

    return "\n".join(
        str(problem.get(field) or "")
        for field in fields
    )


def make_shingles(
    token_list: list[str],
    size: int = 5,
) -> set[tuple[str, ...]]:
    if len(token_list) <= size:
        return {tuple(token_list)} if token_list else set()

    return {
        tuple(token_list[i:i + size])
        for i in range(len(token_list) - size + 1)
    }


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    # Iterate over the smaller set.
    if len(a) > len(b):
        a, b = b, a

    intersection = sum(1 for x in a if x in b)
    union = len(a) + len(b) - intersection

    return intersection / union if union else 0.0


def cosine_counter(
    a: Counter[str],
    b: Counter[str],
    norm_a: float,
    norm_b: float,
) -> float:
    if not a or not b or norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # Iterate through the smaller counter.
    if len(a) > len(b):
        a, b = b, a

    numerator = sum(
        value * b[token]
        for token, value in a.items()
        if token in b
    )

    return numerator / (norm_a * norm_b)


def build_blocks(
    records: list[dict[str, Any]],
    max_postings: int = 40,
) -> dict[tuple[str, ...], set[int]]:
    """
    Build an inverted index over informative 5-token shingles.

    Shingles appearing in too many documents are discarded.
    A lower max_postings prevents common competitive-programming
    phrases from producing enormous candidate sets.
    """

    postings: dict[tuple[str, ...], set[int]] = defaultdict(set)

    for idx, record in enumerate(records):
        for shingle in record["_shingles"]:
            postings[shingle].add(idx)

    return {
        key: ids
        for key, ids in postings.items()
        if len(ids) <= max_postings
    }


def candidate_pairs(
    records: list[dict[str, Any]],
    postings: dict[tuple[str, ...], set[int]],
    min_shared_shingles: int = 2,
) -> set[tuple[int, int]]:
    """
    Generate candidate pairs from shared shingles.

    An additional cheap filter requires at least two shared shingles.
    """

    pair_counts: Counter[tuple[int, int]] = Counter()

    for ids in postings.values():
        ids_list = sorted(ids)

        for i in range(len(ids_list)):
            a = ids_list[i]

            for j in range(i + 1, len(ids_list)):
                b = ids_list[j]

                # Store canonical pair.
                pair_counts[(a, b)] += 1

    return {
        pair
        for pair, count in pair_counts.items()
        if count >= min_shared_shingles
    }


def compare(
    a: dict[str, Any],
    b: dict[str, Any],
) -> dict[str, Any]:

    title_similarity = jaccard(
        a["_title_tokens"],
        b["_title_tokens"],
    )

    statement_similarity = jaccard(
        a["_token_set"],
        b["_token_set"],
    )

    shingle_similarity = jaccard(
        a["_shingles"],
        b["_shingles"],
    )

    cosine_similarity = cosine_counter(
        a["_token_counter"],
        b["_token_counter"],
        a["_token_norm"],
        b["_token_norm"],
    )

    rating_a = a["_problem"].get("rating")
    rating_b = b["_problem"].get("rating")

    same_rating = (
        rating_a is not None
        and rating_b is not None
        and rating_a == rating_b
    )

    family_a = a["dp"].get("primary_family")
    family_b = b["dp"].get("primary_family")

    same_family = (
        family_a is not None
        and family_b is not None
        and family_a == family_b
    )

    score = (
        0.45 * shingle_similarity
        + 0.30 * cosine_similarity
        + 0.15 * statement_similarity
        + 0.10 * title_similarity
    )

    return {
        "score": round(score, 6),
        "title_similarity": round(title_similarity, 6),
        "statement_jaccard": round(statement_similarity, 6),
        "shingle_jaccard": round(shingle_similarity, 6),
        "token_cosine": round(cosine_similarity, 6),
        "same_rating": same_rating,
        "same_dp_family": same_family,
        "dp_family_a": family_a,
        "dp_family_b": family_b,
    }


def prepare_record(
    canonical_record: dict[str, Any],
    problem: dict[str, Any],
) -> dict[str, Any]:

    statement_tokens = tokens(short_text(problem))
    title_tokens = tokens(str(problem.get("title", "")))

    counter = Counter(statement_tokens)

    norm = math.sqrt(
        sum(value * value for value in counter.values())
    )

    return {
        **canonical_record,
        "_problem": problem,
        "_token_set": set(statement_tokens),
        "_token_counter": counter,
        "_token_norm": norm,
        "_title_tokens": set(title_tokens),
        "_shingles": make_shingles(statement_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate near-duplicate problem candidates."
    )

    parser.add_argument(
        "--problems",
        default="data/raw/open_r1_codeforces_train.jsonl",
    )

    parser.add_argument(
        "--canonical",
        default="data/deduplicated/problem_canonical.jsonl",
    )

    parser.add_argument(
        "--output",
        default="data/deduplicated/near_duplicate_candidates.jsonl",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.62,
    )

    parser.add_argument(
        "--max-postings",
        type=int,
        default=40,
        help="Ignore shingles occurring in more than this many problems.",
    )

    parser.add_argument(
        "--min-shared-shingles",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    problems = read_jsonl(Path(args.problems))
    canonical = read_jsonl(Path(args.canonical))

    problem_by_id = {
        str(problem["id"]): problem
        for problem in problems
    }

    print(f"Loaded {len(problems):,} problems.")
    print(f"Loaded {len(canonical):,} canonical records.")

    records: list[dict[str, Any]] = []

    for canonical_record in canonical:
        problem_id = str(canonical_record["problem_id"])
        problem = problem_by_id.get(problem_id)

        if problem is None:
            continue

        records.append(
            prepare_record(
                canonical_record,
                problem,
            )
        )

    print(f"Prepared {len(records):,} records.")

    postings = build_blocks(
        records,
        max_postings=args.max_postings,
    )

    print(
        f"Indexed {len(postings):,} informative shingles "
        f"(max postings={args.max_postings})."
    )

    pairs = candidate_pairs(
        records,
        postings,
        min_shared_shingles=args.min_shared_shingles,
    )

    print(
        f"Candidate pairs before similarity threshold: "
        f"{len(pairs):,}"
    )

    candidates: list[dict[str, Any]] = []

    total_pairs = len(pairs)

    for n, (i, j) in enumerate(pairs, start=1):
        a = records[i]
        b = records[j]

        # Cheap first-stage filters.

        # Two problems sharing only a tiny fraction of their tokens
        # should not reach the expensive scoring stage.
        token_set_similarity = jaccard(
            a["_token_set"],
            b["_token_set"],
        )

        if token_set_similarity < 0.08:
            continue

        comparison = compare(a, b)

        if comparison["score"] < args.threshold:
            continue

        # Exact duplicates are handled separately.
        if (
            a["canonical"]["exact_hash"]
            == b["canonical"]["exact_hash"]
        ):
            continue

        candidates.append(
            {
                "problem_a": a["problem_id"],
                "problem_b": b["problem_id"],
                "similarity": comparison,
                "title_a": a["title"],
                "title_b": b["title"],
            }
        )

        if n % 100_000 == 0:
            print(
                f"Compared {n:,}/{total_pairs:,} pairs; "
                f"kept {len(candidates):,} candidates."
            )

    candidates.sort(
        key=lambda x: (
            -x["similarity"]["score"],
            x["problem_a"],
            x["problem_b"],
        )
    )

    write_jsonl(
        Path(args.output),
        candidates,
    )

    print(
        json.dumps(
            {
                "input_problems": len(records),
                "indexed_shingles": len(postings),
                "candidate_pairs_before_threshold": len(pairs),
                "near_duplicate_candidates": len(candidates),
                "threshold": args.threshold,
                "max_postings": args.max_postings,
                "min_shared_shingles": args.min_shared_shingles,
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
