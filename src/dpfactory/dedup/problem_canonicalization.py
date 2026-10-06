from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


# Fields that describe the actual mathematical/programming problem.
TEXT_FIELDS = (
    "title",
    "description",
    "input_format",
    "output_format",
    "interaction_format",
    "note",
)


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


def normalize_text(value: Any) -> str:
    """
    Conservative deterministic normalization.

    This deliberately does NOT remove mathematical content or aggressively
    rewrite words. The purpose is only to make formatting differences
    irrelevant for exact duplicate detection.
    """
    if value is None:
        return ""

    text = str(value)

    # Unicode normalization.
    text = unicodedata.normalize("NFKC", text)

    # Normalize common line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing whitespace from every line.
    lines = [line.rstrip() for line in text.split("\n")]

    # Collapse runs of horizontal whitespace.
    lines = [re.sub(r"[ \t]+", " ", line) for line in lines]

    # Collapse excessive blank lines.
    result: list[str] = []
    previous_blank = False

    for line in lines:
        blank = not line.strip()

        if blank:
            if previous_blank:
                continue
            result.append("")
        else:
            result.append(line.strip())

        previous_blank = blank

    return "\n".join(result).strip()


def canonical_problem_text(problem: dict[str, Any]) -> str:
    """
    Serialize only problem-defining fields in a stable order.

    Metadata such as rating, contest name, and source are intentionally
    excluded because the same underlying problem can have different metadata.
    """
    parts: list[str] = []

    for field in TEXT_FIELDS:
        value = normalize_text(problem.get(field))
        parts.append(f"[{field}]\n{value}")

    return "\n\n".join(parts)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_record(
    problem: dict[str, Any],
    taxonomy: dict[str, Any] | None,
) -> dict[str, Any]:
    text = canonical_problem_text(problem)
    exact_hash = sha256_text(text)

    tax = taxonomy or {}

    return {
        "problem_id": str(problem["id"]),
        "title": problem.get("title"),
        "rating": problem.get("rating"),
        "contest_id": problem.get("contest_id"),
        "index": problem.get("index"),

        "canonical": {
            "exact_hash": exact_hash,
            "text_length": len(text),
            "fields": list(TEXT_FIELDS),
        },

        "dp": {
            "primary_family": tax.get("primary_family"),
            "families": tax.get("families", []),
            "evidence": tax.get("dp_evidence_level"),
            "consensus": tax.get("dp_consensus"),
        },

        "source": problem.get("source"),
        "source_split": problem.get("source_split"),
    }


def build_duplicate_groups(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)

    for record in records:
        groups[record["canonical"]["exact_hash"]].append(
            record["problem_id"]
        )

    duplicate_groups: list[dict[str, Any]] = []

    for exact_hash, problem_ids in groups.items():
        if len(problem_ids) <= 1:
            continue

        duplicate_groups.append(
            {
                "exact_hash": exact_hash,
                "problem_ids": sorted(problem_ids),
                "count": len(problem_ids),
            }
        )

    duplicate_groups.sort(
        key=lambda x: (-x["count"], x["problem_ids"])
    )

    return duplicate_groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonicalize Codeforces problems and detect exact duplicates."
    )

    parser.add_argument(
        "--problems",
        default="data/raw/open_r1_codeforces_train.jsonl",
    )

    parser.add_argument(
        "--taxonomy",
        default="data/taxonomy/dp_taxonomy.jsonl",
    )

    parser.add_argument(
        "--output",
        default="data/deduplicated/problem_canonical.jsonl",
    )

    parser.add_argument(
        "--duplicates",
        default="data/deduplicated/exact_duplicate_groups.jsonl",
    )

    args = parser.parse_args()

    problems = read_jsonl(Path(args.problems))
    taxonomy_rows = read_jsonl(Path(args.taxonomy))

    taxonomy_by_problem = {
        str(row["problem_id"]): row
        for row in taxonomy_rows
    }

    records: list[dict[str, Any]] = []

    for problem in problems:
        problem_id = str(problem["id"])

        record = canonical_record(
            problem,
            taxonomy_by_problem.get(problem_id),
        )

        records.append(record)

    records.sort(key=lambda x: x["problem_id"])

    duplicate_groups = build_duplicate_groups(records)

    write_jsonl(Path(args.output), records)
    write_jsonl(Path(args.duplicates), duplicate_groups)

    unique_hashes = {
        record["canonical"]["exact_hash"]
        for record in records
    }

    duplicate_problem_count = sum(
        group["count"] - 1
        for group in duplicate_groups
    )

    print(json.dumps(
        {
            "input_problems": len(problems),
            "canonical_records": len(records),
            "unique_exact_hashes": len(unique_hashes),
            "exact_duplicate_groups": len(duplicate_groups),
            "duplicate_problem_count": duplicate_problem_count,
            "output": args.output,
            "duplicates": args.duplicates,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
