from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PASS_STATUSES = {
    "PASSED_AVAILABLE_TESTS",
    "PASSED_EXACT",
    "PASSED_CHECKER",
}

FAILURE_STATUSES = {
    "WRONG_ANSWER",
    "RE",
    "TLE",
    "CHECKER_TLE",
    "CHECKER_ERROR",
    "CE",
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def solution_quality(status: str) -> str:
    if status == "PASSED_EXACT":
        return "exact"
    if status == "PASSED_CHECKER":
        return "checker"
    if status == "PASSED_AVAILABLE_TESTS":
        return "available_tests"

    if status == "WRONG_ANSWER":
        return "wrong_answer"
    if status == "TLE":
        return "tle"
    if status == "RE":
        return "runtime_error"
    if status == "CE":
        return "compile_error"

    return "other"


def build_verified_dataset(
    verified_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
) -> None:
    verified = read_jsonl(verified_path)
    taxonomy = read_jsonl(taxonomy_path)

    taxonomy_by_problem = {
        row["problem_id"]: row
        for row in taxonomy
    }

    verified_rows: list[dict] = []
    failure_rows: list[dict] = []

    seen_verified: set[tuple[str, str]] = set()

    status_counts: Counter[str] = Counter()
    problem_counts: Counter[str] = Counter()

    for row in verified:
        problem_id = str(row["problem_id"])
        source_id = str(row["source_id"])
        status = str(row["status"])

        status_counts[status] += 1

        if status in PASS_STATUSES:
            key = (problem_id, source_id)

            if key in seen_verified:
                continue

            seen_verified.add(key)

            tax = taxonomy_by_problem.get(problem_id, {})

            record = {
                "problem_id": problem_id,
                "source_id": source_id,
                "language": row.get("language"),
                "source": row.get("source"),
                "source_split": row.get("source_split"),

                "source_code": row.get("source_code"),

                "verification": {
                    "status": status,
                    "quality": solution_quality(status),
                    "execution_time_ms": row.get("execution_time_ms"),
                    "verification_method": (
                        "checker"
                        if status == "PASSED_CHECKER"
                        else (
                            "exact"
                            if status == "PASSED_EXACT"
                            else "available_tests"
                        )
                    ),
                },

                "dp": {
                    "evidence": tax.get("dp_evidence_level"),
                    "consensus": tax.get("dp_consensus"),
                    "primary_family": tax.get("primary_family"),
                    "families": tax.get("families", []),
                },
            }

            verified_rows.append(record)
            problem_counts[problem_id] += 1

        elif status in FAILURE_STATUSES:
            failure_rows.append(
                {
                    "problem_id": problem_id,
                    "source_id": source_id,
                    "language": row.get("language"),
                    "source": row.get("source"),
                    "source_code": row.get("source_code"),
                    "status": status,
                    "execution_time_ms": row.get("execution_time_ms"),
                    "stderr": row.get("stderr"),
                    "stdout": row.get("stdout"),
                    "failure_type": solution_quality(status),
                }
            )

    verified_rows.sort(
        key=lambda x: (
            x["problem_id"],
            x["verification"]["quality"],
            x["source_id"],
        )
    )

    failure_rows.sort(
        key=lambda x: (
            x["problem_id"],
            x["failure_type"],
            x["source_id"],
        )
    )

    write_jsonl(
        output_dir / "verified_solutions.jsonl",
        verified_rows,
    )

    write_jsonl(
        output_dir / "failure_solutions.jsonl",
        failure_rows,
    )

    summary = {
        "verified_solutions": len(verified_rows),
        "verified_problems": len(problem_counts),
        "failure_solutions": len(failure_rows),
        "status_counts": dict(status_counts),
        "verification_quality": dict(
            Counter(
                row["verification"]["quality"]
                for row in verified_rows
            )
        ),
        "problems_with_verified_solutions": len(problem_counts),
    }

    with (output_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--verified",
        default="data/verified/verified_solutions.jsonl",
    )

    parser.add_argument(
        "--taxonomy",
        default="data/taxonomy/dp_taxonomy.jsonl",
    )

    parser.add_argument(
        "--output",
        default="data/deduplicated",
    )

    args = parser.parse_args()

    build_verified_dataset(
        verified_path=Path(args.verified),
        taxonomy_path=Path(args.taxonomy),
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
