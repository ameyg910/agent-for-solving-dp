from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


POOL_PATH = Path("data/deduplicated/dp_problem_pool.jsonl")
SOLUTIONS_PATH = Path("data/deduplicated/verified_solutions.jsonl")
OUTPUT_PATH = Path("data/deduplicated/dp_concepts.jsonl")
SUMMARY_PATH = Path("data/deduplicated/dp_concepts_summary.json")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_code(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    return code


def count_patterns(code: str, patterns: dict[str, list[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}

    for name, pats in patterns.items():
        total = 0
        for pattern in pats:
            total += len(re.findall(pattern, code, flags=re.MULTILINE))
        counts[name] = total

    return counts


def extract_code_signals(code: str) -> dict:
    code = normalize_code(code)

    patterns = {
        "array_dp": [
            r"\bvector\s*<[^;\n]+>\s+dp\b",
            r"\bvector\s*<[^;\n]+>\s+dp\s*\[",
            r"\b(?:long long|int|double)\s+dp\s*\[",
            r"\b(?:long long|int|double)\s+dp\d*\s*\[",
        ],
        "two_dimensional_dp": [
            r"\bdp\s*\[[^\]]+\]\s*\[[^\]]+\]",
            r"\bvector\s*<\s*vector\s*<[^>]+>\s*>\s+dp",
        ],
        "memoization": [
            r"\bmemo\b",
            r"\bmem\b",
            r"\bcache\b",
            r"\bunordered_map\s*<[^;]+>\s+memo",
            r"\bmap\s*<[^;]+>\s+memo",
        ],
        "recursive_dp": [
            r"\bdfs\s*\(",
            r"\bsolve\s*\(",
            r"\brec(?:ursive)?\s*\(",
        ],
        "previous_state": [
            r"\bdp\s*\[[^\]]+\]\s*=\s*[^;\n]*\bdp\s*\[",
            r"\bdp\s*\[[^\]]+\]\s*=\s*[^;\n]*\b(?:i\s*-\s*1|i\s*-\s*2|j\s*-\s*1|j\s*-\s*2)\b",
        ],
        "transition": [
            r"\bmax\s*\(",
            r"\bmin\s*\(",
            r"\bdp\s*\[[^\]]+\]\s*(?:\+|-|\*)=",
            r"\bdp\s*\[[^\]]+\]\s*=",
        ],
        "knapsack": [
            r"\bweight\b",
            r"\bvalue\b",
            r"\bcapacity\b",
            r"\bknapsack\b",
            r"\bfor\s*\([^)]*capacity",
        ],
        "bitmask": [
            r"\b(?:1\s*<<|1LL\s*<<)",
            r"\bmask\b",
            r"\bpopcount\b",
            r"__builtin_popcount",
        ],
        "interval": [
            r"\bl\b.*\br\b",
            r"\bleft\b",
            r"\bright\b",
            r"\binterval\b",
            r"\bdp\s*\[[^\]]+\]\s*\[[^\]]+\]",
        ],
        "tree_dp": [
            r"\bsubtree\b",
            r"\bparent\b",
            r"\bchildren\b",
            r"\badj\s*\[",
            r"\bvector\s*<[^>]+>\s+g\s*\[",
        ],
        "digit_dp": [
            r"\bdigit\b",
            r"\btight\b",
            r"\bleading[_ ]?zero",
        ],
        "string_dp": [
            r"\blcs\b",
            r"\blps\b",
            r"\bstring\b",
            r"\bs1\b",
            r"\bs2\b",
        ],
    }

    return count_patterns(code, patterns)


def infer_implementation(signals: dict[str, int]) -> list[str]:
    result: list[str] = []

    if signals["two_dimensional_dp"] > 0:
        result.append("2d_table")

    if signals["memoization"] > 0:
        result.append("memoization")

    if signals["recursive_dp"] > 0 and signals["memoization"] > 0:
        result.append("top_down")

    if signals["array_dp"] > 0 and signals["recursive_dp"] == 0:
        result.append("bottom_up_candidate")

    if signals["bitmask"] > 0:
        result.append("bitmask")

    return result


def aggregate_solution_signals(solutions: list[dict]) -> dict:
    signal_rows = []

    for solution in solutions:
        code = solution.get("source_code")
        if not code:
            continue

        signal_rows.append(extract_code_signals(code))

    if not signal_rows:
        return {
            "solution_count": len(solutions),
            "solutions_with_code": 0,
            "signal_counts": {},
            "signal_consensus": {},
            "implementation_patterns": [],
        }

    signal_names = sorted(
        {
            name
            for row in signal_rows
            for name in row
        }
    )

    signal_counts = {
        name: sum(row.get(name, 0) for row in signal_rows)
        for name in signal_names
    }

    signal_consensus = {
        name: sum(row.get(name, 0) > 0 for row in signal_rows)
        / len(signal_rows)
        for name in signal_names
    }

    implementation_counter: Counter[str] = Counter()

    for row in signal_rows:
        for pattern in infer_implementation(row):
            implementation_counter[pattern] += 1

    implementation_patterns = [
        pattern
        for pattern, count in implementation_counter.most_common()
        if count / len(signal_rows) >= 0.5
    ]

    return {
        "solution_count": len(solutions),
        "solutions_with_code": len(signal_rows),
        "signal_counts": signal_counts,
        "signal_consensus": signal_consensus,
        "implementation_patterns": implementation_patterns,
    }


def main() -> None:
    pool = load_jsonl(POOL_PATH)
    verified = load_jsonl(SOLUTIONS_PATH)

    solutions_by_problem: dict[str, list[dict]] = defaultdict(list)

    for solution in verified:
        solutions_by_problem[solution["problem_id"]].append(solution)

    output: list[dict] = []

    for problem in pool:
        problem_id = problem["problem_id"]
        solutions = solutions_by_problem.get(problem_id, [])

        aggregate = aggregate_solution_signals(solutions)

        output.append(
            {
                "problem_id": problem_id,
                "title": problem.get("title"),
                "rating": problem.get("rating"),
                "canonical_group_id": problem.get("canonical_group_id"),
                "canonical_hash": problem.get("canonical_hash"),
                "family": {
                    "primary": problem.get("primary_family"),
                    "families": problem.get("families", []),
                    "evidence_level": problem.get("dp_evidence_level"),
                    "consensus": problem.get("dp_consensus"),
                },
                "dimensions": problem.get("dimensions", []),
                "transition_style": problem.get("transition_style", []),
                "optimizations": problem.get("optimizations", []),
                "state_signals": problem.get("state_signals", []),
                "verification": {
                    "verified_solution_count": problem.get(
                        "verified_solution_count", 0
                    ),
                    "has_verified_solution": problem.get(
                        "has_verified_solution", False
                    ),
                },
                "code_analysis": aggregate,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in output:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    primary_families = Counter(
        row["family"]["primary"]
        for row in output
        if row["family"]["primary"] is not None
    )

    implementation_patterns = Counter(
        pattern
        for row in output
        for pattern in row["code_analysis"]["implementation_patterns"]
    )

    summary = {
        "total_dp_problems": len(output),
        "problems_with_verified_solutions": sum(
            row["verification"]["has_verified_solution"]
            for row in output
        ),
        "problems_with_code_analysis": sum(
            row["code_analysis"]["solutions_with_code"] > 0
            for row in output
        ),
        "primary_family_counts": dict(primary_families),
        "implementation_pattern_counts": dict(implementation_patterns),
        "output": str(OUTPUT_PATH),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
