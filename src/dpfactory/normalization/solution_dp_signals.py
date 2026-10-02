from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


INPUT_PATH = Path(
    "data/normalized/dp_solutions.jsonl"
)

OUTPUT_PATH = Path(
    "data/normalized/dp_solution_signals.jsonl"
)


PATTERNS: dict[str, str] = {
    "dp_array_name": (
        r"\b(?:dp|f|g|memo|mem)\s*(?:\[|\(|=)"
    ),
    "dp_vector": (
        r"\b(?:vector|array)\s*<[^>]+>\s+"
        r"(?:dp|f|g|memo|mem)\b"
    ),
    "dp_2d": (
        r"\b(?:dp|f|g|memo|mem)\s*"
        r"\[[^]]+\]\s*\[[^]]+\]"
    ),
    "memoization": (
        r"\b(?:memo|memoized|memoization|cache)\b"
    ),
    "recursion": (
        r"\b(?:dfs|solve|go|rec|calc|search)\s*"
        r"\([^;{}]*\)\s*\{"
    ),
    "transition": (
        r"\b(?:dp|f|g)\s*\[[^]]+\]\s*(?:=|\+=|-=)"
    ),
    "previous_state": (
        r"\b(?:dp|f|g)\s*\[[^]]+\].*"
        r"\b(?:dp|f|g)\s*\[[^]]+\]"
    ),
    "knapsack_pattern": (
        r"\b(?:weight|weights|capacity|knapsack|"
        r"\bw\b.*\bc\b)\b"
    ),
    "lis_pattern": (
        r"\b(?:lis|longest increasing|increasing subsequence)\b"
    ),
    "lcs_pattern": (
        r"\b(?:lcs|longest common subsequence)\b"
    ),
    "interval_pattern": (
        r"\b(?:interval|range|l\b.*r\b|left.*right)\b"
    ),
    "bitmask_pattern": (
        r"\b(?:mask|1\s*<<|1LL\s*<<)\b"
    ),
    "tree_dp_pattern": (
        r"\b(?:subtree|tree dp|reroot|children)\b"
    ),
    "digit_dp_pattern": (
        r"\b(?:digit dp|tight|leading.?zero)\b"
    ),
}


def normalize_code(code: str) -> str:
    return re.sub(r"\s+", " ", code.lower())


def find_signals(code: str) -> list[str]:
    text = normalize_code(code)

    signals: list[str] = []

    for name, pattern in PATTERNS.items():
        if re.search(pattern, text):
            signals.append(name)

    return signals


def score_signals(signals: list[str]) -> float:
    strong = {
        "dp_array_name",
        "dp_vector",
        "dp_2d",
        "memoization",
        "transition",
        "previous_state",
    }

    structural = {
        "recursion",
        "knapsack_pattern",
        "lis_pattern",
        "lcs_pattern",
        "interval_pattern",
        "tree_dp_pattern",
        "digit_dp_pattern",
    }

    score = 0.0

    for signal in signals:
        if signal in strong:
            score += 1.0
        elif signal in structural:
            score += 0.5
        elif signal == "bitmask_pattern":
            score += 0.25

    return min(score, 5.0)


def main() -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            row = json.loads(line)

            # Analyze accepted solutions only.
            if not row.get("is_accepted", False):
                continue

            code = str(row.get("source_code", ""))

            signals = find_signals(code)
            score = score_signals(signals)

            grouped[row["problem_id"]].append(
                {
                    "source_id": row.get("source_id"),
                    "language": row.get("language"),
                    "signals": signals,
                    "score": score,
                }
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    problem_count = 0

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        for problem_id, solutions in grouped.items():
            signal_counts = Counter()

            for solution in solutions:
                signal_counts.update(
                    solution["signals"]
                )

            best_score = max(
                float(x["score"])
                for x in solutions
            )

            output = {
                "problem_id": problem_id,
                "accepted_solution_count": len(
                    solutions
                ),
                "best_solution_score": best_score,
                "signal_counts": dict(
                    signal_counts
                ),
                "solutions": solutions,
            }

            f.write(
                json.dumps(
                    output,
                    ensure_ascii=False,
                )
                + "\n"
            )

            problem_count += 1

    print("=" * 70)
    print("DP SOLUTION SIGNAL ANALYSIS")
    print("=" * 70)
    print(
        f"Problems analyzed: "
        f"{problem_count:,}"
    )
    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
