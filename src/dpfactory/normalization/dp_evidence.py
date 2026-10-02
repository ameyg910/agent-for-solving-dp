from __future__ import annotations

import json
from pathlib import Path


CANDIDATE_PATH = Path(
    "data/normalized/dp_candidates.jsonl"
)

SIGNAL_PATH = Path(
    "data/normalized/dp_solution_signals.jsonl"
)

OUTPUT_PATH = Path(
    "data/normalized/dp_evidence.jsonl"
)


STRONG_SIGNALS = {
    "dp_array_name",
    "dp_vector",
    "dp_2d",
    "transition",
    "previous_state",
    "memoization",
}

MODERATE_SIGNALS = {
    "knapsack_pattern",
    "lis_pattern",
    "lcs_pattern",
    "tree_dp_pattern",
    "digit_dp_pattern",
}

WEAK_SIGNALS = {
    "recursion",
    "interval_pattern",
    "bitmask_pattern",
}


def classify_solution(
    signals: list[str],
    score: float,
) -> str:
    """Classify one accepted solution by DP evidence strength."""

    strong_count = sum(
        signal in STRONG_SIGNALS
        for signal in signals
    )

    moderate_count = sum(
        signal in MODERATE_SIGNALS
        for signal in signals
    )

    weak_count = sum(
        signal in WEAK_SIGNALS
        for signal in signals
    )

    # Strong structural evidence.
    if strong_count >= 2:
        return "strong"

    if strong_count == 1 and score >= 1.0:
        return "strong"

    # Moderate DP-specific evidence.
    if moderate_count >= 1:
        return "moderate"

    # One weak signal by itself is not enough.
    if weak_count >= 1:
        return "weak"

    return "none"


def problem_evidence(
    *,
    dp_tag: bool,
    editorial: bool,
    solutions: list[dict],
) -> tuple[str, float]:
    """Determine problem-level evidence using independent solutions."""

    if not solutions:
        if dp_tag or editorial:
            return "metadata_only", 0.0

        return "uncertain", 0.0

    classifications = [
        classify_solution(
            solution.get("signals", []),
            float(solution.get("score", 0.0)),
        )
        for solution in solutions
    ]

    total = len(classifications)

    strong = classifications.count("strong")
    moderate = classifications.count("moderate")
    weak = classifications.count("weak")

    consensus = strong / total

    # Multiple independent implementations agree on DP structure.
    if strong >= 3 and consensus >= 0.5:
        return "very_strong", consensus

    if strong >= 2 and consensus >= 0.4:
        return "strong", consensus

    if strong >= 1 and (
        moderate >= 1 or dp_tag or editorial
    ):
        return "moderate", consensus

    if moderate >= 2:
        return "moderate", consensus

    if weak >= 2:
        return "weak", consensus

    if dp_tag or editorial:
        return "metadata_only", consensus

    return "uncertain", consensus


def main() -> None:
    candidates: dict[str, dict] = {}

    with CANDIDATE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            row = json.loads(line)
            candidates[row["id"]] = row

    solution_evidence: dict[str, dict] = {}

    with SIGNAL_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            row = json.loads(line)
            solution_evidence[row["problem_id"]] = row

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts: dict[str, int] = {}

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as out:

        for problem_id, candidate in candidates.items():

            solution = solution_evidence.get(problem_id)

            if solution is None:
                solutions: list[dict] = []
                accepted_count = 0
                best_solution_score = 0.0
                aggregate_signals: dict[str, int] = {}

            else:
                solutions = solution.get("solutions", [])
                accepted_count = int(
                    solution.get(
                        "accepted_solution_count",
                        0,
                    )
                )
                best_solution_score = float(
                    solution.get(
                        "best_solution_score",
                        0.0,
                    )
                )
                aggregate_signals = solution.get(
                    "signal_counts",
                    {},
                )

            dp_tag = bool(
                candidate.get("dp_tag", False)
            )

            editorial = bool(
                candidate.get(
                    "editorial_dp_evidence"
                )
            )

            level, consensus = problem_evidence(
                dp_tag=dp_tag,
                editorial=editorial,
                solutions=solutions,
            )

            classifications = [
                classify_solution(
                    solution_row.get("signals", []),
                    float(solution_row.get("score", 0.0)),
                )
                for solution_row in solutions
            ]

            strong_count = classifications.count("strong")
            moderate_count = classifications.count("moderate")
            weak_count = classifications.count("weak")

            result = {
                "problem_id": problem_id,
                "title": candidate.get(
                    "title",
                    "",
                ),
                "rating": candidate.get(
                    "rating"
                ),

                "dp_tag": dp_tag,

                "editorial_evidence": candidate.get(
                    "editorial_dp_evidence",
                    [],
                ),

                "metadata_evidence": candidate.get(
                    "metadata_evidence",
                    [],
                ),

                "accepted_solution_count": accepted_count,

                "strong_solution_count": strong_count,
                "moderate_solution_count": moderate_count,
                "weak_solution_count": weak_count,

                "dp_consensus": round(
                    consensus,
                    3,
                ),

                "signal_counts": aggregate_signals,

                "best_solution_score": best_solution_score,

                "evidence_level": level,
            }

            out.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            counts[level] = (
                counts.get(level, 0) + 1
            )

    print("=" * 70)
    print("DP EVIDENCE AGGREGATION — CONSENSUS VERSION")
    print("=" * 70)

    print(
        f"Candidates analyzed: {len(candidates):,}"
    )

    print(
        f"Problems with solutions: "
        f"{len(solution_evidence):,}"
    )

    print()
    print("Evidence levels:")

    for level in [
        "very_strong",
        "strong",
        "moderate",
        "weak",
        "metadata_only",
        "uncertain",
    ]:
        print(
            f"  {level:14s} "
            f"{counts.get(level, 0):,}"
        )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
