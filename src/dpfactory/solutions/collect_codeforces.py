from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm


CANDIDATE_PATH = Path(
    "data/normalized/dp_candidates.jsonl"
)

OUTPUT_PATH = Path(
    "data/normalized/dp_solutions.jsonl"
)

MAX_ACCEPTED_PER_PROBLEM = 5
MAX_INCORRECT_PER_PROBLEM = 3

# Prefer languages useful for our eventual coding model.
LANGUAGE_PRIORITY = {
    "C++20 (GCC 13-64)": 0,
    "C++20 (GCC 11-64)": 1,
    "C++17 (GCC 9-64)": 2,
    "C++17 (GCC 7-32)": 3,
    "GNU C++17": 4,
    "GNU C++14": 5,
    "GNU C++11": 6,
    "GNU C++": 7,
    "Python 3": 10,
    "PyPy 3": 11,
    "Python 2": 12,
    "Java 21": 20,
    "Java 17": 21,
    "Java 11": 22,
    "Java 8": 23,
    "Go": 30,
}


def load_candidate_ids() -> set[str]:
    candidate_ids: set[str] = set()

    with CANDIDATE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            row = json.loads(line)
            candidate_ids.add(row["id"])

    return candidate_ids


def language_rank(language: str) -> int:
    return LANGUAGE_PRIORITY.get(language, 100)


def select_solutions(
    rows: list[dict[str, Any]],
    max_per_problem: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[row["problem_id"]].append(row)

    selected: list[dict[str, Any]] = []

    for problem_id, problem_rows in grouped.items():
        problem_rows.sort(
            key=lambda row: (
                language_rank(
                    str(row.get("programmingLanguage", ""))
                ),
                len(str(row.get("source", ""))),
                str(row.get("submission_id", "")),
            )
        )

        chosen: list[dict[str, Any]] = []
        languages_seen: set[str] = set()

        # First take language-diverse solutions.
        for row in problem_rows:
            language = str(
                row.get("programmingLanguage", "")
            )

            if language in languages_seen:
                continue

            chosen.append(row)
            languages_seen.add(language)

            if len(chosen) >= max_per_problem:
                break

        # Fill remaining slots with additional C++/Python/etc.
        if len(chosen) < max_per_problem:
            chosen_ids = {
                row["submission_id"]
                for row in chosen
            }

            for row in problem_rows:
                if row["submission_id"] in chosen_ids:
                    continue

                chosen.append(row)

                if len(chosen) >= max_per_problem:
                    break

        selected.extend(chosen)

    return selected


def convert_row(
    row: dict[str, Any],
    *,
    failure: bool,
) -> dict[str, Any]:
    return {
        "problem_id": str(row["problem_id"]),
        "language": str(
            row.get("programmingLanguage", "")
        ),
        "source_code": str(
            row.get("source", "")
        ),
        "verdict": str(
            row.get("verdict", "")
        ),
        "execution_time_ms": row.get(
            "timeConsumedMillis"
        ),
        "memory_bytes": row.get(
            "memoryConsumedBytes"
        ),
        "source": "open-r1/codeforces-submissions",
        "source_id": str(
            row.get("submission_id", "")
        ),
        "is_accepted": not failure,
        "is_failure_example": failure,
        "extra": {
            "contest_id": row.get("contestId"),
            "problem_index": row.get(
                "problem_index"
            ),
            "testset": row.get("testset"),
            "passed_test_count": row.get(
                "passedTestCount"
            ),
            "creation_time_seconds": row.get(
                "creationTimeSeconds"
            ),
        },
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading DP candidate IDs...")
    candidate_ids = load_candidate_ids()

    print(
        f"DP candidate problems: "
        f"{len(candidate_ids):,}"
    )

    print()
    print(
        "Loading curated accepted "
        "Codeforces submissions..."
    )

    accepted_ds = load_dataset(
        "open-r1/codeforces-submissions",
        name="selected_accepted",
        split="train",
    )

    accepted_rows: list[dict[str, Any]] = []

    for row in tqdm(
        accepted_ds,
        desc="Filtering accepted",
    ):
        if row["problem_id"] in candidate_ids:
            accepted_rows.append(dict(row))

    print(
        f"Accepted rows matching DP candidates: "
        f"{len(accepted_rows):,}"
    )

    print()
    print(
        "Loading curated incorrect "
        "Codeforces submissions..."
    )

    incorrect_ds = load_dataset(
        "open-r1/codeforces-submissions",
        name="selected_incorrect",
        split="train",
    )

    incorrect_rows: list[dict[str, Any]] = []

    for row in tqdm(
        incorrect_ds,
        desc="Filtering incorrect",
    ):
        if row["problem_id"] in candidate_ids:
            incorrect_rows.append(dict(row))

    print(
        f"Incorrect rows matching DP candidates: "
        f"{len(incorrect_rows):,}"
    )

    selected_accepted = select_solutions(
        accepted_rows,
        MAX_ACCEPTED_PER_PROBLEM,
    )

    selected_incorrect = select_solutions(
        incorrect_rows,
        MAX_INCORRECT_PER_PROBLEM,
    )

    print()
    print(
        f"Selected accepted: "
        f"{len(selected_accepted):,}"
    )

    print(
        f"Selected incorrect: "
        f"{len(selected_incorrect):,}"
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in selected_accepted:
            f.write(
                json.dumps(
                    convert_row(
                        row,
                        failure=False,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )

        for row in selected_incorrect:
            f.write(
                json.dumps(
                    convert_row(
                        row,
                        failure=True,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 70)
    print("CODEFORCES SOLUTION COLLECTION")
    print("=" * 70)
    print(
        f"DP candidate problems: "
        f"{len(candidate_ids):,}"
    )
    print(
        f"Selected accepted:     "
        f"{len(selected_accepted):,}"
    )
    print(
        f"Selected incorrect:     "
        f"{len(selected_incorrect):,}"
    )
    print(
        f"Total solutions:        "
        f"{len(selected_accepted) + len(selected_incorrect):,}"
    )
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
