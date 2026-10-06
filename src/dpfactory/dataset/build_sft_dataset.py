from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

POOL = ROOT / "data/deduplicated/dp_problem_pool.jsonl"
VERIFIED = ROOT / "data/verified/verified_solutions.jsonl"
RAW = ROOT / "data/raw/open_r1_codeforces_train.jsonl"

OUTPUT = ROOT / "data/training"

SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_statement(row: dict) -> str:
    parts = []

    title = row.get("title")
    description = row.get("description")
    input_format = row.get("input_format")
    output_format = row.get("output_format")
    interaction_format = row.get("interaction_format")
    note = row.get("note")

    if title:
        parts.append(f"Title:\n{title}")

    if description:
        parts.append(f"Problem:\n{description}")

    if input_format:
        parts.append(f"Input:\n{input_format}")

    if output_format:
        parts.append(f"Output:\n{output_format}")

    if interaction_format:
        parts.append(
            f"Interaction:\n{interaction_format}"
        )

    if note:
        parts.append(f"Note:\n{note}")

    return "\n\n".join(parts).strip()


def main() -> None:
    print("Loading DP pool...")
    pool_rows = load_jsonl(POOL)

    pool_by_id = {
        row["problem_id"]: row
        for row in pool_rows
    }

    print(f"DP pool: {len(pool_by_id)}")

    print("Loading verified solutions...")
    verified_rows = load_jsonl(VERIFIED)

    # The verified file has no is_accepted field.
    # Its status/verdict fields are the result of our verifier.
    accepted = defaultdict(list)

    for row in verified_rows:
        problem_id = row.get("problem_id")

        if not problem_id:
            continue

        if problem_id not in pool_by_id:
            continue

        if row.get("verdict") != "OK":
            continue

        if row.get("status") not in {
            "PASSED_AVAILABLE_TESTS",
            "PASSED_EXACT",
            "PASSED_CHECKER",
        }:
            continue

        source_code = row.get("source_code")

        if not isinstance(source_code, str):
            continue

        if not source_code.strip():
            continue

        accepted[problem_id].append(row)

    print(
        "Problems with verified accepted solutions:",
        len(accepted),
    )

    print(
        "Verified accepted solutions:",
        sum(len(v) for v in accepted.values()),
    )

    print("Loading raw Codeforces problems...")
    raw_rows = load_jsonl(RAW)

    raw_by_id = {}
    with Path("data/raw/open_r1_codeforces_train.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            problem_id = row.get("id")
            if problem_id:
                raw_by_id[str(problem_id)] = row

    print(f"Raw problems: {len(raw_by_id)}")

    usable = []

    for problem_id in accepted:
        if problem_id not in raw_by_id:
            continue

        statement = build_statement(
            raw_by_id[problem_id]
        )

        if not statement:
            continue

        usable.append(problem_id)

    print(
        "Usable verified DP problems:",
        len(usable),
    )

    if not usable:
        raise RuntimeError(
            "No usable verified DP problems found."
        )

    # ------------------------------------------------------------
    # Problem-level split.
    #
    # Multiple solutions for the same problem always remain
    # in the same split.
    # ------------------------------------------------------------

    rng = random.Random(SEED)

    rng.shuffle(usable)

    n = len(usable)

    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    train_ids = set(
        usable[:n_train]
    )

    val_ids = set(
        usable[n_train:n_train + n_val]
    )

    test_ids = set(
        usable[n_train + n_val:]
    )

    print()
    print("Problem split:")
    print(f"  train: {len(train_ids)}")
    print(f"  val:   {len(val_ids)}")
    print(f"  test:  {len(test_ids)}")

    def make_examples(
        problem_ids: set[str],
    ) -> list[dict]:

        examples = []

        for problem_id in sorted(problem_ids):
            pool = pool_by_id[problem_id]
            raw = raw_by_id[problem_id]

            statement = build_statement(raw)

            family = pool.get("primary_family")

            if not family:
                families = pool.get("families", [])
                family = (
                    families[0]
                    if families
                    else None
                )

            for solution in accepted[problem_id]:

                examples.append(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "Solve the following "
                                    "dynamic programming "
                                    "problem. Provide a "
                                    "correct and efficient "
                                    "solution.\n\n"
                                    + statement
                                ),
                            },
                            {
                                "role": "assistant",
                                "content": solution[
                                    "source_code"
                                ],
                            },
                        ],
                        "metadata": {
                            "problem_id": problem_id,
                            "family": family,
                            "rating": pool.get("rating"),
                            "language": solution.get(
                                "language"
                            ),
                            "source_id": solution.get(
                                "source_id"
                            ),
                        },
                    }
                )

        rng.shuffle(examples)

        return examples

    train = make_examples(train_ids)
    val = make_examples(val_ids)
    test = make_examples(test_ids)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    write_jsonl(
        OUTPUT / "train.jsonl",
        train,
    )

    write_jsonl(
        OUTPUT / "val.jsonl",
        val,
    )

    write_jsonl(
        OUTPUT / "test.jsonl",
        test,
    )

    summary = {
        "seed": SEED,
        "problems": {
            "total": len(usable),
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "examples": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "total": (
                len(train)
                + len(val)
                + len(test)
            ),
        },
        "verified_statuses": {
            status: sum(
                1
                for row in verified_rows
                if row.get("status") == status
            )
            for status in sorted(
                {
                    row.get("status")
                    for row in verified_rows
                }
            )
        },
    }

    with (
        OUTPUT / "dataset_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print("SFT DATASET BUILT")
    print("=" * 60)
    print(f"Train examples:      {len(train)}")
    print(f"Validation examples: {len(val)}")
    print(f"Test examples:       {len(test)}")
    print(
        f"Total examples:      "
        f"{len(train) + len(val) + len(test)}"
    )
    print()
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
