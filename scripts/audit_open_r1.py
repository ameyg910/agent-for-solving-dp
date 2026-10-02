from __future__ import annotations

import json
from collections import Counter

PATH = "data/raw/open_r1_codeforces_train.jsonl"


def main() -> None:
    total = 0
    executable = 0
    with_editorial = 0
    complete_tests = 0

    tag_counts: Counter[str] = Counter()
    rating_counts: Counter[int | None] = Counter()

    dp_problems: list[str] = []

    with open(PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            total += 1

            tags = row.get("tags", [])
            tag_counts.update(tags)
            rating_counts[row.get("rating")] += 1

            if row.get("executable"):
                executable += 1

            if row.get("editorial"):
                with_editorial += 1

            if row.get("official_tests_complete"):
                complete_tests += 1

            if "dp" in tags:
                dp_problems.append(row["id"])

    print("=" * 70)
    print("OPEN-R1 CODEFORCES RAW DATASET AUDIT")
    print("=" * 70)

    print(f"\nTotal problems:           {total:,}")
    print(f"Executable:               {executable:,}")
    print(f"With editorial:           {with_editorial:,}")
    print(f"Complete official tests:  {complete_tests:,}")
    print(f"DP-tagged:                {len(dp_problems):,}")

    print("\nTop 40 tags")
    print("-" * 50)

    for tag, count in tag_counts.most_common(40):
        print(f"{tag:35} {count:>7,}")

    print("\nRating distribution")
    print("-" * 50)

    for rating, count in sorted(
        rating_counts.items(),
        key=lambda x: (x[0] is None, x[0] or 0),
    ):
        print(f"{str(rating):>10} : {count:>7,}")

    print("\nFirst 30 DP problems")
    print("-" * 50)

    for problem_id in dp_problems[:30]:
        print(problem_id)


if __name__ == "__main__":
    main()
