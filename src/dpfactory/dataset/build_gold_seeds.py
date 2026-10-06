from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


INPUT = Path("data/deduplicated/dp_problem_pool.jsonl")
OUTPUT = Path("data/deduplicated/gold_dp_seeds.jsonl")
SUMMARY = Path("data/deduplicated/gold_dp_seeds_summary.json")


HIGH_CONFIDENCE = {
    "very_strong",
    "strong",
    "moderate",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

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


def main() -> None:
    rows = read_jsonl(INPUT)

    seeds = []

    for row in rows:
        evidence = row.get("dp_evidence_level")

        if evidence not in HIGH_CONFIDENCE:
            continue

        if not row.get("has_verified_solution", False):
            continue

        family = row.get("primary_family")

        if family is None:
            role = "training_only_unresolved_family"
        else:
            role = "generation_seed"

        if evidence == "very_strong":
            quality_tier = "A"
        elif evidence == "strong":
            quality_tier = "B"
        else:
            quality_tier = "C"

        seed = {
            "problem_id": row["problem_id"],
            "canonical_group_id": row["canonical_group_id"],
            "canonical_hash": row["canonical_hash"],

            "title": row["title"],
            "rating": row["rating"],

            "primary_family": family,
            "families": row.get("families", []),

            "dimensions": row.get("dimensions", []),
            "transition_style": row.get(
                "transition_style", []
            ),
            "optimizations": row.get(
                "optimizations", []
            ),
            "state_signals": row.get(
                "state_signals", []
            ),

            "dp_evidence_level": evidence,
            "dp_consensus": row.get("dp_consensus"),

            "verified_solution_count": row[
                "verified_solution_count"
            ],
            "verification_statuses": row[
                "verification_statuses"
            ],

            "quality_tier": quality_tier,
            "role": role,

            "relationship_flags": row.get(
                "relationship_flags", []
            ),
            "version_family_members": row.get(
                "version_family_members", []
            ),

            "source": row.get("source"),
            "source_split": row.get("source_split"),
        }

        seeds.append(seed)

    seeds.sort(
        key=lambda x: (
            x["role"],
            x["primary_family"] or "unresolved",
            -x["rating"] if x["rating"] is not None else 0,
            x["problem_id"],
        )
    )

    write_jsonl(OUTPUT, seeds)

    role_counts = Counter(
        row["role"] for row in seeds
    )

    tier_counts = Counter(
        row["quality_tier"] for row in seeds
    )

    family_counts = Counter(
        row["primary_family"] or "unresolved"
        for row in seeds
    )

    summary = {
        "total_gold_seeds": len(seeds),
        "role_counts": dict(sorted(role_counts.items())),
        "quality_tiers": dict(sorted(tier_counts.items())),
        "family_counts": dict(
            sorted(family_counts.items())
        ),
        "generation_seed_count": role_counts[
            "generation_seed"
        ],
        "unresolved_training_seed_count": role_counts[
            "training_only_unresolved_family"
        ],
        "output": str(OUTPUT),
    }

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    with SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
