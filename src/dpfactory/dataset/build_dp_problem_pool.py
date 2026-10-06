from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TAXONOMY = Path("data/taxonomy/dp_taxonomy.jsonl")
GROUPS = Path("data/deduplicated/problem_groups.jsonl")
RELATIONSHIPS = Path("data/deduplicated/problem_relationships.jsonl")
CANONICAL = Path("data/deduplicated/problem_canonical.jsonl")
VERIFIED = Path("data/deduplicated/verified_solutions.jsonl")

OUTPUT = Path("data/deduplicated/dp_problem_pool.jsonl")
SUMMARY = Path("data/deduplicated/dp_problem_pool_summary.json")


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


def main() -> None:
    taxonomy_rows = read_jsonl(TAXONOMY)
    group_rows = read_jsonl(GROUPS)
    relationship_rows = read_jsonl(RELATIONSHIPS)
    canonical_rows = read_jsonl(CANONICAL)
    verified_rows = read_jsonl(VERIFIED)

    taxonomy = {
        str(row["problem_id"]): row
        for row in taxonomy_rows
    }

    canonical = {
        str(row["problem_id"]): row
        for row in canonical_rows
    }

    # ------------------------------------------------------------------
    # Canonical/leakage groups
    # ------------------------------------------------------------------

    group_by_problem: dict[str, dict[str, Any]] = {}

    for row in group_rows:
        pid = str(row["problem_id"])
        group_by_problem[pid] = row

    # ------------------------------------------------------------------
    # Verified solution counts
    # ------------------------------------------------------------------

    verified_count: dict[str, int] = defaultdict(int)

    verification_statuses: dict[str, set[str]] = defaultdict(set)

    for row in verified_rows:
        pid = str(row["problem_id"])
        verified_count[pid] += 1

        verification = row.get("verification", {})
        status = verification.get("status")

        if status:
            verification_statuses[pid].add(str(status))

    # ------------------------------------------------------------------
    # Relationship metadata
    #
    # IMPORTANT:
    # Relationships do NOT merge problems.
    # They are only metadata for later filtering/splitting.
    # ------------------------------------------------------------------

    relationships_by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in relationship_rows:
        a = str(row["problem_a"])
        b = str(row["problem_b"])

        relationships_by_problem[a].append(row)
        relationships_by_problem[b].append(row)

    # ------------------------------------------------------------------
    # Version-family metadata
    # ------------------------------------------------------------------

    version_family_members: dict[str, set[str]] = defaultdict(set)

    for row in relationship_rows:
        if row.get("relationship") != "version_family":
            continue

        a = str(row["problem_a"])
        b = str(row["problem_b"])

        version_family_members[a].add(b)
        version_family_members[b].add(a)

    # ------------------------------------------------------------------
    # Build one row per DP candidate.
    #
    # We intentionally use taxonomy as the definition of the DP pool.
    # ------------------------------------------------------------------

    pool: list[dict[str, Any]] = []

    for pid, tax in taxonomy.items():
        group = group_by_problem.get(pid)
        can = canonical.get(pid)

        if group is None:
            raise RuntimeError(
                f"DP problem {pid} has no problem_groups entry"
            )

        if can is None:
            raise RuntimeError(
                f"DP problem {pid} has no canonical entry"
            )

        canonical_group_id = str(
            group["canonical_group_id"]
        )

        relationships = relationships_by_problem.get(pid, [])

        relationship_flags = sorted(
            {
                str(r["relationship"])
                for r in relationships
                if r.get("relationship")
            }
        )

        version_members = sorted(
            version_family_members.get(pid, set())
        )

        pool.append(
            {
                "problem_id": pid,

                # Exact canonical identity.
                "canonical_group_id": canonical_group_id,

                # Independent problem vs exact duplicate group.
                "group_type": group["group_type"],

                # Useful provenance.
                "base_problem_id": group.get("base_problem_id"),

                # Canonical hash is retained for auditing.
                "canonical_hash": can["canonical"]["exact_hash"],

                # Problem metadata.
                "title": can.get("title"),
                "rating": can.get("rating"),
                "contest_id": can.get("contest_id"),
                "index": can.get("index"),

                # DP taxonomy.
                "primary_family": tax.get("primary_family"),
                "families": tax.get("families", []),
                "dimensions": tax.get("dimensions", []),
                "transition_style": tax.get(
                    "transition_style", []
                ),
                "optimizations": tax.get(
                    "optimizations", []
                ),
                "state_signals": tax.get(
                    "state_signals", []
                ),

                "dp_evidence_level": tax.get(
                    "dp_evidence_level"
                ),
                "dp_consensus": tax.get(
                    "dp_consensus"
                ),

                # Verified solution availability.
                "verified_solution_count": verified_count.get(
                    pid, 0
                ),
                "verification_statuses": sorted(
                    verification_statuses.get(pid, set())
                ),
                "has_verified_solution": (
                    verified_count.get(pid, 0) > 0
                ),

                # Relationship metadata.
                "relationship_flags": relationship_flags,
                "version_family_members": version_members,

                # Number of relationships involving this problem.
                "relationship_count": len(relationships),

                "source": can.get("source"),
                "source_split": can.get("source_split"),
            }
        )

    pool.sort(key=lambda x: x["problem_id"])

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    canonical_groups: dict[str, list[str]] = defaultdict(list)

    for row in pool:
        canonical_groups[
            row["canonical_group_id"]
        ].append(row["problem_id"])

    duplicate_pool_groups = {
        gid: pids
        for gid, pids in canonical_groups.items()
        if len(pids) > 1
    }

    family_counts: dict[str, int] = defaultdict(int)
    evidence_counts: dict[str, int] = defaultdict(int)

    for row in pool:
        family = row["primary_family"]

        if family is not None:
            family_counts[str(family)] += 1

        evidence = row["dp_evidence_level"]

        if evidence is not None:
            evidence_counts[str(evidence)] += 1

    summary = {
        "total_dp_candidates": len(pool),

        "unique_canonical_groups": len(canonical_groups),

        "exact_duplicate_groups_inside_dp_pool": len(
            duplicate_pool_groups
        ),

        "dp_problems_inside_exact_duplicate_groups": sum(
            len(v)
            for v in duplicate_pool_groups.values()
        ),

        "dp_candidates_with_verified_solution": sum(
            1
            for row in pool
            if row["has_verified_solution"]
        ),

        "dp_candidates_without_verified_solution": sum(
            1
            for row in pool
            if not row["has_verified_solution"]
        ),

        "primary_family_counts": dict(
            sorted(family_counts.items())
        ),

        "evidence_level_counts": dict(
            sorted(evidence_counts.items())
        ),

        "relationship_flag_counts": {
            flag: sum(
                1
                for row in pool
                if flag in row["relationship_flags"]
            )
            for flag in sorted(
                {
                    flag
                    for row in pool
                    for flag in row["relationship_flags"]
                }
            )
        },

        "output": str(OUTPUT),
    }

    write_jsonl(OUTPUT, pool)

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
