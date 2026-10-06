import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


CANDIDATES = Path("data/deduplicated/near_duplicate_candidates.jsonl")
CANONICAL = Path("data/deduplicated/problem_canonical.jsonl")

OUT_REL = Path("data/deduplicated/problem_relationships.jsonl")
OUT_GROUPS = Path("data/deduplicated/problem_groups.jsonl")


def base_id(problem_id: str) -> str:
    """
    1356/A1 -> 1356/A
    1356/A2 -> 1356/A
    153/B  -> 153/B
    """
    return re.sub(r"([A-Z])\d+$", r"\1", problem_id)


def canonical_hash(record: dict) -> str:
    """
    Reconstruct the canonical statement representation and hash it.
    Prefer the hash already stored by problem_canonicalization.py.
    """
    for key in ("canonical_hash", "content_hash", "hash"):
        value = record.get(key)
        if value:
            return str(value)

    fields = [
        record.get("title", ""),
        record.get("description", ""),
        record.get("input_format", ""),
        record.get("output_format", ""),
        record.get("interaction_format", ""),
        record.get("note", ""),
    ]

    text = "\n".join(str(x) for x in fields)

    return str(record["canonical"]["exact_hash"])


def main() -> None:
    canonical = {}

    with CANONICAL.open() as f:
        for line in f:
            row = json.loads(line)
            pid = row.get("problem_id") or row.get("id")
            if pid:
                canonical[pid] = row

    print(f"Loaded canonical records: {len(canonical):,}")

    exact_hash_groups = defaultdict(list)

    for pid, row in canonical.items():
        exact_hash_groups[canonical_hash(row)].append(pid)

    # Only hashes with >1 problem are actual exact-duplicate candidates.
    exact_groups = {
        h: sorted(pids)
        for h, pids in exact_hash_groups.items()
        if len(pids) > 1
    }

    print(f"Exact duplicate hash groups: {len(exact_groups)}")

    hash_to_group = {}

    for i, (h, pids) in enumerate(sorted(exact_groups.items()), start=1):
        group_id = f"exact_{i:04d}"
        for pid in pids:
            hash_to_group[pid] = group_id

    relationships = []

    with CANDIDATES.open() as f:
        for line in f:
            r = json.loads(line)

            a = r["problem_a"]
            b = r["problem_b"]
            score = float(r["similarity"]["score"])

            if a not in canonical or b not in canonical:
                continue

            a_hash = canonical_hash(canonical[a])
            b_hash = canonical_hash(canonical[b])

            same_hash = a_hash == b_hash
            same_base = base_id(a) == base_id(b)

            if same_hash:
                relationship = "exact_duplicate"

            elif same_base:
                relationship = "version_family"

            elif score >= 0.95:
                relationship = "reissue_candidate"

            elif score >= 0.90:
                relationship = "related_candidate"

            else:
                relationship = "low_confidence_candidate"

            relationships.append({
                "problem_a": a,
                "problem_b": b,
                "similarity_score": score,
                "relationship": relationship,
                "same_base_id": same_base,
                "same_canonical_hash": same_hash,
                "canonical_group_a": hash_to_group.get(a),
                "canonical_group_b": hash_to_group.get(b),
                "title_a": r.get("title_a"),
                "title_b": r.get("title_b"),
            })

    relationships.sort(
        key=lambda x: (
            x["relationship"],
            -x["similarity_score"],
            x["problem_a"],
            x["problem_b"],
        )
    )

    OUT_REL.parent.mkdir(parents=True, exist_ok=True)

    with OUT_REL.open("w") as f:
        for row in relationships:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Build one group record for every problem.
    groups = []

    for pid in sorted(canonical):
        exact_group = hash_to_group.get(pid)

        if exact_group is not None:
            group_type = "exact_duplicate_group"
        else:
            group_type = "independent_problem"

        groups.append({
            "problem_id": pid,
            "canonical_group_id": exact_group or f"problem_{pid}",
            "group_type": group_type,
            "base_problem_id": base_id(pid),
        })

    with OUT_GROUPS.open("w") as f:
        for row in groups:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = defaultdict(int)

    for r in relationships:
        counts[r["relationship"]] += 1

    print()
    print("Relationship counts:")
    for key in sorted(counts):
        print(f"  {key:25} {counts[key]:6,}")

    print()
    print(f"Wrote: {OUT_REL}")
    print(f"Wrote: {OUT_GROUPS}")

    print()
    print("Exact duplicate groups:")

    for group_id, pids in sorted(
        (
            (gid, [pid for pid, x in hash_to_group.items() if x == gid])
            for gid in set(hash_to_group.values())
        ),
        key=lambda x: x[0],
    ):
        print(f"  {group_id}: {' '.join(sorted(pids))}")


if __name__ == "__main__":
    main()
