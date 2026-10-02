from __future__ import annotations

import json
import re
from pathlib import Path

from tqdm import tqdm

from dpfactory.models import DPCandidate


RAW_PATH = Path("data/raw/open_r1_codeforces_train.jsonl")
OUTPUT_PATH = Path("data/normalized/dp_candidates.jsonl")


# Strong evidence that an editorial is discussing DP.
DP_EDITORIAL_PATTERNS: dict[str, str] = {
    "dynamic programming": "dynamic programming",
    "dynamic-programming": "dynamic-programming",
    "dp solution": "dp solution",
    "dp state": "dp state",
    "dp transition": "dp transition",
    "dp recurrence": "dp recurrence",
    "dynamic programming approach": "dynamic programming approach",
    "memoization": "memoization",
    "state transition": "state transition",
    "recurrence": "recurrence",
    "subproblem": "subproblem",
}


# These are weaker signals. They should never independently
# turn a problem into a high-confidence DP problem.
WEAK_METADATA_TAGS = {
    "bitmasks",
    "probabilities",
    "games",
    "trees",
    "graphs",
    "combinatorics",
    "strings",
    "matrices",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def editorial_evidence(editorial: str | None) -> list[str]:
    if not editorial:
        return []

    text = normalize_text(editorial)

    return [
        label
        for pattern, label in DP_EDITORIAL_PATTERNS.items()
        if pattern in text
    ]


def metadata_evidence(tags: list[str]) -> list[str]:
    return [
        f"tag:{tag}"
        for tag in tags
        if tag in WEAK_METADATA_TAGS
    ]


def score_candidate(
    *,
    dp_tag: bool,
    editorial_signals: list[str],
    metadata_signals: list[str],
) -> float:
    score = 0.0

    # Explicit Codeforces DP tag is our strongest automatic signal.
    if dp_tag:
        score += 1.0

    # Editorial evidence is useful for discovering untagged DP.
    score += min(0.45, 0.15 * len(editorial_signals))

    # Weak metadata evidence only contributes slightly.
    score += min(0.15, 0.03 * len(metadata_signals))

    return min(score, 1.5)


def assign_tier(
    *,
    dp_tag: bool,
    editorial_signals: list[str],
    score: float,
) -> str:
    if dp_tag:
        return "seed"

    if editorial_signals:
        return "editorial_candidate"

    if score >= 0.15:
        return "weak_candidate"

    return "reject"


def extract_candidates() -> tuple[int, int, int]:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    candidates = 0
    seeds = 0

    with (
        RAW_PATH.open("r", encoding="utf-8") as src,
        OUTPUT_PATH.open("w", encoding="utf-8") as dst,
    ):
        for line in tqdm(src, desc="Extracting DP candidates"):
            row = json.loads(line)
            total += 1

            tags = row.get("tags", [])
            dp_tag = "dp" in tags

            editorial_signals = editorial_evidence(
                row.get("editorial")
            )

            metadata_signals = metadata_evidence(tags)

            score = score_candidate(
                dp_tag=dp_tag,
                editorial_signals=editorial_signals,
                metadata_signals=metadata_signals,
            )

            tier = assign_tier(
                dp_tag=dp_tag,
                editorial_signals=editorial_signals,
                score=score,
            )

            if tier == "reject":
                continue

            candidate = DPCandidate(
                id=row["id"],
                title=row.get("title", ""),
                description=row.get("description", ""),
                tags=tags,
                rating=row.get("rating"),
                has_editorial=bool(row.get("editorial")),
                executable=bool(row.get("executable")),
                official_tests_complete=bool(
                    row.get("official_tests_complete")
                ),
                dp_tag=dp_tag,
                editorial_dp_evidence=editorial_signals,
                metadata_evidence=metadata_signals,
                heuristic_score=score,
                tier=tier,
                source="open-r1/codeforces",
                source_split=row.get("source_split", "train"),
            )

            dst.write(
                json.dumps(
                    candidate.model_dump(),
                    ensure_ascii=False,
                )
                + "\n"
            )

            candidates += 1

            if dp_tag:
                seeds += 1

    return total, candidates, seeds


def main() -> None:
    total, candidates, seeds = extract_candidates()

    print()
    print("=" * 70)
    print("DP CANDIDATE EXTRACTION")
    print("=" * 70)
    print(f"Input problems:       {total:,}")
    print(f"DP candidates:        {candidates:,}")
    print(f"Explicit DP seeds:    {seeds:,}")
    print(f"Output:               {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
