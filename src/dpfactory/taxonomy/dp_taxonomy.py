from __future__ import annotations

import json
import re
from pathlib import Path


CANDIDATE_PATH = Path(
    "data/normalized/dp_candidates.jsonl"
)

EVIDENCE_PATH = Path(
    "data/normalized/dp_evidence.jsonl"
)

OUTPUT_PATH = Path(
    "data/taxonomy/dp_taxonomy.jsonl"
)


# ---------------------------------------------------------------------------
# Family detection
#
# Important:
# These rules are intentionally conservative.
# Generic words such as "range", "left", "right", "ways", "path", etc.
# are NOT sufficient evidence for a DP family.
# ---------------------------------------------------------------------------

FAMILY_PATTERNS: dict[str, list[str]] = {
    "knapsack": [
        r"\bknapsack\b",
        r"\bsubset sum\b",
        r"\bsubset-sum\b",
        r"\bbounded knapsack\b",
        r"\bunbounded knapsack\b",
        r"\b0[/ -]?1 knapsack\b",
        r"\bweight\b.{0,80}\bcapacity\b",
        r"\bcapacity\b.{0,80}\bweight\b",
    ],

    "sequence": [
        r"\blongest increasing subsequence\b",
        r"\blongest decreasing subsequence\b",
        r"\bincreasing subsequence\b",
        r"\bdecreasing subsequence\b",
        r"\blis\b",
        r"\blnds\b",
        r"\blcs\b",
        r"\blongest common subsequence\b",
    ],

    "string": [
        r"\bstring\b",
        r"\bstrings\b",
        r"\bsubstring\b",
        r"\bpalindrome\b",
        r"\bpalindromic\b",
        r"\blongest common subsequence\b",
        r"\blcs\b",
        r"\bedit distance\b",
        r"\bstring matching\b",
    ],

    "interval": [
        r"\binterval dp\b",
        r"\brange dp\b",
        r"\binterval dynamic programming\b",
        r"\bdp\s*\[\s*l\s*\]\s*\[\s*r\s*\]",
        r"\bdp\s*\[\s*l\s*\]\s*\[\s*r\s*[+-]",
        r"\bmerge\b.{0,80}\bintervals?\b",
        r"\bpartition\b.{0,80}\binterval\b",
        r"\bpartition\b.{0,80}\bsegment\b",
        r"\bmatrix chain\b",
        r"\boptimal binary search tree\b",
    ],

    "tree": [
        r"\btree dp\b",
        r"\btree dynamic programming\b",
        r"\bsubtree\b",
        r"\breroot(?:ing)?\b",
        r"\bon the tree\b",
        r"\brooted tree\b",
        r"\bchildren of\b",
        r"\bparent of\b",
    ],

    "graph_dag": [
        r"\bdp on (?:a )?dag\b",
        r"\bdag dp\b",
        r"\bdirected acyclic graph\b",
        r"\bdirected acyclic\b",
        r"\btopological sort\b",
        r"\btopological ordering\b",
        r"\btoposort\b",
        r"\bDAG\b",
    ],

    "bitmask": [
        r"\bbitmask dp\b",
        r"\bbitmask dynamic programming\b",
        r"\bmask dp\b",
        r"\bsubset dp\b",
        r"\bSOS DP\b",
        r"\bsum over subsets\b",
        r"\b1\s*<<\s*\w+",
        r"\b1LL\s*<<\s*\w+",
    ],

    "digit": [
        r"\bdigit dp\b",
        r"\bdigit dynamic programming\b",
        r"\btight\b.{0,80}\bdigit\b",
        r"\bdigits?\b.{0,80}\btight\b",
        r"\bleading zeros?\b.{0,80}\bdp\b",
    ],

    "game": [
        r"\bgame dp\b",
        r"\bgame dynamic programming\b",
        r"\bwinning position\b",
        r"\blosing position\b",
        r"\bwinning state\b",
        r"\bgrundy\b",
        r"\bsprague[- ]grundy\b",
        r"\bnim\b",
    ],

    "probability": [
        r"\bprobability dp\b",
        r"\bprobability dynamic programming\b",
        r"\bexpected value\b",
        r"\bexpected number\b",
        r"\bexpected cost\b",
        r"\bexpectation dp\b",
        r"\bprobability\b.{0,100}\bdp\b",
    ],

    "grid": [
        r"\bgrid dp\b",
        r"\bgrid dynamic programming\b",
        r"\bmatrix dp\b",
        r"\bmatrix dynamic programming\b",
        r"\brows?\b.{0,60}\bcolumns?\b.{0,60}\bdp\b",
        r"\bgrid\b.{0,100}\bdp\b",
    ],

    "counting": [
        r"\bcounting dp\b",
        r"\bcounting dynamic programming\b",
        r"\bnumber of ways\b.{0,80}\bdp\b",
        r"\bways\b.{0,80}\bdp\b",
        r"\bcount\b.{0,80}\bdp\b",
        r"\bcombinatorial dp\b",
    ],
}


# ---------------------------------------------------------------------------
# Optimization detection
# ---------------------------------------------------------------------------

OPTIMIZATION_PATTERNS: dict[str, list[str]] = {
    "binary_search": [
        r"\bbinary search\b",
        r"\bparametric search\b",
    ],

    "prefix_sum": [
        r"\bprefix sum\b",
        r"\bprefix sums\b",
        r"\bcumulative sum\b",
    ],

    "monotonic_queue": [
        r"\bmonotonic queue\b",
        r"\bmonotonic deque\b",
        r"\bsliding window\b",
    ],

    "segment_tree": [
        r"\bsegment tree\b",
        r"\bsegtree\b",
    ],

    "fenwick_tree": [
        r"\bfenwick tree\b",
        r"\bbinary indexed tree\b",
    ],

    "divide_conquer_optimization": [
        r"\bdivide and conquer optimization\b",
        r"\bdivide[- ]and[- ]conquer dp\b",
        r"\bd&c optimization\b",
    ],

    "convex_hull_trick": [
        r"\bconvex hull trick\b",
        r"\bcht\b",
    ],

    "knuth_optimization": [
        r"\bknuth optimization\b",
        r"\bknuth dp\b",
    ],
}


STATE_PATTERNS: dict[str, list[str]] = {
    "position": [
        r"\bposition\b",
        r"\bindex\b",
        r"\bidx\b",
    ],

    "capacity": [
        r"\bcapacity\b",
        r"\bbudget\b",
    ],

    "previous_state": [
        r"\bprevious state\b",
        r"\bstate transition\b",
    ],

    "color": [
        r"\bcolor\b",
        r"\bcolors\b",
    ],

    "mask": [
        r"\bbitmask\b",
        r"\bmask dp\b",
    ],

    "last_element": [
        r"\blast element\b",
        r"\bprevious element\b",
        r"\blast value\b",
        r"\bprevious value\b",
    ],

    "left_right": [
        r"\binterval dp\b",
        r"\brange dp\b",
        r"\bdp\s*\[\s*l\s*\]\s*\[\s*r\s*\]",
    ],
}


def compile_patterns(
    patterns: dict[str, list[str]],
) -> dict[str, list[re.Pattern[str]]]:
    return {
        name: [
            re.compile(
                pattern,
                re.IGNORECASE,
            )
            for pattern in regexes
        ]
        for name, regexes in patterns.items()
    }


FAMILY_REGEX = compile_patterns(FAMILY_PATTERNS)
OPTIMIZATION_REGEX = compile_patterns(
    OPTIMIZATION_PATTERNS
)
STATE_REGEX = compile_patterns(
    STATE_PATTERNS
)


def matching_categories(
    text: str,
    patterns: dict[str, list[re.Pattern[str]]],
) -> list[str]:
    matches: list[str] = []

    for category, regexes in patterns.items():
        if any(regex.search(text) for regex in regexes):
            matches.append(category)

    return matches


# ---------------------------------------------------------------------------
# Accepted-code evidence
# ---------------------------------------------------------------------------

def solution_signal_families(
    signal_counts: dict[str, int],
) -> list[str]:
    families: list[str] = []

    if signal_counts.get(
        "knapsack_pattern",
        0,
    ) >= 1:
        families.append("knapsack")

    if signal_counts.get(
        "lis_pattern",
        0,
    ) >= 1:
        families.append("sequence")

    if signal_counts.get(
        "lcs_pattern",
        0,
    ) >= 1:
        families.append("string")

    # IMPORTANT:
    # interval_pattern alone is too noisy.
    #
    # Require multiple occurrences before treating it as family evidence.
    if signal_counts.get(
        "interval_pattern",
        0,
    ) >= 3 and (
        signal_counts.get("dp_2d", 0) >= 1
        or signal_counts.get("previous_state", 0) >= 1
    ):
        families.append("interval")

    if signal_counts.get(
        "tree_dp_pattern",
        0,
    ) >= 1:
        families.append("tree")

    if signal_counts.get(
        "digit_dp_pattern",
        0,
    ) >= 1:
        families.append("digit")

    if signal_counts.get(
        "bitmask_pattern",
        0,
    ) >= 2:
        families.append("bitmask")

    return families


def solution_optimizations(
    signal_counts: dict[str, int],
) -> list[str]:
    result: list[str] = []

    # Bitmask is represented as an optimization/technique too.
    if signal_counts.get(
        "bitmask_pattern",
        0,
    ) >= 2:
        result.append("bitmask")

    return result


# ---------------------------------------------------------------------------
# Dimensions / transitions
# ---------------------------------------------------------------------------

def infer_dimensions(
    *,
    signal_counts: dict[str, int],
    families: list[str],
) -> list[str]:

    dimensions: list[str] = []

    if signal_counts.get(
        "dp_2d",
        0,
    ) >= 1:
        dimensions.append("2d")

    if "grid" in families:
        dimensions.append("grid")

    if "bitmask" in families:
        dimensions.append("mask")

    if "digit" in families:
        dimensions.append("digit")

    if not dimensions:
        dimensions.append("1d_or_unknown")

    return dimensions


def infer_transition_style(
    signal_counts: dict[str, int],
) -> list[str]:

    transitions: list[str] = []

    if signal_counts.get(
        "transition",
        0,
    ) > 0:
        transitions.append(
            "explicit_transition"
        )

    if signal_counts.get(
        "previous_state",
        0,
    ) > 0:
        transitions.append(
            "previous_state_dependency"
        )

    if signal_counts.get(
        "recursion",
        0,
    ) > 0:
        transitions.append(
            "recursive"
        )

    if signal_counts.get(
        "memoization",
        0,
    ) > 0:
        transitions.append(
            "memoized"
        )

    if not transitions:
        transitions.append("unknown")

    return transitions


# ---------------------------------------------------------------------------
# Family confidence
# ---------------------------------------------------------------------------

def family_confidence(
    *,
    family: str,
    text_match: bool,
    code_match: bool,
    code_signal_count: int,
    dp_consensus: float,
    evidence_level: str,
) -> float:

    score = 0.0

    # Strongest evidence: multiple accepted implementations.
    if code_match:
        score += 0.45

        if code_signal_count >= 5:
            score += 0.20
        elif code_signal_count >= 3:
            score += 0.15
        elif code_signal_count >= 1:
            score += 0.08

    # Problem text / editorial evidence.
    if text_match:
        score += 0.30

    # Independent implementation consensus.
    score += 0.20 * min(
        dp_consensus,
        1.0,
    )

    # DP evidence itself matters, but only slightly.
    if evidence_level == "very_strong":
        score += 0.05
    elif evidence_level == "strong":
        score += 0.03

    return round(
        min(score, 0.99),
        3,
    )


def main() -> None:

    candidates: dict[str, dict] = {}

    with CANDIDATE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            row = json.loads(line)
            candidates[row["id"]] = row

    evidence: dict[str, dict] = {}

    with EVIDENCE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            row = json.loads(line)
            evidence[row["problem_id"]] = row

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    family_counts: dict[str, int] = {}

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as out:

        for problem_id, candidate in candidates.items():

            ev = evidence.get(
                problem_id,
                {},
            )

            title = candidate.get(
                "title",
                "",
            )

            description = candidate.get(
                "description",
                "",
            )

            tags = candidate.get(
                "tags",
                [],
            )

            text = " ".join(
                [
                    title,
                    description,
                    " ".join(tags),
                    " ".join(
                        ev.get(
                            "editorial_evidence",
                            [],
                        )
                    ),
                ]
            )

            text_families = matching_categories(
                text,
                FAMILY_REGEX,
            )

            text_optimizations = matching_categories(
                text,
                OPTIMIZATION_REGEX,
            )

            state_signals = matching_categories(
                text,
                STATE_REGEX,
            )

            signal_counts = ev.get(
                "signal_counts",
                {},
            )

            code_families = solution_signal_families(
                signal_counts,
            )

            code_optimizations = solution_optimizations(
                signal_counts,
            )

            families = sorted(
                set(
                    text_families
                    + code_families
                )
            )

            optimizations = sorted(
                set(
                    text_optimizations
                    + code_optimizations
                )
            )

            dimensions = infer_dimensions(
                signal_counts=signal_counts,
                families=families,
            )

            transition_style = infer_transition_style(
                signal_counts,
            )

            dp_consensus = float(
                ev.get(
                    "dp_consensus",
                    0.0,
                )
            )

            evidence_level = ev.get(
                "evidence_level",
                "uncertain",
            )

            family_evidence: dict[str, dict] = {}

            for family in families:

                text_match = (
                    family in text_families
                )

                code_match = (
                    family in code_families
                )

                signal_name = (
                    f"{family}_pattern"
                )

                code_signal_count = int(
                    signal_counts.get(
                        signal_name,
                        0,
                    )
                )

                confidence = family_confidence(
                    family=family,
                    text_match=text_match,
                    code_match=code_match,
                    code_signal_count=code_signal_count,
                    dp_consensus=dp_consensus,
                    evidence_level=evidence_level,
                )

                family_evidence[family] = {
                    "text": text_match,
                    "accepted_code": code_match,
                    "code_signal_count": code_signal_count,
                    "confidence": confidence,
                }

                family_counts[family] = (
                    family_counts.get(
                        family,
                        0,
                    )
                    + 1
                )

            # -----------------------------------------------------------
            # Primary family
            #
            # Select the family with strongest evidence.
            # Do not force a primary family if every family is weak.
            # -----------------------------------------------------------

            ranked_families = sorted(
                families,
                key=lambda family: (
                    family_evidence[family][
                        "confidence"
                    ],
                    family_evidence[family][
                        "code_signal_count"
                    ],
                ),
                reverse=True,
            )

            if ranked_families:
                primary_family = (
                    ranked_families[0]
                )

                primary_confidence = (
                    family_evidence[
                        primary_family
                    ]["confidence"]
                )

                if primary_confidence < 0.45:
                    primary_family = None
            else:
                primary_family = None

            secondary_families = [
                family
                for family in ranked_families
                if family != primary_family
                and family_evidence[family][
                    "confidence"
                ] >= 0.45
            ]

            result = {
                "problem_id": problem_id,
                "title": title,
                "rating": candidate.get(
                    "rating"
                ),

                "primary_family": primary_family,

                "secondary_families": (
                    secondary_families
                ),

                "families": families,

                "dimensions": dimensions,

                "transition_style": (
                    transition_style
                ),

                "optimizations": optimizations,

                "state_signals": state_signals,

                "family_evidence": (
                    family_evidence
                ),

                "dp_evidence_level": (
                    evidence_level
                ),

                "dp_consensus": (
                    dp_consensus
                ),

                "accepted_solution_count": (
                    ev.get(
                        "accepted_solution_count",
                        0,
                    )
                ),

                "strong_solution_count": (
                    ev.get(
                        "strong_solution_count",
                        0,
                    )
                ),

                "tags": tags,

                "source": candidate.get(
                    "source",
                    "open-r1/codeforces",
                ),
            }

            out.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=" * 70)
    print("DP TAXONOMY EXTRACTION — CONSERVATIVE VERSION")
    print("=" * 70)

    print(
        f"Problems classified: "
        f"{len(candidates):,}"
    )

    print()
    print("Family counts:")

    for family, count in sorted(
        family_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"  {family:30s} "
            f"{count:,}"
        )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
