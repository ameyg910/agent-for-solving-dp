from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


RAW_PATH = Path("data/raw/open_r1_codeforces_train.jsonl")
POOL_PATH = Path("data/deduplicated/dp_problem_pool.jsonl")
SOLUTIONS_PATH = Path("data/deduplicated/verified_solutions.jsonl")

OUTPUT_PATH = Path("data/deduplicated/dp_semantic_concepts.jsonl")
SUMMARY_PATH = Path("data/deduplicated/dp_semantic_concepts_summary.json")


SYSTEM_PROMPT = r"""
You are a competitive-programming DP expert.

Your task is to extract the actual dynamic-programming structure from a
problem statement and one verified accepted solution.

Do NOT merely classify based on keywords.

Do NOT infer concepts from variable names alone.

Do NOT call something memoization merely because the code contains words such
as mem, memo, cache, or memset.

Do NOT call something bitmask DP merely because the code contains <<, &, or
bit operations.

Do NOT call something string DP merely because the code manipulates strings.

The goal is to identify the mathematical computational structure actually
used by the solution.

Return ONLY valid JSON matching the requested schema.
"""


SCHEMA_DESCRIPTION = r"""
{
  "family": "null if dp_used is false; otherwise one of:
    bitmask, counting, digit, game, graph_dag, grid, interval, knapsack,
    probability, sequence, string, tree, other, unknown",

IMPORTANT DP GATE:
- First determine whether the solution actually uses dynamic programming.
- Set dp_used=true only when there is genuine DP/state-based reuse of subproblems.
- If dp_used=false, family MUST be null.
- If dp_used=false, state.variables MUST be [], state.meaning MUST be "",
  state.dimensions MUST be 0, and state.representation MUST be "other".
- If dp_used=false, transition.description MUST be "",
  transition.dependencies MUST be [], and transition.operation MUST be "unknown".
- Do not treat a scalar variable, modulo/residual state, loop index,
  greedy state, or mathematical recurrence as DP by itself.

  "dp_used": true or false,

  "state": {
    "variables": ["variable names or symbolic dimensions"],
    "meaning": "precise explanation of what one DP state represents",
    "dimensions": 0,
    "representation": "array | 2d_array | map | hash_map | recursion_cache |
                       scalar_state | other"
  },

  "transition": {
    "description": "mathematical description of how a state is obtained",
    "dependencies": ["previous states or dimensions"],
    "operation": "min | max | sum | probability_sum | boolean |
                  counting | custom | unknown"
  },

  "base_cases": [
    "precise base-case description"
  ],

  "answer": {
    "state": "which state represents the final answer",
    "aggregation": "if an aggregation over states is required"
  },

  "iteration": {
    "style": "bottom_up | top_down | mixed | state_machine | other",
    "order": "description of dependency/evaluation order"
  },

  "optimization": {
    "techniques": [
      "rolling_array | prefix_sum | monotonic_queue | bitmask |
       coordinate_compression | binary_search | matrix_exponentiation |
       divide_and_conquer | none | other"
    ],
    "description": "why the optimization is used"
  },

  "complexity": {
    "time": "Big-O expression",
    "space": "Big-O expression"
  },

  "confidence": "high | medium | low",

  "uncertainties": [
    "specific things that cannot be established confidently"
  ]
}
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def build_problem_text(problem: dict[str, Any]) -> str:
    fields = [
        ("Title", problem.get("title")),
        ("Problem statement", problem.get("description")),
        ("Input", problem.get("input_format")),
        ("Output", problem.get("output_format")),
        ("Interaction", problem.get("interaction_format")),
        ("Note", problem.get("note")),
    ]

    chunks: list[str] = []

    for name, value in fields:
        value = normalize_text(value)

        if value:
            chunks.append(f"{name}:\n{value}")

    return "\n\n".join(chunks)


def strip_markdown_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = strip_markdown_fences(text)

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")

    parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON is not an object")

    return parsed


def ollama_generate(
    *,
    host: str,
    model: str,
    prompt: str,
    timeout: int,
) -> str:
    url = host.rstrip("/") + "/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0,
        },
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))

    result = body.get("response", "")

    if not result.strip():
        raise RuntimeError("Ollama returned an empty response")
    
    return result


def build_prompt(
    *,
    problem_text: str,
    family: dict[str, Any],
    code: str,
) -> str:
    return f"""
Extract the actual dynamic-programming structure from this competitive-programming
problem.

Existing automated taxonomy evidence is provided below.
Treat it as evidence, NOT as unquestionable truth.

TAXONOMY:
{json.dumps(family, indent=2)}

PROBLEM:
{problem_text}

First determine whether the problem genuinely requires dynamic programming.

If dp_used=false:
- family must be null
- state.variables must be []
- state.meaning must be ""
- state.dimensions must be 0
- state.representation must be "other"
- transition.description must be ""
- transition.dependencies must be []
- transition.operation must be "unknown"

If dp_used=true:
- identify the actual DP state
- explain what each state dimension represents
- identify the transition and its dependencies
- identify base cases
- identify the answer state
- identify iteration style
- identify optimization techniques

Do not classify from keywords alone.
Do not invent a recurrence unsupported by the problem.
Distinguish DP from greedy algorithms, graph traversal,
ordinary preprocessing, and simple mathematical recurrences.

Return ONLY valid JSON matching this schema:

{SCHEMA_DESCRIPTION}
"""


def normalize_concept(concept: dict[str, Any]) -> dict[str, Any]:
    """
Keep the output schema stable even if the model omits fields.
"""
    state = concept.get("state")
    if not isinstance(state, dict):
        state = {}
    transition = concept.get("transition")
    if not isinstance(transition, dict):
        transition = {}

    answer = concept.get("answer")
    if not isinstance(answer, dict):
            answer = {}

    iteration = concept.get("iteration")
    if not isinstance(iteration, dict):
        iteration = {}

    optimization = concept.get("optimization")
    if not isinstance(optimization, dict):
        optimization = {}

    complexity = concept.get("complexity")
    if not isinstance(complexity, dict):
        complexity = {}

    variables = state.get("variables", [])
    if not isinstance(variables, list):
        variables = []

    dimensions = state.get("dimensions")
    if not isinstance(dimensions, int):
        dimensions = None

    dependencies = transition.get("dependencies", [])
    if not isinstance(dependencies, list):
        dependencies = []

    base_cases = concept.get("base_cases", [])
    if not isinstance(base_cases, list):
        base_cases = []

    uncertainties = concept.get("uncertainties", [])
    if not isinstance(uncertainties, list):
        uncertainties = []

    techniques = optimization.get("techniques", [])
    if not isinstance(techniques, list):
        techniques = []

    return {
        "family": concept.get("family", "unknown"),
        "dp_used": bool(concept.get("dp_used", False)),
        "state": {
            "variables": variables,
            "meaning": state.get("meaning", ""),
            "dimensions": dimensions,
            "representation": state.get("representation", "other"),
        },
        "transition": {
            "description": transition.get("description", ""),
            "dependencies": dependencies,
            "operation": transition.get("operation", "unknown"),
        },
        "base_cases": base_cases,
        "answer": {
            "state": answer.get("state", ""),
            "aggregation": answer.get("aggregation", ""),
        },
        "iteration": {
            "style": iteration.get("style", "other"),
            "order": iteration.get("order", ""),
        },
        "optimization": {
            "techniques": techniques,
            "description": optimization.get("description", ""),
        },
        "complexity": {
            "time": complexity.get("time", ""),
            "space": complexity.get("space", ""),
        },
        "confidence": concept.get("confidence", "low"),
        "uncertainties": uncertainties,
    }
def extract_one(
    *,
    problem_text: str,
    family: dict[str, Any],
    code: str,
    host: str,
    model: str,
    timeout: int,
    ) -> tuple[dict[str, Any] | None, str | None]:
    prompt = build_prompt(
    problem_text=problem_text,
    family=family,
    code=code,
    )
    try:
        response = ollama_generate(
            host=host,
            model=model,
            prompt=prompt,
            timeout=timeout,
        )

        raw = extract_json_object(response)
        concept = normalize_concept(raw)

        # Enforce the DP gate at the application layer.
        # Non-DP solutions must not carry DP-family/state information.
        if not concept["dp_used"]:
            concept["family"] = None
            concept["state"] = {
                "variables": [],
                "meaning": "",
                "dimensions": 0,
                "representation": "other",
            }
            concept["transition"] = {
                "description": "",
                "dependencies": [],
                "operation": "unknown",
            }

        return concept, None

    except (
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        ) as exc:
        return None, str(exc)
def concepts_agree(
    a: dict[str, Any],
    b: dict[str, Any],
) -> dict[str, Any]:
    """
    Conservative agreement check.

    First gate on whether both solutions agree that the problem uses DP.
    Only compare DP-specific semantics when both solutions say dp_used=True.
    """

    a_dp = bool(a.get("dp_used", False))
    b_dp = bool(b.get("dp_used", False))

    # Both solutions agree that this is NOT DP.
    if not a_dp and not b_dp:
        return {
            "level": "high",
            "score": 1.0,
            "checks": {
                "dp_used": True,
                "family": True,
            },
            "dp_classification": "non_dp",
        }

    # The two solutions disagree about whether DP is used.
    if a_dp != b_dp:
        return {
            "level": "low",
            "score": 0.0,
            "checks": {
                "dp_used": False,
            },
            "dp_classification": "disagreement",
        }

    # Both solutions say this is DP.
    checks: dict[str, bool] = {
        "dp_used": True,
        "family": a.get("family") == b.get("family"),
        "state_dimensions": (
            a.get("state", {}).get("dimensions")
            == b.get("state", {}).get("dimensions")
        ),
        "state_representation": (
            a.get("state", {}).get("representation")
            == b.get("state", {}).get("representation")
        ),
        "iteration_style": (
            a.get("iteration", {}).get("style")
            == b.get("iteration", {}).get("style")
        ),
        "transition_operation": (
            a.get("transition", {}).get("operation")
            == b.get("transition", {}).get("operation")
        ),
    }

    score = sum(checks.values()) / len(checks)

    if score >= 0.83:
        level = "high"
    elif score >= 0.50:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "score": score,
        "checks": checks,
        "dp_classification": "dp",
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default=os.environ.get(
            "OLLAMA_HOST",
            "http://127.0.0.1:11502",
        ),
    )

    parser.add_argument(
        "--model",
        default=os.environ.get(
            "OLLAMA_MODEL",
            "llama3.3:70b",
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--pool",
        type=Path,
        default=POOL_PATH,
    )
    args = parser.parse_args()

    raw_problems = load_jsonl(RAW_PATH)
    pool = load_jsonl(args.pool)
    verified = load_jsonl(SOLUTIONS_PATH)

    raw_by_id = {
        row["id"]: row
        for row in raw_problems
    }

    pool_by_id = {
        row["problem_id"]: row
        for row in pool
    }

    solutions_by_id: dict[str, list[dict[str, Any]]] = {}

    for solution in verified:
        solutions_by_id.setdefault(
            solution["problem_id"],
            [],
        ).append(solution)

    # Only DP candidates with verified code.
    candidate_ids = [
        problem_id
        for problem_id, problem in pool_by_id.items()
        if problem.get("has_verified_solution")
        and problem_id in solutions_by_id
    ]

    candidate_ids.sort()

    if args.limit is not None:
        candidate_ids = candidate_ids[
            args.start : args.start + args.limit
        ]
    else:
        candidate_ids = candidate_ids[args.start :]

    print(f"DP candidates with verified code: {len(candidate_ids)}")
    print(f"Ollama host: {args.host}")
    print(f"Ollama model: {args.model}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Append mode allows the job to be resumed.
    with OUTPUT_PATH.open("a", encoding="utf-8") as out:
        for index, problem_id in enumerate(candidate_ids, 1):
            problem = pool_by_id[problem_id]
            raw_problem = raw_by_id.get(problem_id)

            if raw_problem is None:
                print(
                    f"[{index}/{len(candidate_ids)}] "
                    f"{problem_id}: missing raw problem"
                )
                continue

            problem_text = build_problem_text(raw_problem)

            solutions = solutions_by_id[problem_id]

            # Prefer the first two verified implementations.
            selected = solutions[:2]

            concepts: list[dict[str, Any]] = []
            errors: list[str] = []

            for solution_index, solution in enumerate(selected, 1):
                print(
                    f"[{index}/{len(candidate_ids)}] "
                    f"{problem_id} "
                    f"solution {solution_index}/{len(selected)}",
                    flush=True,
                )

                concept, error = extract_one(
                    problem_text=problem_text,
                    family={
                        "primary": problem.get("primary_family"),
                        "families": problem.get("families", []),
                        "evidence_level": problem.get(
                            "dp_evidence_level"
                        ),
                        "consensus": problem.get(
                            "dp_consensus"
                        ),
                    },
                    code=solution["source_code"],
                    host=args.host,
                    model=args.model,
                    timeout=args.timeout,
                )

                if concept is not None:
                    concepts.append(concept)
                else:
                    errors.append(
                        f"solution_{solution_index}: {error}"
                    )

            row: dict[str, Any] = {
                "problem_id": problem_id,
                "title": problem.get("title"),
                "rating": problem.get("rating"),
                "family_evidence": {
                    "primary": problem.get("primary_family"),
                    "families": problem.get("families", []),
                    "evidence_level": problem.get(
                        "dp_evidence_level"
                    ),
                    "consensus": problem.get(
                        "dp_consensus"
                    ),
                },
                "verified_solution_count": len(solutions),
                "extractions": concepts,
                "agreement": None,
                "status": "error",
                "errors": errors,
            }

            if len(concepts) >= 2:
                agreement = concepts_agree(
                    concepts[0],
                    concepts[1],
                )

                row["agreement"] = agreement

                if agreement["level"] == "high":
                    row["status"] = "high_confidence"
                elif agreement["level"] == "medium":
                    row["status"] = "medium_confidence"
                else:
                    row["status"] = "disagreement"

            elif len(concepts) == 1:
                row["status"] = "single_extraction"

            out.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            out.flush()

            # Small pause to avoid hammering the local server.
            time.sleep(0.05)

    # Build summary from all output rows.
    output_rows = load_jsonl(OUTPUT_PATH)

    status_counts = Counter(
        row.get("status")
        for row in output_rows
    )

    agreement_levels = Counter(
        row["agreement"]["level"]
        for row in output_rows
        if row.get("agreement")
    )

    family_counts = Counter(
        row["family_evidence"].get("primary")
        for row in output_rows
    )

    summary = {
        "rows": len(output_rows),
        "status_counts": dict(status_counts),
        "agreement_levels": dict(agreement_levels),
        "family_counts": dict(family_counts),
        "output": str(OUTPUT_PATH),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
if __name__ == "__main__":
    main()


