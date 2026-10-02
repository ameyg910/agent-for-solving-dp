from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from dpfactory.verification.compiler import (
    compile_solution,
    normalize_language,
)
from dpfactory.verification.runner import run_solution


PROBLEMS_PATH = Path("data/raw/open_r1_codeforces_train.jsonl")
SOLUTIONS_PATH = Path("data/normalized/dp_solutions.jsonl")
OUTPUT_PATH = Path("data/verified/verified_solutions.jsonl")

MAX_PROBLEMS = 20
MAX_SOLUTIONS_PER_PROBLEM = 2


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def verify_solution(
    solution: dict,
    problem: dict,
    work_dir: Path,
) -> dict:
    language = solution["language"]

    normalized_language = normalize_language(language)

    if normalized_language is None:
        return {
            "problem_id": solution["problem_id"],
            "source_id": solution.get("source_id"),
            "language": language,
            "status": "UNSUPPORTED_LANGUAGE",
            "tests_passed": 0,
            "tests_total": 0,
        }

    compile_dir = work_dir / str(solution.get("source_id", "solution"))

    try:
        compiled, executable, compile_error = compile_solution(
            source_code=solution["source_code"],
            language=language,
            work_dir=compile_dir,
        )

        if not compiled:
            return {
                "problem_id": solution["problem_id"],
                "source_id": solution.get("source_id"),
                "language": language,
                "status": "CE",
                "tests_passed": 0,
                "tests_total": 0,
                "compile_error": compile_error,
            }

        official_tests = problem.get("official_tests", [])

        if not official_tests:
            return {
                "problem_id": solution["problem_id"],
                "source_id": solution.get("source_id"),
                "language": language,
                "status": "UNVERIFIED",
                "tests_passed": 0,
                "tests_total": 0,
            }

        passed = 0

        # Codeforces time limits are generally per test.
        # Add a small safety multiplier for local execution.
        time_limit = float(problem.get("time_limit", 2.0))
        timeout_seconds = max(2.0, time_limit * 2.0)

        for test in official_tests:
            result = run_solution(
                executable=executable,
                input_data=test["input"],
                expected_output=test["output"],
                timeout_seconds=timeout_seconds,
                python=normalized_language == "python",
            )

            if result.status != "PASS":
                return {
                    "problem_id": solution["problem_id"],
                    "source_id": solution.get("source_id"),
                    "language": language,
                    "status": result.status,
                    "tests_passed": passed,
                    "tests_total": len(official_tests),
                    "execution_time_ms": result.execution_time_ms,
                    "stderr": result.stderr[-2000:],
                }

            passed += 1

        if problem.get("official_tests_complete", False):
            status = "PASSED_COMPLETE_TESTS"
        else:
            status = "PASSED_AVAILABLE_TESTS"

        return {
            "problem_id": solution["problem_id"],
            "source_id": solution.get("source_id"),
            "language": language,
            "status": status,
            "tests_passed": passed,
            "tests_total": len(official_tests),
        }

    finally:
        # Keep the verification dataset clean.
        shutil.rmtree(compile_dir, ignore_errors=True)


def main() -> None:
    problems = load_jsonl(PROBLEMS_PATH)
    solutions = load_jsonl(SOLUTIONS_PATH)

    problems_by_id = {
        problem["id"]: problem
        for problem in problems
    }

    # Only use problems for which we actually have executable tests.
    selected_problem_ids = [
        problem["id"]
        for problem in problems
        if problem.get("input_mode") == "stdio"
        and problem.get("official_tests")
    ][:MAX_PROBLEMS]

    selected_problem_ids_set = set(selected_problem_ids)

    selected_solutions: list[dict] = []

    counts: Counter[str] = Counter()

    for solution in solutions:
        problem_id = solution["problem_id"]

        if problem_id not in selected_problem_ids_set:
            continue

        if not solution.get("is_accepted", False):
            continue

        if normalize_language(solution["language"]) is None:
            continue

        counts[problem_id] += 1

        if counts[problem_id] <= MAX_SOLUTIONS_PER_PROBLEM:
            selected_solutions.append(solution)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []

    with tempfile.TemporaryDirectory(prefix="dp_verify_") as temp_dir:
        work_dir = Path(temp_dir)

        for index, solution in enumerate(selected_solutions, start=1):
            problem = problems_by_id[solution["problem_id"]]

            print(
                f"[{index}/{len(selected_solutions)}] "
                f"{solution['problem_id']} "
                f"{solution['language']}"
            )

            result = verify_solution(
                solution=solution,
                problem=problem,
                work_dir=work_dir,
            )

            results.append(result)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    status_counts = Counter(result["status"] for result in results)

    print()
    print("Verification complete")
    print(f"Solutions checked: {len(results)}")
    print(f"Output: {OUTPUT_PATH}")
    print()
    print("Statuses:")

    for status, count in status_counts.most_common():
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
