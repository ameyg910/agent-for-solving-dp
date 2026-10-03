from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from dpfactory.verification.checker import run_checker
from dpfactory.verification.compiler import (
    compile_solution,
    normalize_language,
)
from dpfactory.verification.runner import run_solution


PROBLEMS_PATH = Path("data/raw/open_r1_codeforces_train.jsonl")
SOLUTIONS_PATH = Path("data/normalized/dp_solutions.jsonl")
OUTPUT_PATH = Path("data/verified/verified_solutions.jsonl")

MAX_PROBLEMS = 1000
MAX_SOLUTIONS_PER_PROBLEM = 2
RANDOM_SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def output_is_case_insensitive(problem: dict) -> bool:
    output_format = str(problem.get("output_format", ""))
    text = output_format.lower()

    patterns = (
        "any case",
        "case does not matter",
        "case doesn't matter",
        "case-insensitive",
        "case insensitive",
        "uppercase or lowercase",
        "upper or lower case",
    )

    return any(pattern in text for pattern in patterns)

def pypy_available() -> bool:
    return subprocess.run(
        ["which", "pypy3"],
        capture_output=True,
        text=True,
    ).returncode == 0


def choose_solutions(
    candidates: list[dict],
    limit: int,
) -> list[dict]:
    """
    Prefer language diversity.

    First choose one solution from each available language,
    then fill remaining slots deterministically.
    """
    by_language: dict[str, list[dict]] = defaultdict(list)

    for solution in candidates:
        normalized = normalize_language(solution["language"])

        if normalized is None:
            continue

        if normalized == "pypy" and not pypy_available():
            continue

        by_language[normalized].append(solution)

    for values in by_language.values():
        values.sort(
            key=lambda x: str(x.get("source_id", ""))
        )

    selected: list[dict] = []

    for language in sorted(by_language):
        if len(selected) >= limit:
            break

        selected.append(by_language[language][0])

    remaining: list[dict] = []

    for language in sorted(by_language):
        remaining.extend(by_language[language][1:])

    remaining.sort(
        key=lambda x: (
            normalize_language(x["language"]) or "",
            str(x.get("source_id", "")),
        )
    )

    selected.extend(
        remaining[: max(0, limit - len(selected))]
    )

    return selected


def verify_solution(
    solution: dict,
    problem: dict,
    work_dir: Path,
) -> dict:
    language = solution["language"]
    normalized = normalize_language(language)

    base_result = {
        "problem_id": solution["problem_id"],
        "source_id": solution.get("source_id"),
        "language": language,
        "verdict": solution.get("verdict"),
        "source": solution.get("source"),
    }

    if normalized is None:
        return {
            **base_result,
            "status": "UNSUPPORTED_LANGUAGE",
            "tests_passed": 0,
            "tests_total": 0,
        }

    if normalized == "pypy" and not pypy_available():
        return {
            **base_result,
            "status": "PYTHON_INTERPRETER_UNAVAILABLE",
            "tests_passed": 0,
            "tests_total": 0,
        }

    compile_dir = work_dir / str(
        solution.get("source_id", "solution")
    )

    try:
        compiled, executable, compile_error = compile_solution(
            source_code=solution["source_code"],
            language=language,
            work_dir=compile_dir,
        )

        if not compiled:
            # A submission can be accepted by the original
            # dataset while depending on a non-standard header
            # or unavailable local file.
            if (
                "No such file or directory" in compile_error
                or "file not found" in compile_error.lower()
            ):
                return {
                    **base_result,
                    "status": "UNSUPPORTED_DEPENDENCY",
                    "tests_passed": 0,
                    "tests_total": 0,
                    "compile_error": compile_error,
                }

            return {
                **base_result,
                "status": "CE",
                "tests_passed": 0,
                "tests_total": 0,
                "compile_error": compile_error,
            }

        official_tests = problem.get("official_tests", [])

        if not official_tests:
            return {
                **base_result,
                "status": "UNVERIFIED",
                "tests_passed": 0,
                "tests_total": 0,
            }

        time_limit = float(
            problem.get("time_limit", 2.0)
        )

        timeout_seconds = max(
            2.0,
            time_limit * 2.0,
        )

        if normalized == "python":
            interpreter = "python3"
        elif normalized == "pypy":
            interpreter = "pypy3"
        else:
            interpreter = None

        has_checker = bool(
            problem.get("generated_checker")
        )

        passed = 0
        execution_times: list[float] = []

        for test in official_tests:
            result = run_solution(
                executable=executable,
                input_data=test["input"],
                expected_output=test["output"],
                timeout_seconds=timeout_seconds,
                interpreter=interpreter,
                case_insensitive=output_is_case_insensitive(problem),
            )

            execution_times.append(
                result.execution_time_ms
            )

            if result.status == "TLE":
                return {
                    **base_result,
                    "status": "TLE",
                    "tests_passed": passed,
                    "tests_total": len(official_tests),
                    "execution_time_ms": (
                        result.execution_time_ms
                    ),
                    "stderr": result.stderr[-2000:],
                    "stdout": result.stdout[-2000:],
                }

            if result.status == "RE":
                return {
                    **base_result,
                    "status": "RE",
                    "tests_passed": passed,
                    "tests_total": len(official_tests),
                    "execution_time_ms": (
                        result.execution_time_ms
                    ),
                    "stderr": result.stderr[-2000:],
                    "stdout": result.stdout[-2000:],
                }

            if has_checker:
                checker_result = run_checker(
                    checker_source=problem[
                        "generated_checker"
                    ],
                    input_data=test["input"],
                    reference_output=test["output"],
                    submission_output=result.stdout,
                    timeout_seconds=5.0,
                )

                if checker_result.status == "CHECKER_TLE":
                    return {
                        **base_result,
                        "status": "CHECKER_TLE",
                        "tests_passed": passed,
                        "tests_total": len(
                            official_tests
                        ),
                        "checker_stdout": (
                            checker_result.stdout[-2000:]
                        ),
                        "checker_stderr": (
                            checker_result.stderr[-2000:]
                        ),
                    }

                if checker_result.status == "CHECKER_ERROR":
                    return {
                        **base_result,
                        "status": "CHECKER_ERROR",
                        "tests_passed": passed,
                        "tests_total": len(
                            official_tests
                        ),
                        "checker_stdout": (
                            checker_result.stdout[-2000:]
                        ),
                        "checker_stderr": (
                            checker_result.stderr[-2000:]
                        ),
                    }

                if checker_result.status != "PASS":
                    return {
                        **base_result,
                        "status": "WRONG_ANSWER",
                        "tests_passed": passed,
                        "tests_total": len(
                            official_tests
                        ),
                        "execution_time_ms": (
                            result.execution_time_ms
                        ),
                        "checker_stdout": (
                            checker_result.stdout[-2000:]
                        ),
                        "checker_stderr": (
                            checker_result.stderr[-2000:]
                        ),
                        "stdout": result.stdout[-2000:],
                    }

            else:
                # No generated checker exists, so fall back
                # to exact token comparison.
                if result.status != "PASS":
                    return {
                        **base_result,
                        "status": "WRONG_ANSWER",
                        "tests_passed": passed,
                        "tests_total": len(
                            official_tests
                        ),
                        "execution_time_ms": (
                            result.execution_time_ms
                        ),
                        "stderr": result.stderr[-2000:],
                        "stdout": result.stdout[-2000:],
                    }

            passed += 1

        if has_checker:
            status = "PASSED_CHECKER"
        elif problem.get(
            "official_tests_complete",
            False,
        ):
            status = "PASSED_EXACT"
        else:
            status = "PASSED_AVAILABLE_TESTS"

        return {
            **base_result,
            "status": status,
            "tests_passed": passed,
            "tests_total": len(official_tests),
            "max_execution_time_ms": max(
                execution_times
            ),
            "total_execution_time_ms": sum(
                execution_times
            ),
            "official_tests_complete": problem.get(
                "official_tests_complete",
                False,
            ),
            "used_generated_checker": has_checker,
        }

    finally:
        shutil.rmtree(
            compile_dir,
            ignore_errors=True,
        )


def main() -> None:
    problems = load_jsonl(PROBLEMS_PATH)
    solutions = load_jsonl(SOLUTIONS_PATH)

    problems_by_id = {
        problem["id"]: problem
        for problem in problems
    }

    solutions_by_problem: dict[
        str, list[dict]
    ] = defaultdict(list)

    for solution in solutions:
        if not solution.get("is_accepted", False):
            continue

        problem_id = solution["problem_id"]

        if problem_id not in problems_by_id:
            continue

        normalized = normalize_language(
            solution["language"]
        )

        if normalized is None:
            continue

        if (
            normalized == "pypy"
            and not pypy_available()
        ):
            continue

        solutions_by_problem[
            problem_id
        ].append(solution)

    eligible_problems = [
        problem
        for problem in problems
        if problem.get("input_mode") == "stdio"
        and problem.get("official_tests")
        and problem.get("executable", False)
        and problem["id"] in solutions_by_problem
    ]

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(eligible_problems)

    selected_problems = eligible_problems[
        :MAX_PROBLEMS
    ]

    selected_solutions: list[dict] = []

    for problem in selected_problems:
        candidates = solutions_by_problem[
            problem["id"]
        ]

        selected_solutions.extend(
            choose_solutions(
                candidates,
                MAX_SOLUTIONS_PER_PROBLEM,
            )
        )

    print(
        f"Eligible problems: "
        f"{len(eligible_problems)}"
    )
    print(
        f"Unique problems selected: "
        f"{len(selected_problems)}"
    )
    print(
        f"Solutions selected: "
        f"{len(selected_solutions)}"
    )
    print(
        f"PyPy available: "
        f"{pypy_available()}"
    )
    print()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    with tempfile.TemporaryDirectory(
        prefix="dp_verify_"
    ) as temp_dir:
        work_dir = Path(temp_dir)

        for index, solution in enumerate(
            selected_solutions,
            start=1,
        ):
            problem = problems_by_id[
                solution["problem_id"]
            ]

            print(
                f"[{index}/"
                f"{len(selected_solutions)}] "
                f"{solution['problem_id']} "
                f"{solution['language']}"
            )

            result = verify_solution(
                solution=solution,
                problem=problem,
                work_dir=work_dir,
            )

            results.append(result)

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        for result in results:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    status_counts = Counter(
        result["status"]
        for result in results
    )

    language_counts = Counter(
        result["language"]
        for result in results
    )

    print()
    print("Verification complete")
    print(
        f"Unique problems: "
        f"{len(selected_problems)}"
    )
    print(
        f"Solutions checked: "
        f"{len(results)}"
    )
    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print("Languages:")

    for language, count in (
        language_counts.most_common()
    ):
        print(f"  {language}: {count}")

    print()
    print("Statuses:")

    for status, count in (
        status_counts.most_common()
    ):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
