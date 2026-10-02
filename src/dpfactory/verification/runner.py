from __future__ import annotations

import subprocess
import time
from pathlib import Path


class RunResult:
    def __init__(
        self,
        status: str,
        stdout: str,
        stderr: str,
        execution_time_ms: float,
    ) -> None:
        self.status = status
        self.stdout = stdout
        self.stderr = stderr
        self.execution_time_ms = execution_time_ms


def normalize_output(output: str) -> list[str]:
    return output.split()


def outputs_match(actual: str, expected: str) -> bool:
    return normalize_output(actual) == normalize_output(expected)


def run_solution(
    executable: Path,
    input_data: str,
    expected_output: str,
    timeout_seconds: float,
    interpreter: str | None = None,
) -> RunResult:
    if interpreter is not None:
        command = [interpreter, str(executable)]
    else:
        command = [str(executable)]

    start = time.perf_counter()

    try:
        result = subprocess.run(
            command,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=executable.parent,
        )

    except subprocess.TimeoutExpired as exc:
        elapsed = (time.perf_counter() - start) * 1000

        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")

        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")

        return RunResult(
            status="TLE",
            stdout=stdout,
            stderr=stderr,
            execution_time_ms=elapsed,
        )

    elapsed = (time.perf_counter() - start) * 1000

    if result.returncode != 0:
        return RunResult(
            status="RE",
            stdout=result.stdout,
            stderr=result.stderr,
            execution_time_ms=elapsed,
        )

    if not outputs_match(result.stdout, expected_output):
        return RunResult(
            status="WA",
            stdout=result.stdout,
            stderr=result.stderr,
            execution_time_ms=elapsed,
        )

    return RunResult(
        status="PASS",
        stdout=result.stdout,
        stderr=result.stderr,
        execution_time_ms=elapsed,
    )
