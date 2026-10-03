from __future__ import annotations

import os
import resource
import signal
import subprocess
import time
from pathlib import Path


# Conservative safety limits for untrusted competitive-programming submissions.
MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024       # 2 GiB
FILE_SIZE_LIMIT_BYTES = 64 * 1024 * 1024          # 64 MiB
PROCESS_LIMIT = 64


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


def normalize_input(input_data: str) -> str:
    """Normalize dataset line endings before feeding input to submissions."""
    return input_data.replace("\r\n", "\n").replace("\r", "\n")


def normalize_output(output: str) -> list[str]:
    return output.split()


def outputs_match(
    actual: str,
    expected: str,
    case_insensitive: bool = False,
) -> bool:
    actual_tokens = normalize_output(actual)
    expected_tokens = normalize_output(expected)

    if case_insensitive:
        actual_tokens = [token.lower() for token in actual_tokens]
        expected_tokens = [token.lower() for token in expected_tokens]

    return actual_tokens == expected_tokens


def _apply_resource_limits() -> None:
    """Apply Linux resource limits inside the submission process."""

    # Maximum virtual address space.
    resource.setrlimit(
        resource.RLIMIT_AS,
        (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES),
    )

    # Maximum size of any single file created by the submission.
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (FILE_SIZE_LIMIT_BYTES, FILE_SIZE_LIMIT_BYTES),
    )

    # Prevent fork bombs / excessive child processes.
    resource.setrlimit(
        resource.RLIMIT_NPROC,
        (PROCESS_LIMIT, PROCESS_LIMIT),
    )


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    """Kill the submission and all children."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_solution(
    executable: Path,
    input_data: str,
    expected_output: str,
    timeout_seconds: float,
    interpreter: str | None = None,
    case_insensitive: bool = False,
) -> RunResult:
    if interpreter is not None:
        command = [interpreter, str(executable)]
    else:
        command = [str(executable)]

    start = time.perf_counter()

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=executable.parent,
            start_new_session=True,
            preexec_fn=_apply_resource_limits,
        )

        try:
            stdout, stderr = process.communicate(
                input=normalize_input(input_data),
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            _kill_process_group(process)
            stdout, stderr = process.communicate()

            elapsed = (time.perf_counter() - start) * 1000

            return RunResult(
                status="TLE",
                stdout=stdout or "",
                stderr=stderr or "",
                execution_time_ms=elapsed,
            )

    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000

        return RunResult(
            status="RE",
            stdout="",
            stderr=str(exc),
            execution_time_ms=elapsed,
        )

    elapsed = (time.perf_counter() - start) * 1000

    if process.returncode != 0:
        return RunResult(
            status="RE",
            stdout=stdout,
            stderr=stderr,
            execution_time_ms=elapsed,
        )

    if not outputs_match(
        stdout,
        expected_output,
        case_insensitive=case_insensitive,
    ):
        return RunResult(
            status="WA",
            stdout=stdout,
            stderr=stderr,
            execution_time_ms=elapsed,
        )

    return RunResult(
        status="PASS",
        stdout=stdout,
        stderr=stderr,
        execution_time_ms=elapsed,
    )
