from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class CheckerResult:
    def __init__(
        self,
        status: str,
        stdout: str,
        stderr: str,
    ) -> None:
        self.status = status
        self.stdout = stdout
        self.stderr = stderr


def run_checker(
    checker_source: str,
    input_data: str,
    reference_output: str,
    submission_output: str,
    timeout_seconds: float = 5.0,
) -> CheckerResult:
    """
    Run a dataset-provided checker.

    Expected checker interface:

        python3 checker.py input_path reference_output_path submission_path

    Checker convention used by the Open-R1 Codeforces records:
        0   -> rejected
        1   -> accepted
        100 -> accepted (used by at least one checker)
    """

    with tempfile.TemporaryDirectory(prefix="dp_checker_") as temp_dir:
        temp = Path(temp_dir)

        input_path = temp / "input.txt"
        reference_path = temp / "reference.txt"
        submission_path = temp / "submission.txt"
        checker_path = temp / "checker.py"

        input_path.write_text(input_data, encoding="utf-8")
        reference_path.write_text(reference_output, encoding="utf-8")
        submission_path.write_text(submission_output, encoding="utf-8")
        checker_path.write_text(checker_source, encoding="utf-8")

        try:
            result = subprocess.run(
                [
                    "python3",
                    str(checker_path),
                    str(input_path),
                    str(reference_path),
                    str(submission_path),
                ],
                cwd=temp,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""

            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")

            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")

            return CheckerResult(
                status="CHECKER_TLE",
                stdout=stdout,
                stderr=stderr,
            )

        if result.returncode != 0:
            return CheckerResult(
                status="CHECKER_ERROR",
                stdout=result.stdout,
                stderr=result.stderr,
            )

        checker_output = result.stdout.strip()

        if checker_output in {"1", "100"}:
            return CheckerResult(
                status="PASS",
                stdout=result.stdout,
                stderr=result.stderr,
            )

        if checker_output == "0":
            return CheckerResult(
                status="FAIL",
                stdout=result.stdout,
                stderr=result.stderr,
            )

        return CheckerResult(
            status="CHECKER_ERROR",
            stdout=result.stdout,
            stderr=(
                result.stderr
                + "\nUnexpected checker output: "
                + repr(checker_output)
            ),
        )
