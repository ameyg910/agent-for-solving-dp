from __future__ import annotations

import subprocess
from pathlib import Path


COMPILE_COMMANDS: dict[str, list[str]] = {
    "GNU C++": [
        "g++",
        "-std=c++17",
        "-O2",
        "-pipe",
    ],
    "GNU C++11": [
        "g++",
        "-std=c++11",
        "-O2",
        "-pipe",
    ],
    "GNU C++14": [
        "g++",
        "-std=c++14",
        "-O2",
        "-pipe",
    ],
    "C++17": [
        "g++",
        "-std=c++17",
        "-O2",
        "-pipe",
    ],
    "C++20": [
        "g++",
        "-std=c++20",
        "-O2",
        "-pipe",
    ],
}


def normalize_language(language: str) -> str | None:
    language_lower = language.lower()

    if "c++20" in language_lower:
        return "C++20"

    if "c++17" in language_lower:
        return "C++17"

    if "c++14" in language_lower:
        return "GNU C++14"

    if "c++11" in language_lower:
        return "GNU C++11"

    if "c++" in language_lower:
        return "GNU C++"

    if "python" in language_lower or "pypy" in language_lower:
        return "python"

    return None


def compile_solution(
    source_code: str,
    language: str,
    work_dir: Path,
) -> tuple[bool, Path | None, str]:
    normalized = normalize_language(language)

    if normalized is None:
        return False, None, f"Unsupported language: {language}"

    work_dir.mkdir(parents=True, exist_ok=True)

    if normalized == "python":
        source_path = work_dir / "main.py"
        source_path.write_text(source_code, encoding="utf-8")
        return True, source_path, ""

    source_path = work_dir / "main.cpp"
    binary_path = work_dir / "main"

    source_path.write_text(source_code, encoding="utf-8")

    command = COMPILE_COMMANDS[normalized] + [
        str(source_path),
        "-o",
        str(binary_path),
    ]

    result = subprocess.run(
        command,
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        return False, None, result.stderr[-4000:]

    return True, binary_path, ""
