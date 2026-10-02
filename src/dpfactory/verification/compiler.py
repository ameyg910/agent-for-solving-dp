from __future__ import annotations

import subprocess
from pathlib import Path


def normalize_language(language: str) -> str | None:
    value = language.lower()

    if "pypy" in value:
        return "pypy"

    if "python" in value:
        return "python"

    if "c++20" in value:
        return "cpp20"

    if "c++17" in value:
        return "cpp17"

    if "c++14" in value:
        return "cpp14"

    if "c++11" in value:
        return "cpp11"

    if "c++" in value:
        return "cpp"

    return None


CPP_FLAGS = {
    "cpp": ["-std=gnu++17"],
    "cpp11": ["-std=gnu++11"],
    "cpp14": ["-std=gnu++14"],
    "cpp17": ["-std=gnu++17"],
    "cpp20": ["-std=gnu++20"],
}

def compile_solution(
    source_code: str,
    language: str,
    work_dir: Path,
) -> tuple[bool, Path | None, str]:
    normalized = normalize_language(language)

    if normalized is None:
        return False, None, f"Unsupported language: {language}"

    work_dir.mkdir(parents=True, exist_ok=True)

    if normalized in {"python", "pypy"}:
        source_path = work_dir / "main.py"
        source_path.write_text(source_code, encoding="utf-8")
        return True, source_path, ""

    source_path = work_dir / "main.cpp"
    binary_path = work_dir / "main"

    source_path.write_text(source_code, encoding="utf-8")

    command = [
        "g++",
        *CPP_FLAGS[normalized],
        "-O2",
        "-pipe",
        "-DONLINE_JUDGE",
        str(source_path),
        "-o",
        str(binary_path),
    ]

    try:
        result = subprocess.run(
            command,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return False, None, "Compilation timeout"

    if result.returncode != 0:
        return False, None, result.stderr[-4000:]

    return True, binary_path, ""
