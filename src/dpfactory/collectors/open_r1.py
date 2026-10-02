from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

from dpfactory.models import Problem, TestCase


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _parse_examples(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []

    if isinstance(value, list):
        result: list[dict[str, str]] = []

        for item in value:
            if not isinstance(item, dict):
                continue

            result.append(
                {
                    "input": _safe_str(item.get("input")),
                    "output": _safe_str(item.get("output")),
                }
            )

        return result

    return []


def _parse_tests(value: Any) -> list[TestCase]:
    if value is None:
        return []

    if not isinstance(value, list):
        return []

    result: list[TestCase] = []

    for item in value:
        if not isinstance(item, dict):
            continue

        result.append(
            TestCase(
                input=_safe_str(item.get("input")),
                output=_safe_str(item.get("output")),
            )
        )

    return result


def convert_problem(row: dict[str, Any], split: str) -> Problem:
    known_fields = {
        "id",
        "aliases",
        "contest_id",
        "contest_name",
        "contest_type",
        "contest_start",
        "contest_start_year",
        "index",
        "time_limit",
        "memory_limit",
        "title",
        "description",
        "input_format",
        "output_format",
        "interaction_format",
        "note",
        "examples",
        "editorial",
        "rating",
        "tags",
        "testset_size",
        "official_tests",
        "official_tests_complete",
        "input_mode",
        "generated_checker",
        "executable",
        "generated_tests",
    }

    extra = {
        key: value
        for key, value in row.items()
        if key not in known_fields
    }

    return Problem(
        id=_safe_str(row.get("id")),
        aliases=[
            _safe_str(x)
            for x in (row.get("aliases") or [])
        ],
        contest_id=_safe_str(row.get("contest_id")),
        contest_name=_safe_str(row.get("contest_name")),
        contest_type=_safe_str(row.get("contest_type")),
        contest_start_year=int(row.get("contest_start_year") or 0),
        index=_safe_str(row.get("index")),
        title=_safe_str(row.get("title")),
        description=_safe_str(row.get("description")),
        input_format=_safe_str(row.get("input_format")),
        output_format=_safe_str(row.get("output_format")),
        interaction_format=(
            None
            if row.get("interaction_format") is None
            else _safe_str(row.get("interaction_format"))
        ),
        note=(
            None
            if row.get("note") is None
            else _safe_str(row.get("note"))
        ),
        rating=(
            None
            if row.get("rating") is None
            else int(row["rating"])
        ),
        tags=[
            _safe_str(x)
            for x in (row.get("tags") or [])
        ],
        editorial=(
            None
            if row.get("editorial") is None
            else _safe_str(row.get("editorial"))
        ),
        time_limit=float(row.get("time_limit") or 0),
        memory_limit=float(row.get("memory_limit") or 0),
        input_mode=_safe_str(row.get("input_mode")),
        testset_size=(
            None
            if row.get("testset_size") is None
            else int(row["testset_size"])
        ),
        official_tests=_parse_tests(row.get("official_tests")),
        official_tests_complete=bool(
            row.get("official_tests_complete", False)
        ),
        generated_checker=(
            None
            if row.get("generated_checker") is None
            else _safe_str(row.get("generated_checker"))
        ),
        executable=bool(row.get("executable", False)),
        source="open-r1/codeforces",
        source_split=split,
        extra={
            **extra,
            "examples": _parse_examples(row.get("examples")),
            "generated_tests": row.get("generated_tests"),
        },
    )


def collect_open_r1(
    output_path: str | Path,
    split: str = "train",
    config: str = "default",
) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading open-r1/codeforces "
        f"(config={config}, split={split})..."
    )

    dataset = load_dataset(
        "open-r1/codeforces",
        config,
        split=split,
    )

    print(f"Loaded {len(dataset):,} problems.")

    count = 0

    with output.open("w", encoding="utf-8") as f:
        for row in tqdm(dataset, desc="Collecting"):
            problem = convert_problem(row, split)

            f.write(
                json.dumps(
                    problem.model_dump(),
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

    print(f"Wrote {count:,} problems to {output}")

    return count
