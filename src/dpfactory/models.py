from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Example(BaseModel):
    input: str
    output: str


class TestCase(BaseModel):
    input: str
    output: str


class Problem(BaseModel):
    # Identity
    id: str
    aliases: list[str] = Field(default_factory=list)

    # Contest metadata
    contest_id: str
    contest_name: str
    contest_type: str
    contest_start_year: int
    index: str

    # Problem statement
    title: str
    description: str
    input_format: str
    output_format: str
    interaction_format: str | None = None
    note: str | None = None

    # Metadata
    rating: int | None = None
    tags: list[str] = Field(default_factory=list)
    editorial: str | None = None

    # Execution constraints
    time_limit: float
    memory_limit: float
    input_mode: str

    # Verification information
    testset_size: int | None = None
    official_tests: list[TestCase] = Field(default_factory=list)
    official_tests_complete: bool = False
    generated_checker: str | None = None
    executable: bool = False

    # Dataset provenance
    source: str = "open-r1/codeforces"
    source_split: str = "train"

    # Preserve anything we don't explicitly model yet
    extra: dict[str, Any] = Field(default_factory=dict)


class DPCandidate(BaseModel):
    id: str
    title: str
    description: str

    tags: list[str] = Field(default_factory=list)
    rating: int | None = None

    has_editorial: bool = False
    executable: bool = False
    official_tests_complete: bool = False

    dp_tag: bool = False
    editorial_dp_evidence: list[str] = Field(default_factory=list)
    metadata_evidence: list[str] = Field(default_factory=list)

    # This is a heuristic ranking score, NOT a probability.
    heuristic_score: float = 0.0

    tier: str

    source: str = "open-r1/codeforces"
    source_split: str = "train"


class Solution(BaseModel):
    problem_id: str

    language: str
    source_code: str

    verdict: str
    execution_time_ms: float | None = None
    memory_bytes: int | None = None

    source: str
    source_id: str | None = None

    is_accepted: bool = False
    is_failure_example: bool = False

    extra: dict[str, object] = Field(default_factory=dict)
