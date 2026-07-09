from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any

import pytest

from .benchmarking import (
    BENCHMARK_CASES,
    BENCHMARK_RESULTS,
    BenchmarkSettings,
    write_benchmark_json,
)
from .cases import GRAD_CASES, INPLACE_CASES, OP_CASES, VIEW_CASES
from .harness import import_module


@pytest.fixture(scope="session")
def torch_reference(pytestconfig: pytest.Config) -> Any:
    return import_module(pytestconfig.getoption("--compat-reference"))


@pytest.fixture(scope="session")
def torch_candidate(pytestconfig: pytest.Config) -> Any:
    candidate = import_module(pytestconfig.getoption("--compat-candidate"))
    reference = import_module(pytestconfig.getoption("--compat-reference"))
    if candidate is reference:
        raise AssertionError("Candidate module must not be the same module object as the PyTorch reference.")
    return candidate


@pytest.fixture(scope="session")
def allow_reference_backed(pytestconfig: pytest.Config) -> bool:
    return bool(pytestconfig.getoption("--compat-allow-reference-backed"))


@pytest.fixture(scope="session")
def benchmark_settings(pytestconfig: pytest.Config) -> BenchmarkSettings:
    return BenchmarkSettings(
        warmup=max(0, int(pytestconfig.getoption("--compat-benchmark-warmup"))),
        repeat=max(1, int(pytestconfig.getoption("--compat-benchmark-repeat"))),
        slow_ratio=max(0.0, float(pytestconfig.getoption("--compat-benchmark-slow-ratio"))),
        max_ratio=max(0.0, float(pytestconfig.getoption("--compat-benchmark-max-ratio"))),
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    patterns = metafunc.config.getoption("--compat-op") or ["*"]
    benchmark_patterns = metafunc.config.getoption("--compat-benchmark") or ["*"]

    if "api_entry" in metafunc.fixturenames:
        entries = _load_api_entries(Path(metafunc.config.getoption("--compat-api-manifest")))
        metafunc.parametrize("api_entry", entries, ids=[entry["path"] for entry in entries])

    if "op_case" in metafunc.fixturenames:
        cases = [case for case in OP_CASES if _matches(case.id, patterns)]
        metafunc.parametrize("op_case", cases, ids=[case.id for case in cases])

    if "inplace_case" in metafunc.fixturenames:
        cases = [case for case in INPLACE_CASES if _matches(case.id, patterns)]
        metafunc.parametrize("inplace_case", cases, ids=[case.id for case in cases])

    if "view_case" in metafunc.fixturenames:
        cases = [case for case in VIEW_CASES if _matches(case.id, patterns)]
        metafunc.parametrize("view_case", cases, ids=[case.id for case in cases])

    if "grad_case" in metafunc.fixturenames:
        cases = [case for case in GRAD_CASES if _matches(case.id, patterns)]
        metafunc.parametrize("grad_case", cases, ids=[case.id for case in cases])

    if "benchmark_case" in metafunc.fixturenames:
        cases = [case for case in BENCHMARK_CASES if _matches(case.id, benchmark_patterns)]
        metafunc.parametrize("benchmark_case", cases, ids=[case.id for case in cases])


def _matches(case_id: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(case_id, pattern) for pattern in patterns)


def _load_api_entries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"{path}: entries must be a list")
    return entries


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, exitstatus: int, config: pytest.Config) -> None:
    if not BENCHMARK_RESULTS:
        return

    terminalreporter.write_sep("-", "PyTorch benchmark comparison")
    slow_ratio = max(0.0, float(config.getoption("--compat-benchmark-slow-ratio")))
    max_ratio = max(0.0, float(config.getoption("--compat-benchmark-max-ratio")))
    threshold_text = f"SLOW if ratio>{slow_ratio:.2f}"
    if max_ratio > 0.0:
        threshold_text += f", FAIL if ratio>{max_ratio:.2f}"
    terminalreporter.write_line(threshold_text)
    terminalreporter.write_line(
        f"{'case':38} {'status':>6} {'torch ms':>10} {'mtorch ms':>10} {'ratio':>8} {'torch min':>10} {'mtorch min':>10}"
    )
    for result in BENCHMARK_RESULTS:
        terminalreporter.write_line(
            f"{result.id:38} "
            f"{result.status:>6} "
            f"{result.reference_median_ms:10.3f} "
            f"{result.candidate_median_ms:10.3f} "
            f"{result.ratio:8.2f} "
            f"{result.reference_min_ms:10.3f} "
            f"{result.candidate_min_ms:10.3f}"
        )

    output = config.getoption("--compat-benchmark-json")
    if output:
        path = Path(output)
        write_benchmark_json(path)
        terminalreporter.write_line(f"benchmark JSON: {path}")
