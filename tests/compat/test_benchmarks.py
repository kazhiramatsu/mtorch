from __future__ import annotations

import pytest

from .benchmarking import run_benchmark_case


pytestmark = [pytest.mark.compat, pytest.mark.benchmark]


def test_benchmark_against_pytorch(torch_reference, torch_candidate, benchmark_case, benchmark_settings) -> None:
    result = run_benchmark_case(torch_reference, torch_candidate, benchmark_case, benchmark_settings)
    if result.max_ratio_exceeded:
        raise AssertionError(
            f"{result.id}: candidate/reference median ratio {result.ratio:.2f} "
            f"exceeds threshold {benchmark_settings.max_ratio:.2f}"
        )
