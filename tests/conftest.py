from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_MANIFEST = ROOT / "compat" / "api_surface_seed.json"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("pytorch-compat")
    group.addoption("--compat-reference", default="torch", help="Reference PyTorch module import path.")
    group.addoption("--compat-candidate", default="mtorch", help="Candidate clone module import path.")
    group.addoption(
        "--compat-api-manifest",
        default=str(DEFAULT_API_MANIFEST),
        help="JSON manifest of public API entries to compare.",
    )
    group.addoption(
        "--compat-op",
        action="append",
        default=[],
        help="fnmatch pattern for operation cases to run, for example --compat-op 'binary.*'.",
    )
    group.addoption(
        "--compat-allow-reference-backed",
        action="store_true",
        help="Allow candidate tensors to be actual reference torch.Tensor instances during bootstrapping.",
    )
    group.addoption(
        "--compat-benchmark",
        action="append",
        default=[],
        help="fnmatch pattern for benchmark cases to run, for example --compat-benchmark 'bench.matmul*'.",
    )
    group.addoption("--compat-benchmark-warmup", type=int, default=2, help="Warmup iterations per benchmark.")
    group.addoption("--compat-benchmark-repeat", type=int, default=5, help="Measured iterations per benchmark.")
    group.addoption(
        "--compat-benchmark-slow-ratio",
        type=float,
        default=1.0,
        help="Mark benchmark as SLOW if candidate/reference median ratio exceeds this value.",
    )
    group.addoption(
        "--compat-benchmark-max-ratio",
        type=float,
        default=0.0,
        help="Fail if candidate/reference median ratio exceeds this value. 0 disables failure checks.",
    )
    group.addoption(
        "--compat-benchmark-json",
        default="",
        help="Optional path to write benchmark results as JSON.",
    )
