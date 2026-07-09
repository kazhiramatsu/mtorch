from __future__ import annotations

import pytest


pytestmark = pytest.mark.compat


def test_candidate_module_is_not_reference_module(torch_reference, torch_candidate) -> None:
    assert torch_candidate is not torch_reference


def test_candidate_tensor_type_is_not_reference_tensor(torch_reference, torch_candidate, allow_reference_backed) -> None:
    if allow_reference_backed:
        pytest.skip("reference-backed candidate tensors are explicitly allowed for this run")
    if not hasattr(torch_candidate, "tensor"):
        pytest.skip("candidate has no tensor factory yet; API surface tests will report that gap")

    value = torch_candidate.tensor([1.0])
    assert not isinstance(value, torch_reference.Tensor)
