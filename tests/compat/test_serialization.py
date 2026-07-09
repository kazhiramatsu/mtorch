from __future__ import annotations

import pytest

from .harness import assert_values_compatible


pytestmark = [pytest.mark.compat, pytest.mark.serialization]


def test_save_load_tensor_roundtrip_matches_reference(torch_reference, torch_candidate, tmp_path) -> None:
    expected = torch_reference.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch_reference.float32)
    actual = torch_candidate.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch_candidate.float32)
    expected_path = tmp_path / "reference.pt"
    actual_path = tmp_path / "candidate.pt"

    torch_reference.save(expected, expected_path)
    torch_candidate.save(actual, actual_path)

    assert_values_compatible(
        torch_reference,
        torch_reference.load(expected_path),
        torch_candidate.load(actual_path),
        path="save_load.tensor",
        rtol=1e-6,
        atol=1e-6,
    )


def test_save_load_state_dict_roundtrip_matches_reference(torch_reference, torch_candidate, tmp_path) -> None:
    expected = torch_reference.nn.Linear(2, 2)
    actual = torch_candidate.nn.Linear(2, 2)
    expected.weight.data.copy_(torch_reference.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch_reference.float32))
    expected.bias.data.copy_(torch_reference.tensor([0.5, -1.0], dtype=torch_reference.float32))
    actual.weight.copy_(torch_candidate.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch_candidate.float32))
    actual.bias.copy_(torch_candidate.tensor([0.5, -1.0], dtype=torch_candidate.float32))
    expected_path = tmp_path / "reference_state.pt"
    actual_path = tmp_path / "candidate_state.pt"

    torch_reference.save(expected.state_dict(), expected_path)
    torch_candidate.save(actual.state_dict(), actual_path)

    assert_values_compatible(
        torch_reference,
        torch_reference.load(expected_path),
        torch_candidate.load(actual_path),
        path="save_load.state_dict",
        rtol=1e-6,
        atol=1e-6,
    )
