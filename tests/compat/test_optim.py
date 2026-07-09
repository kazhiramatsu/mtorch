from __future__ import annotations

import pytest

from .harness import assert_values_compatible


pytestmark = [pytest.mark.compat, pytest.mark.optim]


def _parameter(module, data):
    return module.tensor(data, dtype=module.float32, requires_grad=True)


def _assign_grad(module, parameter, data):
    parameter.grad = module.tensor(data, dtype=module.float32)


def test_sgd_step_matches_reference(torch_reference, torch_candidate) -> None:
    data = [[1.0, -2.0], [3.0, -4.0]]
    grad = [[0.5, -1.0], [1.5, -2.0]]
    expected = _parameter(torch_reference, data)
    actual = _parameter(torch_candidate, data)
    _assign_grad(torch_reference, expected, grad)
    _assign_grad(torch_candidate, actual, grad)

    torch_reference.optim.SGD([expected], lr=0.1, momentum=0.9, weight_decay=0.01).step()
    torch_candidate.optim.SGD([actual], lr=0.1, momentum=0.9, weight_decay=0.01).step()

    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="optim.SGD.step",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_adam_step_matches_reference(torch_reference, torch_candidate) -> None:
    data = [[1.0, -2.0], [3.0, -4.0]]
    grad = [[0.5, -1.0], [1.5, -2.0]]
    expected = _parameter(torch_reference, data)
    actual = _parameter(torch_candidate, data)
    _assign_grad(torch_reference, expected, grad)
    _assign_grad(torch_candidate, actual, grad)

    torch_reference.optim.Adam([expected], lr=0.01, betas=(0.8, 0.95), weight_decay=0.01).step()
    torch_candidate.optim.Adam([actual], lr=0.01, betas=(0.8, 0.95), weight_decay=0.01).step()

    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="optim.Adam.step",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_adamw_step_matches_reference(torch_reference, torch_candidate) -> None:
    data = [[1.0, -2.0], [3.0, -4.0]]
    grad = [[0.5, -1.0], [1.5, -2.0]]
    expected = _parameter(torch_reference, data)
    actual = _parameter(torch_candidate, data)
    _assign_grad(torch_reference, expected, grad)
    _assign_grad(torch_candidate, actual, grad)

    torch_reference.optim.AdamW([expected], lr=0.01, betas=(0.8, 0.95), weight_decay=0.01).step()
    torch_candidate.optim.AdamW([actual], lr=0.01, betas=(0.8, 0.95), weight_decay=0.01).step()

    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="optim.AdamW.step",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_optimizer_zero_grad_matches_reference(torch_reference, torch_candidate) -> None:
    expected = _parameter(torch_reference, [1.0, -2.0])
    actual = _parameter(torch_candidate, [1.0, -2.0])
    _assign_grad(torch_reference, expected, [0.5, -1.0])
    _assign_grad(torch_candidate, actual, [0.5, -1.0])

    torch_reference.optim.SGD([expected], lr=0.1).zero_grad()
    torch_candidate.optim.SGD([actual], lr=0.1).zero_grad()

    assert expected.grad is None
    assert actual.grad is None
