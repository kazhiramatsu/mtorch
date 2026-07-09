from __future__ import annotations

import pytest

from .harness import assert_first_order_grad, assert_values_compatible


pytestmark = [pytest.mark.compat, pytest.mark.autograd]


def test_first_order_grad_matches_reference(torch_reference, torch_candidate, grad_case) -> None:
    assert_first_order_grad(torch_reference, torch_candidate, grad_case)


def test_batched_matmul_noncontiguous_grad_matches_reference(torch_reference, torch_candidate) -> None:
    left_data = [
        [[1.0, 0.5], [2.0, -2.0], [-1.0, 3.0]],
        [[-1.0, 3.0], [0.25, -0.5], [2.0, 1.0]],
    ]
    right_data = [
        [[0.25, -1.0], [1.5, 0.5], [-0.75, 2.0]],
        [[-1.0, 0.5], [2.0, -0.25], [0.75, 1.5]],
    ]
    expected_left_base = torch_reference.tensor(left_data, dtype=torch_reference.float32, requires_grad=True)
    expected_right = torch_reference.tensor(right_data, dtype=torch_reference.float32, requires_grad=True)
    actual_left_base = torch_candidate.tensor(left_data, dtype=torch_candidate.float32, requires_grad=True)
    actual_right = torch_candidate.tensor(right_data, dtype=torch_candidate.float32, requires_grad=True)

    torch_reference.matmul(expected_left_base.transpose(-1, -2), expected_right).sum().backward()
    torch_candidate.matmul(actual_left_base.transpose(-1, -2), actual_right).sum().backward()

    assert_values_compatible(
        torch_reference,
        expected_left_base.grad,
        actual_left_base.grad,
        path="matmul.noncontiguous_batched_grad.left",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )
    assert_values_compatible(
        torch_reference,
        expected_right.grad,
        actual_right.grad,
        path="matmul.noncontiguous_batched_grad.right",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_bmm_noncontiguous_grad_matches_reference(torch_reference, torch_candidate) -> None:
    left_data = [
        [[1.0, 0.5], [2.0, -2.0], [-1.0, 3.0]],
        [[-1.0, 3.0], [0.25, -0.5], [2.0, 1.0]],
    ]
    right_data = [
        [[0.25, 1.5, -0.75], [-1.0, 0.5, 2.0]],
        [[-1.0, 2.0, 0.75], [0.5, -0.25, 1.5]],
    ]
    expected_left_base = torch_reference.tensor(left_data, dtype=torch_reference.float32, requires_grad=True)
    expected_right_base = torch_reference.tensor(right_data, dtype=torch_reference.float32, requires_grad=True)
    actual_left_base = torch_candidate.tensor(left_data, dtype=torch_candidate.float32, requires_grad=True)
    actual_right_base = torch_candidate.tensor(right_data, dtype=torch_candidate.float32, requires_grad=True)

    torch_reference.bmm(expected_left_base.transpose(-1, -2), expected_right_base.transpose(-1, -2)).sum().backward()
    torch_candidate.bmm(actual_left_base.transpose(-1, -2), actual_right_base.transpose(-1, -2)).sum().backward()

    assert_values_compatible(
        torch_reference,
        expected_left_base.grad,
        actual_left_base.grad,
        path="bmm.noncontiguous_grad.left",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )
    assert_values_compatible(
        torch_reference,
        expected_right_base.grad,
        actual_right_base.grad,
        path="bmm.noncontiguous_grad.right",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_linear_nd_noncontiguous_grad_matches_reference(torch_reference, torch_candidate) -> None:
    input_data = [
        [[1.0, 0.5], [2.0, -2.0], [-1.0, 3.0]],
        [[-1.0, 3.0], [0.25, -0.5], [2.0, 1.0]],
    ]
    weight_data = [[0.25, -1.0, 0.5], [1.5, 0.5, -0.75], [-0.25, 2.0, 1.0]]
    bias_data = [[[0.5, -1.0, 0.25]], [[-0.25, 0.75, -0.5]]]
    expected_input_base = torch_reference.tensor(input_data, dtype=torch_reference.float32, requires_grad=True)
    expected_weight = torch_reference.tensor(weight_data, dtype=torch_reference.float32, requires_grad=True)
    expected_bias = torch_reference.tensor(bias_data, dtype=torch_reference.float32, requires_grad=True)
    actual_input_base = torch_candidate.tensor(input_data, dtype=torch_candidate.float32, requires_grad=True)
    actual_weight = torch_candidate.tensor(weight_data, dtype=torch_candidate.float32, requires_grad=True)
    actual_bias = torch_candidate.tensor(bias_data, dtype=torch_candidate.float32, requires_grad=True)

    torch_reference.nn.functional.linear(
        expected_input_base.transpose(-1, -2),
        expected_weight,
        expected_bias,
    ).sum().backward()
    torch_candidate.nn.functional.linear(
        actual_input_base.transpose(-1, -2),
        actual_weight,
        actual_bias,
    ).sum().backward()

    assert_values_compatible(
        torch_reference,
        expected_input_base.grad,
        actual_input_base.grad,
        path="linear.nd_noncontiguous_grad.input",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )
    assert_values_compatible(
        torch_reference,
        expected_weight.grad,
        actual_weight.grad,
        path="linear.nd_noncontiguous_grad.weight",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )
    assert_values_compatible(
        torch_reference,
        expected_bias.grad,
        actual_bias.grad,
        path="linear.nd_noncontiguous_grad.bias",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_autograd_grad_matches_reference_without_accumulating(torch_reference, torch_candidate) -> None:
    expected_input = torch_reference.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch_reference.float32, requires_grad=True)
    actual_input = torch_candidate.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch_candidate.float32, requires_grad=True)

    expected_grad = torch_reference.autograd.grad((expected_input * expected_input).sum(), (expected_input,))
    actual_grad = torch_candidate.autograd.grad((actual_input * actual_input).sum(), (actual_input,))

    assert_values_compatible(
        torch_reference,
        expected_grad,
        actual_grad,
        path="autograd.grad.square",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )
    assert expected_input.grad is None
    assert actual_input.grad is None


def test_autograd_grad_with_grad_outputs_matches_reference(torch_reference, torch_candidate) -> None:
    data = [[1.0, -2.0], [3.0, -4.0]]
    grad_output = [[0.5, 2.0], [-1.0, 0.25]]
    expected_input = torch_reference.tensor(data, dtype=torch_reference.float32, requires_grad=True)
    actual_input = torch_candidate.tensor(data, dtype=torch_candidate.float32, requires_grad=True)
    expected_upstream = torch_reference.tensor(grad_output, dtype=torch_reference.float32)
    actual_upstream = torch_candidate.tensor(grad_output, dtype=torch_candidate.float32)

    expected_grad = torch_reference.autograd.grad(expected_input * expected_input, expected_input, expected_upstream)
    actual_grad = torch_candidate.autograd.grad(actual_input * actual_input, actual_input, actual_upstream)

    assert_values_compatible(
        torch_reference,
        expected_grad,
        actual_grad,
        path="autograd.grad.grad_outputs",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_backward_accepts_explicit_gradient(torch_reference, torch_candidate) -> None:
    data = [[1.0, -2.0], [3.0, -4.0]]
    grad_output = [[0.5, 2.0], [-1.0, 0.25]]
    expected_input = torch_reference.tensor(data, dtype=torch_reference.float32, requires_grad=True)
    actual_input = torch_candidate.tensor(data, dtype=torch_candidate.float32, requires_grad=True)
    expected_upstream = torch_reference.tensor(grad_output, dtype=torch_reference.float32)
    actual_upstream = torch_candidate.tensor(grad_output, dtype=torch_candidate.float32)

    (expected_input * expected_input).backward(expected_upstream)
    (actual_input * actual_input).backward(actual_upstream)

    assert_values_compatible(
        torch_reference,
        expected_input.grad,
        actual_input.grad,
        path="Tensor.backward.grad",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_no_grad_matches_reference_and_restores_state(torch_reference, torch_candidate) -> None:
    expected_input = torch_reference.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=torch_reference.float32, requires_grad=True)
    actual_input = torch_candidate.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=torch_candidate.float32, requires_grad=True)

    with torch_reference.no_grad():
        expected_result = (expected_input * 2.0).clone()
        expected_view = expected_input.reshape(3, 2)
        expected_factory = torch_reference.ones((2,), dtype=torch_reference.float32, requires_grad=True)

    with torch_candidate.no_grad():
        actual_result = (actual_input * 2.0).clone()
        actual_view = actual_input.reshape(3, 2)
        actual_factory = torch_candidate.ones((2,), dtype=torch_candidate.float32, requires_grad=True)

    assert_values_compatible(
        torch_reference,
        expected_result,
        actual_result,
        path="no_grad.result",
        rtol=1e-6,
        atol=1e-6,
    )
    assert_values_compatible(torch_reference, expected_view, actual_view, path="no_grad.view")
    assert_values_compatible(torch_reference, expected_factory, actual_factory, path="no_grad.factory")
    assert torch_reference.is_grad_enabled() == torch_candidate.is_grad_enabled()


def test_inference_mode_disables_grad_and_restores_state(torch_reference, torch_candidate) -> None:
    expected_input = torch_reference.tensor([1.0, 2.0], dtype=torch_reference.float32, requires_grad=True)
    actual_input = torch_candidate.tensor([1.0, 2.0], dtype=torch_candidate.float32, requires_grad=True)

    assert torch_candidate.is_inference_mode_enabled() is False
    with torch_reference.inference_mode():
        expected_result = expected_input * 2.0
        expected_grad_enabled = torch_reference.is_grad_enabled()
        expected_nested_grad_enabled = None
        with torch_reference.inference_mode(False):
            expected_nested_grad_enabled = torch_reference.is_grad_enabled()

    with torch_candidate.inference_mode():
        actual_result = actual_input * 2.0
        actual_grad_enabled = torch_candidate.is_grad_enabled()
        actual_inference_enabled = torch_candidate.is_inference_mode_enabled()
        with torch_candidate.inference_mode(False):
            actual_nested_grad_enabled = torch_candidate.is_grad_enabled()
            actual_nested_inference_enabled = torch_candidate.is_inference_mode_enabled()

    assert_values_compatible(
        torch_reference,
        expected_result,
        actual_result,
        path="inference_mode.result",
        rtol=1e-6,
        atol=1e-6,
    )
    assert expected_grad_enabled == actual_grad_enabled
    assert expected_nested_grad_enabled == actual_nested_grad_enabled
    assert actual_inference_enabled is True
    assert actual_nested_inference_enabled is False
    assert torch_candidate.is_inference_mode_enabled() is False
    assert torch_reference.is_grad_enabled() == torch_candidate.is_grad_enabled()


def test_set_grad_enabled_matches_reference_immediate_mode(torch_reference, torch_candidate) -> None:
    expected_previous = torch_reference.is_grad_enabled()
    actual_previous = torch_candidate.is_grad_enabled()
    try:
        torch_reference.set_grad_enabled(False)
        torch_candidate.set_grad_enabled(False)
        expected_input = torch_reference.tensor([1.0, 2.0], dtype=torch_reference.float32, requires_grad=True)
        actual_input = torch_candidate.tensor([1.0, 2.0], dtype=torch_candidate.float32, requires_grad=True)

        assert_values_compatible(
            torch_reference,
            expected_input + 1.0,
            actual_input + 1.0,
            path="set_grad_enabled.false_result",
            rtol=1e-6,
            atol=1e-6,
        )
    finally:
        torch_reference.set_grad_enabled(expected_previous)
        torch_candidate.set_grad_enabled(actual_previous)

    assert torch_reference.is_grad_enabled() == torch_candidate.is_grad_enabled()
