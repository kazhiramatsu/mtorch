from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .harness import assert_values_compatible


def _tensor(module: Any, data: Any) -> Any:
    return module.tensor(data, dtype=module.float32)


def _copy_parameter(module: Any, target: Any, source: Any) -> None:
    if hasattr(module, "no_grad"):
        with module.no_grad():
            target.copy_(source)
        return
    target.copy_(source)


def _linear(module: Any, bias: bool = True) -> Any:
    layer = module.nn.Linear(3, 2, bias=bias)
    _copy_parameter(
        module,
        layer.weight,
        _tensor(module, [[0.25, -1.0, 2.0], [1.5, 0.5, -0.75]]),
    )
    if bias:
        _copy_parameter(module, layer.bias, _tensor(module, [0.5, -1.0]))
    return layer


def _conv1d(module: Any, bias: bool = True) -> Any:
    layer = module.nn.Conv1d(2, 3, 3, padding=1, bias=bias)
    _copy_parameter(
        module,
        layer.weight,
        _tensor(
            module,
            [
                [[0.25, -0.5, 0.75], [1.0, 0.0, -1.0]],
                [[-0.25, 0.5, 1.0], [0.75, -0.5, 0.25]],
                [[1.0, 0.5, -0.5], [-0.25, 0.75, 0.25]],
            ],
        ),
    )
    if bias:
        _copy_parameter(module, layer.bias, _tensor(module, [0.5, -1.0, 0.25]))
    return layer


def _conv_transpose1d(module: Any, bias: bool = True) -> Any:
    layer = module.nn.ConvTranspose1d(2, 3, 3, stride=2, padding=1, output_padding=1, bias=bias)
    _copy_parameter(
        module,
        layer.weight,
        _tensor(
            module,
            [
                [[0.25, 1.0, 0.5], [-0.5, 0.0, 0.25], [0.75, -1.0, -0.25]],
                [[1.0, -0.25, 0.5], [0.5, 0.75, -1.0], [-0.5, 0.25, 0.25]],
            ],
        ),
    )
    if bias:
        _copy_parameter(module, layer.bias, _tensor(module, [0.5, -1.0, 0.25]))
    return layer


def _conv2d(module: Any, bias: bool = True) -> Any:
    layer = module.nn.Conv2d(2, 3, 3, padding=1, bias=bias)
    _copy_parameter(
        module,
        layer.weight,
        _tensor(
            module,
            [
                [
                    [[0.25, -0.5, 0.75], [1.0, 0.0, -1.0], [0.5, 0.25, -0.25]],
                    [[-0.25, 0.5, 1.0], [0.75, -0.5, 0.25], [-1.0, 0.25, 0.5]],
                ],
                [
                    [[1.0, 0.5, -0.5], [-0.25, 0.75, 0.25], [0.5, -1.0, 0.25]],
                    [[0.25, -0.75, 0.5], [1.0, -0.5, 0.0], [-0.25, 0.5, 0.75]],
                ],
                [
                    [[-0.5, 0.25, 1.0], [0.5, -0.25, 0.75], [1.0, -0.5, 0.25]],
                    [[0.75, 0.25, -1.0], [-0.5, 1.0, 0.5], [0.25, -0.75, 0.5]],
                ],
            ],
        ),
    )
    if bias:
        _copy_parameter(module, layer.bias, _tensor(module, [0.5, -1.0, 0.25]))
    return layer


def _conv3d(module: Any, bias: bool = True) -> Any:
    layer = module.nn.Conv3d(2, 2, 2, padding=1, bias=bias)
    _copy_parameter(
        module,
        layer.weight,
        _tensor(
            module,
            [
                [
                    [
                        [[0.25, -0.5], [1.0, 0.0]],
                        [[-1.0, 0.5], [0.75, -0.25]],
                    ],
                    [
                        [[0.5, 0.25], [-0.75, 1.0]],
                        [[-0.5, 1.25], [0.0, -1.0]],
                    ],
                ],
                [
                    [
                        [[-0.25, 0.75], [0.5, -1.0]],
                        [[1.0, -0.5], [0.25, 0.5]],
                    ],
                    [
                        [[0.75, -0.25], [1.0, 0.5]],
                        [[-1.25, 0.25], [0.5, -0.75]],
                    ],
                ],
            ],
        ),
    )
    if bias:
        _copy_parameter(module, layer.bias, _tensor(module, [0.5, -1.0]))
    return layer


def _conv_transpose3d(module: Any, bias: bool = True) -> Any:
    layer = module.nn.ConvTranspose3d(2, 2, 2, stride=2, padding=1, output_padding=1, bias=bias)
    _copy_parameter(
        module,
        layer.weight,
        _tensor(
            module,
            [
                [
                    [
                        [[0.25, -0.5], [1.0, 0.0]],
                        [[-1.0, 0.5], [0.75, -0.25]],
                    ],
                    [
                        [[0.5, 0.25], [-0.75, 1.0]],
                        [[-0.5, 1.25], [0.0, -1.0]],
                    ],
                ],
                [
                    [
                        [[-0.25, 0.75], [0.5, -1.0]],
                        [[1.0, -0.5], [0.25, 0.5]],
                    ],
                    [
                        [[0.75, -0.25], [1.0, 0.5]],
                        [[-1.25, 0.25], [0.5, -0.75]],
                    ],
                ],
            ],
        ),
    )
    if bias:
        _copy_parameter(module, layer.bias, _tensor(module, [0.5, -1.0]))
    return layer


def _conv_transpose2d(module: Any, bias: bool = True) -> Any:
    layer = module.nn.ConvTranspose2d(2, 3, 3, stride=2, padding=1, output_padding=1, bias=bias)
    _copy_parameter(
        module,
        layer.weight,
        _tensor(
            module,
            [
                [
                    [[0.25, 1.0, 0.5], [-0.5, 0.0, 0.25], [0.75, -1.0, -0.25]],
                    [[1.0, -0.25, 0.5], [0.5, 0.75, -1.0], [-0.5, 0.25, 0.25]],
                    [[-0.5, 0.5, 1.0], [0.25, -0.25, -0.5], [1.0, 0.75, 0.25]],
                ],
                [
                    [[-0.25, 0.75, -1.0], [0.5, -0.5, 0.25], [1.0, 0.25, 0.5]],
                    [[0.25, 1.0, -0.25], [-0.75, -0.5, 0.5], [0.5, 0.0, 0.75]],
                    [[0.75, -0.5, 0.25], [0.25, 1.0, -0.75], [-1.0, 0.5, 0.5]],
                ],
            ],
        ),
    )
    if bias:
        _copy_parameter(module, layer.bias, _tensor(module, [0.5, -1.0, 0.25]))
    return layer


def _multihead_attention(module: Any, **kwargs: Any) -> Any:
    layer = module.nn.MultiheadAttention(4, 2, dropout=0.0, batch_first=True, **kwargs)
    _copy_parameter(
        module,
        layer.in_proj_weight,
        _tensor(
            module,
            [
                [0.25, -0.5, 0.75, 1.0],
                [-0.25, 0.5, 1.0, -0.75],
                [1.0, 0.25, -0.5, 0.5],
                [-1.0, 0.75, 0.25, -0.25],
                [0.5, -1.0, 0.25, 0.75],
                [1.0, 0.5, -0.5, 1.5],
                [-0.25, 0.75, 1.0, -1.0],
                [0.25, -0.75, 0.5, 1.0],
                [1.0, -0.5, 0.25, 0.75],
                [0.5, 1.5, -1.0, 0.25],
                [-0.25, 0.75, 1.0, -0.5],
                [1.25, -0.25, 0.5, -1.0],
            ],
        ),
    )
    _copy_parameter(module, layer.in_proj_bias, _tensor(module, [0.1, -0.2, 0.3, -0.4, 0.25, -0.5, 0.75, 0.5, -0.1, 0.2, -0.3, 0.4]))
    _copy_parameter(
        module,
        layer.out_proj.weight,
        _tensor(module, [[0.25, -0.5, 0.75, 1.0], [-0.25, 0.5, 1.0, -0.75], [1.0, 0.25, -0.5, 0.5], [-1.0, 0.75, 0.25, -0.25]]),
    )
    _copy_parameter(module, layer.out_proj.bias, _tensor(module, [0.5, -0.25, 0.75, -0.5]))
    if getattr(layer, "bias_k", None) is not None:
        _copy_parameter(module, layer.bias_k, _tensor(module, [[[0.25, -0.5, 0.75, -1.0]]]))
        _copy_parameter(module, layer.bias_v, _tensor(module, [[[1.0, 0.5, -0.25, 0.75]]]))
    return layer


def test_linear_module_forward_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _linear(torch_reference)
    actual_layer = _linear(torch_candidate)
    expected_input = _tensor(torch_reference, [[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]])
    actual_input = _tensor(torch_candidate, [[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]])

    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input),
        actual_layer(actual_input),
        path="nn.Linear.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_linear_module_nd_forward_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _linear(torch_reference)
    actual_layer = _linear(torch_candidate)
    expected_input = _tensor(
        torch_reference,
        [
            [[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]],
            [[-1.0, 0.25, 2.0], [3.0, -0.5, 1.0]],
        ],
    )
    actual_input = _tensor(
        torch_candidate,
        [
            [[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]],
            [[-1.0, 0.25, 2.0], [3.0, -0.5, 1.0]],
        ],
    )

    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input),
        actual_layer(actual_input),
        path="nn.Linear.nd_forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_conv1d_module_forward_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _conv1d(torch_reference)
    actual_layer = _conv1d(torch_candidate)
    expected_input = _tensor(
        torch_reference,
        [[[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, 0.5, 2.0, -2.0, 1.5]]],
    )
    actual_input = _tensor(
        torch_candidate,
        [[[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, 0.5, 2.0, -2.0, 1.5]]],
    )

    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input),
        actual_layer(actual_input),
        path="nn.Conv1d.forward",
        rtol=1e-5,
        atol=1e-5,
    )


def test_conv_transpose1d_module_forward_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _conv_transpose1d(torch_reference)
    actual_layer = _conv_transpose1d(torch_candidate)
    input_data = [[[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]]
    expected_input = _tensor(torch_reference, input_data)
    actual_input = _tensor(torch_candidate, input_data)

    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input),
        actual_layer(actual_input),
        path="nn.ConvTranspose1d.forward",
        rtol=1e-5,
        atol=1e-5,
    )


def test_conv2d_module_forward_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _conv2d(torch_reference)
    actual_layer = _conv2d(torch_candidate)
    expected_input = _tensor(
        torch_reference,
        [
            [
                [[1.0, 2.0, 3.0], [4.0, -1.0, 0.5], [2.5, 1.5, -2.0]],
                [[-1.0, 0.25, 2.0], [3.0, -0.5, 1.0], [0.0, 2.5, -1.5]],
            ]
        ],
    )
    actual_input = _tensor(
        torch_candidate,
        [
            [
                [[1.0, 2.0, 3.0], [4.0, -1.0, 0.5], [2.5, 1.5, -2.0]],
                [[-1.0, 0.25, 2.0], [3.0, -0.5, 1.0], [0.0, 2.5, -1.5]],
            ]
        ],
    )

    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input),
        actual_layer(actual_input),
        path="nn.Conv2d.forward",
        rtol=1e-5,
        atol=1e-5,
    )


def test_conv3d_module_forward_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _conv3d(torch_reference)
    actual_layer = _conv3d(torch_candidate)
    input_data = [
        [
            [
                [[1.0, -2.0, 3.0], [0.5, 1.5, -1.0], [2.0, -0.5, 0.25]],
                [[-1.0, 2.5, 0.0], [3.0, -3.5, 1.0], [0.75, 1.25, -2.0]],
                [[4.0, -1.5, 2.0], [0.0, 0.5, -0.75], [1.5, -2.5, 3.0]],
            ],
            [
                [[-0.5, 1.0, 2.5], [3.5, -1.0, 0.0], [1.25, -2.25, 0.5]],
                [[2.0, -3.0, 1.5], [0.25, 0.75, -1.25], [3.0, 1.0, -0.5]],
                [[-2.0, 0.5, 1.0], [2.25, -0.75, 3.5], [0.0, 1.5, -1.5]],
            ],
        ]
    ]
    expected_input = _tensor(torch_reference, input_data)
    actual_input = _tensor(torch_candidate, input_data)

    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input),
        actual_layer(actual_input),
        path="nn.Conv3d.forward",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_conv_modules_string_padding_match_reference(torch_reference, torch_candidate) -> None:
    expected_conv1d = torch_reference.nn.Conv1d(1, 1, 3, stride=2, padding="valid", bias=False)
    actual_conv1d = torch_candidate.nn.Conv1d(1, 1, 3, stride=2, padding="valid", bias=False)
    _copy_parameter(torch_reference, expected_conv1d.weight, _tensor(torch_reference, [[[0.25, -0.5, 1.0]]]))
    _copy_parameter(torch_candidate, actual_conv1d.weight, _tensor(torch_candidate, [[[0.25, -0.5, 1.0]]]))
    expected_input1d = _tensor(torch_reference, [[[1.0, -2.0, 3.0, 0.5, -1.5, 2.0]]])
    actual_input1d = _tensor(torch_candidate, [[[1.0, -2.0, 3.0, 0.5, -1.5, 2.0]]])

    assert_values_compatible(
        torch_reference,
        expected_conv1d(expected_input1d),
        actual_conv1d(actual_input1d),
        path="nn.Conv1d.padding_valid",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )

    expected_conv2d = torch_reference.nn.Conv2d(1, 1, 3, padding="same", bias=False)
    actual_conv2d = torch_candidate.nn.Conv2d(1, 1, 3, padding="same", bias=False)
    _copy_parameter(
        torch_reference,
        expected_conv2d.weight,
        _tensor(torch_reference, [[[[0.25, -0.5, 1.0], [0.75, 0.0, -1.0], [0.5, 0.25, -0.25]]]]),
    )
    _copy_parameter(
        torch_candidate,
        actual_conv2d.weight,
        _tensor(torch_candidate, [[[[0.25, -0.5, 1.0], [0.75, 0.0, -1.0], [0.5, 0.25, -0.25]]]]),
    )
    expected_input2d = _tensor(torch_reference, [[[[1.0, -2.0, 3.0], [0.5, -1.5, 2.0], [1.25, 0.0, -0.75]]]])
    actual_input2d = _tensor(torch_candidate, [[[[1.0, -2.0, 3.0], [0.5, -1.5, 2.0], [1.25, 0.0, -0.75]]]])

    assert_values_compatible(
        torch_reference,
        expected_conv2d(expected_input2d),
        actual_conv2d(actual_input2d),
        path="nn.Conv2d.padding_same",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_conv_modules_padding_mode_match_reference(torch_reference, torch_candidate) -> None:
    expected_conv1d = torch_reference.nn.Conv1d(1, 1, 3, padding=1, padding_mode="replicate", bias=False)
    actual_conv1d = torch_candidate.nn.Conv1d(1, 1, 3, padding=1, padding_mode="replicate", bias=False)
    _copy_parameter(torch_reference, expected_conv1d.weight, _tensor(torch_reference, [[[0.25, -0.5, 1.0]]]))
    _copy_parameter(torch_candidate, actual_conv1d.weight, _tensor(torch_candidate, [[[0.25, -0.5, 1.0]]]))
    expected_input1d = _tensor(torch_reference, [[[1.0, -2.0, 3.0, 0.5]]])
    actual_input1d = _tensor(torch_candidate, [[[1.0, -2.0, 3.0, 0.5]]])

    assert_values_compatible(
        torch_reference,
        expected_conv1d(expected_input1d),
        actual_conv1d(actual_input1d),
        path="nn.Conv1d.padding_mode_replicate",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )

    expected_conv2d = torch_reference.nn.Conv2d(1, 1, 3, padding="same", padding_mode="reflect", bias=False)
    actual_conv2d = torch_candidate.nn.Conv2d(1, 1, 3, padding="same", padding_mode="reflect", bias=False)
    weight = [[[[0.25, -0.5, 1.0], [0.75, 0.0, -1.0], [0.5, 0.25, -0.25]]]]
    _copy_parameter(torch_reference, expected_conv2d.weight, _tensor(torch_reference, weight))
    _copy_parameter(torch_candidate, actual_conv2d.weight, _tensor(torch_candidate, weight))
    expected_input2d = _tensor(torch_reference, [[[[1.0, -2.0, 3.0], [0.5, -1.5, 2.0], [1.25, 0.0, -0.75]]]])
    actual_input2d = _tensor(torch_candidate, [[[[1.0, -2.0, 3.0], [0.5, -1.5, 2.0], [1.25, 0.0, -0.75]]]])

    assert_values_compatible(
        torch_reference,
        expected_conv2d(expected_input2d),
        actual_conv2d(actual_input2d),
        path="nn.Conv2d.padding_mode_reflect_same",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_conv_transpose2d_module_forward_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _conv_transpose2d(torch_reference)
    actual_layer = _conv_transpose2d(torch_candidate)
    expected_input = _tensor(
        torch_reference,
        [
            [
                [[1.0, 2.0, 3.0], [4.0, -1.0, 0.5], [2.5, 1.5, -2.0]],
                [[-1.0, 0.25, 2.0], [3.0, -0.5, 1.0], [0.0, 2.5, -1.5]],
            ]
        ],
    )
    actual_input = _tensor(
        torch_candidate,
        [
            [
                [[1.0, 2.0, 3.0], [4.0, -1.0, 0.5], [2.5, 1.5, -2.0]],
                [[-1.0, 0.25, 2.0], [3.0, -0.5, 1.0], [0.0, 2.5, -1.5]],
            ]
        ],
    )

    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input),
        actual_layer(actual_input),
        path="nn.ConvTranspose2d.forward",
        rtol=1e-5,
        atol=1e-5,
    )


def test_conv_transpose3d_module_forward_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _conv_transpose3d(torch_reference)
    actual_layer = _conv_transpose3d(torch_candidate)
    input_data = [
        [
            [
                [[1.0, -2.0, 3.0], [0.5, 1.5, -1.0], [2.0, -0.5, 0.25]],
                [[-1.0, 2.5, 0.0], [3.0, -3.5, 1.0], [0.75, 1.25, -2.0]],
                [[4.0, -1.5, 2.0], [0.0, 0.5, -0.75], [1.5, -2.5, 3.0]],
            ],
            [
                [[-0.5, 1.0, 2.5], [3.5, -1.0, 0.0], [1.25, -2.25, 0.5]],
                [[2.0, -3.0, 1.5], [0.25, 0.75, -1.25], [3.0, 1.0, -0.5]],
                [[-2.0, 0.5, 1.0], [2.25, -0.75, 3.5], [0.0, 1.5, -1.5]],
            ],
        ]
    ]
    expected_input = _tensor(torch_reference, input_data)
    actual_input = _tensor(torch_candidate, input_data)

    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input),
        actual_layer(actual_input),
        path="nn.ConvTranspose3d.forward",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_conv_transpose2d_module_output_size_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _conv_transpose2d(torch_reference)
    actual_layer = _conv_transpose2d(torch_candidate)
    input_data = [
        [
            [[1.0, 2.0, 3.0], [4.0, -1.0, 0.5], [2.5, 1.5, -2.0]],
            [[-1.0, 0.25, 2.0], [3.0, -0.5, 1.0], [0.0, 2.5, -1.5]],
        ]
    ]
    expected_input = _tensor(torch_reference, input_data)
    actual_input = _tensor(torch_candidate, input_data)

    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input, output_size=(1, 3, 5, 5)),
        actual_layer(actual_input, output_size=(1, 3, 5, 5)),
        path="nn.ConvTranspose2d.output_size",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_conv_transpose3d_module_output_size_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _conv_transpose3d(torch_reference)
    actual_layer = _conv_transpose3d(torch_candidate)
    input_data = [
        [
            [
                [[1.0, -2.0, 3.0], [0.5, 1.5, -1.0], [2.0, -0.5, 0.25]],
                [[-1.0, 2.5, 0.0], [3.0, -3.5, 1.0], [0.75, 1.25, -2.0]],
                [[4.0, -1.5, 2.0], [0.0, 0.5, -0.75], [1.5, -2.5, 3.0]],
            ],
            [
                [[-0.5, 1.0, 2.5], [3.5, -1.0, 0.0], [1.25, -2.25, 0.5]],
                [[2.0, -3.0, 1.5], [0.25, 0.75, -1.25], [3.0, 1.0, -0.5]],
                [[-2.0, 0.5, 1.0], [2.25, -0.75, 3.5], [0.0, 1.5, -1.5]],
            ],
        ]
    ]
    expected_input = _tensor(torch_reference, input_data)
    actual_input = _tensor(torch_candidate, input_data)

    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input, output_size=(1, 2, 4, 4, 4)),
        actual_layer(actual_input, output_size=(1, 2, 4, 4, 4)),
        path="nn.ConvTranspose3d.output_size",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_conv_transpose1d_module_output_size_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _conv_transpose1d(torch_reference)
    actual_layer = _conv_transpose1d(torch_candidate)
    input_data = [[[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]]
    expected_input = _tensor(torch_reference, input_data)
    actual_input = _tensor(torch_candidate, input_data)

    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input, output_size=(1, 3, 6)),
        actual_layer(actual_input, output_size=(1, 3, 6)),
        path="nn.ConvTranspose1d.output_size",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_pool1d_modules_forward_match_reference(torch_reference, torch_candidate) -> None:
    input_data = [[[1.0, -2.0, 3.0, 0.5, 4.0, 1.5], [0.25, 5.0, -0.5, 1.0, 3.0, -1.5]]]
    expected_input = _tensor(torch_reference, input_data)
    actual_input = _tensor(torch_candidate, input_data)

    for layer_name in ("MaxPool1d", "AvgPool1d"):
        expected_layer = getattr(torch_reference.nn, layer_name)(2, stride=2)
        actual_layer = getattr(torch_candidate.nn, layer_name)(2, stride=2)
        assert_values_compatible(
            torch_reference,
            expected_layer(expected_input),
            actual_layer(actual_input),
            path=f"nn.{layer_name}.forward",
            rtol=1e-6,
            atol=1e-6,
            check_stride=False,
        )


def test_pool2d_modules_forward_match_reference(torch_reference, torch_candidate) -> None:
    input_data = [
        [
            [
                [1.0, -2.0, 3.0, 0.5],
                [4.0, 1.5, -1.0, 2.0],
                [0.25, 5.0, -0.5, 1.0],
                [3.0, -1.5, 2.5, -2.0],
            ]
        ]
    ]
    expected_input = _tensor(torch_reference, input_data)
    actual_input = _tensor(torch_candidate, input_data)

    for layer_name in ("MaxPool2d", "AvgPool2d"):
        expected_layer = getattr(torch_reference.nn, layer_name)(2, stride=2)
        actual_layer = getattr(torch_candidate.nn, layer_name)(2, stride=2)
        assert_values_compatible(
            torch_reference,
            expected_layer(expected_input),
            actual_layer(actual_input),
            path=f"nn.{layer_name}.forward",
            rtol=1e-6,
            atol=1e-6,
            check_stride=False,
        )


def test_upsample_module_forward_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = torch_reference.nn.Upsample(scale_factor=2.0, mode="nearest")
    actual_layer = torch_candidate.nn.Upsample(scale_factor=2.0, mode="nearest")
    expected_input = _tensor(
        torch_reference,
        [[[[1.0, -2.0], [3.0, 0.5]], [[-1.0, 2.5], [0.0, -0.5]]]],
    )
    actual_input = _tensor(
        torch_candidate,
        [[[[1.0, -2.0], [3.0, 0.5]], [[-1.0, 2.5], [0.0, -0.5]]]],
    )

    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input),
        actual_layer(actual_input),
        path="nn.Upsample.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_multihead_attention_batch_first_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _multihead_attention(torch_reference)
    actual_layer = _multihead_attention(torch_candidate)
    expected_input = _tensor(
        torch_reference,
        [[[0.25, -0.5, 1.0, 0.75], [1.5, 0.25, -0.75, 0.5], [-1.0, 0.5, 0.25, 1.25]]],
    )
    actual_input = _tensor(
        torch_candidate,
        [[[0.25, -0.5, 1.0, 0.75], [1.5, 0.25, -0.75, 0.5], [-1.0, 0.5, 0.25, 1.25]]],
    )
    expected, _ = expected_layer(expected_input, expected_input, expected_input, need_weights=False)
    actual, _ = actual_layer(actual_input, actual_input, actual_input, need_weights=False)

    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="nn.MultiheadAttention.forward",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_multihead_attention_cross_same_embed_dim_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _multihead_attention(torch_reference)
    actual_layer = _multihead_attention(torch_candidate)
    expected_query = _tensor(torch_reference, [[[0.25, -0.5, 1.0, 0.75], [1.5, 0.25, -0.75, 0.5]]])
    actual_query = _tensor(torch_candidate, [[[0.25, -0.5, 1.0, 0.75], [1.5, 0.25, -0.75, 0.5]]])
    expected_key = _tensor(
        torch_reference,
        [[[0.5, -1.0, 0.25, 1.25], [1.0, 0.5, -0.5, 0.75], [-0.25, 0.75, 1.0, -1.0]]],
    )
    actual_key = _tensor(
        torch_candidate,
        [[[0.5, -1.0, 0.25, 1.25], [1.0, 0.5, -0.5, 0.75], [-0.25, 0.75, 1.0, -1.0]]],
    )
    expected_value = _tensor(
        torch_reference,
        [[[1.0, -0.5, 0.25, 0.75], [0.5, 1.5, -1.0, 0.25], [-0.25, 0.75, 1.0, -0.5]]],
    )
    actual_value = _tensor(
        torch_candidate,
        [[[1.0, -0.5, 0.25, 0.75], [0.5, 1.5, -1.0, 0.25], [-0.25, 0.75, 1.0, -0.5]]],
    )

    expected_output, expected_weights = expected_layer(expected_query, expected_key, expected_value, need_weights=True)
    actual_output, actual_weights = actual_layer(actual_query, actual_key, actual_value, need_weights=True)

    assert_values_compatible(
        torch_reference,
        expected_output,
        actual_output,
        path="nn.MultiheadAttention.cross_same_embed_dim.output",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )
    assert_values_compatible(
        torch_reference,
        expected_weights,
        actual_weights,
        path="nn.MultiheadAttention.cross_same_embed_dim.weights",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_multihead_attention_bool_masks_match_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _multihead_attention(torch_reference)
    actual_layer = _multihead_attention(torch_candidate)
    expected_input = _tensor(
        torch_reference,
        [[[0.25, -0.5, 1.0, 0.75], [1.5, 0.25, -0.75, 0.5], [-1.0, 0.5, 0.25, 1.25]]],
    )
    actual_input = _tensor(
        torch_candidate,
        [[[0.25, -0.5, 1.0, 0.75], [1.5, 0.25, -0.75, 0.5], [-1.0, 0.5, 0.25, 1.25]]],
    )
    expected_attn_mask = torch_reference.tensor(
        [[False, True, False], [False, False, False], [True, False, False]],
        dtype=torch_reference.bool,
    )
    actual_attn_mask = torch_candidate.tensor(
        [[False, True, False], [False, False, False], [True, False, False]],
        dtype=torch_candidate.bool,
    )
    expected_key_padding_mask = torch_reference.tensor([[False, False, True]], dtype=torch_reference.bool)
    actual_key_padding_mask = torch_candidate.tensor([[False, False, True]], dtype=torch_candidate.bool)

    expected, _ = expected_layer(
        expected_input,
        expected_input,
        expected_input,
        key_padding_mask=expected_key_padding_mask,
        need_weights=False,
        attn_mask=expected_attn_mask,
    )
    actual, _ = actual_layer(
        actual_input,
        actual_input,
        actual_input,
        key_padding_mask=actual_key_padding_mask,
        need_weights=False,
        attn_mask=actual_attn_mask,
    )

    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="nn.MultiheadAttention.bool_masks",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_multihead_attention_add_zero_attn_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _multihead_attention(torch_reference, add_zero_attn=True)
    actual_layer = _multihead_attention(torch_candidate, add_zero_attn=True)
    input_data = [[[0.25, -0.5, 1.0, 0.75], [1.5, 0.25, -0.75, 0.5], [-1.0, 0.5, 0.25, 1.25]]]
    expected_input = _tensor(torch_reference, input_data)
    actual_input = _tensor(torch_candidate, input_data)
    expected_key_padding_mask = torch_reference.tensor([[False, False, True]], dtype=torch_reference.bool)
    actual_key_padding_mask = torch_candidate.tensor([[False, False, True]], dtype=torch_candidate.bool)

    expected, _ = expected_layer(expected_input, expected_input, expected_input, key_padding_mask=expected_key_padding_mask, need_weights=False)
    actual, _ = actual_layer(actual_input, actual_input, actual_input, key_padding_mask=actual_key_padding_mask, need_weights=False)

    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="nn.MultiheadAttention.add_zero_attn",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_multihead_attention_add_bias_kv_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _multihead_attention(torch_reference, add_bias_kv=True)
    actual_layer = _multihead_attention(torch_candidate, add_bias_kv=True)
    input_data = [[[0.25, -0.5, 1.0, 0.75], [1.5, 0.25, -0.75, 0.5], [-1.0, 0.5, 0.25, 1.25]]]
    expected_input = _tensor(torch_reference, input_data)
    actual_input = _tensor(torch_candidate, input_data)
    expected_attn_mask = torch_reference.tensor(
        [[False, True, False], [False, False, False], [True, False, False]],
        dtype=torch_reference.bool,
    )
    actual_attn_mask = torch_candidate.tensor(
        [[False, True, False], [False, False, False], [True, False, False]],
        dtype=torch_candidate.bool,
    )

    expected, _ = expected_layer(expected_input, expected_input, expected_input, need_weights=False, attn_mask=expected_attn_mask)
    actual, _ = actual_layer(actual_input, actual_input, actual_input, need_weights=False, attn_mask=actual_attn_mask)

    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="nn.MultiheadAttention.add_bias_kv",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_multihead_attention_separate_kdim_vdim_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = torch_reference.nn.MultiheadAttention(4, 2, kdim=3, vdim=5, dropout=0.0, batch_first=True)
    actual_layer = torch_candidate.nn.MultiheadAttention(4, 2, kdim=3, vdim=5, dropout=0.0, batch_first=True)
    _copy_parameter(torch_reference, expected_layer.q_proj_weight, _tensor(torch_reference, [[0.25, -0.5, 0.75, 1.0], [-0.25, 0.5, 1.0, -0.75], [1.0, 0.25, -0.5, 0.5], [-1.0, 0.75, 0.25, -0.25]]))
    _copy_parameter(torch_candidate, actual_layer.q_proj_weight, _tensor(torch_candidate, [[0.25, -0.5, 0.75, 1.0], [-0.25, 0.5, 1.0, -0.75], [1.0, 0.25, -0.5, 0.5], [-1.0, 0.75, 0.25, -0.25]]))
    _copy_parameter(torch_reference, expected_layer.k_proj_weight, _tensor(torch_reference, [[0.5, -1.0, 0.25], [1.0, 0.5, -0.5], [-0.25, 0.75, 1.0], [0.25, -0.75, 0.5]]))
    _copy_parameter(torch_candidate, actual_layer.k_proj_weight, _tensor(torch_candidate, [[0.5, -1.0, 0.25], [1.0, 0.5, -0.5], [-0.25, 0.75, 1.0], [0.25, -0.75, 0.5]]))
    _copy_parameter(torch_reference, expected_layer.v_proj_weight, _tensor(torch_reference, [[1.0, -0.5, 0.25, 0.75, -1.0], [0.5, 1.5, -1.0, 0.25, 0.5], [-0.25, 0.75, 1.0, -0.5, 1.25], [1.25, -0.25, 0.5, -1.0, 0.75]]))
    _copy_parameter(torch_candidate, actual_layer.v_proj_weight, _tensor(torch_candidate, [[1.0, -0.5, 0.25, 0.75, -1.0], [0.5, 1.5, -1.0, 0.25, 0.5], [-0.25, 0.75, 1.0, -0.5, 1.25], [1.25, -0.25, 0.5, -1.0, 0.75]]))
    _copy_parameter(torch_reference, expected_layer.in_proj_bias, _tensor(torch_reference, [0.1, -0.2, 0.3, -0.4, 0.25, -0.5, 0.75, 0.5, -0.1, 0.2, -0.3, 0.4]))
    _copy_parameter(torch_candidate, actual_layer.in_proj_bias, _tensor(torch_candidate, [0.1, -0.2, 0.3, -0.4, 0.25, -0.5, 0.75, 0.5, -0.1, 0.2, -0.3, 0.4]))
    _copy_parameter(
        torch_reference,
        expected_layer.out_proj.weight,
        _tensor(torch_reference, [[0.25, -0.5, 0.75, 1.0], [-0.25, 0.5, 1.0, -0.75], [1.0, 0.25, -0.5, 0.5], [-1.0, 0.75, 0.25, -0.25]]),
    )
    _copy_parameter(
        torch_candidate,
        actual_layer.out_proj.weight,
        _tensor(torch_candidate, [[0.25, -0.5, 0.75, 1.0], [-0.25, 0.5, 1.0, -0.75], [1.0, 0.25, -0.5, 0.5], [-1.0, 0.75, 0.25, -0.25]]),
    )
    _copy_parameter(torch_reference, expected_layer.out_proj.bias, _tensor(torch_reference, [0.5, -0.25, 0.75, -0.5]))
    _copy_parameter(torch_candidate, actual_layer.out_proj.bias, _tensor(torch_candidate, [0.5, -0.25, 0.75, -0.5]))
    expected_query = _tensor(torch_reference, [[[0.25, -0.5, 1.0, 0.75], [1.5, 0.25, -0.75, 0.5]]])
    actual_query = _tensor(torch_candidate, [[[0.25, -0.5, 1.0, 0.75], [1.5, 0.25, -0.75, 0.5]]])
    expected_key = _tensor(torch_reference, [[[0.5, -1.0, 0.25], [1.0, 0.5, -0.5], [-0.25, 0.75, 1.0]]])
    actual_key = _tensor(torch_candidate, [[[0.5, -1.0, 0.25], [1.0, 0.5, -0.5], [-0.25, 0.75, 1.0]]])
    expected_value = _tensor(torch_reference, [[[1.0, -0.5, 0.25, 0.75, -1.0], [0.5, 1.5, -1.0, 0.25, 0.5], [-0.25, 0.75, 1.0, -0.5, 1.25]]])
    actual_value = _tensor(torch_candidate, [[[1.0, -0.5, 0.25, 0.75, -1.0], [0.5, 1.5, -1.0, 0.25, 0.5], [-0.25, 0.75, 1.0, -0.5, 1.25]]])

    expected, _ = expected_layer(expected_query, expected_key, expected_value, need_weights=False)
    actual, _ = actual_layer(actual_query, actual_key, actual_value, need_weights=False)

    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="nn.MultiheadAttention.separate_kdim_vdim",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_parameter_defaults_to_requires_grad(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.Parameter(_tensor(torch_reference, [1.0, 2.0]))
    actual = torch_candidate.nn.Parameter(_tensor(torch_candidate, [1.0, 2.0]))

    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="nn.Parameter",
    )


def test_linear_module_without_bias_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _linear(torch_reference, bias=False)
    actual_layer = _linear(torch_candidate, bias=False)
    expected_input = _tensor(torch_reference, [1.0, 2.0, -1.0])
    actual_input = _tensor(torch_candidate, [1.0, 2.0, -1.0])

    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input),
        actual_layer(actual_input),
        path="nn.Linear.forward_no_bias",
        rtol=1e-6,
        atol=1e-6,
    )


def test_module_to_dtype_converts_parameters_and_forward_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _linear(torch_reference).to(dtype=torch_reference.float64, non_blocking=True)
    actual_layer = _linear(torch_candidate).to(dtype=torch_candidate.float64, non_blocking=True)
    expected_input = _tensor(torch_reference, [[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]]).to(dtype=torch_reference.float64)
    actual_input = _tensor(torch_candidate, [[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]]).to(dtype=torch_candidate.float64)

    assert expected_layer.weight.dtype == torch_reference.float64
    assert actual_layer.weight.dtype == torch_candidate.float64
    assert expected_layer.bias.dtype == torch_reference.float64
    assert actual_layer.bias.dtype == torch_candidate.float64
    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input),
        actual_layer(actual_input),
        path="nn.Module.to.dtype.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_module_to_channels_last_matches_reference(torch_reference, torch_candidate) -> None:
    expected_layer = _conv2d(torch_reference).to(memory_format=torch_reference.channels_last)
    actual_layer = _conv2d(torch_candidate).to(memory_format=torch_candidate.channels_last)
    input_data = [
        [
            [[1.0, 2.0, 3.0], [4.0, -1.0, 0.5], [2.5, 1.5, -2.0]],
            [[-1.0, 0.25, 2.0], [3.0, -0.5, 1.0], [0.0, 2.5, -1.5]],
        ]
    ]
    expected_input = _tensor(torch_reference, input_data).to(memory_format=torch_reference.channels_last)
    actual_input = _tensor(torch_candidate, input_data).to(memory_format=torch_candidate.channels_last)

    assert actual_layer.weight.stride() == expected_layer.weight.stride()
    assert actual_layer.weight.is_contiguous(memory_format=torch_candidate.channels_last)
    assert actual_layer.bias.stride() == expected_layer.bias.stride()
    assert_values_compatible(
        torch_reference,
        expected_layer(expected_input),
        actual_layer(actual_input),
        path="nn.Module.to.memory_format.channels_last.forward",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_module_to_dtype_converts_float_buffers_only(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.BatchNorm1d(3).to(dtype=torch_reference.float64)
    actual = torch_candidate.nn.BatchNorm1d(3).to(dtype=torch_candidate.float64)

    assert expected.running_mean.dtype == torch_reference.float64
    assert actual.running_mean.dtype == torch_candidate.float64
    assert expected.running_var.dtype == torch_reference.float64
    assert actual.running_var.dtype == torch_candidate.float64
    assert expected.num_batches_tracked.dtype == torch_reference.int64
    assert actual.num_batches_tracked.dtype == torch_candidate.int64


def test_module_requires_grad_inplace_matches_reference(torch_reference, torch_candidate) -> None:
    expected = _linear(torch_reference).requires_grad_(False)
    actual = _linear(torch_candidate).requires_grad_(False)

    assert [parameter.requires_grad for parameter in expected.parameters()] == [
        parameter.requires_grad for parameter in actual.parameters()
    ]
    assert actual.requires_grad_(True) is actual
    assert all(parameter.requires_grad for parameter in actual.parameters())


def test_module_zero_grad_matches_reference(torch_reference, torch_candidate) -> None:
    expected = _linear(torch_reference)
    actual = _linear(torch_candidate)
    expected_input = _tensor(torch_reference, [[1.0, 2.0, -1.0]])
    actual_input = _tensor(torch_candidate, [[1.0, 2.0, -1.0]])
    expected(expected_input).sum().backward()
    actual(actual_input).sum().backward()

    expected.zero_grad(set_to_none=False)
    actual.zero_grad(set_to_none=False)
    for index, (expected_parameter, actual_parameter) in enumerate(zip(expected.parameters(), actual.parameters(), strict=True)):
        assert_values_compatible(
            torch_reference,
            expected_parameter.grad,
            actual_parameter.grad,
            path=f"nn.Module.zero_grad[{index}]",
            rtol=1e-6,
            atol=1e-6,
        )

    expected(expected_input).sum().backward()
    actual(actual_input).sum().backward()
    expected.zero_grad()
    actual.zero_grad()
    assert [parameter.grad for parameter in expected.parameters()] == [parameter.grad for parameter in actual.parameters()]


def test_embedding_module_matches_reference(torch_reference, torch_candidate) -> None:
    weight = [[0.0, 0.25, -0.5], [1.0, -1.5, 2.0], [0.5, 0.75, -1.0], [-0.25, 1.25, 1.5]]
    expected = torch_reference.nn.Embedding(4, 3)
    actual = torch_candidate.nn.Embedding(4, 3)
    _copy_parameter(torch_reference, expected.weight, _tensor(torch_reference, weight))
    _copy_parameter(torch_candidate, actual.weight, _tensor(torch_candidate, weight))
    expected_input = torch_reference.tensor([[1, 0], [3, 2]], dtype=torch_reference.int64)
    actual_input = torch_candidate.tensor([[1, 0], [3, 2]], dtype=torch_candidate.int64)

    assert_values_compatible(
        torch_reference,
        expected(expected_input),
        actual(actual_input),
        path="nn.Embedding.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_embedding_backward_matches_reference(torch_reference, torch_candidate) -> None:
    weight = [[0.0, 0.25, -0.5], [1.0, -1.5, 2.0], [0.5, 0.75, -1.0], [-0.25, 1.25, 1.5]]
    expected = torch_reference.nn.Embedding(4, 3)
    actual = torch_candidate.nn.Embedding(4, 3)
    _copy_parameter(torch_reference, expected.weight, _tensor(torch_reference, weight))
    _copy_parameter(torch_candidate, actual.weight, _tensor(torch_candidate, weight))
    expected_input = torch_reference.tensor([1, 0, 1, 3], dtype=torch_reference.int64)
    actual_input = torch_candidate.tensor([1, 0, 1, 3], dtype=torch_candidate.int64)

    expected(expected_input).sum().backward()
    actual(actual_input).sum().backward()

    assert_values_compatible(
        torch_reference,
        expected.weight.grad,
        actual.weight.grad,
        path="nn.Embedding.backward",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_embedding_padding_idx_backward_matches_reference(torch_reference, torch_candidate) -> None:
    weight = [[0.0, 0.25, -0.5], [1.0, -1.5, 2.0], [0.5, 0.75, -1.0], [-0.25, 1.25, 1.5]]
    expected = torch_reference.nn.Embedding(4, 3, padding_idx=0)
    actual = torch_candidate.nn.Embedding(4, 3, padding_idx=0)
    _copy_parameter(torch_reference, expected.weight, _tensor(torch_reference, weight))
    _copy_parameter(torch_candidate, actual.weight, _tensor(torch_candidate, weight))
    expected_input = torch_reference.tensor([1, 0, 1, 3, 0], dtype=torch_reference.int64)
    actual_input = torch_candidate.tensor([1, 0, 1, 3, 0], dtype=torch_candidate.int64)

    expected(expected_input).sum().backward()
    actual(actual_input).sum().backward()

    assert_values_compatible(
        torch_reference,
        expected.weight.grad,
        actual.weight.grad,
        path="nn.Embedding.padding_idx.backward",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_embedding_max_norm_module_matches_reference(torch_reference, torch_candidate) -> None:
    weight = [[3.0, 4.0], [6.0, 8.0], [1.0, 2.0], [-8.0, 6.0]]
    expected = torch_reference.nn.Embedding(4, 2, max_norm=5.0, norm_type=2.0)
    actual = torch_candidate.nn.Embedding(4, 2, max_norm=5.0, norm_type=2.0)
    _copy_parameter(torch_reference, expected.weight, _tensor(torch_reference, weight))
    _copy_parameter(torch_candidate, actual.weight, _tensor(torch_candidate, weight))
    expected_input = torch_reference.tensor([0, 1, 1, 3], dtype=torch_reference.int64)
    actual_input = torch_candidate.tensor([0, 1, 1, 3], dtype=torch_candidate.int64)

    assert_values_compatible(
        torch_reference,
        expected(expected_input),
        actual(actual_input),
        path="nn.Embedding.max_norm.forward",
        rtol=1e-6,
        atol=1e-6,
    )
    assert_values_compatible(
        torch_reference,
        expected.weight,
        actual.weight,
        path="nn.Embedding.max_norm.weight",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_embedding_scale_grad_by_freq_backward_matches_reference(torch_reference, torch_candidate) -> None:
    weight = [[0.0, 0.25], [1.0, -1.5], [0.5, 0.75]]
    expected = torch_reference.nn.Embedding(3, 2, scale_grad_by_freq=True)
    actual = torch_candidate.nn.Embedding(3, 2, scale_grad_by_freq=True)
    _copy_parameter(torch_reference, expected.weight, _tensor(torch_reference, weight))
    _copy_parameter(torch_candidate, actual.weight, _tensor(torch_candidate, weight))
    expected_input = torch_reference.tensor([1, 1, 2], dtype=torch_reference.int64)
    actual_input = torch_candidate.tensor([1, 1, 2], dtype=torch_candidate.int64)
    expected_upstream = _tensor(torch_reference, [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    actual_upstream = _tensor(torch_candidate, [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])

    (expected(expected_input) * expected_upstream).sum().backward()
    (actual(actual_input) * actual_upstream).sum().backward()

    assert_values_compatible(
        torch_reference,
        expected.weight.grad,
        actual.weight.grad,
        path="nn.Embedding.scale_grad_by_freq.backward",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_linear_module_parameters_match_reference(torch_reference, torch_candidate) -> None:
    expected_parameters = tuple(_linear(torch_reference).parameters())
    actual_parameters = tuple(_linear(torch_candidate).parameters())

    assert len(expected_parameters) == len(actual_parameters)
    for index, (expected, actual) in enumerate(zip(expected_parameters, actual_parameters, strict=True)):
        assert_values_compatible(
            torch_reference,
            expected,
            actual,
            path=f"nn.Linear.parameters[{index}]",
            rtol=1e-6,
            atol=1e-6,
        )


def test_identity_module_matches_reference(torch_reference, torch_candidate) -> None:
    expected_input = _tensor(torch_reference, [[1.0, -2.0], [3.0, 4.0]])
    actual_input = _tensor(torch_candidate, [[1.0, -2.0], [3.0, 4.0]])

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.Identity()(expected_input),
        torch_candidate.nn.Identity()(actual_input),
        path="nn.Identity.forward",
    )


def test_leaky_relu_module_matches_reference(torch_reference, torch_candidate) -> None:
    expected_input = _tensor(torch_reference, [[1.0, -2.0], [3.0, -4.0]])
    actual_input = _tensor(torch_candidate, [[1.0, -2.0], [3.0, -4.0]])

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.LeakyReLU(negative_slope=0.2)(expected_input),
        torch_candidate.nn.LeakyReLU(negative_slope=0.2)(actual_input),
        path="nn.LeakyReLU.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_silu_module_matches_reference(torch_reference, torch_candidate) -> None:
    expected_input = _tensor(torch_reference, [[1.0, -2.0], [3.0, -4.0]])
    actual_input = _tensor(torch_candidate, [[1.0, -2.0], [3.0, -4.0]])

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.SiLU()(expected_input),
        torch_candidate.nn.SiLU()(actual_input),
        path="nn.SiLU.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_gelu_module_matches_reference(torch_reference, torch_candidate) -> None:
    expected_input = _tensor(torch_reference, [[1.0, -2.0], [3.0, -4.0]])
    actual_input = _tensor(torch_candidate, [[1.0, -2.0], [3.0, -4.0]])

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.GELU(approximate="tanh")(expected_input),
        torch_candidate.nn.GELU(approximate="tanh")(actual_input),
        path="nn.GELU.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_piecewise_activation_modules_match_reference(torch_reference, torch_candidate) -> None:
    expected_input = _tensor(torch_reference, [[-4.0, -2.0, 1.0], [2.0, 4.0, 7.0]])
    actual_input = _tensor(torch_candidate, [[-4.0, -2.0, 1.0], [2.0, 4.0, 7.0]])

    for name in ("ReLU6", "Hardsigmoid", "Hardswish"):
        assert_values_compatible(
            torch_reference,
            getattr(torch_reference.nn, name)()(expected_input),
            getattr(torch_candidate.nn, name)()(actual_input),
            path=f"nn.{name}.forward",
            rtol=1e-6,
            atol=1e-6,
        )
    assert_values_compatible(
        torch_reference,
        torch_reference.nn.Hardtanh(min_val=-0.5, max_val=2.0)(expected_input),
        torch_candidate.nn.Hardtanh(min_val=-0.5, max_val=2.0)(actual_input),
        path="nn.Hardtanh.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_smooth_activation_modules_match_reference(torch_reference, torch_candidate) -> None:
    expected_input = _tensor(torch_reference, [[-4.0, -2.0, 1.0], [2.0, 4.0, 7.0]])
    actual_input = _tensor(torch_candidate, [[-4.0, -2.0, 1.0], [2.0, 4.0, 7.0]])
    cases = (
        (torch_reference.nn.ELU(alpha=0.25), torch_candidate.nn.ELU(alpha=0.25), "ELU"),
        (torch_reference.nn.SELU(), torch_candidate.nn.SELU(), "SELU"),
        (torch_reference.nn.Softplus(beta=0.5, threshold=10.0), torch_candidate.nn.Softplus(beta=0.5, threshold=10.0), "Softplus"),
        (torch_reference.nn.Softsign(), torch_candidate.nn.Softsign(), "Softsign"),
        (torch_reference.nn.Mish(), torch_candidate.nn.Mish(), "Mish"),
    )

    for expected, actual, name in cases:
        assert_values_compatible(
            torch_reference,
            expected(expected_input),
            actual(actual_input),
            path=f"nn.{name}.forward",
            rtol=1e-6,
            atol=1e-6,
        )


def test_dropout_module_eval_matches_reference(torch_reference, torch_candidate) -> None:
    expected_input = _tensor(torch_reference, [[1.0, -2.0], [3.0, -4.0]])
    actual_input = _tensor(torch_candidate, [[1.0, -2.0], [3.0, -4.0]])
    expected = torch_reference.nn.Dropout(p=0.4).eval()
    actual = torch_candidate.nn.Dropout(p=0.4).eval()

    assert_values_compatible(
        torch_reference,
        expected(expected_input),
        actual(actual_input),
        path="nn.Dropout.eval",
        rtol=1e-6,
        atol=1e-6,
    )


def test_dropout_zero_probability_training_matches_reference(torch_reference, torch_candidate) -> None:
    expected_input = _tensor(torch_reference, [[1.0, -2.0], [3.0, -4.0]])
    actual_input = _tensor(torch_candidate, [[1.0, -2.0], [3.0, -4.0]])

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.Dropout(p=0.0)(expected_input),
        torch_candidate.nn.Dropout(p=0.0)(actual_input),
        path="nn.Dropout.p0.training",
        rtol=1e-6,
        atol=1e-6,
    )


def test_dropout_training_reproducibility_and_backward(torch_candidate) -> None:
    first_input = torch_candidate.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch_candidate.float32, requires_grad=True)
    second_input = torch_candidate.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch_candidate.float32, requires_grad=True)

    torch_candidate.manual_seed(123)
    first = torch_candidate.nn.functional.dropout(first_input, p=0.5, training=True)
    torch_candidate.manual_seed(123)
    second = torch_candidate.nn.functional.dropout(second_input, p=0.5, training=True)

    assert first.tolist() == second.tolist()
    for source_row, output_row in zip(first_input.tolist(), first.tolist(), strict=True):
        for source, output in zip(source_row, output_row, strict=True):
            assert output in {0.0, source * 2.0}

    first.sum().backward()
    for grad_row in first_input.grad.tolist():
        for value in grad_row:
            assert value in {0.0, 2.0}


def test_layer_norm_module_matches_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.LayerNorm(3)
    actual = torch_candidate.nn.LayerNorm(3)
    _copy_parameter(torch_reference, expected.weight, _tensor(torch_reference, [0.5, 1.5, -1.0]))
    _copy_parameter(torch_candidate, actual.weight, _tensor(torch_candidate, [0.5, 1.5, -1.0]))
    _copy_parameter(torch_reference, expected.bias, _tensor(torch_reference, [0.25, -0.5, 0.75]))
    _copy_parameter(torch_candidate, actual.bias, _tensor(torch_candidate, [0.25, -0.5, 0.75]))
    expected_input = _tensor(torch_reference, [[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]])
    actual_input = _tensor(torch_candidate, [[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]])

    assert_values_compatible(
        torch_reference,
        expected(expected_input),
        actual(actual_input),
        path="nn.LayerNorm.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_rms_norm_module_matches_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.RMSNorm(3, eps=1e-5)
    actual = torch_candidate.nn.RMSNorm(3, eps=1e-5)
    _copy_parameter(torch_reference, expected.weight, _tensor(torch_reference, [0.5, 1.5, -1.0]))
    _copy_parameter(torch_candidate, actual.weight, _tensor(torch_candidate, [0.5, 1.5, -1.0]))
    expected_input = _tensor(torch_reference, [[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]])
    actual_input = _tensor(torch_candidate, [[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]])

    assert_values_compatible(
        torch_reference,
        expected(expected_input),
        actual(actual_input),
        path="nn.RMSNorm.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_batch_norm1d_eval_module_matches_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.BatchNorm1d(3)
    actual = torch_candidate.nn.BatchNorm1d(3)
    _copy_parameter(torch_reference, expected.weight, _tensor(torch_reference, [0.5, 1.5, -1.0]))
    _copy_parameter(torch_candidate, actual.weight, _tensor(torch_candidate, [0.5, 1.5, -1.0]))
    _copy_parameter(torch_reference, expected.bias, _tensor(torch_reference, [0.25, -0.5, 0.75]))
    _copy_parameter(torch_candidate, actual.bias, _tensor(torch_candidate, [0.25, -0.5, 0.75]))
    _copy_parameter(torch_reference, expected.running_mean, _tensor(torch_reference, [0.25, -0.5, 1.0]))
    _copy_parameter(torch_candidate, actual.running_mean, _tensor(torch_candidate, [0.25, -0.5, 1.0]))
    _copy_parameter(torch_reference, expected.running_var, _tensor(torch_reference, [1.0, 4.0, 0.25]))
    _copy_parameter(torch_candidate, actual.running_var, _tensor(torch_candidate, [1.0, 4.0, 0.25]))
    expected.eval()
    actual.eval()
    expected_input = _tensor(torch_reference, [[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]])
    actual_input = _tensor(torch_candidate, [[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]])

    assert_values_compatible(
        torch_reference,
        expected(expected_input),
        actual(actual_input),
        path="nn.BatchNorm1d.eval.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_batch_norm1d_training_updates_running_stats(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.BatchNorm1d(3, momentum=0.2)
    actual = torch_candidate.nn.BatchNorm1d(3, momentum=0.2)
    _copy_parameter(torch_reference, expected.weight, _tensor(torch_reference, [0.5, 1.5, -1.0]))
    _copy_parameter(torch_candidate, actual.weight, _tensor(torch_candidate, [0.5, 1.5, -1.0]))
    _copy_parameter(torch_reference, expected.bias, _tensor(torch_reference, [0.25, -0.5, 0.75]))
    _copy_parameter(torch_candidate, actual.bias, _tensor(torch_candidate, [0.25, -0.5, 0.75]))
    expected_input = _tensor(torch_reference, [[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]])
    actual_input = _tensor(torch_candidate, [[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]])

    assert_values_compatible(
        torch_reference,
        expected(expected_input),
        actual(actual_input),
        path="nn.BatchNorm1d.training.forward",
        rtol=1e-5,
        atol=1e-5,
    )
    assert_values_compatible(
        torch_reference,
        expected.running_mean,
        actual.running_mean,
        path="nn.BatchNorm1d.running_mean",
        rtol=1e-6,
        atol=1e-6,
    )
    assert_values_compatible(
        torch_reference,
        expected.running_var,
        actual.running_var,
        path="nn.BatchNorm1d.running_var",
        rtol=1e-6,
        atol=1e-6,
    )
    assert_values_compatible(
        torch_reference,
        expected.num_batches_tracked,
        actual.num_batches_tracked,
        path="nn.BatchNorm1d.num_batches_tracked",
        rtol=1e-6,
        atol=1e-6,
    )


def test_batch_norm2d_eval_module_matches_reference(torch_reference, torch_candidate) -> None:
    data = [
        [
            [[1.0, -2.0], [3.0, 4.0]],
            [[0.5, -1.5], [2.0, 1.5]],
            [[-3.0, 0.25], [4.5, -6.0]],
        ],
        [
            [[2.0, 1.0], [-1.0, 0.5]],
            [[-2.0, 3.0], [1.0, -0.5]],
            [[0.75, -1.25], [2.5, 3.5]],
        ],
    ]
    expected = torch_reference.nn.BatchNorm2d(3)
    actual = torch_candidate.nn.BatchNorm2d(3)
    _copy_parameter(torch_reference, expected.weight, _tensor(torch_reference, [0.5, 1.5, -1.0]))
    _copy_parameter(torch_candidate, actual.weight, _tensor(torch_candidate, [0.5, 1.5, -1.0]))
    _copy_parameter(torch_reference, expected.bias, _tensor(torch_reference, [0.25, -0.5, 0.75]))
    _copy_parameter(torch_candidate, actual.bias, _tensor(torch_candidate, [0.25, -0.5, 0.75]))
    _copy_parameter(torch_reference, expected.running_mean, _tensor(torch_reference, [0.25, -0.5, 1.0]))
    _copy_parameter(torch_candidate, actual.running_mean, _tensor(torch_candidate, [0.25, -0.5, 1.0]))
    _copy_parameter(torch_reference, expected.running_var, _tensor(torch_reference, [1.0, 4.0, 0.25]))
    _copy_parameter(torch_candidate, actual.running_var, _tensor(torch_candidate, [1.0, 4.0, 0.25]))
    expected.eval()
    actual.eval()

    assert_values_compatible(
        torch_reference,
        expected(_tensor(torch_reference, data)),
        actual(_tensor(torch_candidate, data)),
        path="nn.BatchNorm2d.eval.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_group_norm_module_matches_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.GroupNorm(2, 4)
    actual = torch_candidate.nn.GroupNorm(2, 4)
    _copy_parameter(torch_reference, expected.weight, _tensor(torch_reference, [0.5, 1.5, -1.0, 0.25]))
    _copy_parameter(torch_candidate, actual.weight, _tensor(torch_candidate, [0.5, 1.5, -1.0, 0.25]))
    _copy_parameter(torch_reference, expected.bias, _tensor(torch_reference, [0.25, -0.5, 0.75, -1.0]))
    _copy_parameter(torch_candidate, actual.bias, _tensor(torch_candidate, [0.25, -0.5, 0.75, -1.0]))
    expected_input = _tensor(torch_reference, [[1.0, -2.0, 3.0, 0.5], [4.0, 0.5, -6.0, 2.0]])
    actual_input = _tensor(torch_candidate, [[1.0, -2.0, 3.0, 0.5], [4.0, 0.5, -6.0, 2.0]])

    assert_values_compatible(
        torch_reference,
        expected(expected_input),
        actual(actual_input),
        path="nn.GroupNorm.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_flatten_module_matches_reference(torch_reference, torch_candidate) -> None:
    data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    expected_input = _tensor(torch_reference, data)
    actual_input = _tensor(torch_candidate, data)

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.Flatten(start_dim=1)(expected_input),
        torch_candidate.nn.Flatten(start_dim=1)(actual_input),
        path="nn.Flatten.forward",
    )


def test_unflatten_module_matches_reference(torch_reference, torch_candidate) -> None:
    data = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    expected_input = _tensor(torch_reference, data)
    actual_input = _tensor(torch_candidate, data)

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.Unflatten(1, (2, 2))(expected_input),
        torch_candidate.nn.Unflatten(1, (2, 2))(actual_input),
        path="nn.Unflatten.forward",
    )


def test_spatial_helper_modules_match_reference(torch_reference, torch_candidate) -> None:
    data = [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]]
    expected_input = _tensor(torch_reference, data)
    actual_input = _tensor(torch_candidate, data)

    cases = (
        (torch_reference.nn.PixelShuffle(2), torch_candidate.nn.PixelShuffle(2), "PixelShuffle"),
        (torch_reference.nn.PixelUnshuffle(2), torch_candidate.nn.PixelUnshuffle(2), "PixelUnshuffle"),
        (torch_reference.nn.ChannelShuffle(2), torch_candidate.nn.ChannelShuffle(2), "ChannelShuffle"),
        (torch_reference.nn.AdaptiveAvgPool2d((1, 1)), torch_candidate.nn.AdaptiveAvgPool2d((1, 1)), "AdaptiveAvgPool2d"),
        (torch_reference.nn.ZeroPad2d((1, 0, 1, 0)), torch_candidate.nn.ZeroPad2d((1, 0, 1, 0)), "ZeroPad2d"),
        (torch_reference.nn.ConstantPad2d((1, 0, 1, 0), -0.5), torch_candidate.nn.ConstantPad2d((1, 0, 1, 0), -0.5), "ConstantPad2d"),
        (torch_reference.nn.ConstantPad2d(1, -0.5), torch_candidate.nn.ConstantPad2d(1, -0.5), "ConstantPad2d.int"),
        (torch_reference.nn.ReflectionPad2d(1), torch_candidate.nn.ReflectionPad2d(1), "ReflectionPad2d"),
        (torch_reference.nn.ReplicationPad2d(1), torch_candidate.nn.ReplicationPad2d(1), "ReplicationPad2d"),
        (torch_reference.nn.CircularPad2d(1), torch_candidate.nn.CircularPad2d(1), "CircularPad2d"),
        (torch_reference.nn.UpsamplingNearest2d(scale_factor=2), torch_candidate.nn.UpsamplingNearest2d(scale_factor=2), "UpsamplingNearest2d"),
    )

    for expected, actual, name in cases:
        assert_values_compatible(
            torch_reference,
            expected(expected_input),
            actual(actual_input),
            path=f"nn.{name}.forward",
            rtol=1e-6,
            atol=1e-6,
            check_stride=False,
        )


def test_adaptive_avg_pool2d_non_global_module_matches_reference(torch_reference, torch_candidate) -> None:
    data = [[[[1.0, -2.0, 3.0, 0.5], [-1.0, 2.5, 0.0, -0.5], [4.0, -3.0, 1.5, 2.0]]]]
    expected_input = _tensor(torch_reference, data)
    actual_input = _tensor(torch_candidate, data)

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.AdaptiveAvgPool2d((2, 3))(expected_input),
        torch_candidate.nn.AdaptiveAvgPool2d((2, 3))(actual_input),
        path="nn.AdaptiveAvgPool2d.non_global.forward",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_adaptive_avg_pool1d_module_matches_reference(torch_reference, torch_candidate) -> None:
    data = [[[1.0, -2.0, 3.0, 0.5], [-1.0, 2.5, 0.0, -0.5]]]
    expected_input = _tensor(torch_reference, data)
    actual_input = _tensor(torch_candidate, data)

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.AdaptiveAvgPool1d(3)(expected_input),
        torch_candidate.nn.AdaptiveAvgPool1d(3)(actual_input),
        path="nn.AdaptiveAvgPool1d.forward",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_fold_unfold_modules_match_reference(torch_reference, torch_candidate) -> None:
    image = [[[[1.0, -2.0, 3.0, 0.5], [-1.0, 2.5, 0.0, -0.5], [4.0, -3.0, 1.5, 2.0]]]]
    expected_image = _tensor(torch_reference, image)
    actual_image = _tensor(torch_candidate, image)

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.Unfold((2, 2), padding=1, stride=2)(expected_image),
        torch_candidate.nn.Unfold((2, 2), padding=1, stride=2)(actual_image),
        path="nn.Unfold.forward",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )

    columns = [[[1.0, -2.0, 3.0, 0.5], [-1.0, 2.5, 0.0, -0.5], [4.0, -3.0, 1.5, 2.0], [0.25, -1.5, 2.5, 3.0]]]
    expected_columns = _tensor(torch_reference, columns)
    actual_columns = _tensor(torch_candidate, columns)
    assert_values_compatible(
        torch_reference,
        torch_reference.nn.Fold((3, 3), (2, 2))(expected_columns),
        torch_candidate.nn.Fold((3, 3), (2, 2))(actual_columns),
        path="nn.Fold.forward",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_cosine_similarity_module_matches_reference(torch_reference, torch_candidate) -> None:
    expected_left = _tensor(torch_reference, [[1.0, -2.0, 3.5], [4.0, 0.5, -6.0]])
    expected_right = _tensor(torch_reference, [[0.5, 2.0, -1.5], [3.0, -4.0, 2.0]])
    actual_left = _tensor(torch_candidate, [[1.0, -2.0, 3.5], [4.0, 0.5, -6.0]])
    actual_right = _tensor(torch_candidate, [[0.5, 2.0, -1.5], [3.0, -4.0, 2.0]])

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.CosineSimilarity(dim=1)(expected_left, expected_right),
        torch_candidate.nn.CosineSimilarity(dim=1)(actual_left, actual_right),
        path="nn.CosineSimilarity.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_parameter_identity_and_module_registration(torch_candidate) -> None:
    parameter = torch_candidate.nn.Parameter([1.0, 2.0])
    assert isinstance(parameter, torch_candidate.Tensor)
    assert isinstance(parameter, torch_candidate.nn.Parameter)
    assert isinstance(parameter, torch_candidate.nn.parameter.Parameter)

    module = torch_candidate.nn.Module()
    module.register_parameter("weight", parameter)
    module.add_module("child", torch_candidate.nn.Linear(2, 1))
    assert module.get_parameter("weight") is parameter
    assert module.get_submodule("child") is module.child
    assert [name for name, _ in module.named_children()] == ["child"]
    assert "child.weight" in dict(module.named_parameters())


def test_module_setattr_registration_matches_reference(torch_reference, torch_candidate) -> None:
    class ExpectedModel(torch_reference.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch_reference.nn.Parameter(_tensor(torch_reference, [1.0, 2.0]))
            self.tensor = _tensor(torch_reference, [3.0, 4.0])
            self.child = _linear(torch_reference)
            self.register_buffer("running", _tensor(torch_reference, [5.0, 6.0]))

    class ActualModel(torch_candidate.nn.Module):
        def __init__(self):
            self.param = torch_candidate.nn.Parameter(_tensor(torch_candidate, [1.0, 2.0]))
            self.tensor = _tensor(torch_candidate, [3.0, 4.0])
            self.child = _linear(torch_candidate)
            self.register_buffer("running", _tensor(torch_candidate, [5.0, 6.0]))

    expected = ExpectedModel()
    actual = ActualModel()

    assert [name for name, _ in expected.named_children()] == [name for name, _ in actual.named_children()]
    assert [name for name, _ in expected.named_parameters()] == [name for name, _ in actual.named_parameters()]
    assert "tensor" not in dict(actual.named_parameters())
    assert_values_compatible(
        torch_reference,
        expected.state_dict(),
        actual.state_dict(),
        path="nn.Module.__setattr__.state_dict",
        rtol=1e-6,
        atol=1e-6,
    )


def test_module_reassignment_and_deletion_match_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.Module()
    actual = torch_candidate.nn.Module()
    expected.child = _linear(torch_reference)
    actual.child = _linear(torch_candidate)
    expected.child = None
    actual.child = None
    expected.param = torch_reference.nn.Parameter(_tensor(torch_reference, [1.0]))
    actual.param = torch_candidate.nn.Parameter(_tensor(torch_candidate, [1.0]))
    expected.param = None
    actual.param = None
    expected.register_buffer("running", _tensor(torch_reference, [2.0]))
    actual.register_buffer("running", _tensor(torch_candidate, [2.0]))
    expected.running = None
    actual.running = None

    assert [name for name, _ in expected.named_children()] == [name for name, _ in actual.named_children()]
    assert [name for name, _ in expected.named_parameters()] == [name for name, _ in actual.named_parameters()]
    assert [name for name, _ in expected.named_buffers()] == [name for name, _ in actual.named_buffers()]

    expected.child = _linear(torch_reference)
    actual.child = _linear(torch_candidate)
    del expected.child
    del actual.child
    assert [name for name, _ in expected.named_children()] == [name for name, _ in actual.named_children()]


def test_module_container_path_lookup_matches_reference(torch_reference, torch_candidate) -> None:
    class ExpectedModel(torch_reference.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch_reference.nn.ModuleList([_linear(torch_reference), torch_reference.nn.Sequential(_linear(torch_reference))])
            self.blocks = torch_reference.nn.ModuleDict({"down": _linear(torch_reference)})
            self.layers[1].register_buffer("running", _tensor(torch_reference, [1.0, 2.0]))

    class ActualModel(torch_candidate.nn.Module):
        def __init__(self):
            self.layers = torch_candidate.nn.ModuleList([_linear(torch_candidate), torch_candidate.nn.Sequential(_linear(torch_candidate))])
            self.blocks = torch_candidate.nn.ModuleDict({"down": _linear(torch_candidate)})
            self.layers[1].register_buffer("running", _tensor(torch_candidate, [1.0, 2.0]))

    expected = ExpectedModel()
    actual = ActualModel()

    assert type(expected.get_submodule("layers.0")).__name__ == type(actual.get_submodule("layers.0")).__name__
    assert type(expected.get_submodule("layers.1.0")).__name__ == type(actual.get_submodule("layers.1.0")).__name__
    assert type(expected.get_submodule("blocks.down")).__name__ == type(actual.get_submodule("blocks.down")).__name__
    assert_values_compatible(
        torch_reference,
        expected.get_parameter("layers.0.weight"),
        actual.get_parameter("layers.0.weight"),
        path="nn.Module.get_parameter.ModuleList",
        rtol=1e-6,
        atol=1e-6,
    )
    assert_values_compatible(
        torch_reference,
        expected.get_buffer("layers.1.running"),
        actual.get_buffer("layers.1.running"),
        path="nn.Module.get_buffer.Sequential",
        rtol=1e-6,
        atol=1e-6,
    )


def test_module_named_parameters_remove_duplicate_matches_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.Module()
    actual = torch_candidate.nn.Module()
    expected.shared = _linear(torch_reference)
    actual.shared = _linear(torch_candidate)
    expected.alias = expected.shared
    actual.alias = actual.shared

    assert [name for name, _ in expected.named_parameters()] == [name for name, _ in actual.named_parameters()]
    assert [name for name, _ in expected.named_parameters(remove_duplicate=False)] == [
        name for name, _ in actual.named_parameters(remove_duplicate=False)
    ]


def test_module_forward_hooks_match_reference(torch_reference, torch_candidate) -> None:
    class ExpectedModel(torch_reference.nn.Module):
        def forward(self, input, scale=1.0):
            return input * scale

    class ActualModel(torch_candidate.nn.Module):
        def forward(self, input, scale=1.0):
            return input * scale

    expected = ExpectedModel()
    actual = ActualModel()

    def expected_pre_hook(module, args, kwargs):
        return args, {"scale": kwargs["scale"] + 1.0}

    def actual_pre_hook(module, args, kwargs):
        return args, {"scale": kwargs["scale"] + 1.0}

    def expected_hook(module, args, kwargs, output):
        return output + 0.5

    def actual_hook(module, args, kwargs, output):
        return output + 0.5

    expected_pre_handle = expected.register_forward_pre_hook(expected_pre_hook, with_kwargs=True)
    actual_pre_handle = actual.register_forward_pre_hook(actual_pre_hook, with_kwargs=True)
    expected_handle = expected.register_forward_hook(expected_hook, with_kwargs=True)
    actual_handle = actual.register_forward_hook(actual_hook, with_kwargs=True)

    expected_input = _tensor(torch_reference, [1.0, 2.0])
    actual_input = _tensor(torch_candidate, [1.0, 2.0])
    assert_values_compatible(
        torch_reference,
        expected(expected_input, scale=2.0),
        actual(actual_input, scale=2.0),
        path="nn.Module.forward_hooks",
        rtol=1e-6,
        atol=1e-6,
    )

    expected_pre_handle.remove()
    actual_pre_handle.remove()
    expected_handle.remove()
    actual_handle.remove()
    assert_values_compatible(
        torch_reference,
        expected(expected_input, scale=2.0),
        actual(actual_input, scale=2.0),
        path="nn.Module.forward_hooks.removed",
        rtol=1e-6,
        atol=1e-6,
    )


def test_module_forward_hook_always_call_matches_reference(torch_reference, torch_candidate) -> None:
    class ExpectedModel(torch_reference.nn.Module):
        def forward(self, input):
            raise RuntimeError("boom")

    class ActualModel(torch_candidate.nn.Module):
        def forward(self, input):
            raise RuntimeError("boom")

    expected = ExpectedModel()
    actual = ActualModel()
    expected_seen: list[str] = []
    actual_seen: list[str] = []

    expected.register_forward_hook(lambda module, args, output: expected_seen.append(str(output)), always_call=True)
    actual.register_forward_hook(lambda module, args, output: actual_seen.append(str(output)), always_call=True)

    for module, input_value in (
        (expected, _tensor(torch_reference, [1.0])),
        (actual, _tensor(torch_candidate, [1.0])),
    ):
        try:
            module(input_value)
        except RuntimeError:
            pass

    assert expected_seen == actual_seen


def test_builtin_module_weights_are_parameters(torch_candidate) -> None:
    modules = (
        torch_candidate.nn.Linear(3, 2),
        torch_candidate.nn.Conv1d(2, 3, 3),
        torch_candidate.nn.Conv2d(2, 3, 3),
        torch_candidate.nn.Conv3d(2, 3, 3),
        torch_candidate.nn.ConvTranspose1d(2, 3, 3),
        torch_candidate.nn.ConvTranspose2d(2, 3, 3),
        torch_candidate.nn.ConvTranspose3d(2, 3, 3),
        torch_candidate.nn.Embedding(4, 3),
        torch_candidate.nn.LayerNorm(3),
        torch_candidate.nn.RMSNorm(3),
        torch_candidate.nn.BatchNorm1d(3),
        torch_candidate.nn.GroupNorm(1, 3),
        torch_candidate.nn.MultiheadAttention(4, 2),
    )
    for module in modules:
        for name, value in module.named_parameters(recurse=False):
            assert isinstance(value, torch_candidate.nn.Parameter), f"{type(module).__name__}.{name}"


def test_sequential_module_matches_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.Sequential(_linear(torch_reference), torch_reference.nn.ReLU())
    actual = torch_candidate.nn.Sequential(_linear(torch_candidate), torch_candidate.nn.ReLU())
    expected_input = _tensor(torch_reference, [[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]])
    actual_input = _tensor(torch_candidate, [[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]])

    assert_values_compatible(
        torch_reference,
        expected(expected_input),
        actual(actual_input),
        path="nn.Sequential.forward",
        rtol=1e-6,
        atol=1e-6,
    )

    expected_named = tuple(expected.named_parameters())
    actual_named = tuple(actual.named_parameters())
    assert [name for name, _ in expected_named] == [name for name, _ in actual_named]
    for index, ((_, expected_parameter), (_, actual_parameter)) in enumerate(
        zip(expected_named, actual_named, strict=True)
    ):
        assert_values_compatible(
            torch_reference,
            expected_parameter,
            actual_parameter,
            path=f"nn.Sequential.named_parameters[{index}]",
            rtol=1e-6,
            atol=1e-6,
        )


def test_sequential_ordered_dict_state_dict_matches_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.Sequential(OrderedDict([("first", _linear(torch_reference)), ("act", torch_reference.nn.ReLU())]))
    actual = torch_candidate.nn.Sequential(OrderedDict([("first", _linear(torch_candidate)), ("act", torch_candidate.nn.ReLU())]))

    assert [name for name, _ in expected.named_modules()] == [name for name, _ in actual.named_modules()]
    assert_values_compatible(
        torch_reference,
        expected.state_dict(),
        actual.state_dict(),
        path="nn.Sequential.OrderedDict.state_dict",
        rtol=1e-6,
        atol=1e-6,
    )


def test_module_state_dict_matches_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.Sequential(_linear(torch_reference), torch_reference.nn.ReLU())
    actual = torch_candidate.nn.Sequential(_linear(torch_candidate), torch_candidate.nn.ReLU())

    assert_values_compatible(
        torch_reference,
        expected.state_dict(),
        actual.state_dict(),
        path="nn.Module.state_dict",
        rtol=1e-6,
        atol=1e-6,
    )


def test_module_load_state_dict_matches_reference(torch_reference, torch_candidate) -> None:
    expected_source = _linear(torch_reference)
    actual_source = _linear(torch_candidate)
    expected_target = torch_reference.nn.Linear(3, 2)
    actual_target = torch_candidate.nn.Linear(3, 2)

    expected_result = expected_target.load_state_dict(expected_source.state_dict())
    actual_result = actual_target.load_state_dict(actual_source.state_dict())
    assert actual_result.missing_keys == expected_result.missing_keys
    assert actual_result.unexpected_keys == expected_result.unexpected_keys
    assert tuple(actual_result) == tuple(expected_result)

    assert_values_compatible(
        torch_reference,
        expected_target.state_dict(),
        actual_target.state_dict(),
        path="nn.Module.load_state_dict",
        rtol=1e-6,
        atol=1e-6,
    )


def test_module_load_state_dict_incompatible_keys_match_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.Linear(3, 2)
    actual = torch_candidate.nn.Linear(3, 2)

    expected_state = expected.state_dict()
    actual_state = actual.state_dict()
    del expected_state["bias"]
    del actual_state["bias"]
    expected_state["extra"] = torch_reference.ones((1,), dtype=torch_reference.float32)
    actual_state["extra"] = torch_candidate.ones((1,), dtype=torch_candidate.float32)

    expected_result = expected.load_state_dict(expected_state, strict=False)
    actual_result = actual.load_state_dict(actual_state, strict=False)

    assert actual_result.missing_keys == expected_result.missing_keys
    assert actual_result.unexpected_keys == expected_result.unexpected_keys
    assert tuple(actual_result) == tuple(expected_result)
    assert repr(actual.load_state_dict(actual.state_dict())) == repr(expected.load_state_dict(expected.state_dict()))


def test_module_state_dict_hooks_match_reference(torch_reference, torch_candidate) -> None:
    expected = _linear(torch_reference)
    actual = _linear(torch_candidate)
    expected_events = []
    actual_events = []

    def expected_pre(module, prefix, keep_vars):
        expected_events.append(("pre", prefix, keep_vars, module.training))

    def actual_pre(module, prefix, keep_vars):
        actual_events.append(("pre", prefix, keep_vars, module.training))

    def expected_post(module, state_dict, prefix, local_metadata):
        expected_events.append(("post", prefix, "version" in local_metadata))
        state_dict[prefix + "extra"] = torch_reference.ones((1,), dtype=torch_reference.float32)

    def actual_post(module, state_dict, prefix, local_metadata):
        actual_events.append(("post", prefix, "version" in local_metadata))
        state_dict[prefix + "extra"] = torch_candidate.ones((1,), dtype=torch_candidate.float32)

    expected_handle = expected.register_state_dict_pre_hook(expected_pre)
    actual_handle = actual.register_state_dict_pre_hook(actual_pre)
    expected.register_state_dict_post_hook(expected_post)
    actual.register_state_dict_post_hook(actual_post)

    expected_state = expected.state_dict(prefix="model.")
    actual_state = actual.state_dict(prefix="model.")

    assert expected_events == actual_events
    assert list(expected_state.keys()) == list(actual_state.keys())
    assert "model.extra" in actual_state

    expected_handle.remove()
    actual_handle.remove()
    expected_events.clear()
    actual_events.clear()
    expected.state_dict()
    actual.state_dict()
    assert [event[0] for event in expected_events] == [event[0] for event in actual_events] == ["post"]


def test_module_load_state_dict_hooks_match_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.Linear(3, 2)
    actual = torch_candidate.nn.Linear(3, 2)
    expected_source = _linear(torch_reference).state_dict()
    actual_source = _linear(torch_candidate).state_dict()
    expected_source["renamed_weight"] = expected_source.pop("weight")
    actual_source["renamed_weight"] = actual_source.pop("weight")
    expected_events = []
    actual_events = []

    def expected_pre(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        expected_events.append(("pre", prefix, strict, "version" in local_metadata))
        state_dict[prefix + "weight"] = state_dict.pop(prefix + "renamed_weight")

    def actual_pre(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        actual_events.append(("pre", prefix, strict, "version" in local_metadata))
        state_dict[prefix + "weight"] = state_dict.pop(prefix + "renamed_weight")

    def expected_post(module, incompatible_keys):
        expected_events.append(("post", tuple(incompatible_keys.missing_keys), tuple(incompatible_keys.unexpected_keys)))

    def actual_post(module, incompatible_keys):
        actual_events.append(("post", tuple(incompatible_keys.missing_keys), tuple(incompatible_keys.unexpected_keys)))

    expected.register_load_state_dict_pre_hook(expected_pre)
    actual.register_load_state_dict_pre_hook(actual_pre)
    expected.register_load_state_dict_post_hook(expected_post)
    actual.register_load_state_dict_post_hook(actual_post)

    expected_result = expected.load_state_dict(expected_source)
    actual_result = actual.load_state_dict(actual_source)

    assert expected_events == actual_events
    assert actual_result.missing_keys == expected_result.missing_keys
    assert actual_result.unexpected_keys == expected_result.unexpected_keys
    assert tuple(actual_result) == tuple(expected_result)

    assert_values_compatible(
        torch_reference,
        expected.state_dict(),
        actual.state_dict(),
        path="nn.Module.load_state_dict_hooks.state_dict",
        rtol=1e-6,
        atol=1e-6,
    )


def test_module_register_buffer_state_dict_matches_reference(torch_reference, torch_candidate) -> None:
    class ExpectedModel(torch_reference.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running", _tensor(torch_reference, [1.0, 2.0]))
            self.register_buffer("scratch", _tensor(torch_reference, [3.0, 4.0]), persistent=False)
            self.layer = _linear(torch_reference)

    class ActualModel(torch_candidate.nn.Module):
        def __init__(self):
            self.register_buffer("running", _tensor(torch_candidate, [1.0, 2.0]))
            self.register_buffer("scratch", _tensor(torch_candidate, [3.0, 4.0]), persistent=False)
            self.layer = _linear(torch_candidate)

    expected = ExpectedModel()
    actual = ActualModel()

    assert [name for name, _ in expected.named_buffers()] == [name for name, _ in actual.named_buffers()]
    assert_values_compatible(
        torch_reference,
        expected.state_dict(),
        actual.state_dict(),
        path="nn.Module.register_buffer.state_dict",
        rtol=1e-6,
        atol=1e-6,
    )


def test_module_named_modules_matches_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.Sequential(_linear(torch_reference), torch_reference.nn.Sequential(torch_reference.nn.ReLU()))
    actual = torch_candidate.nn.Sequential(_linear(torch_candidate), torch_candidate.nn.Sequential(torch_candidate.nn.ReLU()))

    assert [name for name, _ in expected.named_modules()] == [name for name, _ in actual.named_modules()]


def test_module_apply_matches_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.Sequential(_linear(torch_reference), torch_reference.nn.Sequential(torch_reference.nn.ReLU()))
    actual = torch_candidate.nn.Sequential(_linear(torch_candidate), torch_candidate.nn.Sequential(torch_candidate.nn.ReLU()))
    expected_seen: list[str] = []
    actual_seen: list[str] = []

    expected_return = expected.apply(lambda module: expected_seen.append(type(module).__name__))
    actual_return = actual.apply(lambda module: actual_seen.append(type(module).__name__))

    assert expected_return is expected
    assert actual_return is actual
    assert actual_seen == expected_seen


def test_module_list_state_dict_matches_reference(torch_reference, torch_candidate) -> None:
    class ExpectedModel(torch_reference.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch_reference.nn.ModuleList([_linear(torch_reference), _linear(torch_reference)])

    class ActualModel(torch_candidate.nn.Module):
        def __init__(self):
            self.layers = torch_candidate.nn.ModuleList([_linear(torch_candidate), _linear(torch_candidate)])

    expected = ExpectedModel()
    actual = ActualModel()

    assert_values_compatible(
        torch_reference,
        expected.state_dict(),
        actual.state_dict(),
        path="nn.ModuleList.state_dict",
        rtol=1e-6,
        atol=1e-6,
    )


def test_parameter_list_state_dict_matches_reference(torch_reference, torch_candidate) -> None:
    class ExpectedModel(torch_reference.nn.Module):
        def __init__(self):
            super().__init__()
            self.params = torch_reference.nn.ParameterList(
                [
                    torch_reference.nn.Parameter(_tensor(torch_reference, [1.0, 2.0])),
                    torch_reference.nn.Parameter(_tensor(torch_reference, [3.0, 4.0])),
                ]
            )

    class ActualModel(torch_candidate.nn.Module):
        def __init__(self):
            self.params = torch_candidate.nn.ParameterList(
                [
                    torch_candidate.nn.Parameter(_tensor(torch_candidate, [1.0, 2.0])),
                    torch_candidate.nn.Parameter(_tensor(torch_candidate, [3.0, 4.0])),
                ]
            )

    expected = ExpectedModel()
    actual = ActualModel()

    assert_values_compatible(
        torch_reference,
        expected.state_dict(),
        actual.state_dict(),
        path="nn.ParameterList.state_dict",
        rtol=1e-6,
        atol=1e-6,
    )


def test_parameter_containers_convert_tensors_matches_reference(torch_reference, torch_candidate) -> None:
    expected_list = torch_reference.nn.ParameterList([_tensor(torch_reference, [1.0, 2.0])])
    actual_list = torch_candidate.nn.ParameterList([_tensor(torch_candidate, [1.0, 2.0])])
    expected_dict = torch_reference.nn.ParameterDict({"left": _tensor(torch_reference, [3.0, 4.0])})
    actual_dict = torch_candidate.nn.ParameterDict({"left": _tensor(torch_candidate, [3.0, 4.0])})

    assert [name for name, _ in expected_list.named_parameters()] == [name for name, _ in actual_list.named_parameters()]
    assert [name for name, _ in expected_dict.named_parameters()] == [name for name, _ in actual_dict.named_parameters()]
    assert isinstance(actual_list[0], torch_candidate.nn.Parameter)
    assert isinstance(actual_dict["left"], torch_candidate.nn.Parameter)
    assert_values_compatible(
        torch_reference,
        getattr(expected_list, "0"),
        getattr(actual_list, "0"),
        path="nn.ParameterList.__getattr__",
        rtol=1e-6,
        atol=1e-6,
    )


def test_module_dict_state_dict_matches_reference(torch_reference, torch_candidate) -> None:
    class ExpectedModel(torch_reference.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch_reference.nn.ModuleDict({"a": _linear(torch_reference), "b": _linear(torch_reference)})

    class ActualModel(torch_candidate.nn.Module):
        def __init__(self):
            self.layers = torch_candidate.nn.ModuleDict({"a": _linear(torch_candidate), "b": _linear(torch_candidate)})

    expected = ExpectedModel()
    actual = ActualModel()

    assert [name for name, _ in expected.named_modules()] == [name for name, _ in actual.named_modules()]
    assert_values_compatible(
        torch_reference,
        expected.state_dict(),
        actual.state_dict(),
        path="nn.ModuleDict.state_dict",
        rtol=1e-6,
        atol=1e-6,
    )


def test_parameter_dict_state_dict_matches_reference(torch_reference, torch_candidate) -> None:
    class ExpectedModel(torch_reference.nn.Module):
        def __init__(self):
            super().__init__()
            self.params = torch_reference.nn.ParameterDict(
                {"left": torch_reference.nn.Parameter(_tensor(torch_reference, [1.0, 2.0]))}
            )

    class ActualModel(torch_candidate.nn.Module):
        def __init__(self):
            self.params = torch_candidate.nn.ParameterDict(
                {"left": torch_candidate.nn.Parameter(_tensor(torch_candidate, [1.0, 2.0]))}
            )

    expected = ExpectedModel()
    actual = ActualModel()

    assert_values_compatible(
        torch_reference,
        expected.state_dict(),
        actual.state_dict(),
        path="nn.ParameterDict.state_dict",
        rtol=1e-6,
        atol=1e-6,
    )


def test_l1_loss_module_matches_reference(torch_reference, torch_candidate) -> None:
    expected_input = _tensor(torch_reference, [[1.0, -2.0], [3.0, 4.0]])
    actual_input = _tensor(torch_candidate, [[1.0, -2.0], [3.0, 4.0]])
    expected_target = _tensor(torch_reference, [[0.5, -1.0], [4.0, 1.0]])
    actual_target = _tensor(torch_candidate, [[0.5, -1.0], [4.0, 1.0]])

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.L1Loss(reduction="sum")(expected_input, expected_target),
        torch_candidate.nn.L1Loss(reduction="sum")(actual_input, actual_target),
        path="nn.L1Loss.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_nll_loss_module_matches_reference(torch_reference, torch_candidate) -> None:
    log_probs = [[-0.25, -2.0, -3.0], [-1.5, -0.5, -2.25]]
    expected_input = _tensor(torch_reference, log_probs)
    actual_input = _tensor(torch_candidate, log_probs)
    expected_target = torch_reference.tensor([0, 1], dtype=torch_reference.int64)
    actual_target = torch_candidate.tensor([0, 1], dtype=torch_candidate.int64)

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.NLLLoss(reduction="sum")(expected_input, expected_target),
        torch_candidate.nn.NLLLoss(reduction="sum")(actual_input, actual_target),
        path="nn.NLLLoss.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_cross_entropy_loss_module_matches_reference(torch_reference, torch_candidate) -> None:
    logits = [[1.5, -0.5, 0.25], [-1.0, 2.0, 0.5]]
    expected_input = _tensor(torch_reference, logits)
    actual_input = _tensor(torch_candidate, logits)
    expected_target = torch_reference.tensor([0, 2], dtype=torch_reference.int64)
    actual_target = torch_candidate.tensor([0, 2], dtype=torch_candidate.int64)

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.CrossEntropyLoss(reduction="sum")(expected_input, expected_target),
        torch_candidate.nn.CrossEntropyLoss(reduction="sum")(actual_input, actual_target),
        path="nn.CrossEntropyLoss.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_cross_entropy_loss_module_ignore_index_matches_reference(torch_reference, torch_candidate) -> None:
    logits = [[1.5, -0.5, 0.25], [-1.0, 2.0, 0.5]]
    expected_input = _tensor(torch_reference, logits)
    actual_input = _tensor(torch_candidate, logits)
    expected_target = torch_reference.tensor([0, -100], dtype=torch_reference.int64)
    actual_target = torch_candidate.tensor([0, -100], dtype=torch_candidate.int64)

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.CrossEntropyLoss()(expected_input, expected_target),
        torch_candidate.nn.CrossEntropyLoss()(actual_input, actual_target),
        path="nn.CrossEntropyLoss.ignore_index",
        rtol=1e-6,
        atol=1e-6,
    )


def test_nn_init_fill_helpers_match_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.empty((2, 3), dtype=torch_reference.float32)
    actual = torch_candidate.empty((2, 3), dtype=torch_candidate.float32)
    returned = torch_candidate.nn.init.constant_(actual, 2.5)
    torch_reference.nn.init.constant_(expected, 2.5)
    assert returned is actual

    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="nn.init.constant_",
        rtol=1e-6,
        atol=1e-6,
    )

    torch_reference.nn.init.zeros_(expected)
    torch_candidate.nn.init.zeros_(actual)
    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="nn.init.zeros_",
        rtol=1e-6,
        atol=1e-6,
    )

    torch_reference.nn.init.ones_(expected)
    torch_candidate.nn.init.ones_(actual)
    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="nn.init.ones_",
        rtol=1e-6,
        atol=1e-6,
    )


def test_nn_init_random_helpers_are_reproducible(torch_reference, torch_candidate) -> None:
    first = torch_candidate.empty((4, 4), dtype=torch_candidate.float32)
    second = torch_candidate.empty((4, 4), dtype=torch_candidate.float32)

    torch_candidate.manual_seed(123)
    returned = torch_candidate.nn.init.normal_(first, mean=0.5, std=0.25)
    torch_candidate.manual_seed(123)
    torch_candidate.nn.init.normal_(second, mean=0.5, std=0.25)

    assert returned is first
    assert first.tolist() == second.tolist()

    uniform = torch_candidate.empty((16,), dtype=torch_candidate.float32)
    returned = torch_candidate.nn.init.uniform_(uniform, -0.25, 0.75)
    assert returned is uniform
    assert all(-0.25 <= value <= 0.75 for value in uniform.tolist())

    trunc = torch_candidate.empty((32,), dtype=torch_candidate.float32)
    torch_candidate.manual_seed(456)
    returned = torch_candidate.nn.init.trunc_normal_(trunc, mean=0.5, std=0.25, a=0.0, b=1.0)
    assert returned is trunc
    assert all(0.0 <= value <= 1.0 for value in trunc.tolist())

    trunc_again = torch_candidate.empty((32,), dtype=torch_candidate.float32)
    torch_candidate.manual_seed(456)
    torch_candidate.nn.init.trunc_normal_(trunc_again, mean=0.5, std=0.25, a=0.0, b=1.0)
    assert trunc.tolist() == trunc_again.tolist()

    xavier = torch_candidate.empty((4, 8), dtype=torch_candidate.float32)
    returned = torch_candidate.nn.init.xavier_uniform_(xavier)
    assert returned is xavier
    bound = (6.0 / (4.0 + 8.0)) ** 0.5
    assert all(-bound <= value <= bound for row in xavier.tolist() for value in row)

    assert torch_candidate.nn.init.calculate_gain("relu") == torch_reference.nn.init.calculate_gain("relu")


def test_nn_init_random_helpers_accept_generator(torch_candidate) -> None:
    generator = torch_candidate.Generator(device="cpu")

    def assert_init_reproducible(seed, shape, initializer):
        first = torch_candidate.empty(shape, dtype=torch_candidate.float32)
        second = torch_candidate.empty(shape, dtype=torch_candidate.float32)
        generator.manual_seed(seed)
        returned = initializer(first, generator)
        generator.manual_seed(seed)
        initializer(second, generator)
        assert returned is first
        assert first.tolist() == second.tolist()

    assert_init_reproducible(
        11,
        (4, 4),
        lambda tensor, gen: torch_candidate.nn.init.normal_(tensor, mean=0.5, std=0.25, generator=gen),
    )
    assert_init_reproducible(
        12,
        (4, 4),
        lambda tensor, gen: torch_candidate.nn.init.uniform_(tensor, -0.25, 0.75, generator=gen),
    )
    assert_init_reproducible(
        13,
        (4, 4),
        lambda tensor, gen: torch_candidate.nn.init.trunc_normal_(
            tensor, mean=0.5, std=0.25, a=0.0, b=1.0, generator=gen
        ),
    )
    assert_init_reproducible(
        14,
        (4, 8),
        lambda tensor, gen: torch_candidate.nn.init.xavier_uniform_(tensor, generator=gen),
    )
    assert_init_reproducible(
        15,
        (4, 8),
        lambda tensor, gen: torch_candidate.nn.init.xavier_normal_(tensor, generator=gen),
    )
    assert_init_reproducible(
        16,
        (4, 8),
        lambda tensor, gen: torch_candidate.nn.init.kaiming_uniform_(tensor, generator=gen),
    )
    assert_init_reproducible(
        17,
        (4, 8),
        lambda tensor, gen: torch_candidate.nn.init.kaiming_normal_(tensor, generator=gen),
    )


def test_bce_loss_module_matches_reference(torch_reference, torch_candidate) -> None:
    probs = [[0.8, 0.1, 0.6], [0.25, 0.7, 0.4]]
    target = [[1.0, 0.0, 0.25], [0.5, 1.0, 0.0]]
    weight = _tensor(torch_reference, [[1.0, 0.5, 2.0], [1.5, 0.75, 1.25]])
    expected_input = _tensor(torch_reference, probs)
    actual_input = _tensor(torch_candidate, probs)
    expected_target = _tensor(torch_reference, target)
    actual_target = _tensor(torch_candidate, target)
    expected_weight = weight
    actual_weight = _tensor(torch_candidate, [[1.0, 0.5, 2.0], [1.5, 0.75, 1.25]])

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.BCELoss(weight=expected_weight, reduction="sum")(expected_input, expected_target),
        torch_candidate.nn.BCELoss(weight=actual_weight, reduction="sum")(actual_input, actual_target),
        path="nn.BCELoss.forward",
        rtol=1e-6,
        atol=1e-6,
    )


def test_bce_with_logits_loss_module_matches_reference(torch_reference, torch_candidate) -> None:
    logits = [[1.0, -2.0, 0.5], [3.0, 0.25, -1.5]]
    target = [[1.0, 0.0, 0.25], [0.5, 1.0, 0.0]]
    weight = [[1.0, 0.5, 2.0], [1.5, 0.75, 1.25]]
    pos_weight = [1.0, 2.0, 0.5]
    expected_input = _tensor(torch_reference, logits)
    actual_input = _tensor(torch_candidate, logits)
    expected_target = _tensor(torch_reference, target)
    actual_target = _tensor(torch_candidate, target)
    expected_weight = _tensor(torch_reference, weight)
    actual_weight = _tensor(torch_candidate, weight)
    expected_pos_weight = _tensor(torch_reference, pos_weight)
    actual_pos_weight = _tensor(torch_candidate, pos_weight)

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.BCEWithLogitsLoss(weight=expected_weight, pos_weight=expected_pos_weight, reduction="sum")(
            expected_input,
            expected_target,
        ),
        torch_candidate.nn.BCEWithLogitsLoss(weight=actual_weight, pos_weight=actual_pos_weight, reduction="sum")(
            actual_input,
            actual_target,
        ),
        path="nn.BCEWithLogitsLoss.forward",
        rtol=1e-6,
        atol=1e-6,
    )
