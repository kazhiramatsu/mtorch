from __future__ import annotations

import importlib
import warnings

import pytest

from .benchmarking import (
    _case_stable_diffusion_half_channels_last_controlnet_merge_setup,
    _case_stable_diffusion_half_channels_last_dpmpp_2m_scheduler_step_setup,
    _case_stable_diffusion_half_channels_last_dpmpp_sde_scheduler_step_setup,
    _case_stable_diffusion_half_channels_last_guidance_rescale_setup,
    _case_stable_diffusion_half_channels_last_inpaint_module_pipeline_setup,
    _case_stable_diffusion_half_channels_last_lora_conv_setup,
    _case_stable_diffusion_half_channels_last_module_pipeline_setup,
    _case_stable_diffusion_half_channels_last_resnet_scale_shift_setup,
    _case_stable_diffusion_half_channels_last_vae_posterior_sample_setup,
    _case_stable_diffusion_half_clip_large_text_encoder_stack_setup,
    _case_stable_diffusion_half_lora_attention_projection_setup,
    _case_stable_diffusion_inpaint_preprocess_bundle_setup,
    _case_stable_diffusion_sd3_flowmatch_cfg_step_setup,
    _case_stable_diffusion_sd3_joint_transformer_block_setup,
    _case_stable_diffusion_sd3_large_controlnet_transformer_stack_setup,
    _case_stable_diffusion_sd3_large_unpatchify_projection_setup,
    _case_stable_diffusion_sd3_mini_transformer_stack_setup,
    _case_stable_diffusion_sd3_multi_controlnet_transformer_stack_setup,
    _case_stable_diffusion_sd3_patch_embed_setup,
    _case_stable_diffusion_sd3_pooled_controlnet_denoising_step_setup,
    _case_stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_setup,
    _case_stable_diffusion_sd3_pooled_transformer_stack_setup,
    _case_stable_diffusion_sd3_qk_norm_joint_attention_setup,
    _case_stable_diffusion_sd3_rectangular_patch_embed_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_disabled_windowed_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_disabled_windowed_img2img_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_controlnet_long_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_zero_width_windowed_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_zero_width_windowed_img2img_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_disabled_windowed_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_disabled_windowed_img2img_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_zero_width_windowed_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_zero_width_windowed_img2img_denoising_loop_setup,
    _case_stable_diffusion_sd3_rectangular_rotary_joint_attention_setup,
    _case_stable_diffusion_sd3_rectangular_transformer_stack_setup,
    _case_stable_diffusion_sd3_rectangular_unpatchify_projection_setup,
    _case_stable_diffusion_sd3_rotary_joint_attention_setup,
    _case_stable_diffusion_sd3_single_transformer_block_setup,
    _case_stable_diffusion_sd3_time_text_conditioning_setup,
    _case_stable_diffusion_sd3_unpatchify_projection_setup,
    _case_stable_diffusion_sdxl_add_time_conditioning_setup,
    _case_stable_diffusion_sdxl_dual_prompt_encode_setup,
    _case_stable_diffusion_sdxl_half_channels_last_inpaint_controlnet_module_pipeline_setup,
    _case_stable_diffusion_sdxl_ip_adapter_cross_attention_setup,
    _case_stable_diffusion_sdxl_prompt_conditioning_bundle_setup,
    _case_stable_diffusion_sdxl_text_encoder2_stack_setup,
    _case_stable_diffusion_sdxl_unet_cross_attention_setup,
    _stable_diffusion_half_channels_last_controlnet_merge_path,
    _stable_diffusion_half_channels_last_dpmpp_2m_scheduler_step_path,
    _stable_diffusion_half_channels_last_dpmpp_sde_scheduler_step_path,
    _stable_diffusion_half_channels_last_guidance_rescale_path,
    _stable_diffusion_half_channels_last_lora_conv_path,
    _stable_diffusion_inpaint_module_pipeline_path,
    _stable_diffusion_module_pipeline_path,
    _stable_diffusion_half_channels_last_resnet_scale_shift_path,
    _stable_diffusion_half_channels_last_vae_posterior_sample_path,
    _stable_diffusion_half_clip_large_text_encoder_stack_path,
    _stable_diffusion_half_lora_attention_projection_path,
    _stable_diffusion_inpaint_preprocess_bundle_path,
    _stable_diffusion_sd3_flowmatch_cfg_step_path,
    _stable_diffusion_sd3_joint_transformer_block_path,
    _stable_diffusion_sd3_large_controlnet_transformer_stack_path,
    _stable_diffusion_sd3_large_unpatchify_projection_path,
    _stable_diffusion_sd3_mini_transformer_stack_path,
    _stable_diffusion_sd3_multi_controlnet_transformer_stack_path,
    _stable_diffusion_sd3_patch_embed_path,
    _stable_diffusion_sd3_pooled_controlnet_denoising_step_path,
    _stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_path,
    _stable_diffusion_sd3_pooled_transformer_stack_path,
    _stable_diffusion_sd3_qk_norm_joint_attention_path,
    _stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_path,
    _stable_diffusion_sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop_path,
    _stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path,
    _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_path,
    _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_path,
    _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop_path,
    _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_path,
    _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_path,
    _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_path,
    _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_path,
    _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_path,
    _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
    _stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_path,
    _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_path,
    _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_path,
    _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_path,
    _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_path,
    _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
    _stable_diffusion_sd3_rectangular_rotary_joint_attention_path,
    _stable_diffusion_sd3_rectangular_transformer_stack_path,
    _stable_diffusion_sd3_rectangular_unpatchify_projection_path,
    _stable_diffusion_sd3_rotary_joint_attention_path,
    _stable_diffusion_sd3_single_transformer_block_path,
    _stable_diffusion_sd3_time_text_conditioning_path,
    _stable_diffusion_sd3_unpatchify_projection_path,
    _stable_diffusion_sdxl_add_time_conditioning_path,
    _stable_diffusion_sdxl_dual_prompt_encode_path,
    _stable_diffusion_sdxl_inpaint_controlnet_module_pipeline_path,
    _stable_diffusion_sdxl_ip_adapter_cross_attention_path,
    _stable_diffusion_sdxl_prompt_conditioning_bundle_path,
    _stable_diffusion_sdxl_text_encoder2_stack_path,
    _stable_diffusion_sdxl_unet_cross_attention_path,
)
from .harness import assert_same_result, assert_values_compatible


pytestmark = [pytest.mark.compat, pytest.mark.numerics]


def test_operator_matches_reference(torch_reference, torch_candidate, op_case) -> None:
    assert_same_result(torch_reference, torch_candidate, op_case)


def test_maximum_minimum_nan_propagation_matches_reference(torch_reference, torch_candidate) -> None:
    left_data = [[float("nan"), 1.0], [3.0, -2.0]]
    right_data = [[2.0, float("nan")], [1.0, -3.0]]
    expected_left = torch_reference.tensor(left_data, dtype=torch_reference.float32)
    expected_right = torch_reference.tensor(right_data, dtype=torch_reference.float32)
    actual_left = torch_candidate.tensor(left_data, dtype=torch_candidate.float32)
    actual_right = torch_candidate.tensor(right_data, dtype=torch_candidate.float32)

    for op_name in ("maximum", "minimum"):
        expected = getattr(torch_reference, op_name)(expected_left, expected_right)
        actual = getattr(torch_candidate, op_name)(actual_left, actual_right)
        actual_as_reference = torch_reference.tensor(actual.tolist(), dtype=torch_reference.float32)
        torch_reference.testing.assert_close(expected, actual_as_reference, equal_nan=True)

        method_actual = getattr(actual_left, op_name)(actual_right)
        method_as_reference = torch_reference.tensor(method_actual.tolist(), dtype=torch_reference.float32)
        torch_reference.testing.assert_close(expected, method_as_reference, equal_nan=True)

    expected_zero_left = torch_reference.tensor([-0.0, 0.0], dtype=torch_reference.float32)
    expected_zero_right = torch_reference.tensor([0.0, -0.0], dtype=torch_reference.float32)
    actual_zero_left = torch_candidate.tensor([-0.0, 0.0], dtype=torch_candidate.float32)
    actual_zero_right = torch_candidate.tensor([0.0, -0.0], dtype=torch_candidate.float32)
    for op_name in ("maximum", "minimum"):
        expected_signs = torch_reference.signbit(getattr(torch_reference, op_name)(expected_zero_left, expected_zero_right))
        actual_signs = torch_candidate.signbit(getattr(torch_candidate, op_name)(actual_zero_left, actual_zero_right))
        assert actual_signs.tolist() == expected_signs.tolist()


def test_cummax_cummin_nan_and_tie_indices_match_reference(torch_reference, torch_candidate) -> None:
    data = [[1.0, float("nan"), 2.0, float("nan"), 0.0], [2.0, 2.0, 1.0, 1.0, 3.0]]
    expected = torch_reference.tensor(data, dtype=torch_reference.float32)
    actual = torch_candidate.tensor(data, dtype=torch_candidate.float32)

    for op_name in ("cummax", "cummin"):
        expected_values, expected_indices = getattr(torch_reference, op_name)(expected, dim=1)
        actual_values, actual_indices = getattr(torch_candidate, op_name)(actual, dim=1)
        actual_values_as_reference = torch_reference.tensor(actual_values.tolist(), dtype=torch_reference.float32)
        torch_reference.testing.assert_close(expected_values, actual_values_as_reference, equal_nan=True)
        assert actual_indices.tolist() == expected_indices.tolist()


def test_numeric_type_info_matches_reference(torch_reference, torch_candidate) -> None:
    for dtype_name in ("float16", "float32", "float64"):
        expected = torch_reference.finfo(getattr(torch_reference, dtype_name))
        actual = torch_candidate.finfo(getattr(torch_candidate, dtype_name))
        assert actual.bits == expected.bits
        assert actual.eps == expected.eps
        assert actual.max == expected.max
        assert actual.min == expected.min
        assert actual.smallest_normal == expected.smallest_normal
        assert actual.tiny == expected.tiny
        assert actual.resolution == expected.resolution

    assert torch_candidate.finfo().bits == torch_reference.finfo().bits

    for dtype_name in ("int32", "int64"):
        expected = torch_reference.iinfo(getattr(torch_reference, dtype_name))
        actual = torch_candidate.iinfo(getattr(torch_candidate, dtype_name))
        assert actual.bits == expected.bits
        assert actual.min == expected.min
        assert actual.max == expected.max

    with pytest.raises(TypeError):
        torch_candidate.finfo(torch_candidate.int64)
    with pytest.raises(TypeError):
        torch_candidate.iinfo(torch_candidate.bool)


def test_batched_matmul_noncontiguous_operands_match_reference(torch_reference, torch_candidate) -> None:
    left_data = [
        [[1.0, 0.5], [2.0, -2.0], [-1.0, 3.0]],
        [[-1.0, 3.0], [0.25, -0.5], [2.0, 1.0]],
    ]
    right_data = [
        [[0.25, -1.0], [1.5, 0.5], [-0.75, 2.0]],
        [[-1.0, 0.5], [2.0, -0.25], [0.75, 1.5]],
    ]
    expected_left = torch_reference.tensor(left_data, dtype=torch_reference.float32).transpose(-1, -2)
    expected_right = torch_reference.tensor(right_data, dtype=torch_reference.float32)
    actual_left = torch_candidate.tensor(left_data, dtype=torch_candidate.float32).transpose(-1, -2)
    actual_right = torch_candidate.tensor(right_data, dtype=torch_candidate.float32)

    assert_values_compatible(
        torch_reference,
        torch_reference.matmul(expected_left, expected_right),
        torch_candidate.matmul(actual_left, actual_right),
        path="matmul.noncontiguous_batched",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_batched_matmul_empty_shapes_match_reference(torch_reference, torch_candidate) -> None:
    cases = (
        ((0, 2, 3), (0, 3, 4)),
        ((2, 0, 3), (3, 4)),
        ((2, 3, 0), (0, 4)),
        ((2, 1, 2, 3), (0, 3, 4)),
    )
    for left_shape, right_shape in cases:
        expected_left = torch_reference.empty(left_shape, dtype=torch_reference.float32)
        expected_right = torch_reference.empty(right_shape, dtype=torch_reference.float32)
        actual_left = torch_candidate.empty(left_shape, dtype=torch_candidate.float32)
        actual_right = torch_candidate.empty(right_shape, dtype=torch_candidate.float32)
        assert_values_compatible(
            torch_reference,
            torch_reference.matmul(expected_left, expected_right),
            torch_candidate.matmul(actual_left, actual_right),
            path=f"matmul.empty.{left_shape}.{right_shape}",
            rtol=1e-6,
            atol=1e-6,
            check_stride=False,
        )


def test_matmul_operator_matches_reference(torch_reference, torch_candidate) -> None:
    expected_left = torch_reference.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]], dtype=torch_reference.float32)
    expected_right = torch_reference.tensor([[0.25, -1.0], [1.5, 0.5], [-0.75, 2.0]], dtype=torch_reference.float32)
    actual_left = torch_candidate.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]], dtype=torch_candidate.float32)
    actual_right = torch_candidate.tensor([[0.25, -1.0], [1.5, 0.5], [-0.75, 2.0]], dtype=torch_candidate.float32)

    assert_values_compatible(
        torch_reference,
        expected_left @ expected_right,
        actual_left @ actual_right,
        path="matmul.operator",
        rtol=1e-6,
        atol=1e-6,
    )


def test_bitwise_operators_match_reference(torch_reference, torch_candidate) -> None:
    expected_mask = torch_reference.tensor([[True, False, True], [False, True, False]], dtype=torch_reference.bool)
    expected_other_mask = torch_reference.tensor([[False, False, True], [True, True, False]], dtype=torch_reference.bool)
    actual_mask = torch_candidate.tensor([[True, False, True], [False, True, False]], dtype=torch_candidate.bool)
    actual_other_mask = torch_candidate.tensor([[False, False, True], [True, True, False]], dtype=torch_candidate.bool)

    for name, expected, actual in (
        ("bitwise.operator.invert_bool", ~expected_mask, ~actual_mask),
        ("bitwise.operator.and_bool", expected_mask & expected_other_mask, actual_mask & actual_other_mask),
        ("bitwise.operator.or_bool", expected_mask | expected_other_mask, actual_mask | actual_other_mask),
        ("bitwise.operator.xor_bool", expected_mask ^ expected_other_mask, actual_mask ^ actual_other_mask),
    ):
        assert_values_compatible(torch_reference, expected, actual, path=name)

    expected_int = torch_reference.tensor([[1, 2, 4], [8, 16, -1]], dtype=torch_reference.int64)
    expected_other_int = torch_reference.tensor([[3, 1, 8], [2, 0, 4]], dtype=torch_reference.int64)
    actual_int = torch_candidate.tensor([[1, 2, 4], [8, 16, -1]], dtype=torch_candidate.int64)
    actual_other_int = torch_candidate.tensor([[3, 1, 8], [2, 0, 4]], dtype=torch_candidate.int64)

    for name, expected, actual in (
        ("bitwise.operator.invert_int", ~expected_int, ~actual_int),
        ("bitwise.operator.and_int", expected_int & expected_other_int, actual_int & actual_other_int),
        ("bitwise.operator.or_int", expected_int | expected_other_int, actual_int | actual_other_int),
        ("bitwise.operator.xor_int", expected_int ^ expected_other_int, actual_int ^ actual_other_int),
    ):
        assert_values_compatible(torch_reference, expected, actual, path=name)


def test_python_arithmetic_operator_slots_match_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.tensor([[-5, -4, 5], [4, 7, -8]], dtype=torch_reference.int64)
    actual = torch_candidate.tensor([[-5, -4, 5], [4, 7, -8]], dtype=torch_candidate.int64)
    divisor_expected = torch_reference.tensor([[3, -3, -3], [3, 2, 5]], dtype=torch_reference.int64)
    divisor_actual = torch_candidate.tensor([[3, -3, -3], [3, 2, 5]], dtype=torch_candidate.int64)

    for name, expected_value, actual_value in (
        ("operator.floor_divide", expected // divisor_expected, actual // divisor_actual),
        ("operator.remainder", expected % divisor_expected, actual % divisor_actual),
        ("operator.abs", abs(expected), abs(actual)),
        ("operator.positive", +expected, +actual),
    ):
        assert_values_compatible(torch_reference, expected_value, actual_value, path=name)


def test_inplace_arithmetic_operator_updates_alias(torch_candidate) -> None:
    value = torch_candidate.tensor([1.0, 2.0, 3.0], dtype=torch_candidate.float32)
    alias = value
    value += 1.5
    assert value is alias
    assert value.tolist() == [2.5, 3.5, 4.5]
    value *= 2.0
    assert alias.tolist() == [5.0, 7.0, 9.0]
    value -= torch_candidate.tensor([1.0, 2.0, 3.0], dtype=torch_candidate.float32)
    assert alias.tolist() == [4.0, 5.0, 6.0]
    value /= 2.0
    assert alias.tolist() == [2.0, 2.5, 3.0]


def test_tensor_truthiness_and_len_match_reference(torch_reference, torch_candidate) -> None:
    for data in (True, False, [False], [2]):
        expected = torch_reference.tensor(data)
        actual = torch_candidate.tensor(data)
        assert bool(actual) is bool(expected)

    for reference, candidate in (
        (torch_reference.tensor([]), torch_candidate.tensor([])),
        (torch_reference.tensor([1, 2]), torch_candidate.tensor([1, 2])),
    ):
        with pytest.raises(RuntimeError):
            bool(reference)
        with pytest.raises(RuntimeError):
            bool(candidate)

    assert len(torch_candidate.ones((2, 3))) == len(torch_reference.ones((2, 3)))
    with pytest.raises(TypeError):
        len(torch_reference.tensor(1.0))
    with pytest.raises(TypeError):
        len(torch_candidate.tensor(1.0))


def test_tensor_python_scalar_conversions_match_reference(torch_reference, torch_candidate) -> None:
    scalar_cases = (
        (1.5, "float32"),
        ([2.25], "float64"),
        ([[3]], "int64"),
        (True, "bool"),
    )
    for data, dtype_name in scalar_cases:
        expected = torch_reference.tensor(data, dtype=getattr(torch_reference, dtype_name))
        actual = torch_candidate.tensor(data, dtype=getattr(torch_candidate, dtype_name))
        assert float(actual) == float(expected)
        assert int(actual) == int(expected)

    assert [10, 20, 30][torch_candidate.tensor([1], dtype=torch_candidate.long)] == [
        10,
        20,
        30,
    ][torch_reference.tensor([1], dtype=torch_reference.long)]

    with pytest.raises(Exception):
        float(torch_reference.tensor([1.0, 2.0]))
    with pytest.raises(Exception):
        float(torch_candidate.tensor([1.0, 2.0]))
    with pytest.raises(TypeError):
        [10, 20][torch_reference.tensor(1.0)]
    with pytest.raises(TypeError):
        [10, 20][torch_candidate.tensor(1.0)]


def test_tensor_iteration_matches_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch_reference.float32)
    actual = torch_candidate.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch_candidate.float32)

    for index, (expected_row, actual_row) in enumerate(zip(expected, actual, strict=True)):
        assert_values_compatible(torch_reference, expected_row, actual_row, path=f"tensor.iter[{index}]")

    with pytest.raises(TypeError):
        iter(torch_reference.tensor(1.0))
    with pytest.raises(TypeError):
        iter(torch_candidate.tensor(1.0))


def test_bmm_noncontiguous_operands_match_reference(torch_reference, torch_candidate) -> None:
    left_data = [
        [[1.0, 0.5], [2.0, -2.0], [-1.0, 3.0]],
        [[-1.0, 3.0], [0.25, -0.5], [2.0, 1.0]],
    ]
    right_data = [
        [[0.25, 1.5, -0.75], [-1.0, 0.5, 2.0]],
        [[-1.0, 2.0, 0.75], [0.5, -0.25, 1.5]],
    ]
    expected_left = torch_reference.tensor(left_data, dtype=torch_reference.float32).transpose(-1, -2)
    expected_right = torch_reference.tensor(right_data, dtype=torch_reference.float32).transpose(-1, -2)
    actual_left = torch_candidate.tensor(left_data, dtype=torch_candidate.float32).transpose(-1, -2)
    actual_right = torch_candidate.tensor(right_data, dtype=torch_candidate.float32).transpose(-1, -2)

    assert_values_compatible(
        torch_reference,
        torch_reference.bmm(expected_left, expected_right),
        torch_candidate.bmm(actual_left, actual_right),
        path="bmm.noncontiguous",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_bmm_empty_shapes_match_reference(torch_reference, torch_candidate) -> None:
    cases = (
        ((0, 2, 3), (0, 3, 4)),
        ((2, 0, 3), (2, 3, 4)),
        ((2, 3, 0), (2, 0, 4)),
    )
    for left_shape, right_shape in cases:
        expected_left = torch_reference.empty(left_shape, dtype=torch_reference.float32)
        expected_right = torch_reference.empty(right_shape, dtype=torch_reference.float32)
        actual_left = torch_candidate.empty(left_shape, dtype=torch_candidate.float32)
        actual_right = torch_candidate.empty(right_shape, dtype=torch_candidate.float32)
        assert_values_compatible(
            torch_reference,
            torch_reference.bmm(expected_left, expected_right),
            torch_candidate.bmm(actual_left, actual_right),
            path=f"bmm.empty.{left_shape}.{right_shape}",
            rtol=1e-6,
            atol=1e-6,
            check_stride=False,
        )


def test_linear_nd_noncontiguous_input_matches_reference(torch_reference, torch_candidate) -> None:
    input_data = [
        [[1.0, 0.5], [2.0, -2.0], [-1.0, 3.0]],
        [[-1.0, 3.0], [0.25, -0.5], [2.0, 1.0]],
    ]
    weight_data = [[0.25, -1.0, 0.5], [1.5, 0.5, -0.75], [-0.25, 2.0, 1.0]]
    bias_data = [[[0.5, -1.0, 0.25]], [[-0.25, 0.75, -0.5]]]
    expected_input = torch_reference.tensor(input_data, dtype=torch_reference.float32).transpose(-1, -2)
    actual_input = torch_candidate.tensor(input_data, dtype=torch_candidate.float32).transpose(-1, -2)
    expected_weight = torch_reference.tensor(weight_data, dtype=torch_reference.float32)
    actual_weight = torch_candidate.tensor(weight_data, dtype=torch_candidate.float32)
    expected_bias = torch_reference.tensor(bias_data, dtype=torch_reference.float32)
    actual_bias = torch_candidate.tensor(bias_data, dtype=torch_candidate.float32)

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.functional.linear(expected_input, expected_weight, expected_bias),
        torch_candidate.nn.functional.linear(actual_input, actual_weight, actual_bias),
        path="linear.nd_noncontiguous",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_linear_nd_empty_batch_matches_reference(torch_reference, torch_candidate) -> None:
    expected_input = torch_reference.empty((0, 3, 4), dtype=torch_reference.float32)
    actual_input = torch_candidate.empty((0, 3, 4), dtype=torch_candidate.float32)
    expected_weight = torch_reference.ones((5, 4), dtype=torch_reference.float32)
    actual_weight = torch_candidate.ones((5, 4), dtype=torch_candidate.float32)
    expected_bias = torch_reference.ones((5,), dtype=torch_reference.float32)
    actual_bias = torch_candidate.ones((5,), dtype=torch_candidate.float32)

    assert_values_compatible(
        torch_reference,
        torch_reference.nn.functional.linear(expected_input, expected_weight, expected_bias),
        torch_candidate.nn.functional.linear(actual_input, actual_weight, actual_bias),
        path="linear.nd_empty_batch",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_mm_is_2d_only_like_reference(torch_reference, torch_candidate) -> None:
    expected_left = torch_reference.ones((1, 2, 2), dtype=torch_reference.float32)
    expected_right = torch_reference.ones((1, 2, 2), dtype=torch_reference.float32)
    actual_left = torch_candidate.ones((1, 2, 2), dtype=torch_candidate.float32)
    actual_right = torch_candidate.ones((1, 2, 2), dtype=torch_candidate.float32)

    with pytest.raises(Exception):
        torch_reference.mm(expected_left, expected_right)
    with pytest.raises(Exception):
        torch_candidate.mm(actual_left, actual_right)


def test_matrix_products_reject_mixed_dtype_and_bool_like_reference(torch_reference, torch_candidate) -> None:
    checks = (
        (
            lambda module: module.matmul(
                module.ones((2, 3), dtype=module.float32),
                module.ones((3, 2), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.mm(
                module.ones((2, 3), dtype=module.float32),
                module.ones((3, 2), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.bmm(
                module.ones((1, 2, 3), dtype=module.float32),
                module.ones((1, 3, 2), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.matmul(
                module.ones((2, 3), dtype=module.bool),
                module.ones((3, 2), dtype=module.bool),
            ),
        ),
        (
            lambda module: module.einsum(
                "ij,jk->ik",
                module.ones((2, 3), dtype=module.bool),
                module.ones((3, 2), dtype=module.bool),
            ),
        ),
        (
            lambda module: module.addmm(
                module.ones((2, 2), dtype=module.float64),
                module.ones((2, 3), dtype=module.float32),
                module.ones((3, 2), dtype=module.float32),
            ),
        ),
        (
            lambda module: module.addmv(
                module.ones((2,), dtype=module.float32),
                module.ones((2, 3), dtype=module.float32),
                module.ones((3,), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.baddbmm(
                module.ones((1, 2, 2), dtype=module.float64),
                module.ones((1, 2, 3), dtype=module.float32),
                module.ones((1, 3, 2), dtype=module.float32),
            ),
        ),
        (
            lambda module: module.addbmm(
                module.ones((2, 2), dtype=module.float32),
                module.ones((1, 2, 3), dtype=module.float32),
                module.ones((1, 3, 2), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.mv(
                module.ones((2, 3), dtype=module.float32),
                module.ones((3,), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.dot(
                module.ones((3,), dtype=module.float32),
                module.ones((3,), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.vdot(
                module.ones((3,), dtype=module.float32),
                module.ones((3,), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.inner(
                module.ones((2, 3), dtype=module.float32),
                module.ones((4, 3), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.tensordot(
                module.ones((2, 3), dtype=module.float32),
                module.ones((3, 4), dtype=module.float64),
                dims=1,
            ),
        ),
        (
            lambda module: module.nn.functional.linear(
                module.ones((2, 3), dtype=module.float32),
                module.ones((4, 3), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.addmm(
                module.ones((2, 2), dtype=module.bool),
                module.ones((2, 3), dtype=module.bool),
                module.ones((3, 2), dtype=module.bool),
            ),
        ),
        (
            lambda module: module.addmv(
                module.ones((2,), dtype=module.bool),
                module.ones((2, 3), dtype=module.bool),
                module.ones((3,), dtype=module.bool),
            ),
        ),
        (
            lambda module: module.baddbmm(
                module.ones((1, 2, 2), dtype=module.bool),
                module.ones((1, 2, 3), dtype=module.bool),
                module.ones((1, 3, 2), dtype=module.bool),
            ),
        ),
        (
            lambda module: module.addbmm(
                module.ones((2, 2), dtype=module.bool),
                module.ones((1, 2, 3), dtype=module.bool),
                module.ones((1, 3, 2), dtype=module.bool),
            ),
        ),
        (
            lambda module: module.mv(
                module.ones((2, 3), dtype=module.bool),
                module.ones((3,), dtype=module.bool),
            ),
        ),
        (
            lambda module: module.dot(
                module.ones((3,), dtype=module.bool),
                module.ones((3,), dtype=module.bool),
            ),
        ),
        (
            lambda module: module.tensordot(
                module.ones((2, 3), dtype=module.bool),
                module.ones((3, 4), dtype=module.bool),
                dims=1,
            ),
        ),
        (
            lambda module: module.inner(
                module.ones((2, 3), dtype=module.bool),
                module.ones((4, 3), dtype=module.bool),
            ),
        ),
        (
            lambda module: module.nn.functional.linear(
                module.ones((2, 3), dtype=module.bool),
                module.ones((4, 3), dtype=module.bool),
            ),
        ),
        (
            lambda module: module.ones((2, 3), dtype=module.float32).mm(
                module.ones((3, 2), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.ones((2, 2), dtype=module.float64).addmm(
                module.ones((2, 3), dtype=module.float32),
                module.ones((3, 2), dtype=module.float32),
            ),
        ),
        (
            lambda module: module.ones((1, 2, 2), dtype=module.float64).baddbmm(
                module.ones((1, 2, 3), dtype=module.float32),
                module.ones((1, 3, 2), dtype=module.float32),
            ),
        ),
        (
            lambda module: module.ones((2, 3), dtype=module.bool).mv(
                module.ones((3,), dtype=module.bool),
            ),
        ),
    )
    for call in checks:
        with pytest.raises(Exception):
            call(torch_reference)
        with pytest.raises(Exception):
            call(torch_candidate)


def test_addmv_input_dtype_follows_mat_vec_like_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.addmv(
        torch_reference.ones((2,), dtype=torch_reference.float64),
        torch_reference.ones((2, 3), dtype=torch_reference.float32),
        torch_reference.ones((3,), dtype=torch_reference.float32),
    )
    actual = torch_candidate.addmv(
        torch_candidate.ones((2,), dtype=torch_candidate.float64),
        torch_candidate.ones((2, 3), dtype=torch_candidate.float32),
        torch_candidate.ones((3,), dtype=torch_candidate.float32),
    )
    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="addmv.input_dtype_cast",
        rtol=1e-6,
        atol=1e-6,
    )


def test_scaled_dot_product_attention_dtype_rules_match_reference(torch_reference, torch_candidate) -> None:
    reject_checks = (
        (
            lambda module: module.nn.functional.scaled_dot_product_attention(
                module.ones((1, 1, 2, 3), dtype=module.float64),
                module.ones((1, 1, 2, 3), dtype=module.float32),
                module.ones((1, 1, 2, 3), dtype=module.float32),
            ),
        ),
        (
            lambda module: module.nn.functional.scaled_dot_product_attention(
                module.ones((1, 1, 2, 3), dtype=module.float32),
                module.ones((1, 1, 2, 3), dtype=module.float32),
                module.ones((1, 1, 2, 3), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.nn.functional.scaled_dot_product_attention(
                module.ones((1, 1, 2, 3), dtype=module.float32),
                module.ones((1, 1, 2, 3), dtype=module.float32),
                module.ones((1, 1, 2, 3), dtype=module.float32),
                attn_mask=module.zeros((2, 2), dtype=module.float64),
            ),
        ),
    )
    for call in reject_checks:
        with pytest.raises(Exception):
            call(torch_reference)
        with pytest.raises(Exception):
            call(torch_candidate)

    expected = torch_reference.nn.functional.scaled_dot_product_attention(
        torch_reference.ones((1, 1, 2, 3), dtype=torch_reference.float64),
        torch_reference.ones((1, 1, 2, 3), dtype=torch_reference.float64),
        torch_reference.ones((1, 1, 2, 3), dtype=torch_reference.float64),
        attn_mask=torch_reference.zeros((2, 2), dtype=torch_reference.float32),
    )
    actual = torch_candidate.nn.functional.scaled_dot_product_attention(
        torch_candidate.ones((1, 1, 2, 3), dtype=torch_candidate.float64),
        torch_candidate.ones((1, 1, 2, 3), dtype=torch_candidate.float64),
        torch_candidate.ones((1, 1, 2, 3), dtype=torch_candidate.float64),
        attn_mask=torch_candidate.zeros((2, 2), dtype=torch_candidate.float32),
    )
    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="sdpa.mask_float32_with_float64_query",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_conv_and_norm_dtype_rejections_match_reference(torch_reference, torch_candidate) -> None:
    reject_checks = (
        (
            lambda module: module.nn.functional.conv1d(
                module.ones((1, 3, 5), dtype=module.float32),
                module.ones((4, 3, 3), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.nn.functional.conv2d(
                module.ones((1, 3, 5, 5), dtype=module.float32),
                module.ones((4, 3, 3, 3), dtype=module.float32),
                module.ones((4,), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.nn.functional.conv_transpose2d(
                module.ones((1, 3, 5, 5), dtype=module.bool),
                module.ones((3, 4, 3, 3), dtype=module.bool),
            ),
        ),
        (
            lambda module: module.nn.functional.layer_norm(
                module.ones((2, 3), dtype=module.float32),
                (3,),
                module.ones((3,), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.nn.functional.layer_norm(
                module.ones((2, 3), dtype=module.bool),
                (3,),
            ),
        ),
        (
            lambda module: module.nn.functional.rms_norm(
                module.ones((2, 3), dtype=module.bool),
                (3,),
            ),
        ),
        (
            lambda module: module.nn.functional.batch_norm(
                module.ones((2, 3), dtype=module.float32),
                module.zeros((3,), dtype=module.float64),
                module.ones((3,), dtype=module.float32),
                training=False,
            ),
        ),
        (
            lambda module: module.nn.functional.batch_norm(
                module.ones((2, 3), dtype=module.bool),
                None,
                None,
                training=True,
            ),
        ),
        (
            lambda module: module.nn.functional.group_norm(
                module.ones((2, 4, 3), dtype=module.float32),
                2,
                module.ones((4,), dtype=module.float64),
            ),
        ),
        (
            lambda module: module.nn.functional.group_norm(
                module.ones((2, 4, 3), dtype=module.bool),
                2,
            ),
        ),
    )
    for call in reject_checks:
        with pytest.raises(Exception):
            call(torch_reference)
        with pytest.raises(Exception):
            call(torch_candidate)


def test_rms_norm_accepts_weight_dtype_like_reference(torch_reference, torch_candidate) -> None:
    for dtype_name in ("float64", "bool", "int64"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected = torch_reference.nn.functional.rms_norm(
                torch_reference.ones((2, 3), dtype=torch_reference.float32),
                (3,),
                torch_reference.ones((3,), dtype=getattr(torch_reference, dtype_name)),
                eps=1e-5,
            )
        actual = torch_candidate.nn.functional.rms_norm(
            torch_candidate.ones((2, 3), dtype=torch_candidate.float32),
            (3,),
            torch_candidate.ones((3,), dtype=getattr(torch_candidate, dtype_name)),
            eps=1e-5,
        )
        assert_values_compatible(
            torch_reference,
            expected,
            actual,
            path=f"rms_norm.weight_dtype.{dtype_name}",
            rtol=1e-6,
            atol=1e-6,
        )


def test_rms_norm_default_eps_follows_input_dtype_like_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.functional.rms_norm(
        torch_reference.tensor([[1.0e-8, -2.0e-8, 3.0e-8]], dtype=torch_reference.float64),
        (3,),
    )
    actual = torch_candidate.nn.functional.rms_norm(
        torch_candidate.tensor([[1.0e-8, -2.0e-8, 3.0e-8]], dtype=torch_candidate.float64),
        (3,),
    )
    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="rms_norm.default_eps.float64",
        rtol=1e-12,
        atol=1e-12,
    )


def test_rms_norm_half_lastdim_unit_stride_view_matches_reference(torch_reference, torch_candidate) -> None:
    def path(module, with_weight):
        base = module.arange(0, 2 * 5 * 3 * 8, dtype=module.float32).reshape(2, 5, 3, 8)
        hidden = (module.sin(base * 0.013) * 0.25).to(dtype=module.float16).transpose(1, 2)
        weight = None
        if with_weight:
            weight = (1.0 + module.arange(0, 8, dtype=module.float32) / 64.0).to(dtype=module.float16)
        return module.nn.functional.rms_norm(hidden, (8,), weight, eps=1e-6)

    for with_weight in (False, True):
        assert_values_compatible(
            torch_reference,
            path(torch_reference, with_weight),
            path(torch_candidate, with_weight),
            path=f"rms_norm.half_lastdim_unit_stride_view.{with_weight}",
            rtol=3e-3,
            atol=3e-3,
            check_stride=True,
        )


def test_conv_integer_dtype_matches_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.nn.functional.conv2d(
        torch_reference.ones((1, 1, 3, 3), dtype=torch_reference.int64),
        torch_reference.ones((1, 1, 2, 2), dtype=torch_reference.int64),
    )
    actual = torch_candidate.nn.functional.conv2d(
        torch_candidate.ones((1, 1, 3, 3), dtype=torch_candidate.int64),
        torch_candidate.ones((1, 1, 2, 2), dtype=torch_candidate.int64),
    )
    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="conv2d.int64",
        check_stride=False,
    )


def test_conv_string_padding_matches_reference(torch_reference, torch_candidate) -> None:
    cases = (
        (
            "conv1d.valid_stride",
            lambda module: module.nn.functional.conv1d(
                module.arange(1, 1 + 1 * 1 * 6, dtype=module.float32).reshape(1, 1, 6),
                module.ones((1, 1, 3), dtype=module.float32),
                stride=2,
                padding="valid",
            ),
        ),
        (
            "conv1d.same_asymmetric",
            lambda module: module.nn.functional.conv1d(
                module.arange(1, 1 + 1 * 1 * 5, dtype=module.float32).reshape(1, 1, 5),
                module.ones((1, 1, 2), dtype=module.float32),
                padding="same",
            ),
        ),
        (
            "conv2d.same_asymmetric",
            lambda module: module.nn.functional.conv2d(
                module.arange(1, 1 + 1 * 1 * 3 * 4, dtype=module.float32).reshape(1, 1, 3, 4),
                module.ones((1, 1, 2, 4), dtype=module.float32),
                padding="same",
            ),
        ),
        (
            "conv2d.valid_stride",
            lambda module: module.nn.functional.conv2d(
                module.arange(1, 1 + 1 * 1 * 4 * 4, dtype=module.float32).reshape(1, 1, 4, 4),
                module.ones((1, 1, 2, 2), dtype=module.float32),
                stride=2,
                padding="valid",
            ),
        ),
    )
    for name, call in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected = call(torch_reference)
        actual = call(torch_candidate)
        assert_values_compatible(
            torch_reference,
            expected,
            actual,
            path=name,
            rtol=1e-6,
            atol=1e-6,
            check_stride=False,
        )

    reject_checks = (
        lambda module: module.nn.functional.conv1d(
            module.ones((1, 1, 5), dtype=module.float32),
            module.ones((1, 1, 3), dtype=module.float32),
            stride=2,
            padding="same",
        ),
        lambda module: module.nn.functional.conv2d(
            module.ones((1, 1, 5, 5), dtype=module.float32),
            module.ones((1, 1, 3, 3), dtype=module.float32),
            stride=2,
            padding="same",
        ),
    )
    for call in reject_checks:
        with pytest.raises(Exception):
            call(torch_reference)
        with pytest.raises(Exception):
            call(torch_candidate)


def test_einsum_outer_product_promotes_dtype_like_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.einsum(
        "i,j->ij",
        torch_reference.tensor([1, 2], dtype=torch_reference.int64),
        torch_reference.tensor([0.5, -1.0, 2.0], dtype=torch_reference.float32),
    )
    actual = torch_candidate.einsum(
        "i,j->ij",
        torch_candidate.tensor([1, 2], dtype=torch_candidate.int64),
        torch_candidate.tensor([0.5, -1.0, 2.0], dtype=torch_candidate.float32),
    )
    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="einsum.outer_promote",
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.mutation
def test_inplace_operator_matches_reference(torch_reference, torch_candidate, inplace_case) -> None:
    assert_same_result(torch_reference, torch_candidate, inplace_case)


def test_empty_strided_metadata_and_fill_match_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.empty_strided((2, 3), (1, 2), dtype=torch_reference.float32, device="cpu")
    actual = torch_candidate.empty_strided((2, 3), (1, 2), dtype=torch_candidate.float32, device="cpu")
    expected.fill_(7.0)
    actual.fill_(7.0)

    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="empty_strided.fill",
        rtol=1e-6,
        atol=1e-6,
    )

    expected_grad = torch_reference.empty_strided((2, 3), (1, 2), dtype=torch_reference.float32, requires_grad=True)
    actual_grad = torch_candidate.empty_strided((2, 3), (1, 2), dtype=torch_candidate.float32, requires_grad=True)
    assert tuple(expected_grad.shape) == tuple(actual_grad.shape)
    assert expected_grad.stride() == actual_grad.stride()
    assert expected_grad.requires_grad == actual_grad.requires_grad


def test_tensor_data_alias_and_assignment_match_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.tensor([1.0, 2.0], dtype=torch_reference.float32, requires_grad=True)
    actual = torch_candidate.tensor([1.0, 2.0], dtype=torch_candidate.float32, requires_grad=True)

    expected_data = expected.data
    actual_data = actual.data
    expected_data.add_(3.0)
    actual_data.add_(3.0)

    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="Tensor.data.alias",
        rtol=1e-6,
        atol=1e-6,
    )
    assert expected_data.requires_grad == actual_data.requires_grad

    expected_source = torch_reference.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch_reference.float64)
    actual_source = torch_candidate.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch_candidate.float64)
    expected.data = expected_source
    actual.data = actual_source

    assert_values_compatible(
        torch_reference,
        expected,
        actual,
        path="Tensor.data.assign",
        rtol=1e-6,
        atol=1e-6,
    )
    assert expected.requires_grad == actual.requires_grad
    assert expected.data.requires_grad == actual.data.requires_grad


def test_tensor_dtype_compares_with_public_dtype_singletons(torch_reference, torch_candidate) -> None:
    expected = torch_reference.tensor([1.0, 2.0], dtype=torch_reference.float32)
    actual = torch_candidate.tensor([1.0, 2.0], dtype=torch_candidate.float32)

    assert expected.dtype == torch_reference.float32
    assert actual.dtype == torch_candidate.float32
    assert torch_candidate.float32 == actual.dtype
    assert actual.dtype.name == "float32"
    assert repr(torch_candidate.float32) == "mtorch.float32"


def test_tensor_device_compares_with_public_device_objects(torch_reference, torch_candidate) -> None:
    expected = torch_reference.tensor([1.0, 2.0], device=torch_reference.device("cpu"))
    actual = torch_candidate.tensor([1.0, 2.0], device=torch_candidate.device("cpu"))

    assert expected.device == torch_reference.device("cpu")
    assert actual.device == torch_candidate.device("cpu")
    assert torch_candidate.device("cpu") == actual.device
    assert actual.device.type == "cpu"
    assert actual.device.index is None
    assert repr(torch_candidate.device("cpu")) == "device(type='cpu')"


def test_randn_like_metadata_and_manual_seed_reproducibility(torch_reference, torch_candidate) -> None:
    expected_base = torch_reference.zeros((2, 3), dtype=torch_reference.float64)
    actual_base = torch_candidate.zeros((2, 3), dtype=torch_candidate.float64)

    torch_reference.manual_seed(123)
    expected_first = torch_reference.randn_like(expected_base)
    torch_reference.manual_seed(123)
    expected_second = torch_reference.randn_like(expected_base)
    torch_candidate.manual_seed(123)
    actual_first = torch_candidate.randn_like(actual_base)
    torch_candidate.manual_seed(123)
    actual_second = torch_candidate.randn_like(actual_base)

    assert expected_first.shape == actual_first.shape
    assert actual_first.dtype == torch_candidate.float64
    assert actual_first.device == torch_candidate.device("cpu")
    assert_values_compatible(
        torch_reference,
        expected_second,
        expected_first,
        path="torch.randn_like.manual_seed",
        rtol=0.0,
        atol=0.0,
    )
    assert actual_second.tolist() == actual_first.tolist()


def test_randint_metadata_and_manual_seed_reproducibility(torch_reference, torch_candidate) -> None:
    expected = torch_reference.randint(0, 10, (2, 3), dtype=torch_reference.int64)
    expected_keyword = torch_reference.randint(low=2, high=10, size=(2, 3), dtype=torch_reference.int64)
    expected_like = torch_reference.randint_like(torch_reference.zeros((2, 3), dtype=torch_reference.float32), 5)

    torch_candidate.manual_seed(123)
    actual_first = torch_candidate.randint(0, 10, (2, 3), dtype=torch_candidate.int64)
    torch_candidate.manual_seed(123)
    actual_second = torch_candidate.randint(0, 10, (2, 3), dtype=torch_candidate.int64)
    actual_keyword = torch_candidate.randint(low=2, high=10, size=(2, 3), dtype=torch_candidate.int64)
    actual_high_only = torch_candidate.randint(high=5, size=(2, 3), dtype=torch_candidate.int64)
    actual_positional_low = torch_candidate.randint(1, high=5, size=(2, 3), dtype=torch_candidate.int64)
    actual_like_base = torch_candidate.zeros((2, 3), dtype=torch_candidate.float32)
    generator = torch_candidate.Generator(device="cpu").manual_seed(456)
    actual_like_first = torch_candidate.randint_like(actual_like_base, 5, generator=generator)
    generator.manual_seed(456)
    actual_like_second = torch_candidate.randint_like(actual_like_base, low=1, high=5, generator=generator)
    actual_like_int = torch_candidate.randint_like(actual_like_base, 1, high=5, dtype=torch_candidate.long)

    assert tuple(actual_first.shape) == tuple(expected.shape)
    assert actual_first.dtype == torch_candidate.int64
    assert actual_first.device == torch_candidate.device("cpu")
    assert actual_first.tolist() == actual_second.tolist()
    assert all(0 <= value < 10 for row in actual_first.tolist() for value in row)
    assert tuple(actual_keyword.shape) == tuple(expected_keyword.shape)
    assert all(2 <= value < 10 for row in actual_keyword.tolist() for value in row)
    assert all(0 <= value < 5 for row in actual_high_only.tolist() for value in row)
    assert all(1 <= value < 5 for row in actual_positional_low.tolist() for value in row)
    assert tuple(actual_like_first.shape) == tuple(expected_like.shape)
    assert actual_like_first.dtype == torch_candidate.float32
    assert actual_like_first.device == torch_candidate.device("cpu")
    assert actual_like_first.tolist() == torch_candidate.randint_like(actual_like_base, 5, generator=torch_candidate.Generator().manual_seed(456)).tolist()
    assert all(0.0 <= value < 5.0 for row in actual_like_first.tolist() for value in row)
    assert all(1.0 <= value < 5.0 for row in actual_like_second.tolist() for value in row)
    assert actual_like_int.dtype == torch_candidate.long
    assert all(1 <= value < 5 for row in actual_like_int.tolist() for value in row)

    bool_values = torch_candidate.randint(0, 2, (8,), dtype=torch_candidate.bool)
    assert bool_values.dtype == torch_candidate.bool
    assert all(isinstance(value, bool) for value in bool_values.tolist())


def test_rand_and_rand_like_metadata_and_manual_seed_reproducibility(torch_reference, torch_candidate) -> None:
    expected = torch_reference.rand((2, 3), dtype=torch_reference.float64)
    base = torch_candidate.zeros((2, 3), dtype=torch_candidate.float64)

    torch_candidate.manual_seed(321)
    actual_first = torch_candidate.rand((2, 3), dtype=torch_candidate.float64, generator=None, pin_memory=False)
    torch_candidate.manual_seed(321)
    actual_second = torch_candidate.rand_like(base, generator=None, pin_memory=False, layout=None)
    actual_normal = torch_candidate.randn((2, 3), dtype=torch_candidate.float64, generator=None, pin_memory=False, layout=None, out=None)
    actual_normal_like = torch_candidate.randn_like(base, generator=None, pin_memory=False, layout=None)

    assert tuple(actual_first.shape) == tuple(expected.shape)
    assert actual_first.dtype == torch_candidate.float64
    assert actual_first.device == torch_candidate.device("cpu")
    assert actual_first.tolist() == actual_second.tolist()
    assert tuple(actual_normal.shape) == tuple(expected.shape)
    assert tuple(actual_normal_like.shape) == tuple(expected.shape)
    assert all(0.0 <= value < 1.0 for row in actual_first.tolist() for value in row)


def test_bincount_matches_reference(torch_reference, torch_candidate) -> None:
    expected_indices = torch_reference.tensor([0, 1, 1, 3, 2, 3, 3], dtype=torch_reference.long)
    actual_indices = torch_candidate.tensor([0, 1, 1, 3, 2, 3, 3], dtype=torch_candidate.long)
    expected_i32 = torch_reference.tensor([2, 0, 2, 1], dtype=torch_reference.int32)
    actual_i32 = torch_candidate.tensor([2, 0, 2, 1], dtype=torch_candidate.int32)
    expected_weights = torch_reference.tensor([0.5, 1.0, -0.25, 2.0, 0.75, 0.25, 1.5], dtype=torch_reference.float32)
    actual_weights = torch_candidate.tensor([0.5, 1.0, -0.25, 2.0, 0.75, 0.25, 1.5], dtype=torch_candidate.float32)
    expected_half_weights = torch_reference.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch_reference.float16)
    actual_half_weights = torch_candidate.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch_candidate.float16)

    assert_values_compatible(
        torch_reference,
        (
            torch_reference.bincount(expected_indices),
            torch_reference.bincount(expected_i32, minlength=5),
            torch_reference.bincount(expected_indices, weights=expected_weights, minlength=6),
            torch_reference.bincount(expected_i32, weights=expected_half_weights),
            expected_indices.bincount(minlength=4),
            torch_reference.bincount(torch_reference.tensor([], dtype=torch_reference.long), minlength=3),
        ),
        (
            torch_candidate.bincount(actual_indices),
            torch_candidate.bincount(actual_i32, minlength=5),
            torch_candidate.bincount(actual_indices, weights=actual_weights, minlength=6),
            torch_candidate.bincount(actual_i32, weights=actual_half_weights),
            actual_indices.bincount(minlength=4),
            torch_candidate.bincount(torch_candidate.tensor([], dtype=torch_candidate.long), minlength=3),
        ),
        path="torch.bincount",
        rtol=1e-6,
        atol=1e-6,
    )

    grad_weights = torch_candidate.tensor([1.0, 2.0], dtype=torch_candidate.float32, requires_grad=True)
    grad_result = torch_candidate.bincount(torch_candidate.tensor([0, 1], dtype=torch_candidate.long), weights=grad_weights)
    assert grad_result.requires_grad
    with pytest.raises(RuntimeError):
        grad_result.sum().backward()

    for invalid in (
        lambda module: module.bincount(module.tensor([-1], dtype=module.long)),
        lambda module: module.bincount(module.tensor([[1, 2]], dtype=module.long)),
        lambda module: module.bincount(module.tensor([1.0], dtype=module.float32)),
        lambda module: module.bincount(module.tensor([1, 2], dtype=module.long), weights=module.tensor([1.0], dtype=module.float32)),
        lambda module: module.bincount(module.tensor([1, 2], dtype=module.long), minlength=-1),
    ):
        with pytest.raises(Exception):
            invalid(torch_reference)
        with pytest.raises(Exception):
            invalid(torch_candidate)


def test_normal_factory_overloads_match_reference_metadata(torch_reference, torch_candidate) -> None:
    expected = torch_reference.normal(0.5, 0.25, size=(2, 3), dtype=torch_reference.float64, device="cpu")
    out = torch_candidate.empty((2, 3), dtype=torch_candidate.float64)
    generator = torch_candidate.Generator(device="cpu").manual_seed(171)

    first = torch_candidate.normal(0.5, 0.25, size=(2, 3), dtype=torch_candidate.float64, device="cpu", generator=generator)
    generator.manual_seed(171)
    second = torch_candidate.normal(0.5, 0.25, size=(2, 3), dtype=torch_candidate.float64, device="cpu", generator=generator)
    returned = torch_candidate.normal(0.5, 0.25, size=(2, 3), out=out, generator=generator)

    assert tuple(first.shape) == tuple(expected.shape)
    assert first.dtype == torch_candidate.float64
    assert first.device == torch_candidate.device("cpu")
    assert first.tolist() == second.tolist()
    assert returned is out
    assert out.dtype == torch_candidate.float64

    mean = torch_candidate.zeros((2, 1), dtype=torch_candidate.float32)
    std = torch_candidate.ones((1, 3), dtype=torch_candidate.float32)
    generator.manual_seed(23)
    tensor_first = torch_candidate.normal(mean, std, generator=generator)
    generator.manual_seed(23)
    tensor_second = torch_candidate.normal(mean, std, generator=generator)
    assert tuple(tensor_first.shape) == (2, 3)
    assert tensor_first.dtype == torch_candidate.float32
    assert tensor_first.tolist() == tensor_second.tolist()

    generator.manual_seed(24)
    scalar_mean = torch_candidate.normal(0.25, std, generator=generator)
    generator.manual_seed(24)
    scalar_mean_again = torch_candidate.normal(0.25, std, generator=generator)
    assert tuple(scalar_mean.shape) == (1, 3)
    assert scalar_mean.tolist() == scalar_mean_again.tolist()

    generator.manual_seed(25)
    scalar_std = torch_candidate.normal(mean, 0.75, generator=generator)
    generator.manual_seed(25)
    scalar_std_again = torch_candidate.normal(mean, 0.75, generator=generator)
    assert tuple(scalar_std.shape) == (2, 1)
    assert scalar_std.tolist() == scalar_std_again.tolist()


def test_float16_dense_casts_match_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.tensor([[-1.5, 0.0, 2.25], [3.5, -4.75, 0.5]], dtype=torch_reference.float16)
    actual = torch_candidate.tensor([[-1.5, 0.0, 2.25], [3.5, -4.75, 0.5]], dtype=torch_candidate.float16)

    assert_values_compatible(
        torch_reference,
        (
            expected.to(dtype=torch_reference.float32),
            expected.to(dtype=torch_reference.float64),
            expected.to(dtype=torch_reference.long),
            expected.to(dtype=torch_reference.bool),
            (expected > 0).to(dtype=torch_reference.float16),
        ),
        (
            actual.to(dtype=torch_candidate.float32),
            actual.to(dtype=torch_candidate.float64),
            actual.to(dtype=torch_candidate.long),
            actual.to(dtype=torch_candidate.bool),
            (actual > 0).to(dtype=torch_candidate.float16),
        ),
        path="float16.dense_casts",
        rtol=1e-6,
        atol=1e-6,
    )


def test_float16_dense_cast_rounding_boundaries_match_reference(torch_reference, torch_candidate) -> None:
    values = [3.99853515625, 3.9990234375, 7.99853515625, 7.9990234375, -7.99853515625, float("inf")]
    expected = torch_reference.tensor(values, dtype=torch_reference.float32).to(dtype=torch_reference.float16)
    actual = torch_candidate.tensor(values, dtype=torch_candidate.float32).to(dtype=torch_candidate.float16)

    assert_values_compatible(
        torch_reference,
        expected.to(dtype=torch_reference.float32),
        actual.to(dtype=torch_candidate.float32),
        path="float16.dense_cast_rounding_boundaries",
        rtol=1e-6,
        atol=1e-6,
    )


def test_stable_diffusion_dynamic_thresholding_path_matches_reference(torch_reference, torch_candidate) -> None:
    data = [
        [
            [[-2.5, -0.5], [0.25, 1.25]],
            [[2.5, -3.0], [0.75, 1.5]],
            [[-1.75, 0.5], [3.25, -0.25]],
        ],
        [
            [[-0.5, 0.75], [1.25, -1.5]],
            [[2.25, -2.75], [0.0, 1.75]],
            [[-3.25, 2.5], [-0.75, 0.5]],
        ],
    ]

    def dynamic_threshold(module, sample):
        batch, channels, height, width = sample.shape
        flattened = sample.reshape(batch, channels * height * width)
        threshold = module.quantile(flattened.abs(), 0.8, dim=1)
        threshold = module.clamp(threshold, min=1.0, max=2.0).unsqueeze(1)
        flattened = module.clamp(flattened, min=-threshold, max=threshold) / threshold
        return flattened.reshape(batch, channels, height, width)

    expected = torch_reference.tensor(data, dtype=torch_reference.float32)
    actual = torch_candidate.tensor(data, dtype=torch_candidate.float32)

    assert_values_compatible(
        torch_reference,
        dynamic_threshold(torch_reference, expected),
        dynamic_threshold(torch_candidate, actual),
        path="stable_diffusion.dynamic_thresholding",
        rtol=1e-6,
        atol=1e-6,
    )


def test_float32_row_broadcast_binary_path_matches_reference(torch_reference, torch_candidate) -> None:
    expected_matrix = torch_reference.tensor(
        [[-2.0, -0.5, 0.25, 1.5], [2.5, -3.0, 0.75, 4.0]],
        dtype=torch_reference.float32,
    )
    expected_column = torch_reference.tensor([[1.25], [2.0]], dtype=torch_reference.float32)
    actual_matrix = torch_candidate.tensor(
        [[-2.0, -0.5, 0.25, 1.5], [2.5, -3.0, 0.75, 4.0]],
        dtype=torch_candidate.float32,
    )
    actual_column = torch_candidate.tensor([[1.25], [2.0]], dtype=torch_candidate.float32)

    for op_name in ("add", "sub", "mul", "div", "maximum", "minimum"):
        reference_op = getattr(torch_reference, op_name)
        candidate_op = getattr(torch_candidate, op_name)
        assert_values_compatible(
            torch_reference,
            reference_op(expected_matrix, expected_column),
            candidate_op(actual_matrix, actual_column),
            path=f"binary.float32_row_broadcast.{op_name}.matrix_column",
            rtol=1e-6,
            atol=1e-6,
            check_stride=False,
        )
        assert_values_compatible(
            torch_reference,
            reference_op(expected_column, expected_matrix),
            candidate_op(actual_column, actual_matrix),
            path=f"binary.float32_row_broadcast.{op_name}.column_matrix",
            rtol=1e-6,
            atol=1e-6,
            check_stride=False,
        )


def test_stable_diffusion_large_dynamic_thresholding_path_matches_reference(torch_reference, torch_candidate) -> None:
    def dynamic_threshold(module):
        sample = module.arange(0, 2 * 4 * 64 * 64, dtype=module.float32).reshape(2, 4, 64, 64) / 1024.0 - 4.0
        batch = sample.shape[0]
        flattened = sample.reshape(batch, -1)
        threshold = module.quantile(flattened.abs(), 0.8, dim=1)
        threshold = module.clamp(threshold, min=1.0, max=2.0).unsqueeze(1)
        return (module.clamp(flattened, min=-threshold, max=threshold) / threshold).reshape(sample.shape)

    assert_values_compatible(
        torch_reference,
        dynamic_threshold(torch_reference),
        dynamic_threshold(torch_candidate),
        path="stable_diffusion.large_dynamic_thresholding",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_stable_diffusion_guidance_rescale_path_matches_reference(torch_reference, torch_candidate) -> None:
    def guidance_rescale(module):
        noise_cfg = module.arange(0, 2 * 4 * 4 * 4, dtype=module.float32).reshape(2, 4, 4, 4) / 32.0 - 1.0
        noise_text = module.flip(noise_cfg, (-1,)) * 0.75 + 0.125
        cfg_flat = noise_cfg.reshape(noise_cfg.shape[0], -1)
        text_flat = noise_text.reshape(noise_text.shape[0], -1)
        text_std, text_mean = module.std_mean(text_flat, dim=1, correction=0, keepdim=True)
        cfg_std, cfg_mean = module.std_mean(cfg_flat, dim=1, correction=0, keepdim=True)
        cfg_variance, _ = module.var_mean(cfg_flat, dim=1, correction=0, keepdim=True)
        rescaled = (cfg_flat - cfg_mean) * (text_std / module.clamp_min(cfg_std, 1e-6)) + text_mean
        mixed = module.lerp(cfg_flat, rescaled, 0.7)
        return mixed.reshape(noise_cfg.shape), cfg_variance

    assert_values_compatible(
        torch_reference,
        guidance_rescale(torch_reference),
        guidance_rescale(torch_candidate),
        path="stable_diffusion.guidance_rescale",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_stable_diffusion_guidance_rescale_tail_std_path_matches_reference(torch_reference, torch_candidate) -> None:
    def guidance_rescale(module):
        noise_cfg = module.arange(0, 2 * 4 * 4 * 4, dtype=module.float32).reshape(2, 4, 4, 4) / 32.0 - 1.0
        noise_text = module.flip(noise_cfg, (-1,)) * 0.75 + 0.125
        reduce_dims = tuple(range(1, noise_cfg.ndim))
        text_std = noise_text.std(dim=reduce_dims, correction=0, keepdim=True)
        cfg_std = noise_cfg.std(dim=reduce_dims, correction=0, keepdim=True)
        rescaled = noise_cfg * (text_std / module.clamp_min(cfg_std, 1e-6))
        return module.lerp(noise_cfg, rescaled, 0.7)

    assert_values_compatible(
        torch_reference,
        guidance_rescale(torch_reference),
        guidance_rescale(torch_candidate),
        path="stable_diffusion.guidance_rescale_tail_std",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_stable_diffusion_half_channels_last_guidance_rescale_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_half_channels_last_guidance_rescale_setup(torch_reference)
    actual_args = _case_stable_diffusion_half_channels_last_guidance_rescale_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_half_channels_last_guidance_rescale_path(torch_reference, *expected_args),
        _stable_diffusion_half_channels_last_guidance_rescale_path(torch_candidate, *actual_args),
        path="stable_diffusion.half_channels_last_guidance_rescale",
        rtol=5e-3,
        atol=5e-3,
    )


def test_stable_diffusion_scheduler_sigma_path_matches_reference(torch_reference, torch_candidate) -> None:
    def scheduler_path(module):
        sigmas = module.logspace(-2.0, 0.0, 8, dtype=module.float32)
        queries = module.tensor([0.015, 0.1, 0.5], dtype=module.float32)
        upper_index = module.searchsorted(sigmas, queries, right=False)
        lower_index = upper_index - 1
        lower = module.take_along_dim(sigmas, lower_index, dim=0)
        upper = module.take_along_dim(sigmas, upper_index, dim=0)
        fraction = (queries - lower) / (upper - lower)
        interpolated = lower + (upper - lower) * fraction
        deltas = sigmas.diff(dim=0).broadcast_to(2, sigmas.shape[0] - 1)
        return interpolated, deltas

    assert_values_compatible(
        torch_reference,
        scheduler_path(torch_reference),
        scheduler_path(torch_candidate),
        path="stable_diffusion.scheduler_sigma",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_stable_diffusion_scheduler_derivative_path_matches_reference(torch_reference, torch_candidate) -> None:
    def scheduler_path(module):
        timesteps = module.linspace(0.0, 1.0, 8, dtype=module.float32)
        sigmas = module.logspace(-2.0, 0.0, 8, dtype=module.float32).flip(0)
        sigma_derivative = module.gradient(sigmas, spacing=(timesteps,), edge_order=2)[0]
        alpha_cumprod = 1.0 / (sigmas * sigmas + 1.0)
        alpha_derivative = module.gradient(alpha_cumprod, spacing=0.125, dim=0)[0]
        return sigma_derivative, alpha_derivative

    assert_values_compatible(
        torch_reference,
        scheduler_path(torch_reference),
        scheduler_path(torch_candidate),
        path="stable_diffusion.scheduler_derivative",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_stable_diffusion_scheduler_cumulative_sigma_bounds_path_matches_reference(torch_reference, torch_candidate) -> None:
    def scheduler_path(module):
        raw_sigmas = module.tensor([14.0, 10.0, 10.0, 4.0, 5.0, 1.0], dtype=module.float32)
        descending_sigmas, descending_indices = module.cummin(raw_sigmas, dim=0)
        reverse_ceiling, reverse_indices = module.cummax(module.flip(raw_sigmas, (0,)), dim=0)
        ceiling_sigmas = module.flip(reverse_ceiling, (0,))

        timestep_indices = module.tensor([0, 1, 2, 3, 4, 5], dtype=module.long)
        selected_sigmas = module.take_along_dim(descending_sigmas, timestep_indices, dim=0)
        latents = module.arange(0, 1 * 1 * 2 * 3, dtype=module.float32).reshape(1, 1, 2, 3) / 8.0
        scaled_latents = latents / (selected_sigmas.reshape(1, 1, 2, 3) + 1.0)
        return scaled_latents, descending_indices, ceiling_sigmas, reverse_indices

    assert_values_compatible(
        torch_reference,
        scheduler_path(torch_reference),
        scheduler_path(torch_candidate),
        path="stable_diffusion.scheduler_cumulative_sigma_bounds",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_stable_diffusion_ddim_scheduler_step_path_matches_reference(torch_reference, torch_candidate) -> None:
    def scheduler_path(module):
        betas = module.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 16, dtype=module.float32) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = module.cumprod(alphas, dim=0)
        timesteps = module.linspace(0, 15, 4, dtype=module.float32).round().to(dtype=module.long).flip(0)
        sample = module.arange(0, 2 * 4 * 4 * 4, dtype=module.float32).reshape(2, 4, 4, 4) / 64.0 - 1.0
        model_output = module.flip(sample, (-1,)) * 0.75 + 0.1
        interval = module.tensor(4, dtype=module.long)
        zero = module.tensor(0, dtype=module.long)
        current = sample
        for timestep in timesteps:
            previous_timestep = module.clamp(timestep - interval, min=zero)
            alpha_prod_t = alphas_cumprod[timestep]
            alpha_prod_prev = alphas_cumprod[previous_timestep]
            beta_prod_t = 1.0 - alpha_prod_t
            pred_original = (current - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
            pred_original = pred_original.clamp(-1.0, 1.0)
            pred_epsilon = (current - alpha_prod_t.sqrt() * pred_original) / beta_prod_t.sqrt()
            direction = (1.0 - alpha_prod_prev).sqrt() * pred_epsilon
            current = alpha_prod_prev.sqrt() * pred_original + direction
        return current

    assert_values_compatible(
        torch_reference,
        scheduler_path(torch_reference),
        scheduler_path(torch_candidate),
        path="stable_diffusion.ddim_scheduler_step",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_stable_diffusion_karras_euler_scheduler_step_path_matches_reference(torch_reference, torch_candidate) -> None:
    def scheduler_path(module):
        ramp = module.linspace(0.0, 1.0, 32, dtype=module.float32)
        min_inv = 0.0292 ** (1.0 / 7.0)
        max_inv = 14.6146 ** (1.0 / 7.0)
        sigmas = (max_inv + ramp * (min_inv - max_inv)) ** 7.0
        sigmas = module.cat([sigmas, module.zeros((1,), dtype=module.float32)], dim=0)
        sample = module.arange(0, 2 * 4 * 8 * 8, dtype=module.float32).reshape(2, 4, 8, 8) / 128.0 - 1.0
        model_output = module.flip(sample, (-1,)) * 0.75 + 0.1
        index = module.tensor(3, dtype=module.long)

        sigma = sigmas[index].reshape(1, 1, 1, 1)
        scaled_sample = sample / ((sigma * sigma + 1.0) ** 0.5)
        denoised = scaled_sample - model_output * sigma
        derivative = (sample - denoised) / sigma
        dt = sigmas[index + 1] - sigmas[index]
        return sample + derivative * dt.reshape(1, 1, 1, 1)

    assert_values_compatible(
        torch_reference,
        scheduler_path(torch_reference),
        scheduler_path(torch_candidate),
        path="stable_diffusion.karras_euler_scheduler_step",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_stable_diffusion_euler_ancestral_scheduler_step_path_matches_reference(torch_reference, torch_candidate) -> None:
    def scheduler_path(module):
        ramp = module.linspace(0.0, 1.0, 32, dtype=module.float32)
        min_inv = 0.0292 ** (1.0 / 7.0)
        max_inv = 14.6146 ** (1.0 / 7.0)
        sigmas = (max_inv + ramp * (min_inv - max_inv)) ** 7.0
        sigmas = module.cat([sigmas, module.zeros((1,), dtype=module.float32)], dim=0)
        sample = module.arange(0, 2 * 4 * 8 * 8, dtype=module.float32).reshape(2, 4, 8, 8) / 128.0 - 1.0
        denoised = module.flip(sample, (-1,)) * 0.75 + 0.125
        noise = module.sin(sample) * 0.5
        index = module.tensor(4, dtype=module.long)
        sigma_from = sigmas[index]
        sigma_to = sigmas[index + 1]

        sigma_up = module.sqrt(
            (sigma_to * sigma_to) * (sigma_from * sigma_from - sigma_to * sigma_to) / (sigma_from * sigma_from)
        ) * 0.8
        sigma_down = module.sqrt(module.clamp(sigma_to * sigma_to - sigma_up * sigma_up, min=0.0))
        sigma = sigma_from.reshape(1, 1, 1, 1)
        derivative = (sample - denoised) / sigma
        dt = (sigma_down - sigma_from).reshape(1, 1, 1, 1)
        return sample + derivative * dt + noise * sigma_up.reshape(1, 1, 1, 1)

    assert_values_compatible(
        torch_reference,
        scheduler_path(torch_reference),
        scheduler_path(torch_candidate),
        path="stable_diffusion.euler_ancestral_scheduler_step",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_stable_diffusion_half_channels_last_dpmpp_2m_scheduler_step_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_half_channels_last_dpmpp_2m_scheduler_step_setup(torch_reference)
    actual_args = _case_stable_diffusion_half_channels_last_dpmpp_2m_scheduler_step_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_half_channels_last_dpmpp_2m_scheduler_step_path(torch_reference, *expected_args),
        _stable_diffusion_half_channels_last_dpmpp_2m_scheduler_step_path(torch_candidate, *actual_args),
        path="stable_diffusion.half_channels_last_dpmpp_2m_scheduler_step",
        rtol=5e-3,
        atol=5e-3,
    )


def test_stable_diffusion_half_channels_last_dpmpp_sde_scheduler_step_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_half_channels_last_dpmpp_sde_scheduler_step_setup(torch_reference)
    actual_args = _case_stable_diffusion_half_channels_last_dpmpp_sde_scheduler_step_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_half_channels_last_dpmpp_sde_scheduler_step_path(torch_reference, *expected_args),
        _stable_diffusion_half_channels_last_dpmpp_sde_scheduler_step_path(torch_candidate, *actual_args),
        path="stable_diffusion.half_channels_last_dpmpp_sde_scheduler_step",
        rtol=5e-3,
        atol=5e-3,
    )


def test_stable_diffusion_scheduler_membership_and_guidance_path_matches_reference(torch_reference, torch_candidate) -> None:
    def sd_path(module):
        timesteps = module.tensor([999, 999, 750, 500, 500, 250], dtype=module.long)
        unique_steps, inverse, counts = module.unique_consecutive(timesteps, return_inverse=True, return_counts=True)
        active_steps = module.tensor([999, 250], dtype=module.long)
        active_mask = module.isin(timesteps, active_steps)

        prompt = module.arange(0, 2 * 3 * 4, dtype=module.float32).reshape(2, 3, 4) / 16.0
        keep_probability = module.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=module.float32)
        keep = module.bernoulli(keep_probability).unsqueeze(-1) > 0
        prompt = module.where(keep, prompt, module.zeros_like(prompt))

        noise_pred = module.arange(0, 4 * 2 * 2, dtype=module.float32).reshape(4, 2, 2) / 8.0
        uncond, text = noise_pred.chunk(2, dim=0)
        guided = uncond + (text - uncond) * 7.5
        return unique_steps, inverse, counts, active_mask, prompt, guided

    assert_values_compatible(
        torch_reference,
        sd_path(torch_reference),
        sd_path(torch_candidate),
        path="stable_diffusion.scheduler_membership_guidance",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_stable_diffusion_prompt_embedding_repeat_path_matches_reference(torch_reference, torch_candidate) -> None:
    def sd_path(module):
        prompt = module.arange(0, 2 * 77 * 64, dtype=module.float32).reshape(2, 77, 64) / 1024.0
        negative = (prompt + 1.0).to(dtype=module.float16)
        prompt = prompt.to(dtype=module.float16)
        prompt = prompt.repeat(1, 2, 1).reshape(4, 77, 64)
        negative = negative.repeat(1, 2, 1).reshape(4, 77, 64)
        return module.cat([negative, prompt], dim=0)

    assert_values_compatible(
        torch_reference,
        sd_path(torch_reference),
        sd_path(torch_candidate),
        path="stable_diffusion.prompt_embedding_repeat",
        rtol=1e-3,
        atol=1e-3,
        check_stride=False,
    )


def test_stable_diffusion_timestep_embedding_mlp_path_matches_reference(torch_reference, torch_candidate) -> None:
    def sd_path(module):
        dim = 64
        hidden_dim = 192
        timesteps = module.tensor([999.0, 500.0], dtype=module.float32)
        exponent = (
            -module.log(module.tensor(10000.0, dtype=module.float32))
            * module.arange(0, dim // 2, dtype=module.float32)
            / (dim // 2)
        )
        frequencies = module.exp(exponent)
        args = timesteps.reshape(-1, 1) * frequencies.reshape(1, -1)
        embedding = module.cat([module.cos(args), module.sin(args)], dim=-1).to(dtype=module.float16)

        w1_base = module.arange(0, hidden_dim * dim, dtype=module.float32).reshape(hidden_dim, dim)
        w1 = (module.sin(w1_base * 0.0007) * 0.02).to(dtype=module.float16)
        b1 = (module.arange(0, hidden_dim, dtype=module.float32) / 4096.0 - 0.04).to(dtype=module.float16)
        w2_base = module.arange(0, hidden_dim * hidden_dim, dtype=module.float32).reshape(hidden_dim, hidden_dim)
        w2 = (module.cos(w2_base * 0.0005) * 0.015).to(dtype=module.float16)
        b2 = (module.arange(0, hidden_dim, dtype=module.float32) / 8192.0 - 0.02).to(dtype=module.float16)

        hidden = module.nn.functional.linear(embedding, w1, b1)
        hidden = module.nn.functional.silu(hidden)
        return module.nn.functional.linear(hidden, w2, b2)

    assert_values_compatible(
        torch_reference,
        sd_path(torch_reference),
        sd_path(torch_candidate),
        path="stable_diffusion.timestep_embedding_mlp",
        rtol=5e-3,
        atol=5e-3,
        check_stride=False,
    )


def test_half_linear_weight_cache_tracks_inplace_updates(torch_reference, torch_candidate) -> None:
    def cache_path(module):
        input = (module.arange(0, 2 * 128, dtype=module.float32).reshape(2, 128) / 512.0 - 0.25).to(
            dtype=module.float16
        )
        weight = (module.arange(0, 96 * 128, dtype=module.float32).reshape(96, 128) / 4096.0 - 0.5).to(
            dtype=module.float16
        )
        bias = (module.arange(0, 96, dtype=module.float32) / 2048.0 - 0.02).to(dtype=module.float16)
        first = module.nn.functional.linear(input, weight, bias)
        weight.copy_(weight + 0.125)
        second = module.nn.functional.linear(input, weight, bias)
        return first, second

    assert_values_compatible(
        torch_reference,
        cache_path(torch_reference),
        cache_path(torch_candidate),
        path="nn.functional.linear_half_weight_cache_invalidation",
        rtol=2e-3,
        atol=2e-3,
        check_stride=False,
    )


def test_stable_diffusion_scheduler_timestep_bincount_path_matches_reference(torch_reference, torch_candidate) -> None:
    def sd_path(module):
        timesteps = module.tensor([999, 999, 750, 500, 500, 250, 250, 250], dtype=module.long)
        counts = module.bincount(timesteps, minlength=1000)
        sigma_slots = timesteps % 4
        weights = module.arange(0, timesteps.numel(), dtype=module.float32) / 8.0 + 0.25
        weighted_counts = module.bincount(sigma_slots, weights=weights, minlength=4)
        sigmas = module.logspace(-2.0, 0.0, 4, dtype=module.float32).flip(0)
        return counts[timesteps], weighted_counts * sigmas

    assert_values_compatible(
        torch_reference,
        sd_path(torch_reference),
        sd_path(torch_candidate),
        path="stable_diffusion.scheduler_timestep_bincount",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_stable_diffusion_scheduler_sigma_stack_path_matches_reference(torch_reference, torch_candidate) -> None:
    def sd_path(module):
        sigmas = module.logspace(0.0, -2.0, 8, dtype=module.float32)
        sigma_pairs = module.stack([sigmas[:-1], sigmas[1:]], dim=-1)
        sigma_delta = sigma_pairs[:, 1] - sigma_pairs[:, 0]
        latents = module.arange(0, 2 * 4 * 2 * 2, dtype=module.float32).reshape(2, 4, 2, 2) / 16.0
        return latents * sigma_delta[:2].reshape(2, 1, 1, 1)

    assert_values_compatible(
        torch_reference,
        sd_path(torch_reference),
        sd_path(torch_candidate),
        path="stable_diffusion.scheduler_sigma_stack",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_stable_diffusion_scheduler_timestep_table_dedup_path_matches_reference(torch_reference, torch_candidate) -> None:
    def sd_path(module):
        timestep_table = module.tensor(
            [
                [999, 0],
                [999, 0],
                [750, 1],
                [500, 2],
                [500, 2],
                [250, 3],
            ],
            dtype=module.long,
        )
        unique_rows, inverse, counts = module.unique(
            timestep_table,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )
        consecutive_rows, consecutive_inverse, consecutive_counts = module.unique_consecutive(
            timestep_table,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )

        sigmas = module.logspace(-2.0, 0.0, 4, dtype=module.float32).flip(0)
        sigma_indices = unique_rows.select(1, 1)
        sigma_values = module.take_along_dim(sigmas, sigma_indices, dim=0)
        expanded_sigma = module.take_along_dim(sigma_values, inverse, dim=0)

        batched_timestep_table = module.tensor(
            [
                [[999, 0], [999, 0], [750, 1], [500, 2], [500, 2], [250, 3]],
                [[999, 0], [999, 0], [750, 1], [500, 2], [500, 2], [250, 3]],
            ],
            dtype=module.long,
        )
        batched_unique, batched_inverse, batched_counts = module.unique(
            batched_timestep_table,
            dim=1,
            return_inverse=True,
            return_counts=True,
        )
        batched_consecutive, batched_consecutive_inverse, batched_consecutive_counts = module.unique_consecutive(
            batched_timestep_table,
            dim=1,
            return_inverse=True,
            return_counts=True,
        )
        batched_sigma_indices = batched_unique.select(2, 1)
        sigma_grid = sigmas.unsqueeze(0).broadcast_to(batched_sigma_indices.shape[0], sigmas.shape[0])
        batched_sigma_values = module.take_along_dim(sigma_grid, batched_sigma_indices, dim=1)
        inverse_grid = batched_inverse.unsqueeze(0).broadcast_to(batched_sigma_values.shape[0], batched_inverse.shape[0])
        expanded_batched_sigma = module.take_along_dim(batched_sigma_values, inverse_grid, dim=1)
        return (
            unique_rows,
            inverse,
            counts,
            consecutive_rows,
            consecutive_inverse,
            consecutive_counts,
            expanded_sigma,
            batched_unique,
            batched_inverse,
            batched_counts,
            batched_consecutive,
            batched_consecutive_inverse,
            batched_consecutive_counts,
            expanded_batched_sigma,
        )

    assert_values_compatible(
        torch_reference,
        sd_path(torch_reference),
        sd_path(torch_candidate),
        path="stable_diffusion.scheduler_timestep_table_dedup",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_stable_diffusion_clip_causal_attention_path_matches_reference(torch_reference, torch_candidate) -> None:
    def clip_path(module):
        batch, sequence, width, heads = 2, 4, 4, 2
        input_ids = module.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=module.long)
        token_weight = module.arange(0, 8 * width, dtype=module.float32).reshape(8, width) / 16.0
        position_weight = module.arange(0, sequence * width, dtype=module.float32).reshape(sequence, width) / 32.0
        position_ids = module.arange(sequence, dtype=module.long).unsqueeze(0).expand(batch, -1)
        hidden = module.nn.functional.embedding(input_ids, token_weight)
        hidden = hidden + module.nn.functional.embedding(position_ids, position_weight)
        hidden = module.nn.functional.layer_norm(hidden, (width,))

        head_dim = width // heads
        heads_view = hidden.reshape(batch, sequence, heads, head_dim).transpose(1, 2)
        mask = module.empty((batch, sequence, sequence), dtype=module.float32)
        mask.fill_(module.finfo(module.float32).min)
        mask.triu_(1)
        mask = mask.unsqueeze(1)
        return module.nn.functional.scaled_dot_product_attention(heads_view, heads_view, heads_view, attn_mask=mask)

    assert_values_compatible(
        torch_reference,
        clip_path(torch_reference),
        clip_path(torch_candidate),
        path="stable_diffusion.clip_causal_attention",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_stable_diffusion_clip_is_causal_attention_path_matches_reference(torch_reference, torch_candidate) -> None:
    def clip_path(module):
        batch, sequence, width, heads = 2, 5, 8, 2
        input_ids = module.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=module.long)
        token_weight = module.arange(0, 12 * width, dtype=module.float32).reshape(12, width) / 32.0
        position_weight = module.arange(0, sequence * width, dtype=module.float32).reshape(sequence, width) / 64.0
        position_ids = module.arange(sequence, dtype=module.long).unsqueeze(0).expand(batch, -1)
        hidden = module.nn.functional.embedding(input_ids, token_weight)
        hidden = module.nn.functional.layer_norm(hidden + module.nn.functional.embedding(position_ids, position_weight), (width,))

        head_dim = width // heads
        query = hidden.reshape(batch, sequence, heads, head_dim).transpose(1, 2)
        key = module.flip(query, (-1,))
        value = module.flip(query, (-2,))
        return module.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)

    assert_values_compatible(
        torch_reference,
        clip_path(torch_reference),
        clip_path(torch_candidate),
        path="stable_diffusion.clip_is_causal_attention",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_stable_diffusion_clip_half_is_causal_attention_path_matches_reference(torch_reference, torch_candidate) -> None:
    def clip_path(module):
        query = module.arange(0, 1 * 2 * 5 * 8, dtype=module.float16).reshape(1, 2, 5, 8) / 64.0
        key = module.flip(query, (-1,))
        value = module.arange(0, 1 * 2 * 5 * 8, dtype=module.float16).reshape(1, 2, 5, 8) / 128.0
        return module.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)

    assert_values_compatible(
        torch_reference,
        clip_path(torch_reference),
        clip_path(torch_candidate),
        path="stable_diffusion.clip_half_is_causal_attention",
        rtol=1e-3,
        atol=1e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_rank3_cross_attention_path_matches_reference(torch_reference, torch_candidate) -> None:
    def cross_attention_path(module):
        batch_heads, query_tokens, key_tokens, head_dim = 16, 64, 77, 40
        query_base = module.arange(0, batch_heads * query_tokens * head_dim, dtype=module.float32).reshape(
            batch_heads,
            query_tokens,
            head_dim,
        )
        key_base = module.arange(0, batch_heads * key_tokens * head_dim, dtype=module.float32).reshape(
            batch_heads,
            key_tokens,
            head_dim,
        )
        query = (module.sin(query_base * 0.013) * 0.2).to(dtype=module.float16)
        key = (module.cos(key_base * 0.011) * 0.2).to(dtype=module.float16)
        value = (module.sin(key_base * 0.017 + 0.25) * 0.2).to(dtype=module.float16)
        text_mask = module.ones((batch_heads, 1, key_tokens), dtype=module.bool)
        text_mask[:, :, key_tokens - 7 :] = False
        attended = module.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=text_mask,
            scale=0.125,
        )
        return attended + query

    assert_values_compatible(
        torch_reference,
        cross_attention_path(torch_reference),
        cross_attention_path(torch_candidate),
        path="stable_diffusion.half_rank3_cross_attention",
        rtol=3e-3,
        atol=3e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_legacy_baddbmm_attention_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def cross_attention_path(module):
        batch_heads, query_tokens, key_tokens, head_dim = 16, 64, 77, 40
        query_base = module.arange(0, batch_heads * query_tokens * head_dim, dtype=module.float32).reshape(
            batch_heads,
            query_tokens,
            head_dim,
        )
        key_base = module.arange(0, batch_heads * key_tokens * head_dim, dtype=module.float32).reshape(
            batch_heads,
            key_tokens,
            head_dim,
        )
        query = (module.sin(query_base * 0.013) * 0.2).to(dtype=module.float16)
        key = (module.cos(key_base * 0.011) * 0.2).to(dtype=module.float16)
        value = (module.sin(key_base * 0.017 + 0.25) * 0.2).to(dtype=module.float16)
        text_mask = module.ones((batch_heads, 1, key_tokens), dtype=module.bool)
        text_mask[:, :, key_tokens - 7 :] = False
        scores = module.baddbmm(
            module.zeros((batch_heads, query_tokens, key_tokens), dtype=module.float16),
            query,
            key.transpose(1, 2).contiguous(),
            beta=0.0,
            alpha=0.125,
        )
        scores = scores.masked_fill(~text_mask, -10000.0)
        probabilities = module.softmax(scores, dim=-1)
        return module.bmm(probabilities, value) + query

    assert_values_compatible(
        torch_reference,
        cross_attention_path(torch_reference),
        cross_attention_path(torch_candidate),
        path="stable_diffusion.half_legacy_baddbmm_attention",
        rtol=4e-3,
        atol=4e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_upcast_baddbmm_attention_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def cross_attention_path(module):
        batch_heads, query_tokens, key_tokens, head_dim = 16, 64, 77, 40
        query_base = module.arange(0, batch_heads * query_tokens * head_dim, dtype=module.float32).reshape(
            batch_heads,
            query_tokens,
            head_dim,
        )
        key_base = module.arange(0, batch_heads * key_tokens * head_dim, dtype=module.float32).reshape(
            batch_heads,
            key_tokens,
            head_dim,
        )
        query = (module.sin(query_base * 0.013) * 0.2).to(dtype=module.float16)
        key = (module.cos(key_base * 0.011) * 0.2).to(dtype=module.float16)
        value = (module.sin(key_base * 0.017 + 0.25) * 0.2).to(dtype=module.float16)
        text_mask = module.ones((batch_heads, 1, key_tokens), dtype=module.bool)
        text_mask[:, :, key_tokens - 7 :] = False
        scores = module.baddbmm(
            module.zeros((batch_heads, query_tokens, key_tokens), dtype=module.float32),
            query.to(dtype=module.float32),
            key.transpose(1, 2).contiguous().to(dtype=module.float32),
            beta=0.0,
            alpha=0.125,
        )
        scores = scores.masked_fill(~text_mask, -10000.0)
        probabilities = module.softmax(scores, dim=-1).to(dtype=module.float16)
        return module.bmm(probabilities, value) + query

    assert_values_compatible(
        torch_reference,
        cross_attention_path(torch_reference),
        cross_attention_path(torch_candidate),
        path="stable_diffusion.half_upcast_baddbmm_attention",
        rtol=4e-3,
        atol=4e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_unet_cross_attention_path_matches_reference(torch_reference, torch_candidate) -> None:
    def cross_attention_path(module):
        batch, query_tokens, key_tokens, channels, heads = 2, 64, 77, 320, 8
        head_dim = channels // heads
        hidden_base = module.arange(0, batch * query_tokens * channels, dtype=module.float32).reshape(
            batch,
            query_tokens,
            channels,
        )
        encoder_base = module.arange(0, batch * key_tokens * channels, dtype=module.float32).reshape(
            batch,
            key_tokens,
            channels,
        )
        hidden = (module.sin(hidden_base * 0.013) * 0.2).to(dtype=module.float16)
        encoder_hidden = (module.cos(encoder_base * 0.011) * 0.2).to(dtype=module.float16)
        weight_base = module.arange(0, channels * channels, dtype=module.float32).reshape(channels, channels)
        q_weight = (module.sin(weight_base * 0.007) / 64.0).to(dtype=module.float16)
        k_weight = (module.cos(weight_base * 0.009) / 64.0).to(dtype=module.float16)
        v_weight = (module.sin(weight_base * 0.011) / 64.0).to(dtype=module.float16)
        out_weight = module.flip(q_weight, (1,))
        bias = (module.arange(0, channels, dtype=module.float32) / 4096.0 - 0.03).to(dtype=module.float16)
        text_mask = module.ones((batch * heads, 1, key_tokens), dtype=module.bool)
        text_mask[:, :, key_tokens - 7 :] = False

        query = module.nn.functional.linear(hidden, q_weight, bias).reshape(batch, query_tokens, heads, head_dim)
        key = module.nn.functional.linear(encoder_hidden, k_weight, bias).reshape(batch, key_tokens, heads, head_dim)
        value = module.nn.functional.linear(encoder_hidden, v_weight, bias).reshape(batch, key_tokens, heads, head_dim)
        query = query.transpose(1, 2).reshape(batch * heads, query_tokens, head_dim)
        key = key.transpose(1, 2).reshape(batch * heads, key_tokens, head_dim)
        value = value.transpose(1, 2).reshape(batch * heads, key_tokens, head_dim)
        attended = module.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=text_mask,
            scale=0.125,
        )
        attended = attended.reshape(batch, heads, query_tokens, head_dim).transpose(1, 2).reshape(
            batch,
            query_tokens,
            channels,
        )
        return hidden + module.nn.functional.linear(attended, out_weight, bias)

    assert_values_compatible(
        torch_reference,
        cross_attention_path(torch_reference),
        cross_attention_path(torch_candidate),
        path="stable_diffusion.half_unet_cross_attention",
        rtol=4e-3,
        atol=4e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_unet_group_norm_path_matches_reference(torch_reference, torch_candidate) -> None:
    def norm_path(module):
        hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
        hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
        norm_weight = (1.0 + module.arange(0, 320, dtype=module.float32) * 0.0007).to(dtype=module.float16)
        norm_bias = (module.arange(0, 320, dtype=module.float32) * 0.0003 - 0.04).to(dtype=module.float16)
        return module.nn.functional.group_norm(hidden, 32, norm_weight, norm_bias, eps=1e-5)

    assert_values_compatible(
        torch_reference,
        norm_path(torch_reference),
        norm_path(torch_candidate),
        path="stable_diffusion.half_unet_group_norm",
        rtol=3e-3,
        atol=3e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_channels_last_group_norm_path_matches_reference(torch_reference, torch_candidate) -> None:
    def norm_path(module):
        hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
        hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last)
        norm_weight = (1.0 + module.arange(0, 320, dtype=module.float32) * 0.0007).to(dtype=module.float16)
        norm_bias = (module.arange(0, 320, dtype=module.float32) * 0.0003 - 0.04).to(dtype=module.float16)
        return module.nn.functional.group_norm(hidden, 32, norm_weight, norm_bias, eps=1e-5)

    assert_values_compatible(
        torch_reference,
        norm_path(torch_reference),
        norm_path(torch_candidate),
        path="stable_diffusion.half_channels_last_group_norm",
        rtol=3e-3,
        atol=3e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_silu_path_matches_reference(torch_reference, torch_candidate) -> None:
    def silu_path(module):
        hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
        hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last)
        return module.nn.functional.silu(hidden)

    assert_values_compatible(
        torch_reference,
        silu_path(torch_reference),
        silu_path(torch_candidate),
        path="stable_diffusion.half_channels_last_silu",
        rtol=3e-3,
        atol=3e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_3d_norm_silu_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def norm_silu_path(module):
        channels = 4
        hidden_base = module.arange(0, 1 * channels * 4 * 8 * 8, dtype=module.float32).reshape(
            1,
            channels,
            4,
            8,
            8,
        )
        hidden = (module.sin(hidden_base * 0.017) * 0.5).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last_3d)
        norm_weight = module.tensor([0.75 + index * 0.05 for index in range(channels)], dtype=module.float16)
        norm_bias = module.tensor([-0.08 + index * 0.025 for index in range(channels)], dtype=module.float16)
        normed = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
        return normed, module.nn.functional.silu(hidden), module.nn.functional.silu(normed)

    assert_values_compatible(
        torch_reference,
        norm_silu_path(torch_reference),
        norm_silu_path(torch_candidate),
        path="stable_diffusion.half_channels_last_3d_norm_silu",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_3d_residual_timestep_add_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def add_path(module):
        channels = 4
        hidden_base = module.arange(0, 2 * channels * 4 * 8 * 8, dtype=module.float32).reshape(
            2,
            channels,
            4,
            8,
            8,
        )
        hidden = (module.sin(hidden_base * 0.017) * 0.5).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last_3d)
        residual = module.flip(hidden, (-1,)).contiguous(memory_format=module.channels_last_3d)
        temb_base = module.arange(0, 2 * channels, dtype=module.float32).reshape(2, channels, 1, 1, 1)
        temb = (module.cos(temb_base * 0.019) * 0.05).to(dtype=module.float16)
        channel = (module.arange(0, channels, dtype=module.float32).reshape(channels, 1, 1, 1) / 256.0).to(
            dtype=module.float16
        )
        return hidden + residual, hidden - residual, hidden + temb, temb[:1] - hidden, hidden + channel

    assert_values_compatible(
        torch_reference,
        add_path(torch_reference),
        add_path(torch_candidate),
        path="stable_diffusion.half_channels_last_3d_residual_timestep_add",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_3d_classifier_free_guidance_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def guidance_path(module):
        channels = 4
        latent_base = module.arange(0, 2 * channels * 4 * 8 * 8, dtype=module.float32).reshape(
            2,
            channels,
            4,
            8,
            8,
        )
        latents = (module.sin(latent_base * 0.017) * 0.5).to(dtype=module.float16)
        latents = latents.contiguous(memory_format=module.channels_last_3d)
        uncond = latents * 0.75 + 0.125
        text = (module.flip(latents, (-1,)).contiguous(memory_format=module.channels_last_3d) * 0.5) - 0.25
        noise_pred = module.cat([uncond, text], dim=0)
        noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
        return noise_uncond + (noise_text - noise_uncond) * 7.5

    assert_values_compatible(
        torch_reference,
        guidance_path(torch_reference),
        guidance_path(torch_candidate),
        path="stable_diffusion.half_channels_last_3d_classifier_free_guidance",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_scheduler_scale_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def scale_path(module):
        latent_base = module.arange(0, 2 * 4 * 16 * 16, dtype=module.float32).reshape(2, 4, 16, 16)
        latents = (module.sin(latent_base * 0.017) * 0.5).to(dtype=module.float16)
        latents = latents.contiguous(memory_format=module.channels_last)
        sigma = module.tensor(2.5, dtype=module.float16).reshape(1, 1, 1, 1)
        scaled = latents / ((sigma * sigma + 1.0) ** 0.5)

        video_base = module.arange(0, 2 * 4 * 3 * 8 * 8, dtype=module.float32).reshape(2, 4, 3, 8, 8)
        video = (module.cos(video_base * 0.011) * 0.5).to(dtype=module.float16)
        video = video.contiguous(memory_format=module.channels_last_3d)
        video_sigma = module.tensor(1.75, dtype=module.float16).reshape(1, 1, 1, 1, 1)
        video_scaled = video / ((video_sigma * video_sigma + 1.0) ** 0.5)
        return scaled, video_scaled

    assert_values_compatible(
        torch_reference,
        scale_path(torch_reference),
        scale_path(torch_candidate),
        path="stable_diffusion.half_channels_last_scheduler_scale",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_scheduler_add_noise_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def add_noise_path(module):
        latent_base = module.arange(0, 2 * 4 * 16 * 16, dtype=module.float32).reshape(2, 4, 16, 16)
        latents = (module.sin(latent_base * 0.013) * 0.35).to(dtype=module.float16)
        latents = latents.contiguous(memory_format=module.channels_last)
        noise = (module.cos(latent_base * 0.017) * 0.25).to(dtype=module.float16)
        noise = noise.contiguous(memory_format=module.channels_last)
        alphas = module.tensor([0.95, 0.72, 0.33, 0.11], dtype=module.float16)
        timesteps = module.tensor([1, 2], dtype=module.long)
        alpha = module.sqrt(alphas[timesteps]).reshape(2, 1, 1, 1)
        sigma = module.sqrt(1.0 - alphas[timesteps]).reshape(2, 1, 1, 1)
        noised = latents * alpha + noise * sigma

        video_base = module.arange(0, 2 * 4 * 3 * 8 * 8, dtype=module.float32).reshape(2, 4, 3, 8, 8)
        video = (module.cos(video_base * 0.011) * 0.35).to(dtype=module.float16)
        video = video.contiguous(memory_format=module.channels_last_3d)
        video_noise = (module.sin(video_base * 0.019) * 0.25).to(dtype=module.float16)
        video_noise = video_noise.contiguous(memory_format=module.channels_last_3d)
        video_alpha = module.sqrt(alphas[timesteps]).reshape(2, 1, 1, 1, 1)
        video_sigma = module.sqrt(1.0 - alphas[timesteps]).reshape(2, 1, 1, 1, 1)
        video_noised = video * video_alpha + video_noise * video_sigma
        return noised, video_noised

    assert_values_compatible(
        torch_reference,
        add_noise_path(torch_reference),
        add_noise_path(torch_candidate),
        path="stable_diffusion.half_channels_last_scheduler_add_noise",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_mask_blend_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def mask_blend_path(module):
        latent_base = module.arange(0, 2 * 4 * 16 * 16, dtype=module.float32).reshape(2, 4, 16, 16)
        latents = (module.sin(latent_base * 0.017) * 0.5).to(dtype=module.float16)
        latents = latents.contiguous(memory_format=module.channels_last)
        init_latents = module.flip(latents, (-1,)).contiguous(memory_format=module.channels_last)
        mask = (module.arange(0, 2 * 1 * 16 * 16, dtype=module.float32).reshape(2, 1, 16, 16) % 2).to(
            dtype=module.float16
        )
        mask = mask.contiguous(memory_format=module.channels_last)
        blended = init_latents * mask + latents * (1.0 - mask)

        video_base = module.arange(0, 2 * 4 * 3 * 8 * 8, dtype=module.float32).reshape(2, 4, 3, 8, 8)
        video = (module.cos(video_base * 0.011) * 0.5).to(dtype=module.float16)
        video = video.contiguous(memory_format=module.channels_last_3d)
        init_video = module.flip(video, (-1,)).contiguous(memory_format=module.channels_last_3d)
        video_mask = (
            module.arange(0, 2 * 1 * 3 * 8 * 8, dtype=module.float32).reshape(2, 1, 3, 8, 8) % 2
        ).to(dtype=module.float16)
        video_mask = video_mask.contiguous(memory_format=module.channels_last_3d)
        blended_video = init_video * video_mask + video * (1.0 - video_mask)
        return blended, blended_video

    assert_values_compatible(
        torch_reference,
        mask_blend_path(torch_reference),
        mask_blend_path(torch_candidate),
        path="stable_diffusion.half_channels_last_mask_blend",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_controlnet_merge_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_half_channels_last_controlnet_merge_setup(torch_reference)
    actual_args = _case_stable_diffusion_half_channels_last_controlnet_merge_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_half_channels_last_controlnet_merge_path(torch_reference, *expected_args),
        _stable_diffusion_half_channels_last_controlnet_merge_path(torch_candidate, *actual_args),
        path="stable_diffusion.half_channels_last_controlnet_merge",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_3d_video_upsample_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def upsample_path(module):
        channels = 4
        hidden = module.arange(0, 1 * channels * 4 * 8 * 8, dtype=module.float32).reshape(
            1,
            channels,
            4,
            8,
            8,
        )
        hidden = (hidden / 128.0 - 4.0).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last_3d)
        spatial = module.nn.functional.interpolate(hidden, scale_factor=(1.0, 2.0, 2.0), mode="nearest")
        exact = module.nn.functional.interpolate(hidden, size=(3, 9, 11), mode="nearest-exact")
        return spatial, exact

    assert_values_compatible(
        torch_reference,
        upsample_path(torch_reference),
        upsample_path(torch_candidate),
        path="stable_diffusion.half_channels_last_3d_video_upsample",
        rtol=0.0,
        atol=0.0,
        check_stride=True,
    )


def test_stable_diffusion_half_video_upsample_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def upsample_path(module):
        channels = 4
        hidden = module.arange(0, 1 * channels * 4 * 8 * 8, dtype=module.float32).reshape(
            1,
            channels,
            4,
            8,
            8,
        )
        hidden = (hidden / 128.0 - 4.0).to(dtype=module.float16)
        spatial = module.nn.functional.interpolate(hidden, scale_factor=(1.0, 2.0, 2.0), mode="nearest")
        spatiotemporal = module.nn.functional.interpolate(hidden, scale_factor=(2.0, 2.0, 2.0), mode="nearest-exact")
        return spatial, spatiotemporal

    assert_values_compatible(
        torch_reference,
        upsample_path(torch_reference),
        upsample_path(torch_candidate),
        path="stable_diffusion.half_video_upsample",
        rtol=0.0,
        atol=0.0,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_residual_add_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def residual_path(module):
        hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
        hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last)
        residual = module.flip(hidden, (-1,)).contiguous(memory_format=module.channels_last)
        return hidden + residual

    assert_values_compatible(
        torch_reference,
        residual_path(torch_reference),
        residual_path(torch_candidate),
        path="stable_diffusion.half_channels_last_residual_add",
        rtol=3e-3,
        atol=3e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_timestep_add_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def timestep_path(module):
        hidden_base = module.arange(0, 2 * 320 * 16 * 16, dtype=module.float32).reshape(2, 320, 16, 16)
        hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last)
        temb_base = module.arange(0, 2 * 320, dtype=module.float32).reshape(2, 320, 1, 1)
        temb = (module.cos(temb_base * 0.017) * 0.05).to(dtype=module.float16)
        channel = (module.arange(0, 320, dtype=module.float32).reshape(320, 1, 1) / 4096.0).to(
            dtype=module.float16
        )
        return hidden + temb, hidden - temb, temb[:1] - hidden, hidden + channel

    assert_values_compatible(
        torch_reference,
        timestep_path(torch_reference),
        timestep_path(torch_candidate),
        path="stable_diffusion.half_channels_last_timestep_add",
        rtol=3e-3,
        atol=3e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_resnet_scale_shift_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_half_channels_last_resnet_scale_shift_setup(torch_reference)
    actual_args = _case_stable_diffusion_half_channels_last_resnet_scale_shift_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_half_channels_last_resnet_scale_shift_path(torch_reference, *expected_args),
        _stable_diffusion_half_channels_last_resnet_scale_shift_path(torch_candidate, *actual_args),
        path="stable_diffusion.half_channels_last_resnet_scale_shift",
        rtol=1e-2,
        atol=8e-2,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_classifier_free_guidance_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def guidance_path(module):
        latent_base = module.arange(0, 2 * 4 * 16 * 16, dtype=module.float32).reshape(2, 4, 16, 16)
        latents = (module.sin(latent_base * 0.017) * 0.5).to(dtype=module.float16)
        latents = latents.contiguous(memory_format=module.channels_last)
        uncond = latents * 0.75 + 0.125
        text = (module.flip(latents, (-1,)).contiguous(memory_format=module.channels_last) * 0.5) - 0.25
        noise_pred = module.cat([uncond, text], dim=0)
        noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
        return noise_uncond + (noise_text - noise_uncond) * 7.5

    assert_values_compatible(
        torch_reference,
        guidance_path(torch_reference),
        guidance_path(torch_candidate),
        path="stable_diffusion.half_channels_last_classifier_free_guidance",
        rtol=4e-3,
        atol=4e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_conv2d_path_matches_reference(torch_reference, torch_candidate) -> None:
    def conv_path(module):
        hidden_base = module.arange(0, 1 * 4 * 16 * 16, dtype=module.float32).reshape(1, 4, 16, 16)
        hidden = (module.sin(hidden_base * 0.017) * 0.5).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last)
        weight = (module.cos(module.arange(0, 8 * 4 * 3 * 3, dtype=module.float32).reshape(8, 4, 3, 3) * 0.013) * 0.05).to(
            dtype=module.float16
        )
        return module.nn.functional.conv2d(hidden, weight, None, padding=1)

    assert_values_compatible(
        torch_reference,
        conv_path(torch_reference),
        conv_path(torch_candidate),
        path="stable_diffusion.half_channels_last_conv2d",
        rtol=3e-3,
        atol=3e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_projection_conv2d_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def conv_path(module):
        hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
        hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last)
        weight_base = module.arange(0, 320 * 320, dtype=module.float32).reshape(320, 320, 1, 1)
        weight = (module.cos(weight_base * 0.007) * 0.02).to(dtype=module.float16)
        bias = (module.arange(0, 320, dtype=module.float32) / 4096.0 - 0.02).to(dtype=module.float16)
        return module.nn.functional.conv2d(hidden, weight, bias)

    assert_values_compatible(
        torch_reference,
        conv_path(torch_reference),
        conv_path(torch_candidate),
        path="stable_diffusion.half_channels_last_projection_conv2d",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_wide_conv2d_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def conv_path(module):
        hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
        hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last)
        weight_base = module.arange(0, 320 * 320 * 3 * 3, dtype=module.float32).reshape(320, 320, 3, 3)
        weight = (module.cos(weight_base * 0.007) * 0.005).to(dtype=module.float16)
        bias = (module.arange(0, 320, dtype=module.float32) / 4096.0 - 0.02).to(dtype=module.float16)
        return module.nn.functional.conv2d(hidden, weight, bias, padding=1)

    assert_values_compatible(
        torch_reference,
        conv_path(torch_reference),
        conv_path(torch_candidate),
        path="stable_diffusion.half_channels_last_wide_conv2d",
        rtol=8e-3,
        atol=8e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_lora_conv2d_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_half_channels_last_lora_conv_setup(torch_reference)
    actual_args = _case_stable_diffusion_half_channels_last_lora_conv_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_half_channels_last_lora_conv_path(torch_reference, *expected_args),
        _stable_diffusion_half_channels_last_lora_conv_path(torch_candidate, *actual_args),
        path="stable_diffusion.half_channels_last_lora_conv2d",
        rtol=8e-3,
        atol=8e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_skip_cat_conv2d_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def conv_path(module):
        hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
        hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last)
        skip = module.flip(hidden, (-1,)).contiguous(memory_format=module.channels_last)
        merged = module.cat([hidden, skip], dim=1)
        weight_base = module.arange(0, 320 * 640 * 3 * 3, dtype=module.float32).reshape(320, 640, 3, 3)
        weight = (module.cos(weight_base * 0.005) * 0.004).to(dtype=module.float16)
        bias = (module.arange(0, 320, dtype=module.float32) / 4096.0 - 0.02).to(dtype=module.float16)
        return merged, module.nn.functional.conv2d(merged, weight, bias, padding=1)

    assert_values_compatible(
        torch_reference,
        conv_path(torch_reference),
        conv_path(torch_candidate),
        path="stable_diffusion.half_channels_last_skip_cat_conv2d",
        rtol=8e-3,
        atol=8e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_conv2d_weight_cache_tracks_inplace_updates(
    torch_reference,
    torch_candidate,
) -> None:
    def conv_cache_path(module):
        hidden_base = module.arange(0, 1 * 40 * 4 * 4, dtype=module.float32).reshape(1, 40, 4, 4)
        hidden = (module.sin(hidden_base * 0.019) * 0.25).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last)
        weight_base = module.arange(0, 20 * 40 * 3 * 3, dtype=module.float32).reshape(20, 40, 3, 3)
        weight = (module.cos(weight_base * 0.011) * 0.01).to(dtype=module.float16)
        bias = (module.arange(0, 20, dtype=module.float32) / 2048.0 - 0.01).to(dtype=module.float16)
        first = module.nn.functional.conv2d(hidden, weight, bias, padding=1)
        weight.copy_(weight + 0.015625)
        second = module.nn.functional.conv2d(hidden, weight, bias, padding=1)
        return first, second

    assert_values_compatible(
        torch_reference,
        conv_cache_path(torch_reference),
        conv_cache_path(torch_candidate),
        path="stable_diffusion.half_channels_last_conv2d_weight_cache_invalidation",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_downsample_conv2d_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def conv_path(module):
        hidden_base = module.arange(0, 1 * 320 * 32 * 32, dtype=module.float32).reshape(1, 320, 32, 32)
        hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last)
        weight_base = module.arange(0, 320 * 320 * 3 * 3, dtype=module.float32).reshape(320, 320, 3, 3)
        weight = (module.cos(weight_base * 0.007) * 0.005).to(dtype=module.float16)
        bias = (module.arange(0, 320, dtype=module.float32) / 4096.0 - 0.02).to(dtype=module.float16)
        return module.nn.functional.conv2d(hidden, weight, bias, stride=2, padding=1)

    assert_values_compatible(
        torch_reference,
        conv_path(torch_reference),
        conv_path(torch_candidate),
        path="stable_diffusion.half_channels_last_downsample_conv2d",
        rtol=8e-3,
        atol=8e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_upsample_conv2d_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def upsample_conv_path(module):
        hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
        hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last)
        nearest = module.nn.functional.interpolate(hidden, scale_factor=2.0, mode="nearest")
        exact = module.nn.functional.interpolate(hidden, scale_factor=2.0, mode="nearest-exact")
        weight_base = module.arange(0, 320 * 320 * 3 * 3, dtype=module.float32).reshape(320, 320, 3, 3)
        weight = (module.cos(weight_base * 0.007) * 0.005).to(dtype=module.float16)
        bias = (module.arange(0, 320, dtype=module.float32) / 4096.0 - 0.02).to(dtype=module.float16)
        return nearest, exact, module.nn.functional.conv2d(nearest, weight, bias, padding=1)

    assert_values_compatible(
        torch_reference,
        upsample_conv_path(torch_reference),
        upsample_conv_path(torch_candidate),
        path="stable_diffusion.half_channels_last_upsample_conv2d",
        rtol=8e-3,
        atol=8e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_clip_text_encoder_path_matches_reference(torch_reference, torch_candidate) -> None:
    def clip_text_path(module):
        batch, sequence, width, heads = 2, 77, 64, 4
        head_dim = width // heads
        input_ids = module.tensor(
            [[(row * 7 + col * 5) % 32 for col in range(sequence)] for row in range(batch)],
            dtype=module.long,
        )
        position_ids = module.arange(sequence, dtype=module.long).unsqueeze(0).expand(batch, -1)
        token_base = module.arange(0, 32 * width, dtype=module.float32).reshape(32, width)
        position_base = module.arange(0, sequence * width, dtype=module.float32).reshape(sequence, width)
        token_weight = (module.sin(token_base * 0.013) * 0.2).to(dtype=module.float16)
        position_weight = (module.cos(position_base * 0.011) * 0.05).to(dtype=module.float16)
        norm_weight = (1.0 + module.arange(width, dtype=module.float32) / 256.0).to(dtype=module.float16)
        norm_bias = (module.arange(width, dtype=module.float32) / 2048.0 - 0.015).to(dtype=module.float16)
        weight_base = module.arange(0, width * width, dtype=module.float32).reshape(width, width)
        q_weight = (module.sin(weight_base * 0.007) / 64.0).to(dtype=module.float16)
        k_weight = (module.cos(weight_base * 0.009) / 64.0).to(dtype=module.float16)
        v_weight = (module.sin(weight_base * 0.011) / 64.0).to(dtype=module.float16)
        out_weight = module.flip(v_weight, (1,))
        bias = (module.arange(width, dtype=module.float32) / 4096.0 - 0.0075).to(dtype=module.float16)

        hidden = module.nn.functional.embedding(input_ids, token_weight)
        hidden = hidden + module.nn.functional.embedding(position_ids, position_weight)
        hidden = module.nn.functional.layer_norm(hidden, (width,), norm_weight, norm_bias, eps=1e-5)
        query = module.nn.functional.linear(hidden, q_weight, bias).reshape(batch, sequence, heads, head_dim).transpose(1, 2)
        key = module.nn.functional.linear(hidden, k_weight, bias).reshape(batch, sequence, heads, head_dim).transpose(1, 2)
        value = module.nn.functional.linear(hidden, v_weight, bias).reshape(batch, sequence, heads, head_dim).transpose(1, 2)
        attended = module.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, sequence, width)
        return module.nn.functional.linear(attended, out_weight, bias)

    assert_values_compatible(
        torch_reference,
        clip_text_path(torch_reference),
        clip_text_path(torch_candidate),
        path="stable_diffusion.half_clip_text_encoder",
        rtol=3e-3,
        atol=3e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_clip_large_layer_norm_path_matches_reference(torch_reference, torch_candidate) -> None:
    hidden_values = [
        [
            [
                float(((batch * 77 * 768 + token * 768 + col) % 1009) - 504) / 1024.0
                for col in range(768)
            ]
            for token in range(77)
        ]
        for batch in range(2)
    ]
    weight_values = [0.75 + float(index % 257) / 2048.0 for index in range(768)]
    bias_values = [float((index % 127) - 63) / 4096.0 for index in range(768)]

    def layer_norm_path(module):
        hidden = module.tensor(hidden_values, dtype=module.float16)
        weight = module.tensor(weight_values, dtype=module.float16)
        bias = module.tensor(bias_values, dtype=module.float16)
        return module.nn.functional.layer_norm(hidden, (768,), weight, bias, eps=1e-5)

    assert_values_compatible(
        torch_reference,
        layer_norm_path(torch_reference),
        layer_norm_path(torch_candidate),
        path="stable_diffusion.half_clip_large_layer_norm",
        rtol=4e-3,
        atol=4e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_clip_quick_gelu_path_matches_reference(torch_reference, torch_candidate) -> None:
    hidden_values = [
        [
            [
                float(((batch * 77 * 768 + token * 768 + col) % 1009) - 504) / 512.0
                for col in range(768)
            ]
            for token in range(77)
        ]
        for batch in range(2)
    ]

    def quick_gelu_path(module):
        hidden = module.tensor(hidden_values, dtype=module.float16)
        return hidden * module.sigmoid(hidden * 1.702)

    assert_values_compatible(
        torch_reference,
        quick_gelu_path(torch_reference),
        quick_gelu_path(torch_candidate),
        path="stable_diffusion.half_clip_quick_gelu",
        rtol=4e-3,
        atol=4e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_clip_mlp_path_matches_reference(torch_reference, torch_candidate) -> None:
    def mlp_path(module):
        hidden = (module.arange(0, 2 * 77 * 768, dtype=module.float32).reshape(2, 77, 768) % 1009) / 2048.0 - 0.25
        hidden = hidden.to(dtype=module.float16)
        fc1_weight = (
            (module.arange(0, 3072 * 768, dtype=module.float32).reshape(3072, 768) % 1021) / 16384.0 - 0.03
        ).to(dtype=module.float16)
        fc2_weight = (
            (module.arange(0, 768 * 3072, dtype=module.float32).reshape(768, 3072) % 1031) / 16384.0 - 0.03
        ).to(dtype=module.float16)
        fc1_bias = ((module.arange(0, 3072, dtype=module.float32) % 127) / 8192.0 - 0.008).to(dtype=module.float16)
        fc2_bias = ((module.arange(0, 768, dtype=module.float32) % 127) / 8192.0 - 0.008).to(dtype=module.float16)
        hidden = module.nn.functional.linear(hidden, fc1_weight, fc1_bias)
        hidden = hidden * module.sigmoid(hidden * 1.702)
        return module.nn.functional.linear(hidden, fc2_weight, fc2_bias)

    assert_values_compatible(
        torch_reference,
        mlp_path(torch_reference),
        mlp_path(torch_candidate),
        path="stable_diffusion.half_clip_mlp",
        rtol=4e-3,
        atol=4e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_clip_attention_projection_large_path_matches_reference(
    torch_reference, torch_candidate
) -> None:
    def attention_path(module):
        batch, tokens, width, heads, head_dim = 2, 77, 768, 12, 64
        hidden = (module.arange(0, batch * tokens * width, dtype=module.float32).reshape(batch, tokens, width) % 1009)
        hidden = (hidden / 2048.0 - 0.25).to(dtype=module.float16)
        base_weight = (module.arange(0, width * width, dtype=module.float32).reshape(width, width) % 1021) / 16384.0 - 0.03
        q_weight = base_weight.to(dtype=module.float16)
        k_weight = module.flip(base_weight, (0,)).to(dtype=module.float16)
        v_weight = module.flip(base_weight, (1,)).to(dtype=module.float16)
        out_weight = base_weight.transpose(0, 1).contiguous().to(dtype=module.float16)
        bias = ((module.arange(0, width, dtype=module.float32) % 127) / 8192.0 - 0.008).to(dtype=module.float16)
        query = module.nn.functional.linear(hidden, q_weight, bias).reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        key = module.nn.functional.linear(hidden, k_weight, bias).reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        value = module.nn.functional.linear(hidden, v_weight, bias).reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        attended = module.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        return module.nn.functional.linear(attended, out_weight, bias)

    assert_values_compatible(
        torch_reference,
        attention_path(torch_reference),
        attention_path(torch_candidate),
        path="stable_diffusion.half_clip_attention_projection_large",
        rtol=4e-3,
        atol=4e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_lora_attention_projection_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_half_lora_attention_projection_setup(torch_reference)
    actual_args = _case_stable_diffusion_half_lora_attention_projection_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_half_lora_attention_projection_path(torch_reference, *expected_args),
        _stable_diffusion_half_lora_attention_projection_path(torch_candidate, *actual_args),
        path="stable_diffusion.half_lora_attention_projection",
        rtol=5e-3,
        atol=4e-2,
        check_stride=False,
    )


def test_stable_diffusion_half_clip_transformer_block_path_matches_reference(torch_reference, torch_candidate) -> None:
    def transformer_block(module):
        batch, tokens, width, heads, head_dim = 2, 77, 768, 12, 64
        hidden = (module.arange(0, batch * tokens * width, dtype=module.float32).reshape(batch, tokens, width) % 1009)
        hidden = (hidden / 2048.0 - 0.25).to(dtype=module.float16)
        norm1_weight = (0.75 + (module.arange(0, width, dtype=module.float32) % 257) / 2048.0).to(dtype=module.float16)
        norm1_bias = (((module.arange(0, width, dtype=module.float32) % 127) - 63.0) / 4096.0).to(dtype=module.float16)
        norm2_weight = (0.8 + (module.arange(0, width, dtype=module.float32) % 251) / 2048.0).to(dtype=module.float16)
        norm2_bias = (((module.arange(0, width, dtype=module.float32) % 131) - 65.0) / 4096.0).to(dtype=module.float16)

        base_weight = (module.arange(0, width * width, dtype=module.float32).reshape(width, width) % 1021) / 16384.0 - 0.03
        q_weight = base_weight.to(dtype=module.float16)
        k_weight = module.flip(base_weight, (0,)).to(dtype=module.float16)
        v_weight = module.flip(base_weight, (1,)).to(dtype=module.float16)
        out_weight = base_weight.transpose(0, 1).contiguous().to(dtype=module.float16)
        attn_bias = ((module.arange(0, width, dtype=module.float32) % 127) / 8192.0 - 0.008).to(dtype=module.float16)

        fc1_weight = (
            (module.arange(0, 3072 * width, dtype=module.float32).reshape(3072, width) % 1021) / 16384.0 - 0.03
        ).to(dtype=module.float16)
        fc2_weight = (
            (module.arange(0, width * 3072, dtype=module.float32).reshape(width, 3072) % 1031) / 16384.0 - 0.03
        ).to(dtype=module.float16)
        fc1_bias = ((module.arange(0, 3072, dtype=module.float32) % 127) / 8192.0 - 0.008).to(dtype=module.float16)
        fc2_bias = ((module.arange(0, width, dtype=module.float32) % 127) / 8192.0 - 0.008).to(dtype=module.float16)

        normed = module.nn.functional.layer_norm(hidden, (width,), norm1_weight, norm1_bias, eps=1e-5)
        query = module.nn.functional.linear(normed, q_weight, attn_bias).reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        key = module.nn.functional.linear(normed, k_weight, attn_bias).reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        value = module.nn.functional.linear(normed, v_weight, attn_bias).reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        attended = module.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        hidden = hidden + module.nn.functional.linear(attended, out_weight, attn_bias)

        mlp_hidden = module.nn.functional.layer_norm(hidden, (width,), norm2_weight, norm2_bias, eps=1e-5)
        mlp_hidden = module.nn.functional.linear(mlp_hidden, fc1_weight, fc1_bias)
        mlp_hidden = mlp_hidden * module.sigmoid(mlp_hidden * 1.702)
        return hidden + module.nn.functional.linear(mlp_hidden, fc2_weight, fc2_bias)

    assert_values_compatible(
        torch_reference,
        transformer_block(torch_reference),
        transformer_block(torch_candidate),
        path="stable_diffusion.half_clip_transformer_block",
        rtol=5e-3,
        atol=3e-2,
        check_stride=False,
    )


def test_stable_diffusion_half_clip_large_text_encoder_stack_path_matches_reference(
    torch_reference, torch_candidate
) -> None:
    expected_args = _case_stable_diffusion_half_clip_large_text_encoder_stack_setup(torch_reference)
    actual_args = _case_stable_diffusion_half_clip_large_text_encoder_stack_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_half_clip_large_text_encoder_stack_path(torch_reference, *expected_args),
        _stable_diffusion_half_clip_large_text_encoder_stack_path(torch_candidate, *actual_args),
        path="stable_diffusion.half_clip_large_text_encoder_stack",
        rtol=8e-3,
        atol=6e-2,
        check_stride=False,
    )


def test_stable_diffusion_clip_pooled_projection_path_matches_reference(torch_reference, torch_candidate) -> None:
    def pooled_path(module):
        batch, sequence, width = 2, 6, 8
        input_ids = module.tensor([[1, 2, 9, 3, 0, 0], [0, 10, 2, 3, 4, 0]], dtype=module.long)
        hidden_base = module.arange(0, batch * sequence * width, dtype=module.float32).reshape(batch, sequence, width)
        hidden = (module.sin(hidden_base * 0.017) * 0.25).to(dtype=module.float16)
        batch_indices = module.arange(batch, dtype=module.long)
        eos_indices = input_ids.argmax(dim=-1)
        pooled = hidden[batch_indices, eos_indices]
        proj_weight = (module.arange(0, width * width, dtype=module.float32).reshape(width, width) / 512.0).to(
            dtype=module.float16
        )
        proj_bias = (module.arange(0, width, dtype=module.float32) / 1024.0 - 0.01).to(dtype=module.float16)
        return module.nn.functional.linear(pooled, proj_weight, proj_bias)

    assert_values_compatible(
        torch_reference,
        pooled_path(torch_reference),
        pooled_path(torch_candidate),
        path="stable_diffusion.clip_pooled_projection",
        rtol=2e-3,
        atol=2e-3,
        check_stride=False,
    )


def test_stable_diffusion_sdxl_dual_prompt_encode_path_matches_reference(torch_reference, torch_candidate) -> None:
    expected_args = _case_stable_diffusion_sdxl_dual_prompt_encode_setup(torch_reference)
    actual_args = _case_stable_diffusion_sdxl_dual_prompt_encode_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sdxl_dual_prompt_encode_path(torch_reference, *expected_args),
        _stable_diffusion_sdxl_dual_prompt_encode_path(torch_candidate, *actual_args),
        path="stable_diffusion.sdxl_dual_prompt_encode",
        rtol=8e-3,
        atol=6e-2,
        check_stride=False,
    )


def test_stable_diffusion_sdxl_text_encoder2_stack_path_matches_reference(torch_reference, torch_candidate) -> None:
    expected_args = _case_stable_diffusion_sdxl_text_encoder2_stack_setup(torch_reference)
    actual_args = _case_stable_diffusion_sdxl_text_encoder2_stack_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sdxl_text_encoder2_stack_path(torch_reference, *expected_args),
        _stable_diffusion_sdxl_text_encoder2_stack_path(torch_candidate, *actual_args),
        path="stable_diffusion.sdxl_text_encoder2_stack",
        rtol=8e-3,
        atol=6e-2,
        check_stride=False,
    )


def test_stable_diffusion_sdxl_add_time_conditioning_path_matches_reference(torch_reference, torch_candidate) -> None:
    expected_args = _case_stable_diffusion_sdxl_add_time_conditioning_setup(torch_reference)
    actual_args = _case_stable_diffusion_sdxl_add_time_conditioning_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sdxl_add_time_conditioning_path(torch_reference, *expected_args),
        _stable_diffusion_sdxl_add_time_conditioning_path(torch_candidate, *actual_args),
        path="stable_diffusion.sdxl_add_time_conditioning",
        rtol=5e-3,
        atol=5e-2,
        check_stride=False,
    )


def test_stable_diffusion_sdxl_prompt_conditioning_bundle_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sdxl_prompt_conditioning_bundle_setup(torch_reference)
    actual_args = _case_stable_diffusion_sdxl_prompt_conditioning_bundle_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sdxl_prompt_conditioning_bundle_path(torch_reference, *expected_args),
        _stable_diffusion_sdxl_prompt_conditioning_bundle_path(torch_candidate, *actual_args),
        path="stable_diffusion.sdxl_prompt_conditioning_bundle",
        rtol=8e-3,
        atol=6e-2,
        check_stride=False,
    )


def test_stable_diffusion_sdxl_unet_cross_attention_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sdxl_unet_cross_attention_setup(torch_reference)
    actual_args = _case_stable_diffusion_sdxl_unet_cross_attention_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sdxl_unet_cross_attention_path(torch_reference, *expected_args),
        _stable_diffusion_sdxl_unet_cross_attention_path(torch_candidate, *actual_args),
        path="stable_diffusion.sdxl_unet_cross_attention",
        rtol=6e-3,
        atol=6e-2,
        check_stride=False,
    )


def test_stable_diffusion_sdxl_ip_adapter_cross_attention_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sdxl_ip_adapter_cross_attention_setup(torch_reference)
    actual_args = _case_stable_diffusion_sdxl_ip_adapter_cross_attention_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sdxl_ip_adapter_cross_attention_path(torch_reference, *expected_args),
        _stable_diffusion_sdxl_ip_adapter_cross_attention_path(torch_candidate, *actual_args),
        path="stable_diffusion.sdxl_ip_adapter_cross_attention",
        rtol=8e-3,
        atol=7e-2,
        check_stride=False,
    )


def test_stable_diffusion_sd3_joint_transformer_block_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_joint_transformer_block_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_joint_transformer_block_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_joint_transformer_block_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_joint_transformer_block_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_joint_transformer_block",
        rtol=8e-3,
        atol=8e-2,
        check_stride=False,
    )


def test_stable_diffusion_sd3_qk_norm_joint_attention_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_qk_norm_joint_attention_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_qk_norm_joint_attention_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_qk_norm_joint_attention_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_qk_norm_joint_attention_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_qk_norm_joint_attention",
        rtol=8e-3,
        atol=8e-2,
        check_stride=False,
    )


def test_stable_diffusion_sd3_rotary_joint_attention_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rotary_joint_attention_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_rotary_joint_attention_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rotary_joint_attention_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_rotary_joint_attention_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_rotary_joint_attention",
        rtol=8e-3,
        atol=8e-2,
        check_stride=False,
    )


def test_stable_diffusion_sd3_rectangular_rotary_joint_attention_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_rotary_joint_attention_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_rectangular_rotary_joint_attention_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_rotary_joint_attention_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_rectangular_rotary_joint_attention_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_rectangular_rotary_joint_attention",
        rtol=8e-3,
        atol=8e-2,
        check_stride=False,
    )


def test_stable_diffusion_sd3_single_transformer_block_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_single_transformer_block_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_single_transformer_block_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_single_transformer_block_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_single_transformer_block_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_single_transformer_block",
        rtol=8e-3,
        atol=8e-2,
        check_stride=False,
    )


def test_stable_diffusion_sd3_time_text_conditioning_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_time_text_conditioning_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_time_text_conditioning_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_time_text_conditioning_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_time_text_conditioning_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_time_text_conditioning",
        rtol=5e-3,
        atol=5e-2,
        check_stride=False,
    )


def test_stable_diffusion_sd3_unpatchify_projection_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_unpatchify_projection_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_unpatchify_projection_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_unpatchify_projection_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_unpatchify_projection_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_unpatchify_projection",
        rtol=5e-3,
        atol=5e-2,
        check_stride=True,
    )


def test_stable_diffusion_sd3_large_unpatchify_projection_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_large_unpatchify_projection_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_large_unpatchify_projection_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_large_unpatchify_projection_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_large_unpatchify_projection_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_large_unpatchify_projection",
        rtol=5e-3,
        atol=5e-2,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_unpatchify_projection_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_unpatchify_projection_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_rectangular_unpatchify_projection_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_unpatchify_projection_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_rectangular_unpatchify_projection_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_rectangular_unpatchify_projection",
        rtol=5e-3,
        atol=5e-2,
        check_stride=True,
    )


def test_stable_diffusion_sd3_patch_embed_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_patch_embed_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_patch_embed_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_patch_embed_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_patch_embed_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_patch_embed",
        rtol=5e-3,
        atol=5e-2,
        check_stride=True,
    )
    expected_hidden = torch_reference.nn.functional.conv2d(*expected_args[:3], stride=2)
    actual_hidden = torch_candidate.nn.functional.conv2d(*actual_args[:3], stride=2)
    assert_values_compatible(
        torch_reference,
        expected_hidden.flatten(2).transpose(1, 2),
        actual_hidden.flatten(2).transpose(1, 2),
        path="stable_diffusion.sd3_patch_embed_tokens",
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_patch_embed_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_patch_embed_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_rectangular_patch_embed_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_patch_embed_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_patch_embed_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_rectangular_patch_embed",
        rtol=5e-3,
        atol=5e-2,
        check_stride=True,
    )
    expected_hidden = torch_reference.nn.functional.conv2d(*expected_args[:3], stride=2)
    actual_hidden = torch_candidate.nn.functional.conv2d(*actual_args[:3], stride=2)
    assert_values_compatible(
        torch_reference,
        expected_hidden.flatten(2).transpose(1, 2),
        actual_hidden.flatten(2).transpose(1, 2),
        path="stable_diffusion.sd3_rectangular_patch_embed_tokens",
        check_stride=True,
    )


def test_stable_diffusion_sd3_mini_transformer_stack_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_mini_transformer_stack_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_mini_transformer_stack_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_mini_transformer_stack_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_mini_transformer_stack_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_mini_transformer_stack",
        rtol=1e-2,
        atol=1e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_transformer_stack_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_transformer_stack_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_rectangular_transformer_stack_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_transformer_stack_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_rectangular_transformer_stack_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_rectangular_transformer_stack",
        rtol=1e-2,
        atol=1e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_pooled_transformer_stack_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_pooled_transformer_stack_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_pooled_transformer_stack_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_pooled_transformer_stack_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_pooled_transformer_stack_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_pooled_transformer_stack",
        rtol=1e-2,
        atol=1e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_pooled_multi_controlnet_transformer_stack",
        rtol=1e-2,
        atol=1e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_rectangular_pooled_multi_controlnet_transformer_stack",
        rtol=1e-2,
        atol=1e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_setup(
        torch_reference
    )
    actual_args = _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_setup(
        torch_candidate
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_inpaint_controlnet_transformer_stack",
        rtol=1e-2,
        atol=1e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_setup(
        torch_reference
    )
    actual_args = _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_setup(
        torch_candidate
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack",
        rtol=1e-2,
        atol=1e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_setup(
        torch_reference
    )
    actual_args = _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_setup(
        torch_candidate
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack",
        rtol=1e-2,
        atol=1e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_large_controlnet_transformer_stack_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_large_controlnet_transformer_stack_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_large_controlnet_transformer_stack_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_large_controlnet_transformer_stack_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_large_controlnet_transformer_stack_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_large_controlnet_transformer_stack",
        rtol=1e-2,
        atol=1e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_multi_controlnet_transformer_stack_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_multi_controlnet_transformer_stack_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_multi_controlnet_transformer_stack_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_multi_controlnet_transformer_stack_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_multi_controlnet_transformer_stack_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_multi_controlnet_transformer_stack",
        rtol=1e-2,
        atol=1e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_flowmatch_cfg_step_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_flowmatch_cfg_step_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_flowmatch_cfg_step_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_flowmatch_cfg_step_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_flowmatch_cfg_step_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_flowmatch_cfg_step",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_sd3_pooled_controlnet_denoising_step_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_pooled_controlnet_denoising_step_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_pooled_controlnet_denoising_step_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_pooled_controlnet_denoising_step_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_pooled_controlnet_denoising_step_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_pooled_controlnet_denoising_step",
        rtol=1e-2,
        atol=1e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_rectangular_pooled_controlnet_denoising_loop",
        rtol=2e-2,
        atol=2e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_controlnet_long_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_pooled_controlnet_long_denoising_loop_setup(torch_reference)
    actual_args = _case_stable_diffusion_sd3_rectangular_pooled_controlnet_long_denoising_loop_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_rectangular_pooled_controlnet_long_denoising_loop",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_setup(
        torch_reference
    )
    actual_args = _case_stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_setup(
        torch_candidate
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_controlnet_img2img_denoising_loop",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )

    expected_zero_strength_args = list(expected_args)
    actual_zero_strength_args = list(actual_args)
    expected_zero_strength_args[7] = 0.0
    actual_zero_strength_args[7] = 0.0
    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_path(
            torch_reference,
            *expected_zero_strength_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_path(
            torch_candidate,
            *actual_zero_strength_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_controlnet_img2img_denoising_loop_zero_strength",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop_setup(
        torch_reference
    )
    actual_args = _case_stable_diffusion_sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop_setup(
        torch_candidate
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_setup(
        torch_reference
    )
    actual_args = _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_setup(
        torch_candidate
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_path(torch_reference, *expected_args),
        _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_path(torch_candidate, *actual_args),
        path="stable_diffusion.sd3_rectangular_pooled_inpaint_controlnet_denoising_loop",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_setup(
        torch_reference
    )
    actual_args = _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_setup(
        torch_candidate
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )

    expected_zero_strength_args = list(expected_args)
    actual_zero_strength_args = list(actual_args)
    expected_zero_strength_args[9] = 0.0
    actual_zero_strength_args[9] = 0.0
    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_path(
            torch_reference,
            *expected_zero_strength_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_path(
            torch_candidate,
            *actual_zero_strength_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_zero_strength",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop_setup(
        torch_reference
    )
    actual_args = _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop_setup(
        torch_candidate
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = (
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_setup(
            torch_reference
        )
    )
    actual_args = _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_setup(
        torch_candidate
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = (
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_setup(
            torch_reference
        )
    )
    actual_args = (
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_setup(
            torch_candidate
        )
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_setup(
        torch_reference
    )
    actual_args = _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_setup(
        torch_candidate
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = (
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_setup(
            torch_reference
        )
    )
    actual_args = (
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_setup(
            torch_candidate
        )
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = (
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_setup(
            torch_reference
        )
    )
    actual_args = (
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_setup(
            torch_candidate
        )
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )

    expected_zero_strength_args = list(expected_args)
    actual_zero_strength_args = list(actual_args)
    expected_zero_strength_args[11] = 0.0
    actual_zero_strength_args[11] = 0.0
    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_path(
            torch_reference,
            *expected_zero_strength_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_path(
            torch_candidate,
            *actual_zero_strength_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_zero_strength",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = (
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup(
            torch_reference
        )
    )
    actual_args = (
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup(
            torch_candidate
        )
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )

    expected_zero_strength_args = list(expected_args)
    actual_zero_strength_args = list(actual_args)
    expected_zero_strength_args[11] = 0.0
    actual_zero_strength_args[11] = 0.0
    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path(
            torch_reference,
            *expected_zero_strength_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path(
            torch_candidate,
            *actual_zero_strength_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_zero_strength",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = (
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_setup(
            torch_reference
        )
    )
    actual_args = (
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_setup(
            torch_candidate
        )
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )

    expected_zero_strength_args = list(expected_args)
    actual_zero_strength_args = list(actual_args)
    expected_zero_strength_args[10] = 0.0
    actual_zero_strength_args[10] = 0.0
    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_path(
            torch_reference,
            *expected_zero_strength_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_path(
            torch_candidate,
            *actual_zero_strength_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_zero_strength",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = (
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup(
            torch_reference
        )
    )
    actual_args = (
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup(
            torch_candidate
        )
    )

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path(
            torch_reference,
            *expected_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path(
            torch_candidate,
            *actual_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )

    expected_zero_strength_args = list(expected_args)
    actual_zero_strength_args = list(actual_args)
    expected_zero_strength_args[10] = 0.0
    actual_zero_strength_args[10] = 0.0
    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path(
            torch_reference,
            *expected_zero_strength_args,
        ),
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path(
            torch_candidate,
            *actual_zero_strength_args,
        ),
        path="stable_diffusion.sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_zero_strength",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


@pytest.mark.parametrize(
    ("setup", "path", "path_name"),
    (
        (
            _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_zero_width_windowed_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_path,
            "sd3_rectangular_pooled_multi_image_inpaint_controlnet_zero_width_windowed_denoising_loop",
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_disabled_windowed_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_path,
            "sd3_rectangular_pooled_multi_image_inpaint_controlnet_disabled_windowed_denoising_loop",
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_zero_width_windowed_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_path,
            "sd3_rectangular_pooled_triple_image_inpaint_controlnet_zero_width_windowed_denoising_loop",
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_disabled_windowed_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_path,
            "sd3_rectangular_pooled_triple_image_inpaint_controlnet_disabled_windowed_denoising_loop",
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_zero_width_windowed_img2img_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
            "sd3_rectangular_pooled_multi_image_inpaint_controlnet_zero_width_windowed_img2img_denoising_loop",
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_disabled_windowed_img2img_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
            "sd3_rectangular_pooled_multi_image_inpaint_controlnet_disabled_windowed_img2img_denoising_loop",
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_zero_width_windowed_img2img_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
            "sd3_rectangular_pooled_triple_image_inpaint_controlnet_zero_width_windowed_img2img_denoising_loop",
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_disabled_windowed_img2img_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
            "sd3_rectangular_pooled_triple_image_inpaint_controlnet_disabled_windowed_img2img_denoising_loop",
        ),
    ),
)
def test_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_windowed_guidance_edge_paths_match_reference(
    torch_reference,
    torch_candidate,
    setup,
    path,
    path_name,
) -> None:
    expected_args = setup(torch_reference)
    actual_args = setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        path(torch_reference, *expected_args),
        path(torch_candidate, *actual_args),
        path=f"stable_diffusion.{path_name}",
        rtol=3e-2,
        atol=3e-1,
        check_stride=True,
    )


def _assert_stable_diffusion_sd3_path_raises_for_both_modules(
    torch_reference,
    torch_candidate,
    setup,
    path,
    mutate_args,
) -> None:
    expected_args = list(setup(torch_reference))
    actual_args = list(setup(torch_candidate))
    mutate_args(torch_reference, expected_args)
    mutate_args(torch_candidate, actual_args)
    with pytest.raises(ValueError):
        path(torch_reference, *expected_args)
    with pytest.raises(ValueError):
        path(torch_candidate, *actual_args)


def test_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_rejects_malformed_control_tuples(
    torch_reference,
    torch_candidate,
) -> None:
    def remove_one_controlnet(_module, args):
        args[24] = args[24][:-1]

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_path,
        remove_one_controlnet,
    )

    def remove_one_mask(_module, args):
        args[25] = args[25][:-1]

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_path,
        remove_one_mask,
    )

    def make_short_scales(module, args):
        args[34] = module.tensor([0.90, 0.45], dtype=module.float32)

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_path,
        make_short_scales,
    )


def test_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_rejects_malformed_control_tuples(
    torch_reference,
    torch_candidate,
) -> None:
    def remove_one_controlnet(_module, args):
        args[3] = args[3][:-1]

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_path,
        remove_one_controlnet,
    )

    def remove_one_mask(_module, args):
        args[4] = args[4][:-1]

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_path,
        remove_one_mask,
    )


def test_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_rejects_bad_scale_schedules(
    torch_reference,
    torch_candidate,
) -> None:
    def make_rank1_schedule(module, args):
        args[13] = module.tensor([0.90, 0.45, 0.24], dtype=module.float32)

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_path,
        make_rank1_schedule,
    )

    def make_short_width_schedule(module, args):
        args[13] = module.tensor(
            [
                [0.90, 0.45],
                [0.82, 0.50],
                [0.66, 0.42],
                [0.38, 0.25],
                [0.0, 0.0],
            ],
            dtype=module.float32,
        )

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_path,
        make_short_width_schedule,
    )

    def make_short_step_schedule(module, args):
        args[13] = module.tensor(
            [
                [0.90, 0.45, 0.24],
                [0.82, 0.50, 0.31],
                [0.66, 0.42, 0.36],
                [0.0, 0.0, 0.0],
            ],
            dtype=module.float32,
        )

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_path,
        make_short_step_schedule,
    )


def test_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_rejects_bad_guidance_windows(
    torch_reference,
    torch_candidate,
) -> None:
    def make_short_guidance_start(module, args):
        args[14] = module.tensor([0.0, 0.24], dtype=module.float32)

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_path,
        make_short_guidance_start,
    )

    def make_short_guidance_end(module, args):
        args[15] = module.tensor([0.72, 0.88], dtype=module.float32)

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_path,
        make_short_guidance_end,
    )


@pytest.mark.parametrize(
    ("setup", "path", "start_index", "end_index"),
    (
        (
            _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_path,
            15,
            16,
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_path,
            14,
            15,
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
            15,
            16,
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
            14,
            15,
        ),
    ),
)
def test_stable_diffusion_sd3_windowed_controlnet_rejects_invalid_guidance_window_values(
    torch_reference,
    torch_candidate,
    setup,
    path,
    start_index,
    end_index,
) -> None:
    def replace_entry(module, args, index, position, value):
        values = args[index].tolist()
        values[position] = value
        args[index] = module.tensor(values, dtype=module.float32)

    def set_negative_start(module, args):
        replace_entry(module, args, start_index, 0, -0.01)

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        setup,
        path,
        set_negative_start,
    )

    def set_end_above_one(module, args):
        replace_entry(module, args, end_index, 0, 1.01)

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        setup,
        path,
        set_end_above_one,
    )

    def set_start_after_end(module, args):
        replace_entry(module, args, start_index, 0, 0.90)
        replace_entry(module, args, end_index, 0, 0.80)

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        setup,
        path,
        set_start_after_end,
    )

    def set_nonfinite_start(module, args):
        replace_entry(module, args, start_index, 0, float("nan"))

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        setup,
        path,
        set_nonfinite_start,
    )


def test_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_rejects_malformed_control_tuples(
    torch_reference,
    torch_candidate,
) -> None:
    def remove_one_controlnet(_module, args):
        args[3] = args[3][:-1]

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_path,
        remove_one_controlnet,
    )

    def make_rank1_schedule(module, args):
        args[13] = module.tensor([0.90, 0.45, 0.24], dtype=module.float32)

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_path,
        make_rank1_schedule,
    )


@pytest.mark.parametrize(
    ("setup", "path", "sigmas_index", "strength_index", "steps_index"),
    (
        (
            _case_stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_path,
            6,
            7,
            8,
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_path,
            8,
            9,
            10,
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_path,
            10,
            11,
            12,
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
            10,
            11,
            12,
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_path,
            9,
            10,
            11,
        ),
        (
            _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup,
            _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
            9,
            10,
            11,
        ),
    ),
)
def test_stable_diffusion_sd3_img2img_paths_reject_invalid_start_parameters(
    torch_reference,
    torch_candidate,
    setup,
    path,
    sigmas_index,
    strength_index,
    steps_index,
) -> None:
    for bad_strength in (-0.01, 1.01):
        def set_bad_strength(_module, args, bad_strength=bad_strength):
            args[strength_index] = bad_strength

        _assert_stable_diffusion_sd3_path_raises_for_both_modules(
            torch_reference,
            torch_candidate,
            setup,
            path,
            set_bad_strength,
        )

    def set_zero_steps(_module, args):
        args[steps_index] = 0

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        setup,
        path,
        set_zero_steps,
    )

    def remove_terminal_sigma(module, args):
        args[sigmas_index] = module.tensor(args[sigmas_index].tolist()[:-1], dtype=module.float32)

    _assert_stable_diffusion_sd3_path_raises_for_both_modules(
        torch_reference,
        torch_candidate,
        setup,
        path,
        remove_terminal_sigma,
    )


def test_stable_diffusion_attention_backend_selectors_match_reference(torch_reference, torch_candidate) -> None:
    def backend_state(module):
        return (
            module.backends.cuda.flash_sdp_enabled(),
            module.backends.cuda.math_sdp_enabled(),
            module.backends.cuda.mem_efficient_sdp_enabled(),
            module.backends.cuda.cudnn_sdp_enabled(),
        )

    def backend_path(module):
        query = module.arange(0, 1 * 2 * 3 * 4, dtype=module.float32).reshape(1, 2, 3, 4) / 16.0
        key = module.flip(query, (-1,))
        value = module.arange(0, 1 * 2 * 3 * 2, dtype=module.float32).reshape(1, 2, 3, 2) / 8.0
        params = module.backends.cuda.SDPAParams(query, key, value, None, 0.0, False, False)
        before = backend_state(module)
        with module.nn.attention.sdpa_kernel(module.nn.attention.SDPBackend.MATH):
            math_only = backend_state(module)
            attended = module.nn.functional.scaled_dot_product_attention(
                params.query,
                params.key,
                params.value,
                params.attn_mask,
                dropout_p=params.dropout,
                is_causal=params.is_causal,
                enable_gqa=params.enable_gqa,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with module.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
                enable_cudnn=False,
            ):
                flash_only = backend_state(module)
        after = backend_state(module)
        can_flash = module.backends.cuda.can_use_flash_attention(params)
        can_efficient = module.nn.attention.can_use_efficient_attention(params)
        return before, math_only, flash_only, after, can_flash, can_efficient, attended

    assert_values_compatible(
        torch_reference,
        backend_path(torch_reference),
        backend_path(torch_candidate),
        path="stable_diffusion.attention_backend_selectors",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_stable_diffusion_attention_mask_index_put_path_matches_reference(torch_reference, torch_candidate) -> None:
    def attention_path(module):
        query = module.arange(0, 1 * 2 * 4 * 4, dtype=module.float32).reshape(1, 2, 4, 4) / 16.0
        key = module.flip(query, (-1,))
        value = module.arange(0, 1 * 2 * 4 * 3, dtype=module.float32).reshape(1, 2, 4, 3) / 12.0
        rows = module.tensor([0, 1, 2, 3], dtype=module.long)
        cols = module.tensor([3, 2, 1, 0], dtype=module.long)
        batch_indices = module.zeros((4,), dtype=module.long)
        head_indices = module.zeros((4,), dtype=module.long)
        mask_values = module.full((4,), -10000.0, dtype=module.float32)
        mask = module.zeros((1, 1, 4, 4), dtype=module.float32)
        mask = module.index_put(mask, (batch_indices, head_indices, rows, cols), mask_values)
        return module.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)

    assert_values_compatible(
        torch_reference,
        attention_path(torch_reference),
        attention_path(torch_candidate),
        path="stable_diffusion.attention_mask_index_put",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_stable_diffusion_attention_bool_mask_index_put_path_matches_reference(torch_reference, torch_candidate) -> None:
    def attention_path(module):
        query = module.arange(0, 1 * 2 * 4 * 4, dtype=module.float32).reshape(1, 2, 4, 4) / 16.0
        key = module.flip(query, (-1,))
        value = module.arange(0, 1 * 2 * 4 * 3, dtype=module.float32).reshape(1, 2, 4, 3) / 12.0
        rows = module.tensor([0, 1, 2, 3], dtype=module.long)
        cols = module.tensor([3, 2, 1, 0], dtype=module.long)
        batch_indices = module.zeros((4,), dtype=module.long)
        head_indices = module.zeros((4,), dtype=module.long)
        mask_values = module.zeros((4,), dtype=module.bool)
        mask = module.ones((1, 1, 4, 4), dtype=module.bool)
        mask = module.index_put(mask, (batch_indices, head_indices, rows, cols), mask_values)
        return module.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)

    assert_values_compatible(
        torch_reference,
        attention_path(torch_reference),
        attention_path(torch_candidate),
        path="stable_diffusion.attention_bool_mask_index_put",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_stable_diffusion_unet_resblock_attention_path_matches_reference(torch_reference, torch_candidate) -> None:
    def unet_path(module):
        sample = module.arange(0, 1 * 4 * 4 * 4, dtype=module.float32).reshape(1, 4, 4, 4) / 32.0
        norm_weight = module.tensor([0.75, 1.0, 1.25, 1.5], dtype=module.float32)
        norm_bias = module.tensor([-0.1, 0.0, 0.1, 0.2], dtype=module.float32)
        hidden = module.nn.functional.group_norm(sample, 2, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)

        conv_weight = module.arange(0, 4 * 4 * 3 * 3, dtype=module.float32).reshape(4, 4, 3, 3) / 1024.0
        conv_bias = module.tensor([-0.05, 0.0, 0.05, 0.1], dtype=module.float32)
        hidden = module.nn.functional.conv2d(hidden, conv_weight, conv_bias, padding=1)

        timestep = module.arange(0, 8, dtype=module.float32).reshape(1, 8) / 16.0
        temb_weight = module.arange(0, 8 * 4, dtype=module.float32).reshape(8, 4) / 512.0
        temb_bias = module.tensor([0.01, -0.02, 0.03, -0.04], dtype=module.float32)
        temb = module.nn.functional.silu(module.matmul(timestep, temb_weight) + temb_bias).reshape(1, 4, 1, 1)
        hidden = hidden + temb
        hidden = module.nn.functional.interpolate(hidden, scale_factor=2.0, mode="nearest")

        tokens = hidden.permute(0, 2, 3, 1).reshape(1, 64, 4)
        q_weight = module.eye(4, dtype=module.float32)
        k_weight = module.flip(q_weight, (0,))
        v_weight = module.flip(q_weight, (1,))
        query = module.matmul(tokens, q_weight).reshape(1, 64, 2, 2).transpose(1, 2)
        key = module.matmul(tokens, k_weight).reshape(1, 64, 2, 2).transpose(1, 2)
        value = module.matmul(tokens, v_weight).reshape(1, 64, 2, 2).transpose(1, 2)
        attended = module.nn.functional.scaled_dot_product_attention(query, key, value)
        return attended.transpose(1, 2).reshape(1, 8, 8, 4).permute(0, 3, 1, 2)

    assert_values_compatible(
        torch_reference,
        unet_path(torch_reference),
        unet_path(torch_candidate),
        path="stable_diffusion.unet_resblock_attention",
        rtol=1e-4,
        atol=1e-4,
        check_stride=False,
    )


def test_stable_diffusion_vae_encode_decode_path_matches_reference(torch_reference, torch_candidate) -> None:
    def vae_path(module):
        sample = module.arange(0, 1 * 3 * 8 * 8, dtype=module.float32).reshape(1, 3, 8, 8) / 64.0 - 1.0
        enc_weight = module.arange(0, 6 * 3 * 3 * 3, dtype=module.float32).reshape(6, 3, 3, 3) / 512.0
        hidden = module.nn.functional.conv2d(sample, enc_weight, None, padding=1)
        hidden = module.nn.functional.group_norm(hidden, 3, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)

        hidden = module.nn.functional.pad(hidden, (0, 1, 0, 1), mode="constant", value=0.0)
        down_weight = module.arange(0, 8 * 6 * 3 * 3, dtype=module.float32).reshape(8, 6, 3, 3) / 1024.0
        hidden = module.nn.functional.conv2d(hidden, down_weight, None, stride=2)

        quant_weight = module.arange(0, 8 * 8 * 1 * 1, dtype=module.float32).reshape(8, 8, 1, 1) / 2048.0
        moments = module.nn.functional.conv2d(hidden, quant_weight, None)
        mean, logvar = moments.chunk(2, dim=1)
        logvar = module.clamp(logvar, min=-30.0, max=20.0)
        std = module.exp(0.5 * logvar)
        latent = mean + std * 0.0

        dec_weight = module.arange(0, 4 * 4 * 3 * 3, dtype=module.float32).reshape(4, 4, 3, 3) / 1024.0
        decoded = module.nn.functional.conv_transpose2d(latent, dec_weight, None, stride=2, padding=1, output_padding=1)
        decoded = module.nn.functional.group_norm(decoded, 2, eps=1e-5)
        decoded = module.nn.functional.silu(decoded)
        out_weight = module.arange(0, 3 * 4 * 3 * 3, dtype=module.float32).reshape(3, 4, 3, 3) / 512.0
        return module.nn.functional.conv2d(decoded, out_weight, None, padding=1).clamp(-1.0, 1.0)

    assert_values_compatible(
        torch_reference,
        vae_path(torch_reference),
        vae_path(torch_candidate),
        path="stable_diffusion.vae_encode_decode",
        rtol=1e-4,
        atol=1e-4,
        check_stride=False,
    )


def test_stable_diffusion_half_channels_last_vae_posterior_sample_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_half_channels_last_vae_posterior_sample_setup(torch_reference)
    actual_args = _case_stable_diffusion_half_channels_last_vae_posterior_sample_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_half_channels_last_vae_posterior_sample_path(torch_reference, *expected_args),
        _stable_diffusion_half_channels_last_vae_posterior_sample_path(torch_candidate, *actual_args),
        path="stable_diffusion.half_channels_last_vae_posterior_sample",
        rtol=4e-3,
        atol=4e-3,
    )


def test_stable_diffusion_half_vae_encode_down_block_path_matches_reference(torch_reference, torch_candidate) -> None:
    def encode_path(module):
        sample_base = module.arange(0, 1 * 3 * 16 * 16, dtype=module.float32).reshape(1, 3, 16, 16)
        sample = (module.sin(sample_base * 0.017) * 0.5).to(dtype=module.float16)
        enc_weight = (module.sin(module.arange(0, 8 * 3 * 3 * 3, dtype=module.float32).reshape(8, 3, 3, 3) * 0.013) * 0.05).to(
            dtype=module.float16
        )
        res1_weight = (module.cos(module.arange(0, 8 * 8 * 3 * 3, dtype=module.float32).reshape(8, 8, 3, 3) * 0.011) * 0.04).to(
            dtype=module.float16
        )
        res2_weight = (module.sin(module.arange(0, 8 * 8 * 3 * 3, dtype=module.float32).reshape(8, 8, 3, 3) * 0.007) * 0.04).to(
            dtype=module.float16
        )
        down_weight = (module.cos(module.arange(0, 8 * 8 * 3 * 3, dtype=module.float32).reshape(8, 8, 3, 3) * 0.009) * 0.05).to(
            dtype=module.float16
        )
        quant_weight = (module.sin(module.arange(0, 8 * 8, dtype=module.float32).reshape(8, 8, 1, 1) * 0.019) * 0.03).to(
            dtype=module.float16
        )
        norm_weight = module.tensor([0.75 + index * 0.05 for index in range(8)], dtype=module.float16)
        norm_bias = module.tensor([-0.1 + index * 0.025 for index in range(8)], dtype=module.float16)

        hidden = module.nn.functional.conv2d(sample, enc_weight, None, padding=1)
        residual = hidden
        hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)
        hidden = module.nn.functional.conv2d(hidden, res1_weight, None, padding=1)
        hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)
        hidden = module.nn.functional.conv2d(hidden, res2_weight, None, padding=1)
        hidden = hidden + residual
        hidden = module.nn.functional.pad(hidden, (0, 1, 0, 1), mode="constant", value=0.0)
        hidden = module.nn.functional.conv2d(hidden, down_weight, None, stride=2)
        moments = module.nn.functional.conv2d(hidden, quant_weight, None)
        mean, logvar = moments.chunk(2, dim=1)
        return mean + module.exp(0.5 * module.clamp(logvar, min=-30.0, max=20.0)) * 0.0

    assert_values_compatible(
        torch_reference,
        encode_path(torch_reference),
        encode_path(torch_candidate),
        path="stable_diffusion.half_vae_encode_down_block",
        rtol=3e-3,
        atol=3e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_vae_attention_block_path_matches_reference(torch_reference, torch_candidate) -> None:
    def attention_block(module):
        hidden_base = module.arange(0, 1 * 16 * 8 * 8, dtype=module.float32).reshape(1, 16, 8, 8)
        hidden = (module.sin(hidden_base * 0.017) * 0.5).to(dtype=module.float16)
        norm_weight = module.tensor([0.75 + index * 0.025 for index in range(16)], dtype=module.float16)
        norm_bias = module.tensor([-0.08 + index * 0.01 for index in range(16)], dtype=module.float16)
        base_weight = module.arange(0, 16 * 16, dtype=module.float32).reshape(16, 16, 1, 1)
        q_weight = (module.sin(base_weight * 0.011) * 0.04).to(dtype=module.float16)
        k_weight = (module.cos(base_weight * 0.013) * 0.04).to(dtype=module.float16)
        v_weight = (module.sin(base_weight * 0.017 + 0.25) * 0.04).to(dtype=module.float16)
        proj_weight = (module.cos(base_weight * 0.019 + 0.5) * 0.03).to(dtype=module.float16)
        bias = (module.arange(0, 16, dtype=module.float32) / 256.0 - 0.03).to(dtype=module.float16)

        normed = module.nn.functional.group_norm(hidden, 4, norm_weight, norm_bias, eps=1e-5)
        query = module.nn.functional.conv2d(normed, q_weight, bias)
        key = module.nn.functional.conv2d(normed, k_weight, bias)
        value = module.nn.functional.conv2d(normed, v_weight, bias)
        batch, channels, height, width = query.shape
        heads = 4
        head_dim = channels // heads
        query = query.reshape(batch, heads, head_dim, height * width).transpose(2, 3)
        key = key.reshape(batch, heads, head_dim, height * width).transpose(2, 3)
        value = value.reshape(batch, heads, head_dim, height * width).transpose(2, 3)
        attended = module.nn.functional.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(2, 3).reshape(batch, channels, height, width)
        return hidden + module.nn.functional.conv2d(attended, proj_weight, bias)

    assert_values_compatible(
        torch_reference,
        attention_block(torch_reference),
        attention_block(torch_candidate),
        path="stable_diffusion.half_vae_attention_block",
        rtol=4e-3,
        atol=4e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_video_conv3d_block_path_matches_reference(torch_reference, torch_candidate) -> None:
    def video_block(module):
        channels = 4
        hidden_base = module.arange(0, 1 * channels * 4 * 8 * 8, dtype=module.float32).reshape(1, channels, 4, 8, 8)
        hidden = (module.sin(hidden_base * 0.017) * 0.5).to(dtype=module.float16)
        weight1_base = module.arange(0, channels * channels * 3 * 3 * 3, dtype=module.float32).reshape(
            channels, channels, 3, 3, 3
        )
        weight2_base = module.arange(0, channels * channels * 3 * 3 * 3, dtype=module.float32).reshape(
            channels, channels, 3, 3, 3
        )
        weight1 = (module.cos(weight1_base * 0.011) * 0.04).to(dtype=module.float16)
        weight2 = (module.sin(weight2_base * 0.007) * 0.035).to(dtype=module.float16)
        norm_weight = module.tensor([0.75 + index * 0.05 for index in range(channels)], dtype=module.float16)
        norm_bias = module.tensor([-0.08 + index * 0.025 for index in range(channels)], dtype=module.float16)

        residual = hidden
        hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)
        hidden = module.nn.functional.conv3d(hidden, weight1, None, padding=1)
        hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)
        hidden = module.nn.functional.conv3d(hidden, weight2, None, padding=1)
        return hidden + residual

    assert_values_compatible(
        torch_reference,
        video_block(torch_reference),
        video_block(torch_candidate),
        path="stable_diffusion.half_video_conv3d_block",
        rtol=5e-3,
        atol=5e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_channels_last_3d_video_conv3d_block_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def video_block(module):
        channels = 4
        hidden_base = module.arange(0, 1 * channels * 4 * 8 * 8, dtype=module.float32).reshape(1, channels, 4, 8, 8)
        hidden = (module.sin(hidden_base * 0.017) * 0.5).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last_3d)
        weight1_base = module.arange(0, channels * channels * 3 * 3 * 3, dtype=module.float32).reshape(
            channels, channels, 3, 3, 3
        )
        weight2_base = module.arange(0, channels * channels * 3 * 3 * 3, dtype=module.float32).reshape(
            channels, channels, 3, 3, 3
        )
        weight1 = (module.cos(weight1_base * 0.011) * 0.04).to(dtype=module.float16)
        weight2 = (module.sin(weight2_base * 0.007) * 0.035).to(dtype=module.float16)
        norm_weight = module.tensor([0.75 + index * 0.05 for index in range(channels)], dtype=module.float16)
        norm_bias = module.tensor([-0.08 + index * 0.025 for index in range(channels)], dtype=module.float16)

        residual = hidden
        hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)
        hidden = module.nn.functional.conv3d(hidden, weight1, None, padding=1)
        hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)
        hidden = module.nn.functional.conv3d(hidden, weight2, None, padding=1)
        return hidden + residual

    assert_values_compatible(
        torch_reference,
        video_block(torch_reference),
        video_block(torch_candidate),
        path="stable_diffusion.half_channels_last_3d_video_conv3d_block",
        rtol=5e-3,
        atol=5e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_channels_last_3d_skip_cat_conv3d_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def video_block(module):
        channels = 4
        hidden_base = module.arange(0, 1 * channels * 4 * 8 * 8, dtype=module.float32).reshape(
            1,
            channels,
            4,
            8,
            8,
        )
        hidden = (module.sin(hidden_base * 0.017) * 0.5).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last_3d)
        skip = module.flip(hidden, (-1,)).contiguous(memory_format=module.channels_last_3d)
        merged = module.cat([hidden, skip], dim=1)
        weight_base = module.arange(0, channels * (channels * 2) * 3 * 3 * 3, dtype=module.float32).reshape(
            channels,
            channels * 2,
            3,
            3,
            3,
        )
        weight = (module.cos(weight_base * 0.011) * 0.04).to(dtype=module.float16)
        bias = (module.arange(0, channels, dtype=module.float32) / 256.0 - 0.03).to(dtype=module.float16)
        return merged, module.nn.functional.conv3d(merged, weight, bias, padding=1)

    assert_values_compatible(
        torch_reference,
        video_block(torch_reference),
        video_block(torch_candidate),
        path="stable_diffusion.half_channels_last_3d_skip_cat_conv3d",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_bicubic_decode_postprocess_path_matches_reference(torch_reference, torch_candidate) -> None:
    def postprocess_path(module):
        decoded = module.arange(0, 1 * 3 * 8 * 8, dtype=module.float32).reshape(1, 3, 8, 8) / 64.0 - 1.0
        resized = module.nn.functional.interpolate(decoded, size=(16, 16), mode="bicubic", align_corners=False)
        image = (resized / 2.0 + 0.5).clamp(0.0, 1.0)
        return image.permute(0, 2, 3, 1).contiguous()

    assert_values_compatible(
        torch_reference,
        postprocess_path(torch_reference),
        postprocess_path(torch_candidate),
        path="stable_diffusion.bicubic_decode_postprocess",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_stable_diffusion_area_image_preprocess_path_matches_reference(torch_reference, torch_candidate) -> None:
    def preprocess_path(module):
        image = module.arange(0, 1 * 3 * 16 * 16, dtype=module.float32).reshape(1, 3, 16, 16) / 255.0
        downsampled = module.nn.functional.interpolate(image, size=(8, 8), mode="area")
        return downsampled * 2.0 - 1.0

    assert_values_compatible(
        torch_reference,
        preprocess_path(torch_reference),
        preprocess_path(torch_candidate),
        path="stable_diffusion.area_image_preprocess",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_stable_diffusion_constant_pad_crop_path_matches_reference(torch_reference, torch_candidate) -> None:
    def crop_path(module):
        latent = module.arange(0, 1 * 4 * 5 * 7, dtype=module.float32).reshape(1, 4, 5, 7) / 32.0 - 1.0
        cropped = module.nn.functional.pad(latent, (-1, 2, -2, 1), mode="constant", value=0.25)
        weight = module.arange(0, 4 * 4 * 3 * 3, dtype=module.float32).reshape(4, 4, 3, 3) / 256.0
        hidden = module.nn.functional.conv2d(cropped, weight, None, padding=1)
        return module.nn.functional.interpolate(hidden, size=(4, 6), mode="bilinear", align_corners=False)

    assert_values_compatible(
        torch_reference,
        crop_path(torch_reference),
        crop_path(torch_candidate),
        path="stable_diffusion.constant_pad_crop",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_stable_diffusion_nearest_exact_mask_preprocess_path_matches_reference(torch_reference, torch_candidate) -> None:
    def mask_path(module):
        mask = module.arange(0, 1 * 1 * 5 * 7, dtype=module.float32).reshape(1, 1, 5, 7) % 2.0
        resized = module.nn.functional.interpolate(mask, size=(3, 4), mode="nearest-exact")
        return resized > 0.5

    assert_values_compatible(
        torch_reference,
        mask_path(torch_reference),
        mask_path(torch_candidate),
        path="stable_diffusion.nearest_exact_mask_preprocess",
        rtol=1e-6,
        atol=1e-6,
        check_stride=False,
    )


def test_stable_diffusion_inpaint_preprocess_bundle_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_inpaint_preprocess_bundle_setup(torch_reference)
    actual_args = _case_stable_diffusion_inpaint_preprocess_bundle_setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        _stable_diffusion_inpaint_preprocess_bundle_path(torch_reference, *expected_args),
        _stable_diffusion_inpaint_preprocess_bundle_path(torch_candidate, *actual_args),
        path="stable_diffusion.inpaint_preprocess_bundle",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_affine_grid_sample_mask_path_matches_reference(torch_reference, torch_candidate) -> None:
    def spatial_path(module):
        sample = module.arange(0, 1 * 4 * 4 * 4, dtype=module.float32).reshape(1, 4, 4, 4) / 32.0 - 1.0
        theta = module.tensor([[[0.9, 0.0, 0.2], [0.1, 0.85, -0.1]]], dtype=module.float32)
        grid = module.nn.functional.affine_grid(theta, sample.shape, align_corners=False)
        warped = module.nn.functional.grid_sample(
            sample,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        mask = module.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=module.float32)
        mask = module.nn.functional.interpolate(mask, size=(4, 4), mode="nearest-exact") > 0.5
        return module.where(mask, warped, sample)

    assert_values_compatible(
        torch_reference,
        spatial_path(torch_reference),
        spatial_path(torch_candidate),
        path="stable_diffusion.affine_grid_sample_mask",
        rtol=1e-5,
        atol=1e-5,
        check_stride=False,
    )


def test_stable_diffusion_half_interpolate_upsample_path_matches_reference(torch_reference, torch_candidate) -> None:
    def upsample_path(module):
        hidden = module.arange(0, 1 * 4 * 4 * 4, dtype=module.float16).reshape(1, 4, 4, 4) / 32.0
        nearest = module.nn.functional.interpolate(hidden, scale_factor=2.0, mode="nearest")
        smooth = module.nn.functional.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        return nearest, smooth

    assert_values_compatible(
        torch_reference,
        upsample_path(torch_reference),
        upsample_path(torch_candidate),
        path="stable_diffusion.half_interpolate_upsample",
        rtol=2e-3,
        atol=2e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_vae_nearest_upsample_path_matches_reference(torch_reference, torch_candidate) -> None:
    def upsample_path(module):
        hidden_base = module.arange(0, 1 * 8 * 16 * 16, dtype=module.float32).reshape(1, 8, 16, 16)
        hidden = (module.sin(hidden_base * 0.017) * 0.5).to(dtype=module.float16)
        return module.nn.functional.interpolate(hidden, scale_factor=2.0, mode="nearest")

    assert_values_compatible(
        torch_reference,
        upsample_path(torch_reference),
        upsample_path(torch_candidate),
        path="stable_diffusion.half_vae_nearest_upsample",
        rtol=0.0,
        atol=0.0,
        check_stride=False,
    )


def test_stable_diffusion_half_vae_decode_path_matches_reference(torch_reference, torch_candidate) -> None:
    def decode_path(module):
        latent = module.arange(0, 1 * 4 * 4 * 4, dtype=module.float16).reshape(1, 4, 4, 4) / 64.0 - 0.5
        up_weight = module.arange(0, 4 * 8 * 3 * 3, dtype=module.float16).reshape(4, 8, 3, 3) / 512.0
        hidden = module.nn.functional.conv_transpose2d(latent, up_weight, None, stride=2, padding=1, output_padding=1)
        norm_weight = module.tensor([0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45], dtype=module.float16)
        norm_bias = module.tensor([-0.2, -0.1, 0.0, 0.1, 0.2, -0.15, 0.05, 0.15], dtype=module.float16)
        hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)
        out_weight = module.arange(0, 3 * 8 * 3 * 3, dtype=module.float16).reshape(3, 8, 3, 3) / 512.0
        return module.nn.functional.conv2d(hidden, out_weight, None, padding=1).clamp(-1.0, 1.0)

    assert_values_compatible(
        torch_reference,
        decode_path(torch_reference),
        decode_path(torch_candidate),
        path="stable_diffusion.half_vae_decode",
        rtol=3e-3,
        atol=3e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_channels_last_vae_transpose_decode_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def decode_path(module):
        latent = module.arange(0, 1 * 4 * 8 * 8, dtype=module.float32).reshape(1, 4, 8, 8) / 64.0 - 0.5
        latent = latent.to(dtype=module.float16).contiguous(memory_format=module.channels_last)
        up_weight = module.arange(0, 4 * 8 * 3 * 3, dtype=module.float32).reshape(4, 8, 3, 3) / 512.0
        up_weight = up_weight.to(dtype=module.float16)
        norm_weight = module.tensor([0.75 + index * 0.05 for index in range(8)], dtype=module.float16)
        norm_bias = module.tensor([-0.1 + index * 0.025 for index in range(8)], dtype=module.float16)
        out_weight = module.arange(0, 3 * 8 * 3 * 3, dtype=module.float32).reshape(3, 8, 3, 3) / 512.0
        out_weight = out_weight.to(dtype=module.float16)

        hidden = module.nn.functional.conv_transpose2d(
            latent,
            up_weight,
            None,
            stride=2,
            padding=1,
            output_padding=1,
        )
        hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)
        return hidden, module.nn.functional.conv2d(hidden, out_weight, None, padding=1).clamp(-1.0, 1.0)

    assert_values_compatible(
        torch_reference,
        decode_path(torch_reference),
        decode_path(torch_candidate),
        path="stable_diffusion.half_channels_last_vae_transpose_decode",
        rtol=3e-3,
        atol=3e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_vae_nearest_block_path_matches_reference(torch_reference, torch_candidate) -> None:
    def decode_block(module):
        latent = module.arange(0, 1 * 4 * 4 * 4, dtype=module.float32).reshape(1, 4, 4, 4) / 64.0 - 0.5
        latent = latent.to(dtype=module.float16)
        in_weight = module.arange(0, 8 * 4 * 3 * 3, dtype=module.float32).reshape(8, 4, 3, 3) / 512.0
        res1_weight = module.arange(0, 8 * 8 * 3 * 3, dtype=module.float32).reshape(8, 8, 3, 3) / 768.0
        res2_weight = module.arange(0, 8 * 8 * 3 * 3, dtype=module.float32).reshape(8, 8, 3, 3) / 896.0
        up_weight = module.arange(0, 8 * 8 * 3 * 3, dtype=module.float32).reshape(8, 8, 3, 3) / 1024.0
        out_weight = module.arange(0, 3 * 8 * 3 * 3, dtype=module.float32).reshape(3, 8, 3, 3) / 512.0
        norm_weight = module.tensor([0.75 + index * 0.05 for index in range(8)], dtype=module.float16)
        norm_bias = module.tensor([-0.1 + index * 0.025 for index in range(8)], dtype=module.float16)

        hidden = module.nn.functional.conv2d(latent, in_weight.to(dtype=module.float16), None, padding=1)
        residual = hidden
        hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)
        hidden = module.nn.functional.conv2d(hidden, res1_weight.to(dtype=module.float16), None, padding=1)
        hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)
        hidden = module.nn.functional.conv2d(hidden, res2_weight.to(dtype=module.float16), None, padding=1)
        hidden = hidden + residual
        hidden = module.nn.functional.interpolate(hidden, scale_factor=2.0, mode="nearest")
        hidden = module.nn.functional.conv2d(hidden, up_weight.to(dtype=module.float16), None, padding=1)
        hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)
        return module.nn.functional.conv2d(hidden, out_weight.to(dtype=module.float16), None, padding=1).clamp(-1.0, 1.0)

    assert_values_compatible(
        torch_reference,
        decode_block(torch_reference),
        decode_block(torch_candidate),
        path="stable_diffusion.half_vae_nearest_block",
        rtol=3e-3,
        atol=3e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_channels_last_vae_decode_block_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    def decode_block(module):
        channels, spatial = 16, 16
        hidden_base = module.arange(0, 1 * channels * spatial * spatial, dtype=module.float32).reshape(
            1,
            channels,
            spatial,
            spatial,
        )
        hidden = (module.sin(hidden_base * 0.011) * 0.25).to(dtype=module.float16)
        hidden = hidden.contiguous(memory_format=module.channels_last)

        def make_weight(scale):
            base = module.arange(0, channels * channels * 3 * 3, dtype=module.float32).reshape(
                channels,
                channels,
                3,
                3,
            )
            return (module.sin(base * scale) * 0.035).to(dtype=module.float16)

        out_base = module.arange(0, 3 * channels * 3 * 3, dtype=module.float32).reshape(3, channels, 3, 3)
        out_weight = (module.cos(out_base * 0.029) * 0.03).to(dtype=module.float16)
        norm_weight = (0.75 + (module.arange(0, channels, dtype=module.float32) % 257) / 512.0).to(
            dtype=module.float16
        )
        norm_bias = (((module.arange(0, channels, dtype=module.float32) % 127) - 63.0) / 2048.0).to(
            dtype=module.float16
        )

        residual = hidden
        hidden = module.nn.functional.group_norm(hidden, 4, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)
        hidden = module.nn.functional.conv2d(hidden, make_weight(0.013), None, padding=1)
        hidden = module.nn.functional.group_norm(hidden, 4, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)
        hidden = module.nn.functional.conv2d(hidden, make_weight(0.017), None, padding=1)
        hidden = hidden + residual
        hidden = module.nn.functional.interpolate(hidden, scale_factor=2.0, mode="nearest")
        hidden = module.nn.functional.conv2d(hidden, make_weight(0.019), None, padding=1)
        residual = hidden
        hidden = module.nn.functional.group_norm(hidden, 4, norm_weight, norm_bias, eps=1e-5)
        hidden = module.nn.functional.silu(hidden)
        hidden = module.nn.functional.conv2d(hidden, make_weight(0.023), None, padding=1)
        hidden = hidden + residual
        return module.nn.functional.conv2d(hidden, out_weight, None, padding=1).clamp(-1.0, 1.0)

    assert_values_compatible(
        torch_reference,
        decode_block(torch_reference),
        decode_block(torch_candidate),
        path="stable_diffusion.half_channels_last_vae_decode_block",
        rtol=5e-3,
        atol=5e-3,
    )


def test_stable_diffusion_half_attention_projection_path_matches_reference(torch_reference, torch_candidate) -> None:
    def attention_projection(module):
        hidden_base = module.arange(0, 2 * 16 * 64, dtype=module.float32).reshape(2, 16, 64)
        hidden = (module.sin(hidden_base * 0.017) * 0.25).to(dtype=module.float16)
        q_weight = (module.arange(0, 64 * 64, dtype=module.float32).reshape(64, 64) / 4096.0 - 0.5).to(
            dtype=module.float16
        )
        k_weight = module.flip(q_weight, (0,))
        v_weight = (module.cos(module.arange(0, 64 * 64, dtype=module.float32).reshape(64, 64) * 0.011) / 128.0).to(
            dtype=module.float16
        )
        out_weight = module.flip(v_weight, (1,))
        bias = (module.arange(0, 64, dtype=module.float32) / 2048.0 - 0.015).to(dtype=module.float16)

        query = module.nn.functional.linear(hidden, q_weight, bias).reshape(2, 16, 4, 16).transpose(1, 2)
        key = module.nn.functional.linear(hidden, k_weight, bias).reshape(2, 16, 4, 16).transpose(1, 2)
        value = module.nn.functional.linear(hidden, v_weight, bias).reshape(2, 16, 4, 16).transpose(1, 2)
        attended = module.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).reshape(2, 16, 64)
        return module.nn.functional.linear(attended, out_weight, bias)

    assert_values_compatible(
        torch_reference,
        attention_projection(torch_reference),
        attention_projection(torch_candidate),
        path="stable_diffusion.half_attention_projection",
        rtol=3e-3,
        atol=3e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_masked_attention_path_matches_reference(torch_reference, torch_candidate) -> None:
    def attention_path(module):
        base = module.arange(0, 1 * 2 * 16 * 16, dtype=module.float32).reshape(1, 2, 16, 16)
        query = (module.sin(base * 0.017) * 0.1).to(dtype=module.float16)
        key = (module.cos(base * 0.019) * 0.1).to(dtype=module.float16)
        value = (module.sin(base * 0.013) * 0.1).to(dtype=module.float16)
        rows = module.arange(0, 16, dtype=module.long)
        cols = module.flip(rows, (0,))
        batch_indices = module.zeros((16,), dtype=module.long)
        head_indices = module.zeros((16,), dtype=module.long)

        additive_mask = module.zeros((1, 1, 16, 16), dtype=module.float32)
        additive_mask = module.index_put(
            additive_mask,
            (batch_indices, head_indices, rows, cols),
            module.full((16,), -10000.0, dtype=module.float32),
        )
        half_additive_mask = module.zeros((1, 1, 16, 16), dtype=module.float16)
        half_additive_mask = module.index_put(
            half_additive_mask,
            (batch_indices, head_indices, rows, cols),
            module.full((16,), -10000.0, dtype=module.float16),
        )
        bool_mask = module.ones((1, 1, 16, 16), dtype=module.bool)
        bool_mask = module.index_put(
            bool_mask,
            (batch_indices, head_indices, rows, cols),
            module.zeros((16,), dtype=module.bool),
        )
        additive = module.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=additive_mask)
        half_additive = module.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=half_additive_mask,
        )
        boolean = module.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=bool_mask)
        return additive, half_additive, boolean

    assert_values_compatible(
        torch_reference,
        attention_path(torch_reference),
        attention_path(torch_candidate),
        path="stable_diffusion.half_masked_attention",
        rtol=1e-3,
        atol=1e-3,
        check_stride=False,
    )


def test_stable_diffusion_module_pipeline_path_matches_reference(torch_reference, torch_candidate) -> None:
    def matrix(rows, cols):
        return [[float(((row * cols + col) % 19) - 9) / 18.0 for col in range(cols)] for row in range(rows)]

    def vector(cols):
        return [float((col % 13) - 6) / 16.0 for col in range(cols)]

    def copy_parameter(module, target, source):
        with module.no_grad():
            target.copy_(source)

    def fill_parameter(module, parameter):
        if parameter.dim() >= 2:
            values = module.tensor(
                matrix(parameter.shape[0], parameter.numel() // parameter.shape[0]),
                dtype=module.float32,
            ).reshape(parameter.shape)
            copy_parameter(module, parameter, values / 4.0)
        else:
            copy_parameter(module, parameter, module.tensor(vector(parameter.numel()), dtype=module.float32))

    def make_text_encoder(module):
        class MiniTextEncoder(module.nn.Module):
            def __init__(self):
                super().__init__()
                self.token = module.nn.Embedding(16, 8)
                self.position = module.nn.Embedding(6, 8)
                self.norm = module.nn.LayerNorm(8)
                self.proj = module.nn.Linear(8, 8)
                copy_parameter(module, self.token.weight, module.tensor(matrix(16, 8), dtype=module.float32))
                copy_parameter(module, self.position.weight, module.tensor(matrix(6, 8), dtype=module.float32))
                copy_parameter(
                    module,
                    self.norm.weight,
                    module.tensor([1.0 + index / 16.0 for index in range(8)], dtype=module.float32),
                )
                copy_parameter(module, self.norm.bias, module.tensor(vector(8), dtype=module.float32))
                copy_parameter(module, self.proj.weight, module.tensor(matrix(8, 8), dtype=module.float32))
                copy_parameter(module, self.proj.bias, module.tensor(vector(8), dtype=module.float32))

            def forward(self, input_ids):
                positions = module.arange(input_ids.shape[1], dtype=module.long).unsqueeze(0).expand(input_ids.shape[0], -1)
                hidden = self.token(input_ids) + self.position(positions)
                return module.nn.functional.gelu(self.proj(self.norm(hidden)))

        return MiniTextEncoder().eval()

    def make_unet(module):
        class MiniUNet(module.nn.Module):
            def __init__(self):
                super().__init__()
                self.in_conv = module.nn.Conv2d(4, 8, 3, padding=1)
                self.norm = module.nn.GroupNorm(2, 8)
                self.time = module.nn.Linear(4, 8)
                self.q = module.nn.Linear(8, 8)
                self.k = module.nn.Linear(8, 8)
                self.v = module.nn.Linear(8, 8)
                self.out = module.nn.Linear(8, 8)
                self.out_norm = module.nn.GroupNorm(2, 8)
                self.out_conv = module.nn.Conv2d(8, 4, 3, padding=1)
                for _, parameter in self.named_parameters():
                    fill_parameter(module, parameter)

            def forward(self, sample, timestep, encoder_hidden):
                hidden = module.nn.functional.silu(self.norm(self.in_conv(sample)))
                timestep_value = timestep.to(dtype=sample.dtype).reshape(1, 1)
                timestep_embedding = module.cat(
                    [
                        timestep_value,
                        timestep_value * timestep_value,
                        module.sin(timestep_value),
                        module.cos(timestep_value),
                    ],
                    dim=1,
                )
                hidden = hidden + self.time(timestep_embedding).reshape(1, 8, 1, 1)
                batch, channels, height, width = hidden.shape
                tokens = hidden.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
                query = self.q(tokens).reshape(batch, height * width, 2, 4).transpose(1, 2)
                key = self.k(encoder_hidden).reshape(batch, encoder_hidden.shape[1], 2, 4).transpose(1, 2)
                value = self.v(encoder_hidden).reshape(batch, encoder_hidden.shape[1], 2, 4).transpose(1, 2)
                attended = module.nn.functional.scaled_dot_product_attention(query, key, value, scale=0.5)
                attended = attended.transpose(1, 2).reshape(batch, height * width, channels)
                tokens = tokens + self.out(attended)
                hidden = tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
                hidden = module.nn.functional.silu(self.out_norm(hidden))
                return self.out_conv(hidden)

        return MiniUNet().eval()

    def make_vae(module):
        class MiniVAE(module.nn.Module):
            def __init__(self):
                super().__init__()
                self.up = module.nn.ConvTranspose2d(4, 8, 3, stride=2, padding=1, output_padding=1)
                self.norm = module.nn.GroupNorm(2, 8)
                self.out = module.nn.Conv2d(8, 3, 3, padding=1)
                for _, parameter in self.named_parameters():
                    fill_parameter(module, parameter)

            def forward(self, latents):
                hidden = module.nn.functional.silu(self.norm(self.up(latents)))
                return self.out(hidden).clamp(-1.0, 1.0)

        return MiniVAE().eval()

    def make_scheduler(module):
        class MiniScheduler(module.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("timesteps", module.tensor([999, 750, 500], dtype=module.long))
                self.register_buffer("sigmas", module.logspace(-1.0, -2.0, 3, dtype=module.float32))

            def step(self, model_output, index, sample):
                sigma = self.sigmas[index].reshape(1, 1, 1, 1)
                next_index = (index + 1).clamp(max=self.sigmas.shape[0] - 1)
                next_sigma = module.where(
                    index + 1 < self.sigmas.shape[0],
                    self.sigmas[next_index],
                    module.zeros((), dtype=sample.dtype),
                )
                return sample + model_output * (next_sigma - sigma.reshape(())).reshape(1, 1, 1, 1)

        return MiniScheduler().eval()

    def pipeline(module):
        text_encoder = make_text_encoder(module)
        unet = make_unet(module)
        vae = make_vae(module)
        scheduler = make_scheduler(module)
        unconditional_ids = module.tensor([[0, 0, 0, 0, 0, 0]], dtype=module.long)
        conditional_ids = module.tensor([[1, 2, 3, 4, 5, 0]], dtype=module.long)
        input_ids = module.cat([unconditional_ids, conditional_ids], dim=0)
        latents = module.arange(0, 1 * 4 * 4 * 4, dtype=module.float32).reshape(1, 4, 4, 4) / 32.0 - 1.0
        with module.inference_mode():
            encoder_hidden = text_encoder(input_ids)
            for index, timestep in enumerate(scheduler.timesteps):
                latent_model_input = module.cat([latents, latents], dim=0)
                noise_pred = unet(latent_model_input, timestep, encoder_hidden)
                uncond, cond = noise_pred.chunk(2, dim=0)
                guided = uncond + (cond - uncond) * 5.5
                latents = scheduler.step(guided, module.tensor(index, dtype=module.long), latents)
            return vae(latents)

    assert_values_compatible(
        torch_reference,
        pipeline(torch_reference),
        pipeline(torch_candidate),
        path="stable_diffusion.module_pipeline",
        rtol=1e-4,
        atol=1e-4,
        check_stride=False,
    )

    def half_pipeline(module):
        text_encoder = make_text_encoder(module).half()
        unet = make_unet(module).half()
        vae = make_vae(module).half()
        scheduler = make_scheduler(module).half()
        with module.no_grad():
            scheduler.timesteps.copy_(module.tensor([9, 7, 5], dtype=module.long))
        unconditional_ids = module.tensor([[0, 0, 0, 0, 0, 0]], dtype=module.long)
        conditional_ids = module.tensor([[1, 2, 3, 4, 5, 0]], dtype=module.long)
        input_ids = module.cat([unconditional_ids, conditional_ids], dim=0)
        latents = module.arange(0, 1 * 4 * 4 * 4, dtype=module.float32).reshape(1, 4, 4, 4) / 32.0 - 1.0
        latents = latents.to(dtype=module.float16)
        with module.inference_mode():
            encoder_hidden = text_encoder(input_ids)
            for index, timestep in enumerate(scheduler.timesteps):
                latent_model_input = module.cat([latents, latents], dim=0)
                noise_pred = unet(latent_model_input, timestep, encoder_hidden)
                uncond, cond = noise_pred.chunk(2, dim=0)
                guided = uncond + (cond - uncond) * 5.5
                latents = scheduler.step(guided, module.tensor(index, dtype=module.long), latents)
            return vae(latents)

    assert_values_compatible(
        torch_reference,
        half_pipeline(torch_reference),
        half_pipeline(torch_candidate),
        path="stable_diffusion.half_module_pipeline",
        rtol=5e-3,
        atol=5e-3,
        check_stride=False,
    )


def test_stable_diffusion_half_channels_last_module_pipeline_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_half_channels_last_module_pipeline_setup(torch_reference)
    actual_args = _case_stable_diffusion_half_channels_last_module_pipeline_setup(torch_candidate)
    assert_values_compatible(
        torch_reference,
        _stable_diffusion_module_pipeline_path(torch_reference, *expected_args),
        _stable_diffusion_module_pipeline_path(torch_candidate, *actual_args),
        path="stable_diffusion.half_channels_last_module_pipeline",
        rtol=5e-3,
        atol=5e-3,
        check_stride=True,
    )


def test_stable_diffusion_half_channels_last_inpaint_module_pipeline_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_half_channels_last_inpaint_module_pipeline_setup(torch_reference)
    actual_args = _case_stable_diffusion_half_channels_last_inpaint_module_pipeline_setup(torch_candidate)
    assert_values_compatible(
        torch_reference,
        _stable_diffusion_inpaint_module_pipeline_path(torch_reference, *expected_args),
        _stable_diffusion_inpaint_module_pipeline_path(torch_candidate, *actual_args),
        path="stable_diffusion.half_channels_last_inpaint_module_pipeline",
        rtol=6e-3,
        atol=6e-3,
        check_stride=True,
    )


def test_stable_diffusion_sdxl_half_channels_last_inpaint_controlnet_module_pipeline_path_matches_reference(
    torch_reference,
    torch_candidate,
) -> None:
    expected_args = _case_stable_diffusion_sdxl_half_channels_last_inpaint_controlnet_module_pipeline_setup(
        torch_reference
    )
    actual_args = _case_stable_diffusion_sdxl_half_channels_last_inpaint_controlnet_module_pipeline_setup(
        torch_candidate
    )
    assert_values_compatible(
        torch_reference,
        _stable_diffusion_sdxl_inpaint_controlnet_module_pipeline_path(torch_reference, *expected_args),
        _stable_diffusion_sdxl_inpaint_controlnet_module_pipeline_path(torch_candidate, *actual_args),
        path="stable_diffusion.sdxl_half_channels_last_inpaint_controlnet_module_pipeline",
        rtol=8e-3,
        atol=8e-3,
        check_stride=True,
    )


def test_tensor_transpose_properties_match_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch_reference.float32)
    actual = torch_candidate.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch_candidate.float32)
    expected_matrix = torch_reference.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch_reference.float32)
    actual_matrix = torch_candidate.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch_candidate.float32)

    assert_values_compatible(torch_reference, expected.T, actual.T, path="tensor.T")
    assert_values_compatible(torch_reference, expected_matrix.H, actual_matrix.H, path="tensor.H")
    assert_values_compatible(torch_reference, expected.mT, actual.mT, path="tensor.mT")
    assert_values_compatible(torch_reference, expected.mH, actual.mH, path="tensor.mH")


def test_generator_random_factories_are_independent_and_reproducible(torch_candidate) -> None:
    base = torch_candidate.zeros((2, 3), dtype=torch_candidate.float32)
    generator = torch_candidate.Generator(device="cpu").manual_seed(77)
    assert generator.device == torch_candidate.device("cpu")
    assert generator.device.type == "cpu"
    assert generator.initial_seed() == 77

    first = torch_candidate.randn((2, 3), generator=generator)
    generator.manual_seed(77)
    second = torch_candidate.randn_like(base, generator=generator)
    assert first.tolist() == second.tolist()

    torch_candidate.manual_seed(1234)
    global_sample = torch_candidate.randn((2, 3))
    generator.manual_seed(77)
    third = torch_candidate.randn((2, 3), generator=generator)
    assert third.tolist() == first.tolist()
    assert global_sample.tolist() != first.tolist()

    global_generator = torch_candidate.manual_seed(4321)
    assert torch_candidate.initial_seed() == 4321
    assert global_generator.initial_seed() == 4321
    global_generator.manual_seed(4322)
    assert torch_candidate.initial_seed() == 4322

    generator.manual_seed(9)
    randint_first = torch_candidate.randint(0, 10, (2, 3), generator=generator)
    generator.manual_seed(9)
    randint_second = torch_candidate.randint(0, 10, (2, 3), generator=generator)
    assert randint_first.tolist() == randint_second.tolist()

    generator.manual_seed(11)
    permutation_first = torch_candidate.randperm(8, generator=generator)
    generator.manual_seed(11)
    permutation_second = torch_candidate.randperm(8, generator=generator)
    assert permutation_first.tolist() == permutation_second.tolist()
    assert sorted(permutation_first.tolist()) == list(range(8))

    global_generator = torch_candidate.manual_seed(55)
    assert global_generator.device.type == "cpu"
    assert global_generator.initial_seed() == 55
    global_first = torch_candidate.randn((2, 3), generator=global_generator)
    torch_candidate.manual_seed(55)
    global_second = torch_candidate.randn((2, 3))
    assert global_first.tolist() == global_second.tolist()


def test_strided_layout_factories_for_diffusion_helpers_match_reference(torch_reference, torch_candidate) -> None:
    tensor = torch_candidate.zeros((1,), dtype=torch_candidate.float32)
    assert tensor.layout is torch_candidate.strided
    for attribute in ("is_cuda", "is_mps", "is_sparse", "is_sparse_csr", "is_mkldnn", "is_quantized", "is_meta", "is_nested"):
        assert getattr(tensor, attribute) is getattr(torch_reference.zeros((1,), dtype=torch_reference.float32), attribute)

    assert_values_compatible(
        torch_reference,
        torch_reference.zeros((2, 3), dtype=torch_reference.float32, layout=torch_reference.strided),
        torch_candidate.zeros((2, 3), dtype=torch_candidate.float32, layout=torch_candidate.strided),
        path="layout.zeros_strided",
    )
    assert_values_compatible(
        torch_reference,
        torch_reference.ones((2, 3), dtype=torch_reference.float32, layout=torch_reference.strided),
        torch_candidate.ones((2, 3), dtype=torch_candidate.float32, layout=torch_candidate.strided),
        path="layout.ones_strided",
    )
    assert_values_compatible(
        torch_reference,
        torch_reference.logspace(-2.0, 0.0, 4, dtype=torch_reference.float32, layout=torch_reference.strided),
        torch_candidate.logspace(-2.0, 0.0, 4, dtype=torch_candidate.float32, layout=torch_candidate.strided),
        path="layout.logspace_strided",
        rtol=1e-6,
        atol=1e-6,
    )

    base = torch_candidate.zeros((2, 3), dtype=torch_candidate.float32, layout=torch_candidate.strided)
    generator = torch_candidate.Generator(device="cpu").manual_seed(909)
    first = torch_candidate.randn((2, 3), dtype=torch_candidate.float32, layout=base.layout, generator=generator)
    generator.manual_seed(909)
    second = torch_candidate.randn_like(base, layout=base.layout, generator=generator)
    assert first.tolist() == second.tolist()

    generator.manual_seed(910)
    first_int = torch_candidate.randint(0, 10, (2, 3), layout=torch_candidate.strided, generator=generator)
    generator.manual_seed(910)
    second_int = torch_candidate.randint(low=0, high=10, size=(2, 3), layout=torch_candidate.strided, generator=generator)
    assert first_int.tolist() == second_int.tolist()
    generator.manual_seed(910)
    like_int = torch_candidate.randint_like(base, 10, layout=torch_candidate.strided, generator=generator)
    assert like_int.shape == base.shape
    assert like_int.dtype == base.dtype
    assert all(0.0 <= value < 10.0 for row in like_int.tolist() for value in row)

    generator.manual_seed(911)
    first_normal = torch_candidate.normal(
        0.0,
        1.0,
        size=(2, 3),
        layout=torch_candidate.strided,
        generator=generator,
    )
    generator.manual_seed(911)
    second_normal = torch_candidate.normal(0.0, 1.0, size=(2, 3), generator=generator)
    assert first_normal.tolist() == second_normal.tolist()


def test_generator_inplace_random_methods_are_independent_and_reproducible(torch_candidate) -> None:
    generator = torch_candidate.Generator(device="cpu").manual_seed(101)
    first = torch_candidate.empty((3, 4), dtype=torch_candidate.float32)
    second = torch_candidate.empty((3, 4), dtype=torch_candidate.float32)

    returned = first.normal_(mean=0.25, std=0.5, generator=generator)
    generator.manual_seed(101)
    second.normal_(mean=0.25, std=0.5, generator=generator)
    assert returned is first
    assert first.tolist() == second.tolist()

    generator.manual_seed(202)
    uniform_first = torch_candidate.empty((8,), dtype=torch_candidate.float32)
    uniform_second = torch_candidate.empty((8,), dtype=torch_candidate.float32)
    uniform_first.uniform_(-1.0, 2.0, generator=generator)
    generator.manual_seed(202)
    uniform_second.uniform_(-1.0, 2.0, generator=generator)
    assert uniform_first.tolist() == uniform_second.tolist()
    assert all(-1.0 <= value <= 2.0 for value in uniform_first.tolist())

    torch_candidate.manual_seed(303)
    expected_global = torch_candidate.randn((2, 2))
    torch_candidate.manual_seed(303)
    generator.manual_seed(404)
    first.normal_(generator=generator)
    actual_global = torch_candidate.randn((2, 2))
    assert actual_global.tolist() == expected_global.tolist()


def test_autocast_context_tracks_enabled_state(torch_candidate) -> None:
    assert not torch_candidate.is_autocast_enabled("cpu")
    with torch_candidate.autocast("cpu", enabled=True):
        assert torch_candidate.is_autocast_enabled("cpu")
        with torch_candidate.amp.autocast("cpu", enabled=False):
            assert not torch_candidate.is_autocast_enabled("cpu")
        assert torch_candidate.is_autocast_enabled("cpu")
    assert not torch_candidate.is_autocast_enabled("cpu")


def test_memory_format_channels_last_matches_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.arange(24, dtype=torch_reference.float32).reshape(1, 2, 3, 4)
    actual = torch_candidate.arange(24, dtype=torch_candidate.float32).reshape(1, 2, 3, 4)

    expected_channels_last = expected.contiguous(memory_format=torch_reference.channels_last)
    actual_channels_last = actual.contiguous(memory_format=torch_candidate.channels_last)
    assert actual_channels_last.stride() == expected_channels_last.stride()
    assert actual_channels_last.is_contiguous(memory_format=torch_candidate.channels_last)
    assert not actual_channels_last.is_contiguous()
    assert_values_compatible(
        torch_reference,
        expected_channels_last,
        actual_channels_last,
        path="memory_format.channels_last.contiguous",
    )

    expected_to = expected.to(memory_format=torch_reference.channels_last)
    actual_to = actual.to(memory_format=torch_candidate.channels_last)
    assert_values_compatible(
        torch_reference,
        expected_to,
        actual_to,
        path="memory_format.channels_last.to",
    )
    expected_channels_last_other = torch_reference.flip(expected_channels_last, (-1,)).contiguous(
        memory_format=torch_reference.channels_last
    )
    actual_channels_last_other = torch_candidate.flip(actual_channels_last, (-1,)).contiguous(
        memory_format=torch_candidate.channels_last
    )
    for dim in range(4):
        assert_values_compatible(
            torch_reference,
            torch_reference.cat([expected_channels_last, expected_channels_last_other], dim=dim),
            torch_candidate.cat([actual_channels_last, actual_channels_last_other], dim=dim),
            path=f"memory_format.channels_last.cat_dim{dim}",
            check_stride=True,
        )

    expected_like = torch_reference.zeros_like(expected_channels_last, memory_format=torch_reference.contiguous_format)
    actual_like = torch_candidate.zeros_like(actual_channels_last, memory_format=torch_candidate.contiguous_format)
    assert_values_compatible(
        torch_reference,
        expected_like,
        actual_like,
        path="memory_format.zeros_like.contiguous_format",
    )

    expected_3d = torch_reference.arange(120, dtype=torch_reference.float32).reshape(1, 2, 3, 4, 5)
    actual_3d = torch_candidate.arange(120, dtype=torch_candidate.float32).reshape(1, 2, 3, 4, 5)
    expected_channels_last_3d = expected_3d.contiguous(memory_format=torch_reference.channels_last_3d)
    actual_channels_last_3d = actual_3d.contiguous(memory_format=torch_candidate.channels_last_3d)
    assert actual_channels_last_3d.stride() == expected_channels_last_3d.stride()
    assert actual_channels_last_3d.is_contiguous(memory_format=torch_candidate.channels_last_3d)
    assert not actual_channels_last_3d.is_contiguous()
    assert_values_compatible(
        torch_reference,
        expected_channels_last_3d,
        actual_channels_last_3d,
        path="memory_format.channels_last_3d.contiguous",
    )

    expected_to_3d = expected_3d.to(memory_format=torch_reference.channels_last_3d)
    actual_to_3d = actual_3d.to(memory_format=torch_candidate.channels_last_3d)
    assert_values_compatible(
        torch_reference,
        expected_to_3d,
        actual_to_3d,
        path="memory_format.channels_last_3d.to",
    )
    expected_channels_last_3d_other = torch_reference.flip(expected_channels_last_3d, (-1,)).contiguous(
        memory_format=torch_reference.channels_last_3d
    )
    actual_channels_last_3d_other = torch_candidate.flip(actual_channels_last_3d, (-1,)).contiguous(
        memory_format=torch_candidate.channels_last_3d
    )
    for dim in range(5):
        assert_values_compatible(
            torch_reference,
            torch_reference.cat([expected_channels_last_3d, expected_channels_last_3d_other], dim=dim),
            torch_candidate.cat([actual_channels_last_3d, actual_channels_last_3d_other], dim=dim),
            path=f"memory_format.channels_last_3d.cat_dim{dim}",
            check_stride=True,
        )


def test_compile_noop_supports_direct_and_decorator(torch_candidate) -> None:
    def fn(value):
        return value + 1

    assert torch_candidate.compile(fn) is fn
    assert torch_candidate.compile()(fn) is fn


def test_backend_probe_modules_and_checkpoint_noop(torch_candidate) -> None:
    assert not torch_candidate.cuda.is_available()
    assert torch_candidate.cuda.device_count() == 0
    assert torch_candidate.cuda.empty_cache() is None
    assert torch_candidate.cuda.manual_seed(1) is None
    assert not torch_candidate.cuda.is_bf16_supported()
    with torch_candidate.cuda.amp.autocast(enabled=True):
        assert torch_candidate.is_autocast_enabled("cuda")
    assert not torch_candidate.is_autocast_enabled("cuda")
    scaler = torch_candidate.amp.GradScaler()
    assert scaler.scale(3) == 3
    assert scaler.state_dict() == {}
    assert not scaler.is_enabled()
    torch_candidate.backends.cuda.matmul.allow_tf32 = True
    assert torch_candidate.backends.cuda.matmul.allow_tf32 is True
    torch_candidate.backends.cudnn.benchmark = True
    assert torch_candidate.backends.cudnn.benchmark is True
    assert not torch_candidate.mps.is_available()
    assert not torch_candidate.mps.is_built()
    assert not torch_candidate.backends.mps.is_available()
    assert not torch_candidate.backends.mps.is_built()

    value = torch_candidate.tensor([1.0, 2.0], dtype=torch_candidate.float32)
    result = torch_candidate.utils.checkpoint.checkpoint(lambda item: item + 1.0, value, use_reentrant=False)
    assert result.tolist() == [2.0, 3.0]


def test_numpy_bridge_and_legacy_tensor_constructors(torch_reference, torch_candidate) -> None:
    numpy = pytest.importorskip("numpy")
    array = numpy.array([[1.0, -2.0], [3.5, 4.25]], dtype=numpy.float32)

    expected_from_numpy = torch_reference.from_numpy(array)
    actual_from_numpy = torch_candidate.from_numpy(array)
    assert_values_compatible(
        torch_reference,
        expected_from_numpy,
        actual_from_numpy,
        path="from_numpy.float32",
    )

    actual_numpy = actual_from_numpy.numpy()
    assert isinstance(actual_numpy, numpy.ndarray)
    assert actual_numpy.dtype == numpy.float32
    numpy.testing.assert_array_equal(actual_numpy, array)

    constructor_cases = (
        ("FloatTensor", [1.0, 2.0], torch_reference.float32, torch_candidate.float32),
        ("DoubleTensor", [1.0, 2.0], torch_reference.float64, torch_candidate.float64),
        ("HalfTensor", [1.0, 2.0], torch_reference.float16, torch_candidate.float16),
        ("LongTensor", [1, 2], torch_reference.int64, torch_candidate.int64),
        ("IntTensor", [1, 2], torch_reference.int32, torch_candidate.int32),
        ("BoolTensor", [True, False], torch_reference.bool, torch_candidate.bool),
    )
    for name, data, expected_dtype, actual_dtype in constructor_cases:
        expected = getattr(torch_reference, name)(data)
        actual = getattr(torch_candidate, name)(data)
        assert actual.dtype == actual_dtype
        assert_values_compatible(torch_reference, expected, actual, path=f"legacy_constructor.{name}")

    expected_shape = torch_reference.FloatTensor(2, 3)
    actual_shape = torch_candidate.FloatTensor(2, 3)
    assert tuple(actual_shape.shape) == tuple(expected_shape.shape)
    assert actual_shape.dtype == torch_candidate.float32


def test_legacy_tensor_type_and_new_methods_match_reference(torch_reference, torch_candidate) -> None:
    expected = torch_reference.tensor([1.0, 2.0], dtype=torch_reference.float32)
    actual = torch_candidate.tensor([1.0, 2.0], dtype=torch_candidate.float32)

    assert actual.type() == expected.type()
    assert_values_compatible(
        torch_reference,
        expected.type(torch_reference.DoubleTensor),
        actual.type(torch_candidate.DoubleTensor),
        path="tensor.type.DoubleTensor",
    )
    assert_values_compatible(
        torch_reference,
        expected.type(torch_reference.int64),
        actual.type(torch_candidate.int64),
        path="tensor.type.int64",
    )
    assert_values_compatible(
        torch_reference,
        expected.new([3.0, 4.0]),
        actual.new([3.0, 4.0]),
        path="tensor.new.data",
    )
    assert tuple(actual.new(2, 3).shape) == tuple(expected.new(2, 3).shape)
    assert actual.new(2, 3).dtype == actual.dtype


def test_multinomial_metadata_constraints_and_seed_reproducibility(torch_candidate) -> None:
    probabilities = torch_candidate.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0]], dtype=torch_candidate.float32)

    torch_candidate.manual_seed(7)
    first = torch_candidate.multinomial(probabilities, 3, replacement=True)
    torch_candidate.manual_seed(7)
    second = torch_candidate.multinomial(probabilities, 3, replacement=True)

    assert first.dtype == torch_candidate.int64
    assert first.device == torch_candidate.device("cpu")
    assert tuple(first.shape) == (2, 3)
    assert first.tolist() == second.tolist()
    assert first.tolist()[0] == [1, 1, 1]
    assert all(0 <= value < 3 for row in first.tolist() for value in row)

    without_replacement = torch_candidate.multinomial(probabilities, 3, replacement=False)
    for row in without_replacement.tolist():
        assert len(row) == len(set(row))

    generator = torch_candidate.Generator(device="cpu").manual_seed(17)
    generated_first = torch_candidate.multinomial(probabilities, 3, replacement=True, generator=generator)
    generator.manual_seed(17)
    generated_second = torch_candidate.multinomial(probabilities, 3, replacement=True, generator=generator)
    assert generated_first.tolist() == generated_second.tolist()

    torch_candidate.manual_seed(23)
    expected_global = torch_candidate.multinomial(probabilities, 3, replacement=True)
    torch_candidate.manual_seed(23)
    generator.manual_seed(29)
    torch_candidate.multinomial(probabilities, 3, replacement=True, generator=generator)
    actual_global = torch_candidate.multinomial(probabilities, 3, replacement=True)
    assert actual_global.tolist() == expected_global.tolist()


def test_multinomial_error_paths_match_reference(torch_reference, torch_candidate) -> None:
    cases = [
        (lambda module: module.multinomial(module.tensor([0.0, 0.0], dtype=module.float32), 1), RuntimeError),
        (lambda module: module.multinomial(module.tensor([-1.0, 2.0], dtype=module.float32), 1), RuntimeError),
        (lambda module: module.multinomial(module.tensor([1, 2], dtype=module.int64), 1), RuntimeError),
        (lambda module: module.multinomial(module.ones((1, 1, 1), dtype=module.float32), 1), RuntimeError),
        (lambda module: module.multinomial(module.ones((2,), dtype=module.float32), 3, replacement=False), RuntimeError),
    ]

    for call, exception_type in cases:
        with pytest.raises(exception_type):
            call(torch_reference)
        with pytest.raises(exception_type):
            call(torch_candidate)


def test_public_submodule_imports_match_reference(torch_reference, torch_candidate) -> None:
    for suffix in ("nn", "nn.functional", "nn.init", "linalg", "autograd", "optim"):
        expected = importlib.import_module(f"{torch_reference.__name__}.{suffix}")
        actual = importlib.import_module(f"{torch_candidate.__name__}.{suffix}")
        assert actual is _resolve(torch_candidate, suffix)
        assert type(actual).__name__ == type(expected).__name__


def _resolve(module, path):
    current = module
    for part in path.split("."):
        current = getattr(current, part)
    return current
