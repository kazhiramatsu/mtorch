from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Any, Callable

from .harness import assert_values_compatible


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    setup: Callable[[Any], tuple[Any, ...]]
    run: Callable[..., Any]
    rtol: float | None = None
    atol: float | None = None


@dataclass(frozen=True)
class BenchmarkSettings:
    warmup: int = 2
    repeat: int = 5
    slow_ratio: float = 1.0
    max_ratio: float = 0.0


@dataclass(frozen=True)
class BenchmarkResult:
    id: str
    reference_median_ms: float
    candidate_median_ms: float
    ratio: float
    status: str
    slow_ratio: float
    slow_ratio_exceeded: bool
    max_ratio: float
    max_ratio_exceeded: bool
    reference_min_ms: float
    candidate_min_ms: float
    repeat: int


BENCHMARK_RESULTS: list[BenchmarkResult] = []


def run_benchmark_case(
    reference: Any,
    candidate: Any,
    case: BenchmarkCase,
    settings: BenchmarkSettings,
) -> BenchmarkResult:
    if hasattr(reference, "set_num_threads"):
        reference.set_num_threads(1)

    reference_args = case.setup(reference)
    candidate_args = case.setup(candidate)

    expected = case.run(reference, *reference_args)
    actual = case.run(candidate, *candidate_args)
    assert_values_compatible(
        reference,
        expected,
        actual,
        path=f"{case.id}.benchmark_result",
        rtol=case.rtol,
        atol=case.atol,
        check_stride=False,
    )

    reference_times = _measure(lambda: case.run(reference, *reference_args), settings)
    candidate_times = _measure(lambda: case.run(candidate, *candidate_args), settings)
    reference_median_ms = median(reference_times) / 1_000_000
    candidate_median_ms = median(candidate_times) / 1_000_000
    reference_min_ms = min(reference_times) / 1_000_000
    candidate_min_ms = min(candidate_times) / 1_000_000
    ratio = candidate_median_ms / reference_median_ms if reference_median_ms > 0.0 else float("inf")
    slow_ratio_exceeded = ratio > settings.slow_ratio
    max_ratio_exceeded = settings.max_ratio > 0.0 and ratio > settings.max_ratio
    status = "FAIL" if max_ratio_exceeded else "SLOW" if slow_ratio_exceeded else "OK"

    result = BenchmarkResult(
        id=case.id,
        reference_median_ms=reference_median_ms,
        candidate_median_ms=candidate_median_ms,
        ratio=ratio,
        status=status,
        slow_ratio=settings.slow_ratio,
        slow_ratio_exceeded=slow_ratio_exceeded,
        max_ratio=settings.max_ratio,
        max_ratio_exceeded=max_ratio_exceeded,
        reference_min_ms=reference_min_ms,
        candidate_min_ms=candidate_min_ms,
        repeat=settings.repeat,
    )
    BENCHMARK_RESULTS.append(result)
    return result


def write_benchmark_json(path: Path) -> None:
    payload = {"benchmarks": [asdict(result) for result in BENCHMARK_RESULTS]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _measure(fn: Callable[[], Any], settings: BenchmarkSettings) -> list[int]:
    for _ in range(settings.warmup):
        _touch(fn())

    samples: list[int] = []
    for _ in range(settings.repeat):
        start = perf_counter_ns()
        result = fn()
        elapsed = perf_counter_ns() - start
        _touch(result)
        samples.append(elapsed)
    return samples


def _touch(value: Any) -> None:
    if isinstance(value, tuple):
        for item in value:
            _touch(item)
        return
    if hasattr(value, "shape"):
        tuple(value.shape)
        return
    if hasattr(value, "item"):
        value.item()


def _matrix(rows: int, cols: int) -> list[list[float]]:
    return [[float(((row * cols + col) % 17) - 8) / 8.0 for col in range(cols)] for row in range(rows)]


def _unique_matrix(rows: int, cols: int) -> list[list[float]]:
    return [[float(row * cols + col) / float(rows * cols) for col in range(cols)] for row in range(rows)]


def _near_one_matrix(rows: int, cols: int) -> list[list[float]]:
    return [[1.0 + float(((row * cols + col) % 7) - 3) / 1000.0 for col in range(cols)] for row in range(rows)]


def _probability_matrix(rows: int, cols: int) -> list[list[float]]:
    return [[0.05 + 0.9 * float((row * cols + col) % 17) / 16.0 for col in range(cols)] for row in range(rows)]


def _tensor3(rows: int, cols: int, depth: int) -> list[list[list[float]]]:
    return [
        [
            [float(((plane * cols * depth + row * depth + col) % 19) - 9) / 9.0 for col in range(depth)]
            for row in range(cols)
        ]
        for plane in range(rows)
    ]


def _tensor4(batch: int, channels: int, height: int, width: int) -> list[list[list[list[float]]]]:
    return [
        [
            [
                [
                    float(((n * channels * height * width + c * height * width + h * width + w) % 23) - 11) / 11.0
                    for w in range(width)
                ]
                for h in range(height)
            ]
            for c in range(channels)
        ]
        for n in range(batch)
    ]


def _tensor5(batch: int, channels: int, depth: int, height: int, width: int) -> list[list[list[list[list[float]]]]]:
    return [
        [
            [
                [
                    [
                        float(((n * channels * depth * height * width + c * depth * height * width + d * height * width + h * width + w) % 29) - 14) / 14.0
                        for w in range(width)
                    ]
                    for h in range(height)
                ]
                for d in range(depth)
            ]
            for c in range(channels)
        ]
        for n in range(batch)
    ]


def _vector(cols: int) -> list[float]:
    return [float((col % 11) - 5) / 5.0 for col in range(cols)]


def _mask(rows: int, cols: int) -> list[list[bool]]:
    return [[(row + col) % 3 == 0 for col in range(cols)] for row in range(rows)]


def _bool_matrix(rows: int, cols: int, value: bool) -> list[list[bool]]:
    return [[value for _ in range(cols)] for _ in range(rows)]


def _tensor(module: Any, data: Any, dtype_name: str = "float32", requires_grad: bool = False) -> Any:
    return module.tensor(data, dtype=getattr(module, dtype_name), requires_grad=requires_grad)


def _case_binary_add_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _matrix(64, 64))


def _case_logical_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _mask(64, 64), "bool"), _tensor(module, [[(row * 2 + col) % 5 == 0 for col in range(64)] for row in range(64)], "bool")


def _case_logical_not_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _mask(64, 64), "bool"),)


def _case_isclose_setup(module: Any) -> tuple[Any, Any]:
    left = _near_one_matrix(64, 64)
    right = [[value + (1e-6 if (row + col) % 4 == 0 else 0.0) for col, value in enumerate(values)] for row, values in enumerate(left)]
    return _tensor(module, left), _tensor(module, right)


def _case_nonzero_binary_setup(module: Any) -> tuple[Any, Any]:
    left = _matrix(64, 64)
    right = [[1.0 + abs(value) for value in row] for row in _matrix(64, 64)]
    return _tensor(module, left), _tensor(module, right)


def _case_hypot_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _near_one_matrix(64, 64)), _tensor(module, _probability_matrix(64, 64))


def _case_positive_binary_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _probability_matrix(64, 64)), _tensor(module, _near_one_matrix(64, 64))


def _case_heaviside_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _probability_matrix(64, 64))


def _case_lerp_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _near_one_matrix(64, 64)), _tensor(module, _probability_matrix(64, 64))


def _case_addcmul_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _matrix(64, 64)), _tensor(module, _near_one_matrix(64, 64))


def _case_addcdiv_setup(module: Any) -> tuple[Any, Any, Any]:
    denominator = [[1.0 + abs(value) for value in row] for row in _near_one_matrix(64, 64)]
    return _tensor(module, _matrix(64, 64)), _tensor(module, _matrix(64, 64)), _tensor(module, denominator)


def _case_special_float_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, [[float("nan") if (row + col) % 17 == 0 else float("inf") if (row + col) % 19 == 0 else -float("inf") if (row + col) % 23 == 0 else value for col, value in enumerate(values)] for row, values in enumerate(_matrix(64, 64))]),)


def _case_loss_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _matrix(64, 64))


def _case_cross_entropy_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, [row % 64 for row in range(64)], "int64")


def _case_cross_entropy_weight_setup(module: Any) -> tuple[Any, Any, Any]:
    return (
        _tensor(module, _matrix(64, 64)),
        _tensor(module, [row % 64 for row in range(64)], "int64"),
        _tensor(module, [1.0 + (col % 7) * 0.25 for col in range(64)]),
    )


def _case_nll_loss_setup(module: Any) -> tuple[Any, Any]:
    logits = _tensor(module, _matrix(64, 64))
    return module.nn.functional.log_softmax(logits, dim=1), _tensor(module, [row % 64 for row in range(64)], "int64")


def _case_nll_loss_weight_setup(module: Any) -> tuple[Any, Any, Any]:
    logits = _tensor(module, _matrix(64, 64))
    return (
        module.nn.functional.log_softmax(logits, dim=1),
        _tensor(module, [row % 64 for row in range(64)], "int64"),
        _tensor(module, [1.0 + (col % 7) * 0.25 for col in range(64)]),
    )


def _case_bce_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _probability_matrix(64, 64)), _tensor(module, _probability_matrix(64, 64))


def _case_bce_with_logits_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, [[1.0 if (row + col) % 3 == 0 else 0.0 for col in range(64)] for row in range(64)])


def _case_bce_weight_setup(module: Any) -> tuple[Any, Any, Any]:
    input, target = _case_bce_setup(module)
    weight = _tensor(module, [[1.0 + ((row + col) % 5) * 0.25 for col in range(64)] for row in range(64)])
    return input, target, weight


def _case_bce_with_logits_weight_pos_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    input, target = _case_bce_with_logits_setup(module)
    weight = _tensor(module, [[1.0 + ((row + col) % 5) * 0.25 for col in range(64)] for row in range(64)])
    pos_weight = _tensor(module, [1.0 + (col % 3) * 0.5 for col in range(64)])
    return input, target, weight, pos_weight


def _case_broadcast_add_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _vector(64))


def _case_sum_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _matrix(128, 128)),)


def _case_clip_argmax_int64_setup(module: Any) -> tuple[Any]:
    return (
        module.tensor(
            [
                [((row * 7 + col * 5) % 32) if col != 40 + row else 99 for col in range(77)]
                for row in range(2)
            ],
            dtype=module.long,
        ),
    )


def _case_prod_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _near_one_matrix(128, 128)),)


def _case_all_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _bool_matrix(128, 128, True), "bool"),)


def _case_matmul_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(48, 48)), _tensor(module, _matrix(48, 48))


def _case_matmul_batched_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _tensor3(8, 32, 32)), _tensor(module, _tensor3(8, 32, 32))


def _case_matmul_broadcast_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _tensor3(1, 32, 32)), _tensor(module, _tensor3(8, 32, 32))


def _case_dot_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _vector(1024)), _tensor(module, _vector(1024))


def _case_inner_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _matrix(64, 64))


def _case_mv_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _vector(64))


def _case_linear_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _matrix(64, 64)), _tensor(module, _vector(64))


def _case_linear_3d_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _tensor3(8, 32, 64)), _tensor(module, _matrix(64, 64)), _tensor(module, _vector(64))


def _case_linear_half_projection_setup(module: Any) -> tuple[Any, Any, Any]:
    input = (module.arange(0, 2 * 77 * 64, dtype=module.float32).reshape(2, 77, 64) / 1024.0 - 1.0).to(
        dtype=module.float16
    )
    weight = (module.arange(0, 64 * 64, dtype=module.float32).reshape(64, 64) / 4096.0 - 0.5).to(
        dtype=module.float16
    )
    bias = (module.arange(0, 64, dtype=module.float32) / 2048.0 - 0.015).to(dtype=module.float16)
    return input, weight, bias


def _case_conv1d_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _tensor3(1, 4, 32)), _tensor(module, _tensor3(8, 4, 3)), _tensor(module, _vector(8))


def _case_pool1d_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _tensor3(1, 4, 32)),)


def _case_conv_transpose1d_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _tensor3(1, 4, 32)), _tensor(module, _tensor3(4, 4, 3)), _tensor(module, _vector(4))


def _case_conv2d_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _tensor4(1, 4, 8, 8)), _tensor(module, _tensor4(8, 4, 3, 3)), _tensor(module, _vector(8))


def _case_conv2d_half_1x1_setup(module: Any) -> tuple[Any, Any, Any]:
    input = (module.sin(module.arange(0, 1 * 16 * 8 * 8, dtype=module.float32).reshape(1, 16, 8, 8) * 0.017) * 0.5).to(
        dtype=module.float16
    )
    weight = (module.cos(module.arange(0, 16 * 16, dtype=module.float32).reshape(16, 16, 1, 1) * 0.013) * 0.05).to(
        dtype=module.float16
    )
    bias = (module.arange(0, 16, dtype=module.float32) / 128.0 - 0.05).to(dtype=module.float16)
    return input, weight, bias


def _case_conv3d_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _tensor5(1, 2, 4, 4, 4)), _tensor(module, _tensor5(3, 2, 3, 3, 3)), _tensor(module, _vector(3))


def _case_conv3d_half_setup(module: Any) -> tuple[Any, Any, Any]:
    hidden_base = module.arange(0, 1 * 4 * 4 * 8 * 8, dtype=module.float32).reshape(1, 4, 4, 8, 8)
    hidden = (module.sin(hidden_base * 0.017) * 0.5).to(dtype=module.float16)
    weight_base = module.arange(0, 4 * 4 * 3 * 3 * 3, dtype=module.float32).reshape(4, 4, 3, 3, 3)
    weight = (module.cos(weight_base * 0.011) * 0.04).to(dtype=module.float16)
    bias = (module.arange(0, 4, dtype=module.float32) / 256.0 - 0.01).to(dtype=module.float16)
    return hidden, weight, bias


def _case_conv_transpose3d_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _tensor5(1, 2, 4, 4, 4)), _tensor(module, _tensor5(2, 3, 3, 3, 3)), _tensor(module, _vector(3))


def _case_conv_transpose2d_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _tensor4(1, 4, 8, 8)), _tensor(module, _tensor4(4, 4, 3, 3)), _tensor(module, _vector(4))


def _case_pad_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _tensor4(1, 4, 16, 16)),)


def _case_one_hot_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, [[(row + col) % 16 for col in range(16)] for row in range(16)], "int64"),)


def _case_cosine_similarity_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _near_one_matrix(64, 64))


def _case_interpolate_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _tensor4(1, 4, 16, 16)),)


def _case_interpolate_half_setup(module: Any) -> tuple[Any]:
    tensor = module.arange(0, 1 * 4 * 16 * 16, dtype=module.float32).reshape(1, 4, 16, 16) / 128.0
    return (tensor.to(dtype=module.float16),)


def _case_grid_sample_setup(module: Any) -> tuple[Any, Any]:
    height = 16
    width = 16
    grid = [
        [
            [
                [2.0 * (col + 0.5) / width - 1.0, 2.0 * (row + 0.5) / height - 1.0]
                for col in range(width)
            ]
            for row in range(height)
        ]
    ]
    return _tensor(module, _tensor4(1, 4, height, width)), _tensor(module, grid)


def _case_affine_grid_setup(module: Any) -> tuple[Any]:
    theta = module.tensor([[[0.95, 0.05, 0.1], [-0.05, 0.9, -0.1]]], dtype=module.float32)
    return (theta,)


def _case_stable_diffusion_spatial_transform_setup(module: Any) -> tuple[Any, Any, Any]:
    sample = module.arange(0, 1 * 4 * 16 * 16, dtype=module.float32).reshape(1, 4, 16, 16) / 128.0 - 1.0
    theta = module.tensor([[[0.95, 0.05, 0.1], [-0.05, 0.9, -0.1]]], dtype=module.float32)
    mask = module.arange(0, 1 * 1 * 8 * 8, dtype=module.float32).reshape(1, 1, 8, 8) % 3.0
    return sample, theta, mask


def _stable_diffusion_spatial_transform_path(module: Any, sample: Any, theta: Any, mask: Any) -> Any:
    grid = module.nn.functional.affine_grid(theta, sample.shape, align_corners=False)
    warped = module.nn.functional.grid_sample(
        sample,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    resized_mask = module.nn.functional.interpolate(mask, size=(sample.shape[2], sample.shape[3]), mode="nearest-exact")
    return module.where(resized_mask > 0.5, warped, sample)


def _case_fold_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _tensor3(1, 36, 256)),)


def _case_sdpa_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _tensor4(1, 2, 16, 16)), _tensor(module, _tensor4(1, 2, 16, 16)), _tensor(module, _tensor4(1, 2, 16, 16))


def _case_sdpa_gqa_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _tensor4(1, 4, 16, 16)), _tensor(module, _tensor4(1, 2, 16, 16)), _tensor(module, _tensor4(1, 2, 16, 16))


def _case_stable_diffusion_clip_causal_attention_setup(module: Any) -> tuple[Any, Any, Any]:
    query = module.arange(0, 1 * 2 * 77 * 64, dtype=module.float32).reshape(1, 2, 77, 64) / 1024.0
    key = module.flip(query, (-1,))
    value = module.arange(0, 1 * 2 * 77 * 64, dtype=module.float32).reshape(1, 2, 77, 64) / 2048.0
    return query, key, value


def _case_stable_diffusion_clip_causal_attention_half_setup(module: Any) -> tuple[Any, Any, Any]:
    base = module.arange(0, 1 * 2 * 77 * 64, dtype=module.float32).reshape(1, 2, 77, 64)
    query = (module.sin(base * 0.017) * 0.1).to(dtype=module.float16)
    key = (module.cos(base * 0.019) * 0.1).to(dtype=module.float16)
    value = (module.sin(base * 0.013) * 0.1).to(dtype=module.float16)
    return query, key, value


def _case_stable_diffusion_half_additive_masked_attention_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    base = module.arange(0, 1 * 2 * 64 * 64, dtype=module.float32).reshape(1, 2, 64, 64)
    query = (module.sin(base * 0.017) * 0.1).to(dtype=module.float16)
    key = (module.cos(base * 0.019) * 0.1).to(dtype=module.float16)
    value = (module.sin(base * 0.013) * 0.1).to(dtype=module.float16)
    rows = module.arange(0, 64, dtype=module.long)
    cols = module.flip(rows, (0,))
    batch_indices = module.zeros((64,), dtype=module.long)
    head_indices = module.zeros((64,), dtype=module.long)
    mask_values = module.full((64,), -10000.0, dtype=module.float32)
    mask = module.zeros((1, 1, 64, 64), dtype=module.float32)
    mask = module.index_put(mask, (batch_indices, head_indices, rows, cols), mask_values)
    return query, key, value, mask


def _case_stable_diffusion_half_dtype_masked_attention_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    base = module.arange(0, 1 * 2 * 64 * 64, dtype=module.float32).reshape(1, 2, 64, 64)
    query = (module.sin(base * 0.017) * 0.1).to(dtype=module.float16)
    key = (module.cos(base * 0.019) * 0.1).to(dtype=module.float16)
    value = (module.sin(base * 0.013) * 0.1).to(dtype=module.float16)
    rows = module.arange(0, 64, dtype=module.long)
    cols = module.flip(rows, (0,))
    batch_indices = module.zeros((64,), dtype=module.long)
    head_indices = module.zeros((64,), dtype=module.long)
    mask_values = module.full((64,), -10000.0, dtype=module.float16)
    mask = module.zeros((1, 1, 64, 64), dtype=module.float16)
    mask = module.index_put(mask, (batch_indices, head_indices, rows, cols), mask_values)
    return query, key, value, mask


def _case_stable_diffusion_half_bool_masked_attention_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    base = module.arange(0, 1 * 2 * 64 * 64, dtype=module.float32).reshape(1, 2, 64, 64)
    query = (module.sin(base * 0.017) * 0.1).to(dtype=module.float16)
    key = (module.cos(base * 0.019) * 0.1).to(dtype=module.float16)
    value = (module.sin(base * 0.013) * 0.1).to(dtype=module.float16)
    rows = module.arange(0, 64, dtype=module.long)
    cols = module.flip(rows, (0,))
    batch_indices = module.zeros((64,), dtype=module.long)
    head_indices = module.zeros((64,), dtype=module.long)
    mask_values = module.zeros((64,), dtype=module.bool)
    mask = module.ones((1, 1, 64, 64), dtype=module.bool)
    mask = module.index_put(mask, (batch_indices, head_indices, rows, cols), mask_values)
    return query, key, value, mask


def _case_stable_diffusion_half_rank3_cross_attention_setup(module: Any) -> tuple[Any, Any, Any, Any]:
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
    return query, key, value, text_mask


def _stable_diffusion_half_rank3_cross_attention_path(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    text_mask: Any,
) -> Any:
    attended = module.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=text_mask,
        scale=0.125,
    )
    return attended + query


def _stable_diffusion_half_legacy_baddbmm_attention_path(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    text_mask: Any,
) -> Any:
    scores = module.baddbmm(
        module.zeros((query.shape[0], query.shape[1], key.shape[1]), dtype=query.dtype),
        query,
        key.transpose(1, 2).contiguous(),
        beta=0.0,
        alpha=0.125,
    )
    scores = scores.masked_fill(~text_mask, -10000.0)
    probabilities = module.softmax(scores, dim=-1)
    return module.bmm(probabilities, value) + query


def _stable_diffusion_half_upcast_baddbmm_attention_path(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    text_mask: Any,
) -> Any:
    scores = module.baddbmm(
        module.zeros((query.shape[0], query.shape[1], key.shape[1]), dtype=module.float32),
        query.to(dtype=module.float32),
        key.transpose(1, 2).contiguous().to(dtype=module.float32),
        beta=0.0,
        alpha=0.125,
    )
    scores = scores.masked_fill(~text_mask, -10000.0)
    probabilities = module.softmax(scores, dim=-1).to(dtype=module.float16)
    return module.bmm(probabilities, value) + query


def _case_stable_diffusion_half_unet_cross_attention_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    batch, query_tokens, key_tokens, channels, heads = 2, 64, 77, 320, 8
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
    return hidden, encoder_hidden, q_weight, k_weight, v_weight, out_weight, bias, text_mask


def _stable_diffusion_half_unet_cross_attention_path(
    module: Any,
    hidden: Any,
    encoder_hidden: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    out_weight: Any,
    bias: Any,
    text_mask: Any,
) -> Any:
    batch, query_tokens, channels = hidden.shape
    heads = 8
    head_dim = channels // heads
    key_tokens = encoder_hidden.shape[1]
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


def _case_stable_diffusion_sdxl_unet_cross_attention_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    batch, query_tokens, key_tokens, channels, cross_attention_dim, heads = 2, 64, 77, 320, 2048, 8
    hidden_base = module.arange(0, batch * query_tokens * channels, dtype=module.float32).reshape(
        batch,
        query_tokens,
        channels,
    )
    encoder_base = module.arange(0, batch * key_tokens * cross_attention_dim, dtype=module.float32).reshape(
        batch,
        key_tokens,
        cross_attention_dim,
    )
    hidden = (module.sin(hidden_base * 0.013) * 0.2).to(dtype=module.float16)
    encoder_hidden = (module.cos(encoder_base * 0.003) * 0.15).to(dtype=module.float16)
    q_base = module.arange(0, channels * channels, dtype=module.float32).reshape(channels, channels)
    kv_base = module.arange(0, channels * cross_attention_dim, dtype=module.float32).reshape(
        channels,
        cross_attention_dim,
    )
    q_weight = (module.sin(q_base * 0.007) / 64.0).to(dtype=module.float16)
    k_weight = (module.cos(kv_base * 0.005) / 128.0).to(dtype=module.float16)
    v_weight = (module.sin(kv_base * 0.006 + 0.2) / 128.0).to(dtype=module.float16)
    out_weight = module.flip(q_weight, (1,))
    bias = (module.arange(0, channels, dtype=module.float32) / 4096.0 - 0.03).to(dtype=module.float16)
    text_mask = module.ones((batch * heads, 1, key_tokens), dtype=module.bool)
    text_mask[:, :, key_tokens - 5 :] = False
    return hidden, encoder_hidden, q_weight, k_weight, v_weight, out_weight, bias, text_mask


def _stable_diffusion_sdxl_unet_cross_attention_path(
    module: Any,
    hidden: Any,
    encoder_hidden: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    out_weight: Any,
    bias: Any,
    text_mask: Any,
) -> Any:
    batch, query_tokens, channels = hidden.shape
    heads = 8
    head_dim = channels // heads
    key_tokens = encoder_hidden.shape[1]
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


def _case_stable_diffusion_sdxl_ip_adapter_cross_attention_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    (
        hidden,
        encoder_hidden,
        q_weight,
        k_weight,
        v_weight,
        out_weight,
        bias,
        text_mask,
    ) = _case_stable_diffusion_sdxl_unet_cross_attention_setup(module)
    batch, image_tokens, image_dim, channels = 2, 16, 2048, 320
    image_base = module.arange(0, batch * image_tokens * image_dim, dtype=module.float32).reshape(
        batch,
        image_tokens,
        image_dim,
    )
    image_hidden = (module.sin(image_base * 0.004 + 0.5) * 0.12).to(dtype=module.float16)
    ip_base = module.arange(0, channels * image_dim, dtype=module.float32).reshape(channels, image_dim)
    ip_k_weight = (module.cos(ip_base * 0.006 + 0.1) / 128.0).to(dtype=module.float16)
    ip_v_weight = (module.sin(ip_base * 0.007 + 0.3) / 128.0).to(dtype=module.float16)
    return (
        hidden,
        encoder_hidden,
        q_weight,
        k_weight,
        v_weight,
        out_weight,
        bias,
        text_mask,
        image_hidden,
        ip_k_weight,
        ip_v_weight,
    )


def _stable_diffusion_sdxl_ip_adapter_cross_attention_path(
    module: Any,
    hidden: Any,
    encoder_hidden: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    out_weight: Any,
    bias: Any,
    text_mask: Any,
    image_hidden: Any,
    ip_k_weight: Any,
    ip_v_weight: Any,
) -> Any:
    batch, query_tokens, channels = hidden.shape
    heads = 8
    head_dim = channels // heads
    key_tokens = encoder_hidden.shape[1]
    image_tokens = image_hidden.shape[1]
    query = module.nn.functional.linear(hidden, q_weight, bias).reshape(batch, query_tokens, heads, head_dim)
    key = module.nn.functional.linear(encoder_hidden, k_weight, bias).reshape(batch, key_tokens, heads, head_dim)
    value = module.nn.functional.linear(encoder_hidden, v_weight, bias).reshape(batch, key_tokens, heads, head_dim)
    image_key = module.nn.functional.linear(image_hidden, ip_k_weight, None).reshape(
        batch,
        image_tokens,
        heads,
        head_dim,
    )
    image_value = module.nn.functional.linear(image_hidden, ip_v_weight, None).reshape(
        batch,
        image_tokens,
        heads,
        head_dim,
    )
    query = query.transpose(1, 2).reshape(batch * heads, query_tokens, head_dim)
    key = key.transpose(1, 2).reshape(batch * heads, key_tokens, head_dim)
    value = value.transpose(1, 2).reshape(batch * heads, key_tokens, head_dim)
    image_key = image_key.transpose(1, 2).reshape(batch * heads, image_tokens, head_dim)
    image_value = image_value.transpose(1, 2).reshape(batch * heads, image_tokens, head_dim)
    text_attended = module.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=text_mask,
        scale=0.125,
    )
    image_attended = module.nn.functional.scaled_dot_product_attention(
        query,
        image_key,
        image_value,
        scale=0.125,
    )
    attended = text_attended + image_attended * 0.65
    attended = attended.reshape(batch, heads, query_tokens, head_dim).transpose(1, 2).reshape(
        batch,
        query_tokens,
        channels,
    )
    return hidden + module.nn.functional.linear(attended, out_weight, bias)


def _case_stable_diffusion_sd3_joint_transformer_block_setup(
    module: Any,
) -> tuple[Any, ...]:
    batch, image_tokens, text_tokens, width, intermediate = 2, 64, 77, 320, 640
    hidden_base = module.arange(0, batch * image_tokens * width, dtype=module.float32).reshape(
        batch,
        image_tokens,
        width,
    )
    context_base = module.arange(0, batch * text_tokens * width, dtype=module.float32).reshape(
        batch,
        text_tokens,
        width,
    )
    hidden = (module.sin(hidden_base * 0.013) * 0.18).to(dtype=module.float16)
    context = (module.cos(context_base * 0.011) * 0.14).to(dtype=module.float16)

    hidden_norm_weight = (0.82 + (module.arange(0, width, dtype=module.float32) % 251) / 2048.0).to(
        dtype=module.float16
    )
    hidden_norm_bias = (((module.arange(0, width, dtype=module.float32) % 127) - 63.0) / 4096.0).to(
        dtype=module.float16
    )
    context_norm_weight = (0.78 + (module.arange(0, width, dtype=module.float32) % 239) / 2048.0).to(
        dtype=module.float16
    )
    context_norm_bias = (((module.arange(0, width, dtype=module.float32) % 131) - 65.0) / 4096.0).to(
        dtype=module.float16
    )
    mlp_norm_weight = (0.86 + (module.arange(0, width, dtype=module.float32) % 257) / 2048.0).to(
        dtype=module.float16
    )
    mlp_norm_bias = (((module.arange(0, width, dtype=module.float32) % 137) - 68.0) / 4096.0).to(
        dtype=module.float16
    )

    hidden_modulation = (
        module.sin(module.arange(0, batch * 6 * width, dtype=module.float32).reshape(batch, 6, width) * 0.017)
        * 0.08
    ).to(dtype=module.float16)
    context_modulation = (
        module.cos(module.arange(0, batch * 3 * width, dtype=module.float32).reshape(batch, 3, width) * 0.019)
        * 0.07
    ).to(dtype=module.float16)

    qkv_base = (
        module.arange(0, 3 * width * width, dtype=module.float32).reshape(3 * width, width) % 1543
    ) / 65536.0 - 0.012
    context_qkv_base = (
        (module.arange(0, 3 * width * width, dtype=module.float32).reshape(3 * width, width) + 97) % 1553
    ) / 65536.0 - 0.012
    qkv_weight = qkv_base.to(dtype=module.float16)
    qkv_bias = (((module.arange(0, 3 * width, dtype=module.float32) % 181) - 90.0) / 8192.0).to(
        dtype=module.float16
    )
    context_qkv_weight = context_qkv_base.to(dtype=module.float16)
    context_qkv_bias = (((module.arange(0, 3 * width, dtype=module.float32) % 173) - 86.0) / 8192.0).to(
        dtype=module.float16
    )
    out_weight = (
        (module.arange(0, width * width, dtype=module.float32).reshape(width, width) % 1237) / 65536.0 - 0.01
    ).to(dtype=module.float16)
    out_bias = (((module.arange(0, width, dtype=module.float32) % 149) - 74.0) / 8192.0).to(dtype=module.float16)
    context_out_weight = module.flip(out_weight, (1,)).contiguous()
    context_out_bias = (((module.arange(0, width, dtype=module.float32) % 151) - 75.0) / 8192.0).to(
        dtype=module.float16
    )

    ff_in_weight = (
        (module.arange(0, 2 * intermediate * width, dtype=module.float32).reshape(2 * intermediate, width) % 1559)
        / 65536.0
        - 0.012
    ).to(dtype=module.float16)
    ff_in_bias = (((module.arange(0, 2 * intermediate, dtype=module.float32) % 191) - 95.0) / 8192.0).to(
        dtype=module.float16
    )
    ff_out_weight = (
        (module.arange(0, width * intermediate, dtype=module.float32).reshape(width, intermediate) % 1567)
        / 65536.0
        - 0.012
    ).to(dtype=module.float16)
    ff_out_bias = (((module.arange(0, width, dtype=module.float32) % 157) - 78.0) / 8192.0).to(
        dtype=module.float16
    )
    return (
        hidden,
        context,
        hidden_norm_weight,
        hidden_norm_bias,
        context_norm_weight,
        context_norm_bias,
        mlp_norm_weight,
        mlp_norm_bias,
        hidden_modulation,
        context_modulation,
        qkv_weight,
        qkv_bias,
        context_qkv_weight,
        context_qkv_bias,
        out_weight,
        out_bias,
        context_out_weight,
        context_out_bias,
        ff_in_weight,
        ff_in_bias,
        ff_out_weight,
        ff_out_bias,
    )


def _stable_diffusion_sd3_joint_transformer_block_path(
    module: Any,
    hidden: Any,
    context: Any,
    hidden_norm_weight: Any,
    hidden_norm_bias: Any,
    context_norm_weight: Any,
    context_norm_bias: Any,
    mlp_norm_weight: Any,
    mlp_norm_bias: Any,
    hidden_modulation: Any,
    context_modulation: Any,
    qkv_weight: Any,
    qkv_bias: Any,
    context_qkv_weight: Any,
    context_qkv_bias: Any,
    out_weight: Any,
    out_bias: Any,
    context_out_weight: Any,
    context_out_bias: Any,
    ff_in_weight: Any,
    ff_in_bias: Any,
    ff_out_weight: Any,
    ff_out_bias: Any,
) -> Any:
    batch, image_tokens, width = hidden.shape
    heads = 8
    if width % heads != 0:
        raise ValueError("SD3 joint transformer width must be divisible by the head count")
    head_dim = width // heads
    text_tokens = context.shape[1]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = hidden_modulation.chunk(6, dim=1)
    context_shift, context_scale, context_gate = context_modulation.chunk(3, dim=1)

    normed_hidden = module.nn.functional.layer_norm(hidden, (width,), hidden_norm_weight, hidden_norm_bias, eps=1e-5)
    normed_hidden = normed_hidden * (1.0 + scale_msa) + shift_msa
    normed_context = module.nn.functional.layer_norm(
        context,
        (width,),
        context_norm_weight,
        context_norm_bias,
        eps=1e-5,
    )
    normed_context = normed_context * (1.0 + context_scale) + context_shift

    qkv = module.nn.functional.linear(normed_hidden, qkv_weight, qkv_bias)
    query, key, value = qkv.chunk(3, dim=-1)
    context_qkv = module.nn.functional.linear(normed_context, context_qkv_weight, context_qkv_bias)
    context_query, context_key, context_value = context_qkv.chunk(3, dim=-1)

    query = query.reshape(batch, image_tokens, heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, image_tokens, heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, image_tokens, heads, head_dim).transpose(1, 2)
    context_query = context_query.reshape(batch, text_tokens, heads, head_dim).transpose(1, 2)
    context_key = context_key.reshape(batch, text_tokens, heads, head_dim).transpose(1, 2)
    context_value = context_value.reshape(batch, text_tokens, heads, head_dim).transpose(1, 2)

    joint_query = module.cat([context_query, query], dim=2)
    joint_key = module.cat([context_key, key], dim=2)
    joint_value = module.cat([context_value, value], dim=2)
    attended = module.nn.functional.scaled_dot_product_attention(joint_query, joint_key, joint_value, scale=0.125)
    attended_context, attended_hidden = module.split(attended, [text_tokens, image_tokens], dim=2)

    attended_hidden = attended_hidden.transpose(1, 2).reshape(batch, image_tokens, width)
    attended_context = attended_context.transpose(1, 2).reshape(batch, text_tokens, width)
    hidden = hidden + gate_msa * module.nn.functional.linear(attended_hidden, out_weight, out_bias)
    context = context + context_gate * module.nn.functional.linear(attended_context, context_out_weight, context_out_bias)

    mlp_hidden = module.nn.functional.layer_norm(hidden, (width,), mlp_norm_weight, mlp_norm_bias, eps=1e-5)
    mlp_hidden = mlp_hidden * (1.0 + scale_mlp) + shift_mlp
    geglu = module.nn.functional.linear(mlp_hidden, ff_in_weight, ff_in_bias)
    values, gate = geglu.chunk(2, dim=-1)
    mlp_hidden = values * module.nn.functional.gelu(gate)
    hidden = hidden + gate_mlp * module.nn.functional.linear(mlp_hidden, ff_out_weight, ff_out_bias)
    return hidden, context


def _case_stable_diffusion_sd3_qk_norm_joint_attention_shape_setup(
    module: Any,
    image_tokens: int,
    text_tokens: int,
    width: int,
) -> tuple[Any, ...]:
    batch, heads = 2, 8
    if width % heads != 0:
        raise ValueError("SD3 joint attention width must be divisible by the head count")
    head_dim = width // heads
    hidden_base = module.arange(0, batch * image_tokens * width, dtype=module.float32).reshape(
        batch,
        image_tokens,
        width,
    )
    context_base = module.arange(0, batch * text_tokens * width, dtype=module.float32).reshape(
        batch,
        text_tokens,
        width,
    )
    hidden = (module.sin(hidden_base * 0.013) * 0.18).to(dtype=module.float16)
    context = (module.cos(context_base * 0.011) * 0.14).to(dtype=module.float16)

    base_weight = (
        module.arange(0, width * width, dtype=module.float32).reshape(width, width) % 1543
    ) / 65536.0 - 0.012
    q_weight = base_weight.to(dtype=module.float16)
    k_weight = module.flip(base_weight, (0,)).contiguous().to(dtype=module.float16)
    v_weight = module.flip(base_weight, (1,)).contiguous().to(dtype=module.float16)
    context_base_weight = (
        (module.arange(0, width * width, dtype=module.float32).reshape(width, width) + 97) % 1553
    ) / 65536.0 - 0.012
    context_q_weight = context_base_weight.to(dtype=module.float16)
    context_k_weight = module.flip(context_base_weight, (0,)).contiguous().to(dtype=module.float16)
    context_v_weight = module.flip(context_base_weight, (1,)).contiguous().to(dtype=module.float16)
    bias = (((module.arange(0, width, dtype=module.float32) % 181) - 90.0) / 8192.0).to(dtype=module.float16)
    context_bias = (((module.arange(0, width, dtype=module.float32) % 173) - 86.0) / 8192.0).to(
        dtype=module.float16
    )
    q_norm_weight = (0.92 + (module.arange(0, head_dim, dtype=module.float32) % 37) / 512.0).to(
        dtype=module.float16
    )
    k_norm_weight = (0.88 + (module.arange(0, head_dim, dtype=module.float32) % 41) / 512.0).to(
        dtype=module.float16
    )
    out_weight = (
        (module.arange(0, width * width, dtype=module.float32).reshape(width, width) % 1237) / 65536.0 - 0.01
    ).to(dtype=module.float16)
    out_bias = (((module.arange(0, width, dtype=module.float32) % 149) - 74.0) / 8192.0).to(dtype=module.float16)
    context_out_weight = module.flip(out_weight, (1,)).contiguous()
    context_out_bias = (((module.arange(0, width, dtype=module.float32) % 151) - 75.0) / 8192.0).to(
        dtype=module.float16
    )
    return (
        hidden,
        context,
        q_weight,
        k_weight,
        v_weight,
        context_q_weight,
        context_k_weight,
        context_v_weight,
        bias,
        context_bias,
        q_norm_weight,
        k_norm_weight,
        out_weight,
        out_bias,
        context_out_weight,
        context_out_bias,
    )


def _case_stable_diffusion_sd3_qk_norm_joint_attention_setup(module: Any) -> tuple[Any, ...]:
    return _case_stable_diffusion_sd3_qk_norm_joint_attention_shape_setup(
        module,
        image_tokens=64,
        text_tokens=77,
        width=320,
    )


def _stable_diffusion_sd3_qk_norm_joint_attention_path(
    module: Any,
    hidden: Any,
    context: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    context_q_weight: Any,
    context_k_weight: Any,
    context_v_weight: Any,
    bias: Any,
    context_bias: Any,
    q_norm_weight: Any,
    k_norm_weight: Any,
    out_weight: Any,
    out_bias: Any,
    context_out_weight: Any,
    context_out_bias: Any,
) -> Any:
    batch, image_tokens, width = hidden.shape
    heads = 8
    if width % heads != 0:
        raise ValueError("SD3 joint attention width must be divisible by the head count")
    head_dim = width // heads
    text_tokens = context.shape[1]
    query = module.nn.functional.linear(hidden, q_weight, bias).reshape(batch, image_tokens, heads, head_dim).transpose(
        1,
        2,
    )
    key = module.nn.functional.linear(hidden, k_weight, bias).reshape(batch, image_tokens, heads, head_dim).transpose(
        1,
        2,
    )
    value = module.nn.functional.linear(hidden, v_weight, bias).reshape(batch, image_tokens, heads, head_dim).transpose(
        1,
        2,
    )
    context_query = module.nn.functional.linear(context, context_q_weight, context_bias).reshape(
        batch,
        text_tokens,
        heads,
        head_dim,
    ).transpose(1, 2)
    context_key = module.nn.functional.linear(context, context_k_weight, context_bias).reshape(
        batch,
        text_tokens,
        heads,
        head_dim,
    ).transpose(1, 2)
    context_value = module.nn.functional.linear(context, context_v_weight, context_bias).reshape(
        batch,
        text_tokens,
        heads,
        head_dim,
    ).transpose(1, 2)

    query = module.nn.functional.rms_norm(query, (head_dim,), q_norm_weight, eps=1e-6)
    key = module.nn.functional.rms_norm(key, (head_dim,), k_norm_weight, eps=1e-6)
    context_query = module.nn.functional.rms_norm(context_query, (head_dim,), q_norm_weight, eps=1e-6)
    context_key = module.nn.functional.rms_norm(context_key, (head_dim,), k_norm_weight, eps=1e-6)

    joint_query = module.cat([context_query, query], dim=2)
    joint_key = module.cat([context_key, key], dim=2)
    joint_value = module.cat([context_value, value], dim=2)
    attended = module.nn.functional.scaled_dot_product_attention(joint_query, joint_key, joint_value, scale=0.125)
    attended_context, attended_hidden = module.split(attended, [text_tokens, image_tokens], dim=2)
    attended_hidden = attended_hidden.transpose(1, 2).reshape(batch, image_tokens, width)
    attended_context = attended_context.transpose(1, 2).reshape(batch, text_tokens, width)
    return (
        module.nn.functional.linear(attended_hidden, out_weight, out_bias),
        module.nn.functional.linear(attended_context, context_out_weight, context_out_bias),
    )


def _case_stable_diffusion_sd3_rotary_joint_attention_setup(module: Any) -> tuple[Any, ...]:
    base_args = _case_stable_diffusion_sd3_qk_norm_joint_attention_setup(module)
    image_tokens, head_dim = 64, 40
    half_dim = head_dim // 2
    positions = module.arange(0, image_tokens, dtype=module.float32)
    exponent = (
        -module.log(module.tensor(10000.0, dtype=module.float32))
        * module.arange(0, half_dim, dtype=module.float32)
        / half_dim
    )
    frequencies = module.exp(exponent)
    angles = positions.reshape(1, 1, image_tokens, 1) * frequencies.reshape(1, 1, 1, half_dim)
    rotary_cos = module.cat([module.cos(angles), module.cos(angles)], dim=-1).to(dtype=module.float16)
    rotary_sin = module.cat([module.sin(angles), module.sin(angles)], dim=-1).to(dtype=module.float16)
    return (*base_args, rotary_cos, rotary_sin)


def _stable_diffusion_apply_rotary_embedding(module: Any, tensor: Any, rotary_cos: Any, rotary_sin: Any) -> Any:
    first, second = tensor.chunk(2, dim=-1)
    rotated = module.cat([second * -1.0, first], dim=-1)
    return tensor * rotary_cos + rotated * rotary_sin


def _case_stable_diffusion_sd3_rectangular_rotary_joint_attention_setup(module: Any) -> tuple[Any, ...]:
    grid_h, grid_w, width, heads = 8, 12, 320, 8
    image_tokens = grid_h * grid_w
    head_dim = width // heads
    axis_dim = head_dim // 2
    axis_half_dim = axis_dim // 2
    base_args = _case_stable_diffusion_sd3_qk_norm_joint_attention_shape_setup(
        module,
        image_tokens=image_tokens,
        text_tokens=77,
        width=width,
    )
    row_positions = module.tensor([row for row in range(grid_h) for _ in range(grid_w)], dtype=module.float32)
    col_positions = module.tensor([col for _ in range(grid_h) for col in range(grid_w)], dtype=module.float32)
    exponent = (
        -module.log(module.tensor(10000.0, dtype=module.float32))
        * module.arange(0, axis_half_dim, dtype=module.float32)
        / axis_half_dim
    )
    frequencies = module.exp(exponent)
    row_angles = row_positions.reshape(1, 1, image_tokens, 1) * frequencies.reshape(1, 1, 1, axis_half_dim)
    col_angles = col_positions.reshape(1, 1, image_tokens, 1) * frequencies.reshape(1, 1, 1, axis_half_dim)
    row_cos = module.cat([module.cos(row_angles), module.cos(row_angles)], dim=-1)
    row_sin = module.cat([module.sin(row_angles), module.sin(row_angles)], dim=-1)
    col_cos = module.cat([module.cos(col_angles), module.cos(col_angles)], dim=-1)
    col_sin = module.cat([module.sin(col_angles), module.sin(col_angles)], dim=-1)
    rotary_cos = module.cat([row_cos, col_cos], dim=-1).to(dtype=module.float16)
    rotary_sin = module.cat([row_sin, col_sin], dim=-1).to(dtype=module.float16)
    return (*base_args, rotary_cos, rotary_sin)


def _stable_diffusion_apply_2d_rotary_embedding(module: Any, tensor: Any, rotary_cos: Any, rotary_sin: Any) -> Any:
    if tensor.shape[-1] % 4 != 0:
        raise ValueError("SD3 2D rotary head dimension must be divisible by four")
    row_first, row_second, col_first, col_second = tensor.chunk(4, dim=-1)
    rotated = module.cat([row_second * -1.0, row_first, col_second * -1.0, col_first], dim=-1)
    return tensor * rotary_cos + rotated * rotary_sin


def _stable_diffusion_sd3_rotary_joint_attention_path(
    module: Any,
    hidden: Any,
    context: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    context_q_weight: Any,
    context_k_weight: Any,
    context_v_weight: Any,
    bias: Any,
    context_bias: Any,
    q_norm_weight: Any,
    k_norm_weight: Any,
    out_weight: Any,
    out_bias: Any,
    context_out_weight: Any,
    context_out_bias: Any,
    rotary_cos: Any,
    rotary_sin: Any,
) -> Any:
    batch, image_tokens, width = hidden.shape
    heads = 8
    if width % heads != 0:
        raise ValueError("SD3 rotary joint attention width must be divisible by the head count")
    head_dim = width // heads
    text_tokens = context.shape[1]
    query = module.nn.functional.linear(hidden, q_weight, bias).reshape(batch, image_tokens, heads, head_dim).transpose(
        1,
        2,
    )
    key = module.nn.functional.linear(hidden, k_weight, bias).reshape(batch, image_tokens, heads, head_dim).transpose(
        1,
        2,
    )
    value = module.nn.functional.linear(hidden, v_weight, bias).reshape(batch, image_tokens, heads, head_dim).transpose(
        1,
        2,
    )
    context_query = module.nn.functional.linear(context, context_q_weight, context_bias).reshape(
        batch,
        text_tokens,
        heads,
        head_dim,
    ).transpose(1, 2)
    context_key = module.nn.functional.linear(context, context_k_weight, context_bias).reshape(
        batch,
        text_tokens,
        heads,
        head_dim,
    ).transpose(1, 2)
    context_value = module.nn.functional.linear(context, context_v_weight, context_bias).reshape(
        batch,
        text_tokens,
        heads,
        head_dim,
    ).transpose(1, 2)

    query = module.nn.functional.rms_norm(query, (head_dim,), q_norm_weight, eps=1e-6)
    key = module.nn.functional.rms_norm(key, (head_dim,), k_norm_weight, eps=1e-6)
    query = _stable_diffusion_apply_rotary_embedding(module, query, rotary_cos, rotary_sin)
    key = _stable_diffusion_apply_rotary_embedding(module, key, rotary_cos, rotary_sin)
    context_query = module.nn.functional.rms_norm(context_query, (head_dim,), q_norm_weight, eps=1e-6)
    context_key = module.nn.functional.rms_norm(context_key, (head_dim,), k_norm_weight, eps=1e-6)

    joint_query = module.cat([context_query, query], dim=2)
    joint_key = module.cat([context_key, key], dim=2)
    joint_value = module.cat([context_value, value], dim=2)
    attended = module.nn.functional.scaled_dot_product_attention(joint_query, joint_key, joint_value, scale=0.125)
    attended_context, attended_hidden = module.split(attended, [text_tokens, image_tokens], dim=2)
    attended_hidden = attended_hidden.transpose(1, 2).reshape(batch, image_tokens, width)
    attended_context = attended_context.transpose(1, 2).reshape(batch, text_tokens, width)
    return (
        module.nn.functional.linear(attended_hidden, out_weight, out_bias),
        module.nn.functional.linear(attended_context, context_out_weight, context_out_bias),
    )


def _stable_diffusion_sd3_rectangular_rotary_joint_attention_path(
    module: Any,
    hidden: Any,
    context: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    context_q_weight: Any,
    context_k_weight: Any,
    context_v_weight: Any,
    bias: Any,
    context_bias: Any,
    q_norm_weight: Any,
    k_norm_weight: Any,
    out_weight: Any,
    out_bias: Any,
    context_out_weight: Any,
    context_out_bias: Any,
    rotary_cos: Any,
    rotary_sin: Any,
) -> Any:
    batch, image_tokens, width = hidden.shape
    heads = 8
    if width % heads != 0:
        raise ValueError("SD3 rectangular rotary joint attention width must be divisible by the head count")
    head_dim = width // heads
    text_tokens = context.shape[1]
    query = module.nn.functional.linear(hidden, q_weight, bias).reshape(batch, image_tokens, heads, head_dim).transpose(
        1,
        2,
    )
    key = module.nn.functional.linear(hidden, k_weight, bias).reshape(batch, image_tokens, heads, head_dim).transpose(
        1,
        2,
    )
    value = module.nn.functional.linear(hidden, v_weight, bias).reshape(batch, image_tokens, heads, head_dim).transpose(
        1,
        2,
    )
    context_query = module.nn.functional.linear(context, context_q_weight, context_bias).reshape(
        batch,
        text_tokens,
        heads,
        head_dim,
    ).transpose(1, 2)
    context_key = module.nn.functional.linear(context, context_k_weight, context_bias).reshape(
        batch,
        text_tokens,
        heads,
        head_dim,
    ).transpose(1, 2)
    context_value = module.nn.functional.linear(context, context_v_weight, context_bias).reshape(
        batch,
        text_tokens,
        heads,
        head_dim,
    ).transpose(1, 2)

    query = module.nn.functional.rms_norm(query, (head_dim,), q_norm_weight, eps=1e-6)
    key = module.nn.functional.rms_norm(key, (head_dim,), k_norm_weight, eps=1e-6)
    query = _stable_diffusion_apply_2d_rotary_embedding(module, query, rotary_cos, rotary_sin)
    key = _stable_diffusion_apply_2d_rotary_embedding(module, key, rotary_cos, rotary_sin)
    context_query = module.nn.functional.rms_norm(context_query, (head_dim,), q_norm_weight, eps=1e-6)
    context_key = module.nn.functional.rms_norm(context_key, (head_dim,), k_norm_weight, eps=1e-6)

    joint_query = module.cat([context_query, query], dim=2)
    joint_key = module.cat([context_key, key], dim=2)
    joint_value = module.cat([context_value, value], dim=2)
    attended = module.nn.functional.scaled_dot_product_attention(joint_query, joint_key, joint_value, scale=0.125)
    attended_context, attended_hidden = module.split(attended, [text_tokens, image_tokens], dim=2)
    attended_hidden = attended_hidden.transpose(1, 2).reshape(batch, image_tokens, width)
    attended_context = attended_context.transpose(1, 2).reshape(batch, text_tokens, width)
    return (
        module.nn.functional.linear(attended_hidden, out_weight, out_bias),
        module.nn.functional.linear(attended_context, context_out_weight, context_out_bias),
    )


def _stable_diffusion_sd3_rectangular_rotary_joint_transformer_block_path(
    module: Any,
    hidden: Any,
    context: Any,
    hidden_norm_weight: Any,
    hidden_norm_bias: Any,
    context_norm_weight: Any,
    context_norm_bias: Any,
    mlp_norm_weight: Any,
    mlp_norm_bias: Any,
    hidden_modulation: Any,
    context_modulation: Any,
    qkv_weight: Any,
    qkv_bias: Any,
    context_qkv_weight: Any,
    context_qkv_bias: Any,
    out_weight: Any,
    out_bias: Any,
    context_out_weight: Any,
    context_out_bias: Any,
    ff_in_weight: Any,
    ff_in_bias: Any,
    ff_out_weight: Any,
    ff_out_bias: Any,
    q_norm_weight: Any,
    k_norm_weight: Any,
    rotary_cos: Any,
    rotary_sin: Any,
) -> Any:
    batch, image_tokens, width = hidden.shape
    heads = 8
    if width % heads != 0:
        raise ValueError("SD3 rectangular rotary joint transformer width must be divisible by the head count")
    head_dim = width // heads
    text_tokens = context.shape[1]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = hidden_modulation.chunk(6, dim=1)
    context_shift, context_scale, context_gate = context_modulation.chunk(3, dim=1)

    normed_hidden = module.nn.functional.layer_norm(hidden, (width,), hidden_norm_weight, hidden_norm_bias, eps=1e-5)
    normed_hidden = normed_hidden * (1.0 + scale_msa) + shift_msa
    normed_context = module.nn.functional.layer_norm(
        context,
        (width,),
        context_norm_weight,
        context_norm_bias,
        eps=1e-5,
    )
    normed_context = normed_context * (1.0 + context_scale) + context_shift

    qkv = module.nn.functional.linear(normed_hidden, qkv_weight, qkv_bias)
    query, key, value = qkv.chunk(3, dim=-1)
    context_qkv = module.nn.functional.linear(normed_context, context_qkv_weight, context_qkv_bias)
    context_query, context_key, context_value = context_qkv.chunk(3, dim=-1)

    query = query.reshape(batch, image_tokens, heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, image_tokens, heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, image_tokens, heads, head_dim).transpose(1, 2)
    context_query = context_query.reshape(batch, text_tokens, heads, head_dim).transpose(1, 2)
    context_key = context_key.reshape(batch, text_tokens, heads, head_dim).transpose(1, 2)
    context_value = context_value.reshape(batch, text_tokens, heads, head_dim).transpose(1, 2)

    query = module.nn.functional.rms_norm(query, (head_dim,), q_norm_weight, eps=1e-6)
    key = module.nn.functional.rms_norm(key, (head_dim,), k_norm_weight, eps=1e-6)
    query = _stable_diffusion_apply_2d_rotary_embedding(module, query, rotary_cos, rotary_sin)
    key = _stable_diffusion_apply_2d_rotary_embedding(module, key, rotary_cos, rotary_sin)
    context_query = module.nn.functional.rms_norm(context_query, (head_dim,), q_norm_weight, eps=1e-6)
    context_key = module.nn.functional.rms_norm(context_key, (head_dim,), k_norm_weight, eps=1e-6)

    joint_query = module.cat([context_query, query], dim=2)
    joint_key = module.cat([context_key, key], dim=2)
    joint_value = module.cat([context_value, value], dim=2)
    attended = module.nn.functional.scaled_dot_product_attention(joint_query, joint_key, joint_value, scale=0.125)
    attended_context, attended_hidden = module.split(attended, [text_tokens, image_tokens], dim=2)

    attended_hidden = attended_hidden.transpose(1, 2).reshape(batch, image_tokens, width)
    attended_context = attended_context.transpose(1, 2).reshape(batch, text_tokens, width)
    hidden = hidden + gate_msa * module.nn.functional.linear(attended_hidden, out_weight, out_bias)
    context = context + context_gate * module.nn.functional.linear(attended_context, context_out_weight, context_out_bias)

    mlp_hidden = module.nn.functional.layer_norm(hidden, (width,), mlp_norm_weight, mlp_norm_bias, eps=1e-5)
    mlp_hidden = mlp_hidden * (1.0 + scale_mlp) + shift_mlp
    geglu = module.nn.functional.linear(mlp_hidden, ff_in_weight, ff_in_bias)
    values, gate = geglu.chunk(2, dim=-1)
    mlp_hidden = values * module.nn.functional.gelu(gate)
    hidden = hidden + gate_mlp * module.nn.functional.linear(mlp_hidden, ff_out_weight, ff_out_bias)
    return hidden, context


def _case_stable_diffusion_sd3_single_transformer_block_setup(module: Any) -> tuple[Any, ...]:
    batch, image_tokens, text_tokens, width, intermediate = 2, 64, 77, 320, 640
    hidden_base = module.arange(0, batch * image_tokens * width, dtype=module.float32).reshape(
        batch,
        image_tokens,
        width,
    )
    context_base = module.arange(0, batch * text_tokens * width, dtype=module.float32).reshape(
        batch,
        text_tokens,
        width,
    )
    hidden = (module.sin(hidden_base * 0.013) * 0.18).to(dtype=module.float16)
    context = (module.cos(context_base * 0.011) * 0.14).to(dtype=module.float16)

    attn_norm_weight = (0.82 + (module.arange(0, width, dtype=module.float32) % 251) / 2048.0).to(
        dtype=module.float16
    )
    attn_norm_bias = (((module.arange(0, width, dtype=module.float32) % 127) - 63.0) / 4096.0).to(
        dtype=module.float16
    )
    mlp_norm_weight = (0.86 + (module.arange(0, width, dtype=module.float32) % 257) / 2048.0).to(
        dtype=module.float16
    )
    mlp_norm_bias = (((module.arange(0, width, dtype=module.float32) % 137) - 68.0) / 4096.0).to(
        dtype=module.float16
    )
    modulation = (
        module.sin(module.arange(0, batch * 6 * width, dtype=module.float32).reshape(batch, 6, width) * 0.017)
        * 0.08
    ).to(dtype=module.float16)

    qkv_base = (
        module.arange(0, 3 * width * width, dtype=module.float32).reshape(3 * width, width) % 1543
    ) / 65536.0 - 0.012
    qkv_weight = qkv_base.to(dtype=module.float16)
    qkv_bias = (((module.arange(0, 3 * width, dtype=module.float32) % 181) - 90.0) / 8192.0).to(
        dtype=module.float16
    )
    q_norm_weight = (0.92 + (module.arange(0, 40, dtype=module.float32) % 37) / 512.0).to(dtype=module.float16)
    k_norm_weight = (0.88 + (module.arange(0, 40, dtype=module.float32) % 41) / 512.0).to(dtype=module.float16)
    out_weight = (
        (module.arange(0, width * width, dtype=module.float32).reshape(width, width) % 1237) / 65536.0 - 0.01
    ).to(dtype=module.float16)
    out_bias = (((module.arange(0, width, dtype=module.float32) % 149) - 74.0) / 8192.0).to(dtype=module.float16)

    ff_in_weight = (
        (module.arange(0, 2 * intermediate * width, dtype=module.float32).reshape(2 * intermediate, width) % 1559)
        / 65536.0
        - 0.012
    ).to(dtype=module.float16)
    ff_in_bias = (((module.arange(0, 2 * intermediate, dtype=module.float32) % 191) - 95.0) / 8192.0).to(
        dtype=module.float16
    )
    ff_out_weight = (
        (module.arange(0, width * intermediate, dtype=module.float32).reshape(width, intermediate) % 1567)
        / 65536.0
        - 0.012
    ).to(dtype=module.float16)
    ff_out_bias = (((module.arange(0, width, dtype=module.float32) % 157) - 78.0) / 8192.0).to(
        dtype=module.float16
    )
    return (
        hidden,
        context,
        attn_norm_weight,
        attn_norm_bias,
        mlp_norm_weight,
        mlp_norm_bias,
        modulation,
        qkv_weight,
        qkv_bias,
        q_norm_weight,
        k_norm_weight,
        out_weight,
        out_bias,
        ff_in_weight,
        ff_in_bias,
        ff_out_weight,
        ff_out_bias,
    )


def _stable_diffusion_sd3_single_transformer_block_path(
    module: Any,
    hidden: Any,
    context: Any,
    attn_norm_weight: Any,
    attn_norm_bias: Any,
    mlp_norm_weight: Any,
    mlp_norm_bias: Any,
    modulation: Any,
    qkv_weight: Any,
    qkv_bias: Any,
    q_norm_weight: Any,
    k_norm_weight: Any,
    out_weight: Any,
    out_bias: Any,
    ff_in_weight: Any,
    ff_in_bias: Any,
    ff_out_weight: Any,
    ff_out_bias: Any,
) -> Any:
    batch, image_tokens, width = hidden.shape
    heads = 8
    if width % heads != 0:
        raise ValueError("SD3 single transformer width must be divisible by the head count")
    head_dim = width // heads
    text_tokens = context.shape[1]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

    combined = module.cat([context, hidden], dim=1)
    normed = module.nn.functional.layer_norm(combined, (width,), attn_norm_weight, attn_norm_bias, eps=1e-5)
    normed = normed * (1.0 + scale_msa) + shift_msa
    qkv = module.nn.functional.linear(normed, qkv_weight, qkv_bias)
    query, key, value = qkv.chunk(3, dim=-1)
    token_count = text_tokens + image_tokens
    query = query.reshape(batch, token_count, heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, token_count, heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, token_count, heads, head_dim).transpose(1, 2)
    query = module.nn.functional.rms_norm(query, (head_dim,), q_norm_weight, eps=1e-6)
    key = module.nn.functional.rms_norm(key, (head_dim,), k_norm_weight, eps=1e-6)

    attended = module.nn.functional.scaled_dot_product_attention(query, key, value, scale=0.125)
    attended = attended.transpose(1, 2).reshape(batch, token_count, width)
    combined = combined + gate_msa * module.nn.functional.linear(attended, out_weight, out_bias)

    mlp_hidden = module.nn.functional.layer_norm(combined, (width,), mlp_norm_weight, mlp_norm_bias, eps=1e-5)
    mlp_hidden = mlp_hidden * (1.0 + scale_mlp) + shift_mlp
    geglu = module.nn.functional.linear(mlp_hidden, ff_in_weight, ff_in_bias)
    values, gate = geglu.chunk(2, dim=-1)
    mlp_hidden = values * module.nn.functional.gelu(gate)
    combined = combined + gate_mlp * module.nn.functional.linear(mlp_hidden, ff_out_weight, ff_out_bias)

    context_out, hidden_out = module.split(combined, [text_tokens, image_tokens], dim=1)
    return hidden_out, context_out


def _case_stable_diffusion_sd3_time_text_conditioning_setup(module: Any) -> tuple[Any, ...]:
    batch, time_width, pooled_width, out_width = 4, 256, 2048, 1280
    timesteps = module.tensor([999.0, 750.0, 500.0, 250.0], dtype=module.float32)
    pooled_base = module.arange(0, batch * pooled_width, dtype=module.float32).reshape(batch, pooled_width)
    pooled_prompt_embeds = (module.sin((pooled_base % 1543) * 0.005) * 0.2).to(dtype=module.float16)

    time_fc1_base = module.arange(0, out_width * time_width, dtype=module.float32).reshape(out_width, time_width)
    time_fc2_base = module.arange(0, out_width * out_width, dtype=module.float32).reshape(out_width, out_width)
    text_fc1_base = module.arange(0, out_width * pooled_width, dtype=module.float32).reshape(out_width, pooled_width)
    text_fc2_base = module.arange(0, out_width * out_width, dtype=module.float32).reshape(out_width, out_width)
    time_fc1_weight = ((time_fc1_base % 1543) / 65536.0 - 0.012).to(dtype=module.float16)
    time_fc2_weight = ((time_fc2_base % 1553) / 65536.0 - 0.012).to(dtype=module.float16)
    text_fc1_weight = ((text_fc1_base % 1567) / 65536.0 - 0.012).to(dtype=module.float16)
    text_fc2_weight = ((text_fc2_base % 1571) / 65536.0 - 0.012).to(dtype=module.float16)
    time_fc1_bias = ((module.arange(0, out_width, dtype=module.float32) % 173) / 8192.0 - 0.01).to(
        dtype=module.float16
    )
    time_fc2_bias = ((module.arange(0, out_width, dtype=module.float32) % 181) / 8192.0 - 0.011).to(
        dtype=module.float16
    )
    text_fc1_bias = ((module.arange(0, out_width, dtype=module.float32) % 191) / 8192.0 - 0.012).to(
        dtype=module.float16
    )
    text_fc2_bias = ((module.arange(0, out_width, dtype=module.float32) % 193) / 8192.0 - 0.012).to(
        dtype=module.float16
    )
    return (
        timesteps,
        pooled_prompt_embeds,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
    )


def _stable_diffusion_sd3_time_text_conditioning_path(
    module: Any,
    timesteps: Any,
    pooled_prompt_embeds: Any,
    time_fc1_weight: Any,
    time_fc1_bias: Any,
    time_fc2_weight: Any,
    time_fc2_bias: Any,
    text_fc1_weight: Any,
    text_fc1_bias: Any,
    text_fc2_weight: Any,
    text_fc2_bias: Any,
) -> Any:
    batch, time_width = timesteps.shape[0], time_fc1_weight.shape[1]
    half_width = time_width // 2
    exponent = (
        -module.log(module.tensor(10000.0, dtype=module.float32))
        * module.arange(0, half_width, dtype=module.float32)
        / half_width
    )
    frequencies = module.exp(exponent)
    angles = timesteps.reshape(batch, 1) * frequencies.reshape(1, half_width)
    time_embed = module.cat([module.cos(angles), module.sin(angles)], dim=-1).to(dtype=module.float16)
    time_hidden = module.nn.functional.linear(time_embed, time_fc1_weight, time_fc1_bias)
    time_hidden = module.nn.functional.silu(time_hidden)
    time_hidden = module.nn.functional.linear(time_hidden, time_fc2_weight, time_fc2_bias)

    text_hidden = module.nn.functional.linear(pooled_prompt_embeds, text_fc1_weight, text_fc1_bias)
    text_hidden = module.nn.functional.silu(text_hidden)
    text_hidden = module.nn.functional.linear(text_hidden, text_fc2_weight, text_fc2_bias)
    return time_hidden + text_hidden


def _case_stable_diffusion_sd3_pooled_modulation_setup(
    module: Any,
    batch: int,
    pooled_width: int,
    cond_width: int,
    width: int,
    timestep_values: tuple[float, ...],
) -> tuple[Any, ...]:
    if len(timestep_values) != batch:
        raise ValueError("SD3 pooled modulation timestep count must match the batch size")
    timesteps = module.tensor(list(timestep_values), dtype=module.float32)
    pooled_base = module.arange(0, batch * pooled_width, dtype=module.float32).reshape(batch, pooled_width)
    pooled_prompt_embeds = (module.sin((pooled_base % 1543) * 0.005) * 0.2).to(dtype=module.float16)

    time_width = 256
    time_fc1_base = module.arange(0, cond_width * time_width, dtype=module.float32).reshape(cond_width, time_width)
    time_fc2_base = module.arange(0, cond_width * cond_width, dtype=module.float32).reshape(cond_width, cond_width)
    text_fc1_base = module.arange(0, cond_width * pooled_width, dtype=module.float32).reshape(cond_width, pooled_width)
    text_fc2_base = module.arange(0, cond_width * cond_width, dtype=module.float32).reshape(cond_width, cond_width)
    time_fc1_weight = ((time_fc1_base % 1543) / 65536.0 - 0.012).to(dtype=module.float16)
    time_fc2_weight = ((time_fc2_base % 1553) / 65536.0 - 0.012).to(dtype=module.float16)
    text_fc1_weight = ((text_fc1_base % 1567) / 65536.0 - 0.012).to(dtype=module.float16)
    text_fc2_weight = ((text_fc2_base % 1571) / 65536.0 - 0.012).to(dtype=module.float16)
    time_fc1_bias = ((module.arange(0, cond_width, dtype=module.float32) % 173) / 8192.0 - 0.01).to(
        dtype=module.float16
    )
    time_fc2_bias = ((module.arange(0, cond_width, dtype=module.float32) % 181) / 8192.0 - 0.011).to(
        dtype=module.float16
    )
    text_fc1_bias = ((module.arange(0, cond_width, dtype=module.float32) % 191) / 8192.0 - 0.012).to(
        dtype=module.float16
    )
    text_fc2_bias = ((module.arange(0, cond_width, dtype=module.float32) % 193) / 8192.0 - 0.012).to(
        dtype=module.float16
    )

    hidden_mod_base = module.arange(0, 6 * width * cond_width, dtype=module.float32).reshape(6 * width, cond_width)
    context_mod_base = (
        module.arange(0, 3 * width * cond_width, dtype=module.float32).reshape(3 * width, cond_width) + 17
    )
    single_mod_base = (
        module.arange(0, 6 * width * cond_width, dtype=module.float32).reshape(6 * width, cond_width) + 29
    )
    final_mod_base = (
        module.arange(0, 2 * width * cond_width, dtype=module.float32).reshape(2 * width, cond_width) + 43
    )
    hidden_mod_weight = ((hidden_mod_base % 1549) / 131072.0 - 0.006).to(dtype=module.float16)
    context_mod_weight = ((context_mod_base % 1553) / 131072.0 - 0.006).to(dtype=module.float16)
    single_mod_weight = ((single_mod_base % 1559) / 131072.0 - 0.006).to(dtype=module.float16)
    final_mod_weight = ((final_mod_base % 1567) / 131072.0 - 0.006).to(dtype=module.float16)
    hidden_mod_bias = (((module.arange(0, 6 * width, dtype=module.float32) % 181) - 90.0) / 16384.0).to(
        dtype=module.float16
    )
    context_mod_bias = (((module.arange(0, 3 * width, dtype=module.float32) % 173) - 86.0) / 16384.0).to(
        dtype=module.float16
    )
    single_mod_bias = (((module.arange(0, 6 * width, dtype=module.float32) % 191) - 95.0) / 16384.0).to(
        dtype=module.float16
    )
    final_mod_bias = (((module.arange(0, 2 * width, dtype=module.float32) % 157) - 78.0) / 16384.0).to(
        dtype=module.float16
    )

    return (
        timesteps,
        pooled_prompt_embeds,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
        hidden_mod_weight,
        hidden_mod_bias,
        context_mod_weight,
        context_mod_bias,
        single_mod_weight,
        single_mod_bias,
        final_mod_weight,
        final_mod_bias,
    )


def _case_stable_diffusion_sd3_pooled_transformer_stack_setup(module: Any) -> tuple[Any, ...]:
    batch, pooled_width, cond_width, width = 2, 2048, 1280, 320
    latent, patch_weight, patch_bias, pos_embed = _case_stable_diffusion_sd3_mini_transformer_stack_setup(module)[:4]
    joint_full_args = _case_stable_diffusion_sd3_joint_transformer_block_setup(module)
    single_full_args = _case_stable_diffusion_sd3_single_transformer_block_setup(module)
    unpatchify_full_args = _case_stable_diffusion_sd3_unpatchify_projection_setup(module)
    modulation_args = _case_stable_diffusion_sd3_pooled_modulation_setup(
        module,
        batch,
        pooled_width,
        cond_width,
        width,
        (999.0, 333.0),
    )

    return (
        latent,
        patch_weight,
        patch_bias,
        pos_embed,
        joint_full_args[1],
        joint_full_args[2:],
        single_full_args[2:],
        unpatchify_full_args[1:],
        *modulation_args,
    )


def _stable_diffusion_sd3_pooled_transformer_stack_path(
    module: Any,
    latent: Any,
    patch_weight: Any,
    patch_bias: Any,
    pos_embed: Any,
    context: Any,
    joint_args: Any,
    single_args: Any,
    unpatchify_args: Any,
    timesteps: Any,
    pooled_prompt_embeds: Any,
    time_fc1_weight: Any,
    time_fc1_bias: Any,
    time_fc2_weight: Any,
    time_fc2_bias: Any,
    text_fc1_weight: Any,
    text_fc1_bias: Any,
    text_fc2_weight: Any,
    text_fc2_bias: Any,
    hidden_mod_weight: Any,
    hidden_mod_bias: Any,
    context_mod_weight: Any,
    context_mod_bias: Any,
    single_mod_weight: Any,
    single_mod_bias: Any,
    final_mod_weight: Any,
    final_mod_bias: Any,
) -> Any:
    batch, width = 2, 320
    conditioning = _stable_diffusion_sd3_time_text_conditioning_path(
        module,
        timesteps,
        pooled_prompt_embeds,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
    )
    hidden_modulation = module.nn.functional.linear(conditioning, hidden_mod_weight, hidden_mod_bias).reshape(
        batch,
        6,
        width,
    )
    context_modulation = module.nn.functional.linear(conditioning, context_mod_weight, context_mod_bias).reshape(
        batch,
        3,
        width,
    )
    single_modulation = module.nn.functional.linear(conditioning, single_mod_weight, single_mod_bias).reshape(
        batch,
        6,
        width,
    )
    final_modulation = module.nn.functional.linear(conditioning, final_mod_weight, final_mod_bias).reshape(
        batch,
        2,
        width,
    )

    hidden = _stable_diffusion_sd3_patch_embed_path(module, latent, patch_weight, patch_bias, pos_embed)
    hidden, context = _stable_diffusion_sd3_joint_transformer_block_path(
        module,
        hidden,
        context,
        *joint_args[:6],
        hidden_modulation,
        context_modulation,
        *joint_args[8:],
    )
    hidden, context = _stable_diffusion_sd3_single_transformer_block_path(
        module,
        hidden,
        context,
        *single_args[:4],
        single_modulation,
        *single_args[5:],
    )
    return _stable_diffusion_sd3_unpatchify_projection_path(
        module,
        hidden,
        final_modulation,
        *unpatchify_args[1:],
    )


def _case_stable_diffusion_sd3_large_controlnet_transformer_stack_setup(module: Any) -> tuple[Any, ...]:
    batch, token_count, width = 2, 256, 320
    latent, patch_weight, patch_bias, pos_embed = _case_stable_diffusion_sd3_patch_embed_setup(module)
    joint_args = _case_stable_diffusion_sd3_joint_transformer_block_setup(module)
    single_args = _case_stable_diffusion_sd3_single_transformer_block_setup(module)
    unpatchify_args = _case_stable_diffusion_sd3_large_unpatchify_projection_setup(module)
    residual_base = module.arange(0, batch * token_count * width, dtype=module.float32).reshape(
        batch,
        token_count,
        width,
    )
    block_residual = (module.cos(residual_base * 0.017) * 0.025).to(dtype=module.float16)
    controlnet_keep = module.tensor([0.7, 0.35], dtype=module.float32)
    return (
        latent,
        patch_weight,
        patch_bias,
        pos_embed,
        joint_args[1],
        joint_args[2:],
        single_args[2:],
        unpatchify_args[1:],
        block_residual,
        controlnet_keep,
    )


def _stable_diffusion_sd3_large_controlnet_transformer_stack_path(
    module: Any,
    latent: Any,
    patch_weight: Any,
    patch_bias: Any,
    pos_embed: Any,
    context: Any,
    joint_args: Any,
    single_args: Any,
    unpatchify_args: Any,
    block_residual: Any,
    controlnet_keep: Any,
) -> Any:
    hidden = _stable_diffusion_sd3_patch_embed_path(module, latent, patch_weight, patch_bias, pos_embed)
    hidden, context = _stable_diffusion_sd3_joint_transformer_block_path(module, hidden, context, *joint_args)
    hidden = hidden + block_residual * controlnet_keep[0].to(dtype=hidden.dtype)
    hidden, context = _stable_diffusion_sd3_single_transformer_block_path(module, hidden, context, *single_args)
    hidden = hidden + block_residual * controlnet_keep[1].to(dtype=hidden.dtype)
    return _stable_diffusion_sd3_large_unpatchify_projection_path(module, hidden, *unpatchify_args)


def _stable_diffusion_sd3_shift_layer_args(args: tuple[Any, ...], amount: float) -> tuple[Any, ...]:
    return tuple((arg + amount).to(dtype=arg.dtype) for arg in args)


def _case_stable_diffusion_sd3_multi_controlnet_transformer_stack_setup(module: Any) -> tuple[Any, ...]:
    batch, token_count, width = 2, 256, 320
    latent, patch_weight, patch_bias, pos_embed = _case_stable_diffusion_sd3_patch_embed_setup(module)
    joint_args = _case_stable_diffusion_sd3_joint_transformer_block_setup(module)
    single_args = _case_stable_diffusion_sd3_single_transformer_block_setup(module)
    unpatchify_args = _case_stable_diffusion_sd3_large_unpatchify_projection_setup(module)
    joint_layers = tuple(
        _stable_diffusion_sd3_shift_layer_args(joint_args[2:], 0.00018 * float(layer + 1)) for layer in range(2)
    )
    single_layers = tuple(
        _stable_diffusion_sd3_shift_layer_args(single_args[2:], -0.00014 * float(layer + 1)) for layer in range(2)
    )
    residual_base = module.arange(0, batch * token_count * width, dtype=module.float32).reshape(
        batch,
        token_count,
        width,
    )
    block_residuals = tuple(
        (module.cos((residual_base + float(layer * 97)) * (0.013 + float(layer) * 0.001)) * (0.018 - layer * 0.002)).to(
            dtype=module.float16
        )
        for layer in range(4)
    )
    controlnet_keep = module.tensor([0.85, 0.65, 0.45, 0.25], dtype=module.float32)
    return (
        latent,
        patch_weight,
        patch_bias,
        pos_embed,
        joint_args[1],
        joint_layers,
        single_layers,
        unpatchify_args[1:],
        block_residuals,
        controlnet_keep,
    )


def _stable_diffusion_sd3_multi_controlnet_transformer_stack_path(
    module: Any,
    latent: Any,
    patch_weight: Any,
    patch_bias: Any,
    pos_embed: Any,
    context: Any,
    joint_layers: Any,
    single_layers: Any,
    unpatchify_args: Any,
    block_residuals: Any,
    controlnet_keep: Any,
) -> Any:
    hidden = _stable_diffusion_sd3_patch_embed_path(module, latent, patch_weight, patch_bias, pos_embed)
    residual_index = 0
    for joint_args in joint_layers:
        hidden, context = _stable_diffusion_sd3_joint_transformer_block_path(module, hidden, context, *joint_args)
        hidden = hidden + block_residuals[residual_index] * controlnet_keep[residual_index].to(dtype=hidden.dtype)
        residual_index += 1
    for single_args in single_layers:
        hidden, context = _stable_diffusion_sd3_single_transformer_block_path(module, hidden, context, *single_args)
        hidden = hidden + block_residuals[residual_index] * controlnet_keep[residual_index].to(dtype=hidden.dtype)
        residual_index += 1
    return _stable_diffusion_sd3_large_unpatchify_projection_path(module, hidden, *unpatchify_args)


def _case_stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_setup(module: Any) -> tuple[Any, ...]:
    batch, token_count, pooled_width, cond_width, width = 2, 256, 2048, 1280, 320
    latent, patch_weight, patch_bias, pos_embed = _case_stable_diffusion_sd3_patch_embed_setup(module)
    joint_args = _case_stable_diffusion_sd3_joint_transformer_block_setup(module)
    single_args = _case_stable_diffusion_sd3_single_transformer_block_setup(module)
    unpatchify_args = _case_stable_diffusion_sd3_large_unpatchify_projection_setup(module)
    joint_layers = tuple(
        _stable_diffusion_sd3_shift_layer_args(joint_args[2:], 0.00018 * float(layer + 1)) for layer in range(2)
    )
    single_layers = tuple(
        _stable_diffusion_sd3_shift_layer_args(single_args[2:], -0.00014 * float(layer + 1)) for layer in range(2)
    )

    modulation_args = _case_stable_diffusion_sd3_pooled_modulation_setup(
        module,
        batch,
        pooled_width,
        cond_width,
        width,
        (999.0, 333.0),
    )
    conditioning_args = modulation_args[:10]
    hidden_mod_weight, hidden_mod_bias = modulation_args[10], modulation_args[11]
    context_mod_weight, context_mod_bias = modulation_args[12], modulation_args[13]
    single_mod_weight, single_mod_bias = modulation_args[14], modulation_args[15]
    final_mod_weight, final_mod_bias = modulation_args[16], modulation_args[17]
    joint_mod_layers = tuple(
        (
            *_stable_diffusion_sd3_shift_layer_args(
                (hidden_mod_weight, hidden_mod_bias),
                0.000025 * float(layer + 1),
            ),
            *_stable_diffusion_sd3_shift_layer_args(
                (context_mod_weight, context_mod_bias),
                -0.000018 * float(layer + 1),
            ),
        )
        for layer in range(2)
    )
    single_mod_layers = tuple(
        _stable_diffusion_sd3_shift_layer_args(
            (single_mod_weight, single_mod_bias),
            0.000021 * float(layer + 1),
        )
        for layer in range(2)
    )

    residual_base = module.arange(0, batch * token_count * width, dtype=module.float32).reshape(
        batch,
        token_count,
        width,
    )
    block_residuals = tuple(
        (module.sin((residual_base + float(layer * 131)) * (0.011 + float(layer) * 0.001)) * (0.02 - layer * 0.002)).to(
            dtype=module.float16
        )
        for layer in range(4)
    )
    controlnet_keep = module.tensor([0.9, 0.7, 0.5, 0.3], dtype=module.float32)
    return (
        latent,
        patch_weight,
        patch_bias,
        pos_embed,
        joint_args[1],
        joint_layers,
        single_layers,
        unpatchify_args[1:],
        block_residuals,
        controlnet_keep,
        *conditioning_args,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
    )


def _stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_path(
    module: Any,
    latent: Any,
    patch_weight: Any,
    patch_bias: Any,
    pos_embed: Any,
    context: Any,
    joint_layers: Any,
    single_layers: Any,
    unpatchify_args: Any,
    block_residuals: Any,
    controlnet_keep: Any,
    timesteps: Any,
    pooled_prompt_embeds: Any,
    time_fc1_weight: Any,
    time_fc1_bias: Any,
    time_fc2_weight: Any,
    time_fc2_bias: Any,
    text_fc1_weight: Any,
    text_fc1_bias: Any,
    text_fc2_weight: Any,
    text_fc2_bias: Any,
    joint_mod_layers: Any,
    single_mod_layers: Any,
    final_mod_weight: Any,
    final_mod_bias: Any,
) -> Any:
    batch, width = 2, 320
    conditioning = _stable_diffusion_sd3_time_text_conditioning_path(
        module,
        timesteps,
        pooled_prompt_embeds,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
    )

    hidden = _stable_diffusion_sd3_patch_embed_path(module, latent, patch_weight, patch_bias, pos_embed)
    residual_index = 0
    for joint_args, joint_mod_args in zip(joint_layers, joint_mod_layers):
        hidden_mod_weight, hidden_mod_bias, context_mod_weight, context_mod_bias = joint_mod_args
        hidden_modulation = module.nn.functional.linear(conditioning, hidden_mod_weight, hidden_mod_bias).reshape(
            batch,
            6,
            width,
        )
        context_modulation = module.nn.functional.linear(conditioning, context_mod_weight, context_mod_bias).reshape(
            batch,
            3,
            width,
        )
        hidden, context = _stable_diffusion_sd3_joint_transformer_block_path(
            module,
            hidden,
            context,
            *joint_args[:6],
            hidden_modulation,
            context_modulation,
            *joint_args[8:],
        )
        hidden = hidden + block_residuals[residual_index] * controlnet_keep[residual_index].to(dtype=hidden.dtype)
        residual_index += 1

    for single_args, single_mod_args in zip(single_layers, single_mod_layers):
        single_mod_weight, single_mod_bias = single_mod_args
        single_modulation = module.nn.functional.linear(conditioning, single_mod_weight, single_mod_bias).reshape(
            batch,
            6,
            width,
        )
        hidden, context = _stable_diffusion_sd3_single_transformer_block_path(
            module,
            hidden,
            context,
            *single_args[:4],
            single_modulation,
            *single_args[5:],
        )
        hidden = hidden + block_residuals[residual_index] * controlnet_keep[residual_index].to(dtype=hidden.dtype)
        residual_index += 1

    final_modulation = module.nn.functional.linear(conditioning, final_mod_weight, final_mod_bias).reshape(
        batch,
        2,
        width,
    )
    return _stable_diffusion_sd3_large_unpatchify_projection_path(
        module,
        hidden,
        final_modulation,
        *unpatchify_args[1:],
    )


def _case_stable_diffusion_sd3_flowmatch_cfg_step_setup(module: Any) -> tuple[Any, ...]:
    batch, channels, height, width = 2, 16, 16, 16
    text_tokens, text_width, pooled_width = 77, 320, 2048
    latent_base = module.arange(0, batch * channels * height * width, dtype=module.float32).reshape(
        batch,
        channels,
        height,
        width,
    )
    latents = (module.sin(latent_base * 0.007) * 0.8).to(dtype=module.float16)
    latents = latents.contiguous(memory_format=module.channels_last)

    prompt_base = module.arange(0, batch * text_tokens * text_width, dtype=module.float32).reshape(
        batch,
        text_tokens,
        text_width,
    )
    negative_prompt_embeds = (module.cos(prompt_base * 0.011) * 0.13).to(dtype=module.float16)
    prompt_embeds = (module.sin(prompt_base * 0.013) * 0.15).to(dtype=module.float16)
    pooled_base = module.arange(0, batch * pooled_width, dtype=module.float32).reshape(batch, pooled_width)
    negative_pooled_prompt_embeds = (module.cos((pooled_base % 1543) * 0.005) * 0.18).to(dtype=module.float16)
    pooled_prompt_embeds = (module.sin((pooled_base % 1553) * 0.005) * 0.2).to(dtype=module.float16)
    sigmas = module.tensor([1.0, 0.65, 0.25, 0.0], dtype=module.float32)
    index = module.tensor(1, dtype=module.long)
    return (
        latents,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        index,
        5.5,
    )


def _stable_diffusion_sd3_flowmatch_cfg_step_path(
    module: Any,
    latents: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    index: Any,
    guidance_scale: float,
) -> Any:
    latent_model_input = module.cat([latents, latents], dim=0)
    prompt_conditioning = module.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_conditioning = module.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    sigma = sigmas[index]
    scaled_input = latent_model_input / (sigma.reshape(1, 1, 1, 1).to(dtype=latents.dtype) + 1.0)
    prompt_shift = module.mean(prompt_conditioning, dim=(1, 2)).reshape(latent_model_input.shape[0], 1, 1, 1)
    pooled_shift = module.mean(pooled_conditioning, dim=1).reshape(latent_model_input.shape[0], 1, 1, 1)
    timestep_shift = sigma.reshape(1, 1, 1, 1).to(dtype=latents.dtype) * 0.0001
    model_output = (
        scaled_input * 0.72
        + prompt_shift.to(dtype=latents.dtype) * 0.003
        + pooled_shift.to(dtype=latents.dtype) * 0.002
        + timestep_shift
    )
    noise_pred_uncond, noise_pred_text = model_output.chunk(2, dim=0)
    guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    dt = (sigmas[index + 1] - sigmas[index]).reshape(1, 1, 1, 1).to(dtype=latents.dtype)
    return (latents + guided * dt).contiguous(memory_format=module.channels_last)


def _case_stable_diffusion_sd3_pooled_controlnet_denoising_step_setup(module: Any) -> tuple[Any, ...]:
    stack_args = _case_stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_setup(module)
    latent, patch_weight, patch_bias, pos_embed = stack_args[:4]
    context = stack_args[4]
    joint_layers, single_layers, unpatchify_args = stack_args[5:8]
    block_residuals, controlnet_keep = stack_args[8:10]
    pooled_prompt_embeds = stack_args[11]
    conditioning_weights = stack_args[12:20]
    joint_mod_layers, single_mod_layers, final_mod_weight, final_mod_bias = stack_args[20:24]

    latent_uncond, latent_cond = latent.chunk(2, dim=0)
    latents = (latent_uncond * 0.7 + module.flip(latent_cond, (-1,)) * 0.3).contiguous(
        memory_format=module.channels_last
    )
    negative_prompt_embeds, prompt_embeds = context.chunk(2, dim=0)
    negative_prompt_embeds = (negative_prompt_embeds - 0.012).to(dtype=module.float16)
    prompt_embeds = (prompt_embeds + 0.015).to(dtype=module.float16)
    negative_pooled_prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds.chunk(2, dim=0)
    negative_pooled_prompt_embeds = (negative_pooled_prompt_embeds - 0.01).to(dtype=module.float16)
    pooled_prompt_embeds = (pooled_prompt_embeds + 0.012).to(dtype=module.float16)
    sigmas = module.tensor([1.0, 0.72, 0.28, 0.0], dtype=module.float32)
    index = module.tensor(1, dtype=module.long)
    model_args = (
        patch_weight,
        patch_bias,
        pos_embed,
        joint_layers,
        single_layers,
        unpatchify_args,
        block_residuals,
        controlnet_keep,
        *conditioning_weights,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
    )
    return (
        latents,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        index,
        5.5,
        model_args,
    )


def _stable_diffusion_sd3_pooled_controlnet_denoising_step_path(
    module: Any,
    latents: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    index: Any,
    guidance_scale: float,
    model_args: Any,
) -> Any:
    (
        patch_weight,
        patch_bias,
        pos_embed,
        joint_layers,
        single_layers,
        unpatchify_args,
        block_residuals,
        controlnet_keep,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
    ) = model_args

    latent_model_input = module.cat([latents, latents], dim=0).contiguous(memory_format=module.channels_last)
    context = module.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_conditioning = module.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    sigma = sigmas[index]
    timesteps = module.stack([sigma * 1000.0, sigma * 1000.0], dim=0)

    model_output = _stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_path(
        module,
        latent_model_input,
        patch_weight,
        patch_bias,
        pos_embed,
        context,
        joint_layers,
        single_layers,
        unpatchify_args,
        block_residuals,
        controlnet_keep,
        timesteps,
        pooled_conditioning,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
    )
    noise_pred_uncond, noise_pred_text = model_output.chunk(2, dim=0)
    guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    dt = (sigmas[index + 1] - sigma).reshape(1, 1, 1, 1).to(dtype=latents.dtype)
    return (latents + guided * dt).contiguous(memory_format=module.channels_last)


def _case_stable_diffusion_sd3_unpatchify_projection_setup(module: Any) -> tuple[Any, ...]:
    batch, grid_h, grid_w, width, patch, out_channels = 2, 8, 8, 320, 2, 16
    tokens = grid_h * grid_w
    hidden_base = module.arange(0, batch * tokens * width, dtype=module.float32).reshape(batch, tokens, width)
    hidden = (module.sin(hidden_base * 0.013) * 0.18).to(dtype=module.float16)
    modulation = (
        module.cos(module.arange(0, batch * 2 * width, dtype=module.float32).reshape(batch, 2, width) * 0.017)
        * 0.06
    ).to(dtype=module.float16)
    norm_weight = (0.84 + (module.arange(0, width, dtype=module.float32) % 251) / 2048.0).to(dtype=module.float16)
    norm_bias = (((module.arange(0, width, dtype=module.float32) % 127) - 63.0) / 4096.0).to(dtype=module.float16)
    out_width = patch * patch * out_channels
    proj_base = module.arange(0, out_width * width, dtype=module.float32).reshape(out_width, width)
    proj_weight = ((proj_base % 1549) / 65536.0 - 0.012).to(dtype=module.float16)
    proj_bias = (((module.arange(0, out_width, dtype=module.float32) % 181) - 90.0) / 8192.0).to(
        dtype=module.float16
    )
    return hidden, modulation, norm_weight, norm_bias, proj_weight, proj_bias


def _stable_diffusion_sd3_unpatchify_projection_path(
    module: Any,
    hidden: Any,
    modulation: Any,
    norm_weight: Any,
    norm_bias: Any,
    proj_weight: Any,
    proj_bias: Any,
) -> Any:
    batch, grid_h, grid_w, width, patch, out_channels = 2, 8, 8, 320, 2, 16
    shift, scale = modulation.chunk(2, dim=1)
    hidden = module.nn.functional.layer_norm(hidden, (width,), norm_weight, norm_bias, eps=1e-5)
    hidden = hidden * (1.0 + scale) + shift
    patches = module.nn.functional.linear(hidden, proj_weight, proj_bias)
    patches = patches.reshape(batch, grid_h, grid_w, patch, patch, out_channels)
    return patches.permute(0, 5, 1, 3, 2, 4).reshape(batch, out_channels, grid_h * patch, grid_w * patch)


def _case_stable_diffusion_sd3_large_unpatchify_projection_setup(module: Any) -> tuple[Any, ...]:
    batch, grid_h, grid_w, width, patch, out_channels = 2, 16, 16, 320, 2, 16
    tokens = grid_h * grid_w
    hidden_base = module.arange(0, batch * tokens * width, dtype=module.float32).reshape(batch, tokens, width)
    hidden = (module.sin(hidden_base * 0.011) * 0.16).to(dtype=module.float16)
    modulation = (
        module.cos(module.arange(0, batch * 2 * width, dtype=module.float32).reshape(batch, 2, width) * 0.017)
        * 0.06
    ).to(dtype=module.float16)
    norm_weight = (0.84 + (module.arange(0, width, dtype=module.float32) % 251) / 2048.0).to(dtype=module.float16)
    norm_bias = (((module.arange(0, width, dtype=module.float32) % 127) - 63.0) / 4096.0).to(dtype=module.float16)
    out_width = patch * patch * out_channels
    proj_base = module.arange(0, out_width * width, dtype=module.float32).reshape(out_width, width)
    proj_weight = ((proj_base % 1549) / 65536.0 - 0.012).to(dtype=module.float16)
    proj_bias = (((module.arange(0, out_width, dtype=module.float32) % 181) - 90.0) / 8192.0).to(
        dtype=module.float16
    )
    return hidden, modulation, norm_weight, norm_bias, proj_weight, proj_bias


def _stable_diffusion_sd3_large_unpatchify_projection_path(
    module: Any,
    hidden: Any,
    modulation: Any,
    norm_weight: Any,
    norm_bias: Any,
    proj_weight: Any,
    proj_bias: Any,
) -> Any:
    batch, grid_h, grid_w, width, patch, out_channels = 2, 16, 16, 320, 2, 16
    shift, scale = modulation.chunk(2, dim=1)
    hidden = module.nn.functional.layer_norm(hidden, (width,), norm_weight, norm_bias, eps=1e-5)
    hidden = hidden * (1.0 + scale) + shift
    patches = module.nn.functional.linear(hidden, proj_weight, proj_bias)
    patches = patches.reshape(batch, grid_h, grid_w, patch, patch, out_channels)
    return patches.permute(0, 5, 1, 3, 2, 4).reshape(batch, out_channels, grid_h * patch, grid_w * patch)


def _case_stable_diffusion_sd3_rectangular_unpatchify_projection_setup(module: Any) -> tuple[Any, ...]:
    batch, grid_h, grid_w, width, patch, out_channels = 2, 8, 12, 320, 2, 16
    tokens = grid_h * grid_w
    hidden_base = module.arange(0, batch * tokens * width, dtype=module.float32).reshape(batch, tokens, width)
    hidden = (module.sin(hidden_base * 0.012) * 0.17).to(dtype=module.float16)
    modulation = (
        module.cos(module.arange(0, batch * 2 * width, dtype=module.float32).reshape(batch, 2, width) * 0.015)
        * 0.055
    ).to(dtype=module.float16)
    norm_weight = (0.84 + (module.arange(0, width, dtype=module.float32) % 251) / 2048.0).to(dtype=module.float16)
    norm_bias = (((module.arange(0, width, dtype=module.float32) % 127) - 63.0) / 4096.0).to(dtype=module.float16)
    out_width = patch * patch * out_channels
    proj_base = module.arange(0, out_width * width, dtype=module.float32).reshape(out_width, width)
    proj_weight = ((proj_base % 1549) / 65536.0 - 0.012).to(dtype=module.float16)
    proj_bias = (((module.arange(0, out_width, dtype=module.float32) % 181) - 90.0) / 8192.0).to(
        dtype=module.float16
    )
    return hidden, modulation, norm_weight, norm_bias, proj_weight, proj_bias


def _stable_diffusion_sd3_rectangular_unpatchify_projection_path(
    module: Any,
    hidden: Any,
    modulation: Any,
    norm_weight: Any,
    norm_bias: Any,
    proj_weight: Any,
    proj_bias: Any,
) -> Any:
    batch, grid_h, grid_w, width, patch, out_channels = 2, 8, 12, 320, 2, 16
    if hidden.shape[1] != grid_h * grid_w:
        raise ValueError("SD3 rectangular unpatchify token count must match the latent grid")
    shift, scale = modulation.chunk(2, dim=1)
    hidden = module.nn.functional.layer_norm(hidden, (width,), norm_weight, norm_bias, eps=1e-5)
    hidden = hidden * (1.0 + scale) + shift
    patches = module.nn.functional.linear(hidden, proj_weight, proj_bias)
    patches = patches.reshape(batch, grid_h, grid_w, patch, patch, out_channels)
    return patches.permute(0, 5, 1, 3, 2, 4).reshape(batch, out_channels, grid_h * patch, grid_w * patch)


def _case_stable_diffusion_sd3_patch_embed_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    batch, channels, height, width, embed_dim, patch = 2, 16, 32, 32, 320, 2
    latent_base = module.arange(0, batch * channels * height * width, dtype=module.float32).reshape(
        batch,
        channels,
        height,
        width,
    )
    latent = (module.sin(latent_base * 0.009) * 0.22).to(dtype=module.float16)
    latent = latent.contiguous(memory_format=module.channels_last)
    weight_base = module.arange(0, embed_dim * channels * patch * patch, dtype=module.float32).reshape(
        embed_dim,
        channels,
        patch,
        patch,
    )
    weight = ((weight_base % 1549) / 65536.0 - 0.012).to(dtype=module.float16)
    bias = (((module.arange(0, embed_dim, dtype=module.float32) % 181) - 90.0) / 8192.0).to(dtype=module.float16)
    token_count = (height // patch) * (width // patch)
    pos_base = module.arange(0, token_count * embed_dim, dtype=module.float32).reshape(1, token_count, embed_dim)
    pos_embed = (module.cos(pos_base * 0.007) * 0.035).to(dtype=module.float16)
    return latent, weight, bias, pos_embed


def _stable_diffusion_sd3_patch_embed_path(
    module: Any,
    latent: Any,
    weight: Any,
    bias: Any,
    pos_embed: Any,
) -> Any:
    hidden = module.nn.functional.conv2d(latent, weight, bias, stride=2)
    tokens = hidden.flatten(2).transpose(1, 2)
    return tokens + pos_embed


def _case_stable_diffusion_sd3_rectangular_patch_embed_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    batch, channels, height, width, embed_dim, patch = 2, 16, 16, 24, 320, 2
    latent_base = module.arange(0, batch * channels * height * width, dtype=module.float32).reshape(
        batch,
        channels,
        height,
        width,
    )
    latent = (module.sin(latent_base * 0.009) * 0.22).to(dtype=module.float16)
    latent = latent.contiguous(memory_format=module.channels_last)
    weight_base = module.arange(0, embed_dim * channels * patch * patch, dtype=module.float32).reshape(
        embed_dim,
        channels,
        patch,
        patch,
    )
    weight = ((weight_base % 1549) / 65536.0 - 0.012).to(dtype=module.float16)
    bias = (((module.arange(0, embed_dim, dtype=module.float32) % 181) - 90.0) / 8192.0).to(dtype=module.float16)
    token_count = (height // patch) * (width // patch)
    pos_base = module.arange(0, token_count * embed_dim, dtype=module.float32).reshape(1, token_count, embed_dim)
    pos_embed = (module.cos(pos_base * 0.007) * 0.035).to(dtype=module.float16)
    return latent, weight, bias, pos_embed


def _case_stable_diffusion_sd3_mini_transformer_stack_setup(module: Any) -> tuple[Any, ...]:
    batch, channels, height, width, embed_dim, patch = 2, 16, 16, 16, 320, 2
    latent_base = module.arange(0, batch * channels * height * width, dtype=module.float32).reshape(
        batch,
        channels,
        height,
        width,
    )
    latent = (module.sin(latent_base * 0.009) * 0.22).to(dtype=module.float16)
    latent = latent.contiguous(memory_format=module.channels_last)
    weight_base = module.arange(0, embed_dim * channels * patch * patch, dtype=module.float32).reshape(
        embed_dim,
        channels,
        patch,
        patch,
    )
    patch_weight = ((weight_base % 1549) / 65536.0 - 0.012).to(dtype=module.float16)
    patch_bias = (((module.arange(0, embed_dim, dtype=module.float32) % 181) - 90.0) / 8192.0).to(
        dtype=module.float16
    )
    token_count = (height // patch) * (width // patch)
    pos_base = module.arange(0, token_count * embed_dim, dtype=module.float32).reshape(1, token_count, embed_dim)
    pos_embed = (module.cos(pos_base * 0.007) * 0.035).to(dtype=module.float16)

    joint_args = _case_stable_diffusion_sd3_joint_transformer_block_setup(module)
    single_args = _case_stable_diffusion_sd3_single_transformer_block_setup(module)
    unpatchify_args = _case_stable_diffusion_sd3_unpatchify_projection_setup(module)
    context = joint_args[1]
    return (
        latent,
        patch_weight,
        patch_bias,
        pos_embed,
        context,
        joint_args[2:],
        single_args[2:],
        unpatchify_args[1:],
    )


def _stable_diffusion_sd3_mini_transformer_stack_path(
    module: Any,
    latent: Any,
    patch_weight: Any,
    patch_bias: Any,
    pos_embed: Any,
    context: Any,
    joint_args: Any,
    single_args: Any,
    unpatchify_args: Any,
) -> Any:
    hidden = _stable_diffusion_sd3_patch_embed_path(module, latent, patch_weight, patch_bias, pos_embed)
    hidden, context = _stable_diffusion_sd3_joint_transformer_block_path(module, hidden, context, *joint_args)
    hidden, context = _stable_diffusion_sd3_single_transformer_block_path(module, hidden, context, *single_args)
    return _stable_diffusion_sd3_unpatchify_projection_path(module, hidden, *unpatchify_args)


def _case_stable_diffusion_sd3_rectangular_transformer_stack_setup(module: Any) -> tuple[Any, ...]:
    latent, patch_weight, patch_bias, pos_embed = _case_stable_diffusion_sd3_rectangular_patch_embed_setup(module)
    joint_args = _case_stable_diffusion_sd3_joint_transformer_block_setup(module)
    single_args = _case_stable_diffusion_sd3_single_transformer_block_setup(module)
    unpatchify_args = _case_stable_diffusion_sd3_rectangular_unpatchify_projection_setup(module)
    return (
        latent,
        patch_weight,
        patch_bias,
        pos_embed,
        joint_args[1],
        joint_args[2:],
        single_args[2:],
        unpatchify_args[1:],
    )


def _stable_diffusion_sd3_rectangular_transformer_stack_path(
    module: Any,
    latent: Any,
    patch_weight: Any,
    patch_bias: Any,
    pos_embed: Any,
    context: Any,
    joint_args: Any,
    single_args: Any,
    unpatchify_args: Any,
) -> Any:
    hidden = _stable_diffusion_sd3_patch_embed_path(module, latent, patch_weight, patch_bias, pos_embed)
    hidden, context = _stable_diffusion_sd3_joint_transformer_block_path(module, hidden, context, *joint_args)
    hidden, context = _stable_diffusion_sd3_single_transformer_block_path(module, hidden, context, *single_args)
    return _stable_diffusion_sd3_rectangular_unpatchify_projection_path(module, hidden, *unpatchify_args)


def _case_stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_setup(module: Any) -> tuple[Any, ...]:
    batch, token_count, pooled_width, cond_width, width = 2, 96, 2048, 1280, 320
    latent, patch_weight, patch_bias, pos_embed = _case_stable_diffusion_sd3_rectangular_patch_embed_setup(module)
    joint_args = _case_stable_diffusion_sd3_joint_transformer_block_setup(module)
    single_args = _case_stable_diffusion_sd3_single_transformer_block_setup(module)
    unpatchify_args = _case_stable_diffusion_sd3_rectangular_unpatchify_projection_setup(module)
    rotary_args = _case_stable_diffusion_sd3_rectangular_rotary_joint_attention_setup(module)
    q_norm_weight, k_norm_weight = rotary_args[10], rotary_args[11]
    rotary_cos, rotary_sin = rotary_args[-2], rotary_args[-1]

    joint_layers = tuple(
        (
            *_stable_diffusion_sd3_shift_layer_args(joint_args[2:], 0.00018 * float(layer + 1)),
            *_stable_diffusion_sd3_shift_layer_args(
                (q_norm_weight, k_norm_weight),
                0.000007 * float(layer + 1),
            ),
            rotary_cos,
            rotary_sin,
        )
        for layer in range(2)
    )
    single_layers = tuple(
        _stable_diffusion_sd3_shift_layer_args(single_args[2:], -0.00014 * float(layer + 1)) for layer in range(2)
    )

    modulation_args = _case_stable_diffusion_sd3_pooled_modulation_setup(
        module,
        batch,
        pooled_width,
        cond_width,
        width,
        (999.0, 333.0),
    )
    conditioning_args = modulation_args[:10]
    hidden_mod_weight, hidden_mod_bias = modulation_args[10], modulation_args[11]
    context_mod_weight, context_mod_bias = modulation_args[12], modulation_args[13]
    single_mod_weight, single_mod_bias = modulation_args[14], modulation_args[15]
    final_mod_weight, final_mod_bias = modulation_args[16], modulation_args[17]
    joint_mod_layers = tuple(
        (
            *_stable_diffusion_sd3_shift_layer_args(
                (hidden_mod_weight, hidden_mod_bias),
                0.000025 * float(layer + 1),
            ),
            *_stable_diffusion_sd3_shift_layer_args(
                (context_mod_weight, context_mod_bias),
                -0.000018 * float(layer + 1),
            ),
        )
        for layer in range(2)
    )
    single_mod_layers = tuple(
        _stable_diffusion_sd3_shift_layer_args(
            (single_mod_weight, single_mod_bias),
            0.000021 * float(layer + 1),
        )
        for layer in range(2)
    )

    residual_base = module.arange(0, batch * token_count * width, dtype=module.float32).reshape(
        batch,
        token_count,
        width,
    )
    block_residuals = tuple(
        (module.cos((residual_base + float(layer * 149)) * (0.0105 + float(layer) * 0.001)) * (0.019 - layer * 0.002)).to(
            dtype=module.float16
        )
        for layer in range(4)
    )
    controlnet_keep = module.tensor([0.9, 0.7, 0.5, 0.3], dtype=module.float32)
    return (
        latent,
        patch_weight,
        patch_bias,
        pos_embed,
        joint_args[1],
        joint_layers,
        single_layers,
        unpatchify_args[1:],
        block_residuals,
        controlnet_keep,
        *conditioning_args,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
    )


def _stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_path(
    module: Any,
    latent: Any,
    patch_weight: Any,
    patch_bias: Any,
    pos_embed: Any,
    context: Any,
    joint_layers: Any,
    single_layers: Any,
    unpatchify_args: Any,
    block_residuals: Any,
    controlnet_keep: Any,
    timesteps: Any,
    pooled_prompt_embeds: Any,
    time_fc1_weight: Any,
    time_fc1_bias: Any,
    time_fc2_weight: Any,
    time_fc2_bias: Any,
    text_fc1_weight: Any,
    text_fc1_bias: Any,
    text_fc2_weight: Any,
    text_fc2_bias: Any,
    joint_mod_layers: Any,
    single_mod_layers: Any,
    final_mod_weight: Any,
    final_mod_bias: Any,
) -> Any:
    batch, width = latent.shape[0], 320
    conditioning = _stable_diffusion_sd3_time_text_conditioning_path(
        module,
        timesteps,
        pooled_prompt_embeds,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
    )

    hidden = _stable_diffusion_sd3_patch_embed_path(module, latent, patch_weight, patch_bias, pos_embed)
    residual_index = 0
    for joint_args, joint_mod_args in zip(joint_layers, joint_mod_layers):
        hidden_mod_weight, hidden_mod_bias, context_mod_weight, context_mod_bias = joint_mod_args
        hidden_modulation = module.nn.functional.linear(conditioning, hidden_mod_weight, hidden_mod_bias).reshape(
            batch,
            6,
            width,
        )
        context_modulation = module.nn.functional.linear(conditioning, context_mod_weight, context_mod_bias).reshape(
            batch,
            3,
            width,
        )
        hidden, context = _stable_diffusion_sd3_rectangular_rotary_joint_transformer_block_path(
            module,
            hidden,
            context,
            *joint_args[:6],
            hidden_modulation,
            context_modulation,
            *joint_args[8:20],
            *joint_args[20:],
        )
        hidden = hidden + block_residuals[residual_index] * controlnet_keep[residual_index].to(dtype=hidden.dtype)
        residual_index += 1

    for single_args, single_mod_args in zip(single_layers, single_mod_layers):
        single_mod_weight, single_mod_bias = single_mod_args
        single_modulation = module.nn.functional.linear(conditioning, single_mod_weight, single_mod_bias).reshape(
            batch,
            6,
            width,
        )
        hidden, context = _stable_diffusion_sd3_single_transformer_block_path(
            module,
            hidden,
            context,
            *single_args[:4],
            single_modulation,
            *single_args[5:],
        )
        hidden = hidden + block_residuals[residual_index] * controlnet_keep[residual_index].to(dtype=hidden.dtype)
        residual_index += 1

    final_modulation = module.nn.functional.linear(conditioning, final_mod_weight, final_mod_bias).reshape(
        batch,
        2,
        width,
    )
    return _stable_diffusion_sd3_rectangular_unpatchify_projection_path(
        module,
        hidden,
        final_modulation,
        *unpatchify_args[1:],
    )


def _case_stable_diffusion_sd3_rectangular_inpaint_control_conditioning_setup(module: Any) -> tuple[Any, ...]:
    batch, channels, height, width, embed_dim, patch = 2, 16, 16, 24, 320, 2
    control_base = module.arange(0, batch * channels * height * width, dtype=module.float32).reshape(
        batch,
        channels,
        height,
        width,
    )
    control_image_latents = (module.sin((control_base + 37.0) * 0.006) * 0.21).to(dtype=module.float16)
    control_image_latents = control_image_latents.contiguous(memory_format=module.channels_last)

    mask_base = module.arange(0, batch * height * width, dtype=module.float32).reshape(batch, 1, height, width)
    inpaint_mask = ((module.sin(mask_base * 0.031) + 1.0) * 0.5).to(dtype=module.float16)
    inpaint_mask = inpaint_mask.contiguous(memory_format=module.channels_last)

    weight_base = module.arange(0, embed_dim * channels * patch * patch, dtype=module.float32).reshape(
        embed_dim,
        channels,
        patch,
        patch,
    )
    control_patch_weight = ((weight_base % 1543) / 65536.0 - 0.012).to(dtype=module.float16)
    control_patch_bias = (((module.arange(0, embed_dim, dtype=module.float32) % 179) - 89.0) / 8192.0).to(
        dtype=module.float16
    )
    token_count = (height // patch) * (width // patch)
    pos_base = module.arange(0, token_count * embed_dim, dtype=module.float32).reshape(1, token_count, embed_dim)
    control_pos_embed = (module.sin((pos_base + 53.0) * 0.006) * 0.028).to(dtype=module.float16)

    mask_proj_weight = ((module.arange(0, embed_dim, dtype=module.float32).reshape(embed_dim, 1) % 251) / 8192.0 - 0.015).to(
        dtype=module.float16
    )
    mask_proj_bias = (((module.arange(0, embed_dim, dtype=module.float32) % 163) - 81.0) / 8192.0).to(
        dtype=module.float16
    )

    residual_proj_weights = tuple(
        (
            (
                module.arange(0, embed_dim * embed_dim, dtype=module.float32).reshape(embed_dim, embed_dim)
                + float(layer * 113)
            )
            % (1549 + layer * 4)
        )
        / 131072.0
        - (0.0055 + layer * 0.0002)
        for layer in range(4)
    )
    residual_proj_weights = tuple(weight.to(dtype=module.float16) for weight in residual_proj_weights)
    residual_proj_biases = tuple(
        (((module.arange(0, embed_dim, dtype=module.float32) % (173 + layer * 2)) - (86.0 + layer)) / 16384.0).to(
            dtype=module.float16
        )
        for layer in range(4)
    )
    control_conditioning_scale = module.tensor([0.8, 0.55, 0.35, 0.2], dtype=module.float32)
    return (
        control_image_latents,
        inpaint_mask,
        control_patch_weight,
        control_patch_bias,
        control_pos_embed,
        mask_proj_weight,
        mask_proj_bias,
        residual_proj_weights,
        residual_proj_biases,
        control_conditioning_scale,
    )


def _stable_diffusion_sd3_rectangular_inpaint_control_residuals_path(
    module: Any,
    control_image_latents: Any,
    inpaint_mask: Any,
    control_patch_weight: Any,
    control_patch_bias: Any,
    control_pos_embed: Any,
    mask_proj_weight: Any,
    mask_proj_bias: Any,
    residual_proj_weights: Any,
    residual_proj_biases: Any,
    control_conditioning_scale: Any,
) -> tuple[Any, ...]:
    if control_image_latents.shape[0] != inpaint_mask.shape[0]:
        raise ValueError("SD3 inpaint control conditioning batch sizes must match")
    if inpaint_mask.shape[1] != 1:
        raise ValueError("SD3 inpaint mask conditioning expects a single mask channel")
    control_tokens = _stable_diffusion_sd3_patch_embed_path(
        module,
        control_image_latents,
        control_patch_weight,
        control_patch_bias,
        control_pos_embed,
    )
    mask_tokens = module.nn.functional.avg_pool2d(inpaint_mask, 2, stride=2).flatten(2).transpose(1, 2)
    if mask_tokens.shape[1] != control_tokens.shape[1]:
        raise ValueError("SD3 inpaint mask token count must match the control latent token count")
    mask_tokens = module.nn.functional.linear(
        mask_tokens.to(dtype=control_tokens.dtype),
        mask_proj_weight,
        mask_proj_bias,
    )
    conditioning_tokens = control_tokens * (1.0 + mask_tokens * 0.05) + mask_tokens * 0.35

    residuals = []
    for index, (weight, bias) in enumerate(zip(residual_proj_weights, residual_proj_biases)):
        residual = module.nn.functional.linear(conditioning_tokens, weight, bias)
        residual = module.nn.functional.silu(residual) * control_conditioning_scale[index].to(dtype=residual.dtype)
        residuals.append(residual)
    return tuple(residuals)


def _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_setup(
    module: Any,
) -> tuple[Any, ...]:
    stack_args = _case_stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_setup(module)
    control_args = _case_stable_diffusion_sd3_rectangular_inpaint_control_conditioning_setup(module)
    return (*stack_args, *control_args)


def _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_path(
    module: Any,
    latent: Any,
    patch_weight: Any,
    patch_bias: Any,
    pos_embed: Any,
    context: Any,
    joint_layers: Any,
    single_layers: Any,
    unpatchify_args: Any,
    block_residuals: Any,
    controlnet_keep: Any,
    timesteps: Any,
    pooled_prompt_embeds: Any,
    time_fc1_weight: Any,
    time_fc1_bias: Any,
    time_fc2_weight: Any,
    time_fc2_bias: Any,
    text_fc1_weight: Any,
    text_fc1_bias: Any,
    text_fc2_weight: Any,
    text_fc2_bias: Any,
    joint_mod_layers: Any,
    single_mod_layers: Any,
    final_mod_weight: Any,
    final_mod_bias: Any,
    control_image_latents: Any,
    inpaint_mask: Any,
    control_patch_weight: Any,
    control_patch_bias: Any,
    control_pos_embed: Any,
    mask_proj_weight: Any,
    mask_proj_bias: Any,
    residual_proj_weights: Any,
    residual_proj_biases: Any,
    control_conditioning_scale: Any,
) -> Any:
    if latent.shape[0] != control_image_latents.shape[0]:
        raise ValueError("SD3 inpaint control latent batch size must match the model latent batch size")
    batch, width = latent.shape[0], 320
    conditioning = _stable_diffusion_sd3_time_text_conditioning_path(
        module,
        timesteps,
        pooled_prompt_embeds,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
    )
    control_residuals = _stable_diffusion_sd3_rectangular_inpaint_control_residuals_path(
        module,
        control_image_latents,
        inpaint_mask,
        control_patch_weight,
        control_patch_bias,
        control_pos_embed,
        mask_proj_weight,
        mask_proj_bias,
        residual_proj_weights,
        residual_proj_biases,
        control_conditioning_scale,
    )

    hidden = _stable_diffusion_sd3_patch_embed_path(module, latent, patch_weight, patch_bias, pos_embed)
    residual_index = 0
    for joint_args, joint_mod_args in zip(joint_layers, joint_mod_layers):
        hidden_mod_weight, hidden_mod_bias, context_mod_weight, context_mod_bias = joint_mod_args
        hidden_modulation = module.nn.functional.linear(conditioning, hidden_mod_weight, hidden_mod_bias).reshape(
            batch,
            6,
            width,
        )
        context_modulation = module.nn.functional.linear(conditioning, context_mod_weight, context_mod_bias).reshape(
            batch,
            3,
            width,
        )
        hidden, context = _stable_diffusion_sd3_rectangular_rotary_joint_transformer_block_path(
            module,
            hidden,
            context,
            *joint_args[:6],
            hidden_modulation,
            context_modulation,
            *joint_args[8:20],
            *joint_args[20:],
        )
        control_residual = block_residuals[residual_index] + control_residuals[residual_index]
        hidden = hidden + control_residual * controlnet_keep[residual_index].to(dtype=hidden.dtype)
        residual_index += 1

    for single_args, single_mod_args in zip(single_layers, single_mod_layers):
        single_mod_weight, single_mod_bias = single_mod_args
        single_modulation = module.nn.functional.linear(conditioning, single_mod_weight, single_mod_bias).reshape(
            batch,
            6,
            width,
        )
        hidden, context = _stable_diffusion_sd3_single_transformer_block_path(
            module,
            hidden,
            context,
            *single_args[:4],
            single_modulation,
            *single_args[5:],
        )
        control_residual = block_residuals[residual_index] + control_residuals[residual_index]
        hidden = hidden + control_residual * controlnet_keep[residual_index].to(dtype=hidden.dtype)
        residual_index += 1

    final_modulation = module.nn.functional.linear(conditioning, final_mod_weight, final_mod_bias).reshape(
        batch,
        2,
        width,
    )
    return _stable_diffusion_sd3_rectangular_unpatchify_projection_path(
        module,
        hidden,
        final_modulation,
        *unpatchify_args[1:],
    )


def _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_setup(
    module: Any,
) -> tuple[Any, ...]:
    stack_args = _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_setup(module)
    control_image_latents, inpaint_mask = stack_args[24], stack_args[25]
    secondary_control_image_latents = (
        module.flip(control_image_latents, (-1,)) * 0.61 + module.flip(control_image_latents, (-2,)) * 0.39
    ).contiguous(memory_format=module.channels_last)
    secondary_inpaint_mask = (module.flip(inpaint_mask, (-1,)) * 0.58 + module.flip(inpaint_mask, (-2,)) * 0.42).contiguous(
        memory_format=module.channels_last
    )
    multi_control_scales = module.tensor([0.9, 0.45], dtype=module.float32)
    return (*stack_args, secondary_control_image_latents, secondary_inpaint_mask, multi_control_scales)


def _stable_diffusion_sd3_rectangular_multi_image_inpaint_controlnet_residuals_path(
    module: Any,
    latent_batch_size: int,
    control_image_latents_by_controlnet: Any,
    inpaint_masks_by_controlnet: Any,
    control_patch_weight: Any,
    control_patch_bias: Any,
    control_pos_embed: Any,
    mask_proj_weight: Any,
    mask_proj_bias: Any,
    residual_proj_weights: Any,
    residual_proj_biases: Any,
    control_conditioning_scale: Any,
    multi_control_scales: Any,
) -> Any:
    controlnet_count = len(control_image_latents_by_controlnet)
    if controlnet_count <= 0:
        raise ValueError("SD3 multi-image inpaint ControlNet path expects at least one ControlNet")
    if len(inpaint_masks_by_controlnet) != controlnet_count:
        raise ValueError("SD3 multi-image inpaint ControlNet masks must match the ControlNet count")
    if len(multi_control_scales.shape) != 1 or multi_control_scales.shape[0] != controlnet_count:
        raise ValueError("SD3 multi-image inpaint ControlNet scales must have one entry per ControlNet")

    control_residuals = None
    for control_index, (control_image_latents, inpaint_mask) in enumerate(
        zip(control_image_latents_by_controlnet, inpaint_masks_by_controlnet, strict=True)
    ):
        if latent_batch_size != control_image_latents.shape[0]:
            raise ValueError("SD3 multi-image inpaint ControlNet batch size must match the model latent batch size")
        scale = control_conditioning_scale * multi_control_scales[control_index].to(
            dtype=control_conditioning_scale.dtype
        )
        residuals = _stable_diffusion_sd3_rectangular_inpaint_control_residuals_path(
            module,
            control_image_latents,
            inpaint_mask,
            control_patch_weight,
            control_patch_bias,
            control_pos_embed,
            mask_proj_weight,
            mask_proj_bias,
            residual_proj_weights,
            residual_proj_biases,
            scale,
        )
        if control_residuals is None:
            control_residuals = residuals
        else:
            control_residuals = tuple(total + residual for total, residual in zip(control_residuals, residuals))
    return control_residuals


def _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_transformer_stack_path(
    module: Any,
    latent: Any,
    patch_weight: Any,
    patch_bias: Any,
    pos_embed: Any,
    context: Any,
    joint_layers: Any,
    single_layers: Any,
    unpatchify_args: Any,
    block_residuals: Any,
    controlnet_keep: Any,
    timesteps: Any,
    pooled_prompt_embeds: Any,
    time_fc1_weight: Any,
    time_fc1_bias: Any,
    time_fc2_weight: Any,
    time_fc2_bias: Any,
    text_fc1_weight: Any,
    text_fc1_bias: Any,
    text_fc2_weight: Any,
    text_fc2_bias: Any,
    joint_mod_layers: Any,
    single_mod_layers: Any,
    final_mod_weight: Any,
    final_mod_bias: Any,
    control_image_latents_by_controlnet: Any,
    inpaint_masks_by_controlnet: Any,
    control_patch_weight: Any,
    control_patch_bias: Any,
    control_pos_embed: Any,
    mask_proj_weight: Any,
    mask_proj_bias: Any,
    residual_proj_weights: Any,
    residual_proj_biases: Any,
    control_conditioning_scale: Any,
    multi_control_scales: Any,
) -> Any:
    batch, width = latent.shape[0], 320
    conditioning = _stable_diffusion_sd3_time_text_conditioning_path(
        module,
        timesteps,
        pooled_prompt_embeds,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
    )
    control_residuals = _stable_diffusion_sd3_rectangular_multi_image_inpaint_controlnet_residuals_path(
        module,
        batch,
        control_image_latents_by_controlnet,
        inpaint_masks_by_controlnet,
        control_patch_weight,
        control_patch_bias,
        control_pos_embed,
        mask_proj_weight,
        mask_proj_bias,
        residual_proj_weights,
        residual_proj_biases,
        control_conditioning_scale,
        multi_control_scales,
    )

    hidden = _stable_diffusion_sd3_patch_embed_path(module, latent, patch_weight, patch_bias, pos_embed)
    residual_index = 0
    for joint_args, joint_mod_args in zip(joint_layers, joint_mod_layers):
        hidden_mod_weight, hidden_mod_bias, context_mod_weight, context_mod_bias = joint_mod_args
        hidden_modulation = module.nn.functional.linear(conditioning, hidden_mod_weight, hidden_mod_bias).reshape(
            batch,
            6,
            width,
        )
        context_modulation = module.nn.functional.linear(conditioning, context_mod_weight, context_mod_bias).reshape(
            batch,
            3,
            width,
        )
        hidden, context = _stable_diffusion_sd3_rectangular_rotary_joint_transformer_block_path(
            module,
            hidden,
            context,
            *joint_args[:6],
            hidden_modulation,
            context_modulation,
            *joint_args[8:20],
            *joint_args[20:],
        )
        control_residual = block_residuals[residual_index] + control_residuals[residual_index]
        hidden = hidden + control_residual * controlnet_keep[residual_index].to(dtype=hidden.dtype)
        residual_index += 1

    for single_args, single_mod_args in zip(single_layers, single_mod_layers):
        single_mod_weight, single_mod_bias = single_mod_args
        single_modulation = module.nn.functional.linear(conditioning, single_mod_weight, single_mod_bias).reshape(
            batch,
            6,
            width,
        )
        hidden, context = _stable_diffusion_sd3_single_transformer_block_path(
            module,
            hidden,
            context,
            *single_args[:4],
            single_modulation,
            *single_args[5:],
        )
        control_residual = block_residuals[residual_index] + control_residuals[residual_index]
        hidden = hidden + control_residual * controlnet_keep[residual_index].to(dtype=hidden.dtype)
        residual_index += 1

    final_modulation = module.nn.functional.linear(conditioning, final_mod_weight, final_mod_bias).reshape(
        batch,
        2,
        width,
    )
    return _stable_diffusion_sd3_rectangular_unpatchify_projection_path(
        module,
        hidden,
        final_modulation,
        *unpatchify_args[1:],
    )


def _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_path(
    module: Any,
    latent: Any,
    patch_weight: Any,
    patch_bias: Any,
    pos_embed: Any,
    context: Any,
    joint_layers: Any,
    single_layers: Any,
    unpatchify_args: Any,
    block_residuals: Any,
    controlnet_keep: Any,
    timesteps: Any,
    pooled_prompt_embeds: Any,
    time_fc1_weight: Any,
    time_fc1_bias: Any,
    time_fc2_weight: Any,
    time_fc2_bias: Any,
    text_fc1_weight: Any,
    text_fc1_bias: Any,
    text_fc2_weight: Any,
    text_fc2_bias: Any,
    joint_mod_layers: Any,
    single_mod_layers: Any,
    final_mod_weight: Any,
    final_mod_bias: Any,
    control_image_latents: Any,
    inpaint_mask: Any,
    control_patch_weight: Any,
    control_patch_bias: Any,
    control_pos_embed: Any,
    mask_proj_weight: Any,
    mask_proj_bias: Any,
    residual_proj_weights: Any,
    residual_proj_biases: Any,
    control_conditioning_scale: Any,
    secondary_control_image_latents: Any,
    secondary_inpaint_mask: Any,
    multi_control_scales: Any,
) -> Any:
    return _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_transformer_stack_path(
        module,
        latent,
        patch_weight,
        patch_bias,
        pos_embed,
        context,
        joint_layers,
        single_layers,
        unpatchify_args,
        block_residuals,
        controlnet_keep,
        timesteps,
        pooled_prompt_embeds,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
        (control_image_latents, secondary_control_image_latents),
        (inpaint_mask, secondary_inpaint_mask),
        control_patch_weight,
        control_patch_bias,
        control_pos_embed,
        mask_proj_weight,
        mask_proj_bias,
        residual_proj_weights,
        residual_proj_biases,
        control_conditioning_scale,
        multi_control_scales,
    )


def _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_setup(
    module: Any,
) -> tuple[Any, ...]:
    stack_args = _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_setup(
        module
    )
    control_image_latents, inpaint_mask = stack_args[24], stack_args[25]
    secondary_control_image_latents, secondary_inpaint_mask = stack_args[34], stack_args[35]
    tertiary_control_image_latents = (
        module.flip(control_image_latents, (-1,)) * 0.37 + secondary_control_image_latents * 0.63
    ).contiguous(memory_format=module.channels_last)
    tertiary_inpaint_mask = (module.flip(inpaint_mask, (-2,)) * 0.41 + secondary_inpaint_mask * 0.59).contiguous(
        memory_format=module.channels_last
    )
    triple_control_scales = module.tensor([0.9, 0.45, 0.25], dtype=module.float32)
    return (
        *stack_args[:24],
        (control_image_latents, secondary_control_image_latents, tertiary_control_image_latents),
        (inpaint_mask, secondary_inpaint_mask, tertiary_inpaint_mask),
        *stack_args[26:34],
        triple_control_scales,
    )


def _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_path(
    module: Any,
    latent: Any,
    patch_weight: Any,
    patch_bias: Any,
    pos_embed: Any,
    context: Any,
    joint_layers: Any,
    single_layers: Any,
    unpatchify_args: Any,
    block_residuals: Any,
    controlnet_keep: Any,
    timesteps: Any,
    pooled_prompt_embeds: Any,
    time_fc1_weight: Any,
    time_fc1_bias: Any,
    time_fc2_weight: Any,
    time_fc2_bias: Any,
    text_fc1_weight: Any,
    text_fc1_bias: Any,
    text_fc2_weight: Any,
    text_fc2_bias: Any,
    joint_mod_layers: Any,
    single_mod_layers: Any,
    final_mod_weight: Any,
    final_mod_bias: Any,
    control_image_latents_by_controlnet: Any,
    inpaint_masks_by_controlnet: Any,
    control_patch_weight: Any,
    control_patch_bias: Any,
    control_pos_embed: Any,
    mask_proj_weight: Any,
    mask_proj_bias: Any,
    residual_proj_weights: Any,
    residual_proj_biases: Any,
    control_conditioning_scale: Any,
    multi_control_scales: Any,
) -> Any:
    return _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_transformer_stack_path(
        module,
        latent,
        patch_weight,
        patch_bias,
        pos_embed,
        context,
        joint_layers,
        single_layers,
        unpatchify_args,
        block_residuals,
        controlnet_keep,
        timesteps,
        pooled_prompt_embeds,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
        control_image_latents_by_controlnet,
        inpaint_masks_by_controlnet,
        control_patch_weight,
        control_patch_bias,
        control_pos_embed,
        mask_proj_weight,
        mask_proj_bias,
        residual_proj_weights,
        residual_proj_biases,
        control_conditioning_scale,
        multi_control_scales,
    )


def _case_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_setup(module: Any) -> tuple[Any, ...]:
    stack_args = _case_stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_setup(module)
    latent, patch_weight, patch_bias, pos_embed = stack_args[:4]
    context = stack_args[4]
    joint_layers, single_layers, unpatchify_args = stack_args[5:8]
    block_residuals, controlnet_keep = stack_args[8:10]
    pooled_prompt_embeds = stack_args[11]
    conditioning_weights = stack_args[12:20]
    joint_mod_layers, single_mod_layers, final_mod_weight, final_mod_bias = stack_args[20:24]

    latent_uncond, latent_cond = latent.chunk(2, dim=0)
    latents = (latent_uncond * 0.65 + module.flip(latent_cond, (-1,)) * 0.35).contiguous(
        memory_format=module.channels_last
    )
    negative_prompt_embeds, prompt_embeds = context.chunk(2, dim=0)
    negative_prompt_embeds = (negative_prompt_embeds - 0.012).to(dtype=module.float16)
    prompt_embeds = (prompt_embeds + 0.015).to(dtype=module.float16)
    negative_pooled_prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds.chunk(2, dim=0)
    negative_pooled_prompt_embeds = (negative_pooled_prompt_embeds - 0.01).to(dtype=module.float16)
    pooled_prompt_embeds = (pooled_prompt_embeds + 0.012).to(dtype=module.float16)
    sigmas = module.tensor([1.0, 0.76, 0.42, 0.12, 0.0], dtype=module.float32)
    index = module.tensor(0, dtype=module.long)
    model_args = (
        patch_weight,
        patch_bias,
        pos_embed,
        joint_layers,
        single_layers,
        unpatchify_args,
        block_residuals,
        controlnet_keep,
        *conditioning_weights,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
    )
    return (
        latents,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        index,
        2,
        5.5,
        model_args,
    )


def _stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path(
    module: Any,
    latents: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    index: Any,
    step_count: int,
    guidance_scale: float,
    model_args: Any,
) -> Any:
    (
        patch_weight,
        patch_bias,
        pos_embed,
        joint_layers,
        single_layers,
        unpatchify_args,
        block_residuals,
        controlnet_keep,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
    ) = model_args

    context = module.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_conditioning = module.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    for step in range(step_count):
        current_index = index + step
        sigma = sigmas[current_index]
        latent_model_input = module.cat([latents, latents], dim=0).contiguous(memory_format=module.channels_last)
        timesteps = module.stack([sigma * 1000.0, sigma * 1000.0], dim=0)
        model_output = _stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_path(
            module,
            latent_model_input,
            patch_weight,
            patch_bias,
            pos_embed,
            context,
            joint_layers,
            single_layers,
            unpatchify_args,
            block_residuals,
            controlnet_keep,
            timesteps,
            pooled_conditioning,
            time_fc1_weight,
            time_fc1_bias,
            time_fc2_weight,
            time_fc2_bias,
            text_fc1_weight,
            text_fc1_bias,
            text_fc2_weight,
            text_fc2_bias,
            joint_mod_layers,
            single_mod_layers,
            final_mod_weight,
            final_mod_bias,
        )
        noise_pred_uncond, noise_pred_text = model_output.chunk(2, dim=0)
        guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        dt = (sigmas[current_index + 1] - sigma).reshape(1, 1, 1, 1).to(dtype=latents.dtype)
        latents = (latents + guided * dt).contiguous(memory_format=module.channels_last)
    return latents


def _case_stable_diffusion_sd3_rectangular_pooled_controlnet_long_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = list(_case_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_setup(module))
    args[5] = module.tensor([1.0, 0.86, 0.64, 0.38, 0.14, 0.0], dtype=module.float32)
    args[7] = 4
    args[8] = 5.75
    return tuple(args)


def _case_stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_controlnet_long_denoising_loop_setup(module)
    init_latents = args[0]
    latent_noise = (module.flip(init_latents, (-1,)) * 0.73 + module.flip(init_latents, (-2,)) * 0.27).contiguous(
        memory_format=module.channels_last
    )
    return (
        init_latents,
        latent_noise,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        0.6,
        5,
        5.75,
        args[9],
    )


def _stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_path(
    module: Any,
    init_latents: Any,
    latent_noise: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    model_args: Any,
) -> Any:
    init_timestep, start_index, latents = _stable_diffusion_sd3_prepare_img2img_latents_path(
        module,
        init_latents,
        latent_noise,
        sigmas,
        strength,
        num_inference_steps,
        "SD3 img2img",
    )
    if init_timestep == 0:
        return latents

    return _stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path(
        module,
        latents,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        module.tensor(start_index, dtype=module.long),
        init_timestep,
        guidance_scale,
        model_args,
    )


def _stable_diffusion_sd3_prepare_img2img_latents_path(
    module: Any,
    init_latents: Any,
    latent_noise: Any,
    sigmas: Any,
    strength: float,
    num_inference_steps: int,
    error_prefix: str,
) -> tuple[int, int, Any]:
    if strength < 0.0 or strength > 1.0:
        raise ValueError(f"{error_prefix} strength must be in [0, 1]")
    if num_inference_steps <= 0:
        raise ValueError(f"{error_prefix} num_inference_steps must be positive")
    if sigmas.shape[0] < num_inference_steps + 1:
        raise ValueError(f"{error_prefix} sigma schedule must include the terminal sigma")

    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    if init_timestep == 0:
        return init_timestep, num_inference_steps, init_latents.contiguous(memory_format=module.channels_last)

    start_index = num_inference_steps - init_timestep
    sigma = sigmas[start_index].reshape(1, 1, 1, 1).to(dtype=init_latents.dtype)
    latents = (init_latents * (1.0 - sigma) + latent_noise * sigma).contiguous(memory_format=module.channels_last)
    return init_timestep, start_index, latents


def _case_stable_diffusion_sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_controlnet_long_denoising_loop_setup(module)
    step_controlnet_keep = module.tensor([1.0, 0.82, 0.55, 0.28, 0.0], dtype=module.float32)
    return (*args[:9], step_controlnet_keep, args[9])


def _stable_diffusion_sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop_path(
    module: Any,
    latents: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    index: Any,
    step_count: int,
    guidance_scale: float,
    step_controlnet_keep: Any,
    model_args: Any,
) -> Any:
    (
        patch_weight,
        patch_bias,
        pos_embed,
        joint_layers,
        single_layers,
        unpatchify_args,
        block_residuals,
        controlnet_keep,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
    ) = model_args

    context = module.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_conditioning = module.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    for step in range(step_count):
        current_index = index + step
        sigma = sigmas[current_index]
        latent_model_input = module.cat([latents, latents], dim=0).contiguous(memory_format=module.channels_last)
        timesteps = module.stack([sigma * 1000.0, sigma * 1000.0], dim=0)
        scaled_controlnet_keep = controlnet_keep * step_controlnet_keep[current_index].to(dtype=controlnet_keep.dtype)
        model_output = _stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_path(
            module,
            latent_model_input,
            patch_weight,
            patch_bias,
            pos_embed,
            context,
            joint_layers,
            single_layers,
            unpatchify_args,
            block_residuals,
            scaled_controlnet_keep,
            timesteps,
            pooled_conditioning,
            time_fc1_weight,
            time_fc1_bias,
            time_fc2_weight,
            time_fc2_bias,
            text_fc1_weight,
            text_fc1_bias,
            text_fc2_weight,
            text_fc2_bias,
            joint_mod_layers,
            single_mod_layers,
            final_mod_weight,
            final_mod_bias,
        )
        noise_pred_uncond, noise_pred_text = model_output.chunk(2, dim=0)
        guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        dt = (sigmas[current_index + 1] - sigma).reshape(1, 1, 1, 1).to(dtype=latents.dtype)
        latents = (latents + guided * dt).contiguous(memory_format=module.channels_last)
    return latents


def _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    stack_args = _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_setup(module)
    latent, patch_weight, patch_bias, pos_embed = stack_args[:4]
    context = stack_args[4]
    joint_layers, single_layers, unpatchify_args = stack_args[5:8]
    block_residuals, controlnet_keep = stack_args[8:10]
    pooled_prompt_embeds = stack_args[11]
    conditioning_weights = stack_args[12:20]
    joint_mod_layers, single_mod_layers, final_mod_weight, final_mod_bias = stack_args[20:24]
    (
        control_image_latents,
        inpaint_mask,
        control_patch_weight,
        control_patch_bias,
        control_pos_embed,
        mask_proj_weight,
        mask_proj_bias,
        residual_proj_weights,
        residual_proj_biases,
        control_conditioning_scale,
    ) = stack_args[24:34]

    latent_uncond, latent_cond = latent.chunk(2, dim=0)
    init_latents = (latent_uncond * 0.58 + module.flip(latent_cond, (-2,)) * 0.42).contiguous(
        memory_format=module.channels_last
    )
    latents = (init_latents * 0.83 + module.flip(latent_uncond, (-1,)) * 0.17).contiguous(
        memory_format=module.channels_last
    )
    control_uncond, control_cond = control_image_latents.chunk(2, dim=0)
    mask_uncond, mask_cond = inpaint_mask.chunk(2, dim=0)
    inpaint_mask = (mask_uncond * 0.62 + module.flip(mask_cond, (-1,)) * 0.38).contiguous(
        memory_format=module.channels_last
    )
    masked_image_latents = (
        control_uncond * (1.0 - inpaint_mask) + module.flip(control_cond, (-1,)) * inpaint_mask
    ).contiguous(memory_format=module.channels_last)

    negative_prompt_embeds, prompt_embeds = context.chunk(2, dim=0)
    negative_prompt_embeds = (negative_prompt_embeds - 0.014).to(dtype=module.float16)
    prompt_embeds = (prompt_embeds + 0.017).to(dtype=module.float16)
    negative_pooled_prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds.chunk(2, dim=0)
    negative_pooled_prompt_embeds = (negative_pooled_prompt_embeds - 0.011).to(dtype=module.float16)
    pooled_prompt_embeds = (pooled_prompt_embeds + 0.013).to(dtype=module.float16)
    sigmas = module.tensor([1.0, 0.86, 0.64, 0.38, 0.14, 0.0], dtype=module.float32)
    index = module.tensor(0, dtype=module.long)
    model_args = (
        patch_weight,
        patch_bias,
        pos_embed,
        joint_layers,
        single_layers,
        unpatchify_args,
        block_residuals,
        controlnet_keep,
        *conditioning_weights,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
        control_patch_weight,
        control_patch_bias,
        control_pos_embed,
        mask_proj_weight,
        mask_proj_bias,
        residual_proj_weights,
        residual_proj_biases,
        control_conditioning_scale,
    )
    return (
        latents,
        init_latents,
        inpaint_mask,
        masked_image_latents,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        index,
        4,
        5.75,
        model_args,
    )


def _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_path(
    module: Any,
    latents: Any,
    init_latents: Any,
    inpaint_mask: Any,
    control_image_latents: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    index: Any,
    step_count: int,
    guidance_scale: float,
    model_args: Any,
) -> Any:
    (
        patch_weight,
        patch_bias,
        pos_embed,
        joint_layers,
        single_layers,
        unpatchify_args,
        block_residuals,
        controlnet_keep,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
        control_patch_weight,
        control_patch_bias,
        control_pos_embed,
        mask_proj_weight,
        mask_proj_bias,
        residual_proj_weights,
        residual_proj_biases,
        control_conditioning_scale,
    ) = model_args

    context = module.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_conditioning = module.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    preserve_mask = inpaint_mask.to(dtype=latents.dtype)
    for step in range(step_count):
        current_index = index + step
        sigma = sigmas[current_index]
        latent_model_input = module.cat([latents, latents], dim=0).contiguous(memory_format=module.channels_last)
        control_model_input = module.cat([control_image_latents, control_image_latents], dim=0).contiguous(
            memory_format=module.channels_last
        )
        mask_model_input = module.cat([inpaint_mask, inpaint_mask], dim=0).contiguous(memory_format=module.channels_last)
        timesteps = module.stack([sigma * 1000.0, sigma * 1000.0], dim=0)
        model_output = _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_path(
            module,
            latent_model_input,
            patch_weight,
            patch_bias,
            pos_embed,
            context,
            joint_layers,
            single_layers,
            unpatchify_args,
            block_residuals,
            controlnet_keep,
            timesteps,
            pooled_conditioning,
            time_fc1_weight,
            time_fc1_bias,
            time_fc2_weight,
            time_fc2_bias,
            text_fc1_weight,
            text_fc1_bias,
            text_fc2_weight,
            text_fc2_bias,
            joint_mod_layers,
            single_mod_layers,
            final_mod_weight,
            final_mod_bias,
            control_model_input,
            mask_model_input,
            control_patch_weight,
            control_patch_bias,
            control_pos_embed,
            mask_proj_weight,
            mask_proj_bias,
            residual_proj_weights,
            residual_proj_biases,
            control_conditioning_scale,
        )
        noise_pred_uncond, noise_pred_text = model_output.chunk(2, dim=0)
        guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        dt = (sigmas[current_index + 1] - sigma).reshape(1, 1, 1, 1).to(dtype=latents.dtype)
        denoised = (latents + guided * dt).contiguous(memory_format=module.channels_last)
        latents = (init_latents * preserve_mask + denoised * (1.0 - preserve_mask)).contiguous(
            memory_format=module.channels_last
        )
    return latents


def _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_setup(module)
    init_latents = args[1]
    latent_noise = (module.flip(init_latents, (-1,)) * 0.67 + module.flip(init_latents, (-2,)) * 0.33).contiguous(
        memory_format=module.channels_last
    )
    return (
        init_latents,
        latent_noise,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        0.6,
        5,
        args[11],
        args[12],
    )


def _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_path(
    module: Any,
    init_latents: Any,
    latent_noise: Any,
    inpaint_mask: Any,
    control_image_latents: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    model_args: Any,
) -> Any:
    init_timestep, start_index, latents = _stable_diffusion_sd3_prepare_img2img_latents_path(
        module,
        init_latents,
        latent_noise,
        sigmas,
        strength,
        num_inference_steps,
        "SD3 inpaint img2img",
    )
    if init_timestep == 0:
        return latents

    return _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_path(
        module,
        latents,
        init_latents,
        inpaint_mask,
        control_image_latents,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        module.tensor(start_index, dtype=module.long),
        init_timestep,
        guidance_scale,
        model_args,
    )


def _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_setup(module)
    step_controlnet_keep = module.tensor([1.0, 0.78, 0.52, 0.24, 0.0], dtype=module.float32)
    return (*args[:12], step_controlnet_keep, args[12])


def _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop_path(
    module: Any,
    latents: Any,
    init_latents: Any,
    inpaint_mask: Any,
    control_image_latents: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    index: Any,
    step_count: int,
    guidance_scale: float,
    step_controlnet_keep: Any,
    model_args: Any,
) -> Any:
    (
        patch_weight,
        patch_bias,
        pos_embed,
        joint_layers,
        single_layers,
        unpatchify_args,
        block_residuals,
        controlnet_keep,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
        control_patch_weight,
        control_patch_bias,
        control_pos_embed,
        mask_proj_weight,
        mask_proj_bias,
        residual_proj_weights,
        residual_proj_biases,
        control_conditioning_scale,
    ) = model_args

    context = module.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_conditioning = module.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    preserve_mask = inpaint_mask.to(dtype=latents.dtype)
    for step in range(step_count):
        current_index = index + step
        sigma = sigmas[current_index]
        latent_model_input = module.cat([latents, latents], dim=0).contiguous(memory_format=module.channels_last)
        control_model_input = module.cat([control_image_latents, control_image_latents], dim=0).contiguous(
            memory_format=module.channels_last
        )
        mask_model_input = module.cat([inpaint_mask, inpaint_mask], dim=0).contiguous(memory_format=module.channels_last)
        timesteps = module.stack([sigma * 1000.0, sigma * 1000.0], dim=0)
        scaled_controlnet_keep = controlnet_keep * step_controlnet_keep[current_index].to(dtype=controlnet_keep.dtype)
        model_output = _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_path(
            module,
            latent_model_input,
            patch_weight,
            patch_bias,
            pos_embed,
            context,
            joint_layers,
            single_layers,
            unpatchify_args,
            block_residuals,
            scaled_controlnet_keep,
            timesteps,
            pooled_conditioning,
            time_fc1_weight,
            time_fc1_bias,
            time_fc2_weight,
            time_fc2_bias,
            text_fc1_weight,
            text_fc1_bias,
            text_fc2_weight,
            text_fc2_bias,
            joint_mod_layers,
            single_mod_layers,
            final_mod_weight,
            final_mod_bias,
            control_model_input,
            mask_model_input,
            control_patch_weight,
            control_patch_bias,
            control_pos_embed,
            mask_proj_weight,
            mask_proj_bias,
            residual_proj_weights,
            residual_proj_biases,
            control_conditioning_scale,
        )
        noise_pred_uncond, noise_pred_text = model_output.chunk(2, dim=0)
        guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        dt = (sigmas[current_index + 1] - sigma).reshape(1, 1, 1, 1).to(dtype=latents.dtype)
        denoised = (latents + guided * dt).contiguous(memory_format=module.channels_last)
        latents = (init_latents * preserve_mask + denoised * (1.0 - preserve_mask)).contiguous(
            memory_format=module.channels_last
        )
    return latents


def _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    stack_args = _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_setup(
        module
    )
    latent, patch_weight, patch_bias, pos_embed = stack_args[:4]
    context = stack_args[4]
    joint_layers, single_layers, unpatchify_args = stack_args[5:8]
    block_residuals, controlnet_keep = stack_args[8:10]
    pooled_prompt_embeds = stack_args[11]
    conditioning_weights = stack_args[12:20]
    joint_mod_layers, single_mod_layers, final_mod_weight, final_mod_bias = stack_args[20:24]
    (
        control_image_latents,
        inpaint_mask,
        control_patch_weight,
        control_patch_bias,
        control_pos_embed,
        mask_proj_weight,
        mask_proj_bias,
        residual_proj_weights,
        residual_proj_biases,
        control_conditioning_scale,
        secondary_control_image_latents,
        secondary_inpaint_mask,
        _multi_control_scales,
    ) = stack_args[24:37]

    latent_uncond, latent_cond = latent.chunk(2, dim=0)
    init_latents = (latent_uncond * 0.57 + module.flip(latent_cond, (-2,)) * 0.43).contiguous(
        memory_format=module.channels_last
    )
    latents = (init_latents * 0.79 + module.flip(latent_uncond, (-1,)) * 0.21).contiguous(
        memory_format=module.channels_last
    )

    control_uncond, control_cond = control_image_latents.chunk(2, dim=0)
    mask_uncond, mask_cond = inpaint_mask.chunk(2, dim=0)
    inpaint_mask = (mask_uncond * 0.64 + module.flip(mask_cond, (-1,)) * 0.36).contiguous(
        memory_format=module.channels_last
    )
    masked_image_latents = (
        control_uncond * (1.0 - inpaint_mask) + module.flip(control_cond, (-1,)) * inpaint_mask
    ).contiguous(memory_format=module.channels_last)

    secondary_control_uncond, secondary_control_cond = secondary_control_image_latents.chunk(2, dim=0)
    secondary_mask_uncond, secondary_mask_cond = secondary_inpaint_mask.chunk(2, dim=0)
    secondary_inpaint_mask = (
        secondary_mask_uncond * 0.57 + module.flip(secondary_mask_cond, (-2,)) * 0.43
    ).contiguous(memory_format=module.channels_last)
    secondary_masked_image_latents = (
        secondary_control_uncond * (1.0 - secondary_inpaint_mask)
        + module.flip(secondary_control_cond, (-2,)) * secondary_inpaint_mask
    ).contiguous(memory_format=module.channels_last)

    negative_prompt_embeds, prompt_embeds = context.chunk(2, dim=0)
    negative_prompt_embeds = (negative_prompt_embeds - 0.016).to(dtype=module.float16)
    prompt_embeds = (prompt_embeds + 0.019).to(dtype=module.float16)
    negative_pooled_prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds.chunk(2, dim=0)
    negative_pooled_prompt_embeds = (negative_pooled_prompt_embeds - 0.012).to(dtype=module.float16)
    pooled_prompt_embeds = (pooled_prompt_embeds + 0.014).to(dtype=module.float16)

    sigmas = module.tensor([1.0, 0.86, 0.64, 0.38, 0.14, 0.0], dtype=module.float32)
    index = module.tensor(0, dtype=module.long)
    multi_control_scale_schedule = module.tensor(
        [
            [0.90, 0.45],
            [0.82, 0.50],
            [0.66, 0.42],
            [0.38, 0.25],
            [0.0, 0.0],
        ],
        dtype=module.float32,
    )
    model_args = (
        patch_weight,
        patch_bias,
        pos_embed,
        joint_layers,
        single_layers,
        unpatchify_args,
        block_residuals,
        controlnet_keep,
        *conditioning_weights,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
        control_patch_weight,
        control_patch_bias,
        control_pos_embed,
        mask_proj_weight,
        mask_proj_bias,
        residual_proj_weights,
        residual_proj_biases,
        control_conditioning_scale,
    )
    return (
        latents,
        init_latents,
        inpaint_mask,
        masked_image_latents,
        secondary_inpaint_mask,
        secondary_masked_image_latents,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        index,
        4,
        5.75,
        multi_control_scale_schedule,
        model_args,
    )


def _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_path(
    module: Any,
    latents: Any,
    init_latents: Any,
    inpaint_mask: Any,
    control_image_latents: Any,
    secondary_inpaint_mask: Any,
    secondary_control_image_latents: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    index: Any,
    step_count: int,
    guidance_scale: float,
    multi_control_scale_schedule: Any,
    model_args: Any,
) -> Any:
    return _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_denoising_loop_path(
        module,
        latents,
        init_latents,
        inpaint_mask,
        (control_image_latents, secondary_control_image_latents),
        (inpaint_mask, secondary_inpaint_mask),
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        index,
        step_count,
        guidance_scale,
        multi_control_scale_schedule,
        model_args,
    )


def _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_denoising_loop_path(
    module: Any,
    latents: Any,
    init_latents: Any,
    preserve_mask: Any,
    control_image_latents_by_controlnet: Any,
    inpaint_masks_by_controlnet: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    index: Any,
    step_count: int,
    guidance_scale: float,
    multi_control_scale_schedule: Any,
    model_args: Any,
) -> Any:
    (
        patch_weight,
        patch_bias,
        pos_embed,
        joint_layers,
        single_layers,
        unpatchify_args,
        block_residuals,
        controlnet_keep,
        time_fc1_weight,
        time_fc1_bias,
        time_fc2_weight,
        time_fc2_bias,
        text_fc1_weight,
        text_fc1_bias,
        text_fc2_weight,
        text_fc2_bias,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
        control_patch_weight,
        control_patch_bias,
        control_pos_embed,
        mask_proj_weight,
        mask_proj_bias,
        residual_proj_weights,
        residual_proj_biases,
        control_conditioning_scale,
    ) = model_args

    controlnet_count = len(control_image_latents_by_controlnet)
    if controlnet_count <= 0:
        raise ValueError("SD3 multi-image inpaint denoising loop expects at least one ControlNet")
    if len(inpaint_masks_by_controlnet) != controlnet_count:
        raise ValueError("SD3 multi-image inpaint denoising masks must match the ControlNet count")
    if len(multi_control_scale_schedule.shape) != 2 or multi_control_scale_schedule.shape[1] != controlnet_count:
        raise ValueError("SD3 multi-image inpaint control schedule must have shape [steps, controlnets]")
    if multi_control_scale_schedule.shape[0] < sigmas.shape[0] - 1:
        raise ValueError("SD3 multi-image inpaint control schedule must cover all non-terminal sigmas")

    context = module.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_conditioning = module.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    control_model_inputs = tuple(
        module.cat([control_image_latents, control_image_latents], dim=0).contiguous(
            memory_format=module.channels_last
        )
        for control_image_latents in control_image_latents_by_controlnet
    )
    mask_model_inputs = tuple(
        module.cat([inpaint_mask, inpaint_mask], dim=0).contiguous(memory_format=module.channels_last)
        for inpaint_mask in inpaint_masks_by_controlnet
    )
    preserve_mask = preserve_mask.to(dtype=latents.dtype)
    for step in range(step_count):
        current_index = index + step
        sigma = sigmas[current_index]
        latent_model_input = module.cat([latents, latents], dim=0).contiguous(memory_format=module.channels_last)
        timesteps = module.stack([sigma * 1000.0, sigma * 1000.0], dim=0)
        model_output = _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_transformer_stack_path(
            module,
            latent_model_input,
            patch_weight,
            patch_bias,
            pos_embed,
            context,
            joint_layers,
            single_layers,
            unpatchify_args,
            block_residuals,
            controlnet_keep,
            timesteps,
            pooled_conditioning,
            time_fc1_weight,
            time_fc1_bias,
            time_fc2_weight,
            time_fc2_bias,
            text_fc1_weight,
            text_fc1_bias,
            text_fc2_weight,
            text_fc2_bias,
            joint_mod_layers,
            single_mod_layers,
            final_mod_weight,
            final_mod_bias,
            control_model_inputs,
            mask_model_inputs,
            control_patch_weight,
            control_patch_bias,
            control_pos_embed,
            mask_proj_weight,
            mask_proj_bias,
            residual_proj_weights,
            residual_proj_biases,
            control_conditioning_scale,
            multi_control_scale_schedule[current_index],
        )
        noise_pred_uncond, noise_pred_text = model_output.chunk(2, dim=0)
        guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        dt = (sigmas[current_index + 1] - sigma).reshape(1, 1, 1, 1).to(dtype=latents.dtype)
        denoised = (latents + guided * dt).contiguous(memory_format=module.channels_last)
        latents = (init_latents * preserve_mask + denoised * (1.0 - preserve_mask)).contiguous(
            memory_format=module.channels_last
        )
    return latents


def _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_setup(module)
    tertiary_inpaint_mask = (module.flip(args[2], (-2,)) * 0.46 + module.flip(args[4], (-1,)) * 0.54).contiguous(
        memory_format=module.channels_last
    )
    tertiary_control_image_latents = (module.flip(args[3], (-2,)) * 0.52 + module.flip(args[5], (-1,)) * 0.48).contiguous(
        memory_format=module.channels_last
    )
    multi_control_scale_schedule = module.tensor(
        [
            [0.90, 0.45, 0.24],
            [0.82, 0.50, 0.31],
            [0.66, 0.42, 0.36],
            [0.38, 0.25, 0.21],
            [0.0, 0.0, 0.0],
        ],
        dtype=module.float32,
    )
    return (
        args[0],
        args[1],
        args[2],
        (args[3], args[5], tertiary_control_image_latents),
        (args[2], args[4], tertiary_inpaint_mask),
        *args[6:14],
        multi_control_scale_schedule,
        args[15],
    )


def _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_path(
    module: Any,
    latents: Any,
    init_latents: Any,
    preserve_mask: Any,
    control_image_latents_by_controlnet: Any,
    inpaint_masks_by_controlnet: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    index: Any,
    step_count: int,
    guidance_scale: float,
    multi_control_scale_schedule: Any,
    model_args: Any,
) -> Any:
    return _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_denoising_loop_path(
        module,
        latents,
        init_latents,
        preserve_mask,
        control_image_latents_by_controlnet,
        inpaint_masks_by_controlnet,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        index,
        step_count,
        guidance_scale,
        multi_control_scale_schedule,
        model_args,
    )


def _stable_diffusion_sd3_multi_control_guidance_window_schedule_path(
    module: Any,
    multi_control_scale_schedule: Any,
    control_guidance_start: Any,
    control_guidance_end: Any,
) -> Any:
    if len(multi_control_scale_schedule.shape) != 2:
        raise ValueError("SD3 multi-control scale schedule must have shape [steps, controlnets]")
    controlnet_count = multi_control_scale_schedule.shape[1]
    if len(control_guidance_start.shape) != 1 or control_guidance_start.shape[0] != controlnet_count:
        raise ValueError("SD3 control guidance start must have one entry per ControlNet")
    if len(control_guidance_end.shape) != 1 or control_guidance_end.shape[0] != controlnet_count:
        raise ValueError("SD3 control guidance end must have one entry per ControlNet")
    if multi_control_scale_schedule.shape[0] <= 0:
        raise ValueError("SD3 multi-control scale schedule must include at least one denoising step")
    if not module.isfinite(control_guidance_start).all().item() or not module.isfinite(control_guidance_end).all().item():
        raise ValueError("SD3 control guidance windows must be finite")
    if (
        ((control_guidance_start < 0.0) | (control_guidance_start > 1.0)).any().item()
        or ((control_guidance_end < 0.0) | (control_guidance_end > 1.0)).any().item()
    ):
        raise ValueError("SD3 control guidance windows must stay in [0, 1]")
    if (control_guidance_start > control_guidance_end).any().item():
        raise ValueError("SD3 control guidance start must not exceed control guidance end")

    step_count = multi_control_scale_schedule.shape[0]
    step_index = module.arange(0, step_count, dtype=module.float32).reshape(step_count, 1)
    step_start = step_index / float(step_count)
    step_end = (step_index + 1.0) / float(step_count)
    control_start = control_guidance_start.reshape(1, controlnet_count)
    control_end = control_guidance_end.reshape(1, controlnet_count)
    control_active = (step_start >= control_start) & (step_end <= control_end)
    return multi_control_scale_schedule * control_active.to(dtype=multi_control_scale_schedule.dtype)


def _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_setup(module)
    control_guidance_start = module.tensor([0.0, 0.4], dtype=module.float32)
    control_guidance_end = module.tensor([0.76, 1.0], dtype=module.float32)
    return (*args[:15], control_guidance_start, control_guidance_end, args[15])


def _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_zero_width_windowed_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_setup(
        module
    )
    control_guidance_start = module.tensor([0.0, 0.4], dtype=module.float32)
    control_guidance_end = module.tensor([0.76, 0.4], dtype=module.float32)
    return (*args[:15], control_guidance_start, control_guidance_end, args[17])


def _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_disabled_windowed_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_setup(
        module
    )
    control_guidance_start = module.tensor([0.0, 0.4], dtype=module.float32)
    control_guidance_end = module.tensor([0.0, 0.4], dtype=module.float32)
    return (*args[:15], control_guidance_start, control_guidance_end, args[17])


def _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_path(
    module: Any,
    latents: Any,
    init_latents: Any,
    inpaint_mask: Any,
    control_image_latents: Any,
    secondary_inpaint_mask: Any,
    secondary_control_image_latents: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    index: Any,
    step_count: int,
    guidance_scale: float,
    multi_control_scale_schedule: Any,
    control_guidance_start: Any,
    control_guidance_end: Any,
    model_args: Any,
) -> Any:
    windowed_control_scale_schedule = _stable_diffusion_sd3_multi_control_guidance_window_schedule_path(
        module,
        multi_control_scale_schedule,
        control_guidance_start,
        control_guidance_end,
    )
    return _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_path(
        module,
        latents,
        init_latents,
        inpaint_mask,
        control_image_latents,
        secondary_inpaint_mask,
        secondary_control_image_latents,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        index,
        step_count,
        guidance_scale,
        windowed_control_scale_schedule,
        model_args,
    )


def _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_setup(module)
    control_guidance_start = module.tensor([0.0, 0.24, 0.52], dtype=module.float32)
    control_guidance_end = module.tensor([0.72, 0.88, 1.0], dtype=module.float32)
    return (*args[:14], control_guidance_start, control_guidance_end, args[14])


def _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_zero_width_windowed_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_setup(
        module
    )
    control_guidance_start = module.tensor([0.0, 0.24, 0.52], dtype=module.float32)
    control_guidance_end = module.tensor([0.72, 0.24, 1.0], dtype=module.float32)
    return (*args[:14], control_guidance_start, control_guidance_end, args[16])


def _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_disabled_windowed_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_setup(
        module
    )
    control_guidance_start = module.tensor([0.0, 0.24, 0.52], dtype=module.float32)
    control_guidance_end = module.tensor([0.0, 0.24, 0.52], dtype=module.float32)
    return (*args[:14], control_guidance_start, control_guidance_end, args[16])


def _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_path(
    module: Any,
    latents: Any,
    init_latents: Any,
    preserve_mask: Any,
    control_image_latents_by_controlnet: Any,
    inpaint_masks_by_controlnet: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    index: Any,
    step_count: int,
    guidance_scale: float,
    multi_control_scale_schedule: Any,
    control_guidance_start: Any,
    control_guidance_end: Any,
    model_args: Any,
) -> Any:
    windowed_control_scale_schedule = _stable_diffusion_sd3_multi_control_guidance_window_schedule_path(
        module,
        multi_control_scale_schedule,
        control_guidance_start,
        control_guidance_end,
    )
    return _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_denoising_loop_path(
        module,
        latents,
        init_latents,
        preserve_mask,
        control_image_latents_by_controlnet,
        inpaint_masks_by_controlnet,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        index,
        step_count,
        guidance_scale,
        windowed_control_scale_schedule,
        model_args,
    )


def _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_setup(module)
    init_latents = args[1]
    latent_noise = (module.flip(init_latents, (-1,)) * 0.63 + module.flip(init_latents, (-2,)) * 0.37).contiguous(
        memory_format=module.channels_last
    )
    return (
        init_latents,
        latent_noise,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9],
        args[10],
        0.6,
        5,
        args[13],
        args[14],
        args[15],
    )


def _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_prepared_img2img_denoising_loop_path(
    module: Any,
    latents: Any,
    init_latents: Any,
    preserve_mask: Any,
    control_image_latents_by_controlnet: Any,
    inpaint_masks_by_controlnet: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    start_index: int,
    init_timestep: int,
    guidance_scale: float,
    multi_control_scale_schedule: Any,
    model_args: Any,
) -> Any:
    return _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_denoising_loop_path(
        module,
        latents,
        init_latents,
        preserve_mask,
        control_image_latents_by_controlnet,
        inpaint_masks_by_controlnet,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        module.tensor(start_index, dtype=module.long),
        init_timestep,
        guidance_scale,
        multi_control_scale_schedule,
        model_args,
    )


def _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_img2img_denoising_loop_path(
    module: Any,
    init_latents: Any,
    latent_noise: Any,
    preserve_mask: Any,
    control_image_latents_by_controlnet: Any,
    inpaint_masks_by_controlnet: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    multi_control_scale_schedule: Any,
    model_args: Any,
    error_prefix: str,
) -> Any:
    init_timestep, start_index, latents = _stable_diffusion_sd3_prepare_img2img_latents_path(
        module,
        init_latents,
        latent_noise,
        sigmas,
        strength,
        num_inference_steps,
        error_prefix,
    )
    if init_timestep == 0:
        return latents

    return _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_prepared_img2img_denoising_loop_path(
        module,
        latents,
        init_latents,
        preserve_mask,
        control_image_latents_by_controlnet,
        inpaint_masks_by_controlnet,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        start_index,
        init_timestep,
        guidance_scale,
        multi_control_scale_schedule,
        model_args,
    )


def _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_path(
    module: Any,
    init_latents: Any,
    latent_noise: Any,
    inpaint_mask: Any,
    control_image_latents: Any,
    secondary_inpaint_mask: Any,
    secondary_control_image_latents: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    multi_control_scale_schedule: Any,
    model_args: Any,
) -> Any:
    return _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_img2img_denoising_loop_path(
        module,
        init_latents,
        latent_noise,
        inpaint_mask,
        (control_image_latents, secondary_control_image_latents),
        (inpaint_mask, secondary_inpaint_mask),
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        strength,
        num_inference_steps,
        guidance_scale,
        multi_control_scale_schedule,
        model_args,
        "SD3 multi-image inpaint img2img",
    )


def _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_setup(
        module
    )
    control_guidance_start = module.tensor([0.0, 0.4], dtype=module.float32)
    control_guidance_end = module.tensor([0.76, 1.0], dtype=module.float32)
    return (*args[:15], control_guidance_start, control_guidance_end, args[15])


def _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_zero_width_windowed_img2img_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = (
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup(
            module
        )
    )
    control_guidance_start = module.tensor([0.0, 0.4], dtype=module.float32)
    control_guidance_end = module.tensor([0.76, 0.4], dtype=module.float32)
    return (*args[:15], control_guidance_start, control_guidance_end, args[17])


def _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_disabled_windowed_img2img_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = (
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup(
            module
        )
    )
    control_guidance_start = module.tensor([0.0, 0.4], dtype=module.float32)
    control_guidance_end = module.tensor([0.0, 0.4], dtype=module.float32)
    return (*args[:15], control_guidance_start, control_guidance_end, args[17])


def _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path(
    module: Any,
    init_latents: Any,
    latent_noise: Any,
    inpaint_mask: Any,
    control_image_latents: Any,
    secondary_inpaint_mask: Any,
    secondary_control_image_latents: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    multi_control_scale_schedule: Any,
    control_guidance_start: Any,
    control_guidance_end: Any,
    model_args: Any,
) -> Any:
    init_timestep, start_index, latents = _stable_diffusion_sd3_prepare_img2img_latents_path(
        module,
        init_latents,
        latent_noise,
        sigmas,
        strength,
        num_inference_steps,
        "SD3 multi-image inpaint windowed img2img",
    )
    if init_timestep == 0:
        return latents

    windowed_control_scale_schedule = _stable_diffusion_sd3_multi_control_guidance_window_schedule_path(
        module,
        multi_control_scale_schedule,
        control_guidance_start,
        control_guidance_end,
    )
    return _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_prepared_img2img_denoising_loop_path(
        module,
        latents,
        init_latents,
        inpaint_mask,
        (control_image_latents, secondary_control_image_latents),
        (inpaint_mask, secondary_inpaint_mask),
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        start_index,
        init_timestep,
        guidance_scale,
        windowed_control_scale_schedule,
        model_args,
    )


def _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_setup(module)
    init_latents = args[1]
    latent_noise = (module.flip(init_latents, (-1,)) * 0.59 + module.flip(init_latents, (-2,)) * 0.41).contiguous(
        memory_format=module.channels_last
    )
    return (
        init_latents,
        latent_noise,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9],
        0.6,
        5,
        args[12],
        args[13],
        args[14],
    )


def _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_path(
    module: Any,
    init_latents: Any,
    latent_noise: Any,
    preserve_mask: Any,
    control_image_latents_by_controlnet: Any,
    inpaint_masks_by_controlnet: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    multi_control_scale_schedule: Any,
    model_args: Any,
) -> Any:
    return _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_img2img_denoising_loop_path(
        module,
        init_latents,
        latent_noise,
        preserve_mask,
        control_image_latents_by_controlnet,
        inpaint_masks_by_controlnet,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        strength,
        num_inference_steps,
        guidance_scale,
        multi_control_scale_schedule,
        model_args,
        "SD3 triple-image inpaint img2img",
    )


def _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_setup(
        module
    )
    control_guidance_start = module.tensor([0.0, 0.24, 0.52], dtype=module.float32)
    control_guidance_end = module.tensor([0.72, 0.88, 1.0], dtype=module.float32)
    return (*args[:14], control_guidance_start, control_guidance_end, args[14])


def _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_zero_width_windowed_img2img_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = (
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup(
            module
        )
    )
    control_guidance_start = module.tensor([0.0, 0.24, 0.52], dtype=module.float32)
    control_guidance_end = module.tensor([0.72, 0.24, 1.0], dtype=module.float32)
    return (*args[:14], control_guidance_start, control_guidance_end, args[16])


def _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_disabled_windowed_img2img_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    args = (
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup(
            module
        )
    )
    control_guidance_start = module.tensor([0.0, 0.24, 0.52], dtype=module.float32)
    control_guidance_end = module.tensor([0.0, 0.24, 0.52], dtype=module.float32)
    return (*args[:14], control_guidance_start, control_guidance_end, args[16])


def _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path(
    module: Any,
    init_latents: Any,
    latent_noise: Any,
    preserve_mask: Any,
    control_image_latents_by_controlnet: Any,
    inpaint_masks_by_controlnet: Any,
    negative_prompt_embeds: Any,
    prompt_embeds: Any,
    negative_pooled_prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    sigmas: Any,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    multi_control_scale_schedule: Any,
    control_guidance_start: Any,
    control_guidance_end: Any,
    model_args: Any,
) -> Any:
    init_timestep, start_index, latents = _stable_diffusion_sd3_prepare_img2img_latents_path(
        module,
        init_latents,
        latent_noise,
        sigmas,
        strength,
        num_inference_steps,
        "SD3 triple-image inpaint windowed img2img",
    )
    if init_timestep == 0:
        return latents

    windowed_control_scale_schedule = _stable_diffusion_sd3_multi_control_guidance_window_schedule_path(
        module,
        multi_control_scale_schedule,
        control_guidance_start,
        control_guidance_end,
    )
    return _stable_diffusion_sd3_rectangular_pooled_generic_multi_image_inpaint_controlnet_prepared_img2img_denoising_loop_path(
        module,
        latents,
        init_latents,
        preserve_mask,
        control_image_latents_by_controlnet,
        inpaint_masks_by_controlnet,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        start_index,
        init_timestep,
        guidance_scale,
        windowed_control_scale_schedule,
        model_args,
    )


def _case_stable_diffusion_half_unet_group_norm_setup(module: Any) -> tuple[Any, Any, Any]:
    hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
    hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
    norm_weight = (1.0 + module.arange(0, 320, dtype=module.float32) * 0.0007).to(dtype=module.float16)
    norm_bias = (module.arange(0, 320, dtype=module.float32) * 0.0003 - 0.04).to(dtype=module.float16)
    return hidden, norm_weight, norm_bias


def _case_stable_diffusion_half_channels_last_group_norm_setup(module: Any) -> tuple[Any, Any, Any]:
    hidden, norm_weight, norm_bias = _case_stable_diffusion_half_unet_group_norm_setup(module)
    return hidden.contiguous(memory_format=module.channels_last), norm_weight, norm_bias


def _stable_diffusion_half_unet_group_norm_path(module: Any, hidden: Any, norm_weight: Any, norm_bias: Any) -> Any:
    return module.nn.functional.group_norm(hidden, 32, norm_weight, norm_bias, eps=1e-5)


def _case_stable_diffusion_half_channels_last_silu_setup(module: Any) -> tuple[Any]:
    hidden, _, _ = _case_stable_diffusion_half_channels_last_group_norm_setup(module)
    return (hidden,)


def _stable_diffusion_half_channels_last_silu_path(module: Any, hidden: Any) -> Any:
    return module.nn.functional.silu(hidden)


def _case_stable_diffusion_half_attention_projection_setup(module: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    hidden_base = module.arange(0, 2 * 77 * 64, dtype=module.float32).reshape(2, 77, 64)
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
    return hidden, q_weight, k_weight, v_weight, out_weight, bias


def _case_stable_diffusion_half_clip_text_encoder_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    batch, sequence, width = 2, 77, 64
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
    return (
        input_ids,
        position_ids,
        token_weight,
        position_weight,
        norm_weight,
        norm_bias,
        q_weight,
        k_weight,
        v_weight,
        out_weight,
        bias,
    )


def _stable_diffusion_half_clip_text_encoder_path(
    module: Any,
    input_ids: Any,
    position_ids: Any,
    token_weight: Any,
    position_weight: Any,
    norm_weight: Any,
    norm_bias: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    out_weight: Any,
    bias: Any,
) -> Any:
    batch, sequence = input_ids.shape
    width = token_weight.shape[1]
    heads = 4
    head_dim = width // heads
    hidden = module.nn.functional.embedding(input_ids, token_weight)
    hidden = hidden + module.nn.functional.embedding(position_ids, position_weight)
    hidden = module.nn.functional.layer_norm(hidden, (width,), norm_weight, norm_bias, eps=1e-5)
    query = module.nn.functional.linear(hidden, q_weight, bias).reshape(batch, sequence, heads, head_dim).transpose(1, 2)
    key = module.nn.functional.linear(hidden, k_weight, bias).reshape(batch, sequence, heads, head_dim).transpose(1, 2)
    value = module.nn.functional.linear(hidden, v_weight, bias).reshape(batch, sequence, heads, head_dim).transpose(1, 2)
    attended = module.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
    attended = attended.transpose(1, 2).reshape(batch, sequence, width)
    return module.nn.functional.linear(attended, out_weight, bias)


def _case_stable_diffusion_clip_pooled_projection_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    batch, sequence, width = 2, 77, 64
    input_ids = module.tensor(
        [
            [((row * 7 + col * 5) % 32) if col != 40 + row else 99 for col in range(sequence)]
            for row in range(batch)
        ],
        dtype=module.long,
    )
    hidden_base = module.arange(0, batch * sequence * width, dtype=module.float32).reshape(batch, sequence, width)
    hidden = (module.sin(hidden_base * 0.017) * 0.25).to(dtype=module.float16)
    proj_weight = (module.arange(0, width * width, dtype=module.float32).reshape(width, width) / 4096.0 - 0.5).to(
        dtype=module.float16
    )
    proj_bias = (module.arange(0, width, dtype=module.float32) / 2048.0 - 0.015).to(dtype=module.float16)
    return input_ids, hidden, proj_weight, proj_bias


def _stable_diffusion_clip_pooled_projection_path(
    module: Any,
    input_ids: Any,
    hidden: Any,
    proj_weight: Any,
    proj_bias: Any,
) -> Any:
    batch_indices = module.arange(hidden.shape[0], dtype=module.long)
    eos_indices = input_ids.argmax(dim=-1)
    pooled = hidden[batch_indices, eos_indices]
    return module.nn.functional.linear(pooled, proj_weight, proj_bias)


def _case_stable_diffusion_sdxl_dual_prompt_encode_setup(module: Any) -> tuple[Any, ...]:
    batch, tokens, width_1, width_2 = 2, 77, 768, 1280
    input_ids_2 = module.tensor(
        [
            [159 if col == 70 + row else (row * 13 + col * 7) % 128 for col in range(tokens)]
            for row in range(batch)
        ],
        dtype=module.long,
    )
    negative_input_ids_2 = module.tensor(
        [
            [159 if col == 66 + row else (row * 17 + col * 5) % 128 for col in range(tokens)]
            for row in range(batch)
        ],
        dtype=module.long,
    )
    hidden_1_base = module.arange(0, batch * tokens * width_1, dtype=module.float32).reshape(batch, tokens, width_1)
    hidden_2_base = module.arange(0, batch * tokens * width_2, dtype=module.float32).reshape(batch, tokens, width_2)
    hidden_1 = (module.sin((hidden_1_base % 1009) * 0.007) * 0.25).to(dtype=module.float16)
    hidden_2 = (module.cos((hidden_2_base % 1223) * 0.005) * 0.2).to(dtype=module.float16)
    negative_hidden_1 = (module.flip(hidden_1, (1,)) * 0.5 - 0.08).contiguous()
    negative_hidden_2 = (module.flip(hidden_2, (1,)) * 0.5 + 0.04).contiguous()
    norm1_weight = (0.75 + (module.arange(0, width_1, dtype=module.float32) % 257) / 2048.0).to(dtype=module.float16)
    norm1_bias = (((module.arange(0, width_1, dtype=module.float32) % 127) - 63.0) / 4096.0).to(
        dtype=module.float16
    )
    norm2_weight = (0.8 + (module.arange(0, width_2, dtype=module.float32) % 293) / 2048.0).to(dtype=module.float16)
    norm2_bias = (((module.arange(0, width_2, dtype=module.float32) % 149) - 74.0) / 4096.0).to(
        dtype=module.float16
    )
    pooled_base = (
        module.arange(0, width_2 * width_2, dtype=module.float32).reshape(width_2, width_2) % 1543
    ) / 65536.0 - 0.011
    pooled_weight = pooled_base.to(dtype=module.float16)
    pooled_bias = ((module.arange(0, width_2, dtype=module.float32) % 173) / 8192.0 - 0.01).to(
        dtype=module.float16
    )
    return (
        input_ids_2,
        negative_input_ids_2,
        hidden_1,
        hidden_2,
        negative_hidden_1,
        negative_hidden_2,
        norm1_weight,
        norm1_bias,
        norm2_weight,
        norm2_bias,
        pooled_weight,
        pooled_bias,
    )


def _stable_diffusion_sdxl_dual_prompt_encode_path(
    module: Any,
    input_ids_2: Any,
    negative_input_ids_2: Any,
    hidden_1: Any,
    hidden_2: Any,
    negative_hidden_1: Any,
    negative_hidden_2: Any,
    norm1_weight: Any,
    norm1_bias: Any,
    norm2_weight: Any,
    norm2_bias: Any,
    pooled_weight: Any,
    pooled_bias: Any,
) -> Any:
    batch, tokens, width_1, width_2, num_images_per_prompt = 2, 77, 768, 1280, 2
    hidden_1 = module.nn.functional.layer_norm(hidden_1, (width_1,), norm1_weight, norm1_bias, eps=1e-5)
    hidden_2 = module.nn.functional.layer_norm(hidden_2, (width_2,), norm2_weight, norm2_bias, eps=1e-5)
    negative_hidden_1 = module.nn.functional.layer_norm(
        negative_hidden_1, (width_1,), norm1_weight, norm1_bias, eps=1e-5
    )
    negative_hidden_2 = module.nn.functional.layer_norm(
        negative_hidden_2, (width_2,), norm2_weight, norm2_bias, eps=1e-5
    )

    batch_indices = module.arange(batch, dtype=module.long)
    pooled = hidden_2[batch_indices, input_ids_2.argmax(dim=-1)]
    negative_pooled = negative_hidden_2[batch_indices, negative_input_ids_2.argmax(dim=-1)]
    pooled = module.nn.functional.linear(pooled, pooled_weight, pooled_bias)
    negative_pooled = module.nn.functional.linear(negative_pooled, pooled_weight, pooled_bias)

    prompt_embeds = module.cat([hidden_1, hidden_2], dim=-1)
    negative_prompt_embeds = module.cat([negative_hidden_1, negative_hidden_2], dim=-1)
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).reshape(
        batch * num_images_per_prompt, tokens, width_1 + width_2
    )
    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1).reshape(
        batch * num_images_per_prompt, tokens, width_1 + width_2
    )
    pooled = pooled.repeat(1, num_images_per_prompt).reshape(batch * num_images_per_prompt, width_2)
    negative_pooled = negative_pooled.repeat(1, num_images_per_prompt).reshape(batch * num_images_per_prompt, width_2)
    return module.cat([negative_prompt_embeds, prompt_embeds], dim=0), module.cat([negative_pooled, pooled], dim=0)


def _stable_diffusion_half_attention_projection_path(
    module: Any,
    hidden: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    out_weight: Any,
    bias: Any,
) -> Any:
    batch, tokens, channels = hidden.shape
    head_dim = channels // 4
    query = module.nn.functional.linear(hidden, q_weight, bias).reshape(batch, tokens, 4, head_dim).transpose(1, 2)
    key = module.nn.functional.linear(hidden, k_weight, bias).reshape(batch, tokens, 4, head_dim).transpose(1, 2)
    value = module.nn.functional.linear(hidden, v_weight, bias).reshape(batch, tokens, 4, head_dim).transpose(1, 2)
    attended = module.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
    attended = attended.transpose(1, 2).reshape(batch, tokens, channels)
    return module.nn.functional.linear(attended, out_weight, bias)


def _case_stable_diffusion_masked_attention_setup(module: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    query, key, value = _case_sdpa_setup(module)
    rows = module.arange(0, 16, dtype=module.long)
    cols = module.flip(rows, (0,))
    batch_indices = module.zeros((16,), dtype=module.long)
    head_indices = module.zeros((16,), dtype=module.long)
    mask_values = module.full((16,), -10000.0, dtype=module.float32)
    return query, key, value, batch_indices, head_indices, rows, cols, mask_values


def _stable_diffusion_masked_attention_path(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    batch_indices: Any,
    head_indices: Any,
    rows: Any,
    cols: Any,
    mask_values: Any,
) -> Any:
    mask = module.zeros((1, 1, query.shape[-2], key.shape[-2]), dtype=query.dtype)
    mask = module.index_put(mask, (batch_indices, head_indices, rows, cols), mask_values)
    return module.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)


def _case_stable_diffusion_bool_masked_attention_setup(module: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    query, key, value = _case_sdpa_setup(module)
    rows = module.arange(0, 16, dtype=module.long)
    cols = module.flip(rows, (0,))
    batch_indices = module.zeros((16,), dtype=module.long)
    head_indices = module.zeros((16,), dtype=module.long)
    mask_values = module.zeros((16,), dtype=module.bool)
    return query, key, value, batch_indices, head_indices, rows, cols, mask_values


def _stable_diffusion_bool_masked_attention_path(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    batch_indices: Any,
    head_indices: Any,
    rows: Any,
    cols: Any,
    mask_values: Any,
) -> Any:
    mask = module.ones((1, 1, query.shape[-2], key.shape[-2]), dtype=module.bool)
    mask = module.index_put(mask, (batch_indices, head_indices, rows, cols), mask_values)
    return module.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)


def _case_stable_diffusion_dynamic_thresholding_setup(module: Any) -> tuple[Any]:
    sample = module.arange(0, 2 * 4 * 8 * 8, dtype=module.float32).reshape(2, 4, 8, 8) / 64.0 - 4.0
    return (sample,)


def _case_stable_diffusion_dynamic_thresholding_large_setup(module: Any) -> tuple[Any]:
    sample = module.arange(0, 2 * 4 * 64 * 64, dtype=module.float32).reshape(2, 4, 64, 64) / 1024.0 - 4.0
    return (sample,)


def _stable_diffusion_dynamic_thresholding_path(module: Any, sample: Any) -> Any:
    batch = sample.shape[0]
    flattened = sample.reshape(batch, -1)
    threshold = module.quantile(flattened.abs(), 0.8, dim=1)
    threshold = module.clamp(threshold, min=1.0, max=2.0).unsqueeze(1)
    return (module.clamp(flattened, min=-threshold, max=threshold) / threshold).reshape(sample.shape)


def _case_stable_diffusion_guidance_rescale_setup(module: Any) -> tuple[Any, Any]:
    noise_cfg = module.arange(0, 2 * 4 * 8 * 8, dtype=module.float32).reshape(2, 4, 8, 8) / 128.0 - 1.0
    noise_text = module.flip(noise_cfg, (-1,)) * 0.75 + 0.125
    return noise_cfg, noise_text


def _case_stable_diffusion_half_channels_last_guidance_rescale_setup(module: Any) -> tuple[Any, Any]:
    noise_cfg = module.arange(0, 2 * 4 * 64 * 64, dtype=module.float32).reshape(2, 4, 64, 64) / 4096.0 - 1.0
    noise_cfg = noise_cfg.to(dtype=module.float16).contiguous(memory_format=module.channels_last)
    noise_text = (module.flip(noise_cfg, (-1,)) * 0.75 + 0.125).contiguous(memory_format=module.channels_last)
    return noise_cfg, noise_text


def _case_stable_diffusion_latent_flip_setup(module: Any) -> tuple[Any]:
    return (module.arange(0, 2 * 4 * 8 * 8, dtype=module.float32).reshape(2, 4, 8, 8),)


def _case_stable_diffusion_half_channels_last_conv_setup(module: Any) -> tuple[Any, Any]:
    hidden_base = module.arange(0, 1 * 4 * 16 * 16, dtype=module.float32).reshape(1, 4, 16, 16)
    hidden = (module.sin(hidden_base * 0.017) * 0.5).to(dtype=module.float16)
    hidden = hidden.contiguous(memory_format=module.channels_last)
    weight = (module.cos(module.arange(0, 8 * 4 * 3 * 3, dtype=module.float32).reshape(8, 4, 3, 3) * 0.013) * 0.05).to(
        dtype=module.float16
    )
    return hidden, weight


def _case_stable_diffusion_half_channels_last_projection_conv_setup(module: Any) -> tuple[Any, Any, Any]:
    hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
    hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
    hidden = hidden.contiguous(memory_format=module.channels_last)
    weight_base = module.arange(0, 320 * 320, dtype=module.float32).reshape(320, 320, 1, 1)
    weight = (module.cos(weight_base * 0.007) * 0.02).to(dtype=module.float16)
    bias = (module.arange(0, 320, dtype=module.float32) / 4096.0 - 0.02).to(dtype=module.float16)
    return hidden, weight, bias


def _case_stable_diffusion_half_channels_last_wide_conv_setup(module: Any) -> tuple[Any, Any, Any]:
    hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
    hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
    hidden = hidden.contiguous(memory_format=module.channels_last)
    weight_base = module.arange(0, 320 * 320 * 3 * 3, dtype=module.float32).reshape(320, 320, 3, 3)
    weight = (module.cos(weight_base * 0.007) * 0.005).to(dtype=module.float16)
    bias = (module.arange(0, 320, dtype=module.float32) / 4096.0 - 0.02).to(dtype=module.float16)
    return hidden, weight, bias


def _case_stable_diffusion_half_channels_last_lora_conv_setup(module: Any) -> tuple[Any, Any, Any, Any, Any]:
    hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
    hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
    hidden = hidden.contiguous(memory_format=module.channels_last)
    weight_base = module.arange(0, 320 * 320 * 3 * 3, dtype=module.float32).reshape(320, 320, 3, 3)
    weight = (module.cos(weight_base * 0.007) * 0.005).to(dtype=module.float16)
    down_base = module.arange(0, 8 * 320, dtype=module.float32).reshape(8, 320, 1, 1)
    up_base = module.arange(0, 320 * 8, dtype=module.float32).reshape(320, 8, 1, 1)
    down_weight = (module.sin(down_base * 0.011) * 0.02).to(dtype=module.float16)
    up_weight = (module.cos(up_base * 0.017) * 0.02).to(dtype=module.float16)
    bias = (module.arange(0, 320, dtype=module.float32) / 4096.0 - 0.02).to(dtype=module.float16)
    return hidden, weight, bias, down_weight, up_weight


def _case_stable_diffusion_half_channels_last_downsample_conv_setup(module: Any) -> tuple[Any, Any, Any]:
    hidden_base = module.arange(0, 1 * 320 * 32 * 32, dtype=module.float32).reshape(1, 320, 32, 32)
    hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
    hidden = hidden.contiguous(memory_format=module.channels_last)
    weight_base = module.arange(0, 320 * 320 * 3 * 3, dtype=module.float32).reshape(320, 320, 3, 3)
    weight = (module.cos(weight_base * 0.007) * 0.005).to(dtype=module.float16)
    bias = (module.arange(0, 320, dtype=module.float32) / 4096.0 - 0.02).to(dtype=module.float16)
    return hidden, weight, bias


def _case_stable_diffusion_half_channels_last_upsample_conv_setup(module: Any) -> tuple[Any, Any, Any]:
    hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
    hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
    hidden = hidden.contiguous(memory_format=module.channels_last)
    weight_base = module.arange(0, 320 * 320 * 3 * 3, dtype=module.float32).reshape(320, 320, 3, 3)
    weight = (module.cos(weight_base * 0.007) * 0.005).to(dtype=module.float16)
    bias = (module.arange(0, 320, dtype=module.float32) / 4096.0 - 0.02).to(dtype=module.float16)
    return hidden, weight, bias


def _case_stable_diffusion_half_channels_last_skip_cat_conv_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
    hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
    hidden = hidden.contiguous(memory_format=module.channels_last)
    skip = module.flip(hidden, (-1,)).contiguous(memory_format=module.channels_last)
    weight_base = module.arange(0, 320 * 640 * 3 * 3, dtype=module.float32).reshape(320, 640, 3, 3)
    weight = (module.cos(weight_base * 0.005) * 0.004).to(dtype=module.float16)
    bias = (module.arange(0, 320, dtype=module.float32) / 4096.0 - 0.02).to(dtype=module.float16)
    return hidden, skip, weight, bias


def _case_stable_diffusion_half_channels_last_timestep_add_setup(module: Any) -> tuple[Any, Any]:
    hidden_base = module.arange(0, 1 * 320 * 16 * 16, dtype=module.float32).reshape(1, 320, 16, 16)
    hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
    hidden = hidden.contiguous(memory_format=module.channels_last)
    temb = (module.arange(0, 320, dtype=module.float32).reshape(1, 320, 1, 1) / 4096.0 - 0.02).to(
        dtype=module.float16
    )
    return hidden, temb


def _case_stable_diffusion_half_channels_last_resnet_scale_shift_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    channels, spatial = 320, 16
    hidden_base = module.arange(0, 1 * channels * spatial * spatial, dtype=module.float32).reshape(
        1,
        channels,
        spatial,
        spatial,
    )
    hidden = (module.sin(hidden_base * 0.013) * 0.25).to(dtype=module.float16)
    hidden = hidden.contiguous(memory_format=module.channels_last)
    temb_base = module.arange(0, 1 * channels * 2, dtype=module.float32).reshape(1, channels * 2, 1, 1)
    temb = (module.cos(temb_base * 0.017) * 0.04).to(dtype=module.float16)
    norm1_weight = (0.75 + (module.arange(0, channels, dtype=module.float32) % 257) / 2048.0).to(
        dtype=module.float16
    )
    norm1_bias = (((module.arange(0, channels, dtype=module.float32) % 127) - 63.0) / 4096.0).to(
        dtype=module.float16
    )
    norm2_weight = (0.8 + (module.arange(0, channels, dtype=module.float32) % 251) / 2048.0).to(
        dtype=module.float16
    )
    norm2_bias = (((module.arange(0, channels, dtype=module.float32) % 131) - 65.0) / 4096.0).to(
        dtype=module.float16
    )
    conv1_base = module.arange(0, channels * channels * 3 * 3, dtype=module.float32).reshape(
        channels,
        channels,
        3,
        3,
    )
    conv2_base = module.arange(0, channels * channels * 3 * 3, dtype=module.float32).reshape(
        channels,
        channels,
        3,
        3,
    )
    shortcut_base = module.arange(0, channels * channels, dtype=module.float32).reshape(channels, channels, 1, 1)
    conv1_weight = (module.cos(conv1_base * 0.007) * 0.004).to(dtype=module.float16)
    conv2_weight = (module.sin(conv2_base * 0.011) * 0.004).to(dtype=module.float16)
    shortcut_weight = (module.cos(shortcut_base * 0.013) * 0.02).to(dtype=module.float16)
    conv_bias = (module.arange(0, channels, dtype=module.float32) / 8192.0 - 0.015).to(dtype=module.float16)
    return (
        hidden,
        temb,
        norm1_weight,
        norm1_bias,
        norm2_weight,
        norm2_bias,
        conv1_weight,
        conv2_weight,
        shortcut_weight,
        conv_bias,
    )


def _case_stable_diffusion_half_channels_last_classifier_free_guidance_setup(module: Any) -> tuple[Any, Any]:
    latent_base = module.arange(0, 2 * 4 * 64 * 64, dtype=module.float32).reshape(2, 4, 64, 64)
    latents = (module.sin(latent_base * 0.017) * 0.5).to(dtype=module.float16)
    latents = latents.contiguous(memory_format=module.channels_last)
    uncond = latents * 0.75 + 0.125
    text = (module.flip(latents, (-1,)).contiguous(memory_format=module.channels_last) * 0.5) - 0.25
    return uncond, text


def _case_stable_diffusion_half_channels_last_scheduler_scale_setup(module: Any) -> tuple[Any, Any]:
    latent_base = module.arange(0, 2 * 4 * 64 * 64, dtype=module.float32).reshape(2, 4, 64, 64)
    latents = (module.sin(latent_base * 0.017) * 0.5).to(dtype=module.float16)
    latents = latents.contiguous(memory_format=module.channels_last)
    sigma = module.tensor(2.5, dtype=module.float16).reshape(1, 1, 1, 1)
    return latents, sigma


def _case_stable_diffusion_half_channels_last_scheduler_add_noise_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    latent_base = module.arange(0, 2 * 4 * 64 * 64, dtype=module.float32).reshape(2, 4, 64, 64)
    latents = (module.sin(latent_base * 0.013) * 0.35).to(dtype=module.float16)
    latents = latents.contiguous(memory_format=module.channels_last)
    noise = (module.cos(latent_base * 0.017) * 0.25).to(dtype=module.float16)
    noise = noise.contiguous(memory_format=module.channels_last)
    alphas = module.tensor([0.95, 0.72, 0.33, 0.11], dtype=module.float16)
    timesteps = module.tensor([1, 2], dtype=module.long)
    return latents, noise, alphas, timesteps


def _case_stable_diffusion_half_channels_last_mask_blend_setup(module: Any) -> tuple[Any, Any, Any]:
    latent_base = module.arange(0, 2 * 4 * 64 * 64, dtype=module.float32).reshape(2, 4, 64, 64)
    latents = (module.sin(latent_base * 0.017) * 0.5).to(dtype=module.float16)
    latents = latents.contiguous(memory_format=module.channels_last)
    init_latents = module.flip(latents, (-1,)).contiguous(memory_format=module.channels_last)
    mask = (module.arange(0, 2 * 1 * 64 * 64, dtype=module.float32).reshape(2, 1, 64, 64) % 2).to(
        dtype=module.float16
    )
    mask = mask.contiguous(memory_format=module.channels_last)
    return latents, init_latents, mask


def _case_stable_diffusion_half_channels_last_controlnet_merge_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    specs = ((2, 320, 32, 32), (2, 320, 16, 16), (2, 640, 8, 8), (2, 1280, 4, 4))
    samples = []
    residuals = []
    for offset, (batch, channels, height, width) in enumerate(specs):
        base = module.arange(0, batch * channels * height * width, dtype=module.float32).reshape(
            batch,
            channels,
            height,
            width,
        )
        sample = (module.sin(base * 0.011 + offset) * 0.25).to(dtype=module.float16)
        residual = (module.cos(base * 0.013 + offset) * 0.125).to(dtype=module.float16)
        samples.append(sample.contiguous(memory_format=module.channels_last))
        residuals.append(residual.contiguous(memory_format=module.channels_last))

    mid_base = module.arange(0, 2 * 1280 * 4 * 4, dtype=module.float32).reshape(2, 1280, 4, 4)
    mid = (module.sin(mid_base * 0.017) * 0.2).to(dtype=module.float16)
    mid_residual = (module.cos(mid_base * 0.019) * 0.1).to(dtype=module.float16)
    return (
        tuple(samples),
        tuple(residuals),
        mid.contiguous(memory_format=module.channels_last),
        mid_residual.contiguous(memory_format=module.channels_last),
    )


def _case_stable_diffusion_half_channels_last_3d_timestep_add_setup(module: Any) -> tuple[Any, Any]:
    channels = 320
    hidden_base = module.arange(0, 1 * channels * 4 * 8 * 8, dtype=module.float32).reshape(
        1,
        channels,
        4,
        8,
        8,
    )
    hidden = (module.sin(hidden_base * 0.017) * 0.5).to(dtype=module.float16)
    hidden = hidden.contiguous(memory_format=module.channels_last_3d)
    temb = (module.arange(0, channels, dtype=module.float32).reshape(1, channels, 1, 1, 1) / 4096.0 - 0.02).to(
        dtype=module.float16
    )
    return hidden, temb


def _case_stable_diffusion_half_channels_last_3d_classifier_free_guidance_setup(module: Any) -> tuple[Any, Any]:
    channels = 4
    latent_base = module.arange(0, 2 * channels * 4 * 32 * 32, dtype=module.float32).reshape(
        2,
        channels,
        4,
        32,
        32,
    )
    latents = (module.sin(latent_base * 0.017) * 0.5).to(dtype=module.float16)
    latents = latents.contiguous(memory_format=module.channels_last_3d)
    uncond = latents * 0.75 + 0.125
    text = (module.flip(latents, (-1,)).contiguous(memory_format=module.channels_last_3d) * 0.5) - 0.25
    return uncond, text


def _case_stable_diffusion_half_channels_last_3d_scheduler_scale_setup(module: Any) -> tuple[Any, Any]:
    channels = 4
    latent_base = module.arange(0, 2 * channels * 4 * 32 * 32, dtype=module.float32).reshape(
        2,
        channels,
        4,
        32,
        32,
    )
    latents = (module.cos(latent_base * 0.011) * 0.5).to(dtype=module.float16)
    latents = latents.contiguous(memory_format=module.channels_last_3d)
    sigma = module.tensor(1.75, dtype=module.float16).reshape(1, 1, 1, 1, 1)
    return latents, sigma


def _case_stable_diffusion_half_channels_last_3d_scheduler_add_noise_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    channels = 4
    latent_base = module.arange(0, 2 * channels * 4 * 32 * 32, dtype=module.float32).reshape(
        2,
        channels,
        4,
        32,
        32,
    )
    latents = (module.cos(latent_base * 0.011) * 0.35).to(dtype=module.float16)
    latents = latents.contiguous(memory_format=module.channels_last_3d)
    noise = (module.sin(latent_base * 0.019) * 0.25).to(dtype=module.float16)
    noise = noise.contiguous(memory_format=module.channels_last_3d)
    alphas = module.tensor([0.95, 0.72, 0.33, 0.11], dtype=module.float16)
    timesteps = module.tensor([1, 2], dtype=module.long)
    return latents, noise, alphas, timesteps


def _case_stable_diffusion_half_channels_last_3d_mask_blend_setup(module: Any) -> tuple[Any, Any, Any]:
    channels = 4
    latent_base = module.arange(0, 2 * channels * 4 * 32 * 32, dtype=module.float32).reshape(
        2,
        channels,
        4,
        32,
        32,
    )
    latents = (module.cos(latent_base * 0.011) * 0.5).to(dtype=module.float16)
    latents = latents.contiguous(memory_format=module.channels_last_3d)
    init_latents = module.flip(latents, (-1,)).contiguous(memory_format=module.channels_last_3d)
    mask = (module.arange(0, 2 * 1 * 4 * 32 * 32, dtype=module.float32).reshape(2, 1, 4, 32, 32) % 2).to(
        dtype=module.float16
    )
    mask = mask.contiguous(memory_format=module.channels_last_3d)
    return latents, init_latents, mask


def _case_stable_diffusion_half_channels_last_3d_video_upsample_setup(module: Any) -> tuple[Any]:
    channels = 320
    values = [float((index % 257) - 128) / 16.0 for index in range(1 * channels * 4 * 8 * 8)]
    hidden = module.tensor(values, dtype=module.float16).reshape(
        1,
        channels,
        4,
        8,
        8,
    )
    return (hidden.contiguous(memory_format=module.channels_last_3d),)


def _case_stable_diffusion_half_video_upsample_setup(module: Any) -> tuple[Any]:
    channels = 320
    values = [float((index % 257) - 128) / 16.0 for index in range(1 * channels * 4 * 8 * 8)]
    hidden = module.tensor(values, dtype=module.float16).reshape(
        1,
        channels,
        4,
        8,
        8,
    )
    return (hidden,)


def _stable_diffusion_half_channels_last_conv_path(module: Any, hidden: Any, weight: Any) -> Any:
    return module.nn.functional.conv2d(hidden, weight, None, padding=1)


def _stable_diffusion_half_channels_last_timestep_add_path(module: Any, hidden: Any, temb: Any) -> Any:
    return module.nn.functional.silu(hidden + temb)


def _stable_diffusion_half_channels_last_resnet_scale_shift_path(
    module: Any,
    hidden: Any,
    temb: Any,
    norm1_weight: Any,
    norm1_bias: Any,
    norm2_weight: Any,
    norm2_bias: Any,
    conv1_weight: Any,
    conv2_weight: Any,
    shortcut_weight: Any,
    conv_bias: Any,
) -> Any:
    residual = hidden
    hidden = module.nn.functional.group_norm(hidden, 32, norm1_weight, norm1_bias, eps=1e-5)
    hidden = module.nn.functional.silu(hidden)
    hidden = module.nn.functional.conv2d(hidden, conv1_weight, conv_bias, padding=1)
    hidden = module.nn.functional.group_norm(hidden, 32, norm2_weight, norm2_bias, eps=1e-5)
    scale, shift = temb.chunk(2, dim=1)
    hidden = hidden * (1.0 + scale) + shift
    hidden = module.nn.functional.silu(hidden)
    hidden = module.nn.functional.conv2d(hidden, conv2_weight, conv_bias, padding=1)
    residual = module.nn.functional.conv2d(residual, shortcut_weight, conv_bias)
    return (hidden + residual) / 1.4142135623730951


def _stable_diffusion_half_channels_last_classifier_free_guidance_path(module: Any, uncond: Any, text: Any) -> Any:
    noise_pred = module.cat([uncond, text], dim=0)
    noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
    return noise_uncond + (noise_text - noise_uncond) * 7.5


def _stable_diffusion_half_channels_last_scheduler_scale_path(module: Any, latents: Any, sigma: Any) -> Any:
    return latents / ((sigma * sigma + 1.0) ** 0.5)


def _stable_diffusion_half_channels_last_scheduler_add_noise_path(
    module: Any,
    latents: Any,
    noise: Any,
    alphas: Any,
    timesteps: Any,
) -> Any:
    alpha = module.sqrt(alphas[timesteps]).reshape(2, 1, 1, 1)
    sigma = module.sqrt(1.0 - alphas[timesteps]).reshape(2, 1, 1, 1)
    return latents * alpha + noise * sigma


def _stable_diffusion_half_channels_last_mask_blend_path(module: Any, latents: Any, init_latents: Any, mask: Any) -> Any:
    return init_latents * mask + latents * (1.0 - mask)


def _stable_diffusion_half_channels_last_controlnet_merge_path(
    module: Any,
    samples: Any,
    residuals: Any,
    mid: Any,
    mid_residual: Any,
) -> Any:
    scales = module.logspace(-1.0, 0.0, len(samples) + 1, dtype=module.float32).to(dtype=module.float16) * 1.25
    merged = tuple(
        module.addcmul(sample, residual, scales[index])
        for index, (sample, residual) in enumerate(zip(samples, residuals))
    )
    return merged, module.addcmul(mid, mid_residual, scales[-1])


def _stable_diffusion_half_channels_last_3d_timestep_add_path(module: Any, hidden: Any, temb: Any) -> Any:
    return module.nn.functional.silu(hidden + temb)


def _stable_diffusion_half_channels_last_3d_classifier_free_guidance_path(module: Any, uncond: Any, text: Any) -> Any:
    noise_pred = module.cat([uncond, text], dim=0)
    noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
    return noise_uncond + (noise_text - noise_uncond) * 7.5


def _stable_diffusion_half_channels_last_3d_scheduler_scale_path(module: Any, latents: Any, sigma: Any) -> Any:
    return latents / ((sigma * sigma + 1.0) ** 0.5)


def _stable_diffusion_half_channels_last_3d_scheduler_add_noise_path(
    module: Any,
    latents: Any,
    noise: Any,
    alphas: Any,
    timesteps: Any,
) -> Any:
    alpha = module.sqrt(alphas[timesteps]).reshape(2, 1, 1, 1, 1)
    sigma = module.sqrt(1.0 - alphas[timesteps]).reshape(2, 1, 1, 1, 1)
    return latents * alpha + noise * sigma


def _stable_diffusion_half_channels_last_3d_mask_blend_path(
    module: Any,
    latents: Any,
    init_latents: Any,
    mask: Any,
) -> Any:
    return init_latents * mask + latents * (1.0 - mask)


def _stable_diffusion_half_channels_last_3d_video_upsample_path(module: Any, hidden: Any) -> Any:
    return module.nn.functional.interpolate(hidden, scale_factor=(1.0, 2.0, 2.0), mode="nearest")


def _stable_diffusion_half_video_upsample_path(module: Any, hidden: Any) -> Any:
    return module.nn.functional.interpolate(hidden, scale_factor=(1.0, 2.0, 2.0), mode="nearest")


def _stable_diffusion_half_channels_last_projection_conv_path(
    module: Any,
    hidden: Any,
    weight: Any,
    bias: Any,
) -> Any:
    return module.nn.functional.conv2d(hidden, weight, bias)


def _stable_diffusion_half_channels_last_wide_conv_path(
    module: Any,
    hidden: Any,
    weight: Any,
    bias: Any,
) -> Any:
    return module.nn.functional.conv2d(hidden, weight, bias, padding=1)


def _stable_diffusion_half_channels_last_lora_conv_path(
    module: Any,
    hidden: Any,
    weight: Any,
    bias: Any,
    down_weight: Any,
    up_weight: Any,
) -> Any:
    base = module.nn.functional.conv2d(hidden, weight, bias, padding=1)
    adapted = module.nn.functional.conv2d(module.nn.functional.conv2d(hidden, down_weight, None), up_weight, None)
    return base + adapted * 0.75


def _stable_diffusion_half_channels_last_repeated_wide_conv_path(
    module: Any,
    hidden: Any,
    weight: Any,
    bias: Any,
) -> Any:
    first = module.nn.functional.conv2d(hidden, weight, bias, padding=1)
    return module.nn.functional.conv2d(module.nn.functional.silu(first), weight, bias, padding=1)


def _stable_diffusion_half_channels_last_downsample_conv_path(
    module: Any,
    hidden: Any,
    weight: Any,
    bias: Any,
) -> Any:
    return module.nn.functional.conv2d(hidden, weight, bias, stride=2, padding=1)


def _stable_diffusion_half_channels_last_upsample_conv_path(
    module: Any,
    hidden: Any,
    weight: Any,
    bias: Any,
) -> Any:
    upsampled = module.nn.functional.interpolate(hidden, scale_factor=2.0, mode="nearest")
    return module.nn.functional.conv2d(upsampled, weight, bias, padding=1)


def _stable_diffusion_half_channels_last_skip_cat_conv_path(
    module: Any,
    hidden: Any,
    skip: Any,
    weight: Any,
    bias: Any,
) -> Any:
    merged = module.cat([hidden, skip], dim=1)
    return module.nn.functional.conv2d(merged, weight, bias, padding=1)


def _case_stable_diffusion_prompt_embedding_repeat_setup(module: Any) -> tuple[Any]:
    prompt = module.arange(0, 2 * 77 * 320, dtype=module.float32).reshape(2, 77, 320) / 1024.0
    return (prompt.to(dtype=module.float16),)


def _stable_diffusion_prompt_embedding_repeat_path(module: Any, prompt: Any) -> Any:
    return prompt.repeat(1, 2, 1).reshape(4, 77, 320)


def _case_stable_diffusion_timestep_embedding_mlp_setup(module: Any) -> tuple[Any, ...]:
    dim = 320
    hidden_dim = 1280
    timesteps = module.tensor([999.0, 500.0], dtype=module.float32)
    exponent = (
        -module.log(module.tensor(10000.0, dtype=module.float32))
        * module.arange(0, dim // 2, dtype=module.float32)
        / (dim // 2)
    )
    frequencies = module.exp(exponent)
    w1_base = module.arange(0, hidden_dim * dim, dtype=module.float32).reshape(hidden_dim, dim)
    w1 = (module.sin(w1_base * 0.0001) * 0.02).to(dtype=module.float16)
    b1 = (module.arange(0, hidden_dim, dtype=module.float32) / 8192.0 - 0.05).to(dtype=module.float16)
    w2_base = module.arange(0, hidden_dim * hidden_dim, dtype=module.float32).reshape(hidden_dim, hidden_dim)
    w2 = (module.cos(w2_base * 0.00007) * 0.015).to(dtype=module.float16)
    b2 = (module.arange(0, hidden_dim, dtype=module.float32) / 16384.0 - 0.025).to(dtype=module.float16)
    return timesteps, frequencies, w1, b1, w2, b2


def _stable_diffusion_timestep_embedding_mlp_path(
    module: Any,
    timesteps: Any,
    frequencies: Any,
    w1: Any,
    b1: Any,
    w2: Any,
    b2: Any,
) -> Any:
    args = timesteps.reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = module.cat([module.cos(args), module.sin(args)], dim=-1).to(dtype=module.float16)
    hidden = module.nn.functional.linear(embedding, w1, b1)
    hidden = module.nn.functional.silu(hidden)
    return module.nn.functional.linear(hidden, w2, b2)


def _stable_diffusion_guidance_rescale_path(module: Any, noise_cfg: Any, noise_text: Any) -> Any:
    cfg_flat = noise_cfg.reshape(noise_cfg.shape[0], -1)
    text_flat = noise_text.reshape(noise_text.shape[0], -1)
    text_std, text_mean = module.std_mean(text_flat, dim=1, correction=0, keepdim=True)
    cfg_std, cfg_mean = module.std_mean(cfg_flat, dim=1, correction=0, keepdim=True)
    rescaled = (cfg_flat - cfg_mean) * (text_std / module.clamp_min(cfg_std, 1e-6)) + text_mean
    return module.lerp(cfg_flat, rescaled, 0.7).reshape(noise_cfg.shape)


def _stable_diffusion_guidance_rescale_tail_std_path(module: Any, noise_cfg: Any, noise_text: Any) -> Any:
    reduce_dims = tuple(range(1, noise_cfg.ndim))
    text_std = noise_text.std(dim=reduce_dims, correction=0, keepdim=True)
    cfg_std = noise_cfg.std(dim=reduce_dims, correction=0, keepdim=True)
    rescaled = noise_cfg * (text_std / module.clamp_min(cfg_std, 1e-6))
    return module.lerp(noise_cfg, rescaled, 0.7)


def _stable_diffusion_half_channels_last_guidance_rescale_path(module: Any, noise_cfg: Any, noise_text: Any) -> Any:
    reduce_dims = tuple(range(1, noise_cfg.ndim))
    text_std = noise_text.std(dim=reduce_dims, correction=0, keepdim=True)
    cfg_std = noise_cfg.std(dim=reduce_dims, correction=0, keepdim=True)
    rescaled = noise_cfg * (text_std / module.clamp_min(cfg_std, 1e-6))
    return module.lerp(noise_cfg, rescaled, 0.7)


def _case_stable_diffusion_batched_timestep_table_setup(module: Any) -> tuple[Any, Any]:
    table = [
        [[999 - group, group % 64] for group in [index // 2 for index in range(128)]],
        [[999 - group, group % 64] for group in [index // 2 for index in range(128)]],
    ]
    timestep_table = module.tensor(table, dtype=module.long)
    sigmas = module.logspace(-2.0, 0.0, 64, dtype=module.float32).flip(0)
    return timestep_table, sigmas


def _stable_diffusion_batched_timestep_table_path(module: Any, timestep_table: Any, sigmas: Any) -> Any:
    unique_rows, inverse, counts = module.unique(
        timestep_table,
        dim=1,
        return_inverse=True,
        return_counts=True,
    )
    consecutive_rows, consecutive_inverse, consecutive_counts = module.unique_consecutive(
        timestep_table,
        dim=1,
        return_inverse=True,
        return_counts=True,
    )
    sigma_indices = unique_rows.select(2, 1)
    sigma_grid = sigmas.unsqueeze(0).broadcast_to(sigma_indices.shape[0], sigmas.shape[0])
    sigma_values = module.take_along_dim(sigma_grid, sigma_indices, dim=1)
    inverse_grid = inverse.unsqueeze(0).broadcast_to(sigma_values.shape[0], inverse.shape[0])
    expanded_sigma = module.take_along_dim(sigma_values, inverse_grid, dim=1)
    return unique_rows, inverse, counts, consecutive_rows, consecutive_inverse, consecutive_counts, expanded_sigma


def _case_stable_diffusion_scheduler_cumextrema_setup(module: Any) -> tuple[Any, Any]:
    raw_sigmas = module.tensor(
        [16.0 - float(index) / 8.0 + (0.75 if index % 17 == 0 else 0.0) for index in range(128)],
        dtype=module.float32,
    )
    latents = module.arange(0, 2 * 4 * 4 * 4, dtype=module.float32).reshape(2, 4, 4, 4) / 128.0 - 0.5
    return raw_sigmas, latents


def _stable_diffusion_scheduler_cumextrema_path(module: Any, raw_sigmas: Any, latents: Any) -> Any:
    descending_sigmas, descending_indices = module.cummin(raw_sigmas, dim=0)
    reverse_ceiling, reverse_indices = module.cummax(module.flip(raw_sigmas, (0,)), dim=0)
    selected = module.take_along_dim(descending_sigmas, module.arange(0, raw_sigmas.shape[0], dtype=module.long), dim=0)
    scaled = latents / (selected.reshape(latents.shape) + 1.0)
    return scaled, descending_indices, reverse_indices, module.flip(reverse_ceiling, (0,))


def _case_stable_diffusion_timestep_bincount_setup(module: Any) -> tuple[Any, Any, Any]:
    timesteps = _tensor(module, [999 - ((index * 7) % 64) for index in range(1024)], "int64")
    slots = _tensor(module, [(index * 7) % 64 for index in range(1024)], "int64")
    weights = _tensor(module, [float((index % 29) + 1) / 29.0 for index in range(1024)], "float32")
    return timesteps, slots, weights


def _stable_diffusion_timestep_bincount_path(module: Any, timesteps: Any, slots: Any, weights: Any) -> Any:
    return module.bincount(timesteps, minlength=1000), module.bincount(slots, weights=weights, minlength=64)


def _case_stable_diffusion_scheduler_sigma_stack_setup(module: Any) -> tuple[Any]:
    return (module.logspace(0.0, -4.0, 1025, dtype=module.float32),)


def _stable_diffusion_scheduler_sigma_stack_path(module: Any, sigmas: Any) -> Any:
    return module.stack([sigmas[:-1], sigmas[1:]], dim=-1)


def _case_stable_diffusion_ddim_scheduler_step_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    betas = module.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 16, dtype=module.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = module.cumprod(alphas, dim=0)
    timesteps = module.linspace(0, 15, 4, dtype=module.float32).round().to(dtype=module.long).flip(0)
    sample = module.arange(0, 2 * 4 * 8 * 8, dtype=module.float32).reshape(2, 4, 8, 8) / 128.0 - 1.0
    model_output = module.flip(sample, (-1,)) * 0.75 + 0.1
    return alphas_cumprod, timesteps, sample, model_output


def _stable_diffusion_ddim_scheduler_step_path(
    module: Any,
    alphas_cumprod: Any,
    timesteps: Any,
    sample: Any,
    model_output: Any,
) -> Any:
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


def _case_stable_diffusion_karras_euler_scheduler_step_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    ramp = module.linspace(0.0, 1.0, 32, dtype=module.float32)
    min_inv = 0.0292 ** (1.0 / 7.0)
    max_inv = 14.6146 ** (1.0 / 7.0)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** 7.0
    sigmas = module.cat([sigmas, module.zeros((1,), dtype=module.float32)], dim=0)
    sample = module.arange(0, 2 * 4 * 8 * 8, dtype=module.float32).reshape(2, 4, 8, 8) / 128.0 - 1.0
    model_output = module.flip(sample, (-1,)) * 0.75 + 0.1
    index = module.tensor(3, dtype=module.long)
    return sigmas, sample, model_output, index


def _stable_diffusion_karras_euler_scheduler_step_path(
    module: Any,
    sigmas: Any,
    sample: Any,
    model_output: Any,
    index: Any,
) -> Any:
    sigma = sigmas[index].reshape(1, 1, 1, 1)
    scaled_sample = sample / ((sigma * sigma + 1.0) ** 0.5)
    denoised = scaled_sample - model_output * sigma
    derivative = (sample - denoised) / sigma
    dt = sigmas[index + 1] - sigmas[index]
    return sample + derivative * dt.reshape(1, 1, 1, 1)


def _case_stable_diffusion_euler_ancestral_scheduler_step_setup(module: Any) -> tuple[Any, Any, Any, Any, Any]:
    ramp = module.linspace(0.0, 1.0, 32, dtype=module.float32)
    min_inv = 0.0292 ** (1.0 / 7.0)
    max_inv = 14.6146 ** (1.0 / 7.0)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** 7.0
    sigmas = module.cat([sigmas, module.zeros((1,), dtype=module.float32)], dim=0)
    sample = module.arange(0, 2 * 4 * 8 * 8, dtype=module.float32).reshape(2, 4, 8, 8) / 128.0 - 1.0
    denoised = module.flip(sample, (-1,)) * 0.75 + 0.125
    noise = module.sin(sample) * 0.5
    index = module.tensor(4, dtype=module.long)
    return sigmas, sample, denoised, noise, index


def _stable_diffusion_euler_ancestral_scheduler_step_path(
    module: Any,
    sigmas: Any,
    sample: Any,
    denoised: Any,
    noise: Any,
    index: Any,
) -> Any:
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


def _case_stable_diffusion_half_channels_last_dpmpp_2m_scheduler_step_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    ramp = module.linspace(0.0, 1.0, 32, dtype=module.float32)
    min_inv = 0.0292 ** (1.0 / 7.0)
    max_inv = 14.6146 ** (1.0 / 7.0)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** 7.0
    alphas = 1.0 / module.sqrt(sigmas * sigmas + 1.0)
    sample_base = module.arange(0, 2 * 4 * 64 * 64, dtype=module.float32).reshape(2, 4, 64, 64)
    sample = (module.sin(sample_base * 0.005) * 0.75).to(dtype=module.float16)
    sample = sample.contiguous(memory_format=module.channels_last)
    model_output = (module.flip(sample, (-1,)) * 0.75 + 0.1).contiguous(memory_format=module.channels_last)
    prev_model_output = (module.cos(sample_base * 0.007).to(dtype=module.float16) * 0.5).contiguous(
        memory_format=module.channels_last
    )
    index = module.tensor(4, dtype=module.long)
    return sigmas, alphas, sample, model_output, prev_model_output, index


def _case_stable_diffusion_half_channels_last_dpmpp_sde_scheduler_step_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    ramp = module.linspace(0.0, 1.0, 32, dtype=module.float32)
    min_inv = 0.0292 ** (1.0 / 7.0)
    max_inv = 14.6146 ** (1.0 / 7.0)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** 7.0
    alphas = 1.0 / module.sqrt(sigmas * sigmas + 1.0)
    sample_base = module.arange(0, 2 * 4 * 64 * 64, dtype=module.float32).reshape(2, 4, 64, 64)
    sample = (module.sin(sample_base * 0.005) * 0.75).to(dtype=module.float16)
    sample = sample.contiguous(memory_format=module.channels_last)
    model_output = (module.flip(sample, (-1,)) * 0.75 + 0.1).contiguous(memory_format=module.channels_last)
    prev_model_output = (module.cos(sample_base * 0.007).to(dtype=module.float16) * 0.5).contiguous(
        memory_format=module.channels_last
    )
    noise = (module.sin(sample_base * 0.011 + 0.25).to(dtype=module.float16) * 0.35).contiguous(
        memory_format=module.channels_last
    )
    index = module.tensor(4, dtype=module.long)
    return sigmas, alphas, sample, model_output, prev_model_output, noise, index


def _stable_diffusion_half_channels_last_dpmpp_2m_scheduler_step_path(
    module: Any,
    sigmas: Any,
    alphas: Any,
    sample: Any,
    model_output: Any,
    prev_model_output: Any,
    index: Any,
) -> Any:
    sigma_t = sigmas[index + 1]
    sigma_s0 = sigmas[index]
    sigma_s1 = sigmas[index - 1]
    alpha_t = alphas[index + 1]
    alpha_s0 = alphas[index]
    alpha_s1 = alphas[index - 1]
    lambda_t = module.log(alpha_t) - module.log(sigma_t)
    lambda_s0 = module.log(alpha_s0) - module.log(sigma_s0)
    lambda_s1 = module.log(alpha_s1) - module.log(sigma_s1)
    h = lambda_t - lambda_s0
    h_0 = lambda_s0 - lambda_s1
    r0 = h_0 / h
    d0 = model_output
    d1 = (model_output - prev_model_output) / r0.reshape(1, 1, 1, 1)
    sigma_scale = (sigma_t / sigma_s0).reshape(1, 1, 1, 1)
    alpha_coeff = (alpha_t * (module.exp(-h) - 1.0)).reshape(1, 1, 1, 1)
    return sample * sigma_scale - d0 * alpha_coeff - d1 * (0.5 * alpha_coeff)


def _stable_diffusion_half_channels_last_dpmpp_sde_scheduler_step_path(
    module: Any,
    sigmas: Any,
    alphas: Any,
    sample: Any,
    model_output: Any,
    prev_model_output: Any,
    noise: Any,
    index: Any,
) -> Any:
    sigma_t = sigmas[index + 1]
    sigma_s0 = sigmas[index]
    sigma_s1 = sigmas[index - 1]
    alpha_t = alphas[index + 1]
    alpha_s0 = alphas[index]
    alpha_s1 = alphas[index - 1]
    lambda_t = module.log(alpha_t) - module.log(sigma_t)
    lambda_s0 = module.log(alpha_s0) - module.log(sigma_s0)
    lambda_s1 = module.log(alpha_s1) - module.log(sigma_s1)
    h = lambda_t - lambda_s0
    h_0 = lambda_s0 - lambda_s1
    r0 = h_0 / h
    d0 = model_output
    d1 = (model_output - prev_model_output) / r0.reshape(1, 1, 1, 1)
    sigma_scale = (sigma_t / sigma_s0).reshape(1, 1, 1, 1)
    alpha_coeff = (alpha_t * (module.exp(-h) - 1.0)).reshape(1, 1, 1, 1)
    noise_coeff = (sigma_t * module.sqrt(1.0 - module.exp(-2.0 * h))).reshape(1, 1, 1, 1)
    return sample * sigma_scale - d0 * alpha_coeff - d1 * (0.5 * alpha_coeff) + noise * noise_coeff


def _case_stable_diffusion_vae_setup(module: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    sample = module.arange(0, 1 * 3 * 8 * 8, dtype=module.float32).reshape(1, 3, 8, 8) / 64.0 - 1.0
    enc_weight = module.arange(0, 6 * 3 * 3 * 3, dtype=module.float32).reshape(6, 3, 3, 3) / 512.0
    down_weight = module.arange(0, 8 * 6 * 3 * 3, dtype=module.float32).reshape(8, 6, 3, 3) / 1024.0
    quant_weight = module.arange(0, 8 * 8 * 1 * 1, dtype=module.float32).reshape(8, 8, 1, 1) / 2048.0
    dec_weight = module.arange(0, 4 * 4 * 3 * 3, dtype=module.float32).reshape(4, 4, 3, 3) / 1024.0
    out_weight = module.arange(0, 3 * 4 * 3 * 3, dtype=module.float32).reshape(3, 4, 3, 3) / 512.0
    return sample, enc_weight, down_weight, quant_weight, dec_weight, out_weight


def _stable_diffusion_vae_path(
    module: Any,
    sample: Any,
    enc_weight: Any,
    down_weight: Any,
    quant_weight: Any,
    dec_weight: Any,
    out_weight: Any,
) -> Any:
    hidden = module.nn.functional.conv2d(sample, enc_weight, None, padding=1)
    hidden = module.nn.functional.group_norm(hidden, 3, eps=1e-5)
    hidden = module.nn.functional.silu(hidden)
    hidden = module.nn.functional.pad(hidden, (0, 1, 0, 1), mode="constant", value=0.0)
    hidden = module.nn.functional.conv2d(hidden, down_weight, None, stride=2)
    moments = module.nn.functional.conv2d(hidden, quant_weight, None)
    mean, logvar = moments.chunk(2, dim=1)
    logvar = module.clamp(logvar, min=-30.0, max=20.0)
    std = module.exp(0.5 * logvar)
    latent = mean + std * 0.0
    decoded = module.nn.functional.conv_transpose2d(latent, dec_weight, None, stride=2, padding=1, output_padding=1)
    decoded = module.nn.functional.group_norm(decoded, 2, eps=1e-5)
    decoded = module.nn.functional.silu(decoded)
    return module.nn.functional.conv2d(decoded, out_weight, None, padding=1).clamp(-1.0, 1.0)


def _case_stable_diffusion_half_channels_last_vae_posterior_sample_setup(module: Any) -> tuple[Any, Any]:
    spatial = 64
    moments_base = module.arange(0, 1 * 8 * spatial * spatial, dtype=module.float32).reshape(
        1,
        8,
        spatial,
        spatial,
    )
    moments = (module.sin(moments_base * 0.017) * 2.0).to(dtype=module.float16)
    moments = moments.contiguous(memory_format=module.channels_last)
    noise_base = module.arange(0, 1 * 4 * spatial * spatial, dtype=module.float32).reshape(
        1,
        4,
        spatial,
        spatial,
    )
    noise = (module.cos(noise_base * 0.013) * 0.75).to(dtype=module.float16)
    noise = noise.contiguous(memory_format=module.channels_last)
    return moments, noise


def _stable_diffusion_half_channels_last_vae_posterior_sample_path(module: Any, moments: Any, noise: Any) -> Any:
    mean, logvar = moments.chunk(2, dim=1)
    logvar = module.clamp(logvar, min=-30.0, max=20.0)
    std = module.exp(0.5 * logvar)
    return (mean + std * noise) * 0.18215


def _case_stable_diffusion_half_vae_encode_down_block_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
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
    return sample, enc_weight, res1_weight, res2_weight, down_weight, quant_weight, norm_weight, norm_bias


def _stable_diffusion_half_vae_encode_down_block_path(
    module: Any,
    sample: Any,
    enc_weight: Any,
    res1_weight: Any,
    res2_weight: Any,
    down_weight: Any,
    quant_weight: Any,
    norm_weight: Any,
    norm_bias: Any,
) -> Any:
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


def _case_stable_diffusion_half_vae_attention_block_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
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
    return hidden, norm_weight, norm_bias, q_weight, k_weight, v_weight, proj_weight, bias


def _stable_diffusion_half_vae_attention_block_path(
    module: Any,
    hidden: Any,
    norm_weight: Any,
    norm_bias: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    proj_weight: Any,
    bias: Any,
) -> Any:
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


def _case_stable_diffusion_half_video_conv3d_block_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any]:
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
    return hidden, weight1, weight2, norm_weight, norm_bias


def _case_stable_diffusion_half_channels_last_3d_video_conv3d_block_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    hidden, weight1, weight2, norm_weight, norm_bias = _case_stable_diffusion_half_video_conv3d_block_setup(module)
    return hidden.contiguous(memory_format=module.channels_last_3d), weight1, weight2, norm_weight, norm_bias


def _case_stable_diffusion_half_channels_last_3d_skip_cat_conv3d_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any]:
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
    weight_base = module.arange(0, channels * (channels * 2) * 3 * 3 * 3, dtype=module.float32).reshape(
        channels,
        channels * 2,
        3,
        3,
        3,
    )
    weight = (module.cos(weight_base * 0.011) * 0.04).to(dtype=module.float16)
    bias = (module.arange(0, channels, dtype=module.float32) / 256.0 - 0.03).to(dtype=module.float16)
    return hidden, skip, weight, bias


def _stable_diffusion_half_video_conv3d_block_path(
    module: Any,
    hidden: Any,
    weight1: Any,
    weight2: Any,
    norm_weight: Any,
    norm_bias: Any,
) -> Any:
    residual = hidden
    hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
    hidden = module.nn.functional.silu(hidden)
    hidden = module.nn.functional.conv3d(hidden, weight1, None, padding=1)
    hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
    hidden = module.nn.functional.silu(hidden)
    hidden = module.nn.functional.conv3d(hidden, weight2, None, padding=1)
    return hidden + residual


def _stable_diffusion_half_channels_last_3d_skip_cat_conv3d_path(
    module: Any,
    hidden: Any,
    skip: Any,
    weight: Any,
    bias: Any,
) -> Any:
    merged = module.cat([hidden, skip], dim=1)
    return module.nn.functional.conv3d(merged, weight, bias, padding=1)


def _case_stable_diffusion_half_vae_nearest_upsample_setup(module: Any) -> tuple[Any]:
    hidden_base = module.arange(0, 1 * 8 * 16 * 16, dtype=module.float32).reshape(1, 8, 16, 16)
    hidden = (module.sin(hidden_base * 0.017) * 0.5).to(dtype=module.float16)
    return (hidden,)


def _stable_diffusion_half_vae_nearest_upsample_path(module: Any, hidden: Any) -> Any:
    return module.nn.functional.interpolate(hidden, scale_factor=2.0, mode="nearest")


def _case_stable_diffusion_half_vae_decode_shape_setup(module: Any, spatial: int) -> tuple[Any, Any, Any, Any, Any]:
    latent = module.arange(0, 1 * 4 * spatial * spatial, dtype=module.float16).reshape(1, 4, spatial, spatial) / 64.0 - 0.5
    up_weight = module.arange(0, 4 * 8 * 3 * 3, dtype=module.float16).reshape(4, 8, 3, 3) / 512.0
    norm_weight = module.tensor([0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45], dtype=module.float16)
    norm_bias = module.tensor([-0.2, -0.1, 0.0, 0.1, 0.2, -0.15, 0.05, 0.15], dtype=module.float16)
    out_weight = module.arange(0, 3 * 8 * 3 * 3, dtype=module.float16).reshape(3, 8, 3, 3) / 512.0
    return latent, up_weight, norm_weight, norm_bias, out_weight


def _case_stable_diffusion_half_vae_decode_setup(module: Any) -> tuple[Any, Any, Any, Any, Any]:
    return _case_stable_diffusion_half_vae_decode_shape_setup(module, 2)


def _case_stable_diffusion_half_vae_decode_8x8_setup(module: Any) -> tuple[Any, Any, Any, Any, Any]:
    return _case_stable_diffusion_half_vae_decode_shape_setup(module, 8)


def _stable_diffusion_half_vae_decode_path(
    module: Any,
    latent: Any,
    up_weight: Any,
    norm_weight: Any,
    norm_bias: Any,
    out_weight: Any,
) -> Any:
    hidden = module.nn.functional.conv_transpose2d(latent, up_weight, None, stride=2, padding=1, output_padding=1)
    hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
    hidden = module.nn.functional.silu(hidden)
    return module.nn.functional.conv2d(hidden, out_weight, None, padding=1).clamp(-1.0, 1.0)


def _case_stable_diffusion_half_channels_last_vae_transpose_decode_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    latent = module.arange(0, 1 * 4 * 8 * 8, dtype=module.float32).reshape(1, 4, 8, 8) / 64.0 - 0.5
    latent = latent.to(dtype=module.float16).contiguous(memory_format=module.channels_last)
    up_weight = module.arange(0, 4 * 8 * 3 * 3, dtype=module.float32).reshape(4, 8, 3, 3) / 512.0
    norm_weight = module.tensor([0.75 + index * 0.05 for index in range(8)], dtype=module.float16)
    norm_bias = module.tensor([-0.1 + index * 0.025 for index in range(8)], dtype=module.float16)
    out_weight = module.arange(0, 3 * 8 * 3 * 3, dtype=module.float32).reshape(3, 8, 3, 3) / 512.0
    return (
        latent,
        up_weight.to(dtype=module.float16),
        norm_weight,
        norm_bias,
        out_weight.to(dtype=module.float16),
    )


def _stable_diffusion_half_channels_last_vae_transpose_decode_path(
    module: Any,
    latent: Any,
    up_weight: Any,
    norm_weight: Any,
    norm_bias: Any,
    out_weight: Any,
) -> Any:
    hidden = module.nn.functional.conv_transpose2d(latent, up_weight, None, stride=2, padding=1, output_padding=1)
    hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
    hidden = module.nn.functional.silu(hidden)
    return module.nn.functional.conv2d(hidden, out_weight, None, padding=1).clamp(-1.0, 1.0)


def _case_stable_diffusion_half_vae_nearest_block_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    latent = module.arange(0, 1 * 4 * 8 * 8, dtype=module.float32).reshape(1, 4, 8, 8) / 64.0 - 0.5
    latent = latent.to(dtype=module.float16)
    in_weight = module.arange(0, 8 * 4 * 3 * 3, dtype=module.float32).reshape(8, 4, 3, 3) / 512.0
    res1_weight = module.arange(0, 8 * 8 * 3 * 3, dtype=module.float32).reshape(8, 8, 3, 3) / 768.0
    res2_weight = module.arange(0, 8 * 8 * 3 * 3, dtype=module.float32).reshape(8, 8, 3, 3) / 896.0
    up_weight = module.arange(0, 8 * 8 * 3 * 3, dtype=module.float32).reshape(8, 8, 3, 3) / 1024.0
    out_weight = module.arange(0, 3 * 8 * 3 * 3, dtype=module.float32).reshape(3, 8, 3, 3) / 512.0
    norm_weight = module.tensor([0.75 + index * 0.05 for index in range(8)], dtype=module.float16)
    norm_bias = module.tensor([-0.1 + index * 0.025 for index in range(8)], dtype=module.float16)
    return (
        latent,
        in_weight.to(dtype=module.float16),
        res1_weight.to(dtype=module.float16),
        res2_weight.to(dtype=module.float16),
        up_weight.to(dtype=module.float16),
        out_weight.to(dtype=module.float16),
        norm_weight,
        norm_bias,
    )


def _stable_diffusion_half_vae_nearest_block_path(
    module: Any,
    latent: Any,
    in_weight: Any,
    res1_weight: Any,
    res2_weight: Any,
    up_weight: Any,
    out_weight: Any,
    norm_weight: Any,
    norm_bias: Any,
) -> Any:
    hidden = module.nn.functional.conv2d(latent, in_weight, None, padding=1)
    residual = hidden
    hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
    hidden = module.nn.functional.silu(hidden)
    hidden = module.nn.functional.conv2d(hidden, res1_weight, None, padding=1)
    hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
    hidden = module.nn.functional.silu(hidden)
    hidden = module.nn.functional.conv2d(hidden, res2_weight, None, padding=1)
    hidden = hidden + residual
    hidden = module.nn.functional.interpolate(hidden, scale_factor=2.0, mode="nearest")
    hidden = module.nn.functional.conv2d(hidden, up_weight, None, padding=1)
    hidden = module.nn.functional.group_norm(hidden, 2, norm_weight, norm_bias, eps=1e-5)
    hidden = module.nn.functional.silu(hidden)
    return module.nn.functional.conv2d(hidden, out_weight, None, padding=1).clamp(-1.0, 1.0)


def _case_stable_diffusion_half_channels_last_vae_decode_block_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    channels, spatial = 16, 16
    hidden_base = module.arange(0, 1 * channels * spatial * spatial, dtype=module.float32).reshape(
        1,
        channels,
        spatial,
        spatial,
    )
    hidden = (module.sin(hidden_base * 0.011) * 0.25).to(dtype=module.float16)
    hidden = hidden.contiguous(memory_format=module.channels_last)

    def make_weight(scale: float) -> Any:
        base = module.arange(0, channels * channels * 3 * 3, dtype=module.float32).reshape(channels, channels, 3, 3)
        return (module.sin(base * scale) * 0.035).to(dtype=module.float16)

    out_base = module.arange(0, 3 * channels * 3 * 3, dtype=module.float32).reshape(3, channels, 3, 3)
    out_weight = (module.cos(out_base * 0.029) * 0.03).to(dtype=module.float16)
    norm_weight = (0.75 + (module.arange(0, channels, dtype=module.float32) % 257) / 512.0).to(dtype=module.float16)
    norm_bias = (((module.arange(0, channels, dtype=module.float32) % 127) - 63.0) / 2048.0).to(dtype=module.float16)
    return (
        hidden,
        make_weight(0.013),
        make_weight(0.017),
        make_weight(0.019),
        make_weight(0.023),
        out_weight,
        norm_weight,
        norm_bias,
    )


def _stable_diffusion_half_channels_last_vae_decode_block_path(
    module: Any,
    hidden: Any,
    res1_weight: Any,
    res2_weight: Any,
    up_weight: Any,
    res3_weight: Any,
    out_weight: Any,
    norm_weight: Any,
    norm_bias: Any,
) -> Any:
    residual = hidden
    hidden = module.nn.functional.group_norm(hidden, 4, norm_weight, norm_bias, eps=1e-5)
    hidden = module.nn.functional.silu(hidden)
    hidden = module.nn.functional.conv2d(hidden, res1_weight, None, padding=1)
    hidden = module.nn.functional.group_norm(hidden, 4, norm_weight, norm_bias, eps=1e-5)
    hidden = module.nn.functional.silu(hidden)
    hidden = module.nn.functional.conv2d(hidden, res2_weight, None, padding=1)
    hidden = hidden + residual
    hidden = module.nn.functional.interpolate(hidden, scale_factor=2.0, mode="nearest")
    hidden = module.nn.functional.conv2d(hidden, up_weight, None, padding=1)
    residual = hidden
    hidden = module.nn.functional.group_norm(hidden, 4, norm_weight, norm_bias, eps=1e-5)
    hidden = module.nn.functional.silu(hidden)
    hidden = module.nn.functional.conv2d(hidden, res3_weight, None, padding=1)
    hidden = hidden + residual
    return module.nn.functional.conv2d(hidden, out_weight, None, padding=1).clamp(-1.0, 1.0)


def _case_stable_diffusion_bicubic_postprocess_setup(module: Any) -> tuple[Any]:
    decoded = module.arange(0, 1 * 3 * 16 * 16, dtype=module.float32).reshape(1, 3, 16, 16) / 128.0 - 1.0
    return (decoded,)


def _stable_diffusion_bicubic_postprocess_path(module: Any, decoded: Any) -> Any:
    resized = module.nn.functional.interpolate(decoded, scale_factor=2.0, mode="bicubic", align_corners=False)
    image = (resized / 2.0 + 0.5).clamp(0.0, 1.0)
    return image.permute(0, 2, 3, 1).contiguous()


def _case_stable_diffusion_area_preprocess_setup(module: Any) -> tuple[Any]:
    image = module.arange(0, 1 * 3 * 16 * 16, dtype=module.float32).reshape(1, 3, 16, 16) / 255.0
    return (image,)


def _stable_diffusion_area_preprocess_path(module: Any, image: Any) -> Any:
    return module.nn.functional.interpolate(image, size=(8, 8), mode="area") * 2.0 - 1.0


def _case_stable_diffusion_constant_pad_crop_setup(module: Any) -> tuple[Any, Any]:
    latent = module.arange(0, 1 * 4 * 16 * 16, dtype=module.float32).reshape(1, 4, 16, 16) / 64.0 - 1.0
    weight = module.arange(0, 4 * 4 * 3 * 3, dtype=module.float32).reshape(4, 4, 3, 3) / 512.0
    return latent, weight


def _stable_diffusion_constant_pad_crop_path(module: Any, latent: Any, weight: Any) -> Any:
    cropped = module.nn.functional.pad(latent, (-1, 2, -2, 1), mode="constant", value=0.25)
    return module.nn.functional.conv2d(cropped, weight, None, padding=1)


def _case_stable_diffusion_mask_preprocess_setup(module: Any) -> tuple[Any]:
    mask = module.arange(0, 1 * 1 * 32 * 32, dtype=module.float32).reshape(1, 1, 32, 32) % 2.0
    return (mask,)


def _stable_diffusion_mask_nearest_exact_path(module: Any, mask: Any) -> Any:
    return module.nn.functional.interpolate(mask, size=(16, 16), mode="nearest-exact") > 0.5


def _case_stable_diffusion_inpaint_preprocess_bundle_setup(module: Any) -> tuple[Any, Any, Any, Any, Any]:
    image = module.arange(0, 1 * 3 * 32 * 32, dtype=module.float32).reshape(1, 3, 32, 32) / 255.0
    mask = (module.arange(0, 1 * 1 * 32 * 32, dtype=module.float32).reshape(1, 1, 32, 32) % 5.0) / 4.0
    latent_base = module.arange(0, 1 * 4 * 16 * 16, dtype=module.float32).reshape(1, 4, 16, 16)
    latents = (module.sin(latent_base * 0.017) * 0.55).to(dtype=module.float16)
    noise = (module.cos(latent_base * 0.013) * 0.35).to(dtype=module.float16)
    latents = latents.contiguous(memory_format=module.channels_last)
    noise = noise.contiguous(memory_format=module.channels_last)
    weight_base = module.arange(0, 4 * 3 * 3 * 3, dtype=module.float32).reshape(4, 3, 3, 3)
    encoder_weight = (module.cos(weight_base * 0.011) * 0.04).to(dtype=module.float16)
    return image, mask, latents, noise, encoder_weight


def _stable_diffusion_inpaint_preprocess_bundle_path(
    module: Any,
    image: Any,
    mask: Any,
    latents: Any,
    noise: Any,
    encoder_weight: Any,
) -> Any:
    image = module.nn.functional.interpolate(image, size=(16, 16), mode="area") * 2.0 - 1.0
    image = image.to(dtype=latents.dtype).contiguous(memory_format=module.channels_last)
    binary_mask = module.nn.functional.interpolate(mask, size=(16, 16), mode="nearest-exact") > 0.5
    binary_mask = binary_mask.to(dtype=latents.dtype).contiguous(memory_format=module.channels_last)
    masked_image = image * (1.0 - binary_mask)
    masked_image_latents = module.nn.functional.conv2d(masked_image, encoder_weight, None, padding=1)
    noised_latents = latents + noise * 0.15
    latent_model_input = noised_latents * (1.0 - binary_mask) + noise * binary_mask
    return module.cat([latent_model_input, binary_mask, masked_image_latents], dim=1).contiguous(
        memory_format=module.channels_last
    )


def _case_embedding_setup(module: Any) -> tuple[Any, Any]:
    weight = _tensor(module, _matrix(32, 16))
    indices = _tensor(module, [(row * 17) % 32 for row in range(32)], "int64")
    return indices, weight


def _case_embedding_max_norm_setup(module: Any) -> tuple[Any, Any]:
    weight = _tensor(module, [[value * 8.0 for value in row] for row in _matrix(32, 16)])
    indices = _tensor(module, [(row * 17) % 32 for row in range(32)], "int64")
    return indices, weight


def _copy_parameter(module: Any, target: Any, source: Any) -> None:
    if hasattr(module, "no_grad"):
        with module.no_grad():
            target.copy_(source)
        return
    target.copy_(source)


def _fill_stable_diffusion_parameter(module: Any, parameter: Any) -> None:
    if parameter.dim() >= 2:
        values = _tensor(module, _matrix(parameter.shape[0], parameter.numel() // parameter.shape[0])).reshape(parameter.shape)
        _copy_parameter(module, parameter, values / 4.0)
    else:
        _copy_parameter(module, parameter, _tensor(module, _vector(parameter.numel())))


def _make_stable_diffusion_text_encoder(module: Any) -> Any:
    class MiniTextEncoder(module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token = module.nn.Embedding(16, 8)
            self.position = module.nn.Embedding(6, 8)
            self.norm = module.nn.LayerNorm(8)
            self.proj = module.nn.Linear(8, 8)
            _copy_parameter(module, self.token.weight, _tensor(module, _matrix(16, 8)))
            _copy_parameter(module, self.position.weight, _tensor(module, _matrix(6, 8)))
            _copy_parameter(module, self.norm.weight, _tensor(module, [[1.0 + index / 16.0 for index in range(8)]])[0])
            _copy_parameter(module, self.norm.bias, _tensor(module, _vector(8)))
            _copy_parameter(module, self.proj.weight, _tensor(module, _matrix(8, 8)))
            _copy_parameter(module, self.proj.bias, _tensor(module, _vector(8)))

        def forward(self, input_ids: Any) -> Any:
            positions = module.arange(input_ids.shape[1], dtype=module.long).unsqueeze(0).expand(input_ids.shape[0], -1)
            hidden = self.token(input_ids) + self.position(positions)
            return module.nn.functional.gelu(self.proj(self.norm(hidden)))

    return MiniTextEncoder().eval()


def _make_stable_diffusion_unet(module: Any, in_channels: int = 4) -> Any:
    class MiniUNet(module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_conv = module.nn.Conv2d(in_channels, 8, 3, padding=1)
            self.norm = module.nn.GroupNorm(2, 8)
            self.time = module.nn.Linear(4, 8)
            self.q = module.nn.Linear(8, 8)
            self.k = module.nn.Linear(8, 8)
            self.v = module.nn.Linear(8, 8)
            self.out = module.nn.Linear(8, 8)
            self.out_norm = module.nn.GroupNorm(2, 8)
            self.out_conv = module.nn.Conv2d(8, 4, 3, padding=1)
            for _, parameter in self.named_parameters():
                _fill_stable_diffusion_parameter(module, parameter)

        def forward(self, sample: Any, timestep: Any, encoder_hidden: Any) -> Any:
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


def _make_stable_diffusion_sdxl_text_encoder(module: Any) -> Any:
    class MiniSDXLTextEncoder(module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_1 = module.nn.Embedding(32, 8)
            self.position_1 = module.nn.Embedding(6, 8)
            self.norm_1 = module.nn.LayerNorm(8)
            self.proj_1 = module.nn.Linear(8, 8)
            self.token_2 = module.nn.Embedding(32, 8)
            self.position_2 = module.nn.Embedding(6, 8)
            self.norm_2 = module.nn.LayerNorm(8)
            self.proj_2 = module.nn.Linear(8, 8)
            self.pooled_projection = module.nn.Linear(8, 8)
            for _, parameter in self.named_parameters():
                _fill_stable_diffusion_parameter(module, parameter)

        def forward(self, input_ids_1: Any, input_ids_2: Any) -> Any:
            positions = module.arange(input_ids_1.shape[1], dtype=module.long).unsqueeze(0).expand(
                input_ids_1.shape[0],
                -1,
            )
            hidden_1 = self.token_1(input_ids_1) + self.position_1(positions)
            hidden_1 = module.nn.functional.gelu(self.proj_1(self.norm_1(hidden_1)))
            hidden_2 = self.token_2(input_ids_2) + self.position_2(positions)
            hidden_2 = module.nn.functional.gelu(self.proj_2(self.norm_2(hidden_2)))
            pooled = hidden_2[module.arange(input_ids_2.shape[0], dtype=module.long), input_ids_2.argmax(dim=-1)]
            pooled = self.pooled_projection(pooled)
            return module.cat([hidden_1, hidden_2], dim=-1), pooled

    return MiniSDXLTextEncoder().eval()


def _make_stable_diffusion_sdxl_add_embedding(module: Any) -> Any:
    class MiniSDXLAddEmbedding(module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = module.nn.Linear(14, 8)
            self.fc2 = module.nn.Linear(8, 8)
            for _, parameter in self.named_parameters():
                _fill_stable_diffusion_parameter(module, parameter)

        def forward(self, pooled_prompt_embeds: Any, add_time_ids: Any) -> Any:
            time_features = add_time_ids.to(dtype=pooled_prompt_embeds.dtype) / 1024.0
            hidden = module.cat([pooled_prompt_embeds, time_features], dim=-1)
            hidden = module.nn.functional.silu(self.fc1(hidden))
            return self.fc2(hidden)

    return MiniSDXLAddEmbedding().eval()


def _make_stable_diffusion_sdxl_controlnet(module: Any) -> Any:
    class MiniSDXLControlNet(module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_conv = module.nn.Conv2d(5, 8, 3, padding=1)
            self.norm = module.nn.GroupNorm(2, 8)
            self.time = module.nn.Linear(4, 8)
            self.add = module.nn.Linear(8, 8)
            self.down_residual = module.nn.Conv2d(8, 8, 1)
            self.mid_residual = module.nn.Conv2d(8, 8, 3, padding=1)
            for _, parameter in self.named_parameters():
                _fill_stable_diffusion_parameter(module, parameter)

        def forward(self, control_image_latents: Any, mask: Any, timestep: Any, added_cond: Any) -> Any:
            if control_image_latents.shape[0] != mask.shape[0]:
                raise ValueError("SDXL ControlNet conditioning batch sizes must match")
            if mask.shape[1] != 1:
                raise ValueError("SDXL ControlNet inpaint mask must have one channel")
            conditioning = module.cat([control_image_latents, mask.to(dtype=control_image_latents.dtype)], dim=1)
            conditioning = conditioning.contiguous(memory_format=module.channels_last)
            hidden = module.nn.functional.silu(self.norm(self.in_conv(conditioning)))
            timestep_value = timestep.to(dtype=hidden.dtype).reshape(1, 1)
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
            hidden = hidden + self.add(added_cond).reshape(added_cond.shape[0], 8, 1, 1)
            down_residual = self.down_residual(hidden).contiguous(memory_format=module.channels_last)
            mid_residual = self.mid_residual(module.nn.functional.silu(hidden)).contiguous(
                memory_format=module.channels_last
            )
            return down_residual * 0.5, mid_residual * 0.25

    return MiniSDXLControlNet().eval()


def _make_stable_diffusion_sdxl_inpaint_unet(module: Any) -> Any:
    class MiniSDXLInpaintUNet(module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_conv = module.nn.Conv2d(9, 8, 3, padding=1)
            self.norm = module.nn.GroupNorm(2, 8)
            self.time = module.nn.Linear(4, 8)
            self.add = module.nn.Linear(8, 8)
            self.q = module.nn.Linear(8, 8)
            self.k = module.nn.Linear(16, 8)
            self.v = module.nn.Linear(16, 8)
            self.out = module.nn.Linear(8, 8)
            self.out_norm = module.nn.GroupNorm(2, 8)
            self.out_conv = module.nn.Conv2d(8, 4, 3, padding=1)
            for _, parameter in self.named_parameters():
                _fill_stable_diffusion_parameter(module, parameter)

        def forward(
            self,
            sample: Any,
            timestep: Any,
            encoder_hidden: Any,
            added_cond: Any,
            control_residuals: Any | None = None,
        ) -> Any:
            if control_residuals is None:
                down_residual = None
                mid_residual = None
            elif len(control_residuals) == 2:
                down_residual, mid_residual = control_residuals
            else:
                raise ValueError("SDXL inpaint UNet expects two ControlNet residual tensors")

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
            hidden = hidden + self.add(added_cond).reshape(sample.shape[0], 8, 1, 1)
            if down_residual is not None:
                hidden = hidden + down_residual
            batch, channels, height, width = hidden.shape
            tokens = hidden.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            query = self.q(tokens).reshape(batch, height * width, 2, 4).transpose(1, 2)
            key = self.k(encoder_hidden).reshape(batch, encoder_hidden.shape[1], 2, 4).transpose(1, 2)
            value = self.v(encoder_hidden).reshape(batch, encoder_hidden.shape[1], 2, 4).transpose(1, 2)
            attended = module.nn.functional.scaled_dot_product_attention(query, key, value, scale=0.5)
            attended = attended.transpose(1, 2).reshape(batch, height * width, channels)
            tokens = tokens + self.out(attended)
            hidden = tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
            if mid_residual is not None:
                hidden = hidden + mid_residual
            hidden = module.nn.functional.silu(self.out_norm(hidden))
            return self.out_conv(hidden)

    return MiniSDXLInpaintUNet().eval()


def _make_stable_diffusion_vae(module: Any) -> Any:
    class MiniVAE(module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = module.nn.ConvTranspose2d(4, 8, 3, stride=2, padding=1, output_padding=1)
            self.norm = module.nn.GroupNorm(2, 8)
            self.out = module.nn.Conv2d(8, 3, 3, padding=1)
            for _, parameter in self.named_parameters():
                _fill_stable_diffusion_parameter(module, parameter)

        def forward(self, latents: Any) -> Any:
            hidden = module.nn.functional.silu(self.norm(self.up(latents)))
            return self.out(hidden).clamp(-1.0, 1.0)

    return MiniVAE().eval()


def _make_stable_diffusion_scheduler(module: Any) -> Any:
    class MiniScheduler(module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("timesteps", module.tensor([999, 750, 500], dtype=module.long))
            self.register_buffer("sigmas", module.logspace(-1.0, -2.0, 3, dtype=module.float32))

        def step(self, model_output: Any, index: Any, sample: Any) -> Any:
            sigma = self.sigmas[index].reshape(1, 1, 1, 1)
            next_index = (index + 1).clamp(max=self.sigmas.shape[0] - 1)
            next_sigma = module.where(
                index + 1 < self.sigmas.shape[0],
                self.sigmas[next_index],
                module.zeros((), dtype=sample.dtype),
            )
            return sample + model_output * (next_sigma - sigma.reshape(())).reshape(1, 1, 1, 1)

    return MiniScheduler().eval()


def _case_stable_diffusion_module_pipeline_setup(module: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    text_encoder = _make_stable_diffusion_text_encoder(module)
    unet = _make_stable_diffusion_unet(module)
    vae = _make_stable_diffusion_vae(module)
    scheduler = _make_stable_diffusion_scheduler(module)
    unconditional_ids = module.tensor([[0, 0, 0, 0, 0, 0]], dtype=module.long)
    conditional_ids = module.tensor([[1, 2, 3, 4, 5, 0]], dtype=module.long)
    input_ids = module.cat([unconditional_ids, conditional_ids], dim=0)
    latents = module.arange(0, 1 * 4 * 4 * 4, dtype=module.float32).reshape(1, 4, 4, 4) / 32.0 - 1.0
    return text_encoder, unet, vae, scheduler, input_ids, latents


def _case_stable_diffusion_half_module_pipeline_setup(module: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    text_encoder = _make_stable_diffusion_text_encoder(module).half()
    unet = _make_stable_diffusion_unet(module).half()
    vae = _make_stable_diffusion_vae(module).half()
    scheduler = _make_stable_diffusion_scheduler(module).half()
    _copy_parameter(module, scheduler.timesteps, module.tensor([9, 7, 5], dtype=module.long))
    unconditional_ids = module.tensor([[0, 0, 0, 0, 0, 0]], dtype=module.long)
    conditional_ids = module.tensor([[1, 2, 3, 4, 5, 0]], dtype=module.long)
    input_ids = module.cat([unconditional_ids, conditional_ids], dim=0)
    latents = (module.arange(0, 1 * 4 * 4 * 4, dtype=module.float32).reshape(1, 4, 4, 4) / 32.0 - 1.0).to(
        dtype=module.float16
    )
    return text_encoder, unet, vae, scheduler, input_ids, latents


def _case_stable_diffusion_half_channels_last_module_pipeline_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    text_encoder, unet, vae, scheduler, input_ids, latents = _case_stable_diffusion_half_module_pipeline_setup(module)
    unet.to(memory_format=module.channels_last)
    vae.to(memory_format=module.channels_last)
    latents = latents.contiguous(memory_format=module.channels_last)
    return text_encoder, unet, vae, scheduler, input_ids, latents


def _case_stable_diffusion_half_channels_last_inpaint_module_pipeline_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    text_encoder = _make_stable_diffusion_text_encoder(module).half()
    unet = _make_stable_diffusion_unet(module, in_channels=9).half()
    vae = _make_stable_diffusion_vae(module).half()
    scheduler = _make_stable_diffusion_scheduler(module).half()
    _copy_parameter(module, scheduler.timesteps, module.tensor([9, 7, 5], dtype=module.long))
    unet.to(memory_format=module.channels_last)
    vae.to(memory_format=module.channels_last)

    unconditional_ids = module.tensor([[0, 0, 0, 0, 0, 0]], dtype=module.long)
    conditional_ids = module.tensor([[1, 2, 3, 4, 5, 0]], dtype=module.long)
    input_ids = module.cat([unconditional_ids, conditional_ids], dim=0)

    latent_grid = module.arange(0, 1 * 4 * 4 * 4, dtype=module.float32).reshape(1, 4, 4, 4)
    init_latents = (module.sin(latent_grid * 0.11) * 0.7).to(dtype=module.float16)
    latent_noise = (module.cos(latent_grid * 0.07) * 0.4).to(dtype=module.float16)
    mask = (module.arange(0, 1 * 1 * 4 * 4, dtype=module.float32).reshape(1, 1, 4, 4) % 4.0) / 3.0
    mask = (mask > 0.45).to(dtype=module.float16)
    masked_image_latents = (init_latents * (1.0 - mask) + module.flip(init_latents, (-1,)) * mask * 0.25).to(
        dtype=module.float16
    )
    latents = (init_latents + latent_noise * 0.35).contiguous(memory_format=module.channels_last)
    init_latents = init_latents.contiguous(memory_format=module.channels_last)
    mask = mask.contiguous(memory_format=module.channels_last)
    masked_image_latents = masked_image_latents.contiguous(memory_format=module.channels_last)
    return text_encoder, unet, vae, scheduler, input_ids, latents, init_latents, mask, masked_image_latents


def _case_stable_diffusion_sdxl_half_channels_last_inpaint_controlnet_module_pipeline_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    text_encoder = _make_stable_diffusion_sdxl_text_encoder(module).half()
    add_embedding = _make_stable_diffusion_sdxl_add_embedding(module).half()
    controlnet = _make_stable_diffusion_sdxl_controlnet(module).half()
    unet = _make_stable_diffusion_sdxl_inpaint_unet(module).half()
    vae = _make_stable_diffusion_vae(module).half()
    scheduler = _make_stable_diffusion_scheduler(module).half()
    _copy_parameter(module, scheduler.timesteps, module.tensor([11, 8, 5], dtype=module.long))

    controlnet.to(memory_format=module.channels_last)
    unet.to(memory_format=module.channels_last)
    vae.to(memory_format=module.channels_last)

    unconditional_ids_1 = module.tensor([[0, 0, 0, 0, 0, 0]], dtype=module.long)
    conditional_ids_1 = module.tensor([[1, 2, 3, 4, 5, 0]], dtype=module.long)
    input_ids_1 = module.cat([unconditional_ids_1, conditional_ids_1], dim=0)
    unconditional_ids_2 = module.tensor([[0, 0, 0, 0, 0, 31]], dtype=module.long)
    conditional_ids_2 = module.tensor([[7, 11, 13, 17, 19, 31]], dtype=module.long)
    input_ids_2 = module.cat([unconditional_ids_2, conditional_ids_2], dim=0)
    add_time_ids = module.tensor(
        [
            [1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0],
            [768.0, 1024.0, 16.0, 32.0, 768.0, 1024.0],
        ],
        dtype=module.float32,
    )

    latent_grid = module.arange(0, 1 * 4 * 4 * 4, dtype=module.float32).reshape(1, 4, 4, 4)
    init_latents = (module.sin(latent_grid * 0.09) * 0.65).to(dtype=module.float16)
    latent_noise = (module.cos(latent_grid * 0.13) * 0.35).to(dtype=module.float16)
    mask = (module.arange(0, 1 * 1 * 4 * 4, dtype=module.float32).reshape(1, 1, 4, 4) % 5.0) / 4.0
    mask = (mask > 0.35).to(dtype=module.float16)
    masked_image_latents = (init_latents * (1.0 - mask) + module.flip(init_latents, (-2,)) * mask * 0.3).to(
        dtype=module.float16
    )
    control_image_latents = (module.flip(init_latents, (-1,)) * 0.4 + latent_noise * 0.2).to(dtype=module.float16)
    latents = (init_latents + latent_noise * 0.4).contiguous(memory_format=module.channels_last)
    init_latents = init_latents.contiguous(memory_format=module.channels_last)
    mask = mask.contiguous(memory_format=module.channels_last)
    masked_image_latents = masked_image_latents.contiguous(memory_format=module.channels_last)
    control_image_latents = control_image_latents.contiguous(memory_format=module.channels_last)
    return (
        text_encoder,
        add_embedding,
        controlnet,
        unet,
        vae,
        scheduler,
        input_ids_1,
        input_ids_2,
        add_time_ids,
        latents,
        init_latents,
        mask,
        masked_image_latents,
        control_image_latents,
    )


def _stable_diffusion_module_pipeline_path(
    module: Any,
    text_encoder: Any,
    unet: Any,
    vae: Any,
    scheduler: Any,
    input_ids: Any,
    latents: Any,
) -> Any:
    sample = latents.clone()
    with module.inference_mode():
        encoder_hidden = text_encoder(input_ids)
        for index, timestep in enumerate(scheduler.timesteps):
            latent_model_input = module.cat([sample, sample], dim=0)
            noise_pred = unet(latent_model_input, timestep, encoder_hidden)
            uncond, cond = noise_pred.chunk(2, dim=0)
            guided = uncond + (cond - uncond) * 5.5
            sample = scheduler.step(guided, module.tensor(index, dtype=module.long), sample)
        return vae(sample)


def _stable_diffusion_inpaint_module_pipeline_path(
    module: Any,
    text_encoder: Any,
    unet: Any,
    vae: Any,
    scheduler: Any,
    input_ids: Any,
    latents: Any,
    init_latents: Any,
    mask: Any,
    masked_image_latents: Any,
) -> Any:
    sample = latents.clone()
    preserve_mask = mask.to(dtype=sample.dtype).contiguous(memory_format=module.channels_last)
    masked_image_latents = masked_image_latents.to(dtype=sample.dtype).contiguous(memory_format=module.channels_last)
    with module.inference_mode():
        encoder_hidden = text_encoder(input_ids)
        for index, timestep in enumerate(scheduler.timesteps):
            latent_model_input = module.cat([sample, sample], dim=0).contiguous(memory_format=module.channels_last)
            mask_model_input = module.cat([preserve_mask, preserve_mask], dim=0).contiguous(
                memory_format=module.channels_last
            )
            masked_image_model_input = module.cat([masked_image_latents, masked_image_latents], dim=0).contiguous(
                memory_format=module.channels_last
            )
            inpaint_model_input = module.cat(
                [latent_model_input, mask_model_input, masked_image_model_input],
                dim=1,
            ).contiguous(memory_format=module.channels_last)
            noise_pred = unet(inpaint_model_input, timestep, encoder_hidden)
            uncond, cond = noise_pred.chunk(2, dim=0)
            guided = uncond + (cond - uncond) * 5.5
            denoised = scheduler.step(guided, module.tensor(index, dtype=module.long), sample)
            sample = (init_latents * preserve_mask + denoised * (1.0 - preserve_mask)).contiguous(
                memory_format=module.channels_last
            )
        return vae(sample)


def _stable_diffusion_sdxl_inpaint_controlnet_module_pipeline_path(
    module: Any,
    text_encoder: Any,
    add_embedding: Any,
    controlnet: Any,
    unet: Any,
    vae: Any,
    scheduler: Any,
    input_ids_1: Any,
    input_ids_2: Any,
    add_time_ids: Any,
    latents: Any,
    init_latents: Any,
    mask: Any,
    masked_image_latents: Any,
    control_image_latents: Any,
) -> Any:
    sample = latents.clone()
    preserve_mask = mask.to(dtype=sample.dtype).contiguous(memory_format=module.channels_last)
    masked_image_latents = masked_image_latents.to(dtype=sample.dtype).contiguous(memory_format=module.channels_last)
    control_image_latents = control_image_latents.to(dtype=sample.dtype).contiguous(memory_format=module.channels_last)
    with module.inference_mode():
        encoder_hidden, pooled_prompt_embeds = text_encoder(input_ids_1, input_ids_2)
        added_cond = add_embedding(pooled_prompt_embeds, add_time_ids)
        for index, timestep in enumerate(scheduler.timesteps):
            latent_model_input = module.cat([sample, sample], dim=0).contiguous(memory_format=module.channels_last)
            mask_model_input = module.cat([preserve_mask, preserve_mask], dim=0).contiguous(
                memory_format=module.channels_last
            )
            masked_image_model_input = module.cat([masked_image_latents, masked_image_latents], dim=0).contiguous(
                memory_format=module.channels_last
            )
            inpaint_model_input = module.cat(
                [latent_model_input, mask_model_input, masked_image_model_input],
                dim=1,
            ).contiguous(memory_format=module.channels_last)
            control_model_input = module.cat([control_image_latents, control_image_latents], dim=0).contiguous(
                memory_format=module.channels_last
            )
            control_residuals = controlnet(control_model_input, mask_model_input, timestep, added_cond)
            noise_pred = unet(inpaint_model_input, timestep, encoder_hidden, added_cond, control_residuals)
            uncond, cond = noise_pred.chunk(2, dim=0)
            guided = uncond + (cond - uncond) * 5.5
            denoised = scheduler.step(guided, module.tensor(index, dtype=module.long), sample)
            sample = (init_latents * preserve_mask + denoised * (1.0 - preserve_mask)).contiguous(
                memory_format=module.channels_last
            )
        return vae(sample)


def _case_nn_linear_module_setup(module: Any) -> tuple[Any, Any]:
    layer = module.nn.Linear(64, 64)
    _copy_parameter(module, layer.weight, _tensor(module, _matrix(64, 64)))
    _copy_parameter(module, layer.bias, _tensor(module, _vector(64)))
    return layer, _tensor(module, _matrix(64, 64))


def _case_nn_multihead_attention_setup(module: Any) -> tuple[Any, Any]:
    layer = module.nn.MultiheadAttention(32, 4, dropout=0.0, batch_first=True)
    _copy_parameter(module, layer.in_proj_weight, _tensor(module, _matrix(96, 32)))
    _copy_parameter(module, layer.in_proj_bias, _tensor(module, _vector(96)))
    _copy_parameter(module, layer.out_proj.weight, _tensor(module, _matrix(32, 32)))
    _copy_parameter(module, layer.out_proj.bias, _tensor(module, _vector(32)))
    return layer, _tensor(module, _tensor3(1, 8, 32))


def _case_nn_multihead_attention_mask_setup(module: Any) -> tuple[Any, Any, Any]:
    layer, input = _case_nn_multihead_attention_setup(module)
    key_padding_mask = module.tensor([[False, False, False, False, False, False, True, True]], dtype=module.bool)
    return layer, input, key_padding_mask


def _case_nn_multihead_attention_cross_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    layer, query = _case_nn_multihead_attention_setup(module)
    key = _tensor(module, _tensor3(1, 8, 32))
    value = _tensor(module, _tensor3(1, 8, 32))
    return layer, query, key, value


def _case_nn_multihead_attention_kv_setup(module: Any) -> tuple[Any, Any, Any, Any]:
    layer = module.nn.MultiheadAttention(32, 4, dropout=0.0, batch_first=True, kdim=24, vdim=16)
    _copy_parameter(module, layer.q_proj_weight, _tensor(module, _matrix(32, 32)))
    _copy_parameter(module, layer.k_proj_weight, _tensor(module, _matrix(32, 24)))
    _copy_parameter(module, layer.v_proj_weight, _tensor(module, _matrix(32, 16)))
    _copy_parameter(module, layer.in_proj_bias, _tensor(module, _vector(96)))
    _copy_parameter(module, layer.out_proj.weight, _tensor(module, _matrix(32, 32)))
    _copy_parameter(module, layer.out_proj.bias, _tensor(module, _vector(32)))
    return layer, _tensor(module, _tensor3(1, 8, 32)), _tensor(module, _tensor3(1, 8, 24)), _tensor(module, _tensor3(1, 8, 16))


def _case_nn_sequential_setup(module: Any) -> tuple[Any, Any]:
    layer, input = _case_nn_linear_module_setup(module)
    return module.nn.Sequential(layer, module.nn.ReLU()), input


def _case_addmm_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _vector(64)), _tensor(module, _matrix(64, 64)), _tensor(module, _matrix(64, 64))


def _case_addmv_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _vector(64)), _tensor(module, _matrix(64, 64)), _tensor(module, _vector(64))


def _case_addr_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _vector(64)), _tensor(module, [1.0 + value for value in _vector(64)])


def _case_bmm_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _tensor3(8, 32, 32)), _tensor(module, _tensor3(8, 32, 32))


def _case_baddbmm_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _tensor3(8, 32, 32)), _tensor(module, _tensor3(8, 32, 32)), _tensor(module, _tensor3(8, 32, 32))


def _case_addbmm_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _matrix(32, 32)), _tensor(module, _tensor3(8, 32, 32)), _tensor(module, _tensor3(8, 32, 32))


def _case_chain_matmul_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _matrix(32, 32)), _tensor(module, _matrix(32, 32)), _tensor(module, _matrix(32, 32))


def _case_matrix_power_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _matrix(32, 32)),)


def _case_kron_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(16, 16)), _tensor(module, _near_one_matrix(16, 16))


def _case_block_diag_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(32, 32)), _tensor(module, _near_one_matrix(32, 32))


def _case_cartesian_prod_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _vector(32)), _tensor(module, [1.0 + value for value in _vector(32)])


def _case_transpose_contiguous_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _matrix(96, 64)),)


def _case_permute_contiguous_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _tensor3(24, 32, 16)),)


def _case_cast_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _matrix(64, 64)),)


def _case_new_factory_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _matrix(64, 64), "float64"),)


def _case_type_as_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _matrix(64, 64), "float64")


def _case_batch_norm_setup(module: Any) -> tuple[Any, Any, Any, Any, Any]:
    running_mean = _tensor(module, _vector(64))
    running_var = _tensor(module, [1.0 + abs(value) for value in _vector(64)])
    weight = _tensor(module, [1.0 + value / 10.0 for value in _vector(64)])
    bias = _tensor(module, [value / 10.0 for value in _vector(64)])
    return _tensor(module, _matrix(64, 64)), running_mean, running_var, weight, bias


def _case_batch_norm2d_setup(module: Any) -> tuple[Any, Any, Any, Any, Any]:
    running_mean = _tensor(module, _vector(16))
    running_var = _tensor(module, [1.0 + abs(value) for value in _vector(16)])
    weight = _tensor(module, [1.0 + value / 10.0 for value in _vector(16)])
    bias = _tensor(module, [value / 10.0 for value in _vector(16)])
    return _tensor(module, _tensor4(8, 16, 8, 8)), running_mean, running_var, weight, bias


def _case_group_norm_setup(module: Any) -> tuple[Any, Any, Any]:
    weight = _tensor(module, [1.0 + value / 10.0 for value in _vector(16)])
    bias = _tensor(module, [value / 10.0 for value in _vector(16)])
    return _tensor(module, _tensor4(8, 16, 8, 8)), weight, bias


def _case_layer_norm_half_setup(module: Any) -> tuple[Any, Any, Any]:
    tensor = [
        [
            [float(((batch * 77 * 64 + token * 64 + col) % 257) - 128) / 512.0 for col in range(64)]
            for token in range(77)
        ]
        for batch in range(2)
    ]
    weight = [1.0 + float(index) / 256.0 for index in range(64)]
    bias = [float(index) / 2048.0 - 0.015 for index in range(64)]
    return _tensor(module, tensor, "float16"), _tensor(module, weight, "float16"), _tensor(module, bias, "float16")


def _case_layer_norm_half_768_setup(module: Any) -> tuple[Any, Any, Any]:
    tensor = [
        [
            [
                float(((batch * 77 * 768 + token * 768 + col) % 1009) - 504) / 1024.0
                for col in range(768)
            ]
            for token in range(77)
        ]
        for batch in range(2)
    ]
    weight = [0.75 + float(index % 257) / 2048.0 for index in range(768)]
    bias = [float((index % 127) - 63) / 4096.0 for index in range(768)]
    return _tensor(module, tensor, "float16"), _tensor(module, weight, "float16"), _tensor(module, bias, "float16")


def _case_quick_gelu_half_768_setup(module: Any) -> tuple[Any]:
    tensor = [
        [
            [
                float(((batch * 77 * 768 + token * 768 + col) % 1009) - 504) / 512.0
                for col in range(768)
            ]
            for token in range(77)
        ]
        for batch in range(2)
    ]
    return (_tensor(module, tensor, "float16"),)


def _case_stable_diffusion_half_clip_mlp_setup(module: Any) -> tuple[Any, Any, Any, Any, Any]:
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
    return hidden, fc1_weight, fc1_bias, fc2_weight, fc2_bias


def _stable_diffusion_half_clip_mlp_path(
    module: Any,
    hidden: Any,
    fc1_weight: Any,
    fc1_bias: Any,
    fc2_weight: Any,
    fc2_bias: Any,
) -> Any:
    hidden = module.nn.functional.linear(hidden, fc1_weight, fc1_bias)
    hidden = hidden * module.sigmoid(hidden * 1.702)
    return module.nn.functional.linear(hidden, fc2_weight, fc2_bias)


def _case_stable_diffusion_half_clip_attention_large_setup(module: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    batch, tokens, width = 2, 77, 768
    hidden = (module.arange(0, batch * tokens * width, dtype=module.float32).reshape(batch, tokens, width) % 1009)
    hidden = (hidden / 2048.0 - 0.25).to(dtype=module.float16)
    base_weight = (module.arange(0, width * width, dtype=module.float32).reshape(width, width) % 1021) / 16384.0 - 0.03
    q_weight = base_weight.to(dtype=module.float16)
    k_weight = module.flip(base_weight, (0,)).to(dtype=module.float16)
    v_weight = module.flip(base_weight, (1,)).to(dtype=module.float16)
    out_weight = base_weight.transpose(0, 1).contiguous().to(dtype=module.float16)
    bias = ((module.arange(0, width, dtype=module.float32) % 127) / 8192.0 - 0.008).to(dtype=module.float16)
    return hidden, q_weight, k_weight, v_weight, out_weight, bias


def _stable_diffusion_half_clip_attention_large_path(
    module: Any,
    hidden: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    out_weight: Any,
    bias: Any,
) -> Any:
    batch, tokens, width, heads, head_dim = 2, 77, 768, 12, 64
    query = module.nn.functional.linear(hidden, q_weight, bias).reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    key = module.nn.functional.linear(hidden, k_weight, bias).reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    value = module.nn.functional.linear(hidden, v_weight, bias).reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    attended = module.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
    attended = attended.transpose(1, 2).reshape(batch, tokens, width)
    return module.nn.functional.linear(attended, out_weight, bias)


def _case_stable_diffusion_half_lora_attention_projection_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    batch, tokens, width, rank = 2, 77, 768, 8
    hidden = (module.arange(0, batch * tokens * width, dtype=module.float32).reshape(batch, tokens, width) % 1009)
    hidden = (hidden / 2048.0 - 0.25).to(dtype=module.float16)
    base_weight = (module.arange(0, width * width, dtype=module.float32).reshape(width, width) % 1021) / 32768.0 - 0.012
    q_weight = base_weight.to(dtype=module.float16)
    k_weight = module.flip(base_weight, (0,)).to(dtype=module.float16)
    v_weight = module.flip(base_weight, (1,)).to(dtype=module.float16)
    out_weight = base_weight.transpose(0, 1).contiguous().to(dtype=module.float16)
    bias = ((module.arange(0, width, dtype=module.float32) % 127) / 8192.0 - 0.008).to(dtype=module.float16)
    down_weight = ((module.arange(0, rank * width, dtype=module.float32).reshape(rank, width) % 257) / 16384.0 - 0.006).to(
        dtype=module.float16
    )
    up_weight = ((module.arange(0, width * rank, dtype=module.float32).reshape(width, rank) % 251) / 16384.0 - 0.006).to(
        dtype=module.float16
    )
    return hidden, q_weight, k_weight, v_weight, out_weight, bias, down_weight, up_weight


def _stable_diffusion_half_lora_linear_path(
    module: Any,
    hidden: Any,
    weight: Any,
    bias: Any,
    down_weight: Any,
    up_weight: Any,
    scale: float = 0.75,
) -> Any:
    projected = module.nn.functional.linear(hidden, weight, bias)
    adapted = module.nn.functional.linear(module.nn.functional.linear(hidden, down_weight, None), up_weight, None)
    return projected + adapted * scale


def _stable_diffusion_half_lora_attention_projection_path(
    module: Any,
    hidden: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    out_weight: Any,
    bias: Any,
    down_weight: Any,
    up_weight: Any,
) -> Any:
    batch, tokens, width, heads, head_dim = 2, 77, 768, 12, 64
    query = _stable_diffusion_half_lora_linear_path(module, hidden, q_weight, bias, down_weight, up_weight)
    key = _stable_diffusion_half_lora_linear_path(module, hidden, k_weight, bias, down_weight, up_weight)
    value = _stable_diffusion_half_lora_linear_path(module, hidden, v_weight, bias, down_weight, up_weight)
    query = query.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    attended = module.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
    attended = attended.transpose(1, 2).reshape(batch, tokens, width)
    return _stable_diffusion_half_lora_linear_path(module, attended, out_weight, bias, down_weight, up_weight)


def _case_stable_diffusion_half_clip_transformer_block_setup(
    module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    batch, tokens, width = 2, 77, 768
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
    return (
        hidden,
        norm1_weight,
        norm1_bias,
        norm2_weight,
        norm2_bias,
        q_weight,
        k_weight,
        v_weight,
        out_weight,
        attn_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )


def _stable_diffusion_half_clip_transformer_block_path(
    module: Any,
    hidden: Any,
    norm1_weight: Any,
    norm1_bias: Any,
    norm2_weight: Any,
    norm2_bias: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    out_weight: Any,
    attn_bias: Any,
    fc1_weight: Any,
    fc1_bias: Any,
    fc2_weight: Any,
    fc2_bias: Any,
) -> Any:
    batch, tokens, width, heads, head_dim = 2, 77, 768, 12, 64
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


def _stable_diffusion_half_clip_large_stack_layer_setup(module: Any, layer: int) -> tuple[Any, ...]:
    width = 768
    offset = layer + 1
    norm1_weight = (
        0.75 + ((module.arange(0, width, dtype=module.float32) + offset * 17) % 257) / 2048.0
    ).to(dtype=module.float16)
    norm1_bias = (
        (((module.arange(0, width, dtype=module.float32) + offset * 19) % 127) - 63.0) / 4096.0
    ).to(dtype=module.float16)
    norm2_weight = (
        0.8 + ((module.arange(0, width, dtype=module.float32) + offset * 23) % 251) / 2048.0
    ).to(dtype=module.float16)
    norm2_bias = (
        (((module.arange(0, width, dtype=module.float32) + offset * 29) % 131) - 65.0) / 4096.0
    ).to(dtype=module.float16)
    base_weight = (
        (module.arange(0, width * width, dtype=module.float32).reshape(width, width) + offset * 31) % 1021
    ) / 16384.0 - 0.03
    q_weight = base_weight.to(dtype=module.float16)
    k_weight = module.flip(base_weight, (0,)).to(dtype=module.float16)
    v_weight = module.flip(base_weight, (1,)).to(dtype=module.float16)
    out_weight = base_weight.transpose(0, 1).contiguous().to(dtype=module.float16)
    attn_bias = (((module.arange(0, width, dtype=module.float32) + offset * 37) % 127) / 8192.0 - 0.008).to(
        dtype=module.float16
    )
    fc1_weight = (
        ((module.arange(0, 3072 * width, dtype=module.float32).reshape(3072, width) + offset * 41) % 1021)
        / 16384.0
        - 0.03
    ).to(dtype=module.float16)
    fc2_weight = (
        ((module.arange(0, width * 3072, dtype=module.float32).reshape(width, 3072) + offset * 43) % 1031)
        / 16384.0
        - 0.03
    ).to(dtype=module.float16)
    fc1_bias = (((module.arange(0, 3072, dtype=module.float32) + offset * 47) % 127) / 8192.0 - 0.008).to(
        dtype=module.float16
    )
    fc2_bias = (((module.arange(0, width, dtype=module.float32) + offset * 53) % 127) / 8192.0 - 0.008).to(
        dtype=module.float16
    )
    return (
        norm1_weight,
        norm1_bias,
        norm2_weight,
        norm2_bias,
        q_weight,
        k_weight,
        v_weight,
        out_weight,
        attn_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )


def _case_stable_diffusion_half_clip_large_text_encoder_stack_setup(module: Any) -> tuple[Any, ...]:
    batch, tokens, width, vocab = 2, 77, 768, 128
    input_ids = module.tensor(
        [
            [127 if col == 70 + row else (row * 11 + col * 7) % 96 for col in range(tokens)]
            for row in range(batch)
        ],
        dtype=module.long,
    )
    position_ids = module.arange(tokens, dtype=module.long).unsqueeze(0).expand(batch, -1)
    token_base = module.arange(0, vocab * width, dtype=module.float32).reshape(vocab, width)
    position_base = module.arange(0, tokens * width, dtype=module.float32).reshape(tokens, width)
    token_weight = (module.sin(token_base * 0.007) * 0.08).to(dtype=module.float16)
    position_weight = (module.cos(position_base * 0.011) * 0.03).to(dtype=module.float16)
    layers = tuple(_stable_diffusion_half_clip_large_stack_layer_setup(module, layer) for layer in range(2))
    final_norm_weight = (0.85 + (module.arange(0, width, dtype=module.float32) % 241) / 2048.0).to(
        dtype=module.float16
    )
    final_norm_bias = (((module.arange(0, width, dtype=module.float32) % 139) - 69.0) / 4096.0).to(
        dtype=module.float16
    )
    projection_base = (
        module.arange(0, width * width, dtype=module.float32).reshape(width, width) % 1009
    ) / 32768.0 - 0.015
    projection_weight = projection_base.to(dtype=module.float16)
    projection_bias = ((module.arange(0, width, dtype=module.float32) % 113) / 8192.0 - 0.006).to(
        dtype=module.float16
    )
    return (
        input_ids,
        position_ids,
        token_weight,
        position_weight,
        layers,
        final_norm_weight,
        final_norm_bias,
        projection_weight,
        projection_bias,
    )


def _stable_diffusion_half_clip_large_text_encoder_stack_path(
    module: Any,
    input_ids: Any,
    position_ids: Any,
    token_weight: Any,
    position_weight: Any,
    layers: Any,
    final_norm_weight: Any,
    final_norm_bias: Any,
    projection_weight: Any,
    projection_bias: Any,
) -> Any:
    batch, tokens, width, heads, head_dim = 2, 77, 768, 12, 64
    hidden = module.nn.functional.embedding(input_ids, token_weight)
    hidden = hidden + module.nn.functional.embedding(position_ids, position_weight)
    for layer in layers:
        (
            norm1_weight,
            norm1_bias,
            norm2_weight,
            norm2_bias,
            q_weight,
            k_weight,
            v_weight,
            out_weight,
            attn_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        ) = layer
        normed = module.nn.functional.layer_norm(hidden, (width,), norm1_weight, norm1_bias, eps=1e-5)
        query = module.nn.functional.linear(normed, q_weight, attn_bias).reshape(batch, tokens, heads, head_dim).transpose(
            1, 2
        )
        key = module.nn.functional.linear(normed, k_weight, attn_bias).reshape(batch, tokens, heads, head_dim).transpose(
            1, 2
        )
        value = module.nn.functional.linear(normed, v_weight, attn_bias).reshape(
            batch, tokens, heads, head_dim
        ).transpose(1, 2)
        attended = module.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        hidden = hidden + module.nn.functional.linear(attended, out_weight, attn_bias)
        mlp_hidden = module.nn.functional.layer_norm(hidden, (width,), norm2_weight, norm2_bias, eps=1e-5)
        mlp_hidden = module.nn.functional.linear(mlp_hidden, fc1_weight, fc1_bias)
        mlp_hidden = mlp_hidden * module.sigmoid(mlp_hidden * 1.702)
        hidden = hidden + module.nn.functional.linear(mlp_hidden, fc2_weight, fc2_bias)

    hidden = module.nn.functional.layer_norm(hidden, (width,), final_norm_weight, final_norm_bias, eps=1e-5)
    pooled = hidden[module.arange(batch, dtype=module.long), input_ids.argmax(dim=-1)]
    return hidden, module.nn.functional.linear(pooled, projection_weight, projection_bias)


def _stable_diffusion_sdxl_text_encoder2_layer_setup(module: Any, layer: int) -> tuple[Any, ...]:
    width, intermediate = 1280, 5120
    offset = layer + 1
    norm1_weight = (
        0.75 + ((module.arange(0, width, dtype=module.float32) + offset * 17) % 293) / 2048.0
    ).to(dtype=module.float16)
    norm1_bias = (
        (((module.arange(0, width, dtype=module.float32) + offset * 19) % 149) - 74.0) / 4096.0
    ).to(dtype=module.float16)
    norm2_weight = (
        0.8 + ((module.arange(0, width, dtype=module.float32) + offset * 23) % 307) / 2048.0
    ).to(dtype=module.float16)
    norm2_bias = (
        (((module.arange(0, width, dtype=module.float32) + offset * 29) % 157) - 78.0) / 4096.0
    ).to(dtype=module.float16)
    base_weight = (
        (module.arange(0, width * width, dtype=module.float32).reshape(width, width) + offset * 31) % 1543
    ) / 32768.0 - 0.018
    q_weight = base_weight.to(dtype=module.float16)
    k_weight = module.flip(base_weight, (0,)).to(dtype=module.float16)
    v_weight = module.flip(base_weight, (1,)).to(dtype=module.float16)
    out_weight = base_weight.transpose(0, 1).contiguous().to(dtype=module.float16)
    attn_bias = (((module.arange(0, width, dtype=module.float32) + offset * 37) % 173) / 8192.0 - 0.01).to(
        dtype=module.float16
    )
    fc1_weight = (
        ((module.arange(0, intermediate * width, dtype=module.float32).reshape(intermediate, width) + offset * 41)
        % 1553)
        / 32768.0
        - 0.018
    ).to(dtype=module.float16)
    fc2_weight = (
        ((module.arange(0, width * intermediate, dtype=module.float32).reshape(width, intermediate) + offset * 43)
        % 1567)
        / 32768.0
        - 0.018
    ).to(dtype=module.float16)
    fc1_bias = (((module.arange(0, intermediate, dtype=module.float32) + offset * 47) % 181) / 8192.0 - 0.011).to(
        dtype=module.float16
    )
    fc2_bias = (((module.arange(0, width, dtype=module.float32) + offset * 53) % 191) / 8192.0 - 0.011).to(
        dtype=module.float16
    )
    return (
        norm1_weight,
        norm1_bias,
        norm2_weight,
        norm2_bias,
        q_weight,
        k_weight,
        v_weight,
        out_weight,
        attn_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )


def _case_stable_diffusion_sdxl_text_encoder2_stack_setup(module: Any) -> tuple[Any, ...]:
    batch, tokens, width, vocab = 2, 77, 1280, 160
    input_ids = module.tensor(
        [
            [vocab - 1 if col == 70 + row else (row * 13 + col * 7) % (vocab - 32) for col in range(tokens)]
            for row in range(batch)
        ],
        dtype=module.long,
    )
    position_ids = module.arange(tokens, dtype=module.long).unsqueeze(0).expand(batch, -1)
    token_base = module.arange(0, vocab * width, dtype=module.float32).reshape(vocab, width)
    position_base = module.arange(0, tokens * width, dtype=module.float32).reshape(tokens, width)
    token_weight = (module.sin(token_base * 0.005) * 0.06).to(dtype=module.float16)
    position_weight = (module.cos(position_base * 0.009) * 0.025).to(dtype=module.float16)
    layers = tuple(_stable_diffusion_sdxl_text_encoder2_layer_setup(module, layer) for layer in range(1))
    final_norm_weight = (0.85 + (module.arange(0, width, dtype=module.float32) % 293) / 2048.0).to(
        dtype=module.float16
    )
    final_norm_bias = (((module.arange(0, width, dtype=module.float32) % 151) - 75.0) / 4096.0).to(
        dtype=module.float16
    )
    projection_base = (
        module.arange(0, width * width, dtype=module.float32).reshape(width, width) % 1543
    ) / 65536.0 - 0.011
    projection_weight = projection_base.to(dtype=module.float16)
    projection_bias = ((module.arange(0, width, dtype=module.float32) % 173) / 8192.0 - 0.01).to(
        dtype=module.float16
    )
    return (
        input_ids,
        position_ids,
        token_weight,
        position_weight,
        layers,
        final_norm_weight,
        final_norm_bias,
        projection_weight,
        projection_bias,
    )


def _stable_diffusion_sdxl_text_encoder2_stack_path(
    module: Any,
    input_ids: Any,
    position_ids: Any,
    token_weight: Any,
    position_weight: Any,
    layers: Any,
    final_norm_weight: Any,
    final_norm_bias: Any,
    projection_weight: Any,
    projection_bias: Any,
) -> Any:
    batch, tokens, width, heads, head_dim = 2, 77, 1280, 20, 64
    hidden = module.nn.functional.embedding(input_ids, token_weight)
    hidden = hidden + module.nn.functional.embedding(position_ids, position_weight)
    for layer in layers:
        (
            norm1_weight,
            norm1_bias,
            norm2_weight,
            norm2_bias,
            q_weight,
            k_weight,
            v_weight,
            out_weight,
            attn_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        ) = layer
        normed = module.nn.functional.layer_norm(hidden, (width,), norm1_weight, norm1_bias, eps=1e-5)
        query = module.nn.functional.linear(normed, q_weight, attn_bias).reshape(batch, tokens, heads, head_dim).transpose(
            1, 2
        )
        key = module.nn.functional.linear(normed, k_weight, attn_bias).reshape(batch, tokens, heads, head_dim).transpose(
            1, 2
        )
        value = module.nn.functional.linear(normed, v_weight, attn_bias).reshape(
            batch, tokens, heads, head_dim
        ).transpose(1, 2)
        attended = module.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        hidden = hidden + module.nn.functional.linear(attended, out_weight, attn_bias)
        mlp_hidden = module.nn.functional.layer_norm(hidden, (width,), norm2_weight, norm2_bias, eps=1e-5)
        mlp_hidden = module.nn.functional.linear(mlp_hidden, fc1_weight, fc1_bias)
        mlp_hidden = mlp_hidden * module.sigmoid(mlp_hidden * 1.702)
        hidden = hidden + module.nn.functional.linear(mlp_hidden, fc2_weight, fc2_bias)

    hidden = module.nn.functional.layer_norm(hidden, (width,), final_norm_weight, final_norm_bias, eps=1e-5)
    pooled = hidden[module.arange(batch, dtype=module.long), input_ids.argmax(dim=-1)]
    return hidden, module.nn.functional.linear(pooled, projection_weight, projection_bias)


def _case_stable_diffusion_sdxl_add_time_conditioning_setup(module: Any) -> tuple[Any, ...]:
    batch, pooled_width, time_ids, time_embed_width, out_width = 8, 1280, 6, 256, 1280
    pooled_base = module.arange(0, batch * pooled_width, dtype=module.float32).reshape(batch, pooled_width)
    pooled_prompt_embeds = (module.sin((pooled_base % 1543) * 0.005) * 0.2).to(dtype=module.float16)
    add_time_ids = module.tensor(
        [
            [1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0],
            [768.0, 1024.0, 32.0, 16.0, 768.0, 1024.0],
            [1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0],
            [768.0, 1024.0, 32.0, 16.0, 768.0, 1024.0],
            [1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0],
            [768.0, 1024.0, 32.0, 16.0, 768.0, 1024.0],
            [1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0],
            [768.0, 1024.0, 32.0, 16.0, 768.0, 1024.0],
        ],
        dtype=module.float32,
    )
    input_width = pooled_width + time_ids * time_embed_width
    fc1_base = module.arange(0, out_width * input_width, dtype=module.float32).reshape(out_width, input_width)
    fc2_base = module.arange(0, out_width * out_width, dtype=module.float32).reshape(out_width, out_width)
    fc1_weight = ((fc1_base % 1553) / 65536.0 - 0.012).to(dtype=module.float16)
    fc2_weight = ((fc2_base % 1543) / 65536.0 - 0.012).to(dtype=module.float16)
    fc1_bias = ((module.arange(0, out_width, dtype=module.float32) % 173) / 8192.0 - 0.01).to(dtype=module.float16)
    fc2_bias = ((module.arange(0, out_width, dtype=module.float32) % 181) / 8192.0 - 0.011).to(dtype=module.float16)
    return pooled_prompt_embeds, add_time_ids, fc1_weight, fc1_bias, fc2_weight, fc2_bias


def _stable_diffusion_sdxl_add_time_conditioning_path(
    module: Any,
    pooled_prompt_embeds: Any,
    add_time_ids: Any,
    fc1_weight: Any,
    fc1_bias: Any,
    fc2_weight: Any,
    fc2_bias: Any,
) -> Any:
    batch, time_ids, time_embed_width = pooled_prompt_embeds.shape[0], add_time_ids.shape[1], 256
    half_width = time_embed_width // 2
    frequencies = (module.arange(0, half_width, dtype=module.float32) + 1.0) / 512.0
    angles = add_time_ids.reshape(batch, time_ids, 1) * frequencies.reshape(1, 1, half_width)
    time_embeds = module.cat([module.sin(angles), module.cos(angles)], dim=-1).reshape(
        batch,
        time_ids * time_embed_width,
    )
    conditioning = module.cat([pooled_prompt_embeds, time_embeds.to(dtype=module.float16)], dim=-1)
    hidden = module.nn.functional.linear(conditioning, fc1_weight, fc1_bias)
    hidden = module.nn.functional.silu(hidden)
    return module.nn.functional.linear(hidden, fc2_weight, fc2_bias)


def _case_stable_diffusion_sdxl_prompt_conditioning_bundle_setup(module: Any) -> tuple[Any, ...]:
    dual_prompt_args = _case_stable_diffusion_sdxl_dual_prompt_encode_setup(module)
    _, add_time_ids, fc1_weight, fc1_bias, fc2_weight, fc2_bias = (
        _case_stable_diffusion_sdxl_add_time_conditioning_setup(module)
    )
    return dual_prompt_args, add_time_ids, fc1_weight, fc1_bias, fc2_weight, fc2_bias


def _stable_diffusion_sdxl_prompt_conditioning_bundle_path(
    module: Any,
    dual_prompt_args: Any,
    add_time_ids: Any,
    fc1_weight: Any,
    fc1_bias: Any,
    fc2_weight: Any,
    fc2_bias: Any,
) -> Any:
    prompt_embeds, pooled_prompt_embeds = _stable_diffusion_sdxl_dual_prompt_encode_path(module, *dual_prompt_args)
    add_embeds = _stable_diffusion_sdxl_add_time_conditioning_path(
        module,
        pooled_prompt_embeds,
        add_time_ids,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )
    return prompt_embeds, add_embeds


def _case_expand_contiguous_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _vector(64)),)


def _case_view_copy_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _matrix(64, 64)),)


def _case_bool_mask_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _mask(64, 64), "bool")


def _case_where_setup(module: Any) -> tuple[Any, Any, Any]:
    return _tensor(module, _mask(64, 64), "bool"), _tensor(module, _matrix(64, 64)), _tensor(module, _matrix(64, 64))


def _case_multinomial_one_hot_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, [[1.0 if col == 7 else 0.0 for col in range(64)] for _ in range(64)]),)


def _case_bincount_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, [(index * 17) % 257 for index in range(4096)], "int64"),)


def _case_bincount_weighted_setup(module: Any) -> tuple[Any, Any]:
    indices = _tensor(module, [(index * 17) % 257 for index in range(4096)], "int64")
    weights = _tensor(module, [float((index % 23) - 11) / 23.0 for index in range(4096)], "float32")
    return indices, weights


def _case_mixed_int_slice_setup(module: Any) -> tuple[Any, Any]:
    rows = [63, 0, 12, 31, 47, 5, 19, 28]
    return _tensor(module, _matrix(64, 64)), _tensor(module, rows, "int64")


def _case_nonleading_int_columns_setup(module: Any) -> tuple[Any, Any]:
    cols = [63, 0, 12, 31, 47, 5, 19, 28]
    return _tensor(module, _matrix(64, 64)), _tensor(module, cols, "int64")


def _case_setitem_nonleading_int_columns_setup(module: Any) -> tuple[Any, Any, Any]:
    cols = [63, 0, 12, 31, 47, 5, 19, 28]
    return _tensor(module, _matrix(64, 64)), _tensor(module, cols, "int64"), _tensor(module, _matrix(64, 8))


def _assign_nonleading_int_columns(module: Any, tensor: Any, cols: Any, source: Any) -> Any:
    tensor[:, cols] = source
    return tensor


def _case_copy_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _matrix(64, 64))


def _case_inplace_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _matrix(64, 64)),)


def _case_masked_fill_inplace_setup(module: Any) -> tuple[Any, Any]:
    return _tensor(module, _matrix(64, 64)), _tensor(module, _mask(64, 64), "bool")


def _case_autograd_grad_setup(module: Any) -> tuple[Any]:
    return (_tensor(module, _matrix(32, 32), requires_grad=True),)


def _case_optim_sgd_setup(module: Any) -> tuple[Any, Any]:
    parameter = _tensor(module, _matrix(32, 32), requires_grad=True)
    optimizer = module.optim.SGD([parameter], lr=0.01)
    return parameter, optimizer


def _sgd_training_step(module: Any, parameter: Any, optimizer: Any) -> Any:
    optimizer.zero_grad()
    loss = (parameter * parameter).sum()
    loss.backward()
    optimizer.step()
    return parameter


def _iadd_scalar(module: Any, tensor: Any) -> Any:
    tensor += 0.125
    return tensor


BENCHMARK_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        "bench.binary_add_64x64",
        _case_binary_add_setup,
        lambda module, left, right: module.add(left, right),
    ),
    BenchmarkCase(
        "bench.logical_and_64x64",
        _case_logical_setup,
        lambda module, left, right: module.logical_and(left, right),
    ),
    BenchmarkCase(
        "bench.logical_not_64x64",
        _case_logical_not_setup,
        lambda module, tensor: module.logical_not(tensor),
    ),
    BenchmarkCase(
        "bench.bitwise_and_bool_64x64",
        _case_logical_setup,
        lambda module, left, right: left & right,
    ),
    BenchmarkCase(
        "bench.bitwise_invert_bool_64x64",
        _case_logical_not_setup,
        lambda module, tensor: ~tensor,
    ),
    BenchmarkCase(
        "bench.isclose_64x64",
        _case_isclose_setup,
        lambda module, left, right: module.isclose(left, right),
    ),
    BenchmarkCase(
        "bench.allclose_64x64",
        _case_isclose_setup,
        lambda module, left, right: module.allclose(left, right),
    ),
    BenchmarkCase(
        "bench.remainder_64x64",
        _case_nonzero_binary_setup,
        lambda module, left, right: module.remainder(left, right),
    ),
    BenchmarkCase(
        "bench.fmod_64x64",
        _case_nonzero_binary_setup,
        lambda module, left, right: module.fmod(left, right),
    ),
    BenchmarkCase(
        "bench.floor_divide_64x64",
        _case_nonzero_binary_setup,
        lambda module, left, right: module.floor_divide(left, right),
    ),
    BenchmarkCase(
        "bench.atan2_64x64",
        _case_nonzero_binary_setup,
        lambda module, left, right: module.atan2(left, right),
    ),
    BenchmarkCase(
        "bench.hypot_64x64",
        _case_hypot_setup,
        lambda module, left, right: module.hypot(left, right),
    ),
    BenchmarkCase(
        "bench.ldexp_64x64",
        _case_binary_add_setup,
        lambda module, left, right: module.ldexp(left, right),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nextafter_64x64",
        _case_binary_add_setup,
        lambda module, left, right: module.nextafter(left, right),
    ),
    BenchmarkCase(
        "bench.float_power_64x64",
        _case_positive_binary_setup,
        lambda module, left, right: module.float_power(left, right),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.copysign_64x64",
        _case_binary_add_setup,
        lambda module, left, right: module.copysign(left, right),
    ),
    BenchmarkCase(
        "bench.heaviside_64x64",
        _case_heaviside_setup,
        lambda module, left, right: module.heaviside(left, right),
    ),
    BenchmarkCase(
        "bench.logaddexp_64x64",
        _case_binary_add_setup,
        lambda module, left, right: module.logaddexp(left, right),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.logaddexp2_64x64",
        _case_binary_add_setup,
        lambda module, left, right: module.logaddexp2(left, right),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.xlogy_64x64",
        _case_positive_binary_setup,
        lambda module, left, right: module.xlogy(left, right),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.fmax_64x64",
        _case_binary_add_setup,
        lambda module, left, right: module.fmax(left, right),
    ),
    BenchmarkCase(
        "bench.fmin_64x64",
        _case_binary_add_setup,
        lambda module, left, right: module.fmin(left, right),
    ),
    BenchmarkCase(
        "bench.lerp_tensor_weight_64x64",
        _case_lerp_setup,
        lambda module, left, right, weight: module.lerp(left, right, weight),
    ),
    BenchmarkCase(
        "bench.add_scalar_64x64",
        _case_cast_setup,
        lambda module, tensor: tensor.add(2.0),
    ),
    BenchmarkCase(
        "bench.broadcast_add_64x64",
        _case_broadcast_add_setup,
        lambda module, left, right: module.add(left, right),
    ),
    BenchmarkCase(
        "bench.broadcast_shapes",
        lambda module: ((64, 1, 16), (1, 32, 1), (64, 32, 16)),
        lambda module, left, right, third: module.broadcast_shapes(left, right, third),
    ),
    BenchmarkCase(
        "bench.broadcast_tensors_64x64",
        _case_broadcast_add_setup,
        lambda module, left, right: module.broadcast_tensors(left, right),
    ),
    BenchmarkCase(
        "bench.index_select_dim0_64x64",
        lambda module: (_tensor(module, _matrix(64, 64)), _tensor(module, [(i * 7) % 64 for i in range(32)], "int64")),
        lambda module, tensor, indices: module.index_select(tensor, 0, indices),
    ),
    BenchmarkCase(
        "bench.take_64x64",
        lambda module: (
            _tensor(module, _matrix(64, 64)),
            _tensor(module, [(i * 17) % (64 * 64) for i in range(1024)], "int64"),
        ),
        lambda module, tensor, indices: module.take(tensor, indices),
    ),
    BenchmarkCase(
        "bench.gather_dim1_64x64",
        lambda module: (
            _tensor(module, _matrix(64, 64)),
            _tensor(module, [[(row + col * 3) % 64 for col in range(32)] for row in range(64)], "int64"),
        ),
        lambda module, tensor, indices: module.gather(tensor, 1, indices),
    ),
    BenchmarkCase(
        "bench.index_put_pair_64x64",
        lambda module: (
            module.zeros((64, 64), dtype=module.float32),
            module.arange(0, 64, dtype=module.long),
            (module.arange(0, 64, dtype=module.long) * 7) % 64,
            module.ones((64,), dtype=module.float32),
        ),
        lambda module, tensor, rows, cols, values: module.index_put(tensor, (rows, cols), values),
    ),
    BenchmarkCase(
        "bench.take_along_dim_dim1_64x64",
        lambda module: (
            _tensor(module, _matrix(64, 64)),
            _tensor(module, [[(row + col * 3) % 64 for col in range(32)] for row in range(64)], "int64"),
        ),
        lambda module, tensor, indices: module.take_along_dim(tensor, indices, dim=1),
    ),
    BenchmarkCase(
        "bench.isin_int64_64x64",
        lambda module: (
            _tensor(module, [[(row * 64 + col) % 257 for col in range(64)] for row in range(64)], "int64"),
            module.arange(0, 128, dtype=module.int64),
        ),
        lambda module, tensor, test_values: module.isin(tensor, test_values),
    ),
    BenchmarkCase(
        "bench.scatter_dim1_64x64",
        lambda module: (
            _tensor(module, _matrix(64, 64)),
            _tensor(module, [[(row + col * 3) % 64 for col in range(32)] for row in range(64)], "int64"),
            _tensor(module, _matrix(64, 32)),
        ),
        lambda module, tensor, indices, source: module.scatter(tensor, 1, indices, source),
    ),
    BenchmarkCase(
        "bench.scatter_add_dim1_64x64",
        lambda module: (
            _tensor(module, [[0.0 for _ in range(64)] for _ in range(64)]),
            _tensor(module, [[(row + col * 3) % 64 for col in range(32)] for row in range(64)], "int64"),
            _tensor(module, _matrix(64, 32)),
        ),
        lambda module, tensor, indices, source: module.scatter_add(tensor, 1, indices, source),
    ),
    BenchmarkCase(
        "bench.masked_select_64x64",
        lambda module: (
            _tensor(module, _matrix(64, 64)),
            _tensor(module, [[(row + col) % 3 == 0 for col in range(64)] for row in range(64)], "bool"),
        ),
        lambda module, tensor, mask: module.masked_select(tensor, mask),
    ),
    BenchmarkCase(
        "bench.masked_fill_64x64",
        lambda module: (
            _tensor(module, _matrix(64, 64)),
            _tensor(module, [[(row + col) % 3 == 0 for col in range(64)] for row in range(64)], "bool"),
        ),
        lambda module, tensor, mask: module.masked_fill(tensor, mask, -1.0),
    ),
    BenchmarkCase(
        "bench.nonzero_64x64",
        lambda module: (_tensor(module, [[0.0 if (row + col) % 3 else 1.0 for col in range(64)] for row in range(64)]),),
        lambda module, tensor: module.nonzero(tensor),
    ),
    BenchmarkCase(
        "bench.count_nonzero_64x64",
        lambda module: (_tensor(module, [[0.0 if (row + col) % 3 else 1.0 for col in range(64)] for row in range(64)]),),
        lambda module, tensor: module.count_nonzero(tensor),
    ),
    BenchmarkCase(
        "bench.method_mul_64x64",
        _case_binary_add_setup,
        lambda module, left, right: left.mul(right),
    ),
    BenchmarkCase(
        "bench.split_64x64_dim0_size16",
        _case_cast_setup,
        lambda module, tensor: module.split(tensor, 16, dim=0),
    ),
    BenchmarkCase(
        "bench.chunk_64x64_dim0_chunks8",
        _case_cast_setup,
        lambda module, tensor: module.chunk(tensor, 8, dim=0),
    ),
    BenchmarkCase(
        "bench.unbind_8x64_dim0",
        lambda module: (_tensor(module, _matrix(8, 64)),),
        lambda module, tensor: module.unbind(tensor, dim=0),
    ),
    BenchmarkCase(
        "bench.as_strided_64x64",
        _case_cast_setup,
        lambda module, tensor: module.as_strided(tensor, (32, 32), (64, 1), storage_offset=16),
    ),
    BenchmarkCase(
        "bench.diagonal_64x64",
        _case_cast_setup,
        lambda module, tensor: module.diagonal(tensor),
    ),
    BenchmarkCase(
        "bench.diag_64x64",
        _case_cast_setup,
        lambda module, tensor: module.diag(tensor),
    ),
    BenchmarkCase(
        "bench.diagflat_vector_128",
        lambda module: (_tensor(module, _vector(128)),),
        lambda module, tensor: module.diagflat(tensor),
    ),
    BenchmarkCase(
        "bench.tril_64x64",
        _case_cast_setup,
        lambda module, tensor: module.tril(tensor),
    ),
    BenchmarkCase(
        "bench.triu_64x64",
        _case_cast_setup,
        lambda module, tensor: tensor.triu(),
    ),
    BenchmarkCase(
        "bench.ravel_transpose_64x64",
        _case_cast_setup,
        lambda module, tensor: module.ravel(module.transpose(tensor, 0, 1)),
    ),
    BenchmarkCase(
        "bench.empty_strided_fill_64x64",
        lambda module: (),
        lambda module: module.empty_strided((64, 64), (1, 64), dtype=module.float32).fill_(1.0),
    ),
    BenchmarkCase(
        "bench.trace_64x64",
        _case_cast_setup,
        lambda module, tensor: module.trace(tensor),
    ),
    BenchmarkCase(
        "bench.movedim_8x16x8x8",
        lambda module: (_tensor(module, _tensor4(8, 16, 8, 8)),),
        lambda module, tensor: module.movedim(tensor, (0, 1), (2, 0)),
    ),
    BenchmarkCase(
        "bench.swapaxes_8x16x8x8",
        lambda module: (_tensor(module, _tensor4(8, 16, 8, 8)),),
        lambda module, tensor: module.swapaxes(tensor, 1, 3),
    ),
    BenchmarkCase(
        "bench.maximum_64x64",
        _case_binary_add_setup,
        lambda module, left, right: module.maximum(left, right),
    ),
    BenchmarkCase(
        "bench.relu_64x64",
        _case_cast_setup,
        lambda module, tensor: tensor.relu(),
    ),
    BenchmarkCase(
        "bench.nn_functional_leaky_relu_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.leaky_relu(tensor, negative_slope=0.2),
    ),
    BenchmarkCase(
        "bench.nn_functional_silu_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.silu(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_elu_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.elu(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_selu_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.selu(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_softplus_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.softplus(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_hardtanh_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.hardtanh(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_gelu_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.gelu(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_relu6_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.relu6(tensor),
    ),
    BenchmarkCase(
        "bench.nn_functional_hardsigmoid_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.hardsigmoid(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_hardswish_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.hardswish(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_softsign_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.softsign(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_mish_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.mish(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_layer_norm_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.layer_norm(tensor, (64,)),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.nn_functional_layer_norm_half_2x77x64",
        _case_layer_norm_half_setup,
        lambda module, tensor, weight, bias: module.nn.functional.layer_norm(tensor, (64,), weight, bias, eps=1e-5),
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.nn_functional_layer_norm_half_2x77x768",
        _case_layer_norm_half_768_setup,
        lambda module, tensor, weight, bias: module.nn.functional.layer_norm(tensor, (768,), weight, bias, eps=1e-5),
        rtol=4e-3,
        atol=4e-3,
    ),
    BenchmarkCase(
        "bench.sigmoid_half_2x77x768",
        _case_quick_gelu_half_768_setup,
        lambda module, tensor: module.sigmoid(tensor),
        rtol=4e-3,
        atol=4e-3,
    ),
    BenchmarkCase(
        "bench.nn_functional_rms_norm_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.rms_norm(tensor, (64,), eps=1e-5),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.nn_functional_batch_norm_eval_64x64",
        _case_batch_norm_setup,
        lambda module, tensor, running_mean, running_var, weight, bias: module.nn.functional.batch_norm(
            tensor,
            running_mean,
            running_var,
            weight,
            bias,
            training=False,
        ),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.nn_functional_batch_norm2d_eval_8x16x8x8",
        _case_batch_norm2d_setup,
        lambda module, tensor, running_mean, running_var, weight, bias: module.nn.functional.batch_norm(
            tensor,
            running_mean,
            running_var,
            weight,
            bias,
            training=False,
        ),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.nn_functional_group_norm_8x16x8x8",
        _case_group_norm_setup,
        lambda module, tensor, weight, bias: module.nn.functional.group_norm(tensor, 4, weight, bias),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.nn_functional_normalize_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.normalize(tensor, p=2.0, dim=1),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_dropout_eval_64x64",
        _case_cast_setup,
        lambda module, tensor: module.nn.functional.dropout(tensor, p=0.5, training=False),
    ),
    BenchmarkCase(
        "bench.sin_64x64",
        _case_cast_setup,
        lambda module, tensor: module.sin(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.cos_64x64",
        _case_cast_setup,
        lambda module, tensor: module.cos(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.tan_64x64",
        _case_cast_setup,
        lambda module, tensor: module.tan(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.sinh_64x64",
        _case_cast_setup,
        lambda module, tensor: module.sinh(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.cosh_64x64",
        _case_cast_setup,
        lambda module, tensor: module.cosh(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.tanh_64x64",
        _case_cast_setup,
        lambda module, tensor: module.tanh(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.sigmoid_64x64",
        _case_cast_setup,
        lambda module, tensor: module.sigmoid(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.deg2rad_64x64",
        _case_cast_setup,
        lambda module, tensor: module.deg2rad(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.rad2deg_64x64",
        _case_cast_setup,
        lambda module, tensor: module.rad2deg(tensor),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.frac_64x64",
        _case_cast_setup,
        lambda module, tensor: module.frac(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.isfinite_64x64",
        _case_cast_setup,
        lambda module, tensor: module.isfinite(tensor),
    ),
    BenchmarkCase(
        "bench.signbit_64x64",
        _case_cast_setup,
        lambda module, tensor: module.signbit(tensor),
    ),
    BenchmarkCase(
        "bench.nan_to_num_64x64",
        _case_special_float_setup,
        lambda module, tensor: module.nan_to_num(tensor),
    ),
    BenchmarkCase(
        "bench.square_64x64",
        _case_cast_setup,
        lambda module, tensor: module.square(tensor),
    ),
    BenchmarkCase(
        "bench.floor_64x64",
        _case_cast_setup,
        lambda module, tensor: module.floor(tensor),
    ),
    BenchmarkCase(
        "bench.log1p_64x64",
        _case_cast_setup,
        lambda module, tensor: module.log1p(tensor.add(2.0)),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.expm1_64x64",
        _case_cast_setup,
        lambda module, tensor: module.expm1(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.log2_64x64",
        _case_cast_setup,
        lambda module, tensor: module.log2(tensor.add(2.0)),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.log10_64x64",
        _case_cast_setup,
        lambda module, tensor: module.log10(tensor.add(2.0)),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.rsqrt_64x64",
        _case_cast_setup,
        lambda module, tensor: module.rsqrt(tensor.add(2.0)),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.asin_64x64",
        _case_cast_setup,
        lambda module, tensor: module.asin(tensor.mul(0.5)),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.erf_64x64",
        _case_cast_setup,
        lambda module, tensor: module.erf(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.reciprocal_64x64",
        _case_cast_setup,
        lambda module, tensor: module.reciprocal(tensor.add(2.0)),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.clamp_64x64",
        _case_cast_setup,
        lambda module, tensor: module.clamp(tensor, min=-0.25, max=0.5),
    ),
    BenchmarkCase(
        "bench.clamp_min_64x64",
        _case_cast_setup,
        lambda module, tensor: module.clamp_min(tensor, 0.0),
    ),
    BenchmarkCase(
        "bench.clamp_max_64x64",
        _case_cast_setup,
        lambda module, tensor: module.clamp_max(tensor, 0.5),
    ),
    BenchmarkCase(
        "bench.clone_float32_64x64",
        _case_cast_setup,
        lambda module, tensor: module.clone(tensor),
    ),
    BenchmarkCase(
        "bench.sum_128x128",
        _case_sum_setup,
        lambda module, tensor: module.sum(tensor),
    ),
    BenchmarkCase(
        "bench.sum_dim1_128x128",
        _case_sum_setup,
        lambda module, tensor: module.sum(tensor, dim=1),
    ),
    BenchmarkCase(
        "bench.mean_dim1_128x128",
        _case_sum_setup,
        lambda module, tensor: module.mean(tensor, dim=1),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.cumsum_dim1_128x128",
        _case_sum_setup,
        lambda module, tensor: module.cumsum(tensor, dim=1),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.cumprod_dim1_128x128",
        lambda module: (_tensor(module, _near_one_matrix(128, 128)),),
        lambda module, tensor: module.cumprod(tensor, dim=1),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.cummax_dim1_128x128",
        _case_sum_setup,
        lambda module, tensor: module.cummax(tensor, dim=1),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.cummin_dim1_128x128",
        _case_sum_setup,
        lambda module, tensor: module.cummin(tensor, dim=1),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.argmax_int64_clip_2x77",
        _case_clip_argmax_int64_setup,
        lambda module, tensor: module.argmax(tensor, dim=-1),
    ),
    BenchmarkCase(
        "bench.diff_dim1_128x128",
        _case_sum_setup,
        lambda module, tensor: module.diff(tensor, dim=1),
    ),
    BenchmarkCase(
        "bench.gradient_dim1_128x128",
        _case_sum_setup,
        lambda module, tensor: module.gradient(tensor, dim=1)[0],
    ),
    BenchmarkCase(
        "bench.gradient_edge2_1024",
        lambda module: (module.arange(0, 1024, dtype=module.float32),),
        lambda module, tensor: module.gradient(tensor, spacing=0.01, dim=0, edge_order=2)[0],
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.trapezoid_dim1_128x128",
        _case_sum_setup,
        lambda module, tensor: module.trapezoid(tensor, dx=0.5, dim=1),
    ),
    BenchmarkCase(
        "bench.cumulative_trapezoid_dim1_128x128",
        _case_sum_setup,
        lambda module, tensor: module.cumulative_trapezoid(tensor, dx=0.5, dim=1),
    ),
    BenchmarkCase(
        "bench.amax_dim1_256x256",
        lambda module: (_tensor(module, _matrix(256, 256)),),
        lambda module, tensor: module.amax(tensor, dim=1),
    ),
    BenchmarkCase(
        "bench.prod_128x128",
        _case_prod_setup,
        lambda module, tensor: module.prod(tensor),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.sort_dim1_64x64",
        lambda module: (_tensor(module, _unique_matrix(64, 64)),),
        lambda module, tensor: module.sort(tensor, dim=1),
    ),
    BenchmarkCase(
        "bench.argsort_dim1_64x64",
        lambda module: (_tensor(module, _unique_matrix(64, 64)),),
        lambda module, tensor: module.argsort(tensor, dim=1),
    ),
    BenchmarkCase(
        "bench.topk_dim1_64x64",
        lambda module: (_tensor(module, _unique_matrix(64, 64)),),
        lambda module, tensor: module.topk(tensor, 8, dim=1),
    ),
    BenchmarkCase(
        "bench.searchsorted_1024",
        lambda module: (
            module.arange(0, 2048, 2, dtype=module.float32),
            module.arange(0, 2048, 3, dtype=module.float32),
        ),
        lambda module, boundaries, values: module.searchsorted(boundaries, values),
    ),
    BenchmarkCase(
        "bench.logspace_1024",
        lambda module: (),
        lambda module: module.logspace(-3.0, 1.0, 1024, dtype=module.float32),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.quantile_dim1_64x64",
        lambda module: (_tensor(module, _unique_matrix(64, 64)),),
        lambda module, tensor: module.quantile(tensor, 0.75, dim=1),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.quantile_all_64x64",
        lambda module: (_tensor(module, _unique_matrix(64, 64)),),
        lambda module, tensor: module.quantile(tensor, 0.75),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.multinomial_one_hot_64x64",
        _case_multinomial_one_hot_setup,
        lambda module, tensor: module.multinomial(tensor, 16, replacement=True),
    ),
    BenchmarkCase(
        "bench.bincount_int64_4096",
        _case_bincount_setup,
        lambda module, tensor: module.bincount(tensor, minlength=512),
    ),
    BenchmarkCase(
        "bench.bincount_weighted_float32_4096",
        _case_bincount_weighted_setup,
        lambda module, indices, weights: module.bincount(indices, weights=weights, minlength=512),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.bernoulli_binary_64x64",
        lambda module: (_tensor(module, [[float((row + col) % 2) for col in range(64)] for row in range(64)]),),
        lambda module, tensor: module.bernoulli(tensor),
    ),
    BenchmarkCase(
        "bench.unique_int64_4096",
        lambda module: (_tensor(module, [i % 257 for i in range(4096)], "int64"),),
        lambda module, tensor: module.unique(tensor, return_inverse=True, return_counts=True),
    ),
    BenchmarkCase(
        "bench.unique_dim0_int64_128x2",
        lambda module: (_tensor(module, [[i % 64, (i // 2) % 17] for i in range(128)], "int64"),),
        lambda module, tensor: module.unique(tensor, dim=0, return_inverse=True, return_counts=True),
    ),
    BenchmarkCase(
        "bench.unique_dim1_int64_2x128",
        lambda module: (_tensor(module, [[i % 64 for i in range(128)], [(i // 2) % 17 for i in range(128)]], "int64"),),
        lambda module, tensor: module.unique(tensor, dim=1, return_inverse=True, return_counts=True),
    ),
    BenchmarkCase(
        "bench.unique_consecutive_int64_1024",
        lambda module: (_tensor(module, [(i // 2) % 257 for i in range(1024)], "int64"),),
        lambda module, tensor: module.unique_consecutive(tensor, return_inverse=True, return_counts=True),
    ),
    BenchmarkCase(
        "bench.unique_consecutive_dim0_int64_128x2",
        lambda module: (_tensor(module, [[(i // 2) % 64, (i // 4) % 17] for i in range(128)], "int64"),),
        lambda module, tensor: module.unique_consecutive(tensor, dim=0, return_inverse=True, return_counts=True),
    ),
    BenchmarkCase(
        "bench.unique_consecutive_dim1_int64_2x128",
        lambda module: (
            _tensor(module, [[(i // 2) % 64 for i in range(128)], [(i // 4) % 17 for i in range(128)]], "int64"),
        ),
        lambda module, tensor: module.unique_consecutive(tensor, dim=1, return_inverse=True, return_counts=True),
    ),
    BenchmarkCase(
        "bench.softmax_dim1_64x64",
        _case_cast_setup,
        lambda module, tensor: module.softmax(tensor, dim=1),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.log_softmax_dim1_64x64",
        _case_cast_setup,
        lambda module, tensor: module.log_softmax(tensor, dim=1),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.norm_dim1_64x64",
        _case_cast_setup,
        lambda module, tensor: tensor.norm(dim=1),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.norm_p0_dim1_64x64",
        _case_cast_setup,
        lambda module, tensor: tensor.norm(p=0.0, dim=1),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.var_128x128",
        _case_sum_setup,
        lambda module, tensor: module.var(tensor),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.std_128x128",
        _case_sum_setup,
        lambda module, tensor: module.std(tensor),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.var_mean_dim1_128x128",
        _case_sum_setup,
        lambda module, tensor: module.var_mean(tensor, dim=1, correction=0),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.std_mean_dim1_128x128",
        _case_sum_setup,
        lambda module, tensor: module.std_mean(tensor, dim=1, correction=0),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.all_bool_128x128",
        _case_all_setup,
        lambda module, tensor: module.all(tensor),
    ),
    BenchmarkCase(
        "bench.matmul_48x48",
        _case_matmul_setup,
        lambda module, left, right: module.matmul(left, right),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.matmul_operator_48x48",
        _case_matmul_setup,
        lambda module, left, right: left @ right,
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.matmul_batched_8x32x32",
        _case_matmul_batched_setup,
        lambda module, left, right: module.matmul(left, right),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.matmul_broadcast_8x32x32",
        _case_matmul_broadcast_setup,
        lambda module, left, right: module.matmul(left, right),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.dot_1024",
        _case_dot_setup,
        lambda module, left, right: module.dot(left, right),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.vdot_1024",
        _case_dot_setup,
        lambda module, left, right: module.vdot(left, right),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.inner_64x64",
        _case_inner_setup,
        lambda module, left, right: module.inner(left, right),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.tensordot_64x64",
        _case_inner_setup,
        lambda module, left, right: module.tensordot(left, right, dims=1),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.einsum_matmul_48x48",
        _case_matmul_setup,
        lambda module, left, right: module.einsum("ij,jk->ik", left, right),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.einsum_batched_matmul_8x32x32",
        _case_bmm_setup,
        lambda module, left, right: module.einsum("bij,bjk->bik", left, right),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.mv_64x64",
        _case_mv_setup,
        lambda module, matrix, vector: module.mv(matrix, vector),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.outer_64x64",
        lambda module: (
            _tensor(module, [float((i % 13) - 6) / 7.0 for i in range(64)]),
            _tensor(module, [float((i % 11) - 5) / 6.0 for i in range(64)]),
        ),
        lambda module, left, right: module.outer(left, right),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.addmm_64x64",
        _case_addmm_setup,
        lambda module, input, mat1, mat2: module.addmm(input, mat1, mat2),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.addmv_64x64",
        _case_addmv_setup,
        lambda module, input, mat, vec: module.addmv(input, mat, vec),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.addr_64x64",
        _case_addr_setup,
        lambda module, input, vec1, vec2: module.addr(input, vec1, vec2),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.addcmul_64x64",
        _case_addcmul_setup,
        lambda module, input, tensor1, tensor2: module.addcmul(input, tensor1, tensor2, value=0.5),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.addcdiv_64x64",
        _case_addcdiv_setup,
        lambda module, input, tensor1, tensor2: module.addcdiv(input, tensor1, tensor2, value=0.5),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.bmm_8x32x32",
        _case_bmm_setup,
        lambda module, left, right: module.bmm(left, right),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.baddbmm_8x32x32",
        _case_baddbmm_setup,
        lambda module, input, batch1, batch2: module.baddbmm(input, batch1, batch2),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.addbmm_8x32x32",
        _case_addbmm_setup,
        lambda module, input, batch1, batch2: module.addbmm(input, batch1, batch2),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.chain_matmul_32x32",
        _case_chain_matmul_setup,
        lambda module, left, middle, right: module.chain_matmul(left, middle, right),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.linalg_multi_dot_32x32",
        _case_chain_matmul_setup,
        lambda module, left, middle, right: module.linalg.multi_dot([left, middle, right]),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.matrix_power_32x32",
        _case_matrix_power_setup,
        lambda module, matrix: module.matrix_power(matrix, 3),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.kron_16x16",
        _case_kron_setup,
        lambda module, left, right: module.kron(left, right),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.diag_embed_64x64",
        _case_cast_setup,
        lambda module, tensor: module.diag_embed(tensor),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.block_diag_32x32",
        _case_block_diag_setup,
        lambda module, left, right: module.block_diag(left, right),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_linear_64x64",
        _case_linear_setup,
        lambda module, input, weight, bias: module.nn.functional.linear(input, weight, bias),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_functional_linear_8x32x64",
        _case_linear_3d_setup,
        lambda module, input, weight, bias: module.nn.functional.linear(input, weight, bias),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_functional_linear_half_2x77x64",
        _case_linear_half_projection_setup,
        lambda module, input, weight, bias: module.nn.functional.linear(input, weight, bias),
        rtol=2e-3,
        atol=2e-3,
    ),
    BenchmarkCase(
        "bench.nn_functional_conv1d_1x4x32",
        _case_conv1d_setup,
        lambda module, input, weight, bias: module.nn.functional.conv1d(input, weight, bias, padding=1),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_functional_conv_transpose1d_1x4x32",
        _case_conv_transpose1d_setup,
        lambda module, input, weight, bias: module.nn.functional.conv_transpose1d(input, weight, bias, stride=2, padding=1, output_padding=1),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_functional_max_pool1d_1x4x32",
        _case_pool1d_setup,
        lambda module, input: module.nn.functional.max_pool1d(input, 2, stride=2),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_avg_pool1d_1x4x32",
        _case_pool1d_setup,
        lambda module, input: module.nn.functional.avg_pool1d(input, 2, stride=2),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_adaptive_avg_pool1d_1x4x32",
        _case_pool1d_setup,
        lambda module, input: module.nn.functional.adaptive_avg_pool1d(input, 8),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_conv2d_1x4x8x8",
        _case_conv2d_setup,
        lambda module, input, weight, bias: module.nn.functional.conv2d(input, weight, bias, padding=1),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_functional_conv2d_half_1x1_1x16x8x8",
        _case_conv2d_half_1x1_setup,
        lambda module, input, weight, bias: module.nn.functional.conv2d(input, weight, bias),
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.nn_functional_conv2d_same_1x4x8x8",
        _case_conv2d_setup,
        lambda module, input, weight, bias: module.nn.functional.conv2d(input, weight, bias, padding="same"),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_functional_conv3d_1x2x4x4x4",
        _case_conv3d_setup,
        lambda module, input, weight, bias: module.nn.functional.conv3d(input, weight, bias, padding=1),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_functional_conv3d_half_1x4x4x8x8",
        _case_conv3d_half_setup,
        lambda module, input, weight, bias: module.nn.functional.conv3d(input, weight, bias, padding=1),
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.nn_functional_conv_transpose2d_1x4x8x8",
        _case_conv_transpose2d_setup,
        lambda module, input, weight, bias: module.nn.functional.conv_transpose2d(input, weight, bias, stride=2, padding=1, output_padding=1),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_functional_conv_transpose3d_1x2x4x4x4",
        _case_conv_transpose3d_setup,
        lambda module, input, weight, bias: module.nn.functional.conv_transpose3d(input, weight, bias, stride=2, padding=1, output_padding=1),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_functional_max_pool2d_1x4x16x16",
        _case_interpolate_setup,
        lambda module, input: module.nn.functional.max_pool2d(input, 2, stride=2),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_avg_pool2d_1x4x16x16",
        _case_interpolate_setup,
        lambda module, input: module.nn.functional.avg_pool2d(input, 2, stride=2),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_adaptive_avg_pool2d_1x4x16x16",
        _case_interpolate_setup,
        lambda module, input: module.nn.functional.adaptive_avg_pool2d(input, (1, 1)),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_adaptive_avg_pool2d_8x8_1x4x16x16",
        _case_interpolate_setup,
        lambda module, input: module.nn.functional.adaptive_avg_pool2d(input, (8, 8)),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_unfold_1x4x16x16",
        _case_interpolate_setup,
        lambda module, input: module.nn.functional.unfold(input, (3, 3), padding=1, stride=1),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_fold_1x36x256",
        _case_fold_setup,
        lambda module, input: module.nn.functional.fold(input, (16, 16), (3, 3), padding=1, stride=1),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_pad_1x4x16x16",
        _case_pad_setup,
        lambda module, input: module.nn.functional.pad(input, (1, 1, 1, 1), value=-0.5),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_pad_reflect_1x4x16x16",
        _case_pad_setup,
        lambda module, input: module.nn.functional.pad(input, (1, 1, 1, 1), mode="reflect"),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_one_hot_16x16",
        _case_one_hot_setup,
        lambda module, input: module.nn.functional.one_hot(input, 16),
    ),
    BenchmarkCase(
        "bench.nn_functional_pixel_shuffle_1x4x16x16",
        _case_pad_setup,
        lambda module, input: module.nn.functional.pixel_shuffle(input, 2),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_pixel_unshuffle_1x4x16x16",
        _case_pad_setup,
        lambda module, input: module.nn.functional.pixel_unshuffle(input, 2),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_channel_shuffle_1x4x16x16",
        _case_pad_setup,
        lambda module, input: module.nn.functional.channel_shuffle(input, 2),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_cosine_similarity_64x64",
        _case_cosine_similarity_setup,
        lambda module, left, right: module.nn.functional.cosine_similarity(left, right, dim=1),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_interpolate_nearest_1x4x16x16",
        _case_interpolate_setup,
        lambda module, input: module.nn.functional.interpolate(input, scale_factor=2.0, mode="nearest"),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_interpolate_nearest_exact_1x4x16x16",
        _case_interpolate_setup,
        lambda module, input: module.nn.functional.interpolate(input, scale_factor=2.0, mode="nearest-exact"),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_interpolate_nearest_half_1x4x16x16",
        _case_interpolate_half_setup,
        lambda module, input: module.nn.functional.interpolate(input, scale_factor=2.0, mode="nearest"),
        rtol=1e-3,
        atol=1e-3,
    ),
    BenchmarkCase(
        "bench.nn_functional_interpolate_bilinear_1x4x16x16",
        _case_interpolate_setup,
        lambda module, input: module.nn.functional.interpolate(input, scale_factor=2.0, mode="bilinear", align_corners=False),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.nn_functional_interpolate_bilinear_half_1x4x16x16",
        _case_interpolate_half_setup,
        lambda module, input: module.nn.functional.interpolate(input, scale_factor=2.0, mode="bilinear", align_corners=False),
        rtol=2e-3,
        atol=2e-3,
    ),
    BenchmarkCase(
        "bench.nn_functional_interpolate_bicubic_1x4x16x16",
        _case_interpolate_setup,
        lambda module, input: module.nn.functional.interpolate(input, scale_factor=2.0, mode="bicubic", align_corners=False),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.nn_functional_interpolate_bicubic_half_1x4x16x16",
        _case_interpolate_half_setup,
        lambda module, input: module.nn.functional.interpolate(input, scale_factor=2.0, mode="bicubic", align_corners=False),
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.nn_functional_interpolate_area_1x4x16x16",
        _case_interpolate_setup,
        lambda module, input: module.nn.functional.interpolate(input, size=(8, 8), mode="area"),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_grid_sample_bilinear_1x4x16x16",
        _case_grid_sample_setup,
        lambda module, input, grid: module.nn.functional.grid_sample(input, grid, align_corners=False),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.nn_functional_affine_grid_1x4x16x16",
        _case_affine_grid_setup,
        lambda module, theta: module.nn.functional.affine_grid(theta, (1, 4, 16, 16), align_corners=False),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_sdpa_1x2x16x16",
        _case_sdpa_setup,
        lambda module, query, key, value: module.nn.functional.scaled_dot_product_attention(query, key, value),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_clip_causal_attention_1x2x77x64",
        _case_stable_diffusion_clip_causal_attention_setup,
        lambda module, query, key, value: module.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        ),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_clip_causal_attention_half_1x2x77x64",
        _case_stable_diffusion_clip_causal_attention_half_setup,
        lambda module, query, key, value: module.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        ),
        rtol=1e-3,
        atol=1e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_additive_masked_attention_1x2x64x64",
        _case_stable_diffusion_half_additive_masked_attention_setup,
        lambda module, query, key, value, mask: module.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
        ),
        rtol=1e-3,
        atol=1e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_dtype_masked_attention_1x2x64x64",
        _case_stable_diffusion_half_dtype_masked_attention_setup,
        lambda module, query, key, value, mask: module.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
        ),
        rtol=1e-3,
        atol=1e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_bool_masked_attention_1x2x64x64",
        _case_stable_diffusion_half_bool_masked_attention_setup,
        lambda module, query, key, value, mask: module.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
        ),
        rtol=1e-3,
        atol=1e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_rank3_cross_attention_16x64x77x40",
        _case_stable_diffusion_half_rank3_cross_attention_setup,
        _stable_diffusion_half_rank3_cross_attention_path,
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_legacy_baddbmm_attention_16x64x77x40",
        _case_stable_diffusion_half_rank3_cross_attention_setup,
        _stable_diffusion_half_legacy_baddbmm_attention_path,
        rtol=4e-3,
        atol=4e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_upcast_baddbmm_attention_16x64x77x40",
        _case_stable_diffusion_half_rank3_cross_attention_setup,
        _stable_diffusion_half_upcast_baddbmm_attention_path,
        rtol=4e-3,
        atol=4e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_unet_cross_attention_2x64x77x320",
        _case_stable_diffusion_half_unet_cross_attention_setup,
        _stable_diffusion_half_unet_cross_attention_path,
        rtol=4e-3,
        atol=4e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sdxl_unet_cross_attention_half_2x64x77x2048",
        _case_stable_diffusion_sdxl_unet_cross_attention_setup,
        _stable_diffusion_sdxl_unet_cross_attention_path,
        rtol=6e-3,
        atol=6e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sdxl_ip_adapter_cross_attention_half_2x64x77x2048",
        _case_stable_diffusion_sdxl_ip_adapter_cross_attention_setup,
        _stable_diffusion_sdxl_ip_adapter_cross_attention_path,
        rtol=8e-3,
        atol=7e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_joint_transformer_block_half_2x64x77x320",
        _case_stable_diffusion_sd3_joint_transformer_block_setup,
        _stable_diffusion_sd3_joint_transformer_block_path,
        rtol=8e-3,
        atol=8e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_qk_norm_joint_attention_half_2x64x77x320",
        _case_stable_diffusion_sd3_qk_norm_joint_attention_setup,
        _stable_diffusion_sd3_qk_norm_joint_attention_path,
        rtol=8e-3,
        atol=8e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rotary_joint_attention_half_2x64x77x320",
        _case_stable_diffusion_sd3_rotary_joint_attention_setup,
        _stable_diffusion_sd3_rotary_joint_attention_path,
        rtol=8e-3,
        atol=8e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_rotary_joint_attention_half_2x96x77x320",
        _case_stable_diffusion_sd3_rectangular_rotary_joint_attention_setup,
        _stable_diffusion_sd3_rectangular_rotary_joint_attention_path,
        rtol=8e-3,
        atol=8e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_single_transformer_block_half_2x64x77x320",
        _case_stable_diffusion_sd3_single_transformer_block_setup,
        _stable_diffusion_sd3_single_transformer_block_path,
        rtol=8e-3,
        atol=8e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_time_text_conditioning_half_4x2048x1280",
        _case_stable_diffusion_sd3_time_text_conditioning_setup,
        _stable_diffusion_sd3_time_text_conditioning_path,
        rtol=5e-3,
        atol=5e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_unpatchify_projection_half_2x64x320",
        _case_stable_diffusion_sd3_unpatchify_projection_setup,
        _stable_diffusion_sd3_unpatchify_projection_path,
        rtol=5e-3,
        atol=5e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_large_unpatchify_projection_half_2x256x320",
        _case_stable_diffusion_sd3_large_unpatchify_projection_setup,
        _stable_diffusion_sd3_large_unpatchify_projection_path,
        rtol=5e-3,
        atol=5e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_unpatchify_projection_half_2x96x320",
        _case_stable_diffusion_sd3_rectangular_unpatchify_projection_setup,
        _stable_diffusion_sd3_rectangular_unpatchify_projection_path,
        rtol=5e-3,
        atol=5e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_patch_embed_half_channels_last_2x16x32x32",
        _case_stable_diffusion_sd3_patch_embed_setup,
        _stable_diffusion_sd3_patch_embed_path,
        rtol=5e-3,
        atol=5e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_patch_embed_half_channels_last_2x16x16x24",
        _case_stable_diffusion_sd3_rectangular_patch_embed_setup,
        _stable_diffusion_sd3_patch_embed_path,
        rtol=5e-3,
        atol=5e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_mini_transformer_stack_half_2x16x16x16",
        _case_stable_diffusion_sd3_mini_transformer_stack_setup,
        _stable_diffusion_sd3_mini_transformer_stack_path,
        rtol=1e-2,
        atol=1e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_transformer_stack_half_2x16x16x24",
        _case_stable_diffusion_sd3_rectangular_transformer_stack_setup,
        _stable_diffusion_sd3_rectangular_transformer_stack_path,
        rtol=1e-2,
        atol=1e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_pooled_transformer_stack_half_2x16x16x16",
        _case_stable_diffusion_sd3_pooled_transformer_stack_setup,
        _stable_diffusion_sd3_pooled_transformer_stack_path,
        rtol=1e-2,
        atol=1e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_large_controlnet_transformer_stack_half_2x16x32x32",
        _case_stable_diffusion_sd3_large_controlnet_transformer_stack_setup,
        _stable_diffusion_sd3_large_controlnet_transformer_stack_path,
        rtol=1e-2,
        atol=1e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_multi_controlnet_transformer_stack_half_2x16x32x32",
        _case_stable_diffusion_sd3_multi_controlnet_transformer_stack_setup,
        _stable_diffusion_sd3_multi_controlnet_transformer_stack_path,
        rtol=1e-2,
        atol=1e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_half_2x16x32x32",
        _case_stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_setup,
        _stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_path,
        rtol=1e-2,
        atol=1e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_half_2x16x16x24",
        _case_stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_setup,
        _stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_path,
        rtol=1e-2,
        atol=1e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_half_2x16x16x24",
        _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_setup,
        _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_path,
        rtol=1e-2,
        atol=1e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_half_2x16x16x24",
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_setup,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_path,
        rtol=1e-2,
        atol=1e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_half_2x16x16x24",
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_path,
        rtol=1e-2,
        atol=1e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_flowmatch_cfg_step_half_2x16x16x16",
        _case_stable_diffusion_sd3_flowmatch_cfg_step_setup,
        _stable_diffusion_sd3_flowmatch_cfg_step_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_pooled_controlnet_denoising_step_half_1x16x32x32",
        _case_stable_diffusion_sd3_pooled_controlnet_denoising_step_setup,
        _stable_diffusion_sd3_pooled_controlnet_denoising_step_path,
        rtol=1e-2,
        atol=1e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_half_1x16x16x24_2step",
        _case_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path,
        rtol=2e-2,
        atol=2e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_half_1x16x16x24_4step",
        _case_stable_diffusion_sd3_rectangular_pooled_controlnet_long_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_half_1x16x16x24_strength060_3step",
        _case_stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop_half_1x16x16x24_4step",
        _case_stable_diffusion_sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_half_1x16x16x24_4step",
        _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_half_1x16x16x24_strength060_3step",
        _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop_half_1x16x16x24_4step",
        _case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_half_1x16x16x24_4step",
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_half_1x16x16x24_4step",
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_zero_width_windowed_denoising_loop_half_1x16x16x24_4step",
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_zero_width_windowed_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_disabled_windowed_denoising_loop_half_1x16x16x24_4step",
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_disabled_windowed_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_half_1x16x16x24_4step",
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_half_1x16x16x24_4step",
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_zero_width_windowed_denoising_loop_half_1x16x16x24_4step",
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_zero_width_windowed_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_disabled_windowed_denoising_loop_half_1x16x16x24_4step",
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_disabled_windowed_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_half_1x16x16x24_strength060_3step",
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_half_1x16x16x24_strength060_3step",
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_zero_width_windowed_img2img_denoising_loop_half_1x16x16x24_strength060_3step",
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_zero_width_windowed_img2img_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_disabled_windowed_img2img_denoising_loop_half_1x16x16x24_strength060_3step",
        _case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_disabled_windowed_img2img_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_half_1x16x16x24_strength060_3step",
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_half_1x16x16x24_strength060_3step",
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_zero_width_windowed_img2img_denoising_loop_half_1x16x16x24_strength060_3step",
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_zero_width_windowed_img2img_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_disabled_windowed_img2img_denoising_loop_half_1x16x16x24_strength060_3step",
        _case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_disabled_windowed_img2img_denoising_loop_setup,
        _stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path,
        rtol=3e-2,
        atol=3e-1,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_unet_group_norm_1x320x16x16",
        _case_stable_diffusion_half_unet_group_norm_setup,
        _stable_diffusion_half_unet_group_norm_path,
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_group_norm_1x320x16x16",
        _case_stable_diffusion_half_channels_last_group_norm_setup,
        _stable_diffusion_half_unet_group_norm_path,
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_silu_1x320x16x16",
        _case_stable_diffusion_half_channels_last_silu_setup,
        _stable_diffusion_half_channels_last_silu_path,
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_timestep_add_silu_1x320x16x16",
        _case_stable_diffusion_half_channels_last_timestep_add_setup,
        _stable_diffusion_half_channels_last_timestep_add_path,
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_resnet_scale_shift_1x320x16x16",
        _case_stable_diffusion_half_channels_last_resnet_scale_shift_setup,
        _stable_diffusion_half_channels_last_resnet_scale_shift_path,
        rtol=1e-2,
        atol=8e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_classifier_free_guidance_2x4x64x64",
        _case_stable_diffusion_half_channels_last_classifier_free_guidance_setup,
        _stable_diffusion_half_channels_last_classifier_free_guidance_path,
        rtol=4e-3,
        atol=4e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_scheduler_scale_2x4x64x64",
        _case_stable_diffusion_half_channels_last_scheduler_scale_setup,
        _stable_diffusion_half_channels_last_scheduler_scale_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_scheduler_add_noise_2x4x64x64",
        _case_stable_diffusion_half_channels_last_scheduler_add_noise_setup,
        _stable_diffusion_half_channels_last_scheduler_add_noise_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_mask_blend_2x4x64x64",
        _case_stable_diffusion_half_channels_last_mask_blend_setup,
        _stable_diffusion_half_channels_last_mask_blend_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_controlnet_merge",
        _case_stable_diffusion_half_channels_last_controlnet_merge_setup,
        _stable_diffusion_half_channels_last_controlnet_merge_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_3d_timestep_add_silu_1x320x4x8x8",
        _case_stable_diffusion_half_channels_last_3d_timestep_add_setup,
        _stable_diffusion_half_channels_last_3d_timestep_add_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_3d_classifier_free_guidance_2x4x4x32x32",
        _case_stable_diffusion_half_channels_last_3d_classifier_free_guidance_setup,
        _stable_diffusion_half_channels_last_3d_classifier_free_guidance_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_3d_scheduler_scale_2x4x4x32x32",
        _case_stable_diffusion_half_channels_last_3d_scheduler_scale_setup,
        _stable_diffusion_half_channels_last_3d_scheduler_scale_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_3d_scheduler_add_noise_2x4x4x32x32",
        _case_stable_diffusion_half_channels_last_3d_scheduler_add_noise_setup,
        _stable_diffusion_half_channels_last_3d_scheduler_add_noise_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_3d_mask_blend_2x4x4x32x32",
        _case_stable_diffusion_half_channels_last_3d_mask_blend_setup,
        _stable_diffusion_half_channels_last_3d_mask_blend_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_3d_video_upsample_1x320x4x8x8",
        _case_stable_diffusion_half_channels_last_3d_video_upsample_setup,
        _stable_diffusion_half_channels_last_3d_video_upsample_path,
        rtol=0.0,
        atol=0.0,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_video_upsample_1x320x4x8x8",
        _case_stable_diffusion_half_video_upsample_setup,
        _stable_diffusion_half_video_upsample_path,
        rtol=0.0,
        atol=0.0,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_conv2d_1x4x16x16",
        _case_stable_diffusion_half_channels_last_conv_setup,
        _stable_diffusion_half_channels_last_conv_path,
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_projection_conv2d_1x320x16x16",
        _case_stable_diffusion_half_channels_last_projection_conv_setup,
        _stable_diffusion_half_channels_last_projection_conv_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_wide_conv2d_1x320x16x16",
        _case_stable_diffusion_half_channels_last_wide_conv_setup,
        _stable_diffusion_half_channels_last_wide_conv_path,
        rtol=8e-3,
        atol=8e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_lora_conv2d_1x320x16x16",
        _case_stable_diffusion_half_channels_last_lora_conv_setup,
        _stable_diffusion_half_channels_last_lora_conv_path,
        rtol=8e-3,
        atol=8e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_repeated_wide_conv2d_1x320x16x16",
        _case_stable_diffusion_half_channels_last_wide_conv_setup,
        _stable_diffusion_half_channels_last_repeated_wide_conv_path,
        rtol=1e-2,
        atol=1e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_downsample_conv2d_1x320x32x32",
        _case_stable_diffusion_half_channels_last_downsample_conv_setup,
        _stable_diffusion_half_channels_last_downsample_conv_path,
        rtol=8e-3,
        atol=8e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_upsample_conv2d_1x320x16x16",
        _case_stable_diffusion_half_channels_last_upsample_conv_setup,
        _stable_diffusion_half_channels_last_upsample_conv_path,
        rtol=8e-3,
        atol=8e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_skip_cat_conv2d_1x320x16x16",
        _case_stable_diffusion_half_channels_last_skip_cat_conv_setup,
        _stable_diffusion_half_channels_last_skip_cat_conv_path,
        rtol=8e-3,
        atol=8e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_attention_projection_2x4x77x16",
        _case_stable_diffusion_half_attention_projection_setup,
        _stable_diffusion_half_attention_projection_path,
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_clip_text_encoder_2x77x64",
        _case_stable_diffusion_half_clip_text_encoder_setup,
        _stable_diffusion_half_clip_text_encoder_path,
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_clip_layer_norm_2x77x768",
        _case_layer_norm_half_768_setup,
        lambda module, tensor, weight, bias: module.nn.functional.layer_norm(tensor, (768,), weight, bias, eps=1e-5),
        rtol=4e-3,
        atol=4e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_clip_quick_gelu_2x77x768",
        _case_quick_gelu_half_768_setup,
        lambda module, tensor: tensor * module.sigmoid(tensor * 1.702),
        rtol=4e-3,
        atol=4e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_clip_mlp_2x77x768",
        _case_stable_diffusion_half_clip_mlp_setup,
        _stable_diffusion_half_clip_mlp_path,
        rtol=4e-3,
        atol=4e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_clip_attention_projection_2x77x768",
        _case_stable_diffusion_half_clip_attention_large_setup,
        _stable_diffusion_half_clip_attention_large_path,
        rtol=4e-3,
        atol=4e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_lora_attention_projection_2x77x768",
        _case_stable_diffusion_half_lora_attention_projection_setup,
        _stable_diffusion_half_lora_attention_projection_path,
        rtol=5e-3,
        atol=4e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_clip_transformer_block_2x77x768",
        _case_stable_diffusion_half_clip_transformer_block_setup,
        _stable_diffusion_half_clip_transformer_block_path,
        rtol=5e-3,
        atol=3e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_clip_large_text_encoder_stack_2x77x768",
        _case_stable_diffusion_half_clip_large_text_encoder_stack_setup,
        _stable_diffusion_half_clip_large_text_encoder_stack_path,
        rtol=8e-3,
        atol=6e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_clip_pooled_projection_half_2x77x64",
        _case_stable_diffusion_clip_pooled_projection_setup,
        _stable_diffusion_clip_pooled_projection_path,
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sdxl_dual_prompt_encode_half_2x77x2048",
        _case_stable_diffusion_sdxl_dual_prompt_encode_setup,
        _stable_diffusion_sdxl_dual_prompt_encode_path,
        rtol=8e-3,
        atol=6e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sdxl_text_encoder2_stack_half_2x77x1280",
        _case_stable_diffusion_sdxl_text_encoder2_stack_setup,
        _stable_diffusion_sdxl_text_encoder2_stack_path,
        rtol=8e-3,
        atol=6e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sdxl_add_time_conditioning_half_8x2816x1280",
        _case_stable_diffusion_sdxl_add_time_conditioning_setup,
        _stable_diffusion_sdxl_add_time_conditioning_path,
        rtol=5e-3,
        atol=5e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sdxl_prompt_conditioning_bundle_half_4x77x2048",
        _case_stable_diffusion_sdxl_prompt_conditioning_bundle_setup,
        _stable_diffusion_sdxl_prompt_conditioning_bundle_path,
        rtol=8e-3,
        atol=6e-2,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_prompt_embedding_repeat_half_2x77x320",
        _case_stable_diffusion_prompt_embedding_repeat_setup,
        _stable_diffusion_prompt_embedding_repeat_path,
        rtol=1e-3,
        atol=1e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_timestep_embedding_mlp_half_2x320x1280",
        _case_stable_diffusion_timestep_embedding_mlp_setup,
        _stable_diffusion_timestep_embedding_mlp_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_masked_attention_index_put_1x2x16x16",
        _case_stable_diffusion_masked_attention_setup,
        _stable_diffusion_masked_attention_path,
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_bool_masked_attention_index_put_1x2x16x16",
        _case_stable_diffusion_bool_masked_attention_setup,
        _stable_diffusion_bool_masked_attention_path,
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_dynamic_thresholding_quantile_2x4x8x8",
        _case_stable_diffusion_dynamic_thresholding_setup,
        _stable_diffusion_dynamic_thresholding_path,
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_dynamic_thresholding_quantile_2x4x64x64",
        _case_stable_diffusion_dynamic_thresholding_large_setup,
        _stable_diffusion_dynamic_thresholding_path,
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_guidance_rescale_std_mean_2x4x8x8",
        _case_stable_diffusion_guidance_rescale_setup,
        _stable_diffusion_guidance_rescale_path,
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_guidance_rescale_tail_std_2x4x8x8",
        _case_stable_diffusion_guidance_rescale_setup,
        _stable_diffusion_guidance_rescale_tail_std_path,
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_guidance_rescale_2x4x64x64",
        _case_stable_diffusion_half_channels_last_guidance_rescale_setup,
        _stable_diffusion_half_channels_last_guidance_rescale_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_batched_timestep_table_unique_dim1_2x128x2",
        _case_stable_diffusion_batched_timestep_table_setup,
        _stable_diffusion_batched_timestep_table_path,
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_scheduler_cumextrema_2x4x4x4",
        _case_stable_diffusion_scheduler_cumextrema_setup,
        _stable_diffusion_scheduler_cumextrema_path,
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_timestep_bincount_1024",
        _case_stable_diffusion_timestep_bincount_setup,
        _stable_diffusion_timestep_bincount_path,
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_scheduler_sigma_stack_1024",
        _case_stable_diffusion_scheduler_sigma_stack_setup,
        _stable_diffusion_scheduler_sigma_stack_path,
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_ddim_scheduler_step_2x4x8x8",
        _case_stable_diffusion_ddim_scheduler_step_setup,
        _stable_diffusion_ddim_scheduler_step_path,
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_karras_euler_scheduler_step_2x4x8x8",
        _case_stable_diffusion_karras_euler_scheduler_step_setup,
        _stable_diffusion_karras_euler_scheduler_step_path,
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_euler_ancestral_scheduler_step_2x4x8x8",
        _case_stable_diffusion_euler_ancestral_scheduler_step_setup,
        _stable_diffusion_euler_ancestral_scheduler_step_path,
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_dpmpp_2m_scheduler_step_2x4x64x64",
        _case_stable_diffusion_half_channels_last_dpmpp_2m_scheduler_step_setup,
        _stable_diffusion_half_channels_last_dpmpp_2m_scheduler_step_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_dpmpp_sde_scheduler_step_2x4x64x64",
        _case_stable_diffusion_half_channels_last_dpmpp_sde_scheduler_step_setup,
        _stable_diffusion_half_channels_last_dpmpp_sde_scheduler_step_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_vae_path_1x3x8x8",
        _case_stable_diffusion_vae_setup,
        _stable_diffusion_vae_path,
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_vae_posterior_sample_1x8x64x64",
        _case_stable_diffusion_half_channels_last_vae_posterior_sample_setup,
        _stable_diffusion_half_channels_last_vae_posterior_sample_path,
        rtol=4e-3,
        atol=4e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_vae_encode_down_block_1x3x16x16",
        _case_stable_diffusion_half_vae_encode_down_block_setup,
        _stable_diffusion_half_vae_encode_down_block_path,
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_vae_attention_block_1x16x8x8",
        _case_stable_diffusion_half_vae_attention_block_setup,
        _stable_diffusion_half_vae_attention_block_path,
        rtol=4e-3,
        atol=4e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_video_conv3d_block_1x4x4x8x8",
        _case_stable_diffusion_half_video_conv3d_block_setup,
        _stable_diffusion_half_video_conv3d_block_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_3d_video_conv3d_block_1x4x4x8x8",
        _case_stable_diffusion_half_channels_last_3d_video_conv3d_block_setup,
        _stable_diffusion_half_video_conv3d_block_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_3d_skip_cat_conv3d_1x4x4x8x8",
        _case_stable_diffusion_half_channels_last_3d_skip_cat_conv3d_setup,
        _stable_diffusion_half_channels_last_3d_skip_cat_conv3d_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_vae_nearest_upsample_1x8x16x16",
        _case_stable_diffusion_half_vae_nearest_upsample_setup,
        _stable_diffusion_half_vae_nearest_upsample_path,
        rtol=0.0,
        atol=0.0,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_vae_decode_1x4x2x2",
        _case_stable_diffusion_half_vae_decode_setup,
        _stable_diffusion_half_vae_decode_path,
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_vae_decode_1x4x8x8",
        _case_stable_diffusion_half_vae_decode_8x8_setup,
        _stable_diffusion_half_vae_decode_path,
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_vae_transpose_decode_1x4x8x8",
        _case_stable_diffusion_half_channels_last_vae_transpose_decode_setup,
        _stable_diffusion_half_channels_last_vae_transpose_decode_path,
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_vae_nearest_block_1x4x8x8",
        _case_stable_diffusion_half_vae_nearest_block_setup,
        _stable_diffusion_half_vae_nearest_block_path,
        rtol=3e-3,
        atol=3e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_vae_decode_block_1x16x16x16",
        _case_stable_diffusion_half_channels_last_vae_decode_block_setup,
        _stable_diffusion_half_channels_last_vae_decode_block_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_bicubic_postprocess_1x3x16x16",
        _case_stable_diffusion_bicubic_postprocess_setup,
        _stable_diffusion_bicubic_postprocess_path,
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_area_preprocess_1x3x16x16",
        _case_stable_diffusion_area_preprocess_setup,
        _stable_diffusion_area_preprocess_path,
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_constant_pad_crop_1x4x16x16",
        _case_stable_diffusion_constant_pad_crop_setup,
        _stable_diffusion_constant_pad_crop_path,
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_mask_nearest_exact_1x1x32x32",
        _case_stable_diffusion_mask_preprocess_setup,
        _stable_diffusion_mask_nearest_exact_path,
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_inpaint_preprocess_bundle_half_channels_last_1x4x16x16",
        _case_stable_diffusion_inpaint_preprocess_bundle_setup,
        _stable_diffusion_inpaint_preprocess_bundle_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_affine_grid_sample_mask_1x4x16x16",
        _case_stable_diffusion_spatial_transform_setup,
        _stable_diffusion_spatial_transform_path,
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_module_pipeline_1x4x4x4",
        _case_stable_diffusion_module_pipeline_setup,
        _stable_diffusion_module_pipeline_path,
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_module_pipeline_1x4x4x4",
        _case_stable_diffusion_half_module_pipeline_setup,
        _stable_diffusion_module_pipeline_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_module_pipeline_1x4x4x4",
        _case_stable_diffusion_half_channels_last_module_pipeline_setup,
        _stable_diffusion_module_pipeline_path,
        rtol=5e-3,
        atol=5e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_half_channels_last_inpaint_module_pipeline_1x4x4x4",
        _case_stable_diffusion_half_channels_last_inpaint_module_pipeline_setup,
        _stable_diffusion_inpaint_module_pipeline_path,
        rtol=6e-3,
        atol=6e-3,
    ),
    BenchmarkCase(
        "bench.stable_diffusion_sdxl_half_channels_last_inpaint_controlnet_module_pipeline_1x4x4x4",
        _case_stable_diffusion_sdxl_half_channels_last_inpaint_controlnet_module_pipeline_setup,
        _stable_diffusion_sdxl_inpaint_controlnet_module_pipeline_path,
        rtol=8e-3,
        atol=8e-3,
    ),
    BenchmarkCase(
        "bench.nn_functional_sdpa_dropout_one_1x2x16x16",
        _case_sdpa_setup,
        lambda module, query, key, value: module.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=1.0),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.einsum_attention_scores_1x2x16x16",
        _case_sdpa_setup,
        lambda module, query, key, value: module.einsum("bhid,bhjd->bhij", query, key),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.einsum_attention_scores_generic_labels_1x2x16x16",
        _case_sdpa_setup,
        lambda module, query, key, value: module.einsum("bhqd,bhkd->bhqk", query, key),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.nn_functional_sdpa_gqa_1x4x16x16",
        _case_sdpa_gqa_setup,
        lambda module, query, key, value: module.nn.functional.scaled_dot_product_attention(query, key, value, enable_gqa=True),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_functional_embedding_32x16",
        _case_embedding_setup,
        lambda module, input, weight: module.nn.functional.embedding(input, weight),
    ),
    BenchmarkCase(
        "bench.nn_functional_embedding_max_norm_32x16",
        _case_embedding_max_norm_setup,
        lambda module, input, weight: module.nn.functional.embedding(input, weight, max_norm=1.0, norm_type=2.0),
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.nn_linear_module_64x64",
        _case_nn_linear_module_setup,
        lambda module, layer, input: layer(input),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_multihead_attention_1x8x32",
        _case_nn_multihead_attention_setup,
        lambda module, layer, input: layer(input, input, input, need_weights=False)[0],
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_multihead_attention_key_padding_mask_1x8x32",
        _case_nn_multihead_attention_mask_setup,
        lambda module, layer, input, mask: layer(input, input, input, key_padding_mask=mask, need_weights=False)[0],
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_multihead_attention_cross_1x8x32",
        _case_nn_multihead_attention_cross_setup,
        lambda module, layer, query, key, value: layer(query, key, value, need_weights=False)[0],
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_multihead_attention_kv_1x8x32",
        _case_nn_multihead_attention_kv_setup,
        lambda module, layer, query, key, value: layer(query, key, value, need_weights=False)[0],
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_sequential_linear_relu_64x64",
        _case_nn_sequential_setup,
        lambda module, model, input: model(input),
        rtol=1e-4,
        atol=1e-4,
    ),
    BenchmarkCase(
        "bench.nn_functional_mse_loss_64x64",
        _case_loss_setup,
        lambda module, input, target: module.nn.functional.mse_loss(input, target),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_l1_loss_64x64",
        _case_loss_setup,
        lambda module, input, target: module.nn.functional.l1_loss(input, target),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_nll_loss_64x64",
        _case_nll_loss_setup,
        lambda module, input, target: module.nn.functional.nll_loss(input, target),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_nll_loss_weight_64x64",
        _case_nll_loss_weight_setup,
        lambda module, input, target, weight: module.nn.functional.nll_loss(input, target, weight=weight),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_cross_entropy_64x64",
        _case_cross_entropy_setup,
        lambda module, input, target: module.nn.functional.cross_entropy(input, target),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_cross_entropy_label_smoothing_64x64",
        _case_cross_entropy_setup,
        lambda module, input, target: module.nn.functional.cross_entropy(input, target, label_smoothing=0.2),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_cross_entropy_weight_64x64",
        _case_cross_entropy_weight_setup,
        lambda module, input, target, weight: module.nn.functional.cross_entropy(input, target, weight=weight),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_cross_entropy_weight_label_smoothing_64x64",
        _case_cross_entropy_weight_setup,
        lambda module, input, target, weight: module.nn.functional.cross_entropy(
            input,
            target,
            weight=weight,
            label_smoothing=0.2,
        ),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_bce_64x64",
        _case_bce_setup,
        lambda module, input, target: module.nn.functional.binary_cross_entropy(input, target),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_bce_weight_64x64",
        _case_bce_weight_setup,
        lambda module, input, target, weight: module.nn.functional.binary_cross_entropy(input, target, weight=weight),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_bce_with_logits_64x64",
        _case_bce_with_logits_setup,
        lambda module, input, target: module.nn.functional.binary_cross_entropy_with_logits(input, target),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.nn_functional_bce_with_logits_weight_pos_64x64",
        _case_bce_with_logits_weight_pos_setup,
        lambda module, input, target, weight, pos_weight: module.nn.functional.binary_cross_entropy_with_logits(
            input,
            target,
            weight=weight,
            pos_weight=pos_weight,
        ),
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.autograd_grad_square_32x32",
        _case_autograd_grad_setup,
        lambda module, tensor: module.autograd.grad((tensor * tensor).sum(), tensor)[0],
        rtol=1e-6,
        atol=1e-6,
    ),
    BenchmarkCase(
        "bench.optim_sgd_square_step_32x32",
        _case_optim_sgd_setup,
        _sgd_training_step,
        rtol=1e-5,
        atol=1e-5,
    ),
    BenchmarkCase(
        "bench.transpose_contiguous_96x64",
        _case_transpose_contiguous_setup,
        lambda module, tensor: module.transpose(tensor, 0, 1).contiguous(),
    ),
    BenchmarkCase(
        "bench.permute_contiguous_24x32x16",
        _case_permute_contiguous_setup,
        lambda module, tensor: module.permute(tensor, (2, 0, 1)).contiguous(),
    ),
    BenchmarkCase(
        "bench.cast_float64_64x64",
        _case_cast_setup,
        lambda module, tensor: tensor.double(),
    ),
    BenchmarkCase(
        "bench.tensor_new_zeros_64x64",
        _case_new_factory_setup,
        lambda module, tensor: tensor.new_zeros((64, 64)),
    ),
    BenchmarkCase(
        "bench.tensor_new_full_64x64",
        _case_new_factory_setup,
        lambda module, tensor: tensor.new_full((64, 64), -3.0),
    ),
    BenchmarkCase(
        "bench.tensor_type_as_64x64",
        _case_type_as_setup,
        lambda module, tensor, other: tensor.type_as(other),
    ),
    BenchmarkCase(
        "bench.expand_contiguous_64x64",
        _case_expand_contiguous_setup,
        lambda module, tensor: module.broadcast_to(tensor, (64, 64)).contiguous(),
    ),
    BenchmarkCase(
        "bench.repeat_vector_64x64",
        _case_expand_contiguous_setup,
        lambda module, tensor: tensor.repeat(64, 1),
    ),
    BenchmarkCase(
        "bench.repeat_interleave_64x64",
        _case_cast_setup,
        lambda module, tensor: module.repeat_interleave(tensor, 2, dim=1),
    ),
    BenchmarkCase(
        "bench.flip_dim1_64x64",
        _case_cast_setup,
        lambda module, tensor: module.flip(tensor, (1,)),
    ),
    BenchmarkCase(
        "bench.stable_diffusion_flip_latent_lastdim_2x4x8x8",
        _case_stable_diffusion_latent_flip_setup,
        lambda module, tensor: module.flip(tensor, (-1,)),
    ),
    BenchmarkCase(
        "bench.rot90_64x64",
        _case_cast_setup,
        lambda module, tensor: module.rot90(tensor, 1, (0, 1)),
    ),
    BenchmarkCase(
        "bench.roll_dim1_64x64",
        _case_cast_setup,
        lambda module, tensor: module.roll(tensor, 7, dims=1),
    ),
    BenchmarkCase(
        "bench.cat_dim0_64x64",
        _case_binary_add_setup,
        lambda module, left, right: module.cat([left, right], dim=0),
    ),
    BenchmarkCase(
        "bench.stack_dim0_64x64",
        _case_binary_add_setup,
        lambda module, left, right: module.stack([left, right], dim=0),
    ),
    BenchmarkCase(
        "bench.unflatten_64x64",
        _case_cast_setup,
        lambda module, tensor: module.unflatten(tensor, 1, (8, 8)),
    ),
    BenchmarkCase(
        "bench.is_signed_64x64",
        _case_cast_setup,
        lambda module, tensor: module.is_signed(tensor),
    ),
    BenchmarkCase(
        "bench.atleast_2d_1024",
        _case_dot_setup,
        lambda module, left, right: module.atleast_2d(left),
    ),
    BenchmarkCase(
        "bench.hstack_64x64",
        _case_binary_add_setup,
        lambda module, left, right: module.hstack([left, right]),
    ),
    BenchmarkCase(
        "bench.vstack_64x64",
        _case_binary_add_setup,
        lambda module, left, right: module.vstack([left, right]),
    ),
    BenchmarkCase(
        "bench.dstack_64x64",
        _case_binary_add_setup,
        lambda module, left, right: module.dstack([left, right]),
    ),
    BenchmarkCase(
        "bench.column_stack_1024",
        _case_dot_setup,
        lambda module, left, right: module.column_stack([left, right]),
    ),
    BenchmarkCase(
        "bench.meshgrid_1024",
        _case_dot_setup,
        lambda module, left, right: module.meshgrid(left, right, indexing="ij"),
    ),
    BenchmarkCase(
        "bench.cartesian_prod_32x32",
        _case_cartesian_prod_setup,
        lambda module, left, right: module.cartesian_prod(left, right),
    ),
    BenchmarkCase(
        "bench.narrow_contiguous_32x64",
        _case_view_copy_setup,
        lambda module, tensor: module.narrow(tensor, 0, 16, 32).contiguous(),
    ),
    BenchmarkCase(
        "bench.select_contiguous_64",
        _case_view_copy_setup,
        lambda module, tensor: tensor.select(0, 32).contiguous(),
    ),
    BenchmarkCase(
        "bench.bool_mask_64x64",
        _case_bool_mask_setup,
        lambda module, tensor, mask: tensor[mask],
    ),
    BenchmarkCase(
        "bench.where_64x64",
        _case_where_setup,
        lambda module, condition, left, right: module.where(condition, left, right),
    ),
    BenchmarkCase(
        "bench.mixed_int_slice_8x32",
        _case_mixed_int_slice_setup,
        lambda module, tensor, rows: tensor[rows, 16:48],
    ),
    BenchmarkCase(
        "bench.nonleading_int_columns_64x8",
        _case_nonleading_int_columns_setup,
        lambda module, tensor, cols: tensor[:, cols],
    ),
    BenchmarkCase(
        "bench.setitem_nonleading_int_columns_64x8",
        _case_setitem_nonleading_int_columns_setup,
        _assign_nonleading_int_columns,
    ),
    BenchmarkCase(
        "bench.copy_float32_64x64",
        _case_copy_setup,
        lambda module, target, source: target.copy_(source),
    ),
    BenchmarkCase(
        "bench.fill_inplace_float32_64x64",
        _case_inplace_setup,
        lambda module, tensor: tensor.fill_(1.25),
    ),
    BenchmarkCase(
        "bench.masked_fill_inplace_float32_64x64",
        _case_masked_fill_inplace_setup,
        lambda module, tensor, mask: tensor.masked_fill_(mask, -7.0),
    ),
    BenchmarkCase(
        "bench.add_inplace_float32_64x64",
        _case_inplace_setup,
        lambda module, tensor: tensor.add_(0.125),
    ),
    BenchmarkCase(
        "bench.add_inplace_operator_float32_64x64",
        _case_inplace_setup,
        _iadd_scalar,
    ),
    BenchmarkCase(
        "bench.sub_inplace_float32_64x64",
        _case_inplace_setup,
        lambda module, tensor: tensor.sub_(0.125),
    ),
    BenchmarkCase(
        "bench.mul_inplace_float32_64x64",
        _case_inplace_setup,
        lambda module, tensor: tensor.mul_(1.0001),
    ),
    BenchmarkCase(
        "bench.div_inplace_float32_64x64",
        _case_inplace_setup,
        lambda module, tensor: tensor.div_(1.0001),
    ),
    BenchmarkCase(
        "bench.zero_inplace_float32_64x64",
        _case_inplace_setup,
        lambda module, tensor: tensor.zero_(),
    ),
)
