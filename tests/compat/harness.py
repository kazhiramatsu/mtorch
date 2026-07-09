from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import importlib
import inspect
from typing import Any, Callable
import warnings

import pytest


_MISSING = object()


@dataclass(frozen=True)
class DType:
    name: str


@dataclass(frozen=True)
class TensorSpec:
    data: Any
    dtype: str | None = "float32"
    requires_grad: bool = False


@dataclass(frozen=True)
class FactorySpec:
    target: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpCase:
    id: str
    target: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    call: str = "function"
    rtol: float | None = None
    atol: float | None = None
    check_stride: bool = True
    check_mutation: bool = True


@dataclass(frozen=True)
class GradCase:
    id: str
    inputs: tuple[TensorSpec, ...]
    expression: Callable[..., Any]
    rtol: float | None = None
    atol: float | None = None


@dataclass(frozen=True)
class ViewCase:
    id: str
    target: str
    base: TensorSpec
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    call: str = "method"
    mutation_value: float = 99.0


@dataclass
class CapturedCall:
    returned: Any = _MISSING
    args: tuple[Any, ...] = ()
    exception: BaseException | None = None


def import_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - pytest prints the chained error.
        raise AssertionError(f"Could not import module {name!r}") from exc


def resolve_path(root: Any, path: str) -> Any:
    current = root
    walked: list[str] = []
    for part in path.split("."):
        walked.append(part)
        if not hasattr(current, part):
            owner = getattr(current, "__name__", type(current).__name__)
            raise AttributeError(f"{owner!r} has no attribute {part!r} while resolving {'.'.join(walked)!r}")
        current = getattr(current, part)
    return current


def api_kind(value: Any) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if inspect.ismodule(value):
            return "module"
        if inspect.isclass(value):
            return "class"
    if callable(value):
        return "callable"
    return "value"


def build_value(module: Any, value: Any) -> Any:
    if isinstance(value, DType):
        return resolve_path(module, value.name)
    if isinstance(value, TensorSpec):
        kwargs: dict[str, Any] = {}
        if value.dtype is not None:
            kwargs["dtype"] = resolve_path(module, value.dtype)
        if value.requires_grad:
            kwargs["requires_grad"] = True
        return module.tensor(value.data, **kwargs)
    if isinstance(value, FactorySpec):
        fn = resolve_path(module, value.target)
        args = tuple(build_value(module, item) for item in value.args)
        kwargs = {key: build_value(module, item) for key, item in value.kwargs.items()}
        return fn(*args, **kwargs)
    if isinstance(value, tuple):
        return tuple(build_value(module, item) for item in value)
    if isinstance(value, list):
        return [build_value(module, item) for item in value]
    if isinstance(value, dict):
        return {key: build_value(module, item) for key, item in value.items()}
    return value


def invoke(module: Any, case: OpCase | ViewCase, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    if case.call == "function":
        return resolve_path(module, case.target)(*args, **kwargs)
    if case.call == "method":
        if not args:
            raise AssertionError(f"{case.id}: method calls require a receiver argument")
        return getattr(args[0], case.target)(*args[1:], **kwargs)
    raise AssertionError(f"{case.id}: unknown call mode {case.call!r}")


def capture_call(module: Any, case: OpCase) -> CapturedCall:
    try:
        args = tuple(build_value(module, item) for item in case.args)
        kwargs = {key: build_value(module, item) for key, item in case.kwargs.items()}
        returned = invoke(module, case, args, kwargs)
        return CapturedCall(returned=returned, args=args)
    except BaseException as exc:
        return CapturedCall(exception=exc)


def assert_same_result(reference: Any, candidate: Any, case: OpCase) -> None:
    expected = capture_call(reference, case)
    actual = capture_call(candidate, case)

    if expected.exception is not None:
        _assert_exception_compatible(case.id, expected.exception, actual.exception)
        return
    if actual.exception is not None:
        raise AssertionError(f"{case.id}: candidate raised {actual.exception!r}, reference returned normally") from actual.exception

    assert_values_compatible(
        reference,
        expected.returned,
        actual.returned,
        path=f"{case.id}.return",
        rtol=case.rtol,
        atol=case.atol,
        check_stride=case.check_stride,
    )

    if case.check_mutation:
        if len(expected.args) != len(actual.args):
            raise AssertionError(f"{case.id}: argument count changed during capture")
        for index, (expected_arg, actual_arg) in enumerate(zip(expected.args, actual.args, strict=True)):
            assert_values_compatible(
                reference,
                expected_arg,
                actual_arg,
                path=f"{case.id}.arg{index}",
                rtol=case.rtol,
                atol=case.atol,
                check_stride=False,
            )


def assert_first_order_grad(reference: Any, candidate: Any, case: GradCase) -> None:
    try:
        expected_inputs = tuple(build_value(reference, item) for item in case.inputs)
        expected_output = case.expression(reference, *expected_inputs)
        expected_output.backward()
    except BaseException as exc:
        raise AssertionError(f"{case.id}: reference grad case is invalid: {exc!r}") from exc

    try:
        actual_inputs = tuple(build_value(candidate, item) for item in case.inputs)
        actual_output = case.expression(candidate, *actual_inputs)
        actual_output.backward()
    except BaseException as exc:
        raise AssertionError(f"{case.id}: candidate failed during forward/backward: {exc!r}") from exc

    assert_values_compatible(
        reference,
        expected_output,
        actual_output,
        path=f"{case.id}.output",
        rtol=case.rtol,
        atol=case.atol,
    )

    for index, (expected_input, actual_input) in enumerate(zip(expected_inputs, actual_inputs, strict=True)):
        assert_values_compatible(
            reference,
            expected_input.grad,
            getattr(actual_input, "grad", None),
            path=f"{case.id}.input{index}.grad",
            rtol=case.rtol,
            atol=case.atol,
            check_stride=False,
        )


def assert_view_alias_parity(reference: Any, candidate: Any, case: ViewCase) -> None:
    expected_base = build_value(reference, case.base)
    actual_base = build_value(candidate, case.base)
    expected_args = (expected_base, *(build_value(reference, item) for item in case.args))
    actual_args = (actual_base, *(build_value(candidate, item) for item in case.args))
    expected_kwargs = {key: build_value(reference, item) for key, item in case.kwargs.items()}
    actual_kwargs = {key: build_value(candidate, item) for key, item in case.kwargs.items()}

    try:
        expected_view = invoke(reference, case, expected_args, expected_kwargs)
        actual_view = invoke(candidate, case, actual_args, actual_kwargs)
    except BaseException as exc:
        raise AssertionError(f"{case.id}: candidate/reference view invocation failed: {exc!r}") from exc

    assert_values_compatible(
        reference,
        expected_view,
        actual_view,
        path=f"{case.id}.view",
    )

    _mutate_tensor(expected_view, case.mutation_value)
    _mutate_tensor(actual_view, case.mutation_value)

    assert_values_compatible(
        reference,
        expected_base,
        actual_base,
        path=f"{case.id}.base_after_view_mutation",
        check_stride=False,
    )


def assert_values_compatible(
    reference: Any,
    expected: Any,
    actual: Any,
    *,
    path: str,
    rtol: float | None = None,
    atol: float | None = None,
    check_stride: bool = True,
) -> None:
    if expected is None:
        if actual is not None:
            raise AssertionError(f"{path}: expected None, got {actual!r}")
        return

    if isinstance(expected, reference.Tensor):
        _assert_tensor_compatible(reference, expected, actual, path=path, rtol=rtol, atol=atol, check_stride=check_stride)
        return

    if isinstance(expected, reference.Size):
        if tuple(expected) != tuple(actual):
            raise AssertionError(f"{path}: expected Size {tuple(expected)!r}, got {actual!r}")
        return

    if isinstance(expected, (bool, int, float, str)):
        if expected != actual:
            raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")
        return

    if isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(expected) != len(actual):
            raise AssertionError(f"{path}: expected tuple of length {len(expected)}, got {actual!r}")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            assert_values_compatible(
                reference,
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
                check_stride=check_stride,
            )
        return

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"{path}: expected list of length {len(expected)}, got {actual!r}")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            assert_values_compatible(
                reference,
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
                check_stride=check_stride,
            )
        return

    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise AssertionError(f"{path}: expected keys {set(expected)!r}, got {set(actual)!r}")
        for key in expected:
            assert_values_compatible(
                reference,
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                rtol=rtol,
                atol=atol,
                check_stride=check_stride,
            )
        return

    if _dtype_name(expected).startswith(("float", "int", "bool", "complex", "bfloat")):
        if _dtype_name(expected) != _dtype_name(actual):
            raise AssertionError(f"{path}: expected dtype-like value {expected!r}, got {actual!r}")
        return

    if expected != actual:
        raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")


def _assert_tensor_compatible(
    reference: Any,
    expected: Any,
    actual: Any,
    *,
    path: str,
    rtol: float | None,
    atol: float | None,
    check_stride: bool,
) -> None:
    if not hasattr(actual, "shape"):
        raise AssertionError(f"{path}: candidate result is not tensor-like: {actual!r}")

    expected_shape = tuple(expected.shape)
    actual_shape = tuple(actual.shape)
    if expected_shape != actual_shape:
        raise AssertionError(f"{path}: expected shape {expected_shape!r}, got {actual_shape!r}")

    expected_dtype = _dtype_name(expected.dtype)
    actual_dtype = _dtype_name(getattr(actual, "dtype", None))
    if expected_dtype != actual_dtype:
        raise AssertionError(f"{path}: expected dtype {expected_dtype!r}, got {actual_dtype!r}")

    expected_device = getattr(expected.device, "type", str(expected.device).split(":", 1)[0])
    actual_device_obj = getattr(actual, "device", "cpu")
    actual_device = getattr(actual_device_obj, "type", str(actual_device_obj).split(":", 1)[0])
    if expected_device != actual_device:
        raise AssertionError(f"{path}: expected device {expected_device!r}, got {actual_device!r}")

    expected_requires_grad = bool(getattr(expected, "requires_grad", False))
    actual_requires_grad = bool(getattr(actual, "requires_grad", False))
    if expected_requires_grad != actual_requires_grad:
        raise AssertionError(
            f"{path}: expected requires_grad={expected_requires_grad!r}, got {actual_requires_grad!r}"
        )

    if check_stride and hasattr(actual, "stride"):
        expected_stride = tuple(expected.stride())
        actual_stride = tuple(actual.stride())
        if expected_stride != actual_stride:
            raise AssertionError(f"{path}: expected stride {expected_stride!r}, got {actual_stride!r}")

    actual_as_reference = _coerce_to_reference_tensor(reference, actual, expected.dtype)
    reference.testing.assert_close(
        expected.detach().cpu(),
        actual_as_reference,
        rtol=rtol,
        atol=atol,
        check_dtype=False,
        msg=lambda msg: f"{path}: tensor values differ\n{msg}",
    )


def _coerce_to_reference_tensor(reference: Any, value: Any, dtype: Any) -> Any:
    if isinstance(value, reference.Tensor):
        return value.detach().cpu()

    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "shape"):
        shape = tuple(current.shape)
        if any(dim == 0 for dim in shape):
            return reference.empty(shape, dtype=dtype)
    if hasattr(current, "tolist"):
        return reference.tensor(current.tolist(), dtype=dtype)
    if hasattr(current, "item"):
        return reference.tensor(current.item(), dtype=dtype)
    return reference.tensor(current, dtype=dtype)


def _mutate_tensor(value: Any, fill_value: float) -> None:
    if hasattr(value, "fill_"):
        value.fill_(fill_value)
        return
    if hasattr(value, "add_"):
        value.add_(fill_value)
        return
    raise AssertionError(f"Tensor-like value does not support an in-place mutation probe: {value!r}")


def _assert_exception_compatible(case_id: str, expected: BaseException, actual: BaseException | None) -> None:
    if actual is None:
        raise AssertionError(f"{case_id}: reference raised {expected!r}, candidate returned normally")
    expected_name = type(expected).__name__
    actual_name = type(actual).__name__
    if expected_name != actual_name:
        raise AssertionError(f"{case_id}: expected exception {expected_name}, got {actual_name}: {actual!r}")


def _dtype_name(value: Any) -> str:
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def maybe_signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def pytest_skip_missing_internal_opinfo() -> None:
    pytest.importorskip(
        "expecttest",
        reason="PyTorch internal OpInfo tests require optional dependency expecttest",
    )
