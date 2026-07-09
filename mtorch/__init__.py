"""PyTorch-compatible public API backed by the native mtorch core."""

from __future__ import annotations

import builtins as _builtins
import functools as _functools
import math as _math
import pickle as _pickle
import sys as _sys
from collections import OrderedDict, namedtuple
from enum import IntEnum as _IntEnum
from types import ModuleType

from . import _C


__version__ = "0.0.0"


class dtype(str):
    def __new__(cls, name: str):
        obj = str.__new__(cls, name)
        obj.name = name
        return obj

    def __repr__(self) -> str:
        return f"mtorch.{self.name}"

    def __str__(self) -> str:
        return self.name


class device(str):
    def __new__(cls, spec: str = "cpu"):
        text = str(spec)
        device_type, _, index = text.partition(":")
        if device_type == "metal":
            device_type = "mps"
        if device_type not in {"cpu", "mps"}:
            raise NotImplementedError(f"{device_type!r} device is not implemented")
        normalized = device_type if not index else f"{device_type}:{index}"
        obj = str.__new__(cls, normalized)
        obj.type = device_type
        obj.index = _builtins.int(index) if index else None
        return obj

    def __repr__(self) -> str:
        if self.index is None:
            return f"device(type='{self.type}')"
        return f"device(type='{self.type}', index={self.index})"

    def __str__(self) -> str:
        if self.index is None:
            return self.type
        return f"{self.type}:{self.index}"


class memory_format:
    __slots__ = ("name",)

    def __new__(cls, name: str | None = None, *, _internal: bool = False):
        if not _internal:
            raise TypeError("cannot create 'mtorch.memory_format' instances")
        obj = object.__new__(cls)
        obj.name = str(name)
        return obj

    def __repr__(self) -> str:
        return f"mtorch.{self.name}"

    def __str__(self) -> str:
        return f"mtorch.{self.name}"

    def __eq__(self, other) -> bool:
        return isinstance(other, memory_format) and self.name == other.name

    def __hash__(self) -> int:
        return hash((memory_format, self.name))


class layout:
    __slots__ = ("name",)

    def __new__(cls, name: str | None = None, *, _internal: bool = False):
        if not _internal:
            raise TypeError("cannot create 'mtorch.layout' instances")
        obj = object.__new__(cls)
        obj.name = str(name)
        return obj

    def __repr__(self) -> str:
        return f"mtorch.{self.name}"

    def __str__(self) -> str:
        return f"mtorch.{self.name}"

    def __eq__(self, other) -> bool:
        return isinstance(other, layout) and self.name == other.name

    def __hash__(self) -> int:
        return hash((layout, self.name))


class Size(tuple):
    pass


class finfo:
    _DATA = {
        "float16": (16, 0.0009765625, 65504.0, 6.103515625e-05, 0.001),
        "float32": (32, 1.1920928955078125e-07, 3.4028234663852886e38, 1.1754943508222875e-38, 1e-06),
        "float64": (64, 2.220446049250313e-16, 1.7976931348623157e308, 2.2250738585072014e-308, 1e-15),
    }

    def __init__(self, type=None):
        dtype_object = float32 if type is None else type
        name = getattr(dtype_object, "name", str(dtype_object))
        if name not in self._DATA:
            raise TypeError(f"torch.finfo() requires a floating point input type. Got {name}")
        bits, eps, max_value, tiny, resolution = self._DATA[name]
        self.bits = bits
        self.eps = eps
        self.max = max_value
        self.min = -max_value
        self.smallest_normal = tiny
        self.tiny = tiny
        self.resolution = resolution
        self.dtype = globals()[name]

    def __repr__(self) -> str:
        return (
            f"finfo(resolution={self.resolution:g}, min={self.min:g}, max={self.max:g}, "
            f"eps={self.eps:g}, smallest_normal={self.smallest_normal:g}, tiny={self.tiny:g}, "
            f"dtype={self.dtype})"
        )


class iinfo:
    _DATA = {
        "int32": (32, -(2**31), 2**31 - 1),
        "int64": (64, -(2**63), 2**63 - 1),
    }

    def __init__(self, type):
        name = getattr(type, "name", str(type))
        if name not in self._DATA:
            raise TypeError(f"torch.iinfo() requires an integer input type. Got {name}")
        bits, min_value, max_value = self._DATA[name]
        self.bits = bits
        self.min = min_value
        self.max = max_value
        self.dtype = globals()[name]

    def __repr__(self) -> str:
        return f"iinfo(min={_builtins.float(self.min):g}, max={_builtins.float(self.max):g}, dtype={self.dtype})"


float16 = dtype("float16")
float32 = dtype("float32")
float64 = dtype("float64")
int32 = dtype("int32")
int64 = dtype("int64")
bool = dtype("bool")
float = float32
double = float64
half = float16
long = int64
int = int32
contiguous_format = memory_format("contiguous_format", _internal=True)
preserve_format = memory_format("preserve_format", _internal=True)
channels_last = memory_format("channels_last", _internal=True)
channels_last_3d = memory_format("channels_last_3d", _internal=True)
strided = layout("strided", _internal=True)

Tensor = _C.Tensor
_tensor_dtype_descriptor = Tensor.dtype
_tensor_device_descriptor = Tensor.device


def _tensor_dtype_property(tensor):
    name = _tensor_dtype_descriptor.__get__(tensor, type(tensor))
    return globals().get(str(name), dtype(str(name)))


def _tensor_device_property(tensor):
    return device(_tensor_device_descriptor.__get__(tensor, type(tensor)))


Tensor.dtype = property(_tensor_dtype_property)
Tensor.device = property(_tensor_device_property)
Tensor.layout = property(lambda self: strided)
Tensor.is_cuda = property(lambda self: self.device.type == "cuda")
Tensor.is_mps = property(lambda self: self.device.type == "mps")
Tensor.is_sparse = property(lambda self: False)
Tensor.is_sparse_csr = property(lambda self: False)
Tensor.is_mkldnn = property(lambda self: False)
Tensor.is_quantized = property(lambda self: False)
Tensor.is_meta = property(lambda self: False)
Tensor.is_nested = property(lambda self: False)
Generator = _C.Generator
Generator.device = property(lambda self: device("cpu"))
tensor = _C.tensor
as_tensor = _C.as_tensor
asarray = _C.as_tensor
_zeros = _C.zeros
_ones = _C.ones
_empty = _C.empty
empty_strided = _C.empty_strided
_full = _C.full
empty_like = _C.empty_like
zeros_like = _C.zeros_like
ones_like = _C.ones_like
full_like = _C.full_like
arange = _C.arange
linspace = _C.linspace
eye = _C.eye
_randint = _C.randint
_rand = _C.rand
_randn = _C.randn
randperm = _C.randperm
manual_seed = _C.manual_seed
initial_seed = _C.initial_seed
multinomial = _C.multinomial
bernoulli = _C.bernoulli


def _normalize_layout_argument(value, op_name):
    if value is None or value is strided:
        return None
    if isinstance(value, layout):
        raise NotImplementedError(f"{op_name} layout={value!r} is not implemented")
    raise TypeError(f"{op_name} layout must be mtorch.layout")


def _strip_strided_layout(kwargs, op_name):
    kwargs = dict(kwargs)
    layout_value = kwargs.pop("layout", None)
    _normalize_layout_argument(layout_value, op_name)
    return kwargs


def empty(*args, **kwargs):
    return _empty(*args, **_strip_strided_layout(kwargs, "empty"))


def zeros(*args, **kwargs):
    return _zeros(*args, **_strip_strided_layout(kwargs, "zeros"))


def ones(*args, **kwargs):
    return _ones(*args, **_strip_strided_layout(kwargs, "ones"))


def full(*args, **kwargs):
    return _full(*args, **_strip_strided_layout(kwargs, "full"))


def rand(*args, **kwargs):
    return _rand(*args, **_strip_strided_layout(kwargs, "rand"))


def randn(*args, **kwargs):
    return _randn(*args, **_strip_strided_layout(kwargs, "randn"))


def randint(*args, **kwargs):
    kwargs = dict(kwargs)
    layout_value = kwargs.pop("layout", None)
    _normalize_layout_argument(layout_value, "randint")
    low_kw = kwargs.pop("low", None)
    high_kw = kwargs.pop("high", None)
    size_kw = kwargs.pop("size", None)
    if low_kw is None and high_kw is None and size_kw is None:
        return _randint(*args, **kwargs)
    values = list(args)
    if low_kw is not None:
        if values:
            raise TypeError("randint received both positional and keyword low")
        values.append(low_kw)
    if high_kw is not None:
        if len(values) > 1:
            raise TypeError("randint received both positional and keyword high")
        values.append(high_kw)
    elif low_kw is not None:
        raise TypeError("randint missing required keyword high")
    if size_kw is not None:
        if len(values) >= 3:
            raise TypeError("randint received both positional and keyword size")
        kwargs["size"] = size_kw
    return _randint(*tuple(values), **kwargs)


def randint_like(
    input,
    *args,
    low=0,
    high=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=preserve_format,
    pin_memory=False,
    generator=None,
):
    _normalize_layout_argument(layout, "randint_like")
    if pin_memory:
        raise NotImplementedError("randint_like pin_memory=True is not implemented")
    values = list(args)
    if high is None:
        if len(values) == 1:
            low_value = 0
            high_value = values[0]
        elif len(values) == 2:
            low_value = values[0]
            high_value = values[1]
        else:
            raise TypeError("randint_like expected high or low, high")
    else:
        if len(values) == 0:
            low_value = low
            high_value = high
        elif len(values) == 1:
            if low != 0:
                raise TypeError("randint_like received both positional and keyword low")
            low_value = values[0]
            high_value = high
        else:
            raise TypeError("randint_like received too many positional arguments")
    result = randint(
        low_value,
        high_value,
        tuple(input.shape),
        dtype=dtype or input.dtype,
        device=device or input.device,
        requires_grad=requires_grad,
        generator=generator,
    )
    target_format = _like_memory_format(input, memory_format)
    if target_format is None or target_format is preserve_format:
        return result
    return result.contiguous(memory_format=target_format)


def _numpy_dtype_for_mtorch(dtype_value):
    import numpy as _np

    name = str(dtype_value)
    mapping = {
        "float16": _np.float16,
        "float32": _np.float32,
        "float64": _np.float64,
        "int32": _np.int32,
        "int64": _np.int64,
        "bool": _np.bool_,
    }
    if name not in mapping:
        raise TypeError(f"cannot convert mtorch dtype {dtype_value!r} to a NumPy dtype")
    return mapping[name]


def _mtorch_dtype_from_numpy_dtype(numpy_dtype):
    name = str(numpy_dtype)
    mapping = {
        "float16": float16,
        "float32": float32,
        "float64": float64,
        "int32": int32,
        "int64": int64,
        "bool": bool,
        "bool_": bool,
    }
    if name not in mapping:
        raise TypeError(f"from_numpy does not support NumPy dtype {name!r}")
    return mapping[name]


def from_numpy(ndarray):
    import numpy as _np

    if not isinstance(ndarray, _np.ndarray):
        raise TypeError("from_numpy expected a numpy.ndarray")
    return tensor(ndarray.tolist(), dtype=_mtorch_dtype_from_numpy_dtype(ndarray.dtype))


def _tensor_numpy(self, *, force=False):
    import numpy as _np

    return _np.array(self.tolist(), dtype=_numpy_dtype_for_mtorch(self.dtype))


Tensor.numpy = _tensor_numpy


def _tensor_float(self):
    return _builtins.float(self.item())


def _tensor_int(self):
    return _builtins.int(self.item())


def _tensor_index(self):
    if str(self.dtype) not in {"int32", "int64", "bool"}:
        raise TypeError("only integer tensors of a single element can be converted to an index")
    try:
        return _builtins.int(self.item())
    except ValueError as exc:
        raise TypeError("only integer tensors of a single element can be converted to an index") from exc


Tensor.__float__ = _tensor_float
Tensor.__int__ = _tensor_int
Tensor.__index__ = _tensor_index


def _tensor_clamp_(self, min=None, max=None):
    self.copy_(clamp(self, min=min, max=max))
    return self


def _tensor_clamp_min_(self, min):
    self.copy_(clamp_min(self, min))
    return self


def _tensor_clamp_max_(self, max):
    self.copy_(clamp_max(self, max))
    return self


Tensor.clamp_ = _tensor_clamp_
Tensor.clip_ = _tensor_clamp_
Tensor.clamp_min_ = _tensor_clamp_min_
Tensor.clamp_max_ = _tensor_clamp_max_


class _TensorConstructor:
    _dtype = float32

    def __new__(cls, *args, **kwargs):
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"{cls.__name__} got an unexpected keyword argument {unexpected!r}")
        if not args:
            return empty((0,), dtype=cls._dtype)
        if len(args) == 1:
            value = args[0]
            if isinstance(value, Tensor):
                return value.to(dtype=cls._dtype)
            if isinstance(value, _builtins.int):
                return empty((value,), dtype=cls._dtype)
            return tensor(value, dtype=cls._dtype)
        return empty(tuple(_builtins.int(item) for item in args), dtype=cls._dtype)


class FloatTensor(_TensorConstructor):
    _dtype = float32


class DoubleTensor(_TensorConstructor):
    _dtype = float64


class HalfTensor(_TensorConstructor):
    _dtype = float16


class LongTensor(_TensorConstructor):
    _dtype = int64


class IntTensor(_TensorConstructor):
    _dtype = int32


class BoolTensor(_TensorConstructor):
    _dtype = bool


_TENSOR_TYPE_NAME_BY_DTYPE = {
    "float16": "torch.HalfTensor",
    "float32": "torch.FloatTensor",
    "float64": "torch.DoubleTensor",
    "int32": "torch.IntTensor",
    "int64": "torch.LongTensor",
    "bool": "torch.BoolTensor",
}

_TENSOR_DTYPE_BY_TYPE_NAME = {
    "halftensor": float16,
    "floattensor": float32,
    "doubletensor": float64,
    "inttensor": int32,
    "longtensor": int64,
    "booltensor": bool,
    "float16": float16,
    "float32": float32,
    "float64": float64,
    "int32": int32,
    "int64": int64,
    "bool": bool,
}


def _dtype_from_type_request(request):
    if isinstance(request, dtype):
        return request
    if isinstance(request, type) and issubclass(request, _TensorConstructor):
        return request._dtype
    text = str(request)
    normalized = text.rsplit(".", 1)[-1].lower().replace("'", "").replace(">", "")
    if normalized in _TENSOR_DTYPE_BY_TYPE_NAME:
        return _TENSOR_DTYPE_BY_TYPE_NAME[normalized]
    raise TypeError(f"invalid type: {request!r}")


def _tensor_type(self, dtype=None, non_blocking=False, **kwargs):
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"type() got an unexpected keyword argument {unexpected!r}")
    if dtype is None:
        return _TENSOR_TYPE_NAME_BY_DTYPE.get(str(self.dtype), "torch.Tensor")
    return self.to(dtype=_dtype_from_type_request(dtype))


def _tensor_new(self, *args):
    if not args:
        return empty((0,), dtype=self.dtype, device=self.device)
    if len(args) == 1:
        value = args[0]
        if isinstance(value, _builtins.int):
            return empty((value,), dtype=self.dtype, device=self.device)
        return tensor(value, dtype=self.dtype, device=self.device)
    return empty(tuple(_builtins.int(item) for item in args), dtype=self.dtype, device=self.device)


Tensor.type = _tensor_type
Tensor.new = _tensor_new


def _normalize_memory_format(value):
    if value is None:
        return None
    if not isinstance(value, memory_format):
        raise TypeError("memory_format must be mtorch.memory_format")
    return value


def _like_memory_format(input, requested):
    format_value = _normalize_memory_format(requested)
    if format_value is preserve_format:
        if len(input.shape) == 4 and input.is_contiguous(memory_format=channels_last):
            return channels_last
        if len(input.shape) == 5 and input.is_contiguous(memory_format=channels_last_3d):
            return channels_last_3d
        return None
    return format_value


def rand_like(
    input,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=preserve_format,
    pin_memory=False,
    generator=None,
):
    _normalize_layout_argument(layout, "rand_like")
    if pin_memory:
        raise NotImplementedError("rand_like pin_memory=True is not implemented")
    result = rand(
        tuple(input.shape),
        dtype=dtype or input.dtype,
        device=device or input.device,
        requires_grad=requires_grad,
        generator=generator,
    )
    target_format = _like_memory_format(input, memory_format)
    if target_format is None or target_format is preserve_format:
        return result
    return result.contiguous(memory_format=target_format)


def randn_like(
    input,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=preserve_format,
    pin_memory=False,
    generator=None,
):
    _normalize_layout_argument(layout, "randn_like")
    if pin_memory:
        raise NotImplementedError("randn_like pin_memory=True is not implemented")
    result = randn(
        tuple(input.shape),
        dtype=dtype or input.dtype,
        device=device or input.device,
        requires_grad=requires_grad,
        generator=generator,
    )
    target_format = _like_memory_format(input, memory_format)
    if target_format is None or target_format is preserve_format:
        return result
    return result.contiguous(memory_format=target_format)


def normal(
    mean,
    std=1.0,
    size=None,
    *,
    generator=None,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    requires_grad=False,
):
    mean_is_tensor = isinstance(mean, Tensor)
    std_is_tensor = isinstance(std, Tensor)
    if mean_is_tensor or std_is_tensor:
        if size is not None or dtype is not None or layout is not None or device is not None or pin_memory or requires_grad:
            raise TypeError("normal tensor overloads only accept generator and out keyword arguments")
        if mean_is_tensor and std_is_tensor:
            mean, std = broadcast_tensors(mean, std)
            result = randn_like(mean, generator=generator) * std + mean
        elif mean_is_tensor:
            result = randn_like(mean, generator=generator) * _builtins.float(std) + mean
        else:
            result = randn_like(std, generator=generator) * std + _builtins.float(mean)
    else:
        if size is None:
            raise TypeError("normal scalar overload requires size")
        _normalize_layout_argument(layout, "normal")
        if pin_memory:
            raise NotImplementedError("normal pin_memory=True is not implemented")
        target_dtype = dtype or (out.dtype if out is not None else float32)
        target_device = device or (out.device if out is not None else None)
        result = empty(size, dtype=target_dtype, device=target_device, requires_grad=requires_grad)
        result.normal_(_builtins.float(mean), _builtins.float(std), generator=generator)
    if out is not None:
        out.copy_(result)
        return out
    return result


neg = _C.neg
negative = _C.neg
abs = _C.abs
absolute = _C.abs
exp = _C.exp
expm1 = _C.expm1
log = _C.log
log1p = _C.log1p
log2 = _C.log2
log10 = _C.log10
sqrt = _C.sqrt
rsqrt = _C.rsqrt
reciprocal = _C.reciprocal
sign = _C.sign
floor = _C.floor
ceil = _C.ceil
trunc = _C.trunc
fix = _C.trunc
round = _C.round
sin = _C.sin
cos = _C.cos
tan = _C.tan
sinh = _C.sinh
cosh = _C.cosh
tanh = _C.tanh
asin = _C.asin
arcsin = _C.asin
acos = _C.acos
arccos = _C.acos
atan = _C.atan
arctan = _C.atan
sigmoid = _C.sigmoid
erf = _C.erf
erfc = _C.erfc
deg2rad = _C.deg2rad
rad2deg = _C.rad2deg
frac = _C.frac
isnan = _C.isnan
isinf = _C.isinf
isfinite = _C.isfinite
signbit = _C.signbit
isposinf = _C.isposinf
isneginf = _C.isneginf
logical_not = _C.logical_not
bitwise_not = _C.bitwise_not
square = _C.square
nan_to_num = _C.nan_to_num
_clamp = _C.clamp
_clamp_min = _C.clamp_min
_clamp_max = _C.clamp_max
softmax = _C.softmax
log_softmax = _C.log_softmax
norm = _C.norm
_normalize_l2 = _C._normalize_l2
layer_norm = _C.layer_norm
rms_norm = _C.rms_norm
_add = _C.add
_sub = _C.sub
mul = _C.mul
multiply = _C.mul
div = _C.div
divide = _C.div
true_divide = _C.div
pow = _C.pow
floor_divide = _C.floor_divide
float_power = _C.float_power
remainder = _C.remainder
fmod = _C.fmod
atan2 = _C.atan2
arctan2 = _C.atan2
hypot = _C.hypot
ldexp = _C.ldexp
nextafter = _C.nextafter
copysign = _C.copysign
heaviside = _C.heaviside
logaddexp = _C.logaddexp
logaddexp2 = _C.logaddexp2
xlogy = _C.xlogy
fmax = _C.fmax
fmin = _C.fmin
addcmul = _C.addcmul
addcdiv = _C.addcdiv
maximum = _C.maximum
minimum = _C.minimum


def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    _normalize_layout_argument(layout, "logspace")
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    exponents = linspace(start, end, steps, **kwargs)
    result = pow(base, exponents)
    if out is not None:
        out.copy_(result)
        if requires_grad:
            out.requires_grad_(True)
        return out
    if requires_grad:
        result.requires_grad_(True)
    return result


_Tensor_add = Tensor.add
_Tensor_sub = Tensor.sub
_Tensor_add_ = Tensor.add_
_Tensor_sub_ = Tensor.sub_


def _alpha_scaled_other(other, alpha):
    alpha_value = _builtins.float(alpha)
    if alpha_value == 1.0:
        return other
    return other * alpha_value


def add(input, other, *, alpha=1, out=None):
    result = _add(input, _alpha_scaled_other(other, alpha))
    if out is not None:
        out.copy_(result)
        return out
    return result


def sub(input, other, *, alpha=1, out=None):
    result = _sub(input, _alpha_scaled_other(other, alpha))
    if out is not None:
        out.copy_(result)
        return out
    return result


subtract = sub


def _tensor_add(self, other, *, alpha=1):
    if _builtins.float(alpha) == 1.0:
        return _Tensor_add(self, other)
    return add(self, other, alpha=alpha)


def _tensor_sub(self, other, *, alpha=1):
    if _builtins.float(alpha) == 1.0:
        return _Tensor_sub(self, other)
    return sub(self, other, alpha=alpha)


def _tensor_add_alpha_(self, other, *, alpha=1):
    if _builtins.float(alpha) == 1.0:
        return _Tensor_add_(self, other)
    self.copy_(add(self, other, alpha=alpha))
    return self


def _tensor_sub_alpha_(self, other, *, alpha=1):
    if _builtins.float(alpha) == 1.0:
        return _Tensor_sub_(self, other)
    self.copy_(sub(self, other, alpha=alpha))
    return self


Tensor.add = _tensor_add
Tensor.sub = _tensor_sub
Tensor.subtract = _tensor_sub
Tensor.add_ = _tensor_add_alpha_
Tensor.sub_ = _tensor_sub_alpha_


def clamp(input, min=None, max=None):
    if min is None and max is None:
        raise ValueError("clamp expected min, max, or both")
    if isinstance(min, Tensor) or isinstance(max, Tensor):
        result = input
        if min is not None:
            result = maximum(result, min) if isinstance(min, Tensor) else _clamp_min(result, min)
        if max is not None:
            result = minimum(result, max) if isinstance(max, Tensor) else _clamp_max(result, max)
        return result
    return _clamp(input, min=min, max=max)


def clamp_min(input, min):
    return maximum(input, min) if isinstance(min, Tensor) else _clamp_min(input, min)


def clamp_max(input, max):
    return minimum(input, max) if isinstance(max, Tensor) else _clamp_max(input, max)


clip = clamp
Tensor.clamp = lambda self, min=None, max=None: clamp(self, min=min, max=max)
Tensor.clip = Tensor.clamp
Tensor.clamp_min = lambda self, min: clamp_min(self, min)
Tensor.clamp_max = lambda self, max: clamp_max(self, max)
eq = _C.eq
ne = _C.ne
not_equal = _C.ne
lt = _C.lt
less = _C.lt
le = _C.le
less_equal = _C.le
gt = _C.gt
greater = _C.gt
ge = _C.ge
greater_equal = _C.ge
logical_and = _C.logical_and
logical_or = _C.logical_or
logical_xor = _C.logical_xor
bitwise_and = _C.bitwise_and
bitwise_or = _C.bitwise_or
bitwise_xor = _C.bitwise_xor
isclose = _C.isclose
allclose = _C.allclose
equal = _C.equal
is_nonzero = _C.is_nonzero
lerp = _C.lerp
matmul = _C.matmul
mm = _C.mm
bmm = _C.bmm
addmm = _C.addmm
addmv = _C.addmv
addr = _C.addr
baddbmm = _C.baddbmm
addbmm = _C.addbmm
vdot = _C.vdot
inner = _C.inner
tensordot = _C.tensordot
kron = _C.kron
chain_matmul = _C.chain_matmul
matrix_power = _C.matrix_power
dot = _C.dot
mv = _C.mv
outer = _C.outer
ger = _C.ger


_EINSUM_ELLIPSIS = object()


def _einsum_prod(values):
    result = 1
    for value in values:
        result *= _builtins.int(value)
    return result


def _einsum_unique(labels):
    result = []
    for label in labels:
        if label not in result:
            result.append(label)
    return result


def _einsum_parse_subscript(spec):
    labels = []
    index = 0
    while index < len(spec):
        if spec.startswith("...", index):
            labels.append(_EINSUM_ELLIPSIS)
            index += 3
            continue
        if spec[index] == ".":
            raise ValueError("einsum found '.' outside an ellipsis")
        labels.append(spec[index])
        index += 1
    if labels.count(_EINSUM_ELLIPSIS) > 1:
        raise ValueError("einsum subscript contains more than one ellipsis")
    return labels


def _einsum_expand_labels(tokens, tensor, ellipsis_labels):
    explicit_rank = _builtins.sum(1 for label in tokens if label is not _EINSUM_ELLIPSIS)
    ellipsis_rank = tensor.ndim - explicit_rank
    if ellipsis_rank < 0:
        raise ValueError("einsum subscript rank does not match operand rank")
    labels = []
    for label in tokens:
        if label is _EINSUM_ELLIPSIS:
            labels.extend(ellipsis_labels[len(ellipsis_labels) - ellipsis_rank :])
        else:
            labels.append(label)
    if len(labels) != tensor.ndim:
        raise ValueError("einsum subscript rank does not match operand rank")
    return labels


def _einsum_expand_output(tokens, ellipsis_labels):
    labels = []
    for label in tokens:
        if label is _EINSUM_ELLIPSIS:
            labels.extend(ellipsis_labels)
        else:
            labels.append(label)
    if len(labels) != len(set(labels)):
        raise ValueError("einsum output subscript must not contain repeated labels")
    return labels


def _einsum_apply_repeated_labels(tensor, labels):
    labels = list(labels)
    while len(labels) != len(set(labels)):
        repeated = next(label for label in labels if labels.count(label) > 1)
        first = labels.index(repeated)
        second = labels.index(repeated, first + 1)
        if tensor.shape[first] != tensor.shape[second]:
            raise ValueError("einsum repeated subscript dimensions must match")
        tensor = diagonal(tensor, 0, first, second)
        labels = [label for index, label in enumerate(labels) if index not in (first, second)]
        labels.append(repeated)
    return tensor, labels


def _einsum_update_label_size(label_sizes, label, size):
    size = _builtins.int(size)
    if label not in label_sizes:
        label_sizes[label] = size
        return
    previous = label_sizes[label]
    if previous != size and previous != 1 and size != 1:
        raise ValueError("einsum operands could not be broadcast together")
    label_sizes[label] = _builtins.max(previous, size)


def _einsum_label_dim(tensor, labels, target):
    return _builtins.int(tensor.shape[labels.index(target)])


def _einsum_permute_if_needed(tensor, labels, ordered_labels):
    order = [labels.index(label) for label in ordered_labels]
    if order == list(range(len(order))):
        return tensor
    return permute(tensor, order)


def _einsum_align_operand(tensor, labels, target_labels):
    present = [label for label in target_labels if label in labels]
    tensor = _einsum_permute_if_needed(tensor, labels, present) if present else tensor
    if present == target_labels:
        return tensor
    shape = []
    present_index = 0
    for label in target_labels:
        if label in labels:
            shape.append(_builtins.int(tensor.shape[present_index]))
            present_index += 1
        else:
            shape.append(1)
    return reshape(tensor, tuple(shape))


def _einsum_try_binary_matmul(operands, labels_by_operand, output_labels, label_sizes, contract_labels):
    if len(operands) != 2 or not contract_labels:
        return None
    left, right = operands
    left_labels, right_labels = labels_by_operand
    if _builtins.any(label not in left_labels or label not in right_labels for label in contract_labels):
        return None
    for label in contract_labels:
        if _einsum_label_dim(left, left_labels, label) != _einsum_label_dim(right, right_labels, label):
            return None

    batch_labels = [label for label in output_labels if label in left_labels and label in right_labels]
    left_free = [label for label in output_labels if label in left_labels and label not in right_labels]
    right_free = [label for label in output_labels if label in right_labels and label not in left_labels]
    supported_left = set(batch_labels + left_free + contract_labels)
    supported_right = set(batch_labels + right_free + contract_labels)
    if set(left_labels) != supported_left or set(right_labels) != supported_right:
        return None

    left_order = batch_labels + left_free + contract_labels
    right_order = batch_labels + contract_labels + right_free
    left_arg = _einsum_permute_if_needed(left, left_labels, left_order)
    right_arg = _einsum_permute_if_needed(right, right_labels, right_order)

    batch_rank = len(batch_labels)
    left_batch_shape = tuple(_builtins.int(size) for size in left_arg.shape[:batch_rank])
    right_batch_shape = tuple(_builtins.int(size) for size in right_arg.shape[:batch_rank])
    left_free_shape = tuple(_builtins.int(size) for size in left_arg.shape[batch_rank : batch_rank + len(left_free)])
    right_free_shape = tuple(_builtins.int(size) for size in right_arg.shape[batch_rank + len(contract_labels) :])
    contract_size = _einsum_prod(label_sizes[label] for label in contract_labels)
    left_outer = _einsum_prod(left_free_shape)
    right_outer = _einsum_prod(right_free_shape)

    left_matrix = reshape(left_arg, left_batch_shape + (left_outer, contract_size))
    right_matrix = reshape(right_arg, right_batch_shape + (contract_size, right_outer))
    result = matmul(left_matrix, right_matrix)

    canonical = batch_labels + left_free + right_free
    canonical_shape = tuple(label_sizes[label] for label in canonical)
    result = reshape(result, canonical_shape)
    if output_labels != canonical and output_labels:
        result = permute(result, [canonical.index(label) for label in output_labels])
    return result


def _einsum_generic_elementwise(operands, labels_by_operand, output_labels, contract_labels):
    target_labels = output_labels + contract_labels
    aligned = [_einsum_align_operand(tensor, labels, target_labels) for tensor, labels in zip(operands, labels_by_operand)]
    result = aligned[0]
    for tensor in aligned[1:]:
        result = result * tensor
    for axis in range(len(target_labels) - 1, len(output_labels) - 1, -1):
        result = sum(result, dim=axis)
    return result


def einsum(equation, *operands):
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = tuple(operands[0])
    try:
        return _C.einsum(equation, *operands)
    except NotImplementedError:
        pass
    if not isinstance(equation, str):
        raise TypeError("einsum equation must be a string")
    if not operands:
        raise ValueError("einsum expected at least one operand")

    compact = equation.replace(" ", "")
    if "->" in compact:
        inputs_text, output_text = compact.split("->", 1)
        has_explicit_output = True
    else:
        inputs_text, output_text = compact, ""
        has_explicit_output = False
    input_specs = inputs_text.split(",")
    if len(input_specs) != len(operands):
        raise ValueError("einsum operand count does not match equation")

    parsed_inputs = [_einsum_parse_subscript(spec) for spec in input_specs]
    ellipsis_rank = 0
    for tokens, operand in zip(parsed_inputs, operands):
        explicit_rank = _builtins.sum(1 for label in tokens if label is not _EINSUM_ELLIPSIS)
        ellipsis_rank = _builtins.max(ellipsis_rank, operand.ndim - explicit_rank)
    ellipsis_labels = [f"@ellipsis{index}" for index in range(ellipsis_rank)]

    normalized_operands = []
    labels_by_operand = []
    label_sizes = {}
    input_label_order = []
    for tokens, operand in zip(parsed_inputs, operands):
        labels = _einsum_expand_labels(tokens, operand, ellipsis_labels)
        operand, labels = _einsum_apply_repeated_labels(operand, labels)
        normalized_operands.append(operand)
        labels_by_operand.append(labels)
        input_label_order.extend(labels)
        for label, size in zip(labels, operand.shape):
            _einsum_update_label_size(label_sizes, label, size)

    if has_explicit_output:
        output_labels = _einsum_expand_output(_einsum_parse_subscript(output_text), ellipsis_labels)
        missing = [label for label in output_labels if label not in label_sizes]
        if missing:
            raise ValueError("einsum output subscript appears in no input operand")
    else:
        counts = {}
        for label in input_label_order:
            if label not in ellipsis_labels:
                counts[label] = counts.get(label, 0) + 1
        output_labels = ellipsis_labels + sorted(label for label, count in counts.items() if count == 1)

    contract_labels = [label for label in _einsum_unique(input_label_order) if label not in output_labels]
    if contract_labels and _builtins.any(operand.dtype != normalized_operands[0].dtype for operand in normalized_operands[1:]):
        raise RuntimeError("expected einsum operands to have the same dtype for contraction")
    if contract_labels and normalized_operands[0].dtype == bool:
        raise RuntimeError("einsum contraction is not implemented for bool tensors")

    result = _einsum_try_binary_matmul(
        normalized_operands,
        labels_by_operand,
        output_labels,
        label_sizes,
        contract_labels,
    )
    if result is not None:
        return result
    return _einsum_generic_elementwise(normalized_operands, labels_by_operand, output_labels, contract_labels)


conv_transpose1d = _C.conv_transpose1d
conv2d = _C.conv2d
conv_transpose2d = _C.conv_transpose2d
conv_transpose3d = _C.conv_transpose3d
max_pool1d = _C.max_pool1d
avg_pool1d = _C.avg_pool1d
max_pool2d = _C.max_pool2d
_sum = _C.sum
trace = _C.trace
_diff_float32 = _C._diff_float32
cumsum = _C.cumsum
cumprod = _C.cumprod
cummax = _C.cummax
cummin = _C.cummin
_trapezoid_dx = _C._trapezoid_dx
_cumulative_trapezoid_dx = _C._cumulative_trapezoid_dx
_gradient_uniform = _C._gradient_uniform


def _slice_for_diff(input, dim, start, stop):
    index = [slice(None)] * input.ndim
    index[dim] = slice(start, stop)
    return input[tuple(index)]


def _normalize_dim_for_input(input, dim):
    rank = input.ndim
    axis = _builtins.int(dim)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise IndexError("dimension out of range")
    return axis


def diff(input, n=1, dim=-1, prepend=None, append=None, *, out=None):
    order = _builtins.int(n)
    if order < 0:
        raise RuntimeError(f"order must be non-negative but got {order}")
    axis = _normalize_dim_for_input(input, dim)
    result = input
    if prepend is not None or append is not None:
        pieces = []
        if prepend is not None:
            pieces.append(prepend)
        pieces.append(result)
        if append is not None:
            pieces.append(append)
        result = cat(tuple(pieces), dim=axis)
    if order == 0:
        result = result.clone()
    elif (
        order == 1
        and prepend is None
        and append is None
        and result.dtype == float32
        and result.ndim == 2
        and not result.requires_grad
    ):
        result = _diff_float32(result, axis)
    else:
        for _ in range(order):
            result = _slice_for_diff(result, axis, 1, None) - _slice_for_diff(result, axis, None, -1)
    if out is not None:
        out.copy_(result)
        return out
    return result


Tensor.diff = lambda self, n=1, dim=-1, prepend=None, append=None: diff(
    self, n=n, dim=dim, prepend=prepend, append=append
)


def _trapezoid_areas(y, x, dx, dim):
    axis = _normalize_dim_for_input(y, dim)
    left = _slice_for_diff(y, axis, None, -1)
    right = _slice_for_diff(y, axis, 1, None)
    pair_average = (left + right) * 0.5
    if x is None:
        return pair_average * dx, axis
    if x.ndim == 1:
        widths = diff(x, dim=0)
        shape = [1] * y.ndim
        shape[axis] = widths.shape[0]
        widths = widths.reshape(tuple(shape))
    else:
        widths = diff(x, dim=axis)
    return pair_average * widths, axis


def trapezoid(y, x=None, *, dx=1.0, dim=-1):
    if x is None and not y.requires_grad:
        return _trapezoid_dx(y, dx=dx, dim=dim)
    areas, axis = _trapezoid_areas(y, x, dx, dim)
    return areas.sum(dim=axis)


trapz = trapezoid


def cumulative_trapezoid(y, x=None, *, dx=1.0, dim=-1):
    if x is None and not y.requires_grad:
        return _cumulative_trapezoid_dx(y, dx=dx, dim=dim)
    areas, axis = _trapezoid_areas(y, x, dx, dim)
    return areas.cumsum(dim=axis)


def _gradient_dims(input, dim):
    if dim is None:
        return tuple(range(input.ndim))
    if isinstance(dim, (tuple, list, Size)):
        normalized = []
        for item in dim:
            value = _normalize_dim_for_input(input, item)
            if value in normalized:
                raise RuntimeError(f"dim {value} appears multiple times in the list of dims")
            normalized.append(value)
        return tuple(normalized)
    return (_normalize_dim_for_input(input, dim),)


def _gradient_scalar_spacing(value):
    if isinstance(value, Tensor):
        if value.ndim != 0:
            raise TypeError("gradient spacing tensor must be a scalar when spacing is not a sequence")
        return _builtins.float(value.item())
    return _builtins.float(value)


def _gradient_spacing(input, spacing, dims, dim_was_none):
    if isinstance(spacing, (tuple, list, Size)):
        values = tuple(spacing)
        if len(values) != len(dims):
            if dim_was_none:
                raise RuntimeError(
                    "torch.gradient expected spacing to be unspecified, a scalar, or a list of length "
                    f"equal to 'self.dim() = {input.ndim}', since dim argument was not given, "
                    f"but got a list of length {len(values)}"
                )
            raise RuntimeError(
                "torch.gradient expected spacing to be unspecified, a scalar or it's spacing and dim arguments "
                f"to have the same length, but got a spacing argument of length {len(values)} and a dim argument "
                f"of length {len(dims)}."
            )
        if _builtins.any(isinstance(item, Tensor) for item in values):
            for index, item in enumerate(values):
                if not isinstance(item, Tensor):
                    raise TypeError(f"expected Tensor as element {index} in argument 1, but got {type(item).__name__}")
                if item.ndim != 1:
                    raise RuntimeError(
                        "torch.gradient expected each element of spacing to have one dimension, "
                        f"but got an element with {item.ndim} dimensions!"
                    )
                if item.shape[0] != input.shape[dims[index]]:
                    raise RuntimeError("torch.gradient expected each element of spacing to have the same length as the dimension")
            return tuple(("coordinate", item) for item in values)
        return tuple(("scalar", _builtins.float(item)) for item in values)
    scalar = _gradient_scalar_spacing(spacing)
    return tuple(("scalar", scalar) for _ in dims)


def _gradient_shape_for_axis(input, axis, length):
    shape = [1] * input.ndim
    shape[axis] = length
    return tuple(shape)


def _gradient_uniform_python(input, axis, spacing, edge_order):
    inv_spacing = 1.0 / _builtins.float(spacing)
    if edge_order == 1:
        first = (input.select(axis, 1) - input.select(axis, 0)) * inv_spacing
        last = (input.select(axis, input.shape[axis] - 1) - input.select(axis, input.shape[axis] - 2)) * inv_spacing
    else:
        first = (
            input.select(axis, 0) * -1.5
            + input.select(axis, 1) * 2.0
            - input.select(axis, 2) * 0.5
        ) * inv_spacing
        last = (
            input.select(axis, input.shape[axis] - 1) * 1.5
            - input.select(axis, input.shape[axis] - 2) * 2.0
            + input.select(axis, input.shape[axis] - 3) * 0.5
        ) * inv_spacing

    middle = (
        _slice_for_diff(input, axis, 2, None) - _slice_for_diff(input, axis, None, -2)
    ) * (0.5 * inv_spacing)
    return cat((first.unsqueeze(axis), middle, last.unsqueeze(axis)), dim=axis)


def _gradient_coordinate(input, axis, coordinates, edge_order):
    previous_width = _slice_for_diff(coordinates, 0, 1, -1) - _slice_for_diff(coordinates, 0, None, -2)
    next_width = _slice_for_diff(coordinates, 0, 2, None) - _slice_for_diff(coordinates, 0, 1, -1)
    denom = previous_width * next_width * (previous_width + next_width)
    left_weight = -(next_width * next_width) / denom
    center_weight = (next_width * next_width - previous_width * previous_width) / denom
    right_weight = (previous_width * previous_width) / denom
    coefficient_shape = _gradient_shape_for_axis(input, axis, input.shape[axis] - 2)
    middle = (
        _slice_for_diff(input, axis, None, -2) * left_weight.reshape(coefficient_shape)
        + _slice_for_diff(input, axis, 1, -1) * center_weight.reshape(coefficient_shape)
        + _slice_for_diff(input, axis, 2, None) * right_weight.reshape(coefficient_shape)
    )

    if edge_order == 1:
        first = (input.select(axis, 1) - input.select(axis, 0)) / (coordinates.select(0, 1) - coordinates.select(0, 0))
        last = (input.select(axis, input.shape[axis] - 1) - input.select(axis, input.shape[axis] - 2)) / (
            coordinates.select(0, input.shape[axis] - 1) - coordinates.select(0, input.shape[axis] - 2)
        )
    else:
        first_width = coordinates.select(0, 1) - coordinates.select(0, 0)
        second_width = coordinates.select(0, 2) - coordinates.select(0, 1)
        first = (
            input.select(axis, 0) * (-(2.0 * first_width + second_width) / (first_width * (first_width + second_width)))
            + input.select(axis, 1) * ((first_width + second_width) / (first_width * second_width))
            - input.select(axis, 2) * (first_width / (second_width * (first_width + second_width)))
        )

        penultimate_width = coordinates.select(0, input.shape[axis] - 2) - coordinates.select(0, input.shape[axis] - 3)
        last_width = coordinates.select(0, input.shape[axis] - 1) - coordinates.select(0, input.shape[axis] - 2)
        last = (
            input.select(axis, input.shape[axis] - 3)
            * (last_width / (penultimate_width * (penultimate_width + last_width)))
            - input.select(axis, input.shape[axis] - 2)
            * ((penultimate_width + last_width) / (penultimate_width * last_width))
            + input.select(axis, input.shape[axis] - 1)
            * ((2.0 * last_width + penultimate_width) / (last_width * (penultimate_width + last_width)))
        )

    return cat((first.unsqueeze(axis), middle, last.unsqueeze(axis)), dim=axis)


def gradient(input, *, spacing=1, dim=None, edge_order=1):
    edge = _builtins.int(edge_order)
    if edge not in (1, 2):
        raise RuntimeError("torch.gradient only supports edge_order=1 and edge_order=2.")
    dims = _gradient_dims(input, dim)
    spacing_kinds = _gradient_spacing(input, spacing, dims, dim is None)
    if not dims:
        return ()
    if input.dtype == bool:
        raise RuntimeError("Subtraction, the `-` operator, with two bool tensors is not supported")
    for axis in dims:
        if input.shape[axis] < edge + 1:
            raise RuntimeError("torch.gradient expected each dimension size to be at least edge_order+1")

    results = []
    for axis, (kind, value) in zip(dims, spacing_kinds, strict=True):
        if kind == "scalar":
            if not input.requires_grad:
                results.append(_gradient_uniform(input, spacing=value, dim=axis, edge_order=edge))
            else:
                results.append(_gradient_uniform_python(input, axis, value, edge))
        else:
            results.append(_gradient_coordinate(input, axis, value, edge))
    return tuple(results)


_mean = _C.mean
prod = _C.prod
_var = _C.var
_std = _C.std
_var_mean_native = _C._var_mean
_std_mean_native = _C._std_mean
_var_tail_native = _C._var_tail
_std_tail_native = _C._std_tail
_all = _C.all
_any = _C.any
amax = _C.amax
amin = _C.amin
max = _C.max
argmax = _C.argmax
min = _C.min
argmin = _C.argmin
sort = _C.sort
argsort = _C.argsort
topk = _C.topk
_quantile_flat = _C._quantile_flat
_quantile_dim_2d = _C._quantile_dim_2d
searchsorted = _C.searchsorted
unique = _C.unique
unique_consecutive = _C.unique_consecutive


def bucketize(input, boundaries, *, out_int32=False, right=False, out=None):
    return searchsorted(boundaries, input, out_int32=out_int32, right=right, out=out)


def _quantile_value(q):
    value = q.item() if isinstance(q, Tensor) else q
    value = _builtins.float(value)
    if value < 0.0 or value > 1.0:
        raise RuntimeError("quantile q must be in the range [0, 1]")
    return value


def _quantile_index_pair(length, q):
    if length <= 0:
        raise RuntimeError("quantile input must be non-empty")
    position = q * _builtins.float(length - 1)
    lower = _builtins.int(_math.floor(position))
    upper = _builtins.int(_math.ceil(position))
    return lower, upper, position - _builtins.float(lower)


def _quantile_select(sorted_values, dim, q, interpolation):
    length = sorted_values.shape[dim]
    lower, upper, weight = _quantile_index_pair(length, q)
    mode = str(interpolation)
    if mode == "lower" or lower == upper:
        return sorted_values.select(dim, lower)
    if mode == "higher":
        return sorted_values.select(dim, upper)
    if mode == "nearest":
        return sorted_values.select(dim, _builtins.int(_builtins.round(q * _builtins.float(length - 1))))
    lower_values = sorted_values.select(dim, lower)
    upper_values = sorted_values.select(dim, upper)
    if mode == "midpoint":
        return (lower_values + upper_values) * 0.5
    if mode != "linear":
        raise ValueError("interpolation must be linear, lower, higher, midpoint, or nearest")
    return lower_values + (upper_values - lower_values) * weight


def quantile(input, q, dim=None, keepdim=False, *, interpolation="linear", out=None):
    if input.dtype not in {float32, float64}:
        raise RuntimeError("quantile input tensor must be either float or double dtype")
    q_value = _quantile_value(q)
    if dim is None:
        if (
            not input.requires_grad
            and input.is_contiguous()
            and interpolation in {"linear", "lower", "higher", "midpoint", "nearest"}
        ):
            result = _quantile_flat(input, q_value, interpolation)
        else:
            result = _quantile_select(input.flatten().sort(dim=0)[0], 0, q_value, interpolation)
        if keepdim:
            result = result.reshape(tuple(1 for _ in range(input.ndim)))
    else:
        axis = _builtins.int(dim)
        if axis < 0:
            axis += input.ndim
        if axis < 0 or axis >= input.ndim:
            raise IndexError("dimension out of range")
        if (
            not input.requires_grad
            and input.ndim == 2
            and input.is_contiguous()
            and interpolation in {"linear", "lower", "higher", "midpoint", "nearest"}
        ):
            result = _quantile_dim_2d(input, q_value, axis, interpolation)
        else:
            result = _quantile_select(input.sort(dim=axis)[0], axis, q_value, interpolation)
        if keepdim:
            result = result.unsqueeze(axis)
    result = result.contiguous()
    if out is not None:
        out.copy_(result)
        return out
    return result


Tensor.quantile = lambda self, q, dim=None, keepdim=False, *, interpolation="linear", out=None: quantile(
    self, q, dim=dim, keepdim=keepdim, interpolation=interpolation, out=out
)
reshape = _C.reshape
unflatten = _C.unflatten
transpose = _C.transpose
permute = _C.permute
movedim = _C.movedim
moveaxis = _C.moveaxis
swapaxes = _C.swapaxes
swapdims = _C.swapdims
flatten = _C.flatten
ravel = _C.ravel
t = _C.t
broadcast_to = _C.broadcast_to
broadcast_shapes = _C.broadcast_shapes
broadcast_tensors = _C.broadcast_tensors


def _tensor_broadcast_to(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, Size)):
        shape = tuple(shape[0])
    return broadcast_to(self, tuple(shape))


Tensor.broadcast_to = _tensor_broadcast_to
tile = _C.tile
repeat_interleave = _C.repeat_interleave
flip = _C.flip
fliplr = _C.fliplr
flipud = _C.flipud
rot90 = _C.rot90
roll = _C.roll
squeeze = _C.squeeze
unsqueeze = _C.unsqueeze
narrow = _C.narrow
select = _C.select
as_strided = _C.as_strided
diagonal = _C.diagonal
diag = _C.diag
diagflat = _C.diagflat
diag_embed = _C.diag_embed
block_diag = _C.block_diag
tril = _C.tril
triu = _C.triu


def _tensor_tril_(self, diagonal=0):
    self.copy_(tril(self, diagonal=diagonal))
    return self


def _tensor_triu_(self, diagonal=0):
    self.copy_(triu(self, diagonal=diagonal))
    return self


Tensor.tril_ = _tensor_tril_
Tensor.triu_ = _tensor_triu_
Tensor.ndimension = Tensor.dim
Tensor.nelement = Tensor.numel
split = _C.split
chunk = _C.chunk
unbind = _C.unbind
cat = _C.cat
concat = _C.cat
concatenate = _C.cat
stack = _C.stack
where = _C.where
take = _C.take
index_select = _C.index_select
gather = _C.gather
_index_put_native = _C.index_put
isin = _C.isin


def _normalize_index_put_indices(indices):
    if not isinstance(indices, (tuple, list)):
        raise TypeError("index_put(): argument 'indices' must be tuple of Tensors")
    if len(indices) == 0:
        raise RuntimeError("index_put expected a non-empty indices tuple")
    normalized = tuple(indices)
    for index, item in enumerate(normalized):
        if not isinstance(item, Tensor):
            raise TypeError(f"expected Tensor as element {index} in argument 1")
    return normalized


def index_put(input, indices, values, accumulate=False):
    if not isinstance(values, Tensor):
        raise TypeError("index_put(): argument 'values' must be Tensor")
    return _index_put_native(input, _normalize_index_put_indices(indices), values, accumulate=accumulate)


def _tensor_index_put(self, indices, values, accumulate=False):
    return index_put(self, indices, values, accumulate=accumulate)


def _tensor_index_put_(self, indices, values, accumulate=False):
    if not isinstance(values, Tensor):
        raise TypeError("index_put_(): argument 'values' must be Tensor")
    self.copy_(_index_put_native(self, _normalize_index_put_indices(indices), values, accumulate=accumulate))
    return self


def take_along_dim(input, indices, dim=None, *, out=None):
    if dim is None:
        result = input.flatten().gather(0, indices.flatten())
    else:
        result = gather(input, dim, indices)
    if out is not None:
        out.copy_(result)
        return out
    return result


Tensor.take_along_dim = lambda self, indices, dim=None: take_along_dim(self, indices, dim=dim)
Tensor.index_put = _tensor_index_put
Tensor.index_put_ = _tensor_index_put_
Tensor.unique = lambda self, sorted=True, return_inverse=False, return_counts=False, dim=None: unique(
    self, sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim
)
Tensor.unique_consecutive = lambda self, return_inverse=False, return_counts=False, dim=None: unique_consecutive(
    self, return_inverse=return_inverse, return_counts=return_counts, dim=dim
)
Tensor.bernoulli = lambda self, *, generator=None: bernoulli(self, generator=generator)
scatter = _C.scatter
scatter_add = _C.scatter_add
masked_select = _C.masked_select
masked_fill = _C.masked_fill
nonzero = _C.nonzero
argwhere = _C.argwhere


def _tensor_T(self):
    if self.ndim == 0:
        return self
    return permute(self, tuple(reversed(range(self.ndim))))


def _tensor_mT(self):
    if self.ndim < 2:
        raise RuntimeError("tensor.mT is only supported on matrices or batches of matrices")
    return transpose(self, -2, -1)


def _tensor_H(self):
    if self.ndim != 2:
        raise RuntimeError("tensor.H is only supported on matrices (2-D tensors)")
    return transpose(self, 0, 1)


Tensor.T = property(_tensor_T)
Tensor.H = property(_tensor_H)
Tensor.mT = property(_tensor_mT)
Tensor.mH = property(_tensor_mT)


def _is_dim_sequence(dim):
    return isinstance(dim, (tuple, list, Size))


def _normalized_dims_for(input, dim, *, empty_reduces_all):
    if not _is_dim_sequence(dim):
        return None
    if len(dim) == 0:
        return tuple(range(input.ndim)) if empty_reduces_all else ()
    normalized = []
    rank = input.ndim
    for item in dim:
        value = _builtins.int(item)
        if value < 0:
            value += rank
        if value < 0 or value >= rank:
            raise IndexError("dimension out of range")
        if value in normalized:
            raise RuntimeError("dim appears multiple times in the list of dims")
        normalized.append(value)
    return tuple(normalized)


def _tail_reduction_start(input, dims):
    if not dims:
        return None
    ordered = tuple(sorted(dims))
    start = ordered[0]
    if ordered == tuple(range(start, input.ndim)):
        return start
    return None


def _tail_reduction_native_supported(input, start):
    if input.is_contiguous():
        return True
    if start == 1 and input.ndim == 4 and input.is_contiguous(memory_format=channels_last):
        return True
    if start == 1 and input.ndim == 5 and input.is_contiguous(memory_format=channels_last_3d):
        return True
    return False


def _reduce_sequence(input, dim, keepdim, reducer, *, dtype=None, empty_reduces_all):
    dims = _normalized_dims_for(input, dim, empty_reduces_all=empty_reduces_all)
    if dims is None:
        if dtype is None:
            return reducer(input, dim=dim, keepdim=keepdim)
        return reducer(input, dim=dim, keepdim=keepdim, dtype=dtype)
    if not dims:
        return input
    result = input
    for axis in sorted(dims, reverse=True):
        if dtype is None:
            result = reducer(result, dim=axis, keepdim=keepdim)
        else:
            result = reducer(result, dim=axis, keepdim=keepdim, dtype=dtype)
    return result


def sum(input, dim=None, keepdim=False, dtype=None):
    return _reduce_sequence(input, dim, keepdim, _sum, dtype=dtype, empty_reduces_all=True)


def mean(input, dim=None, keepdim=False, dtype=None):
    return _reduce_sequence(input, dim, keepdim, _mean, dtype=dtype, empty_reduces_all=True)


def all(input, dim=None, keepdim=False):
    dims = _normalized_dims_for(input, dim, empty_reduces_all=False)
    if dims == ():
        return input.bool()
    return _reduce_sequence(input, dim, keepdim, _all, empty_reduces_all=False)


def any(input, dim=None, keepdim=False):
    dims = _normalized_dims_for(input, dim, empty_reduces_all=False)
    if dims == ():
        return input.bool()
    return _reduce_sequence(input, dim, keepdim, _any, empty_reduces_all=False)


def _reduction_correction(unbiased=None, correction=None):
    if correction is not None:
        return _builtins.float(correction)
    if unbiased is None:
        return 1.0
    return 1.0 if _builtins.bool(unbiased) else 0.0


def _variance_over_dims(input, dim, unbiased=None, keepdim=False, correction=None, *, take_sqrt=False):
    dims = _normalized_dims_for(input, dim, empty_reduces_all=True)
    if dims is None:
        reducer = _std if take_sqrt else _var
        return reducer(input, dim=dim, unbiased=True if unbiased is None else unbiased, keepdim=keepdim, correction=correction)
    if not dims:
        dims = tuple(range(input.ndim))
    tail_start = _tail_reduction_start(input, dims)
    if (
        tail_start is not None
        and not getattr(input, "requires_grad", False)
        and _tail_reduction_native_supported(input, tail_start)
    ):
        reducer = _std_tail_native if take_sqrt else _var_tail_native
        return reducer(
            input,
            tail_start,
            keepdim=keepdim,
            correction=_reduction_correction(unbiased, correction),
        )
    centered = input - mean(input, dim=dims, keepdim=True)
    total = sum(centered * centered, dim=dims, keepdim=keepdim)
    count = 1
    for axis in dims:
        count *= input.shape[axis]
    variance = total / (_builtins.float(count) - _reduction_correction(unbiased, correction))
    return sqrt(variance) if take_sqrt else variance


def var(input, dim=None, unbiased=None, keepdim=False, *, correction=None):
    return _variance_over_dims(input, dim, unbiased, keepdim, correction)


def std(input, dim=None, unbiased=None, keepdim=False, *, correction=None):
    return _variance_over_dims(input, dim, unbiased, keepdim, correction, take_sqrt=True)


def _variance_pair_over_dims(input, dim, unbiased=None, keepdim=False, correction=None, *, take_sqrt=False):
    if correction is not None and unbiased is not None:
        name = "std_mean" if take_sqrt else "var_mean"
        raise TypeError(f"{name}() received both correction and unbiased")
    if isinstance(dim, _builtins.bool) and unbiased is None:
        unbiased = dim
        dim = None

    dims = _normalized_dims_for(input, dim, empty_reduces_all=True)
    if dims is None and not getattr(input, "requires_grad", False):
        native = _std_mean_native if take_sqrt else _var_mean_native
        return native(input, dim=dim, unbiased=True if unbiased is None else unbiased, keepdim=keepdim, correction=correction)

    variance = _variance_over_dims(input, dim, unbiased, keepdim, correction)
    if take_sqrt:
        variance = sqrt(variance)
    return variance, mean(input, dim=dim, keepdim=keepdim)


def var_mean(input, dim=None, unbiased=None, keepdim=False, *, correction=None):
    return _variance_pair_over_dims(input, dim, unbiased, keepdim, correction, take_sqrt=False)


def std_mean(input, dim=None, unbiased=None, keepdim=False, *, correction=None):
    return _variance_pair_over_dims(input, dim, unbiased, keepdim, correction, take_sqrt=True)


Tensor.sum = lambda self, dim=None, keepdim=False, dtype=None: sum(self, dim=dim, keepdim=keepdim, dtype=dtype)
Tensor.mean = lambda self, dim=None, keepdim=False, dtype=None: mean(self, dim=dim, keepdim=keepdim, dtype=dtype)
Tensor.all = lambda self, dim=None, keepdim=False: all(self, dim=dim, keepdim=keepdim)
Tensor.any = lambda self, dim=None, keepdim=False: any(self, dim=dim, keepdim=keepdim)
Tensor.var = lambda self, dim=None, unbiased=None, keepdim=False, *, correction=None: var(
    self, dim=dim, unbiased=unbiased, keepdim=keepdim, correction=correction
)
Tensor.std = lambda self, dim=None, unbiased=None, keepdim=False, *, correction=None: std(
    self, dim=dim, unbiased=unbiased, keepdim=keepdim, correction=correction
)
count_nonzero = _C.count_nonzero
bincount = _C.bincount
Tensor.bincount = lambda self, weights=None, minlength=0: bincount(self, weights=weights, minlength=minlength)
relu = _C.relu
selu = _C.selu
clone = _C.clone
numel = _C.numel
is_tensor = _C.is_tensor
is_floating_point = _C.is_floating_point
is_complex = _C.is_complex
is_conj = _C.is_conj
is_signed = _C.is_signed


def _single_or_tuple(values):
    return values[0] if len(values) == 1 else tuple(values)


def atleast_1d(*tensors):
    values = tuple(reshape(tensor, (1,)) if tensor.ndim == 0 else tensor for tensor in tensors)
    return _single_or_tuple(values)


def atleast_2d(*tensors):
    values = []
    for tensor in tensors:
        if tensor.ndim == 0:
            values.append(reshape(tensor, (1, 1)))
        elif tensor.ndim == 1:
            values.append(unsqueeze(tensor, 0))
        else:
            values.append(tensor)
    return _single_or_tuple(tuple(values))


def atleast_3d(*tensors):
    values = []
    for tensor in tensors:
        if tensor.ndim == 0:
            values.append(reshape(tensor, (1, 1, 1)))
        elif tensor.ndim == 1:
            values.append(reshape(tensor, (1, tensor.shape[0], 1)))
        elif tensor.ndim == 2:
            values.append(unsqueeze(tensor, -1))
        else:
            values.append(tensor)
    return _single_or_tuple(tuple(values))


def hstack(tensors):
    values = atleast_1d(*tuple(tensors))
    if not isinstance(values, tuple):
        values = (values,)
    dim = 0 if values[0].ndim == 1 else 1
    return cat(values, dim=dim)


def vstack(tensors):
    values = atleast_2d(*tuple(tensors))
    if not isinstance(values, tuple):
        values = (values,)
    return cat(values, dim=0)


def row_stack(tensors):
    return vstack(tensors)


def dstack(tensors):
    values = atleast_3d(*tuple(tensors))
    if not isinstance(values, tuple):
        values = (values,)
    return cat(values, dim=2)


def column_stack(tensors):
    values = []
    for tensor in tuple(tensors):
        if tensor.ndim == 0:
            values.append(reshape(tensor, (1, 1)))
        elif tensor.ndim == 1:
            values.append(reshape(tensor, (tensor.shape[0], 1)))
        else:
            values.append(tensor)
    return cat(tuple(values), dim=1)


hstack = _C.hstack
vstack = _C.vstack
row_stack = _C.row_stack
dstack = _C.dstack
column_stack = _C.column_stack
cartesian_prod = _C.cartesian_prod


def meshgrid(*tensors, indexing=None):
    if len(tensors) == 1 and isinstance(tensors[0], (tuple, list)):
        tensors = tuple(tensors[0])
    if indexing is None:
        indexing = "ij"
    if indexing not in {"ij", "xy"}:
        raise ValueError("meshgrid indexing must be 'ij' or 'xy'")
    if not tensors:
        return ()
    order = list(range(len(tensors)))
    if indexing == "xy" and len(tensors) >= 2:
        order[0], order[1] = order[1], order[0]
    ordered = [tensors[index] for index in order]
    full_shape = tuple(tensor.shape[0] for tensor in ordered)
    outputs = []
    for index, tensor in enumerate(ordered):
        shape = [1] * len(ordered)
        shape[index] = tensor.shape[0]
        outputs.append(broadcast_to(reshape(tensor, tuple(shape)), full_shape))
    if indexing == "xy" and len(outputs) >= 2:
        outputs[0], outputs[1] = outputs[1], outputs[0]
    return tuple(outputs)


def _linalg_multi_dot(tensors, *, out=None):
    if out is not None:
        raise NotImplementedError("linalg.multi_dot out= is not implemented yet")
    return _C.chain_matmul(*tuple(tensors))


def _linalg_vector_norm(input, ord=2, dim=None, keepdim=False, *, dtype=None, out=None):
    return norm(input, p=ord, dim=dim, keepdim=keepdim, dtype=dtype, out=out)


def _linalg_norm(input, ord=None, dim=None, keepdim=False, *, dtype=None, out=None):
    return norm(input, p=2.0 if ord is None else ord, dim=dim, keepdim=keepdim, dtype=dtype, out=out)


save = _C.save
load = _C.load


class _GradModeContext:
    def __init__(self, mode, set_immediately=False):
        self.mode = _builtins.bool(mode)
        self.previous = None
        if set_immediately:
            self.previous = _C._set_grad_enabled(self.mode)

    def __enter__(self):
        if self.previous is None:
            self.previous = _C._set_grad_enabled(self.mode)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.previous is not None:
            _C._set_grad_enabled(self.previous)
            self.previous = None
        return False

    def __call__(self, func):
        @_functools.wraps(func)
        def wrapped(*args, **kwargs):
            with _GradModeContext(self.mode):
                return func(*args, **kwargs)

        return wrapped


def is_grad_enabled():
    return _C._is_grad_enabled()


def positive(input):
    return input


def no_grad():
    return _GradModeContext(False)


def enable_grad():
    return _GradModeContext(True)


def set_grad_enabled(mode):
    return _GradModeContext(mode, set_immediately=True)


_inference_mode_enabled = False


class _InferenceModeContext:
    def __init__(self, mode=True):
        self.mode = _builtins.bool(mode)
        self.previous_inference = None
        self.previous_grad = None

    def __enter__(self):
        global _inference_mode_enabled
        self.previous_inference = _inference_mode_enabled
        self.previous_grad = _C._set_grad_enabled(not self.mode)
        _inference_mode_enabled = self.mode
        return self

    def __exit__(self, exc_type, exc, tb):
        global _inference_mode_enabled
        if self.previous_grad is not None:
            _C._set_grad_enabled(self.previous_grad)
            self.previous_grad = None
        if self.previous_inference is not None:
            _inference_mode_enabled = self.previous_inference
            self.previous_inference = None
        return False

    def __call__(self, func):
        @_functools.wraps(func)
        def wrapped(*args, **kwargs):
            with _InferenceModeContext(self.mode):
                return func(*args, **kwargs)

        return wrapped


def inference_mode(mode=True):
    return _InferenceModeContext(mode)


def is_inference_mode_enabled():
    return _inference_mode_enabled


nn = ModuleType("mtorch.nn")
functional = ModuleType("mtorch.nn.functional")
attention = ModuleType("mtorch.nn.attention")
init = ModuleType("mtorch.nn.init")
linalg = ModuleType("mtorch.linalg")
amp = ModuleType("mtorch.amp")
cuda = ModuleType("mtorch.cuda")
cuda_amp = ModuleType("mtorch.cuda.amp")
mps = ModuleType("mtorch.mps")
backends = ModuleType("mtorch.backends")
backends_cuda = ModuleType("mtorch.backends.cuda")
backends_mps = ModuleType("mtorch.backends.mps")
backends_cudnn = ModuleType("mtorch.backends.cudnn")
utils = ModuleType("mtorch.utils")
utils_checkpoint = ModuleType("mtorch.utils.checkpoint")


_autocast_stack = []
_autocast_enabled_by_device = {}


class autocast:
    def __init__(self, device_type="cuda", dtype=None, enabled=True, cache_enabled=None):
        self.device_type = str(device_type)
        self.dtype = dtype
        self.enabled = _builtins.bool(enabled)
        self.cache_enabled = cache_enabled

    def __enter__(self):
        previous = _autocast_enabled_by_device.get(self.device_type, False)
        _autocast_stack.append((self.device_type, previous))
        _autocast_enabled_by_device[self.device_type] = self.enabled
        return self

    def __exit__(self, exc_type, exc, tb):
        if _autocast_stack:
            device_type, previous = _autocast_stack.pop()
            _autocast_enabled_by_device[device_type] = previous
        return False

    def __call__(self, func):
        @_functools.wraps(func)
        def wrapped(*args, **kwargs):
            with type(self)(self.device_type, dtype=self.dtype, enabled=self.enabled, cache_enabled=self.cache_enabled):
                return func(*args, **kwargs)

        return wrapped


def is_autocast_enabled(device_type=None):
    if device_type is None:
        return _builtins.any(_autocast_enabled_by_device.values())
    return _autocast_enabled_by_device.get(str(device_type), False)


class _CudaAutocast(autocast):
    def __init__(self, enabled=True, dtype=None, cache_enabled=None):
        super().__init__("cuda", dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)


class GradScaler:
    def __init__(self, device="cuda", init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
        self.device = device
        self._scale = _builtins.float(init_scale)
        self._enabled = _builtins.bool(enabled)

    def scale(self, outputs):
        return outputs

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        if new_scale is not None:
            self._scale = _builtins.float(new_scale)
        return None

    def get_scale(self):
        return 1.0

    def is_enabled(self):
        return False

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None


class _BackendFlags:
    def __init__(self, **flags):
        for key, value in flags.items():
            setattr(self, key, value)


class SDPBackend(_IntEnum):
    ERROR = -1
    MATH = 0
    FLASH_ATTENTION = 1
    EFFICIENT_ATTENTION = 2
    CUDNN_ATTENTION = 3
    OVERRIDEABLE = 4


class SDPAParams:
    def __init__(self, query, key, value, attn_mask, dropout, is_causal, enable_gqa):
        self.query = query
        self.key = key
        self.value = value
        self.attn_mask = attn_mask
        self.dropout = _builtins.float(dropout)
        self.is_causal = _builtins.bool(is_causal)
        self.enable_gqa = _builtins.bool(enable_gqa)


_sdp_flash_enabled = True
_sdp_math_enabled = True
_sdp_mem_efficient_enabled = True
_sdp_cudnn_enabled = True


def _enable_flash_sdp(enabled):
    global _sdp_flash_enabled
    _sdp_flash_enabled = _builtins.bool(enabled)


def _flash_sdp_enabled():
    return _sdp_flash_enabled


def _enable_math_sdp(enabled):
    global _sdp_math_enabled
    _sdp_math_enabled = _builtins.bool(enabled)


def _math_sdp_enabled():
    return _sdp_math_enabled


def _enable_mem_efficient_sdp(enabled):
    global _sdp_mem_efficient_enabled
    _sdp_mem_efficient_enabled = _builtins.bool(enabled)


def _mem_efficient_sdp_enabled():
    return _sdp_mem_efficient_enabled


def _enable_cudnn_sdp(enabled):
    global _sdp_cudnn_enabled
    _sdp_cudnn_enabled = _builtins.bool(enabled)


def _cudnn_sdp_enabled():
    return _sdp_cudnn_enabled


def _can_use_flash_attention(params, debug=False):
    return False


def _can_use_efficient_attention(params, debug=False):
    return False


def _can_use_cudnn_attention(params, debug=False):
    return False


class _SdpKernelContext:
    def __init__(self, enable_flash=True, enable_math=True, enable_mem_efficient=True, enable_cudnn=True):
        self.requested = (
            _builtins.bool(enable_flash),
            _builtins.bool(enable_math),
            _builtins.bool(enable_mem_efficient),
            _builtins.bool(enable_cudnn),
        )
        self.previous = None

    def __enter__(self):
        global _sdp_flash_enabled, _sdp_math_enabled, _sdp_mem_efficient_enabled, _sdp_cudnn_enabled
        self.previous = (
            _sdp_flash_enabled,
            _sdp_math_enabled,
            _sdp_mem_efficient_enabled,
            _sdp_cudnn_enabled,
        )
        (
            _sdp_flash_enabled,
            _sdp_math_enabled,
            _sdp_mem_efficient_enabled,
            _sdp_cudnn_enabled,
        ) = self.requested
        return self

    def __exit__(self, exc_type, exc, tb):
        global _sdp_flash_enabled, _sdp_math_enabled, _sdp_mem_efficient_enabled, _sdp_cudnn_enabled
        if self.previous is not None:
            (
                _sdp_flash_enabled,
                _sdp_math_enabled,
                _sdp_mem_efficient_enabled,
                _sdp_cudnn_enabled,
            ) = self.previous
            self.previous = None
        return False

    def __call__(self, func):
        @_functools.wraps(func)
        def wrapped(*args, **kwargs):
            with type(self)(*self.requested):
                return func(*args, **kwargs)

        return wrapped


def _sdp_kernel(
    enable_flash=True,
    enable_math=True,
    enable_mem_efficient=True,
    enable_cudnn=True,
):
    return _SdpKernelContext(enable_flash, enable_math, enable_mem_efficient, enable_cudnn)


def _normalize_sdpa_backends(backends):
    if isinstance(backends, SDPBackend):
        return {backends}
    if isinstance(backends, (list, tuple, set, frozenset)):
        normalized = set()
        for backend in backends:
            try:
                normalized.add(backend if isinstance(backend, SDPBackend) else SDPBackend(backend))
            except ValueError:
                continue
        return normalized
    raise AssertionError("Backend must be an instance of SDPBackend or a list of SDPBackend instances")


def _sdpa_kernel(backends, set_priority=False):
    normalized = _normalize_sdpa_backends(backends)
    return _SdpKernelContext(
        SDPBackend.FLASH_ATTENTION in normalized,
        SDPBackend.MATH in normalized,
        SDPBackend.EFFICIENT_ATTENTION in normalized,
        SDPBackend.CUDNN_ATTENTION in normalized,
    )


def compile(
    model=None,
    *,
    fullgraph=False,
    dynamic=None,
    backend="inductor",
    mode=None,
    options=None,
    disable=False,
):
    def decorator(fn):
        return fn

    if model is None:
        return decorator
    return model


def _cuda_is_available():
    return False


def _cuda_device_count():
    return 0


def _cuda_empty_cache():
    return None


def _cuda_manual_seed(seed):
    return None


def _cuda_is_bf16_supported():
    return False


def _mps_is_available():
    return False


def _mps_is_built():
    return False


def _checkpoint(function, *args, use_reentrant=None, **kwargs):
    return function(*args, **kwargs)


class _RemovableHandle:
    _next_id = 0

    def __init__(self, hooks_dict):
        self.hooks_dict = hooks_dict
        self.id = _RemovableHandle._next_id
        _RemovableHandle._next_id += 1

    def remove(self):
        hooks_dict = self.hooks_dict
        if hooks_dict is not None:
            hooks_dict.pop(self.id, None)
            self.hooks_dict = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False


class _IncompatibleKeys(namedtuple("_IncompatibleKeys", ["missing_keys", "unexpected_keys"])):
    __slots__ = ()

    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()


class Module:
    training = True
    _state_registry_names = {"_modules", "_parameters", "_buffers", "_non_persistent_buffers_set", "_module_names"}

    def __init__(self):
        object.__setattr__(self, "training", True)

    def __call__(self, *args, **kwargs):
        forward_pre_hooks = self.__dict__.get("_forward_pre_hooks")
        if forward_pre_hooks:
            args, kwargs = self._call_forward_pre_hooks(args, kwargs, forward_pre_hooks)
        forward_hooks = self.__dict__.get("_forward_hooks")
        try:
            output = self.forward(*args, **kwargs)
        except Exception:
            if forward_hooks:
                self._call_forward_hooks(args, kwargs, None, forward_hooks, only_always=True)
            raise
        if forward_hooks:
            output = self._call_forward_hooks(args, kwargs, output, forward_hooks)
        return output

    def _call_forward_pre_hooks(self, args, kwargs, hooks):
        for hook, with_kwargs in list(hooks.values()):
            if with_kwargs:
                result = hook(self, args, kwargs)
                if result is not None:
                    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[1], dict):
                        raise RuntimeError("forward pre-hook with kwargs must return None or (args, kwargs)")
                    args, kwargs = tuple(result[0]), dict(result[1])
            else:
                result = hook(self, args)
                if result is not None:
                    args = result if isinstance(result, tuple) else (result,)
        return args, kwargs

    def _call_forward_hooks(self, args, kwargs, output, hooks, only_always=False):
        for hook, with_kwargs, always_call in list(hooks.values()):
            if only_always and not always_call:
                continue
            if with_kwargs:
                result = hook(self, args, kwargs, output)
            else:
                result = hook(self, args, output)
            if result is not None:
                output = result
        return output

    def register_forward_pre_hook(self, hook, *, prepend=False, with_kwargs=False):
        hooks = self.__dict__.get("_forward_pre_hooks")
        if not isinstance(hooks, OrderedDict):
            hooks = OrderedDict()
            object.__setattr__(self, "_forward_pre_hooks", hooks)
        handle = _RemovableHandle(hooks)
        hooks[handle.id] = (hook, _builtins.bool(with_kwargs))
        if prepend:
            hooks.move_to_end(handle.id, last=False)
        return handle

    def register_forward_hook(self, hook, *, prepend=False, with_kwargs=False, always_call=False):
        hooks = self.__dict__.get("_forward_hooks")
        if not isinstance(hooks, OrderedDict):
            hooks = OrderedDict()
            object.__setattr__(self, "_forward_hooks", hooks)
        handle = _RemovableHandle(hooks)
        hooks[handle.id] = (hook, _builtins.bool(with_kwargs), _builtins.bool(always_call))
        if prepend:
            hooks.move_to_end(handle.id, last=False)
        return handle

    def _register_hook(self, name, hook, prepend=False):
        hooks = self.__dict__.get(name)
        if not isinstance(hooks, OrderedDict):
            hooks = OrderedDict()
            object.__setattr__(self, name, hooks)
        handle = _RemovableHandle(hooks)
        hooks[handle.id] = hook
        if prepend:
            hooks.move_to_end(handle.id, last=False)
        return handle

    def register_state_dict_pre_hook(self, hook):
        return self._register_hook("_state_dict_pre_hooks", hook)

    def register_state_dict_post_hook(self, hook):
        return self._register_hook("_state_dict_post_hooks", hook)

    def register_load_state_dict_pre_hook(self, hook):
        return self._register_hook("_load_state_dict_pre_hooks", hook)

    def register_load_state_dict_post_hook(self, hook):
        return self._register_hook("_load_state_dict_post_hooks", hook)

    def __getattr__(self, name):
        parameters = self.__dict__.get("_parameters")
        if isinstance(parameters, dict) and name in parameters:
            return parameters[name]
        if isinstance(parameters, list) and name.isdigit():
            index = _builtins.int(name)
            if index < len(parameters):
                return parameters[index]
        buffers = self.__dict__.get("_buffers")
        if isinstance(buffers, dict) and name in buffers:
            return buffers[name]
        modules = self.__dict__.get("_modules")
        if isinstance(modules, dict) and name in modules:
            return modules[name]
        if isinstance(modules, list) and name.isdigit():
            index = _builtins.int(name)
            if index < len(modules):
                return modules[index]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in Module._state_registry_names:
            object.__setattr__(self, name, value)
            return

        parameters = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            self._remove_from_state_registries(name)
            if not isinstance(parameters, OrderedDict):
                parameters = OrderedDict()
                object.__setattr__(self, "_parameters", parameters)
            parameters[name] = value
            object.__setattr__(self, name, value)
            return
        if isinstance(parameters, dict) and name in parameters:
            if value is not None:
                raise TypeError(
                    f"cannot assign {type(value).__name__!r} as parameter {name!r} "
                    "(Parameter or None expected)"
                )
            parameters[name] = None
            object.__setattr__(self, name, None)
            return

        modules = self.__dict__.get("_modules")
        if isinstance(value, Module):
            self._remove_from_state_registries(name)
            if not isinstance(modules, OrderedDict):
                modules = OrderedDict()
                object.__setattr__(self, "_modules", modules)
            modules[name] = value
            object.__setattr__(self, name, value)
            return
        if isinstance(modules, dict) and name in modules:
            if value is not None:
                raise TypeError(
                    f"cannot assign {type(value).__name__!r} as child module {name!r} "
                    "(Module or None expected)"
                )
            modules[name] = None
            object.__setattr__(self, name, None)
            return

        buffers = self.__dict__.get("_buffers")
        if isinstance(buffers, dict) and name in buffers:
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(
                    f"cannot assign {type(value).__name__!r} as buffer {name!r} "
                    "(Tensor or None expected)"
                )
            buffers[name] = value
            object.__setattr__(self, name, value)
            return

        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        removed = self._remove_from_state_registries(name)
        if name in self.__dict__:
            object.__delattr__(self, name)
            return
        if removed:
            return
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _remove_from_state_registries(self, name):
        removed = False
        parameters = self.__dict__.get("_parameters")
        if isinstance(parameters, dict) and name in parameters:
            del parameters[name]
            removed = True
        modules = self.__dict__.get("_modules")
        if isinstance(modules, dict) and name in modules:
            del modules[name]
            removed = True
        buffers = self.__dict__.get("_buffers")
        if isinstance(buffers, dict) and name in buffers:
            del buffers[name]
            removed = True
        non_persistent = self.__dict__.get("_non_persistent_buffers_set")
        if isinstance(non_persistent, set):
            non_persistent.discard(name)
        return removed

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.forward is not implemented")

    def parameters(self, recurse=True):
        for _, parameter in self.named_parameters(recurse=recurse):
            yield parameter

    def register_parameter(self, name, param):
        self._validate_state_name(name)
        if param is not None and not isinstance(param, Parameter):
            raise TypeError("parameter must be a Parameter or None")
        self._remove_from_state_registries(name)
        parameters = self.__dict__.get("_parameters")
        if not isinstance(parameters, OrderedDict):
            parameters = OrderedDict()
            object.__setattr__(self, "_parameters", parameters)
        parameters[name] = param
        object.__setattr__(self, name, param)

    def add_module(self, name, module):
        self._validate_state_name(name)
        if module is not None and not isinstance(module, Module):
            raise TypeError(f"{type(module).__name__} is not a Module subclass")
        self._remove_from_state_registries(name)
        modules = self.__dict__.get("_modules")
        if not isinstance(modules, OrderedDict):
            modules = OrderedDict()
            object.__setattr__(self, "_modules", modules)
        modules[name] = module
        object.__setattr__(self, name, module)

    def register_module(self, name, module):
        return self.add_module(name, module)

    def register_buffer(self, name, tensor, persistent=True):
        self._validate_state_name(name)
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError("buffer must be a Tensor or None")
        self._remove_from_state_registries(name)
        buffers = self.__dict__.get("_buffers")
        if not isinstance(buffers, OrderedDict):
            buffers = OrderedDict()
            object.__setattr__(self, "_buffers", buffers)
        non_persistent = self.__dict__.get("_non_persistent_buffers_set")
        if not isinstance(non_persistent, set):
            non_persistent = set()
            object.__setattr__(self, "_non_persistent_buffers_set", non_persistent)
        buffers[name] = tensor
        if persistent:
            non_persistent.discard(name)
        else:
            non_persistent.add(name)
        object.__setattr__(self, name, tensor)

    def _validate_state_name(self, name):
        if not isinstance(name, str) or not name:
            raise KeyError("state name must be a non-empty string")
        if "." in name:
            raise KeyError("state name cannot contain '.'")

    def buffers(self, recurse=True):
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def children(self):
        for _, module in self.named_children():
            yield module

    def named_children(self):
        seen = set()
        registered_modules = getattr(self, "_modules", None)
        if isinstance(registered_modules, dict):
            for name, value in registered_modules.items():
                if isinstance(value, Module):
                    identity = id(value)
                    if identity not in seen:
                        seen.add(identity)
                        yield name, value
        for name, value in self.__dict__.items():
            if name in {"_modules", "_parameters", "_buffers", "_non_persistent_buffers_set"}:
                continue
            if isinstance(value, Module):
                identity = id(value)
                if identity not in seen:
                    seen.add(identity)
                    yield name, value
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    if isinstance(item, Module):
                        identity = id(item)
                        if identity not in seen:
                            seen.add(identity)
                            yield f"{name}.{index}", item
            elif isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, Module):
                        identity = id(item)
                        if identity not in seen:
                            seen.add(identity)
                            yield f"{name}.{key}", item

    def get_submodule(self, target):
        if not target:
            return self
        module = self
        for item in target.split("."):
            module = module._get_child_module(item) if isinstance(module, Module) else None
            if module is None:
                raise AttributeError(f"{type(self).__name__} has no submodule {target!r}")
        if not isinstance(module, Module):
            raise AttributeError(f"{target!r} is not a submodule")
        return module

    def _get_child_module(self, name):
        modules = self.__dict__.get("_modules")
        if isinstance(modules, dict) and name in modules:
            value = modules[name]
            return value if isinstance(value, Module) else None
        if isinstance(modules, list) and name.isdigit():
            index = _builtins.int(name)
            if index < len(modules):
                value = modules[index]
                return value if isinstance(value, Module) else None
        value = self.__dict__.get(name)
        return value if isinstance(value, Module) else None

    def _should_yield_state_value(self, seen, value):
        if seen is None:
            return True
        identity = id(value)
        if identity in seen:
            return False
        seen.add(identity)
        return True

    def get_parameter(self, target):
        module_path, _, parameter_name = target.rpartition(".")
        module = self.get_submodule(module_path) if module_path else self
        parameters = dict(module.named_parameters(recurse=False))
        if parameter_name not in parameters:
            raise AttributeError(f"{type(module).__name__} has no parameter {parameter_name!r}")
        return parameters[parameter_name]

    def get_buffer(self, target):
        module_path, _, buffer_name = target.rpartition(".")
        module = self.get_submodule(module_path) if module_path else self
        buffers = dict(module.named_buffers(recurse=False))
        if buffer_name not in buffers:
            raise AttributeError(f"{type(module).__name__} has no buffer {buffer_name!r}")
        return buffers[buffer_name]

    def modules(self):
        for _, module in self.named_modules():
            yield module

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        if memo is None:
            memo = set()
        if not remove_duplicate or self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self.named_children():
                submodule_prefix = f"{prefix}.{name}" if prefix else name
                yield from module.named_modules(memo, submodule_prefix, remove_duplicate)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        seen = set() if remove_duplicate else None
        yield from self._named_parameters(prefix, recurse, seen)

    def _named_parameters(self, prefix, recurse, seen):
        registered_parameters = getattr(self, "_parameters", {})
        registered_parameter_names = set(registered_parameters) if isinstance(registered_parameters, dict) else set()
        if isinstance(registered_parameters, dict):
            for name, value in registered_parameters.items():
                if value is None:
                    continue
                if self._should_yield_state_value(seen, value):
                    yield (f"{prefix}.{name}" if prefix else name, value)
        registered_modules = getattr(self, "_modules", None)
        registered_module_names = set(registered_modules) if isinstance(registered_modules, dict) else set()
        if recurse and isinstance(registered_modules, dict):
            for name, value in registered_modules.items():
                if isinstance(value, Module):
                    child_prefix = f"{prefix}.{name}" if prefix else name
                    yield from value._named_parameters(child_prefix, recurse, seen)
        registered_buffer_names = set(getattr(self, "_buffers", {}))
        for name, value in self.__dict__.items():
            if (
                name in Module._state_registry_names
                or name in registered_parameter_names
                or name in registered_module_names
                or name in registered_buffer_names
            ):
                continue
            if isinstance(value, Parameter):
                if self._should_yield_state_value(seen, value):
                    yield (f"{prefix}.{name}" if prefix else name, value)
            elif recurse and isinstance(value, Module):
                child_prefix = f"{prefix}.{name}" if prefix else name
                yield from value._named_parameters(child_prefix, recurse, seen)
            elif recurse and isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    if isinstance(item, Module):
                        child_prefix = f"{prefix}.{name}.{index}" if prefix else f"{name}.{index}"
                        yield from item._named_parameters(child_prefix, recurse, seen)
            elif recurse and isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, Module):
                        child_prefix = f"{prefix}.{name}.{key}" if prefix else f"{name}.{key}"
                        yield from item._named_parameters(child_prefix, recurse, seen)

    def named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        seen = set()
        yield from self._named_buffers(prefix, recurse, seen, remove_duplicate, persistent_only=False)

    def _named_buffers(self, prefix, recurse, seen, remove_duplicate, persistent_only):
        non_persistent = getattr(self, "_non_persistent_buffers_set", set())
        for name, value in getattr(self, "_buffers", {}).items():
            if value is None or (persistent_only and name in non_persistent):
                continue
            identity = id(value)
            if not remove_duplicate or identity not in seen:
                seen.add(identity)
                yield (f"{prefix}.{name}" if prefix else name, value)
        if recurse:
            for name, module in self.named_children():
                child_prefix = f"{prefix}.{name}" if prefix else name
                yield from module._named_buffers(child_prefix, recurse, seen, remove_duplicate, persistent_only)

    def train(self, mode=True):
        self.training = _builtins.bool(mode)
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.train(mode)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.train(mode)
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def requires_grad_(self, requires_grad=True):
        for parameter in self.parameters():
            parameter.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters():
            if parameter.grad is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad.zero_()

    def to(self, *args, **kwargs):
        target_device, target_dtype, target_memory_format = _parse_module_to_args(args, kwargs)
        self._to_inplace(target_device, target_dtype, target_memory_format, set())
        return self

    def cpu(self):
        return self.to(device="cpu")

    def float(self):
        return self.to(dtype=float32)

    def double(self):
        return self.to(dtype=float64)

    def half(self):
        return self.to(dtype=float16)

    def _to_inplace(self, target_device, target_dtype, target_memory_format, seen_modules):
        identity = id(self)
        if identity in seen_modules:
            return
        seen_modules.add(identity)

        parameter_names = set()
        registered_parameters = getattr(self, "_parameters", None)
        if isinstance(registered_parameters, dict):
            for name, value in list(registered_parameters.items()):
                parameter_names.add(name)
                converted = _convert_module_tensor(value, target_device, target_dtype, target_memory_format)
                registered_parameters[name] = converted
                if name in self.__dict__:
                    object.__setattr__(self, name, converted)
        elif isinstance(registered_parameters, list):
            for index, value in enumerate(list(registered_parameters)):
                registered_parameters[index] = _convert_module_tensor(
                    value,
                    target_device,
                    target_dtype,
                    target_memory_format,
                )

        buffer_names = set()
        registered_buffers = getattr(self, "_buffers", None)
        if isinstance(registered_buffers, dict):
            for name, value in list(registered_buffers.items()):
                buffer_names.add(name)
                converted = _convert_module_tensor(value, target_device, target_dtype, target_memory_format)
                registered_buffers[name] = converted
                if name in self.__dict__:
                    object.__setattr__(self, name, converted)

        for name, value in list(self.__dict__.items()):
            if name in {"_parameters", "_buffers", "_non_persistent_buffers_set"} or name in parameter_names or name in buffer_names:
                continue
            converted = _convert_module_state_value(value, target_device, target_dtype, target_memory_format, seen_modules)
            if converted is not value:
                setattr(self, name, converted)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for hook in list(self.__dict__.get("_state_dict_pre_hooks", {}).values()):
            hook(self, prefix, keep_vars)
        for name, parameter in self.named_parameters():
            key = f"{prefix}{name}"
            destination[key] = parameter if keep_vars else parameter.detach()
        buffer_seen = set()
        for name, buffer in self._named_buffers("", True, buffer_seen, True, persistent_only=True):
            key = f"{prefix}{name}"
            destination[key] = buffer if keep_vars else buffer.detach()
        local_metadata = {"version": getattr(self, "_version", None)}
        for hook in list(self.__dict__.get("_state_dict_post_hooks", {}).values()):
            result = hook(self, destination, prefix, local_metadata)
            if result is not None:
                raise RuntimeError("state_dict post-hook must return None")
        return destination

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if assign:
            raise NotImplementedError("load_state_dict assign=True is not implemented yet")
        missing = []
        unexpected = []
        error_msgs = []
        local_metadata = {"version": getattr(self, "_version", None)}
        for hook in list(self.__dict__.get("_load_state_dict_pre_hooks", {}).values()):
            hook(self, state_dict, "", local_metadata, strict, missing, unexpected, error_msgs)
        named = dict(self.named_parameters())
        buffer_seen = set()
        named.update(dict(self._named_buffers("", True, buffer_seen, True, persistent_only=True)))
        for name, parameter in named.items():
            if name not in state_dict:
                missing.append(name)
                continue
            parameter.copy_(state_dict[name])
        for name in state_dict:
            if name not in named:
                unexpected.append(name)
        incompatible_keys = _IncompatibleKeys(missing, unexpected)
        for hook in list(self.__dict__.get("_load_state_dict_post_hooks", {}).values()):
            hook(self, incompatible_keys)
        if error_msgs:
            raise RuntimeError("; ".join(error_msgs))
        if strict:
            if missing or unexpected:
                details = []
                if missing:
                    details.append(f"Missing key(s): {missing!r}")
                if unexpected:
                    details.append(f"Unexpected key(s): {unexpected!r}")
                raise RuntimeError("; ".join(details))
        return incompatible_keys


class _ParameterMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, Tensor) and _C._is_parameter(instance)


class Parameter(metaclass=_ParameterMeta):
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = []
        if isinstance(data, Tensor):
            result = data.detach()
            result.requires_grad_(requires_grad)
        else:
            result = tensor(data, requires_grad=requires_grad)
        _C._mark_parameter(result)
        return result


def _as_parameter(value):
    if value is None or isinstance(value, Parameter):
        return value
    return Parameter(value)


_DTYPE_NAMES = {"float16", "float32", "float64", "int32", "int64", "bool"}
_DEVICE_NAMES = {"cpu", "mps", "metal"}


def _is_dtype_arg(value):
    return isinstance(value, dtype) or isinstance(value, str) and value in _DTYPE_NAMES


def _is_device_arg(value):
    if isinstance(value, device):
        return True
    if isinstance(value, str):
        return value.partition(":")[0] in _DEVICE_NAMES
    return False


def _parse_module_to_args(args, kwargs):
    kwargs = dict(kwargs)
    memory_format = kwargs.pop("memory_format", None)
    memory_format = _normalize_memory_format(memory_format)
    kwargs.pop("non_blocking", None)
    target_device = kwargs.pop("device", None)
    target_dtype = kwargs.pop("dtype", None)
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"to() got an unexpected keyword argument {unexpected!r}")
    for arg in args:
        if isinstance(arg, Tensor):
            target_device = arg.device
            target_dtype = arg.dtype
        elif _is_dtype_arg(arg):
            target_dtype = arg
        elif _is_device_arg(arg):
            target_device = arg
        else:
            raise TypeError("Module.to expected a dtype, device, Tensor, or keyword arguments")
    if target_dtype is not None and str(target_dtype) not in {"float16", "float32", "float64"}:
        raise TypeError("Module.to only accepts floating point dtypes")
    return target_device, target_dtype, memory_format


def _module_tensor_memory_format(value, requested):
    if requested is None or requested is preserve_format:
        return None
    if requested is channels_last:
        return requested if len(value.shape) == 4 else None
    if requested is channels_last_3d:
        return requested if len(value.shape) == 5 else None
    return requested


def _convert_module_tensor(value, target_device, target_dtype, target_memory_format):
    if value is None:
        return None
    was_parameter = isinstance(value, Parameter)
    kwargs = {}
    if target_device is not None:
        kwargs["device"] = target_device
    if target_dtype is not None and value.is_floating_point():
        kwargs["dtype"] = target_dtype
    memory_format_value = _module_tensor_memory_format(value, target_memory_format)
    if memory_format_value is not None:
        kwargs["memory_format"] = memory_format_value
    if not kwargs:
        return value
    converted = value.to(**kwargs)
    if was_parameter:
        _C._mark_parameter(converted)
    return converted


def _convert_module_state_value(value, target_device, target_dtype, target_memory_format, seen_modules):
    if isinstance(value, Tensor):
        return _convert_module_tensor(value, target_device, target_dtype, target_memory_format)
    if isinstance(value, Module):
        value._to_inplace(target_device, target_dtype, target_memory_format, seen_modules)
        return value
    if isinstance(value, list):
        changed = False
        converted = []
        for item in value:
            new_item = _convert_module_state_value(item, target_device, target_dtype, target_memory_format, seen_modules)
            changed = changed or new_item is not item
            converted.append(new_item)
        return converted if changed else value
    if isinstance(value, tuple):
        converted = tuple(
            _convert_module_state_value(item, target_device, target_dtype, target_memory_format, seen_modules)
            for item in value
        )
        return converted if _builtins.any(new_item is not item for new_item, item in zip(converted, value, strict=True)) else value
    if isinstance(value, dict):
        changed = False
        converted = type(value)()
        for key, item in value.items():
            new_item = _convert_module_state_value(item, target_device, target_dtype, target_memory_format, seen_modules)
            changed = changed or new_item is not item
            converted[key] = new_item
        return converted if changed else value
    return value


def _pair(value):
    if isinstance(value, (tuple, list, Size)):
        if len(value) != 2:
            raise ValueError("expected an int or a pair")
        return (_builtins.int(value[0]), _builtins.int(value[1]))
    item = _builtins.int(value)
    return (item, item)


def _triple(value):
    if isinstance(value, (tuple, list, Size)):
        if len(value) != 3:
            raise ValueError("expected an int or a triple")
        return (_builtins.int(value[0]), _builtins.int(value[1]), _builtins.int(value[2]))
    item = _builtins.int(value)
    return (item, item, item)


def _single(value):
    if isinstance(value, (tuple, list, Size)):
        if len(value) != 1:
            raise ValueError("expected an int or a single-element tuple")
        return (_builtins.int(value[0]),)
    return (_builtins.int(value),)


def _one_hot(input, num_classes=-1):
    return _C.one_hot(input, _builtins.int(num_classes))


def _cosine_similarity(x1, x2, dim=1, eps=1e-8):
    numerator = sum(x1 * x2, dim=dim)
    x1_norm = sqrt(sum(x1 * x1, dim=dim))
    x2_norm = sqrt(sum(x2 * x2, dim=dim))
    denominator = clamp_min(x1_norm, eps) * clamp_min(x2_norm, eps)
    return numerator / denominator


def tensor_split(input, indices_or_sections, dim=0):
    rank = input.dim()
    split_dim = _builtins.int(dim)
    if split_dim < 0:
        split_dim += rank
    if split_dim < 0 or split_dim >= rank:
        raise IndexError("dimension out of range")
    dim_size = _builtins.int(input.shape[split_dim])

    if isinstance(indices_or_sections, _builtins.int):
        sections = _builtins.int(indices_or_sections)
        if sections <= 0:
            raise ValueError("number of sections must be larger than 0")
        base = dim_size // sections
        remainder = dim_size % sections
        result = []
        start = 0
        for section in range(sections):
            length = base + (1 if section < remainder else 0)
            result.append(narrow(input, split_dim, start, length))
            start += length
        return tuple(result)

    previous = 0
    result = []
    for raw_index in indices_or_sections:
        index = _builtins.int(raw_index)
        if index < 0:
            index += dim_size
        start = _builtins.min(_builtins.max(previous, 0), dim_size)
        end = _builtins.min(_builtins.max(index, 0), dim_size)
        result.append(narrow(input, split_dim, start, _builtins.max(0, end - start)))
        previous = index
    start = _builtins.min(_builtins.max(previous, 0), dim_size)
    result.append(narrow(input, split_dim, start, dim_size - start))
    return tuple(result)


def _tensor_tensor_split(self, indices_or_sections, dim=0):
    return tensor_split(self, indices_or_sections, dim=dim)


Tensor.tensor_split = _tensor_tensor_split


def _pixel_shuffle(input, upscale_factor):
    factor = _builtins.int(upscale_factor)
    return _C.pixel_shuffle(input, factor)


def _pixel_unshuffle(input, downscale_factor):
    factor = _builtins.int(downscale_factor)
    return _C.pixel_unshuffle(input, factor)


def _channel_shuffle(input, groups):
    group_count = _builtins.int(groups)
    return _C.channel_shuffle(input, group_count)


def _adaptive_avg_pool1d(input, output_size):
    if isinstance(output_size, (tuple, list, Size)):
        if len(output_size) != 1:
            raise ValueError("expected an int or a single-element tuple")
        size = (_builtins.int(output_size[0]),)
    else:
        size = (_builtins.int(output_size),)
    return _C.adaptive_avg_pool1d(input, size)


def _adaptive_avg_pool2d(input, output_size):
    if isinstance(output_size, (tuple, list, Size)):
        if len(output_size) != 2:
            raise ValueError("expected an int or a pair")
        size = tuple(
            _builtins.int(input.shape[-2 + index]) if value is None else _builtins.int(value)
            for index, value in enumerate(output_size)
        )
    else:
        size = _pair(output_size)
    return _C.adaptive_avg_pool2d(input, size)


cosine_similarity = _cosine_similarity
pixel_shuffle = _pixel_shuffle
pixel_unshuffle = _pixel_unshuffle
channel_shuffle = _channel_shuffle
adaptive_avg_pool1d = _adaptive_avg_pool1d


def _validate_conv_string_padding(padding, stride):
    if padding not in {"same", "valid"}:
        raise ValueError("padding must be 'same', 'valid', an int, or a tuple")
    if padding == "same" and _builtins.any(_builtins.int(item) != 1 for item in stride):
        raise ValueError("padding='same' is not supported for strided convolutions")


def _reverse_repeat_padding(padding):
    values = tuple(_builtins.int(value) for value in padding)
    result = []
    for value in reversed(values):
        result.extend((value, value))
    return tuple(result)


def _conv_string_padding(input, weight, stride, padding, dilation, spatial_dims, mode="constant", force_explicit=False):
    if not isinstance(padding, str):
        return None
    if spatial_dims == 1:
        stride_values = _single(stride)
    elif spatial_dims == 2:
        stride_values = _pair(stride)
    else:
        stride_values = _triple(stride)
    _validate_conv_string_padding(padding, stride_values)
    if padding == "valid":
        return tuple(0 for _ in range(spatial_dims)), None

    if spatial_dims == 1:
        dilation_values = _single(dilation)
    elif spatial_dims == 2:
        dilation_values = _pair(dilation)
    else:
        dilation_values = _triple(dilation)
    kernel_shape = tuple(_builtins.int(item) for item in weight.shape[-spatial_dims:])
    before_after = []
    symmetric_padding = []
    for kernel, dilation_value in zip(kernel_shape, dilation_values, strict=True):
        total = dilation_value * (kernel - 1)
        before = total // 2
        after = total - before
        before_after.append((before, after))
        symmetric_padding.append(before)
    if not force_explicit and _builtins.all(before == after for before, after in before_after):
        return tuple(symmetric_padding), None

    pad_values = []
    for before, after in reversed(before_after):
        pad_values.extend((before, after))
    return tuple(0 for _ in range(spatial_dims)), _C.pad(input, tuple(pad_values), mode, 0.0 if mode == "constant" else None)


def _conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if not isinstance(padding, str):
        return _C.conv1d(input, weight, bias, stride, padding, dilation, groups)
    string_padding = _conv_string_padding(input, weight, stride, padding, dilation, 1)
    if string_padding is not None:
        padding, padded_input = string_padding
        input = input if padded_input is None else padded_input
    return _C.conv1d(input, weight, bias, stride, padding, dilation, groups)


conv1d = _conv1d


def _conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if not isinstance(padding, str):
        return _C.conv2d(input, weight, bias, stride, padding, dilation, groups)
    string_padding = _conv_string_padding(input, weight, stride, padding, dilation, 2)
    if string_padding is not None:
        padding, padded_input = string_padding
        input = input if padded_input is None else padded_input
    return _C.conv2d(input, weight, bias, stride, padding, dilation, groups)


conv2d = _conv2d


def _conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if not isinstance(padding, str):
        return _C.conv3d(input, weight, bias, stride, padding, dilation, groups)
    string_padding = _conv_string_padding(input, weight, stride, padding, dilation, 3)
    if string_padding is not None:
        padding, padded_input = string_padding
        input = input if padded_input is None else padded_input
    return _C.conv3d(input, weight, bias, stride, padding, dilation, groups)


conv3d = _conv3d


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        self.in_features = _builtins.int(in_features)
        self.out_features = _builtins.int(out_features)
        self.weight = Parameter(tensor(
            [
                [self._initial_weight(row, col) for col in range(self.in_features)]
                for row in range(self.out_features)
            ],
            dtype=dtype or float32,
            requires_grad=True,
            device=device or "cpu",
        ))
        self.bias = (
            Parameter(tensor(
                [0.0 for _ in range(self.out_features)],
                dtype=dtype or float32,
                requires_grad=True,
                device=device or "cpu",
            ))
            if bias
            else None
        )

    def _initial_weight(self, row, col):
        span = _builtins.max(1, self.in_features)
        return (((row * self.in_features + col) % (2 * span + 1)) - span) / (span * span)

    def forward(self, input):
        return functional.linear(input, self.weight, self.bias)


class Conv1d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        self.in_channels = _builtins.int(in_channels)
        self.out_channels = _builtins.int(out_channels)
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        if isinstance(padding, str):
            _validate_conv_string_padding(padding, self.stride)
            self.padding = padding
        else:
            self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.groups = _builtins.int(groups)
        self.padding_mode = padding_mode
        if self.groups <= 0:
            raise ValueError("groups must be a positive integer")
        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if self.out_channels % self.groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError("padding_mode must be one of 'zeros', 'reflect', 'replicate', or 'circular'")
        channels_per_group = self.in_channels // self.groups
        self.weight = Parameter(tensor(
            [
                [
                    [self._initial_weight(out, channel, kernel_x) for kernel_x in range(self.kernel_size[0])]
                    for channel in range(channels_per_group)
                ]
                for out in range(self.out_channels)
            ],
            dtype=dtype or float32,
            requires_grad=True,
            device=device or "cpu",
        ))
        self.bias = (
            Parameter(tensor(
                [0.0 for _ in range(self.out_channels)],
                dtype=dtype or float32,
                requires_grad=True,
                device=device or "cpu",
            ))
            if bias
            else None
        )

    def _initial_weight(self, out, channel, kernel_x):
        channels_per_group = self.in_channels // self.groups
        fan_in = _builtins.max(1, channels_per_group * self.kernel_size[0])
        position = (out * channels_per_group + channel) * self.kernel_size[0] + kernel_x
        return (((position % (2 * fan_in + 1)) - fan_in) / (fan_in * fan_in))

    def forward(self, input):
        padding = self.padding
        if self.padding_mode != "zeros":
            string_padding = _conv_string_padding(
                input,
                self.weight,
                self.stride,
                padding,
                self.dilation,
                1,
                mode=self.padding_mode,
                force_explicit=True,
            )
            if string_padding is not None:
                padding, padded_input = string_padding
                input = input if padded_input is None else padded_input
            else:
                input = functional.pad(input, _reverse_repeat_padding(padding), self.padding_mode)
                padding = (0,)
        return functional.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )


class Conv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        self.in_channels = _builtins.int(in_channels)
        self.out_channels = _builtins.int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        if isinstance(padding, str):
            _validate_conv_string_padding(padding, self.stride)
            self.padding = padding
        else:
            self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = _builtins.int(groups)
        self.padding_mode = padding_mode
        if self.groups <= 0:
            raise ValueError("groups must be a positive integer")
        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if self.out_channels % self.groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError("padding_mode must be one of 'zeros', 'reflect', 'replicate', or 'circular'")
        channels_per_group = self.in_channels // self.groups
        self.weight = Parameter(tensor(
            [
                [
                    [
                        [
                            self._initial_weight(out, channel, kernel_y, kernel_x)
                            for kernel_x in range(self.kernel_size[1])
                        ]
                        for kernel_y in range(self.kernel_size[0])
                    ]
                    for channel in range(channels_per_group)
                ]
                for out in range(self.out_channels)
            ],
            dtype=dtype or float32,
            requires_grad=True,
            device=device or "cpu",
        ))
        self.bias = (
            Parameter(tensor(
                [0.0 for _ in range(self.out_channels)],
                dtype=dtype or float32,
                requires_grad=True,
                device=device or "cpu",
            ))
            if bias
            else None
        )

    def _initial_weight(self, out, channel, kernel_y, kernel_x):
        channels_per_group = self.in_channels // self.groups
        fan_in = _builtins.max(1, channels_per_group * self.kernel_size[0] * self.kernel_size[1])
        position = ((out * channels_per_group + channel) * self.kernel_size[0] + kernel_y) * self.kernel_size[1] + kernel_x
        return (((position % (2 * fan_in + 1)) - fan_in) / (fan_in * fan_in))

    def forward(self, input):
        padding = self.padding
        if self.padding_mode != "zeros":
            string_padding = _conv_string_padding(
                input,
                self.weight,
                self.stride,
                padding,
                self.dilation,
                2,
                mode=self.padding_mode,
                force_explicit=True,
            )
            if string_padding is not None:
                padding, padded_input = string_padding
                input = input if padded_input is None else padded_input
            else:
                input = functional.pad(input, _reverse_repeat_padding(padding), self.padding_mode)
                padding = (0, 0)
        return functional.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )


class Conv3d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        self.in_channels = _builtins.int(in_channels)
        self.out_channels = _builtins.int(out_channels)
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        if isinstance(padding, str):
            _validate_conv_string_padding(padding, self.stride)
            self.padding = padding
        else:
            self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = _builtins.int(groups)
        self.padding_mode = padding_mode
        if self.groups <= 0:
            raise ValueError("groups must be a positive integer")
        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if self.out_channels % self.groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError("padding_mode must be one of 'zeros', 'reflect', 'replicate', or 'circular'")
        channels_per_group = self.in_channels // self.groups
        self.weight = Parameter(tensor(
            [
                [
                    [
                        [
                            [
                                self._initial_weight(out, channel, kernel_z, kernel_y, kernel_x)
                                for kernel_x in range(self.kernel_size[2])
                            ]
                            for kernel_y in range(self.kernel_size[1])
                        ]
                        for kernel_z in range(self.kernel_size[0])
                    ]
                    for channel in range(channels_per_group)
                ]
                for out in range(self.out_channels)
            ],
            dtype=dtype or float32,
            requires_grad=True,
            device=device or "cpu",
        ))
        self.bias = (
            Parameter(tensor(
                [0.0 for _ in range(self.out_channels)],
                dtype=dtype or float32,
                requires_grad=True,
                device=device or "cpu",
            ))
            if bias
            else None
        )

    def _initial_weight(self, out, channel, kernel_z, kernel_y, kernel_x):
        channels_per_group = self.in_channels // self.groups
        fan_in = _builtins.max(1, channels_per_group * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2])
        position = (
            (((out * channels_per_group + channel) * self.kernel_size[0] + kernel_z) * self.kernel_size[1] + kernel_y)
            * self.kernel_size[2]
            + kernel_x
        )
        return (((position % (2 * fan_in + 1)) - fan_in) / (fan_in * fan_in))

    def forward(self, input):
        padding = self.padding
        if self.padding_mode != "zeros":
            string_padding = _conv_string_padding(
                input,
                self.weight,
                self.stride,
                padding,
                self.dilation,
                3,
                mode=self.padding_mode,
                force_explicit=True,
            )
            if string_padding is not None:
                padding, padded_input = string_padding
                input = input if padded_input is None else padded_input
            else:
                input = functional.pad(input, _reverse_repeat_padding(padding), self.padding_mode)
                padding = (0, 0, 0)
        return functional.conv3d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )


class ConvTranspose1d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        self.in_channels = _builtins.int(in_channels)
        self.out_channels = _builtins.int(out_channels)
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.output_padding = _single(output_padding)
        self.groups = _builtins.int(groups)
        self.dilation = _single(dilation)
        self.padding_mode = padding_mode
        if self.groups <= 0:
            raise ValueError("groups must be a positive integer")
        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if self.out_channels % self.groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if padding_mode != "zeros":
            raise ValueError("ConvTranspose1d only supports padding_mode='zeros'")
        out_channels_per_group = self.out_channels // self.groups
        self.weight = Parameter(tensor(
            [
                [
                    [self._initial_weight(channel, out, kernel_x) for kernel_x in range(self.kernel_size[0])]
                    for out in range(out_channels_per_group)
                ]
                for channel in range(self.in_channels)
            ],
            dtype=dtype or float32,
            requires_grad=True,
            device=device or "cpu",
        ))
        self.bias = (
            Parameter(tensor(
                [0.0 for _ in range(self.out_channels)],
                dtype=dtype or float32,
                requires_grad=True,
                device=device or "cpu",
            ))
            if bias
            else None
        )

    def _initial_weight(self, channel, out, kernel_x):
        out_channels_per_group = self.out_channels // self.groups
        fan_in = _builtins.max(1, out_channels_per_group * self.kernel_size[0])
        position = (channel * out_channels_per_group + out) * self.kernel_size[0] + kernel_x
        return (((position % (2 * fan_in + 1)) - fan_in) / (fan_in * fan_in))

    def _output_padding_for_size(self, input, output_size):
        requested = tuple(_builtins.int(value) for value in output_size)
        spatial_dims = len(self.kernel_size)
        if len(requested) == spatial_dims + 2:
            requested = requested[2:]
        elif len(requested) != spatial_dims:
            raise ValueError(
                f"ConvTranspose1d output_size must have {spatial_dims} or {spatial_dims + 2} elements"
            )
        input_size = _builtins.int(input.shape[-1])
        min_size = (
            (input_size - 1) * self.stride[0]
            - 2 * self.padding[0]
            + self.dilation[0] * (self.kernel_size[0] - 1)
            + 1
        )
        max_size = min_size + self.stride[0] - 1
        size = requested[0]
        if size < min_size or size > max_size:
            raise ValueError(f"requested output_size {requested!r} is invalid: valid range is [{min_size}, {max_size}]")
        return (size - min_size,)

    def forward(self, input, output_size=None):
        output_padding = self.output_padding if output_size is None else self._output_padding_for_size(input, output_size)
        return functional.conv_transpose1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class ConvTranspose2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        self.in_channels = _builtins.int(in_channels)
        self.out_channels = _builtins.int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.output_padding = _pair(output_padding)
        self.groups = _builtins.int(groups)
        self.dilation = _pair(dilation)
        self.padding_mode = padding_mode
        if self.groups <= 0:
            raise ValueError("groups must be a positive integer")
        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if self.out_channels % self.groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if padding_mode != "zeros":
            raise NotImplementedError("only zeros padding_mode is implemented")
        out_channels_per_group = self.out_channels // self.groups
        self.weight = Parameter(tensor(
            [
                [
                    [
                        [
                            self._initial_weight(channel, out, kernel_y, kernel_x)
                            for kernel_x in range(self.kernel_size[1])
                        ]
                        for kernel_y in range(self.kernel_size[0])
                    ]
                    for out in range(out_channels_per_group)
                ]
                for channel in range(self.in_channels)
            ],
            dtype=dtype or float32,
            requires_grad=True,
            device=device or "cpu",
        ))
        self.bias = (
            Parameter(tensor(
                [0.0 for _ in range(self.out_channels)],
                dtype=dtype or float32,
                requires_grad=True,
                device=device or "cpu",
            ))
            if bias
            else None
        )

    def _initial_weight(self, channel, out, kernel_y, kernel_x):
        out_channels_per_group = self.out_channels // self.groups
        fan_in = _builtins.max(1, out_channels_per_group * self.kernel_size[0] * self.kernel_size[1])
        position = ((channel * out_channels_per_group + out) * self.kernel_size[0] + kernel_y) * self.kernel_size[1] + kernel_x
        return (((position % (2 * fan_in + 1)) - fan_in) / (fan_in * fan_in))

    def _output_padding_for_size(self, input, output_size):
        requested = tuple(_builtins.int(value) for value in output_size)
        spatial_dims = len(self.kernel_size)
        if len(requested) == spatial_dims + 2:
            requested = requested[2:]
        elif len(requested) != spatial_dims:
            raise ValueError(
                f"ConvTranspose2d output_size must have {spatial_dims} or {spatial_dims + 2} elements"
            )

        input_spatial = tuple(_builtins.int(value) for value in input.shape[-spatial_dims:])
        output_padding = []
        for index, input_size in enumerate(input_spatial):
            min_size = (
                (input_size - 1) * self.stride[index]
                - 2 * self.padding[index]
                + self.dilation[index] * (self.kernel_size[index] - 1)
                + 1
            )
            max_size = min_size + self.stride[index] - 1
            size = requested[index]
            if size < min_size or size > max_size:
                raise ValueError(
                    f"requested output_size {requested!r} is invalid for spatial dim {index}: "
                    f"valid range is [{min_size}, {max_size}]"
                )
            output_padding.append(size - min_size)
        return tuple(output_padding)

    def forward(self, input, output_size=None):
        output_padding = self.output_padding if output_size is None else self._output_padding_for_size(input, output_size)
        return functional.conv_transpose2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class ConvTranspose3d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        self.in_channels = _builtins.int(in_channels)
        self.out_channels = _builtins.int(out_channels)
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.output_padding = _triple(output_padding)
        self.groups = _builtins.int(groups)
        self.dilation = _triple(dilation)
        self.padding_mode = padding_mode
        if self.groups <= 0:
            raise ValueError("groups must be a positive integer")
        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if self.out_channels % self.groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if padding_mode != "zeros":
            raise NotImplementedError("only zeros padding_mode is implemented")
        out_channels_per_group = self.out_channels // self.groups
        self.weight = Parameter(tensor(
            [
                [
                    [
                        [
                            [
                                self._initial_weight(channel, out, kernel_z, kernel_y, kernel_x)
                                for kernel_x in range(self.kernel_size[2])
                            ]
                            for kernel_y in range(self.kernel_size[1])
                        ]
                        for kernel_z in range(self.kernel_size[0])
                    ]
                    for out in range(out_channels_per_group)
                ]
                for channel in range(self.in_channels)
            ],
            dtype=dtype or float32,
            requires_grad=True,
            device=device or "cpu",
        ))
        self.bias = (
            Parameter(tensor(
                [0.0 for _ in range(self.out_channels)],
                dtype=dtype or float32,
                requires_grad=True,
                device=device or "cpu",
            ))
            if bias
            else None
        )

    def _initial_weight(self, channel, out, kernel_z, kernel_y, kernel_x):
        out_channels_per_group = self.out_channels // self.groups
        fan_in = _builtins.max(
            1,
            out_channels_per_group * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2],
        )
        position = (
            (((channel * out_channels_per_group + out) * self.kernel_size[0] + kernel_z) * self.kernel_size[1] + kernel_y)
            * self.kernel_size[2]
            + kernel_x
        )
        return (((position % (2 * fan_in + 1)) - fan_in) / (fan_in * fan_in))

    def _output_padding_for_size(self, input, output_size):
        requested = tuple(_builtins.int(value) for value in output_size)
        spatial_dims = len(self.kernel_size)
        if len(requested) == spatial_dims + 2:
            requested = requested[2:]
        elif len(requested) != spatial_dims:
            raise ValueError(
                f"ConvTranspose3d output_size must have {spatial_dims} or {spatial_dims + 2} elements"
            )

        input_spatial = tuple(_builtins.int(value) for value in input.shape[-spatial_dims:])
        output_padding = []
        for index, input_size in enumerate(input_spatial):
            min_size = (
                (input_size - 1) * self.stride[index]
                - 2 * self.padding[index]
                + self.dilation[index] * (self.kernel_size[index] - 1)
                + 1
            )
            max_size = min_size + self.stride[index] - 1
            size = requested[index]
            if size < min_size or size > max_size:
                raise ValueError(
                    f"requested output_size {requested!r} is invalid for spatial dim {index}: "
                    f"valid range is [{min_size}, {max_size}]"
                )
            output_padding.append(size - min_size)
        return tuple(output_padding)

    def forward(self, input, output_size=None):
        output_padding = self.output_padding if output_size is None else self._output_padding_for_size(input, output_size)
        return functional.conv_transpose3d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class Upsample(Module):
    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
    ):
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, input):
        return functional.interpolate(
            input,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )


class UpsamplingNearest2d(Upsample):
    def __init__(self, size=None, scale_factor=None):
        super().__init__(size=size, scale_factor=scale_factor, mode="nearest")


class UpsamplingBilinear2d(Upsample):
    def __init__(self, size=None, scale_factor=None):
        super().__init__(size=size, scale_factor=scale_factor, mode="bilinear", align_corners=True)


class MaxPool1d(Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        self.kernel_size = _single(kernel_size)
        self.stride = self.kernel_size if stride is None else _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.return_indices = _builtins.bool(return_indices)
        self.ceil_mode = _builtins.bool(ceil_mode)

    def forward(self, input):
        return functional.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )


class AvgPool1d(Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        self.kernel_size = _single(kernel_size)
        self.stride = self.kernel_size if stride is None else _single(stride)
        self.padding = _single(padding)
        self.ceil_mode = _builtins.bool(ceil_mode)
        self.count_include_pad = _builtins.bool(count_include_pad)

    def forward(self, input):
        return functional.avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )


class AdaptiveAvgPool1d(Module):
    def __init__(self, output_size):
        self.output_size = output_size

    def forward(self, input):
        return functional.adaptive_avg_pool1d(input, self.output_size)


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        self.kernel_size = _pair(kernel_size)
        self.stride = self.kernel_size if stride is None else _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.return_indices = _builtins.bool(return_indices)
        self.ceil_mode = _builtins.bool(ceil_mode)

    def forward(self, input):
        return functional.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )


class AvgPool2d(Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        self.kernel_size = _pair(kernel_size)
        self.stride = self.kernel_size if stride is None else _pair(stride)
        self.padding = _pair(padding)
        self.ceil_mode = _builtins.bool(ceil_mode)
        self.count_include_pad = _builtins.bool(count_include_pad)
        self.divisor_override = divisor_override

    def forward(self, input):
        return functional.avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        self.output_size = output_size

    def forward(self, input):
        return functional.adaptive_avg_pool2d(input, self.output_size)


class Unfold(Module):
    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.stride = _pair(stride)

    def forward(self, input):
        return functional.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)


class Fold(Module):
    def __init__(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        self.output_size = _pair(output_size)
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.stride = _pair(stride)

    def forward(self, input):
        return functional.fold(input, self.output_size, self.kernel_size, self.dilation, self.padding, self.stride)


def _pad_tuple(padding, spatial_dims):
    expected = 2 * _builtins.int(spatial_dims)
    if isinstance(padding, _builtins.int):
        return tuple([_builtins.int(padding)] * expected)
    values = tuple(_builtins.int(item) for item in padding)
    if len(values) != expected:
        raise ValueError(f"padding must be an int or a tuple of length {expected}")
    return values


class _ConstantPadNd(Module):
    _spatial_dims = 1

    def __init__(self, padding, value):
        self.padding = _pad_tuple(padding, self._spatial_dims)
        self.value = _builtins.float(value)

    def forward(self, input):
        return functional.pad(input, self.padding, "constant", self.value)


class _NonConstantPadNd(Module):
    _spatial_dims = 1
    _mode = "constant"

    def __init__(self, padding):
        self.padding = _pad_tuple(padding, self._spatial_dims)

    def forward(self, input):
        return functional.pad(input, self.padding, self._mode)


class ConstantPad1d(_ConstantPadNd):
    _spatial_dims = 1


class ConstantPad2d(_ConstantPadNd):
    _spatial_dims = 2


class ConstantPad3d(_ConstantPadNd):
    _spatial_dims = 3


class ZeroPad1d(_ConstantPadNd):
    _spatial_dims = 1

    def __init__(self, padding):
        super().__init__(padding, 0.0)


class ZeroPad2d(_ConstantPadNd):
    _spatial_dims = 2

    def __init__(self, padding):
        super().__init__(padding, 0.0)


class ZeroPad3d(_ConstantPadNd):
    _spatial_dims = 3

    def __init__(self, padding):
        super().__init__(padding, 0.0)


class ReflectionPad1d(_NonConstantPadNd):
    _spatial_dims = 1
    _mode = "reflect"


class ReflectionPad2d(_NonConstantPadNd):
    _spatial_dims = 2
    _mode = "reflect"


class ReflectionPad3d(_NonConstantPadNd):
    _spatial_dims = 3
    _mode = "reflect"


class ReplicationPad1d(_NonConstantPadNd):
    _spatial_dims = 1
    _mode = "replicate"


class ReplicationPad2d(_NonConstantPadNd):
    _spatial_dims = 2
    _mode = "replicate"


class ReplicationPad3d(_NonConstantPadNd):
    _spatial_dims = 3
    _mode = "replicate"


class CircularPad1d(_NonConstantPadNd):
    _spatial_dims = 1
    _mode = "circular"


class CircularPad2d(_NonConstantPadNd):
    _spatial_dims = 2
    _mode = "circular"


class CircularPad3d(_NonConstantPadNd):
    _spatial_dims = 3
    _mode = "circular"


class PixelShuffle(Module):
    def __init__(self, upscale_factor):
        self.upscale_factor = _builtins.int(upscale_factor)

    def forward(self, input):
        return functional.pixel_shuffle(input, self.upscale_factor)


class PixelUnshuffle(Module):
    def __init__(self, downscale_factor):
        self.downscale_factor = _builtins.int(downscale_factor)

    def forward(self, input):
        return functional.pixel_unshuffle(input, self.downscale_factor)


class ChannelShuffle(Module):
    def __init__(self, groups):
        self.groups = _builtins.int(groups)

    def forward(self, input):
        return functional.channel_shuffle(input, self.groups)


class MultiheadAttention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ):
        self.embed_dim = _builtins.int(embed_dim)
        self.num_heads = _builtins.int(num_heads)
        self.dropout = _builtins.float(dropout)
        self.batch_first = batch_first
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.kdim = self.embed_dim if kdim is None else _builtins.int(kdim)
        self.vdim = self.embed_dim if vdim is None else _builtins.int(vdim)
        self._qkv_same_embed_dim = self.kdim == self.embed_dim and self.vdim == self.embed_dim
        self.add_zero_attn = _builtins.bool(add_zero_attn)
        if self._qkv_same_embed_dim:
            self.in_proj_weight = Parameter(tensor(
                [
                    [self._initial_weight(row, col, self.embed_dim) for col in range(self.embed_dim)]
                    for row in range(3 * self.embed_dim)
                ],
                dtype=dtype or float32,
                requires_grad=True,
                device=device or "cpu",
            ))
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None
        else:
            self.in_proj_weight = None
            self.q_proj_weight = Parameter(tensor(
                [[self._initial_weight(row, col, self.embed_dim) for col in range(self.embed_dim)] for row in range(self.embed_dim)],
                dtype=dtype or float32,
                requires_grad=True,
                device=device or "cpu",
            ))
            self.k_proj_weight = Parameter(tensor(
                [[self._initial_weight(row, col, self.kdim) for col in range(self.kdim)] for row in range(self.embed_dim)],
                dtype=dtype or float32,
                requires_grad=True,
                device=device or "cpu",
            ))
            self.v_proj_weight = Parameter(tensor(
                [[self._initial_weight(row, col, self.vdim) for col in range(self.vdim)] for row in range(self.embed_dim)],
                dtype=dtype or float32,
                requires_grad=True,
                device=device or "cpu",
            ))
        self.in_proj_bias = (
            Parameter(tensor(
                [0.0 for _ in range(3 * self.embed_dim)],
                dtype=dtype or float32,
                requires_grad=True,
                device=device or "cpu",
            ))
            if bias
            else None
        )
        if add_bias_kv:
            self.bias_k = Parameter(tensor(
                [[[self._initial_weight(0, col, self.embed_dim) for col in range(self.embed_dim)]]],
                dtype=dtype or float32,
                requires_grad=True,
                device=device or "cpu",
            ))
            self.bias_v = Parameter(tensor(
                [[[self._initial_weight(1, col, self.embed_dim) for col in range(self.embed_dim)]]],
                dtype=dtype or float32,
                requires_grad=True,
                device=device or "cpu",
            ))
        else:
            self.bias_k = None
            self.bias_v = None
        self.out_proj = Linear(self.embed_dim, self.embed_dim, bias=bias, device=device, dtype=dtype)

    def _initial_weight(self, row, col, width):
        span = _builtins.max(1, width)
        return (((row * width + col) % (2 * span + 1)) - span) / (span * span)

    def _as_batch_first(self, input):
        if input.ndim == 2:
            return reshape(input, (1, input.shape[0], input.shape[1])), True
        if input.ndim != 3:
            raise ValueError("MultiheadAttention input must be 2-D or 3-D")
        if self.batch_first:
            return input, False
        return transpose(input, 0, 1), False

    def _restore_layout(self, input, unbatched):
        if unbatched:
            return squeeze(input, 0)
        if self.batch_first:
            return input
        return transpose(input, 0, 1)

    def _project_3d(self, input, weight, bias):
        return functional.linear(input, weight, bias)

    def _split_heads(self, input):
        batch, length, _ = input.shape
        return transpose(reshape(input, (batch, length, self.num_heads, self.head_dim)), 1, 2)

    def _merge_heads(self, input):
        batch, _, length, _ = input.shape
        return reshape(transpose(input, 1, 2).contiguous(), (batch, length, self.embed_dim))

    def _mask_is_bool(self, mask):
        return str(mask.dtype) == "bool"

    def _keep_mask_to_bias(self, mask):
        return where(
            mask,
            zeros(mask.shape, dtype=float32, device=mask.device),
            full(mask.shape, -_math.inf, dtype=float32, device=mask.device),
        )

    def _pad_source_mask(self, mask):
        if mask is None:
            return None
        pad_shape = tuple(mask.shape[:-1]) + (1,)
        padding = zeros(pad_shape, dtype=mask.dtype, device=mask.device)
        return cat((mask, padding), dim=mask.ndim - 1)

    def _append_bias_kv(self, key, value, key_padding_mask, attn_mask):
        if self.bias_k is None:
            return key, value, key_padding_mask, attn_mask
        batch = key.shape[0]
        key = cat((key, self.bias_k.repeat(batch, 1, 1)), dim=1)
        value = cat((value, self.bias_v.repeat(batch, 1, 1)), dim=1)
        return key, value, self._pad_source_mask(key_padding_mask), self._pad_source_mask(attn_mask)

    def _append_zero_kv(self, key, value, key_padding_mask, attn_mask):
        if not self.add_zero_attn:
            return key, value, key_padding_mask, attn_mask
        zero_shape = (key.shape[0], 1, key.shape[2])
        key = cat((key, zeros(zero_shape, dtype=key.dtype, device=key.device)), dim=1)
        value = cat((value, zeros(zero_shape, dtype=value.dtype, device=value.device)), dim=1)
        return key, value, self._pad_source_mask(key_padding_mask), self._pad_source_mask(attn_mask)

    def _merge_attention_masks(self, attn_mask, key_padding_mask, batch, target_length, source_length):
        merged_mask = None
        if attn_mask is not None:
            merged_mask = logical_not(attn_mask) if self._mask_is_bool(attn_mask) else attn_mask

        if key_padding_mask is not None:
            if key_padding_mask.ndim == 1:
                if key_padding_mask.shape[0] != source_length:
                    raise ValueError("key_padding_mask source length does not match key")
                key_padding_mask = reshape(key_padding_mask, (1, source_length))
            elif key_padding_mask.ndim != 2:
                raise ValueError("key_padding_mask must be 1-D or 2-D")
            if key_padding_mask.shape[0] != batch or key_padding_mask.shape[1] != source_length:
                raise ValueError("key_padding_mask shape must be (batch, source_length)")
            key_mask = logical_not(key_padding_mask) if self._mask_is_bool(key_padding_mask) else key_padding_mask
            key_mask = reshape(key_mask, (batch, 1, 1, source_length))
            if merged_mask is None:
                merged_mask = key_mask
            elif self._mask_is_bool(merged_mask) and self._mask_is_bool(key_mask):
                merged_mask = logical_and(merged_mask, key_mask)
            else:
                if self._mask_is_bool(merged_mask):
                    merged_mask = self._keep_mask_to_bias(merged_mask)
                if self._mask_is_bool(key_mask):
                    key_mask = self._keep_mask_to_bias(key_mask)
                merged_mask = merged_mask + key_mask

        return merged_mask

    def _attention_weights(self, query, key, attn_mask, is_causal, average_attn_weights, unbatched):
        batch, heads, _, _ = query.shape
        source_length = key.shape[2]
        identity = tensor(
            [
                [
                    [[1.0 if row == col else 0.0 for col in range(source_length)] for row in range(source_length)]
                    for _ in range(heads)
                ]
                for _ in range(batch)
            ],
            dtype=float32,
            device="cpu",
        )
        weights = functional.scaled_dot_product_attention(query, key, identity, attn_mask, 0.0, is_causal)
        if average_attn_weights:
            weights = mean(weights, dim=1)
        if unbatched:
            weights = squeeze(weights, 0)
        return weights

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        query_bf, query_unbatched = self._as_batch_first(query)
        key_bf, key_unbatched = self._as_batch_first(key)
        value_bf, value_unbatched = self._as_batch_first(value)
        if key_unbatched != query_unbatched or value_unbatched != query_unbatched:
            raise ValueError("query, key, and value must agree on batching")
        if query_bf.shape[2] != self.embed_dim or key_bf.shape[2] != self.kdim or value_bf.shape[2] != self.vdim:
            raise ValueError("query, key, and value embedding dimensions must match configured dimensions")

        if self._qkv_same_embed_dim and query is key and key is value:
            qkv = self._project_3d(query_bf, self.in_proj_weight, self.in_proj_bias)
            q_proj = narrow(qkv, 2, 0, self.embed_dim)
            k_proj = narrow(qkv, 2, self.embed_dim, self.embed_dim)
            v_proj = narrow(qkv, 2, 2 * self.embed_dim, self.embed_dim)
        elif self._qkv_same_embed_dim:
            q_proj = self._project_3d(
                query_bf,
                narrow(self.in_proj_weight, 0, 0, self.embed_dim),
                None if self.in_proj_bias is None else narrow(self.in_proj_bias, 0, 0, self.embed_dim),
            )
            k_proj = self._project_3d(
                key_bf,
                narrow(self.in_proj_weight, 0, self.embed_dim, self.embed_dim),
                None if self.in_proj_bias is None else narrow(self.in_proj_bias, 0, self.embed_dim, self.embed_dim),
            )
            v_proj = self._project_3d(
                value_bf,
                narrow(self.in_proj_weight, 0, 2 * self.embed_dim, self.embed_dim),
                None if self.in_proj_bias is None else narrow(self.in_proj_bias, 0, 2 * self.embed_dim, self.embed_dim),
            )
        else:
            q_proj = self._project_3d(
                query_bf,
                self.q_proj_weight,
                None if self.in_proj_bias is None else narrow(self.in_proj_bias, 0, 0, self.embed_dim),
            )
            k_proj = self._project_3d(
                key_bf,
                self.k_proj_weight,
                None if self.in_proj_bias is None else narrow(self.in_proj_bias, 0, self.embed_dim, self.embed_dim),
            )
            v_proj = self._project_3d(
                value_bf,
                self.v_proj_weight,
                None if self.in_proj_bias is None else narrow(self.in_proj_bias, 0, 2 * self.embed_dim, self.embed_dim),
            )

        k_proj, v_proj, key_padding_mask, attn_mask = self._append_bias_kv(k_proj, v_proj, key_padding_mask, attn_mask)
        k_proj, v_proj, key_padding_mask, attn_mask = self._append_zero_kv(k_proj, v_proj, key_padding_mask, attn_mask)
        q_heads = self._split_heads(q_proj)
        k_heads = self._split_heads(k_proj)
        v_heads = self._split_heads(v_proj)
        merged_mask = None
        if attn_mask is not None or key_padding_mask is not None:
            merged_mask = self._merge_attention_masks(
                attn_mask,
                key_padding_mask,
                q_heads.shape[0],
                q_heads.shape[2],
                k_heads.shape[2],
            )
        dropout_p = self.dropout if self.training else 0.0
        attended = functional.scaled_dot_product_attention(q_heads, k_heads, v_heads, merged_mask, dropout_p, is_causal)
        merged = self._merge_heads(attended)
        batch, length, _ = merged.shape
        output = self.out_proj(reshape(merged, (batch * length, self.embed_dim)))
        output = reshape(output, (batch, length, self.embed_dim))
        weights = None
        if need_weights:
            weights = self._attention_weights(q_heads, k_heads, merged_mask, is_causal, average_attn_weights, query_unbatched)
        return self._restore_layout(output, query_unbatched), weights


class Embedding(Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        _freeze=False,
        device=None,
        dtype=None,
    ):
        self.num_embeddings = _builtins.int(num_embeddings)
        self.embedding_dim = _builtins.int(embedding_dim)
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        if _weight is None:
            values = [
                [self._initial_weight(row, col) for col in range(self.embedding_dim)]
                for row in range(self.num_embeddings)
            ]
            self.weight = Parameter(tensor(
                values,
                dtype=dtype or float32,
                requires_grad=not _freeze,
                device=device or "cpu",
            ), requires_grad=not _freeze)
            if padding_idx is not None:
                self.weight[_builtins.int(padding_idx)].zero_()
        else:
            self.weight = Parameter(_weight, requires_grad=not _freeze)

    def _initial_weight(self, row, col):
        span = _builtins.max(1, self.embedding_dim)
        return (((row * self.embedding_dim + col) % (2 * span + 1)) - span) / span

    def forward(self, input):
        return functional.embedding(
            input,
            self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


class ReLU(Module):
    def forward(self, input):
        return functional.relu(input)


class ReLU6(Module):
    def __init__(self, inplace=False):
        self.inplace = _builtins.bool(inplace)

    def forward(self, input):
        return functional.relu6(input, inplace=self.inplace)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input):
        return functional.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)


class SiLU(Module):
    def __init__(self, inplace=False):
        self.inplace = inplace

    def forward(self, input):
        return functional.silu(input, inplace=self.inplace)


class ELU(Module):
    def __init__(self, alpha=1.0, inplace=False):
        self.alpha = _builtins.float(alpha)
        self.inplace = _builtins.bool(inplace)

    def forward(self, input):
        return functional.elu(input, alpha=self.alpha, inplace=self.inplace)


class SELU(Module):
    def __init__(self, inplace=False):
        self.inplace = _builtins.bool(inplace)

    def forward(self, input):
        return functional.selu(input, inplace=self.inplace)


class Softplus(Module):
    def __init__(self, beta=1.0, threshold=20.0):
        self.beta = _builtins.float(beta)
        self.threshold = _builtins.float(threshold)

    def forward(self, input):
        return functional.softplus(input, beta=self.beta, threshold=self.threshold)


class Hardtanh(Module):
    def __init__(self, min_val=-1.0, max_val=1.0, inplace=False, min_value=None, max_value=None):
        if min_value is not None:
            min_val = min_value
        if max_value is not None:
            max_val = max_value
        self.min_val = _builtins.float(min_val)
        self.max_val = _builtins.float(max_val)
        self.inplace = _builtins.bool(inplace)

    def forward(self, input):
        return functional.hardtanh(input, min_val=self.min_val, max_val=self.max_val, inplace=self.inplace)


class GELU(Module):
    def __init__(self, approximate="none"):
        self.approximate = approximate

    def forward(self, input):
        return functional.gelu(input, approximate=self.approximate)


class Hardsigmoid(Module):
    def __init__(self, inplace=False):
        self.inplace = _builtins.bool(inplace)

    def forward(self, input):
        return functional.hardsigmoid(input, inplace=self.inplace)


class Hardswish(Module):
    def __init__(self, inplace=False):
        self.inplace = _builtins.bool(inplace)

    def forward(self, input):
        return functional.hardswish(input, inplace=self.inplace)


class Softsign(Module):
    def forward(self, input):
        return functional.softsign(input)


class LogSigmoid(Module):
    def forward(self, input):
        return functional.logsigmoid(input)


class GLU(Module):
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, input):
        return functional.glu(input, dim=self.dim)


class Mish(Module):
    def __init__(self, inplace=False):
        self.inplace = _builtins.bool(inplace)

    def forward(self, input):
        return functional.mish(input, inplace=self.inplace)


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        self.p = _builtins.float(p)
        self.inplace = _builtins.bool(inplace)

    def forward(self, input):
        return functional.dropout(input, self.p, self.training, self.inplace)


class Identity(Module):
    def forward(self, input):
        return input


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return flatten(input, self.start_dim, self.end_dim)


class Unflatten(Module):
    def __init__(self, dim, unflattened_size):
        self.dim = dim
        self.unflattened_size = tuple(unflattened_size)

    def forward(self, input):
        return unflatten(input, self.dim, self.unflattened_size)


class Sequential(Module):
    def __init__(self, *modules):
        if len(modules) == 1 and isinstance(modules[0], OrderedDict):
            self._module_names = [str(name) for name in modules[0].keys()]
            self._modules = list(modules[0].values())
        else:
            self._modules = list(modules)
            self._module_names = [str(index) for index in range(len(self._modules))]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Sequential(OrderedDict(zip(self._module_names[index], self._modules[index], strict=True)))
        if isinstance(index, str):
            return self._modules[self._module_names.index(index)]
        return self._modules[index]

    def append(self, module):
        self._module_names.append(str(len(self._modules)))
        self._modules.append(module)
        return self

    def named_children(self):
        for name, module in zip(self._module_names, self._modules, strict=True):
            if isinstance(module, Module):
                yield name, module

    def forward(self, input):
        output = input
        for module in self._modules:
            output = module(output)
        return output

    def _named_parameters(self, prefix, recurse, seen):
        if not recurse:
            return
        for name, module in zip(self._module_names, self._modules, strict=True):
            if isinstance(module, Module):
                child_prefix = f"{prefix}.{name}" if prefix else name
                yield from module._named_parameters(child_prefix, recurse, seen)


class ModuleList(Module):
    def __init__(self, modules=None):
        self._modules = list(modules or [])

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return ModuleList(self._modules[index])
        return self._modules[index]

    def __setitem__(self, index, module):
        self._modules[index] = module

    def append(self, module):
        self._modules.append(module)
        return self

    def extend(self, modules):
        self._modules.extend(modules)
        return self

    def named_children(self):
        for index, module in enumerate(self._modules):
            if isinstance(module, Module):
                yield str(index), module

    def forward(self, input):
        raise NotImplementedError("ModuleList is not callable")

    def _named_parameters(self, prefix, recurse, seen):
        if not recurse:
            return
        for index, module in enumerate(self._modules):
            if isinstance(module, Module):
                child_prefix = f"{prefix}.{index}" if prefix else str(index)
                yield from module._named_parameters(child_prefix, recurse, seen)


class ModuleDict(Module):
    def __init__(self, modules=None):
        self._modules = OrderedDict()
        if modules is not None:
            self.update(modules)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self._modules[str(key)] = module

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def update(self, modules):
        items = modules.items() if hasattr(modules, "items") else modules
        for key, module in items:
            self[key] = module
        return self

    def named_children(self):
        for name, module in self._modules.items():
            if isinstance(module, Module):
                yield name, module

    def forward(self, input):
        raise NotImplementedError("ModuleDict is not callable")

    def _named_parameters(self, prefix, recurse, seen):
        if not recurse:
            return
        for name, module in self._modules.items():
            if isinstance(module, Module):
                child_prefix = f"{prefix}.{name}" if prefix else name
                yield from module._named_parameters(child_prefix, recurse, seen)


class ParameterList(Module):
    def __init__(self, parameters=None):
        self._parameters = [_as_parameter(item) for item in (parameters or [])]

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return ParameterList(self._parameters[index])
        return self._parameters[index]

    def __setitem__(self, index, parameter):
        self._parameters[index] = _as_parameter(parameter)

    def append(self, parameter):
        self._parameters.append(_as_parameter(parameter))
        return self

    def extend(self, parameters):
        for parameter in parameters:
            self.append(parameter)
        return self

    def forward(self, input):
        raise NotImplementedError("ParameterList is not callable")

    def _named_parameters(self, prefix, recurse, seen):
        for index, parameter in enumerate(self._parameters):
            if parameter is None:
                continue
            if self._should_yield_state_value(seen, parameter):
                yield (f"{prefix}.{index}" if prefix else str(index), parameter)


class ParameterDict(Module):
    def __init__(self, parameters=None):
        self._parameters = OrderedDict()
        if parameters is not None:
            self.update(parameters)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters)

    def __getitem__(self, key):
        return self._parameters[key]

    def __setitem__(self, key, parameter):
        self._parameters[str(key)] = _as_parameter(parameter)

    def keys(self):
        return self._parameters.keys()

    def values(self):
        return self._parameters.values()

    def items(self):
        return self._parameters.items()

    def update(self, parameters):
        items = parameters.items() if hasattr(parameters, "items") else parameters
        for key, parameter in items:
            self[key] = parameter
        return self

    def forward(self, input):
        raise NotImplementedError("ParameterDict is not callable")

    def _named_parameters(self, prefix, recurse, seen):
        for name, parameter in self._parameters.items():
            if parameter is None:
                continue
            if self._should_yield_state_value(seen, parameter):
                yield (f"{prefix}.{name}" if prefix else name, parameter)


class Sigmoid(Module):
    def forward(self, input):
        return functional.sigmoid(input)


class Tanh(Module):
    def forward(self, input):
        return functional.tanh(input)


class Softmax(Module):
    def __init__(self, dim=None):
        self.dim = dim

    def forward(self, input):
        return functional.softmax(input, dim=self.dim)


class Softmin(Module):
    def __init__(self, dim=None):
        self.dim = dim

    def forward(self, input):
        return functional.softmin(input, dim=self.dim)


class LogSoftmax(Module):
    def __init__(self, dim=None):
        self.dim = dim

    def forward(self, input):
        return functional.log_softmax(input, dim=self.dim)


class Softmax2d(Module):
    def forward(self, input):
        if input.dim() not in (3, 4):
            raise ValueError(f"Softmax2d: expected input to be 3D or 4D, got {input.dim()}D instead")
        return functional.softmax(input, dim=0 if input.dim() == 3 else 1)


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        elementwise_affine=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        if isinstance(normalized_shape, _builtins.int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = _builtins.bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = Parameter(ones(self.normalized_shape, dtype=dtype or float32, requires_grad=True, device=device or "cpu"))
            self.bias = (
                Parameter(zeros(self.normalized_shape, dtype=dtype or float32, requires_grad=True, device=device or "cpu"))
                if bias
                else None
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, input):
        return functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


class RMSNorm(Module):
    def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
        if isinstance(normalized_shape, _builtins.int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = _builtins.bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = Parameter(ones(self.normalized_shape, dtype=dtype or float32, requires_grad=True, device=device or "cpu"))
        else:
            self.weight = None

    def forward(self, input):
        return functional.rms_norm(input, self.normalized_shape, self.weight, self.eps)


class BatchNorm1d(Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        self.num_features = _builtins.int(num_features)
        self.eps = _builtins.float(eps)
        self.momentum = momentum
        self.affine = _builtins.bool(affine)
        self.track_running_stats = _builtins.bool(track_running_stats)
        tensor_dtype = dtype or float32
        tensor_device = device or "cpu"
        if self.affine:
            self.weight = Parameter(ones((self.num_features,), dtype=tensor_dtype, requires_grad=True, device=tensor_device))
            self.bias = Parameter(zeros((self.num_features,), dtype=tensor_dtype, requires_grad=True, device=tensor_device))
        else:
            self.weight = None
            self.bias = None
        if self.track_running_stats:
            self.register_buffer("running_mean", zeros((self.num_features,), dtype=tensor_dtype, device=tensor_device))
            self.register_buffer("running_var", ones((self.num_features,), dtype=tensor_dtype, device=tensor_device))
            self.register_buffer("num_batches_tracked", tensor(0, dtype=int64, device=tensor_device))
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def _update_exponential_average_factor(self):
        exponential_average_factor = 0.0 if self.momentum is None else _builtins.float(self.momentum)
        if self.training and self.track_running_stats:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                exponential_average_factor = 1.0 / _builtins.float(self.num_batches_tracked.item())
        return exponential_average_factor

    def forward(self, input):
        if input.dim() not in (2, 3):
            raise ValueError("expected 2D or 3D input")
        use_batch_stats = self.training or not self.track_running_stats
        running_mean = self.running_mean if self.track_running_stats else None
        running_var = self.running_var if self.track_running_stats else None
        return functional.batch_norm(
            input,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            use_batch_stats,
            self._update_exponential_average_factor(),
            self.eps,
        )


class BatchNorm2d(BatchNorm1d):
    def forward(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input")
        return functional.batch_norm(
            input,
            self.running_mean if self.track_running_stats else None,
            self.running_var if self.track_running_stats else None,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self._update_exponential_average_factor(),
            self.eps,
        )


class BatchNorm3d(BatchNorm1d):
    def forward(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input")
        return functional.batch_norm(
            input,
            self.running_mean if self.track_running_stats else None,
            self.running_var if self.track_running_stats else None,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self._update_exponential_average_factor(),
            self.eps,
        )


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, device=None, dtype=None):
        self.num_groups = _builtins.int(num_groups)
        self.num_channels = _builtins.int(num_channels)
        self.eps = _builtins.float(eps)
        self.affine = _builtins.bool(affine)
        tensor_dtype = dtype or float32
        tensor_device = device or "cpu"
        if self.affine:
            self.weight = Parameter(ones((self.num_channels,), dtype=tensor_dtype, requires_grad=True, device=tensor_device))
            self.bias = Parameter(zeros((self.num_channels,), dtype=tensor_dtype, requires_grad=True, device=tensor_device))
        else:
            self.weight = None
            self.bias = None

    def forward(self, input):
        return functional.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)


class CosineSimilarity(Module):
    def __init__(self, dim=1, eps=1e-8):
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return functional.cosine_similarity(x1, x2, self.dim, self.eps)


class MSELoss(Module):
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        if reduce is not None:
            reduction = "mean" if reduce else "none"
        self.reduction = reduction

    def forward(self, input, target):
        return functional.mse_loss(input, target, reduction=self.reduction)


class L1Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        if reduce is not None:
            reduction = "mean" if reduce else "none"
        self.reduction = reduction

    def forward(self, input, target):
        return functional.l1_loss(input, target, reduction=self.reduction)


class NLLLoss(Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean"):
        if reduce is not None:
            reduction = "mean" if reduce else "none"
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        return functional.nll_loss(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class CrossEntropyLoss(Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
    ):
        if reduce is not None:
            reduction = "mean" if reduce else "none"
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        return functional.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class BCELoss(Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction="mean"):
        if reduce is not None:
            reduction = "mean" if reduce else "none"
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return functional.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


class BCEWithLogitsLoss(Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction="mean", pos_weight=None):
        if reduce is not None:
            reduction = "mean" if reduce else "none"
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, input, target):
        return functional.binary_cross_entropy_with_logits(
            input,
            target,
            weight=self.weight,
            reduction=self.reduction,
            pos_weight=self.pos_weight,
        )


def _dropout(input, p=0.5, training=True, inplace=False):
    probability = _builtins.float(p)
    if probability < 0.0 or probability > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1")
    if not training or probability == 0.0:
        return input
    if inplace:
        raise NotImplementedError("dropout inplace=True is not implemented for training=True")
    if probability == 1.0:
        return input * 0.0

    scale = 1.0 / (1.0 - probability)
    counter = [0]

    def build(shape):
        if not shape:
            counter[0] += 1
            sample = _math.sin(counter[0] * 12.9898) * 43758.5453
            sample = sample - _math.floor(sample)
            return scale if sample >= probability else 0.0
        return [build(shape[1:]) for _ in range(shape[0])]

    mask = tensor(build(tuple(input.shape)), dtype=input.dtype, device=input.device)
    return input * mask


def _scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    probability = _builtins.float(dropout_p)
    if probability < 0.0 or probability > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1")
    if probability == 0.0 or probability == 1.0:
        return _C.scaled_dot_product_attention(query, key, value, attn_mask, probability, is_causal, scale, enable_gqa)

    if enable_gqa:
        if query.ndim < 3:
            raise ValueError("scaled_dot_product_attention enable_gqa requires a head dimension")
        query_heads = query.shape[-3]
        key_heads = key.shape[-3]
        value_heads = value.shape[-3]
        if key_heads != value_heads or key_heads <= 0 or query_heads % key_heads != 0:
            raise ValueError("scaled_dot_product_attention query heads must be divisible by key/value heads for enable_gqa")
        repeat = query_heads // key_heads
        key = repeat_interleave(key, repeat, dim=key.ndim - 3)
        value = repeat_interleave(value, repeat, dim=value.ndim - 3)

    scale_factor = _builtins.float(scale) if scale is not None else 1.0 / _math.sqrt(_builtins.float(query.shape[-1]))
    scores = matmul(query, transpose(key, -2, -1)) * scale_factor
    if is_causal:
        query_length = query.shape[-2]
        key_length = key.shape[-2]
        causal_mask = tensor(
            [[col <= row for col in range(key_length)] for row in range(query_length)],
            dtype=bool,
            device=query.device,
        )
        scores = where(causal_mask, scores, full(scores.shape, -_math.inf, dtype=query.dtype, device=query.device))
    if attn_mask is not None:
        if attn_mask.dtype == bool:
            scores = where(attn_mask, scores, full(scores.shape, -_math.inf, dtype=query.dtype, device=query.device))
        else:
            scores = scores + attn_mask
    probabilities = softmax(scores, dim=-1)
    probabilities = _C.dropout(probabilities, probability, True)
    return matmul(probabilities, value)


def _normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    norm_order = _builtins.float(p)
    epsilon = _builtins.float(eps)
    if (
        out is None
        and not input.requires_grad
        and norm_order == 2.0
        and not isinstance(dim, (tuple, list, Size))
        and input.dtype in {float16, float32, float64}
    ):
        return _normalize_l2(input, dim=dim, eps=epsilon)
    denominator = norm(input, p=norm_order, dim=dim, keepdim=True)
    result = input / clamp_min(denominator, epsilon)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _glu(input, dim=-1):
    split_dim = _builtins.int(dim)
    dim_index = split_dim if split_dim >= 0 else input.ndim + split_dim
    if dim_index < 0 or dim_index >= input.ndim:
        raise IndexError("glu dim is out of range")
    dim_size = _builtins.int(input.shape[dim_index])
    if dim_size % 2 != 0:
        raise RuntimeError("Halving dimension must be even")
    first, second = split(input, dim_size // 2, dim=split_dim)
    return first * sigmoid(second)


def _logsigmoid(input):
    return -functional.softplus(-input)


def _softmin(input, dim=None, dtype=None):
    return softmax(-input, dim=dim, dtype=dtype)


def _init_constant_(tensor, val):
    return tensor.fill_(val)


def _init_zeros_(tensor):
    return tensor.zero_()


def _init_ones_(tensor):
    return tensor.fill_(1.0)


def _init_normal_(tensor, mean=0.0, std=1.0, generator=None):
    return tensor.normal_(mean, std, generator=generator)


def _init_uniform_(tensor, a=0.0, b=1.0, generator=None):
    return tensor.uniform_(a, b, generator=generator)


def _init_trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
    return _C.trunc_normal_(tensor, mean, std, a, b, generator=generator)


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError("fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    receptive_field_size = 1
    if dimensions > 2:
        for value in tensor.shape[2:]:
            receptive_field_size *= value
    fan_in = tensor.shape[1] * receptive_field_size
    fan_out = tensor.shape[0] * receptive_field_size
    return fan_in, fan_out


def _calculate_gain(nonlinearity, param=None):
    name = str(nonlinearity).lower()
    if name in {"linear", "sigmoid", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d"}:
        return 1.0
    if name == "tanh":
        return 5.0 / 3.0
    if name == "relu":
        return _math.sqrt(2.0)
    if name == "leaky_relu":
        negative_slope = 0.01 if param is None else _builtins.float(param)
        return _math.sqrt(2.0 / (1.0 + negative_slope * negative_slope))
    if name == "selu":
        return 0.75
    raise ValueError(f"unsupported nonlinearity {nonlinearity!r}")


def _init_xavier_uniform_(tensor, gain=1.0, generator=None):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = _builtins.float(gain) * _math.sqrt(6.0 / _builtins.float(fan_in + fan_out))
    return tensor.uniform_(-bound, bound, generator=generator)


def _init_xavier_normal_(tensor, gain=1.0, generator=None):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = _builtins.float(gain) * _math.sqrt(2.0 / _builtins.float(fan_in + fan_out))
    return tensor.normal_(0.0, std, generator=generator)


def _init_kaiming_uniform_(tensor, a=0.0, mode="fan_in", nonlinearity="leaky_relu", generator=None):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        fan = fan_in
    elif mode == "fan_out":
        fan = fan_out
    else:
        raise ValueError("mode must be 'fan_in' or 'fan_out'")
    gain = _calculate_gain(nonlinearity, a)
    std = gain / _math.sqrt(_builtins.float(fan))
    bound = _math.sqrt(3.0) * std
    return tensor.uniform_(-bound, bound, generator=generator)


def _init_kaiming_normal_(tensor, a=0.0, mode="fan_in", nonlinearity="leaky_relu", generator=None):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        fan = fan_in
    elif mode == "fan_out":
        fan = fan_out
    else:
        raise ValueError("mode must be 'fan_in' or 'fan_out'")
    gain = _calculate_gain(nonlinearity, a)
    return tensor.normal_(0.0, gain / _math.sqrt(_builtins.float(fan)), generator=generator)


functional.relu = _C.relu
functional.leaky_relu = _C.leaky_relu
functional.silu = _C.silu
functional.elu = _C.elu
functional.selu = _C.selu
functional.softplus = _C.softplus
functional.hardtanh = _C.hardtanh
functional.gelu = _C.gelu
functional.relu6 = _C.relu6
functional.hardsigmoid = _C.hardsigmoid
functional.hardswish = _C.hardswish
functional.softsign = _C.softsign
functional.logsigmoid = _logsigmoid
functional.glu = _glu
functional.mish = _C.mish
functional.dropout = _C.dropout
functional.linear = _C.linear
functional.conv1d = _conv1d
functional.conv_transpose1d = _C.conv_transpose1d
functional.conv2d = _conv2d
functional.conv3d = _conv3d
functional.conv_transpose2d = _C.conv_transpose2d
functional.conv_transpose3d = _C.conv_transpose3d
functional.max_pool1d = _C.max_pool1d
functional.avg_pool1d = _C.avg_pool1d
functional.max_pool2d = _C.max_pool2d
functional.avg_pool2d = _C.avg_pool2d
functional.adaptive_avg_pool1d = _adaptive_avg_pool1d
functional.adaptive_avg_pool2d = _adaptive_avg_pool2d
functional.unfold = _C.unfold
functional.fold = _C.fold
functional.pad = _C.pad
functional.interpolate = _C.interpolate
functional.grid_sample = _C.grid_sample
functional.affine_grid = _C.affine_grid
functional.upsample = _C.interpolate
functional.scaled_dot_product_attention = _scaled_dot_product_attention
functional.normalize = _normalize
functional.cosine_similarity = _cosine_similarity
functional.one_hot = _one_hot
functional.pixel_shuffle = _pixel_shuffle
functional.pixel_unshuffle = _pixel_unshuffle
functional.channel_shuffle = _channel_shuffle
functional.layer_norm = _C.layer_norm
functional.rms_norm = _C.rms_norm
functional.batch_norm = _C.batch_norm
functional.group_norm = _C.group_norm
functional.embedding = _C.embedding
functional.mse_loss = _C.mse_loss
functional.l1_loss = _C.l1_loss
functional.nll_loss = _C.nll_loss
functional.cross_entropy = _C.cross_entropy
functional.binary_cross_entropy = _C.binary_cross_entropy
functional.binary_cross_entropy_with_logits = _C.binary_cross_entropy_with_logits
functional.sigmoid = _C.sigmoid
functional.tanh = _C.tanh
functional.softmin = _softmin
functional.softmax = _C.softmax
functional.log_softmax = _C.log_softmax
linalg.matrix_power = _C.matrix_power
linalg.multi_dot = _linalg_multi_dot
linalg.matmul = _C.matmul
linalg.diagonal = _C.diagonal
linalg.norm = _linalg_norm
linalg.vector_norm = _linalg_vector_norm

amp.autocast = autocast
init.calculate_gain = _calculate_gain
init.constant_ = _init_constant_
init.zeros_ = _init_zeros_
init.ones_ = _init_ones_
init.normal_ = _init_normal_
init.uniform_ = _init_uniform_
init.trunc_normal_ = _init_trunc_normal_
init.xavier_uniform_ = _init_xavier_uniform_
init.xavier_normal_ = _init_xavier_normal_
init.kaiming_uniform_ = _init_kaiming_uniform_
init.kaiming_normal_ = _init_kaiming_normal_
nn.Module = Module
nn.Parameter = Parameter
nn.Linear = Linear
nn.Conv1d = Conv1d
nn.ConvTranspose1d = ConvTranspose1d
nn.Conv2d = Conv2d
nn.Conv3d = Conv3d
nn.ConvTranspose2d = ConvTranspose2d
nn.ConvTranspose3d = ConvTranspose3d
nn.Upsample = Upsample
nn.UpsamplingNearest2d = UpsamplingNearest2d
nn.UpsamplingBilinear2d = UpsamplingBilinear2d
nn.MaxPool1d = MaxPool1d
nn.AvgPool1d = AvgPool1d
nn.AdaptiveAvgPool1d = AdaptiveAvgPool1d
nn.MaxPool2d = MaxPool2d
nn.AvgPool2d = AvgPool2d
nn.AdaptiveAvgPool2d = AdaptiveAvgPool2d
nn.Unfold = Unfold
nn.Fold = Fold
nn.ConstantPad1d = ConstantPad1d
nn.ConstantPad2d = ConstantPad2d
nn.ConstantPad3d = ConstantPad3d
nn.ZeroPad1d = ZeroPad1d
nn.ZeroPad2d = ZeroPad2d
nn.ZeroPad3d = ZeroPad3d
nn.ReflectionPad1d = ReflectionPad1d
nn.ReflectionPad2d = ReflectionPad2d
nn.ReflectionPad3d = ReflectionPad3d
nn.ReplicationPad1d = ReplicationPad1d
nn.ReplicationPad2d = ReplicationPad2d
nn.ReplicationPad3d = ReplicationPad3d
nn.CircularPad1d = CircularPad1d
nn.CircularPad2d = CircularPad2d
nn.CircularPad3d = CircularPad3d
nn.PixelShuffle = PixelShuffle
nn.PixelUnshuffle = PixelUnshuffle
nn.ChannelShuffle = ChannelShuffle
nn.MultiheadAttention = MultiheadAttention
nn.Embedding = Embedding
nn.ReLU = ReLU
nn.ReLU6 = ReLU6
nn.LeakyReLU = LeakyReLU
nn.SiLU = SiLU
nn.ELU = ELU
nn.SELU = SELU
nn.Softplus = Softplus
nn.Hardtanh = Hardtanh
nn.GELU = GELU
nn.Hardsigmoid = Hardsigmoid
nn.Hardswish = Hardswish
nn.Softsign = Softsign
nn.LogSigmoid = LogSigmoid
nn.GLU = GLU
nn.Mish = Mish
nn.Dropout = Dropout
nn.Identity = Identity
nn.Flatten = Flatten
nn.Unflatten = Unflatten
nn.Sequential = Sequential
nn.ModuleList = ModuleList
nn.ModuleDict = ModuleDict
nn.ParameterList = ParameterList
nn.ParameterDict = ParameterDict
nn.Sigmoid = Sigmoid
nn.Tanh = Tanh
nn.Softmax = Softmax
nn.Softmin = Softmin
nn.LogSoftmax = LogSoftmax
nn.Softmax2d = Softmax2d
nn.LayerNorm = LayerNorm
nn.RMSNorm = RMSNorm
nn.BatchNorm1d = BatchNorm1d
nn.BatchNorm2d = BatchNorm2d
nn.BatchNorm3d = BatchNorm3d
nn.GroupNorm = GroupNorm
nn.CosineSimilarity = CosineSimilarity
nn.MSELoss = MSELoss
nn.L1Loss = L1Loss
nn.NLLLoss = NLLLoss
nn.CrossEntropyLoss = CrossEntropyLoss
nn.BCELoss = BCELoss
nn.BCEWithLogitsLoss = BCEWithLogitsLoss
nn.functional = functional
nn.init = init

parameter = ModuleType("mtorch.nn.parameter")
parameter.Parameter = Parameter
nn.parameter = parameter


autograd = ModuleType("mtorch.autograd")


def _as_tensor_tuple(value, name):
    if isinstance(value, Tensor):
        return (value,)
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a Tensor or an iterable of Tensors") from exc
    for item in result:
        if not isinstance(item, Tensor):
            raise TypeError(f"{name} must contain only Tensors")
    return result


def _as_optional_tensor_tuple(value, length):
    if value is None:
        return tuple(None for _ in range(length))
    if isinstance(value, Tensor) or value is None:
        if length != 1:
            raise RuntimeError("grad_outputs must match outputs length")
        return (value,)
    result = tuple(value)
    if len(result) != length:
        raise RuntimeError("grad_outputs must match outputs length")
    for item in result:
        if item is not None and not isinstance(item, Tensor):
            raise TypeError("grad_outputs must contain Tensors or None")
    return result


def _autograd_grad(
    outputs,
    inputs,
    grad_outputs=None,
    retain_graph=None,
    create_graph=False,
    only_inputs=True,
    allow_unused=None,
    is_grads_batched=False,
    materialize_grads=False,
):
    if create_graph:
        raise NotImplementedError("autograd.grad create_graph=True is not implemented yet")
    if is_grads_batched:
        raise NotImplementedError("autograd.grad is_grads_batched=True is not implemented yet")
    outputs_tuple = _as_tensor_tuple(outputs, "outputs")
    inputs_tuple = _as_tensor_tuple(inputs, "inputs")
    grad_outputs_tuple = _as_optional_tensor_tuple(grad_outputs, len(outputs_tuple))
    if allow_unused is None:
        allow_unused = materialize_grads
    return _C._autograd_grad(
        outputs_tuple,
        inputs_tuple,
        grad_outputs_tuple,
        allow_unused=allow_unused,
        materialize_grads=materialize_grads,
    )


autograd.grad = _autograd_grad


optim = ModuleType("mtorch.optim")


def _ensure_param_list(params):
    if isinstance(params, Tensor):
        return [params]
    return list(params)


class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = dict(defaults)
        self.param_groups = []
        self.state = {}
        params = list(params) if not isinstance(params, Tensor) else [params]
        if params and isinstance(params[0], dict):
            for group in params:
                merged = dict(self.defaults)
                merged.update(group)
                merged["params"] = _ensure_param_list(merged["params"])
                self.param_groups.append(merged)
        else:
            group = dict(self.defaults)
            group["params"] = params
            self.param_groups.append(group)
        if not self.param_groups:
            raise ValueError("optimizer got an empty parameter list")

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()

    def state_dict(self):
        return {"state": self.state, "param_groups": self.param_groups}

    def load_state_dict(self, state_dict):
        self.state = state_dict["state"]
        self.param_groups = state_dict["param_groups"]


class SGD:
    def __init__(
        self,
        params,
        lr=0.001,
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        maximize=False,
        foreach=None,
        differentiable=False,
        fused=None,
    ):
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self._optimizer = Optimizer(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "dampening": dampening,
                "weight_decay": weight_decay,
                "nesterov": nesterov,
                "maximize": maximize,
            },
        )
        self.defaults = self._optimizer.defaults
        self.param_groups = self._optimizer.param_groups
        self.state = self._optimizer.state

    def zero_grad(self, set_to_none=True):
        return self._optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self._optimizer.load_state_dict(state_dict)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                update = grad
                if weight_decay != 0.0:
                    update = update + parameter * weight_decay
                if momentum != 0.0:
                    state = self.state.setdefault(id(parameter), {})
                    buffer = state.get("momentum_buffer")
                    if buffer is None:
                        buffer = update.clone().detach()
                        state["momentum_buffer"] = buffer
                    else:
                        buffer.mul_(momentum)
                        if dampening != 1.0:
                            buffer.add_(update * (1.0 - dampening))
                    update = update + buffer * momentum if nesterov else buffer
                parameter.copy_(parameter + update * lr if maximize else parameter - update * lr)
        return loss


class Adam:
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        maximize=False,
        foreach=None,
        capturable=False,
        differentiable=False,
        fused=None,
    ):
        beta1, beta2 = betas
        if not 0.0 <= lr:
            raise ValueError("invalid learning rate")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("invalid beta parameter at index 0")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("invalid beta parameter at index 1")
        if not 0.0 <= eps:
            raise ValueError("invalid epsilon value")
        self._optimizer = Optimizer(
            params,
            {
                "lr": lr,
                "betas": (beta1, beta2),
                "eps": eps,
                "weight_decay": weight_decay,
                "amsgrad": amsgrad,
                "maximize": maximize,
            },
        )
        self.defaults = self._optimizer.defaults
        self.param_groups = self._optimizer.param_groups
        self.state = self._optimizer.state

    def zero_grad(self, set_to_none=True):
        return self._optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self._optimizer.load_state_dict(state_dict)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if maximize:
                    grad = -grad
                if weight_decay != 0.0:
                    grad = grad + parameter * weight_decay
                state = self.state.setdefault(id(parameter), {})
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = zeros_like(parameter)
                    state["exp_avg_sq"] = zeros_like(parameter)
                    if amsgrad:
                        state["max_exp_avg_sq"] = zeros_like(parameter)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1)
                exp_avg.add_(grad * (1.0 - beta1))
                exp_avg_sq.mul_(beta2)
                exp_avg_sq.add_((grad * grad) * (1.0 - beta2))

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    max_exp_avg_sq.copy_(maximum(max_exp_avg_sq, exp_avg_sq))
                    denom = max_exp_avg_sq.sqrt() / _math.sqrt(bias_correction2) + eps
                else:
                    denom = exp_avg_sq.sqrt() / _math.sqrt(bias_correction2) + eps
                parameter.copy_(parameter - (exp_avg / denom) * (lr / bias_correction1))
        return loss


class AdamW(Adam):
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        maximize=False,
        foreach=None,
        capturable=False,
        differentiable=False,
        fused=None,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if weight_decay != 0.0:
                    parameter.copy_(parameter * (1.0 - lr * weight_decay))
                if maximize:
                    grad = -grad
                state = self.state.setdefault(id(parameter), {})
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = zeros_like(parameter)
                    state["exp_avg_sq"] = zeros_like(parameter)
                    if amsgrad:
                        state["max_exp_avg_sq"] = zeros_like(parameter)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1)
                exp_avg.add_(grad * (1.0 - beta1))
                exp_avg_sq.mul_(beta2)
                exp_avg_sq.add_((grad * grad) * (1.0 - beta2))

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    max_exp_avg_sq.copy_(maximum(max_exp_avg_sq, exp_avg_sq))
                    denom = max_exp_avg_sq.sqrt() / _math.sqrt(bias_correction2) + eps
                else:
                    denom = exp_avg_sq.sqrt() / _math.sqrt(bias_correction2) + eps
                parameter.copy_(parameter - (exp_avg / denom) * (lr / bias_correction1))
        return loss


optim.Optimizer = Optimizer
optim.SGD = SGD
optim.Adam = Adam
optim.AdamW = AdamW

cuda.is_available = _cuda_is_available
cuda.device_count = _cuda_device_count
cuda.empty_cache = _cuda_empty_cache
cuda.manual_seed = _cuda_manual_seed
cuda.manual_seed_all = _cuda_manual_seed
cuda.is_bf16_supported = _cuda_is_bf16_supported
cuda.amp = cuda_amp
amp.GradScaler = GradScaler
cuda_amp.autocast = _CudaAutocast
cuda_amp.GradScaler = GradScaler
mps.is_available = _mps_is_available
mps.is_built = _mps_is_built
backends.cuda = backends_cuda
backends_cuda.matmul = _BackendFlags(allow_tf32=False)
backends_cuda.SDPBackend = SDPBackend
backends_cuda.SDPAParams = SDPAParams
backends_cuda.enable_flash_sdp = _enable_flash_sdp
backends_cuda.flash_sdp_enabled = _flash_sdp_enabled
backends_cuda.enable_math_sdp = _enable_math_sdp
backends_cuda.math_sdp_enabled = _math_sdp_enabled
backends_cuda.enable_mem_efficient_sdp = _enable_mem_efficient_sdp
backends_cuda.mem_efficient_sdp_enabled = _mem_efficient_sdp_enabled
backends_cuda.enable_cudnn_sdp = _enable_cudnn_sdp
backends_cuda.cudnn_sdp_enabled = _cudnn_sdp_enabled
backends_cuda.can_use_flash_attention = _can_use_flash_attention
backends_cuda.can_use_efficient_attention = _can_use_efficient_attention
backends_cuda.can_use_cudnn_attention = _can_use_cudnn_attention
backends_cuda.sdp_kernel = _sdp_kernel
backends.mps = backends_mps
backends_mps.is_available = _mps_is_available
backends_mps.is_built = _mps_is_built
backends.cudnn = backends_cudnn
backends_cudnn.enabled = False
backends_cudnn.benchmark = False
backends_cudnn.deterministic = False
backends_cudnn.allow_tf32 = False
nn.attention = attention
attention.SDPBackend = SDPBackend
attention.SDPAParams = SDPAParams
attention.WARN_FOR_UNFUSED_KERNELS = False
attention.sdpa_kernel = _sdpa_kernel
attention.can_use_flash_attention = _can_use_flash_attention
attention.can_use_efficient_attention = _can_use_efficient_attention
utils.checkpoint = utils_checkpoint
utils_checkpoint.checkpoint = _checkpoint

_sys.modules[__name__ + ".nn"] = nn
_sys.modules[__name__ + ".nn.functional"] = functional
_sys.modules[__name__ + ".nn.attention"] = attention
_sys.modules[__name__ + ".nn.init"] = init
_sys.modules[__name__ + ".nn.parameter"] = parameter
_sys.modules[__name__ + ".linalg"] = linalg
_sys.modules[__name__ + ".amp"] = amp
_sys.modules[__name__ + ".cuda"] = cuda
_sys.modules[__name__ + ".cuda.amp"] = cuda_amp
_sys.modules[__name__ + ".mps"] = mps
_sys.modules[__name__ + ".backends"] = backends
_sys.modules[__name__ + ".backends.cuda"] = backends_cuda
_sys.modules[__name__ + ".backends.mps"] = backends_mps
_sys.modules[__name__ + ".backends.cudnn"] = backends_cudnn
_sys.modules[__name__ + ".utils"] = utils
_sys.modules[__name__ + ".utils.checkpoint"] = utils_checkpoint
_sys.modules[__name__ + ".autograd"] = autograd
_sys.modules[__name__ + ".optim"] = optim


def _serialize_value(value):
    if isinstance(value, Tensor):
        return {
            "__mtorch_tensor__": True,
            "data": value.tolist(),
            "dtype": str(value.dtype),
            "requires_grad": _builtins.bool(value.requires_grad),
        }
    if isinstance(value, OrderedDict):
        return {"__mtorch_ordered_dict__": True, "items": [(key, _serialize_value(item)) for key, item in value.items()]}
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__mtorch_tuple__": True, "items": [_serialize_value(item) for item in value]}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def _deserialize_value(value, map_location=None):
    if isinstance(value, dict) and value.get("__mtorch_tensor__"):
        dtype_object = globals().get(value["dtype"], float32)
        return tensor(value["data"], dtype=dtype_object, requires_grad=value["requires_grad"], device=map_location or "cpu")
    if isinstance(value, dict) and value.get("__mtorch_ordered_dict__"):
        return OrderedDict((key, _deserialize_value(item, map_location)) for key, item in value["items"])
    if isinstance(value, dict) and value.get("__mtorch_tuple__"):
        return tuple(_deserialize_value(item, map_location) for item in value["items"])
    if isinstance(value, dict):
        return {key: _deserialize_value(item, map_location) for key, item in value.items()}
    if isinstance(value, list):
        return [_deserialize_value(item, map_location) for item in value]
    return value


def save(obj, f, pickle_module=_pickle, pickle_protocol=2, **pickle_save_args):
    payload = _serialize_value(obj)
    if hasattr(f, "write"):
        pickle_module.dump(payload, f, protocol=pickle_protocol, **pickle_save_args)
        return None
    with _builtins.open(f, "wb") as handle:
        pickle_module.dump(payload, handle, protocol=pickle_protocol, **pickle_save_args)
    return None


def load(f, map_location=None, pickle_module=_pickle, **pickle_load_args):
    if hasattr(f, "read"):
        payload = pickle_module.load(f, **pickle_load_args)
    else:
        with _builtins.open(f, "rb") as handle:
            payload = pickle_module.load(handle, **pickle_load_args)
    return _deserialize_value(payload, map_location=map_location)


__all__ = [
    "Tensor",
    "Size",
    "dtype",
    "device",
    "memory_format",
    "layout",
    "Generator",
    "finfo",
    "iinfo",
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "bool",
    "float",
    "double",
    "half",
    "long",
    "int",
    "contiguous_format",
    "preserve_format",
    "channels_last",
    "channels_last_3d",
    "strided",
    "tensor",
    "as_tensor",
    "asarray",
    "zeros",
    "ones",
    "empty",
    "empty_strided",
    "full",
    "from_numpy",
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "arange",
    "linspace",
    "logspace",
    "eye",
    "randint",
    "randint_like",
    "rand",
    "rand_like",
    "randn",
    "randn_like",
    "normal",
    "randperm",
    "manual_seed",
    "initial_seed",
    "multinomial",
    "bernoulli",
    "FloatTensor",
    "DoubleTensor",
    "HalfTensor",
    "LongTensor",
    "IntTensor",
    "BoolTensor",
    "is_grad_enabled",
    "no_grad",
    "inference_mode",
    "is_inference_mode_enabled",
    "enable_grad",
    "set_grad_enabled",
    "autocast",
    "is_autocast_enabled",
    "compile",
    "neg",
    "negative",
    "abs",
    "absolute",
    "exp",
    "expm1",
    "log",
    "log1p",
    "log2",
    "log10",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "sign",
    "floor",
    "ceil",
    "trunc",
    "fix",
    "round",
    "positive",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "asin",
    "arcsin",
    "acos",
    "arccos",
    "atan",
    "arctan",
    "sigmoid",
    "erf",
    "erfc",
    "deg2rad",
    "rad2deg",
    "frac",
    "isnan",
    "isinf",
    "isfinite",
    "signbit",
    "isposinf",
    "isneginf",
    "logical_not",
    "bitwise_not",
    "square",
    "nan_to_num",
    "clamp",
    "clip",
    "clamp_min",
    "clamp_max",
    "softmax",
    "log_softmax",
    "norm",
    "layer_norm",
    "rms_norm",
    "add",
    "sub",
    "subtract",
    "mul",
    "multiply",
    "div",
    "divide",
    "true_divide",
    "pow",
    "floor_divide",
    "float_power",
    "remainder",
    "fmod",
    "atan2",
    "arctan2",
    "hypot",
    "ldexp",
    "nextafter",
    "copysign",
    "heaviside",
    "logaddexp",
    "logaddexp2",
    "xlogy",
    "fmax",
    "fmin",
    "addcmul",
    "addcdiv",
    "maximum",
    "minimum",
    "eq",
    "ne",
    "not_equal",
    "lt",
    "less",
    "le",
    "less_equal",
    "gt",
    "greater",
    "ge",
    "greater_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "isclose",
    "allclose",
    "equal",
    "is_nonzero",
    "lerp",
    "cosine_similarity",
    "matmul",
    "mm",
    "bmm",
    "addmm",
    "addmv",
    "addr",
    "baddbmm",
    "addbmm",
    "vdot",
    "inner",
    "tensordot",
    "kron",
    "chain_matmul",
    "matrix_power",
    "einsum",
    "dot",
    "mv",
    "outer",
    "ger",
    "conv1d",
    "conv_transpose1d",
    "conv2d",
    "conv3d",
    "conv_transpose2d",
    "conv_transpose3d",
    "max_pool1d",
    "avg_pool1d",
    "max_pool2d",
    "adaptive_avg_pool1d",
    "pixel_shuffle",
    "pixel_unshuffle",
    "channel_shuffle",
    "sum",
    "trace",
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
    "gradient",
    "diff",
    "trapezoid",
    "trapz",
    "cumulative_trapezoid",
    "mean",
    "prod",
    "var",
    "std",
    "var_mean",
    "std_mean",
    "all",
    "any",
    "amax",
    "amin",
    "max",
    "argmax",
    "min",
    "argmin",
    "sort",
    "argsort",
    "topk",
    "searchsorted",
    "unique",
    "unique_consecutive",
    "bucketize",
    "quantile",
    "reshape",
    "unflatten",
    "transpose",
    "permute",
    "movedim",
    "moveaxis",
    "swapaxes",
    "swapdims",
    "flatten",
    "ravel",
    "t",
    "broadcast_to",
    "broadcast_shapes",
    "broadcast_tensors",
    "tile",
    "repeat_interleave",
    "flip",
    "fliplr",
    "flipud",
    "rot90",
    "roll",
    "squeeze",
    "unsqueeze",
    "narrow",
    "select",
    "as_strided",
    "diagonal",
    "diag",
    "diagflat",
    "diag_embed",
    "block_diag",
    "tril",
    "triu",
    "split",
    "tensor_split",
    "chunk",
    "unbind",
    "cat",
    "concat",
    "concatenate",
    "stack",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "hstack",
    "vstack",
    "row_stack",
    "dstack",
    "column_stack",
    "meshgrid",
    "cartesian_prod",
    "where",
    "isin",
    "take",
    "index_select",
    "gather",
    "index_put",
    "take_along_dim",
    "scatter",
    "scatter_add",
    "masked_select",
    "masked_fill",
    "nonzero",
    "argwhere",
    "count_nonzero",
    "bincount",
    "relu",
    "selu",
    "clone",
    "numel",
    "is_tensor",
    "is_floating_point",
    "is_complex",
    "is_conj",
    "is_signed",
    "nn",
    "linalg",
    "amp",
    "cuda",
    "mps",
    "backends",
    "utils",
    "autograd",
    "optim",
    "save",
    "load",
]
