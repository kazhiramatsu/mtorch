# Native Core Implementation Notes

The initial implementation follows the planned layering:

```text
mtorch Python API
  -> mtorch._C CPython extension
    -> cpp/mtorch/core Tensor implementation
```

`pybind11` and `nanobind` are not installed in the current environment, so the first binding layer uses the CPython C API directly. The C++ core is intentionally separate from `cpp/mtorch/python/module.cpp`, so the binding can be replaced later without rewriting the tensor implementation.

## Implemented First Slice

- CPU-only `Tensor`
- typed CPU storage with shape/stride/offset metadata
- view aliasing for `reshape`, `transpose`, `squeeze`, `unsqueeze`, `narrow`
- dtype-backed storage for `float16`, `float32`, `float64`, `int32`, `int64`, `bool`
- default dtype inference for Python literals matching the common PyTorch cases: float input to `float32`, integer input to `int64`, bool input to `bool`
- dtype casts at tensor creation and `element_size()`
- creation ops: `tensor`, `zeros`, `ones`, `empty`, `full`, `arange`, `linspace`, `eye`, `randn`
- unary/binary ops: `neg`, `abs`, `exp`, `log`, `sqrt`, `add`, `sub`, `mul`, `div`, `pow`
- reductions: `sum`, `mean`, `max(dim=...)`, `argmax(dim=...)`
- shape ops: `reshape`, `transpose`, `squeeze`, `unsqueeze`, `cat`, `stack`
- indexing helpers: `where`, `take`
- 2-D `matmul` / `mm`
- in-place mutation: `add_`, `mul_`, `copy_`, `fill_`
- minimal reverse-mode autograd for the seed harness: elementwise multiply, matmul, relu, sum, mean
- Python API placeholders for `nn`, `autograd`, `optim`, `save`, `load`

## Build

```bash
python3 setup.py build_ext --inplace
```

## Current Compatibility Gate

```bash
pytest tests/compat
```

The seed compatibility harness currently passes. This is not yet PyTorch compatibility; it means the first CPU-only slice satisfies the bootstrap manifest and deterministic seed cases.

## Next Core Milestones

1. Expand dtype promotion rules for scalar/tensor mixed arithmetic, reductions, comparisons, and unsupported dtype errors.
2. Add advanced indexing, slicing, comparison ops, broadcasting gradient reductions, and non-contiguous reshape semantics.
3. Add version counters and PyTorch-compatible in-place/autograd error behavior.
4. Add `nn.Module`, `Parameter`, `state_dict`, serialization, and optimizers.
5. Introduce backend abstraction for CPU, then CUDA/HIP/Metal.
