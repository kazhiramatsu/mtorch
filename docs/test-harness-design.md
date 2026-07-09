# PyTorch Compatibility Test Harness Design

## Purpose

`mtorch` aims to be a fully compatible clone of PyTorch. Implementation correctness is not judged by a human on an API-by-API basis; instead, a pinned reference PyTorch is used as a runtime oracle and compared against the candidate implementation.

The completion criteria for this harness are made explicit.

1. Pin the target PyTorch version.
2. Turn that version's public API surface into a manifest.
3. For each API, compare value, dtype, shape, stride, device, exceptions, side effects, view aliasing, autograd, randomness, serialization, and module/optimizer behavior.
4. Once `xfail` and unregistered APIs reach 0 and all compatibility tests pass, consider the implementation compatible with that pinned version.

## Basic Structure

- Reference implementation: `--compat-reference=torch`
- Candidate implementation: `--compat-candidate=mtorch`
- API manifest: `compat/api_surface_seed.json` or a generated full manifest
- Entry point: `pytest tests/compat`

Comparisons are done not against golden files, but by building both the reference PyTorch and the candidate implementation from the same inputs within the same process. This lets us treat the observable behavior itself as the specification, rather than re-implementing PyTorch's fine-grained semantics from documentation.

## Test Layers

### 1. API Surface

`tests/compat/test_api_surface.py` checks the following for each path in the manifest.

- The attribute exists
- The kind (module/class/callable/value) matches
- If the manifest has a signature, the signature matches

The initial manifest is `compat/api_surface_seed.json`, used for bootstrapping. To increase completeness, run the following to widen the manifest.

```bash
python3 tools/generate_api_manifest.py --module torch --out compat/api_surface_torch_2_10_0.json --max-depth 3
pytest tests/compat --compat-api-manifest compat/api_surface_torch_2_10_0.json
```

### 2. Operator Semantics

`OP_CASES` in `tests/compat/cases.py` holds deterministic operator cases. Each case builds the same input for the reference and the candidate, then compares the return value and the arguments after the call.

Compared items:

- tensor values
- shape
- dtype
- device
- `requires_grad`
- stride
- Python scalar/list/tuple/dict/`torch.Size`
- the exception class, when the reference raises an exception

Cases can be narrowed with `--compat-op`.

```bash
pytest tests/compat --compat-op 'binary.*'
```

### 3. In-place And View Aliasing

`INPLACE_CASES` compares side effects such as `add_`, `mul_`, and `copy_`. It compares not only the return value but also the input tensor after the call.

`VIEW_CASES` mutates the return values of `reshape`, `transpose`, and `narrow` in place and checks whether the effect on the base tensor matches the reference. For PyTorch compatibility, the storage-alias behavior matters as much as the values.

### 4. Autograd

`GRAD_CASES` compares first-order backward.

- Build the same `requires_grad=True` inputs
- Build a scalar loss
- Call `backward()`
- Compare the output and each leaf tensor's `.grad`

Extend this layer to add the following.

- gradients involving broadcasting
- view/in-place and the version counter
- `torch.autograd.grad`
- higher-order gradients
- no-grad/inference-mode
- custom `Function`

### 5. Reference Leakage

`tests/compat/test_candidate_identity.py` verifies that the candidate module is not the reference PyTorch itself. By default, it also fails an implementation whose candidate `tensor()` returns a `torch.Tensor`.

Only when you want to temporarily allow a PyTorch wrapper during the initial bootstrap, explicitly use the following.

```bash
pytest tests/compat --compat-allow-reference-backed
```

### 6. PyTorch Benchmark Comparison

`tests/compat/test_benchmarks.py` runs lightweight microbenchmarks of the reference PyTorch and the candidate implementation on every run.

By default it does not fail because of a speed difference. Performance numbers are printed in pytest's terminal summary, and when the candidate is slower than PyTorch it is shown as `SLOW`. This is because, even though the initial implementation is known to be slower than PyTorch, we want to keep watching the trend continuously while passing the compatibility tests.

Compared items:

- binary add
- broadcasting add
- reduction sum
- 2-D matmul
- transpose + contiguous
- boolean mask indexing
- mixed integer advanced indexing

Example runs:

```bash
pytest tests/compat
pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10
pytest tests/compat/test_benchmarks.py --compat-benchmark 'bench.matmul*'
pytest tests/compat/test_benchmarks.py --compat-benchmark-json compat/benchmark_results.json
```

The ratio is `candidate median / PyTorch median`. `--compat-benchmark-slow-ratio` sets the threshold for showing `SLOW` in the summary. The default is `1.0`, so anything even slightly slower than PyTorch is shown as `SLOW`.

```bash
pytest tests/compat/test_benchmarks.py --compat-benchmark-slow-ratio=2.0
```

If you want performance regressions to fail, specify `--compat-benchmark-max-ratio`. In that case, benchmarks exceeding the threshold are shown as `FAIL` and pytest also fails.

```bash
pytest tests/compat/test_benchmarks.py --compat-benchmark-max-ratio=1000
```

## Order for Advancing Toward Full Compatibility

1. Get a minimal Tensor object with `tensor`, dtype, shape, stride, storage, and device passing.
2. Get creation ops passing.
3. Get unary/binary/reduction/broadcasting passing.
4. Get view and in-place aliasing / version counter passing.
5. Get the autograd backward graph passing.
6. Get `nn.Module`, `Parameter`, state dict, and optimizers passing.
7. Get RNG, serialization, dtype promotion, indexing, and advanced indexing passing.
8. Widen the API surface with `tools/generate_api_manifest.py` and drive missing APIs to 0.
9. In environments where PyTorch's internal OpInfo is available, add exhaustive operator cases derived from OpInfo.
10. Finally, make the strict run a mandatory condition in CI.

## CI Policy

CI is split into at least 3 tiers.

- `compat-smoke`: seed manifest, deterministic cases, lightweight benchmark summary. Mandatory on every PR.
- `compat-expanded`: generated manifest and expanded operator cases. Mandatory before merging into main.
- `compat-release`: OpInfo, randomized/property tests, serialization, a large dtype/device matrix, and thresholded benchmarks. Mandatory before release.

Compatibility gaps are not hidden behind an allowlist; they are left as failing tests. Only when the volume of failures makes development difficult, narrow the working set with `--compat-op`.
