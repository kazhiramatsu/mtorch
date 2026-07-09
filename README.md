# mtorch

A PyTorch-compatible tensor library built from scratch: a C++20 core with
CPython C API bindings and a pure-Python API layer that mirrors `torch`.
CPU-only, optimized for Apple Silicon (NEON intrinsics + the Accelerate
framework). It has enough operator, autograd, `nn`, and `optim` coverage to
run Stable Diffusion-style inference workloads.

The codebase was written almost entirely by an autonomous coding-agent loop
(`run-codex-loop.sh` + `prompt.md`), with upstream PyTorch used only as a
runtime test oracle — see [How it was built](#how-it-was-built).

> **Status: experimental.** The library is functional and its compatibility
> suite is green, but the code is concentrated in a few very large files. A
> full behavior-preserving refactoring is planned and documented in
> [`docs/design/`](docs/design/00-overview.md).

## Quick example

```python
import mtorch
import mtorch.nn as nn

x = mtorch.randn(128, 64)
w = mtorch.randn(64, 32, requires_grad=True)
y = (x @ w).relu().sum()
y.backward()                     # reverse-mode autograd
print(w.grad.shape)              # (64, 32)

model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))
opt = mtorch.optim.SGD(model.parameters(), lr=0.1)
loss = model(x).square().mean()
loss.backward()
opt.step()
```

## Requirements

- Python >= 3.10
- A C++20 compiler
- Developed and tested on macOS / Apple Silicon (uses `-DMTORCH_USE_ACCELERATE`
  and NEON fast paths there; other platforms fall back to portable code paths)
- `torch` and `pytest` — only for running the compatibility test suite;
  mtorch itself has no runtime dependencies

## Build

```bash
python3 -m pip install -e '.[dev]'      # pulls pytest + torch (test oracle)
python3 setup.py build_ext --inplace    # builds mtorch/_C
python3 -c "import mtorch; print(mtorch.__version__)"
```

## Tests and benchmarks

The compatibility harness imports both `torch` and `mtorch` in the same
process, runs each case through both, and compares results (values, dtypes,
shapes, strides, gradients, serialized state) — see
[`docs/test-harness-design.md`](docs/test-harness-design.md).

```bash
pytest tests/compat                                  # full suite
pytest tests/compat -m "not slow"                    # skip large generated suites
pytest tests/compat -m autograd                      # one dimension only

# Benchmarks (median mtorch/torch time ratio per case):
pytest tests/compat/test_benchmarks.py \
  --compat-benchmark 'bench.matmul*' \
  --compat-benchmark-json benchmark-results/matmul.json
```

## Repository layout

| Path | Contents |
|---|---|
| `mtorch/` | Python API layer (`nn`, `optim`, `autograd`, serialization, …) |
| `cpp/mtorch/core/` | C++ tensor core (storage, kernels, autograd) |
| `cpp/mtorch/python/` | CPython C API bindings (`mtorch._C`) |
| `tests/compat/` | PyTorch-compatibility harness, test tables, benchmark cases |
| `compat/` | Input data for the harness (API surface seed, benchmark snapshots) |
| `tools/` | Maintenance scripts |
| `docs/` | Design notes and the refactoring plan ([index](docs/README.md)) |

## How it was built

`run-codex-loop.sh` repeatedly invoked a coding agent with `prompt.md`, which
instructed it to extend PyTorch compatibility, keep the test suite green, and
re-benchmark against PyTorch after every change. The result is a working
library with the characteristic scars of that process (a handful of very
large files, duplicated fast paths). The refactoring plan in
[`docs/design/00-overview.md`](docs/design/00-overview.md) addresses exactly
that, phase by phase, with the compatibility harness as the safety net.

## License

[BSD 3-Clause](LICENSE).

mtorch is an independent, from-scratch reimplementation of a subset of the
PyTorch API. It is not affiliated with, endorsed by, or derived from the
PyTorch project. "PyTorch" is a trademark of The Linux Foundation. PyTorch
itself is used in this repository solely as a reference oracle for the
compatibility test suite.
