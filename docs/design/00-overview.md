# mtorch Refactoring Design Document (Master)

Created: 2026-07-08
Target revision: the working tree before git management was started (~76,000 lines)

> This document presents the overall picture as the master design. For the actual work steps, follow
> `01-rules-and-verification.md` (common rules) and the per-phase procedure guides `02`–`07`.
> Progress is tracked in `PROGRESS.md`.

## 1. Background and Purpose

mtorch is a clone of a PyTorch-compatible tensor library, developed by codex's automated loop
(`run-codex-loop.sh` + `prompt.md`). In terms of functionality it has reached a level that can run
Stable Diffusion-style workloads. On the other hand, due to the nature of the automated loop —
"add code wherever it can be added in a single turn" — the code has become concentrated in a few
huge files, and copy-paste proliferation of isomorphic code has advanced.

This refactoring moves to a structure where both humans and AI agents can safely continue development,
**without changing externally observable behavior (API, numerics, performance)**.

### Goals

1. Split up the huge files: organize into files by responsibility and keep each file's size within a range a human can review
2. Eliminate duplication: resolve the state where "adding one operator requires synchronized edits in 3–6 places"
3. Correct layer responsibilities: return the numeric kernels that leaked into the binding layer, and the test scenario definitions co-located in the benchmark infrastructure, to their proper layers
4. Normalize repository operations: start git management, and separate generated artifacts (403 benchmark-result JSON files) from input data

### Non-goals (what we will not do this time)

- Adding new features or expanding operator coverage
- Implementing a Metal/CUDA backend (keep the current state where `DeviceType::Metal` is only declared). Planned as a follow-on project for AFTER this refactoring completes: phases 5–7, guides `08-phase5-device-layer.md` / `09-phase6-metal-backend.md` / `10-phase7-cuda-backend.md`
- Migrating to pybind11/nanobind (keep the policy of writing directly against the CPython C API; only split and consolidate)
- Converting string-based op dispatch to enums (a future task; see Appendix B)

## 2. Current-State Analysis

### 2.1 The Overall Scale

| File | Lines | Contents |
|---|---|---|
| `cpp/mtorch/core/tensor.cpp` | 32,762 | The entire C++ core (storage/view/elementwise/reduction/conv/attention/autograd) |
| `cpp/mtorch/python/module.cpp` | 12,560 | All CPython C API bindings + some numeric kernels |
| `tests/compat/benchmarking.py` | 10,931 | Benchmark execution engine (~130 lines) + case definitions (~10,800 lines) |
| `mtorch/__init__.py` | 6,484 | The entire Python API layer (nn/optim/autograd/serialization co-located in a single file) |
| `tests/compat/test_tensor_ops.py` | 6,016 | 2 table-driven tests + 195 hand-written individual test functions |
| `tests/compat/cases.py` | 4,107 | The compatibility test data table (healthy) |
| `tests/compat/test_nn_modules.py` | 2,423 | 89 hand-written individual test functions |
| `cpp/mtorch/core/tensor.h` | ~600 | The public header (~250 free-function declarations) — **this is the only cross-layer interface** |
| `compat/*.json` | 403 files / 11MB | Benchmark-result snapshots. **Zero references from code** (only `api_surface_seed.json` is live) |

### 2.2 Problems by Layer

#### C++ core (`tensor.cpp`)

- **Single TU**: ~8,000 lines of anonymous-namespace helper layer + ~25,000 lines of public API implementation. No section comments
- **op dispatch by string comparison**: `op == "..."` appears in 312 places. dtype dispatch is a 2-tier scheme: "a slow path that processes all dtypes via double + 107 hand-written Float32/Float16-only fast paths"
- **Triple duplication of the same op list**: the if-else chain for unary operators (~30 kinds) runs in parallel across 3 lineages — (1) the float32 kernel (L4285–), (2) the double fallback (L14391–), (3) backward differentiation (L14497–) (4 lineages if you include the float16 kernel at L4460–). **Adding one op requires synchronized edits in 3–4 places**
- Copy-paste examples: `half_bits_to_float_array` (L346) and `half_bits_to_neg_float_array` (L370) are identical 23-line ×2 except for the sign. `try_binary_sub/mul_float32_fast_path` (L5654/L5714) are identical 59-line ×2 except for the operator
- Platform dependencies (`__ARM_NEON` in 89 places, `MTORCH_USE_ACCELERATE` in 85 places) are scattered as inline `#if` inside functions
- **A good sign**: `tensor.h` is established as the public interface, and all implementation details are hidden in the anonymous namespace. **The cpp can be split without touching the header**

#### Binding layer (`module.cpp`)

- The whole thing is a single anonymous namespace. The only external symbol is `PyInit__C`. No macros used
- Massive duplication of boilerplate: the identical 5 lines `try { ... } catch (...) { translate_exception(); }` appear in 328 places. `PyArg_ParseTuple(AndKeywords)` in a total of 436 places
- Common helpers (`dtype_from_py`, `factory_options_from_args`, transfer helpers, etc.) exist but are applied inconsistently
- **Leakage of numeric kernels**: the RNG kernels (L118–367), `bernoulli/dropout/multinomial_tensor` (L378–745), `norm_tensor` (L3363), the einsum fast path (L6106–6332), and channels-last conversion (L984–1118) are implemented in the binding layer even though they belong in core

#### Python API layer (`mtorch/__init__.py`)

- Everything corresponding to torch's `torch/nn/`, `torch/optim/`, `torch/autograd/`, etc., is co-located in a single file. Submodules use a pseudo-structure that dynamically generates `types.ModuleType` and manually registers them into `sys.modules` (L2049–2063, L6081–6098)
- As a result, every class's `__module__` becomes `"mtorch"` (not `"mtorch.nn"`), diverging from torch compatibility for pickle/repr
- Dead code: the double definition of `save/load` (the `_C` binding at L1955–1956 is overwritten by the Python implementation at L6135/6145 and becomes unreachable), and the unused `_dropout` (L5310–5334)
- Copy-paste: `Conv1d/2d/3d` + `ConvTranspose1d/2d/3d` are near-exact copies of ~100 lines each (no base class equivalent to torch's `_ConvNd`)
- **torch-incompatible structure**: `SGD`/`Adam`/`AdamW` use composition rather than being subclasses of `Optimizer`. `isinstance(opt, optim.Optimizer)` returns False

#### Test/benchmark infrastructure

- In `benchmarking.py`, against ~130 lines of execution engine, case definitions are ~10,800 lines. Of these, the Stable Diffusion-style scenario functions (setup/path) are **7,425 lines = 68%**. SD variants have combinatorially exploded as copies of functions differing only in name (139 of 415 benchmark IDs are SD-related)
- **Inverted dependency**: `test_tensor_ops.py` imports ~120 private functions (`_case_*_setup`/`_*_path`) from `benchmarking.py` (L8–129)
- `harness.py` (the comparison engine), `cases.py` (the data table), and the conftest parametrize machinery are well designed. However, the hand-written individual tests (195 + 89 functions) do not use this machinery and instead copy-paste isomorphic boilerplate
- The 403 `compat/benchmark_*.json` files are work history left by the codex loop; there is no code that references them

#### Repository operations

- **Source tree not yet under git**. Git now exists with a housekeeping commit (README/LICENSE/pyproject.toml), but the source tree is untracked until the Phase 0 snapshot commit — the history and rollback means that are prerequisites for refactoring are not yet in place
- The compatibility harness in `docs/test-harness-design.md` (using PyTorch itself as a runtime oracle for in-process comparison) is functional, and **this becomes the safety net for the refactoring**

## 3. Refactoring Principles

1. **Behavior preservation**: before and after each step, the results of `pytest tests/compat` (down to the pass/xfail breakdown) must match
2. **Performance preservation**: at the completion of each phase, run the benchmarks and confirm that the median-ratio degradation relative to the baseline is within 5%. Mandatory for phases that touch C++
3. **Do not mix "mechanical moves" and "rewrites" in the same commit**: Phases 1–2 are cut-and-paste only; only from Phase 3 onward is code actually rewritten
4. **Freeze the public interfaces**: `cpp/mtorch/core/tensor.h`, the symbol table of `mtorch._C`, and the public API of `mtorch.*` are not changed within each phase
5. **Each phase can be stopped independently**: even if you stop partway, the repository always remains buildable and all-tests-green

## 4. Target Architecture (Summary)

The detailed split tables and code templates are in each phase's procedure guide. Here we show only the end state.

### 4.1 C++ core: `cpp/mtorch/core/`

- The public header `tensor.h` is kept as-is
- Cross-cutting helpers are promoted to internal headers `detail/*.h` in `namespace mtorch::detail`
- The implementation is split into ~18 files (storage/elementwise/views/indexing/reductions/sorting/linalg/conv/pooling/attention/losses/activations/normalization/resample/random, etc.). Each file under 5,000 lines
- → Guide: `05-phase2b-tensor-cpp-split.md`

### 4.2 Binding layer: `cpp/mtorch/python/`

- `py_common.{h,cpp}` (PyTensor, exception translation, parse helpers) + ~14 responsibility-specific `py_*.cpp` files
- For method tables, each file exposes an extern array, and `module_init.cpp` concatenates them
- The 328 try/catch boilerplate sites are unified into a template wrapper (Phase 3)
- Numeric kernels (RNG, etc.) are relocated to core (Phase 3)
- → Guides: `04-phase2a-module-cpp-split.md`, `06-phase3-dedup.md`

### 4.3 Python package: `mtorch/`

Move to a real package structure conforming to upstream torch:

```
mtorch/
  __init__.py          # metadata, _C re-export, factory wrappers, and at the end `from mtorch import nn, ...`
  _tensor_patches.py   # consolidate the ~60 scattered `Tensor.xxx = ...` patches
  _numeric.py          # einsum fallback, gradient/diff/trapezoid/quantile
  nn/{__init__.py, modules/*.py, functional.py, init.py, parameter.py}
  optim.py, autograd.py, serialization.py, amp.py
  backends/__init__.py, cuda.py, mps.py, utils/checkpoint.py
```

- Circular imports are resolved by "root-tail import + lazy attribute lookup in functional" (the same solution as torch)
- → Guide: `03-phase1-python-package.md`

### 4.4 Test/benchmark infrastructure: `tests/compat/`

- Add a new `scenarios/` package and promote the setup/path functions to "shared scenario definitions" (referenced as public API from both benchmarks and correctness tests)
- Generate SD variants from the Cartesian product of dataclass parameters, compressing the 139 name-differing functions into single digits
- Make `compat/` input-data only, and delete the 403 `benchmark_*.json` files. Benchmark output goes to `benchmark-results/`, which is gitignored
- → Guides: `02-phase0-baseline.md` (JSON cleanup), `07-phase4-test-infra.md`

### 4.5 Build system

Glob the `sources` in `setup.py`. The full build of the 33k-line single TU becomes a per-file parallel/incremental build.

## 5. Phase Plan

| Phase | Contents | Estimate | Guide |
|---|---|---|---|
| 0 | Full-tree snapshot commit, record baseline, JSON cleanup, prepare comparison scripts | Half a day | `02-phase0-baseline.md` |
| 1 | Split the Python package (mechanical moves only) | 1–2 days | `03-phase1-python-package.md` |
| 2a | Split `module.cpp` (mechanical moves only) | 1–2 days | `04-phase2a-module-cpp-split.md` |
| 2b | Split `tensor.cpp` (promote detail headers + mechanical moves) | 3–5 days | `05-phase2b-tensor-cpp-split.md` |
| 3 | Deduplication and responsibility correction (rewrite code for the first time) | 1–2 weeks | `06-phase3-dedup.md` |
| 4 | Test/benchmark infrastructure cleanup (can run in parallel with other phases) | 1 week | `07-phase4-test-infra.md` |

Follow-on project (GPU backends) — starts only after the §7 completion criteria are all met:

| Phase | Contents | Estimate | Guide |
|---|---|---|---|
| 5 | Device/backend infrastructure (allocator registry, transfer API, Python device surface) | 2–3 days | `08-phase5-device-layer.md` |
| 6 | Metal backend, Python device `"mps"` (Apple Silicon; local machine) | 1–2 weeks | `09-phase6-metal-backend.md` |
| 7 | CUDA backend, device `"cuda"` (requires a separate Linux/NVIDIA machine) | 1–2 weeks | `10-phase7-cuda-backend.md` |

Dependencies:

```
Phase 0 ──→ Phase 1 (Python) ──────→ Phases 3-5 (Python rewrite)
       └──→ Phase 2a → Phase 2b ──→ Phases 3-1–3-4 (C++ rewrite)
       └──→ Phase 4 (can run in parallel; 4-2 after 4-1)
```

## 6. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Performance degradation from lost inlining due to TU splitting | Gate on the full benchmark at Phase 2b completion. Address in the order: make hot helpers `inline` in headers → `-flto` |
| Mistaking guard conditions when merging fast paths (silent numeric behavior change) | 1 fast path = 1 commit. Scrutinize guard conditions via diff before merging. Since the compatibility harness does oracle comparison, value divergences are detected by tests |
| ODR violations / symbol collisions when splitting the anonymous namespace | Explicitly promote cross-cutting helpers to `mtorch::detail`. Things kept file-local do not collide because each TU has its own anonymous namespace |
| Backward incompatibility of the custom serialization due to `__module__` changes | Verify with `test_serialization.py`. If compatibility with previously saved files is needed, place old-name aliases on the load side |
| Circular imports in the Python split | An import-order convention (root-tail import + lazy lookup in functional). Successful import itself is verified by `test_api_surface.py` |
| Dropped cases during tabularization/parametrization | Machine-compare the test collection count and the benchmark case ID list before and after the migration |
| Un-reviewable huge diffs | Separate "mechanical move" and "rewrite" commits. Move commits can be checked by matching line counts |

## 7. Completion Criteria and Success Metrics

1. The pass/xfail breakdown of `pytest tests/compat` matches the baseline (excluding intended improvement diffs)
2. The median ratio of the full benchmark is within 5% of the baseline
3. Maximum file size: C++ implementation under 5,000 lines per file, Python under 2,000 lines per file, test definitions mainly data tables
4. When "adding one unary/binary operator," the number of files to edit drops from the current 6+ places to 3 or fewer
5. No numeric kernels exist in the binding layer
6. No generated JSON exists in `compat/`, and git management is functioning

## Appendix B: Future Tasks Deferred This Time

- **Convert op dispatch to enums**: replace `std::string` op names (312 string comparisons) with an enum + table. Since this involves changing the public signatures of `tensor.h`, make it a separate project after the Phase 3 tabularization has stabilized
- **Unify dtype dispatch with templates**: unify into something equivalent to PyTorch's `AT_DISPATCH_*`. Prerequisite is the completion of fast-path merging (Phase 3-3)
- **pybind11/nanobind migration**: once consolidation into `py_common` is done, the migration cost drops, but after the boilerplate is unified, direct C API code should be maintainable at a manageable scale
- **Metal backend**: the backend abstraction comes after the directory split is done. Now fully planned: see `08-phase5-device-layer.md` (device layer), `09-phase6-metal-backend.md` (Metal), and `10-phase7-cuda-backend.md` (CUDA)
