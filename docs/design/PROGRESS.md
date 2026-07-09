# Refactoring Progress Checklist

## How to use (for the working agent)

- Read this file at the start of a work session. **The first incomplete (`[ ]`) step** is the next work target
- When you complete a step, mark it `[x]`, fill in `commit:` with the hash (`git rev-parse --short HEAD`)
  and `date:` with the date, then commit
- If it fails or is interrupted, leave it as `[ ]`, append `**BLOCKED**` at the end of the line, and write the situation in the notes field within 3 lines
- For procedure details, see each phase's procedure document (02–07 in this directory)
- For issues or observations not covered by the procedure documents, leave a note in the relevant phase's "Notes" field (do not fix them on your own)

## Phase 0: Laying the foundation (procedure: 02-phase0-baseline.md)

- [ ] 0-1 Environment check — commit: / date:
- [ ] 0-2 .gitignore and full-tree snapshot commit — commit: / date:
- [ ] 0-3 Create tools/compare_benchmarks.py — commit: / date:
- [ ] 0-4 Record baseline (tests / collect-count / benchmark) — commit: / date:
- [ ] 0-5 Delete compat/benchmark_*.json — commit: / date:
- [ ] 0-6 Phase completion check — commit: / date:

Notes:

## Phase 1: Splitting the Python package (procedure: 03-phase1-python-package.md)

- [ ] 1-0 Insert markers in the root trailing section (§0-b) — commit: / date:
- [ ] 1-1 mtorch/serialization.py — commit: / date:
- [ ] 1-2 mtorch/optim.py — commit: / date:
- [ ] 1-3 mtorch/autograd.py — commit: / date:
- [ ] 1-4 amp.py / cuda.py / mps.py / backends/ / utils/ — commit: / date:
- [ ] 1-5 mtorch/_numeric.py — commit: / date:
- [ ] 1-6-a Extract the nn-scoped Tensor patches — commit: / date:
- [ ] 1-6-b nn/parameter.py and nn/modules/module.py — commit: / date:
- [ ] 1-6-c nn/functional.py and nn/init.py (including removing the _dropout dead code) — commit: / date:
- [ ] 1-6-d nn/modules/ layers (linear/conv/pooling/padding/attention/sparse/activation/container/normalization/loss) — commit: / date:
- [ ] 1-6-e Complete nn/__init__.py and remove the pseudo-modules — commit: / date:
- [ ] 1-7 mtorch/_tensor_patches.py — commit: / date:
- [ ] 1-8 Root finishing (fully remove ModuleType/sys.modules, verify __all__) — commit: / date:

Notes:

## Phase 2a: Splitting module.cpp (procedure: 04-phase2a-module-cpp-split.md)

- [ ] 2a-1 Glob-ify setup.py — commit: / date:
- [ ] 2a-2 py_common.{h,cpp} — commit: / date:
- [ ] 2a-3 module_init.cpp and the table-concatenation mechanism — commit: / date:
- [ ] 2a-4 py_random.cpp — commit: / date:
- [ ] 2a-5 py_creation.cpp — commit: / date:
- [ ] 2a-6 py_pointwise.cpp — commit: / date:
- [ ] 2a-7 py_reduction.cpp — commit: / date:
- [ ] 2a-8 py_sort_search.cpp — commit: / date:
- [ ] 2a-9 py_linalg.cpp — commit: / date:
- [ ] 2a-10 py_index_select.cpp — commit: / date:
- [ ] 2a-11 py_shape.cpp — commit: / date:
- [ ] 2a-12 py_nn_functional.cpp — commit: / date:
- [ ] 2a-13 py_autograd.cpp — commit: / date:
- [ ] 2a-14 py_indexing.{h,cpp} — commit: / date:
- [ ] 2a-15 py_tensor_type.cpp (including module.cpp cleanup) — commit: / date:
- [ ] 2a-16 Phase completion check (full benchmark comparison) — commit: / date:

Notes:

## Phase 2b: Splitting tensor.cpp (procedure: 05-phase2b-tensor-cpp-split.md)

- [ ] 2b-1a detail/platform.h — commit: / date:
- [ ] 2b-1b detail/common.{h,cpp} — commit: / date:
- [ ] 2b-1c detail/half.h — commit: / date:
- [ ] 2b-1d detail/storage.h — commit: / date:
- [ ] 2b-1e detail/accelerate.{h,cpp} — commit: / date:
- [ ] 2b-1f detail/broadcast.{h,cpp} — commit: / date:
- [ ] 2b-1g detail/promotion.{h,cpp} — commit: / date:
- [ ] 2b-1h detail/factory.{h,cpp} — commit: / date:
- [ ] 2b-1i detail/elementwise.{h,cpp} — commit: / date:
- [ ] 2b-2 tensor_core.cpp — commit: / date:
- [ ] 2b-3 losses.cpp — commit: / date:
- [ ] 2b-4 activations.cpp — commit: / date:
- [ ] 2b-5 normalization.cpp — commit: / date:
- [ ] 2b-6 sorting.cpp — commit: / date:
- [ ] 2b-7 pooling.cpp — commit: / date:
- [ ] 2b-8 attention.cpp — commit: / date:
- [ ] 2b-9 linalg.cpp — commit: / date:
- [ ] 2b-10 reductions.cpp — commit: / date:
- [ ] 2b-11 indexing.cpp — commit: / date:
- [ ] 2b-12 resample.cpp — commit: / date:
- [ ] 2b-13 conv.cpp — commit: / date:
- [ ] 2b-14 views.cpp — commit: / date:
- [ ] 2b-15 elementwise_ops.cpp — commit: / date:
- [ ] 2b-16 Confirm and remove the tensor.cpp remnants — commit: / date:
- [ ] 2b-17 Performance gate (full benchmark comparison) — commit: / date:

Notes:

## Phase 3: Deduplication and responsibility fixing (procedure: 06-phase3-dedup.md)

- [ ] 3-1-a Unify the try/catch wrapper (all binding files) — commit: / date:
- [ ] 3-1-b Table-ify creation ops — commit: / date:
- [ ] 3-1-c Unify Tensor method forwarding — commit: / date:
- [ ] 3-2 Table-ify unary ops (float32 → double → backward → float16, 4 commits + cleanup) — commit: / date:
- [ ] 3-3 Consolidate binary fast paths (starting from the sub/mul pair) — commit: / date:
- [ ] 3-4 Move numeric kernels into core (random → norm → einsum → channels-last) — commit: / date:
- [ ] 3-5-a _ConvNd base class — commit: / date:
- [ ] 3-5-b Make Optimizer a true subclass — commit: / date:
- [ ] 3-6 Resolve small-grained duplication — commit: / date:

Notes:

## Phase 4: Organizing the test/benchmark infrastructure (procedure: 07-phase4-test-infra.md)

- [ ] 4-0 Pre-capture the benchmark ID list — commit: / date:
- [ ] 4-1-a Create the scenarios/ shim and rewire the test_tensor_ops imports — commit: / date:
- [ ] 4-1-b Physically move the scenario definitions (per domain) — commit: / date:
- [ ] 4-1-c Split BENCHMARK_CASES (including extracting bench_types.py) — commit: / date:
- [ ] 4-2 Parameterize SD variants (per family) — commit: / date:
- [ ] 4-3 Table-ify PATH_CASES (driver + gradual migration) — commit: / date:
- [ ] 4-4 Convert to ModuleCase — commit: / date:
- [ ] 4-5 Operationalize the API manifest (optional) — commit: / date:

Notes:

## Overall completion check

- [ ] Completion criterion 1: Full test matches the baseline (00-overview.md §7)
- [ ] Completion criterion 2: Full benchmark within 5%
- [ ] Completion criterion 3: File size limits (C++ 5,000 lines / Python 2,000 lines)
- [ ] Completion criterion 4: 3 or fewer edit sites for adding an op
- [ ] Completion criterion 5: No numeric kernels in the binding layer
- [ ] Completion criterion 6: No generated artifacts in compat/, git workflow established

Notes:
