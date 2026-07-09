# 06. Phase 3: Deduplication and Fixing Layer Responsibilities

Prerequisites: 3-1 through 3-4 require Phase 2a/2b to be complete (all boxes of those phases
checked in `docs/design/PROGRESS.md`); 3-5 requires Phase 1 to be complete. 3-6 items 1 and 3
require Phase 2b; 3-6 item 2 requires Phase 1. `01-rules-and-verification.md` read.

**This is the first phase in which we modify code** (Phases 1/2a/2b only moved it).
Precisely because of that, commit and verify in even smaller increments than before.
The subtasks (3-1 through 3-6) are mutually independent, so you may stop at any point.
Proceed in numerical order.

How to read the anchors in this document: every target is identified two ways.

1. **Primary (post-split tree)**: a `grep -n '<symbol>'` command against the file the symbol
   lives in after Phases 1/2a/2b, using the target file names from
   `04-phase2a-module-cpp-split.md` §1 and `05-phase2b-tensor-cpp-split.md` §1 and
   `03-phase1-python-package.md`. Always run the grep; never trust a line number.
2. **Hint (pre-split location)**: the file and line where the symbol sat in the 2026-07-08
   snapshot (`module.cpp` = `cpp/mtorch/python/module.cpp`, `tensor.cpp` =
   `cpp/mtorch/core/tensor.cpp`, `__init__.py` = `mtorch/__init__.py`). Use it only to
   recognize the code when you find it. If a grep finds nothing in the expected post-split
   file, follow `01-rules-and-verification.md` §9 (search the whole tree, check PROGRESS.md,
   and if still missing, mark BLOCKED — do not guess).

All code blocks labeled BEFORE in this document were copied verbatim from the snapshot.
After the splits the same bytes exist in the post-split file (the split phases were
mechanical moves), possibly with different surrounding namespaces. If what you find differs
from a BEFORE block by more than namespace/`#include` context, stop and mark the step
**BLOCKED** in PROGRESS.md.

Common verification (for **every** commit in this phase):

```bash
python3 setup.py build_ext --inplace
pytest tests/compat 2>&1 | tee /tmp/tests-current.txt
tail -1 docs/design/baseline/tests-baseline.txt | tr -d '=' | sed 's/ in .*//' | xargs
tail -1 /tmp/tests-current.txt | tr -d '=' | sed 's/ in .*//' | xargs
```

The two printed summary lines must be character-for-character identical
(`01-rules-and-verification.md` §5.1; run the collect-count check there too).
For subtasks that touch numeric kernels (3-2, 3-3, 3-4), additionally run the benchmark
commands given inside those steps. Benchmark-subset comparisons in this phase always pipe
through `grep -E '^REGRESSION|^compared'` — when the current JSON contains only a subset of
cases, `MISSING` lines and exit code 1 are produced by design and must be ignored; only
`REGRESSION` lines matter, and each one is subject to the 3-rerun noise rule of
`01-rules-and-verification.md` §5.2.

Common failure protocol (referenced by every "On failure" below): follow
`01-rules-and-verification.md` §6 exactly. Uncommitted changes: `git restore .` (plus
`git clean -fd -- <path>` per new file). Committed changes: `git revert --no-edit HEAD`
(one revert per commit, newest first). Then re-run §5.1, mark the step's line in
`docs/design/PROGRESS.md` with ` **BLOCKED**`, add ≤3 note lines, commit the PROGRESS.md
edit, and stop. (`git reset --hard` is forbidden by `01-rules-and-verification.md` §2 rule 7.)

Commit message format: `refactor(phase3-<step>): <what happened>`, e.g.
`refactor(phase3-1a): route py_creation.cpp bindings through guarded_kw`.

---

## 3-1. Unifying binding boilerplate (target: `cpp/mtorch/python/`)

### 3-1-a: try/catch wrapper (~321 sites)

**Goal**: Every `catch (...) { translate_exception(); return nullptr; }` shell that wraps a
whole binding function is replaced by a single pair of wrapper templates, so
`grep -c 'translate_exception();'` across `cpp/mtorch/python/*.cpp` drops from ~321
(the count in the snapshot; recount first) to a small remainder.

**Preconditions**:

- Phase 2a complete: `ls cpp/mtorch/python/` shows the ~14 files of
  `04-phase2a-module-cpp-split.md` §1 and `module.cpp` no longer exists
  (renamed to `py_tensor_type.cpp` in 2a-15).
- `grep -n 'void translate_exception' cpp/mtorch/python/py_common.cpp` finds the definition
  (hint: old module.cpp L792) and `py_common.h` declares it.
- Record the starting count: `cat cpp/mtorch/python/*.cpp | grep -c 'translate_exception();'`
  (321 in the snapshot).

**Actions**:

1. Append the following to `cpp/mtorch/python/py_common.h`, inside the existing
   `namespace mtorch::py { ... }` block (this is the complete wrapper; the codebase is built
   with `-std=c++20` per `setup.py`, so template arguments with internal linkage are legal):

   ```cpp
   // Wrappers that translate C++ exceptions at the binding boundary.
   // guarded_kw wraps METH_VARARGS | METH_KEYWORDS functions, guarded wraps METH_VARARGS.
   // Usage in a method table:
   //   {"zeros", reinterpret_cast<PyCFunction>(guarded_kw<py_zeros>), METH_VARARGS | METH_KEYWORDS, nullptr},
   //   {"matmul", guarded<py_matmul>, METH_VARARGS, nullptr},

   template <PyObject* (*Fn)(PyObject*, PyObject*, PyObject*)>
   PyObject* guarded_kw(PyObject* self, PyObject* args, PyObject* kwargs) {
       try {
           return Fn(self, args, kwargs);
       } catch (...) {
           translate_exception();
           return nullptr;
       }
   }

   template <PyObject* (*Fn)(PyObject*, PyObject*)>
   PyObject* guarded(PyObject* self, PyObject* args) {
       try {
           return Fn(self, args);
       } catch (...) {
           translate_exception();
           return nullptr;
       }
   }
   ```

   Commit this header change together with the first migrated file (step 4).

2. **Select the functions to migrate, one binding file at a time** (**1 binding file =
   1 commit**). Process the files in this order, skipping any file with zero qualifying
   functions: `py_creation.cpp`, `py_pointwise.cpp`, `py_reduction.cpp`,
   `py_sort_search.cpp`, `py_linalg.cpp`, `py_index_select.cpp`, `py_shape.cpp`,
   `py_nn_functional.cpp`, `py_autograd.cpp`, `py_indexing.cpp`, `py_random.cpp`,
   `py_tensor_type.cpp`, `py_common.cpp`. In each file list the candidates:

   ```bash
   grep -n -B3 'translate_exception();' cpp/mtorch/python/py_creation.cpp
   ```

   A function **qualifies** only if ALL of the following hold (check mechanically, function
   by function):

   - it appears in a `PyMethodDef` table (module table, `Tensor_methods`/`kTensorMethods`,
     or `Generator_methods`) — pure helpers such as `binary_dispatch`, `unary_dispatch`,
     `parse_tensor_call` (old module.cpp L2004/L2024/L2037) keep their internal try/catch
     as-is;
   - its signature is `PyObject* f(PyObject*, PyObject*, PyObject*)` (kwargs form) or
     `PyObject* f(PyObject*, PyObject*)` (varargs form) — functions returning `int`
     (e.g. `Generator_init`, which returns `-1` in its catch) do **not** qualify;
   - it contains exactly one `try {` and its catch block is byte-for-byte:

     ```cpp
       } catch (...) {
         translate_exception();
         return nullptr;
       }
     ```

   Any function with additional handling in the catch (resource release, different return,
   multiple try blocks — e.g. `parse_tensor_call` has two) is **left unchanged**, and you
   add one line to the Phase 3 `Notes:` field of PROGRESS.md naming it.

3. For each qualifying function in the file: delete the `try {` line and the whole
   `} catch (...) { ... }` block, re-indent the former try-body one level left, and change
   nothing else in the body. Worked example (BEFORE is the verbatim snapshot code of
   `py_zeros`, old module.cpp L2093; locate with
   `grep -n 'PyObject\* py_zeros' cpp/mtorch/python/py_creation.cpp`):

   BEFORE:

   ```cpp
   PyObject* py_zeros(PyObject*, PyObject* args, PyObject* kwargs) {
     try {
       const auto options = factory_options_from_args(args, kwargs, "zeros");
       auto result = mtorch::zeros(options.shape, dtype_from_py(options.dtype), device_from_py(options.device));
       result->requires_grad = options.requires_grad;
       return wrap_tensor(result);
     } catch (...) {
       translate_exception();
       return nullptr;
     }
   }
   ```

   AFTER:

   ```cpp
   PyObject* py_zeros(PyObject*, PyObject* args, PyObject* kwargs) {
     const auto options = factory_options_from_args(args, kwargs, "zeros");
     auto result = mtorch::zeros(options.shape, dtype_from_py(options.dtype), device_from_py(options.device));
     result->requires_grad = options.requires_grad;
     return wrap_tensor(result);
   }
   ```

   Note: many functions (e.g. `py_sum`, old L3815) have `PyArg_ParseTupleAndKeywords` and
   `is_tensor` checks *before* the `try`. That is fine — leave those lines where they are;
   after migration the wrapper simply guards them too (they do not throw C++ exceptions, so
   behavior is unchanged).

4. In the same file's method table(s), change each migrated entry:

   - kwargs form — BEFORE:
     `{"zeros", reinterpret_cast<PyCFunction>(py_zeros), METH_VARARGS | METH_KEYWORDS, nullptr},`
     AFTER:
     `{"zeros", reinterpret_cast<PyCFunction>(guarded_kw<py_zeros>), METH_VARARGS | METH_KEYWORDS, nullptr},`
   - varargs form — BEFORE (note: varargs entries in the snapshot have either no cast, e.g.
     `{"matmul", py_matmul, METH_VARARGS, nullptr},`, or a `reinterpret_cast`; keep whichever
     cast style the entry already uses):
     AFTER: `{"matmul", guarded<py_matmul>, METH_VARARGS, nullptr},`

   If a table entry lives in a different file than the function (should not happen after
   2a; if it does, mark BLOCKED), do not migrate that function.

**Verification** (after each file):

1. `python3 setup.py build_ext --inplace`
2. Exception path still live (must print a Python `ValueError` traceback, not a segfault):

   ```bash
   python3 -c "import mtorch; mtorch.zeros((2,2)).reshape(3)"
   ```

   Expected last line: `ValueError: shape is invalid for input size`.
3. Full §5.1 test comparison (commands at the top of this document).
4. `grep -c 'translate_exception();' cpp/mtorch/python/<file>` decreased by exactly the
   number of functions you migrated in that file.

**On failure**: common failure protocol. This step's commits are not mechanical moves, so
§5.3 does not apply.

**Commit** (per file; repeat for each file in the step-2 order):

```bash
git add cpp/mtorch/python/py_common.h cpp/mtorch/python/py_creation.cpp
git commit -m "refactor(phase3-1a): route py_creation.cpp bindings through guarded/guarded_kw"
```

(only the first commit includes `py_common.h`). After the last file, update the 3-1-a line
in PROGRESS.md per `01-rules-and-verification.md` §8.

### 3-1-b: table-ify creation ops

**Goal**: `py_zeros` / `py_ones` / `py_empty` share one implementation parameterized by a
core factory pointer, and the four `*_like` functions share one tail helper.

**Preconditions**: 3-1-a completed for `py_creation.cpp` (the code below assumes the
functions are already wrapped by `guarded_kw`, so the helpers need no try/catch).
Locate the targets: `grep -n 'PyObject\* py_zeros\|PyObject\* py_ones\|PyObject\* py_empty(' cpp/mtorch/python/py_creation.cpp`
(hint: old module.cpp L2093–2127) and
`grep -n '_like(PyObject' cpp/mtorch/python/py_creation.cpp` (hint: old L2184–2348).

**Actions**:

1. **Confirm the three factory bodies still match the snapshot.** `py_zeros`, `py_ones`,
   `py_empty` are 10-line triplets identical except for the name string and the callee;
   verified snapshot facts: all three call `factory_options_from_args`, and **`py_empty`
   calls `mtorch::zeros`** (empty is zero-filled today — keep that; do not "fix" it to an
   uninitialized factory). The core signatures (from `cpp/mtorch/core/tensor.h`, unchanged):

   ```cpp
   TensorPtr zeros(const std::vector<int64_t>& sizes, ScalarType dtype, Device device = Device{});
   TensorPtr ones(const std::vector<int64_t>& sizes, ScalarType dtype, Device device = Device{});
   ```

2. Add to the anonymous namespace of `py_creation.cpp`, above the three functions:

   ```cpp
   PyObject* py_factory_common(
       PyObject* args,
       PyObject* kwargs,
       const char* name,
       mtorch::TensorPtr (*factory)(const std::vector<int64_t>&, mtorch::ScalarType, mtorch::Device)) {
     const auto options = factory_options_from_args(args, kwargs, name);
     auto result = factory(options.shape, dtype_from_py(options.dtype), device_from_py(options.device));
     result->requires_grad = options.requires_grad;
     return wrap_tensor(result);
   }
   ```

   and replace the three bodies with:

   ```cpp
   PyObject* py_zeros(PyObject*, PyObject* args, PyObject* kwargs) {
     return py_factory_common(args, kwargs, "zeros", mtorch::zeros);
   }

   PyObject* py_ones(PyObject*, PyObject* args, PyObject* kwargs) {
     return py_factory_common(args, kwargs, "ones", mtorch::ones);
   }

   PyObject* py_empty(PyObject*, PyObject* args, PyObject* kwargs) {
     return py_factory_common(args, kwargs, "empty", mtorch::zeros);  // empty is zero-filled today; behavior preserved
   }
   ```

   Method table entries do not change (they already point at `guarded_kw<py_zeros>` etc.).

3. **Diff the four `*_like` implementations before consolidating** (do not skip). The
   verified snapshot differences are exactly these and nothing else:

   - all four parse `{"input", ..., "dtype", "layout", "device", "requires_grad",
     "memory_format"}` and call `make_like_with_memory_format(...)`;
   - `py_empty_like` returns the result directly (no fill);
   - `py_zeros_like` additionally calls `result->fill_inplace(0.0)`;
   - `py_ones_like` additionally calls `result->fill_inplace(1.0)`;
   - `py_full_like` parses one extra positional `fill_value` (`"OO|OOOpO:full_like"`) and
     calls `result->fill_inplace(scalar_from_py(fill_value))`.

   Re-verify on the current tree by reading the four functions. **If you find any
   difference not in the list above, do not consolidate**; add a note to PROGRESS.md and
   skip to Verification.

4. Add the shared tail helper to the anonymous namespace of `py_creation.cpp`:

   ```cpp
   PyObject* like_factory_common(
       PyObject* input,
       PyObject* dtype,
       PyObject* device,
       int requires_grad,
       PyObject* memory_format,
       const char* name,
       std::optional<double> fill) {
     if (!is_tensor(input)) {
       PyErr_Format(PyExc_TypeError, "%s expected Tensor", name);
       return nullptr;
     }
     const auto& source = tensor_ref(input);
     auto result = make_like_with_memory_format(
         source,
         dtype_from_py(dtype, source->dtype),
         device_from_py(device, source->device),
         requires_grad != 0,
         optional_memory_format_from_py(memory_format));
     if (fill.has_value()) {
       result->fill_inplace(*fill);
     }
     return wrap_tensor(result);
   }
   ```

   (add `#include <optional>` at the top of the file if not already present). Keep each
   `py_*_like` function's `PyArg_ParseTupleAndKeywords` block exactly as it is (the format
   strings differ per function) and replace everything after the parse + nothing-else with
   one return:

   - `py_empty_like`: `return like_factory_common(input, dtype, device, requires_grad, memory_format, "empty_like", std::nullopt);`
   - `py_zeros_like`: `... "zeros_like", 0.0);`
   - `py_ones_like`: `... "ones_like", 1.0);`
   - `py_full_like`: `... "full_like", scalar_from_py(fill_value));`
     (`scalar_from_py` may throw; that is translated by the `guarded_kw` wrapper from 3-1-a
     exactly as the old in-function catch did).

**Verification**: common verification, plus:

```bash
pytest tests/compat/test_tensor_ops.py --compat-op 'creation.*'
```

**On failure**: common failure protocol.

**Commit**:

```bash
git add cpp/mtorch/python/py_creation.cpp
git commit -m "refactor(phase3-1b): table-ify zeros/ones/empty and consolidate *_like tails"
```

### 3-1-c: unify Tensor method forwarding

**Goal**: `Tensor_forward_keyword_method` (old module.cpp L10485) is the single standard
for "prepend self, call the module-level `py_*` function"; the hand-inlined copies of that
pattern and the matmul-family re-implementations are replaced by forwards.

**Preconditions**: Phase 2a complete; the Tensor method group lives in
`cpp/mtorch/python/py_tensor_type.cpp`. Locate the helper:
`grep -n 'Tensor_forward_keyword_method' cpp/mtorch/python/py_tensor_type.cpp`.
**1 group = 1 commit** (3 commits below).

**Actions**:

1. **Commit 1 — kwargs inliners.** The following six methods inline the exact body of
   `Tensor_forward_keyword_method` instead of calling it (verified in the snapshot; the
   callees are in parentheses): `Tensor_mean_method` (`py_mean`, old L11332),
   `Tensor_all_method` (`py_all`), `Tensor_max_method` (`py_max`), `Tensor_min_method`
   (`py_min`), `Tensor_argmax_method` (`py_argmax`), `Tensor_argmin_method` (`py_argmin`).
   For each, first confirm the body is byte-identical to `Tensor_forward_keyword_method`'s
   body with `function` replaced by the concrete `py_*` name (if not, leave it and add a
   PROGRESS.md note). Then replace. Worked example — BEFORE (verbatim snapshot):

   ```cpp
   PyObject* Tensor_mean_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
     PyObject* tuple = PyTuple_New(PyTuple_GET_SIZE(args) + 1);
     if (tuple == nullptr) {
       return nullptr;
     }
     Py_INCREF(self);
     PyTuple_SET_ITEM(tuple, 0, reinterpret_cast<PyObject*>(self));
     for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
       PyObject* item = PyTuple_GET_ITEM(args, i);
       Py_INCREF(item);
       PyTuple_SET_ITEM(tuple, i + 1, item);
     }
     PyObject* result = py_mean(nullptr, tuple, kwargs);
     Py_DECREF(tuple);
     return result;
   }
   ```

   AFTER:

   ```cpp
   PyObject* Tensor_mean_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
     return Tensor_forward_keyword_method(self, args, kwargs, py_mean);
   }
   ```

   Method-table flags stay untouched (both sides are `METH_VARARGS | METH_KEYWORDS` for all
   six — verified). If a `py_*` callee now lives in another section file and is not visible
   from `py_tensor_type.cpp`, it was already promoted/declared during 2a (these calls
   existed before the split); if the build says "undeclared", declare it in that section's
   header exactly as 2a step 4 prescribes.

2. **Commit 2 — varargs forward helper + equal/lerp.** Directly below
   `Tensor_forward_keyword_method`, add:

   ```cpp
   using TensorVarargsForward = PyObject* (*)(PyObject*, PyObject*);

   PyObject* Tensor_forward_varargs_method(PyTensor* self, PyObject* args, TensorVarargsForward function) {
     PyObject* tuple = PyTuple_New(PyTuple_GET_SIZE(args) + 1);
     if (tuple == nullptr) {
       return nullptr;
     }
     Py_INCREF(self);
     PyTuple_SET_ITEM(tuple, 0, reinterpret_cast<PyObject*>(self));
     for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
       PyObject* item = PyTuple_GET_ITEM(args, i);
       Py_INCREF(item);
       PyTuple_SET_ITEM(tuple, i + 1, item);
     }
     PyObject* result = function(nullptr, tuple);
     Py_DECREF(tuple);
     return result;
   }
   ```

   Convert the two varargs inliners `Tensor_equal_method` (→ `py_equal`) and
   `Tensor_lerp_method` (→ `py_lerp`) to one-line
   `return Tensor_forward_varargs_method(self, args, py_equal);` etc., after the same
   byte-identity check as in step 1. **Do not convert `Tensor_is_nonzero_method`**: its
   table entry is `METH_NOARGS` while `py_is_nonzero` is `METH_VARARGS` (verified) — the
   flags differ, so per the rule below it stays; add a PROGRESS.md note.

3. **Commit 3 — matmul family.** These methods re-implement parsing + checks instead of
   forwarding (old module.cpp L10908–11209 band). For each pair below, extract both
   function bodies and diff them before touching anything:

   ```bash
   awk '/^PyObject\* Tensor_mm_method\(/,/^}$/' cpp/mtorch/python/py_tensor_type.cpp > /tmp/m_method.txt
   awk '/^PyObject\* py_mm\(/,/^}$/' cpp/mtorch/python/py_linalg.cpp > /tmp/m_py.txt
   diff /tmp/m_method.txt /tmp/m_py.txt
   ```

   Replacement rule (apply mechanically): the method may be replaced by a forward **iff**
   the diff shows only (a) the function signature line, (b) `self->value` / the first
   parsed argument appearing as the left operand, (c) parse format arity (`"O:mm"` vs
   `"OO:mm"`), and (d) TypeError message wording (e.g. `"mm expected Tensor"` vs
   `"mm expected tensors"`). Message wording differences are acceptable because the compat
   harness compares only the raised exception *type* name, never the message
   (`01-rules-and-verification.md` §5.0). Any other difference (extra checks, different
   core call, different defaults, different METH flags between the two table entries) →
   do NOT replace; add a PROGRESS.md note naming the method.

   Apply, in this order, with these expected outcomes (verified against the snapshot):

   - varargs, forward with `Tensor_forward_varargs_method`: `Tensor_matmul_method` →
     `py_matmul`, `Tensor_mm_method` → `py_mm`, `Tensor_bmm_method` → `py_bmm`.
     Worked example — BEFORE (verbatim snapshot):

     ```cpp
     PyObject* Tensor_matmul_method(PyTensor* self, PyObject* args) {
       PyObject* other = nullptr;
       if (!PyArg_ParseTuple(args, "O:matmul", &other)) {
         return nullptr;
       }
       if (!is_tensor(other)) {
         PyErr_SetString(PyExc_TypeError, "matmul expected Tensor");
         return nullptr;
       }
       try {
         const auto right = tensor_ref(other);
         ensure_same_dtype_matrix_contraction("matmul", *self->value, right);
         return wrap_tensor(mtorch::matmul(*self->value, right));
       } catch (...) {
         translate_exception();
         return nullptr;
       }
     }
     ```

     AFTER:

     ```cpp
     PyObject* Tensor_matmul_method(PyTensor* self, PyObject* args) {
       return Tensor_forward_varargs_method(self, args, py_matmul);
     }
     ```

   - kwargs, forward with `Tensor_forward_keyword_method` after the same diff check:
     `Tensor_addmm_method` → `py_addmm`, `Tensor_addmv_method` → `py_addmv`,
     `Tensor_addr_method` → `py_addr`, `Tensor_baddbmm_method` → `py_baddbmm`,
     `Tensor_addbmm_method` → `py_addbmm`.
   - varargs, same recipe: `Tensor_vdot_method` → `py_vdot`, `Tensor_inner_method` →
     `py_inner`, `Tensor_kron_method` → `py_kron`, `Tensor_matrix_power_method` →
     `py_matrix_power`, `Tensor_dot_method` → `py_dot`, `Tensor_mv_method` → `py_mv`,
     `Tensor_outer_method` → `py_outer`.

   Note: if a `py_*` target was already wrapped by 3-1-a (`guarded<py_matmul>` in the
   module table), you still forward to the **plain** `py_matmul` function — the exception
   translation then happens in the *method* table's own `guarded<...>` wrapper if you also
   migrated `py_tensor_type.cpp` in 3-1-a, or in `py_matmul`'s remaining try/catch if you
   did not. Either way exactly one translation layer runs; never wrap twice inside the
   forward helper.

**Verification** (after each of the three commits): common verification, plus:

```bash
pytest tests/compat/test_tensor_ops.py --compat-op 'method.*' --compat-op 'reduction.*' --compat-op 'linalg.*'
```

**On failure**: common failure protocol; revert only the group commit that failed.

**Commit** (one per group):

```bash
git add cpp/mtorch/python/py_tensor_type.cpp
git commit -m "refactor(phase3-1c): forward reduction-family Tensor methods through helper"
# group 2: "refactor(phase3-1c): add varargs forward helper; forward equal/lerp"
# group 3: "refactor(phase3-1c): forward matmul-family Tensor methods to py_* bindings"
```

---

## 3-2. table-ify unary ops (target: `detail/elementwise.*`, `elementwise_ops.cpp`)

**Goal**: The per-op if-else chains for simple unary ops are replaced by one table
(`UnaryOpDef`), so adding a unary op means one table row instead of four chain edits.

The op-name chains exist in 4 places (post-split locations first, snapshot hints in
parentheses):

1. float32 kernel — inside `run_unary_float32_kernel`, `cpp/mtorch/core/detail/elementwise.cpp`
   (old tensor.cpp L4273–4458)
2. float16 kernel — `run_unary_float16_kernel`, same file (old L4460–4535). **Verified
   snapshot fact: this kernel handles only `op == "sigmoid"`** (a NEON/Accelerate special
   case); it contains no if-else op chain. See commit D below.
3. double fallback — the lambda inside `unary()`, `cpp/mtorch/core/elementwise_ops.cpp`
   (old L14390–14488)
4. backward differentiation — the `backward_fn` lambda inside `unary()`, same file
   (old L14492–14558)

**Preconditions**: Phase 2b complete. Locate everything:

```bash
grep -n 'run_unary_float32_kernel\|run_unary_float16_kernel' cpp/mtorch/core/detail/elementwise.cpp cpp/mtorch/core/detail/elementwise.h
grep -n 'TensorPtr unary(' cpp/mtorch/core/elementwise_ops.cpp
grep -n 'op == "' cpp/mtorch/core/detail/elementwise.cpp cpp/mtorch/core/elementwise_ops.cpp
```

**Actions**:

1. **Build the inventory (no commit; scratch only).** From the `grep -n 'op == "'` output,
   list the op names per place. The expected result (verified against the snapshot) is:

   - float32 kernel and double fallback both handle exactly these 32 ops:
     `neg abs exp expm1 log log1p log2 log10 sqrt rsqrt reciprocal sign floor ceil trunc
     round frac deg2rad rad2deg sin cos tan sinh cosh tanh asin acos atan sigmoid square
     erf erfc`
   - the backward chain handles the same set **minus** `sign floor ceil trunc round`
     (for those five, `local` stays `0.0` — i.e. their gradient is the constant 0);
   - the float16 kernel handles only `sigmoid`.

   **Migration targets are these 24 ops** — the ones whose float32 branch body is exactly
   one `run_coalesced_unary_float32_kernel(plan, result, <lambda>); return true;` with no
   `#if` block inside the branch:

   `neg abs exp expm1 log log2 log10 sqrt rsqrt reciprocal sign floor ceil trunc round
   frac deg2rad rad2deg tan sinh cosh asin acos atan`

   **Out of scope (stay in the existing chains in every place)**: `log1p sin cos tanh
   sigmoid square erf erfc` — each of these has an in-branch
   `#if defined(MTORCH_USE_ACCELERATE)` dispatch in the float32 kernel
   (`try_accelerate_vforce_float32_kernel` / `try_accelerate_sigmoid_float32_kernel` /
   `try_accelerate_square_float32_kernel` / `try_accelerate_erf_float32_kernel`), and `erf`
   additionally uses `erf_approx_float` in float32 but `std::erf` in double. Ops with
   extra arguments (`clamp` etc.) are not part of `unary()` at all and are untouched.
   The NEON kernels (old L3901–4260) are also **out of scope** — the table replaces only
   the scalar path.

   If your grep output does not match the lists above, STOP: do not proceed with a
   different list; mark 3-2 **BLOCKED** in PROGRESS.md with a note showing the mismatch.

2. **Commit A — define the table and switch the float32 kernel.**
   In `cpp/mtorch/core/detail/elementwise.h` (namespace `mtorch::detail`), add:

   ```cpp
   // Simple elementwise unary ops that need no Accelerate/NEON special-casing.
   // grad: d(op)/dx evaluated at the input value x (the existing backward chain only ever
   // reads the input value, so one argument is sufficient). For sign/floor/ceil/trunc/round
   // the derivative is the constant 0, matching the old chain where `local` stayed 0.0.
   struct UnaryOpDef {
     const char* name;
     float (*f32)(float);
     double (*f64)(double);
     double (*grad)(double x);
   };
   const UnaryOpDef* find_unary_op(const std::string& name);  // nullptr if not in the table
   ```

   In `cpp/mtorch/core/detail/elementwise.cpp`, add (the formulas below are transcribed
   1:1 from the three verified chains — float32 lambda / double lambda / backward branch;
   do not "simplify" any of them):

   ```cpp
   namespace {
   double unary_grad_zero(double) { return 0.0; }
   }  // namespace

   const UnaryOpDef kUnaryOpTable[] = {
       {"neg", [](float v) { return -v; }, [](double v) { return -v; }, [](double) { return -1.0; }},
       {"abs", [](float v) { return std::abs(v); }, [](double v) { return std::abs(v); },
        [](double x) { return x < 0.0 ? -1.0 : 1.0; }},
       {"exp", [](float v) { return std::exp(v); }, [](double v) { return std::exp(v); },
        [](double x) { return std::exp(x); }},
       {"expm1", [](float v) { return std::expm1(v); }, [](double v) { return std::expm1(v); },
        [](double x) { return std::exp(x); }},
       {"log", [](float v) { return std::log(v); }, [](double v) { return std::log(v); },
        [](double x) { return 1.0 / x; }},
       {"log2", [](float v) { return std::log2(v); }, [](double v) { return std::log2(v); },
        [](double x) { return 1.0 / (x * std::log(2.0)); }},
       {"log10", [](float v) { return std::log10(v); }, [](double v) { return std::log10(v); },
        [](double x) { return 1.0 / (x * std::log(10.0)); }},
       {"sqrt", [](float v) { return std::sqrt(v); }, [](double v) { return std::sqrt(v); },
        [](double x) { return 0.5 / std::sqrt(x); }},
       {"rsqrt", [](float v) { return 1.0f / std::sqrt(v); }, [](double v) { return 1.0 / std::sqrt(v); },
        [](double x) { return -0.5 / (x * std::sqrt(x)); }},
       {"reciprocal", [](float v) { return 1.0f / v; }, [](double v) { return 1.0 / v; },
        [](double x) { return -1.0 / (x * x); }},
       {"sign",
        [](float v) { return v > 0.0f ? 1.0f : v < 0.0f ? -1.0f : 0.0f; },
        [](double v) { return v > 0.0 ? 1.0 : v < 0.0 ? -1.0 : 0.0; }, unary_grad_zero},
       {"floor", [](float v) { return std::floor(v); }, [](double v) { return std::floor(v); }, unary_grad_zero},
       {"ceil", [](float v) { return std::ceil(v); }, [](double v) { return std::ceil(v); }, unary_grad_zero},
       {"trunc", [](float v) { return std::trunc(v); }, [](double v) { return std::trunc(v); }, unary_grad_zero},
       {"round", [](float v) { return std::nearbyint(v); }, [](double v) { return std::nearbyint(v); }, unary_grad_zero},
       {"frac", [](float v) { return v - std::trunc(v); }, [](double v) { return v - std::trunc(v); },
        [](double) { return 1.0; }},
       {"deg2rad", [](float v) { return v * 0.017453292519943295769f; },
        [](double v) { return v * 0.017453292519943295769; },
        [](double) { return 0.017453292519943295769; }},
       {"rad2deg", [](float v) { return v * 57.295779513082320876f; },
        [](double v) { return v * 57.295779513082320876; },
        [](double) { return 57.295779513082320876; }},
       {"tan", [](float v) { return std::tan(v); }, [](double v) { return std::tan(v); },
        [](double x) { const double c = std::cos(x); return 1.0 / (c * c); }},
       {"sinh", [](float v) { return std::sinh(v); }, [](double v) { return std::sinh(v); },
        [](double x) { return std::cosh(x); }},
       {"cosh", [](float v) { return std::cosh(v); }, [](double v) { return std::cosh(v); },
        [](double x) { return std::sinh(x); }},
       {"asin", [](float v) { return std::asin(v); }, [](double v) { return std::asin(v); },
        [](double x) { return 1.0 / std::sqrt(1.0 - x * x); }},
       {"acos", [](float v) { return std::acos(v); }, [](double v) { return std::acos(v); },
        [](double x) { return -1.0 / std::sqrt(1.0 - x * x); }},
       {"atan", [](float v) { return std::atan(v); }, [](double v) { return std::atan(v); },
        [](double x) { return 1.0 / (1.0 + x * x); }},
   };

   const UnaryOpDef* find_unary_op(const std::string& name) {
     for (const UnaryOpDef& def : kUnaryOpTable) {  // linear search is fine (~24 entries)
       if (name == def.name) {
         return &def;
       }
     }
     return nullptr;
   }
   ```

   Then, in `run_unary_float32_kernel`, insert the table dispatch **immediately after** the
   leading Accelerate attempt (the `#if defined(MTORCH_USE_ACCELERATE) ...
   try_accelerate_vforce_float32_kernel ... #endif` block at the top of the function) and
   **before** the first `if (op == "neg")`:

   ```cpp
   if (const UnaryOpDef* def = find_unary_op(op)) {
     run_coalesced_unary_float32_kernel(plan, result, def->f32);
     return true;
   }
   ```

   Do not delete any chain branches yet (that is Commit E); the table shadows the 24
   branches, and non-table ops fall through to the existing chain.

3. **Commit B — double fallback.** In `unary()` in `cpp/mtorch/core/elementwise_ops.cpp`,
   hoist one lookup above the `run_coalesced_unary_double_kernel` call and use it first
   inside the lambda:

   BEFORE (start of the fallback, verbatim):

   ```cpp
   run_coalesced_unary_double_kernel(plan, *result, [&op](double value) {
     if (op == "neg") {
       return -value;
     }
   ```

   AFTER:

   ```cpp
   const detail::UnaryOpDef* table_def = detail::find_unary_op(op);
   run_coalesced_unary_double_kernel(plan, *result, [&op, table_def](double value) {
     if (table_def != nullptr) {
       return table_def->f64(value);
     }
     if (op == "neg") {
       return -value;
     }
   ```

   (the trailing `throw std::invalid_argument("unknown unary op: " + op);` and everything
   else stays; the lookup is hoisted so it runs once, not per element).

4. **Commit C — backward.** In the same `unary()`, inside `result->backward_fn`, hoist a
   lookup after the `grad` allocation and consult it before the chain:

   BEFORE (verbatim):

   ```cpp
   result->backward_fn = [input, op](const Tensor& upstream) {
     auto grad = zeros(input->sizes, ScalarType::Float32, input->device);
     for (int64_t i = 0; i < input->numel(); ++i) {
       const double value = input->value_at_linear(i);
       double local = 0.0;
       if (op == "neg") {
         local = -1.0;
       } else if (op == "abs") {
   ```

   AFTER:

   ```cpp
   result->backward_fn = [input, op](const Tensor& upstream) {
     auto grad = zeros(input->sizes, ScalarType::Float32, input->device);
     const detail::UnaryOpDef* table_def = detail::find_unary_op(op);
     for (int64_t i = 0; i < input->numel(); ++i) {
       const double value = input->value_at_linear(i);
       double local = 0.0;
       if (table_def != nullptr) {
         local = table_def->grad(value);
       } else if (op == "neg") {
         local = -1.0;
       } else if (op == "abs") {
   ```

   (turn the old first `if` into an `else if`; everything downstream unchanged.)

5. **Commit D — float16.** Verified: `run_unary_float16_kernel` handles only `sigmoid`,
   which is an out-of-scope op; there is no chain to table-ify. Confirm on the current
   tree (`grep -n 'op != "sigmoid"' cpp/mtorch/core/detail/elementwise.cpp` hits inside
   that function), change no code, and record in the Phase 3 `Notes:` of PROGRESS.md:
   `3-2 float16: run_unary_float16_kernel is a sigmoid-only NEON kernel; nothing to table-ify.`
   Commit only the PROGRESS.md note
   (`git add docs/design/PROGRESS.md && git commit -m "refactor(phase3-2): note float16 unary kernel has no op chain"`).

6. **Commit E — cleanup.** Now remove the shadowed branches for exactly the 24 table ops:

   - in `run_unary_float32_kernel`: delete the 24 `if (op == "...") { ... return true; }`
     blocks for the table ops (the 8 out-of-scope branches stay);
   - in the double-fallback lambda: delete the 24 corresponding `if (op == "...")` blocks
     (the 8 out-of-scope ones and the final `throw` stay);
   - in the backward lambda: delete the corresponding `else if (op == "...")` arms
     (19 arms — remember `sign/floor/ceil/trunc/round` never had one).

**Verification** (after **every** commit A/B/C/E; benchmark confirmation is mandatory
because the table routes the scalar path through a function pointer):

```bash
python3 setup.py build_ext --inplace
pytest tests/compat/test_tensor_ops.py --compat-op 'unary.*'
pytest tests/compat/test_autograd.py --compat-op 'grad.*'
pytest tests/compat 2>&1 | tee /tmp/tests-current.txt   # then §5.1 line comparison
mkdir -p benchmark-results
pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
    --compat-benchmark 'bench.expm1_64x64' --compat-benchmark 'bench.log2_64x64' \
    --compat-benchmark 'bench.log10_64x64' --compat-benchmark 'bench.rsqrt_64x64' \
    --compat-benchmark 'bench.reciprocal_64x64' --compat-benchmark 'bench.floor_64x64' \
    --compat-benchmark 'bench.frac_64x64' --compat-benchmark 'bench.deg2rad_64x64' \
    --compat-benchmark 'bench.rad2deg_64x64' --compat-benchmark 'bench.tan_64x64' \
    --compat-benchmark 'bench.sinh_64x64' --compat-benchmark 'bench.cosh_64x64' \
    --compat-benchmark 'bench.asin_64x64' --compat-benchmark 'bench.sin_64x64' \
    --compat-benchmark 'bench.cos_64x64' --compat-benchmark 'bench.tanh_64x64' \
    --compat-benchmark 'bench.log1p_64x64' --compat-benchmark 'bench.sigmoid_64x64' \
    --compat-benchmark 'bench.square_64x64' --compat-benchmark 'bench.erf_64x64' \
    --compat-benchmark-json benchmark-results/phase3-2-unary.json
python3 tools/compare_benchmarks.py docs/design/baseline/benchmark-baseline.json \
    benchmark-results/phase3-2-unary.json | grep -E '^REGRESSION|^compared' || true
```

PASS: zero `REGRESSION` lines (apply the 3-rerun noise rule of §5.2 to any that appear).

**On failure**: common failure protocol. For a numeric divergence or reproduced benchmark
regression, `git revert --no-edit HEAD` the single commit (A, B, C, or E) that introduced
it, note it in PROGRESS.md, and stop 3-2 there (earlier commits of 3-2 may stand — they
each passed verification on their own).

**Commit** messages:

```bash
git add cpp/mtorch/core/detail/elementwise.h cpp/mtorch/core/detail/elementwise.cpp
git commit -m "refactor(phase3-2): add UnaryOpDef table; route float32 unary kernel through it"
# B: git add cpp/mtorch/core/elementwise_ops.cpp && git commit -m "refactor(phase3-2): route unary double fallback through UnaryOpDef"
# C: git add cpp/mtorch/core/elementwise_ops.cpp && git commit -m "refactor(phase3-2): route unary backward through UnaryOpDef"
# D: PROGRESS.md note commit (see above)
# E: git add cpp/mtorch/core/detail/elementwise.cpp cpp/mtorch/core/elementwise_ops.cpp && git commit -m "refactor(phase3-2): remove table-ized branches from unary op chains"
```

---

## 3-3. consolidating binary fast paths (target: `detail/elementwise.cpp`)

**Goal**: `try_binary_sub_float32_fast_path` and `try_binary_mul_float32_fast_path`
(verified: identical except operator and callee kernel) become 3-line forwards to one
common implementation. **1 pair = 1 commit.**

**Preconditions**: Phase 2b complete. Locate:

```bash
grep -n 'try_binary_sub_float32_fast_path\|try_binary_mul_float32_fast_path\|sub_float32_contiguous\|mul_float32_contiguous' cpp/mtorch/core/detail/elementwise.cpp
```

(hints: old tensor.cpp L5654 / L5714; the contiguous kernels have signature
`void sub_float32_contiguous(const float* left, const float* right, float* target, int64_t elements)`,
old L5483/L5501).

**Actions**:

1. **Mechanically confirm identity (do not skip):**

   ```bash
   awk '/^TensorPtr try_binary_sub_float32_fast_path\(/,/^}$/' cpp/mtorch/core/detail/elementwise.cpp > /tmp/fp_sub.txt
   awk '/^TensorPtr try_binary_mul_float32_fast_path\(/,/^}$/' cpp/mtorch/core/detail/elementwise.cpp > /tmp/fp_mul.txt
   diff /tmp/fp_sub.txt /tmp/fp_mul.txt
   ```

   Expected: every diff line contains only one of these three substitutions —
   `try_binary_sub_float32_fast_path` ↔ `try_binary_mul_float32_fast_path` (1 line),
   `sub_float32_contiguous(` ↔ `mul_float32_contiguous(` (2 call sites), or the scalar
   operator `-` ↔ `*` (2 arithmetic lines: the dim-1 loop and the dim-2 strided loop).
   Verified against the snapshot: the guard conditions, remainder handling, and loop
   bounds are byte-identical. **If there is even one diff line outside those three
   substitutions, do not consolidate**; note it in PROGRESS.md and go to step 4.

2. Add the common implementation to `cpp/mtorch/core/detail/elementwise.cpp`, directly
   above `try_binary_sub_float32_fast_path` (it is the sub body with the operator and the
   contiguous kernel parameterized; every guard is copied verbatim):

   ```cpp
   template <typename ScalarOp>
   TensorPtr try_binary_same_shape_float32_fast_path(
       const TensorPtr& left,
       const TensorPtr& right,
       const std::vector<int64_t>& out_shape,
       bool requires_grad,
       ScalarOp scalar_op,
       void (*contiguous_kernel)(const float*, const float*, float*, int64_t)) {
     if (requires_grad || left->dtype != ScalarType::Float32 || right->dtype != ScalarType::Float32 ||
         left->sizes != right->sizes || left->sizes != out_shape) {
       return nullptr;
     }

     auto result = make_empty_contiguous_tensor(out_shape, ScalarType::Float32, false, left->device);
     float* target = mutable_float_data_at(*result->storage, result->offset);

     if (left->is_contiguous() && right->is_contiguous()) {
       contiguous_kernel(
           float_data_at(*left->storage, left->offset),
           float_data_at(*right->storage, right->offset),
           target,
           product(out_shape));
       return result;
     }
     if (left->dim() == 1) {
       const float* left_data = float_data_at(*left->storage, 0);
       const float* right_data = float_data_at(*right->storage, 0);
       const int64_t size = left->sizes[0];
       for (int64_t i = 0; i < size; ++i) {
         target[i] = scalar_op(
             left_data[left->offset + i * left->strides[0]],
             right_data[right->offset + i * right->strides[0]]);
       }
       return result;
     }
     if (left->dim() == 2) {
       const float* left_data = float_data_at(*left->storage, 0);
       const float* right_data = float_data_at(*right->storage, 0);
       const int64_t rows = left->sizes[0];
       const int64_t cols = left->sizes[1];
       if (left->strides[1] == 1 && right->strides[1] == 1) {
         for (int64_t row = 0; row < rows; ++row) {
           contiguous_kernel(
               left_data + left->offset + row * left->strides[0],
               right_data + right->offset + row * right->strides[0],
               target + row * cols,
               cols);
         }
         return result;
       }
       for (int64_t row = 0; row < rows; ++row) {
         const int64_t left_base = left->offset + row * left->strides[0];
         const int64_t right_base = right->offset + row * right->strides[0];
         for (int64_t col = 0; col < cols; ++col) {
           target[row * cols + col] = scalar_op(
               left_data[left_base + col * left->strides[1]],
               right_data[right_base + col * right->strides[1]]);
         }
       }
       return result;
     }

     return nullptr;
   }
   ```

3. Replace the two function bodies with forwards (signatures unchanged; both functions are
   called from `binary_tensor_tensor` in `elementwise_ops.cpp` — old L18978–19064 — which
   must not be touched):

   ```cpp
   TensorPtr try_binary_sub_float32_fast_path(
       const TensorPtr& left,
       const TensorPtr& right,
       const std::vector<int64_t>& out_shape,
       bool requires_grad) {
     return try_binary_same_shape_float32_fast_path(
         left, right, out_shape, requires_grad,
         [](float a, float b) { return a - b; }, sub_float32_contiguous);
   }

   TensorPtr try_binary_mul_float32_fast_path(
       const TensorPtr& left,
       const TensorPtr& right,
       const std::vector<int64_t>& out_shape,
       bool requires_grad) {
     return try_binary_same_shape_float32_fast_path(
         left, right, out_shape, requires_grad,
         [](float a, float b) { return a * b; }, mul_float32_contiguous);
   }
   ```

   Build + verify + commit (this is the "sub/mul pair" commit; the PROGRESS.md 3-3 line is
   checked off after this pair; additional pairs from step 4 get their own commits under
   the same step ID).

4. **Subsequent pair candidates** (each one: re-run the step-1 awk/diff first with the two
   new function names; consolidate only under the same "three substitutions only" rule;
   1 pair = 1 commit). Real names in `detail/elementwise.cpp` (snapshot lines in
   parentheses):

   - fp16 channels-last group: `try_binary_channels_last_compatible_same_shape_float16_fast_path`
     (L5897/L6636), `try_binary_channels_last_singleton_broadcast_float16_fast_path` (L6714),
     `try_binary_channels_last_channel_broadcast_float16_fast_path` (L6099),
     `try_binary_channels_last_batch_broadcast_float16_fast_path` (L7099)
   - rank3 group: `try_binary_rank3_same_shape_float16_fast_path` (L5904/L6350),
     `try_binary_rank3_token_broadcast_float16_fast_path` (L6449),
     `try_binary_rank3_batch_token_broadcast_float16_fast_path` (L6511)

   If no pair inside a group passes the diff rule, leave the group alone and add one
   PROGRESS.md note line per group examined.

   `try_binary_add_float32_fast_path` (old L5519) is an extended version with row/column
   broadcast cases and Accelerate branching — **out of scope for consolidation, leave
   it as-is.**

**Verification** (per pair commit):

```bash
python3 setup.py build_ext --inplace
pytest tests/compat/test_tensor_ops.py --compat-op 'binary.*' --compat-op 'inplace.*'
pytest tests/compat 2>&1 | tee /tmp/tests-current.txt   # then §5.1 line comparison
mkdir -p benchmark-results
pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
    --compat-benchmark 'bench.binary_add_64x64' \
    --compat-benchmark 'bench.sub_inplace_float32_64x64' \
    --compat-benchmark 'bench.mul_inplace_float32_64x64' \
    --compat-benchmark 'bench.stable_diffusion_guidance_rescale_std_mean_2x4x8x8' \
    --compat-benchmark 'bench.stable_diffusion_ddim_scheduler_step_2x4x8x8' \
    --compat-benchmark 'bench.stable_diffusion_euler_ancestral_scheduler_step_2x4x8x8' \
    --compat-benchmark-json benchmark-results/phase3-3-binary.json
python3 tools/compare_benchmarks.py docs/design/baseline/benchmark-baseline.json \
    benchmark-results/phase3-3-binary.json | grep -E '^REGRESSION|^compared' || true
```

(for fp16 pairs, additionally include
`--compat-benchmark 'bench.stable_diffusion_half_channels_last_classifier_free_guidance_2x4x64x64'`
and `--compat-benchmark 'bench.stable_diffusion_half_channels_last_mask_blend_2x4x64x64'`).

**On failure**: for any test mismatch, numeric divergence, or reproduced (2-of-3)
regression: **revert exactly the one fast-path commit** (`git revert --no-edit HEAD`),
re-verify §5.1, note it in PROGRESS.md, and continue with the next pair (or stop if it was
the first pair).

**Commit** (first pair):

```bash
git add cpp/mtorch/core/detail/elementwise.cpp
git commit -m "refactor(phase3-3): merge sub/mul float32 fast paths into a common template"
```

---

## 3-4. moving numeric kernels into core (target: bindings → core)

**Goal**: No numeric kernels remain in `cpp/mtorch/python/`; the binding layer only parses
arguments and calls `mtorch::*` functions. **In this subtask, *adding* declarations to
`cpp/mtorch/core/tensor.h` is permitted (appending new declarations before the closing
`}  // namespace mtorch`); modifying existing declarations is not.**

**Preconditions**: Phase 2a and 2b complete. Order (1 target = 1 commit):
random → norm → einsum → channels-last. The destination files `core/random.cpp` is new;
`setup.py` already globs `cpp/mtorch/core/*.cpp` (2a-1), so no build-config change is
needed. Common procedure for every target: (1) move the kernel functions into the core
file and make the entry points public in `namespace mtorch`; helpers used only internally
go into the destination's anonymous namespace; (2) confirm no Python dependency remains in
the moved code (`grep -n 'PyObject\|Py_' cpp/mtorch/core/<file>.cpp` must print nothing) —
if a Python-dependent fragment exists, keep that fragment in the binding and move only the
pure part; (3) add the declarations to `tensor.h`; (4) make the binding a thin wrapper
that calls `mtorch::xxx(...)`. On "undeclared identifier" during the build, apply the 2b
decision algorithm: the missing symbol is either declared in a `detail/*.h` you must
include, or is another moved helper that needs a declaration added to `tensor.h`.

### Commit 1 — random → `core/random.cpp`

Move targets, all currently in `cpp/mtorch/python/py_random.cpp` / `py_common.{h,cpp}`
(snapshot hints: old module.cpp L47–367 and L368–745). Locate each with
`grep -n '<name>' cpp/mtorch/python/py_random.cpp cpp/mtorch/python/py_common.cpp cpp/mtorch/python/py_common.h`:

- `struct RandomState` (old L47) and `global_rng()` (old L60)
- `seed_random_state` (L70), `next_random_u64` (L76), `next_uniform_open` (L84),
  `next_standard_normal_pair` (L90), `next_standard_normal` (L104),
  `next_random_int64` (L185)
- `fill_random_normal_contiguous` / `fill_random_uniform_contiguous` (templates, L118/L147)
- `fill_randn_result` (L159), `fill_randint_result` (L193), `normal_inplace` (L245),
  `next_truncated_normal` (L276), `trunc_normal_inplace` (L284), `uniform_inplace` (L335),
  `fill_bernoulli_contiguous` (template, L368)
- `bernoulli_tensor` (L378), `dropout_tensor` (L436), `multinomial_tensor` (L688)

Stays in the binding: `global_initial_seed()` (Python-facing seed bookkeeping),
`PyGenerator` (old L53 — after this commit its `RandomState rng;` member is the
`mtorch::RandomState` from `tensor.h`, which is the doc-mandated "PyGenerator holds a
`mtorch::RandomState`"), `random_state_from_generator` (old L1375; takes `PyObject*`),
`py_manual_seed` / `Generator_*` / all `py_*` functions.

Create `cpp/mtorch/core/random.cpp` in the standard 2b form
(`#include "mtorch/core/tensor.h"` etc.; `namespace mtorch { ... }`). Paste the moved
definitions unchanged, except: delete the `RandomState` struct from the paste (it moves to
`tensor.h`, next paragraph). Keep the two `fill_random_*_contiguous` templates,
`fill_bernoulli_contiguous`, `next_standard_normal_pair`, and `next_truncated_normal` in
the **anonymous namespace** of `random.cpp` (they are only called by the moved entry
points). The moved code calls `is_floating_scalar_type` (defined in the binding's
anonymous namespace, old module.cpp L37); copy that 3-line helper into `random.cpp`'s
anonymous namespace:

```cpp
bool is_floating_scalar_type(ScalarType dtype) {
  return dtype == ScalarType::Float16 || dtype == ScalarType::Float32 || dtype == ScalarType::Float64;
}
```

Append to `tensor.h` (immediately before `}  // namespace mtorch`; defaults preserved
verbatim from the snapshot so no call site changes):

```cpp
// --- random (kernels moved from the binding layer in phase 3-4) ---
struct RandomState {
  uint64_t state = 0;
  bool has_spare_normal = false;
  double spare_normal = 0.0;
};
RandomState& global_rng();
void seed_random_state(RandomState& rng, uint64_t seed);
uint64_t next_random_u64(RandomState& rng = global_rng());
double next_uniform_open(RandomState& rng = global_rng());
double next_standard_normal(RandomState& rng = global_rng());
int64_t next_random_int64(int64_t low, int64_t high, RandomState& rng = global_rng());
void fill_randn_result(const TensorPtr& result, RandomState& rng = global_rng());
void fill_randint_result(const TensorPtr& result, int64_t low, int64_t high, RandomState& rng = global_rng());
void normal_inplace(Tensor& tensor, double mean, double std, RandomState& rng = global_rng());
void trunc_normal_inplace(Tensor& tensor, double mean, double std, double a, double b, RandomState& rng = global_rng());
void uniform_inplace(Tensor& tensor, double from, double to, RandomState& rng = global_rng());
TensorPtr bernoulli_tensor(const TensorPtr& input, RandomState& rng = global_rng());
TensorPtr dropout_tensor(const TensorPtr& input, double p, bool training);
TensorPtr multinomial_tensor(
    const TensorPtr& input,
    int64_t num_samples,
    bool replacement,
    RandomState& rng = global_rng());
```

Binding side: delete the moved definitions; in `py_common.h` remove the local
`RandomState`/`global_rng` declarations and rely on the `tensor.h` ones (the binding files
already include `tensor.h` transitively; if a name lookup fails, add
`using mtorch::RandomState;` / `using mtorch::global_rng;` inside `namespace mtorch::py`
in `py_common.h`). The `py_*` bodies keep calling `fill_randn_result(...)` etc. — those
calls now resolve to `mtorch::` (binding files open `using namespace mtorch::py;` and the
functions are found via the `mtorch::` qualification already present or via the argument-
dependent `TensorPtr`; if the build complains, qualify the call sites as
`mtorch::fill_randn_result(...)` — that is the thin-wrapper form this step wants anyway).

Extra verification for this commit (seed reproducibility — existing tests also cover it):

```bash
python3 - <<'EOF'
import mtorch
mtorch.manual_seed(1234)
a = mtorch.randn((4, 4)).tolist()
mtorch.manual_seed(1234)
b = mtorch.randn((4, 4)).tolist()
assert a == b, "manual_seed no longer reproduces the same sequence"
print("rng determinism ok")
EOF
pytest tests/compat/test_tensor_ops.py --compat-op 'sampling.*'
```

Commit:

```bash
git add cpp/mtorch/core/tensor.h cpp/mtorch/core/random.cpp cpp/mtorch/python
git commit -m "refactor(phase3-4): move RNG/bernoulli/dropout/multinomial kernels into core/random.cpp"
```

### Commit 2 — norm → `core/reductions.cpp`

Move targets from `cpp/mtorch/python/py_pointwise.cpp`
(`grep -n 'norm_tensor\|reduce_sum_dims_for_norm\|norm_zero_tensor\|normalized_dims_for_tensor' cpp/mtorch/python/*.cpp`;
hints: old module.cpp L3234–3396): `normalized_dims_for_tensor`,
`reduce_sum_dims_for_norm`, `norm_zero_tensor` → anonymous namespace of
`cpp/mtorch/core/reductions.cpp` (verified: they have no other binding callers);
`norm_tensor` → public. It is implemented on top of `mtorch::unary` / `reduce_sum` /
`amax` / `amin` (reduction content → `reductions.cpp`, not `linalg.cpp`). Copy the
`is_floating_scalar_type` helper into the anonymous namespace as in commit 1 (unless
`reductions.cpp` already has one — check first; if a same-named helper with an identical
body exists there, reuse it).

Append to `tensor.h`:

```cpp
// --- norm (moved from the binding layer in phase 3-4) ---
TensorPtr norm_tensor(
    const TensorPtr& input,
    double p,
    const std::optional<std::vector<int64_t>>& dims,
    bool keepdim,
    ScalarType dtype);
```

Binding side: `py_norm` (old L3405) and the Tensor `norm` method (old L10771) now call
`mtorch::norm_tensor(...)`; `optional_dims_from_py` (old L3398) stays in the binding (it
reads `PyObject*`).

Extra verification:

```bash
pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
    --compat-benchmark 'bench.norm_dim1_64x64' --compat-benchmark 'bench.norm_p0_dim1_64x64' \
    --compat-benchmark-json benchmark-results/phase3-4-norm.json
python3 tools/compare_benchmarks.py docs/design/baseline/benchmark-baseline.json \
    benchmark-results/phase3-4-norm.json | grep -E '^REGRESSION|^compared' || true
```

Commit: `git add cpp/mtorch/core/tensor.h cpp/mtorch/core/reductions.cpp cpp/mtorch/python && git commit -m "refactor(phase3-4): move norm_tensor into core/reductions.cpp"`

### Commit 3 — einsum fast path → `core/linalg.cpp`

Move targets from `cpp/mtorch/python/py_linalg.cpp`
(`grep -n 'einsum_attention_scores\|einsum_attention_values\|SimpleBinaryEinsum\|try_simple_binary_einsum_fast_path\|ensure_einsum_contraction_supported' cpp/mtorch/python/py_linalg.cpp`;
hints: old module.cpp L6106–6331):

- anonymous namespace of `core/linalg.cpp`: `struct SimpleBinaryEinsum`,
  `split_simple_binary_einsum`, `einsum_labels_are_unique`, `einsum_labels_match_rank`
- public `namespace mtorch`: `einsum_attention_scores`, `einsum_attention_values`,
  `ensure_einsum_contraction_supported`, `try_simple_binary_einsum_fast_path`
  (all four are called directly by `py_einsum`, old L6333)

The scores/values kernels use Accelerate under `#if defined(MTORCH_USE_ACCELERATE)`
(`accelerate_int_ok` etc.); add `#include "mtorch/core/detail/accelerate.h"` (and
`detail/platform.h`) to `linalg.cpp` if the build asks for them.

Append to `tensor.h`:

```cpp
// --- einsum fast paths (moved from the binding layer in phase 3-4) ---
TensorPtr einsum_attention_scores(const TensorPtr& left, const TensorPtr& right);
TensorPtr einsum_attention_values(const TensorPtr& left, const TensorPtr& right);
void ensure_einsum_contraction_supported(const TensorPtr& left, const TensorPtr& right);
TensorPtr try_simple_binary_einsum_fast_path(
    const std::string& text,
    const TensorPtr& left,
    const TensorPtr& right);
```

Binding side: `normalized_equation` (old L6092; uses `PyUnicode_AsUTF8`) stays in
`py_linalg.cpp`; `py_einsum` keeps its structure and calls the four `mtorch::` functions.

Extra verification:

```bash
pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
    --compat-benchmark 'bench.einsum_matmul_48x48' \
    --compat-benchmark 'bench.einsum_batched_matmul_8x32x32' \
    --compat-benchmark 'bench.einsum_attention_scores_1x2x16x16' \
    --compat-benchmark 'bench.einsum_attention_scores_generic_labels_1x2x16x16' \
    --compat-benchmark-json benchmark-results/phase3-4-einsum.json
python3 tools/compare_benchmarks.py docs/design/baseline/benchmark-baseline.json \
    benchmark-results/phase3-4-einsum.json | grep -E '^REGRESSION|^compared' || true
```

Commit: `git add cpp/mtorch/core/tensor.h cpp/mtorch/core/linalg.cpp cpp/mtorch/python && git commit -m "refactor(phase3-4): move einsum fast-path kernels into core/linalg.cpp"`

### Commit 4 — channels-last conversion → `core/views.cpp`

Move targets from `cpp/mtorch/python/py_common.cpp` (2a-2 moved the memory_format family
there; hints: old module.cpp L918–1117). Locate:
`grep -n 'MemoryFormat\|channels_last_strides\|tensor_to_memory_format\|make_like_with_memory_format' cpp/mtorch/python/py_common.cpp cpp/mtorch/python/py_common.h`.

- `enum class MemoryFormat { Preserve, Contiguous, ChannelsLast, ChannelsLast3d };`
  (old L918) → **moves into `tensor.h`** (this is a new addition to `tensor.h`, permitted;
  in `py_common.h`, replace the enum definition with
  `using MemoryFormat = mtorch::MemoryFormat;` so all binding signatures keep compiling)
- anonymous namespace of `core/views.cpp`: `copy_storage_element` (old L1005),
  `try_copy_contiguous_to_channels_last_4d` (old L1013). If `views.cpp` already contains
  an anonymous-namespace helper with the same name, compare bodies: identical → delete the
  moved copy and use the existing one; different → rename the moved one by appending
  `_memory_format` and update its (moved) callers only.
- public `namespace mtorch`: `channels_last_strides_2d` (old L984),
  `channels_last_strides_3d` (L988), `tensor_is_contiguous_memory_format` (L992),
  `tensor_to_memory_format` (L1050), `strides_for_like_memory_format` (L1089),
  `make_like_with_memory_format` (L1110)

Stays in the binding (`py_common.cpp`): `memory_format_from_py`,
`optional_memory_format_from_py` (old L953/L977; they read `PyObject*`).

Append to `tensor.h` (plus the `MemoryFormat` enum above the `Tensor` struct or directly
before this block — anywhere inside `namespace mtorch` above its first use):

```cpp
// --- memory-format conversion (moved from the binding layer in phase 3-4) ---
std::vector<int64_t> channels_last_strides_2d(const std::vector<int64_t>& sizes);
std::vector<int64_t> channels_last_strides_3d(const std::vector<int64_t>& sizes);
bool tensor_is_contiguous_memory_format(const Tensor& tensor, MemoryFormat memory_format);
TensorPtr tensor_to_memory_format(const TensorPtr& input, MemoryFormat memory_format, bool copy = false);
std::vector<int64_t> strides_for_like_memory_format(const Tensor& source, std::optional<MemoryFormat> memory_format);
TensorPtr make_like_with_memory_format(
    const TensorPtr& source,
    ScalarType dtype,
    Device device,
    bool requires_grad,
    std::optional<MemoryFormat> memory_format);
```

Binding call sites that keep working through the new `mtorch::` functions (verified list):
the four `*_like` bindings in `py_creation.cpp` (`make_like_with_memory_format`),
`Tensor_is_contiguous` in `py_tensor_type.cpp` (old L9175,
`tensor_is_contiguous_memory_format`), and the `contiguous`/`to` method paths in
`py_tensor_type.cpp` (old L9219/L9268, `tensor_to_memory_format`).

Extra verification (SD family, channels-last heavy — the "pick 2–3
`bench.stable_diffusion*`" requirement):

```bash
pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
    --compat-benchmark 'bench.stable_diffusion_half_channels_last_conv2d_1x4x16x16' \
    --compat-benchmark 'bench.stable_diffusion_half_channels_last_scheduler_add_noise_2x4x64x64' \
    --compat-benchmark 'bench.stable_diffusion_half_channels_last_3d_video_upsample_1x320x4x8x8' \
    --compat-benchmark-json benchmark-results/phase3-4-cl.json
python3 tools/compare_benchmarks.py docs/design/baseline/benchmark-baseline.json \
    benchmark-results/phase3-4-cl.json | grep -E '^REGRESSION|^compared' || true
```

Commit: `git add cpp/mtorch/core/tensor.h cpp/mtorch/core/views.cpp cpp/mtorch/python && git commit -m "refactor(phase3-4): move channels-last conversion kernels into core/views.cpp"`

**Verification (whole step, after every commit)**: common verification (full suite §5.1),
plus the per-commit extras above.

**On failure**: common failure protocol; revert only the current target's commit — earlier
3-4 commits stand.

---

## 3-5. fixing the Python class hierarchy (target: `mtorch/nn/modules/conv.py`, `mtorch/optim.py`)

### 3-5-a: `_ConvNd` base class

**Goal**: The duplicated `__init__` logic of `Conv1d/2d/3d` and `ConvTranspose1d/2d/3d`
(attribute conversion, groups validation, weight/bias `Parameter` creation) lives once in
a `_ConvNd` base class.

**Preconditions**: Phase 1 complete; the six classes live in `mtorch/nn/modules/conv.py`
(`grep -n '^class ' mtorch/nn/modules/conv.py`; hints: old `__init__.py` L3338, L3433,
L3534, L3642, L3741, L3853). Verified snapshot facts you must preserve:

- **None of the six classes call `super().__init__()`** (Module.`__init__` is skipped;
  `training` resolves to the `Module.training = True` class attribute). `_ConvNd.__init__`
  must therefore not call `super().__init__()` either.
- **There is no `extra_repr` anywhere** (`grep -c 'def extra_repr' mtorch/` prints 0 for
  every file). Nothing to lift; do not add one. Record in PROGRESS.md Notes:
  `3-5-a: no extra_repr exists in the codebase; lifted __init__/params only.`
- The plain convs validate `padding_mode` against
  `{"zeros", "reflect", "replicate", "circular"}` with a `ValueError`; `ConvTranspose1d`
  raises `ValueError("ConvTranspose1d only supports padding_mode='zeros'")`;
  `ConvTranspose2d/3d` raise `NotImplementedError("only zeros padding_mode is implemented")`.
  These messages/types differ per class and must stay byte-identical → padding-mode
  validation is a per-class hook, not shared code.
- Parameter names stay `weight` / `bias` and shapes are unchanged (state_dict keys
  invariant); public `__init__` signatures unchanged; forward numerics unchanged (the
  `forward`, `_initial_weight`, and `_output_padding_for_size` methods are NOT touched).

**Actions**:

1. At the top of `mtorch/nn/modules/conv.py`, above `class Conv1d`, add the base class
   (uses the same names the file already imports — `Module`, `Parameter`, `tensor`,
   `float32`, `_builtins`; if the file spells any of these differently after Phase 1,
   use the file's spelling):

   ```python
   class _ConvNd(Module):
       """Shared __init__ logic for Conv1d/2d/3d and ConvTranspose1d/2d/3d.

       Subclasses provide:
         _weight_data()            -- nested list for the initial weight (verbatim from the
                                      old per-class __init__)
         _validate_padding_mode()  -- per-class padding_mode check (messages differ per class)
       """

       def __init__(
           self,
           in_channels,
           out_channels,
           kernel_size,
           stride,
           padding,
           dilation,
           transposed,
           output_padding,
           groups,
           bias,
           padding_mode,
           device,
           dtype,
           size_tuple,
       ):
           # NOTE: deliberately no super().__init__() call -- the original classes never
           # called Module.__init__ and behavior must not change.
           self.in_channels = _builtins.int(in_channels)
           self.out_channels = _builtins.int(out_channels)
           self.kernel_size = size_tuple(kernel_size)
           self.stride = size_tuple(stride)
           if not transposed and isinstance(padding, str):
               _validate_conv_string_padding(padding, self.stride)
               self.padding = padding
           else:
               self.padding = size_tuple(padding)
           if transposed:
               self.output_padding = size_tuple(output_padding)
           self.dilation = size_tuple(dilation)
           self.groups = _builtins.int(groups)
           self.padding_mode = padding_mode
           if self.groups <= 0:
               raise ValueError("groups must be a positive integer")
           if self.in_channels % self.groups != 0:
               raise ValueError("in_channels must be divisible by groups")
           if self.out_channels % self.groups != 0:
               raise ValueError("out_channels must be divisible by groups")
           self._validate_padding_mode(padding_mode)
           self.weight = Parameter(tensor(
               self._weight_data(),
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

       def _validate_padding_mode(self, padding_mode):
           if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
               raise ValueError("padding_mode must be one of 'zeros', 'reflect', 'replicate', or 'circular'")
   ```

   (`_validate_conv_string_padding` is the helper the plain convs already call; it must be
   importable in this file — it already is, because the old `Conv1d.__init__` used it here.)

2. Rewrite `Conv1d` — keep `_initial_weight` and `forward` byte-identical; replace only
   `__init__` and add `_weight_data`:

   ```python
   class Conv1d(_ConvNd):
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
           super().__init__(
               in_channels, out_channels, kernel_size, stride, padding, dilation,
               False, 0, groups, bias, padding_mode, device, dtype, _single,
           )

       def _weight_data(self):
           channels_per_group = self.in_channels // self.groups
           return [
               [
                   [self._initial_weight(out, channel, kernel_x) for kernel_x in range(self.kernel_size[0])]
                   for channel in range(channels_per_group)
               ]
               for out in range(self.out_channels)
           ]

       # _initial_weight and forward: unchanged from the current file.
   ```

3. `Conv2d` and `Conv3d`: identical transformation — `super().__init__(..., False, 0,
   groups, bias, padding_mode, device, dtype, _pair)` (Conv2d) / `..., _triple)` (Conv3d),
   and `_weight_data` is the class's existing nested weight comprehension moved verbatim
   out of the old `__init__` (with the leading
   `channels_per_group = self.in_channels // self.groups` line), returning the list that
   was previously passed to `tensor(...)`.

4. `ConvTranspose1d`:

   ```python
   class ConvTranspose1d(_ConvNd):
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
           super().__init__(
               in_channels, out_channels, kernel_size, stride, padding, dilation,
               True, output_padding, groups, bias, padding_mode, device, dtype, _single,
           )

       def _validate_padding_mode(self, padding_mode):
           if padding_mode != "zeros":
               raise ValueError("ConvTranspose1d only supports padding_mode='zeros'")

       def _weight_data(self):
           out_channels_per_group = self.out_channels // self.groups
           return [
               [
                   [self._initial_weight(channel, out, kernel_x) for kernel_x in range(self.kernel_size[0])]
                   for out in range(out_channels_per_group)
               ]
               for channel in range(self.in_channels)
           ]

       # _initial_weight, _output_padding_for_size, forward: unchanged.
   ```

5. `ConvTranspose2d` / `ConvTranspose3d`: same transformation with `_pair` / `_triple`,
   `True, output_padding`, their own verbatim `_weight_data`, and this override for both:

   ```python
       def _validate_padding_mode(self, padding_mode):
           if padding_mode != "zeros":
               raise NotImplementedError("only zeros padding_mode is implemented")
   ```

6. Do not export `_ConvNd` from `mtorch/nn/__init__.py` (it is private, like torch's).

**Verification**:

```bash
python3 setup.py build_ext --inplace
pytest tests/compat/test_nn_modules.py
pytest tests/compat/test_api_surface.py | tail -3
pytest tests/compat 2>&1 | tee /tmp/tests-current.txt   # then §5.1 line comparison
python3 - <<'EOF'
import mtorch
c = mtorch.nn.Conv2d(4, 8, 3, padding=1)
sd = c.state_dict()
assert sorted(sd.keys()) == ["bias", "weight"], sorted(sd.keys())
assert list(c.weight.shape) == [8, 4, 3, 3], list(c.weight.shape)
t = mtorch.nn.ConvTranspose2d(4, 8, 3)
assert list(t.weight.shape) == [4, 8, 3, 3], list(t.weight.shape)
print("state_dict keys and shapes ok")
EOF
```

**On failure**: common failure protocol (single commit → `git revert --no-edit HEAD` if
already committed).

**Commit**:

```bash
git add mtorch/nn/modules/conv.py
git commit -m "refactor(phase3-5a): introduce _ConvNd base class for the six conv modules"
```

### 3-5-b: making Optimizers true subclasses

**Goal**: `SGD` / `Adam` inherit from `Optimizer` instead of wrapping one
(`self._optimizer = Optimizer(...)`), and the duplicated delegation methods
(`zero_grad` / `state_dict` / `load_state_dict`) are deleted. `AdamW` already subclasses
`Adam` (verified) and needs no structural change.

**Preconditions**: Phase 1 complete; the classes live in `mtorch/optim.py`
(`grep -n '^class ' mtorch/optim.py`; hints: old `__init__.py` L5757/L5794/L5865/L5957).
Constraints: the structure (key names, nesting) of `param_groups` / `state` / `defaults`
must not change, and `state_dict()` output must be identical before/after.

**Actions**:

1. **Capture the BEFORE state_dict snapshots** (run this while the tree is still
   unchanged). The `state` dict is keyed by `id(parameter)`, which differs between
   processes, so the script normalizes keys to parameter indices before serializing:

   ```bash
   python3 - <<'EOF'
   import json
   import mtorch

   def snapshot(opt_name, path, **kwargs):
       model = mtorch.nn.Linear(4, 3)
       opt = getattr(mtorch.optim, opt_name)(model.parameters(), lr=0.01, **kwargs)
       for _ in range(3):
           opt.zero_grad()
           loss = model(mtorch.ones((2, 4))).sum()
           loss.backward()
           opt.step()
       params = [p for g in opt.param_groups for p in g["params"]]
       index = {id(p): i for i, p in enumerate(params)}
       sd = opt.state_dict()
       def enc(obj):
           if hasattr(obj, "tolist"):
               return obj.tolist()
           raise TypeError(str(type(obj)))
       normalized = {
           "state": {str(index[key]): value for key, value in sd["state"].items()},
           "param_groups": [
               {k: (["<param>"] * len(v) if k == "params" else v) for k, v in group.items()}
               for group in sd["param_groups"]
           ],
       }
       with open(path, "w") as f:
           json.dump(normalized, f, sort_keys=True, default=enc)
       print("wrote", path)

   snapshot("SGD", "/tmp/optim_before_sgd.json", momentum=0.9)
   snapshot("Adam", "/tmp/optim_before_adam.json")
   snapshot("AdamW", "/tmp/optim_before_adamw.json")
   EOF
   ```

2. Rewrite `SGD`. BEFORE (verbatim snapshot; the only Phase-1 difference may be
   `mtorch.`-qualification of root functions inside `step`):

   ```python
   class SGD:
       def __init__(self, params, lr=0.001, momentum=0.0, dampening=0.0, weight_decay=0.0,
                    nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
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
           ...  # unchanged
   ```

   AFTER (the `step` method body is not touched at all):

   ```python
   class SGD(Optimizer):
       def __init__(self, params, lr=0.001, momentum=0.0, dampening=0.0, weight_decay=0.0,
                    nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
           if nesterov and (momentum <= 0.0 or dampening != 0.0):
               raise ValueError("Nesterov momentum requires a momentum and zero dampening")
           super().__init__(
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

       def step(self, closure=None):
           ...  # unchanged
   ```

   Deleted: the `self._optimizer = Optimizer(...)` composition, the three
   `self.defaults/param_groups/state = self._optimizer...` copies (the inherited
   `Optimizer.__init__` sets those attributes directly), and the three delegation methods
   (`zero_grad`, `state_dict`, `load_state_dict` are inherited).

3. Rewrite `Adam` identically: `class Adam(Optimizer):`, keep its four validation `raise`
   lines, replace `self._optimizer = Optimizer(params, {...})` + attribute copies with
   `super().__init__(params, {...})` (same defaults dict:
   `lr, betas, eps, weight_decay, amsgrad, maximize`), delete the three delegation
   methods, keep `step` untouched.

4. `AdamW(Adam)`: no change (its `__init__` already calls `super().__init__(...)` with
   keyword arguments that still match; its `step` stays).

5. `Optimizer` itself: no change.

6. **Compare state_dicts.** Re-run the step-1 script with paths
   `/tmp/optim_after_sgd.json` etc., then:

   ```bash
   diff /tmp/optim_before_sgd.json /tmp/optim_after_sgd.json && \
   diff /tmp/optim_before_adam.json /tmp/optim_after_adam.json && \
   diff /tmp/optim_before_adamw.json /tmp/optim_after_adamw.json && echo STATE_DICT_OK
   ```

   `STATE_DICT_OK` must print. Any diff → failure path.

7. Record the intended behavior change in PROGRESS.md Phase 3 Notes:
   `3-5-b: isinstance(SGD/Adam/AdamW instance, Optimizer) changed False -> True (intended torch-compat improvement).`

8. **Optional** (only if everything above is green and you have spare capacity — may also
   be skipped with a note): `Adam.step` and `AdamW.step` are copies except for the
   weight-decay/maximize prologue (verified: Adam does `maximize → grad = -grad` first and
   then `grad = grad + parameter * weight_decay`; AdamW does
   `parameter.copy_(parameter * (1.0 - lr * weight_decay))` first and then
   `maximize → grad = -grad`). Consolidate into
   `Adam._step_impl(self, closure, decoupled_weight_decay)` whose per-parameter loop
   begins:

   ```python
   if decoupled_weight_decay:
       if weight_decay != 0.0:
           parameter.copy_(parameter * (1.0 - lr * weight_decay))
       if maximize:
           grad = -grad
   else:
       if maximize:
           grad = -grad
       if weight_decay != 0.0:
           grad = grad + parameter * weight_decay
   ```

   with the rest of the loop taken verbatim from the current `Adam.step`; then reduce the
   two step methods to:

   ```python
   # in Adam
   def step(self, closure=None):
       return self._step_impl(closure, decoupled_weight_decay=False)

   # in AdamW
   def step(self, closure=None):
       return self._step_impl(closure, decoupled_weight_decay=True)
   ```

   Re-run the state_dict comparison (step 6) afterwards. If you do this, make it a
   separate second commit.

**Verification**:

```bash
python3 setup.py build_ext --inplace
pytest tests/compat/test_optim.py -v
pytest tests/compat 2>&1 | tee /tmp/tests-current.txt   # then §5.1 line comparison
python3 - <<'EOF'
import mtorch
opt = mtorch.optim.AdamW([mtorch.ones((2, 2), requires_grad=True)])
assert isinstance(opt, mtorch.optim.Optimizer)
assert isinstance(opt, mtorch.optim.Adam)
print("isinstance ok")
EOF
```

**On failure**: common failure protocol.

**Commit**:

```bash
git add mtorch/optim.py
git commit -m "refactor(phase3-5b): make SGD/Adam true Optimizer subclasses, drop delegation"
# optional follow-up: git commit -m "refactor(phase3-5b): consolidate Adam/AdamW step into _step_impl"
```

---

## 3-6. resolving small-grained duplication (independent, in order of increasing risk)

**Goal**: Three known micro-duplications are removed (items 1–2) or explicitly deferred
with a note (item 3 is optional). 1 item = 1 commit.

**Preconditions**: item 1 needs Phase 2b (`cpp/mtorch/core/detail/half.h` exists);
item 2 needs Phase 1 (`mtorch/autograd.py` exists); item 3 needs Phase 2b.

**Actions**:

1. **`detail/half.h`: merge the fp16→fp32 array converters.** Locate:
   `grep -n 'half_bits_to_float_array\|half_bits_to_neg_float_array' cpp/mtorch/core/detail/half.h`
   (hints: old tensor.cpp L346/L370; verified: the two bodies are identical except that the
   neg variant wraps each NEON store in `vnegq_f32` and negates in the scalar loops).
   Replace the two definitions with (keep the `inline` the 2b move added; the surrounding
   `mtorch::detail` namespace stays):

   ```cpp
   template <bool Negate>
   inline void half_bits_to_float_array_impl(const uint16_t* source, float* target, int64_t count) {
   #if defined(__ARM_NEON) && defined(__aarch64__)
     int64_t i = 0;
     for (; i + 7 < count; i += 8) {
       const uint16x8_t bits = vld1q_u16(source + i);
       const float16x8_t half_values = vreinterpretq_f16_u16(bits);
       float32x4_t low = vcvt_f32_f16(vget_low_f16(half_values));
       float32x4_t high = vcvt_f32_f16(vget_high_f16(half_values));
       if constexpr (Negate) {
         low = vnegq_f32(low);
         high = vnegq_f32(high);
       }
       vst1q_f32(target + i, low);
       vst1q_f32(target + i + 4, high);
     }
     for (; i + 3 < count; i += 4) {
       const uint16x4_t bits = vld1_u16(source + i);
       const float16x4_t half_values = vreinterpret_f16_u16(bits);
       float32x4_t values = vcvt_f32_f16(half_values);
       if constexpr (Negate) {
         values = vnegq_f32(values);
       }
       vst1q_f32(target + i, values);
     }
     for (; i < count; ++i) {
       target[i] = Negate ? -half_bits_to_float(source[i]) : half_bits_to_float(source[i]);
     }
   #else
     for (int64_t i = 0; i < count; ++i) {
       target[i] = Negate ? -half_bits_to_float(source[i]) : half_bits_to_float(source[i]);
     }
   #endif
   }

   inline void half_bits_to_float_array(const uint16_t* source, float* target, int64_t count) {
     half_bits_to_float_array_impl<false>(source, target, count);
   }

   inline void half_bits_to_neg_float_array(const uint16_t* source, float* target, int64_t count) {
     half_bits_to_float_array_impl<true>(source, target, count);
   }
   ```

   The two existing names remain as inline forwards, so no caller (dozens across the core)
   changes. Verify + commit:
   `git add cpp/mtorch/core/detail/half.h && git commit -m "refactor(phase3-6): merge half_bits_to_float_array variants via template<bool Negate>"`

2. **`mtorch/autograd.py`: remove the unreachable condition.** Locate:
   `grep -n 'or value is None' mtorch/autograd.py` (hint: old `__init__.py` L5703, inside
   `_as_optional_tensor_tuple`; the function's first branch `if value is None: return ...`
   has already returned, so the second `value is None` test is dead). BEFORE / AFTER
   (one line):

   ```python
   # BEFORE
       if isinstance(value, Tensor) or value is None:
   # AFTER
       if isinstance(value, Tensor):
   ```

   Verify (`pytest tests/compat/test_autograd.py` + full §5.1) + commit:
   `git add mtorch/autograd.py && git commit -m "refactor(phase3-6): drop unreachable 'or value is None' in _as_optional_tensor_tuple"`

3. **(optional) dimension-parallel implementations.** Candidates (post-split locations):
   `copy_cat_channels_last_4d` / `copy_cat_channels_last_5d` in `cpp/mtorch/core/views.cpp`
   (`grep -n 'copy_cat_channels_last' cpp/mtorch/core/views.cpp`; old tensor.cpp
   L31860/L31942) and `conv_transpose1d/2d/3d` in `cpp/mtorch/core/conv.cpp`
   (old L26242/L28178/L28723). **Only consolidate a pair if an awk-extract + diff (same
   technique as 3-3 step 1) shows the bodies are same-shaped apart from the dimension
   count.** These functions differ in loop nesting depth, so a clean diff is unlikely; if
   you judge the risk high relative to the payoff, defer: add one PROGRESS.md note line
   (`3-6-3 deferred: copy_cat_channels_last / conv_transpose consolidation not same-shaped under diff`)
   and finish the step without a code commit for this item.

**Verification** (per item): common verification; item 1 additionally:

```bash
pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
    --compat-benchmark 'bench.sigmoid_half_2x77x768' \
    --compat-benchmark 'bench.stable_diffusion_half_channels_last_silu_1x320x16x16' \
    --compat-benchmark-json benchmark-results/phase3-6-half.json
python3 tools/compare_benchmarks.py docs/design/baseline/benchmark-baseline.json \
    benchmark-results/phase3-6-half.json | grep -E '^REGRESSION|^compared' || true
```

**On failure**: common failure protocol, per item.

---

## Phase completion criteria

1. The try/catch boilerplate and forwarding boilerplate are unified through common
   helpers. Check:

   ```bash
   cat cpp/mtorch/python/*.cpp | grep -c 'translate_exception();'
   ```

   The count has dropped sharply from the recorded 3-1-a starting count (~321) — expected
   remaining: the wrapper templates, the internal-helper catches (`binary_dispatch` etc.),
   and the functions individually noted in PROGRESS.md; roughly ≤ 40. If the number is
   materially higher, list the offenders with `grep -n -B3` and either migrate them
   (3-1-a recipe) or confirm each has a PROGRESS.md note.

2. Adding a unary op is now "1 table row (`kUnaryOpTable` in `detail/elementwise.cpp`)
   + 1 binding entry + 1 Python re-export".

3. No computation loops remain in the binding layer:

   ```bash
   grep -n 'for (' cpp/mtorch/python/*.cpp
   ```

   Review every hit; what remains must only be argument processing and Python list/tuple
   construction (`PyTuple_SET_ITEM`, `PyList_...`, iteration over parsed args). Any hit
   that indexes tensor storage or does arithmetic on element values is a leftover kernel —
   assign it to a 3-4 destination and move it (same recipe), or note it in PROGRESS.md.

4. Full test run matches the baseline per §5.1 (the only permitted behavioral difference
   in this phase is the 3-5-b `isinstance` improvement, which is invisible to the test
   suite and recorded in PROGRESS.md), and the full benchmark is within 5%:

   ```bash
   mkdir -p benchmark-results
   pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
       --compat-benchmark-json benchmark-results/after-phase3.json
   python3 tools/compare_benchmarks.py \
       docs/design/baseline/benchmark-baseline.json benchmark-results/after-phase3.json
   ```

   PASS iff the final line reads `... regressions=0 missing=0 new=0` (apply the §5.2
   3-rerun rule to any REGRESSION before concluding).

Check off the Phase 3 lines in PROGRESS.md as each step completed (per
`01-rules-and-verification.md` §8), and commit the final PROGRESS.md update.
