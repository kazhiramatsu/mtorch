# 04. Phase 2a: Splitting `module.cpp` (the binding layer)

Prerequisite: Phase 0 complete (can be started in any order relative to Phase 1). `01-rules-and-verification.md` read.
Progress tracking: check off steps 2a-1 … 2a-16 in `docs/design/PROGRESS.md` as described in each step's **Commit** section.

Objective: Mechanically split `cpp/mtorch/python/module.cpp` (12,560 lines, one single
anonymous namespace from L25 to L12525, plus `PyInit__C` at L12527) into about 14 files
organized by responsibility.

**This phase is purely mechanical moves.** Unifying the try/catch boilerplate, standardizing
the forwarding helpers, and relocating the numeric kernels into core are done in Phase 3 (3-1, 3-4).
The only rewrites allowed during the move are "linkage adjustments" (taking symbols out of the
anonymous namespace into `namespace mtorch::py`, adding declarations to headers, and moving a
default argument from a definition to its header declaration).

Every function name, struct name, and table row named in this document has been verified against
the current `module.cpp`. Line numbers given as "hint LNNNN" refer to the ORIGINAL 12,560-line
file; they shift as steps complete. **Always locate code with the given grep command, never with
the line hint alone.**

## 0. C++ splitting fundamentals (required reading)

- **Symbols inside an anonymous namespace have internal linkage** and cannot be referenced from
  other .cpp files. Symbols used from multiple files must live in a named namespace (in this
  phase, `mtorch::py`) and be declared in a header.
- **In this phase every new section file places ALL of its moved code inside
  `namespace mtorch::py { ... }`** (not an anonymous namespace). This is safe by construction:
  every symbol name in the original `module.cpp` was unique inside its single anonymous
  namespace, so no two section files can define the same external symbol. It also means a
  section never breaks the build just because a sibling file needs one of its helpers — the
  helper only needs a declaration added to that section's header. Phase 3 may re-tighten
  visibility. `module.cpp` itself keeps its anonymous namespace throughout; it gets
  `using namespace mtorch::py;` at file scope (added in step 2a-2) so its remaining code can
  call the moved symbols without any call-site rewrites.
- The global variables `TensorType` and `GeneratorType`, and the process-wide RNG accessors
  `global_rng()` / `global_initial_seed()`, are **defined once in `py_common.cpp`** and declared
  in `py_common.h`. **Putting a variable definition in a header causes a duplicate-symbol link
  error**; headers hold `extern` declarations and function declarations only. Struct definitions
  (`PyTensor`, `RandomState`, …) go in the header ONLY — delete them from the .cpp when moving.
- **Default arguments** may appear in the header declaration or the in-file definition, but a
  cross-file caller only sees the header. Rule used here: if a function moves to a header, the
  default argument moves to the header declaration and is deleted from the definition. Each step
  lists exactly which defaults to delete.
- The top of every new .cpp must start exactly like this (`PY_SSIZE_T_CLEAN` before `Python.h`):

  ```cpp
  #define PY_SSIZE_T_CLEAN
  #include <Python.h>
  ```

- This machine is macOS: in-place sed is `sed -i '' ...` (with the empty string argument).

## 1. Target file structure

All files live in `cpp/mtorch/python/`. "Anchor" = first symbol of the main block, usable with
`grep -nF '<anchor>' cpp/mtorch/python/module.cpp`.

| Step | File | Contents (anchor in original module.cpp) | Approx. lines |
|---|---|---|---|
| 2a-2 | `py_common.{h,cpp}` | `PyTensor`, exceptions, `translate_exception`, wrap/unwrap, dtype/device/shape/memory-format parse helpers, `binary_dispatch`/`unary_dispatch`, `TensorType`/`GeneratorType`/`global_rng` (anchors `struct TypeErrorException`, hint L732; `PyObject* scalar_to_py(`, hint L1888) | ~900 |
| 2a-3 | `py_registry.h`, `module_init.cpp` | table concatenation, `PyInit__C`, `module_def` (anchor `PyMODINIT_FUNC PyInit__C`, hint L12527) | ~150 |
| 2a-4 | `py_random.cpp` (+`py_random.h`) | RNG kernels, `py_rand*`, Generator type (anchors `void seed_random_state(`, hint L70; `PyObject* py_randint(`, hint L2413) | ~1,100 |
| 2a-5 | `py_creation.cpp` | `py_tensor` … `py_eye` (anchor `PyObject* py_tensor(`, hint L2085) | ~330 |
| 2a-6 | `py_pointwise.cpp` (+`py_pointwise.h`) | unary/binary/comparison/clamp/softmax/norm (anchor `PyObject* py_unary(`, hint L2779) | ~1,050 |
| 2a-7 | `py_reduction.cpp` (+`py_reduction.h`) | sum/mean/var/std/arg* (anchor `PyObject* py_sum(`, hint L3815) | ~690 |
| 2a-8 | `py_sort_search.cpp` (+`py_sort_search.h`) | sort/topk/searchsorted/unique/isin (anchor `PyObject* py_sort(`, hint L5413) | ~340 |
| 2a-9 | `py_linalg.cpp` (+`py_linalg.h`) | matmul/einsum/addmm family (anchor `void ensure_not_bool_matrix_contraction(`, hint L5994) | ~790 |
| 2a-10 | `py_index_select.cpp` (+`py_index_select.h`) | where/gather/scatter/masked/bincount (anchor `PyObject* py_where(`, hint L8082) | ~370 |
| 2a-11 | `py_shape.cpp` | view/layout/pad/interpolate/split/cat, incl. `trace` and the adaptive pools that sit in this range (anchors `PyObject* py_reshape(`, hint L4498; `PyObject* py_broadcast_shapes(`, hint L5763) | ~1,130 |
| 2a-12 | `py_nn_functional.cpp` | linear/conv/pool/norm/loss + activations (anchors `PyObject* py_linear(`, hint L6778; `PyObject* py_relu(`, hint L8447) | ~1,580 |
| 2a-13 | `py_autograd.cpp` | grad mode, `autograd.grad`, plus the module-level utilities `py_clone` … `py_is_signed` and `py_not_implemented` that sit in the same range (anchor `PyObject* py_clone(`, hint L8716) | ~300 |
| 2a-14 | `py_indexing.{h,cpp}` | index-key parsing (anchor `mtorch::TensorIndex make_select_index(`, hint L1457) + subscript protocol (anchor `PyObject* Tensor_subscript(`, hint L10290) | ~600 |
| 2a-15 | `py_tensor_type.cpp` | everything that remains of module.cpp: `Tensor_*` methods, number/sequence protocol, `Tensor_getset`, `kTensorMethods`, `Tensor_spec`, `make_tensor_type` | ~3,000 |

Module-method table row totals per section (sum = 287 rows, all of `module_methods`):
creation 14, random 9, pointwise 91, reduction 25, shape 54, sort/search 9, linalg 19,
nn 38, index/select 14, autograd/util 14.

## 2. Conventions used by every step

### 2.1 The BLOCK CUT recipe

A "block" is a contiguous run of definitions, identified by the signature line of its FIRST
symbol and the signature line of its LAST symbol. Functions end at the first line that is
exactly `}` in column 0 after the last signature; tables/specs end at the first line that is
exactly `};`. To extract and delete a block:

```bash
F='cpp/mtorch/python/module.cpp'
START=$(grep -nF '<FIRST-SIGNATURE>' "$F" | cut -d: -f1)
LAST=$(grep -nF '<LAST-SIGNATURE>' "$F" | cut -d: -f1)
END=$(awk -v s="$LAST" 'NR>s && /^}$/ {print NR; exit}' "$F")      # use /^};$/ when the step says the block ends in a table or spec
echo "START=$START LAST=$LAST END=$END"                             # all three must be non-empty, START < LAST < END
sed -n "${START},${END}p" "$F" > /tmp/block.cpp
sed -i '' "${START},${END}d" "$F"
```

When a step moves TWO blocks: extract BOTH to `/tmp/blockN.cpp` first (line numbers are valid
simultaneously), then delete the block with the LARGER start line first, then the other one.
Paste blocks into the new file in their original file order (block with the smaller original
start line first).

After extracting, sanity-check the block's function list with
`grep -E '^[A-Za-z].*\(' /tmp/block.cpp` and compare against the function list printed in the step.

### 2.2 The NEW FILE template

Every new section .cpp is created with this exact top part (add the extra `#include` lines the
step specifies after the `py_common.h` line):

```cpp
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "mtorch/core/tensor.h"
#include "mtorch/python/py_common.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(MTORCH_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

namespace mtorch::py {
```

then the pasted block(s), then the section's method table (steps 2a-4 … 2a-13 only), then:

```cpp
}  // namespace mtorch::py
```

The `using mtorch::Tensor;` etc. aliases are inherited from `py_common.h` (they are declared
inside `namespace mtorch::py` there), so pasted code compiles unchanged.

### 2.3 The ROW MOVER script

Steps 2a-4 … 2a-13 move rows out of `kLegacyModuleMethods`. Run this from the repo root with the
step's name list; it deletes the rows from `module.cpp` and writes them to `/tmp/rows.txt`
(safe: only rows whose implementation is a `py_*` function are touched, so `Tensor_*` /
`Generator_*` rows can never match):

```bash
python3 - <<'EOF'
import re
names = {NAMES}                      # <-- replace with the step's set, e.g. {"randint", "rand"}
expected = 0                         # <-- replace with the step's expected row count
path = "cpp/mtorch/python/module.cpp"
lines = open(path).readlines()
moved, kept = [], []
for line in lines:
    m = re.match(r'\s*\{"([^"]+)",', line)
    if m and m.group(1) in names and re.search(r"\bpy_[a-z0-9_]+\b", line):
        moved.append(line)
    else:
        kept.append(line)
assert len(moved) == expected, f"moved {len(moved)} rows, expected {expected}"
open(path, "w").writelines(kept)
open("/tmp/rows.txt", "w").writelines(moved)
print(f"moved {len(moved)} rows to /tmp/rows.txt")
EOF
```

Paste the content of `/tmp/rows.txt` verbatim into the new file's table body.

### 2.4 VERIFICATION block (run after every step, from the repo root)

```bash
python3 setup.py build_ext --inplace
python3 -c "from mtorch import _C; print(len([n for n in dir(_C) if not n.startswith('__')]))"
pytest tests/compat -q | tail -3
```

Pass criteria, all three required:
1. The build finishes with no errors.
2. The printed count equals METHOD_COUNT recorded in PROGRESS.md at step 2a-1.
3. The pytest summary line shows the same passed/failed/skipped counts as
   `tail -3 docs/design/baseline/tests-baseline.txt` (recorded in Phase 0, step 0-4).

### 2.5 ON FAILURE block

If verification fails and the cause is not an obvious typo you can fix in under 15 minutes
(see the pitfall list in §3 first):

```bash
git restore --staged --worktree .
git clean -fd cpp/mtorch/python
```

Then edit `docs/design/PROGRESS.md`: append ` — **BLOCKED**: <one-line reason>` to that step's
line, run `git add docs/design/PROGRESS.md && git commit -m "docs: mark 2a-N BLOCKED"`, and stop.

### 2.6 COMMIT recipe

```bash
git add -A
git commit -m "refactor(phase2a-N): <message given in the step>"
git log -1 --format=%h        # copy this hash
```

Then edit `docs/design/PROGRESS.md`: change the step's line from `- [ ]` to `- [x]` and fill in
`commit: <hash> / date: <YYYY-MM-DD>`. Commit that too:
`git add docs/design/PROGRESS.md && git commit -m "docs: PROGRESS 2a-N done"`.

## Step 2a-1: Making setup.py use glob

**Goal**: `setup.py` picks up every `.cpp` under `cpp/` automatically so that steps 2a-2 … 2a-15
never need to touch it again.

**Preconditions**:

```bash
git status --porcelain          # must print nothing
grep -n 'sources=' setup.py     # must show the explicit two-file list
```

**Actions**:

1. Replace the entire content of `setup.py` with exactly this
   (`cpp/mtorch/core/detail/` does not exist yet — the glob is a no-op until Phase 2b creates it;
   keep it anyway):

   ```python
   from __future__ import annotations

   import glob
   import sys

   from setuptools import Extension, setup


   extra_compile_args = ["-std=c++20", "-O3"]
   extra_link_args: list[str] = []
   if sys.platform == "darwin":
       extra_compile_args.append("-DMTORCH_USE_ACCELERATE")
       extra_link_args.extend(["-framework", "Accelerate"])

   setup(
       packages=["mtorch"],
       ext_modules=[
           Extension(
               "mtorch._C",
               sources=sorted(
                   glob.glob("cpp/mtorch/core/*.cpp")
                   + glob.glob("cpp/mtorch/core/detail/*.cpp")
                   + glob.glob("cpp/mtorch/python/*.cpp")
               ),
               include_dirs=["cpp"],
               language="c++",
               extra_compile_args=extra_compile_args,
               extra_link_args=extra_link_args,
           )
       ],
   )
   ```

2. Confirm the glob resolves to exactly the two current sources:

   ```bash
   python3 - <<'EOF'
   import glob
   sources = sorted(
       glob.glob("cpp/mtorch/core/*.cpp")
       + glob.glob("cpp/mtorch/core/detail/*.cpp")
       + glob.glob("cpp/mtorch/python/*.cpp")
   )
   print(sources)
   assert sources == ["cpp/mtorch/core/tensor.cpp", "cpp/mtorch/python/module.cpp"], sources
   EOF
   ```

**Verification**: run the §2.4 block. Additionally verify one object file per translation unit:

```bash
find build -type f -name '*.o' | sort
```

Expected: exactly two files, `.../cpp/mtorch/core/tensor.o` and `.../cpp/mtorch/python/module.o`.
Record the method count: write `— METHOD_COUNT: <printed number>` at the end of the 2a-1 line in
`docs/design/PROGRESS.md`. Every later step compares against this number.

**On failure**: §2.5 (for this step `git checkout -- setup.py` is enough).

**Commit**: §2.6 with message `refactor(phase2a-1): glob sources in setup.py`.

## Step 2a-2: Carving out `py_common.{h,cpp}`

**Goal**: shared structs, parse helpers, wrap/unwrap, exception translation, and the global type
objects move to `py_common.{h,cpp}` in `namespace mtorch::py`. `module.cpp` keeps compiling via a
file-scope `using namespace mtorch::py;`.

**Preconditions**:

```bash
git status --porcelain                                              # nothing
grep -cF 'struct TypeErrorException' cpp/mtorch/python/module.cpp   # prints 1
grep -cF 'PyObject* make_size_tuple(' cpp/mtorch/python/module.cpp  # prints 1
```

**Actions**:

1. Create `cpp/mtorch/python/py_common.h` with exactly this content (every signature below was
   verified against module.cpp — do not "fix" anything):

   ```cpp
   #pragma once
   #define PY_SSIZE_T_CLEAN
   #include <Python.h>

   #include <cstdint>
   #include <optional>
   #include <stdexcept>
   #include <string>
   #include <vector>

   #include "mtorch/core/tensor.h"

   namespace mtorch::py {

   using mtorch::Device;
   using mtorch::DeviceType;
   using mtorch::ScalarType;
   using mtorch::Tensor;
   using mtorch::TensorPtr;
   using mtorch::UniqueResult;

   // --- global type objects (defined in py_common.cpp) ---
   extern PyObject* TensorType;
   extern PyObject* GeneratorType;

   // --- structs / enums (definitions live here ONLY; moved verbatim from module.cpp) ---
   struct RandomState {
     uint64_t state = 0;
     bool has_spare_normal = false;
     double spare_normal = 0.0;
   };

   struct PyGenerator {
     PyObject_HEAD
     RandomState rng;
     bool uses_global = false;
     uint64_t initial_seed = 0;
   };

   struct TypeErrorException : public std::runtime_error {
     using std::runtime_error::runtime_error;
   };

   struct NotImplementedException : public std::runtime_error {
     using std::runtime_error::runtime_error;
   };

   struct PyTensor {
     PyObject_HEAD
     TensorPtr* value;
     bool is_parameter;
   };

   enum class MemoryFormat {
     Preserve,
     Contiguous,
     ChannelsLast,
     ChannelsLast3d,
   };

   struct FactoryOptions {
     std::vector<int64_t> shape;
     PyObject* dtype = Py_None;
     PyObject* device = Py_None;
     PyObject* generator = Py_None;
     bool requires_grad = false;
   };

   struct ToRequest {
     ScalarType dtype;
     Device device;
     bool copy = false;
     std::optional<MemoryFormat> memory_format;
   };

   // --- RNG state ---
   RandomState& global_rng();
   uint64_t& global_initial_seed();
   RandomState& random_state_from_generator(PyObject* generator);

   // --- dtype helpers ---
   bool is_floating_scalar_type(ScalarType dtype);
   bool dtype_is_signed(ScalarType dtype);
   bool dtype_allows_requires_grad(ScalarType dtype);
   ScalarType dtype_from_py(PyObject* object, ScalarType fallback = ScalarType::Float32);
   ScalarType default_arange_dtype(PyObject* dtype, PyObject* start, PyObject* end, PyObject* step);
   ScalarType default_sum_dtype(ScalarType input_dtype);
   ScalarType infer_scalar_dtype(PyObject* object);
   ScalarType merge_inferred_dtype(ScalarType left, ScalarType right);

   // --- tensor wrap/unwrap ---
   bool is_tensor(PyObject* object);
   TensorPtr& tensor_ref(PyObject* object);
   PyObject* wrap_tensor(const TensorPtr& tensor);
   std::vector<TensorPtr> tensor_sequence_from_py(PyObject* object, const char* name, bool allow_single_tensor = true);

   // --- exception translation (call inside catch(...) of every binding) ---
   void translate_exception();

   // --- text / device parsing ---
   std::string lowercase_ascii(std::string text);
   std::string object_text_lower(PyObject* object);
   bool text_looks_like_device(const std::string& text);
   std::string py_type_name_lower(PyObject* object);
   Device device_from_py(PyObject* object, Device fallback = mtorch::cpu_device());

   // --- memory format ---
   MemoryFormat memory_format_from_py(PyObject* object);
   std::optional<MemoryFormat> optional_memory_format_from_py(PyObject* object);
   std::vector<int64_t> channels_last_strides_2d(const std::vector<int64_t>& sizes);
   std::vector<int64_t> channels_last_strides_3d(const std::vector<int64_t>& sizes);
   bool tensor_is_contiguous_memory_format(const Tensor& tensor, MemoryFormat memory_format);
   void copy_storage_element(const Tensor& source, int64_t source_offset, Tensor& target, int64_t target_offset);
   TensorPtr tensor_to_memory_format(const TensorPtr& input, MemoryFormat memory_format, bool copy = false);
   TensorPtr make_like_with_memory_format(
       const TensorPtr& source,
       ScalarType dtype,
       Device device,
       bool requires_grad,
       std::optional<MemoryFormat> memory_format);

   // --- scalar / shape parsing ---
   bool object_is_sequence(PyObject* object);
   double scalar_from_py(PyObject* object);
   std::optional<double> optional_scalar_from_py(PyObject* object);
   std::optional<int64_t> optional_int64_from_py(PyObject* object, const char* name);
   double correction_from_py(PyObject* correction, PyObject* unbiased);
   void parse_nested_data(
       PyObject* object,
       std::vector<double>& values,
       std::vector<int64_t>& shape,
       ScalarType& inferred_dtype,
       int64_t depth);
   std::vector<int64_t> shape_from_object(PyObject* object);
   std::vector<double> double_vector_from_object(PyObject* object);
   std::vector<int64_t> shape_from_args(PyObject* args, Py_ssize_t start = 0);
   std::vector<int64_t> dims_from_args(PyObject* args, Py_ssize_t start = 0);
   FactoryOptions factory_options_from_args(
       PyObject* args,
       PyObject* kwargs,
       const char* name,
       bool allow_random_options = false);

   // --- Python object construction ---
   PyObject* scalar_to_py(double value, ScalarType dtype);
   PyObject* tensor_to_nested_list(const Tensor& tensor, size_t depth, std::vector<int64_t>& index);
   bool pyobject_to_scalar(PyObject* object, double& value, ScalarType* dtype = nullptr);
   PyObject* make_size_tuple(const std::vector<int64_t>& values);
   PyObject* tuple_from_tensors(const std::vector<TensorPtr>& tensors);
   PyObject* tuple_from_int64s(const std::vector<int64_t>& values);

   // --- .to() argument parsing ---
   void apply_to_argument(PyObject* object, ToRequest& request);
   ToRequest parse_to_request(const Tensor& source, PyObject* args, PyObject* kwargs);

   // --- operator dispatch ---
   PyObject* binary_dispatch(PyObject* left, PyObject* right, const std::string& op);
   PyObject* unary_dispatch(PyObject* object, const std::string& op);
   PyObject* parse_tensor_call(PyObject* args, PyObject* kwargs);

   }  // namespace mtorch::py
   ```

2. Create `cpp/mtorch/python/py_common.cpp` from the §2.2 template. The template already ends
   with the `namespace mtorch::py {` opener — directly below that opener (do NOT write a second
   one) put the two global definitions and the paste marker, then the closing line:

   ```cpp
   PyObject* TensorType = nullptr;
   PyObject* GeneratorType = nullptr;

   // === P2, P3, P4, P5, P6, P7, P8 pasted here, in this order ===

   }  // namespace mtorch::py
   ```

3. Extract these blocks from `module.cpp` with the §2.1 recipe (all end patterns are `^}$`)
   and append them, in order, into `py_common.cpp` at the paste marker:

   | Block | FIRST signature (grep -F) | LAST signature (grep -F) | hint |
   |---|---|---|---|
   | P2 | `bool is_floating_scalar_type(` | (same — single function) | L37 |
   | P3 | `struct RandomState {` | `uint64_t& global_initial_seed(` | L47–L68 |
   | P4 | `struct TypeErrorException` | `std::vector<int64_t> dims_from_args(` | L732–L1456 |
   | P5 | `PyObject* scalar_to_py(` | `PyObject* make_size_tuple(` | L1888–L2083 |
   | P6 | `PyObject* tuple_from_tensors(` | (single function) | L5397 |
   | P7 | `PyObject* tuple_from_int64s(` | (single function) | L5747 |
   | P8 | `bool dtype_is_signed(` | (single function) | L8788 |

   Note: P4 deliberately stops at `dims_from_args` — the index-key helpers that follow
   (`make_select_index` onward, hint L1457) STAY in module.cpp until step 2a-14.

4. Also delete these two lines from `module.cpp` (hint L34–35); their definitions are now in
   `py_common.cpp`:

   ```cpp
   PyObject* TensorType = nullptr;
   PyObject* GeneratorType = nullptr;
   ```

5. In `py_common.cpp`, delete the pieces that now live in the header (they came along inside
   blocks P3/P4/P5). Delete exactly these definitions, whole, nothing else:
   `struct RandomState { ... };` (4 lines), `struct PyGenerator { ... };` (6 lines),
   `struct TypeErrorException ... };` (3 lines), `struct NotImplementedException ... };` (3 lines),
   `struct PyTensor { ... };` (5 lines), `enum class MemoryFormat { ... };` (6 lines),
   `struct FactoryOptions { ... };` (7 lines), `struct ToRequest { ... };` (6 lines), and the
   one-line forward declaration `bool object_is_sequence(PyObject* object);` (hint L754; keep the
   full DEFINITION of `object_is_sequence` that appears later in P4, hint L1154).

6. In `py_common.cpp`, delete the default arguments from these definitions (they are now on the
   header declarations). Exact edits:
   - `tensor_sequence_from_py`: `bool allow_single_tensor = true` → `bool allow_single_tensor`
   - `dtype_from_py`: `ScalarType fallback = ScalarType::Float32` → `ScalarType fallback`
   - `device_from_py`: `Device fallback = mtorch::cpu_device()` → `Device fallback`
   - `tensor_to_memory_format`: `bool copy = false` → `bool copy`
   - `shape_from_args`: `Py_ssize_t start = 0` → `Py_ssize_t start`
   - `dims_from_args`: `Py_ssize_t start = 0` → `Py_ssize_t start`
   - `factory_options_from_args`: `bool allow_random_options = false` → `bool allow_random_options`
   - `pyobject_to_scalar`: `ScalarType* dtype = nullptr` → `ScalarType* dtype`

7. In `module.cpp`, immediately after the `#endif` of the Accelerate include block (hint L23) and
   BEFORE the `namespace {` line, insert:

   ```cpp
   #include "mtorch/python/py_common.h"

   using namespace mtorch::py;
   ```

8. Build (`python3 setup.py build_ext --inplace`). If the compiler reports an undeclared
   identifier in module.cpp, that symbol was missed — find its definition in py_common.cpp order
   above; if it is genuinely still in module.cpp and used only there, leave it (do NOT move
   extra symbols beyond the lists in this document).

**Verification**: §2.4 block.

**On failure**: §2.5.

**Commit**: §2.6, message `refactor(phase2a-2): extract py_common.{h,cpp}`.

## Step 2a-3: `module_init.cpp` and the table concatenation mechanism

**Goal**: `PyInit__C` moves to a new `module_init.cpp` that assembles the module and Tensor
method tables from per-section tables at import time. `module.cpp`'s two big tables become
`kLegacyModuleMethods` / `kLegacyTensorMethods` with external linkage.

**Preconditions**:

```bash
git status --porcelain                                                  # nothing
grep -cF 'PyMethodDef module_methods[] = {' cpp/mtorch/python/module.cpp  # 1  (hint L12161 originally)
grep -cF 'PyMethodDef Tensor_methods[] = {' cpp/mtorch/python/module.cpp  # 1  (hint L11920 originally)
grep -cF 'PyMODINIT_FUNC PyInit__C' cpp/mtorch/python/module.cpp           # 1
```

**Actions**:

1. Create `cpp/mtorch/python/py_registry.h` with exactly:

   ```cpp
   #pragma once
   #define PY_SSIZE_T_CLEAN
   #include <Python.h>

   namespace mtorch::py {

   // Method tables exposed by section files. Every table is terminated by the
   // {nullptr, nullptr, 0, nullptr} sentinel row.
   // Steps 2a-4 .. 2a-13 add one extern declaration here per split-out section.
   extern PyMethodDef kLegacyModuleMethods[];
   extern PyMethodDef kLegacyTensorMethods[];

   // Defined in module_init.cpp: concatenate all registered tables.
   PyMethodDef* build_module_methods();
   PyMethodDef* build_tensor_methods();

   // Defined next to the corresponding PyType_Spec:
   // make_tensor_type in module.cpp (later py_tensor_type.cpp),
   // make_generator_type in module.cpp (moves to py_random.cpp in 2a-4).
   PyObject* make_tensor_type();
   PyObject* make_generator_type();

   }  // namespace mtorch::py
   ```

2. In `module.cpp`, add `#include "mtorch/python/py_registry.h"` on the line directly after the
   `#include "mtorch/python/py_common.h"` added in 2a-2.

3. Move the two big tables out of the anonymous namespace. Locate
   `grep -nF 'PyMethodDef Tensor_methods[] = {' cpp/mtorch/python/module.cpp` and insert
   directly ABOVE that line:

   ```cpp
   }  // namespace

   namespace mtorch::py {
   ```

   Then locate the END of `module_methods` — the table order in the file is `Tensor_methods`
   (hint L11920) immediately followed by `module_methods` (hint L12161), so find
   `grep -nF 'PyMethodDef module_methods[] = {' cpp/mtorch/python/module.cpp` and the first
   line that is exactly `};` after it — and insert directly BELOW that `};` line:

   ```cpp
   }  // namespace mtorch::py

   namespace {
   ```

4. Rename the two tables (definition lines only — nothing else references these names except
   the two places fixed in actions 5 and 7):
   - `PyMethodDef Tensor_methods[] = {` → `PyMethodDef kLegacyTensorMethods[] = {`
   - `PyMethodDef module_methods[] = {` → `PyMethodDef kLegacyModuleMethods[] = {`
   Confirm the sentinel row `{nullptr, nullptr, 0, nullptr},` is the last row of each
   (it is; do not remove it).

5. In `Tensor_slots` (hint L12474), change the one line
   `{Py_tp_methods, reinterpret_cast<void*>(Tensor_methods)},` to
   `{Py_tp_methods, reinterpret_cast<void*>(mtorch::py::kLegacyTensorMethods)},`.

6. Delete from `module.cpp`: the whole `PyModuleDef module_def = { ... };` definition
   (7 lines, hint L12517) and the whole `PyMODINIT_FUNC PyInit__C() { ... }` function
   (hint L12527 to end of file). The file now ends with the `}  // namespace` that closes the
   anonymous namespace.

7. Append at the very end of `module.cpp`:

   ```cpp
   namespace mtorch::py {

   PyObject* make_tensor_type() {
     for (PyType_Slot* slot = Tensor_slots; slot->slot != 0; ++slot) {
       if (slot->slot == Py_tp_methods) {
         slot->pfunc = reinterpret_cast<void*>(build_tensor_methods());
       }
     }
     return PyType_FromSpec(&Tensor_spec);
   }

   PyObject* make_generator_type() {
     return PyType_FromSpec(&Generator_spec);
   }

   }  // namespace mtorch::py
   ```

   (`Tensor_slots`, `Tensor_spec`, `Generator_spec` are anonymous-namespace names in the same
   translation unit, so this compiles. The `Py_tp_methods` slot is finalized before
   `PyType_FromSpec` runs — the vector behind `build_tensor_methods()` never grows afterward.)

8. Create `cpp/mtorch/python/module_init.cpp` with exactly:

   ```cpp
   #define PY_SSIZE_T_CLEAN
   #include <Python.h>

   #include <vector>

   #include "mtorch/python/py_common.h"
   #include "mtorch/python/py_registry.h"

   namespace {

   void append_table(std::vector<PyMethodDef>& out, const PyMethodDef* table) {
     for (; table->ml_name != nullptr; ++table) {
       out.push_back(*table);
     }
   }

   PyModuleDef module_def = {
       PyModuleDef_HEAD_INIT,
       "mtorch._C",
       "Native C++ core bindings for mtorch.",
       -1,
       nullptr,
   };

   }  // namespace

   namespace mtorch::py {

   PyMethodDef* build_module_methods() {
     static std::vector<PyMethodDef> methods;  // module lifetime; must not grow after PyModule_Create
     if (methods.empty()) {
       append_table(methods, kLegacyModuleMethods);
       // Steps 2a-4 .. 2a-13 add one append_table(...) line here per split-out section.
       methods.push_back({nullptr, nullptr, 0, nullptr});
     }
     return methods.data();
   }

   PyMethodDef* build_tensor_methods() {
     static std::vector<PyMethodDef> methods;  // must be finalized before PyType_FromSpec runs
     if (methods.empty()) {
       append_table(methods, kLegacyTensorMethods);
       methods.push_back({nullptr, nullptr, 0, nullptr});
     }
     return methods.data();
   }

   }  // namespace mtorch::py

   PyMODINIT_FUNC PyInit__C() {
     module_def.m_methods = mtorch::py::build_module_methods();
     PyObject* module = PyModule_Create(&module_def);
     if (module == nullptr) {
       return nullptr;
     }

     mtorch::py::TensorType = mtorch::py::make_tensor_type();
     if (mtorch::py::TensorType == nullptr) {
       Py_DECREF(module);
       return nullptr;
     }

     Py_INCREF(mtorch::py::TensorType);
     if (PyModule_AddObject(module, "Tensor", mtorch::py::TensorType) < 0) {
       Py_DECREF(mtorch::py::TensorType);
       Py_DECREF(module);
       return nullptr;
     }

     mtorch::py::GeneratorType = mtorch::py::make_generator_type();
     if (mtorch::py::GeneratorType == nullptr) {
       Py_DECREF(module);
       return nullptr;
     }

     Py_INCREF(mtorch::py::GeneratorType);
     if (PyModule_AddObject(module, "Generator", mtorch::py::GeneratorType) < 0) {
       Py_DECREF(mtorch::py::GeneratorType);
       Py_DECREF(module);
       return nullptr;
     }

     return module;
   }
   ```

**Verification**: §2.4 block (this is where the METHOD_COUNT comparison first proves the
concatenation works). Also `find build -type f -name '*.o' | sort` must now list a `.o` for
`module.cpp`, `py_common.cpp`, and `module_init.cpp`.

**On failure**: §2.5. Most likely causes: forgot the sentinel, forgot action 5, or the
`}  // namespace` / `namespace {` insertions from action 3 are unbalanced
(check with `grep -c '^namespace {' cpp/mtorch/python/module.cpp` vs `grep -c '^}  // namespace$' ...`).

**Commit**: §2.6, message `refactor(phase2a-3): module_init.cpp and method-table registry`.

## Steps 2a-4 … 2a-13: carving out the sections (1 file = 1 commit)

Fixed order (matches PROGRESS.md): 2a-4 `py_random` → 2a-5 `py_creation` → 2a-6 `py_pointwise`
→ 2a-7 `py_reduction` → 2a-8 `py_sort_search` → 2a-9 `py_linalg` → 2a-10 `py_index_select`
→ 2a-11 `py_shape` → 2a-12 `py_nn_functional` → 2a-13 `py_autograd`. Then 2a-14 `py_indexing`
and 2a-15 `py_tensor_type`.

Common procedure (fully worked in 2a-4; later steps give only the data):

1. Create the section header (if the step lists exported symbols) with the exact content given.
2. Create the new .cpp from the §2.2 template (plus the step's extra includes).
3. Cut the step's block(s) from module.cpp (§2.1) and paste inside `namespace mtorch::py {`.
4. Move the step's method-table rows with the §2.3 script; append the section table before the
   closing `}  // namespace mtorch::py`.
5. Add the step's `extern` line to `py_registry.h` and the step's `append_table` line to
   `build_module_methods()` in `module_init.cpp`.
6. Add the step's `#include` line(s) to `module.cpp` (needed when code remaining in module.cpp
   calls the section's exported symbols).
7. Verify (§2.4), commit (§2.6).

**Never decide category membership yourself** — the function lists below are exhaustive; a
function not listed for the current step stays in module.cpp.

### Step 2a-4: `py_random.cpp` (worked example)

**Goal**: RNG kernels, random ops, and the Generator type move to `py_random.cpp`.

**Preconditions**:

```bash
git status --porcelain                                              # nothing
grep -cF 'void seed_random_state(' cpp/mtorch/python/module.cpp     # 1
grep -cF 'PyObject* py_randint(' cpp/mtorch/python/module.cpp       # 1
grep -cF 'PyObject* make_generator_type() {' cpp/mtorch/python/module.cpp  # 1 (added in 2a-3)
```

**Actions**:

1. Create `cpp/mtorch/python/py_random.h` with exactly:

   ```cpp
   #pragma once
   #define PY_SSIZE_T_CLEAN
   #include <Python.h>

   #include "mtorch/core/tensor.h"
   #include "mtorch/python/py_common.h"

   namespace mtorch::py {

   // Random kernels also called from outside py_random.cpp:
   // normal_inplace / uniform_inplace by Tensor_normal_inplace / Tensor_uniform_inplace
   // (module.cpp -> py_tensor_type.cpp), dropout_tensor by py_dropout (py_nn_functional.cpp).
   void normal_inplace(Tensor& tensor, double mean, double std, RandomState& rng);
   void uniform_inplace(Tensor& tensor, double from, double to, RandomState& rng);
   TensorPtr dropout_tensor(const TensorPtr& input, double p, bool training);

   }  // namespace mtorch::py
   ```

2. Create `cpp/mtorch/python/py_random.cpp`: §2.2 template, with these two lines added after the
   `py_common.h` include:

   ```cpp
   #include "mtorch/python/py_random.h"
   #include "mtorch/python/py_registry.h"
   ```

3. Extract four blocks. Blocks A, B, D end at `^}$`; block C ends at `^};$`.
   Extract all four to `/tmp/blockA.cpp` … `/tmp/blockD.cpp` FIRST, then delete from module.cpp
   in the order D, C, B, A (largest start line first):

   | Block | FIRST signature | LAST signature | hint | end |
   |---|---|---|---|---|
   | A | `void seed_random_state(` | `TensorPtr multinomial_tensor(` | L70–L731 | `^}$` |
   | B | `PyObject* py_randint(` | `PyObject* py_bernoulli(` | L2413–L2777 | `^}$` |
   | C | `PyMethodDef Generator_methods[] = {` | `PyType_Spec Generator_spec = {` | L12452–L12472 | `^};$` |
   | D | `PyObject* make_generator_type() {` | (single function) | end of file | `^}$` |

   Concretely, for block A:

   ```bash
   F='cpp/mtorch/python/module.cpp'
   START=$(grep -nF 'void seed_random_state(' "$F" | cut -d: -f1)
   LAST=$(grep -nF 'TensorPtr multinomial_tensor(' "$F" | cut -d: -f1)
   END=$(awk -v s="$LAST" 'NR>s && /^}$/ {print NR; exit}' "$F")
   sed -n "${START},${END}p" "$F" > /tmp/blockA.cpp
   ```

   Repeat for B, C (`/^};$/` in the awk pattern), D. Then delete: D range, C range, B range, A
   range (recompute each range right before its own deletion, or delete strictly in descending
   START order using the numbers you already printed).

   Paste into `py_random.cpp` inside `namespace mtorch::py {`, in the order A, B, C, D.

   Expected functions in A (17): `seed_random_state`, `next_uniform_open`,
   `next_standard_normal_pair`, `next_standard_normal`, `fill_random_normal_contiguous`,
   `fill_random_uniform_contiguous`, `fill_randn_result`, `fill_randint_result`,
   `normal_inplace`, `next_truncated_normal`, `trunc_normal_inplace`, `uniform_inplace`,
   `fill_bernoulli_contiguous`, `bernoulli_tensor`, `dropout_tensor`,
   `validate_multinomial_weight`, `multinomial_tensor`.
   Expected in B (14): `py_randint`, `py_rand`, `py_randn`, `py_randperm`,
   `py_trunc_normal_inplace`, `py_manual_seed`, `py_initial_seed`, `Generator_new`,
   `Generator_init`, `Generator_manual_seed`, `Generator_initial_seed`, `Generator_repr`,
   `py_multinomial`, `py_bernoulli`.
   Check with `grep -E '^[A-Za-z].*\(' /tmp/blockA.cpp` etc.

4. In `py_random.cpp`, delete the default argument `= global_rng()` from the definitions of
   `normal_inplace` and `uniform_inplace` ONLY (every call site in the codebase passes the rng
   explicitly; the header declares them without defaults). Leave every other default in block A
   untouched.

5. Move the 9 module-table rows with the §2.3 script using:

   ```python
   names = {"randint", "rand", "randn", "randperm", "trunc_normal_",
            "manual_seed", "initial_seed", "multinomial", "bernoulli"}
   expected = 9
   ```

   Append to `py_random.cpp`, after block D and before the closing `}  // namespace mtorch::py`:

   ```cpp
   PyMethodDef kRandomModuleMethods[] = {
       // <content of /tmp/rows.txt pasted here, verbatim>
       {nullptr, nullptr, 0, nullptr},
   };
   ```

6. In `py_registry.h`, below `extern PyMethodDef kLegacyTensorMethods[];`, add:

   ```cpp
   extern PyMethodDef kRandomModuleMethods[];
   ```

7. In `module_init.cpp`, inside `build_module_methods()`, directly below
   `append_table(methods, kLegacyModuleMethods);`, add:

   ```cpp
   append_table(methods, kRandomModuleMethods);
   ```

8. In `module.cpp`, add below the `py_registry.h` include:

   ```cpp
   #include "mtorch/python/py_random.h"
   ```

**Verification**: §2.4 block. Sanity: `grep -cF 'PyObject* py_' cpp/mtorch/python/py_random.cpp`
prints 9 and `grep -rn 'seed_random_state' cpp/mtorch/python/module.cpp` prints nothing.

**On failure**: §2.5.

**Commit**: §2.6, message `refactor(phase2a-4): split py_random.cpp out of module.cpp`.

### Step 2a-5: `py_creation.cpp`

- **Goal**: factory functions move to `py_creation.cpp`. No header (nothing is exported).
- **Preconditions**: `git status --porcelain` empty;
  `grep -cF 'PyObject* py_tensor(' cpp/mtorch/python/module.cpp` prints 1.
- **Blocks** (end `^}$`): one block, `PyObject* py_tensor(` → `PyObject* py_eye(` (hint L2085–L2412).
  Functions (14): `py_tensor`, `py_as_tensor`, `py_zeros`, `py_ones`, `py_empty`,
  `py_empty_strided`, `py_full`, `py_empty_like`, `py_zeros_like`, `py_ones_like`,
  `py_full_like`, `py_arange`, `py_linspace`, `py_eye`.
- **Extra includes for the new file**: none beyond the template.
- **Rows** (§2.3): `names = {"tensor", "as_tensor", "zeros", "ones", "empty", "empty_strided",
  "full", "empty_like", "zeros_like", "ones_like", "full_like", "arange", "linspace", "eye"}`,
  `expected = 14`. Table name: `kCreationModuleMethods`.
- **Registry/init**: add `extern PyMethodDef kCreationModuleMethods[];` to `py_registry.h`;
  add `append_table(methods, kCreationModuleMethods);` to `build_module_methods()`.
- **module.cpp includes**: none needed.
- **Verification**: §2.4; `grep -cF 'PyObject* py_' cpp/mtorch/python/py_creation.cpp` prints 14.
- **On failure**: §2.5. **Commit**: §2.6, `refactor(phase2a-5): split py_creation.cpp out of module.cpp`.

### Step 2a-6: `py_pointwise.cpp` (+ `py_pointwise.h`)

- **Goal**: unary/binary/comparison/clamp/softmax/norm bindings move out. Tensor methods still
  in module.cpp forward to 14 of these `py_*` functions and 4 helpers, so this step has a header.
- **Preconditions**: `git status --porcelain` empty;
  `grep -cF 'PyObject* py_unary(' cpp/mtorch/python/module.cpp` prints 1.
- **Header** — create `cpp/mtorch/python/py_pointwise.h`:

  ```cpp
  #pragma once
  #define PY_SSIZE_T_CLEAN
  #include <Python.h>

  #include <optional>
  #include <vector>

  #include "mtorch/core/tensor.h"
  #include "mtorch/python/py_common.h"

  namespace mtorch::py {

  // Helpers also used by Tensor methods in py_tensor_type.cpp.
  double norm_order_from_py(PyObject* object);
  std::optional<std::vector<int64_t>> optional_dims_from_py(PyObject* dim);
  TensorPtr norm_tensor(
      const TensorPtr& input,
      double p,
      const std::optional<std::vector<int64_t>>& dims,
      bool keepdim,
      ScalarType dtype);
  int tensor_truth_value(const TensorPtr& tensor);

  // Module functions forwarded to by Tensor methods.
  PyObject* py_nan_to_num(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_clamp(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_clip(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_clamp_min(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_clamp_max(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_softmax(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_log_softmax(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_addcmul(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_addcdiv(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_isclose(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_allclose(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_equal(PyObject*, PyObject* args);
  PyObject* py_is_nonzero(PyObject*, PyObject* args);
  PyObject* py_lerp(PyObject*, PyObject* args);

  }  // namespace mtorch::py
  ```

- **Extra include for py_pointwise.cpp**: `#include "mtorch/python/py_pointwise.h"`.
- **Blocks** (end `^}$`): one block, `PyObject* py_unary(` → `PyObject* py_lerp(`
  (hint L2779–L3814). It contains ~95 functions, including the helpers `py_unary`,
  `py_unary_predicate`, `py_binary`, `norm_order_from_py`, `reduce_sum_dims_for_norm`,
  `norm_zero_tensor`, `norm_tensor`, `optional_dims_from_py`, `py_normalize_l2`,
  `tensor_truth_value`, and every `py_<op>` from `py_neg` through `py_lerp`. Everything in the
  block moves; nothing else does.
- **Rows** (§2.3), `expected = 91`:

  ```python
  names = {"neg", "abs", "exp", "expm1", "log", "log1p", "log2", "log10", "sqrt", "rsqrt",
           "reciprocal", "sign", "floor", "ceil", "trunc", "round", "sin", "cos", "tan",
           "sinh", "cosh", "tanh", "asin", "acos", "atan", "sigmoid", "erf", "erfc",
           "deg2rad", "rad2deg", "frac", "isnan", "isinf", "isfinite", "signbit",
           "isposinf", "isneginf", "logical_not", "bitwise_not", "square", "nan_to_num",
           "gelu", "clamp", "clip", "clamp_min", "clamp_max", "softmax", "log_softmax",
           "norm", "_normalize_l2", "add", "sub", "mul", "div", "pow", "floor_divide",
           "float_power", "remainder", "fmod", "atan2", "hypot", "ldexp", "nextafter",
           "copysign", "heaviside", "logaddexp", "logaddexp2", "xlogy", "fmax", "fmin",
           "addcmul", "addcdiv", "maximum", "minimum", "eq", "ne", "lt", "le", "gt", "ge",
           "logical_and", "logical_or", "logical_xor", "bitwise_and", "bitwise_or",
           "bitwise_xor", "isclose", "allclose", "equal", "is_nonzero", "lerp"}
  ```

  Table name: `kPointwiseModuleMethods`.
- **Registry/init**: `extern PyMethodDef kPointwiseModuleMethods[];` /
  `append_table(methods, kPointwiseModuleMethods);`.
- **module.cpp includes**: add `#include "mtorch/python/py_pointwise.h"` below the
  `py_random.h` include.
- **Verification**: §2.4; `grep -cF 'PyObject* py_' cpp/mtorch/python/py_pointwise.cpp` prints 94
  (91 row-backed functions plus the helpers `py_unary`, `py_unary_predicate`, `py_binary`).
- **On failure**: §2.5. **Commit**: §2.6, `refactor(phase2a-6): split py_pointwise.cpp out of module.cpp`.

### Step 2a-7: `py_reduction.cpp` (+ `py_reduction.h`)

- **Preconditions**: `git status --porcelain` empty;
  `grep -cF 'PyObject* py_sum(' cpp/mtorch/python/module.cpp` prints 1.
- **Header** — create `cpp/mtorch/python/py_reduction.h` (all 17 are forwarded to by Tensor
  methods; all have the 3-argument kwargs signature):

  ```cpp
  #pragma once
  #define PY_SSIZE_T_CLEAN
  #include <Python.h>

  namespace mtorch::py {

  PyObject* py_sum(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_cumsum(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_cumprod(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_cummax(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_cummin(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_mean(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_prod(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_var(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_std(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_all(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_any(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_amax(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_amin(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_max(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_argmax(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_min(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_argmin(PyObject*, PyObject* args, PyObject* kwargs);

  }  // namespace mtorch::py
  ```

- **Extra include for py_reduction.cpp**: `#include "mtorch/python/py_reduction.h"`.
- **Blocks** (end `^}$`): one block, `PyObject* py_sum(` → `PyObject* py_argmin(`
  (hint L3815–L4497). Functions (27): the 17 above plus `py_diff_float32`,
  `py_cumulative_extreme`, `py_trapezoid_dx`, `py_cumulative_trapezoid_dx`,
  `py_gradient_uniform`, `py_var_tail`, `py_std_tail`, `wrap_tensor_pair`, `py_var_mean`,
  `py_std_mean`.
- **Rows** (§2.3), `expected = 25`. Note: the `trace` row sits inside this run of the table but
  is NOT in this list — it moves in 2a-11:

  ```python
  names = {"sum", "_diff_float32", "cumsum", "cumprod", "cummax", "cummin", "_trapezoid_dx",
           "_cumulative_trapezoid_dx", "_gradient_uniform", "mean", "prod", "var", "std",
           "_var_tail", "_std_tail", "_var_mean", "_std_mean", "all", "any", "amax", "amin",
           "max", "argmax", "min", "argmin"}
  ```

  Table name: `kReductionModuleMethods`.
- **Registry/init**: `extern PyMethodDef kReductionModuleMethods[];` /
  `append_table(methods, kReductionModuleMethods);`.
- **module.cpp includes**: add `#include "mtorch/python/py_reduction.h"`.
- **Verification**: §2.4; `grep -cF 'PyObject* py_' cpp/mtorch/python/py_reduction.cpp` prints 26
  (25 rows reference 25 distinct functions; `py_cumulative_extreme` has no row).
- **On failure**: §2.5. **Commit**: §2.6, `refactor(phase2a-7): split py_reduction.cpp out of module.cpp`.

### Step 2a-8: `py_sort_search.cpp` (+ `py_sort_search.h`)

- **Preconditions**: `git status --porcelain` empty;
  `grep -cF 'PyObject* py_sort(' cpp/mtorch/python/module.cpp` prints 1.
- **Header** — create `cpp/mtorch/python/py_sort_search.h`:

  ```cpp
  #pragma once
  #define PY_SSIZE_T_CLEAN
  #include <Python.h>

  namespace mtorch::py {

  PyObject* py_sort(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_argsort(PyObject*, PyObject* args, PyObject* kwargs);
  PyObject* py_topk(PyObject*, PyObject* args, PyObject* kwargs);

  }  // namespace mtorch::py
  ```

- **Extra include for py_sort_search.cpp**: `#include "mtorch/python/py_sort_search.h"`.
- **Blocks** (end `^}$`): one block, `PyObject* py_sort(` → `PyObject* py_unique_consecutive(`
  (hint L5413–L5745). Functions (11): `py_sort`, `py_argsort`, `py_quantile_dim_2d`,
  `py_quantile_flat`, `py_topk`, `py_searchsorted`, `tensor_or_scalar_for_isin`, `py_isin`,
  `unique_result_to_py`, `py_unique`, `py_unique_consecutive`.
- **Rows** (§2.3), `expected = 9` (the `isin` row sits between `where` and `take` in the table —
  the script finds it regardless of position):

  ```python
  names = {"sort", "argsort", "_quantile_flat", "_quantile_dim_2d", "topk", "searchsorted",
           "unique", "unique_consecutive", "isin"}
  ```

  Table name: `kSortSearchModuleMethods`.
- **Registry/init**: `extern PyMethodDef kSortSearchModuleMethods[];` /
  `append_table(methods, kSortSearchModuleMethods);`.
- **module.cpp includes**: add `#include "mtorch/python/py_sort_search.h"`.
- **Verification**: §2.4; `grep -cF 'PyObject* py_' cpp/mtorch/python/py_sort_search.cpp` prints 9.
- **On failure**: §2.5. **Commit**: §2.6, `refactor(phase2a-8): split py_sort_search.cpp out of module.cpp`.

### Step 2a-9: `py_linalg.cpp` (+ `py_linalg.h`)

- **Preconditions**: `git status --porcelain` empty;
  `grep -cF 'void ensure_not_bool_matrix_contraction(' cpp/mtorch/python/module.cpp` prints 1;
  `grep -cF 'bool accelerate_int_ok(' cpp/mtorch/python/module.cpp` prints 1.
- **Header** — create `cpp/mtorch/python/py_linalg.h` (these four helpers are used by nn code
  until 2a-12 and by Tensor methods permanently):

  ```cpp
  #pragma once
  #define PY_SSIZE_T_CLEAN
  #include <Python.h>

  #include <vector>

  #include "mtorch/core/tensor.h"
  #include "mtorch/python/py_common.h"

  namespace mtorch::py {

  void ensure_same_dtype_matrix_contraction(const char* op, const TensorPtr& left, const TensorPtr& right);
  void ensure_all_same_dtype_matrix_contraction(const char* op, const std::vector<TensorPtr>& tensors);
  void ensure_all_same_dtype_non_bool(const char* op, const std::vector<TensorPtr>& tensors);
  double default_rms_norm_eps(ScalarType dtype);

  }  // namespace mtorch::py
  ```

- **Extra include for py_linalg.cpp**: `#include "mtorch/python/py_linalg.h"`.
- **Blocks**: extract both first, delete B then A:
  - Block A — the Accelerate-guarded helper at the top of module.cpp (hint L41–L45). Cut by
    arithmetic, and verify the extracted text is exactly the 5 lines shown:

    ```bash
    F='cpp/mtorch/python/module.cpp'
    A=$(grep -nF 'bool accelerate_int_ok(' "$F" | cut -d: -f1)
    sed -n "$((A-1)),$((A+3))p" "$F" > /tmp/blockA.cpp
    cat /tmp/blockA.cpp
    # must print exactly:
    # #if defined(MTORCH_USE_ACCELERATE)
    # bool accelerate_int_ok(int64_t value) {
    #   return value >= 0 && value <= std::numeric_limits<int>::max();
    # }
    # #endif
    ```

    Delete with `sed -i '' "$((A-1)),$((A+3))d" "$F"` (AFTER extracting and deleting block B).
  - Block B (end `^}$`): `void ensure_not_bool_matrix_contraction(` → `PyObject* py_outer(`
    (hint L5994–L6777). Functions (31): `ensure_not_bool_matrix_contraction`,
    `ensure_same_dtype_matrix_contraction`, `ensure_all_same_dtype_matrix_contraction`,
    `ensure_all_same_dtype_non_bool`, `default_rms_norm_eps`, `py_matmul`, `py_mm`,
    `einsum_attention_scores`, `einsum_attention_values`, `ensure_einsum_contraction_supported`,
    `split_simple_binary_einsum` (and its `SimpleBinaryEinsum` struct),
    `einsum_labels_are_unique`, `einsum_labels_match_rank`, `try_simple_binary_einsum_fast_path`,
    `py_einsum`, `py_bmm`, `py_addmm`, `py_addmv`, `py_addr`, `py_baddbmm`, `py_addbmm`,
    `py_vdot`, `py_inner`, `py_chain_matmul`, `py_matrix_power`, `parse_dim_sequence`,
    `parse_tensordot_dims`, `py_tensordot`, `py_kron`, `py_dot`, `py_mv`, `py_outer`.
  - Paste order in the new file: A then B.
- **Rows** (§2.3), `expected = 19` (the `ger` row also references `py_outer`; the script moves it):

  ```python
  names = {"matmul", "mm", "einsum", "bmm", "addmm", "addmv", "addr", "baddbmm", "addbmm",
           "vdot", "inner", "tensordot", "kron", "chain_matmul", "matrix_power", "dot",
           "mv", "outer", "ger"}
  ```

  Table name: `kLinalgModuleMethods`.
- **Registry/init**: `extern PyMethodDef kLinalgModuleMethods[];` /
  `append_table(methods, kLinalgModuleMethods);`.
- **module.cpp includes**: add `#include "mtorch/python/py_linalg.h"` (the conv/norm code still
  in module.cpp calls `ensure_all_same_dtype_non_bool` and `default_rms_norm_eps`; Tensor
  methods call the matrix-contraction checks).
- **Verification**: §2.4; `grep -cF 'PyObject* py_' cpp/mtorch/python/py_linalg.cpp` prints 18.
- **On failure**: §2.5. **Commit**: §2.6, `refactor(phase2a-9): split py_linalg.cpp out of module.cpp`.

### Step 2a-10: `py_index_select.cpp` (+ `py_index_select.h`)

- **Preconditions**: `git status --porcelain` empty;
  `grep -cF 'PyObject* py_where(' cpp/mtorch/python/module.cpp` prints 1.
- **Header** — create `cpp/mtorch/python/py_index_select.h`:

  ```cpp
  #pragma once
  #define PY_SSIZE_T_CLEAN
  #include <Python.h>

  #include "mtorch/core/tensor.h"
  #include "mtorch/python/py_common.h"

  namespace mtorch::py {

  // Also used by Tensor_scatter* methods in py_tensor_type.cpp.
  TensorPtr scatter_source_from_py(const TensorPtr& input, PyObject* source, const char* name);

  }  // namespace mtorch::py
  ```

- **Extra include for py_index_select.cpp**: `#include "mtorch/python/py_index_select.h"`.
- **Blocks** (end `^}$`): one block, `PyObject* py_where(` → `PyObject* py_one_hot(`
  (hint L8082–L8446). Functions (16): `py_where`, `py_take`, `py_index_select`, `py_gather`,
  `index_put_indices_from_py`, `py_index_put`, `scatter_source_from_py`, `py_scatter`,
  `py_scatter_add`, `py_masked_select`, `py_masked_fill`, `py_nonzero`, `py_argwhere`,
  `py_count_nonzero`, `py_bincount`, `py_one_hot`. (The `isin` row/function already moved in 2a-8.)
- **Rows** (§2.3), `expected = 14`:

  ```python
  names = {"where", "take", "index_select", "gather", "index_put", "scatter", "scatter_add",
           "masked_select", "masked_fill", "nonzero", "argwhere", "count_nonzero",
           "bincount", "one_hot"}
  ```

  Table name: `kIndexSelectModuleMethods`.
- **Registry/init**: `extern PyMethodDef kIndexSelectModuleMethods[];` /
  `append_table(methods, kIndexSelectModuleMethods);`.
- **module.cpp includes**: add `#include "mtorch/python/py_index_select.h"`.
- **Verification**: §2.4; `grep -cF 'PyObject* py_' cpp/mtorch/python/py_index_select.cpp` prints 14.
- **On failure**: §2.5. **Commit**: §2.6, `refactor(phase2a-10): split py_index_select.cpp out of module.cpp`.

### Step 2a-11: `py_shape.cpp`

- **Preconditions**: `git status --porcelain` empty;
  `grep -cF 'PyObject* py_reshape(' cpp/mtorch/python/module.cpp` prints 1;
  `grep -cF 'PyObject* py_broadcast_shapes(' cpp/mtorch/python/module.cpp` prints 1.
- **Header**: none (nothing exported).
- **Extra includes for py_shape.cpp**: none beyond the template.
- **Blocks** (both end `^}$`; extract both, delete B2 then B1, paste B1 then B2):
  - B1: `PyObject* py_reshape(` → `PyObject* py_trace(` (hint L4498–L5396). Functions (40):
    `py_reshape`, `py_unflatten`, `py_transpose`, `py_permute`, `py_movedim`, `py_swapaxes`,
    `py_swapdims`, `py_flatten`, `py_ravel`, `py_t`, `py_broadcast_to`, `py_tile`,
    `py_repeat_interleave`, `py_flip`, `py_pad`, `py_adaptive_avg_pool1d`,
    `py_adaptive_avg_pool2d`, `py_pixel_shuffle`, `py_pixel_unshuffle`, `py_channel_shuffle`,
    `py_interpolate`, `py_grid_sample`, `py_affine_grid`, `py_fliplr`, `py_flipud`, `py_rot90`,
    `py_roll`, `py_squeeze`, `py_unsqueeze`, `py_narrow`, `py_select`, `py_as_strided`,
    `py_diagonal`, `py_diag`, `py_diagflat`, `py_diag_embed`, `py_block_diag`, `py_tril`,
    `py_triu`, `py_trace`. (Yes, the adaptive pools and `trace` belong to this file — assignment
    is by block, not by name.)
  - B2: `PyObject* py_broadcast_shapes(` → `PyObject* py_cartesian_prod(` (hint L5763–L5993).
    Functions (13): `py_broadcast_shapes`, `py_broadcast_tensors`, `py_split`, `py_chunk`,
    `py_unbind`, `tensor_list_from_py`, `py_cat`, `py_stack`, `py_hstack`, `py_vstack`,
    `py_dstack`, `py_column_stack`, `py_cartesian_prod`.
- **Rows** (§2.3), `expected = 54` (`moveaxis` shares `py_movedim`, `row_stack` shares
  `py_vstack`; the `trace` row sits up in the reduction run of the table — all are matched by
  name regardless of position):

  ```python
  names = {"trace", "reshape", "unflatten", "transpose", "permute", "movedim", "moveaxis",
           "swapaxes", "swapdims", "flatten", "ravel", "t", "broadcast_to",
           "broadcast_shapes", "broadcast_tensors", "tile", "repeat_interleave", "pad",
           "adaptive_avg_pool1d", "adaptive_avg_pool2d", "pixel_shuffle", "pixel_unshuffle",
           "channel_shuffle", "interpolate", "grid_sample", "affine_grid", "flip", "fliplr",
           "flipud", "rot90", "roll", "squeeze", "unsqueeze", "narrow", "select",
           "as_strided", "diagonal", "diag", "diagflat", "diag_embed", "block_diag", "tril",
           "triu", "split", "chunk", "unbind", "cat", "stack", "hstack", "vstack",
           "row_stack", "dstack", "column_stack", "cartesian_prod"}
  ```

  Table name: `kShapeModuleMethods`.
- **Registry/init**: `extern PyMethodDef kShapeModuleMethods[];` /
  `append_table(methods, kShapeModuleMethods);`.
- **module.cpp includes**: none needed.
- **Verification**: §2.4; `grep -cF 'PyObject* py_' cpp/mtorch/python/py_shape.cpp` prints 52
  (54 rows minus the two alias rows `moveaxis` and `row_stack`).
- **On failure**: §2.5. **Commit**: §2.6, `refactor(phase2a-11): split py_shape.cpp out of module.cpp`.

### Step 2a-12: `py_nn_functional.cpp`

- **Preconditions**: `git status --porcelain` empty;
  `grep -cF 'PyObject* py_linear(' cpp/mtorch/python/module.cpp` prints 1;
  `grep -cF 'PyObject* py_relu(' cpp/mtorch/python/module.cpp` prints 1.
- **Header**: none (nothing exported).
- **Extra includes for py_nn_functional.cpp** (this code calls `dropout_tensor` and the linalg
  dtype checks):

  ```cpp
  #include "mtorch/python/py_linalg.h"
  #include "mtorch/python/py_random.h"
  ```

- **Blocks** (both end `^}$`; extract both, delete B2 then B1, paste B1 then B2):
  - B1: `PyObject* py_linear(` → `PyObject* py_binary_cross_entropy_with_logits(`
    (hint L6778–L8081): `py_linear`, `py_conv1d`, `py_conv_transpose1d`, `py_conv2d`,
    `py_conv3d`, `py_conv_transpose2d`, `py_conv_transpose3d`, `py_max_pool1d`,
    `py_avg_pool1d`, `py_max_pool2d`, `py_avg_pool2d`, `py_unfold`, `py_fold`,
    `py_scaled_dot_product_attention`, `py_layer_norm`, `py_rms_norm`, `py_batch_norm`,
    `py_group_norm`, `py_embedding`, `py_dropout`, `py_mse_loss`, `py_l1_loss`, `py_nll_loss`,
    `py_cross_entropy`, `py_binary_cross_entropy`, `py_binary_cross_entropy_with_logits`,
    plus every indented helper between them (they move with the block automatically).
  - B2: `PyObject* py_relu(` → `PyObject* py_mish(` (hint L8447–L8715): `py_relu`,
    `py_leaky_relu`, `py_silu`, `py_elu`, `py_selu`, `py_softplus`, `py_hardtanh`, `py_relu6`,
    `py_hardsigmoid`, `py_hardswish`, `py_softsign`, `py_mish`.
- **Rows** (§2.3), `expected = 38`:

  ```python
  names = {"linear", "conv1d", "conv_transpose1d", "conv2d", "conv3d", "conv_transpose2d",
           "conv_transpose3d", "max_pool1d", "avg_pool1d", "max_pool2d", "avg_pool2d",
           "unfold", "fold", "scaled_dot_product_attention", "layer_norm", "rms_norm",
           "batch_norm", "group_norm", "embedding", "dropout", "mse_loss", "l1_loss",
           "nll_loss", "cross_entropy", "binary_cross_entropy",
           "binary_cross_entropy_with_logits", "relu", "leaky_relu", "silu", "elu", "selu",
           "softplus", "hardtanh", "relu6", "hardsigmoid", "hardswish", "softsign", "mish"}
  ```

  Table name: `kNnFunctionalModuleMethods`.
- **Registry/init**: `extern PyMethodDef kNnFunctionalModuleMethods[];` /
  `append_table(methods, kNnFunctionalModuleMethods);`.
- **module.cpp includes**: none added.
- **Verification**: §2.4; `grep -cF 'PyObject* py_' cpp/mtorch/python/py_nn_functional.cpp` prints 38.
- **On failure**: §2.5. **Commit**: §2.6, `refactor(phase2a-12): split py_nn_functional.cpp out of module.cpp`.

### Step 2a-13: `py_autograd.cpp`

- **Preconditions**: `git status --porcelain` empty;
  `grep -cF 'PyObject* py_clone(' cpp/mtorch/python/module.cpp` prints 1.
- **Header**: none (nothing exported; `dtype_is_signed` already went to py_common in 2a-2).
- **Extra includes for py_autograd.cpp**: none beyond the template.
- **Blocks** (end `^}$`): one block, `PyObject* py_clone(` → `PyObject* py_not_implemented(`
  (hint L8716–L9008). Functions (15): `py_clone`, `py_numel`, `py_is_tensor`,
  `py_mark_parameter`, `py_is_parameter`, `py_is_floating_point`, `py_is_complex`,
  `py_is_conj`, `py_is_signed`, `py_is_grad_enabled`, `py_set_grad_enabled`,
  `collect_autograd_nodes` (and the `GradSnapshot` struct), `grad_outputs_from_py`,
  `py_autograd_grad`, `py_not_implemented`. This step deliberately includes the module-level
  utilities `clone`/`numel`/`is_*` because they live in this contiguous range.
- **Rows** (§2.3), `expected = 14` (`save` and `load` both reference `py_not_implemented`):

  ```python
  names = {"clone", "numel", "is_tensor", "_mark_parameter", "_is_parameter",
           "is_floating_point", "is_complex", "is_conj", "is_signed", "_is_grad_enabled",
           "_set_grad_enabled", "_autograd_grad", "save", "load"}
  ```

  Table name: `kAutogradModuleMethods`.
- **Registry/init**: `extern PyMethodDef kAutogradModuleMethods[];` /
  `append_table(methods, kAutogradModuleMethods);`.
- **module.cpp includes**: none needed.
- **Verification**: §2.4; additionally `kLegacyModuleMethods` must now contain ONLY the sentinel:

  ```bash
  awk '/PyMethodDef kLegacyModuleMethods/{f=1} f{print} f&&/^};$/{exit}' cpp/mtorch/python/module.cpp
  ```

  must print exactly the header line, the sentinel row, and `};` (3 lines).
- **On failure**: §2.5. **Commit**: §2.6, `refactor(phase2a-13): split py_autograd.cpp out of module.cpp`.

## Step 2a-14: `py_indexing.{h,cpp}`

**Goal**: the index-key parsing helpers and the subscript protocol move to `py_indexing.{h,cpp}`.
The four functions referenced by `Tensor_slots` / `kLegacyTensorMethods` are exported.

**Preconditions**:

```bash
git status --porcelain                                                        # nothing
grep -cF 'mtorch::TensorIndex make_select_index(' cpp/mtorch/python/module.cpp  # 1
grep -cF 'PyObject* Tensor_subscript(' cpp/mtorch/python/module.cpp              # 1
```

**Actions**:

1. Create `cpp/mtorch/python/py_indexing.h`:

   ```cpp
   #pragma once
   #define PY_SSIZE_T_CLEAN
   #include <Python.h>

   #include "mtorch/python/py_common.h"

   namespace mtorch::py {

   // Referenced by Tensor_slots (Py_mp_subscript / Py_mp_ass_subscript) and by the
   // __getitem__ / __setitem__ rows of kLegacyTensorMethods in py_tensor_type.cpp.
   PyObject* Tensor_subscript(PyObject* self, PyObject* key);
   int Tensor_ass_subscript(PyObject* self, PyObject* key, PyObject* value);
   PyObject* Tensor_getitem_method(PyTensor* self, PyObject* args);
   PyObject* Tensor_setitem_method(PyTensor* self, PyObject* args);

   }  // namespace mtorch::py
   ```

2. Create `cpp/mtorch/python/py_indexing.cpp`: §2.2 template plus
   `#include "mtorch/python/py_indexing.h"` after the `py_common.h` include.

3. Cut two blocks (both end `^}$`; extract both, delete B2 then B1, paste B1 then B2):
   - B1: `mtorch::TensorIndex make_select_index(` → `TensorPtr full_row_slice_int_columns_key(`
     (hint L1457–L1887). Functions (17): `make_select_index`, `make_slice_index`,
     `make_new_axis_index`, `single_bool_mask_key`, `bool_mask_from_key`,
     `bool_mask_from_candidate`, `parse_int_index_data`, `int_tensor_from_candidate`,
     `int_tensor_from_key`, `int_tensor_tuple_from_key`, `is_basic_integer_select_candidate`,
     `parse_tensor_indices`, `MixedAdvancedKey` + `DimIntAdvancedKey` (structs),
     `parse_tail_indices`, `parse_mixed_advanced_key`, `parse_dim_int_advanced_key`,
     `full_row_slice_int_columns_key`.
   - B2: `PyObject* Tensor_subscript(` → `PyObject* Tensor_setitem_method(`
     (hint L10290–L10445). Functions (5): `Tensor_subscript`, `assign_tensor_subscript`,
     `Tensor_ass_subscript`, `Tensor_getitem_method`, `Tensor_setitem_method`.

4. In `module.cpp`, add `#include "mtorch/python/py_indexing.h"` below the other section
   includes. The `Tensor_slots` rows for `Py_mp_subscript` / `Py_mp_ass_subscript` and the
   `__getitem__` / `__setitem__` rows of `kLegacyTensorMethods` need no edits — the names now
   resolve through the header plus the file-scope `using namespace mtorch::py;`.

5. No method-table rows move and there are no registry/module_init changes in this step.

**Verification**: §2.4 block. Sanity: run
`python3 -c "import mtorch; t = mtorch.tensor([[1.0, 2.0], [3.0, 4.0]]); t[0, 1] = 9; print(t[0].tolist())"`
— must print `[1.0, 9.0]`.

**On failure**: §2.5.

**Commit**: §2.6, message `refactor(phase2a-14): split py_indexing.{h,cpp} out of module.cpp`.

## Step 2a-15: `py_tensor_type.cpp` (rename + cleanup)

**Goal**: what is left of `module.cpp` is exactly the Tensor type: `Tensor_dealloc` …
`Tensor_richcompare`, `Tensor_getset`, `kLegacyTensorMethods`, `Tensor_slots`, `Tensor_spec`,
`make_tensor_type`, and the empty `kLegacyModuleMethods`. Rename the file and retire the legacy
module table.

**Preconditions**:

```bash
git status --porcelain    # nothing
awk '/PyMethodDef kLegacyModuleMethods/{f=1} f{print} f&&/^};$/{exit}' cpp/mtorch/python/module.cpp
# must print exactly:
# PyMethodDef kLegacyModuleMethods[] = {
#     {nullptr, nullptr, 0, nullptr},
# };
grep -cF 'PyObject* py_' cpp/mtorch/python/module.cpp   # must print 0
```

**Actions**:

1. `git mv cpp/mtorch/python/module.cpp cpp/mtorch/python/py_tensor_type.cpp`
2. In `py_tensor_type.cpp`, delete the 3-line `kLegacyModuleMethods` definition shown above.
3. In `py_tensor_type.cpp`, rename `kLegacyTensorMethods` → `kTensorMethods` (2 occurrences:
   the definition and the `Tensor_slots` reference; find them with
   `grep -n 'kLegacyTensorMethods' cpp/mtorch/python/py_tensor_type.cpp`).
4. In `cpp/mtorch/python/py_registry.h`: delete the line
   `extern PyMethodDef kLegacyModuleMethods[];` and rename
   `extern PyMethodDef kLegacyTensorMethods[];` → `extern PyMethodDef kTensorMethods[];`.
5. In `cpp/mtorch/python/module_init.cpp`: delete the line
   `append_table(methods, kLegacyModuleMethods);` and change
   `append_table(methods, kLegacyTensorMethods);` → `append_table(methods, kTensorMethods);`.
6. Confirm nothing legacy remains: `grep -rn kLegacy cpp/` must print nothing.
7. If a stale object file confuses the build, remove it:
   `rm -f build/temp.*/cpp/mtorch/python/module.o`.

**Verification**: §2.4 block, plus the target-structure check:

```bash
ls cpp/mtorch/python/
# expected files:
# module_init.cpp  py_autograd.cpp  py_common.cpp  py_common.h  py_creation.cpp
# py_index_select.cpp  py_index_select.h  py_indexing.cpp  py_indexing.h  py_linalg.cpp
# py_linalg.h  py_nn_functional.cpp  py_pointwise.cpp  py_pointwise.h  py_random.cpp
# py_random.h  py_reduction.cpp  py_reduction.h  py_registry.h  py_shape.cpp
# py_sort_search.cpp  py_sort_search.h  py_tensor_type.cpp
wc -l cpp/mtorch/python/*.cpp        # every file well under 5,000 lines
find build -type f -name '*.o' | sort   # one .o per .cpp listed above plus core/tensor.o
```

**On failure**: §2.5 (note: `git restore --staged --worktree .` also undoes the `git mv`).

**Commit**: §2.6, message `refactor(phase2a-15): rename module.cpp to py_tensor_type.cpp and drop legacy table`.

## Step 2a-16: Phase completion check (full benchmark comparison)

**Goal**: prove the split is performance-neutral against the Phase 0 baseline.

**Preconditions**: 2a-1 … 2a-15 all checked off in PROGRESS.md;
`ls docs/design/baseline/benchmark-baseline.json tools/compare_benchmarks.py` shows both files.

**Actions**:

```bash
mkdir -p benchmark-results
python3 setup.py build_ext --inplace
pytest tests/compat -q | tail -3
pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
    --compat-benchmark-json benchmark-results/after-phase2a.json
python3 tools/compare_benchmarks.py \
    docs/design/baseline/benchmark-baseline.json benchmark-results/after-phase2a.json
```

**Verification**: the compare script reports no `REGRESSION` lines (the binding split does not
touch operator kernels, so results should be equivalent to the baseline), and the pytest summary
matches `docs/design/baseline/tests-baseline.txt`.

**On failure**: if a `REGRESSION` appears, follow the reproduction-check procedure in
`01-rules-and-verification.md` §5.2 (re-run the benchmark; only a reproducible regression
counts). If reproducible, mark 2a-16 **BLOCKED** in PROGRESS.md per §2.5 (do NOT reset the
phase's commits — the code is functionally green; the block is for performance investigation).

**Commit**: §2.6, message `refactor(phase2a-16): record post-split benchmark results`
(commit `benchmark-results/after-phase2a.json` too).

## 3. Pitfall list

| Symptom | Cause and fix |
|---|---|
| Link error: duplicate symbol | A function definition sits in a header (without `inline`), or a struct was left in both py_common.h and py_common.cpp, or a global variable is defined in a header. Definitions go in one .cpp; headers hold declarations only (structs/enums live in the header ONLY). |
| Link error: undefined symbol | A file references a symbol still inside module.cpp's anonymous namespace, or a section header declaration was not added. Check the step's export list; the symbol must be defined inside `namespace mtorch::py` in exactly one .cpp and declared in that section's header. |
| Compile error: default argument given twice | A default was left on the definition after being added to the header declaration. Delete it from the definition (see 2a-2 action 6, 2a-4 action 4). |
| Builds but SystemError / missing method at import | Forgot the table sentinel, or forgot the `append_table` line in `build_module_methods()`, or forgot the `extern` in py_registry.h. Compare `python3 -c "from mtorch import _C; print(len([n for n in dir(_C) if not n.startswith('__')]))"` with METHOD_COUNT from 2a-1. |
| Tensor method raises AttributeError | The `Py_tp_methods` slot was not patched before `PyType_FromSpec` — `make_tensor_type()` must overwrite the slot with `build_tensor_methods()` (2a-3 action 7), and the vector must never grow afterward. |
| Warning at `PyArg_ParseTupleAndKeywords` | The const-ness of `keywords` arrays differs across compilers. Do not change how the original code was written. |
| `error: use of undeclared identifier 'MemoryFormat'` (or similar) in module.cpp | The `using namespace mtorch::py;` line from 2a-2 action 7 is missing or placed after `namespace {` instead of before it. |
| Row count assertion fails in the ROW MOVER script | A row was already moved in an earlier step (re-check PROGRESS.md), or the step's deletion of blocks accidentally removed table rows. `git diff` module.cpp and compare against the step's block boundaries. |
