# 05. Phase 2b: Splitting `tensor.cpp` (the C++ core)

Prerequisite: Phase 2a complete (table concatenation and the glob build are working).
`01-rules-and-verification.md` and `04-phase2a-module-cpp-split.md` §0 (C++ splitting fundamentals) read.

Objective: Split `cpp/mtorch/core/tensor.cpp` (32,762 lines) into internal headers under
`cpp/mtorch/core/detail/` and 14 implementation files organized by responsibility.
**The public header `tensor.h` (593 lines) is not changed** (the only exception in this phase is
confirming that it is "not changed").

**This phase is purely mechanical moves.** Turning ops into a table and integrating fast paths are done
in Phase 3. The only rewrites allowed are linkage adjustments (anonymous namespace → `mtorch::detail`),
adding `#include` / `using` / declarations, and the two keyword edits explicitly listed in 2b-1e
(`static` → `inline`) — nothing else. Every commit in this phase must pass the §5.3 move check of
`01-rules-and-verification.md`.

## 0. Fundamentals specific to this phase

### 0.1 Verified current structure of tensor.cpp

All line numbers in this document are **hints valid for the original 32,762-line file** (the state at
the start of Phase 2b). Lines shift after every step. **Never trust a line hint; always re-run the
given `grep` and use the line it prints.**

Namespace map (re-verify with `grep -n 'namespace' cpp/mtorch/core/tensor.cpp`):

| Lines (original) | Content |
|---|---|
| 1–23 | includes; `#if defined(__ARM_NEON)` / `#if defined(MTORCH_USE_ACCELERATE)` guarded includes at 17–23 |
| 25 | `namespace mtorch {` |
| 26–7989 | the big anonymous namespace (helper layer, ~7,960 lines) — split into `detail/` by step 2b-1 |
| 7991–32761 | public API implementations (the ~250 functions declared in `tensor.h`) plus namespace-scope helpers, with four small extra anonymous namespaces at 9206–9245 (pad), 10054–10481 (resample), 25899–26062 (conv), 30297–30928 (attention) |
| 32762 | `}  // namespace mtorch` |

Key verified anchors (re-check each with the exact grep shown):

- `grep -n 'void half_bits_to_float_array' cpp/mtorch/core/tensor.cpp` → currently ~L346
- `grep -n 'bool run_unary_float32_kernel' cpp/mtorch/core/tensor.cpp` → currently ~L4273
- `grep -n 'bool run_unary_float16_kernel' cpp/mtorch/core/tensor.cpp` → currently ~L4460
- `grep -n 'try_binary_sub_float32_fast_path' cpp/mtorch/core/tensor.cpp` → definition ~L5654
- `grep -n 'try_binary_mul_float32_fast_path' cpp/mtorch/core/tensor.cpp` → definition ~L5714
- `grep -n 'run_coalesced_unary_double_kernel' cpp/mtorch/core/tensor.cpp` → definition ~L3643; the
  double-fallback call inside `unary()` is ~L14390; the `unary()` backward lambda is ~L14492

### 0.2 Linkage rules (recap of 04 §0, applied to this phase)

- Symbols inside the anonymous namespace have internal linkage. Every symbol promoted to
  `namespace mtorch::detail` and declared/defined in a `detail/*.h` (+ optional `.cpp`) becomes usable
  from all files.
- **Chosen convention (use everywhere, no exceptions): add `using namespace detail;` once,
  immediately after `namespace mtorch {`, in tensor.cpp (done in step 2b-1b) and in every new
  section .cpp (part of the template below). Never rewrite call sites to qualified `detail::` calls.**
- Qualified calls such as `::mtorch::mark_storage_modified(*storage)` (present in
  `Tensor::mark_storage_modified`, ~L8342) keep compiling after promotion: qualified lookup in
  `mtorch` follows the `using namespace detail;` directive. Do not edit such lines.
- Function templates always live in the header, definition and all.
- **Inline rule**: every non-template function definition pasted into a `detail/*.h` gets `inline `
  inserted at the start of its return type, unless it already has `inline`. Pure declarations (lines
  ending in `;` with no body) are left untouched.
- **Declaration rule**: for every function whose definition goes into a `detail/*.cpp`, add a
  declaration to the matching `detail/*.h`: copy the definition's signature from the return type up
  to and including the matching `)`, then write `;` instead of the `{`.
- **Default-argument rule**: if a signature contains `= <value>` defaults and the definition goes to
  a `.cpp`, keep the defaults in the header declaration and delete the `= <value>` parts from the
  `.cpp` definition. (In this phase the only affected function is `can_accelerate_vector`, and it is
  placed header-inline precisely to avoid this; the `make_*_tensor` factory functions with defaults
  are also header-inline. So in practice this rule should never fire — if it does, follow it.)
- All preprocessor lines (`#if` / `#else` / `#endif`, including `MTORCH_USE_ACCELERATE`,
  `__ARM_NEON`, `__clang__` pragmas) move verbatim with the code they wrap.

### 0.3 Build, tests, glob

- Build (run from repo root, always): `python3 setup.py build_ext --inplace`
- Compile flags (already in setup.py, do not touch until 2b-17 remediation): `-std=c++20 -O3`,
  plus `-DMTORCH_USE_ACCELERATE` and `-framework Accelerate` on darwin. `include_dirs=["cpp"]`,
  therefore every new include is written as `#include "mtorch/core/detail/xxx.h"`.
- Step 2a-1 replaced `sources=[...]` in setup.py with
  `sorted(glob.glob("cpp/mtorch/core/*.cpp") + glob.glob("cpp/mtorch/core/detail/*.cpp") + glob.glob("cpp/mtorch/python/*.cpp"))`.
  **This glob covers both `cpp/mtorch/core/*.cpp` (all new section files, e.g. `losses.cpp`) and
  `cpp/mtorch/core/detail/*.cpp` (all new detail files). No setup.py edit is needed anywhere in
  Phase 2b except the optional `-flto` remediation in 2b-17.** Precondition check (must print a line):

  ```bash
  grep -Fn 'glob.glob("cpp/mtorch/core/detail/*.cpp")' setup.py
  ```

- Tests: `pytest tests/compat`. Baseline artifacts (created in Phase 0, see
  `02-phase0-baseline.md`): `docs/design/baseline/tests-baseline.txt`,
  `docs/design/baseline/collect-count.txt`, `docs/design/baseline/benchmark-baseline.json`.
  Comparison script: `tools/compare_benchmarks.py` (5% median-ratio gate, threshold 1.05).

### 0.4 Standard verification for every step (referenced below as "STANDARD VERIFY")

```bash
python3 setup.py build_ext --inplace
pytest tests/compat -q | tail -3
tail -1 docs/design/baseline/tests-baseline.txt
```

PASS if and only if the build exits 0 AND every count (passed / failed / xfailed / skipped / error)
in the pytest summary line equals the corresponding count in the baseline line. Any difference is a
failure. (This is the quick form of `01-rules-and-verification.md` §5.1; if the two lines are hard to
compare, run the full §5.1 procedure — it is the authoritative one.)

### 0.5 Standard failure procedure for every step (referenced below as "STANDARD FAIL")

1. Discard the step completely (`<new files>` = the exact files the step created):

   ```bash
   git restore --staged --worktree .
   rm -f <new files>
   python3 setup.py build_ext --inplace
   pytest tests/compat -q | tail -3
   ```

   Confirm the tree is green again (matches baseline).
2. In `docs/design/PROGRESS.md`, append ` **BLOCKED**` to the current step's line and write up to
   3 lines describing the failure in the Phase 2b "Notes:" field. Commit only PROGRESS.md:
   `git add docs/design/PROGRESS.md && git commit -m "refactor(phase2b): mark <step-id> BLOCKED"`.
3. Stop. Do not start the next step.

### 0.6 Standard include block (referenced below as "STD-INCLUDES")

Every new detail header and every new .cpp file starts with exactly this block (it is the include set
tensor.cpp compiles with today, so no body can be missing a header):

```cpp
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "mtorch/core/tensor.h"
#include "mtorch/core/detail/platform.h"
```

### 0.7 Detail header template (referenced below as "DETAIL-HEADER template")

```cpp
#pragma once

// STD-INCLUDES block here (§0.6)
// plus #include lines for previously created detail headers, as listed per step

namespace mtorch::detail {

// pasted declarations / inline definitions go here, in the original tensor.cpp order

}  // namespace mtorch::detail
```

The matching `detail/xxx.cpp` (when the step has one):

```cpp
// STD-INCLUDES block here (§0.6)
#include "mtorch/core/detail/xxx.h"
// plus #include lines for previously created detail headers, as listed per step

namespace mtorch::detail {

// pasted function definitions go here, in the original tensor.cpp order

}  // namespace mtorch::detail
```

### 0.8 Section file template (referenced below as "SECTION-FILE template", used by 2b-2 … 2b-15)

Every new `cpp/mtorch/core/<name>.cpp` starts with exactly this (the nine detail includes are always
all present — do not try to minimize them):

```cpp
// STD-INCLUDES block here (§0.6)
#include "mtorch/core/detail/common.h"
#include "mtorch/core/detail/half.h"
#include "mtorch/core/detail/storage.h"
#include "mtorch/core/detail/accelerate.h"
#include "mtorch/core/detail/broadcast.h"
#include "mtorch/core/detail/promotion.h"
#include "mtorch/core/detail/factory.h"
#include "mtorch/core/detail/elementwise.h"

namespace mtorch {
using namespace detail;   // avoids rewriting calls in the existing code

// pasted code goes here, byte-for-byte, in the original tensor.cpp order.
// Blocks that were inside an anonymous namespace in tensor.cpp keep (or regain)
// their `namespace { ... }  // namespace` wrapper; namespace-scope helpers stay
// at namespace scope.

}  // namespace mtorch
```

### 0.9 Two compile/link error rules for section steps (2b-2 … 2b-15)

- **`use of undeclared identifier 'X'` while compiling a new section file**: `X` is still defined in
  tensor.cpp.
  - If `X` is inside an anonymous namespace of tensor.cpp: it belongs to a block scheduled for a
    later step. Move it NOW into the current file's anonymous namespace instead, and write one note
    line in PROGRESS.md ("moved X early into <file>").
  - If `X` is at namespace scope in tensor.cpp (not in `tensor.h`, not anonymous): do not move it.
    Add a matching declaration (signature + `;`) inside `namespace mtorch {` near the top of the new
    file, and write one note line in PROGRESS.md.
- **`undefined symbol` at link time**: an anonymous-namespace helper is referenced from another file.
  Promote it: move its definition into the matching `detail/*.h` (inline rule) or `detail/*.{h,cpp}`
  (declaration rule), whichever detail file the §1 table assigns its topic to. This is the entire
  decision algorithm; there is no other fix.

## 1. Target file structure

Under `cpp/mtorch/core/` (create the subdirectory once, in step 2b-1a:
`mkdir -p cpp/mtorch/core/detail`):

```
tensor.h                  (do not change)
detail/platform.h         aggregation of platform-dependent includes (2b-1a)
detail/common.{h,cpp}     small items: dim normalization, pooling output size, require_cpu_device,
                          storage version counter; .cpp holds only the grad_enabled thread_local (2b-1b)
detail/half.h             fp16⇔fp32 conversion + half caches (header-only, inline, per-NEON #if) (2b-1c)
detail/storage.h          load/store templates, storage_get/set, typed_data_at (header-only, inline) (2b-1d)
detail/accelerate.{h,cpp} vDSP/BLAS wrappers (the MTORCH_USE_ACCELERATE #if stays confined here) (2b-1e)
detail/broadcast.{h,cpp}  broadcast index computation, reduce_grad_to_shape (2b-1f)
detail/promotion.h        apply_binary, op-kind decision, dtype promotion rules (header-only) (2b-1g)
detail/factory.{h,cpp}    make_empty_contiguous_tensor, strided/dense copy+cast, reduce fast path (2b-1h)
detail/elementwise.{h,cpp} execution plan, coalesce, run_*_kernel, binary fast-path group (2b-1i)
tensor_core.cpp           dtype/device/grad-mode functions, Storage/Tensor members, factories, to() (2b-2)
losses.cpp (2b-3) / activations.cpp (2b-4) / normalization.cpp (2b-5) / sorting.cpp (2b-6) /
pooling.cpp (2b-7) / attention.cpp (2b-8) / linalg.cpp (2b-9) / reductions.cpp (2b-10) /
indexing.cpp (2b-11) / resample.cpp (2b-12) / conv.cpp (2b-13) / views.cpp (2b-14) /
elementwise_ops.cpp (2b-15)
```

All new files are picked up automatically by the setup.py glob (§0.3).

## Step 2b-1: Promoting the detail layer (one commit per sub-step)

This moves the big anonymous namespace (L26–7989) into `detail/`, ordered least-dependent first.
Three blocks are **deliberately left behind** in the anonymous namespace of tensor.cpp because only
the indexing family uses them; they move in step 2b-11:

- nonzero counters, original L683–840 (`linear_to_index` forward declaration through
  `try_fill_nonzero_contiguous`) — the forward declaration at L683 itself is deleted in 2b-1f
- mask/int-index tail helpers, original L1663–1762 (`tail_shape_after_mask` through
  `tail_view_for_prefix`)
- index fast paths, original L3134–3437 (`broadcast_storage_offset_1d` through
  `try_index_put_int_tensor_dim_1d_fast_path`)

If you are unsure about any helper not listed in a sub-step: leave it in tensor.cpp. Promote it only
when a later step's build fails with `undefined symbol` (§0.9).

---

### Step 2b-1a: `detail/platform.h`

**Goal**: All platform-conditional includes live in one header.

**Preconditions**: Phase 2a complete;
`grep -Fn 'glob.glob("cpp/mtorch/core/detail/*.cpp")' setup.py` prints a line;
`git status --porcelain` is empty.

**Actions**:

1. `mkdir -p cpp/mtorch/core/detail`
2. Create `cpp/mtorch/core/detail/platform.h` with exactly this content:

   ```cpp
   #pragma once

   #if defined(__ARM_NEON)
   #include <arm_neon.h>
   #endif

   #if defined(MTORCH_USE_ACCELERATE)
   #include <Accelerate/Accelerate.h>
   #endif
   ```

3. In tensor.cpp, locate the two guarded include blocks:
   `grep -n 'arm_neon.h\|Accelerate/Accelerate.h' cpp/mtorch/core/tensor.cpp` → currently ~L18 and
   ~L22. Delete the two 3-line blocks (`#if defined(__ARM_NEON)` … `#endif` and
   `#if defined(MTORCH_USE_ACCELERATE)` … `#endif`, original L17–23) and insert in their place the
   single line:

   ```cpp
   #include "mtorch/core/detail/platform.h"
   ```

**Verification**: STANDARD VERIFY (§0.4).

**On failure**: STANDARD FAIL (§0.5) with `<new files>` = `cpp/mtorch/core/detail/platform.h`.

**Commit**:

```bash
git add cpp/mtorch/core/detail/platform.h cpp/mtorch/core/tensor.cpp
git commit -m "refactor(phase2b-1a): extract detail/platform.h"
```

Then run the §5.3 move check of `01-rules-and-verification.md` and update the 2b-1a line in
`docs/design/PROGRESS.md` (both apply to every commit below; they are not repeated).

---

### Step 2b-1b: `detail/common.{h,cpp}` — **fully worked example for all detail promotions**

**Goal**: The small generic helpers (original L28–275) live in `mtorch::detail`, and tensor.cpp gains
its single `using namespace detail;`.

**Preconditions**: 2b-1a committed. `ls cpp/mtorch/core/detail/platform.h` succeeds.

**Actions**:

1. Locate the block. Start:
   `grep -n 'thread_local bool grad_enabled' cpp/mtorch/core/tensor.cpp` → currently ~L28. End:
   `grep -n 'uint16_t float_to_half_bits' cpp/mtorch/core/tensor.cpp` → currently ~L276; the block
   ends on the line before that (the closing `}` of the second `ensure_same_device` overload).
   The block contains exactly these 26 symbols (verify each name exists with
   `grep -n '<name>' cpp/mtorch/core/tensor.cpp` before moving):

   `grad_enabled` (thread_local variable, ~L28), `track_grad` (~L30), `positive_dim` (~L34),
   `product` (~L44), `is_channels_last_contiguous_4d` (~L51), `is_channels_last_compatible_4d`
   (~L62), `is_channels_last_contiguous_5d` (~L77), `channels_last_strides_4d` (~L90),
   `channels_last_strides_5d` (~L100), `floor_divide_int` (~L111), `pair_from_pool_arg` (~L121),
   `single_from_pool_arg` (~L131), `stride_from_pool_arg_1d` (~L138), `stride_from_pool_arg`
   (~L145), `require_positive_value` (~L154), `require_nonnegative_value` (~L160),
   `require_positive_pair` (~L166), `require_nonnegative_pair` (~L172), `pooling_output_size`
   (~L178), `strided_required_storage_size` (~L200), `require_cpu_device` (~L231),
   `require_cpu_storage` (~L237), `mark_storage_modified` (~L241), `mark_storage_modified_if_cached`
   (~L259), `ensure_same_device` (2 overloads, ~L265 and ~L271).

2. Create `cpp/mtorch/core/detail/common.h`:

   ```cpp
   #pragma once

   #include <algorithm>
   #include <array>
   #include <bit>
   #include <cmath>
   #include <cstring>
   #include <limits>
   #include <numeric>
   #include <optional>
   #include <stdexcept>
   #include <type_traits>
   #include <unordered_map>
   #include <unordered_set>
   #include <utility>

   #include "mtorch/core/tensor.h"
   #include "mtorch/core/detail/platform.h"

   namespace mtorch::detail {

   extern thread_local bool grad_enabled;

   inline bool track_grad(bool requires_grad) {
     return grad_enabled && requires_grad;
   }

   // ... paste the remaining 24 function definitions here, byte-for-byte, in the
   // original tensor.cpp order, applying the inline rule (§0.2) to each:
   // e.g. `int64_t positive_dim(...)` becomes `inline int64_t positive_dim(...)`.

   }  // namespace mtorch::detail
   ```

   The only definition that does NOT go in the header is the `grad_enabled` variable itself (a
   thread_local defined in a header would be duplicated per TU — see the pitfall list).

3. Create `cpp/mtorch/core/detail/common.cpp` with exactly this content:

   ```cpp
   #include "mtorch/core/detail/common.h"

   namespace mtorch::detail {

   thread_local bool grad_enabled = true;

   }  // namespace mtorch::detail
   ```

4. In tensor.cpp:
   - Delete the moved block (everything from the `thread_local bool grad_enabled = true;` line
     through the closing `}` of the second `ensure_same_device`, inclusive). Do NOT delete the
     `namespace {` line above it.
   - Add `#include "mtorch/core/detail/common.h"` immediately after the
     `#include "mtorch/core/detail/platform.h"` line.
   - Immediately after the `namespace mtorch {` line (currently L25), add a new line:
     `using namespace detail;`
     (This is added ONCE, in this step. Later steps must not add it again.)

**Verification**: STANDARD VERIFY. Additionally
`grep -c 'using namespace detail;' cpp/mtorch/core/tensor.cpp` must print `1`.

**On failure**: STANDARD FAIL, `<new files>` = `cpp/mtorch/core/detail/common.h cpp/mtorch/core/detail/common.cpp`.

**Commit**:

```bash
git add cpp/mtorch/core/detail/common.h cpp/mtorch/core/detail/common.cpp cpp/mtorch/core/tensor.cpp
git commit -m "refactor(phase2b-1b): extract detail/common"
```

**Every following sub-step (2b-1c … 2b-1i) repeats this exact mechanic**: locate block by grep →
create header (DETAIL-HEADER template, §0.7) → paste, applying the inline/declaration/default-arg
rules (§0.2) → delete the block from tensor.cpp → add one `#include` line after the previous detail
include → STANDARD VERIFY → commit. Only the specifics are listed below.

---

### Step 2b-1c: `detail/half.h` (header-only)

**Goal**: fp16⇔fp32 conversion and the half caches, all `inline` in the header (the NEON `#if`
branches move verbatim).

**Preconditions**: 2b-1b committed. `ls cpp/mtorch/core/detail/common.h` succeeds.

**Actions**:

1. Block: from `grep -n 'uint16_t float_to_half_bits' cpp/mtorch/core/tensor.cpp` (~L276) through
   the `#endif` that closes `horizontal_sum_f32x4`
   (`grep -n 'float horizontal_sum_f32x4' cpp/mtorch/core/tensor.cpp` → ~L540; its
   `#if defined(__ARM_NEON) && defined(__aarch64__)` opener at ~L539 and `#endif` at ~L545 move
   with it). Exactly 12 symbols:

   `float_to_half_bits` (~L276), `half_bits_to_float` (~L317), `half_bits_to_float_array` (~L346),
   `half_bits_to_neg_float_array` (~L370), `float_array_to_half_bits` (~L394),
   `cached_contiguous_half_as_float` (~L416), `is_supported_half_conv_weight_4d` (~L435),
   `copy_half_4d_to_contiguous_float` (~L440), `copy_half_4d_to_contiguous_float_or_fast` (~L457),
   `pack_conv2d_nhwc_half_weight` (~L467), `cached_conv2d_nhwc_packed_half_weight` (~L494),
   `horizontal_sum_f32x4` (~L540, keep its NEON guard).

2. Create `cpp/mtorch/core/detail/half.h` per the DETAIL-HEADER template, with one extra include
   after the STD-INCLUDES block: `#include "mtorch/core/detail/common.h"`
   (needed by `is_supported_half_conv_weight_4d`, which calls `is_channels_last_contiguous_4d`).
   Paste all 12 definitions, applying the inline rule. There is no `half.cpp`.
3. tensor.cpp: delete the block; add `#include "mtorch/core/detail/half.h"` after the common.h
   include.

**Verification**: STANDARD VERIFY. Additionally
`grep -c 'uint16_t float_to_half_bits' cpp/mtorch/core/tensor.cpp` must print `0`.

**On failure**: STANDARD FAIL, `<new files>` = `cpp/mtorch/core/detail/half.h`.

**Commit**:

```bash
git add cpp/mtorch/core/detail/half.h cpp/mtorch/core/tensor.cpp
git commit -m "refactor(phase2b-1c): extract detail/half.h"
```

---

### Step 2b-1d: `detail/storage.h` (header-only)

**Goal**: The typed load/store templates and raw-pointer accessors, all in the header (templates +
hot path).

**Preconditions**: 2b-1c committed. `ls cpp/mtorch/core/detail/half.h` succeeds.

**Actions**:

1. Block: from the `template <typename T>` line above
   `grep -n 'T load_value' cpp/mtorch/core/tensor.cpp` (~L548) through the closing `}` of
   `copy_channels_last_5d_half_to_float_contiguous_ncdhw`
   (`grep -n 'copy_channels_last_5d_half_to_float_contiguous_ncdhw' cpp/mtorch/core/tensor.cpp`
   → definition ~L656; it ends on the line before the
   `std::vector<int64_t> linear_to_index(int64_t linear, const std::vector<int64_t>& sizes);`
   forward declaration, ~L683). **Do NOT move or delete that forward-declaration line in this step**
   (2b-1f deletes it). Exactly 11 symbols:

   `load_value` (template, ~L548), `store_value` (template, ~L555), `storage_get_unchecked`
   (~L559), `storage_get_integral_unchecked` (~L577), `storage_set_unchecked` (~L593),
   `storage_set_integral_unchecked` (~L616), `float_data_at` (~L634), `mutable_float_data_at`
   (~L639), `typed_data_at` (template, ~L645), `mutable_typed_data_at` (template, ~L651),
   `copy_channels_last_5d_half_to_float_contiguous_ncdhw` (~L656).

2. Create `cpp/mtorch/core/detail/storage.h` per the DETAIL-HEADER template, with one extra include:
   `#include "mtorch/core/detail/half.h"` (needed by `storage_get_unchecked` /
   `storage_set_unchecked`, which call `half_bits_to_float` / `float_to_half_bits`). Templates move
   verbatim; the non-template functions get the inline rule. No `storage.cpp`.
3. tensor.cpp: delete the block; add `#include "mtorch/core/detail/storage.h"`.

**Verification**: STANDARD VERIFY. Additionally
`grep -c 'double storage_get_unchecked' cpp/mtorch/core/tensor.cpp` must print `0`.

**On failure**: STANDARD FAIL, `<new files>` = `cpp/mtorch/core/detail/storage.h`.

**Commit**:

```bash
git add cpp/mtorch/core/detail/storage.h cpp/mtorch/core/tensor.cpp
git commit -m "refactor(phase2b-1d): extract detail/storage.h"
```

---

### Step 2b-1e: `detail/accelerate.{h,cpp}`

**Goal**: All BLAS/vDSP wrappers in one pair of files. The `MTORCH_USE_ACCELERATE` guard structure
is preserved exactly (group 1 below only exists when Accelerate is enabled, mirroring today's code;
the header stays includable when it is disabled because the guarded region is simply empty then).

**Preconditions**: 2b-1d committed. `ls cpp/mtorch/core/detail/storage.h` succeeds.

**Actions**:

1. Two blocks move.
   **Group 1** — the whole `#if defined(MTORCH_USE_ACCELERATE)` region:
   `grep -n '#if defined(MTORCH_USE_ACCELERATE)' cpp/mtorch/core/tensor.cpp` → the first hit inside
   the anonymous namespace, currently ~L842; its matching `#endif` is currently ~L1292 (the line
   before `std::vector<int64_t> linear_to_index(...) {` at ~L1294). 17 symbols:

   `accelerate_int_ok` (~L843), `accelerate_stride_ok` (~L847), `accelerate_scopy` (~L851),
   `accelerate_saxpy` (~L870), `can_accelerate_vector` (~L890, has default arg `target_stride = 1`),
   `try_accelerate_copy_float32_1d` (~L894), `try_accelerate_copy_float32_2d` (~L908),
   `try_accelerate_contiguous_copy_float32` (~L927), `try_accelerate_axpy_float32_same_shape`
   (~L946), `try_accelerate_sum_float32` (~L988), `sum_float32_neon` (~L1029, already `inline`,
   wrapped in its own `#if defined(__ARM_NEON)`), `sum_float16_as_float_contiguous` (~L1055,
   currently `static float ...`), `try_accelerate_cumsum_float32_2d` (~L1088),
   `try_accelerate_fill_float32` (~L1138), `try_accelerate_add_scalar_float32` (~L1170),
   `try_accelerate_mul_scalar_float32` (~L1197), `try_accelerate_dense_cast` (~L1224).

   **Group 2** — the fp16 GEMM wrappers (defined unconditionally, with `#if`/`#else` inside their
   bodies): `try_bmm_float16_accelerate`
   (`grep -n 'try_bmm_float16_accelerate' cpp/mtorch/core/tensor.cpp` → ~L1529) and
   `try_baddbmm_float16_accelerate` (~L1589), i.e. original L1529 through the closing `}` at ~L1660
   (the line before `std::vector<int64_t> tail_shape_after_mask(...)` at ~L1663).

2. Create `cpp/mtorch/core/detail/accelerate.h` per the DETAIL-HEADER template, extra includes:
   `common.h`, `half.h`, `storage.h`. Content layout:

   ```cpp
   namespace mtorch::detail {

   #if defined(MTORCH_USE_ACCELERATE)
   // header-inline definitions (inline rule):
   //   accelerate_int_ok, accelerate_stride_ok, can_accelerate_vector (keep its default arg),
   //   sum_float32_neon (with its inner #if defined(__ARM_NEON) wrapper),
   //   sum_float16_as_float_contiguous  <-- EDIT: change `static float` to `inline float`.
   //       This is one of the two allowed keyword edits in this phase.
   // declarations (declaration rule) for the 12 remaining group-1 functions defined in the .cpp
   #endif

   // unconditional declarations:
   bool try_bmm_float16_accelerate(const TensorPtr& left, const TensorPtr& right, Tensor& result);
   bool try_baddbmm_float16_accelerate(/* copy exact signature from tensor.cpp */);

   }  // namespace mtorch::detail
   ```

3. Create `cpp/mtorch/core/detail/accelerate.cpp`: STD-INCLUDES +
   `#include "mtorch/core/detail/accelerate.h"` + includes of `common.h`, `half.h`, `storage.h`;
   inside `namespace mtorch::detail { ... }` paste, in order:
   `#if defined(MTORCH_USE_ACCELERATE)` + the 12 group-1 function definitions that were not made
   header-inline (in original order, all `#pragma clang` lines verbatim) + `#endif`, then the two
   group-2 definitions verbatim (their internal `#if`/`#else` untouched).
4. tensor.cpp: delete both blocks; add `#include "mtorch/core/detail/accelerate.h"`.

**Verification**: STANDARD VERIFY. Additionally
`grep -c 'void accelerate_scopy\|bool try_bmm_float16_accelerate(' cpp/mtorch/core/tensor.cpp` must
print `0` (no definitions left; remaining call sites are expected and fine).

**On failure**: STANDARD FAIL, `<new files>` = `cpp/mtorch/core/detail/accelerate.h cpp/mtorch/core/detail/accelerate.cpp`.

**Commit**:

```bash
git add cpp/mtorch/core/detail/accelerate.h cpp/mtorch/core/detail/accelerate.cpp cpp/mtorch/core/tensor.cpp
git commit -m "refactor(phase2b-1e): extract detail/accelerate"
```

---

### Step 2b-1f: `detail/broadcast.{h,cpp}`

**Goal**: Broadcast index computation and grad reduction.

**Preconditions**: 2b-1e committed. `ls cpp/mtorch/core/detail/accelerate.h` succeeds.

**Actions**:

1. Block: from `grep -n 'std::vector<int64_t> linear_to_index(int64_t linear' cpp/mtorch/core/tensor.cpp`
   — the DEFINITION (with `{`), currently ~L1294 — through the closing `}` of
   `initialize_scaled_broadcasted_half_as_float` (~L1527, the line before
   `bool try_bmm_float16_accelerate` used to sit; after 2b-1e the block is immediately followed by
   `std::vector<int64_t> tail_shape_after_mask`). 10 symbols:

   `linear_to_index` (~L1294), `storage_index_from_multi` (~L1304), `broadcast_shape` (~L1315),
   `value_for_broadcast` (~L1334), `add_for_broadcast` (~L1349), `reduce_grad_to_shape` (~L1365),
   `transpose_2d_copy` (~L1376), `initialize_scaled_broadcasted_result` (~L1391, contains an
   `#if defined(MTORCH_USE_ACCELERATE)` region — moves verbatim), `copy_contiguous_half_to_float_buffer`
   (~L1457), `initialize_scaled_broadcasted_half_as_float` (~L1466).

2. Create `cpp/mtorch/core/detail/broadcast.h` (DETAIL-HEADER template; extra includes: `common.h`,
   `half.h`, `storage.h`, `accelerate.h`).
   - Header-inline (inline rule): `linear_to_index`, `storage_index_from_multi`,
     `value_for_broadcast`, `add_for_broadcast`.
   - Declarations (declaration rule) for: `broadcast_shape`, `reduce_grad_to_shape`,
     `transpose_2d_copy`, `initialize_scaled_broadcasted_result`,
     `copy_contiguous_half_to_float_buffer`, `initialize_scaled_broadcasted_half_as_float`.
3. Create `cpp/mtorch/core/detail/broadcast.cpp` (same includes as the header plus
   `#include "mtorch/core/detail/broadcast.h"`) with the 6 declared definitions, original order.
4. tensor.cpp:
   - delete the block;
   - **delete the forward-declaration line**
     `std::vector<int64_t> linear_to_index(int64_t linear, const std::vector<int64_t>& sizes);`
     (originally ~L683, just above `count_nonzero_dense_values`) — if it stayed, the anonymous
     namespace would declare a never-defined function and the nonzero helpers would fail to link;
   - add `#include "mtorch/core/detail/broadcast.h"`.

**Verification**: STANDARD VERIFY. Additionally
`grep -c 'linear_to_index(int64_t linear, const std::vector<int64_t>& sizes);' cpp/mtorch/core/tensor.cpp`
must print `0`.

**On failure**: STANDARD FAIL, `<new files>` = `cpp/mtorch/core/detail/broadcast.h cpp/mtorch/core/detail/broadcast.cpp`.

**Commit**:

```bash
git add cpp/mtorch/core/detail/broadcast.h cpp/mtorch/core/detail/broadcast.cpp cpp/mtorch/core/tensor.cpp
git commit -m "refactor(phase2b-1f): extract detail/broadcast"
```

---

### Step 2b-1g: `detail/promotion.h` (header-only)

**Goal**: Scalar binary-op evaluation, op-kind predicates, and dtype promotion rules in one header.
(The original plan said `promotion.{h,cpp}`; every function here is small and hot, so the header
alone satisfies it — no `promotion.cpp` is created.)

**Preconditions**: 2b-1f committed. `ls cpp/mtorch/core/detail/broadcast.h` succeeds.

**Actions**:

1. Block: from `grep -n 'double logaddexp_value(double left, double right);' cpp/mtorch/core/tensor.cpp`
   (the group of 7 forward declarations, currently ~L1764 — they move along verbatim because
   `apply_binary` uses them before their definitions) through the closing `}` of
   `binary_result_dtype` (`grep -n 'ScalarType binary_result_dtype' cpp/mtorch/core/tensor.cpp` →
   ~L2194; ends ~L2240, the line before `TensorPtr make_empty_contiguous_tensor(` at ~L2242).
   Symbols (38 functions + 2 enums + 7 forward declarations):

   forward declarations of `logaddexp_value`, `logaddexp2_value`, `xlogy_value`,
   `logaddexp_float_value`, `logaddexp2_float_value`, `xlogy_float_value`, `nextafter_float_value`
   (~L1764–1770 — leave them as plain declarations, no inline keyword);
   `maximum_value` (~L1772), `minimum_value` (~L1776), `maximum_float_value` (~L1780),
   `minimum_float_value` (~L1784), `apply_binary` (~L1788), `is_comparison_op` (~L1885),
   `apply_comparison` (~L1889), `is_logical_binary_op` (~L1911), `is_bitwise_binary_op` (~L1915),
   `is_boolean_result_binary_op` (~L1919), `is_nondifferentiable_binary_op` (~L1923),
   `enum class LogicalBinaryOp` (~L1928), `parse_logical_binary_op` (~L1934),
   `apply_logical_binary` (~L1947), `enum class BitwiseBinaryOp` (~L1959),
   `parse_bitwise_binary_op` (~L1965), `bool_logical_equivalent_for_bitwise_op` (~L1978),
   `apply_bitwise_binary` (~L1991), `validate_close_tolerances` (~L2003), `close_values` (~L2009),
   `logaddexp_value` (~L2021), `logaddexp2_value` (~L2032), `binary_softmax_left` (~L2043),
   `xlogy_value` (~L2055), `logaddexp_float_value` (~L2059), `logaddexp2_float_value` (~L2070),
   `xlogy_float_value` (~L2081), `nextafter_float_value` (~L2085), `fmax_grad_left` (~L2108),
   `fmin_grad_left` (~L2118), `is_floating_dtype` (~L2128), `is_integral_dtype` (~L2132),
   `dtype_category` (~L2136), `regular_promote_dtype` (~L2146), `is_bitwise_dtype` (~L2165),
   `bitwise_result_dtype` (~L2169), `arithmetic_result_dtype` (~L2179), `binary_result_dtype`
   (~L2194).

2. Create `cpp/mtorch/core/detail/promotion.h` (DETAIL-HEADER template; no extra detail includes
   needed beyond the template). Paste the whole block in original order; inline rule on every
   function DEFINITION; enums and the 7 forward declarations verbatim.
3. tensor.cpp: delete the block; add `#include "mtorch/core/detail/promotion.h"`.

**Verification**: STANDARD VERIFY. Additionally
`grep -c 'double apply_binary' cpp/mtorch/core/tensor.cpp` must print `0`.

**On failure**: STANDARD FAIL, `<new files>` = `cpp/mtorch/core/detail/promotion.h`.

**Commit**:

```bash
git add cpp/mtorch/core/detail/promotion.h cpp/mtorch/core/tensor.cpp
git commit -m "refactor(phase2b-1g): extract detail/promotion.h"
```

---

### Step 2b-1h: `detail/factory.{h,cpp}`

**Goal**: Tensor construction, strided/dense copy+cast, and the float32 reduce fast paths.

**Preconditions**: 2b-1g committed. `ls cpp/mtorch/core/detail/promotion.h` succeeds.

**Actions**:

1. Block: from `grep -n 'TensorPtr make_empty_contiguous_tensor' cpp/mtorch/core/tensor.cpp` — the
   DEFINITION at ~L2242 — through the closing `}` of `try_reduce_var_mean_dim_float32`
   (`grep -n 'try_reduce_var_mean_dim_float32' cpp/mtorch/core/tensor.cpp` → ~L3076; ends ~L3132,
   the line before `int64_t broadcast_storage_offset_1d(` at ~L3134 — that index fast-path block
   STAYS in tensor.cpp until 2b-11).

2. Create `cpp/mtorch/core/detail/factory.h` (DETAIL-HEADER template; extra includes: `common.h`,
   `half.h`, `storage.h`, `accelerate.h`).
   - Header-inline (inline rule; these keep their default arguments):
     `make_empty_contiguous_tensor` (~L2242), `make_uninitialized_contiguous_tensor` (~L2251),
     `storage_elements_for_shape_strides` (~L2260), `covers_dense_storage_without_holes` (~L2264),
     `make_empty_strided_tensor` (~L2269), `make_uninitialized_strided_tensor` (~L2279),
     `make_scalar_int64_tensor` (~L2289), `copy_storage_element_bytes` (~L2313).
   - Templates (verbatim, in header): `cast_dense_numeric` (~L2633), `cast_dense_numeric_to_half`
     (~L2642), `cast_dense_half_to_numeric` (~L2651), `cast_dense_to_bool` (~L2660),
     `cast_dense_from_bool` (~L2669), `cast_dense_from_numeric_source` (~L2694).
   - Declarations (declaration rule) for the .cpp definitions listed in 3.
3. Create `cpp/mtorch/core/detail/factory.cpp` (same includes + `factory.h`) with these 18
   definitions in original order:
   `try_dense_byte_copy` (~L2295), `try_ranked_strided_copy` (~L2321), `try_ranked_strided_cast`
   (~L2548), `cast_dense_half_to_bool` (~L2677), `cast_dense_bool_to_half` (~L2685),
   `cast_dense_from_half_source` (~L2718), `cast_dense_from_bool_source` (~L2741), `try_dense_cast`
   (~L2764), `product_float32_unit_stride` (~L2798), `sum_float32_unit_stride` (~L2842),
   `max_float32_unit_stride` (~L2886), `sum_float32_strided` (~L2919), `product_float32_strided`
   (~L2941), `try_reduce_prod_float32` (~L2963), `variance_from_biased` (~L2990),
   `try_reduce_variance_float32` (~L2998), `try_reduce_var_mean_float32` (~L3036),
   `try_reduce_var_mean_dim_float32` (~L3076).
4. tensor.cpp: delete the block; add `#include "mtorch/core/detail/factory.h"`.

**Verification**: STANDARD VERIFY. Additionally
`grep -c 'TensorPtr make_empty_contiguous_tensor(' cpp/mtorch/core/tensor.cpp` must print `0`.

**On failure**: STANDARD FAIL, `<new files>` = `cpp/mtorch/core/detail/factory.h cpp/mtorch/core/detail/factory.cpp`.

**Commit**:

```bash
git add cpp/mtorch/core/detail/factory.h cpp/mtorch/core/detail/factory.cpp cpp/mtorch/core/tensor.cpp
git commit -m "refactor(phase2b-1h): extract detail/factory"
```

---

### Step 2b-1i: `detail/elementwise.{h,cpp}`

**Goal**: The elementwise execution machinery: plan structs, coalescing, kernels, and the whole
binary fast-path group. Plan structs, plan builders, and the four generic-kernel templates go in the
header; every other body goes in the .cpp.

**Preconditions**: 2b-1h committed. `ls cpp/mtorch/core/detail/factory.h` succeeds.

**Actions**:

1. Block: from `grep -n 'struct BroadcastOperand' cpp/mtorch/core/tensor.cpp` (~L3438) through the
   closing `}` of `attach_reduce_prod_backward`
   (`grep -n 'attach_reduce_prod_backward' cpp/mtorch/core/tensor.cpp` → ~L7958; it is the last
   definition before the `}  // namespace` that closes the big anonymous namespace, ~L7989).

2. Create `cpp/mtorch/core/detail/elementwise.h` (DETAIL-HEADER template; extra includes:
   `common.h`, `half.h`, `storage.h`, `accelerate.h`, `broadcast.h`, `promotion.h`, `factory.h`).
   **Header content, in this order** (inline rule on non-templates, templates verbatim):
   - the 7 structs: `BroadcastOperand` (~L3438), `UnaryElementwisePlan` (~L3443),
     `BinaryElementwisePlan` (~L3448), `TernaryElementwisePlan` (~L3454),
     `CoalescedBinaryElementwisePlan` (~L3461), `CoalescedUnaryElementwisePlan` (~L3469),
     `CoalescedTernaryElementwisePlan` (~L3475)
   - `make_broadcast_operand` (~L3485), `can_coalesce_stride` (~L3501),
     `make_unary_elementwise_plan` (~L3505), `make_binary_elementwise_plan` (~L3509),
     `make_ternary_elementwise_plan` (~L3516), `coalesce_unary_elementwise_plan` (~L3528),
     `coalesce_binary_elementwise_plan` (~L3551), `coalesce_ternary_elementwise_plan` (~L3580),
     `broadcast_storage_offset` (~L3615)
   - the 4 templates: `run_coalesced_unary_double_kernel` (~L3643),
     `run_coalesced_unary_float32_kernel` (~L3689), `run_coalesced_binary_double_kernel` (~L4749),
     `run_coalesced_binary_float32_kernel` (~L4807)
   - declarations (declaration rule) for every .cpp definition listed in 3.
3. Create `cpp/mtorch/core/detail/elementwise.cpp` (same includes + `elementwise.h`). Paste
   everything else from the block, **in original order** (the two forward declarations at ~L5897 and
   ~L5904 move verbatim too). Full definition list (85 functions):

   `try_accelerate_square_float32_kernel` (~L3735), `try_accelerate_vforce_float32_kernel` (~L3770),
   `erf_approx_float` (~L3853), `try_accelerate_sigmoid_float32_kernel` (~L3875),
   `try_accelerate_erf_float32_kernel` (~L3893), `try_accelerate_clamp_float32` (~L3973),
   `run_clamp_bound_float32_kernel` (~L4028), `run_clamp_bound_float16_dense_kernel` (~L4078),
   `run_clamp_float16_dense_kernel` (~L4135), `run_clamp_float16_channels_last_compatible_4d_kernel`
   (~L4206), `run_unary_float32_kernel` (~L4273), `run_unary_float16_kernel` (~L4460),
   `try_unary_channels_last_compatible_float16_4d_fast_path` (~L4537),
   `run_unary_predicate_float32_kernel` (~L4591), `run_unary_logical_not_bool_kernel` (~L4694),
   `try_accelerate_binary_special_float32_kernel` (~L4859),
   `try_neon_binary_minmax_float32_kernel` (~L4925), `run_binary_float32_kernel` (~L4996),
   `run_binary_logical_kernel` (~L5117), `try_binary_logical_bool_fast_path` (~L5161),
   `run_binary_bitwise_kernel` (~L5218), `run_isclose_float32_kernel` (~L5276),
   `try_relu_float32_kernel` (~L5329), `run_coalesced_where_kernel` (~L5406),
   `add_float32_contiguous` (~L5465), `sub_float32_contiguous` (~L5483), `mul_float32_contiguous`
   (~L5501), `try_binary_add_float32_fast_path` (~L5519), `try_binary_sub_float32_fast_path`
   (~L5654), `try_binary_mul_float32_fast_path` (~L5714), `add_half_scalar_contiguous` (~L5774),
   `subtract_half_scalar_contiguous` (~L5809), `add_half_tensors_contiguous` (~L5838),
   `try_binary_addsub_float16_fast_path` (~L5911), `is_half_channel_broadcast_for_channels_last`
   (~L5941), `binary_channels_last_channel_broadcast_half` (~L5974),
   `try_binary_channels_last_channel_broadcast_float16_fast_path` (~L6099),
   `is_channels_last_dense_float16` (~L6128), `is_singleton_broadcast_float16` (~L6133),
   `is_channels_last_spatial_mask_float16` (~L6145), `is_batch_scalar_broadcast_float16` (~L6158),
   `negate_half_contiguous` (~L6171), `multiply_half_scalar_contiguous` (~L6193),
   `divide_half_scalar_contiguous` (~L6222), `multiply_half_tensors_contiguous` (~L6251),
   `binary_half_unit_stride_block` (~L6275), `try_binary_rank3_same_shape_float16_fast_path`
   (~L6350), `is_rank3_token_broadcast_float16` (~L6385), `rank3_token_broadcast_offset` (~L6406),
   `binary_rank3_token_broadcast_half` (~L6417),
   `try_binary_rank3_token_broadcast_float16_fast_path` (~L6449),
   `is_rank3_batch_token_broadcast_float16` (~L6474), `binary_rank3_batch_token_broadcast_half`
   (~L6481), `try_binary_rank3_batch_token_broadcast_float16_fast_path` (~L6511),
   `binary_half_channels_last_compatible_4d_same_shape` (~L6536),
   `try_binary_channels_last_compatible_same_shape_float16_fast_path` (~L6636),
   `try_binary_mul_float16_fast_path` (~L6654), `binary_channels_last_singleton_broadcast_half`
   (~L6683), `try_binary_channels_last_singleton_broadcast_float16_fast_path` (~L6714),
   `is_singleton_broadcast_tensor` (~L6739), `is_channels_last_dense_floating` (~L6751),
   `read_float_or_half_value` (~L6755), `apply_float_binary_value` (~L6762),
   `try_binary_channels_last_singleton_broadcast_floating_fast_path` (~L6775),
   `try_binary_channels_last_same_shape_float32_fast_path` (~L6908),
   `apply_half_scalar_binary_block` (~L6978), `binary_channels_last_batch_broadcast_half` (~L7075),
   `try_binary_channels_last_batch_broadcast_float16_fast_path` (~L7099),
   `addcmul_half_channels_last_scalar_contiguous` (~L7124),
   `try_addcmul_channels_last_scalar_float16_fast_path` (~L7164),
   `binary_channels_last_spatial_mask_mul_half` (~L7186),
   `try_binary_channels_last_spatial_mask_float16_fast_path` (~L7244),
   `try_binary_minmax_float32_fast_path` (~L7269), `store_float32_row_broadcast_binary` (~L7322),
   `try_binary_row_broadcast_float32_fast_path` (~L7470), `try_softmax_float32_kernel` (~L7508),
   `try_softmax_float16_kernel` (~L7743), `try_binary_grad_same_shape_float32` (~L7804),
   `try_binary_grad_same_operand_mul_float32` (~L7859), `binary_grad_left` (~L7874),
   `binary_grad_right` (~L7918), `attach_reduce_prod_backward` (~L7958).

4. tensor.cpp: delete the block; add `#include "mtorch/core/detail/elementwise.h"`.

**Verification**: STANDARD VERIFY. Additionally, the big anonymous namespace must now contain ONLY
the three indexing-family blocks:

```bash
grep -c 'count_nonzero_dense_values\|tail_view_for_prefix\|try_index_int_tensor_dim_1d_fast_path' cpp/mtorch/core/tensor.cpp
grep -c 'struct BroadcastOperand\|bool run_unary_float32_kernel' cpp/mtorch/core/tensor.cpp
```

The first count is nonzero; the second must print `0`. `wc -l cpp/mtorch/core/tensor.cpp` should
now report roughly 25,500 lines.

**On failure**: STANDARD FAIL, `<new files>` = `cpp/mtorch/core/detail/elementwise.h cpp/mtorch/core/detail/elementwise.cpp`.

**Commit**:

```bash
git add cpp/mtorch/core/detail/elementwise.h cpp/mtorch/core/detail/elementwise.cpp cpp/mtorch/core/tensor.cpp
git commit -m "refactor(phase2b-1i): extract detail/elementwise"
```

---

## Step 2b-2: `tensor_core.cpp` — **fully worked example for all section carve-outs**

**Goal**: The dtype/device/grad-mode free functions, all `Storage` and `Tensor` member functions,
the tensor factories, and `to()` live in `cpp/mtorch/core/tensor_core.cpp`.

**Preconditions**: 2b-1i committed.
`ls cpp/mtorch/core/detail/elementwise.h` succeeds; `git status --porcelain` is empty.

**Actions**:

1. Determine the block boundaries (one contiguous block):

   ```bash
   grep -n '^int64_t element_size(ScalarType dtype)' cpp/mtorch/core/tensor.cpp
   grep -n '^TensorPtr reshape(' cpp/mtorch/core/tensor.cpp
   ```

   Start = the first line printed (originally ~L7991, immediately after the anonymous namespace's
   closing `}  // namespace`). End = the line BEFORE the second line printed (originally ~L8583,
   the closing `}` of `to()`).

2. Checklist — the block implements exactly these declarations from `tensor.h` (enumerate the
   public surface any time with `grep -nE '^[A-Za-z].*\(' cpp/mtorch/core/tensor.h`):
   `element_size` (~L7991), `dtype_name` (~L8009), `promote_dtype` (~L8027), `device_type_name`
   (~L8031), `device_name` (~L8041), `cpu_device` (~L8049), `metal_device` (~L8053),
   `devices_equal` (~L8057), `is_grad_enabled` (~L8061), `set_grad_enabled` (~L8065),
   `Storage::Storage` (~L8071), `Storage::numel` (~L8082), `Storage::get` (~L8086), `Storage::set`
   (~L8094), the 23 `Tensor::` members from `Tensor::Tensor` (~L8102) through
   `Tensor::backward_with` (~L8352) — enumerate them with
   `grep -n 'Tensor::' cpp/mtorch/core/tensor.cpp` before cutting — plus `contiguous_strides`
   (~L8368), `make_tensor` (~L8378), `full` (~L8394), `zeros` (~L8403), `ones` (~L8407),
   `empty_strided` (~L8411), `empty_like` (~L8420), `full_like` (~L8424), `zeros_like` (~L8430),
   `ones_like` (~L8434), `arange` (~L8438), `linspace` (~L8455), `eye` (~L8471), `to` (~L8479).

3. Create `cpp/mtorch/core/tensor_core.cpp` from the SECTION-FILE template (§0.8) and paste the
   block between `using namespace detail;` and the final `}  // namespace mtorch`, byte-for-byte.
   Notes for this file:
   - `Tensor::mark_storage_modified` contains `::mtorch::mark_storage_modified(*storage);` — keep it
     verbatim (§0.2 explains why it still resolves).
   - There is no anonymous namespace to move for this file.
4. Delete the block from tensor.cpp.

**Verification**: STANDARD VERIFY. Additionally
`grep -cE '^[a-zA-Z].*Tensor::' cpp/mtorch/core/tensor.cpp` must print `0` (no member-function
definitions left in tensor.cpp).

**On failure**: STANDARD FAIL, `<new files>` = `cpp/mtorch/core/tensor_core.cpp`.

**Commit**:

```bash
git add cpp/mtorch/core/tensor_core.cpp cpp/mtorch/core/tensor.cpp
git commit -m "refactor(phase2b-2): carve tensor_core.cpp out of tensor.cpp"
```

---

## Steps 2b-3 … 2b-15: Carving out the public API sections (1 file = 1 commit)

Every step below repeats the 2b-2 mechanic exactly: print block boundaries with the two (or more)
greps given → create the file from the SECTION-FILE template → paste the block(s) in order,
byte-for-byte → delete from tensor.cpp → STANDARD VERIFY → commit
(`git add cpp/mtorch/core/<file> cpp/mtorch/core/tensor.cpp` +
`git commit -m "refactor(phase2b-<n>): carve <file> out of tensor.cpp"`) → §5.3 move check →
PROGRESS.md. On failure: STANDARD FAIL with `<new files>` = the new .cpp. Compile/link errors:
apply §0.9. **Do the steps in the exact order listed** — later steps' boundary descriptions assume
earlier blocks are already gone.

Membership is authoritative from `tensor.h`: the per-step checklists below list every `tensor.h`
function the file must implement. After pasting, verify each name appears in the new file
(`grep -c '<name>' cpp/mtorch/core/<file>` ≥ 1). Helpers that sit inside a block always move with it.

### Step 2b-3: `losses.cpp` (~1,150 lines)

- Block: from `grep -n '^TensorPtr mse_loss(' cpp/mtorch/core/tensor.cpp` (~L15223) to the line
  before `grep -n '^TensorPtr relu(' cpp/mtorch/core/tensor.cpp` (~L16369).
- tensor.h checklist (6): `mse_loss`, `l1_loss` (~L15327), `nll_loss` (~L15446),
  `cross_entropy_loss` (~L15575), `binary_cross_entropy_loss` (~L16001),
  `binary_cross_entropy_with_logits_loss` (~L16157).
- Helpers moving with the block (namespace scope, stay at namespace scope):
  `validate_loss_class_weight`, `loss_class_weight_value`, `validate_bce_optional_weight`,
  `optional_broadcast_weight_value`, `binary_cross_entropy_element`,
  `binary_cross_entropy_with_logits_element`.

### Step 2b-4: `activations.cpp` (~925 lines)

- Block: from `grep -n '^TensorPtr relu(' cpp/mtorch/core/tensor.cpp` (~L16369) to the line before
  `grep -n 'bool try_layer_norm_half_row_neon' cpp/mtorch/core/tensor.cpp` (~L17294).
- tensor.h checklist (13): `relu`, `leaky_relu` (~L16391), `silu` (~L16480), `elu` (~L16770),
  `selu` (~L16774), `softplus` (~L16780), `hardtanh` (~L16974), `relu6` (~L16996), `hardsigmoid`
  (~L17013), `hardswish` (~L17030), `softsign` (~L17061), `mish` (~L17081), `gelu` (~L17207).
- Helpers moving: `unary_activation` (~L16616), `try_elu_like_float32_accelerate` (~L16664),
  `elu_like` (~L16745), `gelu_value` (~L17103), `gelu_grad_value` (~L17115), and any other
  definitions inside the block.

### Step 2b-5: `normalization.cpp` (~1,810 lines)

- Two blocks; after 2b-3 and 2b-4 they are ADJACENT in tensor.cpp, so this is one contiguous cut:
  from `grep -n 'void attach_softmax_backward' cpp/mtorch/core/tensor.cpp` (~L15085 originally) to
  the line before `grep -n '^TensorPtr binary_tensor_tensor(' cpp/mtorch/core/tensor.cpp`
  (~L18965 originally). (Originally these were L15085–15222 — softmax — and L17294–18964 —
  layer/rms/batch/group norm — with losses and activations in between.)
- tensor.h checklist (7): `softmax` (~L15215), `log_softmax` (~L15219), `layer_norm` (~L17386),
  `rms_norm` (~L17860), `normalize_l2` (~L18081), `batch_norm` (~L18137), `group_norm` (~L18643).
- Helpers moving: `attach_softmax_backward`, `softmax_impl` (~L15145),
  `try_layer_norm_half_row_neon` (~L17294), `rms_norm_half_unit_stride_row` (~L17715),
  `try_rms_norm_half_lastdim_unit_stride` (~L17770), `batch_norm_inner_size` (~L18126),
  `batch_norm_channel_for_linear` (~L18133), `sum_and_sumsq_channels_last_half_group` (~L18520),
  `normalize_channels_last_half_group` (~L18589).
- (The fp32/fp16 softmax kernels `try_softmax_float32_kernel` / `try_softmax_float16_kernel` are
  already in detail/elementwise since 2b-1i; nothing to do.)

### Step 2b-6: `sorting.cpp` (~1,835 lines)

- Block: from `grep -n 'struct SortEntry' cpp/mtorch/core/tensor.cpp` (~L21967) to the line before
  `grep -n 'TensorPtr clone_with_identity_backward' cpp/mtorch/core/tensor.cpp` (~L23799).
- tensor.h checklist (8): `sort` (~L22123), `argsort` (~L22376), `quantile_flat` (~L22339),
  `quantile_dim_2d` (~L22357), `topk` (~L22483), `searchsorted` (~L22641), `unique` (~L23766),
  `unique_consecutive` (~L23784).
- Helpers moving: `SortEntry`, `sort_entry_before`, `order_sort_entries`, the `quantile_*` group
  (~L22158–22357), `searchsorted_contiguous_kernel` group (~L22531–22640), the whole `unique_*`
  helper group (~L22668–23764).

### Step 2b-7: `pooling.cpp` (~1,340 lines)

- Two non-adjacent blocks (paste P1 then P2 into the same file):
  - P1: from `grep -n '^TensorPtr adaptive_avg_pool1d(' cpp/mtorch/core/tensor.cpp` (~L9497) to
    the line before `grep -n '^TensorPtr pixel_shuffle(' cpp/mtorch/core/tensor.cpp` (~L9729).
  - P2: from `grep -n '^TensorPtr max_pool2d(' cpp/mtorch/core/tensor.cpp` (~L29192) to the line
    before the `namespace {` line that immediately precedes
    `grep -n 'TensorPtr detached_contiguous_for_read' cpp/mtorch/core/tensor.cpp` (~L30297; the
    `namespace {` is 2 lines above the grep hit — the cut ends just BEFORE that `namespace {`).
- tensor.h checklist (8): `adaptive_avg_pool1d` (~L9497), `adaptive_avg_pool2d` (~L9597),
  `max_pool2d` (~L29192), `max_pool1d` (~L29402), `avg_pool1d` (~L29566), `avg_pool2d` (~L29728),
  `unfold2d` (~L29931), `fold2d` (~L30106).
- `pooling_output_size` and the pool-arg helpers are already in detail/common (2b-1b).

### Step 2b-8: `attention.cpp` (~1,760 lines)

- Block: from the `namespace {` line that immediately precedes
  `grep -n 'TensorPtr detached_contiguous_for_read' cpp/mtorch/core/tensor.cpp` (~L30297,
  including that `namespace {` line) to the line before
  `grep -n 'TensorPtr try_cat_channels_last_fast_path' cpp/mtorch/core/tensor.cpp` (~L32057).
- tensor.h checklist (3): `scaled_dot_product_attention` (~L30930), `linear` (~L31401),
  `embedding` (~L31748).
- The block carries its own anonymous namespace (~L30297–30928: `detached_contiguous_for_read`,
  `SdpaPrefixPlan`, `sdpa_prefix_plan`, `sdpa_attention_shape`, `validate_sdpa_mask`,
  `apply_sdpa_mask_value`, `softmax_scores`, `softmax_scores_float`,
  `copy_rank4_half_to_float_contiguous`, `copy_rank3_half_to_float_contiguous`,
  `try_scaled_dot_product_attention_float16_rank4` (~L30535),
  `try_scaled_dot_product_attention_float16_rank3` (~L30724)) — the `namespace { ... }  // namespace`
  wrapper moves verbatim. Namespace-scope helpers `try_small_batch_half_linear` (~L31326),
  `embedding_row_norm` (~L31696), `renorm_embedding_weight` (~L31720) also move.

### Step 2b-9: `linalg.cpp` (~2,100 lines)

- Block: from `grep -n 'TensorPtr clone_with_identity_backward' cpp/mtorch/core/tensor.cpp`
  (~L23799) to the line before the `namespace {` line that immediately precedes
  `grep -n 'normalize_conv2d_pair' cpp/mtorch/core/tensor.cpp` (~L25899; the cut ends just BEFORE
  that `namespace {`).
- tensor.h checklist (16): `dot` (~L23837), `vdot` (~L23912), `inner` (~L23916), `tensordot`
  (~L24091), `kron` (~L24241), `mv` (~L24385), `outer` (~L24441), `matmul` (~L24896), `bmm`
  (~L25066), `addmm` (~L25184), `addmv` (~L25368), `addr` (~L25466), `baddbmm` (~L25557), `addbmm`
  (~L25695), `chain_matmul` (~L25828), `matrix_power` (~L25848).
- Helpers moving: `clone_with_identity_backward`, `matrix_power_identity` (~L23811),
  `vector_matmul` (~L24482), `struct MatmulBatchPlan` (~L24537), `make_matmul_batch_plan` (~L24556),
  the `matmul_*` helper group (~L24596–24821), `try_accelerate_batched_matmul` (~L24712),
  `batched_matmul` (~L24822).
- fp16 GEMM (`try_bmm_float16_accelerate` / `try_baddbmm_float16_accelerate`) and
  `initialize_scaled_broadcasted_result` / `transpose_2d_copy` are already in detail (2b-1e/2b-1f).

### Step 2b-10: `reductions.cpp` (~2,315 lines)

- Block: from `grep -n '^TensorPtr reduce_sum(' cpp/mtorch/core/tensor.cpp` (~L19654) through the
  closing `}` of `argmin_dim` (`grep -n 'TensorPtr argmin_dim' cpp/mtorch/core/tensor.cpp` →
  ~L21956; `argmin_dim` is the last function of the block — by this point in the order, the sorting
  block that used to follow it is already gone).
- tensor.h checklist (38): `reduce_sum` (~L19654), `reduce_sum_dim` (~L19737), `diff_float32`
  (~L19983), `cumsum` (~L19884), `cumprod` (~L20044), `cummax` (~L20673), `cummin` (~L20677),
  `trapezoid_dx` (~L20697), `cumulative_trapezoid_dx` (~L20773), `gradient_uniform` (~L21005),
  `reduce_mean` (~L21053), `reduce_mean_dim` (~L21073), `reduce_prod` (~L21122), `reduce_prod_dim`
  (~L21164), `reduce_var` (~L21188), `reduce_var_dim` (~L21211), `reduce_var_tail` (~L21412),
  `reduce_std` (~L21416), `reduce_std_dim` (~L21422), `reduce_std_tail` (~L21430),
  `reduce_var_mean` (~L21434), `reduce_var_mean_dim` (~L21462), `reduce_std_mean` (~L21507),
  `reduce_std_mean_dim` (~L21513), `reduce_all` (~L21521), `reduce_all_dim` (~L21584),
  `reduce_any` (~L21537), `reduce_any_dim` (~L21588), `reduce_max` (~L21592), `amax` (~L21746),
  `reduce_max_dim` (~L21750), `argmax` (~L21782), `argmax_dim` (~L21872), `reduce_min` (~L21883),
  `amin` (~L21897), `reduce_min_dim` (~L21901), `argmin` (~L21933), `argmin_dim` (~L21956).
- Helpers moving: `try_reduce_sum_float16_last_dim` (~L19710), the cumulative/NaN helper group
  (`is_nan_bits`, `is_nan_half_bits`, `cumulative_value_is_nan`,
  `should_update_cumulative_extreme`, `should_update_cumulative_extreme_half`,
  `cumulative_extreme_should_update_value`, `cumulative_extreme` ~L20156–20679),
  `trapezoid_result_dtype`, `gradient_result_dtype`, `validate_gradient_axis`,
  `write_gradient_uniform_line`, `reduce_var_tail_impl` (~L21250), `reduce_bool_dim` (~L21553),
  `reduce_extreme_values_dim` (~L21654), `reduce_extreme_values_dims` (~L21733).

### Step 2b-11: `indexing.cpp` (~2,570 lines)

- TWO cuts:
  1. Main block: from `grep -n '^TensorPtr index(' cpp/mtorch/core/tensor.cpp` (~L12595) to the
     line before `grep -n '^TensorPtr unary(' cpp/mtorch/core/tensor.cpp` (~L14373). Paste at
     namespace scope.
  2. The ENTIRE remaining content of the first anonymous namespace of tensor.cpp (the three legacy
     blocks left behind by 2b-1: nonzero counters `count_nonzero_dense_values` …
     `try_fill_nonzero_contiguous` (originally L685–840), mask/int-index tail helpers
     `tail_shape_after_mask` … `tail_view_for_prefix` (originally L1663–1762), and index fast paths
     `broadcast_storage_offset_1d` … `try_index_put_int_tensor_dim_1d_fast_path` (originally
     L3134–3437)). Paste all of it inside a `namespace { ... }  // namespace` block placed ABOVE the
     main block in indexing.cpp, then delete the now-empty `namespace {` / `}  // namespace` pair
     from tensor.cpp.
- tensor.h checklist (25): `index` (~L12595), `index_integer_tuple` (~L12963), `index_bool_mask`
  (~L12654), `index_put_bool_mask` (~L12744), `index_int_tensor` (~L12825), `index_put_int_tensor`
  (~L12851), `index_int_tensor_dim` (~L12882), `index_put_int_tensor_dim` (~L13014),
  `index_put_integer_tuple` (~L13111), `index_put` (~L13165), `one_hot` (~L13192), `index_select`
  (~L13273), `gather` (~L13332), `masked_select` (~L13431), `masked_fill` (~L13497), `nonzero`
  (~L13646), `nonzero_tuple` (~L13681), `count_nonzero` (~L13715), `count_nonzero_dim` (~L13729),
  `bincount` (~L14011), `isin` (~L14137), `scatter` (~L14353), `scatter_inplace` (~L14359),
  `scatter_add` (~L14363), `scatter_add_inplace` (~L14369).
- Namespace-scope helpers inside the main block move with it: `is_integer_index_tensor`
  (forward declaration ~L12921 AND definition ~L13053 — keep both, verbatim),
  `try_index_integer_tuple_pair_tail_fast_path` (~L12924), `broadcast_index_shape`
  (declaration ~L12922, definition ~L13069), `normalized_index_for_dim` (~L13057),
  `bincount_result_dtype`, `attach_bincount_backward`, `bincount_contiguous_kernel`,
  `is_exact_integral_or_bool_dtype`, `scalar_tensor_values_equal`,
  `isin_same_dtype_contiguous_kernel`, `try_isin_same_dtype_contiguous`, `validate_scatter_args`,
  `scatter_inplace_impl`.

### Step 2b-12: `resample.cpp` (~1,920 lines)

- Block: from `grep -n '^TensorPtr pixel_shuffle(' cpp/mtorch/core/tensor.cpp` (~L9729) to the
  line before `grep -n 'bool try_flip_contiguous_single_dim' cpp/mtorch/core/tensor.cpp` (~L11651).
- tensor.h checklist (6): `pixel_shuffle` (~L9729), `pixel_unshuffle` (~L9854), `channel_shuffle`
  (~L9979), `affine_grid` (~L10483), `grid_sample` (~L10622), `interpolate` (~L10897).
  (Note: `grid_sample` comes BEFORE `interpolate` in the file — an earlier draft of this document
  had the two line numbers swapped.)
- The block carries its own anonymous namespace (~L10054–10481: `interpolation_bounds`,
  `interpolation_source_coordinate`, `cubic_convolution1/2`, `cubic_interpolation_bounds`,
  `nearest_interpolation_index*`, the `grid_*` helper group, `affine_grid_*` fill helpers,
  `try_affine_grid_contiguous`) — its wrapper moves verbatim.

### Step 2b-13: `conv.cpp` (~3,295 lines)

- Block: from the `namespace {` line that immediately precedes
  `grep -n 'normalize_conv2d_pair' cpp/mtorch/core/tensor.cpp` (~L25899, including the
  `namespace {` line) through the closing `}` of `conv_transpose3d`
  (`grep -n '^TensorPtr conv_transpose3d(' cpp/mtorch/core/tensor.cpp` → ~L28723;
  `conv_transpose3d` is the last function of the block — the pooling block that used to follow it
  was removed in 2b-7).
- tensor.h checklist (6): `conv1d` (~L26064), `conv2d` (~L26490), `conv3d` (~L27530),
  `conv_transpose1d` (~L26242), `conv_transpose2d` (~L28178), `conv_transpose3d` (~L28723).
- The block carries its own anonymous namespace (~L25899–26062: `normalize_conv2d_pair`,
  `normalize_conv3d_triple`, `conv2d_output_size`, `normalize_conv1d_single`,
  `conv2d_fill_im2col_float32` (~L25960), `conv3d_fill_im2col_float32` (~L26005)) — wrapper moves
  verbatim. The fp16 weight-cache helpers are already in detail/half (2b-1c).

### Step 2b-14: `views.cpp` (~2,560 lines)

- THREE blocks, pasted into the file in this order:
  - V1: from `grep -n '^TensorPtr reshape(' cpp/mtorch/core/tensor.cpp` (~L8584) through the
    closing `}` of the second `pad` overload
    (`grep -n '^TensorPtr pad(' cpp/mtorch/core/tensor.cpp` → overloads ~L9247 and ~L9251; cut
    through the end of the second overload's body — the adaptive-pool block that used to follow it
    was removed in 2b-7). V1 contains the small pad anonymous namespace (~L9206–9245:
    `positive_mod`, `nonconstant_pad_index`, `nonconstant_pad_indices`) — wrapper moves verbatim.
  - V2: from `grep -n 'bool try_flip_contiguous_single_dim' cpp/mtorch/core/tensor.cpp` (~L11651)
    through the closing `}` of `unbind`
    (`grep -n 'std::vector<TensorPtr> unbind' cpp/mtorch/core/tensor.cpp` → ~L12584; cut through
    the end of its body — the indexing block that used to follow it was removed in 2b-11).
  - V3: from `grep -n 'TensorPtr try_cat_channels_last_fast_path' cpp/mtorch/core/tensor.cpp`
    (~L32057) through the closing `}` of `take` (~L32760) — i.e. everything up to but NOT
    including the final `}  // namespace mtorch` line of tensor.cpp.
- tensor.h checklist (48): `reshape` (~L8584), `unflatten` (~L8613), `transpose` (~L8653),
  `permute` (~L8680), `movedim` (~L8725), `flatten` (~L8771), `ravel` (~L8805), `t` (~L8809),
  `expand` (~L8820), `broadcast_to` (~L8870), `broadcast_shapes` (~L8874), `broadcast_tensors`
  (~L8901), `repeat` (~L8917), `repeat_interleave` (2 overloads, ~L9158/~L9171), `tile` (~L9198),
  `pad` (2 overloads, ~L9247/~L9251), `flip` (~L11742), `fliplr` (~L11779), `flipud` (~L11786),
  `rot90` (~L11865), `roll` (~L11970), `squeeze` (2 overloads, ~L12043/~L12055), `unsqueeze`
  (~L12068), `narrow` (~L12084), `select` (~L12095), `as_strided` (~L12112), `diagonal` (~L12125),
  `diag` (~L12196), `diagflat` (~L12217), `diag_embed` (~L12221), `block_diag` (~L12304), `tril`
  (~L12460), `triu` (~L12486), `trace` (~L12512), `split` (2 overloads, ~L12519/~L12537), `chunk`
  (~L12561), `unbind` (~L12584), `cat` (~L32141), `cat_pair` (~L32251), `stack` (~L32297),
  `hstack` (~L32449), `vstack` (~L32462), `dstack` (~L32490), `column_stack` (~L32544),
  `cartesian_prod` (~L32594), `where` (~L32676), `take` (~L32704).
- Helpers moving: `repeat_interleave_from_counts` (~L9006), `try_flip_contiguous_single_dim`
  (~L11651), `positive_mod_shift` (~L11920), `try_cat_channels_last_fast_path` (~L32057).

### Step 2b-15: `elementwise_ops.cpp` (~1,400 lines)

- At this point tensor.cpp contains nothing but includes, `namespace mtorch {`,
  `using namespace detail;`, the two ops blocks, and the closing brace. Cut EVERYTHING between
  `using namespace detail;` and the final `}  // namespace mtorch` into the new file.
  (Originally these were two blocks: `unary` … `clamp_max`, L14373–15084, and
  `binary_tensor_tensor` … `addcdiv`, L18965–19653.)
- tensor.h checklist (22): `unary` (~L14373), `unary_predicate` (~L14563), `logical_not`
  (~L14616), `bitwise_not` (~L14643), `deg2rad` (~L14720), `rad2deg` (~L14731), `frac` (~L14742),
  `nan_to_num` (~L14772), `reciprocal` (~L14929), `clamp` (~L15003), `clamp_min` (~L15077),
  `clamp_max` (~L15081), `binary_tensor_tensor` (~L18965), `binary_tensor_scalar` (~L19106),
  `binary_scalar_tensor` (~L19178), `isclose` (~L19217), `allclose` (~L19231), `lerp`
  (2 overloads, ~L19340/~L19467), `addcmul` (~L19646), `addcdiv` (~L19650).
- Helpers moving: `finite_max_for_dtype` (~L14762), `clamp_bound` (~L14951),
  `lerp_half_tensors_same_storage_order` (~L19295),
  `try_lerp_half_scalar_same_storage_order_fast_path` (~L19323), `fused_addcmul_addcdiv`
  (~L19472).

---

## Step 2b-16: Confirm and remove the tensor.cpp remnants

**Goal**: Prove tensor.cpp is empty and delete it; the build must still link.

**Preconditions**: 2b-15 committed; `git status --porcelain` is empty.

**Actions & Verification (this step's actions ARE the verification)**:

1. Prove nothing is left. Both commands must produce the shown results:

   ```bash
   wc -l cpp/mtorch/core/tensor.cpp
   grep -vE '^[[:space:]]*$|^#include|^namespace mtorch \{|^using namespace detail;|^\}  // namespace mtorch$|^//' cpp/mtorch/core/tensor.cpp
   ```

   The `grep -v` must print NOTHING (exit code 1). If it prints anything, an implementation was
   missed: find which step's checklist it belongs to in the tables above, move it into that file
   now (one extra commit `refactor(phase2b-16): move straggler <name> into <file>`), and re-run.
2. Delete the file and rebuild (the setup.py glob simply stops picking it up):

   ```bash
   git rm cpp/mtorch/core/tensor.cpp
   python3 setup.py build_ext --inplace
   pytest tests/compat -q | tail -3
   tail -1 docs/design/baseline/tests-baseline.txt
   ```

   PASS = build exits 0 (proving every symbol still links from the new files) AND the test summary
   matches the baseline. Also confirm the file census:

   ```bash
   ls cpp/mtorch/core/ cpp/mtorch/core/detail/
   wc -l cpp/mtorch/core/*.cpp cpp/mtorch/core/detail/*.{h,cpp}
   ```

   Expected: 14 section .cpp files + tensor.h under core/, 9 headers + 5 .cpp under detail/, and no
   file over 5,000 lines.

**On failure**: `git restore --staged --worktree .` (restores tensor.cpp), rebuild, mark 2b-16 **BLOCKED** in
`docs/design/PROGRESS.md` per §0.5.

**Commit**:

```bash
git commit -m "refactor(phase2b-16): remove emptied tensor.cpp"
```

---

## Step 2b-17: Performance gate (required)

**Goal**: Prove the split caused no benchmark case to degrade by more than 5% (median-ratio gate)
against the Phase 0 baseline.

**Preconditions**: 2b-16 committed.
`ls tools/compare_benchmarks.py docs/design/baseline/benchmark-baseline.json` succeeds.

**Actions**:

```bash
python3 setup.py build_ext --inplace
pytest tests/compat -q | tail -3
mkdir -p benchmark-results
pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
    --compat-benchmark-json benchmark-results/after-phase2b.json
python3 tools/compare_benchmarks.py \
    docs/design/baseline/benchmark-baseline.json benchmark-results/after-phase2b.json
```

**Verification**: The compare script's final line must read `... regressions=0 missing=0 new=0` and
its exit code must be 0 (the script's threshold defaults to 1.05 = the 5% gate). If `REGRESSION`
lines appear, first apply the 3× reproduction check of `01-rules-and-verification.md` §5.2 —
noise (reproduced 0–1 of 3) passes with a PROGRESS.md note.

**On failure** (a REGRESSION reproduces in 2 of 3 or more reruns), apply the remedies in this exact
order, one commit per attempt, re-running the failed case's 3× recheck after each:

1. **Inline promotion.** Move the hot helper's definition from its detail .cpp into the matching
   detail header. Candidate order (stop at the first that fixes the case):
   `run_unary_float32_kernel`, `run_binary_float32_kernel`, `try_binary_add_float32_fast_path`,
   `try_binary_sub_float32_fast_path`, `try_binary_mul_float32_fast_path` (all in
   `cpp/mtorch/core/detail/elementwise.cpp`), then `try_dense_cast` and `sum_float32_unit_stride`
   (in `cpp/mtorch/core/detail/factory.cpp`). Mechanics per helper: cut the full definition from the
   .cpp, paste it at the END of the matching header (immediately before `}  // namespace
   mtorch::detail`), prefix `inline `, and delete the now-duplicate declaration from the header.
   Build + recheck. Commit as
   `git add cpp/mtorch/core/detail && git commit -m "refactor(phase2b-17): inline <name> into header (perf)"`.
2. **LTO.** In `setup.py`, change
   `extra_compile_args = ["-std=c++20", "-O3"]` to
   `extra_compile_args = ["-std=c++20", "-O3", "-flto"]` and
   `extra_link_args: list[str] = []` to
   `extra_link_args: list[str] = ["-flto"]`. Then:

   ```bash
   rm -rf build && python3 setup.py build_ext --inplace
   pytest tests/compat -q | tail -3
   ```

   Re-run the full benchmark + compare. Commit as
   `git add setup.py && git commit -m "refactor(phase2b-17): enable -flto (perf)"`.
3. **Still failing**: `git revert --no-edit <hash>` the most recent carve-out commit related to the
   degraded op (find it with `git log --oneline | head -30`), record 2b-17 as **BLOCKED** in
   `docs/design/PROGRESS.md` with the case IDs and measured ratios, commit PROGRESS.md, and stop.

**Commit** (on success — the JSON itself is gitignored, only PROGRESS.md changes):

```bash
git add docs/design/PROGRESS.md
git commit -m "refactor(phase2b-17): pass performance gate for tensor.cpp split"
```

---

## Pitfall list

| Symptom | Cause and fix |
|---|---|
| undefined symbol (at link time) | An anonymous-namespace helper is referenced from another file. Promote it to `mtorch::detail` and declare it in the topic's detail header (§0.9) |
| duplicate symbol (at link time) | A function definition without `inline` in a header, or a variable definition in a header. Apply the inline rule, or move the definition to a .cpp |
| undefined symbol involving a template | The template definition landed in a .cpp. Move it, definition and all, to the header |
| `redefinition of default argument` | A default arg appears in both the header declaration and the .cpp definition. Delete it from the .cpp (default-argument rule, §0.2) |
| `use of undeclared identifier` in a new section file | The helper is still in tensor.cpp. Follow §0.9 exactly (anon-ns → move it now; namespace scope → add a declaration) |
| Build error around `#if MTORCH_USE_ACCELERATE` | The guard structure was not moved verbatim. detail/accelerate.h keeps group 1 inside the guard exactly as 2b-1e specifies |
| Wrong grad-mode behavior / linker error on `grad_enabled` | `thread_local bool grad_enabled` was defined in the header (one copy per TU) instead of `extern` in common.h + a single definition in common.cpp |
| Builds but numeric mismatch in tests | Code was "tidied" during the move. `git diff HEAD` and eliminate every difference that is not pure cut-and-paste plus the allowed include/using/declaration/inline additions |
| Benchmark degradation | Procedure in 2b-17. Confirm reproduction 3× before acting (it is usually noise) |
