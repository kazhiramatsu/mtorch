# 08. Phase 5: Device/Backend Infrastructure (the layer Phases 6 and 7 plug into)

Prerequisite: **Phases 0–4 are complete AND every item under "Overall completion check" in
`docs/design/PROGRESS.md` is checked.** `01-rules-and-verification.md` read in full.

Objective: build the device/backend plumbing that the Metal backend (Phase 6, guide
`09-phase6-metal.md`) and the CUDA backend (Phase 7, guide `10-phase7-cuda.md`) plug into,
**without changing any observable CPU behavior**. Concretely, this phase adds:

1. `DeviceType::CUDA` (the enum and the `Device` struct already exist — see §1).
2. An allocator interface + a process-wide allocator registry
   (`cpp/mtorch/core/detail/allocator.{h,cpp}`), with the CPU allocator always registered.
3. Allocator-backed storage for non-CPU devices (the CPU storage path stays byte-for-byte
   identical).
4. A cross-device transfer API `to_device()` (declared in `tensor.h` — an explicitly allowed
   addition), wired into the existing `to()` and the `Tensor.cpu()` binding.
5. The torch-compatible Python surface: `mtorch.device` accepting `"cuda"`,
   `mtorch._C._backend_available()`, and `mtorch.backends.mps.is_available()/is_built()` /
   `mtorch.cuda.is_available()` wired to the allocator registry (all `False` until Phases 6/7
   register a backend).
6. The self-parity test scaffolding `tests/compat/test_device_parity.py` (runs as a
   cpu-vs-cpu pass until a backend exists).
7. The **dispatch hook-point convention** for Phases 6/7, documented (not applied) in §4.

What this phase does **NOT** do:

- It does not touch any public op in `tensor.h` (no kernel gains a device branch; the hook
  pattern in §4 is documentation for Phases 6/7 only).
- It does not implement any Metal or CUDA code. After Phase 5, `mtorch.zeros((1,), device="mps")`
  still raises `NotImplementedError`, exactly as pinned by `tests/test_device.py`.
- It does not create `cpp/mtorch/core/device.h`. The design contract says to add a `Device`
  struct there only "if nothing equivalent exists" — an equivalent **does** exist
  (`struct Device` in `cpp/mtorch/core/tensor.h`, verified in §1), so no new header is created.

## When to start (gate — check before doing anything)

Run all of the following from the repository root (`cd /Users/hiramatsu/dev/mtorch`):

```bash
grep -c -- '- \[ \]' docs/design/PROGRESS.md
```

Expected output: `0` (every step of Phases 0–4 and every overall completion criterion is
checked). If the output is not `0`, **stop — Phase 5 must not start**. Do not add the Phase 5
checklist (step 5-0) before this prints `0`, because the Phase 5 checklist itself introduces
new `- [ ]` lines.

```bash
ls cpp/mtorch/core/tensor_core.cpp cpp/mtorch/core/detail/storage.h \
   cpp/mtorch/python/py_common.cpp cpp/mtorch/python/py_registry.h \
   cpp/mtorch/python/module_init.cpp cpp/mtorch/python/py_tensor_type.cpp \
   mtorch/cuda/__init__.py mtorch/mps.py mtorch/backends/mps.py \
   tests/compat/harness.py docs/design/baseline/tests-baseline.txt \
   docs/design/baseline/collect-count.txt docs/design/baseline/benchmark-baseline.json \
   tools/compare_benchmarks.py
grep -Fn 'glob.glob("cpp/mtorch/core/detail/*.cpp")' setup.py
```

Expected: the `ls` lists every file with no "No such file" error (these are the post-Phase-1/2a/2b
homes of the code this phase edits, plus the Phase 0 baseline artifacts), and the `grep` prints
one line (the setup.py glob from step 2a-1 — it automatically picks up every new `.cpp` this
phase creates; **no setup.py edit is needed anywhere in Phase 5**).

Then run the environment check of `01-rules-and-verification.md` §3. Only after all of the
above pass, do step 5-0.

## 0. Conventions specific to this phase

### 0.1 One rule above all

This phase adds code next to a fully green, baseline-matched tree. Every step below gives the
complete content of every new file and exact BEFORE/AFTER text for every edit. If the BEFORE
text of an edit cannot be found with the given grep, **do not improvise** — treat the step as
failed and follow STANDARD FAIL (§0.4). The same applies if any expected output differs.

### 0.2 Include convention for new Phase 5 files

Phase 2b's STD-INCLUDES block (05 §0.6) exists to guarantee that *moved* code keeps compiling.
Phase 5 writes **new** code, so new Phase 5 files use exactly the minimal include lists shown
in each step — copy them verbatim, do not add or remove includes.

### 0.3 STANDARD VERIFY (two forms)

**STANDARD VERIFY-A** (used by steps 5-0 through 5-5 — the tree must match the baseline
exactly):

```bash
python3 setup.py build_ext --inplace
pytest tests/compat -q 2>&1 | tee /tmp/tests-current.txt | tail -3
tail -1 docs/design/baseline/tests-baseline.txt | tr -d '=' | sed 's/ in .*//' | xargs
tail -1 /tmp/tests-current.txt | tr -d '=' | sed 's/ in .*//' | xargs
```

PASS if and only if the build exits 0 AND the last two printed lines are
character-for-character identical (every count — passed / failed / xfailed / skipped / error —
matches). This is the quick form of `01-rules-and-verification.md` §5.1; if in doubt, run the
full §5.1 procedure — it is authoritative. The full run takes minutes to tens of minutes; set a
command timeout of at least 60 minutes and never abort partway.

**STANDARD VERIFY-B** (used by steps 5-6 and 5-7 — after the new parity test file exists, the
tree must match the baseline **plus exactly 21 new passing tests**):

```bash
python3 setup.py build_ext --inplace
pytest tests/compat -q 2>&1 | tee /tmp/tests-current.txt | tail -3
tail -1 docs/design/baseline/tests-baseline.txt | tr -d '=' | sed 's/ in .*//' | xargs
tail -1 /tmp/tests-current.txt | tr -d '=' | sed 's/ in .*//' | xargs
pytest tests/compat --collect-only -q | grep -oE '[0-9]+ tests collected'
grep -oE '[0-9]+ tests collected' docs/design/baseline/collect-count.txt
```

PASS if and only if ALL of:

1. The build exits 0.
2. The current summary line equals the baseline summary line with the `passed` count increased
   by **exactly 21** and every other count (failed / xfailed / skipped / error) unchanged.
   Worked example: if the baseline line normalizes to `2200 passed, 72 xfailed`, the current
   line must normalize to exactly `2221 passed, 72 xfailed`.
3. The current collected count equals the baseline collected count plus **exactly 21**.
   Worked example: baseline `2272 tests collected` → current `2293 tests collected`.

Any other difference (passed +20, passed +22, a new `failed`, a new `skipped`, ...) is a
FAILURE — go to STANDARD FAIL. The number 21 is fixed by the content of
`tests/compat/test_device_parity.py` given verbatim in step 5-6: 5 surface tests + 16 parity
cases × 1 device (`cpu`). Do not adjust it.

### 0.4 STANDARD FAIL (the only failure procedure)

1. Discard the step completely. `<new files>` below means the exact new files/directories the
   current step created (each step's **On failure** names them):

   ```bash
   git restore --staged --worktree .
   git status --short
   ```

   For every `??` line that `git status --short` still shows, remove it explicitly by path,
   one command per path (never run `git clean -fd` without a path):

   ```bash
   git clean -fd -- <path>
   ```

2. Confirm the tree is green again:

   ```bash
   python3 setup.py build_ext --inplace
   pytest tests/compat -q | tail -3
   ```

   The summary must match the baseline again (STANDARD VERIFY-A form; after step 5-6 has been
   committed, VERIFY-B form).

3. In `docs/design/PROGRESS.md`, keep the step's checkbox `[ ]`, append ` **BLOCKED**` at the
   end of the step's line, and write at most 3 lines describing the failure under the Phase 5
   `Notes:` field. Commit only PROGRESS.md:

   ```bash
   git add docs/design/PROGRESS.md && git commit -m "progress: mark 5-N BLOCKED"
   ```

   (replace `5-N` with the actual step ID).

4. **Stop working.** Never improvise a workaround (01 §6 item 7).

If the failed step's work was already committed, use `git revert --no-edit HEAD` per
`01-rules-and-verification.md` §6 item 3 instead of `git restore`.

### 0.5 Commit and PROGRESS discipline

Exactly as `01-rules-and-verification.md` §7 and §8: one work commit per step with message
`refactor(phase5-N): <what happened>`, staged explicitly by path; then a separate
`progress: complete 5-N` commit that flips the checkbox and records
`git rev-parse --short HEAD` + `date +%F`.

### 0.6 File map of this phase

| Step | New files | Edited files |
|---|---|---|
| 5-0 | — | `docs/design/PROGRESS.md`, `docs/README.md` |
| 5-1 | — | `cpp/mtorch/core/tensor.h`, `cpp/mtorch/core/tensor_core.cpp` |
| 5-2 | `cpp/mtorch/core/detail/allocator.h`, `cpp/mtorch/core/detail/allocator.cpp` | — |
| 5-3 | — | `cpp/mtorch/core/tensor.h`, `cpp/mtorch/core/tensor_core.cpp` |
| 5-4 | `cpp/mtorch/core/device_transfer.cpp` | `cpp/mtorch/core/tensor.h`, `cpp/mtorch/core/tensor_core.cpp`, `cpp/mtorch/python/py_tensor_type.cpp` |
| 5-5 | `cpp/mtorch/python/py_device.cpp` | `cpp/mtorch/python/py_common.cpp`, `cpp/mtorch/python/py_registry.h`, `cpp/mtorch/python/module_init.cpp`, `mtorch/__init__.py`, `mtorch/cuda/__init__.py`, `mtorch/mps.py` |
| 5-6 | `tests/compat/test_device_parity.py` | — |
| 5-7 | — | `docs/design/baseline/tests-baseline.txt`, `docs/design/baseline/collect-count.txt` |

All new `.cpp` files are picked up automatically by the setup.py glob (gate check above).

## 1. Verified facts (grep-verified; re-verify each before relying on it)

Every fact below was verified against the tree on 2026-07-09. The "verify with" commands are
written against the **post-Phase-4 tree** (the tree this guide executes on). Parenthetical
hints give the pre-split location where the same code lived when this guide was written; use
them only if a post-split grep finds nothing (then follow 01 §9 — check `git log`, and if the
symbol is truly gone, STANDARD FAIL, do not guess).

1. **`DeviceType` exists with members `CPU` and `Metal`; `CUDA` is absent.** In
   `cpp/mtorch/core/tensor.h` (unchanged by Phases 1–4):

   ```cpp
   enum class DeviceType {
     CPU,
     Metal,
   };
   ```

   Verify with: `grep -n -A 4 'enum class DeviceType' cpp/mtorch/core/tensor.h`
   Enumerator order matters: `CPU` = 0, `Metal` = 1; step 5-1 appends `CUDA` = 2, and the
   allocator registry in step 5-2 indexes a 3-slot table with these values.

2. **A `Device` struct equivalent already exists — no `cpp/mtorch/core/device.h` is created.**
   In `cpp/mtorch/core/tensor.h`, directly below the enum:

   ```cpp
   struct Device {
     DeviceType type = DeviceType::CPU;
     int64_t index = -1;
   };
   ```

   Verify with: `grep -n -A 4 'struct Device' cpp/mtorch/core/tensor.h`
   Note for cross-referencing the shared architecture contract (and guides 09/10): the
   contract's `struct Device { DeviceType type; int index; }` maps to this existing struct;
   the index field is `int64_t`, and `index == -1` means "no index given" (torch's
   `device.index is None`). The CPU device is never indexed.

3. **Device helper functions already exist** and live in `cpp/mtorch/core/tensor_core.cpp`
   after the Phase 2b split (moved there by step 2b-2; pre-split: `cpp/mtorch/core/tensor.cpp`
   ~L8031–L8060): `device_type_name` (returns `"cpu"` for `CPU` and — already — `"mps"` for
   `Metal`), `device_name` (appends `:<index>` when `index >= 0`), `cpu_device()` (returns
   `{CPU, -1}`), `metal_device(int64_t index = 0)`, `devices_equal`.
   Verify with: `grep -n 'std::string device_type_name\|Device cpu_device\|Device metal_device\|bool devices_equal' cpp/mtorch/core/tensor_core.cpp`

4. **`Storage` already carries a `Device device` tag** (so the contract item "Storage gains a
   device tag" is about *wiring*, not about adding the field), and **the current CPU
   allocation path is a `std::vector<uint8_t> bytes` member**:

   ```cpp
   struct Storage {
     ScalarType dtype;
     Device device;
     std::vector<uint8_t> bytes;
     ...
     Storage(ScalarType dtype, int64_t elements, Device device = Device{}, bool zero_initialize = true);
   ```

   The constructor body (in `tensor_core.cpp` post-2b-2; pre-split tensor.cpp ~L8071) is:

   ```cpp
   Storage::Storage(ScalarType dtype, int64_t elements, Device device, bool zero_initialize)
       : dtype(dtype), device(device) {
     require_cpu_device(device);
     const size_t byte_count = static_cast<size_t>(elements * element_size(dtype));
     if (zero_initialize) {
       bytes.assign(byte_count, 0);
     } else {
       bytes.resize(byte_count);
     }
   }
   ```

   Verify with: `grep -n -A 10 'Storage::Storage' cpp/mtorch/core/tensor_core.cpp`

5. **The vector `bytes` cannot be replaced.** Inventory of member operations on it (verify
   with `grep -rhoE 'bytes\.[a-z_]+' cpp/mtorch/core/ cpp/mtorch/python/ | sort | uniq -c`):
   `.data()` ×146, `.assign()` ×1, `.resize()` ×1, `.size()` ×1 as of writing — plus the
   load/store templates in `cpp/mtorch/core/detail/storage.h` (post 2b-1d) take
   `const std::vector<uint8_t>&` / `std::vector<uint8_t>&` parameters and are called with
   `tensor.storage->bytes` at hundreds of sites. Therefore this phase keeps `bytes` exactly
   as-is for CPU storages and adds a **separate allocator-owned buffer for non-CPU storages**
   (step 5-3). This is the contract's "behavior-neutral swap" reconciled with the code:
   the CPU path is preserved exactly (the strictest reading of behavior-neutral), and all
   non-CPU allocation routes through the registry. A non-CPU storage has an **empty** `bytes`
   vector; CPU kernels that read `storage->bytes` therefore cannot silently touch device
   tensors (they would index an empty vector, and every public entry point is expected to
   route via the §4 hook before that can happen — see the Phase 6 note in §5.3).

6. **`Storage` objects are never copied** — every construction site is
   `std::make_shared<Storage>(...)` (6 sites pre-split) or the constructor itself.
   Verify with: `grep -rn 'Storage(' cpp/mtorch/core/*.cpp cpp/mtorch/python/*.cpp | grep -v 'make_shared<Storage>' | grep -v 'Storage::Storage'`
   — expected: no output (only the constructor definition matches otherwise). This makes it
   safe for step 5-3 to add a destructor and delete the copy operations.

7. **`Tensor` derives its `device` member from its storage** — the `Tensor` constructor
   initializer list contains `device(this->storage->device)` (tensor_core.cpp post-2b-2;
   pre-split tensor.cpp ~L8114). So tagging the storage tags the tensor; nothing else to do.
   Verify with: `grep -n 'device(this->storage->device)' cpp/mtorch/core/tensor_core.cpp`

8. **`to()` is the only public dtype/device conversion** —
   `TensorPtr to(const TensorPtr& input, ScalarType dtype, Device device, bool copy = false);`
   in tensor.h; its implementation (tensor_core.cpp post-2b-2; pre-split ~L8479) starts with
   `require_cpu_device(device);`. `require_cpu_device` lives in `detail/common.h` (post
   2b-1b) and throws `std::runtime_error("Metal/MPS device execution is not implemented yet")`.
   Verify with: `grep -n -A 4 'TensorPtr to(const TensorPtr& input, ScalarType dtype, Device device, bool copy)' cpp/mtorch/core/tensor_core.cpp`

9. **The single Python-visible gate for non-CPU devices is `device_from_py`** in
   `cpp/mtorch/python/py_common.cpp` (moved by 2a-2; pre-split module.cpp ~L1119). Today it
   maps any text containing `"cpu"` to `cpu_device()` and **throws
   `NotImplementedException("Metal/MPS device execution is not implemented yet")` /
   `NotImplementedException("CUDA device execution is not implemented")`** for mps/metal/cuda
   text. `NotImplementedException` is translated to Python `NotImplementedError` by
   `translate_exception` (py_common.cpp post-2a-2; pre-split module.cpp ~L792). **Every**
   binding that accepts a device (`FactoryOptions.device` consumers, `py_tensor`,
   `parse_to_request` for `Tensor.to`, `*_like` factories, ...) funnels through
   `device_from_py`. This is why `mtorch.zeros((1,), device="mps")` raises
   `NotImplementedError` today, and why editing this one function (step 5-5) plus the storage
   path (5-3) is sufficient plumbing for the whole factory surface.
   Verify with: `grep -n 'Device device_from_py' cpp/mtorch/python/py_common.cpp` and
   `grep -rn 'device_from_py' cpp/mtorch/python/*.cpp | wc -l` (20+ call sites).

10. **`FactoryOptions` already carries `device`** (a `PyObject*`, parsed per call with
    `device_from_py`) — `factory_options_from_args` in py_common.cpp (pre-split module.cpp
    ~L1367/L1386) recognizes the `device` keyword. **No factory binding needs any edit in this
    phase**; the `device=` kwarg plumbs through end-to-end already.
    Verify with: `grep -n 'struct FactoryOptions' -A 6 cpp/mtorch/python/py_common.h` (or
    `.cpp`, wherever 2a-2 placed the struct) — the `PyObject* device = Py_None;` member exists.

11. **Tensor bindings**: `Tensor_get_device` returns
    `PyUnicode_FromString(mtorch::device_name(...))` (a plain string like `"cpu"` or
    `"mps:0"`); `Tensor_cpu` returns `self` unconditionally (incref, no copy); `Tensor_to`
    parses via `parse_to_request` then calls `mtorch::to(...)`. All three live in
    `cpp/mtorch/python/py_tensor_type.cpp` after step 2a-15 (pre-split module.cpp ~L9043,
    ~L11930s, and the `Tensor_to` definition near the method table).
    Verify with: `grep -n 'Tensor_get_device\|PyObject\* Tensor_cpu\|PyObject\* Tensor_to' cpp/mtorch/python/py_tensor_type.cpp`

12. **The Python `device` class already exists and is a `str` subclass** in the root
    `mtorch/__init__.py` (not moved by Phase 1; verify with
    `grep -n "class device(str)" mtorch/__init__.py`). It normalizes `"metal"` → `"mps"`,
    accepts only `{"cpu", "mps"}` (raising `NotImplementedError` otherwise), and already has
    `.type`, `.index` (`None` when no index), `__repr__` = `device(type='mps')` /
    `device(type='mps', index=0)`, `__str__`, and `str`-inherited `__eq__` (this is what makes
    the pinned assertion `tensor.device == "cpu"` true). The only Phase 5 change is adding
    `"cuda"` to the accepted set. The `Tensor.device` property patch (wrapping the `_C` string
    into this class) lives in `mtorch/_tensor_patches.py` after step 1-7 — it needs **no**
    change.

13. **The Python availability stubs post-Phase-1** (created by steps 1-4-c/1-4-d):
    `mtorch/cuda/__init__.py` defines `def is_available(): return False` (plus
    `device_count`, `empty_cache`, `manual_seed`, `is_bf16_supported`, `manual_seed_all`);
    `mtorch/mps.py` defines `is_available`/`is_built` returning `False`;
    `mtorch/backends/mps.py` re-exports both names from `mtorch.mps` — so rewiring
    `mtorch/mps.py` automatically rewires `mtorch.backends.mps`.

14. **`tests/test_device.py` pins exactly this behavior** (outside `tests/compat`; it is NOT
    part of the compat baseline, but Iron Rule 1 forbids changing it, and this phase must keep
    it green):

    - `mtorch.zeros((2, 3), dtype=mtorch.float32, device=mtorch.device("cpu")).device == "cpu"`
    - `str(mtorch.device("metal")) == "mps"`
    - `mtorch.zeros((1,), device=mtorch.device("mps"))` raises `NotImplementedError`

    Run it with `pytest tests/test_device.py -q` (expected: `2 passed`).

15. **Harness comparison helpers** in `tests/compat/harness.py` (unchanged by Phase 4):
    `assert_values_compatible(reference, expected, actual, *, path, rtol=None, atol=None,
    check_stride=True)` — when `expected` is a torch (`reference`) tensor it delegates to
    `_assert_tensor_compatible`, which compares shape, dtype name, device type,
    `requires_grad`, optional stride, and values via `reference.testing.assert_close`; and
    `_coerce_to_reference_tensor(reference, value, dtype)` which converts an mtorch tensor to
    a torch tensor (via `.detach()/.cpu()/.tolist()`). Step 5-6 reuses exactly these two.
    Verify with: `grep -n 'def assert_values_compatible\|def _coerce_to_reference_tensor' tests/compat/harness.py`

16. **Binding method-table registry** (from step 2a-3): each binding section file exposes an
    `extern PyMethodDef k<Name>Methods[]` declared in `cpp/mtorch/python/py_registry.h`, and
    `build_module_methods()` in `cpp/mtorch/python/module_init.cpp` concatenates the tables
    with `append_table(...)` calls. Step 5-5 adds one new section (`py_device.cpp`,
    `kDeviceMethods`) through this exact mechanism.
    Verify with: `grep -n 'append_table' cpp/mtorch/python/module_init.cpp`

17. **The exception mapping** in `translate_exception` (py_common.cpp): `TypeErrorException` →
    `TypeError`, `NotImplementedException` → `NotImplementedError`, `std::out_of_range` →
    `IndexError`, `std::invalid_argument` → `ValueError`, any other `std::exception` →
    `RuntimeError`. Phase 5 keeps `NotImplementedException` as the *binding-layer* way to
    signal an unregistered backend; the core throws `std::runtime_error` in defensive paths
    that the bindings gate off (see steps 5-3 and 5-5).

18. **All chosen parity ops exist and run today** (verified by executing them):
    `+`, `-`, `*`, `/`, unary `-`, scalar `+`/`*`, `@` with `.t()`, `.sum()`, `.mean()`,
    `mtorch.exp`, `mtorch.relu`, `mtorch.softmax(x, dim)`, `mtorch.cat([a, b], dim)`,
    `.transpose(0, 1).contiguous()`, `.reshape((r, c))`, plus `.tolist()`, `.detach()`,
    `.cpu()`, `.to("cpu")` on `mtorch.Tensor`. The step 5-6 test table uses only these.

## 2. Background primer (read once before step 5-1)

**The device model.** A "device" is where a tensor's bytes physically live: `"cpu"` = ordinary
process memory, `"mps"` = memory managed by Apple's Metal API, `"cuda"` = memory on an NVIDIA
GPU. PyTorch encodes this as a `(type, index)` pair — the index distinguishes multiple GPUs
(`"cuda:1"`), and is `None`/absent for singletons like the CPU. mtorch already mirrors this
with `Device{DeviceType, int64_t index}` where `-1` plays the role of "no index". Every tensor
answers `tensor.device`, and computations require their operands on the same device — that
discipline is what makes it possible to know, per op, which backend must run it.

**What an allocator is.** CPU code gets memory from the heap (`std::vector`, `new`). GPU
memory cannot be obtained that way — each backend has its own allocation call (Metal:
`MTLDevice newBuffer...`; CUDA: `cudaMalloc`) and its own copy primitives. An *allocator* is a
small interface — allocate / deallocate / copy-in / copy-out / copy-on-device — that hides
those per-backend calls behind virtual functions. Core code then only ever says "give me N
bytes on device X", and a *registry* maps `DeviceType` → allocator instance. A backend
"exists" at runtime exactly when its allocator is registered; that single fact drives
`is_available()` all the way up in Python.

**Why the device tag lives in `Storage`.** Many tensors can share one buffer (views:
`reshape`, `transpose`, slicing). The buffer's location is a property of the buffer, not of
the view — so the authoritative tag sits on `Storage`, and `Tensor::device` is copied from
`storage->device` at construction (§1 fact 7). Tagging anywhere else would let two views of
one buffer disagree about where the bytes are.

**Unified vs. discrete memory, and why it shapes the dispatch policy.** On Apple Silicon, the
CPU and GPU share one physical memory pool ("unified memory"); a Metal buffer allocated with
shared storage mode is directly readable and writable by CPU code — that is what
`Allocator::host_accessible() == true` means, and it is why a Metal backend may legally fall
back to running the existing CPU kernel *in place* on a device tensor for ops it has no GPU
kernel for. A discrete CUDA GPU has its own memory across a PCIe bus: host pointers and device
pointers are different address spaces, CPU code dereferencing a device pointer crashes, and
every host<->device move is an explicit `cudaMemcpy`. Hence `host_accessible() == false` for
CUDA, no silent fallback (it would require a hidden round-trip copy — a performance and
semantics trap), and instead a loud
`"mtorch: operator '<name>' is not implemented for device 'cuda'"` error plus an explicit
transfer API (`to_device`, `Tensor.to`, `.cpu()`).

**Why Metal is called `"mps"`.** PyTorch's Apple-GPU backend is named after the Metal
Performance Shaders framework, so the torch-compatible *device string* is `"mps"`. mtorch's
C++ enum keeps the historical name `DeviceType::Metal`, and `device_type_name` already maps it
to `"mps"` (§1 fact 3). Keep that split: C++ says `Metal`, every user-visible string says
`"mps"` (with `"metal"` accepted as an input alias, pinned by `tests/test_device.py`).

## 3. The steps

## Step 5-0: Register Phase 5 in PROGRESS.md and the docs index

**Goal**: `docs/design/PROGRESS.md` gains the Phase 5 checklist (so the 01 §1 work cycle can
drive the remaining steps) and `docs/README.md` indexes this guide.

**Preconditions**: The "When to start" gate above passed, including
`grep -c -- '- \[ \]' docs/design/PROGRESS.md` printing `0` and the 01 §3 environment check.
`git status --porcelain` prints nothing.

**Actions**:

1. In `docs/design/PROGRESS.md`, find the line `## Overall completion check` (verify with
   `grep -n '## Overall completion check' docs/design/PROGRESS.md`) and insert directly ABOVE
   it, keeping one blank line before and after the inserted block:

   ```markdown
   ## Phase 5: Device/backend infrastructure (procedure: 08-phase5-device-layer.md)

   - [ ] 5-0 Register Phase 5 in PROGRESS.md and the docs index — commit: / date:
   - [ ] 5-1 DeviceType::CUDA and cuda_device() — commit: / date:
   - [ ] 5-2 detail/allocator.{h,cpp} (Allocator interface, CPU allocator, registry) — commit: / date:
   - [ ] 5-3 Storage device wiring (allocator-backed non-CPU storage, data()/nbytes()) — commit: / date:
   - [ ] 5-4 to_device() transfer API + to()/Tensor.cpu() rerouting — commit: / date:
   - [ ] 5-5 Python device surface (device class, _backend_available, mps/cuda wiring) — commit: / date:
   - [ ] 5-6 tests/compat/test_device_parity.py scaffolding — commit: / date:
   - [ ] 5-7 Phase gate (full tests + full benchmark + baseline re-record) — commit: / date:

   Notes:
   ```

2. Check whether `docs/README.md` already indexes this guide (the row may have been added
   when the guide was authored):

   ```bash
   grep -c '08-phase5-device-layer' docs/README.md
   ```

   If this prints `1` or more, the index row already exists — skip the rest of this action.
   Only if it prints `0`: find the row for `design/07-phase4-test-infra.md` (verify with
   `grep -n '07-phase4-test-infra' docs/README.md`) and insert directly BELOW that table row:

   ```markdown
   | [design/08-phase5-device-layer.md](design/08-phase5-device-layer.md) | Phase 5: device/backend infrastructure (Device/allocator registry, transfer API, Python device surface, parity test scaffolding) | While working on Phase 5 |
   ```

**Verification**:

```bash
grep -c -- '- \[ \]' docs/design/PROGRESS.md      # 8  (the eight new Phase 5 steps)
grep -c '08-phase5-device-layer' docs/README.md    # 1
```

No build/test run is needed (documentation-only change), but run
`python3 setup.py build_ext --inplace` anyway per the 01 §1 cycle (it finishes in seconds).

**On failure**: STANDARD FAIL. No new files were created; the `git restore` alone reverts both
edits.

**Commit**:

```bash
git add docs/design/PROGRESS.md docs/README.md && git commit -m "refactor(phase5-0): register Phase 5 checklist and docs index entry"
```

Then complete 5-0 in PROGRESS.md per §0.5.

## Step 5-1: `DeviceType::CUDA` and `cuda_device()`

**Goal**: The enum gains the `CUDA` member; `device_type_name` maps it to `"cuda"`; a
`cuda_device()` helper mirrors `metal_device()`. No behavior changes (nothing constructs a
CUDA `Device` yet).

**Background:** an enum member costs nothing at runtime; adding it now (before the allocator
registry in 5-2) lets the registry size its table (`CPU`=0, `Metal`=1, `CUDA`=2) and lets
every later step use `DeviceType::CUDA` without forward references. The overview (00 §
non-goals) said `DeviceType::Metal` is "declared but unused" — §1 fact 1 verified the enum's
actual spelling, and facts 3/9 show `Metal` *is* consumed by `device_type_name`/
`metal_device`/`device_from_py`, so `CUDA` gets the same three touchpoints (the third comes in
step 5-5).

**Preconditions**: 5-0 committed; `git status --porcelain` prints nothing; and:

```bash
grep -c 'CUDA' cpp/mtorch/core/tensor.h        # 0
grep -n 'Metal,' cpp/mtorch/core/tensor.h      # prints exactly one line (inside the enum)
```

**Actions**:

1. `cpp/mtorch/core/tensor.h` — extend the enum. BEFORE:

   ```cpp
   enum class DeviceType {
     CPU,
     Metal,
   };
   ```

   AFTER:

   ```cpp
   enum class DeviceType {
     CPU,
     Metal,
     CUDA,
   };
   ```

2. `cpp/mtorch/core/tensor.h` — declare the helper. Locate with
   `grep -n 'Device metal_device' cpp/mtorch/core/tensor.h`. BEFORE:

   ```cpp
   Device metal_device(int64_t index = 0);
   ```

   AFTER:

   ```cpp
   Device metal_device(int64_t index = 0);
   Device cuda_device(int64_t index = 0);
   ```

3. `cpp/mtorch/core/tensor_core.cpp` — extend `device_type_name`. Locate with
   `grep -n -A 8 'std::string device_type_name' cpp/mtorch/core/tensor_core.cpp`
   (pre-split hint: tensor.cpp ~L8031). BEFORE:

   ```cpp
   std::string device_type_name(DeviceType type) {
     switch (type) {
       case DeviceType::CPU:
         return "cpu";
       case DeviceType::Metal:
         return "mps";
     }
     return "cpu";
   }
   ```

   AFTER:

   ```cpp
   std::string device_type_name(DeviceType type) {
     switch (type) {
       case DeviceType::CPU:
         return "cpu";
       case DeviceType::Metal:
         return "mps";
       case DeviceType::CUDA:
         return "cuda";
     }
     return "cpu";
   }
   ```

4. `cpp/mtorch/core/tensor_core.cpp` — define the helper. Locate with
   `grep -n -A 3 'Device metal_device' cpp/mtorch/core/tensor_core.cpp`
   (pre-split hint: tensor.cpp ~L8053). BEFORE:

   ```cpp
   Device metal_device(int64_t index) {
     return Device{DeviceType::Metal, index};
   }
   ```

   AFTER:

   ```cpp
   Device metal_device(int64_t index) {
     return Device{DeviceType::Metal, index};
   }

   Device cuda_device(int64_t index) {
     return Device{DeviceType::CUDA, index};
   }
   ```

**Verification**:

```bash
grep -c 'CUDA,' cpp/mtorch/core/tensor.h                                    # 1  (the enum member)
grep -c 'cuda_device' cpp/mtorch/core/tensor.h                              # 1  (the declaration)
grep -c 'cuda_device' cpp/mtorch/core/tensor_core.cpp                       # 1  (the definition)
grep -c 'DeviceType::CUDA' cpp/mtorch/core/tensor_core.cpp                  # 2  (switch case + cuda_device body)
```

Then run STANDARD VERIFY-A (§0.3), and:

```bash
python3 -c "import mtorch; t = mtorch.zeros((1,), dtype=mtorch.float32); print(t.device)"   # cpu
pytest tests/test_device.py -q | tail -1                                                    # 2 passed ...
```

**On failure**: STANDARD FAIL. No new files; `git restore --staged --worktree .` reverts both
edited files.

**Commit**:

```bash
git add cpp/mtorch/core/tensor.h cpp/mtorch/core/tensor_core.cpp && git commit -m "refactor(phase5-1): add DeviceType::CUDA and cuda_device()"
```

Then complete 5-1 in PROGRESS.md per §0.5.

## Step 5-2: `detail/allocator.{h,cpp}` — the Allocator interface, the CPU allocator, and the registry

**Goal**: The two new files exist, compile, and export `mtorch::detail::Allocator`,
`allocator_for`, `register_allocator`, and `unsupported_device`. Nothing calls them yet
(that starts in 5-3), so this step is purely additive.

**Background:** this is the seam between the device-agnostic core and the backends. The six
virtual functions are the complete set of memory operations a backend must provide: get/free a
buffer, move bytes host→device, device→host, and device→device, plus one capability flag
(`host_accessible`) that tells the dispatcher whether CPU code may touch the buffer directly
(unified memory) or not (discrete GPU). The registry is a fixed 3-slot table indexed by the
`DeviceType` value; the CPU slot is pre-filled with a singleton whose operations are exactly
the heap + `memcpy` semantics `Storage` has always had, so "CPU allocator" is a wrapper name
for the existing behavior, not new behavior. Phases 6/7 each implement `Allocator` once and
call `register_allocator` at module init — that single call flips `is_available()` to `True`
everywhere.

**Preconditions**: 5-1 committed; `git status --porcelain` prints nothing; and:

```bash
test ! -e cpp/mtorch/core/detail/allocator.h && test ! -e cpp/mtorch/core/detail/allocator.cpp && echo PRECONDITION-OK
```

Expected: `PRECONDITION-OK`.

**Actions**:

1. Create `cpp/mtorch/core/detail/allocator.h` with exactly this content:

   ```cpp
   #pragma once

   #include <cstddef>

   #include "mtorch/core/tensor.h"

   namespace mtorch::detail {

   // Backend memory-management interface (Phase 5 device infrastructure; see
   // docs/design/08-phase5-device-layer.md). One long-lived Allocator instance
   // per DeviceType is registered in a process-wide registry. Phase 6 (Metal)
   // and Phase 7 (CUDA) implement this interface and call register_allocator()
   // during module initialization; a backend is "available" exactly when its
   // allocator is registered.
   struct Allocator {
     virtual ~Allocator();
     // Returns a buffer of `nbytes` bytes on this allocator's device.
     // May return nullptr when nbytes == 0. Throws on allocation failure.
     virtual void* allocate(size_t nbytes) = 0;
     // Releases a buffer previously returned by allocate() on the same allocator.
     virtual void deallocate(void* p, size_t nbytes) = 0;
     // dst is a device pointer of this allocator; src is a host pointer.
     virtual void copy_from_host(void* dst, const void* src, size_t n) = 0;
     // dst is a host pointer; src is a device pointer of this allocator.
     virtual void copy_to_host(void* dst, const void* src, size_t n) = 0;
     // dst and src are both device pointers of this allocator (same backend).
     virtual void copy_on_device(void* dst, const void* src, size_t n) = 0;
     // True when the host CPU can directly dereference pointers returned by
     // allocate() (unified memory, e.g. Metal on Apple Silicon). False for
     // discrete-memory backends (CUDA).
     virtual bool host_accessible() const = 0;
   };

   // Returns the registered allocator for `type`, or nullptr when the backend
   // is not registered. The CPU allocator is always registered.
   Allocator* allocator_for(DeviceType type);

   // Registers (or replaces) the allocator for `type`. The registry does not
   // take ownership; backends must register a long-lived singleton.
   void register_allocator(DeviceType type, Allocator* allocator);

   // Throws std::runtime_error with the fixed message
   //   mtorch: operator '<op>' is not implemented for device '<device>'
   // Used at the public-op dispatch hook points (see the Phase 6/7 convention in
   // docs/design/08-phase5-device-layer.md section 4).
   [[noreturn]] void unsupported_device(const char* op, DeviceType type);

   }  // namespace mtorch::detail
   ```

2. Create `cpp/mtorch/core/detail/allocator.cpp` with exactly this content:

   ```cpp
   #include "mtorch/core/detail/allocator.h"

   #include <cstring>
   #include <new>
   #include <stdexcept>
   #include <string>

   namespace mtorch::detail {

   Allocator::~Allocator() = default;

   namespace {

   // The CPU allocator wraps the process heap with plain memcpy transfers —
   // the same semantics Storage's std::vector<uint8_t> buffer has always had
   // (zero-initialization stays the caller's job, exactly as with
   // bytes.resize()). host_accessible() is trivially true.
   struct CpuAllocator final : Allocator {
     void* allocate(size_t nbytes) override {
       if (nbytes == 0) {
         return nullptr;
       }
       return ::operator new(nbytes);
     }
     void deallocate(void* p, size_t) override {
       ::operator delete(p);
     }
     void copy_from_host(void* dst, const void* src, size_t n) override {
       if (n > 0) {
         std::memcpy(dst, src, n);
       }
     }
     void copy_to_host(void* dst, const void* src, size_t n) override {
       if (n > 0) {
         std::memcpy(dst, src, n);
       }
     }
     void copy_on_device(void* dst, const void* src, size_t n) override {
       if (n > 0) {
         std::memcpy(dst, src, n);
       }
     }
     bool host_accessible() const override {
       return true;
     }
   };

   // Keep in sync with enum class DeviceType in mtorch/core/tensor.h:
   // CPU = 0, Metal = 1, CUDA = 2.
   constexpr int kDeviceTypeCount = 3;

   Allocator** allocator_table() {
     static CpuAllocator cpu_allocator;
     static Allocator* table[kDeviceTypeCount] = {&cpu_allocator, nullptr, nullptr};
     return table;
   }

   }  // namespace

   Allocator* allocator_for(DeviceType type) {
     const int slot = static_cast<int>(type);
     if (slot < 0 || slot >= kDeviceTypeCount) {
       return nullptr;
     }
     return allocator_table()[slot];
   }

   void register_allocator(DeviceType type, Allocator* allocator) {
     const int slot = static_cast<int>(type);
     if (slot < 0 || slot >= kDeviceTypeCount) {
       throw std::invalid_argument("register_allocator: unknown DeviceType");
     }
     allocator_table()[slot] = allocator;
   }

   [[noreturn]] void unsupported_device(const char* op, DeviceType type) {
     throw std::runtime_error(
         std::string("mtorch: operator '") + op + "' is not implemented for device '" +
         device_type_name(type) + "'");
   }

   }  // namespace mtorch::detail
   ```

**Verification**:

```bash
python3 setup.py build_ext --inplace
find build -name 'allocator.o' | head -1
```

The build exits 0 and `find` prints one path (the setup.py glob picked the new file up). Then
STANDARD VERIFY-A (§0.3).

**On failure**: STANDARD FAIL. New files to remove:

```bash
git clean -fd -- cpp/mtorch/core/detail/allocator.h
git clean -fd -- cpp/mtorch/core/detail/allocator.cpp
```

**Commit**:

```bash
git add cpp/mtorch/core/detail/allocator.h cpp/mtorch/core/detail/allocator.cpp && git commit -m "refactor(phase5-2): add allocator interface, CPU allocator, and registry"
```

Then complete 5-2 in PROGRESS.md per §0.5.

## Step 5-3: Storage device wiring (allocator-backed non-CPU storage)

**Goal**: `Storage` can hold its data either in the legacy CPU vector (`bytes`, unchanged) or
in an allocator-owned device buffer, chosen by its existing `device` tag; uniform accessors
`data()` / `nbytes()` work for both. Default remains CPU everywhere, so behavior is unchanged
— the compat suite must still match the baseline exactly.

**Background:** why keep `bytes` at all? §1 fact 5: ~150 call sites plus the `detail/storage.h`
load/store templates are typed against `std::vector<uint8_t>&`. Replacing the vector with raw
allocator memory would be a mass rewrite of exactly the kind Phase 5 must not do. Instead the
storage becomes a two-compartment box: CPU storages use `bytes` precisely as before (same
`assign`/`resize` calls, same zero-initialization), and non-CPU storages — which cannot exist
until a backend registers an allocator — use `device_data`. The new `data()`/`nbytes()`
accessors give backends and the transfer code one way to reach either compartment. The
destructor returns the device buffer to its allocator; copies of `Storage` are deleted
(fact 6: nothing copies it) so the raw pointer cannot be double-freed.

**Preconditions**: 5-2 committed; `git status --porcelain` prints nothing; and:

```bash
grep -c 'std::vector<uint8_t> bytes;' cpp/mtorch/core/tensor.h     # 1
grep -n 'require_cpu_device(device);' cpp/mtorch/core/tensor_core.cpp
```

The second command prints exactly two lines: one inside `Storage::Storage` and one inside
`to()` (confirm by line proximity to `grep -n 'Storage::Storage\|TensorPtr to(' cpp/mtorch/core/tensor_core.cpp`).
This step edits only the one inside `Storage::Storage`; the one inside `to()` is edited in 5-4.

**Actions**:

1. `cpp/mtorch/core/tensor.h` — forward-declare the allocator. Locate the single line
   `namespace mtorch {` (`grep -n '^namespace mtorch {' cpp/mtorch/core/tensor.h`). BEFORE:

   ```cpp
   namespace mtorch {

   enum class ScalarType {
   ```

   AFTER:

   ```cpp
   namespace mtorch {

   namespace detail {
   struct Allocator;
   }  // namespace detail

   enum class ScalarType {
   ```

2. `cpp/mtorch/core/tensor.h` — add the storage members. Locate with
   `grep -n -A 4 'struct Storage {' cpp/mtorch/core/tensor.h`. BEFORE:

   ```cpp
   struct Storage {
     ScalarType dtype;
     Device device;
     std::vector<uint8_t> bytes;
     uint64_t version = 0;
   ```

   AFTER:

   ```cpp
   struct Storage {
     ScalarType dtype;
     Device device;
     // CPU storages keep their data in `bytes` (the unchanged legacy path).
     // Non-CPU storages keep their data in `device_data`, allocated and owned
     // through `allocator` (a registered mtorch::detail::Allocator; see
     // core/detail/allocator.h). Exactly one of the two is in use, selected by
     // device.type; use data()/nbytes() for device-agnostic access.
     std::vector<uint8_t> bytes;
     detail::Allocator* allocator = nullptr;
     void* device_data = nullptr;
     int64_t device_nbytes = 0;
     uint64_t version = 0;
   ```

3. `cpp/mtorch/core/tensor.h` — declare the new members. Locate with
   `grep -n 'bool zero_initialize = true);' cpp/mtorch/core/tensor.h`. BEFORE:

   ```cpp
     Storage(ScalarType dtype, int64_t elements, Device device = Device{}, bool zero_initialize = true);

     int64_t numel() const;
   ```

   AFTER:

   ```cpp
     Storage(ScalarType dtype, int64_t elements, Device device = Device{}, bool zero_initialize = true);
     Storage(const Storage&) = delete;
     Storage& operator=(const Storage&) = delete;
     ~Storage();

     void* data();
     const void* data() const;
     int64_t nbytes() const;
     int64_t numel() const;
   ```

4. `cpp/mtorch/core/tensor_core.cpp` — add the include. The file starts with the SECTION-FILE
   template of 05 §0.8; locate the last detail include with
   `grep -n 'detail/elementwise.h' cpp/mtorch/core/tensor_core.cpp`. BEFORE:

   ```cpp
   #include "mtorch/core/detail/elementwise.h"
   ```

   AFTER:

   ```cpp
   #include "mtorch/core/detail/elementwise.h"
   #include "mtorch/core/detail/allocator.h"
   ```

5. `cpp/mtorch/core/tensor_core.cpp` — rewrite the constructor and add the new members.
   Locate with `grep -n -A 10 'Storage::Storage' cpp/mtorch/core/tensor_core.cpp`
   (pre-split hint: tensor.cpp ~L8071). BEFORE:

   ```cpp
   Storage::Storage(ScalarType dtype, int64_t elements, Device device, bool zero_initialize)
       : dtype(dtype), device(device) {
     require_cpu_device(device);
     const size_t byte_count = static_cast<size_t>(elements * element_size(dtype));
     if (zero_initialize) {
       bytes.assign(byte_count, 0);
     } else {
       bytes.resize(byte_count);
     }
   }
   ```

   AFTER:

   ```cpp
   Storage::Storage(ScalarType dtype, int64_t elements, Device device, bool zero_initialize)
       : dtype(dtype), device(device) {
     const size_t byte_count = static_cast<size_t>(elements * element_size(dtype));
     if (device.type == DeviceType::CPU) {
       // Unchanged legacy CPU path.
       if (zero_initialize) {
         bytes.assign(byte_count, 0);
       } else {
         bytes.resize(byte_count);
       }
       return;
     }
     allocator = detail::allocator_for(device.type);
     if (allocator == nullptr) {
       throw std::runtime_error(
           device_type_name(device.type) + " device is not implemented: no allocator registered");
     }
     device_data = allocator->allocate(byte_count);
     device_nbytes = static_cast<int64_t>(byte_count);
     if (zero_initialize && byte_count > 0) {
       std::vector<uint8_t> zero_bytes(byte_count, 0);
       allocator->copy_from_host(device_data, zero_bytes.data(), byte_count);
     }
   }

   Storage::~Storage() {
     if (allocator != nullptr && device_data != nullptr) {
       allocator->deallocate(device_data, static_cast<size_t>(device_nbytes));
       device_data = nullptr;
     }
   }

   void* Storage::data() {
     return device.type == DeviceType::CPU ? static_cast<void*>(bytes.data()) : device_data;
   }

   const void* Storage::data() const {
     return device.type == DeviceType::CPU ? static_cast<const void*>(bytes.data()) : device_data;
   }

   int64_t Storage::nbytes() const {
     return device.type == DeviceType::CPU ? static_cast<int64_t>(bytes.size()) : device_nbytes;
   }
   ```

   (Note: the `require_cpu_device(device);` call is deleted — for CPU devices it was a no-op,
   and for non-CPU devices the allocator lookup above is the new, more precise guard.
   `require_cpu_device` itself stays in `detail/common.h`; its 100+ other call sites are
   untouched.)

6. `cpp/mtorch/core/tensor_core.cpp` — make `numel()` storage-agnostic. Locate with
   `grep -n -A 3 'int64_t Storage::numel' cpp/mtorch/core/tensor_core.cpp`. BEFORE:

   ```cpp
   int64_t Storage::numel() const {
     return static_cast<int64_t>(bytes.size()) / element_size(dtype);
   }
   ```

   AFTER:

   ```cpp
   int64_t Storage::numel() const {
     return nbytes() / element_size(dtype);
   }
   ```

**Verification**: STANDARD VERIFY-A (§0.3) — this is the critical behavior-neutrality gate for
the storage swap — plus:

```bash
python3 - <<'PY'
import mtorch
t = mtorch.zeros((2, 2), dtype=mtorch.float32)
assert t.device == "cpu"
assert t.tolist() == [[0.0, 0.0], [0.0, 0.0]]
try:
    mtorch.zeros((1,), device="mps")
except NotImplementedError:
    print("phase5-3-check OK")   # still gated by device_from_py, unchanged so far
PY
pytest tests/test_device.py -q | tail -1     # 2 passed ...
```

**On failure**: STANDARD FAIL. No new files; `git restore --staged --worktree .` reverts both
edited files.

**Commit**:

```bash
git add cpp/mtorch/core/tensor.h cpp/mtorch/core/tensor_core.cpp && git commit -m "refactor(phase5-3): route non-CPU Storage allocation through the allocator registry"
```

Then complete 5-3 in PROGRESS.md per §0.5.

## Step 5-4: `to_device()` transfer API, `to()` rerouting, and `Tensor.cpu()`

**Goal**: `TensorPtr to_device(const Tensor& source, Device device)` exists in core (declared
in `tensor.h` — the explicitly allowed tensor.h addition of this phase), `to()` routes
cross-device requests through it, and the `Tensor.cpu()` binding transfers instead of blindly
returning `self`. On a CPU-only tree every one of these paths is either identical to before or
unreachable, so the compat suite must still match the baseline exactly.

**Background — why explicit transfer:** on a discrete-memory backend (CUDA) bytes move only
when someone calls a copy primitive; there is no page-fault magic. torch therefore exposes
movement as an explicit, user-visible operation (`Tensor.to(device)`, `.cpu()`), and mtorch
mirrors that with one core function, `to_device`, that all surfaces funnel through. The
transfer copies the **whole storage** and preserves the view metadata (sizes/strides/offset)
unchanged — this is the simplest scheme that is correct for every layout, because it never has
to repack non-contiguous views on a device that has no kernels for that. dtype conversion is
deliberately NOT part of `to_device`: `to()` stages any dtype change on the host (where all
casting kernels live) and calls `to_device` for the raw moves.

A note on the signature: the shared architecture contract names this function
`Tensor to_device(const Tensor&, Device)`. Every tensor-producing function in `tensor.h`
returns `TensorPtr` (§1 fact 8 and the whole factory family), so the mtorch spelling of that
contract entry is `TensorPtr to_device(const Tensor& source, Device device)` — same name, same
parameters, house return convention. Guides 09/10 consume it under this exact signature (see
§5).

**Preconditions**: 5-3 committed; `git status --porcelain` prints nothing; and:

```bash
grep -c 'to_device' cpp/mtorch/core/tensor.h                       # 0
test ! -e cpp/mtorch/core/device_transfer.cpp && echo PRECONDITION-OK
grep -c 'require_cpu_device(device);' cpp/mtorch/core/tensor_core.cpp   # 1  (the one left inside to())
```

**Actions**:

1. `cpp/mtorch/core/tensor.h` — declare the transfer API. Locate with
   `grep -n 'TensorPtr to(const TensorPtr& input' cpp/mtorch/core/tensor.h`. BEFORE:

   ```cpp
   TensorPtr to(const TensorPtr& input, ScalarType dtype, Device device, bool copy = false);
   ```

   AFTER:

   ```cpp
   TensorPtr to(const TensorPtr& input, ScalarType dtype, Device device, bool copy = false);
   // Phase 5 transfer API (docs/design/08-phase5-device-layer.md): copies the
   // whole storage to `device` and preserves sizes/strides/offset. Same-device
   // calls return a new Tensor sharing the existing storage. dtype conversion
   // is handled by to(), not here.
   TensorPtr to_device(const Tensor& source, Device device);
   ```

2. Create `cpp/mtorch/core/device_transfer.cpp` with exactly this content:

   ```cpp
   // Cross-device tensor transfer (Phase 5 device infrastructure).
   // See docs/design/08-phase5-device-layer.md.

   #include <cstddef>
   #include <memory>
   #include <stdexcept>

   #include "mtorch/core/tensor.h"
   #include "mtorch/core/detail/allocator.h"

   namespace mtorch {

   TensorPtr to_device(const Tensor& source, Device device) {
     Device target = device;
     if (target.type == DeviceType::CPU) {
       target.index = -1;  // the CPU device is never indexed
     }
     if (devices_equal(source.device, target)) {
       // Already resident: hand out a view on the same storage (mirrors
       // torch's no-op Tensor.to(device) for an already-resident tensor).
       return std::make_shared<Tensor>(
           source.storage, source.sizes, source.strides, source.offset, source.dtype,
           source.requires_grad && is_grad_enabled());
     }
     if (source.device.type != DeviceType::CPU && target.type != DeviceType::CPU &&
         source.device.type != target.type) {
       throw std::runtime_error(
           "mtorch: direct transfer between different accelerator backends is not supported; "
           "move the tensor through 'cpu' first");
     }

     // Copy the WHOLE storage and keep the view metadata (sizes/strides/offset)
     // untouched, so every layout stays valid on the target device.
     const Storage& src_storage = *source.storage;
     auto dst_storage = std::make_shared<Storage>(
         src_storage.dtype, src_storage.numel(), target, /*zero_initialize=*/false);
     const size_t transfer_bytes = static_cast<size_t>(src_storage.nbytes());
     if (transfer_bytes > 0) {
       if (source.device.type == DeviceType::CPU) {
         // host -> device (dst_storage's constructor guaranteed the allocator exists)
         detail::allocator_for(target.type)
             ->copy_from_host(dst_storage->data(), src_storage.data(), transfer_bytes);
       } else if (target.type == DeviceType::CPU) {
         // device -> host
         detail::allocator_for(source.device.type)
             ->copy_to_host(dst_storage->data(), src_storage.data(), transfer_bytes);
       } else {
         // device -> device, same backend, different index
         detail::allocator_for(target.type)
             ->copy_on_device(dst_storage->data(), src_storage.data(), transfer_bytes);
       }
     }
     return std::make_shared<Tensor>(
         dst_storage, source.sizes, source.strides, source.offset, source.dtype,
         source.requires_grad && is_grad_enabled());
   }

   }  // namespace mtorch
   ```

   (Note: no autograd edge is attached — a transferred tensor does not backpropagate into its
   source. That is sufficient for Phase 5, where non-CPU tensors cannot exist; Phase 6/7 own
   revisiting this if their workloads need a differentiable `.to()`. Recorded in §5.)

3. `cpp/mtorch/core/tensor_core.cpp` — reroute `to()`. Locate with
   `grep -n -A 5 'TensorPtr to(const TensorPtr& input, ScalarType dtype, Device device, bool copy)' cpp/mtorch/core/tensor_core.cpp`
   (pre-split hint: tensor.cpp ~L8479). BEFORE:

   ```cpp
   TensorPtr to(const TensorPtr& input, ScalarType dtype, Device device, bool copy) {
     require_cpu_device(device);
     if (devices_equal(input->device, device) && input->dtype == dtype && !copy) {
       return input;
     }
   ```

   AFTER:

   ```cpp
   TensorPtr to(const TensorPtr& input, ScalarType dtype, Device device, bool copy) {
     if (input->device.type != DeviceType::CPU || device.type != DeviceType::CPU) {
       // Phase 5 device-transfer path: dtype conversion always runs on the
       // host; to_device() performs the raw transfers. Unreachable while no
       // backend allocator is registered (device_from_py gates the bindings).
       if (devices_equal(input->device, device) && input->dtype == dtype && !copy) {
         return input;
       }
       TensorPtr host =
           input->device.type == DeviceType::CPU ? input : to_device(*input, cpu_device());
       TensorPtr converted = to(host, dtype, cpu_device(), copy);
       if (device.type == DeviceType::CPU) {
         return converted;
       }
       return to_device(*converted, device);
     }
     if (devices_equal(input->device, device) && input->dtype == dtype && !copy) {
       return input;
     }
   ```

   (The rest of the function body — everything after this early-return `if` — is untouched.
   For CPU→CPU calls the new branch is never entered, so behavior is bit-identical.)

4. `cpp/mtorch/python/py_tensor_type.cpp` — make `Tensor.cpu()` transfer. Locate with
   `grep -n -A 4 'PyObject\* Tensor_cpu' cpp/mtorch/python/py_tensor_type.cpp`
   (pre-split hint: module.cpp ~L11930s, directly above the `Tensor_to` definition). BEFORE:

   ```cpp
   PyObject* Tensor_cpu(PyTensor* self, PyObject*) {
     Py_INCREF(self);
     return reinterpret_cast<PyObject*>(self);
   }
   ```

   AFTER:

   ```cpp
   PyObject* Tensor_cpu(PyTensor* self, PyObject*) {
     if (self->value->get()->device.type == mtorch::DeviceType::CPU) {
       Py_INCREF(self);
       return reinterpret_cast<PyObject*>(self);
     }
     try {
       return wrap_tensor(mtorch::to_device(*self->value->get(), mtorch::cpu_device()));
     } catch (...) {
       translate_exception();
       return nullptr;
     }
   }
   ```

**Verification**:

```bash
python3 setup.py build_ext --inplace
python3 - <<'PY'
import mtorch
t = mtorch.ones((2, 2), dtype=mtorch.float32)
assert t.cpu() is t                       # cpu fast path is still identity
u = t.to("cpu")
assert u.device == "cpu" and u.tolist() == t.tolist()
v = t.to(dtype=mtorch.float64)
assert str(v.dtype) == "float64"
print("phase5-4-check OK")
PY
grep -c 'require_cpu_device(device);' cpp/mtorch/core/tensor_core.cpp    # 0
pytest tests/test_device.py -q | tail -1                                  # 2 passed ...
```

Then STANDARD VERIFY-A (§0.3).

**On failure**: STANDARD FAIL. New file to remove:

```bash
git clean -fd -- cpp/mtorch/core/device_transfer.cpp
```

**Commit**:

```bash
git add cpp/mtorch/core/tensor.h cpp/mtorch/core/device_transfer.cpp cpp/mtorch/core/tensor_core.cpp cpp/mtorch/python/py_tensor_type.cpp && git commit -m "refactor(phase5-4): add to_device() transfer API and reroute to()/Tensor.cpu()"
```

Then complete 5-4 in PROGRESS.md per §0.5.

## Step 5-5: The Python device surface

**Goal**: `mtorch.device` accepts `"cuda"`; `device_from_py` parses `"mps"`/`"cuda"` into real
`Device` values whenever the backend's allocator is registered (and keeps raising the exact
same `NotImplementedError`s when it is not — which is always, in Phase 5);
`mtorch._C._backend_available(name)` reports the registry; `mtorch.cuda.is_available()`,
`mtorch.backends.mps.is_available()` and `.is_built()` return real registry-backed values.
Observable behavior today is unchanged (everything still reports unavailable / raises), so the
compat suite must still match the baseline exactly.

**Background — one gate, one truth:** after this step there is exactly one runtime truth about
backend existence — "is an allocator registered for this `DeviceType`" — and three consumers
of it: `device_from_py` (decides whether a device string becomes a `Device` or a
`NotImplementedError`), `Storage`'s constructor (defensive core-side check, added in 5-3), and
`_backend_available` (the Python-visible flag). When Phase 6 calls
`register_allocator(DeviceType::Metal, ...)` during `PyInit__C`, all three flip at once:
factories start building mps storages, `Tensor.to("mps")`/`.cpu()` start transferring, and
`mtorch.backends.mps.is_available()` turns `True` — with **zero further changes** to any file
this step touches.

**Preconditions**: 5-4 committed; `git status --porcelain` prints nothing; and:

```bash
grep -rl '_backend_available' cpp/mtorch/python/ || echo NOT-FOUND-OK      # prints NOT-FOUND-OK
test ! -e cpp/mtorch/python/py_device.cpp && echo PRECONDITION-OK
grep -n 'Device device_from_py' cpp/mtorch/python/py_common.cpp    # prints the definition line
grep -n 'lowercase_ascii' cpp/mtorch/python/py_common.cpp | head -1   # helper is present in this TU
```

**Actions**:

1. `cpp/mtorch/python/py_common.cpp` — add the include. Locate the line
   `#include "mtorch/python/py_common.h"` near the top of the file. BEFORE:

   ```cpp
   #include "mtorch/python/py_common.h"
   ```

   AFTER:

   ```cpp
   #include "mtorch/python/py_common.h"

   #include "mtorch/core/detail/allocator.h"
   ```

2. `cpp/mtorch/python/py_common.cpp` — rewrite `device_from_py`. Locate the definition with
   `grep -n 'Device device_from_py' cpp/mtorch/python/py_common.cpp` (pre-split hint:
   module.cpp ~L1119). **Keep the signature line exactly as it appears in the file** (post
   2a-2 it reads `Device device_from_py(PyObject* object, Device fallback) {` because the
   default argument moved to the header; if you instead find the pre-split form with
   `= mtorch::cpu_device()` in it, keep that line too) and replace the entire body — from the
   line after the signature down to and including the function's closing `}` — with:

   ```cpp
     if (object == nullptr || object == Py_None) {
       return fallback;
     }
     PyObject* text_object = PyObject_Str(object);
     if (text_object == nullptr) {
       throw std::invalid_argument("could not parse device");
     }
     const char* text = PyUnicode_AsUTF8(text_object);
     if (text == nullptr) {
       Py_DECREF(text_object);
       throw std::invalid_argument("could not parse device");
     }
     const std::string device_text = lowercase_ascii(text);
     Py_DECREF(text_object);

     std::string type_text = device_text;
     int64_t index = -1;
     const size_t colon = device_text.find(':');
     if (colon != std::string::npos) {
       type_text = device_text.substr(0, colon);
       const std::string index_text = device_text.substr(colon + 1);
       if (index_text.empty() || index_text.find_first_not_of("0123456789") != std::string::npos) {
         throw std::invalid_argument("could not parse device index");
       }
       index = static_cast<int64_t>(std::stoll(index_text));
     }

     mtorch::DeviceType type = mtorch::DeviceType::CPU;
     if (type_text == "cpu") {
       type = mtorch::DeviceType::CPU;
       index = -1;  // the CPU device is never indexed
     } else if (type_text == "mps" || type_text == "metal") {
       type = mtorch::DeviceType::Metal;
     } else if (type_text == "cuda") {
       type = mtorch::DeviceType::CUDA;
     } else {
       throw std::invalid_argument("unsupported device");
     }

     if (type != mtorch::DeviceType::CPU && mtorch::detail::allocator_for(type) == nullptr) {
       // Keep the exact pre-Phase-5 error messages (pinned by tests/test_device.py
       // via NotImplementedError). These fire until Phase 6/7 register a backend.
       if (type == mtorch::DeviceType::Metal) {
         throw NotImplementedException("Metal/MPS device execution is not implemented yet");
       }
       throw NotImplementedException("CUDA device execution is not implemented");
     }
     return mtorch::Device{type, index};
   }
   ```

   (Behavior notes, all verified against §1 fact 9: for `"cpu"` in any form the result is
   unchanged; for `"mps"`/`"metal"`/`"cuda"` with no registered allocator the raised exception
   type and message are byte-identical to before; the only observable difference is for
   malformed strings such as `"mps:abc"`, which previously matched the `"mps"` substring and
   raised `NotImplementedError` and now raise `ValueError("could not parse device index")` —
   no test exercises those.)

3. Create `cpp/mtorch/python/py_device.cpp` with exactly this content:

   ```cpp
   #define PY_SSIZE_T_CLEAN
   #include <Python.h>

   #include <string>

   #include "mtorch/core/tensor.h"
   #include "mtorch/core/detail/allocator.h"
   #include "mtorch/python/py_common.h"
   #include "mtorch/python/py_registry.h"

   namespace mtorch::py {

   namespace {

   // _backend_available(name) -> bool
   // True iff an allocator is registered for the named backend ("cpu" is always
   // True; "mps"/"cuda" become True when Phase 6/7 call register_allocator).
   // Unknown names return False (they cannot have a registered allocator).
   PyObject* py_backend_available(PyObject*, PyObject* args) {
     const char* name = nullptr;
     if (!PyArg_ParseTuple(args, "s:_backend_available", &name)) {
       return nullptr;
     }
     const std::string text(name);
     mtorch::DeviceType type = mtorch::DeviceType::CPU;
     if (text == "cpu") {
       type = mtorch::DeviceType::CPU;
     } else if (text == "mps" || text == "metal") {
       type = mtorch::DeviceType::Metal;
     } else if (text == "cuda") {
       type = mtorch::DeviceType::CUDA;
     } else {
       Py_RETURN_FALSE;
     }
     if (mtorch::detail::allocator_for(type) != nullptr) {
       Py_RETURN_TRUE;
     }
     Py_RETURN_FALSE;
   }

   }  // namespace

   PyMethodDef kDeviceMethods[] = {
       {"_backend_available", reinterpret_cast<PyCFunction>(py_backend_available), METH_VARARGS, nullptr},
       {nullptr, nullptr, 0, nullptr},
   };

   }  // namespace mtorch::py
   ```

4. `cpp/mtorch/python/py_registry.h` — declare the table. Locate with
   `grep -n 'PyMethodDef\* build_module_methods();' cpp/mtorch/python/py_registry.h` and
   insert directly ABOVE that line (after whatever `extern PyMethodDef ...` lines the earlier
   phases left there):

   ```cpp
   extern PyMethodDef kDeviceMethods[];
   ```

5. `cpp/mtorch/python/module_init.cpp` — concatenate the table. Locate
   `grep -n -A 12 'PyMethodDef\* build_module_methods' cpp/mtorch/python/module_init.cpp`;
   inside that function (and ONLY that function — `build_tensor_methods` below it has an
   identical sentinel line, do not touch it) insert directly ABOVE the
   `methods.push_back({nullptr, nullptr, 0, nullptr});` line:

   ```cpp
       append_table(methods, kDeviceMethods);
   ```

6. `mtorch/__init__.py` — accept `"cuda"` in the device class. Locate with
   `grep -n 'if device_type not in' mtorch/__init__.py`. BEFORE:

   ```python
           if device_type not in {"cpu", "mps"}:
               raise NotImplementedError(f"{device_type!r} device is not implemented")
   ```

   AFTER:

   ```python
           if device_type not in {"cpu", "mps", "cuda"}:
               raise NotImplementedError(f"{device_type!r} device is not implemented")
   ```

7. `mtorch/cuda/__init__.py` — wire `is_available` to the registry. Locate with
   `grep -n -A 2 'def is_available' mtorch/cuda/__init__.py`. BEFORE:

   ```python
   def is_available():
       return False
   ```

   AFTER:

   ```python
   def is_available():
       from mtorch import _C

       return bool(_C._backend_available("cuda"))
   ```

   (The import is intentionally local: `mtorch/cuda` is imported from the root package tail,
   and a module-level `from mtorch import _C` could run while the root module is still
   initializing. The docstring line
   `"""mtorch.cuda stub (equivalent to torch/cuda/; CUDA is never available in this build)."""`
   at the top of the file must also be replaced with
   `"""mtorch.cuda (equivalent to torch/cuda/); availability is read from the native allocator registry."""`.
   Everything else in the file — `device_count`, `empty_cache`, `manual_seed`,
   `is_bf16_supported`, `manual_seed_all`, the `amp` import — stays untouched; Phase 7 owns
   making `device_count` real.)

8. Replace the entire content of `mtorch/mps.py` with exactly:

   ```python
   """mtorch.mps (equivalent to torch/mps/); availability is read from the native allocator registry."""


   def is_available():
       from mtorch import _C

       return bool(_C._backend_available("mps"))


   def is_built():
       from mtorch import _C

       return bool(_C._backend_available("mps"))
   ```

   (`mtorch/backends/mps.py` re-exports these two names — §1 fact 13 — so
   `mtorch.backends.mps.is_available()/is_built()` are now registry-backed with no edit to the
   backends package. In Phase 5 "built" and "available" are deliberately the same registry
   fact; Phase 6 may split them with a compile-time flag — recorded in §5.)

**Verification**:

```bash
python3 setup.py build_ext --inplace
python3 - <<'PY'
import mtorch
import mtorch.backends.mps
import mtorch.cuda

assert mtorch._C._backend_available("cpu") is True
assert mtorch._C._backend_available("mps") is False
assert mtorch._C._backend_available("cuda") is False
assert mtorch._C._backend_available("nonsense") is False

d = mtorch.device("cuda:1")
assert d.type == "cuda" and d.index == 1 and str(d) == "cuda:1"
assert repr(mtorch.device("mps")) == "device(type='mps')"
assert repr(d) == "device(type='cuda', index=1)"
assert mtorch.device("cpu") == "cpu"
assert mtorch.device("metal") == mtorch.device("mps")

assert mtorch.cuda.is_available() is False
assert mtorch.backends.mps.is_available() is False
assert mtorch.backends.mps.is_built() is False

for name in ("mps", "cuda"):
    try:
        mtorch.zeros((1,), device=name)
    except NotImplementedError:
        pass
    else:
        raise SystemExit(f"FAIL: zeros(device={name!r}) did not raise")

t = mtorch.ones((2,), dtype=mtorch.float32, device="cpu")
assert t.device == "cpu" and t.device.type == "cpu"
print("phase5-5-check OK")
PY
pytest tests/test_device.py -q | tail -1     # 2 passed ...
```

Then STANDARD VERIFY-A (§0.3).

**On failure**: STANDARD FAIL. New file to remove:

```bash
git clean -fd -- cpp/mtorch/python/py_device.cpp
```

**Commit**:

```bash
git add cpp/mtorch/python/py_common.cpp cpp/mtorch/python/py_device.cpp cpp/mtorch/python/py_registry.h cpp/mtorch/python/module_init.cpp mtorch/__init__.py mtorch/cuda/__init__.py mtorch/mps.py && git commit -m "refactor(phase5-5): registry-backed device parsing, _backend_available, and Python availability wiring"
```

Then complete 5-5 in PROGRESS.md per §0.5.

## Step 5-6: `tests/compat/test_device_parity.py`

**Goal**: The self-parity framework exists: a fixed table of 16 ops is executed on every
available device and compared against the CPU result using the existing harness comparators,
plus 5 surface tests for the Phase 5 API. With no backend registered it collects and passes
exactly **21** tests (16 parity × 1 device + 5 surface), all cpu-vs-cpu.

**Background:** the compat harness compares mtorch against live torch; the parity framework
compares **mtorch against mtorch** across devices. That is the right oracle for Phases 6/7:
a Metal kernel is correct when it reproduces mtorch's own CPU result (which the compat suite
already pins to torch), and this decouples backend bring-up from torch's own device quirks.
The device list follows the architecture contract verbatim — `["cpu"]` plus `"mps"`/`"cuda"`
when the registry reports them — so when Phase 6 registers its allocator, the same table
re-runs on mps with zero test-code changes (16 more tests appear per device; guides 09/10 must
account for that in their expected counts).

**Preconditions**: 5-5 committed; `git status --porcelain` prints nothing; and:

```bash
test ! -e tests/compat/test_device_parity.py && echo PRECONDITION-OK
grep -n 'def assert_values_compatible\|def _coerce_to_reference_tensor' tests/compat/harness.py
```

The grep prints both definition lines (§1 fact 15). If either name is missing, STANDARD FAIL —
do not substitute a different comparator.

**Actions**:

1. Create `tests/compat/test_device_parity.py` with exactly this content:

   ```python
   """Cross-device self-parity scaffolding (Phase 5).

   Compares mtorch-on-<device> against mtorch-on-cpu for a fixed table of ops,
   using the compat harness comparators for the value checks. Until Phase 6/7
   register a backend allocator, DEVICES == ["cpu"] and every parity case runs
   as a cpu-vs-cpu pass (16 parity tests + 5 surface tests = 21 tests). When
   mtorch.backends.mps.is_available() or mtorch.cuda.is_available() becomes
   True, the same table re-runs on that device with no changes to this file
   (adding 16 tests per newly available device).

   See docs/design/08-phase5-device-layer.md (step 5-6).
   """

   from __future__ import annotations

   from dataclasses import dataclass
   from typing import Any, Callable

   import pytest

   import mtorch
   import mtorch.backends.mps
   import mtorch.cuda

   from .harness import _coerce_to_reference_tensor, assert_values_compatible

   pytestmark = [pytest.mark.compat, pytest.mark.numerics]


   DEVICES = (
       ["cpu"]
       + (["mps"] if mtorch.backends.mps.is_available() else [])
       + (["cuda"] if mtorch.cuda.is_available() else [])
   )


   _LEFT = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
   _RIGHT = [[0.5, 1.5, 2.5, 3.5], [4.5, 5.5, 6.5, 7.5], [8.5, 9.5, 10.5, 11.5]]


   def _single(m: Any) -> tuple:
       return (m.tensor(_LEFT, dtype=m.float32),)


   def _pair(m: Any) -> tuple:
       return (m.tensor(_LEFT, dtype=m.float32), m.tensor(_RIGHT, dtype=m.float32))


   @dataclass(frozen=True)
   class ParityCase:
       id: str
       make_inputs: Callable[[Any], tuple]
       run: Callable[..., Any]


   PARITY_CASES = [
       ParityCase("add", _pair, lambda m, a, b: a + b),
       ParityCase("sub", _pair, lambda m, a, b: a - b),
       ParityCase("mul", _pair, lambda m, a, b: a * b),
       ParityCase("div", _pair, lambda m, a, b: a / (b + 2.0)),
       ParityCase("add_scalar", _single, lambda m, a: a + 1.5),
       ParityCase("mul_scalar", _single, lambda m, a: a * 0.5),
       ParityCase("neg", _single, lambda m, a: -a),
       ParityCase("matmul", _pair, lambda m, a, b: a @ b.t()),
       ParityCase("sum", _single, lambda m, a: a.sum()),
       ParityCase("mean", _single, lambda m, a: a.mean()),
       ParityCase("exp", _single, lambda m, a: m.exp(a * 0.1)),
       ParityCase("relu", _single, lambda m, a: m.relu(a - 6.0)),
       ParityCase("softmax", _single, lambda m, a: m.softmax(a, 1)),
       ParityCase("cat", _pair, lambda m, a, b: m.cat([a, b], 0)),
       ParityCase("transpose_contiguous", _single, lambda m, a: a.transpose(0, 1).contiguous()),
       ParityCase("reshape", _single, lambda m, a: a.reshape((4, 3))),
   ]


   def _device_type_of(value: Any) -> str:
       device = getattr(value, "device", "cpu")
       return str(device).split(":", 1)[0]


   def _assert_parity(reference: Any, cpu_result: Any, device_result: Any, *, path: str) -> None:
       """assert_close-equivalent built on the harness comparators.

       Converts the CPU-side mtorch result to a torch tensor (the harness's
       reference representation) and compares the device-side result — moved
       back to cpu — against it with assert_values_compatible (shape, dtype,
       device type, requires_grad, values via torch.testing.assert_close).
       Strides are not compared: a backend kernel may legally produce a
       different-but-equivalent layout.
       """
       result_on_cpu = device_result.cpu() if hasattr(device_result, "cpu") else device_result
       torch_dtype = getattr(reference, str(cpu_result.dtype))
       expected = _coerce_to_reference_tensor(reference, cpu_result, torch_dtype)
       assert_values_compatible(
           reference, expected, result_on_cpu, path=path, check_stride=False
       )


   # ---------------------------------------------------------------------------
   # Surface tests (device-count independent: always exactly 5 tests)
   # ---------------------------------------------------------------------------


   def test_device_class_surface() -> None:
       d = mtorch.device("mps")
       assert d.type == "mps"
       assert d.index is None
       assert repr(d) == "device(type='mps')"
       assert str(mtorch.device("cuda:1")) == "cuda:1"
       assert mtorch.device("cuda:1").index == 1
       assert repr(mtorch.device("cuda:1")) == "device(type='cuda', index=1)"
       assert mtorch.device("cpu") == "cpu"
       assert mtorch.device("metal") == mtorch.device("mps")


   def test_backend_availability_flags() -> None:
       assert mtorch.backends.mps.is_available() in (True, False)
       assert mtorch.backends.mps.is_built() in (True, False)
       assert mtorch.cuda.is_available() in (True, False)
       assert mtorch._C._backend_available("cpu") is True
       assert mtorch._C._backend_available("nonsense") is False


   def test_factory_device_kwarg_and_device_property() -> None:
       t = mtorch.ones((2, 3), dtype=mtorch.float32, device=mtorch.device("cpu"))
       assert t.device == "cpu"
       assert t.device.type == "cpu"
       assert t.device.index is None


   def test_to_and_cpu_round_trip() -> None:
       t = mtorch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=mtorch.float32)
       assert t.cpu() is t
       moved = t.to("cpu")
       assert _device_type_of(moved) == "cpu"
       assert moved.tolist() == t.tolist()


   def test_unavailable_backend_raises_not_implemented() -> None:
       if not mtorch.backends.mps.is_available():
           with pytest.raises(NotImplementedError):
               mtorch.zeros((1,), device="mps")
       if not mtorch.cuda.is_available():
           with pytest.raises(NotImplementedError):
               mtorch.zeros((1,), device="cuda")


   # ---------------------------------------------------------------------------
   # Parity tests (16 cases x len(DEVICES) tests)
   # ---------------------------------------------------------------------------


   @pytest.mark.parametrize("case", PARITY_CASES, ids=[case.id for case in PARITY_CASES])
   @pytest.mark.parametrize("device_name", DEVICES)
   def test_op_parity_matches_cpu(torch_reference, device_name: str, case: ParityCase) -> None:
       cpu_inputs = case.make_inputs(mtorch)
       cpu_result = case.run(mtorch, *cpu_inputs)

       moved_inputs = tuple(item.to(device_name) for item in cpu_inputs)
       for item in moved_inputs:
           assert _device_type_of(item) == device_name
       device_result = case.run(mtorch, *moved_inputs)
       assert _device_type_of(device_result) == device_name

       _assert_parity(
           torch_reference,
           cpu_result,
           device_result,
           path=f"device_parity.{case.id}[{device_name}]",
       )
   ```

**Verification**:

```bash
pytest tests/compat/test_device_parity.py -q 2>&1 | tail -2
```

Expected: the summary line reads exactly `21 passed` (plus timing). Then run
**STANDARD VERIFY-B** (§0.3) — the full suite must show the baseline breakdown with `passed`
increased by exactly 21 and the collected count increased by exactly 21, nothing else changed.

**On failure**: STANDARD FAIL. New file to remove:

```bash
git clean -fd -- tests/compat/test_device_parity.py
```

**Commit**:

```bash
git add tests/compat/test_device_parity.py && git commit -m "refactor(phase5-6): add cross-device self-parity test scaffolding"
```

Then complete 5-6 in PROGRESS.md per §0.5.

## Step 5-7: Phase gate — full tests, full benchmark, baseline re-record

**Goal**: Prove the whole phase is behavior- and performance-neutral (Storage's allocation
path was touched, so the full benchmark gate is mandatory), then re-record the test baseline
so Phases 6/7 compare against the post-Phase-5 truth (which includes the 21 parity tests).

**Preconditions**: 5-6 committed; `git status --porcelain` prints nothing.

**Actions**:

1. Full test run and comparison (STANDARD VERIFY-B form; keep the `tee` output — action 4
   reuses it):

   ```bash
   python3 setup.py build_ext --inplace
   pytest tests/compat 2>&1 | tee /tmp/tests-current.txt | tail -3
   tail -1 docs/design/baseline/tests-baseline.txt | tr -d '=' | sed 's/ in .*//' | xargs
   tail -1 /tmp/tests-current.txt | tr -d '=' | sed 's/ in .*//' | xargs
   ```

   PASS rule: current = baseline with `passed` + 21, every other count identical (see §0.3
   VERIFY-B, including the worked example). Note this run uses the plain (non `-q`) form so
   that `/tmp/tests-current.txt` has the same format as `tests-baseline.txt` was recorded with
   in Phase 0 step 0-4.

2. Full benchmark comparison, exactly per `01-rules-and-verification.md` §5.2 (do not run
   anything else on the machine in parallel; command timeout at least 60 minutes):

   ```bash
   mkdir -p benchmark-results
   pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
       --compat-benchmark-json benchmark-results/current.json
   python3 tools/compare_benchmarks.py \
       docs/design/baseline/benchmark-baseline.json benchmark-results/current.json
   ```

   PASS rule: final line shows `regressions=0 missing=0` and `new=0` (this phase adds no
   benchmark cases). If `regressions=` is nonzero, apply the 3-rerun noise procedure of 01
   §5.2 verbatim; a reproducible regression (2/3 or more) means the phase failed its
   performance gate — revert per 01 §6 starting from the most recent Phase 5 commit and mark
   5-7 **BLOCKED** (the plausible culprits are the Storage constructor edit in 5-3 and the
   `to()` reroute in 5-4).

3. Public-API-additions audit — confirm the surface grew by exactly the allowed names:

   ```bash
   python3 - <<'PY'
   import mtorch
   assert callable(mtorch._C._backend_available)
   assert mtorch.device("cuda").type == "cuda"
   assert mtorch.cuda.is_available() is False
   assert mtorch.backends.mps.is_available() is False and mtorch.backends.mps.is_built() is False
   print("phase5-api-audit OK")
   PY
   grep -c 'to_device' cpp/mtorch/core/tensor.h    # 1  (the single allowed tensor.h addition)
   pytest tests/test_device.py -q | tail -1         # 2 passed ...
   ```

4. Re-record the test baseline (tests + collection count only; the benchmark baseline is NOT
   re-recorded — no benchmark case changed). This intentionally overwrites two Phase 0
   artifacts so that Phases 6/7 verify against the post-Phase-5 breakdown:

   ```bash
   cp /tmp/tests-current.txt docs/design/baseline/tests-baseline.txt
   pytest tests/compat --collect-only -q 2>&1 | tail -2 > docs/design/baseline/collect-count.txt
   cat docs/design/baseline/collect-count.txt
   ```

   Expected: the `cat` shows a collected count equal to the old baseline + 21 (e.g.
   `2293 tests collected` if the old baseline was `2272 tests collected`).

**Verification**: Actions 1–3 all passed with the exact PASS rules above, and:

```bash
tail -1 docs/design/baseline/tests-baseline.txt | tr -d '=' | sed 's/ in .*//' | xargs
tail -1 /tmp/tests-current.txt | tr -d '=' | sed 's/ in .*//' | xargs
```

prints two identical lines (the new baseline file matches the run that produced it).

**On failure**: If action 1 or 2 failed, STANDARD FAIL (for action 4's partial edits,
`git restore --staged --worktree .` restores both baseline files). Mark 5-7 **BLOCKED**.

**Commit**:

```bash
git add docs/design/baseline/tests-baseline.txt docs/design/baseline/collect-count.txt && git commit -m "refactor(phase5-7): phase gate passed; re-record test baseline including device parity tests"
```

Then complete 5-7 in PROGRESS.md per §0.5. **Phase 5 is done.**

## 4. The dispatch hook-point convention for Phases 6/7 (documentation only — do NOT apply in Phase 5)

Nothing in this section is executed during Phase 5. It is the fixed convention that guides
09 (Metal) and 10 (CUDA) implement, shown fully worked on `add` so there is exactly one
pattern to copy. mtorch's `add` is `binary_tensor_tensor(left, right, "add")` (declared in
`tensor.h`; implementation in `cpp/mtorch/core/elementwise_ops.cpp` after step 2b-15 — locate
with `grep -n 'TensorPtr binary_tensor_tensor' cpp/mtorch/core/elementwise_ops.cpp`).

The convention:

1. The hook sits at the **top of the public op's implementation**, before any storage access.
   Public signatures in `tensor.h` never change.
2. If any input tensor is non-CPU and the backend implements the op, route to the backend
   namespace function: `mtorch::metal::<op>(...)` (Phase 6) / `mtorch::cuda::<op>(...)`
   (Phase 7). Backend entry points mirror the public signature.
3. Otherwise, if the device's allocator reports `host_accessible()` (Metal on unified
   memory), the backend MAY fall back to the existing CPU kernel in place. **Phase 6 owns
   making that fall-through actually safe**: after Phase 5, CPU kernels read
   `storage->bytes`, which is empty for non-CPU storages (§1 fact 5) — so doc 09 must either
   back unified-memory storages with a host-visible `bytes`-compatible buffer or route the
   fallback through explicit transfers. Until it does, treat option 4 as the default.
4. Otherwise raise via the helper — `detail::unsupported_device("add", device_type)` — which
   throws `std::runtime_error` with the fixed message format
   `mtorch: operator 'add' is not implemented for device 'cuda'`. CUDA never falls back
   silently.

The fully worked example (what Phase 6/7 will insert; shown for `add`):

```cpp
TensorPtr binary_tensor_tensor(const TensorPtr& left, const TensorPtr& right, const std::string& op) {
  if (left->device.type != DeviceType::CPU || right->device.type != DeviceType::CPU) {
    const DeviceType device_type =
        left->device.type != DeviceType::CPU ? left->device.type : right->device.type;
    if (device_type == DeviceType::Metal && mtorch::metal::implements_binary(op)) {
      return mtorch::metal::binary_tensor_tensor(left, right, op);   // Phase 6 kernel
    }
    detail::Allocator* backend_allocator = detail::allocator_for(device_type);
    if (backend_allocator != nullptr && backend_allocator->host_accessible()) {
      // Unified memory: fall through to the CPU implementation below, operating
      // on the device-resident buffer in place (Phase 6 must guarantee the CPU
      // kernel can reach those bytes -- see convention rule 3).
    } else {
      detail::unsupported_device(op.c_str(), device_type);
      // e.g. mtorch: operator 'add' is not implemented for device 'cuda'
    }
  }
  // ... existing CPU implementation, byte-for-byte unchanged ...
}
```

Phases 6/7 apply this pattern op by op, each application gated by the parity tests of step
5-6 (and each backend adds ops to its `implements_*` set incrementally). Phase 5 applies it
nowhere.

## 5. Handoff to Phase 6 (Metal, guide 09) and Phase 7 (CUDA, guide 10)

Everything a backend needs exists after step 5-7. The consumable surface, by file:

**`cpp/mtorch/core/tensor.h`**
- `enum class DeviceType { CPU, Metal, CUDA }` (values 0/1/2 — the registry table depends on
  this order).
- `struct Device { DeviceType type; int64_t index = -1; }` — the contract's `Device`;
  `index == -1` means "unindexed" (Python `None`). The CPU device is always `{CPU, -1}`.
  Backends should treat `index == -1` as "default device 0".
- `cpu_device()`, `metal_device(int64_t index = 0)`, `cuda_device(int64_t index = 0)`,
  `devices_equal`, `device_type_name` (`"cpu"` / `"mps"` / `"cuda"`), `device_name`.
- `struct Storage`: `Device device` tag; `bytes` (CPU data, empty for non-CPU);
  `allocator` / `device_data` / `device_nbytes` (non-CPU data); `data()` / `nbytes()` /
  `numel()` device-agnostic accessors; destructor returns the device buffer to its allocator.
  A non-CPU Storage can only be constructed while an allocator is registered.
- `TensorPtr to_device(const Tensor& source, Device device)` — the transfer API (the
  contract's `to_device`; returns `TensorPtr` per house convention). Whole-storage copy,
  view metadata preserved; same-device call returns a storage-sharing view; cross-backend
  device-to-device transfer throws (route through cpu). **No autograd edge is attached** —
  if Phase 6/7 workloads need a differentiable `.to()`, that is backend-phase work.
- `to(input, dtype, device, copy)` — already routes any non-CPU request through host-side
  dtype staging + `to_device`; backends get `Tensor.to(...)` working without touching it.

**`cpp/mtorch/core/detail/allocator.h`** (`namespace mtorch::detail`)
- `struct Allocator` — implement all six members:
  `allocate` / `deallocate` / `copy_from_host` / `copy_to_host` / `copy_on_device` /
  `host_accessible`.
- `void register_allocator(DeviceType, Allocator*)` — call it once with a long-lived
  singleton. Recommended registration point: inside `PyInit__C` in
  `cpp/mtorch/python/module_init.cpp`, before the module object is created (i.e. add a
  `mtorch::metal::register_backend();` call there in Phase 6). Registering the allocator is
  the single switch that makes: factories accept the device, `to_device`/`to()`/`.cpu()`
  transfer, `device_from_py` stop raising, `_backend_available` return True, and the parity
  suite expand.
- `Allocator* allocator_for(DeviceType)` — nullptr means "not available".
- `[[noreturn]] void unsupported_device(const char* op, DeviceType)` — the mandatory error
  for unimplemented ops (message format fixed:
  `mtorch: operator '<name>' is not implemented for device '<device>'`).

**Bindings (no further changes needed for basic bring-up)**
- `device_from_py` (`cpp/mtorch/python/py_common.cpp`) parses `"cpu" | "mps" | "metal" |
  "cuda"` with optional `:<index>` and gates on the registry; while unregistered it raises
  the exact messages `Metal/MPS device execution is not implemented yet` /
  `CUDA device execution is not implemented` as `NotImplementedError`.
- `mtorch._C._backend_available(name)` (`cpp/mtorch/python/py_device.cpp`,
  `kDeviceMethods`, registered in `py_registry.h` + `module_init.cpp`).
- `Tensor_get_device` returns `device_name(...)` strings (`"mps"`, `"mps:0"`), which the
  Python `device` class wraps; `Tensor_cpu` transfers via `to_device`.

**Python**
- `mtorch.device` accepts `"cpu"/"mps"/"cuda"` (+ `"metal"` alias) with optional index; it is
  a `str` subclass with `.type`/`.index`/torch-format `repr`.
- `mtorch.backends.mps.is_available()` / `is_built()` and `mtorch.cuda.is_available()` are
  registry-backed. In Phase 5, mps "built" == "available"; Phase 6 may differentiate them
  (e.g. a compile-time flag exposed through a new `_C` function). `mtorch.cuda.device_count()`
  is still a stub returning 0 — Phase 7 owns it.

**Tests and gates**
- `tests/compat/test_device_parity.py`: `DEVICES` auto-includes each backend the moment
  `is_available()` turns True; each new device adds exactly 16 parity tests (the surface
  tests stay at 5). Guides 09/10 must state their own expected counts accordingly and
  re-record the baseline at their phase gates, as 5-7 did.
- `docs/design/baseline/tests-baseline.txt` and `collect-count.txt` were re-recorded in step
  5-7 and now INCLUDE the 21 Phase 5 tests; `benchmark-baseline.json` is unchanged.
- The dispatch hook pattern to copy is §4 (with the `host_accessible()` fallback caveat that
  CPU kernels read `storage->bytes`, not `Storage::data()` — Phase 6 must bridge that before
  using in-place fallback).
- `tests/test_device.py` still pins: `"metal"` normalizes to `"mps"`, and factories raise
  `NotImplementedError` for unavailable backends. Once Phase 6 registers Metal, the second
  pinned test (`zeros(device="mps")` raises) will start failing **by design** — doc 09 must
  address that test explicitly at its own gate (it is the one pre-Phase-5 test whose pinned
  behavior is "backend absent").
