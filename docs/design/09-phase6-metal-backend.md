# 09. Phase 6: Apple-Silicon Metal Backend (Python device name: "mps")

Prerequisites: **Phases 0–5 are fully complete** (all checkboxes for phases 0–4 in
`docs/design/PROGRESS.md` are `[x]`, and the Phase 5 section from
`08-phase5-device-layer.md` is `[x]` as well). `01-rules-and-verification.md` has
been read in full. This guide executes against the **post-Phase-5 tree**: the C++
core is split into section files (`cpp/mtorch/core/elementwise_ops.cpp`,
`linalg.cpp`, `normalization.cpp`, `conv.cpp`, `attention.cpp`,
`activations.cpp`, `tensor_core.cpp`, …, from `05-phase2b-tensor-cpp-split.md`),
and the device layer from Phase 5 exists (the `Device`/`DeviceType` definitions
in `cpp/mtorch/core/tensor.h` — Phase 5 found them already present and did NOT
create a separate `device.h` — plus `cpp/mtorch/core/detail/allocator.h` and
`tests/compat/test_device_parity.py`).

Objective: implement a working Metal compute backend for Apple Silicon, exposed
to Python as device `"mps"`. After this phase:

- `mtorch.backends.mps.is_available()` returns `True` on this machine.
- Tensors on device `"mps"` live in `MTLBuffer`s (unified memory,
  `storageModeShared`).
- Eight op families run as real GPU kernels: fill/copy, binary elementwise
  (add/sub/mul/div), unary elementwise (+ silu/gelu), matmul, softmax,
  layer_norm/group_norm, conv2d, scaled_dot_product_attention.
- Everything else on `"mps"` keeps working through the Phase 5 zero-copy CPU
  fallback (the Metal allocator is host-accessible, so the existing CPU kernels
  operate directly on the shared buffer).

Non-goals of this phase (all recorded as future work in §5, never improvised):

- Asynchronous execution / kernel batching. **Every launch in this phase is
  synchronous** (`waitUntilCompleted`).
- Autograd on the GPU. **Metal kernels never run for gradient-tracked tensors**;
  the guard falls back to the CPU path (§0.8).
- bfloat16, int dtypes, strided (non-contiguous) kernels, multi-GPU.
- Beating PyTorch's MPS backend on speed. **The gate for this phase is
  correctness, not speed** (§1 step 6-14).

Commit-message prefix for this phase is `feat(phase6-N): ...` instead of
`refactor(...)`. This is an intentional extension of `01-rules-and-verification.md`
§7 (which predates feature phases), not a contradiction — everything else in
§7 (one step = one commit, explicit `git add` by path, no build artifacts)
applies unchanged.

**This guide is deliberately more verbose than 02–07.** Metal is unfamiliar
territory: every Metal API call is explained where it first appears, and every
new concept gets a `**Background:**` call-out. If something here disagrees with
what you observe in the tree, do not improvise — follow the failure protocol
(`01-rules-and-verification.md` §6) and mark the step **BLOCKED**.

---

## 0. Fundamentals (read this whole section once before step 6-0)

### 0.1 The Metal execution model, from zero

Metal is Apple's GPU API. To run computation on the GPU you build a small
object pipeline. Every object below is an Objective-C object you obtain from
the previous one:

```
MTLDevice  →  MTLCommandQueue  →  MTLCommandBuffer  →  MTLComputeCommandEncoder
   │                                                        │
   └── MTLLibrary → MTLFunction → MTLComputePipelineState ──┘
```

1. **`MTLDevice`** represents the GPU itself. You obtain it once with
   `MTLCreateSystemDefaultDevice()`. On Apple Silicon there is exactly one, and
   it shares memory with the CPU. All other objects are created from the device.
2. **`MTLCommandQueue`** (`[device newCommandQueue]`) is a thread-safe queue
   the GPU pulls work from. Create one, keep it forever.
3. **`MTLCommandBuffer`** (`[queue commandBuffer]`) is a single batch of GPU
   work. You fill it, `commit` it (hand it to the GPU), and can then block on
   `waitUntilCompleted`. A command buffer is single-use: one per launch.
4. **`MTLComputeCommandEncoder`** (`[commandBuffer computeCommandEncoder]`) is
   the object you use to write commands *into* the command buffer: which
   kernel to run, which buffers it reads/writes, and how many threads to
   launch. You must call `endEncoding` when done.
5. **`MTLLibrary`** is a container of compiled GPU functions. It comes either
   from a precompiled `.metallib` file (`newLibraryWithURL:error:`) or from
   compiling Metal Shading Language source at runtime
   (`newLibraryWithSource:options:error:`). We support both (§0.6, step 6-4).
6. **`MTLFunction`** (`[library newFunctionWithName:@"mtorch_add_f32"]`) is one
   named `kernel` function from the library.
7. **`MTLComputePipelineState`** ("PSO"), from
   `newComputePipelineStateWithFunction:error:`, is the fully compiled,
   GPU-specific executable form of one function. Creating a PSO is expensive
   (milliseconds); using it is cheap. **We therefore cache PSOs forever in a
   map keyed by function name** (step 6-4).

**Background: grid, threadgroups, and `thread_position_in_grid` (4–8 lines).**
A compute dispatch launches a *grid* of threads; each thread runs the same
kernel function with a different index. The grid is divided into equally-sized
*threadgroups*. Threads in the same threadgroup can share fast on-chip
`threadgroup` memory and synchronize with barriers; threads in different
threadgroups cannot. Inside the kernel, `uint gid [[thread_position_in_grid]]`
gives the thread its global index — our elementwise kernels simply compute
element `gid`. Because the grid is dispatched in whole threadgroups, the total
thread count is rounded up; every kernel therefore starts with
`if (gid >= n) return;` so the overhang threads do nothing.

**Background: choosing the threadgroup size (4 lines).** Each PSO reports
`threadExecutionWidth` (the SIMD width `w`, 32 on Apple GPUs — the analogue of a
CUDA warp) and `maxTotalThreadsPerThreadgroup` (upper limit, usually 1024, but
*lower if the kernel uses many registers*). A good 1-D threadgroup size is "a
multiple of `w`, as large as allowed, but not larger than the problem":
`tg = min(maxTotalThreadsPerThreadgroup, ceil(n / w) * w)`, then dispatch
`ceil(n / tg)` threadgroups. This exact formula is implemented once in
`launch_1d` (step 6-6) and never repeated.

### 0.2 CUDA → Metal vocabulary table

If you know CUDA, translate as follows:

| CUDA | Metal | Notes |
|---|---|---|
| `cudaGetDevice` / device | `MTLDevice` | one device on Apple Silicon |
| stream | `MTLCommandQueue` (+ command buffers) | queue is persistent; command buffer is per-submission |
| kernel launch `<<<blocks, threads>>>` | `dispatchThreadgroups:threadsPerThreadgroup:` on an encoder | encoder → commit → wait |
| `__global__` function | `kernel` function in `.metal` source | compiled into `MTLLibrary` |
| compiled module / cubin / PTX | `.metallib` / runtime-compiled `MTLLibrary` | PTX ≈ AIR (intermediate), cubin ≈ metallib |
| `cuModuleGetFunction` + launch config | `MTLFunction` → `MTLComputePipelineState` | PSO is cached, like a loaded kernel |
| block | threadgroup | |
| warp (32 threads) | SIMD-group (`threadExecutionWidth`, 32) | |
| `threadIdx` / `blockIdx` / global index | `thread_position_in_threadgroup` / `threadgroup_position_in_grid` / `thread_position_in_grid` | attributes on kernel parameters |
| `__shared__` memory | `threadgroup` address space | |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | |
| `cudaMalloc` | `[device newBufferWithLength:options:]` → `MTLBuffer` | |
| `cudaMemcpy` H2D/D2H | **not needed** — unified memory (§0.3) | `memcpy` on `[buffer contents]` |
| `cudaDeviceSynchronize` | `[commandBuffer waitUntilCompleted]` | per command buffer, not global |
| cuBLAS `gemm` | `MPSMatrixMultiplication` (MetalPerformanceShaders) | used for matmul/conv/attention |
| `CUDA_LAUNCH_BLOCKING=1` etc. | `MTL_SHADER_VALIDATION=1`, `MTL_DEBUG_LAYER=1` | §3 debugging appendix |

### 0.3 Unified memory and `storageModeShared` — why there is no H2D copy

**Background (6 lines).** On Apple Silicon the CPU and GPU are on the same die
and share one physical RAM pool. An `MTLBuffer` created with
`MTLResourceStorageModeShared` is a single allocation that both processors
access at the same address: `[buffer contents]` returns a plain CPU pointer
into it. There is no PCIe bus and no `cudaMemcpy` equivalent — "moving a tensor
to the GPU" only means "allocating its bytes inside an MTLBuffer instead of
`malloc`". The only correctness rule is *temporal*: the CPU must not read
results until the GPU work is complete. Because every launch in this phase ends
with `waitUntilCompleted`, that rule is trivially satisfied everywhere.

This is what makes the Phase 5 fallback contract sound: the Metal allocator
reports `host_accessible() == true`, so any op *without* a Metal kernel simply
runs the existing CPU kernel directly on the shared buffer — zero copies, bit-
identical results. Phase 6 only ever *adds* fast paths on top of an already
correct baseline. That is also the debugging superpower of this phase: **you
can always turn every Metal kernel off** (`MTORCH_METAL_FORCE_FALLBACK=1`, §0.8)
**and the whole suite must still be green.**

### 0.4 Metal Shading Language (MSL) basics

MSL is C++14 with GPU extensions. A minimal kernel:

```
#include <metal_stdlib>
using namespace metal;

struct Params { uint n; };

kernel void mtorch_example_f32(
    device const float* x   [[buffer(0)]],   // input,  device memory
    device float*       y   [[buffer(1)]],   // output, device memory
    constant Params&    p   [[buffer(2)]],   // small read-only parameters
    uint gid [[thread_position_in_grid]]) {  // this thread's global index
  if (gid >= p.n) { return; }                // grid overhang guard (§0.1)
  y[gid] = x[gid] * 2.0f;
}
```

- `kernel` marks a compute entry point (CUDA `__global__`).
- `[[buffer(n)]]` binds the parameter to the buffer the host set at index `n`
  with `setBuffer:offset:atIndex:` or `setBytes:length:atIndex:`. **Our fixed
  convention: tensor buffers occupy indices `0..k-1` in the order listed at the
  launch site, and the parameter struct is always the last index `k`, set with
  `setBytes`** (which copies a small struct by value — no buffer needed).
- Address spaces: `device` = normal GPU-visible memory (read/write, any size);
  `constant` = small read-only data with faster broadcast access (use it for
  parameter structs); `threadgroup` = fast on-chip memory shared within one
  threadgroup (used by the reduction kernels in step 6-11).
- `half` is IEEE fp16. Arithmetic on `half` is fast but low-precision; **our
  fixed numeric rule (§2): every f16 kernel converts to `float`, computes in
  `float`, and converts back with `half(...)` only at the final store.**
- Parameter structs are shared between MSL and C++ by writing the *same struct
  twice* (once in `kernels.metal`, once in the `.mm` file). To make the layouts
  match with zero padding surprises, **every field is exactly 4 bytes** (`uint`
  ↔ `uint32_t`, `float` ↔ `float`) **and the field order is identical**.
- Fast math: Metal compiles with aggressive fast-math *by default*, which
  changes `exp`/`log`/division results enough to break our 1e-6 parity
  tolerances. We disable it in both compile paths (`-fno-fast-math` offline,
  `fastMathEnabled = NO` at runtime — step 6-3/6-4). Never remove those flags.

### 0.5 The v1 design in one page

- **Synchronous:** every op = encode → commit → `waitUntilCompleted`. Slow but
  totally ordered; no fences, no hazards, no races with the CPU fallback path.
- **Contiguous only:** kernels index linearly. Non-contiguous, broadcast,
  type-promoting, or gradient-tracked calls fail the guard and take the CPU
  fallback. Falling back is *correct*, not an error (§0.3).
- **Guards are total:** a Metal entry point is only called when its `use_*`
  guard returned true; the guard re-checks *everything* the kernel assumes
  (device, dtype, contiguity, shapes, element-count limits). §0.8 lists the
  common conditions verbatim.
- **f16 = compute in f32:** see §0.4 and §2.
- **MPSMatrixMultiplication for all GEMMs**, always on freshly *staged* f32
  buffers at offset 0 (a staging copy/cast kernel runs first). This costs a
  copy but removes every buffer-offset alignment question from the design.
  conv2d is im2col + this GEMM; attention is stage + GEMM + mask + softmax +
  GEMM. **Chosen over MPSCNNConvolution because it is deterministic, reuses the
  already-parity-tested matmul path, and needs no descriptor/weight-layout
  reverse engineering** (the one-line justification required by the plan).
- **Environment switches** (all read once, value must be exactly `1`):
  - `MTORCH_DISABLE_MPS=1` — kill switch: Metal never initializes, allocator is
    not registered, `mtorch.backends.mps.is_available()` is `False`.
  - `MTORCH_METAL_FORCE_FALLBACK=1` — device stays available, but every
    dispatch guard returns false: mps tensors run entirely on the CPU fallback.
    This is the first tool to reach for when debugging (§3).
  - `MTORCH_METAL_LOG=1` — prints one `[mtorch-metal] launch <kernel> n=<n>`
    line to stderr per kernel launch. Used by every step's verification to
    prove the GPU path actually ran.

### 0.6 Phase 6 file map

```
cpp/mtorch/core/metal/
  kernels.metal            all MSL kernels (grows step by step)      6-3
  kernels_metal_source.h   GENERATED by setup.py, never committed    6-3
  context.h                ObjC++-only internal header: Context,
                           storage_base, launch/gemm declarations    6-4
  context.mm               Context singleton implementation          6-4
  allocator.mm             MetalAllocator + registration+buffer_for  6-5
  dispatch.mm              launch_1d / launch_rows / guards infra    6-6
  ops.h                    the ONLY header core .cpp files include
                           (pure C++, no Objective-C)                6-6
  fill_copy.mm             op family (a)                             6-7
  binary.mm                op family (b)                             6-8
  unary.mm                 op family (c) incl. silu/gelu             6-9
  matmul.mm                op family (d) + staging + run_gemm_f32    6-10
  normalization.mm         op family (e) softmax/layer/group norm    6-11
  conv.mm                  op family (f) im2col + gemm               6-12
  attention.mm             op family (g) sdpa composition            6-13
tools/metal_ratios.py      mps-vs-cpu-vs-torch-mps timing table      6-6
mtorch/default.metallib    GENERATED at build time, never committed  6-3
docs/design/baseline/tests-phase6-cpu.txt   frozen CPU-only summary  6-2
docs/design/baseline/metal-ratios.md        final ratio table        6-14
```

Layering rule: core section files (`elementwise_ops.cpp` etc.) include **only**
`mtorch/core/metal/ops.h`. `ops.h` contains no Objective-C and compiles on any
platform (non-Apple builds get inline stubs). `context.h` is Objective-C++ only
and is included **only** by the `.mm` files; it `#error`s otherwise.

### 0.7 Standard verification and failure for this phase

**STANDARD VERIFY-6** (referenced by every step below):

```bash
python3 setup.py build_ext --inplace
pytest tests/compat/test_device_parity.py -q | tail -3
pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
tail -1 docs/design/baseline/tests-phase6-cpu.txt
```

PASS if and only if ALL of:

1. the build exits 0;
2. the parity-suite summary line contains **no `failed` and no `error`**
   (its `passed` count *grows* during this phase — that is expected and is the
   only reason the parity file is excluded from the frozen comparison below);
3. every count in the `--ignore` run's summary line equals the corresponding
   count in `docs/design/baseline/tests-phase6-cpu.txt` (captured in step 6-2),
   character-for-character after stripping bars/time exactly as in
   `01-rules-and-verification.md` §5.1 Action 2.

**STANDARD FAIL-6** (referenced by every step below):

1. Discard the step (`<new files>` = the exact files the step created):

   ```bash
   git restore --staged --worktree .
   git status --short
   ```

   For every `??` (untracked) line that the step created, remove it explicitly
   by path, e.g. `git clean -fd -- cpp/mtorch/core/metal/binary.mm`. Never run
   `git clean -fd` without a path.
2. Rebuild and re-run STANDARD VERIFY-6; it must PASS (with the parity count
   back at its previous value) before you continue.
3. In `docs/design/PROGRESS.md`, append ` **BLOCKED**` to the current step's
   line and write ≤3 note lines; commit only PROGRESS.md:
   `git add docs/design/PROGRESS.md && git commit -m "progress: mark 6-N BLOCKED"`.
4. Stop. Do not start the next step.

### 0.8 THE GUARD — the dispatch-hook convention (memorize this)

Phase 5 fixed the convention: *at the top of a public op, mps tensors route to
`mtorch::metal::<op>(...)` if implemented; otherwise the existing CPU kernel
runs directly on the shared buffer.* Phase 6 implements the "if implemented"
part as a pair of functions per family, declared in `ops.h`:

```cpp
bool metal::use_<op>(<the exact argument tensors>, ..., bool grad_tracked);
<result> metal::<op>(...);   // may assume everything use_<op> checked
```

Every `use_*` guard is a conjunction that includes, in this order (cheapest
first — the first line must be the device check so pure-CPU calls pay one
predictable branch and nothing else):

1. every argument tensor has `device.type == DeviceType::Metal`;
2. `metal_ready()` is true (context initialized, `MTORCH_DISABLE_MPS` unset,
   `MTORCH_METAL_FORCE_FALLBACK` unset);
3. `grad_tracked` is false — **Metal never runs for gradient-tracked calls**
   (the CPU paths own autograd; at the hook site pass
   `track_grad(<input>->requires_grad || ...)` exactly as shown in each step);
4. all dtypes are `ScalarType::Float32` or `ScalarType::Float16` and equal
   across arguments (no promotion on the GPU);
5. every tensor is contiguous (`is_contiguous()`), with `numel() > 0`, and
   `offset` and `offset + numel()` fit in `uint32_t` (kernel params use 32-bit
   element offsets);
6. family-specific shape conditions (exact same shapes for binary, 2-D for
   matmul, last-dim for softmax, etc. — spelled out per step).

When a guard returns false **nothing happens** — the function body continues
into the existing CPU implementation, which is correct on the shared buffer.
Guards therefore must never throw and never allocate.

### 0.9 Two recurring mini-procedures

**PARITY-ROWS** (used by steps 6-7 … 6-13). Phase 5 created
`tests/compat/test_device_parity.py` with a device-parametrized op table that
compares each op's result on `"mps"` against the same op on `"cpu"`. Phase 6
appends rows to that table (this is explicitly allowed by this guide and does
not violate Iron Rule 1 — we *add* coverage, we never weaken a comparison):

1. Find the table: `grep -n 'CASES' tests/compat/test_device_parity.py | head`.
2. Read a 20-line window around the table start and identify the exact row
   syntax Phase 5 used (constructor name, field order, how ops/args/dtypes and
   `rtol`/`atol` are expressed).
3. Append one row per line of the step's "Parity rows" table, copying the
   existing syntax exactly and substituting op / shapes / dtype / rtol / atol.
4. If the row schema cannot express something a row needs (e.g. there is no
   per-row tolerance field and the table's default tolerance is *tighter* than
   the step's table demands), do NOT bend the semantics — STANDARD FAIL-6 and
   note exactly which field was missing.

**BENCH-GUARD `<ids>`** (used by steps 6-7 … 6-13; this is the §5.2 recheck
pattern from `01-rules-and-verification.md`, narrowed to the family's cases —
it proves the *CPU* path did not slow down; the full-suite §5.2 runs once at
the phase gate 6-14):

```bash
mkdir -p benchmark-results
pytest tests/compat/test_benchmarks.py <one --compat-benchmark '<id>' per id> \
    --compat-benchmark-repeat=10 \
    --compat-benchmark-json benchmark-results/phase6-check.json
python3 tools/compare_benchmarks.py \
    docs/design/baseline/benchmark-baseline.json benchmark-results/phase6-check.json \
    | grep '^REGRESSION' || true
```

PASS if the `grep` prints nothing. (`MISSING` lines and the nonzero exit code
of `compare_benchmarks.py` are expected and ignored here — the JSON contains
only the selected cases, so every other baseline case is "missing" by design;
this is the same rule as §5.2's recheck.) If a `REGRESSION` line appears, apply
§5.2's 3-rerun noise rule verbatim; a reproducible regression is a step failure
(STANDARD FAIL-6). These benchmarks run the ops on **CPU tensors** — they exist
to prove the added guard branches cost nothing measurable.

---

## 1. Steps

### Step 6-0: Register the Phase 6 checklist in PROGRESS.md

**Goal**: `docs/design/PROGRESS.md` gains the Phase 6 section so the normal
work cycle (`01-rules-and-verification.md` §1) can drive this phase.

**Preconditions**: Phase 5's section in PROGRESS.md is fully `[x]`. Check:

```bash
grep -n 'Phase 5' docs/design/PROGRESS.md
```

(must print a section header line; visually confirm every `- [` line under it
is `- [x]`).

**Actions**:

1. Append the following block to the END of `docs/design/PROGRESS.md`, after
   the last existing line (if an "Overall completion check" section is the last
   section, append AFTER it — the overall check remains where it is):

   ```
   ## Phase 6: Metal backend "mps" (procedure: 09-phase6-metal-backend.md)

   - [ ] 6-0 Register the Phase 6 checklist in PROGRESS.md — commit: / date:
   - [ ] 6-1 Environment and toolchain check — commit: / date:
   - [ ] 6-2 Phase 5 contract audit + frozen CPU-only baseline — commit: / date:
   - [ ] 6-3 Build integration (setup.py, kernels.metal, .gitignore) — commit: / date:
   - [ ] 6-4 Metal context singleton (context.h / context.mm) — commit: / date:
   - [ ] 6-5 MetalAllocator + registry hookup ("mps" becomes available) — commit: / date:
   - [ ] 6-6 Dispatch helpers (ops.h, dispatch.mm) + tools/metal_ratios.py — commit: / date:
   - [ ] 6-7 Op family: fill/copy — commit: / date:
   - [ ] 6-8 Op family: binary elementwise add/sub/mul/div (f32/f16) — commit: / date:
   - [ ] 6-9 Op family: unary elementwise + silu + gelu(tanh) — commit: / date:
   - [ ] 6-10 Op family: matmul via MPSMatrixMultiplication — commit: / date:
   - [ ] 6-11 Op family: softmax + layer_norm + group_norm — commit: / date:
   - [ ] 6-12 Op family: conv2d via im2col + matmul — commit: / date:
   - [ ] 6-13 Op family: scaled_dot_product_attention — commit: / date:
   - [ ] 6-14 Phase gate: full suites, benchmark guard, metal-ratios.md — commit: / date:

   Notes:
   ```

**Verification**:

```bash
grep -c '^- \[ \] 6-' docs/design/PROGRESS.md
```

Expected output: `15`.

**On failure**: STANDARD FAIL-6 (the only touched file is PROGRESS.md; restore
it with `git restore docs/design/PROGRESS.md`).

**Commit**:

```bash
git add docs/design/PROGRESS.md && git commit -m "feat(phase6-0): register Phase 6 checklist"
```

Then mark 6-0 `[x]` per `01-rules-and-verification.md` §8 (separate
`progress: complete 6-0` commit), as with every step below.

---

### Step 6-1: Environment and toolchain check

**Goal**: Confirm the machine can run this phase and record which of the two
Metal-compile paths (offline `.metallib` vs runtime source compile) will be
active. **Both outcomes are supported**; this step only records which one you
have.

**Preconditions**: `pwd` prints `/Users/hiramatsu/dev/mtorch`; the
`01-rules-and-verification.md` §3 session check passed.

**Actions** (run in this order):

1. Confirm the hardware and OS:

   ```bash
   uname -m
   ```

   Expected output: `arm64`. Anything else (e.g. `x86_64`) → this phase cannot
   run on this machine; mark 6-1 **BLOCKED** and stop.

2. Confirm the reference PyTorch has a working MPS backend (it is used by
   `tools/metal_ratios.py` and the §4 secondary oracle):

   ```bash
   python3 -c "import torch; print(torch.backends.mps.is_available())"
   ```

   Expected output: `True`. (Verified `True` on this machine with torch 2.10.0
   at the time this guide was written.) `False` → **BLOCKED**.

3. Probe the offline Metal shader compiler:

   ```bash
   xcrun -sdk macosx metal --version
   ```

   Two supported outcomes:

   - It prints a version banner (full Xcode with the Metal toolchain
     installed) → the build will additionally produce
     `mtorch/default.metallib` (fast library load). Record
     `metal-toolchain: present` in the Phase 6 Notes.
   - It prints `xcrun: error: unable to find utility "metal"` (Command Line
     Tools only — **this is what this machine prints at the time of writing**)
     → the build skips the `.metallib` and the runtime compiles the embedded
     MSL source on first use instead (adds ~100 ms once per process; nothing
     else changes). Record `metal-toolchain: absent (runtime compile path)` in
     the Phase 6 Notes. **This is NOT a blocker.** (Optional, human decision
     only, do not do it yourself: installing full Xcode plus
     `xcodebuild -downloadComponent MetalToolchain` would enable the offline
     path.)

4. Confirm the Metal *runtime* (which is part of macOS and present regardless
   of outcome 3) can create a device, using PyTorch as the probe:

   ```bash
   python3 -c "import torch; print(torch.zeros(2, device='mps').cpu().tolist())"
   ```

   Expected output: `[0.0, 0.0]`.

**Verification**: Actions 1, 2, 4 produced exactly the expected outputs, and
one Notes line from action 3 was written.

**On failure**: Do not start the phase. Record the failing command and first
error line in the Phase 6 Notes, mark 6-1 **BLOCKED** (§6 protocol), stop.

**Commit**:

```bash
git add docs/design/PROGRESS.md && git commit -m "feat(phase6-1): record environment check result"
```

---

### Step 6-2: Phase 5 contract audit + frozen CPU-only baseline

**Goal**: (a) Verify every Phase 5 artifact this guide builds on actually
exists under the expected names; (b) determine the one placeholder this guide
cannot pin down in advance — how `Storage` exposes its raw byte pointer — via
a closed decision table; (c) freeze the CPU-only test summary that STANDARD
VERIFY-6 compares against for the rest of the phase.

**Preconditions**: 6-1 complete.

**Actions**:

1. Existence check (every path must exist; `ls` must print both):

   ```bash
   ls cpp/mtorch/core/detail/allocator.h tests/compat/test_device_parity.py
   ```

2. Device enum check (both greps must print a line; the enum lives in
   `tensor.h` — Phase 5 did not create a separate `device.h`):

   ```bash
   grep -rn 'enum class DeviceType' cpp/mtorch/core/tensor.h
   grep -rn 'Metal' cpp/mtorch/core/tensor.h | head -5
   ```

   Confirm the enum has a `Metal` member and that `device_type_name` maps it to
   the string `"mps"`:

   ```bash
   grep -rn -A 3 'case DeviceType::Metal' cpp/mtorch/core/ | head -8
   ```

   (must show `return "mps";` — this was already true before Phase 5 and Phase
   5 keeps it.)

3. Allocator contract check (every symbol must appear):

   ```bash
   grep -n 'allocate\|deallocate\|copy_from_host\|copy_to_host\|copy_on_device\|host_accessible' cpp/mtorch/core/detail/allocator.h
   grep -n 'allocator_for\|register_allocator' cpp/mtorch/core/detail/allocator.h
   ```

   Then print the full interface for later reference (keep the output in your
   working notes for step 6-5 — the `MetalAllocator` overrides must match these
   signatures exactly):

   ```bash
   sed -n '1,80p' cpp/mtorch/core/detail/allocator.h
   ```

4. Python surface check:

   ```bash
   python3 -c "import mtorch; print(mtorch.backends.mps.is_built(), mtorch.backends.mps.is_available())"
   ```

   Expected at this point (allocator not yet registered): first value per
   Phase 5's definition of `is_built` (record it), second value `False`.

5. **The storage-base decision** (the ONLY placeholder in this guide). Print
   the storage struct:

   ```bash
   grep -n -A 40 'struct Storage' cpp/mtorch/core/tensor.h
   ```

   Match the output against this closed table and record the chosen row letter
   in the Phase 6 Notes (`storage-base: row A/B/C`); step 6-4 substitutes the
   expression into one macro, once:

   | Row | What the struct shows | Expression for `MTORCH_METAL_STORAGE_BASE(s)` |
   |---|---|---|
   | A | a member function that returns the base data pointer, e.g. `void* data();` / `uint8_t* data();` (any pointer return type) | `((const void*)(s).data())` (use the actual method name) |
   | B | a raw pointer data member, e.g. `void* data;` / `uint8_t* data_;` | `((const void*)(s).data)` (use the actual member name) |
   | C | only a `std::vector<uint8_t> bytes;` member and **no** pointer/accessor that Phase 5 routes device allocations through | **BLOCKED** — a `std::vector` cannot adopt allocator-owned memory, so Phase 5's Metal storage path must expose a pointer somewhere; re-read the struct and `allocator.h`; if genuinely absent, STANDARD FAIL-6 with a note quoting the struct |

   If the struct shows BOTH a vector and a pointer (e.g. CPU keeps `bytes`,
   devices use a pointer), the pointer is the answer (row B) **only if** Phase
   5's allocation path stores the allocator-returned pointer there for Metal
   storages — confirm by grepping the allocation call:
   `grep -rn 'allocator_for' cpp/mtorch/core/ | head -5` and reading the
   10-line window around the hit inside the Storage constructor.

6. Freeze the CPU-only baseline for STANDARD VERIFY-6 (the parity suite is
   excluded because its row count legitimately grows during this phase; the
   canonical full baseline `tests-baseline.txt` is re-captured once, at the
   phase gate 6-14):

   ```bash
   pytest tests/compat -q --ignore=tests/compat/test_device_parity.py 2>&1 | tail -1 > docs/design/baseline/tests-phase6-cpu.txt
   cat docs/design/baseline/tests-phase6-cpu.txt
   ```

   The printed line must contain no `failed` / `error`.

7. Record in the Phase 6 Notes (≤3 lines total for this step): the
   storage-base row letter and the `is_built` value from action 4.

**Verification**: Actions 1–4 printed the expected symbols; action 5 resolved
to row A or B; action 6 wrote a one-line file with no failures in it.

**On failure**: STANDARD FAIL-6 (`<new files>` =
`docs/design/baseline/tests-phase6-cpu.txt`).

**Commit**:

```bash
git add docs/design/baseline/tests-phase6-cpu.txt docs/design/PROGRESS.md
git commit -m "feat(phase6-2): audit Phase 5 contract, freeze CPU-only baseline"
```

---

### Step 6-3: Build integration (setup.py, kernels.metal, .gitignore)

**Goal**: The build system (a) compiles `cpp/mtorch/core/metal/*.mm` as
Objective-C++ with ARC on darwin only, (b) links the Metal, Foundation, and
MetalPerformanceShaders frameworks, (c) embeds the MSL source into a generated
C++ header (so runtime compilation always works), and (d) additionally
compiles `kernels.metal` → `mtorch/default.metallib` when the offline
toolchain exists (6-1 outcome). No behavior of the extension changes yet.

**Background: the two shader-compile paths (6 lines).** A `.metal` file can be
compiled *offline* (`xcrun metal -c` produces `.air`, Apple's IR — the PTX
analogue — and `xcrun metallib` packs it into a `.metallib` binary the runtime
loads instantly), or *at runtime* from source text via
`newLibraryWithSource:options:error:` (the driver compiles it on first use,
~100 ms, then it behaves identically). We always generate the embedded-source
header so a plain rebuild works on any machine, and treat the `.metallib` as an
optional accelerator. The runtime prefers the `.metallib` and transparently
falls back to the embedded source **per function** — so even a stale
`.metallib` that lacks a newly added kernel cannot break the build (step 6-4
implements that lookup order).

**Background: Objective-C++ and ARC (5 lines).** A `.mm` file is C++ that may
also use Objective-C objects (`id<MTLDevice>` etc.). ARC (Automatic Reference
Counting, enabled with `-fobjc-arc`) makes the compiler insert retain/release
calls automatically — Objective-C object pointers then behave like
`std::shared_ptr`: storing one in a C++ member or `std::unordered_map` value
retains it; erasing/destroying releases it. We rely on that for the PSO cache
and the buffer map. `setuptools` does not know the `.mm` extension, so the
custom `build_ext` below registers it and injects the Objective-C++ flags for
`.mm` sources only.

**Preconditions** (both must print a line):

```bash
grep -Fn 'glob.glob("cpp/mtorch/core/detail/*.cpp")' setup.py
grep -c 'objective' setup.py   # must print 0 (no prior Metal integration)
```

If the first grep prints nothing, the Phase 2a glob was changed by a later
phase in a way this guide did not anticipate → STANDARD FAIL-6 with a note
quoting the current `sources=` expression.

**Actions**:

1. Create the directory and the initial kernel file. Create
   `cpp/mtorch/core/metal/kernels.metal` with exactly this content (the probe
   kernel exists so the very first library compile has something to validate;
   op families append their kernels to this file in later steps):

   ```
   // mtorch Metal kernels (Metal Shading Language).
   // Conventions (docs/design/09-phase6-metal-backend.md §0.4):
   //  - kernel names: mtorch_<op>_f32 / mtorch_<op>_f16
   //  - tensor buffers at [[buffer(0..k-1)]], parameter struct last at [[buffer(k)]]
   //  - every param-struct field is exactly 4 bytes; field order matches the
   //    C++ mirror struct in the corresponding .mm file
   //  - every kernel begins with the grid-overhang guard: if (gid >= p.n) return;
   //  - f16 kernels load half, compute in float, store half (accumulate-in-float rule)

   #include <metal_stdlib>
   using namespace metal;

   struct ProbeParams {
     uint n;
   };

   // Writes out[i] = i. Used only by build/debug verification.
   kernel void mtorch_probe_iota_f32(
       device float* out [[buffer(0)]],
       constant ProbeParams& p [[buffer(1)]],
       uint gid [[thread_position_in_grid]]) {
     if (gid >= p.n) {
       return;
     }
     out[gid] = float(gid);
   }
   ```

2. Replace the entire contents of `setup.py` with the following. (Full-file
   replacement is deliberate: the current file is small and the precondition
   grep proved it still has the Phase 2a shape. If your `setup.py` contains
   anything beyond imports, the flag/source setup matched by the precondition,
   and the single `setup(...)` call — stop and STANDARD FAIL-6 instead of
   merging by hand.)

   ```python
   from __future__ import annotations

   import glob
   import os
   import subprocess
   import sys

   from setuptools import Extension, setup
   from setuptools.command.build_ext import build_ext


   IS_DARWIN = sys.platform == "darwin"

   extra_compile_args = ["-std=c++20", "-O3"]
   extra_link_args: list[str] = []
   if IS_DARWIN:
       extra_compile_args.append("-DMTORCH_USE_ACCELERATE")
       extra_link_args.extend(["-framework", "Accelerate"])
       # Phase 6: Metal backend frameworks. Metal = GPU API, Foundation =
       # base Objective-C runtime types (NSString/NSError/NSURL),
       # MetalPerformanceShaders = Apple's GPU BLAS (MPSMatrixMultiplication).
       extra_link_args.extend([
           "-framework", "Metal",
           "-framework", "Foundation",
           "-framework", "MetalPerformanceShaders",
       ])

   sources = sorted(
       glob.glob("cpp/mtorch/core/*.cpp")
       + glob.glob("cpp/mtorch/core/detail/*.cpp")
       + glob.glob("cpp/mtorch/python/*.cpp")
   )
   if IS_DARWIN:
       # Objective-C++ backend files; compiled only on darwin.
       sources += sorted(glob.glob("cpp/mtorch/core/metal/*.mm"))

   METAL_DIR = os.path.join("cpp", "mtorch", "core", "metal")
   METAL_SOURCE = os.path.join(METAL_DIR, "kernels.metal")
   METAL_AIR = os.path.join("build", "mtorch-kernels.air")
   METALLIB_OUT = os.path.join("mtorch", "default.metallib")
   EMBEDDED_HEADER = os.path.join(METAL_DIR, "kernels_metal_source.h")

   # Flags injected for .mm sources only (see MetalBuildExt below):
   # -x objective-c++ forces the Objective-C++ language mode (belt-and-braces;
   # clang already infers it from the .mm suffix), -fobjc-arc enables Automatic
   # Reference Counting so id<...> pointers are memory-managed automatically.
   OBJCXX_FLAGS = ["-x", "objective-c++", "-fobjc-arc"]


   def generate_embedded_source() -> None:
       """Embed kernels.metal into a C++ header as a raw string literal.

       This guarantees the runtime can always compile the kernels from source
       (newLibraryWithSource:) even when the offline metal toolchain is not
       installed or default.metallib is missing/stale.
       """
       with open(METAL_SOURCE, "r", encoding="utf-8") as fh:
           text = fh.read()
       delimiter = "MTORCH_MSL"
       if (")" + delimiter + '"') in text:
           raise RuntimeError("kernels.metal must not contain the raw-string delimiter")
       header = (
           "// Auto-generated by setup.py from kernels.metal.\n"
           "// Do not edit and do not commit (listed in .gitignore).\n"
           "#pragma once\n"
           "namespace mtorch::metal {\n"
           "inline const char* embedded_metal_source() {\n"
           '  return R"' + delimiter + "(" + text + ")" + delimiter + '";\n'
           "}\n"
           "}  // namespace mtorch::metal\n"
       )
       with open(EMBEDDED_HEADER, "w", encoding="utf-8") as fh:
           fh.write(header)


   def try_build_metallib() -> None:
       """Offline-compile kernels.metal to mtorch/default.metallib if possible.

       Absence of the toolchain is NOT an error: the runtime falls back to the
       embedded source (see generate_embedded_source). A stale metallib is also
       harmless: function lookup falls back per function at runtime.
       """
       probe = subprocess.run(
           ["xcrun", "-sdk", "macosx", "metal", "--version"],
           capture_output=True,
       )
       if probe.returncode != 0:
           # Remove a stale library so runtime behavior is predictable on
           # toolchain-less machines: source compile only.
           if os.path.exists(METALLIB_OUT):
               os.remove(METALLIB_OUT)
           print("mtorch: metal offline toolchain not found; "
                 "kernels will be compiled from embedded source at runtime")
           return
       os.makedirs("build", exist_ok=True)
       # -fno-fast-math: keep GPU exp/log/div IEEE-conformant so results stay
       # within the parity tolerances against the CPU kernels.
       subprocess.run(
           ["xcrun", "-sdk", "macosx", "metal", "-fno-fast-math",
            "-c", METAL_SOURCE, "-o", METAL_AIR],
           check=True,
       )
       subprocess.run(
           ["xcrun", "-sdk", "macosx", "metallib", METAL_AIR, "-o", METALLIB_OUT],
           check=True,
       )
       print(f"mtorch: built {METALLIB_OUT}")


   class MetalBuildExt(build_ext):
       def build_extensions(self):
           if IS_DARWIN:
               generate_embedded_source()
               try_build_metallib()
               # setuptools' UnixCCompiler does not know .mm; register it and
               # inject the Objective-C++ flags for .mm sources only.
               if ".mm" not in self.compiler.src_extensions:
                   self.compiler.src_extensions.append(".mm")
               original_compile = self.compiler._compile

               def patched_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
                   if src.endswith(".mm"):
                       # Prepend: -x must appear BEFORE the input file on the
                       # command line (cc_args precede the source; the
                       # extension's extra_compile_args come after it).
                       cc_args = OBJCXX_FLAGS + list(cc_args)
                   return original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

               self.compiler._compile = patched_compile
           super().build_extensions()


   setup(
       packages=["mtorch"],
       cmdclass={"build_ext": MetalBuildExt},
       ext_modules=[
           Extension(
               "mtorch._C",
               sources=sources,
               include_dirs=["cpp"],
               language="c++",
               extra_compile_args=extra_compile_args,
               extra_link_args=extra_link_args,
           )
       ],
   )
   ```

3. Append these exact lines to the END of `.gitignore` (generated artifacts;
   `build/` is already ignored, which covers the `.air` intermediate):

   ```
   mtorch/default.metallib
   cpp/mtorch/core/metal/kernels_metal_source.h
   ```

**Verification**:

```bash
python3 setup.py build_ext --inplace
ls cpp/mtorch/core/metal/kernels_metal_source.h
ls mtorch/default.metallib 2>/dev/null || echo "no metallib (runtime-compile path)"
git status --short | grep -v '^??' | cat
pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
tail -1 docs/design/baseline/tests-phase6-cpu.txt
pytest tests/compat/test_device_parity.py -q | tail -3
```

PASS iff: the build exits 0 (and printed either `mtorch: built
mtorch/default.metallib` or the "compiled from embedded source at runtime"
notice, matching your 6-1 record); the generated header exists; `git status`
shows only the files this step touched (the two generated artifacts must be
invisible thanks to `.gitignore` — if they appear as `??`, the `.gitignore`
edit is wrong); and the two test runs match 6-2's baselines (STANDARD VERIFY-6
conditions 2–3).

**On failure**: STANDARD FAIL-6 (`<new files>` =
`cpp/mtorch/core/metal/kernels.metal`; also
`rm -f cpp/mtorch/core/metal/kernels_metal_source.h mtorch/default.metallib`).

**Commit**:

```bash
git add setup.py .gitignore cpp/mtorch/core/metal/kernels.metal
git commit -m "feat(phase6-3): metal build integration (objc++, metallib, embedded source)"
```

---

### Step 6-4: The Metal context singleton (context.h / context.mm)

**Goal**: One lazily-initialized singleton owns the `MTLDevice`, the command
queue, the kernel libraries, and the PSO cache. Every `NSError**` out-parameter
is checked and converted into a C++ exception (`std::runtime_error`), so the
existing binding-layer try/catch translation (the one unified in Phase 3)
turns Metal failures into ordinary Python `RuntimeError`s with no new binding
code.

**Background: why a lazy singleton (5 lines).** Creating an `MTLDevice` and
compiling libraries costs tens of milliseconds; doing it at import would tax
every CPU-only run. Instead the first *use* initializes everything under a
mutex, and a failed initialization (no GPU, kill switch) is remembered so
subsequent calls are a cheap boolean check. The singleton is allocated with
`new` and intentionally never destroyed: releasing Metal objects during static
destruction at interpreter exit is a well-known crash source, and the OS
reclaims everything anyway.

**Preconditions**: 6-3 committed. Check: `git log --oneline -1` shows
`feat(phase6-3)`. Also have the storage-base row letter from 6-2 at hand.

**Actions**:

1. Create `cpp/mtorch/core/metal/context.h` with exactly this content, THEN
   apply the one substitution described in action 2:

   ```objc
   #pragma once
   // Internal Objective-C++ header for the Metal backend. Only .mm files may
   // include this. Core C++ files include mtorch/core/metal/ops.h instead.
   #if !defined(__OBJC__)
   #error "mtorch/core/metal/context.h is Objective-C++ only; include it from .mm files"
   #endif

   #import <Foundation/Foundation.h>
   #import <Metal/Metal.h>
   #import <MetalPerformanceShaders/MetalPerformanceShaders.h>

   #include <cstddef>
   #include <cstdint>
   #include <initializer_list>
   #include <mutex>
   #include <string>
   #include <unordered_map>

   #include "mtorch/core/tensor.h"

   namespace mtorch::metal {

   // ---------------------------------------------------------------------
   // Storage bridge (single edit point; see step 6-2 decision table).
   // Returns the base byte pointer of a Storage's data. For Metal storages
   // this is exactly the pointer MetalAllocator::allocate returned, i.e.
   // [buffer contents]; buffer_for() uses it to find the MTLBuffer.
   // ---------------------------------------------------------------------
   #define MTORCH_METAL_STORAGE_BASE(s) (MTORCH_P6_STORAGE_BASE_EXPR)

   inline const void* storage_base(const Storage& s) {
     return MTORCH_METAL_STORAGE_BASE(s);
   }

   // ---------------------------------------------------------------------
   // Context: owns device / queue / libraries / PSO cache. Thread-safe.
   // ---------------------------------------------------------------------
   class Context {
    public:
     static Context& instance();

     // True iff a Metal device exists and MTORCH_DISABLE_MPS != "1".
     // Never throws; initializes lazily on first call.
     bool enabled();

     // The following throw std::runtime_error if !enabled().
     id<MTLDevice> device();
     id<MTLCommandQueue> queue();

     // Cached compute-pipeline-state lookup by kernel function name.
     // Lookup order: default.metallib (if loaded) -> embedded-source library
     // (compiled on first miss). Throws if the function exists nowhere.
     id<MTLComputePipelineState> pipeline(const std::string& name);

     // A tiny permanent shared buffer bound in place of absent optional inputs
     // (e.g. layer_norm without weight); kernels never read it (flag-gated).
     id<MTLBuffer> dummy_buffer();

     // A fresh storageModeShared scratch buffer (matmul staging, conv col
     // buffer, attention scores...). Not cached; ARC frees it when the last
     // reference dies. Throws on allocation failure.
     id<MTLBuffer> temp_buffer(size_t bytes);

    private:
     Context() = default;
     void ensure_initialized_locked();
     void ensure_source_library_locked();
     void require_enabled_locked(const char* what);

     std::mutex mutex_;
     bool init_attempted_ = false;
     bool disabled_by_env_ = false;
     id<MTLDevice> device_ = nil;
     id<MTLCommandQueue> queue_ = nil;
     id<MTLLibrary> binary_library_ = nil;   // from default.metallib (optional)
     id<MTLLibrary> source_library_ = nil;   // from embedded source (lazy)
     id<MTLBuffer> dummy_ = nil;
     std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
   };

   // ---- defined in allocator.mm (step 6-5) ----
   // The MTLBuffer backing a Metal storage. Throws if the storage's base
   // pointer is unknown to the Metal allocator.
   id<MTLBuffer> buffer_for(const Storage& storage);

   // ---- defined in dispatch.mm (step 6-6) ----
   // Synchronous 1-D dispatch: bind buffers at [[buffer(0..k-1)]], params at
   // [[buffer(k)]] via setBytes, threadgroup size from the §0.1 formula,
   // commit, waitUntilCompleted, translate cb.error into an exception.
   void launch_1d(const char* fn,
                  std::initializer_list<id<MTLBuffer>> buffers,
                  const void* params, size_t params_size, size_t n);

   // Synchronous row dispatch for threadgroup-reduction kernels: exactly one
   // 256-thread threadgroup per row (kernels declare threadgroup float sh[256]).
   void launch_rows(const char* fn,
                    std::initializer_list<id<MTLBuffer>> buffers,
                    const void* params, size_t params_size, size_t rows);

   // Throws (with kernel/op name) if the committed command buffer failed.
   void commit_and_wait(id<MTLCommandBuffer> cb, const char* what);

   // ---- defined in matmul.mm (step 6-10) ----
   // result(M x N) = alpha * A(a_rows x a_cols) * op(B) + beta * result,
   // fp32, packed row-major, offset 0 in every buffer.
   // op(B) = B(b_rows x b_cols) transposed iff transpose_right.
   void run_gemm_f32(id<MTLBuffer> a, size_t a_rows, size_t a_cols,
                     id<MTLBuffer> b, size_t b_rows, size_t b_cols,
                     bool transpose_right,
                     id<MTLBuffer> result, double alpha, double beta);

   // Stage n elements of tensor data (f32 copy or f16->f32 cast) starting at
   // element offset elem_off into a fresh offset-0 f32 scratch buffer.
   id<MTLBuffer> stage_to_f32(const Tensor& t, size_t elem_off, size_t n);

   // Inverse: write n f32 elements from src into out (f32 copy or f32->f16
   // cast) starting at element offset elem_off of out.
   void store_from_f32(id<MTLBuffer> src, const Tensor& out, size_t elem_off, size_t n);

   }  // namespace mtorch::metal
   ```

2. In the file you just created, replace the single token
   `MTORCH_P6_STORAGE_BASE_EXPR` with the expression from the step 6-2
   decision-table row you recorded, written against the parameter name `s` —
   for example row A with method name `data` becomes:

   ```objc
   #define MTORCH_METAL_STORAGE_BASE(s) ((const void*)(s).data())
   ```

   (One small substitution is unavoidable here — the actual member name — but
   the *shape* of the expression is fixed by the table.)

3. Create `cpp/mtorch/core/metal/context.mm` with exactly this content:

   ```objc
   #include "mtorch/core/metal/context.h"

   #include <cstdlib>
   #include <cstring>
   #include <stdexcept>

   #include <dlfcn.h>

   #include "mtorch/core/metal/kernels_metal_source.h"

   namespace mtorch::metal {

   namespace {

   bool env_flag_set(const char* name) {
     const char* value = std::getenv(name);
     return value != nullptr && std::strcmp(value, "1") == 0;
   }

   bool metal_log_enabled() {
     static const bool enabled = env_flag_set("MTORCH_METAL_LOG");
     return enabled;
   }

   std::string error_text(NSError* error) {
     if (error == nil) {
       return "unknown error";
     }
     // localizedDescription is an NSString; UTF8String yields a C string
     // owned by the autorelease pool, so copy it into a std::string now.
     return std::string([[error localizedDescription] UTF8String]);
   }

   // Anchor symbol: dladdr() on its address reports the path of the shared
   // object this code was linked into (mtorch/_C.cpython-*.so). The metallib
   // is expected next to it, as arranged by setup.py.
   void metallib_path_anchor() {}

   std::string metallib_path_next_to_extension() {
     Dl_info info;
     if (dladdr(reinterpret_cast<void*>(&metallib_path_anchor), &info) == 0 ||
         info.dli_fname == nullptr) {
       return std::string();
     }
     std::string path(info.dli_fname);
     const size_t slash = path.rfind('/');
     if (slash == std::string::npos) {
       return std::string();
     }
     return path.substr(0, slash + 1) + "default.metallib";
   }

   }  // namespace

   Context& Context::instance() {
     // Intentionally leaked: never run Metal teardown in static destructors.
     static Context* context = new Context();
     return *context;
   }

   void Context::ensure_initialized_locked() {
     if (init_attempted_) {
       return;
     }
     init_attempted_ = true;
     disabled_by_env_ = env_flag_set("MTORCH_DISABLE_MPS");
     if (disabled_by_env_) {
       return;
     }
     // MTLCreateSystemDefaultDevice(): obtain the system GPU. Returns nil in
     // environments without Metal (then the backend simply stays disabled).
     device_ = MTLCreateSystemDefaultDevice();
     if (device_ == nil) {
       return;
     }
     // One persistent queue; command buffers are created per launch.
     queue_ = [device_ newCommandQueue];
     if (queue_ == nil) {
       device_ = nil;
       return;
     }
     // Optional fast path: the offline-compiled library next to the .so.
     const std::string lib_path = metallib_path_next_to_extension();
     if (!lib_path.empty()) {
       NSURL* url = [NSURL fileURLWithPath:
           [NSString stringWithUTF8String:lib_path.c_str()]];
       if ([[NSFileManager defaultManager] fileExistsAtPath:[url path]]) {
         NSError* error = nil;
         // newLibraryWithURL: loads a precompiled .metallib from disk.
         binary_library_ = [device_ newLibraryWithURL:url error:&error];
         if (binary_library_ == nil && metal_log_enabled()) {
           fprintf(stderr, "[mtorch-metal] failed to load %s: %s (falling back to source)\n",
                   lib_path.c_str(), error_text(error).c_str());
         }
       }
     }
     if (metal_log_enabled()) {
       fprintf(stderr, "[mtorch-metal] device: %s, metallib: %s\n",
               [[device_ name] UTF8String],
               binary_library_ != nil ? "loaded" : "absent (source compile)");
     }
   }

   void Context::ensure_source_library_locked() {
     if (source_library_ != nil) {
       return;
     }
     // MTLCompileOptions configures the runtime shader compiler.
     MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
     // Disable fast math to match CPU numerics (§0.4 / §2). fastMathEnabled is
     // marked deprecated on new SDKs in favor of mathMode, but still works;
     // silence just that warning.
   #pragma clang diagnostic push
   #pragma clang diagnostic ignored "-Wdeprecated-declarations"
     options.fastMathEnabled = NO;
   #pragma clang diagnostic pop
     NSError* error = nil;
     // newLibraryWithSource: compiles the embedded MSL text at runtime.
     source_library_ = [device_
         newLibraryWithSource:[NSString stringWithUTF8String:embedded_metal_source()]
                      options:options
                        error:&error];
     if (source_library_ == nil) {
       throw std::runtime_error(
           "mtorch metal: runtime compilation of kernels.metal failed: " +
           error_text(error));
     }
   }

   void Context::require_enabled_locked(const char* what) {
     ensure_initialized_locked();
     if (disabled_by_env_) {
       throw std::runtime_error(std::string("mtorch metal: ") + what +
                                " requested but MTORCH_DISABLE_MPS=1");
     }
     if (device_ == nil) {
       throw std::runtime_error(std::string("mtorch metal: ") + what +
                                " requested but no Metal device is available");
     }
   }

   bool Context::enabled() {
     std::lock_guard<std::mutex> lock(mutex_);
     ensure_initialized_locked();
     return device_ != nil && !disabled_by_env_;
   }

   id<MTLDevice> Context::device() {
     std::lock_guard<std::mutex> lock(mutex_);
     require_enabled_locked("device");
     return device_;
   }

   id<MTLCommandQueue> Context::queue() {
     std::lock_guard<std::mutex> lock(mutex_);
     require_enabled_locked("command queue");
     return queue_;
   }

   id<MTLComputePipelineState> Context::pipeline(const std::string& name) {
     std::lock_guard<std::mutex> lock(mutex_);
     require_enabled_locked("compute pipeline");
     auto it = pipelines_.find(name);
     if (it != pipelines_.end()) {
       return it->second;
     }
     NSString* fn_name = [NSString stringWithUTF8String:name.c_str()];
     id<MTLFunction> fn = nil;
     if (binary_library_ != nil) {
       // Per-function fallback: a stale metallib missing a newly added kernel
       // must not be fatal — we retry against the embedded source below.
       fn = [binary_library_ newFunctionWithName:fn_name];
     }
     if (fn == nil) {
       ensure_source_library_locked();
       fn = [source_library_ newFunctionWithName:fn_name];
     }
     if (fn == nil) {
       throw std::runtime_error("mtorch metal: kernel function not found: " + name);
     }
     NSError* error = nil;
     // The PSO is the GPU-specific compiled executable of one kernel.
     id<MTLComputePipelineState> pso =
         [device_ newComputePipelineStateWithFunction:fn error:&error];
     if (pso == nil) {
       throw std::runtime_error("mtorch metal: pipeline creation failed for '" +
                                name + "': " + error_text(error));
     }
     pipelines_.emplace(name, pso);
     return pso;
   }

   id<MTLBuffer> Context::dummy_buffer() {
     std::lock_guard<std::mutex> lock(mutex_);
     require_enabled_locked("dummy buffer");
     if (dummy_ == nil) {
       dummy_ = [device_ newBufferWithLength:16
                                     options:MTLResourceStorageModeShared];
       if (dummy_ == nil) {
         throw std::runtime_error("mtorch metal: dummy buffer allocation failed");
       }
     }
     return dummy_;
   }

   id<MTLBuffer> Context::temp_buffer(size_t bytes) {
     // device() takes the lock itself; do not hold ours around it.
     id<MTLDevice> dev = device();
     if (bytes == 0) {
       bytes = 16;  // Metal rejects zero-length buffers.
     }
     id<MTLBuffer> buffer =
         [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
     if (buffer == nil) {
       throw std::runtime_error("mtorch metal: scratch allocation failed (" +
                                std::to_string(bytes) + " bytes)");
     }
     return buffer;
   }

   }  // namespace mtorch::metal
   ```

**Verification**:

```bash
python3 setup.py build_ext --inplace
python3 -c "import mtorch; print('import ok')"
pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
tail -1 docs/design/baseline/tests-phase6-cpu.txt
```

PASS iff the build compiles the two new files without errors (watch for the
first error only, per §6 rule 1), `import ok` prints, and the summary matches
the frozen baseline. Nothing calls the context yet, so no runtime behavior may
change.

**On failure**: STANDARD FAIL-6 (`<new files>` =
`cpp/mtorch/core/metal/context.h cpp/mtorch/core/metal/context.mm`). Typical
first errors: a wrong storage-base expression (compile error naming the member
— re-check the 6-2 table) or a leftover `MTORCH_P6_STORAGE_BASE_EXPR` token.

**Commit**:

```bash
git add cpp/mtorch/core/metal/context.h cpp/mtorch/core/metal/context.mm
git commit -m "feat(phase6-4): metal context singleton (device/queue/library/PSO cache)"
```

---

### Step 6-5: MetalAllocator + registry hookup — device "mps" becomes available

**Goal**: Implement the Phase 5 `mtorch::detail::Allocator` interface on top of
shared `MTLBuffer`s and register it for `DeviceType::Metal` at import. After
this step `mtorch.backends.mps.is_available()` returns `True` and **every**
mtorch op already works on `"mps"` tensors via the zero-copy CPU fallback —
before a single op kernel exists. The Phase 5 parity suite goes from "skipped"
to "running" here.

**Background: how a host pointer and an MTLBuffer stay glued together (6
lines).** `allocate(n)` creates an `MTLBuffer` (`newBufferWithLength:options:`
with `MTLResourceStorageModeShared`) and returns `[buffer contents]` — the CPU
address of the same memory. The Storage layer only ever sees that raw pointer.
The allocator keeps a `pointer → id<MTLBuffer>` map (mutex-protected): kernels
call `buffer_for(storage)` to translate back, and `deallocate(ptr)` erases the
map entry — under ARC, removing the last `id<MTLBuffer>` reference from the map
releases the buffer and frees the memory. Nothing else owns the buffer, so
lifetime is exactly the Storage's lifetime.

**Background: why the copy_* hooks are plain memcpy (4 lines).** On unified
memory the "host" and "device" bytes are the same bytes, so
`copy_from_host` / `copy_to_host` / `copy_on_device` are `std::memcpy`. This is
race-free in this phase because every kernel launch is synchronous
(`waitUntilCompleted` returns before any op returns), so no GPU work is ever in
flight while the CPU touches a buffer.

**Preconditions**: 6-4 committed; the allocator interface printout from step
6-2 action 3 at hand.

**Actions**:

1. Create `cpp/mtorch/core/metal/allocator.mm` with exactly this content.
   **Signature rule**: the `override` methods below use the contract
   signatures (`allocate(size_t)` etc.). If Phase 5's `allocator.h` declares a
   different parameter or return type, the build will fail with
   "...marked 'override' but does not override..." — in that case copy the
   exact virtual signatures from `allocator.h` into these overrides,
   changing ONLY parameter/return types, never names or behavior. If the
   *names* differ from the contract, STANDARD FAIL-6 instead.

   ```objc
   #include "mtorch/core/detail/allocator.h"
   #include "mtorch/core/metal/context.h"

   #include <cstring>
   #include <mutex>
   #include <stdexcept>
   #include <unordered_map>

   namespace mtorch::metal {

   class MetalAllocator final : public mtorch::detail::Allocator {
    public:
     void* allocate(size_t bytes) override {
       @autoreleasepool {
         if (bytes == 0) {
           bytes = 16;  // Metal rejects zero-length buffers; empty tensors
                        // still need a live pointer identity.
         }
         id<MTLDevice> device = Context::instance().device();
         // newBufferWithLength:options: allocates GPU-visible memory;
         // storageModeShared = one allocation visible to CPU and GPU (§0.3).
         id<MTLBuffer> buffer =
             [device newBufferWithLength:bytes
                                 options:MTLResourceStorageModeShared];
         if (buffer == nil) {
           throw std::runtime_error(
               "mtorch metal: MTLBuffer allocation failed (" +
               std::to_string(bytes) + " bytes)");
         }
         // [buffer contents]: the CPU address of the shared allocation.
         void* pointer = [buffer contents];
         std::lock_guard<std::mutex> lock(mutex_);
         buffers_[pointer] = buffer;  // ARC: the map retains the buffer.
         return pointer;
       }
     }

     void deallocate(void* pointer) override {
       if (pointer == nullptr) {
         return;
       }
       std::lock_guard<std::mutex> lock(mutex_);
       // Erasing drops the map's (only) strong reference; ARC releases the
       // MTLBuffer and the shared memory is freed.
       buffers_.erase(pointer);
     }

     void copy_from_host(void* dst, const void* src, size_t bytes) override {
       std::memcpy(dst, src, bytes);  // unified memory: H2D is memcpy (§0.3)
     }

     void copy_to_host(void* dst, const void* src, size_t bytes) override {
       std::memcpy(dst, src, bytes);
     }

     void copy_on_device(void* dst, const void* src, size_t bytes) override {
       std::memcpy(dst, src, bytes);
     }

     bool host_accessible() const override {
       // THE key contract bit: true means ops without a Metal kernel may run
       // their CPU implementation directly on the shared buffer.
       return true;
     }

     id<MTLBuffer> lookup(const void* pointer) {
       std::lock_guard<std::mutex> lock(mutex_);
       auto it = buffers_.find(pointer);
       return it == buffers_.end() ? nil : it->second;
     }

    private:
     std::mutex mutex_;
     std::unordered_map<const void*, id<MTLBuffer>> buffers_;
   };

   namespace {

   MetalAllocator* allocator_instance() {
     // Leaked on purpose, same rationale as the Context singleton.
     static MetalAllocator* allocator = new MetalAllocator();
     return allocator;
   }

   // Static registrar: runs when mtorch/_C is loaded (= import time). If the
   // context cannot initialize (no GPU, MTORCH_DISABLE_MPS=1) nothing is
   // registered and mtorch.backends.mps.is_available() stays False — the
   // Phase 5 registry is the single source of truth for availability.
   struct MetalRegistrar {
     MetalRegistrar() {
       if (Context::instance().enabled()) {
         mtorch::detail::register_allocator(mtorch::DeviceType::Metal,
                                            allocator_instance());
       }
     }
   };
   MetalRegistrar metal_registrar;

   }  // namespace

   id<MTLBuffer> buffer_for(const Storage& storage) {
     id<MTLBuffer> buffer = allocator_instance()->lookup(storage_base(storage));
     if (buffer == nil) {
       throw std::runtime_error(
           "mtorch metal: internal error: storage bytes are not backed by a "
           "Metal buffer (storage_base mismatch or non-Metal storage reached "
           "a Metal kernel)");
     }
     return buffer;
   }

   }  // namespace mtorch::metal
   ```

2. If step 6-2's audit showed `register_allocator` lives in a namespace other
   than `mtorch::detail`, or `DeviceType` outside `mtorch`, adjust ONLY those
   two qualified names in the registrar (mechanical rename; anything more →
   STANDARD FAIL-6).

**Verification**:

```bash
python3 setup.py build_ext --inplace
python3 -c "
import mtorch
print(mtorch.backends.mps.is_available())
x = mtorch.ones(4, 4, device='mps')
print(x.device)
y = (x + x).cpu()
print(float(y.sum().item()))
z = mtorch.zeros(3, device='cpu').to('mps').to('cpu')
print(z.tolist())
"
MTORCH_DISABLE_MPS=1 python3 -c "import mtorch; print(mtorch.backends.mps.is_available())"
pytest tests/compat/test_device_parity.py -q | tail -3
pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
tail -1 docs/design/baseline/tests-phase6-cpu.txt
```

Expected: `True`, `mps` (or `mps:0`, whichever Phase 5 prints), `32.0`,
`[0.0, 0.0, 0.0]`, then `False` for the kill-switch run; the parity suite now
actually executes (summary shows a nonzero `passed`, zero `failed` — every op
takes the CPU-fallback path on the shared buffer); the `--ignore` run matches
the frozen baseline. Any crash here means the allocator/registry glue is wrong
— do NOT proceed.

**On failure**: STANDARD FAIL-6 (`<new files>` =
`cpp/mtorch/core/metal/allocator.mm`).

**Commit**:

```bash
git add cpp/mtorch/core/metal/allocator.mm
git commit -m "feat(phase6-5): MetalAllocator over shared MTLBuffers; register mps device"
```

---

### Step 6-6: Dispatch helpers (ops.h, dispatch.mm) + tools/metal_ratios.py

**Goal**: (a) `ops.h` — the single pure-C++ header the core section files will
include, declaring `metal_ready()`, the shared dense-tensor guard helper, and
(added family by family) every `use_*`/op pair; (b) `dispatch.mm` —
`launch_1d`, `launch_rows`, `commit_and_wait`, the environment switches, the
launch logger; (c) `tools/metal_ratios.py` — the timing/ratio tool used by
every op-family step and by the phase gate.

**Preconditions**: 6-5 committed
(`python3 -c "import mtorch; print(mtorch.backends.mps.is_available())"`
prints `True`).

**Actions**:

1. Create `cpp/mtorch/core/metal/ops.h` with exactly this content:

   ```cpp
   #pragma once
   // The ONLY Metal-backend header core C++ files may include. Pure C++:
   // no Objective-C anywhere. On non-Apple builds every guard is an inline
   // stub returning false, so call sites compile everywhere.
   //
   // Convention (docs/design/09-phase6-metal-backend.md §0.8): each op family
   // contributes a use_<op>() guard and a metal::<op>() entry point. The entry
   // point may assume every condition its guard checked. Guards never throw
   // and never allocate; their first check is always the device type.

   #include <cstdint>
   #include <string>

   #include "mtorch/core/tensor.h"

   namespace mtorch::metal {

   #if defined(__APPLE__)

   // True iff the Metal context is usable AND MTORCH_METAL_FORCE_FALLBACK is
   // not set. Every guard calls this after its device check.
   bool metal_ready();

   // Shared guard core: Metal device, dtype f32/f16, contiguous, numel > 0,
   // offset and offset+numel within uint32 range.
   bool metal_dense_f32f16(const Tensor& t);

   // ---- op families are appended here, one block per step 6-7 .. 6-13 ----

   #else  // !defined(__APPLE__)

   inline bool metal_ready() { return false; }
   inline bool metal_dense_f32f16(const Tensor&) { return false; }

   #endif  // defined(__APPLE__)

   }  // namespace mtorch::metal
   ```

   Note the append protocol used by later steps: real declarations go
   immediately after the `---- op families are appended here ----` comment
   line; matching non-Apple inline stubs go immediately after the
   `inline bool metal_dense_f32f16` stub line. Each later step spells out both
   blocks verbatim.

2. Create `cpp/mtorch/core/metal/dispatch.mm` with exactly this content:

   ```objc
   #include "mtorch/core/metal/ops.h"
   #include "mtorch/core/metal/context.h"

   #include <algorithm>
   #include <cstdio>
   #include <cstdlib>
   #include <cstring>
   #include <stdexcept>

   namespace mtorch::metal {

   namespace {

   bool env_flag(const char* name) {
     const char* value = std::getenv(name);
     return value != nullptr && std::strcmp(value, "1") == 0;
   }

   bool launch_log_enabled() {
     static const bool enabled = env_flag("MTORCH_METAL_LOG");
     return enabled;
   }

   }  // namespace

   bool metal_ready() {
     static const bool forced_fallback = env_flag("MTORCH_METAL_FORCE_FALLBACK");
     if (forced_fallback) {
       return false;
     }
     return Context::instance().enabled();
   }

   bool metal_dense_f32f16(const Tensor& t) {
     if (t.device.type != DeviceType::Metal) {
       return false;
     }
     if (t.dtype != ScalarType::Float32 && t.dtype != ScalarType::Float16) {
       return false;
     }
     if (!t.is_contiguous()) {
       return false;
     }
     const int64_t n = t.numel();
     if (n <= 0) {
       return false;  // empty: the CPU no-op path is fine and simpler
     }
     constexpr int64_t kLimit = 0xffffffffLL;  // params carry uint32 offsets
     if (t.offset < 0 || n > kLimit || t.offset > kLimit - n) {
       return false;
     }
     return true;
   }

   void commit_and_wait(id<MTLCommandBuffer> cb, const char* what) {
     // commit: hand the recorded work to the GPU. waitUntilCompleted: block
     // this thread until the GPU finished it (synchronous v1 contract).
     [cb commit];
     [cb waitUntilCompleted];
     // cb.error is non-nil if the GPU work failed (page fault, timeout...).
     if (cb.error != nil) {
       throw std::runtime_error(
           std::string("mtorch metal: command buffer failed in ") + what +
           ": " + [[cb.error localizedDescription] UTF8String]);
     }
   }

   void launch_1d(const char* fn,
                  std::initializer_list<id<MTLBuffer>> buffers,
                  const void* params, size_t params_size, size_t n) {
     if (n == 0) {
       return;
     }
     @autoreleasepool {
       Context& ctx = Context::instance();
       if (launch_log_enabled()) {
         fprintf(stderr, "[mtorch-metal] launch %s n=%zu\n", fn, n);
       }
       id<MTLComputePipelineState> pso = ctx.pipeline(fn);
       // One command buffer per launch (single-use by design).
       id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
       id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
       [enc setComputePipelineState:pso];
       NSUInteger index = 0;
       for (id<MTLBuffer> buffer : buffers) {
         // Bind each tensor buffer at [[buffer(index)]]; offsets are always 0
         // here — element offsets travel inside the param struct instead,
         // which sidesteps every buffer-offset alignment rule.
         [enc setBuffer:buffer offset:0 atIndex:index];
         index += 1;
       }
       // setBytes copies a small parameter struct by value into the command
       // stream; it appears as the last [[buffer(k)]] (constant address space).
       [enc setBytes:params length:params_size atIndex:index];
       // §0.1 threadgroup-size formula:
       //   w   = SIMD width (threadExecutionWidth, 32 on Apple GPUs)
       //   cap = maxTotalThreadsPerThreadgroup (per-PSO, register-dependent)
       //   tg  = min(cap, round_up(n, w))  -- multiple of w, <= cap
       //   threadgroups = ceil(n / tg)     -- overhang guarded in the kernel
       const NSUInteger width = pso.threadExecutionWidth;
       const NSUInteger cap = pso.maxTotalThreadsPerThreadgroup;
       NSUInteger tg = ((n + width - 1) / width) * width;
       tg = std::min<NSUInteger>(tg, cap);
       if (tg == 0) {
         tg = width;
       }
       const NSUInteger groups = (n + tg - 1) / tg;
       [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
       [enc endEncoding];
       commit_and_wait(cb, fn);
     }
   }

   void launch_rows(const char* fn,
                    std::initializer_list<id<MTLBuffer>> buffers,
                    const void* params, size_t params_size, size_t rows) {
     if (rows == 0) {
       return;
     }
     @autoreleasepool {
       Context& ctx = Context::instance();
       if (launch_log_enabled()) {
         fprintf(stderr, "[mtorch-metal] launch %s rows=%zu\n", fn, rows);
       }
       id<MTLComputePipelineState> pso = ctx.pipeline(fn);
       // Reduction kernels hard-code `threadgroup float sh[256]` and a
       // 256-thread tree reduction; the PSO must allow 256 threads. If heavy
       // register use ever lowers the cap below 256, fail loudly (the kernel
       // would silently compute garbage otherwise).
       if (pso.maxTotalThreadsPerThreadgroup < 256) {
         throw std::runtime_error(
             std::string("mtorch metal: ") + fn +
             " needs a 256-thread threadgroup but the pipeline allows only " +
             std::to_string(pso.maxTotalThreadsPerThreadgroup));
       }
       id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
       id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
       [enc setComputePipelineState:pso];
       NSUInteger index = 0;
       for (id<MTLBuffer> buffer : buffers) {
         [enc setBuffer:buffer offset:0 atIndex:index];
         index += 1;
       }
       [enc setBytes:params length:params_size atIndex:index];
       // Exactly one threadgroup per row; each threadgroup has 256 threads
       // that stride across the row's columns.
       [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
       [enc endEncoding];
       commit_and_wait(cb, fn);
     }
   }

   }  // namespace mtorch::metal
   ```

3. Create `tools/metal_ratios.py` with exactly this content (used by every op
   family for the "vs torch MPS" measurement, and by the 6-14 gate to write
   `docs/design/baseline/metal-ratios.md`):

   ```python
   #!/usr/bin/env python3
   """Time mtorch ops on cpu and mps, and torch on mps; print a ratio table.

   Usage (from the repo root):
     python3 tools/metal_ratios.py                    # all families
     python3 tools/metal_ratios.py --family binary    # one family
     python3 tools/metal_ratios.py --check            # also print max|diff|
     python3 tools/metal_ratios.py --write docs/design/baseline/metal-ratios.md

   Ratios are informational (v1 Metal is allowed to be slower than torch MPS);
   the correctness gates are the parity suite and the CPU suite. --check is an
   advisory numeric cross-check against torch on mps, never a pass/fail gate.
   """
   from __future__ import annotations

   import argparse
   import statistics
   import time


   def base_tensor(mod, shape, dtype_name):
       """Deterministic, non-degenerate values, identical for torch and mtorch:
       ((arange(n) % 7) - 3) * 0.25 reshaped to `shape`."""
       n = 1
       for s in shape:
           n *= s
       t = mod.arange(n, dtype=mod.float32)
       t = ((t % 7.0) - 3.0) * 0.25
       t = t.reshape(shape)
       if dtype_name == "float16":
           t = t.to(mod.float16)
       return t


   def positive_tensor(mod, shape, dtype_name):
       return base_tensor(mod, shape, dtype_name).abs() + 0.5


   # Each case: (family, case_id, build) where build(mod, device) returns a
   # zero-argument callable performing one op invocation.
   def _cases():
       C = []

       def add(family, case_id, build):
           C.append((family, case_id, build))

       def unary_method(name):
           def make(shape, dtype):
               def build(mod, device):
                   x = positive_tensor(mod, shape, dtype).to(device)
                   return lambda: getattr(x, name)()
               return build
           return make

       def fill_case(shape, dtype):
           def build(mod, device):
               x = base_tensor(mod, shape, dtype).to(device)
               return lambda: x.fill_(1.5)
           return build

       def copy_case(shape, dtype):
           def build(mod, device):
               x = base_tensor(mod, shape, dtype).to(device)
               y = base_tensor(mod, shape, dtype).to(device)
               return lambda: y.copy_(x)
           return build

       add("fill", "fill_f32_1024x1024", fill_case((1024, 1024), "float32"))
       add("fill", "copy_f32_1024x1024", copy_case((1024, 1024), "float32"))

       def binop(op, shape, dtype):
           def build(mod, device):
               a = base_tensor(mod, shape, dtype).to(device)
               b = (positive_tensor(mod, shape, dtype)).to(device)
               if op == "add":
                   return lambda: a + b
               if op == "mul":
                   return lambda: a * b
               if op == "div":
                   return lambda: a / b
               return lambda: a - b
           return build

       add("binary", "add_f32_1024x1024", binop("add", (1024, 1024), "float32"))
       add("binary", "mul_f16_2x4x64x64", binop("mul", (2, 4, 64, 64), "float16"))
       add("binary", "div_f32_1024x1024", binop("div", (1024, 1024), "float32"))

       add("unary", "exp_f32_1024x1024", unary_method("exp")((1024, 1024), "float32"))
       add("unary", "sqrt_f16_2x77x768", unary_method("sqrt")((2, 77, 768), "float16"))

       def silu(shape, dtype):
           def build(mod, device):
               x = base_tensor(mod, shape, dtype).to(device)
               fn = mod.nn.functional.silu
               return lambda: fn(x)
           return build

       def gelu_tanh(shape, dtype):
           def build(mod, device):
               x = base_tensor(mod, shape, dtype).to(device)
               fn = mod.nn.functional.gelu
               return lambda: fn(x, approximate="tanh")
           return build

       add("unary", "silu_f16_2x320x32x32", silu((2, 320, 32, 32), "float16"))
       add("unary", "gelu_tanh_f32_2x77x768", gelu_tanh((2, 77, 768), "float32"))

       def matmul(m, k, n, dtype):
           def build(mod, device):
               a = base_tensor(mod, (m, k), dtype).to(device)
               b = base_tensor(mod, (k, n), dtype).to(device)
               return lambda: a @ b
           return build

       add("matmul", "matmul_f32_512x512", matmul(512, 512, 512, "float32"))
       add("matmul", "matmul_f16_256x256", matmul(256, 256, 256, "float16"))

       def softmax(shape, dtype):
           def build(mod, device):
               x = base_tensor(mod, shape, dtype).to(device)
               fn = mod.nn.functional.softmax
               return lambda: fn(x, dim=-1)
           return build

       def layer_norm(shape, dtype):
           def build(mod, device):
               x = base_tensor(mod, shape, dtype).to(device)
               w = mod.ones(shape[-1]).to(x.dtype).to(device)
               b = mod.zeros(shape[-1]).to(x.dtype).to(device)
               fn = mod.nn.functional.layer_norm
               return lambda: fn(x, [shape[-1]], w, b)
           return build

       def group_norm(shape, groups, dtype):
           def build(mod, device):
               x = base_tensor(mod, shape, dtype).to(device)
               fn = mod.nn.functional.group_norm
               return lambda: fn(x, groups)
           return build

       add("norm", "softmax_f32_512x512", softmax((512, 512), "float32"))
       add("norm", "layer_norm_f16_2x77x768", layer_norm((2, 77, 768), "float16"))
       add("norm", "group_norm_f32_2x64x32x32_g8", group_norm((2, 64, 32, 32), 8, "float32"))

       def conv2d(shape, oc, k, dtype):
           def build(mod, device):
               x = base_tensor(mod, shape, dtype).to(device)
               w = base_tensor(mod, (oc, shape[1], k, k), dtype).to(device)
               fn = mod.nn.functional.conv2d
               return lambda: fn(x, w, padding=1)
           return build

       add("conv", "conv2d_f32_1x64x32x32_k3", conv2d((1, 64, 32, 32), 64, 3, "float32"))
       add("conv", "conv2d_f16_1x64x32x32_k3", conv2d((1, 64, 32, 32), 64, 3, "float16"))

       def sdpa(shape, dtype):
           def build(mod, device):
               q = base_tensor(mod, shape, dtype).to(device)
               k = positive_tensor(mod, shape, dtype).to(device)
               v = base_tensor(mod, shape, dtype).to(device)
               fn = mod.nn.functional.scaled_dot_product_attention
               return lambda: fn(q, k, v, is_causal=True)
           return build

       add("sdpa", "sdpa_f32_2x8x64x64", sdpa((2, 8, 64, 64), "float32"))
       add("sdpa", "sdpa_f16_2x8x77x64", sdpa((2, 8, 77, 64), "float16"))
       return C


   def measure(run, sync, warmup, repeat):
       for _ in range(warmup):
           run()
           sync()
       samples = []
       for _ in range(repeat):
           start = time.perf_counter()
           run()
           sync()
           samples.append(time.perf_counter() - start)
       return statistics.median(samples)


   def flatten(value):
       if isinstance(value, (list, tuple)):
           out = []
           for item in value:
               out.extend(flatten(item))
           return out
       return [float(value)]


   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument("--family", default="", help="run only this family")
       parser.add_argument("--repeat", type=int, default=10)
       parser.add_argument("--warmup", type=int, default=3)
       parser.add_argument("--check", action="store_true",
                           help="also print max|mtorch_mps - torch_mps| (advisory)")
       parser.add_argument("--write", default="", help="write the table to this file")
       args = parser.parse_args()

       import mtorch
       import torch

       if not mtorch.backends.mps.is_available():
           raise SystemExit("mtorch mps is not available (MTORCH_DISABLE_MPS set, "
                            "or the Metal allocator is not registered)")
       if not torch.backends.mps.is_available():
           raise SystemExit("torch mps is not available")

       def no_sync():
           pass

       lines = [
           "| case | mtorch cpu ms | mtorch mps ms | torch mps ms | mps/cpu | mtorch-mps/torch-mps |",
           "|---|---|---|---|---|---|",
       ]
       for family, case_id, build in _cases():
           if args.family and family != args.family:
               continue
           run_cpu = build(mtorch, "cpu")
           run_mps = build(mtorch, "mps")
           run_torch = build(torch, "mps")
           t_cpu = measure(run_cpu, no_sync, args.warmup, args.repeat)
           t_mps = measure(run_mps, no_sync, args.warmup, args.repeat)  # sync launches
           t_torch = measure(run_torch, torch.mps.synchronize, args.warmup, args.repeat)
           line = "| {} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.2f} |".format(
               case_id, t_cpu * 1e3, t_mps * 1e3, t_torch * 1e3,
               t_mps / t_cpu, t_mps / t_torch)
           if args.check:
               a = flatten(run_mps().cpu().tolist())
               b = flatten(run_torch().cpu().tolist())
               diff = max(abs(x - y) for x, y in zip(a, b)) if a else 0.0
               line += "  <!-- max|diff| vs torch-mps: {:.3e} -->".format(diff)
           lines.append(line)
           print(lines[-1])

       if args.write:
           with open(args.write, "w", encoding="utf-8") as fh:
               fh.write("# mtorch Metal (mps) timing ratios\n\n")
               fh.write("Produced by `python3 tools/metal_ratios.py --write ...`.\n")
               fh.write("Informational only; the phase gate is correctness "
                        "(docs/design/09-phase6-metal-backend.md step 6-14).\n\n")
               fh.write("\n".join(lines) + "\n")
           print(f"wrote {args.write}")


   if __name__ == "__main__":
       main()
   ```

   Note: the in-place `fill`/`copy` cases mutate their operand between
   repeats; that is fine for timing. `--check` skips nothing — in-place ops
   return the mutated tensor in both libraries.

**Verification**:

```bash
python3 setup.py build_ext --inplace
pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
tail -1 docs/design/baseline/tests-phase6-cpu.txt
pytest tests/compat/test_device_parity.py -q | tail -3
python3 tools/metal_ratios.py --family fill --repeat 3
```

PASS iff build + both suites behave per STANDARD VERIFY-6, and the ratio tool
prints its header plus the two `fill` rows (every op still takes the CPU
fallback, so `mps/cpu` will be ≈1.0 — that is expected before 6-7).

**On failure**: STANDARD FAIL-6 (`<new files>` =
`cpp/mtorch/core/metal/ops.h cpp/mtorch/core/metal/dispatch.mm tools/metal_ratios.py`).

**Commit**:

```bash
git add cpp/mtorch/core/metal/ops.h cpp/mtorch/core/metal/dispatch.mm tools/metal_ratios.py
git commit -m "feat(phase6-6): metal dispatch helpers and ratio benchmark tool"
```

---

### Step 6-7: Op family (a) — fill / copy

**Goal**: First real GPU execution. `Tensor::fill_inplace` and
`Tensor::copy_from` on dense f32/f16 mps tensors run Metal kernels; everything
else falls back. This step also proves the whole 6-3…6-6 machinery end to end
— if anything in the plumbing is wrong, it surfaces here, on the two simplest
possible kernels.

**Preconditions**:

```bash
grep -n 'void Tensor::fill_inplace(double value) {' cpp/mtorch/core/tensor_core.cpp
grep -n 'void Tensor::copy_from(const Tensor& source) {' cpp/mtorch/core/tensor_core.cpp
```

Both must print exactly one line each (these member functions moved to
`tensor_core.cpp` in step 2b-2). If either prints nothing, locate the symbol
with `grep -rn 'void Tensor::fill_inplace' cpp/mtorch/core/` and use that file
consistently below; if it is still in a file this guide does not expect,
STANDARD FAIL-6.

**Actions**:

1. Append to `cpp/mtorch/core/metal/kernels.metal` (at the end of the file):

   ```
   // ---------------- family (a): fill / copy ----------------

   struct FillParams {
     uint n;
     uint out_off;
     float value;
   };

   kernel void mtorch_fill_f32(
       device float* out [[buffer(0)]],
       constant FillParams& p [[buffer(1)]],
       uint gid [[thread_position_in_grid]]) {
     if (gid >= p.n) {
       return;
     }
     out[p.out_off + gid] = p.value;
   }

   kernel void mtorch_fill_f16(
       device half* out [[buffer(0)]],
       constant FillParams& p [[buffer(1)]],
       uint gid [[thread_position_in_grid]]) {
     if (gid >= p.n) {
       return;
     }
     out[p.out_off + gid] = half(p.value);
   }

   struct CopyParams {
     uint n;
     uint src_off;
     uint dst_off;
   };

   kernel void mtorch_copy_f32(
       device const float* src [[buffer(0)]],
       device float* dst [[buffer(1)]],
       constant CopyParams& p [[buffer(2)]],
       uint gid [[thread_position_in_grid]]) {
     if (gid >= p.n) {
       return;
     }
     dst[p.dst_off + gid] = src[p.src_off + gid];
   }

   kernel void mtorch_copy_f16(
       device const half* src [[buffer(0)]],
       device half* dst [[buffer(1)]],
       constant CopyParams& p [[buffer(2)]],
       uint gid [[thread_position_in_grid]]) {
     if (gid >= p.n) {
       return;
     }
     dst[p.dst_off + gid] = src[p.src_off + gid];
   }
   ```

2. In `cpp/mtorch/core/metal/ops.h`, insert immediately AFTER the line
   `// ---- op families are appended here, one block per step 6-7 .. 6-13 ----`:

   ```cpp
   // family (a): fill / copy (step 6-7)
   bool use_fill(const Tensor& t);
   void fill(Tensor& t, double value);
   bool use_copy(const Tensor& dst, const Tensor& src);
   void copy(Tensor& dst, const Tensor& src);
   ```

   and insert immediately AFTER the line
   `inline bool metal_dense_f32f16(const Tensor&) { return false; }`:

   ```cpp
   inline bool use_fill(const Tensor&) { return false; }
   inline void fill(Tensor&, double) {}
   inline bool use_copy(const Tensor&, const Tensor&) { return false; }
   inline void copy(Tensor&, const Tensor&) {}
   ```

3. Create `cpp/mtorch/core/metal/fill_copy.mm` with exactly this content:

   ```objc
   #include "mtorch/core/metal/ops.h"
   #include "mtorch/core/metal/context.h"

   #include <cstdint>

   namespace mtorch::metal {

   namespace {

   // Mirrors of the MSL structs (same field order, all fields 4 bytes; §0.4).
   struct FillParams {
     uint32_t n;
     uint32_t out_off;
     float value;
   };

   struct CopyParams {
     uint32_t n;
     uint32_t src_off;
     uint32_t dst_off;
   };

   }  // namespace

   bool use_fill(const Tensor& t) {
     if (t.device.type != DeviceType::Metal) {
       return false;
     }
     if (!metal_ready()) {
       return false;
     }
     if (t.requires_grad) {
       return false;  // §0.8 rule 3: Metal never touches grad-tracked tensors
     }
     return metal_dense_f32f16(t);
   }

   void fill(Tensor& t, double value) {
     FillParams params{static_cast<uint32_t>(t.numel()),
                       static_cast<uint32_t>(t.offset),
                       static_cast<float>(value)};
     const char* fn =
         t.dtype == ScalarType::Float32 ? "mtorch_fill_f32" : "mtorch_fill_f16";
     launch_1d(fn, {buffer_for(*t.storage)}, &params, sizeof(params),
               static_cast<size_t>(t.numel()));
   }

   bool use_copy(const Tensor& dst, const Tensor& src) {
     if (dst.device.type != DeviceType::Metal ||
         src.device.type != DeviceType::Metal) {
       return false;
     }
     if (!metal_ready()) {
       return false;
     }
     if (dst.requires_grad || src.requires_grad) {
       return false;
     }
     if (dst.dtype != src.dtype) {
       return false;  // dtype-converting copy_ stays on the CPU path
     }
     if (dst.sizes != src.sizes) {
       return false;  // broadcasting copy_ stays on the CPU path
     }
     return metal_dense_f32f16(dst) && metal_dense_f32f16(src);
   }

   void copy(Tensor& dst, const Tensor& src) {
     CopyParams params{static_cast<uint32_t>(dst.numel()),
                       static_cast<uint32_t>(src.offset),
                       static_cast<uint32_t>(dst.offset)};
     const char* fn =
         dst.dtype == ScalarType::Float32 ? "mtorch_copy_f32" : "mtorch_copy_f16";
     launch_1d(fn, {buffer_for(*src.storage), buffer_for(*dst.storage)},
               &params, sizeof(params), static_cast<size_t>(dst.numel()));
   }

   }  // namespace mtorch::metal
   ```

4. Hook `Tensor::fill_inplace` in `cpp/mtorch/core/tensor_core.cpp`.
   First add the include: find the last `#include` line at the top of the file
   (`grep -n '#include' cpp/mtorch/core/tensor_core.cpp | tail -1`) and insert
   after it:

   ```cpp
   #include "mtorch/core/metal/ops.h"
   ```

   Then edit the function. BEFORE (locate with the precondition grep):

   ```cpp
   void Tensor::fill_inplace(double value) {
     ::mtorch::mark_storage_modified(*storage);
   ```

   AFTER:

   ```cpp
   void Tensor::fill_inplace(double value) {
     ::mtorch::mark_storage_modified(*storage);
     if (metal::use_fill(*this)) {
       metal::fill(*this, value);
       return;
     }
   ```

   (If the first line inside your `fill_inplace` is not the
   `mark_storage_modified` call — e.g. Phase 3/5 reshaped it — insert the
   3-line guard immediately after whatever the first statement is, keeping the
   `mark_storage_modified` call before the guard; if there is no such call at
   all, insert the guard as the first statement and add
   `::mtorch::mark_storage_modified(*storage);` inside the `if`, before
   `metal::fill`.)

5. Hook `Tensor::copy_from` in the same file. BEFORE:

   ```cpp
   void Tensor::copy_from(const Tensor& source) {
     ensure_same_device(*this, source);
   ```

   AFTER:

   ```cpp
   void Tensor::copy_from(const Tensor& source) {
     ensure_same_device(*this, source);
     if (metal::use_copy(*this, source)) {
       ::mtorch::mark_storage_modified(*storage);
       metal::copy(*this, source);
       return;
     }
   ```

   (Same adaptation note as action 4 if the `ensure_same_device` line moved.)

6. PARITY-ROWS (§0.9). Add rows covering:

   | op expression (per device) | shape | dtype | rtol | atol |
   |---|---|---|---|---|
   | `full((64, 64), 3.5)` on the device (exercises fill) | 64×64 | float32 | 0 | 0 |
   | `full((64, 64), 3.5)` | 64×64 | float16 | 0 | 0 |
   | `zeros_like(x)` of a device tensor | 33×7 | float32 | 0 | 0 |
   | in-place `x.fill_(-2.25)` | 128 | float16 | 0 | 0 |
   | in-place `y.copy_(x)` (same shape/dtype) | 64×64 | float32 | 0 | 0 |
   | in-place `y.copy_(x)` | 64×64 | float16 | 0 | 0 |

   Fill and copy are exact operations — tolerance 0 (use exact-compare rtol=0,
   atol=0 if the schema takes numbers; if the schema treats "no tolerance
   given" as assert_close defaults, that is also acceptable for these rows).

**Verification** (this is the template every family step 6-8…6-13 follows):

```bash
python3 setup.py build_ext --inplace
MTORCH_METAL_LOG=1 python3 -c "
import mtorch
x = mtorch.zeros(8, device='mps')
x.fill_(2.5)
y = mtorch.zeros(8, device='mps')
y.copy_(x)
print(y.cpu().tolist())
" 2>&1 | tee /tmp/p6-fill.txt
grep -c 'launch mtorch_fill_f32' /tmp/p6-fill.txt
grep -c 'launch mtorch_copy_f32' /tmp/p6-fill.txt
MTORCH_METAL_FORCE_FALLBACK=1 python3 -c "
import mtorch
x = mtorch.zeros(8, device='mps'); x.fill_(2.5)
print(x.cpu().tolist())
"
pytest tests/compat/test_device_parity.py -q | tail -3
pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
tail -1 docs/design/baseline/tests-phase6-cpu.txt
```

PASS iff: the smoke prints `[2.5, ...×8]` and both grep counts are ≥1 (proof
the GPU kernels actually ran — note `mtorch.zeros(..., device='mps')` itself
may or may not log a fill depending on how Phase 5 zero-initializes; only the
two counted kernels matter); the force-fallback run prints the same values
(proof the guard collapses cleanly); parity suite green with the new rows;
CPU-only suite matches the frozen baseline. Then run BENCH-GUARD (§0.9) with
ids `bench.fill_inplace_float32_64x64` and `bench.copy_float32_64x64`, and
record ratios:

```bash
python3 tools/metal_ratios.py --family fill
```

Expected-outcome guidance: for 1024×1024 fills the GPU should be at worst a
small factor from CPU (`memset`-class work is memory-bound on both); for tiny
tensors CPU wins by a lot because each launch pays command-buffer +
`waitUntilCompleted` overhead (~50–200 µs). Both outcomes are acceptable —
record one `metal_ratios fill: mps/cpu=<...>, vs torch-mps=<...>` line in the
Phase 6 Notes.

**On failure**: STANDARD FAIL-6 (`<new files>` =
`cpp/mtorch/core/metal/fill_copy.mm`; also restore
`cpp/mtorch/core/metal/kernels.metal cpp/mtorch/core/metal/ops.h
cpp/mtorch/core/tensor_core.cpp tests/compat/test_device_parity.py`). If the
smoke test hangs or the machine's GPU faults, read §3 (debugging appendix)
FIRST, then decide whether to attempt a fix (§6 allows 3 attempts).

**Commit**:

```bash
git add cpp/mtorch/core/metal/kernels.metal cpp/mtorch/core/metal/ops.h \
        cpp/mtorch/core/metal/fill_copy.mm cpp/mtorch/core/tensor_core.cpp \
        tests/compat/test_device_parity.py
git commit -m "feat(phase6-7): metal fill/copy kernels + dispatch hooks"
```

---

### Step 6-8: Op family (b) — binary elementwise add / sub / mul / div (f32 + f16)

**Goal**: `binary_tensor_tensor` on two dense, same-shape, same-dtype f32/f16
mps tensors runs a Metal kernel. **v1 requires contiguous same-shape inputs**;
broadcasts, promotions, scalar variants, in-place variants and everything else
keep taking the CPU shared-buffer fallback (that includes the
`binary_tensor_scalar` / `binary_scalar_tensor` entry points — not hooked in
v1).

**Preconditions**:

```bash
grep -n 'TensorPtr binary_tensor_tensor(const TensorPtr& left, const TensorPtr& right, const std::string& op) {' cpp/mtorch/core/elementwise_ops.cpp
```

Must print exactly one line (`elementwise_ops.cpp` is the 2b-15 section file).

**Actions**:

1. Append to `cpp/mtorch/core/metal/kernels.metal`:

   ```
   // ---------------- family (b): binary elementwise ----------------
   // f16 rule (§0.4): load half -> compute float -> store half.

   struct BinParams {
     uint n;
     uint a_off;
     uint b_off;
     uint out_off;
   };

   #define MTORCH_BINARY(NAME, EXPR)                                          \
   kernel void mtorch_##NAME##_f32(                                           \
       device const float* a [[buffer(0)]],                                   \
       device const float* b [[buffer(1)]],                                   \
       device float* out [[buffer(2)]],                                       \
       constant BinParams& p [[buffer(3)]],                                   \
       uint gid [[thread_position_in_grid]]) {                                \
     if (gid >= p.n) {                                                        \
       return;                                                                \
     }                                                                        \
     const float x = a[p.a_off + gid];                                        \
     const float y = b[p.b_off + gid];                                        \
     out[p.out_off + gid] = (EXPR);                                           \
   }                                                                          \
   kernel void mtorch_##NAME##_f16(                                           \
       device const half* a [[buffer(0)]],                                    \
       device const half* b [[buffer(1)]],                                    \
       device half* out [[buffer(2)]],                                        \
       constant BinParams& p [[buffer(3)]],                                   \
       uint gid [[thread_position_in_grid]]) {                                \
     if (gid >= p.n) {                                                        \
       return;                                                                \
     }                                                                        \
     const float x = float(a[p.a_off + gid]);                                 \
     const float y = float(b[p.b_off + gid]);                                 \
     out[p.out_off + gid] = half(EXPR);                                       \
   }

   MTORCH_BINARY(add, x + y)
   MTORCH_BINARY(sub, x - y)
   MTORCH_BINARY(mul, x * y)
   MTORCH_BINARY(div, x / y)

   #undef MTORCH_BINARY
   ```

2. In `cpp/mtorch/core/metal/ops.h`, insert after the family (a) declaration
   block (i.e. after the `void copy(Tensor& dst, const Tensor& src);` line):

   ```cpp

   // family (b): binary elementwise (step 6-8)
   bool use_binary(const Tensor& a, const Tensor& b, const std::string& op,
                   bool grad_tracked);
   TensorPtr binary(const TensorPtr& a, const TensorPtr& b, const std::string& op);
   ```

   and after the family (a) stub block (i.e. after the
   `inline void copy(Tensor&, const Tensor&) {}` line):

   ```cpp
   inline bool use_binary(const Tensor&, const Tensor&, const std::string&, bool) {
     return false;
   }
   inline TensorPtr binary(const TensorPtr&, const TensorPtr&, const std::string&) {
     return nullptr;
   }
   ```

3. Create `cpp/mtorch/core/metal/binary.mm` with exactly this content:

   ```objc
   #include "mtorch/core/metal/ops.h"
   #include "mtorch/core/metal/context.h"

   #include <cstdint>
   #include <string>

   namespace mtorch::metal {

   namespace {

   struct BinParams {
     uint32_t n;
     uint32_t a_off;
     uint32_t b_off;
     uint32_t out_off;
   };

   bool supported_binary_op(const std::string& op) {
     return op == "add" || op == "sub" || op == "mul" || op == "div";
   }

   }  // namespace

   bool use_binary(const Tensor& a, const Tensor& b, const std::string& op,
                   bool grad_tracked) {
     if (a.device.type != DeviceType::Metal ||
         b.device.type != DeviceType::Metal) {
       return false;
     }
     if (!metal_ready() || grad_tracked) {
       return false;
     }
     if (!supported_binary_op(op)) {
       return false;
     }
     if (a.dtype != b.dtype) {
       return false;  // no dtype promotion on the GPU
     }
     if (a.sizes != b.sizes) {
       return false;  // no broadcasting on the GPU (the v1 contiguity guard)
     }
     return metal_dense_f32f16(a) && metal_dense_f32f16(b);
   }

   TensorPtr binary(const TensorPtr& a, const TensorPtr& b, const std::string& op) {
     // Fresh contiguous result on the same device; requires_grad=false is
     // guaranteed by the guard (grad-tracked calls never reach here).
     TensorPtr out = empty_like(a, a->dtype, a->device, false);
     BinParams params{static_cast<uint32_t>(a->numel()),
                      static_cast<uint32_t>(a->offset),
                      static_cast<uint32_t>(b->offset),
                      static_cast<uint32_t>(out->offset)};
     const std::string fn = "mtorch_" + op +
         (a->dtype == ScalarType::Float32 ? "_f32" : "_f16");
     launch_1d(fn.c_str(),
               {buffer_for(*a->storage), buffer_for(*b->storage),
                buffer_for(*out->storage)},
               &params, sizeof(params), static_cast<size_t>(a->numel()));
     return out;
   }

   }  // namespace mtorch::metal
   ```

4. Hook `binary_tensor_tensor` in `cpp/mtorch/core/elementwise_ops.cpp`.
   Add the include after the last `#include` line
   (`grep -n '#include' cpp/mtorch/core/elementwise_ops.cpp | tail -1`):

   ```cpp
   #include "mtorch/core/metal/ops.h"
   ```

   Then, BEFORE (the function's first two lines, from the precondition grep):

   ```cpp
   TensorPtr binary_tensor_tensor(const TensorPtr& left, const TensorPtr& right, const std::string& op) {
     ensure_same_device(*left, *right);
   ```

   AFTER:

   ```cpp
   TensorPtr binary_tensor_tensor(const TensorPtr& left, const TensorPtr& right, const std::string& op) {
     ensure_same_device(*left, *right);
     if (metal::use_binary(*left, *right, op,
                           track_grad(left->requires_grad || right->requires_grad))) {
       return metal::binary(left, right, op);
     }
   ```

   **This is the exact v1 contiguity guard demanded by the plan**: any
   non-contiguous / broadcast / promoted / grad-tracked call fails
   `use_binary` and continues into the existing CPU implementation, which is
   correct on the shared buffer. (`track_grad` is the detail-layer helper the
   file already uses — visible via `using namespace detail;`; verify with
   `grep -n 'track_grad' cpp/mtorch/core/elementwise_ops.cpp | head -3`.)

5. PARITY-ROWS (§0.9). Add rows covering (inputs must avoid zeros in the div
   denominator — use an offset like `+2.0` per the existing rows' input
   conventions):

   | op | shapes | dtype | rtol | atol |
   |---|---|---|---|---|
   | `add` | (64, 64) + (64, 64) | float32 | 1e-6 | 1e-6 |
   | `sub` | (64, 64) − (64, 64) | float32 | 1e-6 | 1e-6 |
   | `mul` | (2, 4, 64, 64) ∗ (2, 4, 64, 64) | float32 | 1e-6 | 1e-6 |
   | `div` | (64, 64) / (64, 64) (denom ≥ 0.5) | float32 | 1e-6 | 1e-6 |
   | `add` | (2, 77, 768) + (2, 77, 768) | float16 | 1e-3 | 1e-3 |
   | `mul` | (2, 4, 64, 64) ∗ (2, 4, 64, 64) | float16 | 1e-3 | 1e-3 |
   | `div` | (64, 64) / (64, 64) (denom ≥ 0.5) | float16 | 1e-3 | 1e-3 |
   | `add` broadcast (64, 64) + (64,) — MUST fall back, still must be correct | mixed | float32 | 1e-6 | 1e-6 |

   The broadcast row is deliberate: it proves the guard's fallback path stays
   green, not just the kernel path.

**Verification**: as the 6-7 template, with this smoke:

```bash
python3 setup.py build_ext --inplace
MTORCH_METAL_LOG=1 python3 -c "
import mtorch
a = mtorch.ones(4, 4, device='mps') * 3.0
b = mtorch.ones(4, 4, device='mps') * 2.0
print((a + b).cpu().tolist()[0][0], (a - b).cpu().tolist()[0][0],
      (a * b).cpu().tolist()[0][0], (a / b).cpu().tolist()[0][0])
" 2>&1 | tee /tmp/p6-bin.txt
grep -c 'launch mtorch_add_f32' /tmp/p6-bin.txt
grep -c 'launch mtorch_div_f32' /tmp/p6-bin.txt
pytest tests/compat/test_device_parity.py -q | tail -3
pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
tail -1 docs/design/baseline/tests-phase6-cpu.txt
```

Expected smoke values: `5.0 1.0 6.0 1.5`, both grep counts ≥1. Then BENCH-GUARD
(§0.9) with ids `bench.binary_add_64x64`, `bench.sub_inplace_float32_64x64`,
`bench.method_mul_64x64`, `bench.div_inplace_float32_64x64`; then:

```bash
python3 tools/metal_ratios.py --family binary
```

Expected-outcome guidance: 1024×1024 f32 add/mul on the GPU should land within
roughly 0.3×–3× of mtorch CPU (Accelerate is strong; synchronous launches are
expensive); torch-MPS will typically be several times faster than our
synchronous v1 — **acceptable**. Record one Notes line as in 6-7.

**On failure**: STANDARD FAIL-6 (`<new files>` =
`cpp/mtorch/core/metal/binary.mm`; restore `kernels.metal ops.h
elementwise_ops.cpp test_device_parity.py`).

**Commit**:

```bash
git add cpp/mtorch/core/metal/kernels.metal cpp/mtorch/core/metal/ops.h \
        cpp/mtorch/core/metal/binary.mm cpp/mtorch/core/elementwise_ops.cpp \
        tests/compat/test_device_parity.py
git commit -m "feat(phase6-8): metal binary elementwise add/sub/mul/div (f32/f16)"
```

---

### Step 6-9: Op family (c) — unary elementwise + silu + gelu(tanh)

**Goal**: The string-dispatched `unary()` entry point gains a Metal path for a
fixed subset of ops, and the separate public functions `silu` and `gelu`
(approximate="tanh" only) gain theirs.

The full CPU unary-kernel op list, enumerated from the real code (re-verify
with `grep -oE 'op == "[a-z0-9_]+"' cpp/mtorch/core/detail/elementwise.cpp
cpp/mtorch/core/elementwise_ops.cpp 2>/dev/null | sort -u` — the float32 kernel
family historically handled: abs, acos, asin, atan, ceil, cos, cosh, deg2rad,
erf, erfc, exp, expm1, floor, frac, log, log10, log1p, log2, neg, rad2deg,
reciprocal, round, rsqrt, sigmoid, sign, sin, sinh, sqrt, square, tan, tanh,
trunc): **the v1 Metal subset is `neg, abs, exp, sqrt, rsqrt, log, sigmoid,
tanh`** — the ones on the Stable-Diffusion hot path. Every other unary op
falls back (correct by §0.3). `erf` is deliberately excluded: MSL has no
built-in `erf`, and a polynomial would strain the 1e-6 parity tolerance —
which is also why Metal `gelu` only accelerates the `"tanh"` approximation and
leaves exact-`erf` gelu (`approximate="none"`, the default) on the CPU.

**Preconditions**:

```bash
grep -n 'TensorPtr unary(const TensorPtr& input, const std::string& op) {' cpp/mtorch/core/elementwise_ops.cpp
grep -n 'TensorPtr silu(const TensorPtr& input) {' cpp/mtorch/core/activations.cpp
grep -n 'TensorPtr gelu(const TensorPtr& input, const std::string& approximate) {' cpp/mtorch/core/activations.cpp
```

Each must print exactly one line.

**Actions**:

1. Append to `cpp/mtorch/core/metal/kernels.metal`:

   ```
   // ---------------- family (c): unary elementwise ----------------

   struct UnaryParams {
     uint n;
     uint in_off;
     uint out_off;
   };

   #define MTORCH_UNARY(NAME, EXPR)                                           \
   kernel void mtorch_##NAME##_f32(                                           \
       device const float* x [[buffer(0)]],                                   \
       device float* out [[buffer(1)]],                                       \
       constant UnaryParams& p [[buffer(2)]],                                 \
       uint gid [[thread_position_in_grid]]) {                                \
     if (gid >= p.n) {                                                        \
       return;                                                                \
     }                                                                        \
     const float v = x[p.in_off + gid];                                       \
     out[p.out_off + gid] = (EXPR);                                           \
   }                                                                          \
   kernel void mtorch_##NAME##_f16(                                           \
       device const half* x [[buffer(0)]],                                    \
       device half* out [[buffer(1)]],                                        \
       constant UnaryParams& p [[buffer(2)]],                                 \
       uint gid [[thread_position_in_grid]]) {                                \
     if (gid >= p.n) {                                                        \
       return;                                                                \
     }                                                                        \
     const float v = float(x[p.in_off + gid]);                                \
     out[p.out_off + gid] = half(EXPR);                                       \
   }

   MTORCH_UNARY(neg, -v)
   MTORCH_UNARY(abs, fabs(v))
   MTORCH_UNARY(exp, exp(v))
   MTORCH_UNARY(sqrt, sqrt(v))
   MTORCH_UNARY(rsqrt, rsqrt(v))
   MTORCH_UNARY(log, log(v))
   MTORCH_UNARY(sigmoid, 1.0f / (1.0f + exp(-v)))
   MTORCH_UNARY(tanh, tanh(v))
   MTORCH_UNARY(silu, v / (1.0f + exp(-v)))
   // tanh-approximate GELU, exactly the CPU formula:
   // 0.5*v*(1 + tanh(sqrt(2/pi)*(v + 0.044715*v^3)))
   MTORCH_UNARY(gelu_tanh,
       0.5f * v * (1.0f + tanh(0.7978845608028654f * (v + 0.044715f * v * v * v))))

   #undef MTORCH_UNARY
   ```

2. `cpp/mtorch/core/metal/ops.h` — after the family (b) declaration block:

   ```cpp

   // family (c): unary elementwise + silu + gelu(tanh) (step 6-9)
   bool use_unary(const Tensor& t, const std::string& op, bool grad_tracked);
   TensorPtr unary(const TensorPtr& t, const std::string& op);
   bool use_silu(const Tensor& t, bool grad_tracked);
   TensorPtr silu(const TensorPtr& t);
   bool use_gelu(const Tensor& t, const std::string& approximate, bool grad_tracked);
   TensorPtr gelu(const TensorPtr& t);
   ```

   and after the family (b) stub block:

   ```cpp
   inline bool use_unary(const Tensor&, const std::string&, bool) { return false; }
   inline TensorPtr unary(const TensorPtr&, const std::string&) { return nullptr; }
   inline bool use_silu(const Tensor&, bool) { return false; }
   inline TensorPtr silu(const TensorPtr&) { return nullptr; }
   inline bool use_gelu(const Tensor&, const std::string&, bool) { return false; }
   inline TensorPtr gelu(const TensorPtr&) { return nullptr; }
   ```

3. Create `cpp/mtorch/core/metal/unary.mm`:

   ```objc
   #include "mtorch/core/metal/ops.h"
   #include "mtorch/core/metal/context.h"

   #include <cstdint>
   #include <string>

   namespace mtorch::metal {

   namespace {

   struct UnaryParams {
     uint32_t n;
     uint32_t in_off;
     uint32_t out_off;
   };

   bool supported_unary_op(const std::string& op) {
     return op == "neg" || op == "abs" || op == "exp" || op == "sqrt" ||
            op == "rsqrt" || op == "log" || op == "sigmoid" || op == "tanh";
   }

   TensorPtr run_unary_kernel(const TensorPtr& t, const std::string& kernel_base) {
     TensorPtr out = empty_like(t, t->dtype, t->device, false);
     UnaryParams params{static_cast<uint32_t>(t->numel()),
                        static_cast<uint32_t>(t->offset),
                        static_cast<uint32_t>(out->offset)};
     const std::string fn = "mtorch_" + kernel_base +
         (t->dtype == ScalarType::Float32 ? "_f32" : "_f16");
     launch_1d(fn.c_str(),
               {buffer_for(*t->storage), buffer_for(*out->storage)},
               &params, sizeof(params), static_cast<size_t>(t->numel()));
     return out;
   }

   bool dense_metal_no_grad(const Tensor& t, bool grad_tracked) {
     if (t.device.type != DeviceType::Metal) {
       return false;
     }
     if (!metal_ready() || grad_tracked) {
       return false;
     }
     return metal_dense_f32f16(t);
   }

   }  // namespace

   bool use_unary(const Tensor& t, const std::string& op, bool grad_tracked) {
     if (t.device.type != DeviceType::Metal) {
       return false;
     }
     if (!supported_unary_op(op)) {
       return false;
     }
     return dense_metal_no_grad(t, grad_tracked);
   }

   TensorPtr unary(const TensorPtr& t, const std::string& op) {
     return run_unary_kernel(t, op);
   }

   bool use_silu(const Tensor& t, bool grad_tracked) {
     return dense_metal_no_grad(t, grad_tracked);
   }

   TensorPtr silu(const TensorPtr& t) {
     return run_unary_kernel(t, "silu");
   }

   bool use_gelu(const Tensor& t, const std::string& approximate,
                 bool grad_tracked) {
     if (approximate != "tanh") {
       return false;  // exact (erf) gelu stays on the CPU (no MSL erf)
     }
     return dense_metal_no_grad(t, grad_tracked);
   }

   TensorPtr gelu(const TensorPtr& t) {
     return run_unary_kernel(t, "gelu_tanh");
   }

   }  // namespace mtorch::metal
   ```

4. Hook `unary()` in `cpp/mtorch/core/elementwise_ops.cpp` (the include from
   6-8 action 4 is already present). BEFORE (locate with the precondition
   grep; these are the function's first two lines):

   ```cpp
   TensorPtr unary(const TensorPtr& input, const std::string& op) {
     const bool requires_grad = track_grad(input->requires_grad);
   ```

   AFTER:

   ```cpp
   TensorPtr unary(const TensorPtr& input, const std::string& op) {
     const bool requires_grad = track_grad(input->requires_grad);
     if (metal::use_unary(*input, op, requires_grad)) {
       return metal::unary(input, op);
     }
   ```

5. Hook `silu` and `gelu` in `cpp/mtorch/core/activations.cpp`. Add the
   include after the last `#include` line of that file
   (`grep -n '#include' cpp/mtorch/core/activations.cpp | tail -1`):

   ```cpp
   #include "mtorch/core/metal/ops.h"
   ```

   silu BEFORE:

   ```cpp
   TensorPtr silu(const TensorPtr& input) {
     const bool requires_grad = track_grad(input->requires_grad);
   ```

   silu AFTER:

   ```cpp
   TensorPtr silu(const TensorPtr& input) {
     const bool requires_grad = track_grad(input->requires_grad);
     if (metal::use_silu(*input, requires_grad)) {
       return metal::silu(input);
     }
   ```

   gelu BEFORE (the validation stays first so error behavior is unchanged):

   ```cpp
   TensorPtr gelu(const TensorPtr& input, const std::string& approximate) {
     if (approximate != "none" && approximate != "tanh") {
       throw std::invalid_argument("gelu approximate must be 'none' or 'tanh'");
     }
   ```

   gelu AFTER:

   ```cpp
   TensorPtr gelu(const TensorPtr& input, const std::string& approximate) {
     if (approximate != "none" && approximate != "tanh") {
       throw std::invalid_argument("gelu approximate must be 'none' or 'tanh'");
     }
     if (metal::use_gelu(*input, approximate, track_grad(input->requires_grad))) {
       return metal::gelu(input);
     }
   ```

6. PARITY-ROWS (§0.9). Inputs for exp/log/sqrt/rsqrt must respect domains
   (positive inputs for log/sqrt/rsqrt; magnitudes ≤ 4 for exp — §2 explains
   why the range matters at 1e-6):

   | op | shape | dtype | rtol | atol |
   |---|---|---|---|---|
   | `neg` | (64, 64) | float32 | 0 | 0 |
   | `abs` | (64, 64) | float32 | 0 | 0 |
   | `exp` (inputs in [−4, 4]) | (64, 64) | float32 | 1e-6 | 1e-6 |
   | `sqrt` (inputs ≥ 0.25) | (64, 64) | float32 | 1e-6 | 1e-6 |
   | `rsqrt` (inputs ≥ 0.25) | (64, 64) | float32 | 1e-6 | 1e-6 |
   | `log` (inputs ≥ 0.25) | (64, 64) | float32 | 1e-6 | 1e-6 |
   | `sigmoid` | (64, 64) | float32 | 1e-6 | 1e-6 |
   | `tanh` | (64, 64) | float32 | 1e-6 | 1e-6 |
   | `exp` | (2, 77, 768) | float16 | 1e-3 | 1e-3 |
   | `sigmoid` | (2, 77, 768) | float16 | 1e-3 | 1e-3 |
   | `silu` (via nn.functional) | (2, 320, 8, 8) | float32 | 1e-6 | 1e-6 |
   | `silu` | (2, 320, 8, 8) | float16 | 1e-3 | 1e-3 |
   | `gelu(approximate="tanh")` | (2, 77, 768) | float32 | 1e-6 | 1e-6 |
   | `gelu(approximate="tanh")` | (2, 77, 768) | float16 | 1e-3 | 1e-3 |
   | `sin` — unsupported op, MUST fall back and stay correct | (64, 64) | float32 | 1e-6 | 1e-6 |

**Verification**: template of 6-7, smoke:

```bash
python3 setup.py build_ext --inplace
MTORCH_METAL_LOG=1 python3 -c "
import mtorch
x = mtorch.ones(4, device='mps') * 4.0
print(x.sqrt().cpu().tolist(), x.neg().cpu().tolist())
print(mtorch.nn.functional.silu(x).cpu().tolist()[0])
print(mtorch.nn.functional.gelu(x, approximate='tanh').cpu().tolist()[0])
" 2>&1 | tee /tmp/p6-un.txt
grep -c 'launch mtorch_sqrt_f32' /tmp/p6-un.txt
grep -c 'launch mtorch_silu_f32' /tmp/p6-un.txt
grep -c 'launch mtorch_gelu_tanh_f32' /tmp/p6-un.txt
pytest tests/compat/test_device_parity.py -q | tail -3
pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
tail -1 docs/design/baseline/tests-phase6-cpu.txt
```

Expected smoke: `[2.0, 2.0, 2.0, 2.0] [-4.0, ...]`, silu(4)≈3.928, gelu(4)≈4.0
(any value 3.99–4.0), all grep counts ≥1. BENCH-GUARD ids: `bench.rsqrt_64x64`,
`bench.nn_functional_silu_64x64`, `bench.nn_functional_gelu_64x64`,
`bench.sigmoid_half_2x77x768`. Ratios: `python3 tools/metal_ratios.py
--family unary`; guidance as in 6-8 (record one Notes line).

**On failure**: STANDARD FAIL-6 (`<new files>` =
`cpp/mtorch/core/metal/unary.mm`; restore `kernels.metal ops.h
elementwise_ops.cpp activations.cpp test_device_parity.py`).

**Commit**:

```bash
git add cpp/mtorch/core/metal/kernels.metal cpp/mtorch/core/metal/ops.h \
        cpp/mtorch/core/metal/unary.mm cpp/mtorch/core/elementwise_ops.cpp \
        cpp/mtorch/core/activations.cpp tests/compat/test_device_parity.py
git commit -m "feat(phase6-9): metal unary elementwise + silu + gelu(tanh)"
```

---

### Step 6-10: Op family (d) — matmul via MPSMatrixMultiplication

**Goal**: 2-D × 2-D `matmul` on dense f32/f16 mps tensors runs on the GPU
through MetalPerformanceShaders' GEMM. This step also builds the three shared
primitives that conv2d (6-12) and attention (6-13) reuse: `stage_to_f32`,
`store_from_f32`, and `run_gemm_f32`. Batched matmul, mv, addmm, bmm etc. are
NOT hooked in v1 (they fall back).

**Background: MetalPerformanceShaders (MPS) matrices (7 lines).**
MetalPerformanceShaders is Apple's library of pre-tuned GPU kernels — its
`MPSMatrixMultiplication` is the cuBLAS-gemm analogue and is far faster than
anything we could hand-write here. An `MPSMatrix` is a lightweight view over
an `MTLBuffer`: the `MPSMatrixDescriptor` says "this buffer contains `rows`
rows of `columns` elements, each row `rowBytes` bytes apart, of `dataType`".
Our tensors are packed row-major, so `rowBytes = columns * 4` maps a
contiguous f32 tensor 1:1 onto a descriptor — that IS the row-major mapping,
no transposition or copy needed. The kernel object is created per call with
`initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:
interiorColumns:alpha:beta:` and encoded with
`encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:`.

**Background: why we stage into fresh f32 buffers (6 lines).** Two reasons.
(1) fp16 accumulate-in-float (§2): letting MPS multiply f16 matrices directly
would accumulate in unspecified precision; casting operands to f32, running an
f32 GEMM, and casting the result back makes the numerics match the CPU
kernels' float accumulation within ordinary rounding. (2) Offset avoidance:
`MPSMatrix` buffer offsets come with alignment rules; by always copying
operands into fresh scratch buffers that start at offset 0 (using our own
`copy`/`cast` kernels, which take *element* offsets in their params), no MPS
alignment rule can ever bite. The extra copies are a known, accepted v1 cost.

**Preconditions**:

```bash
grep -n 'TensorPtr matmul(const TensorPtr& left, const TensorPtr& right) {' cpp/mtorch/core/linalg.cpp
```

Must print exactly one line (`linalg.cpp` is the 2b-9 section file).

**Actions**:

1. Append to `cpp/mtorch/core/metal/kernels.metal`:

   ```
   // ---------------- family (d): dtype casts for GEMM staging ----------------
   // Reuses CopyParams {n, src_off, dst_off} from family (a).

   kernel void mtorch_cast_f16_f32(
       device const half* src [[buffer(0)]],
       device float* dst [[buffer(1)]],
       constant CopyParams& p [[buffer(2)]],
       uint gid [[thread_position_in_grid]]) {
     if (gid >= p.n) {
       return;
     }
     dst[p.dst_off + gid] = float(src[p.src_off + gid]);
   }

   kernel void mtorch_cast_f32_f16(
       device const float* src [[buffer(0)]],
       device half* dst [[buffer(1)]],
       constant CopyParams& p [[buffer(2)]],
       uint gid [[thread_position_in_grid]]) {
     if (gid >= p.n) {
       return;
     }
     dst[p.dst_off + gid] = half(src[p.src_off + gid]);
   }
   ```

2. `cpp/mtorch/core/metal/ops.h` — after the family (c) declaration block:

   ```cpp

   // family (d): matmul (step 6-10)
   bool use_matmul(const Tensor& a, const Tensor& b, bool grad_tracked);
   TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);
   ```

   and after the family (c) stub block:

   ```cpp
   inline bool use_matmul(const Tensor&, const Tensor&, bool) { return false; }
   inline TensorPtr matmul(const TensorPtr&, const TensorPtr&) { return nullptr; }
   ```

3. Create `cpp/mtorch/core/metal/matmul.mm`:

   ```objc
   #include "mtorch/core/metal/ops.h"
   #include "mtorch/core/metal/context.h"

   #include <cstdint>
   #include <stdexcept>

   namespace mtorch::metal {

   namespace {

   struct CopyParams {
     uint32_t n;
     uint32_t src_off;
     uint32_t dst_off;
   };

   }  // namespace

   id<MTLBuffer> stage_to_f32(const Tensor& t, size_t elem_off, size_t n) {
     id<MTLBuffer> scratch = Context::instance().temp_buffer(n * sizeof(float));
     CopyParams params{static_cast<uint32_t>(n),
                       static_cast<uint32_t>(elem_off),
                       0};
     const char* fn = t.dtype == ScalarType::Float32 ? "mtorch_copy_f32"
                                                     : "mtorch_cast_f16_f32";
     launch_1d(fn, {buffer_for(*t.storage), scratch}, &params, sizeof(params), n);
     return scratch;
   }

   void store_from_f32(id<MTLBuffer> src, const Tensor& out, size_t elem_off,
                       size_t n) {
     CopyParams params{static_cast<uint32_t>(n),
                       0,
                       static_cast<uint32_t>(elem_off)};
     const char* fn = out.dtype == ScalarType::Float32 ? "mtorch_copy_f32"
                                                       : "mtorch_cast_f32_f16";
     launch_1d(fn, {src, buffer_for(*out.storage)}, &params, sizeof(params), n);
   }

   void run_gemm_f32(id<MTLBuffer> a, size_t a_rows, size_t a_cols,
                     id<MTLBuffer> b, size_t b_rows, size_t b_cols,
                     bool transpose_right,
                     id<MTLBuffer> result, double alpha, double beta) {
     @autoreleasepool {
       const size_t m = a_rows;
       const size_t k = a_cols;
       const size_t n = transpose_right ? b_rows : b_cols;
       const size_t b_interior = transpose_right ? b_cols : b_rows;
       if (b_interior != k) {
         throw std::runtime_error("mtorch metal: run_gemm_f32 shape mismatch");
       }
       Context& ctx = Context::instance();
       // Row-major mapping (see Background): a packed (r x c) f32 tensor is
       // exactly rows=r, columns=c, rowBytes=c*sizeof(float).
       MPSMatrixDescriptor* da = [MPSMatrixDescriptor
           matrixDescriptorWithRows:m
                            columns:k
                           rowBytes:k * sizeof(float)
                           dataType:MPSDataTypeFloat32];
       MPSMatrixDescriptor* db = [MPSMatrixDescriptor
           matrixDescriptorWithRows:b_rows
                            columns:b_cols
                           rowBytes:b_cols * sizeof(float)
                           dataType:MPSDataTypeFloat32];
       MPSMatrixDescriptor* dc = [MPSMatrixDescriptor
           matrixDescriptorWithRows:m
                            columns:n
                           rowBytes:n * sizeof(float)
                           dataType:MPSDataTypeFloat32];
       MPSMatrix* ma = [[MPSMatrix alloc] initWithBuffer:a descriptor:da];
       MPSMatrix* mb = [[MPSMatrix alloc] initWithBuffer:b descriptor:db];
       MPSMatrix* mc = [[MPSMatrix alloc] initWithBuffer:result descriptor:dc];
       // The GEMM kernel: result = alpha * A * op(B) + beta * result.
       MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
           initWithDevice:ctx.device()
            transposeLeft:NO
           transposeRight:(transpose_right ? YES : NO)
               resultRows:m
            resultColumns:n
          interiorColumns:k
                    alpha:alpha
                     beta:beta];
       id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
       [gemm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];
       commit_and_wait(cb, "MPSMatrixMultiplication");
     }
   }

   bool use_matmul(const Tensor& a, const Tensor& b, bool grad_tracked) {
     if (a.device.type != DeviceType::Metal ||
         b.device.type != DeviceType::Metal) {
       return false;
     }
     if (!metal_ready() || grad_tracked) {
       return false;
     }
     if (a.dim() != 2 || b.dim() != 2) {
       return false;  // v1: plain 2-D GEMM only; batched/mv fall back
     }
     if (a.dtype != b.dtype) {
       return false;
     }
     if (a.sizes[1] != b.sizes[0]) {
       return false;  // let the CPU path raise its usual shape error
     }
     return metal_dense_f32f16(a) && metal_dense_f32f16(b);
   }

   TensorPtr matmul(const TensorPtr& a, const TensorPtr& b) {
     const int64_t m = a->sizes[0];
     const int64_t k = a->sizes[1];
     const int64_t n = b->sizes[1];
     TensorPtr out = empty_strided({m, n}, contiguous_strides({m, n}),
                                   a->dtype, false, a->device);
     id<MTLBuffer> sa = stage_to_f32(*a, static_cast<size_t>(a->offset),
                                     static_cast<size_t>(m * k));
     id<MTLBuffer> sb = stage_to_f32(*b, static_cast<size_t>(b->offset),
                                     static_cast<size_t>(k * n));
     id<MTLBuffer> sc = Context::instance().temp_buffer(
         static_cast<size_t>(m * n) * sizeof(float));
     run_gemm_f32(sa, m, k, sb, k, n, /*transpose_right=*/false, sc,
                  /*alpha=*/1.0, /*beta=*/0.0);
     store_from_f32(sc, *out, static_cast<size_t>(out->offset),
                    static_cast<size_t>(m * n));
     return out;
   }

   }  // namespace mtorch::metal
   ```

   Note the guard also bounds `m*k`, `k*n`, `m*n` implicitly through
   `metal_dense_f32f16`'s uint32 check on each operand and the fresh result.

4. Hook `matmul` in `cpp/mtorch/core/linalg.cpp`. Add the include after the
   last `#include` line (`grep -n '#include' cpp/mtorch/core/linalg.cpp |
   tail -1`):

   ```cpp
   #include "mtorch/core/metal/ops.h"
   ```

   BEFORE (first two lines of the function, from the precondition grep):

   ```cpp
   TensorPtr matmul(const TensorPtr& left, const TensorPtr& right) {
     ensure_same_device(*left, *right);
   ```

   AFTER:

   ```cpp
   TensorPtr matmul(const TensorPtr& left, const TensorPtr& right) {
     ensure_same_device(*left, *right);
     if (metal::use_matmul(*left, *right,
                           track_grad(left->requires_grad || right->requires_grad))) {
       return metal::matmul(left, right);
     }
   ```

5. PARITY-ROWS (§0.9):

   | op | shapes | dtype | rtol | atol |
   |---|---|---|---|---|
   | `matmul` | (48, 48) @ (48, 48) | float32 | 1e-5 | 1e-5 |
   | `matmul` | (64, 256) @ (256, 32) — non-square, K=256 | float32 | 1e-5 | 1e-5 |
   | `matmul` | (48, 48) @ (48, 48) | float16 | 1e-3 | 1e-3 |
   | `matmul` | (77, 64) @ (64, 77) — SD attention shape | float16 | 1e-3 | 1e-3 |
   | `matmul` batched (2, 8, 16, 16) @ (2, 8, 16, 16) — MUST fall back | — | float32 | 1e-5 | 1e-5 |

   The f32 tolerance is 1e-5 (not 1e-6): GEMM sums K products in a different
   order than the CPU kernel — §2 explains the expectation.

**Verification**: template of 6-7, smoke:

```bash
python3 setup.py build_ext --inplace
MTORCH_METAL_LOG=1 python3 -c "
import mtorch
a = mtorch.ones(8, 8, device='mps')
print((a @ a).cpu().tolist()[0][0])
" 2>&1 | tee /tmp/p6-mm.txt
grep -c 'launch mtorch_copy_f32' /tmp/p6-mm.txt
pytest tests/compat/test_device_parity.py -q | tail -3
pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
tail -1 docs/design/baseline/tests-phase6-cpu.txt
```

Expected smoke: `8.0` and a copy-kernel count ≥3 (two stagings + one store —
the GEMM itself logs nothing because MPS encodes it, which is expected).
BENCH-GUARD ids: `bench.matmul_48x48`, `bench.matmul_operator_48x48`,
`bench.addmm_64x64`. Ratios: `python3 tools/metal_ratios.py --family matmul`.
Expected-outcome guidance: at 512×512 f32 the GPU GEMM should beat mtorch CPU
comfortably despite staging; torch-MPS (MPSGraph, no staging, async) will
still be faster — acceptable. Record the Notes line.

**On failure**: STANDARD FAIL-6 (`<new files>` =
`cpp/mtorch/core/metal/matmul.mm`; restore `kernels.metal ops.h linalg.cpp
test_device_parity.py`).

**Commit**:

```bash
git add cpp/mtorch/core/metal/kernels.metal cpp/mtorch/core/metal/ops.h \
        cpp/mtorch/core/metal/matmul.mm cpp/mtorch/core/linalg.cpp \
        tests/compat/test_device_parity.py
git commit -m "feat(phase6-10): metal matmul via MPSMatrixMultiplication (f32 staging)"
```

---

### Step 6-11: Op family (e) — softmax + layer_norm + group_norm

**Goal**: Row-structured reductions on the GPU. `softmax` (last dim),
`layer_norm` (1-D normalized shape = last dim) and `group_norm` (4-D NCHW)
gain Metal kernels using threadgroup reductions. `log_softmax`, `rms_norm`,
`batch_norm` and every unsupported configuration fall back.

**Background: threadgroup memory and barriers (8 lines).** A reduction (max,
sum) needs threads to combine values, and threads can only communicate through
memory. `threadgroup float sh[256];` declares 256 floats of fast on-chip
memory shared by the (exactly 256) threads of one threadgroup — the CUDA
`__shared__` analogue. The pattern used below: each thread strides over the
row accumulating a private partial value; writes it to `sh[lid]`; then a
`threadgroup_barrier(mem_flags::mem_threadgroup)` makes all writes visible;
then a log2(256)=8-step "tree" folds `sh[lid] += sh[lid + s]` for
s=128,64,…,1, with a barrier after each step because step i reads what step
i−1 wrote. After the last fold, `sh[0]` holds the row total. **Every thread
must reach every barrier** — that is why the strided loops have no early
`return` and why `launch_rows` hard-pins 256 threads per threadgroup.

**Preconditions**:

```bash
grep -n 'TensorPtr softmax(const TensorPtr& input, int64_t dim, ScalarType dtype) {' cpp/mtorch/core/normalization.cpp
grep -n 'TensorPtr layer_norm(' cpp/mtorch/core/normalization.cpp
grep -n 'TensorPtr group_norm(' cpp/mtorch/core/normalization.cpp
```

Each must print at least one line (the first exactly one; for the other two,
use the definition line, not the header declaration — `normalization.cpp` is
the 2b-5 section file).

**Actions**:

1. Append to `cpp/mtorch/core/metal/kernels.metal`:

   ```
   // ---------------- family (e): softmax / layer_norm / group_norm ----------------
   // One 256-thread threadgroup per row (launch_rows). sh[] reductions per §0.
   // f16 variants read half, reduce in float, store half.

   struct RowParams {
     uint rows;
     uint cols;
     uint in_off;
     uint out_off;
   };

   #define MTORCH_SOFTMAX(SUFFIX, T)                                          \
   kernel void mtorch_softmax_rows_##SUFFIX(                                  \
       device const T* x [[buffer(0)]],                                       \
       device T* y [[buffer(1)]],                                             \
       constant RowParams& p [[buffer(2)]],                                   \
       uint row [[threadgroup_position_in_grid]],                             \
       uint lid [[thread_position_in_threadgroup]]) {                         \
     threadgroup float sh[256];                                               \
     const uint base_in = p.in_off + row * p.cols;                            \
     const uint base_out = p.out_off + row * p.cols;                          \
     /* pass 1: row max (for numerical stability) */                          \
     float local_max = -INFINITY;                                             \
     for (uint c = lid; c < p.cols; c += 256u) {                              \
       local_max = max(local_max, float(x[base_in + c]));                     \
     }                                                                        \
     sh[lid] = local_max;                                                     \
     threadgroup_barrier(mem_flags::mem_threadgroup);                         \
     for (uint s = 128u; s > 0u; s >>= 1u) {                                  \
       if (lid < s) {                                                         \
         sh[lid] = max(sh[lid], sh[lid + s]);                                 \
       }                                                                      \
       threadgroup_barrier(mem_flags::mem_threadgroup);                       \
     }                                                                        \
     const float row_max = sh[0];                                             \
     threadgroup_barrier(mem_flags::mem_threadgroup);                         \
     /* pass 2: sum of exp */                                                 \
     float local_sum = 0.0f;                                                  \
     for (uint c = lid; c < p.cols; c += 256u) {                              \
       local_sum += exp(float(x[base_in + c]) - row_max);                     \
     }                                                                        \
     sh[lid] = local_sum;                                                     \
     threadgroup_barrier(mem_flags::mem_threadgroup);                         \
     for (uint s = 128u; s > 0u; s >>= 1u) {                                  \
       if (lid < s) {                                                         \
         sh[lid] += sh[lid + s];                                              \
       }                                                                      \
       threadgroup_barrier(mem_flags::mem_threadgroup);                       \
     }                                                                        \
     const float inv_sum = 1.0f / sh[0];                                      \
     /* pass 3: write normalized values (safe in place: each element is */    \
     /* read and written only by this same thread in this same loop) */       \
     for (uint c = lid; c < p.cols; c += 256u) {                              \
       y[base_out + c] = T(exp(float(x[base_in + c]) - row_max) * inv_sum);   \
     }                                                                        \
   }

   MTORCH_SOFTMAX(f32, float)
   MTORCH_SOFTMAX(f16, half)
   #undef MTORCH_SOFTMAX

   struct LayerNormParams {
     uint rows;
     uint cols;
     uint in_off;
     uint out_off;
     uint w_off;
     uint b_off;
     uint has_weight;
     uint has_bias;
     float eps;
   };

   #define MTORCH_LAYER_NORM(SUFFIX, T)                                       \
   kernel void mtorch_layer_norm_rows_##SUFFIX(                               \
       device const T* x [[buffer(0)]],                                       \
       device T* y [[buffer(1)]],                                             \
       device const T* w [[buffer(2)]],                                       \
       device const T* b [[buffer(3)]],                                       \
       constant LayerNormParams& p [[buffer(4)]],                             \
       uint row [[threadgroup_position_in_grid]],                             \
       uint lid [[thread_position_in_threadgroup]]) {                         \
     threadgroup float sh[256];                                               \
     const uint base_in = p.in_off + row * p.cols;                            \
     const uint base_out = p.out_off + row * p.cols;                          \
     /* mean */                                                               \
     float local_sum = 0.0f;                                                  \
     for (uint c = lid; c < p.cols; c += 256u) {                              \
       local_sum += float(x[base_in + c]);                                    \
     }                                                                        \
     sh[lid] = local_sum;                                                     \
     threadgroup_barrier(mem_flags::mem_threadgroup);                         \
     for (uint s = 128u; s > 0u; s >>= 1u) {                                  \
       if (lid < s) {                                                         \
         sh[lid] += sh[lid + s];                                              \
       }                                                                      \
       threadgroup_barrier(mem_flags::mem_threadgroup);                       \
     }                                                                        \
     const float mean = sh[0] / float(p.cols);                                \
     threadgroup_barrier(mem_flags::mem_threadgroup);                         \
     /* biased variance (matches torch layer_norm) */                         \
     float local_sq = 0.0f;                                                   \
     for (uint c = lid; c < p.cols; c += 256u) {                              \
       const float d = float(x[base_in + c]) - mean;                          \
       local_sq += d * d;                                                     \
     }                                                                        \
     sh[lid] = local_sq;                                                      \
     threadgroup_barrier(mem_flags::mem_threadgroup);                         \
     for (uint s = 128u; s > 0u; s >>= 1u) {                                  \
       if (lid < s) {                                                         \
         sh[lid] += sh[lid + s];                                              \
       }                                                                      \
       threadgroup_barrier(mem_flags::mem_threadgroup);                       \
     }                                                                        \
     const float rstd = rsqrt(sh[0] / float(p.cols) + p.eps);                 \
     for (uint c = lid; c < p.cols; c += 256u) {                              \
       float v = (float(x[base_in + c]) - mean) * rstd;                       \
       if (p.has_weight != 0u) {                                              \
         v *= float(w[p.w_off + c]);                                          \
       }                                                                      \
       if (p.has_bias != 0u) {                                                \
         v += float(b[p.b_off + c]);                                          \
       }                                                                      \
       y[base_out + c] = T(v);                                                \
     }                                                                        \
   }

   MTORCH_LAYER_NORM(f32, float)
   MTORCH_LAYER_NORM(f16, half)
   #undef MTORCH_LAYER_NORM

   struct GroupNormStatsParams {
     uint group_elems;   // channels_per_group * H * W
     uint in_off;
     float eps;
   };

   // Kernel 1: one threadgroup per (batch, group); writes stats[2*g]=mean,
   // stats[2*g+1]=rstd. Valid because in contiguous NCHW the (n, group) block
   // is one contiguous run of group_elems elements.
   #define MTORCH_GROUP_NORM_STATS(SUFFIX, T)                                 \
   kernel void mtorch_group_norm_stats_##SUFFIX(                              \
       device const T* x [[buffer(0)]],                                       \
       device float* stats [[buffer(1)]],                                     \
       constant GroupNormStatsParams& p [[buffer(2)]],                        \
       uint group [[threadgroup_position_in_grid]],                           \
       uint lid [[thread_position_in_threadgroup]]) {                         \
     threadgroup float sh[256];                                               \
     const uint base = p.in_off + group * p.group_elems;                      \
     float local_sum = 0.0f;                                                  \
     float local_sq = 0.0f;                                                   \
     for (uint i = lid; i < p.group_elems; i += 256u) {                       \
       const float v = float(x[base + i]);                                    \
       local_sum += v;                                                        \
       local_sq += v * v;                                                     \
     }                                                                        \
     sh[lid] = local_sum;                                                     \
     threadgroup_barrier(mem_flags::mem_threadgroup);                         \
     for (uint s = 128u; s > 0u; s >>= 1u) {                                  \
       if (lid < s) {                                                         \
         sh[lid] += sh[lid + s];                                              \
       }                                                                      \
       threadgroup_barrier(mem_flags::mem_threadgroup);                       \
     }                                                                        \
     const float mean = sh[0] / float(p.group_elems);                         \
     threadgroup_barrier(mem_flags::mem_threadgroup);                         \
     sh[lid] = local_sq;                                                      \
     threadgroup_barrier(mem_flags::mem_threadgroup);                         \
     for (uint s = 128u; s > 0u; s >>= 1u) {                                  \
       if (lid < s) {                                                         \
         sh[lid] += sh[lid + s];                                              \
       }                                                                      \
       threadgroup_barrier(mem_flags::mem_threadgroup);                       \
     }                                                                        \
     if (lid == 0u) {                                                         \
       const float var = max(sh[0] / float(p.group_elems) - mean * mean, 0.0f); \
       stats[2u * group] = mean;                                              \
       stats[2u * group + 1u] = rsqrt(var + p.eps);                           \
     }                                                                        \
   }

   MTORCH_GROUP_NORM_STATS(f32, float)
   MTORCH_GROUP_NORM_STATS(f16, half)
   #undef MTORCH_GROUP_NORM_STATS

   struct GroupNormApplyParams {
     uint n;              // total elements N*C*H*W
     uint channels;       // C
     uint hw;             // H*W
     uint channels_per_group;
     uint groups;         // G
     uint in_off;
     uint out_off;
     uint w_off;
     uint b_off;
     uint has_weight;
     uint has_bias;
   };

   // Kernel 2: plain elementwise normalize using the per-group stats.
   #define MTORCH_GROUP_NORM_APPLY(SUFFIX, T)                                 \
   kernel void mtorch_group_norm_apply_##SUFFIX(                              \
       device const T* x [[buffer(0)]],                                       \
       device T* y [[buffer(1)]],                                             \
       device const float* stats [[buffer(2)]],                               \
       device const T* w [[buffer(3)]],                                       \
       device const T* b [[buffer(4)]],                                       \
       constant GroupNormApplyParams& p [[buffer(5)]],                        \
       uint gid [[thread_position_in_grid]]) {                                \
     if (gid >= p.n) {                                                        \
       return;                                                                \
     }                                                                        \
     const uint c = (gid / p.hw) % p.channels;                                \
     const uint batch = gid / (p.hw * p.channels);                            \
     const uint group = batch * p.groups + c / p.channels_per_group;          \
     float v = (float(x[p.in_off + gid]) - stats[2u * group]) *               \
               stats[2u * group + 1u];                                        \
     if (p.has_weight != 0u) {                                                \
       v *= float(w[p.w_off + c]);                                            \
     }                                                                        \
     if (p.has_bias != 0u) {                                                  \
       v += float(b[p.b_off + c]);                                            \
     }                                                                        \
     y[p.out_off + gid] = T(v);                                               \
   }

   MTORCH_GROUP_NORM_APPLY(f32, float)
   MTORCH_GROUP_NORM_APPLY(f16, half)
   #undef MTORCH_GROUP_NORM_APPLY
   ```

2. `cpp/mtorch/core/metal/ops.h` — after the family (d) declaration block:

   ```cpp

   // family (e): softmax / layer_norm / group_norm (step 6-11)
   bool use_softmax(const Tensor& t, int64_t dim, ScalarType dtype,
                    bool grad_tracked);
   TensorPtr softmax(const TensorPtr& t, int64_t dim);
   bool use_layer_norm(const Tensor& t, const std::vector<int64_t>& normalized_shape,
                       const TensorPtr& weight, const TensorPtr& bias,
                       bool grad_tracked);
   TensorPtr layer_norm(const TensorPtr& t, const TensorPtr& weight,
                        const TensorPtr& bias, double eps);
   bool use_group_norm(const Tensor& t, int64_t num_groups,
                       const TensorPtr& weight, const TensorPtr& bias,
                       bool grad_tracked);
   TensorPtr group_norm(const TensorPtr& t, int64_t num_groups,
                        const TensorPtr& weight, const TensorPtr& bias,
                        double eps);
   ```

   and after the family (d) stub block:

   ```cpp
   inline bool use_softmax(const Tensor&, int64_t, ScalarType, bool) { return false; }
   inline TensorPtr softmax(const TensorPtr&, int64_t) { return nullptr; }
   inline bool use_layer_norm(const Tensor&, const std::vector<int64_t>&,
                              const TensorPtr&, const TensorPtr&, bool) { return false; }
   inline TensorPtr layer_norm(const TensorPtr&, const TensorPtr&,
                               const TensorPtr&, double) { return nullptr; }
   inline bool use_group_norm(const Tensor&, int64_t, const TensorPtr&,
                              const TensorPtr&, bool) { return false; }
   inline TensorPtr group_norm(const TensorPtr&, int64_t, const TensorPtr&,
                               const TensorPtr&, double) { return nullptr; }
   ```

   (`ops.h` already includes `tensor.h`, which provides `std::vector` via its
   own includes; if the build complains, add `#include <vector>` to `ops.h`'s
   include block.)

3. Create `cpp/mtorch/core/metal/normalization.mm`:

   ```objc
   #include "mtorch/core/metal/ops.h"
   #include "mtorch/core/metal/context.h"

   #include <cstdint>
   #include <vector>

   namespace mtorch::metal {

   namespace {

   struct RowParams {
     uint32_t rows;
     uint32_t cols;
     uint32_t in_off;
     uint32_t out_off;
   };

   struct LayerNormParams {
     uint32_t rows;
     uint32_t cols;
     uint32_t in_off;
     uint32_t out_off;
     uint32_t w_off;
     uint32_t b_off;
     uint32_t has_weight;
     uint32_t has_bias;
     float eps;
   };

   struct GroupNormStatsParams {
     uint32_t group_elems;
     uint32_t in_off;
     float eps;
   };

   struct GroupNormApplyParams {
     uint32_t n;
     uint32_t channels;
     uint32_t hw;
     uint32_t channels_per_group;
     uint32_t groups;
     uint32_t in_off;
     uint32_t out_off;
     uint32_t w_off;
     uint32_t b_off;
     uint32_t has_weight;
     uint32_t has_bias;
   };

   const char* suffix_for(ScalarType dtype) {
     return dtype == ScalarType::Float32 ? "_f32" : "_f16";
   }

   // An optional affine parameter is usable iff absent, or a dense metal
   // tensor of the same dtype as the input with exactly `expected` elements.
   bool affine_param_ok(const TensorPtr& param, const Tensor& input,
                        int64_t expected) {
     if (param == nullptr) {
       return true;
     }
     if (param->dtype != input.dtype || param->numel() != expected ||
         param->requires_grad) {
       return false;
     }
     return metal_dense_f32f16(*param);
   }

   id<MTLBuffer> affine_buffer_or_dummy(const TensorPtr& param) {
     return param != nullptr ? buffer_for(*param->storage)
                             : Context::instance().dummy_buffer();
   }

   }  // namespace

   bool use_softmax(const Tensor& t, int64_t dim, ScalarType dtype,
                    bool grad_tracked) {
     if (t.device.type != DeviceType::Metal) {
       return false;
     }
     if (!metal_ready() || grad_tracked) {
       return false;
     }
     if (dtype != t.dtype) {
       return false;  // dtype-converting softmax stays on CPU
     }
     if (t.dim() < 1) {
       return false;
     }
     const int64_t normalized_dim = dim < 0 ? dim + t.dim() : dim;
     if (normalized_dim != t.dim() - 1) {
       return false;  // v1: last-dim softmax only
     }
     return metal_dense_f32f16(t);
   }

   TensorPtr softmax(const TensorPtr& t, int64_t /*dim*/) {
     TensorPtr out = empty_like(t, t->dtype, t->device, false);
     const int64_t cols = t->sizes.back();
     const int64_t rows = t->numel() / cols;
     RowParams params{static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
                      static_cast<uint32_t>(t->offset),
                      static_cast<uint32_t>(out->offset)};
     const std::string fn = std::string("mtorch_softmax_rows") + suffix_for(t->dtype);
     launch_rows(fn.c_str(),
                 {buffer_for(*t->storage), buffer_for(*out->storage)},
                 &params, sizeof(params), static_cast<size_t>(rows));
     return out;
   }

   bool use_layer_norm(const Tensor& t,
                       const std::vector<int64_t>& normalized_shape,
                       const TensorPtr& weight, const TensorPtr& bias,
                       bool grad_tracked) {
     if (t.device.type != DeviceType::Metal) {
       return false;
     }
     if (!metal_ready() || grad_tracked) {
       return false;
     }
     if (normalized_shape.size() != 1 || t.dim() < 1 ||
         normalized_shape[0] != t.sizes.back()) {
       return false;  // v1: normalize over the last dim only
     }
     if (!affine_param_ok(weight, t, normalized_shape[0]) ||
         !affine_param_ok(bias, t, normalized_shape[0])) {
       return false;
     }
     return metal_dense_f32f16(t);
   }

   TensorPtr layer_norm(const TensorPtr& t, const TensorPtr& weight,
                        const TensorPtr& bias, double eps) {
     TensorPtr out = empty_like(t, t->dtype, t->device, false);
     const int64_t cols = t->sizes.back();
     const int64_t rows = t->numel() / cols;
     LayerNormParams params{
         static_cast<uint32_t>(rows),
         static_cast<uint32_t>(cols),
         static_cast<uint32_t>(t->offset),
         static_cast<uint32_t>(out->offset),
         weight != nullptr ? static_cast<uint32_t>(weight->offset) : 0u,
         bias != nullptr ? static_cast<uint32_t>(bias->offset) : 0u,
         weight != nullptr ? 1u : 0u,
         bias != nullptr ? 1u : 0u,
         static_cast<float>(eps)};
     const std::string fn =
         std::string("mtorch_layer_norm_rows") + suffix_for(t->dtype);
     launch_rows(fn.c_str(),
                 {buffer_for(*t->storage), buffer_for(*out->storage),
                  affine_buffer_or_dummy(weight), affine_buffer_or_dummy(bias)},
                 &params, sizeof(params), static_cast<size_t>(rows));
     return out;
   }

   bool use_group_norm(const Tensor& t, int64_t num_groups,
                       const TensorPtr& weight, const TensorPtr& bias,
                       bool grad_tracked) {
     if (t.device.type != DeviceType::Metal) {
       return false;
     }
     if (!metal_ready() || grad_tracked) {
       return false;
     }
     if (t.dim() != 4) {
       return false;  // v1: NCHW only
     }
     if (num_groups <= 0 || t.sizes[1] % num_groups != 0) {
       return false;  // invalid config: let the CPU path raise its error
     }
     if (!affine_param_ok(weight, t, t.sizes[1]) ||
         !affine_param_ok(bias, t, t.sizes[1])) {
       return false;
     }
     return metal_dense_f32f16(t);
   }

   TensorPtr group_norm(const TensorPtr& t, int64_t num_groups,
                        const TensorPtr& weight, const TensorPtr& bias,
                        double eps) {
     TensorPtr out = empty_like(t, t->dtype, t->device, false);
     const int64_t batch = t->sizes[0];
     const int64_t channels = t->sizes[1];
     const int64_t hw = t->sizes[2] * t->sizes[3];
     const int64_t channels_per_group = channels / num_groups;
     const int64_t total_groups = batch * num_groups;
     // 2 floats (mean, rstd) per (batch, group) in a scratch buffer.
     id<MTLBuffer> stats = Context::instance().temp_buffer(
         static_cast<size_t>(total_groups) * 2 * sizeof(float));
     GroupNormStatsParams stats_params{
         static_cast<uint32_t>(channels_per_group * hw),
         static_cast<uint32_t>(t->offset),
         static_cast<float>(eps)};
     const std::string stats_fn =
         std::string("mtorch_group_norm_stats") + suffix_for(t->dtype);
     launch_rows(stats_fn.c_str(), {buffer_for(*t->storage), stats},
                 &stats_params, sizeof(stats_params),
                 static_cast<size_t>(total_groups));
     GroupNormApplyParams apply_params{
         static_cast<uint32_t>(t->numel()),
         static_cast<uint32_t>(channels),
         static_cast<uint32_t>(hw),
         static_cast<uint32_t>(channels_per_group),
         static_cast<uint32_t>(num_groups),
         static_cast<uint32_t>(t->offset),
         static_cast<uint32_t>(out->offset),
         weight != nullptr ? static_cast<uint32_t>(weight->offset) : 0u,
         bias != nullptr ? static_cast<uint32_t>(bias->offset) : 0u,
         weight != nullptr ? 1u : 0u,
         bias != nullptr ? 1u : 0u};
     const std::string apply_fn =
         std::string("mtorch_group_norm_apply") + suffix_for(t->dtype);
     launch_1d(apply_fn.c_str(),
               {buffer_for(*t->storage), buffer_for(*out->storage), stats,
                affine_buffer_or_dummy(weight), affine_buffer_or_dummy(bias)},
               &apply_params, sizeof(apply_params),
               static_cast<size_t>(t->numel()));
     return out;
   }

   }  // namespace mtorch::metal
   ```

4. Hook the three ops in `cpp/mtorch/core/normalization.cpp`. Add the include
   after the last `#include` line
   (`grep -n '#include' cpp/mtorch/core/normalization.cpp | tail -1`):

   ```cpp
   #include "mtorch/core/metal/ops.h"
   ```

   softmax BEFORE (whole function, from the precondition grep):

   ```cpp
   TensorPtr softmax(const TensorPtr& input, int64_t dim, ScalarType dtype) {
     return softmax_impl(input, dim, dtype, false);
   }
   ```

   softmax AFTER:

   ```cpp
   TensorPtr softmax(const TensorPtr& input, int64_t dim, ScalarType dtype) {
     if (metal::use_softmax(*input, dim, dtype, track_grad(input->requires_grad))) {
       return metal::softmax(input, dim);
     }
     return softmax_impl(input, dim, dtype, false);
   }
   ```

   (`log_softmax` is intentionally NOT hooked — it keeps calling
   `softmax_impl(..., true)` on the CPU fallback.)

   layer_norm: locate the definition
   (`grep -n -A 6 'TensorPtr layer_norm(' cpp/mtorch/core/normalization.cpp`),
   whose signature block ends with `    double eps) {`. Insert immediately
   after that `{`, BEFORE the existing first statement:

   ```cpp
     if (metal::use_layer_norm(*input, normalized_shape, weight, bias,
                               track_grad(input->requires_grad))) {
       return metal::layer_norm(input, weight, bias, eps);
     }
   ```

   group_norm: same pattern
   (`grep -n -A 6 'TensorPtr group_norm(' cpp/mtorch/core/normalization.cpp`),
   insert immediately after its `    double eps) {`:

   ```cpp
     if (metal::use_group_norm(*input, num_groups, weight, bias,
                               track_grad(input->requires_grad))) {
       return metal::group_norm(input, num_groups, weight, bias, eps);
     }
   ```

   (If the parameter names in your tree differ from `input` /
   `normalized_shape` / `weight` / `bias` / `num_groups` / `eps`, use the
   names from the actual signature — nothing else may change.)

5. PARITY-ROWS (§0.9). Note the norm-family f32 tolerance is 1e-4: single-pass
   float accumulation over up to ~10k elements against the CPU's accumulation
   order (§2):

   | op | shape | dtype | rtol | atol |
   |---|---|---|---|---|
   | `softmax(dim=-1)` | (64, 64) | float32 | 1e-6 | 1e-6 |
   | `softmax(dim=-1)` | (2, 8, 77, 77) | float32 | 1e-6 | 1e-6 |
   | `softmax(dim=-1)` | (2, 77, 768) | float16 | 1e-3 | 1e-3 |
   | `softmax(dim=0)` — MUST fall back | (16, 16) | float32 | 1e-6 | 1e-6 |
   | `layer_norm([D])` with weight+bias | (2, 77, 768) | float32 | 1e-4 | 1e-4 |
   | `layer_norm([D])` no affine | (2, 77, 64) | float32 | 1e-4 | 1e-4 |
   | `layer_norm([D])` with weight+bias | (2, 77, 768) | float16 | 1e-2 | 1e-2 |
   | `group_norm(groups=8)` with weight+bias | (2, 64, 8, 8) | float32 | 1e-4 | 1e-4 |
   | `group_norm(groups=32)` no affine | (1, 320, 8, 8) | float32 | 1e-4 | 1e-4 |
   | `group_norm(groups=8)` | (2, 64, 8, 8) | float16 | 1e-2 | 1e-2 |

**Verification**: template of 6-7, smoke:

```bash
python3 setup.py build_ext --inplace
MTORCH_METAL_LOG=1 python3 -c "
import mtorch
x = mtorch.ones(2, 8, device='mps')
s = mtorch.nn.functional.softmax(x, dim=-1)
print(s.cpu().tolist()[0][0])
ln = mtorch.nn.functional.layer_norm(x, [8])
print(ln.cpu().tolist()[0][0])
g = mtorch.ones(1, 8, 4, 4, device='mps')
gn = mtorch.nn.functional.group_norm(g, 4)
print(gn.cpu().tolist()[0][0][0][0])
" 2>&1 | tee /tmp/p6-norm.txt
grep -c 'launch mtorch_softmax_rows_f32' /tmp/p6-norm.txt
grep -c 'launch mtorch_layer_norm_rows_f32' /tmp/p6-norm.txt
grep -c 'launch mtorch_group_norm_stats_f32' /tmp/p6-norm.txt
pytest tests/compat/test_device_parity.py -q | tail -3
pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
tail -1 docs/design/baseline/tests-phase6-cpu.txt
```

Expected smoke: `0.125`, `0.0`, `0.0`, all grep counts ≥1. BENCH-GUARD ids:
`bench.softmax_dim1_64x64`, `bench.nn_functional_layer_norm_64x64`,
`bench.nn_functional_layer_norm_half_2x77x768`,
`bench.nn_functional_group_norm_8x16x8x8`. Ratios:
`python3 tools/metal_ratios.py --family norm`; record the Notes line.

**On failure**: STANDARD FAIL-6 (`<new files>` =
`cpp/mtorch/core/metal/normalization.mm`; restore `kernels.metal ops.h
normalization.cpp test_device_parity.py`). A hang or garbage values here is
almost always a barrier bug — see §3 "threadgroup-size crashes".

**Commit**:

```bash
git add cpp/mtorch/core/metal/kernels.metal cpp/mtorch/core/metal/ops.h \
        cpp/mtorch/core/metal/normalization.mm cpp/mtorch/core/normalization.cpp \
        tests/compat/test_device_parity.py
git commit -m "feat(phase6-11): metal softmax/layer_norm/group_norm (threadgroup reductions)"
```

---

### Step 6-12: Op family (f) — conv2d via im2col + matmul

**Goal**: `conv2d` (4-D NCHW, groups == 1) runs on the GPU as im2col followed
by the 6-10 GEMM. **Chosen over MPSCNNConvolution because it is deterministic,
reuses the already-parity-tested GEMM, and avoids MPSCNN's weight-layout and
descriptor complexity** (the required one-line justification). conv1d/3d,
transposed convs, and grouped convs all fall back.

**Background: im2col (7 lines).** A convolution is a matrix multiply in
disguise. For one image, gather every kernel-sized input patch into a column
of a matrix: the "col" matrix has `C*KH*KW` rows (one per input-channel ×
kernel-position) and `OH*OW` columns (one per output pixel); out-of-bounds
(padding) positions contribute 0. The weight tensor `(OC, C, KH, KW)`, viewed
row-major, is already an `(OC, C*KH*KW)` matrix — no rearrangement needed.
Then `weight_matrix @ col = (OC, OH*OW)`, which is exactly the output image
laid out row-major. We do this once per batch image, entirely on the GPU:
im2col kernel → `run_gemm_f32` → a store kernel that adds the bias and writes
(f32) or casts (f16) into the output tensor.

**Preconditions**:

```bash
grep -n 'TensorPtr conv2d(' cpp/mtorch/core/conv.cpp
```

Must print the definition line (`conv.cpp` is the 2b-13 section file; the
signature spans 8 lines ending `    int64_t groups) {`).

**Actions**:

1. Append to `cpp/mtorch/core/metal/kernels.metal`:

   ```
   // ---------------- family (f): conv2d im2col ----------------
   // Everything below runs in f32: f16 convs are staged through the family
   // (d) cast kernels first (accumulate-in-float rule).

   struct Im2colParams {
     uint total;    // C*KH*KW*OH*OW (one thread per col-matrix element)
     uint c;        // input channels
     uint h;        // input height
     uint w;        // input width
     uint kh;
     uint kw;
     uint oh;
     uint ow;
     uint stride_h;
     uint stride_w;
     uint pad_h;
     uint pad_w;
     uint dil_h;
     uint dil_w;
     uint in_off;   // element offset of this image inside the input buffer
   };

   kernel void mtorch_im2col_f32(
       device const float* img [[buffer(0)]],
       device float* col [[buffer(1)]],
       constant Im2colParams& p [[buffer(2)]],
       uint gid [[thread_position_in_grid]]) {
     if (gid >= p.total) {
       return;
     }
     const uint ohow = p.oh * p.ow;
     const uint r = gid / ohow;          // row in the col matrix: (c, ky, kx)
     const uint pos = gid % ohow;        // output pixel: (oy, ox)
     const uint ox = pos % p.ow;
     const uint oy = pos / p.ow;
     const uint kx = r % p.kw;
     const uint ky = (r / p.kw) % p.kh;
     const uint c = r / (p.kw * p.kh);
     const int iy = int(oy * p.stride_h) - int(p.pad_h) + int(ky * p.dil_h);
     const int ix = int(ox * p.stride_w) - int(p.pad_w) + int(kx * p.dil_w);
     float v = 0.0f;   // padding contributes zero
     if (iy >= 0 && iy < int(p.h) && ix >= 0 && ix < int(p.w)) {
       v = img[p.in_off + (c * p.h + uint(iy)) * p.w + uint(ix)];
     }
     col[gid] = v;     // col is a fresh offset-0 scratch buffer
   }

   struct ConvStoreParams {
     uint n;         // OC*OH*OW (this image's output elements)
     uint ohow;      // OH*OW
     uint out_off;   // element offset of this image inside the output tensor
     uint has_bias;
   };

   #define MTORCH_CONV_STORE(SUFFIX, T)                                       \
   kernel void mtorch_conv_store_##SUFFIX(                                    \
       device const float* acc [[buffer(0)]],                                 \
       device const float* bias [[buffer(1)]],                                \
       device T* out [[buffer(2)]],                                           \
       constant ConvStoreParams& p [[buffer(3)]],                             \
       uint gid [[thread_position_in_grid]]) {                                \
     if (gid >= p.n) {                                                        \
       return;                                                                \
     }                                                                        \
     float v = acc[gid];                                                      \
     if (p.has_bias != 0u) {                                                  \
       v += bias[gid / p.ohow];                                               \
     }                                                                        \
     out[p.out_off + gid] = T(v);                                             \
   }

   MTORCH_CONV_STORE(f32, float)
   MTORCH_CONV_STORE(f16, half)
   #undef MTORCH_CONV_STORE
   ```

2. `cpp/mtorch/core/metal/ops.h` — after the family (e) declaration block:

   ```cpp

   // family (f): conv2d (step 6-12)
   bool use_conv2d(const Tensor& input, const Tensor& weight,
                   const TensorPtr& bias, const std::vector<int64_t>& stride,
                   const std::vector<int64_t>& padding,
                   const std::vector<int64_t>& dilation, int64_t groups,
                   bool grad_tracked);
   TensorPtr conv2d(const TensorPtr& input, const TensorPtr& weight,
                    const TensorPtr& bias, const std::vector<int64_t>& stride,
                    const std::vector<int64_t>& padding,
                    const std::vector<int64_t>& dilation);
   ```

   and after the family (e) stub block:

   ```cpp
   inline bool use_conv2d(const Tensor&, const Tensor&, const TensorPtr&,
                          const std::vector<int64_t>&, const std::vector<int64_t>&,
                          const std::vector<int64_t>&, int64_t, bool) {
     return false;
   }
   inline TensorPtr conv2d(const TensorPtr&, const TensorPtr&, const TensorPtr&,
                           const std::vector<int64_t>&, const std::vector<int64_t>&,
                           const std::vector<int64_t>&) {
     return nullptr;
   }
   ```

3. Create `cpp/mtorch/core/metal/conv.mm`:

   ```objc
   #include "mtorch/core/metal/ops.h"
   #include "mtorch/core/metal/context.h"

   #include <cstdint>
   #include <vector>

   namespace mtorch::metal {

   namespace {

   struct Im2colParams {
     uint32_t total;
     uint32_t c;
     uint32_t h;
     uint32_t w;
     uint32_t kh;
     uint32_t kw;
     uint32_t oh;
     uint32_t ow;
     uint32_t stride_h;
     uint32_t stride_w;
     uint32_t pad_h;
     uint32_t pad_w;
     uint32_t dil_h;
     uint32_t dil_w;
     uint32_t in_off;
   };

   struct ConvStoreParams {
     uint32_t n;
     uint32_t ohow;
     uint32_t out_off;
     uint32_t has_bias;
   };

   int64_t conv_out_size(int64_t in, int64_t kernel, int64_t stride,
                         int64_t pad, int64_t dil) {
     return (in + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
   }

   }  // namespace

   bool use_conv2d(const Tensor& input, const Tensor& weight,
                   const TensorPtr& bias, const std::vector<int64_t>& stride,
                   const std::vector<int64_t>& padding,
                   const std::vector<int64_t>& dilation, int64_t groups,
                   bool grad_tracked) {
     if (input.device.type != DeviceType::Metal ||
         weight.device.type != DeviceType::Metal) {
       return false;
     }
     if (!metal_ready() || grad_tracked) {
       return false;
     }
     if (groups != 1) {
       return false;  // v1: no grouped/depthwise convs
     }
     if (input.dim() != 4 || weight.dim() != 4) {
       return false;
     }
     if (stride.size() != 2 || padding.size() != 2 || dilation.size() != 2) {
       return false;
     }
     if (stride[0] <= 0 || stride[1] <= 0 || dilation[0] <= 0 ||
         dilation[1] <= 0 || padding[0] < 0 || padding[1] < 0) {
       return false;  // invalid config: CPU path raises the proper error
     }
     if (input.dtype != weight.dtype) {
       return false;
     }
     if (input.sizes[1] != weight.sizes[1]) {
       return false;  // channel mismatch: CPU path raises
     }
     if (bias != nullptr &&
         (bias->dtype != input.dtype || bias->numel() != weight.sizes[0] ||
          bias->requires_grad || !metal_dense_f32f16(*bias))) {
       return false;
     }
     if (!metal_dense_f32f16(input) || !metal_dense_f32f16(weight)) {
       return false;
     }
     const int64_t oh = conv_out_size(input.sizes[2], weight.sizes[2],
                                      stride[0], padding[0], dilation[0]);
     const int64_t ow = conv_out_size(input.sizes[3], weight.sizes[3],
                                      stride[1], padding[1], dilation[1]);
     if (oh <= 0 || ow <= 0) {
       return false;  // degenerate output: CPU path raises
     }
     // The col matrix (C*KH*KW x OH*OW) must also fit the uint32 indexing.
     const int64_t col_elems =
         input.sizes[1] * weight.sizes[2] * weight.sizes[3] * oh * ow;
     if (col_elems > 0xffffffffLL) {
       return false;
     }
     return true;
   }

   TensorPtr conv2d(const TensorPtr& input, const TensorPtr& weight,
                    const TensorPtr& bias, const std::vector<int64_t>& stride,
                    const std::vector<int64_t>& padding,
                    const std::vector<int64_t>& dilation) {
     const int64_t batch = input->sizes[0];
     const int64_t channels = input->sizes[1];
     const int64_t in_h = input->sizes[2];
     const int64_t in_w = input->sizes[3];
     const int64_t out_channels = weight->sizes[0];
     const int64_t kh = weight->sizes[2];
     const int64_t kw = weight->sizes[3];
     const int64_t oh = conv_out_size(in_h, kh, stride[0], padding[0], dilation[0]);
     const int64_t ow = conv_out_size(in_w, kw, stride[1], padding[1], dilation[1]);
     const int64_t ohow = oh * ow;
     const int64_t col_rows = channels * kh * kw;

     TensorPtr out = empty_strided({batch, out_channels, oh, ow},
                                   contiguous_strides({batch, out_channels, oh, ow}),
                                   input->dtype, false, input->device);

     Context& ctx = Context::instance();
     // Stage the whole input and weight (and bias) into f32 scratch buffers
     // (identity copy for f32, cast for f16) — see the 6-10 Background.
     id<MTLBuffer> img = stage_to_f32(*input, static_cast<size_t>(input->offset),
                                      static_cast<size_t>(input->numel()));
     id<MTLBuffer> wmat = stage_to_f32(*weight, static_cast<size_t>(weight->offset),
                                       static_cast<size_t>(weight->numel()));
     id<MTLBuffer> bias_buf =
         bias != nullptr
             ? stage_to_f32(*bias, static_cast<size_t>(bias->offset),
                            static_cast<size_t>(bias->numel()))
             : ctx.dummy_buffer();
     id<MTLBuffer> col = ctx.temp_buffer(
         static_cast<size_t>(col_rows * ohow) * sizeof(float));
     id<MTLBuffer> acc = ctx.temp_buffer(
         static_cast<size_t>(out_channels * ohow) * sizeof(float));

     for (int64_t n = 0; n < batch; ++n) {
       Im2colParams ip{static_cast<uint32_t>(col_rows * ohow),
                       static_cast<uint32_t>(channels),
                       static_cast<uint32_t>(in_h),
                       static_cast<uint32_t>(in_w),
                       static_cast<uint32_t>(kh),
                       static_cast<uint32_t>(kw),
                       static_cast<uint32_t>(oh),
                       static_cast<uint32_t>(ow),
                       static_cast<uint32_t>(stride[0]),
                       static_cast<uint32_t>(stride[1]),
                       static_cast<uint32_t>(padding[0]),
                       static_cast<uint32_t>(padding[1]),
                       static_cast<uint32_t>(dilation[0]),
                       static_cast<uint32_t>(dilation[1]),
                       static_cast<uint32_t>(n * channels * in_h * in_w)};
       launch_1d("mtorch_im2col_f32", {img, col}, &ip, sizeof(ip),
                 static_cast<size_t>(col_rows * ohow));
       // (OC x col_rows) @ (col_rows x ohow) -> (OC x ohow)
       run_gemm_f32(wmat, out_channels, col_rows, col, col_rows, ohow,
                    /*transpose_right=*/false, acc, 1.0, 0.0);
       ConvStoreParams sp{static_cast<uint32_t>(out_channels * ohow),
                          static_cast<uint32_t>(ohow),
                          static_cast<uint32_t>(out->offset +
                                                n * out_channels * ohow),
                          bias != nullptr ? 1u : 0u};
       const char* store_fn = input->dtype == ScalarType::Float32
                                  ? "mtorch_conv_store_f32"
                                  : "mtorch_conv_store_f16";
       launch_1d(store_fn, {acc, bias_buf, buffer_for(*out->storage)},
                 &sp, sizeof(sp), static_cast<size_t>(out_channels * ohow));
     }
     return out;
   }

   }  // namespace mtorch::metal
   ```

4. Hook `conv2d` in `cpp/mtorch/core/conv.cpp`. Add the include after the last
   `#include` line (`grep -n '#include' cpp/mtorch/core/conv.cpp | tail -1`):

   ```cpp
   #include "mtorch/core/metal/ops.h"
   ```

   BEFORE (the function's opening — signature ends `    int64_t groups) {`,
   followed by its device checks; locate with
   `grep -n -A 12 '^TensorPtr conv2d(' cpp/mtorch/core/conv.cpp`):

   ```cpp
     ensure_same_device(*input, *weight);
     if (bias) {
       ensure_same_device(*input, *bias);
     }
   ```

   AFTER (insert the guard right after that block):

   ```cpp
     ensure_same_device(*input, *weight);
     if (bias) {
       ensure_same_device(*input, *bias);
     }
     if (metal::use_conv2d(*input, *weight, bias, stride, padding, dilation,
                           groups,
                           track_grad(input->requires_grad ||
                                      weight->requires_grad ||
                                      (bias && bias->requires_grad)))) {
       return metal::conv2d(input, weight, bias, stride, padding, dilation);
     }
   ```

   (If the `ensure_same_device` block's exact shape differs in your tree,
   insert the guard immediately after the last device check at the top of
   `conv2d`, using the actual parameter names.)

5. PARITY-ROWS (§0.9). f32 tolerance 1e-4 (GEMM-reassociated accumulation over
   `C*KH*KW` terms), f16 1e-2:

   | op | input / weight | extras | dtype | rtol | atol |
   |---|---|---|---|---|---|
   | `conv2d` | (1, 4, 8, 8) / (8, 4, 3, 3) | padding=1 | float32 | 1e-4 | 1e-4 |
   | `conv2d` | (2, 16, 16, 16) / (32, 16, 3, 3) | stride=2, padding=1, bias | float32 | 1e-4 | 1e-4 |
   | `conv2d` | (1, 16, 8, 8) / (16, 16, 1, 1) | 1×1, bias | float16 | 1e-2 | 1e-2 |
   | `conv2d` | (1, 4, 8, 8) / (8, 4, 3, 3) | dilation=2, padding=2 | float32 | 1e-4 | 1e-4 |
   | `conv2d` | (1, 4, 8, 8) / (4, 2, 3, 3) | groups=2 — MUST fall back | float32 | 1e-4 | 1e-4 |

**Verification**: template of 6-7, smoke (a 1×1×3×3 all-ones conv with a
1×1×2×2 all-ones kernel: every interior output = 4.0):

```bash
python3 setup.py build_ext --inplace
MTORCH_METAL_LOG=1 python3 -c "
import mtorch
x = mtorch.ones(1, 1, 3, 3, device='mps')
w = mtorch.ones(1, 1, 2, 2, device='mps')
print(mtorch.nn.functional.conv2d(x, w).cpu().tolist())
" 2>&1 | tee /tmp/p6-conv.txt
grep -c 'launch mtorch_im2col_f32' /tmp/p6-conv.txt
grep -c 'launch mtorch_conv_store_f32' /tmp/p6-conv.txt
pytest tests/compat/test_device_parity.py -q | tail -3
pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
tail -1 docs/design/baseline/tests-phase6-cpu.txt
```

Expected smoke: `[[[[4.0, 4.0], [4.0, 4.0]]]]`, both grep counts ≥1.
BENCH-GUARD ids: `bench.nn_functional_conv2d_1x4x8x8`,
`bench.nn_functional_conv2d_same_1x4x8x8`,
`bench.nn_functional_conv2d_half_1x1_1x16x8x8`. Ratios:
`python3 tools/metal_ratios.py --family conv`. Expected-outcome guidance: the
per-image synchronous loop (im2col + gemm + store, each with its own
`waitUntilCompleted`) makes small convs launch-bound; torch MPS will be much
faster — acceptable; record the Notes line.

**On failure**: STANDARD FAIL-6 (`<new files>` =
`cpp/mtorch/core/metal/conv.mm`; restore `kernels.metal ops.h conv.cpp
test_device_parity.py`).

**Commit**:

```bash
git add cpp/mtorch/core/metal/kernels.metal cpp/mtorch/core/metal/ops.h \
        cpp/mtorch/core/metal/conv.mm cpp/mtorch/core/conv.cpp \
        tests/compat/test_device_parity.py
git commit -m "feat(phase6-12): metal conv2d via im2col + MPS gemm"
```

---

### Step 6-13: Op family (g) — scaled_dot_product_attention

**Goal**: `scaled_dot_product_attention` (4-D, no mask tensor, no dropout, no
GQA; optional `is_causal`) runs on the GPU **composed entirely from the
already-tested primitives**: stage → GEMM (family d) → causal-mask kernel →
softmax rows (family e kernel) → GEMM → store. No new reduction code is
introduced; the only new kernel is the trivial causal mask.

**Background: the composition (6 lines).** For each of the `B*H` attention
slices: `scores = scale * Q(Lq×D) @ K(Lk×D)ᵀ` — one GEMM with
`transposeRight:YES` and `alpha = scale` (scale defaults to `1/sqrt(D)`);
`is_causal` overwrites `scores[r, c] = -INF` for `c > r` (upper triangle), so
softmax assigns those positions probability 0; `softmax_rows` normalizes each
of the `Lq` rows (reusing the 6-11 kernel on a `Lq×Lk` scratch); finally
`out = scores(Lq×Lk) @ V(Lk×D)` — a second GEMM. Everything runs on f32
scratch buffers; f16 tensors are cast in and out (accumulate-in-float), which
also matches how the CPU f16 attention path stages through float.

**Preconditions**:

```bash
grep -n 'TensorPtr scaled_dot_product_attention(' cpp/mtorch/core/attention.cpp
```

Must print the definition line (`attention.cpp` is the 2b-8 section file; the
signature spans 9 lines ending `    bool enable_gqa) {`).

**Actions**:

1. Append to `cpp/mtorch/core/metal/kernels.metal`:

   ```
   // ---------------- family (g): attention causal mask ----------------

   struct CausalParams {
     uint total;   // rows*cols of one score slice
     uint cols;
   };

   kernel void mtorch_causal_mask_f32(
       device float* scores [[buffer(0)]],
       constant CausalParams& p [[buffer(1)]],
       uint gid [[thread_position_in_grid]]) {
     if (gid >= p.total) {
       return;
     }
     const uint col = gid % p.cols;
     const uint row = gid / p.cols;
     if (col > row) {
       scores[gid] = -INFINITY;
     }
   }
   ```

2. `cpp/mtorch/core/metal/ops.h` — after the family (f) declaration block:

   ```cpp

   // family (g): scaled_dot_product_attention (step 6-13)
   bool use_sdpa(const Tensor& query, const Tensor& key, const Tensor& value,
                 const TensorPtr& attn_mask, double dropout_p, bool is_causal,
                 bool enable_gqa, bool grad_tracked);
   TensorPtr sdpa(const TensorPtr& query, const TensorPtr& key,
                  const TensorPtr& value, bool is_causal, double scale);
   ```

   and after the family (f) stub block:

   ```cpp
   inline bool use_sdpa(const Tensor&, const Tensor&, const Tensor&,
                        const TensorPtr&, double, bool, bool, bool) {
     return false;
   }
   inline TensorPtr sdpa(const TensorPtr&, const TensorPtr&, const TensorPtr&,
                         bool, double) {
     return nullptr;
   }
   ```

3. Create `cpp/mtorch/core/metal/attention.mm`:

   ```objc
   #include "mtorch/core/metal/ops.h"
   #include "mtorch/core/metal/context.h"

   #include <cstdint>

   namespace mtorch::metal {

   namespace {

   struct CausalParams {
     uint32_t total;
     uint32_t cols;
   };

   struct RowParams {          // mirror of the 6-11 softmax params
     uint32_t rows;
     uint32_t cols;
     uint32_t in_off;
     uint32_t out_off;
   };

   }  // namespace

   bool use_sdpa(const Tensor& query, const Tensor& key, const Tensor& value,
                 const TensorPtr& attn_mask, double dropout_p, bool is_causal,
                 bool enable_gqa, bool grad_tracked) {
     if (query.device.type != DeviceType::Metal ||
         key.device.type != DeviceType::Metal ||
         value.device.type != DeviceType::Metal) {
       return false;
     }
     if (!metal_ready() || grad_tracked) {
       return false;
     }
     if (attn_mask != nullptr || dropout_p != 0.0 || enable_gqa) {
       return false;  // v1: no mask tensor, no dropout, no GQA
     }
     if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
       return false;
     }
     if (query.dtype != key.dtype || query.dtype != value.dtype) {
       return false;
     }
     // Shapes: q (B,H,Lq,D), k (B,H,Lk,D), v (B,H,Lk,D).
     if (query.sizes[0] != key.sizes[0] || query.sizes[1] != key.sizes[1] ||
         key.sizes != value.sizes || query.sizes[3] != key.sizes[3]) {
       return false;
     }
     if (is_causal && query.sizes[2] != key.sizes[2]) {
       return false;  // causal with Lq != Lk: alignment subtleties -> CPU
     }
     return metal_dense_f32f16(query) && metal_dense_f32f16(key) &&
            metal_dense_f32f16(value);
   }

   TensorPtr sdpa(const TensorPtr& query, const TensorPtr& key,
                  const TensorPtr& value, bool is_causal, double scale) {
     const int64_t batch = query->sizes[0];
     const int64_t heads = query->sizes[1];
     const int64_t lq = query->sizes[2];
     const int64_t lk = key->sizes[2];
     const int64_t d = query->sizes[3];

     TensorPtr out = empty_strided({batch, heads, lq, d},
                                   contiguous_strides({batch, heads, lq, d}),
                                   query->dtype, false, query->device);
     Context& ctx = Context::instance();
     // Per-slice f32 scratch, reused across slices (each launch is
     // synchronous, so reuse cannot race).
     id<MTLBuffer> scores =
         ctx.temp_buffer(static_cast<size_t>(lq * lk) * sizeof(float));

     for (int64_t slice = 0; slice < batch * heads; ++slice) {
       const size_t q_off =
           static_cast<size_t>(query->offset) + static_cast<size_t>(slice * lq * d);
       const size_t kv_off =
           static_cast<size_t>(key->offset) + static_cast<size_t>(slice * lk * d);
       const size_t v_off =
           static_cast<size_t>(value->offset) + static_cast<size_t>(slice * lk * d);
       id<MTLBuffer> qb = stage_to_f32(*query, q_off, static_cast<size_t>(lq * d));
       id<MTLBuffer> kb = stage_to_f32(*key, kv_off, static_cast<size_t>(lk * d));
       id<MTLBuffer> vb = stage_to_f32(*value, v_off, static_cast<size_t>(lk * d));

       // scores(Lq x Lk) = scale * Q @ K^T
       run_gemm_f32(qb, lq, d, kb, lk, d, /*transpose_right=*/true, scores,
                    scale, 0.0);

       if (is_causal) {
         CausalParams cp{static_cast<uint32_t>(lq * lk),
                         static_cast<uint32_t>(lk)};
         launch_1d("mtorch_causal_mask_f32", {scores}, &cp, sizeof(cp),
                   static_cast<size_t>(lq * lk));
       }

       // In-place row softmax on the score slice (6-11 kernel, f32 variant).
       RowParams rp{static_cast<uint32_t>(lq), static_cast<uint32_t>(lk), 0u, 0u};
       launch_rows("mtorch_softmax_rows_f32", {scores, scores}, &rp, sizeof(rp),
                   static_cast<size_t>(lq));

       // out_slice(Lq x D) = scores @ V
       id<MTLBuffer> acc =
           ctx.temp_buffer(static_cast<size_t>(lq * d) * sizeof(float));
       run_gemm_f32(scores, lq, lk, vb, lk, d, /*transpose_right=*/false, acc,
                    1.0, 0.0);
       store_from_f32(acc, *out,
                      static_cast<size_t>(out->offset) +
                          static_cast<size_t>(slice * lq * d),
                      static_cast<size_t>(lq * d));
     }
     return out;
   }

   }  // namespace mtorch::metal
   ```

4. Hook the op in `cpp/mtorch/core/attention.cpp`. Add the include after the
   last `#include` line
   (`grep -n '#include' cpp/mtorch/core/attention.cpp | tail -1`):

   ```cpp
   #include "mtorch/core/metal/ops.h"
   ```

   BEFORE (the device checks at the top of the function body; locate with
   `grep -n -A 14 '^TensorPtr scaled_dot_product_attention(' cpp/mtorch/core/attention.cpp`):

   ```cpp
     ensure_same_device(*query, *key, *value);
     if (attn_mask) {
       ensure_same_device(*query, *attn_mask);
     }
   ```

   AFTER (insert the guard right after that block; the scale resolution
   `scale.value_or(...)` reproduces torch's default `1/sqrt(head_dim)`):

   ```cpp
     ensure_same_device(*query, *key, *value);
     if (attn_mask) {
       ensure_same_device(*query, *attn_mask);
     }
     if (metal::use_sdpa(*query, *key, *value, attn_mask, dropout_p, is_causal,
                         enable_gqa,
                         track_grad(query->requires_grad || key->requires_grad ||
                                    value->requires_grad))) {
       const double resolved_scale =
           scale.has_value() ? *scale
                             : 1.0 / std::sqrt(static_cast<double>(query->sizes[3]));
       return metal::sdpa(query, key, value, is_causal, resolved_scale);
     }
   ```

   (If `<cmath>` is somehow not reachable in this file the build will say so —
   then also add `#include <cmath>` next to the ops.h include. If the device-
   check block's shape differs, insert after the last device check with the
   actual parameter names.)

5. PARITY-ROWS (§0.9). f32 tolerance 1e-4, f16 1e-2 (two chained GEMMs plus a
   softmax; §2):

   | op | q/k/v shape | extras | dtype | rtol | atol |
   |---|---|---|---|---|---|
   | `scaled_dot_product_attention` | (1, 2, 16, 16) | — | float32 | 1e-4 | 1e-4 |
   | `scaled_dot_product_attention` | (2, 8, 64, 64) | is_causal=True | float32 | 1e-4 | 1e-4 |
   | `scaled_dot_product_attention` | (1, 2, 77, 64) | is_causal=True | float16 | 1e-2 | 1e-2 |
   | `scaled_dot_product_attention` | (1, 2, 16, 16) | scale=0.5 | float32 | 1e-4 | 1e-4 |
   | `scaled_dot_product_attention` | (1, 2, 16, 16) | dropout_p=1.0 — MUST fall back | float32 | 1e-4 | 1e-4 |
   | `scaled_dot_product_attention` | q (1,2,8,16), k/v (1,2,32,16) cross-attn, non-causal | — | float32 | 1e-4 | 1e-4 |

**Verification**: template of 6-7, smoke (uniform inputs: softmax of equal
scores is uniform, so output = mean of V = 1.0):

```bash
python3 setup.py build_ext --inplace
MTORCH_METAL_LOG=1 python3 -c "
import mtorch
q = mtorch.ones(1, 2, 4, 8, device='mps')
out = mtorch.nn.functional.scaled_dot_product_attention(q, q, q)
print(out.cpu().tolist()[0][0][0][0])
out_c = mtorch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
print(out_c.cpu().tolist()[0][0][0][0])
" 2>&1 | tee /tmp/p6-sdpa.txt
grep -c 'launch mtorch_softmax_rows_f32' /tmp/p6-sdpa.txt
grep -c 'launch mtorch_causal_mask_f32' /tmp/p6-sdpa.txt
pytest tests/compat/test_device_parity.py -q | tail -3
pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
tail -1 docs/design/baseline/tests-phase6-cpu.txt
```

Expected smoke: `1.0` twice, both grep counts ≥1. BENCH-GUARD ids:
`bench.nn_functional_sdpa_1x2x16x16`, `bench.nn_functional_sdpa_gqa_1x4x16x16`,
`bench.stable_diffusion_clip_causal_attention_half_1x2x77x64`. Ratios:
`python3 tools/metal_ratios.py --family sdpa`. Expected-outcome guidance: the
per-slice loop issues ~7 synchronous submissions per (batch, head) — expect
mtorch-mps to trail torch-MPS by a large factor on small shapes; acceptable;
record the Notes line.

**On failure**: STANDARD FAIL-6 (`<new files>` =
`cpp/mtorch/core/metal/attention.mm`; restore `kernels.metal ops.h
attention.cpp test_device_parity.py`).

**Commit**:

```bash
git add cpp/mtorch/core/metal/kernels.metal cpp/mtorch/core/metal/ops.h \
        cpp/mtorch/core/metal/attention.mm cpp/mtorch/core/attention.cpp \
        tests/compat/test_device_parity.py
git commit -m "feat(phase6-13): metal scaled_dot_product_attention (gemm+softmax composition)"
```

---

### Step 6-14: Phase gate

**Goal**: Prove, in one sitting: (1) the full CPU suite is unchanged, (2) the
parity suite is green on mps, (3) the full CPU benchmark set is within the 5%
gate (the Metal code must not have slowed the CPU path), (4) the mps-vs-cpu
ratio table is recorded. Then re-capture the canonical full-suite baseline so
future phases inherit a single source of truth.

**Preconditions**: 6-13 complete; no uncommitted changes
(`git status --short` prints nothing).

**Actions**:

1. Full test suite (this is `01-rules-and-verification.md` §5.1 in spirit; the
   *full* run includes the parity tests, whose count has grown — verify the
   CPU part against the frozen file and the parity part for green):

   ```bash
   pytest tests/compat -q --ignore=tests/compat/test_device_parity.py 2>&1 | tail -1
   tail -1 docs/design/baseline/tests-phase6-cpu.txt
   pytest tests/compat/test_device_parity.py -q | tail -1
   ```

   Gate: the first two lines match exactly (§5.1 normalization); the parity
   line has no `failed`/`error`.

2. Kill-switch and forced-fallback sanity (the §3 debugging contract must hold
   at the gate):

   ```bash
   MTORCH_DISABLE_MPS=1 python3 -c "import mtorch; print(mtorch.backends.mps.is_available())"
   MTORCH_METAL_FORCE_FALLBACK=1 pytest tests/compat/test_device_parity.py -q | tail -1
   ```

   Gate: `False`, then a green parity summary (every op falls back and still
   matches the CPU — this catches any op whose CPU fallback was accidentally
   broken by a hook edit).

3. Full benchmark comparison, exactly `01-rules-and-verification.md` §5.2:

   ```bash
   mkdir -p benchmark-results
   pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
       --compat-benchmark-json benchmark-results/current.json
   python3 tools/compare_benchmarks.py \
       docs/design/baseline/benchmark-baseline.json benchmark-results/current.json
   ```

   Gate: final line shows `regressions=0 missing=0` (`new=0` as well — Phase 6
   adds no benchmark cases). On regressions, apply §5.2's 3-rerun noise rule;
   a reproducible regression means a guard is too hot — STANDARD FAIL-6 (the
   likely culprit is a guard doing work before its device check; fix under the
   §6 3-attempt budget or revert the offending family).

4. Record the ratio table (the exact command required by the plan):

   ```bash
   python3 tools/metal_ratios.py --write docs/design/baseline/metal-ratios.md
   cat docs/design/baseline/metal-ratios.md
   ```

   There is no numeric gate on these ratios — v1 Metal being slower than torch
   MPS (and, on small shapes, slower than CPU) is explicitly acceptable. The
   file is the recorded evidence.

5. Re-capture the canonical full-suite baseline (the parity rows added in
   6-7…6-13 legitimately increased the totals; from here on, future phases
   compare against this):

   ```bash
   pytest tests/compat 2>&1 | tee /tmp/tests-current.txt
   cp /tmp/tests-current.txt docs/design/baseline/tests-baseline.txt
   pytest tests/compat --collect-only -q | tail -2 > docs/design/baseline/collect-count.txt
   ```

6. Fill in the Phase 6 Notes with the one-line ratio summaries collected in
   6-7…6-13 if any are missing.

**Verification**: every gate in actions 1–3 passed; both baseline files and
`metal-ratios.md` exist and reflect this run.

**On failure**: Do not commit any baseline update. STANDARD FAIL-6 against
whichever action failed (for action 3, the revert target is the specific
family commit, newest first, per §6 item 3).

**Commit**:

```bash
git add docs/design/baseline/metal-ratios.md docs/design/baseline/tests-baseline.txt \
        docs/design/baseline/collect-count.txt docs/design/PROGRESS.md
git commit -m "feat(phase6-14): phase gate passed; record metal ratios and new baseline"
```

---

## 2. Numerics: what "equal" means between CPU and GPU

### 2.1 The fp16 accumulate-in-float rule

Every f16 Metal path in this phase computes in `float` and converts to `half`
only at the final store: the elementwise kernels via
`half(f(float(x)))`, the reductions by accumulating `float` partials, and the
GEMM-based ops (matmul, conv2d, sdpa) by staging operands to f32 buffers and
running f32 GEMMs. This matches the CPU kernels, which likewise stage f16
through float. Never "optimize" this away: accumulating in `half` loses ~3
decimal digits and will blow the parity tolerances on any nontrivial K.

### 2.2 Expected divergence vs the CPU results

The GPU never produces bit-identical floats to the CPU for transcendental and
reduction ops; the differences are bounded and expected:

- **Exact ops** (fill, copy, neg, abs, add, sub, mul): bit-identical.
  IEEE-754 basic operations are correctly rounded on both processors; parity
  rows use rtol=atol=0 where marked.
- **div and transcendentals** (exp, log, sqrt, rsqrt, sigmoid, tanh, silu,
  gelu): with fast math disabled (we disable it in BOTH compile paths — 6-3
  `-fno-fast-math`, 6-4 `fastMathEnabled = NO`) the GPU implementations are
  faithfully rounded but not identical to libm — expect a few ULP, i.e.
  ~1e-7 relative for f32. The 1e-6 rtol used by the parity rows (mirroring the
  1e-6 the existing harness uses for f32 unary cases — see
  `grep -n 'unary.exp' tests/compat/cases.py`, which sets `rtol=1e-6,
  atol=1e-6`) holds comfortably **provided inputs are tame**; that is why the
  6-9 parity rows pin input ranges (|x| ≤ 4 for exp, x ≥ 0.25 for log/sqrt).
- **Reductions and GEMMs** (softmax sums, layer/group-norm moments, matmul,
  conv, sdpa): floating-point addition is not associative; the GPU's
  tree/strided summation order differs from the CPU loop, giving relative
  errors that grow ~√K for K summed terms. Hence the graded tolerances: 1e-5
  for plain matmul f32, 1e-4 for the norm family and conv/sdpa f32, and 1e-2
  for their f16 variants (where the final store rounds to a 10-bit mantissa,
  ~1e-3 relative, and the compared *values* passed through two such stores).

### 2.3 Tolerance table (authoritative for the parity rows)

The per-row `rtol`/`atol` values in steps 6-7…6-13 mirror the conventions the
existing harness already uses (verified in today's tree: `OpCase` rows carry
explicit `rtol=`/`atol=` — 409 of them in `tests/compat/cases.py`; f32 unary
rows use 1e-6/1e-6; rows that end in matmul-class accumulation use 1e-3 to
1e-5; where a row sets nothing, `torch.testing.assert_close` defaults apply —
f32 rtol=1.3e-6/atol=1e-5, f16 rtol=1e-3/atol=1e-5, via `assert_same_result`
in `tests/compat/harness.py`):

| family | f32 rtol/atol | f16 rtol/atol |
|---|---|---|
| fill / copy / neg / abs | 0 / 0 | 0 / 0 |
| binary add/sub/mul/div | 1e-6 / 1e-6 | 1e-3 / 1e-3 |
| unary transcendentals, silu, gelu(tanh) | 1e-6 / 1e-6 | 1e-3 / 1e-3 |
| matmul | 1e-5 / 1e-5 | 1e-3 / 1e-3 |
| softmax | 1e-6 / 1e-6 | 1e-3 / 1e-3 |
| layer_norm / group_norm | 1e-4 / 1e-4 | 1e-2 / 1e-2 |
| conv2d | 1e-4 / 1e-4 | 1e-2 / 1e-2 |
| scaled_dot_product_attention | 1e-4 / 1e-4 | 1e-2 / 1e-2 |

Iron Rule 1 still applies with full force: these tolerances are fixed by this
guide *before* any kernel runs. If a parity test fails at these tolerances,
the kernel is wrong — never widen a tolerance to make a red test green.

---

## 3. Debugging appendix

Work through this list top to bottom; each item is cheap and eliminates a
whole class of causes.

1. **First, always: force the fallback.**

   ```bash
   MTORCH_METAL_FORCE_FALLBACK=1 pytest tests/compat/test_device_parity.py -q | tail -3
   ```

   Green → the harness, allocator, device plumbing, and hook edits are all
   fine, and the bug is inside exactly one kernel/wrapper — the one your
   failing test exercises. Red → the problem is NOT your kernel: suspect the
   hook edit (did you change behavior on the fallback path?) or the allocator.
   This single command cuts the search space in half; use it before reading
   any GPU trace.

2. **See what actually launched.** `MTORCH_METAL_LOG=1 <failing command>`
   prints every kernel launch. No launch line → the guard rejected the call
   (check each guard condition against the input); an unexpected kernel name
   or `n` → the wrapper computed wrong params.

3. **Shader validation.**

   ```bash
   MTL_SHADER_VALIDATION=1 MTORCH_METAL_LOG=1 python3 -c "<repro>"
   ```

   `MTL_SHADER_VALIDATION=1` makes the GPU compiler instrument every buffer
   access; out-of-bounds reads/writes (the classic wrong-offset bug) abort
   with a message naming the kernel and the bad address instead of silently
   corrupting a neighboring tensor. Slow — use on small repros only.

4. **API validation.** `MTL_DEBUG_LAYER=1` enables the Metal validation layer
   on the API side: mismatched buffer indices, missing `endEncoding`, encoder
   misuse, and setBytes-size mismatches raise immediately with a readable
   assertion instead of undefined behavior later.

5. **GPU trace capture.** For a kernel that runs but computes wrong values,
   capture a trace and inspect buffer contents per dispatch in Xcode's Metal
   debugger. Command-line processes must opt in with
   `METAL_CAPTURE_ENABLED=1`. Temporary snippet (add inside the failing
   wrapper in the `.mm`, wrap the suspect launches, delete after use — never
   commit it):

   ```objc
   MTLCaptureManager* manager = [MTLCaptureManager sharedCaptureManager];
   MTLCaptureDescriptor* descriptor = [[MTLCaptureDescriptor alloc] init];
   descriptor.captureObject = Context::instance().device();
   descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
   descriptor.outputURL = [NSURL fileURLWithPath:@"/tmp/mtorch.gputrace"];
   NSError* error = nil;
   if (![manager startCaptureWithDescriptor:descriptor error:&error]) {
     NSLog(@"capture failed: %@", error);
   }
   // ... the suspect launches ...
   [manager stopCapture];   // then: open /tmp/mtorch.gputrace
   ```

6. **"Compiler encountered an internal error"** from
   `newLibraryWithSource:` means the *shader compiler itself* crashed — almost
   always provoked by malformed macro expansions or extreme constructs in
   `kernels.metal`, not by your host code. Bisect: comment out the most
   recently added kernel block, rebuild, retry. Also check that the generated
   `kernels_metal_source.h` is intact (`head -5
   cpp/mtorch/core/metal/kernels_metal_source.h`) — a stale or truncated
   embed produces exactly this class of error; `rm` it and rebuild.

7. **Threadgroup-size crashes / hangs.** Symptoms: `waitUntilCompleted` never
   returns, or "Compute function exceeds available threadgroup memory", or
   garbage from a reduction kernel. Causes, in likelihood order: a barrier
   inside a divergent branch (some threads return early and never reach the
   barrier — our row kernels avoid early `return` for exactly this reason);
   dispatching a reduction kernel with anything other than 256 threads
   (`launch_rows` pins this — never dispatch those kernels via `launch_1d`);
   `maxTotalThreadsPerThreadgroup` dropping below 256 for a register-heavy
   kernel (`launch_rows` throws a clear error for this; simplify the kernel).

8. **A GPU fault takes down later launches too.** After a command-buffer
   error, the queue may refuse subsequent work in the same process. Fix the
   first error first; ignore the cascade (same principle as §6 rule 1 for
   compile errors).

---

## 4. Oracle strategy

- **Primary gate (deterministic, always available):**
  `tests/compat/test_device_parity.py` — every op result on `"mps"` is
  compared against the same mtorch op on `"cpu"`, which the main compat suite
  in turn pins against live PyTorch. This self-parity is the ONLY pass/fail
  oracle for Metal kernels, and it is deterministic: same inputs, same
  tolerances, no cross-library noise.
- **Secondary oracle (advisory, opt-in): torch on MPS.** The existing compat
  harness cannot express "reference on mps": its options (verified in
  `tests/conftest.py` — `--compat-reference`, `--compat-candidate`,
  `--compat-api-manifest`, `--compat-op`, `--compat-allow-reference-backed`,
  `--compat-benchmark*`) select *modules* and case filters, not reference
  devices, and the case tables construct CPU tensors. **Do not invent new
  pytest options for this** — that would be test-infrastructure work outside
  this phase. Instead, the concrete opt-in cross-check is:

  ```bash
  python3 tools/metal_ratios.py --check
  ```

  which evaluates each ratio case on mtorch-mps AND torch-mps and prints
  `max|diff|` per case. Use it as a smell test when a parity failure makes you
  suspect the CPU reference itself; it never gates a commit (torch's MPS
  backend has its own numerics and its own bugs).
- **Why self-parity is sound here:** the fallback path shares 100% of its code
  with the CPU device, so "mps result == cpu result" verifies exactly the new
  code (kernels + guards + allocator) and nothing else. Combined with the
  force-fallback run at the gate (6-14 action 2), a green phase means: CPU
  behavior unchanged, fallback correct, kernels equal to CPU within stated
  tolerances.

---

## 5. Future work (recorded, NOT part of this phase)

1. **Asynchronous execution**: drop the per-launch `waitUntilCompleted`,
   encode multiple ops into one command buffer, and synchronize only when the
   CPU actually reads (`.cpu()`, `.item()`, fallback entry). This is the big
   perf lever (today every op pays ~50–200 µs of submission latency) but it
   needs a completion-tracking design on Storage — out of scope for v1.
2. **Batched GEMM**: `MPSMatrixDescriptor` supports `matrices`/`matrixBytes`
   for batched multiplication — would collapse the sdpa/conv per-slice loops
   into single encodes.
3. **Direct (non-staged) GEMM**: operate MPSMatrix views directly on tensor
   buffers with offsets, eliminating the staging copies, once the offset
   alignment rules are nailed down by tests.
4. **Wider guards**: broadcast binary kernels, strided elementwise kernels,
   grouped conv, log_softmax, non-last-dim softmax, GQA/masked sdpa.
5. **bfloat16** once the CPU core grows it.
6. **Packaged (non-inplace) builds**: ship `default.metallib` inside wheels;
   today the build writes it next to the in-place extension only, and the
   embedded source keeps every other install mode working.

---

End of Phase 6. After step 6-14 is committed and PROGRESS.md updated, the
repository has a correct, fully-tested, synchronous Metal backend with a
recorded performance profile — the platform for the async follow-up.











