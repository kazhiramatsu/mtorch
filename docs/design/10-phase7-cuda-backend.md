# 10. Phase 7: CUDA Backend (device "cuda")

Prerequisites: **Phase 5 complete** (`08-phase5-device-layer.md` — the device layer:
`mtorch::Device`, `DeviceType::CUDA`, the allocator registry, `to_device`, the Python
device surface, and `tests/compat/test_device_parity.py`). Phase 6
(`09-phase6-metal-backend.md`, the Metal backend) **should** also be complete — this
guide assumes its dispatch hooks may already be present and tells you how to insert
the CUDA hooks next to them — but Phase 6 is not a hard requirement; every
"if a Metal hook exists" instruction below has an explicit "if it does not" branch.

`01-rules-and-verification.md` must be read in full before starting. Everything in it
(iron rules, commit discipline, PROGRESS.md rules, the protocol on failure) applies to
this phase, with the machine-specific amendments in §0.1 and §0.8 below.

Objective: implement a CUDA backend for mtorch — a `CudaAllocator` registered for
`DeviceType::CUDA`, CUDA kernels for eight op families (fill/copy, binary elementwise,
unary, matmul, reductions, softmax/layer_norm/group_norm, conv2d,
scaled_dot_product_attention), the build-system integration that compiles `.cu` files
with `nvcc` only when CUDA is present, and the parity/oracle tests that prove each
family correct against PyTorch's own CUDA implementation.

**v1 scope (fixed, no exceptions)**: synchronous execution on the default CUDA stream,
contiguous-only kernels, no broadcasting, no autograd on CUDA, no atomics, correctness
over speed. Every restriction is enforced by an explicit guard that raises through
`unsupported_device` (§0.6). Being slower than PyTorch-CUDA is acceptable in v1;
ratios are recorded, not gated (§7-13).

---

## 0. Fundamentals specific to this phase

### 0.1 THIS PHASE RUNS ON A DIFFERENT MACHINE. Read this first.

The development machine used for Phases 0–6 is an Apple-Silicon Mac. **It has no
NVIDIA GPU. Nothing in this phase can be built, run, or tested on it.** Do not try;
`nvcc` does not exist there and never will.

Every step in this guide (7-1 through 7-13) is executed on a **separate Linux machine**
with:

- Linux, x86_64 architecture,
- an NVIDIA GPU with compute capability 8.0 or newer (checked in 7-1; older GPUs get
  one extra `-gencode` line by the exact rule in 7-2),
- CUDA Toolkit version 12 or newer (`nvcc` on `PATH` or `CUDA_HOME` set),
- Python 3 with `pytest`, `numpy`, and a **CUDA build of PyTorch** (this is the test
  oracle; `torch.cuda.is_available()` must be `True`),
- a `g++` new enough for `-std=c++20` (g++ 11 or newer),
- the mtorch repository cloned from `https://github.com/kazhiramatsu/mtorch`,
  checked out at the commit where Phase 6 finished (the human maintainer pushes that
  commit; Iron Rule 6 of `01-rules-and-verification.md` — the agent never pushes —
  still holds on BOTH machines. Getting code between the Mac and the Linux box is
  always the human's job, via `origin`).

Only step 7-14 runs back on the Mac, and it changes nothing — it is a verification
that this phase did not disturb the macOS build.

All commands in this guide are run from the repository root **on the Linux machine**
(except 7-14). The repository root on the Linux machine is wherever the clone landed;
the guide writes commands relative to it, so start every shell session with
`cd <repo-root>` and run everything from there. Where
`01-rules-and-verification.md` says `cd /Users/hiramatsu/dev/mtorch`, substitute the
Linux clone's path.

**Baseline substitution rule (applies to the whole phase):** the committed baseline
`docs/design/baseline/tests-baseline.txt` was recorded on macOS. On the Linux machine
every §5.1-style comparison uses the **Linux baseline**
`docs/design/baseline/tests-baseline-linux.txt` (and
`docs/design/baseline/collect-count-linux.txt`), which step 7-1 records. Wherever
`01-rules-and-verification.md` or the STANDARD VERIFY block below names a baseline
file, use the `-linux` variant while on the Linux machine. Step 7-14 (on the Mac) uses
the original macOS files.

### 0.2 CUDA execution model primer (read once, before writing any kernel)

You do not need prior CUDA experience for this phase; every concept used later is
defined here, and every later use of a CUDA API call is accompanied by a sentence
saying what it does.

**Background: host vs device.**
CUDA programs run on two processors at once. The *host* is the CPU running your
normal C++ code; the *device* is the GPU. They have physically separate memories:
host RAM and device (GPU) RAM. A pointer returned by `cudaMalloc` points into device
RAM and **must never be dereferenced by host code** — doing so is undefined behavior
(usually a segfault). Data moves between the two memories only through explicit
copies (`cudaMemcpy`). This is exactly why the Phase 5 contract sets
`host_accessible() == false` for the CUDA allocator: the rest of mtorch must never
touch CUDA bytes directly, only through the allocator's copy functions or a kernel.

**Background: kernels and the launch syntax `<<<grid, block>>>`.**
A *kernel* is a C++ function marked `__global__` that runs on the GPU. Calling
`my_kernel<<<num_blocks, threads_per_block>>>(args...)` launches
`num_blocks * threads_per_block` parallel GPU threads, each executing the kernel body.
The threads are organized in a *grid* of *blocks*: inside the kernel,
`blockIdx.x` is the index of this thread's block (0 … num_blocks-1), `blockDim.x` is
the number of threads per block, and `threadIdx.x` is this thread's index within its
block (0 … blockDim.x-1). The conventional global thread id is
`blockIdx.x * blockDim.x + threadIdx.x`. A kernel launch is *asynchronous*: the host
call returns immediately, and the kernel runs in the background on a *stream* (below).
This phase uses 256 threads per block for every kernel — a safe, standard choice.

**Background: grid-stride loops.**
The number of elements a kernel must process (say `n = 10^8`) usually exceeds the
number of threads launched. A *grid-stride loop* makes each thread process elements
`i, i + total_threads, i + 2*total_threads, …`:

```cpp
for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
     i < n;
     i += (int64_t)gridDim.x * blockDim.x) { ... }
```

This decouples the launch size from the data size (any grid handles any `n`,
including `n` = 0 — the loop body just never runs), so we can cap the block count
(this guide caps it at 4096) without correctness consequences. Every elementwise
kernel in this phase uses this exact loop.

**Background: warps.**
The GPU hardware executes threads in groups of 32 called *warps*. All 32 threads of a
warp execute the same instruction at the same time. Two consequences matter here:
(1) `if`/`else` divergence within a warp serializes both branches — keep kernel
bodies branch-light; (2) special *warp shuffle* instructions
(`__shfl_down_sync(mask, value, offset)`) let threads of one warp exchange register
values directly, without going through memory — the reduction kernels in 7-9 use this
for their final combine step. The `mask` argument (`0xffffffffu` = all 32 threads)
declares which threads participate.

**Background: memory coalescing.**
Device RAM is read in large aligned transactions. When the 32 threads of a warp read
32 *consecutive* addresses (thread `i` reads element `i`), the hardware merges
("coalesces") them into a few transactions and bandwidth is maximal. When threads
read scattered addresses, each read becomes its own transaction and effective
bandwidth collapses by an order of magnitude. All v1 kernels operate on contiguous
tensors indexed by the global thread id precisely so that every access is coalesced —
this is the performance reason behind the contiguous-only guard (§0.6), not just an
implementation convenience.

**Background: shared memory and `__syncthreads()`.**
Each block has access to a small (48 KB+) on-chip scratchpad called *shared memory*,
declared with `__shared__` inside a kernel. It is visible to all threads of the same
block only, and it is roughly 100x faster than device RAM. Threads within a block
cooperate through it — e.g. a *tree reduction* where 256 threads combine 256 partial
values into 1 (worked example, fully annotated, in 7-9). Because threads run
asynchronously, any hand-off through shared memory must be bracketed by
`__syncthreads()`, a barrier that all threads of the block must reach before any may
continue. Forgetting a `__syncthreads()` produces intermittent wrong answers, not
crashes — the debugging appendix (§A) tells you how to hunt these.

**Background: streams and synchronization (v1 policy).**
A *stream* is an ordered queue of GPU work; operations in one stream run in order,
operations in different streams may overlap. Everything in this phase uses the
*default stream* (stream 0) — no stream object is ever created — and after **every**
kernel launch we call `cudaDeviceSynchronize()`, which blocks the host until all
previously launched GPU work has finished. This makes execution fully synchronous:
slower, but deterministic and trivially debuggable (an error surfaces at the launch
that caused it, see §0.3). Asynchronous execution (per-op streams, events, avoiding
the sync) is explicitly **future work, out of scope for v1** — do not attempt it.

**Background: the CUDA error model.**
Every CUDA runtime API call (`cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`, …)
returns a `cudaError_t` status code; `cudaSuccess` (0) means OK, anything else is an
error whose human-readable text `cudaGetErrorString(err)` returns. Kernel launches
are different: the `<<<>>>` syntax returns nothing. A launch-configuration error
(e.g. too many threads) is retrieved by calling `cudaGetLastError()` immediately
after the launch, and an error *inside* the kernel (e.g. an out-of-bounds access) only
surfaces later, from the next synchronizing call — which, under the v1
sync-after-every-launch policy, is our own `cudaDeviceSynchronize()`. Unchecked
errors are *sticky*: they sit in a per-thread slot and get mis-attributed to a later,
innocent call. Therefore the iron rule of this phase: **every CUDA API call is
wrapped in `MTORCH_CUDA_CHECK(...)`, and every kernel launch is immediately followed
by `MTORCH_CUDA_LAUNCH_CHECK()`** (both defined in §0.3 and created in step 7-2).
No naked CUDA calls anywhere, ever.

### 0.3 The two checking macros (authoritative text; created verbatim in 7-2)

`MTORCH_CUDA_CHECK(expr)` evaluates a CUDA runtime call and throws a C++
`std::runtime_error` (which pybind11 converts into a Python `RuntimeError`) if it did
not return `cudaSuccess`:

```cpp
#define MTORCH_CUDA_CHECK(expr)                                                   \
  do {                                                                            \
    cudaError_t mtorch_cuda_err__ = (expr);                                       \
    if (mtorch_cuda_err__ != cudaSuccess) {                                       \
      throw std::runtime_error(std::string("CUDA error: ") +                      \
                               cudaGetErrorString(mtorch_cuda_err__) +            \
                               " at " __FILE__ ":" + std::to_string(__LINE__) +   \
                               " in `" #expr "`");                                \
    }                                                                             \
  } while (0)
```

`MTORCH_CUDA_LAUNCH_CHECK()` is placed on the line immediately after every kernel
launch. It first checks `cudaGetLastError()` (did the launch itself fail — bad grid
size, no kernel image for this GPU, …), then `cudaDeviceSynchronize()` (did the
kernel body fail — illegal memory access, …). Under the v1 synchronous policy the
synchronize is unconditional, in debug and release alike; this is what makes error
messages point at the guilty kernel instead of a random later call:

```cpp
#define MTORCH_CUDA_LAUNCH_CHECK()                                                \
  do {                                                                            \
    MTORCH_CUDA_CHECK(cudaGetLastError());                                        \
    MTORCH_CUDA_CHECK(cudaDeviceSynchronize());                                   \
  } while (0)
```

(When async execution becomes future work, the `cudaDeviceSynchronize()` line is the
one that would become debug-only. Not in v1.)

### 0.4 File layout, naming, and fixed conventions

All new CUDA code lives under `cpp/mtorch/core/cuda/`:

| File | Kind | Compiled by | Created in | Contents |
|---|---|---|---|---|
| `cpp/mtorch/core/cuda/common.h` | header | both | 7-2 | `MTORCH_CUDA_CHECK`, `MTORCH_CUDA_LAUNCH_CHECK` |
| `cpp/mtorch/core/cuda/kernels.h` | header (plain C++, no CUDA types) | host g++ and nvcc | 7-2, grows each family | `DtypeCode` enum + `launch_*` wrapper declarations |
| `cpp/mtorch/core/cuda/dispatch.cuh` | CUDA header | nvcc only | 7-5 | dtype-switch macros, `__half` accumulation helpers |
| `cpp/mtorch/core/cuda/allocator.cpp` | host C++ | g++ | 7-3 | `CudaAllocator` + registration |
| `cpp/mtorch/core/cuda/ops.h` | header | g++ | 7-5, grows | `mtorch::cuda::<op>` declarations (TensorPtr level) |
| `cpp/mtorch/core/cuda/ops.cpp` | host C++ | g++ | 7-5, grows | guards, pointer extraction, calls into `launch_*` |
| `cpp/mtorch/core/cuda/blas.h` / `blas.cpp` | host C++ | g++ | 7-8 | cuBLAS handle singleton + row-major GEMM wrapper |
| `cpp/mtorch/core/cuda/fill_copy.cu` | CUDA | nvcc | 7-5 | fill kernel |
| `cpp/mtorch/core/cuda/binary.cu` | CUDA | nvcc | 7-6 | grid-stride binary elementwise kernel |
| `cpp/mtorch/core/cuda/unary.cu` | CUDA | nvcc | 7-7 | grid-stride unary kernel (32 ops) |
| `cpp/mtorch/core/cuda/reduce.cu` | CUDA | nvcc | 7-9 | two-pass reduction kernels (teaching example) |
| `cpp/mtorch/core/cuda/rowwise.cu` | CUDA | nvcc | 7-10 | softmax / layer_norm / group_norm row-per-block kernels |
| `cpp/mtorch/core/cuda/im2col.cu` | CUDA | nvcc | 7-11 | im2col kernel + bias-add kernel |

Fixed conventions used by every code listing below (do not vary them):

- Namespace: everything CUDA-side is `namespace mtorch { namespace cuda { ... } }`,
  written `mtorch::cuda` in prose.
- Threads per block: the constant 256, spelled `kThreads` where a constant is
  declared. Block count for grid-stride kernels:
  `(int)std::min<int64_t>((n + kThreads - 1) / kThreads, (int64_t)4096)`, and never
  launch with 0 blocks (use `if (n == 0) return;` before every launch).
- The host↔kernel boundary is **raw pointers + sizes only**. `.cu` files never
  include `tensor.h` (they are compiled as C++17 by nvcc; the core is C++20). The
  translation from `Tensor` to raw pointers happens exclusively in `ops.cpp`.
- Dtype codes crossing that boundary use the `DtypeCode` enum in `kernels.h`, whose
  numeric values **must equal** the declaration order of `enum class ScalarType` in
  `cpp/mtorch/core/tensor.h` (`Float16, Float32, Float64, Int32, Int64, Bool` → 0…5).
  Verification command (run whenever in doubt; it must print the six members in that
  order): `grep -n -A 8 'enum class ScalarType' cpp/mtorch/core/tensor.h`.
- The macro `MTORCH_WITH_CUDA` is defined (by setup.py, step 7-2) for **host** g++
  compilations only when CUDA was detected at build time. Every reference to
  `mtorch::cuda::*` from core files sits inside `#if defined(MTORCH_WITH_CUDA)`.
  On a machine without CUDA the compiled bytes are unchanged.

### 0.5 The dispatch-hook convention (fixed by Phase 5; how to insert hooks)

Phase 5 fixed the convention: at the **top of a public op** in the core section files,
tensors on a non-CPU device are routed to that device's implementation if one exists,
else the op raises through the Phase 5 helper `unsupported_device(op, DeviceType)`.
There is **no silent CPU fallback** — CUDA memory is not host-accessible, so any
"fallback" would read garbage; raising is the only correct behavior.

The CUDA hook inserted by steps 7-5 … 7-12 always has this exact shape (shown here
for `matmul`; each step gives its own concrete block):

```cpp
  if (left->device.type == DeviceType::CUDA || right->device.type == DeviceType::CUDA) {
#if defined(MTORCH_WITH_CUDA)
    return ::mtorch::cuda::matmul(left, right);
#else
    unsupported_device("matmul", DeviceType::CUDA);
#endif
  }
```

Placement rule (mechanical, no judgment):

1. Locate the function with the step's grep command (always given).
2. Run `grep -n 'DeviceType::Metal' <file>` limited to that function's first 30 lines
   (`sed -n '<start>,<start+30>p' <file>`). If a Metal hook block exists there, insert
   the CUDA block **immediately after the closing `}` of the Metal hook block**.
   If none exists, insert **immediately after the function's opening `{` line**.
3. The exact spelling of the raise: before writing your first hook (step 7-5), run

   ```bash
   grep -rn "unsupported_device" cpp/mtorch/core --include='*.cpp' | head -3
   grep -rn "unsupported_device" cpp/mtorch/core --include='*.h' | head -3
   ```

   Copy the call form the existing (Phase 5/6) call sites use — same namespace
   qualification, same argument style — and the `#include` line of the header that
   declares it (add that include to the section file if it is not already there).
   Use that exact form in every hook of this phase. If `unsupported_device` is
   declared `[[noreturn]]`, the `#else` branch above needs no `return`; if the
   existing call sites write `return unsupported_device(...)` or
   `throw ...`, mirror them character for character.
4. Also add, next to the other includes at the top of the section file (only once per
   file, the first time that file gets a hook):

   ```cpp
   #if defined(MTORCH_WITH_CUDA)
   #include "mtorch/core/cuda/ops.h"
   #endif
   ```

Which core section file owns which hook (post-Phase-2b names; re-verify each with the
grep shown in the step itself):

| Family | Public symbol(s) | File |
|---|---|---|
| fill/copy | `Tensor::fill_inplace`, `Tensor::copy_from` | `cpp/mtorch/core/tensor_core.cpp` |
| binary elementwise | `binary_tensor_tensor`, `binary_tensor_scalar`, `binary_scalar_tensor` | `cpp/mtorch/core/elementwise_ops.cpp` |
| unary | `unary` | `cpp/mtorch/core/elementwise_ops.cpp` |
| matmul | `matmul` | `cpp/mtorch/core/linalg.cpp` |
| reductions | `reduce_sum`, `reduce_mean`, `reduce_max` | `cpp/mtorch/core/reductions.cpp` |
| normalization | `softmax`, `layer_norm`, `group_norm` | `cpp/mtorch/core/normalization.cpp` |
| conv | `conv2d` | `cpp/mtorch/core/conv.cpp` |
| attention | `scaled_dot_product_attention` | `cpp/mtorch/core/attention.cpp` |

If a grep for a symbol finds nothing in the named file, follow
`01-rules-and-verification.md` §9 (the code moved; find it with
`grep -rn '<symbol>' cpp/mtorch/core/`; if still absent, BLOCK — never guess).

### 0.6 The v1 guards (fixed rules; identical in spirit to Phase 6)

Every `mtorch::cuda::<op>` function begins with guards. A guard that fails raises
through `unsupported_device` (same spelling as the hooks, §0.5 item 3). The guards
are **temporary v1 restrictions mirrored from the Metal backend (Phase 6)** — they
keep the kernels simple (contiguous, coalesced, no index arithmetic) and are meant to
be lifted in a later phase, never silently worked around now. The fixed list:

- G1 **Grad**: if any input has `requires_grad == true` and `is_grad_enabled()` is
  true → unsupported. (Autograd graphs record CPU lambdas; running them against CUDA
  storage would touch device memory from host code.)
- G2 **Contiguity**: every tensor input must satisfy `is_contiguous()` → else
  unsupported. (Coalescing, §0.2.)
- G3 **No broadcasting / same shape**: where an op takes two tensors, their `sizes`
  must be identical (elementwise family) or match the op's exact v1 shape contract
  (matmul: rank-2 × rank-2; conv2d/sdpa: the shapes listed in their steps).
- G4 **Same device**: all tensor inputs on CUDA; mixed cpu/cuda inputs →
  unsupported. (Transfers are explicit — the user calls `.to()`/`.cuda()`.)
- G5 **Dtype allowlist**: each family's step lists the exact allowed `ScalarType`s;
  anything else → unsupported.
- G6 **Op-specific**: extra per-op restrictions (e.g. softmax only over the last
  dim, conv2d only `groups == 1`) — listed in each step.

Each family's dtype allowlist (G5), fixed here once:

| Family | float32 | float16 | float64 | int32 | int64 | bool |
|---|---|---|---|---|---|---|
| fill / copy | yes | yes | yes | yes | yes | yes |
| binary add/sub/mul | yes | yes | yes | yes | yes | no |
| binary div | yes | yes | yes | no | no | no |
| unary (32 ops) | yes | yes | yes | no | no | no |
| matmul | yes | yes | no | no | no | no |
| reductions sum/mean | yes | yes | yes | sum only | sum only | no |
| reductions max | yes | yes | yes | yes | yes | no |
| softmax / layer_norm / group_norm | yes | yes | no | no | no | no |
| conv2d | yes | yes | no | no | no | no |
| scaled_dot_product_attention | yes | yes | no | no | no | no |

### 0.7 Parity rows, the oracle script, and what "green" means here

Three layers of testing, all mandatory:

1. **The existing CPU suite** (`pytest tests/compat`) — proves the CPU behavior did
   not change. Compared against the **Linux** baseline (§0.1) after every step.
2. **The device-parity suite** `tests/compat/test_device_parity.py` (created by
   Phase 5): a device-parametrized op table; for each row it computes the op on the
   device, copies the result back with an explicit transfer, and compares against
   the same op computed on cpu. Run with
   `pytest tests/compat/test_device_parity.py -q`. Steps 7-5 … 7-12 **append rows**
   to its case table. Appending rows is explicitly authorized by this guide (it is
   the same kind of exception `01-rules-and-verification.md` Iron Rule 1 grants
   Phase 4: additions only — never edit or remove an existing row, tolerance, or
   comparison).
   Mechanical procedure to append a row: `sed -n '1,80p' tests/compat/test_device_parity.py`
   to see the table's row constructor and field names, then append rows at the **end**
   of the table using that exact constructor, filling in the fields from the step's
   "Parity rows" table (case id, op, args, dtype, and — only if the constructor has
   tolerance fields — the rtol/atol given). Do not reformat existing rows.
3. **The oracle script** `tests/compat/tools/cuda_oracle_check.py` (created complete
   in 7-4): compares mtorch-on-CUDA against **torch-on-CUDA** directly. This is
   needed because the pytest harness cannot express "reference runs on cuda" today:
   the only harness options are `--compat-reference`, `--compat-candidate`,
   `--compat-api-manifest`, `--compat-op`, `--compat-allow-reference-backed`, and the
   five `--compat-benchmark-*` options (verify:
   `grep -n 'compat-' tests/conftest.py` — there is no `--compat-device`). So
   reference-on-cuda is **not expressible** through the harness; the parity suite
   (device vs cpu) plus this hand-written oracle (mtorch-cuda vs torch-cuda) together
   cover it. Run with `python3 tests/compat/tools/cuda_oracle_check.py` — expected
   final line `ORACLE OK (<n> checks)`.

**Benchmark note (applies to every family step):** correctness is the gate; speed is
not. v1 kernels being slower than torch-CUDA is expected and acceptable. The oracle
script's `--bench` mode records per-op time ratios (mtorch-cuda / torch-cuda); step
7-13 writes them to `docs/design/baseline/cuda-ratios.md`, and each family step's
Verification tells you to paste one observed ratio into the Phase 7 `Notes:` field of
PROGRESS.md (one line per family, e.g. `cuda binary add 1M f32: ratio 3.4x`). A ratio
of 30x is a note, not a failure.

### 0.8 STANDARD VERIFY (Linux) and STANDARD FAIL (referenced by every step)

**STANDARD VERIFY** (run after the build in every step from 7-2 on):

```bash
python3 setup.py build_ext --inplace
pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
tail -1 docs/design/baseline/tests-baseline-linux.txt
pytest tests/compat/test_device_parity.py -q
```

The `--ignore` is deliberate and permanent for this phase: steps 7-5 … 7-12 *append
rows* to `test_device_parity.py`, so its counts legitimately grow — a fixed baseline
comparison over it would go red on every family step. It is therefore excluded from
the frozen-baseline comparison and verified by its own rule instead (the parity-debt
mechanism, 7-3 Action 4 / 7-5 Action 9). The Linux baseline recorded in 7-1 uses the
same `--ignore`, so the two summary lines are directly comparable.

PASS iff: the build exits 0; every count (passed / failed / xfailed / skipped /
error) in the ignored-run summary line equals the corresponding count in the Linux
baseline line; and the parity suite's failing set equals
`docs/design/baseline/parity-debt.txt` exactly (0 failed once the debt is empty).
(From 7-4 on, also `python3 tests/compat/tools/cuda_oracle_check.py` must end with
`ORACLE OK`, i.e. `failed=0`.) Any difference is a failure. The full-authority
procedure remains `01-rules-and-verification.md` §5.1, with the `-linux` baseline
files per §0.1 and the `--ignore` amendment above.

**STANDARD FAIL** (what "On failure" means in every step, after 3 fix attempts or 30
minutes per `01-rules-and-verification.md` §6):

1. Discard the step completely:

   ```bash
   git restore --staged --worktree .
   git status --short
   ```

   For every `??` (untracked) line that the step created, remove it explicitly by
   path, one at a time — never `git clean -fd` without a path:

   ```bash
   git clean -fd -- <path>
   ```

2. Rebuild and re-run STANDARD VERIFY; it must pass (you are back at the previous
   step's green state) before you continue.
3. In `docs/design/PROGRESS.md`, append ` **BLOCKED**` to the step's line (checkbox
   stays `[ ]`) and add ≤3 lines to the Phase 7 `Notes:` field (what you changed,
   the first error message, what you tried). Commit only PROGRESS.md:
   `git add docs/design/PROGRESS.md && git commit -m "progress: mark <step-id> BLOCKED"`.
4. Stop. A BLOCKED step waits for a human.

### 0.9 Numerics rules (fixed; violations are bugs)

- **N1 — fp16 accumulates in fp32.** No arithmetic is ever performed in `__half`.
  Kernels load `__half`, convert to `float` (`__half2float`), compute in `float`,
  convert back on store (`__float2half`). The dispatch header (7-5) bakes this in as
  `acc_t`. cuBLAS fp16 GEMMs use `CUBLAS_COMPUTE_32F` (fp32 accumulation inside the
  GEMM) for the same reason. Rationale: fp16 has a 10-bit mantissa; accumulating a
  1024-element dot product in fp16 loses ~all precision and parity fails.
- **N2 — no atomics, hard rule.** v1 kernels must not use `atomicAdd` or any other
  atomic. Floating-point atomics make summation order run-dependent, so results
  become non-deterministic run to run, and a parity/oracle failure could never be
  reproduced. The two-pass reduction in 7-9 exists precisely to avoid the atomic
  one-pass shortcut. If you ever "need" an atomic in this phase, the design is
  wrong — stop and re-read the step.
- **N3 — TF32 is disabled everywhere.** On Ampere+ GPUs cuBLAS may silently execute
  fp32 GEMMs in TF32 (a 10-bit-mantissa format), which perturbs results at the ~1e-3
  level — far outside our fp32 tolerances, so parity would fail mysteriously. Two
  independent switches, both mandatory: (a) mtorch side — the cuBLAS handle
  singleton (7-8) calls `cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)` right
  after creation, and every `cublasGemmEx` call passes `CUBLAS_COMPUTE_32F`;
  (b) torch side — the oracle script sets
  `torch.backends.cuda.matmul.allow_tf32 = False` and
  `torch.backends.cudnn.allow_tf32 = False` before any computation.
- **N4 — tolerances.** The harness's own numbers, for calibration: case rows in
  `tests/compat/cases.py` use `rtol=1e-6, atol=1e-6` for float32 math ops and
  `rtol=1e-12, atol=1e-12` for float64 (verify:
  `grep -n 'rtol=1e-6' tests/compat/cases.py | head -3` and
  `grep -n 'rtol=1e-12' tests/compat/cases.py | head -1`); rows without explicit
  values get `torch.testing.assert_close` defaults via
  `_assert_tensor_compatible` in `tests/compat/harness.py` (verify the pass-through:
  `grep -n 'assert_close' tests/compat/harness.py`), i.e. fp32 rtol=1.3e-6/atol=1e-5
  and fp16 rtol=1e-3/atol=1e-5. Cross-implementation GPU comparisons cannot hit
  1e-6 on reduction-shaped ops (different summation orders), so this phase fixes:
  elementwise/unary/fill fp32 → rtol=1e-6, atol=1e-6; matmul/reduction/normalization/
  conv/sdpa fp32 → rtol=1e-4, atol=1e-5; all fp16 → rtol=1e-2, atol=1e-3; float64 →
  rtol=1e-12, atol=1e-12. These exact numbers appear in the oracle script (7-4) and
  in every "Parity rows" table. Never loosen an existing row's tolerance to make it
  pass (Iron Rule 1); if a new row cannot meet these numbers, the kernel is wrong.

---

## Step 7-1: Linux environment gate + Linux baseline

**Goal**: Prove the Linux/CUDA machine is usable — toolchain present, the CPU build
of mtorch compiles and the **full CPU suite passes on Linux** — and record the Linux
test baseline that every later step compares against. If the CPU suite does not pass
here, **the entire phase is BLOCKED at 7-1**: CUDA work must never start on top of a
broken CPU tree, because every later verification would be uninterpretable.

**Preconditions**: You are on the Linux machine, in the repo root of a fresh clone of
`https://github.com/kazhiramatsu/mtorch` at the post-Phase-6 (or post-Phase-5, if
Phase 6 is deferred) commit. Exact check — every command must succeed and print what
the comment says:

```bash
git log --oneline -1                                   # prints the post-Phase-5/6 commit
git status --short                                     # prints nothing (clean tree)
uname -sm                                              # prints: Linux x86_64
nvidia-smi --query-gpu=name,compute_cap --format=csv   # prints the GPU name and e.g. 8.6
nvcc --version | tail -1                               # prints a CUDA 12.x release line
g++ --version | head -1                                # g++ 11 or newer
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # e.g. 2.x.x+cu121 True
python3 -c "import pytest, numpy; print('deps ok')"    # prints: deps ok
```

If `nvcc` is not on `PATH`, set `CUDA_HOME` to the toolkit root (the directory that
contains `bin/nvcc`, typically `/usr/local/cuda`) and export it in every session:
`export CUDA_HOME=/usr/local/cuda`. If `torch.cuda.is_available()` prints `False`,
stop — the oracle does not work; mark 7-1 **BLOCKED** (§0.8 item 3) with the output
of `nvidia-smi` in the note.

**Background: why a separate Linux baseline?** The committed
`docs/design/baseline/tests-baseline.txt` was recorded on macOS, where the C++ core
compiles with `-DMTORCH_USE_ACCELERATE` (Apple's BLAS) and takes `__ARM_NEON` SIMD
paths. On Linux/x86_64 neither macro is defined, so the *portable fallback* code
paths compile and run instead — a genuinely different binary. Timing lines and the
platform header of the pytest output differ too. The pass/fail breakdown is the only
thing we compare, and it must be established once, here, on this machine.

**The complete list of macOS-only assumptions in the tree, and their handling**
(counts verified on the 2026-07-09 pre-split tree; after the Phase 2b split the same
guards are distributed across `cpp/mtorch/core/*.cpp` and `cpp/mtorch/core/detail/*` —
re-count with the commands shown, totals must be in the same ballpark, exact numbers
do not matter):

| # | Assumption | Where / how to verify today | Handling on Linux |
|---|---|---|---|
| 1 | `setup.py` adds `-DMTORCH_USE_ACCELERATE` and `-framework Accelerate` only under `if sys.platform == "darwin":` | `grep -n 'darwin' setup.py` (1 hit) | None. On Linux the branch is not taken; no Accelerate flags, no framework link. Do NOT try to install an Accelerate substitute. |
| 2 | ~89 `#if defined(MTORCH_USE_ACCELERATE)` blocks in the C++ core (85 in the pre-split tensor.cpp + 4 in module.cpp) | `grep -rc 'MTORCH_USE_ACCELERATE' cpp/ \| grep -v ':0'` | None in code. The macro is undefined on Linux, so the `#else`/fallback (portable scalar/loop) paths compile. The CPU code *claims* every Accelerate fast path has a portable fallback — the claim is **verified, not assumed**, by Actions 3–4 below: the build must link and the full suite must pass. |
| 3 | ~89 `#if defined(__ARM_NEON)` blocks (pre-split tensor.cpp count) | `grep -rc '__ARM_NEON' cpp/ \| grep -v ':0'` | None in code. x86_64 g++ never defines `__ARM_NEON`; portable paths compile. Same verification as row 2. |
| 4 | Test expectations | `grep -rn 'darwin\|Darwin\|macos\|Accelerate' tests/ compat/` | Must print nothing (verified: it prints nothing on the 2026-07-09 tree). If a Phase 5/6 step introduced a hit, read it: if it is a device-availability guard (e.g. skips metal when `mtorch.backends.metal` is unavailable) it is fine; anything that hard-codes a darwin expectation → BLOCKED, note the file:line. |
| 5 | `-std=c++20` host flag | `grep -n 'c++20' setup.py` | Needs g++ ≥ 11 (precondition above). No change. |
| 6 | Metal sources from Phase 6 (`cpp/mtorch/core/metal/…`, Objective-C++) | `ls cpp/mtorch/core/metal/ 2>/dev/null` | Phase 6 is contract-bound to compile them only on darwin. If the build in Action 2 fails on a Metal file, that is a Phase 6 defect: BLOCKED, note the first error line. Do not patch Metal code. |
| 7 | Build artifacts from the Mac (`*.so`, `build/`) | `ls mtorch/*.so 2>/dev/null` | A fresh clone contains none (they are gitignored). Never copy `.so`/`build/` between machines. |

**Actions**:

1. Append the Phase 7 section to `docs/design/PROGRESS.md` if it is not already
   there (`grep -n 'Phase 7' docs/design/PROGRESS.md`; if the grep prints a line,
   skip this action). Add at the end of the file, before the "Overall completion
   check" section:

   ```
   ## Phase 7: CUDA backend (procedure: 10-phase7-cuda-backend.md)

   - [ ] 7-1 Linux environment gate + Linux baseline — commit: / date:
   - [ ] 7-2 CUDA build integration (nvcc + setup.py) — commit: / date:
   - [ ] 7-3 CudaAllocator + registry hookup — commit: / date:
   - [ ] 7-4 Oracle tool tests/compat/tools/cuda_oracle_check.py — commit: / date:
   - [ ] 7-5 Op family: fill/copy — commit: / date:
   - [ ] 7-6 Op family: binary elementwise add/sub/mul/div — commit: / date:
   - [ ] 7-7 Op family: unary (32 ops) — commit: / date:
   - [ ] 7-8 Op family: matmul via cuBLAS — commit: / date:
   - [ ] 7-9 Op family: reductions sum/mean/max — commit: / date:
   - [ ] 7-10 Op family: softmax/layer_norm/group_norm — commit: / date:
   - [ ] 7-11 Op family: conv2d (im2col + cuBLAS) — commit: / date:
   - [ ] 7-12 Op family: scaled_dot_product_attention — commit: / date:
   - [ ] 7-13 CUDA ratio table + Linux phase gate — commit: / date:
   - [ ] 7-14 macOS regression check — commit: / date:

   Notes:
   ```

2. Build the CPU tree on Linux (this is the first compile of the portable fallback
   paths on this platform; expect it to take a few minutes):

   ```bash
   python3 setup.py build_ext --inplace
   python3 -c "import mtorch; print(mtorch.__version__)"
   ```

   Both must exit 0; the second prints `0.0.0`.

3. Run the full suite and record the Linux test baseline (mirrors
   `02-phase0-baseline.md` step 0-4 Action 2, with `-linux` file names and one
   deliberate difference: `test_device_parity.py` is excluded, because Phase 7
   appends rows to it and it is gated by its own mechanism — see §0.8. Tens of
   minutes — set a command timeout of at least 60 minutes, never abort):

   ```bash
   pytest tests/compat --ignore=tests/compat/test_device_parity.py 2>&1 | tee docs/design/baseline/tests-baseline-linux.txt
   grep -E "^=+ .*(passed|failed|xfailed|error).* =+$" docs/design/baseline/tests-baseline-linux.txt | tail -1
   ```

4. **THE GATE.** The summary line printed by Action 3 must contain **no `failed` and
   no `error` counts** (only `passed`, and possibly `xfailed`/`skipped`). Mechanical
   check — this must print `GATE OK`:

   ```bash
   tail -1 docs/design/baseline/tests-baseline-linux.txt | grep -qE '(^|[^x])[0-9]+ (failed|error)' && echo GATE FAILED || echo GATE OK
   ```

   `GATE FAILED` means the portable-fallback claim of the CPU code is false on this
   platform (or the environment is broken). Delete the recorded files
   (`rm -f docs/design/baseline/tests-baseline-linux.txt docs/design/baseline/collect-count-linux.txt docs/design/baseline/benchmark-baseline-linux.json`),
   mark 7-1 **BLOCKED** per §0.8 item 3 quoting the first failing test id from the
   pytest output, and stop. **No CUDA work starts while this gate is red.**
   (Note: benchmark cases cannot fail for slowness — `--compat-benchmark-max-ratio`
   defaults to `0.0` = disabled, verify with
   `grep -n 'max-ratio' tests/conftest.py` — so any failure here is a correctness
   failure.)

5. Record the collection count and the Linux benchmark JSON (mirrors step 0-4
   Actions 3–4, with the same `--ignore` as Action 3; the JSON is informational
   for CPU-perf-on-Linux, it gates nothing):

   ```bash
   pytest tests/compat --ignore=tests/compat/test_device_parity.py --collect-only -q 2>&1 | tail -2 > docs/design/baseline/collect-count-linux.txt
   cat docs/design/baseline/collect-count-linux.txt
   pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
       --compat-benchmark-json docs/design/baseline/benchmark-baseline-linux.json
   ```

   Cross-platform sanity check: test collection (excluding the parity file) is
   determined by the committed test files, so the Linux count must be
   reconcilable with the committed macOS `collect-count.txt`:

   ```bash
   grep -oE '[0-9]+ tests collected' docs/design/baseline/collect-count.txt
   grep -oE '[0-9]+ tests collected' docs/design/baseline/collect-count-linux.txt
   pytest tests/compat/test_device_parity.py --collect-only -q | tail -2
   ```

   Rule: if the committed macOS baseline predates the parity suite
   (`grep -c 'test_device_parity' docs/design/baseline/tests-baseline.txt`
   prints `0`), linux_count must equal macos_count exactly; otherwise
   linux_count must equal macos_count minus the parity file's own collected
   count (third command). Any other difference means tests are being skipped or
   lost on Linux → treat as gate failure (Action 4's failure path).

6. Run the parity suite once; the cpu rows must pass, and every cuda row (if Phase 5
   parametrized cuda already) must fail-or-skip **cleanly** via
   `mtorch.cuda.is_available()` being `False` (no allocator is registered yet):

   ```bash
   python3 -c "import mtorch; print(mtorch.cuda.is_available())"   # prints: False
   pytest tests/compat/test_device_parity.py -q | tail -3
   ```

   Expected: `False`, and a summary with 0 failed / 0 error (cuda rows skipped or
   simply not parametrized while unavailable — whatever Phase 5 set up; record the
   summary line in the Phase 7 Notes, you will compare against it in 7-3).

**Verification**: Action 4 printed `GATE OK`; Action 5's three files exist
(`ls docs/design/baseline/*-linux*` lists `tests-baseline-linux.txt`,
`collect-count-linux.txt`, `benchmark-baseline-linux.json`); Action 6 printed `False`
and a 0-failed summary.

**On failure**: STANDARD FAIL (§0.8). The only files this step creates are the three
`-linux` baseline files and the PROGRESS.md edit; remove the baseline files with
`git clean -fd -- docs/design/baseline/tests-baseline-linux.txt`,
`git clean -fd -- docs/design/baseline/collect-count-linux.txt`,
`git clean -fd -- docs/design/baseline/benchmark-baseline-linux.json` as needed.
Mark 7-1 **BLOCKED** in `docs/design/PROGRESS.md` and stop.

**Commit**:

```bash
git add docs/design/baseline/tests-baseline-linux.txt \
        docs/design/baseline/collect-count-linux.txt \
        docs/design/baseline/benchmark-baseline-linux.json \
        docs/design/PROGRESS.md
git commit -m "refactor(phase7-1): record Linux CPU baseline (environment gate)"
```

Then update PROGRESS.md per `01-rules-and-verification.md` §8 (separate commit).

---

## Step 7-2: CUDA build integration (nvcc + setup.py)

**Goal**: Teach `setup.py` to detect CUDA, compile `cpp/mtorch/core/cuda/*.cu` with
`nvcc`, compile `cpp/mtorch/core/cuda/*.cpp` with the host compiler, define
`MTORCH_WITH_CUDA` for host code, and link `libcudart` + `libcublas` — **only when
CUDA is detected**. On a machine without CUDA (the Mac, CI without GPUs) the build
must be byte-identical to before: same sources, same flags, no new defines. Also
create `cpp/mtorch/core/cuda/common.h` (the checking macros of §0.3), the empty-shell
`kernels.h`, and a tiny probe kernel that proves the whole pipeline works.

**Background: what nvcc, `-gencode`, and "compute capability" are.**
`nvcc` is NVIDIA's compiler driver: it splits a `.cu` file into host code (which it
hands to g++) and device code (which it compiles to GPU machine code). GPU machine
code is architecture-specific; each GPU generation has a *compute capability* (CC)
number — Ampere is 8.0/8.6, Ada is 8.9, Hopper is 9.0. A flag pair
`-gencode arch=compute_86,code=sm_86` means "compile device code for CC 8.6 and embed
the native binary (`sm_86`) in the executable". We embed binaries for 8.0, 8.6, 8.9
and 9.0, which covers all common data-center and desktop GPUs since 2020. If the
binary contains no image for the running GPU, every kernel launch fails with
`no kernel image is available for execution on the device` — that is what the CC
check below prevents. Check your GPU's CC with:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

If the printed value is not one of `8.0`, `8.6`, `8.9`, `9.0`: take the printed value,
drop the dot (e.g. `7.5` → `75`), and add one more line
`"-gencode", "arch=compute_75,code=sm_75",` to the `NVCC_FLAGS` list below, right
after the other `-gencode` lines. That is the only permitted deviation. (CC below
7.0 is out of scope — BLOCKED, note the GPU name.)

Meaning of the other nvcc flags used below: `-O3` optimizes; `-std=c++17` sets the
C++ dialect for `.cu` files (our `.cu` files are deliberately tensor.h-free so C++17
suffices, §0.4); `--expt-relaxed-constexpr` lets device code call `constexpr` host
functions (needed by `<cuda_fp16.h>` arithmetic helpers); `-Xcompiler -fPIC` forwards
`-fPIC` to the host compiler half of nvcc's output so the objects can be linked into
a Python shared library.

**Preconditions**: 7-1 committed and `git status --short` prints nothing. Check:

```bash
git log --oneline -3        # top commits are the 7-1 pair
grep -n 'glob.glob' setup.py   # prints the Phase 2a source-glob line(s)
nvcc --version | tail -1    # CUDA 12.x
```

**Actions**:

1. Create the directory and `cpp/mtorch/core/cuda/common.h` with exactly this
   content:

   ```cpp
   #pragma once

   // Phase 7 (10-phase7-cuda-backend.md). Error-checking macros for all CUDA code.
   // Include this from every cuda/*.cpp and cuda/*.cu file. See guide sections
   // 0.2 (error model) and 0.3 (authoritative macro text).

   #include <cuda_runtime.h>

   #include <stdexcept>
   #include <string>

   #define MTORCH_CUDA_CHECK(expr)                                                   \
     do {                                                                            \
       cudaError_t mtorch_cuda_err__ = (expr);                                       \
       if (mtorch_cuda_err__ != cudaSuccess) {                                       \
         throw std::runtime_error(std::string("CUDA error: ") +                      \
                                  cudaGetErrorString(mtorch_cuda_err__) +            \
                                  " at " __FILE__ ":" + std::to_string(__LINE__) +   \
                                  " in `" #expr "`");                                \
       }                                                                             \
     } while (0)

   #define MTORCH_CUDA_LAUNCH_CHECK()                                                \
     do {                                                                            \
       MTORCH_CUDA_CHECK(cudaGetLastError());                                        \
       MTORCH_CUDA_CHECK(cudaDeviceSynchronize());                                   \
     } while (0)
   ```

2. Create `cpp/mtorch/core/cuda/kernels.h` with exactly this content (the dtype
   codes mirror `enum class ScalarType` order — verify per §0.4 — and the file grows
   by appended declarations in later steps):

   ```cpp
   #pragma once

   // Phase 7. Host<->kernel boundary: plain C++ declarations of the nvcc-compiled
   // launch wrappers, raw pointers + sizes only. NEVER include tensor.h or any
   // CUDA header here: this header is included by host C++20 code and by .cu code.

   #include <cstdint>

   namespace mtorch {
   namespace cuda {

   // Must match the declaration order of mtorch::ScalarType (tensor.h):
   //   Float16, Float32, Float64, Int32, Int64, Bool
   enum DtypeCode : int {
     kF16 = 0,
     kF32 = 1,
     kF64 = 2,
     kI32 = 3,
     kI64 = 4,
     kBool = 5,
   };

   // 7-2 probe (removed by no one; harmless):
   // writes value into out[0..n) as float. Proves nvcc compile+link+launch works.
   void launch_probe_fill_float(float* out_device, int64_t n, float value);

   }  // namespace cuda
   }  // namespace mtorch
   ```

3. Create `cpp/mtorch/core/cuda/probe.cu` with exactly this content:

   ```cpp
   // Phase 7 step 7-2. Smallest possible kernel; exists to prove the build works
   // end to end (nvcc compile -> link into mtorch._C -> launch on the GPU).

   #include "mtorch/core/cuda/common.h"
   #include "mtorch/core/cuda/kernels.h"

   namespace mtorch {
   namespace cuda {

   namespace {

   // __global__ marks a kernel: code that runs on the GPU, launched from the host.
   __global__ void probe_fill_float_kernel(float* out, int64_t n, float value) {
     // Standard grid-stride loop (guide section 0.2).
     for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
          i += (int64_t)gridDim.x * blockDim.x) {
       out[i] = value;
     }
   }

   }  // namespace

   void launch_probe_fill_float(float* out_device, int64_t n, float value) {
     if (n == 0) return;
     const int kThreads = 256;
     const int blocks =
         (int)((n + kThreads - 1) / kThreads > 4096 ? 4096 : (n + kThreads - 1) / kThreads);
     probe_fill_float_kernel<<<blocks, kThreads>>>(out_device, n, value);
     MTORCH_CUDA_LAUNCH_CHECK();
   }

   }  // namespace cuda
   }  // namespace mtorch
   ```

4. Edit `setup.py`. Because Phase 6 may have restructured it, this edit is specified
   as three mechanical insertions with grep anchors, not a whole-file replacement.
   First print the current file (it is short): `cat setup.py`.

   4a. **Detection block.** Immediately after the last `import` line at the top of
   the file (anchor: `grep -n '^from setuptools' setup.py`), insert:

   ```python
   # ---- Phase 7: CUDA detection (10-phase7-cuda-backend.md step 7-2) ----------
   # CUDA participates in the build ONLY when detected here. On machines without
   # CUDA every value computed below is empty/False and the build is unchanged.
   import glob as _glob
   import os as _os
   import shutil as _shutil
   import subprocess as _subprocess


   def _find_cuda_home() -> str | None:
       cuda_home = _os.environ.get("CUDA_HOME") or _os.environ.get("CUDA_PATH")
       if cuda_home and _os.path.isdir(cuda_home):
           return cuda_home
       nvcc = _shutil.which("nvcc")
       if nvcc:
           # .../bin/nvcc -> toolkit root
           return _os.path.dirname(_os.path.dirname(nvcc))
       return None


   MTORCH_CUDA_HOME = _find_cuda_home()
   MTORCH_WITH_CUDA = MTORCH_CUDA_HOME is not None

   MTORCH_NVCC_FLAGS = [
       "-O3",
       "-std=c++17",
       "--expt-relaxed-constexpr",
       "-Xcompiler",
       "-fPIC",
       "-gencode", "arch=compute_80,code=sm_80",
       "-gencode", "arch=compute_86,code=sm_86",
       "-gencode", "arch=compute_89,code=sm_89",
       "-gencode", "arch=compute_90,code=sm_90",
   ]

   MTORCH_CUDA_CU_SOURCES = (
       sorted(_glob.glob("cpp/mtorch/core/cuda/*.cu")) if MTORCH_WITH_CUDA else []
   )
   # -----------------------------------------------------------------------------
   ```

   (If `import glob` already exists in setup.py from Phase 2a, the `_glob` alias
   import above is still added as written — aliased imports cannot collide.)

   4b. **Extension arguments.** Locate the `Extension(` call
   (`grep -n 'Extension(' setup.py`) and the lists feeding it
   (`extra_compile_args`, `extra_link_args`, the `sources=` expression,
   `include_dirs`). Immediately **before** the `setup(` line
   (anchor: `grep -n '^setup(' setup.py`), insert:

   ```python
   # ---- Phase 7: CUDA build arguments (only when detected) ---------------------
   mtorch_cuda_host_sources: list[str] = []
   mtorch_cuda_libraries: list[str] = []
   mtorch_cuda_library_dirs: list[str] = []
   mtorch_cuda_include_dirs: list[str] = []
   if MTORCH_WITH_CUDA:
       extra_compile_args.append("-DMTORCH_WITH_CUDA")
       mtorch_cuda_host_sources = sorted(_glob.glob("cpp/mtorch/core/cuda/*.cpp"))
       mtorch_cuda_libraries = ["cudart", "cublas"]
       mtorch_cuda_library_dirs = [_os.path.join(MTORCH_CUDA_HOME, "lib64")]
       mtorch_cuda_include_dirs = [_os.path.join(MTORCH_CUDA_HOME, "include")]
       extra_link_args.append("-Wl,-rpath," + _os.path.join(MTORCH_CUDA_HOME, "lib64"))
   # -----------------------------------------------------------------------------
   ```

   Then edit the `Extension(...)` call itself:

   - `sources=` — BEFORE (the Phase 2a form; your exact expression may differ, keep
     it and only append the new term):

     ```python
     sources=sorted(glob.glob("cpp/mtorch/core/*.cpp") + glob.glob("cpp/mtorch/core/detail/*.cpp") + glob.glob("cpp/mtorch/python/*.cpp")),
     ```

     AFTER:

     ```python
     sources=sorted(glob.glob("cpp/mtorch/core/*.cpp") + glob.glob("cpp/mtorch/core/detail/*.cpp") + glob.glob("cpp/mtorch/python/*.cpp")) + mtorch_cuda_host_sources,
     ```

     (Note `cpp/mtorch/core/cuda/*.cpp` is NOT in the Phase 2a glob — the `cuda/`
     subdirectory is picked up only via `mtorch_cuda_host_sources`, i.e. only when
     CUDA is detected. If Phase 6 changed the sources expression, apply the same
     rule: append `+ mtorch_cuda_host_sources` to whatever expression is there.)

   - `include_dirs=["cpp"],` → `include_dirs=["cpp"] + mtorch_cuda_include_dirs,`
   - Add these two keyword arguments to the `Extension(` call if absent (if Phase 6
     already added `libraries=`/`library_dirs=` keywords, append `+ mtorch_cuda_...`
     to their values instead):

     ```python
     libraries=mtorch_cuda_libraries,
     library_dirs=mtorch_cuda_library_dirs,
     ```

   4c. **The custom `build_ext` (the fiddly part — complete, use verbatim).**
   setuptools does not know what a `.cu` file is, so we compile them ourselves with
   nvcc into object files and hand those objects to the linker via
   `ext.extra_objects`. Determine the base class first:
   `grep -n 'cmdclass' setup.py`.

   - **Case A — no match** (Phase 6 added no cmdclass): insert the block below
     immediately before the `setup(` line, and add `cmdclass={"build_ext": MtorchCudaBuildExt},`
     as a new argument of `setup(` (anywhere among its keyword arguments).
   - **Case B — a match exists**, e.g. `cmdclass={"build_ext": MtorchMetalBuildExt}`:
     insert the same block, but replace the base-class import line as instructed in
     the block's first comment, and change the *value* in the existing `cmdclass=`
     dict to `MtorchCudaBuildExt` (the CUDA class then inherits the Metal behavior).

   ```python
   # ---- Phase 7: compile .cu files with nvcc, feed objects to the linker --------
   # Case A: no pre-existing cmdclass -> use setuptools' build_ext as the base.
   # Case B: a cmdclass exists (Phase 6) -> DELETE the import line below and set
   #         `_MtorchBuildExtBase = <the existing class name>` instead, so Metal
   #         behavior is inherited.
   from setuptools.command.build_ext import build_ext as _MtorchBuildExtBase


   class MtorchCudaBuildExt(_MtorchBuildExtBase):
       """build_ext that additionally compiles cpp/mtorch/core/cuda/*.cu with nvcc.

       Why this shape: setuptools' compiler object only understands host C/C++.
       The robust, portable trick is to pre-compile every .cu into a .o in
       build_temp using nvcc directly, then append those .o paths to
       ext.extra_objects, which distutils passes verbatim to the final link line.
       Host sources, flags and link behavior are untouched, so when
       MTORCH_WITH_CUDA is False this class is exactly its base class.
       """

       def build_extensions(self):
           if MTORCH_WITH_CUDA and MTORCH_CUDA_CU_SOURCES:
               nvcc = _os.path.join(MTORCH_CUDA_HOME, "bin", "nvcc")
               cuda_temp = _os.path.join(self.build_temp, "mtorch_cuda")
               _os.makedirs(cuda_temp, exist_ok=True)
               cuda_objects = []
               for cu_source in MTORCH_CUDA_CU_SOURCES:
                   object_path = _os.path.join(
                       cuda_temp,
                       _os.path.splitext(_os.path.basename(cu_source))[0] + ".o",
                   )
                   command = (
                       [nvcc, "-c", cu_source, "-o", object_path, "-Icpp",
                        "-I" + _os.path.join(MTORCH_CUDA_HOME, "include")]
                       + MTORCH_NVCC_FLAGS
                   )
                   # Always recompile: .cu count is small, correctness beats
                   # incremental-build cleverness (stale-object bugs are brutal).
                   print("nvcc:", " ".join(command))
                   _subprocess.check_call(command)
                   cuda_objects.append(object_path)
               for ext in self.extensions:
                   ext.extra_objects = list(getattr(ext, "extra_objects", []) or []) + cuda_objects
           super().build_extensions()
   # -----------------------------------------------------------------------------
   ```

5. Build and smoke-test the probe end to end:

   ```bash
   python3 setup.py build_ext --inplace 2>&1 | tee /tmp/build-7-2.log
   grep -c '^nvcc:' /tmp/build-7-2.log        # prints 1 (probe.cu compiled by nvcc)
   grep -o '\-DMTORCH_WITH_CUDA' /tmp/build-7-2.log | head -1   # prints -DMTORCH_WITH_CUDA
   python3 - <<'PY'
   import ctypes, mtorch
   # The .so must now depend on libcudart/libcublas; importing proves the link worked.
   print("import ok")
   PY
   ldd mtorch/_C.cpython-*.so | grep -E 'cudart|cublas'   # prints both libraries
   ```

**Verification**: STANDARD VERIFY (§0.8) — the full suite still matches the Linux
baseline (no behavior changed; the probe is dead code until 7-3) and the parity
suite summary equals the one recorded in 7-1 Notes. Additionally, the
no-CUDA-machine guarantee, checked *on this machine* by simulating absence of CUDA
(nvcc hidden ⇒ detection fails ⇒ build must not mention nvcc or MTORCH_WITH_CUDA):

```bash
rm -rf build mtorch/_C*.so
env -u CUDA_HOME -u CUDA_PATH PATH=/usr/bin:/bin python3 setup.py build_ext --inplace 2>&1 | tee /tmp/build-7-2-nocuda.log
grep -c 'nvcc\|MTORCH_WITH_CUDA' /tmp/build-7-2-nocuda.log   # prints 0
python3 -c "import mtorch; print(mtorch.cuda.is_available())" # prints False
# then restore the CUDA build:
python3 setup.py build_ext --inplace
```

(If `/usr/bin:/bin` does not contain your python3/g++, prepend their directories to
that PATH value — but never a directory containing nvcc.)

**On failure**: STANDARD FAIL (§0.8). New paths for `git clean -fd --`:
`cpp/mtorch/core/cuda/`. Mark 7-2 **BLOCKED**.

**Commit**:

```bash
git add setup.py cpp/mtorch/core/cuda/common.h cpp/mtorch/core/cuda/kernels.h cpp/mtorch/core/cuda/probe.cu
git commit -m "refactor(phase7-2): CUDA build integration (nvcc + custom build_ext)"
```

Then update PROGRESS.md per §8 (separate commit).

---

## Step 7-3: CudaAllocator + registry hookup

**Goal**: Implement the CUDA memory allocator behind the Phase 5 allocator
interface, register it for `DeviceType::CUDA` at import time when a CUDA device is
present, and prove that explicit transfers (`.cuda()` / `.cpu()`) round-trip data
bit-exactly. After this step `mtorch.cuda.is_available()` returns `True` on the
Linux box.

**Background: why a caching allocator (and why `cudaMalloc` hurts).**
`cudaMalloc` asks the GPU driver for memory. Unlike host `malloc`, it implicitly
synchronizes with the device and can take on the order of *milliseconds* — often
longer than the kernel that will use the buffer. `cudaFree` also synchronizes.
A program that allocates and frees a tensor per op (exactly what mtorch does) would
spend most of its time inside the driver. The standard fix (PyTorch does the same,
far more elaborately) is a *caching allocator*: `deallocate` does not call
`cudaFree`, it parks the block on a free list keyed by size bucket; the next
`allocate` of a similar size pops it back off, costing nanoseconds. We bucket sizes
by rounding up to powers of two (min 512 bytes) so a 1000-byte request can reuse a
1024-byte block. The escape hatch `MTORCH_CUDA_DISABLE_CACHING=1` reverts to raw
`cudaMalloc`/`cudaFree` — invaluable when hunting memory bugs with
`compute-sanitizer` (§A), because cached reuse masks use-after-free errors.

CUDA API calls used in this step, one sentence each: `cudaGetDeviceCount(&n)` asks
the driver how many CUDA GPUs exist (it fails cleanly, without crashing, on machines
with no driver — which is why registration is conditional on it). `cudaMalloc(&p, n)`
allocates `n` bytes of device memory. `cudaFree(p)` releases it.
`cudaMemcpy(dst, src, n, kind)` copies `n` bytes, where the `kind` argument names the
direction: `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, or
`cudaMemcpyDeviceToDevice` — passing the wrong kind is undefined behavior, so each
allocator method hard-codes its correct kind.

**Preconditions**: 7-2 committed; clean tree; the Phase 5 allocator header exists:

```bash
git status --short                                     # empty
ls cpp/mtorch/core/detail/allocator.h                  # exists
sed -n '1,60p' cpp/mtorch/core/detail/allocator.h      # READ THIS OUTPUT NOW (see Action 2)
grep -rn 'register_allocator' cpp/mtorch/core/detail/allocator.h
```

**Actions**:

1. Create `cpp/mtorch/core/cuda/allocator.cpp` with exactly this content:

   ```cpp
   // Phase 7 step 7-3 (10-phase7-cuda-backend.md).
   // CUDA implementation of the Phase 5 allocator interface, with a simple
   // size-bucketed caching free list (see the step's Background note for why
   // caching matters: cudaMalloc/cudaFree synchronize the device and are slow).
   //
   // Registration happens at import time (static initializer at the bottom of
   // this file) and only when cudaGetDeviceCount reports >= 1 device, which is
   // what makes mtorch.cuda.is_available() come out true through the registry.

   #include "mtorch/core/detail/allocator.h"

   #include "mtorch/core/cuda/common.h"
   #include "mtorch/core/tensor.h"

   #include <cstdlib>
   #include <mutex>
   #include <unordered_map>
   #include <vector>

   namespace mtorch {
   namespace cuda {
   namespace {

   // MTORCH_CUDA_DISABLE_CACHING=1 -> raw cudaMalloc/cudaFree for every call.
   // Read once; the value cannot change after import.
   bool caching_disabled() {
     static const bool disabled = []() {
       const char* value = std::getenv("MTORCH_CUDA_DISABLE_CACHING");
       return value != nullptr && value[0] == '1';
     }();
     return disabled;
   }

   // Round a byte count up to the free-list bucket size: the next power of two,
   // but never below 512 bytes (tiny allocations all share one bucket, and 512
   // keeps every returned pointer usefully aligned for coalesced access).
   int64_t bucket_bytes(int64_t nbytes) {
     int64_t bucket = 512;
     while (bucket < nbytes) {
       bucket *= 2;
     }
     return bucket;
   }

   class CudaAllocator final : public ::mtorch::detail::Allocator {
    public:
     void* allocate(int64_t nbytes) override {
       if (nbytes <= 0) {
         nbytes = 1;  // keep zero-element storages harmless and non-null
       }
       const int64_t bucket = bucket_bytes(nbytes);
       std::lock_guard<std::mutex> lock(mutex_);
       if (!caching_disabled()) {
         auto it = free_lists_.find(bucket);
         if (it != free_lists_.end() && !it->second.empty()) {
           void* pointer = it->second.back();
           it->second.pop_back();
           live_bucket_[pointer] = bucket;
           return pointer;
         }
       }
       void* pointer = nullptr;
       cudaError_t err = cudaMalloc(&pointer, (size_t)bucket);
       if (err == cudaErrorMemoryAllocation) {
         // Out of memory: return every cached block to the driver, retry once.
         cudaGetLastError();  // clear the sticky error slot
         release_cached_blocks_locked();
         err = cudaMalloc(&pointer, (size_t)bucket);
       }
       MTORCH_CUDA_CHECK(err);
       live_bucket_[pointer] = bucket;
       return pointer;
     }

     void deallocate(void* pointer) override {
       if (pointer == nullptr) {
         return;
       }
       std::lock_guard<std::mutex> lock(mutex_);
       auto it = live_bucket_.find(pointer);
       // Unknown pointer: not ours; freeing it would corrupt the driver heap.
       if (it == live_bucket_.end()) {
         return;
       }
       const int64_t bucket = it->second;
       live_bucket_.erase(it);
       if (caching_disabled()) {
         MTORCH_CUDA_CHECK(cudaFree(pointer));
       } else {
         free_lists_[bucket].push_back(pointer);
       }
     }

     void copy_from_host(void* device_dst, const void* host_src, int64_t nbytes) override {
       if (nbytes <= 0) return;
       MTORCH_CUDA_CHECK(
           cudaMemcpy(device_dst, host_src, (size_t)nbytes, cudaMemcpyHostToDevice));
     }

     void copy_to_host(void* host_dst, const void* device_src, int64_t nbytes) override {
       if (nbytes <= 0) return;
       MTORCH_CUDA_CHECK(
           cudaMemcpy(host_dst, device_src, (size_t)nbytes, cudaMemcpyDeviceToHost));
     }

     void copy_on_device(void* device_dst, const void* device_src, int64_t nbytes) override {
       if (nbytes <= 0) return;
       MTORCH_CUDA_CHECK(
           cudaMemcpy(device_dst, device_src, (size_t)nbytes, cudaMemcpyDeviceToDevice));
     }

     bool host_accessible() const override {
       return false;  // CUDA pointers must never be dereferenced by host code.
     }

    private:
     void release_cached_blocks_locked() {
       for (auto& entry : free_lists_) {
         for (void* pointer : entry.second) {
           MTORCH_CUDA_CHECK(cudaFree(pointer));
         }
         entry.second.clear();
       }
     }

     std::mutex mutex_;
     std::unordered_map<int64_t, std::vector<void*>> free_lists_;  // bucket -> parked blocks
     std::unordered_map<void*, int64_t> live_bucket_;              // outstanding ptr -> bucket
   };

   // Import-time registration. A static initializer in a linked object file runs
   // when Python dlopens mtorch/_C*.so, i.e. at `import mtorch`.
   bool register_cuda_allocator() {
     int device_count = 0;
     cudaError_t err = cudaGetDeviceCount(&device_count);
     if (err != cudaSuccess) {
       cudaGetLastError();  // no driver / no GPU: clear the sticky error, stay unregistered
       return false;
     }
     if (device_count <= 0) {
       return false;
     }
     static CudaAllocator allocator;
     ::mtorch::detail::register_allocator(DeviceType::CUDA, &allocator);
     return true;
   }

   const bool cuda_allocator_registered = register_cuda_allocator();

   }  // namespace
   }  // namespace cuda
   }  // namespace mtorch
   ```

2. **Reconcile the override signatures with the real Phase 5 header** (mechanical;
   the listing above uses the contract names, the header is authoritative). From
   the precondition's `sed -n '1,60p' cpp/mtorch/core/detail/allocator.h` output,
   compare each pure-virtual declaration with the corresponding `override` line in
   the new file: method name, parameter order, integer type (`int64_t` vs
   `size_t`), `const`-ness. Where they differ, edit **only the signature line** in
   `allocator.cpp` to match the header exactly; bodies stay as written. Do the
   same for the `register_allocator(DeviceType, Allocator*)` call (argument order
   and namespace as declared in the header) and for the base-class name in
   `class CudaAllocator final : public ...`. If the header's allocate also takes a
   device index or alignment argument, add it to the signature and ignore it in
   the body (v1 is single-GPU, device 0).

3. Build and smoke-test transfers:

   ```bash
   python3 setup.py build_ext --inplace
   python3 - <<'PY'
   import mtorch
   print("available:", mtorch.cuda.is_available())          # True
   x = mtorch.tensor([[1.0, 2.0], [3.0, 4.0]])
   y = x.cuda()
   print("device:", str(y.device))                          # cuda / cuda:0 (Phase 5 spelling)
   z = y.cpu()
   print("roundtrip:", z.tolist())                          # [[1.0, 2.0], [3.0, 4.0]]
   h = x.to(mtorch.float16).cuda().cpu()
   print("f16 roundtrip:", h.tolist())                      # [[1.0, 2.0], [3.0, 4.0]]
   PY
   MTORCH_CUDA_DISABLE_CACHING=1 python3 - <<'PY'
   import mtorch
   x = mtorch.tensor([1.0, 2.0, 3.0]).cuda().cpu()
   print("nocache roundtrip:", x.tolist())                  # [1.0, 2.0, 3.0]
   PY
   ```

4. **Record the parity debt.** Now that cuda is available, the Phase 5 parity
   table's cuda rows run for the first time — and every row whose op has no CUDA
   implementation yet fails through `unsupported_device`. That is expected *only
   at this step*, and it is tracked, not ignored: record the exact failing set,
   commit it, and shrink it to empty by 7-12. This file is the amendment to "tree
   stays green" for steps 7-3 … 7-12: **green on the parity suite means "the
   failing set equals the committed `parity-debt.txt` exactly."**

   ```bash
   pytest tests/compat/test_device_parity.py -q -rf --tb=no | grep '^FAILED' | sort > docs/design/baseline/parity-debt.txt
   wc -l docs/design/baseline/parity-debt.txt
   pytest tests/compat/test_device_parity.py -q -rf --tb=line | grep -A0 'FAILED' | head -5
   ```

   Inspect the last command's output: every failure reason must be the
   `unsupported_device` raise (its message text — whatever Phase 5 chose — appears
   in each line, mentioning cuda). If ANY failure is something else (crash,
   segfault, wrong values on a *cpu* row), that is a real regression: STANDARD
   FAIL. If the parity suite instead skips unavailable ops and reports 0 failed,
   `parity-debt.txt` is legitimately empty — commit it anyway; the mechanism
   degenerates gracefully.

**Verification**: STANDARD VERIFY (§0.8) with the 7-3 amendment: the full compat
suite matches the Linux baseline exactly (the CPU suite never touches cuda), and
the parity suite's failing set equals `docs/design/baseline/parity-debt.txt`:

```bash
pytest tests/compat/test_device_parity.py -q -rf --tb=no | grep '^FAILED' | sort > /tmp/parity-now.txt
diff docs/design/baseline/parity-debt.txt /tmp/parity-now.txt && echo PARITY-DEBT-MATCH
```

Must print `PARITY-DEBT-MATCH`. Also Action 3 printed `True` and both round-trip
lists exactly.

**On failure**: STANDARD FAIL (§0.8); `git clean -fd -- cpp/mtorch/core/cuda/allocator.cpp`
and `git clean -fd -- docs/design/baseline/parity-debt.txt`. Mark 7-3 **BLOCKED**.

**Commit**:

```bash
git add cpp/mtorch/core/cuda/allocator.cpp docs/design/baseline/parity-debt.txt
git commit -m "refactor(phase7-3): CudaAllocator with caching free list + registration"
```

Then update PROGRESS.md per §8 (separate commit).

---

## Step 7-4: The oracle tool `tests/compat/tools/cuda_oracle_check.py`

**Goal**: Create the standalone script that compares mtorch-on-CUDA against
**torch-on-CUDA** (the pytest harness cannot express reference-on-cuda, §0.7 item 3)
and doubles as the benchmark-ratio recorder for 7-13. Created now, before any op
family, so that every family step can immediately run it; ops not yet implemented
are counted as `UNSUPPORTED`, and the count must reach 0 by step 7-12.

**Preconditions**: 7-3 committed; `mtorch.cuda.is_available()` is `True`; clean tree.

```bash
git status --short          # empty
python3 -c "import mtorch; assert mtorch.cuda.is_available()"
mkdir -p tests/compat/tools
```

**Actions**:

1. Create `tests/compat/tools/cuda_oracle_check.py` with exactly this content:

   ```python
   #!/usr/bin/env python3
   """Phase 7 oracle: compare mtorch-on-CUDA against torch-on-CUDA directly.

   The pytest compat harness has no --compat-device option (see tests/conftest.py),
   so "reference computed on cuda" is not expressible there. This script fills the
   gap: every case runs the same op through torch (on cuda) and mtorch (on cuda),
   copies both results back to cpu, and compares values, shape, and dtype.

   Usage:
     python3 tests/compat/tools/cuda_oracle_check.py                # all cases
     python3 tests/compat/tools/cuda_oracle_check.py --only 'unary.*'
     python3 tests/compat/tools/cuda_oracle_check.py --bench        # ratio table (markdown)

   Exit code 0 iff no case FAILED. UNSUPPORTED cases (mtorch raised its
   unsupported-device error because that family is not implemented yet) do not
   fail the run; step 7-13 requires the UNSUPPORTED count to be 0.
   """
   from __future__ import annotations

   import argparse
   import fnmatch
   import math
   import statistics
   import sys
   import time

   import torch
   import mtorch

   # N3 (guide section 0.9): TF32 off, or fp32 matmul parity fails at ~1e-3.
   torch.backends.cuda.matmul.allow_tf32 = False
   torch.backends.cudnn.allow_tf32 = False

   # If mtorch's unsupported_device message uses a different word, append it here
   # (this tuple is the ONLY line of this file a later step may edit; see 7-4
   # Verification).
   UNSUPPORTED_MARKERS = ("unsupported", "not supported", "not implemented")

   # Tolerances per (family_kind, dtype) — fixed in guide section 0.9 N4.
   TOLERANCES = {
       ("exact", "float32"): (0.0, 0.0),
       ("exact", "float16"): (0.0, 0.0),
       ("exact", "float64"): (0.0, 0.0),
       ("exact", "int32"): (0.0, 0.0),
       ("exact", "int64"): (0.0, 0.0),
       ("exact", "bool"): (0.0, 0.0),
       ("elementwise", "float32"): (1e-6, 1e-6),
       ("elementwise", "float64"): (1e-12, 1e-12),
       ("elementwise", "float16"): (1e-2, 1e-3),
       ("elementwise", "int32"): (0.0, 0.0),
       ("elementwise", "int64"): (0.0, 0.0),
       ("reductionish", "float32"): (1e-4, 1e-5),
       ("reductionish", "float64"): (1e-12, 1e-12),
       ("reductionish", "float16"): (1e-2, 1e-3),
       ("reductionish", "int32"): (0.0, 0.0),
       ("reductionish", "int64"): (0.0, 0.0),
   }

   UNARY_OPS = [
       # exactly the op list of the CPU unary() chain, elementwise_ops.cpp
       "abs", "acos", "asin", "atan", "ceil", "cos", "cosh", "deg2rad", "erf",
       "erfc", "exp", "expm1", "floor", "frac", "log", "log10", "log1p", "log2",
       "neg", "rad2deg", "reciprocal", "round", "rsqrt", "sigmoid", "sign", "sin",
       "sinh", "sqrt", "square", "tan", "tanh", "trunc",
   ]

   # Inputs whose domain is safe for every unary op above (positive, in (-1, 1)
   # after the *0.4, away from poles): base grid scaled per op group.
   def unary_input(op: str) -> torch.Tensor:
       g = torch.arange(1, 65, dtype=torch.float32).reshape(8, 8)
       if op in ("acos", "asin", "atan"):        # domain [-1, 1]
           return (g / 100.0) - 0.3
       if op in ("log", "log10", "log1p", "log2", "sqrt", "rsqrt", "reciprocal"):
           return g / 7.0                        # strictly positive
       return (g / 16.0) - 2.0                   # mixed signs, moderate magnitude


   def to_mtorch(t: torch.Tensor):
       """Rebuild a torch cpu tensor as an mtorch tensor of the same dtype."""
       name = str(t.dtype).split(".")[-1]
       if name == "float16":
           return mtorch.tensor(t.float().tolist()).to(mtorch.float16)
       return mtorch.tensor(t.tolist(), dtype=getattr(mtorch, name))


   def back_to_torch(result, like: torch.Tensor) -> torch.Tensor:
       return torch.tensor(result.cpu().tolist(), dtype=like.dtype)


   class Case:
       def __init__(self, case_id, kind, dtype, make_inputs, run, bench_inputs=None):
           self.case_id = case_id          # e.g. "binary.add_f32"
           self.kind = kind                # tolerance family: exact/elementwise/reductionish
           self.dtype = dtype              # dtype name string
           self.make_inputs = make_inputs  # () -> list[torch cpu tensor]
           self.run = run                  # (module, [device tensors]) -> device tensor
           self.bench_inputs = bench_inputs  # bigger () -> inputs, or None: skip in --bench


   def cast(t: torch.Tensor, dtype: str) -> torch.Tensor:
       return t.to(getattr(torch, dtype))


   def build_cases() -> list[Case]:
       torch.manual_seed(0)
       cases: list[Case] = []

       # -- transfers (work from 7-3 on) --------------------------------------
       for dt in ("float32", "float16", "float64", "int32", "int64", "bool"):
           base = torch.arange(0, 24).reshape(4, 6)
           src = (base % 2 == 0) if dt == "bool" else cast(base, dt)
           cases.append(Case(
               f"transfer.roundtrip_{dt}", "exact", dt,
               lambda src=src: [src],
               lambda m, inputs: inputs[0],   # .cuda() then .cpu() done by driver
           ))

       # -- fill (7-5) ---------------------------------------------------------
       for dt in ("float32", "float16", "int64", "bool"):
           cases.append(Case(
               f"fill.zeros_{dt}", "exact", dt, lambda: [],
               lambda m, inputs, dt=dt: m.zeros((4, 5), dtype=getattr(m, dt), device="cuda"),
           ))
           cases.append(Case(
               f"fill.ones_{dt}", "exact", dt, lambda: [],
               lambda m, inputs, dt=dt: m.ones((4, 5), dtype=getattr(m, dt), device="cuda"),
           ))
       cases.append(Case(
           "fill.full_float32", "exact", "float32", lambda: [],
           lambda m, inputs: m.full((3, 7), 3.5, dtype=m.float32, device="cuda"),
       ))

       # -- binary elementwise (7-6) ------------------------------------------
       bin_a32 = torch.randn(64, 64)
       bin_b32 = torch.randn(64, 64) + 2.5     # away from zero: safe divisor
       big_a = torch.randn(1024, 1024)
       big_b = torch.randn(1024, 1024) + 2.5
       for op in ("add", "sub", "mul", "div"):
           for dt in ("float32", "float16", "float64"):
               cases.append(Case(
                   f"binary.{op}_{dt}", "elementwise", dt,
                   lambda dt=dt: [cast(bin_a32, dt), cast(bin_b32, dt)],
                   lambda m, inputs, op=op: getattr(m, op)(inputs[0], inputs[1]),
                   bench_inputs=(lambda dt=dt: [cast(big_a, dt), cast(big_b, dt)])
                   if dt in ("float32", "float16") else None,
               ))
       for op in ("add", "sub", "mul"):
           for dt in ("int32", "int64"):
               ia = torch.arange(-32, 32, dtype=torch.int64).reshape(8, 8)
               ib = torch.arange(1, 65, dtype=torch.int64).reshape(8, 8)
               cases.append(Case(
                   f"binary.{op}_{dt}", "exact", dt,
                   lambda ia=ia, ib=ib, dt=dt: [cast(ia, dt), cast(ib, dt)],
                   lambda m, inputs, op=op: getattr(m, op)(inputs[0], inputs[1]),
               ))

       # -- unary (7-7) ---------------------------------------------------------
       for op in UNARY_OPS:
           for dt in ("float32", "float16"):
               src = unary_input(op)
               cases.append(Case(
                   f"unary.{op}_{dt}", "elementwise", dt,
                   lambda src=src, dt=dt: [cast(src, dt)],
                   lambda m, inputs, op=op: getattr(m, op)(inputs[0]),
                   bench_inputs=(lambda op=op, dt=dt: [cast(unary_input(op).repeat(64, 64), dt)])
                   if (op == "exp" and dt == "float32") else None,
               ))

       # -- matmul (7-8) ---------------------------------------------------------
       mm_a = torch.randn(48, 32)
       mm_b = torch.randn(32, 56)
       for dt in ("float32", "float16"):
           cases.append(Case(
               f"matmul.mm_{dt}", "reductionish", dt,
               lambda dt=dt: [cast(mm_a, dt), cast(mm_b, dt)],
               lambda m, inputs: m.matmul(inputs[0], inputs[1]),
               bench_inputs=lambda dt=dt: [cast(torch.randn(1024, 1024), dt),
                                           cast(torch.randn(1024, 1024), dt)],
           ))

       # -- reductions (7-9) ------------------------------------------------------
       red = torch.randn(37, 53)   # deliberately not a multiple of the block size
       for op in ("sum", "mean", "max"):
           for dt in ("float32", "float16"):
               cases.append(Case(
                   f"reduce.{op}_{dt}", "reductionish", dt,
                   lambda dt=dt: [cast(red, dt)],
                   lambda m, inputs, op=op: getattr(m, op)(inputs[0]),
                   bench_inputs=(lambda dt=dt: [cast(torch.randn(2048, 2048), dt)])
                   if op == "sum" else None,
               ))
       ri = torch.arange(0, 60, dtype=torch.int64).reshape(6, 10)
       for dt in ("int32", "int64"):
           cases.append(Case(
               f"reduce.sum_{dt}", "exact", dt,
               lambda ri=ri, dt=dt: [cast(ri, dt)],
               lambda m, inputs: m.sum(inputs[0]),
           ))

       # -- softmax / layer_norm / group_norm (7-10) -----------------------------
       sm = torch.randn(16, 33)
       ln_x = torch.randn(8, 24)
       ln_w = torch.randn(24)
       ln_b = torch.randn(24)
       gn_x = torch.randn(2, 8, 5, 5)
       gn_w = torch.randn(8)
       gn_b = torch.randn(8)
       for dt in ("float32", "float16"):
           cases.append(Case(
               f"norm.softmax_{dt}", "reductionish", dt,
               lambda dt=dt: [cast(sm, dt)],
               lambda m, inputs: m.nn.functional.softmax(inputs[0], dim=-1),
               bench_inputs=lambda dt=dt: [cast(torch.randn(512, 1024), dt)],
           ))
           cases.append(Case(
               f"norm.layer_norm_{dt}", "reductionish", dt,
               lambda dt=dt: [cast(ln_x, dt), cast(ln_w, dt), cast(ln_b, dt)],
               lambda m, inputs: m.nn.functional.layer_norm(
                   inputs[0], [24], inputs[1], inputs[2], 1e-5),
           ))
           cases.append(Case(
               f"norm.group_norm_{dt}", "reductionish", dt,
               lambda dt=dt: [cast(gn_x, dt), cast(gn_w, dt), cast(gn_b, dt)],
               lambda m, inputs: m.nn.functional.group_norm(
                   inputs[0], 4, inputs[1], inputs[2], 1e-5),
           ))

       # -- conv2d (7-11) ----------------------------------------------------------
       cv_x = torch.randn(2, 3, 12, 12)
       cv_w = torch.randn(6, 3, 3, 3)
       cv_b = torch.randn(6)
       for dt in ("float32", "float16"):
           cases.append(Case(
               f"conv.conv2d_{dt}", "reductionish", dt,
               lambda dt=dt: [cast(cv_x, dt), cast(cv_w, dt), cast(cv_b, dt)],
               lambda m, inputs: m.nn.functional.conv2d(
                   inputs[0], inputs[1], inputs[2], stride=(1, 1), padding=(1, 1)),
               bench_inputs=lambda dt=dt: [cast(torch.randn(8, 32, 64, 64), dt),
                                           cast(torch.randn(64, 32, 3, 3), dt),
                                           cast(torch.randn(64), dt)],
           ))

       # -- scaled_dot_product_attention (7-12) -------------------------------------
       q = torch.randn(2, 4, 16, 8)
       k = torch.randn(2, 4, 16, 8)
       v = torch.randn(2, 4, 16, 8)
       for dt in ("float32", "float16"):
           cases.append(Case(
               f"sdpa.basic_{dt}", "reductionish", dt,
               lambda dt=dt: [cast(q, dt), cast(k, dt), cast(v, dt)],
               lambda m, inputs: m.nn.functional.scaled_dot_product_attention(
                   inputs[0], inputs[1], inputs[2]),
               bench_inputs=lambda dt=dt: [cast(torch.randn(4, 8, 256, 64), dt)] * 3,
           ))
       return cases


   def is_unsupported(error: BaseException) -> bool:
       text = str(error).lower()
       return any(marker in text for marker in UNSUPPORTED_MARKERS)


   def run_case(case: Case, bench: bool):
       inputs = (case.bench_inputs or case.make_inputs)() if bench else case.make_inputs()
       # torch side, on cuda
       t_dev = [t.cuda() for t in inputs]
       expected = case.run(torch, t_dev)
       torch.cuda.synchronize()
       expected = expected.cpu()
       # mtorch side, on cuda (transfers are explicit; sync is implicit in v1)
       m_dev = [to_mtorch(t).cuda() for t in inputs]
       actual = case.run(mtorch, m_dev).cpu()
       # compare dtype name, shape, values
       actual_dtype = str(actual.dtype).split(".")[-1]
       expected_dtype = str(expected.dtype).split(".")[-1]
       if actual_dtype != expected_dtype:
           raise AssertionError(f"dtype: torch={expected_dtype} mtorch={actual_dtype}")
       if tuple(actual.shape) != tuple(expected.shape):
           raise AssertionError(f"shape: torch={tuple(expected.shape)} mtorch={tuple(actual.shape)}")
       rtol, atol = TOLERANCES[(case.kind, case.dtype)]
       torch.testing.assert_close(
           back_to_torch(actual, expected), expected, rtol=rtol, atol=atol,
           check_dtype=False)


   def median_ms(fn, repeat=10, warmup=3) -> float:
       for _ in range(warmup):
           fn()
       torch.cuda.synchronize()
       samples = []
       for _ in range(repeat):
           start = time.perf_counter()
           fn()
           torch.cuda.synchronize()
           samples.append((time.perf_counter() - start) * 1000.0)
       return statistics.median(samples)


   def bench_case(case: Case):
       inputs = case.bench_inputs()
       t_dev = [t.cuda() for t in inputs]
       m_dev = [to_mtorch(t).cuda() for t in inputs]
       torch_ms = median_ms(lambda: case.run(torch, t_dev))
       mtorch_ms = median_ms(lambda: case.run(mtorch, m_dev))
       return torch_ms, mtorch_ms


   def main() -> int:
       parser = argparse.ArgumentParser()
       parser.add_argument("--only", default="*", help="fnmatch pattern on case ids")
       parser.add_argument("--bench", action="store_true",
                           help="print a markdown ratio table instead of checking")
       args = parser.parse_args()

       cases = [c for c in build_cases() if fnmatch.fnmatch(c.case_id, args.only)]
       if not cases:
           print(f"no case matches {args.only!r}", file=sys.stderr)
           return 2

       if args.bench:
           print("| case | dtype | torch ms | mtorch ms | ratio (mtorch/torch) |")
           print("|---|---|---|---|---|")
           for case in cases:
               if case.bench_inputs is None:
                   continue
               try:
                   torch_ms, mtorch_ms = bench_case(case)
               except Exception as error:  # noqa: BLE001
                   if is_unsupported(error):
                       print(f"| {case.case_id} | {case.dtype} | - | - | UNSUPPORTED |")
                       continue
                   raise
               print(f"| {case.case_id} | {case.dtype} | {torch_ms:.3f} | "
                     f"{mtorch_ms:.3f} | {mtorch_ms / torch_ms:.2f}x |")
           return 0

       failed, unsupported, passed = [], [], 0
       for case in cases:
           try:
               run_case(case, bench=False)
               passed += 1
           except Exception as error:  # noqa: BLE001
               if is_unsupported(error):
                   unsupported.append(case.case_id)
               else:
                   failed.append((case.case_id, str(error).splitlines()[0]))
       for case_id, message in failed:
           print(f"FAIL {case_id}: {message}")
       for case_id in unsupported:
           print(f"UNSUPPORTED {case_id}")
       print(f"passed={passed} unsupported={len(unsupported)} failed={len(failed)}")
       if failed:
           print("ORACLE FAILED")
           return 1
       print(f"ORACLE OK ({passed} checks, {len(unsupported)} unsupported)")
       return 0


   if __name__ == "__main__":
       sys.exit(main())
   ```

2. Run it:

   ```bash
   python3 tests/compat/tools/cuda_oracle_check.py | tail -5
   ```

   Expected now (only transfers implemented): a final line
   `ORACLE OK (6 checks, N unsupported)` where the 6 passed checks are the
   `transfer.*` cases and N is everything else. Two deviation rules, both
   mechanical: (a) if a `transfer.*` case FAILs, that is a 7-3 regression —
   STANDARD FAIL; (b) if op cases are reported as `FAIL` rather than
   `UNSUPPORTED` and the printed message is clearly the Phase 5
   unsupported-device raise, the marker tuple missed the message's wording: add
   the message's distinctive word (lower-cased) to `UNSUPPORTED_MARKERS` — that
   tuple is the only permitted edit — and re-run until every unimplemented op
   case shows `UNSUPPORTED`.

**Verification**: STANDARD VERIFY (§0.8, with the parity-debt comparison from 7-3),
plus Action 2's `ORACLE OK` line with `failed=0`.

**On failure**: STANDARD FAIL (§0.8); `git clean -fd -- tests/compat/tools/`.
Mark 7-4 **BLOCKED**.

**Commit**:

```bash
git add tests/compat/tools/cuda_oracle_check.py
git commit -m "refactor(phase7-4): torch-CUDA oracle + ratio bench script"
```

Then update PROGRESS.md per §8 (separate commit).

---

## Step 7-5: Op family: fill/copy (+ the dtype dispatch header, + `ops.cpp`)

**Goal**: First real op family. Create the dtype-dispatch header `dispatch.cuh`
(used by every later kernel), the TensorPtr-level layer `ops.h`/`ops.cpp` with its
guard helpers, the fill kernel (all six dtypes), and the `copy_from` device path;
hook both into `Tensor::fill_inplace` and `Tensor::copy_from` in
`cpp/mtorch/core/tensor_core.cpp`; add parity rows; oracle `fill.*` cases go green.

**Preconditions**: 7-4 committed; clean tree.

```bash
git status --short                                             # empty
grep -n 'void Tensor::fill_inplace' cpp/mtorch/core/tensor_core.cpp   # prints a line
grep -n 'void Tensor::copy_from' cpp/mtorch/core/tensor_core.cpp      # prints a line
```

(If either grep prints nothing, find the symbol with
`grep -rn 'void Tensor::fill_inplace' cpp/mtorch/core/` per §0.5's relocation rule.)

**Actions**:

1. Create `cpp/mtorch/core/cuda/dispatch.cuh` with exactly this content:

   ```cpp
   #pragma once

   // Phase 7 step 7-5. The dtype-switch used by every kernel launcher in the
   // .cu files. nvcc-only (includes cuda_fp16.h); never include from host .cpp.
   //
   // Pattern: a launcher receives the runtime dtype code (kernels.h DtypeCode,
   // numerically equal to mtorch::ScalarType) and expands one case per dtype,
   // aliasing two types inside the case body:
   //   scalar_t — the storage element type in device memory
   //   acc_t    — the type arithmetic is performed in (guide 0.9 N1: __half
   //              storage ALWAYS computes in float; small ints accumulate in
   //              int64_t so reductions cannot overflow)
   // __half <-> float conversions are implicit via cuda_fp16.h's C++ operators,
   // so `static_cast<acc_t>(x)` and `static_cast<scalar_t>(r)` are all you need.

   #include <cuda_fp16.h>

   #include <cstdint>
   #include <stdexcept>
   #include <string>

   #include "mtorch/core/cuda/kernels.h"

   #define MTORCH_CUDA_PRIVATE_CASE(CODE, SCALAR_T, ACC_T, ...)                    \
     case (CODE): {                                                                \
       using scalar_t = SCALAR_T;                                                  \
       using acc_t = ACC_T;                                                        \
       __VA_ARGS__;                                                                \
       break;                                                                      \
     }

   // All six dtypes (fill / copy).
   #define MTORCH_CUDA_DISPATCH_ALL(DTYPE_CODE, NAME, ...)                         \
     switch (DTYPE_CODE) {                                                         \
       MTORCH_CUDA_PRIVATE_CASE(::mtorch::cuda::kF16, __half, float, __VA_ARGS__)  \
       MTORCH_CUDA_PRIVATE_CASE(::mtorch::cuda::kF32, float, float, __VA_ARGS__)   \
       MTORCH_CUDA_PRIVATE_CASE(::mtorch::cuda::kF64, double, double, __VA_ARGS__) \
       MTORCH_CUDA_PRIVATE_CASE(::mtorch::cuda::kI32, int32_t, int64_t, __VA_ARGS__) \
       MTORCH_CUDA_PRIVATE_CASE(::mtorch::cuda::kI64, int64_t, int64_t, __VA_ARGS__) \
       MTORCH_CUDA_PRIVATE_CASE(::mtorch::cuda::kBool, bool, int64_t, __VA_ARGS__) \
       default:                                                                    \
         throw std::runtime_error(std::string(NAME) + ": unexpected dtype code");  \
     }

   // Floating dtypes only (unary, binary div).
   #define MTORCH_CUDA_DISPATCH_FLOATS(DTYPE_CODE, NAME, ...)                      \
     switch (DTYPE_CODE) {                                                         \
       MTORCH_CUDA_PRIVATE_CASE(::mtorch::cuda::kF16, __half, float, __VA_ARGS__)  \
       MTORCH_CUDA_PRIVATE_CASE(::mtorch::cuda::kF32, float, float, __VA_ARGS__)   \
       MTORCH_CUDA_PRIVATE_CASE(::mtorch::cuda::kF64, double, double, __VA_ARGS__) \
       default:                                                                    \
         throw std::runtime_error(std::string(NAME) + ": unexpected dtype code");  \
     }

   // Floats + signed ints (binary add/sub/mul, reductions).
   #define MTORCH_CUDA_DISPATCH_NUMERIC(DTYPE_CODE, NAME, ...)                     \
     switch (DTYPE_CODE) {                                                         \
       MTORCH_CUDA_PRIVATE_CASE(::mtorch::cuda::kF16, __half, float, __VA_ARGS__)  \
       MTORCH_CUDA_PRIVATE_CASE(::mtorch::cuda::kF32, float, float, __VA_ARGS__)   \
       MTORCH_CUDA_PRIVATE_CASE(::mtorch::cuda::kF64, double, double, __VA_ARGS__) \
       MTORCH_CUDA_PRIVATE_CASE(::mtorch::cuda::kI32, int32_t, int64_t, __VA_ARGS__) \
       MTORCH_CUDA_PRIVATE_CASE(::mtorch::cuda::kI64, int64_t, int64_t, __VA_ARGS__) \
       default:                                                                    \
         throw std::runtime_error(std::string(NAME) + ": unexpected dtype code");  \
     }

   namespace mtorch {
   namespace cuda {

   // Shared launch-shape helper for all grid-stride kernels (guide 0.4).
   inline int grid_blocks_for(int64_t n) {
     const int64_t blocks = (n + 255) / 256;
     return (int)(blocks > 4096 ? 4096 : blocks);
   }

   }  // namespace cuda
   }  // namespace mtorch
   ```

2. Append to `cpp/mtorch/core/cuda/kernels.h`, immediately before the line
   `}  // namespace cuda` (this is the standing "append declarations" anchor for
   every later step too):

   ```cpp
   // 7-5 fill/copy.
   // Fills out[0..n) with `value` converted to the dtype named by dtype_code.
   void launch_fill(void* out_device, int64_t n, double value, int dtype_code);
   ```

3. Create `cpp/mtorch/core/cuda/fill_copy.cu` with exactly this content:

   ```cpp
   // Phase 7 step 7-5: fill kernel (all six dtypes). Device-to-device / host
   // copies need no kernel — they are cudaMemcpy calls made by the allocator —
   // so "copy" contributes no code here; see ops.cpp.

   #include "mtorch/core/cuda/common.h"
   #include "mtorch/core/cuda/dispatch.cuh"
   #include "mtorch/core/cuda/kernels.h"

   namespace mtorch {
   namespace cuda {
   namespace {

   template <typename scalar_t>
   __global__ void fill_kernel(scalar_t* out, int64_t n, scalar_t value) {
     for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
          i += (int64_t)gridDim.x * blockDim.x) {
       out[i] = value;
     }
   }

   }  // namespace

   void launch_fill(void* out_device, int64_t n, double value, int dtype_code) {
     if (n == 0) return;
     const int blocks = grid_blocks_for(n);
     MTORCH_CUDA_DISPATCH_ALL(dtype_code, "launch_fill",
       // double -> acc_t -> scalar_t: for __half this goes double->float->half,
       // matching how the CPU core narrows fill values.
       fill_kernel<scalar_t><<<blocks, 256>>>(
           reinterpret_cast<scalar_t*>(out_device), n,
           static_cast<scalar_t>(static_cast<acc_t>(value)));
       MTORCH_CUDA_LAUNCH_CHECK();
     );
   }

   }  // namespace cuda
   }  // namespace mtorch
   ```

4. Create `cpp/mtorch/core/cuda/ops.h` with exactly this content (later steps
   append declarations before the closing namespace brace, same convention as
   `kernels.h`):

   ```cpp
   #pragma once

   // Phase 7. TensorPtr-level CUDA ops: v1 guards (guide 0.6), Tensor -> raw
   // pointer translation, and calls into the nvcc-compiled launchers. Host C++.
   // Core section files include this ONLY inside #if defined(MTORCH_WITH_CUDA).

   #include "mtorch/core/tensor.h"

   namespace mtorch {
   namespace cuda {

   // 7-5 fill/copy.
   void fill_inplace(Tensor& self, double value);
   void copy_from(Tensor& self, const Tensor& source);

   }  // namespace cuda
   }  // namespace mtorch
   ```

5. Create `cpp/mtorch/core/cuda/ops.cpp`. Two placeholders must be resolved while
   pasting — `@STORAGE_DATA@` and `@RAISE@` — by the two mechanical discoveries
   below the listing. Content:

   ```cpp
   // Phase 7. See ops.h. Guard order in every function: grad (G1), device (G4),
   // contiguity (G2), shape (G3), dtype (G5), op-specific (G6) — guide 0.6.

   #include "mtorch/core/cuda/ops.h"

   #include "mtorch/core/cuda/kernels.h"
   #include "mtorch/core/detail/allocator.h"

   #include <cstdint>

   namespace mtorch {
   namespace cuda {
   namespace {

   // Raw byte pointer of a tensor's storage plus the view offset. Works for CPU
   // and CUDA storages alike (Phase 5 unified the accessor).
   void* tensor_data(const Tensor& t) {
     return static_cast<void*>(
         static_cast<uint8_t*>(@STORAGE_DATA@) + t.offset * t.element_size());
   }

   int64_t tensor_nbytes(const Tensor& t) { return t.numel() * t.element_size(); }

   int dtype_code(ScalarType dtype) { return static_cast<int>(dtype); }

   bool on_cuda(const Tensor& t) { return t.device.type == DeviceType::CUDA; }

   [[noreturn]] void raise_unsupported(const char* op) {
     @RAISE@;
   }

   ::mtorch::detail::Allocator& cuda_allocator() {
     return *::mtorch::detail::allocator_for(DeviceType::CUDA);
   }

   void guard_grad(const Tensor& t, const char* op) {
     if (t.requires_grad && is_grad_enabled()) raise_unsupported(op);  // G1
   }

   void guard_contiguous(const Tensor& t, const char* op) {
     if (!t.is_contiguous()) raise_unsupported(op);  // G2
   }

   }  // namespace

   // ---- 7-5 fill/copy ---------------------------------------------------------

   void fill_inplace(Tensor& self, double value) {
     guard_grad(self, "fill_");
     guard_contiguous(self, "fill_");
     if (self.numel() > 0) {
       launch_fill(tensor_data(self), self.numel(), value, dtype_code(self.dtype));
     }
     self.mark_storage_modified();
   }

   void copy_from(Tensor& self, const Tensor& source) {
     guard_grad(self, "copy_");
     guard_contiguous(self, "copy_");
     guard_contiguous(source, "copy_");
     if (self.sizes != source.sizes) raise_unsupported("copy_");   // G3
     if (self.dtype != source.dtype) raise_unsupported("copy_");   // G5 (no cast in v1)
     const int64_t bytes = tensor_nbytes(self);
     if (bytes == 0) { self.mark_storage_modified(); return; }
     if (on_cuda(self) && on_cuda(source)) {
       cuda_allocator().copy_on_device(tensor_data(self), tensor_data(source), bytes);
     } else if (on_cuda(self)) {   // cpu -> cuda
       cuda_allocator().copy_from_host(tensor_data(self), tensor_data(source), bytes);
     } else {                      // cuda -> cpu
       cuda_allocator().copy_to_host(tensor_data(self), tensor_data(source), bytes);
     }
     self.mark_storage_modified();
   }

   }  // namespace cuda
   }  // namespace mtorch
   ```

   **Discovery A — `@STORAGE_DATA@`** (the expression yielding the storage's raw
   byte pointer). Copy it from existing device code, in this priority order; use
   the first that produces a hit and substitute the expression verbatim
   (with `t.storage->` / `(*t.storage).` spelled the way the hit spells it):

   ```bash
   grep -rn 'copy_from_host\|copy_to_host' cpp/mtorch/core/*.cpp cpp/mtorch/core/detail/*.cpp | head -5
   #   -> open the calling line; the argument it passes as the device pointer is
   #      the expression. (This is how Phase 5's to_device gets it.)
   grep -rn 'data_ptr\|raw_data\|device_data' cpp/mtorch/core/tensor.h cpp/mtorch/core/detail/*.h | head -5
   #   -> a method: use t.storage-><method>()
   grep -n 'void\* *[a-z_]*;' cpp/mtorch/core/tensor.h
   #   -> a member: use t.storage-><member>
   ```

   If all three greps miss, BLOCK 7-5 (the Phase 5 storage layout is not what the
   contract promised) — do not invent an accessor.

   **Discovery B — `@RAISE@`** (the unsupported raise). Per §0.5 item 3:

   ```bash
   grep -rn 'unsupported_device' cpp/mtorch/core --include='*.cpp' | grep -v cuda/ | head -3
   ```

   Substitute the exact call the existing sites use, with `op` as the op-name
   argument and `DeviceType::CUDA` as the device, e.g.
   `::mtorch::detail::unsupported_device(op, DeviceType::CUDA)` — whatever the
   real namespace is. Add the corresponding `#include` to ops.cpp if the compiler
   asks for it. If `unsupported_device` returns (is not `[[noreturn]]`), append
   `throw std::runtime_error(std::string(op) + ": unreachable");` on the next
   line so `raise_unsupported` still satisfies `[[noreturn]]` (and add
   `#include <stdexcept>`, `#include <string>`).

   **Discovery C — the registry call forms.** The listing assumes
   `allocator_for(DeviceType)` returns `Allocator*` (the Phase 5 contract). Check
   the real declaration (`grep -n 'allocator_for' cpp/mtorch/core/detail/allocator.h`)
   and, if it returns a reference or a smart pointer, adjust ONLY the one-line
   body of `cuda_allocator()` accordingly (drop or add the `*`). Same
   reconciliation rule as 7-3 Action 2.

6. Insert the dispatch hooks in `cpp/mtorch/core/tensor_core.cpp` (placement per
   §0.5: after the Metal hook if one exists in the function, else right after the
   opening `{`). First add the include block of §0.5 item 4 to `tensor_core.cpp`
   if not present. Then:

   6a. In `void Tensor::fill_inplace(double value)`
   (`grep -n 'void Tensor::fill_inplace' cpp/mtorch/core/tensor_core.cpp`) —
   BEFORE (first lines of the body, whatever they are — shown schematically; do
   not edit them):

   ```cpp
   void Tensor::fill_inplace(double value) {
     <existing first statement>
   ```

   AFTER:

   ```cpp
   void Tensor::fill_inplace(double value) {
     if (device.type == DeviceType::CUDA) {
   #if defined(MTORCH_WITH_CUDA)
       ::mtorch::cuda::fill_inplace(*this, value);
       return;
   #else
       @RAISE-with-op-"fill_"@   // same call form as Discovery B, inline
   #endif
     }
     <existing first statement>
   ```

   (Here and in every later hook, `@RAISE-with-op-"..."@` means: write the
   Discovery-B call with that op-name string literal. If a Metal hook block for
   this function exists, the CUDA block goes immediately after it instead.)

   6b. In `void Tensor::copy_from(const Tensor& source)`
   (`grep -n 'void Tensor::copy_from' cpp/mtorch/core/tensor_core.cpp`), same
   pattern with the condition covering both sides:

   ```cpp
     if (device.type == DeviceType::CUDA || source.device.type == DeviceType::CUDA) {
   #if defined(MTORCH_WITH_CUDA)
       ::mtorch::cuda::copy_from(*this, source);
       return;
   #else
       @RAISE-with-op-"copy_"@
   #endif
     }
   ```

7. Build and run the family's oracle slice:

   ```bash
   python3 setup.py build_ext --inplace
   python3 tests/compat/tools/cuda_oracle_check.py --only 'fill.*'
   python3 tests/compat/tools/cuda_oracle_check.py --only 'transfer.*'
   ```

   Expected: both end `ORACLE OK` with `unsupported=0`.
   **Conditional remediation (apply ONLY if a `fill.*` case FAILs with wrong
   values or a crash, meaning the Phase 5 factories do not route device fills
   through `Tensor::fill_inplace`):** locate the factory,
   `grep -rn 'TensorPtr full(' cpp/mtorch/core/ cpp/mtorch/core/detail/ | grep -v cuda/`,
   and read its body (small window). If it writes elements through a host loop
   (`set`/`set_at_linear`/`storage->set`), insert at the top of `full(...)`:

   ```cpp
   #if defined(MTORCH_WITH_CUDA)
     if (device.type == DeviceType::CUDA) {
       TensorPtr result = zeros(sizes, dtype, device);  // Phase 5 device alloc, zeroed
       if (value != 0.0) result->fill_inplace(value);
       return result;
     }
   #endif
   ```

   and re-run this Action. If `zeros` itself fails, BLOCK — that is a Phase 5
   defect, not yours to fix here.

8. Append parity rows to `tests/compat/test_device_parity.py` (procedure §0.7
   item 2). Rows to add — ids, semantics, tolerance kind `exact`:

   | case id | op / expression | dtype |
   |---|---|---|
   | `fill.zeros_f32` | `zeros((4, 5))` | float32 |
   | `fill.ones_f16` | `ones((4, 5))` | float16 |
   | `fill.full_f32` | `full((3, 7), 3.5)` | float32 |
   | `fill.zeros_i64` | `zeros((4, 5))` | int64 |
   | `fill.zeros_bool` | `zeros((4, 5))` | bool |

9. **DEBT UPDATE** (this exact block recurs in every family step; referenced
   below as "DEBT UPDATE"):

   ```bash
   pytest tests/compat/test_device_parity.py -q -rf --tb=no | grep '^FAILED' | sort > /tmp/parity-now.txt
   comm -13 docs/design/baseline/parity-debt.txt /tmp/parity-now.txt
   cp /tmp/parity-now.txt docs/design/baseline/parity-debt.txt
   ```

   The `comm -13` (lines only in the *new* failing set) must print **nothing** —
   no new failures, ever; lines may only disappear. If it prints anything:
   STANDARD FAIL.

10. Benchmark note: run
    `python3 tests/compat/tools/cuda_oracle_check.py --bench --only 'binary.add_float32'`
    is not yet possible (7-6); for this family there is no bench case — record in
    the Phase 7 Notes one line: `7-5 fill/copy: no bench case (memset-class op)`.

**Verification**: STANDARD VERIFY (§0.8) with the parity-debt rule; plus Action 7's
two `ORACLE OK ... unsupported=0` lines; plus the new parity rows pass on both cpu
and cuda (`pytest tests/compat/test_device_parity.py -q -k 'fill' | tail -3`).

**On failure**: STANDARD FAIL (§0.8). Paths for `git clean -fd --`:
`cpp/mtorch/core/cuda/dispatch.cuh`, `cpp/mtorch/core/cuda/fill_copy.cu`,
`cpp/mtorch/core/cuda/ops.h`, `cpp/mtorch/core/cuda/ops.cpp`; restore edited files
via `git restore --staged --worktree .`. Mark 7-5 **BLOCKED**.

**Commit**:

```bash
git add cpp/mtorch/core/cuda/dispatch.cuh cpp/mtorch/core/cuda/fill_copy.cu \
        cpp/mtorch/core/cuda/ops.h cpp/mtorch/core/cuda/ops.cpp \
        cpp/mtorch/core/cuda/kernels.h cpp/mtorch/core/tensor_core.cpp \
        tests/compat/test_device_parity.py docs/design/baseline/parity-debt.txt
git commit -m "refactor(phase7-5): cuda fill/copy family"
```

(If Action 7's conditional remediation edited a factory file, add that file too.)
Then update PROGRESS.md per §8 (separate commit).

---

## Step 7-6: Op family: binary elementwise add/sub/mul/div

**Goal**: The grid-stride binary elementwise template kernel — contiguous,
same-shape, same-dtype only — hooked into `binary_tensor_tensor`,
`binary_tensor_scalar`, and `binary_scalar_tensor` in
`cpp/mtorch/core/elementwise_ops.cpp`.

**The v1 guard, stated plainly (it is temporary, and mirrored from Phase 6):**
non-contiguous inputs, broadcast shapes (any `sizes` mismatch), and mixed dtypes
raise `unsupported_device`. Why: a contiguous same-shape kernel is a
perfectly-coalesced straight-line loop (§0.2); broadcasting needs per-element index
arithmetic (divides/mods per dimension) that is easy to get subtly wrong and slow to
run — the Metal backend (doc 09) made the same v1 cut, and lifting both guards
together, with a shared strided-index helper, is a designated later phase. Do not
partially lift it here.

**Preconditions**: 7-5 committed; clean tree.

```bash
git status --short   # empty
grep -n '^TensorPtr binary_tensor_tensor(' cpp/mtorch/core/elementwise_ops.cpp
grep -n '^TensorPtr binary_tensor_scalar(' cpp/mtorch/core/elementwise_ops.cpp
grep -n '^TensorPtr binary_scalar_tensor(' cpp/mtorch/core/elementwise_ops.cpp
```

**Actions**:

1. Append to `cpp/mtorch/core/cuda/kernels.h` (before `}  // namespace cuda`):

   ```cpp
   // 7-6 binary elementwise. op_code: 0=add 1=sub 2=mul 3=div.
   // All three pointers same dtype (dtype_code), all contiguous, length n.
   void launch_binary(void* out_device, const void* lhs_device, const void* rhs_device,
                      int64_t n, int op_code, int dtype_code);
   // Tensor (+|-|*|/) scalar and scalar (+|-|*|/) tensor; the scalar rides along
   // as double and is converted inside the kernel's acc_t.
   void launch_binary_scalar(void* out_device, const void* input_device, double scalar,
                             bool scalar_on_left, int64_t n, int op_code, int dtype_code);
   ```

2. Create `cpp/mtorch/core/cuda/binary.cu` with exactly this content:

   ```cpp
   // Phase 7 step 7-6: contiguous same-shape binary elementwise kernels.
   // Arithmetic in acc_t (float for __half — guide 0.9 N1; int64_t for int32,
   // which cannot overflow intermediate int32 products of parity-sized inputs
   // and matches how torch promotes accumulators internally for mul on cuda).

   #include "mtorch/core/cuda/common.h"
   #include "mtorch/core/cuda/dispatch.cuh"
   #include "mtorch/core/cuda/kernels.h"

   namespace mtorch {
   namespace cuda {
   namespace {

   // op_code contract (keep in sync with ops.cpp binary_op_code()):
   // 0=add 1=sub 2=mul 3=div
   template <typename acc_t>
   __device__ __forceinline__ acc_t apply_binary(acc_t x, acc_t y, int op_code) {
     switch (op_code) {
       case 0: return x + y;
       case 1: return x - y;
       case 2: return x * y;
       default: return x / y;  // 3: guarded to float dtypes by ops.cpp
     }
   }

   template <typename scalar_t, typename acc_t>
   __global__ void binary_kernel(scalar_t* out, const scalar_t* lhs,
                                 const scalar_t* rhs, int64_t n, int op_code) {
     for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
          i += (int64_t)gridDim.x * blockDim.x) {
       out[i] = static_cast<scalar_t>(apply_binary<acc_t>(
           static_cast<acc_t>(lhs[i]), static_cast<acc_t>(rhs[i]), op_code));
     }
   }

   template <typename scalar_t, typename acc_t>
   __global__ void binary_scalar_kernel(scalar_t* out, const scalar_t* input,
                                        acc_t scalar, bool scalar_on_left,
                                        int64_t n, int op_code) {
     for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
          i += (int64_t)gridDim.x * blockDim.x) {
       const acc_t x = static_cast<acc_t>(input[i]);
       const acc_t r = scalar_on_left ? apply_binary<acc_t>(scalar, x, op_code)
                                      : apply_binary<acc_t>(x, scalar, op_code);
       out[i] = static_cast<scalar_t>(r);
     }
   }

   }  // namespace

   void launch_binary(void* out_device, const void* lhs_device, const void* rhs_device,
                      int64_t n, int op_code, int dtype_code) {
     if (n == 0) return;
     const int blocks = grid_blocks_for(n);
     MTORCH_CUDA_DISPATCH_NUMERIC(dtype_code, "launch_binary",
       binary_kernel<scalar_t, acc_t><<<blocks, 256>>>(
           reinterpret_cast<scalar_t*>(out_device),
           reinterpret_cast<const scalar_t*>(lhs_device),
           reinterpret_cast<const scalar_t*>(rhs_device), n, op_code);
       MTORCH_CUDA_LAUNCH_CHECK();
     );
   }

   void launch_binary_scalar(void* out_device, const void* input_device, double scalar,
                             bool scalar_on_left, int64_t n, int op_code, int dtype_code) {
     if (n == 0) return;
     const int blocks = grid_blocks_for(n);
     MTORCH_CUDA_DISPATCH_NUMERIC(dtype_code, "launch_binary_scalar",
       binary_scalar_kernel<scalar_t, acc_t><<<blocks, 256>>>(
           reinterpret_cast<scalar_t*>(out_device),
           reinterpret_cast<const scalar_t*>(input_device),
           static_cast<acc_t>(scalar), scalar_on_left, n, op_code);
       MTORCH_CUDA_LAUNCH_CHECK();
     );
   }

   }  // namespace cuda
   }  // namespace mtorch
   ```

3. Append to `cpp/mtorch/core/cuda/ops.h` (before `}  // namespace cuda`):

   ```cpp
   // 7-6 binary elementwise.
   TensorPtr binary_tensor_tensor(const TensorPtr& left, const TensorPtr& right,
                                  const std::string& op);
   TensorPtr binary_tensor_scalar(const TensorPtr& left, double scalar,
                                  ScalarType scalar_dtype, const std::string& op);
   TensorPtr binary_scalar_tensor(double scalar, ScalarType scalar_dtype,
                                  const TensorPtr& right, const std::string& op);
   ```

   (These mirror the public CPU signatures in `cpp/mtorch/core/tensor.h` — verify
   with `grep -n 'binary_tensor_tensor' cpp/mtorch/core/tensor.h`; if the real
   signatures differ, copy the real ones.)

4. Append to `cpp/mtorch/core/cuda/ops.cpp`, immediately before the final
   `}  // namespace cuda` line (standing anchor for all later families), the
   implementation:

   ```cpp
   // ---- 7-6 binary elementwise --------------------------------------------------

   namespace {

   int binary_op_code(const std::string& op) {
     if (op == "add") return 0;
     if (op == "sub") return 1;
     if (op == "mul") return 2;
     if (op == "div") return 3;
     return -1;
   }

   bool binary_dtype_allowed(ScalarType dtype, int op_code) {
     switch (dtype) {
       case ScalarType::Float16:
       case ScalarType::Float32:
       case ScalarType::Float64:
         return true;
       case ScalarType::Int32:
       case ScalarType::Int64:
         return op_code != 3;  // int div promotes on CPU; unsupported in v1 (G5)
       default:
         return false;  // bool
     }
   }

   TensorPtr make_cuda_result_like(const Tensor& t) {
     // zeros() allocates device storage through the Phase 5 allocator path
     // (proven working since 7-5); the fill-with-zero cost is irrelevant in v1.
     return zeros(t.sizes, t.dtype, t.device);
   }

   }  // namespace

   TensorPtr binary_tensor_tensor(const TensorPtr& left, const TensorPtr& right,
                                  const std::string& op) {
     const char* name = "binary_tensor_tensor";
     const int op_code = binary_op_code(op);
     if (op_code < 0) raise_unsupported(name);                       // G6
     guard_grad(*left, name); guard_grad(*right, name);              // G1
     if (!on_cuda(*left) || !on_cuda(*right)) raise_unsupported(name);  // G4
     guard_contiguous(*left, name); guard_contiguous(*right, name);  // G2
     if (left->sizes != right->sizes) raise_unsupported(name);       // G3 (no broadcast)
     if (left->dtype != right->dtype) raise_unsupported(name);       // G5 (no promote)
     if (!binary_dtype_allowed(left->dtype, op_code)) raise_unsupported(name);
     TensorPtr out = make_cuda_result_like(*left);
     launch_binary(tensor_data(*out), tensor_data(*left), tensor_data(*right),
                   out->numel(), op_code, dtype_code(out->dtype));
     return out;
   }

   TensorPtr binary_tensor_scalar(const TensorPtr& left, double scalar,
                                  ScalarType scalar_dtype, const std::string& op) {
     const char* name = "binary_tensor_scalar";
     const int op_code = binary_op_code(op);
     if (op_code < 0) raise_unsupported(name);
     guard_grad(*left, name);
     guard_contiguous(*left, name);
     // v1: the scalar must not force a dtype change (e.g. int tensor / float
     // scalar promotes on CPU) — if the scalar's dtype class differs from the
     // tensor's, refuse (G5).
     const bool tensor_is_float = left->dtype == ScalarType::Float16 ||
                                  left->dtype == ScalarType::Float32 ||
                                  left->dtype == ScalarType::Float64;
     const bool scalar_is_float = scalar_dtype == ScalarType::Float16 ||
                                  scalar_dtype == ScalarType::Float32 ||
                                  scalar_dtype == ScalarType::Float64;
     if (tensor_is_float != scalar_is_float) raise_unsupported(name);
     if (!binary_dtype_allowed(left->dtype, op_code)) raise_unsupported(name);
     TensorPtr out = make_cuda_result_like(*left);
     launch_binary_scalar(tensor_data(*out), tensor_data(*left), scalar,
                          /*scalar_on_left=*/false, out->numel(), op_code,
                          dtype_code(out->dtype));
     return out;
   }

   TensorPtr binary_scalar_tensor(double scalar, ScalarType scalar_dtype,
                                  const TensorPtr& right, const std::string& op) {
     const char* name = "binary_scalar_tensor";
     const int op_code = binary_op_code(op);
     if (op_code < 0) raise_unsupported(name);
     guard_grad(*right, name);
     guard_contiguous(*right, name);
     const bool tensor_is_float = right->dtype == ScalarType::Float16 ||
                                  right->dtype == ScalarType::Float32 ||
                                  right->dtype == ScalarType::Float64;
     const bool scalar_is_float = scalar_dtype == ScalarType::Float16 ||
                                  scalar_dtype == ScalarType::Float32 ||
                                  scalar_dtype == ScalarType::Float64;
     if (tensor_is_float != scalar_is_float) raise_unsupported(name);
     if (!binary_dtype_allowed(right->dtype, op_code)) raise_unsupported(name);
     TensorPtr out = make_cuda_result_like(*right);
     launch_binary_scalar(tensor_data(*out), tensor_data(*right), scalar,
                          /*scalar_on_left=*/true, out->numel(), op_code,
                          dtype_code(out->dtype));
     return out;
   }
   ```

5. Insert the three dispatch hooks in `cpp/mtorch/core/elementwise_ops.cpp`
   (include block of §0.5 item 4 first, once for this file). For each function
   found by the Precondition greps, the hook right after the opening `{` (or
   after the Metal block, §0.5):

   - `binary_tensor_tensor`:

     ```cpp
       if (left->device.type == DeviceType::CUDA || right->device.type == DeviceType::CUDA) {
     #if defined(MTORCH_WITH_CUDA)
         return ::mtorch::cuda::binary_tensor_tensor(left, right, op);
     #else
         @RAISE-with-op-"binary_tensor_tensor"@
     #endif
       }
     ```

   - `binary_tensor_scalar`:

     ```cpp
       if (left->device.type == DeviceType::CUDA) {
     #if defined(MTORCH_WITH_CUDA)
         return ::mtorch::cuda::binary_tensor_scalar(left, scalar, scalar_dtype, op);
     #else
         @RAISE-with-op-"binary_tensor_scalar"@
     #endif
       }
     ```

   - `binary_scalar_tensor`:

     ```cpp
       if (right->device.type == DeviceType::CUDA) {
     #if defined(MTORCH_WITH_CUDA)
         return ::mtorch::cuda::binary_scalar_tensor(scalar, scalar_dtype, right, op);
     #else
         @RAISE-with-op-"binary_scalar_tensor"@
     #endif
       }
     ```

     (Parameter names inside the hooks must match the function's actual parameter
     names — read the signature the grep shows and adjust the identifiers only.)

6. Build, oracle slice, parity rows, DEBT UPDATE:

   ```bash
   python3 setup.py build_ext --inplace
   python3 tests/compat/tools/cuda_oracle_check.py --only 'binary.*'
   ```

   Expected `ORACLE OK` with `unsupported=0`. Then append parity rows (§0.7):

   | case id | op / expression | dtype | tolerance |
   |---|---|---|---|
   | `binary.add_f32` | `a + b`, shapes (64, 64) | float32 | rtol=1e-6, atol=1e-6 |
   | `binary.sub_f32` | `a - b` | float32 | rtol=1e-6, atol=1e-6 |
   | `binary.mul_f16` | `a * b` | float16 | rtol=1e-2, atol=1e-3 |
   | `binary.div_f32` | `a / b`, b offset +2.5 | float32 | rtol=1e-6, atol=1e-6 |
   | `binary.add_i64` | `a + b` | int64 | exact |
   | `binary.add_scalar_f32` | `a + 1.5` | float32 | rtol=1e-6, atol=1e-6 |
   | `binary.rsub_scalar_f32` | `1.5 - a` | float32 | rtol=1e-6, atol=1e-6 |

   Then DEBT UPDATE (7-5 Action 9 block; `comm -13` must print nothing — and this
   time lines for the binary ops should *disappear* from the debt file).

7. Benchmark note:
   `python3 tests/compat/tools/cuda_oracle_check.py --bench --only 'binary.add_float32'`
   — record the printed ratio as one Notes line, e.g.
   `7-6 binary add 1024x1024 f32: <ratio>x`.

**Verification**: STANDARD VERIFY (§0.8) + parity-debt rule + Action 6's
`ORACLE OK ... unsupported=0`.

**On failure**: STANDARD FAIL. New path: `git clean -fd -- cpp/mtorch/core/cuda/binary.cu`.
Mark 7-6 **BLOCKED**.

**Commit**:

```bash
git add cpp/mtorch/core/cuda/binary.cu cpp/mtorch/core/cuda/kernels.h \
        cpp/mtorch/core/cuda/ops.h cpp/mtorch/core/cuda/ops.cpp \
        cpp/mtorch/core/elementwise_ops.cpp \
        tests/compat/test_device_parity.py docs/design/baseline/parity-debt.txt
git commit -m "refactor(phase7-6): cuda binary elementwise family"
```

Then update PROGRESS.md per §8 (separate commit).

---

## Step 7-7: Op family: unary (the 32-op chain)

**Goal**: One grid-stride kernel handling all 32 ops of the CPU `unary()` chain,
dtypes float16/float32/float64, hooked into `TensorPtr unary(...)` in
`cpp/mtorch/core/elementwise_ops.cpp`.

**The op list is not invented — it is read off the CPU code.** The CPU `unary()`
dispatches on an op *string*; the full set as of the 2026-07-09 tree (32 ops):

```
abs acos asin atan ceil cos cosh deg2rad erf erfc exp expm1 floor frac
log log10 log1p log2 neg rad2deg reciprocal round rsqrt sigmoid sign sin
sinh sqrt square tan tanh trunc
```

Re-verify against the post-split tree before coding (must print exactly that set,
one string per line, quotes included):

```bash
L=$(grep -n '^TensorPtr unary(' cpp/mtorch/core/elementwise_ops.cpp | head -1 | cut -d: -f1)
sed -n "${L},$((L+190))p" cpp/mtorch/core/elementwise_ops.cpp | grep -oE '"[a-z0-9_]+"' | sort -u
```

If the printed set differs (Phase 3's table-ification may have moved or extended
it): implement exactly the 32 ops above; any op string outside the code table below
falls through to `raise_unsupported` automatically, which is the correct v1
behavior for stragglers. If an op from the list above is *missing* from the CPU
set, still implement it — dead table entries are harmless.

**Preconditions**: 7-6 committed; clean tree; the re-verify grep above ran.

**Actions**:

1. Append to `cpp/mtorch/core/cuda/kernels.h` (before `}  // namespace cuda`):

   ```cpp
   // 7-7 unary. op_code = index into the fixed 32-op table (ops.cpp
   // unary_op_code() and unary.cu apply_unary() — keep the three in sync).
   void launch_unary(void* out_device, const void* input_device, int64_t n,
                     int op_code, int dtype_code);
   ```

2. Create `cpp/mtorch/core/cuda/unary.cu` with exactly this content:

   ```cpp
   // Phase 7 step 7-7: unary elementwise kernel, 32 ops, float dtypes only.
   // All math runs in acc_t (float for f16/f32 storage, double for f64); CUDA
   // provides float/double overloads of every std math function in device code.

   #include "mtorch/core/cuda/common.h"
   #include "mtorch/core/cuda/dispatch.cuh"
   #include "mtorch/core/cuda/kernels.h"

   namespace mtorch {
   namespace cuda {
   namespace {

   // Codes 0..31, alphabetical, matching ops.cpp unary_op_code():
   //  0 abs        1 acos      2 asin     3 atan     4 ceil      5 cos
   //  6 cosh       7 deg2rad   8 erf      9 erfc    10 exp      11 expm1
   // 12 floor     13 frac     14 log     15 log10   16 log1p    17 log2
   // 18 neg       19 rad2deg  20 reciprocal 21 round 22 rsqrt   23 sigmoid
   // 24 sign      25 sin      26 sinh    27 sqrt    28 square   29 tan
   // 30 tanh      31 trunc

   template <typename acc_t>
   __device__ __forceinline__ acc_t apply_unary(acc_t x, int op_code) {
     const acc_t kDegToRad = (acc_t)0.017453292519943295;  // pi / 180
     const acc_t kRadToDeg = (acc_t)57.29577951308232;     // 180 / pi
     switch (op_code) {
       case 0: return fabs(x);
       case 1: return acos(x);
       case 2: return asin(x);
       case 3: return atan(x);
       case 4: return ceil(x);
       case 5: return cos(x);
       case 6: return cosh(x);
       case 7: return x * kDegToRad;
       case 8: return erf(x);
       case 9: return erfc(x);
       case 10: return exp(x);
       case 11: return expm1(x);
       case 12: return floor(x);
       case 13: return x - trunc(x);                    // frac, torch semantics
       case 14: return log(x);
       case 15: return log10(x);
       case 16: return log1p(x);
       case 17: return log2(x);
       case 18: return -x;
       case 19: return x * kRadToDeg;
       case 20: return (acc_t)1 / x;
       case 21: return rint(x);                          // round: half-to-even, like torch
       case 22: return rsqrt(x);                         // CUDA device intrinsic
       case 23: return (acc_t)1 / ((acc_t)1 + exp(-x));  // sigmoid
       case 24: return (acc_t)((x > (acc_t)0) - (x < (acc_t)0));  // sign
       case 25: return sin(x);
       case 26: return sinh(x);
       case 27: return sqrt(x);
       case 28: return x * x;                            // square
       case 29: return tan(x);
       case 30: return tanh(x);
       default: return trunc(x);                         // 31
     }
   }

   template <typename scalar_t, typename acc_t>
   __global__ void unary_kernel(scalar_t* out, const scalar_t* input, int64_t n,
                                int op_code) {
     for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
          i += (int64_t)gridDim.x * blockDim.x) {
       out[i] = static_cast<scalar_t>(
           apply_unary<acc_t>(static_cast<acc_t>(input[i]), op_code));
     }
   }

   }  // namespace

   void launch_unary(void* out_device, const void* input_device, int64_t n,
                     int op_code, int dtype_code) {
     if (n == 0) return;
     const int blocks = grid_blocks_for(n);
     MTORCH_CUDA_DISPATCH_FLOATS(dtype_code, "launch_unary",
       unary_kernel<scalar_t, acc_t><<<blocks, 256>>>(
           reinterpret_cast<scalar_t*>(out_device),
           reinterpret_cast<const scalar_t*>(input_device), n, op_code);
       MTORCH_CUDA_LAUNCH_CHECK();
     );
   }

   }  // namespace cuda
   }  // namespace mtorch
   ```

3. Append to `cpp/mtorch/core/cuda/ops.h` (before `}  // namespace cuda`):

   ```cpp
   // 7-7 unary.
   TensorPtr unary(const TensorPtr& input, const std::string& op);
   ```

4. Append to `cpp/mtorch/core/cuda/ops.cpp` (before the final
   `}  // namespace cuda`):

   ```cpp
   // ---- 7-7 unary -----------------------------------------------------------------

   namespace {

   int unary_op_code(const std::string& op) {
     static const char* kUnaryOps[32] = {
         "abs",  "acos",    "asin",       "atan",  "ceil",  "cos",     "cosh",
         "deg2rad", "erf",  "erfc",       "exp",   "expm1", "floor",   "frac",
         "log",  "log10",   "log1p",      "log2",  "neg",   "rad2deg",
         "reciprocal", "round", "rsqrt",  "sigmoid", "sign", "sin",    "sinh",
         "sqrt", "square",  "tan",        "tanh",  "trunc"};
     for (int code = 0; code < 32; ++code) {
       if (op == kUnaryOps[code]) return code;
     }
     return -1;
   }

   }  // namespace

   TensorPtr unary(const TensorPtr& input, const std::string& op) {
     const char* name = "unary";
     const int op_code = unary_op_code(op);
     if (op_code < 0) raise_unsupported(name);            // op outside the v1 table (G6)
     guard_grad(*input, name);                            // G1
     guard_contiguous(*input, name);                      // G2
     if (input->dtype != ScalarType::Float16 && input->dtype != ScalarType::Float32 &&
         input->dtype != ScalarType::Float64) {
       raise_unsupported(name);                           // G5
     }
     TensorPtr out = make_cuda_result_like(*input);
     launch_unary(tensor_data(*out), tensor_data(*input), out->numel(), op_code,
                  dtype_code(out->dtype));
     return out;
   }
   ```

5. Hook `TensorPtr unary(...)` in `cpp/mtorch/core/elementwise_ops.cpp`
   (`grep -n '^TensorPtr unary(' cpp/mtorch/core/elementwise_ops.cpp`; §0.5
   placement; the file's include block already exists from 7-6):

   ```cpp
     if (input->device.type == DeviceType::CUDA) {
   #if defined(MTORCH_WITH_CUDA)
       return ::mtorch::cuda::unary(input, op);
   #else
       @RAISE-with-op-"unary"@
   #endif
     }
   ```

   (`unary_predicate` is deliberately NOT hooked in v1 — its ops return bool
   tensors and are out of the 32-op scope; a cuda tensor reaching it follows
   whatever Phase 5's default no-hook behavior is.)

6. Build; oracle slice (all 64 unary cases — 32 ops x f32/f16); parity rows;
   DEBT UPDATE; bench note:

   ```bash
   python3 setup.py build_ext --inplace
   python3 tests/compat/tools/cuda_oracle_check.py --only 'unary.*'
   python3 tests/compat/tools/cuda_oracle_check.py --bench --only 'unary.exp_float32'
   ```

   Parity rows to append (§0.7) — a spot-check subset; the oracle covers all 32:

   | case id | op | dtype | tolerance |
   |---|---|---|---|
   | `unary.exp_f32` | `exp` on (8, 8) mixed-sign input | float32 | rtol=1e-6, atol=1e-6 |
   | `unary.sqrt_f32` | `sqrt` on positive input | float32 | rtol=1e-6, atol=1e-6 |
   | `unary.tanh_f16` | `tanh` | float16 | rtol=1e-2, atol=1e-3 |
   | `unary.round_f32` | `round` on half-integer values `[0.5, 1.5, 2.5, -0.5]` | float32 | exact |
   | `unary.sign_f32` | `sign` on `[-2.0, 0.0, 3.0]` | float32 | exact |

   Notes line: `7-7 unary exp 512x512 f32: <ratio>x`.

**Verification**: STANDARD VERIFY (§0.8) + parity-debt rule + `ORACLE OK` with
`unsupported=0` for `unary.*`. The `unary.round_f32` parity row is the canary for
the `rint` (half-to-even) choice — if it fails, the kernel used `round()`
(half-away-from-zero) somewhere; fix the kernel, never the row.

**On failure**: STANDARD FAIL. New path: `git clean -fd -- cpp/mtorch/core/cuda/unary.cu`.
Mark 7-7 **BLOCKED**.

**Commit**:

```bash
git add cpp/mtorch/core/cuda/unary.cu cpp/mtorch/core/cuda/kernels.h \
        cpp/mtorch/core/cuda/ops.h cpp/mtorch/core/cuda/ops.cpp \
        cpp/mtorch/core/elementwise_ops.cpp \
        tests/compat/test_device_parity.py docs/design/baseline/parity-debt.txt
git commit -m "refactor(phase7-7): cuda unary family (32 ops)"
```

Then update PROGRESS.md per §8 (separate commit).

---

## Step 7-8: Op family: matmul via cuBLAS

**Goal**: `matmul` for rank-2 x rank-2 contiguous float32/float16, implemented with
cuBLAS (NVIDIA's BLAS library — we do not hand-write a GEMM kernel; nobody beats
cuBLAS, and correctness is the point). Adds the cuBLAS handle singleton and the
row-major GEMM wrapper `gemm_rm` that 7-11 (conv2d) and 7-12 (sdpa) reuse.

**Background: cuBLAS in five sentences.**
cuBLAS is a host-side library: you call ordinary C functions from `.cpp` code and
*it* launches the GPU kernels. All calls go through a `cublasHandle_t`, an opaque
context created once with `cublasCreate` (expensive — hence a process-wide
singleton) and configured with `cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)`,
which forbids TF32 down-conversion (guide 0.9 N3). Every call returns a
`cublasStatus_t`, checked by a dedicated macro (`MTORCH_CUBLAS_CHECK` below),
because `cudaGetErrorString` does not understand cuBLAS codes —
`cublasGetStatusString` does. cuBLAS uses the default stream unless told otherwise,
so the v1 synchronous model needs no extra plumbing; `gemm_rm` still ends with a
checked `cudaDeviceSynchronize()` to keep the everything-synchronous invariant.

**Background: the column-major / row-major transpose trick.**
BLAS (since Fortran, 1979) assumes *column-major* matrices; mtorch tensors are
*row-major*. No data is ever transposed or copied — a bit of algebra does it:
the bytes of a row-major `M x N` matrix, read column-major, are exactly `M^T`
(an `N x M` matrix). We want row-major `C = op(A) x op(B)`. Take the transpose of
both sides: `C^T = op(B)^T x op(A)^T`. Every term of the right-hand side is "the
column-major reading of the row-major bytes we already have". So: call cuBLAS with
**operands swapped (B first), dimensions swapped (n, m, k), and each ld set to the
row-major row length** — cuBLAS then computes column-major `C^T`, and writes bytes
that read row-major as exactly `C`. Zero copies, zero transposes in memory. The
full derivation lives as a comment in `blas.cpp` so it is never re-derived wrongly.

**Why `cublasGemmEx` + `CUBLAS_COMPUTE_32F` for fp16, not `cublasHgemm`:**
`cublasHgemm` accumulates in fp16, violating rule N1 (guide 0.9) and failing parity
on any non-trivial `k`. `cublasGemmEx` with fp16 inputs/outputs and
`CUBLAS_COMPUTE_32F` accumulates every dot product in fp32 — same as torch.

**Preconditions**: 7-7 committed; clean tree.

```bash
git status --short   # empty
grep -n '^TensorPtr matmul(' cpp/mtorch/core/linalg.cpp   # prints a line
```

**Actions**:

1. Create `cpp/mtorch/core/cuda/blas.h` with exactly this content:

   ```cpp
   #pragma once

   // Phase 7 step 7-8. Row-major GEMM on top of cuBLAS. Host C++ (no kernels).

   #include <cstdint>

   namespace mtorch {
   namespace cuda {

   // C(m x n) = alpha * op(A) x op(B) + beta * C, ALL matrices row-major,
   // contiguous, on the device. dtype_code: kF32 or kF16 (kernels.h codes);
   // fp16 accumulates in fp32 (CUBLAS_COMPUTE_32F). trans_a/trans_b transpose
   // logically (no data movement): if trans_a, A is stored (k x m).
   void gemm_rm(int dtype_code, bool trans_a, bool trans_b, int64_t m, int64_t n,
                int64_t k, double alpha, const void* a, const void* b, double beta,
                void* c);

   }  // namespace cuda
   }  // namespace mtorch
   ```

2. Create `cpp/mtorch/core/cuda/blas.cpp` with exactly this content:

   ```cpp
   // Phase 7 step 7-8. See blas.h. TF32 is disabled here (PEDANTIC math mode)
   // per guide 0.9 N3 — do not remove that line, parity depends on it.

   #include "mtorch/core/cuda/blas.h"

   #include "mtorch/core/cuda/common.h"
   #include "mtorch/core/cuda/kernels.h"

   #include <cublas_v2.h>

   #include <stdexcept>
   #include <string>

   // cuBLAS calls return cublasStatus_t, not cudaError_t; dedicated checker.
   #define MTORCH_CUBLAS_CHECK(expr)                                                \
     do {                                                                           \
       cublasStatus_t mtorch_cublas_status__ = (expr);                              \
       if (mtorch_cublas_status__ != CUBLAS_STATUS_SUCCESS) {                       \
         throw std::runtime_error(std::string("cuBLAS error: ") +                   \
                                  cublasGetStatusString(mtorch_cublas_status__) +   \
                                  " at " __FILE__ ":" + std::to_string(__LINE__) +  \
                                  " in `" #expr "`");                               \
       }                                                                            \
     } while (0)

   namespace mtorch {
   namespace cuda {
   namespace {

   // Process-wide handle. cublasCreate is expensive; created on first use,
   // deliberately never destroyed (lives until process exit, like torch's).
   cublasHandle_t cublas_handle() {
     static cublasHandle_t handle = []() {
       cublasHandle_t created;
       MTORCH_CUBLAS_CHECK(cublasCreate(&created));
       // N3: forbid TF32; fp32 GEMMs stay true fp32 or parity fails at ~1e-3.
       MTORCH_CUBLAS_CHECK(cublasSetMathMode(created, CUBLAS_PEDANTIC_MATH));
       return created;
     }();
     return handle;
   }

   }  // namespace

   void gemm_rm(int dtype_code, bool trans_a, bool trans_b, int64_t m, int64_t n,
                int64_t k, double alpha, const void* a, const void* b, double beta,
                void* c) {
     // ------------------------------------------------------------------------
     // Row-major C = op(A) x op(B) via column-major cuBLAS. Derivation:
     //
     //   Bytes of row-major X (r x c), read column-major, are X^T (c x r).
     //
     //           row-major world                 column-major world (cuBLAS)
     //   C (m x n) = op(A)(m x k) x op(B)(k x n)
     //   transpose both sides:
     //   C^T (n x m) = op(B)^T (n x k) x op(A)^T (k x m)
     //        ^                ^                  ^
     //        |                |                  +-- col-major view of A's bytes
     //        |                +-- col-major view of B's bytes
     //        +-- what cuBLAS writes = bytes that read row-major as C. QED.
     //
     // Therefore: swap operands (B first), swap m<->n, keep k; each leading
     // dimension (ld) is the ROW-MAJOR row length of that operand's storage:
     //   A stored (m x k) if !trans_a, (k x m) if trans_a  ->  lda = trans_a ? m : k
     //   B stored (k x n) if !trans_b, (n x k) if trans_b  ->  ldb = trans_b ? k : n
     //   C stored (m x n)                                  ->  ldc = n
     // ------------------------------------------------------------------------
     const cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
     const cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
     const int lda = (int)(trans_a ? m : k);
     const int ldb = (int)(trans_b ? k : n);
     const int ldc = (int)n;

     if (dtype_code == kF32) {
       const float alpha_f = (float)alpha;
       const float beta_f = (float)beta;
       // cublasSgemm: single-precision column-major GEMM.
       MTORCH_CUBLAS_CHECK(cublasSgemm(cublas_handle(), op_b, op_a, (int)n, (int)m,
                                       (int)k, &alpha_f,
                                       (const float*)b, ldb,
                                       (const float*)a, lda, &beta_f,
                                       (float*)c, ldc));
     } else if (dtype_code == kF16) {
       const float alpha_f = (float)alpha;
       const float beta_f = (float)beta;
       // cublasGemmEx: mixed-precision GEMM; fp16 in/out, fp32 accumulation
       // (CUBLAS_COMPUTE_32F), alpha/beta given as float per the compute type.
       MTORCH_CUBLAS_CHECK(cublasGemmEx(cublas_handle(), op_b, op_a, (int)n, (int)m,
                                        (int)k, &alpha_f,
                                        b, CUDA_R_16F, ldb,
                                        a, CUDA_R_16F, lda, &beta_f,
                                        c, CUDA_R_16F, ldc,
                                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
     } else {
       throw std::runtime_error("gemm_rm: unexpected dtype code");
     }
     // Keep the v1 everything-synchronous invariant (guide 0.2, streams note).
     MTORCH_CUDA_CHECK(cudaDeviceSynchronize());
   }

   }  // namespace cuda
   }  // namespace mtorch
   ```

3. Append to `cpp/mtorch/core/cuda/ops.h`:

   ```cpp
   // 7-8 matmul (rank-2 x rank-2 only in v1).
   TensorPtr matmul(const TensorPtr& left, const TensorPtr& right);
   ```

4. Append to `cpp/mtorch/core/cuda/ops.cpp` (before the final
   `}  // namespace cuda`), and add `#include "mtorch/core/cuda/blas.h"` next to
   the other includes at the top of the file:

   ```cpp
   // ---- 7-8 matmul -------------------------------------------------------------

   TensorPtr matmul(const TensorPtr& left, const TensorPtr& right) {
     const char* name = "matmul";
     guard_grad(*left, name); guard_grad(*right, name);              // G1
     if (!on_cuda(*left) || !on_cuda(*right)) raise_unsupported(name);  // G4
     guard_contiguous(*left, name); guard_contiguous(*right, name);  // G2
     // G3/G6: v1 handles exactly the rank-2 x rank-2 case. Vectors, batched
     // (rank>=3) and broadcast matmul stay on the unsupported path for now.
     if (left->sizes.size() != 2 || right->sizes.size() != 2) raise_unsupported(name);
     if (left->sizes[1] != right->sizes[0]) raise_unsupported(name);
     if (left->dtype != right->dtype) raise_unsupported(name);       // G5
     if (left->dtype != ScalarType::Float32 && left->dtype != ScalarType::Float16) {
       raise_unsupported(name);
     }
     const int64_t m = left->sizes[0];
     const int64_t k = left->sizes[1];
     const int64_t n = right->sizes[1];
     TensorPtr out = zeros({m, n}, left->dtype, left->device);
     if (m > 0 && n > 0 && k > 0) {
       gemm_rm(dtype_code(left->dtype), /*trans_a=*/false, /*trans_b=*/false,
               m, n, k, /*alpha=*/1.0, tensor_data(*left), tensor_data(*right),
               /*beta=*/0.0, tensor_data(*out));
     }
     return out;
   }
   ```

5. Hook `matmul` in `cpp/mtorch/core/linalg.cpp` (include block §0.5 item 4
   first; placement per §0.5):

   ```cpp
     if (left->device.type == DeviceType::CUDA || right->device.type == DeviceType::CUDA) {
   #if defined(MTORCH_WITH_CUDA)
       return ::mtorch::cuda::matmul(left, right);
   #else
       @RAISE-with-op-"matmul"@
   #endif
     }
   ```

6. Build; oracle slice; parity rows; DEBT UPDATE; bench note:

   ```bash
   python3 setup.py build_ext --inplace
   python3 tests/compat/tools/cuda_oracle_check.py --only 'matmul.*'
   python3 tests/compat/tools/cuda_oracle_check.py --bench --only 'matmul.*'
   ```

   Parity rows (§0.7):

   | case id | op | dtype | tolerance |
   |---|---|---|---|
   | `matmul.mm_f32` | `matmul(a, b)`, a (48, 32), b (32, 56) | float32 | rtol=1e-4, atol=1e-5 |
   | `matmul.mm_f16` | same shapes | float16 | rtol=1e-2, atol=1e-3 |
   | `matmul.mm_f32_odd` | a (17, 31), b (31, 13) | float32 | rtol=1e-4, atol=1e-5 |

   Notes line: `7-8 matmul 1024^3 f32: <ratio>x, f16: <ratio>x`.

**Verification**: STANDARD VERIFY (§0.8) + parity-debt rule + `ORACLE OK`
(`unsupported=0`) for `matmul.*`. If `matmul.mm_f32` is off by ~1e-3: TF32 leaked —
check that `CUBLAS_PEDANTIC_MATH` is set (mtorch side) AND that the oracle's
`allow_tf32 = False` lines were not touched (torch side); see 0.9 N3.

**On failure**: STANDARD FAIL. New paths: `git clean -fd -- cpp/mtorch/core/cuda/blas.h`
and `git clean -fd -- cpp/mtorch/core/cuda/blas.cpp`. Mark 7-8 **BLOCKED**.

**Commit**:

```bash
git add cpp/mtorch/core/cuda/blas.h cpp/mtorch/core/cuda/blas.cpp \
        cpp/mtorch/core/cuda/ops.h cpp/mtorch/core/cuda/ops.cpp \
        cpp/mtorch/core/linalg.cpp \
        tests/compat/test_device_parity.py docs/design/baseline/parity-debt.txt
git commit -m "refactor(phase7-8): cuda matmul via cuBLAS (TF32 off)"
```

Then update PROGRESS.md per §8 (separate commit).

---

## Step 7-9: Op family: reductions sum / mean / max (the fully-worked teaching example)

**Goal**: Full-tensor reductions `reduce_sum`, `reduce_mean`, `reduce_max` on CUDA.
This is the guide's teaching example for cooperative kernels: **every line of the
two kernels is annotated** — read it slowly once, because 7-10's row-wise kernels
reuse the same ideas in compressed form. Dim-wise reductions (`reduce_sum_dim`
etc.) stay unsupported in v1 (they hit the unhooked default path; do not hook them).

**Background: why two passes and not one atomicAdd.**
A reduction combines n values into 1, but 4096 blocks cannot all write one output
cell without coordination. The tempting shortcut — each block does `atomicAdd` on
the single output — is banned by rule N2 (guide 0.9): floating-point addition is
not associative, atomics arrive in random order, and the result changes from run
to run. The deterministic scheme is **two passes**: pass 1 launches B blocks, each
reducing its grid-stride slice to ONE partial value in `partial[blockIdx.x]`; pass
2 launches exactly ONE block that reduces the B partials to the final value. Both
passes combine values in a fixed tree order, so the answer is bit-identical on
every run.

**Background: the inside of one block's reduction (shared-memory tree + warp
shuffle).** 256 threads each hold one running value in a register. Step 1: every
thread parks its value in shared memory (`shm[tid]`), barrier. Step 2: a *tree*:
128 threads add `shm[tid + 128]` into `shm[tid]`, barrier; 64 threads fold
`+64`, barrier — halving until 64 values remain. Step 3: when only warps-worth of
data is left (the last 64 → 32 fold happens as the warp enters), the final 32
values live in one warp, and `__shfl_down_sync` folds 16, 8, 4, 2, 1 —
register-to-register, no shared memory, no barriers needed because a warp executes
in lockstep. Thread 0 then owns the block's result.

**Preconditions**: 7-8 committed; clean tree.

```bash
git status --short   # empty
grep -n '^TensorPtr reduce_sum(' cpp/mtorch/core/reductions.cpp
grep -n '^TensorPtr reduce_mean(' cpp/mtorch/core/reductions.cpp
grep -n '^TensorPtr reduce_max(' cpp/mtorch/core/reductions.cpp
grep -n 'TensorPtr reduce_sum' cpp/mtorch/core/tensor.h   # copy the real signatures for Action 3
```

**Actions**:

1. Append to `cpp/mtorch/core/cuda/kernels.h` (before `}  // namespace cuda`):

   ```cpp
   // 7-9 full-tensor reductions. op_code: 0 = sum, 1 = max (mean is sum/n, done
   // by the caller). Returns the reduced value converted to double on the HOST —
   // the caller writes it into the 0-dim result tensor via launch_fill. (For
   // int64 sums beyond 2^53 the double round-trip loses low bits; v1 parity
   // inputs are far below that. Revisit when dim-reductions arrive.)
   double launch_reduce_all(const void* input_device, int64_t n, int op_code,
                            int dtype_code);
   ```

2. Create `cpp/mtorch/core/cuda/reduce.cu` with exactly this content:

   ```cpp
   // Phase 7 step 7-9: deterministic two-pass full reduction. Teaching example —
   // every line annotated. No atomics anywhere (guide 0.9 N2).

   #include "mtorch/core/cuda/common.h"
   #include "mtorch/core/cuda/dispatch.cuh"
   #include "mtorch/core/cuda/kernels.h"

   #include <climits>
   #include <cmath>

   namespace mtorch {
   namespace cuda {
   namespace {

   // The neutral starting value of the reduction: 0 for sum; the most negative
   // representable value for max (so any real element beats it).
   template <typename acc_t>
   __device__ __forceinline__ acc_t reduce_identity(int op_code) {
     if (op_code == 0) return (acc_t)0;                       // sum
     return (acc_t)-INFINITY;                                 // max (float/double)
   }
   template <>
   __device__ __forceinline__ int64_t reduce_identity<int64_t>(int op_code) {
     if (op_code == 0) return 0;                              // sum
     return LLONG_MIN;                                        // max over int storage
   }

   // The combining operator. NaN note: `a > b ? a : b` does not propagate NaN
   // the way torch.max does; v1 parity inputs contain no NaN by construction.
   template <typename acc_t>
   __device__ __forceinline__ acc_t reduce_combine(acc_t a, acc_t b, int op_code) {
     if (op_code == 0) return a + b;
     return a > b ? a : b;
   }

   // PASS-1 KERNEL: grid of B blocks; block b writes its slice's reduction into
   // partial[b]. Template: scalar_t = storage dtype, acc_t = accumulation dtype
   // (float for __half — N1; int64_t for int32 — no overflow).
   template <typename scalar_t, typename acc_t>
   __global__ void reduce_partial_kernel(const scalar_t* input, acc_t* partial,
                                         int64_t n, int op_code) {
     // Each block owns 256 slots of shared memory — one per thread (guide 0.2,
     // shared memory Background). This is the tree's workspace.
     __shared__ acc_t shm[256];

     // Phase A — sequential accumulation into a register. The grid-stride loop
     // gives this thread elements tid, tid+G, tid+2G, ... (G = total threads).
     // Consecutive threads read consecutive addresses -> fully coalesced.
     acc_t value = reduce_identity<acc_t>(op_code);
     for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
          i += (int64_t)gridDim.x * blockDim.x) {
       value = reduce_combine<acc_t>(value, (acc_t)input[i], op_code);
     }

     // Phase B — park the register value in shared memory and wait for everyone
     // (the barrier is mandatory: without it some thread would read a slot that
     // has not been written yet).
     shm[threadIdx.x] = value;
     __syncthreads();

     // Phase C — shared-memory tree: fold the top half onto the bottom half,
     // halving the active width: 256->128->64. We stop at 64 because the final
     // 64->32 fold below hands the rest to a single warp.
     for (int stride = 128; stride > 32; stride >>= 1) {
       if (threadIdx.x < stride) {
         shm[threadIdx.x] =
             reduce_combine<acc_t>(shm[threadIdx.x], shm[threadIdx.x + stride], op_code);
       }
       __syncthreads();  // every fold level needs a barrier before the next read
     }

     // Phase D — warp-shuffle finish. Only the first warp (threads 0..31) works
     // now; a warp runs in lockstep so no __syncthreads() is needed inside.
     if (threadIdx.x < 32) {
       // the 64->32 fold, from shared memory into a register:
       acc_t warp_value =
           reduce_combine<acc_t>(shm[threadIdx.x], shm[threadIdx.x + 32], op_code);
       // 32->16->8->4->2->1: __shfl_down_sync(mask, v, off) hands each thread
       // the value of the thread `off` lanes above it, register-to-register.
       for (int offset = 16; offset > 0; offset >>= 1) {
         warp_value = reduce_combine<acc_t>(
             warp_value, __shfl_down_sync(0xffffffffu, warp_value, offset), op_code);
       }
       // Lane 0 now holds the whole block's reduction.
       if (threadIdx.x == 0) {
         partial[blockIdx.x] = warp_value;
       }
     }
   }

   // PASS-2 KERNEL: launched with exactly ONE block; reduces the B partials the
   // same way and writes the single result. Same structure as pass 1 with the
   // grid-stride loop degenerated to a block-stride loop (gridDim.x == 1).
   template <typename acc_t>
   __global__ void reduce_final_kernel(const acc_t* partial, int64_t num_partials,
                                       acc_t* result, int op_code) {
     __shared__ acc_t shm[256];
     acc_t value = reduce_identity<acc_t>(op_code);
     for (int64_t i = threadIdx.x; i < num_partials; i += blockDim.x) {
       value = reduce_combine<acc_t>(value, partial[i], op_code);
     }
     shm[threadIdx.x] = value;
     __syncthreads();
     for (int stride = 128; stride > 32; stride >>= 1) {
       if (threadIdx.x < stride) {
         shm[threadIdx.x] =
             reduce_combine<acc_t>(shm[threadIdx.x], shm[threadIdx.x + stride], op_code);
       }
       __syncthreads();
     }
     if (threadIdx.x < 32) {
       acc_t warp_value =
           reduce_combine<acc_t>(shm[threadIdx.x], shm[threadIdx.x + 32], op_code);
       for (int offset = 16; offset > 0; offset >>= 1) {
         warp_value = reduce_combine<acc_t>(
             warp_value, __shfl_down_sync(0xffffffffu, warp_value, offset), op_code);
       }
       if (threadIdx.x == 0) {
         result[0] = warp_value;
       }
     }
   }

   }  // namespace

   double launch_reduce_all(const void* input_device, int64_t n, int op_code,
                            int dtype_code) {
     double host_result = 0.0;
     const int blocks = grid_blocks_for(n);
     MTORCH_CUDA_DISPATCH_NUMERIC(dtype_code, "launch_reduce_all",
       // Scratch: B partials + 1 final slot, in acc_t. Plain cudaMalloc (not the
       // caching allocator): tiny, short-lived, and keeps the .cu layer
       // dependency-free. cudaMalloc allocates device memory; cudaFree returns it.
       acc_t* scratch = nullptr;
       MTORCH_CUDA_CHECK(cudaMalloc(&scratch, sizeof(acc_t) * (blocks + 1)));
       reduce_partial_kernel<scalar_t, acc_t><<<blocks, 256>>>(
           reinterpret_cast<const scalar_t*>(input_device), scratch, n, op_code);
       MTORCH_CUDA_LAUNCH_CHECK();
       reduce_final_kernel<acc_t><<<1, 256>>>(scratch, blocks, scratch + blocks,
                                              op_code);
       MTORCH_CUDA_LAUNCH_CHECK();
       // Bring the single result value back to the host (8 bytes at most).
       acc_t final_value;
       MTORCH_CUDA_CHECK(cudaMemcpy(&final_value, scratch + blocks, sizeof(acc_t),
                                    cudaMemcpyDeviceToHost));
       MTORCH_CUDA_CHECK(cudaFree(scratch));
       host_result = (double)final_value;
     );
     return host_result;
   }

   }  // namespace cuda
   }  // namespace mtorch
   ```

3. Append to `cpp/mtorch/core/cuda/ops.h` (copy the parameter lists from the real
   `tensor.h` declarations found in the Preconditions — shown here as verified on
   the 2026-07-09 tree):

   ```cpp
   // 7-9 full reductions.
   TensorPtr reduce_sum(const TensorPtr& input, ScalarType dtype);
   TensorPtr reduce_mean(const TensorPtr& input, ScalarType dtype);
   TensorPtr reduce_max(const TensorPtr& input);
   ```

4. Append to `cpp/mtorch/core/cuda/ops.cpp` (before the final
   `}  // namespace cuda`):

   ```cpp
   // ---- 7-9 full reductions ------------------------------------------------------

   namespace {

   bool reduce_result_dtype_ok(ScalarType input, ScalarType requested) {
     if (input == requested) {
       return input == ScalarType::Float16 || input == ScalarType::Float32 ||
              input == ScalarType::Float64 || input == ScalarType::Int64;
     }
     // torch's default integer promotion: sum(int32) -> int64.
     return input == ScalarType::Int32 && requested == ScalarType::Int64;
   }

   TensorPtr make_cuda_scalar_result(double value, ScalarType dtype, Device device) {
     TensorPtr out = zeros(std::vector<int64_t>{}, dtype, device);
     if (value != 0.0) {
       out->fill_inplace(value);  // routes to the 7-5 cuda fill hook
     }
     return out;
   }

   }  // namespace

   TensorPtr reduce_sum(const TensorPtr& input, ScalarType dtype) {
     const char* name = "sum";
     guard_grad(*input, name);
     guard_contiguous(*input, name);
     if (!reduce_result_dtype_ok(input->dtype, dtype)) raise_unsupported(name);  // G5
     const double value =
         input->numel() == 0
             ? 0.0
             : launch_reduce_all(tensor_data(*input), input->numel(), /*op=*/0,
                                 dtype_code(input->dtype));
     return make_cuda_scalar_result(value, dtype, input->device);
   }

   TensorPtr reduce_mean(const TensorPtr& input, ScalarType dtype) {
     const char* name = "mean";
     guard_grad(*input, name);
     guard_contiguous(*input, name);
     if (input->dtype != dtype ||
         (dtype != ScalarType::Float16 && dtype != ScalarType::Float32 &&
          dtype != ScalarType::Float64)) {
       raise_unsupported(name);  // G5: mean over ints is a promoting op, not in v1
     }
     if (input->numel() == 0) {
       return make_cuda_scalar_result(std::nan(""), dtype, input->device);
     }
     const double total = launch_reduce_all(tensor_data(*input), input->numel(),
                                            /*op=*/0, dtype_code(input->dtype));
     return make_cuda_scalar_result(total / (double)input->numel(), dtype,
                                    input->device);
   }

   TensorPtr reduce_max(const TensorPtr& input) {
     const char* name = "max";
     guard_grad(*input, name);
     guard_contiguous(*input, name);
     if (input->dtype == ScalarType::Bool) raise_unsupported(name);  // G5
     if (input->numel() == 0) raise_unsupported(name);  // torch errors here too
     const double value = launch_reduce_all(tensor_data(*input), input->numel(),
                                            /*op=*/1, dtype_code(input->dtype));
     return make_cuda_scalar_result(value, input->dtype, input->device);
   }
   ```

   Add `#include <cmath>` next to `#include <cstdint>` at the top of ops.cpp
   (for `std::nan`).

5. Hook the three functions in `cpp/mtorch/core/reductions.cpp` (include block
   §0.5 item 4 first; placement per §0.5). One hook each, pattern:

   ```cpp
     if (input->device.type == DeviceType::CUDA) {
   #if defined(MTORCH_WITH_CUDA)
       return ::mtorch::cuda::reduce_sum(input, dtype);
   #else
       @RAISE-with-op-"sum"@
   #endif
     }
   ```

   (`reduce_mean` passes `(input, dtype)`, `reduce_max` passes `(input)`; op-name
   strings `"mean"` / `"max"`. Match parameter identifiers to the real
   signatures.)

6. Build; oracle slice; parity rows; DEBT UPDATE; bench note:

   ```bash
   python3 setup.py build_ext --inplace
   python3 tests/compat/tools/cuda_oracle_check.py --only 'reduce.*'
   python3 tests/compat/tools/cuda_oracle_check.py --bench --only 'reduce.sum_float32'
   ```

   Note the oracle's reduce input is deliberately 37 x 53 = 1961 elements — not a
   multiple of 256 — so the tree's tail handling is exercised. Parity rows
   (§0.7):

   | case id | op | dtype | tolerance |
   |---|---|---|---|
   | `reduce.sum_f32` | `sum` over (37, 53) | float32 | rtol=1e-4, atol=1e-5 |
   | `reduce.mean_f16` | `mean` | float16 | rtol=1e-2, atol=1e-3 |
   | `reduce.max_f32` | `max` | float32 | exact |
   | `reduce.sum_i32` | `sum` (result dtype int64) | int32 | exact |

   Notes line: `7-9 sum 2048x2048 f32: <ratio>x`.

7. Determinism spot-check (N2 in action — run the same sum 5 times, all outputs
   must be bit-identical):

   ```bash
   python3 - <<'PY'
   import mtorch
   x = mtorch.rand((999, 777)).cuda()
   values = {mtorch.sum(x).cpu().item() for _ in range(5)}
   print("distinct results:", len(values))   # must print 1
   PY
   ```

   (If `mtorch.rand` needs a seed argument in this tree, any fixed tensor works —
   the point is 5 identical launches. `distinct results: 1` is the requirement;
   anything else means an atomic or a data race crept in: STANDARD FAIL.)

**Verification**: STANDARD VERIFY (§0.8) + parity-debt rule + `ORACLE OK`
(`unsupported=0`) for `reduce.*` + Action 7 printing `distinct results: 1`.

**On failure**: STANDARD FAIL. New path: `git clean -fd -- cpp/mtorch/core/cuda/reduce.cu`.
Mark 7-9 **BLOCKED**.

**Commit**:

```bash
git add cpp/mtorch/core/cuda/reduce.cu cpp/mtorch/core/cuda/kernels.h \
        cpp/mtorch/core/cuda/ops.h cpp/mtorch/core/cuda/ops.cpp \
        cpp/mtorch/core/reductions.cpp \
        tests/compat/test_device_parity.py docs/design/baseline/parity-debt.txt
git commit -m "refactor(phase7-9): cuda reductions sum/mean/max (two-pass, deterministic)"
```

Then update PROGRESS.md per §8 (separate commit).

---

## Step 7-10: Op family: softmax / layer_norm / group_norm (row-per-block)

**Goal**: The three normalization ops as *row-per-block* kernels: one thread block
owns one row (softmax row, layer_norm row, or group_norm group), cooperating
through the block-reduce helpers distilled from 7-9. float32/float16 only, float
accumulation.

**Background: row-per-block.** These ops normalize independent slices (rows or
groups): each slice needs its own max/sum/mean/variance. Instead of two global
passes per slice, we launch one block per slice; the block's 256 threads stride
across the slice's columns, then combine via a block-wide reduction, then stride
across again to write normalized outputs. The intermediate (row max, row sum,
mean, variance) never leaves on-chip memory — this is both faster and simpler
than composing global reductions. The helpers `block_reduce_sum_f32` /
`block_reduce_max_f32` below are 7-9's Phase C+D compressed into a warp-first
version: reduce within each warp by shuffle, park one value per warp in shared
memory (32 slots), then let warp 0 reduce those.

**Preconditions**: 7-9 committed; clean tree.

```bash
git status --short   # empty
grep -n '^TensorPtr softmax(' cpp/mtorch/core/normalization.cpp
grep -n '^TensorPtr layer_norm(' cpp/mtorch/core/normalization.cpp
grep -n '^TensorPtr group_norm(' cpp/mtorch/core/normalization.cpp
grep -n 'TensorPtr layer_norm' cpp/mtorch/core/tensor.h   # copy real parameter lists
```

**Actions**:

1. Append to `cpp/mtorch/core/cuda/kernels.h` (before `}  // namespace cuda`):

   ```cpp
   // 7-10 row-per-block normalization (f16/f32 only; gamma/beta may be null).
   void launch_softmax_rows(const void* input_device, void* out_device, int64_t rows,
                            int64_t cols, int dtype_code);
   void launch_layer_norm_rows(const void* input_device, void* out_device,
                               const void* gamma_or_null, const void* beta_or_null,
                               int64_t rows, int64_t cols, double eps, int dtype_code);
   void launch_group_norm(const void* input_device, void* out_device,
                          const void* gamma_or_null, const void* beta_or_null,
                          int64_t batch, int64_t groups, int64_t channels_per_group,
                          int64_t spatial, double eps, int dtype_code);
   ```

2. Create `cpp/mtorch/core/cuda/rowwise.cu` with exactly this content:

   ```cpp
   // Phase 7 step 7-10: softmax / layer_norm / group_norm, one block per row
   // (or group). Reuses the 7-9 reduction pattern as block-wide helpers.
   // Storage f16/f32; ALL arithmetic in float (guide 0.9 N1).

   #include "mtorch/core/cuda/common.h"
   #include "mtorch/core/cuda/dispatch.cuh"
   #include "mtorch/core/cuda/kernels.h"

   #include <cmath>

   namespace mtorch {
   namespace cuda {
   namespace {

   // Block-wide sum of each thread's float. Warp-first version of 7-9 C+D:
   // shuffle-reduce inside each warp, park per-warp results (<= 8 for 256
   // threads) in shared memory, reduce those in warp 0, broadcast via shm[0].
   // The leading __syncthreads() protects shm when the helper is called twice
   // in a row (write-after-read hazard between consecutive calls).
   __device__ float block_reduce_sum_f32(float value) {
     __shared__ float shm[32];
     __syncthreads();
     const int lane = threadIdx.x & 31;
     const int warp = threadIdx.x >> 5;
     for (int offset = 16; offset > 0; offset >>= 1) {
       value += __shfl_down_sync(0xffffffffu, value, offset);
     }
     if (lane == 0) shm[warp] = value;
     __syncthreads();
     value = (threadIdx.x < (blockDim.x >> 5)) ? shm[lane] : 0.0f;
     if (warp == 0) {
       for (int offset = 16; offset > 0; offset >>= 1) {
         value += __shfl_down_sync(0xffffffffu, value, offset);
       }
       if (lane == 0) shm[0] = value;
     }
     __syncthreads();
     return shm[0];
   }

   __device__ float block_reduce_max_f32(float value) {
     __shared__ float shm[32];
     __syncthreads();
     const int lane = threadIdx.x & 31;
     const int warp = threadIdx.x >> 5;
     for (int offset = 16; offset > 0; offset >>= 1) {
       value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
     }
     if (lane == 0) shm[warp] = value;
     __syncthreads();
     value = (threadIdx.x < (blockDim.x >> 5)) ? shm[lane] : -INFINITY;
     if (warp == 0) {
       for (int offset = 16; offset > 0; offset >>= 1) {
         value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
       }
       if (lane == 0) shm[0] = value;
     }
     __syncthreads();
     return shm[0];
   }

   // softmax over the last dim: out = exp(x - rowmax) / sum(exp(x - rowmax)).
   // Subtracting the row max first is the standard overflow guard.
   template <typename scalar_t>
   __global__ void softmax_rows_kernel(const scalar_t* input, scalar_t* out,
                                       int64_t cols) {
     const scalar_t* row_in = input + (int64_t)blockIdx.x * cols;
     scalar_t* row_out = out + (int64_t)blockIdx.x * cols;
     float local_max = -INFINITY;
     for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
       local_max = fmaxf(local_max, (float)row_in[c]);
     }
     const float row_max = block_reduce_max_f32(local_max);
     float local_sum = 0.0f;
     for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
       local_sum += expf((float)row_in[c] - row_max);
     }
     const float inv_sum = 1.0f / block_reduce_sum_f32(local_sum);
     for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
       row_out[c] = (scalar_t)(expf((float)row_in[c] - row_max) * inv_sum);
     }
   }

   // layer_norm over the trailing `cols` elements of each row, with optional
   // per-element affine (gamma/beta of length cols). Biased variance (divide by
   // cols, not cols-1) — same as torch.nn.functional.layer_norm.
   template <typename scalar_t>
   __global__ void layer_norm_rows_kernel(const scalar_t* input, scalar_t* out,
                                          const scalar_t* gamma,
                                          const scalar_t* beta, int64_t cols,
                                          float eps) {
     const scalar_t* row_in = input + (int64_t)blockIdx.x * cols;
     scalar_t* row_out = out + (int64_t)blockIdx.x * cols;
     float local_sum = 0.0f;
     float local_sumsq = 0.0f;
     for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
       const float x = (float)row_in[c];
       local_sum += x;
       local_sumsq += x * x;
     }
     const float mean = block_reduce_sum_f32(local_sum) / (float)cols;
     const float sumsq = block_reduce_sum_f32(local_sumsq);
     // var = E[x^2] - mean^2 (one-pass; adequate in fp32 for v1 input scales).
     const float variance = sumsq / (float)cols - mean * mean;
     const float inv_std = rsqrtf(variance + eps);
     for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
       const float normalized = ((float)row_in[c] - mean) * inv_std;
       const float g = gamma != nullptr ? (float)gamma[c] : 1.0f;
       const float b = beta != nullptr ? (float)beta[c] : 0.0f;
       row_out[c] = (scalar_t)(normalized * g + b);
     }
   }

   // group_norm: block (n, g) normalizes the contiguous chunk of
   // channels_per_group * spatial elements; affine is PER CHANNEL, so the
   // channel index is recovered from the position inside the chunk.
   template <typename scalar_t>
   __global__ void group_norm_kernel(const scalar_t* input, scalar_t* out,
                                     const scalar_t* gamma, const scalar_t* beta,
                                     int64_t groups, int64_t channels_per_group,
                                     int64_t spatial, float eps) {
     const int64_t group_size = channels_per_group * spatial;
     const scalar_t* group_in = input + (int64_t)blockIdx.x * group_size;
     scalar_t* group_out = out + (int64_t)blockIdx.x * group_size;
     const int64_t group_index = blockIdx.x % groups;
     float local_sum = 0.0f;
     float local_sumsq = 0.0f;
     for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
       const float x = (float)group_in[i];
       local_sum += x;
       local_sumsq += x * x;
     }
     const float mean = block_reduce_sum_f32(local_sum) / (float)group_size;
     const float sumsq = block_reduce_sum_f32(local_sumsq);
     const float variance = sumsq / (float)group_size - mean * mean;
     const float inv_std = rsqrtf(variance + eps);
     for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
       const int64_t channel =
           group_index * channels_per_group + i / spatial;  // global channel id
       const float normalized = ((float)group_in[i] - mean) * inv_std;
       const float g = gamma != nullptr ? (float)gamma[channel] : 1.0f;
       const float b = beta != nullptr ? (float)beta[channel] : 0.0f;
       group_out[i] = (scalar_t)(normalized * g + b);
     }
   }

   // f16/f32-only dispatch used by the three launchers below.
   #define MTORCH_ROWWISE_DISPATCH(DTYPE_CODE, NAME, ...)                        \
     switch (DTYPE_CODE) {                                                       \
       case ::mtorch::cuda::kF32: { using scalar_t = float; __VA_ARGS__; break; }\
       case ::mtorch::cuda::kF16: { using scalar_t = __half; __VA_ARGS__; break; }\
       default: throw std::runtime_error(std::string(NAME) + ": dtype");         \
     }

   }  // namespace

   void launch_softmax_rows(const void* input_device, void* out_device, int64_t rows,
                            int64_t cols, int dtype_code) {
     if (rows == 0 || cols == 0) return;
     MTORCH_ROWWISE_DISPATCH(dtype_code, "launch_softmax_rows",
       softmax_rows_kernel<scalar_t><<<(unsigned)rows, 256>>>(
           reinterpret_cast<const scalar_t*>(input_device),
           reinterpret_cast<scalar_t*>(out_device), cols);
       MTORCH_CUDA_LAUNCH_CHECK();
     );
   }

   void launch_layer_norm_rows(const void* input_device, void* out_device,
                               const void* gamma_or_null, const void* beta_or_null,
                               int64_t rows, int64_t cols, double eps, int dtype_code) {
     if (rows == 0 || cols == 0) return;
     MTORCH_ROWWISE_DISPATCH(dtype_code, "launch_layer_norm_rows",
       layer_norm_rows_kernel<scalar_t><<<(unsigned)rows, 256>>>(
           reinterpret_cast<const scalar_t*>(input_device),
           reinterpret_cast<scalar_t*>(out_device),
           reinterpret_cast<const scalar_t*>(gamma_or_null),
           reinterpret_cast<const scalar_t*>(beta_or_null), cols, (float)eps);
       MTORCH_CUDA_LAUNCH_CHECK();
     );
   }

   void launch_group_norm(const void* input_device, void* out_device,
                          const void* gamma_or_null, const void* beta_or_null,
                          int64_t batch, int64_t groups, int64_t channels_per_group,
                          int64_t spatial, double eps, int dtype_code) {
     if (batch == 0 || groups == 0 || channels_per_group * spatial == 0) return;
     MTORCH_ROWWISE_DISPATCH(dtype_code, "launch_group_norm",
       group_norm_kernel<scalar_t><<<(unsigned)(batch * groups), 256>>>(
           reinterpret_cast<const scalar_t*>(input_device),
           reinterpret_cast<scalar_t*>(out_device),
           reinterpret_cast<const scalar_t*>(gamma_or_null),
           reinterpret_cast<const scalar_t*>(beta_or_null), groups,
           channels_per_group, spatial, (float)eps);
       MTORCH_CUDA_LAUNCH_CHECK();
     );
   }

   }  // namespace cuda
   }  // namespace mtorch
   ```

3. Append to `cpp/mtorch/core/cuda/ops.h` (parameter lists copied from
   `tensor.h`, verified 2026-07-09; re-check with the Preconditions grep):

   ```cpp
   // 7-10 normalization.
   TensorPtr softmax(const TensorPtr& input, int64_t dim, ScalarType dtype);
   TensorPtr layer_norm(const TensorPtr& input,
                        const std::vector<int64_t>& normalized_shape,
                        const TensorPtr& weight, const TensorPtr& bias, double eps);
   TensorPtr group_norm(const TensorPtr& input, int64_t num_groups,
                        const TensorPtr& weight, const TensorPtr& bias, double eps);
   ```

4. Append to `cpp/mtorch/core/cuda/ops.cpp` (before the final
   `}  // namespace cuda`):

   ```cpp
   // ---- 7-10 normalization ---------------------------------------------------------

   namespace {

   void guard_float16_or_32(const Tensor& t, const char* op) {
     if (t.dtype != ScalarType::Float16 && t.dtype != ScalarType::Float32) {
       raise_unsupported(op);  // G5
     }
   }

   // Optional affine parameter: if present it must be cuda, contiguous, same
   // dtype as input, with exactly `expected_numel` elements. Returns the device
   // pointer or nullptr.
   const void* affine_data_or_null(const TensorPtr& parameter, const Tensor& input,
                                   int64_t expected_numel, const char* op) {
     if (parameter == nullptr) return nullptr;
     if (!on_cuda(*parameter)) raise_unsupported(op);          // G4
     guard_grad(*parameter, op);                               // G1
     guard_contiguous(*parameter, op);                         // G2
     if (parameter->dtype != input.dtype) raise_unsupported(op);
     if (parameter->numel() != expected_numel) raise_unsupported(op);
     return tensor_data(*parameter);
   }

   }  // namespace

   TensorPtr softmax(const TensorPtr& input, int64_t dim, ScalarType dtype) {
     const char* name = "softmax";
     guard_grad(*input, name);
     guard_contiguous(*input, name);
     guard_float16_or_32(*input, name);
     if (dtype != input->dtype) raise_unsupported(name);  // v1: no dtype override
     const int64_t rank = (int64_t)input->sizes.size();
     if (rank == 0) raise_unsupported(name);
     const int64_t normalized_dim = dim < 0 ? dim + rank : dim;
     if (normalized_dim != rank - 1) raise_unsupported(name);  // G6: last dim only
     const int64_t cols = input->sizes[rank - 1];
     const int64_t rows = cols == 0 ? 0 : input->numel() / cols;
     TensorPtr out = make_cuda_result_like(*input);
     launch_softmax_rows(tensor_data(*input), tensor_data(*out), rows, cols,
                         dtype_code(input->dtype));
     return out;
   }

   TensorPtr layer_norm(const TensorPtr& input,
                        const std::vector<int64_t>& normalized_shape,
                        const TensorPtr& weight, const TensorPtr& bias, double eps) {
     const char* name = "layer_norm";
     guard_grad(*input, name);
     guard_contiguous(*input, name);
     guard_float16_or_32(*input, name);
     // G6: normalized_shape must equal the trailing sizes of input.
     const int64_t rank = (int64_t)input->sizes.size();
     const int64_t norm_rank = (int64_t)normalized_shape.size();
     if (norm_rank == 0 || norm_rank > rank) raise_unsupported(name);
     int64_t cols = 1;
     for (int64_t i = 0; i < norm_rank; ++i) {
       if (input->sizes[rank - norm_rank + i] != normalized_shape[i]) {
         raise_unsupported(name);
       }
       cols *= normalized_shape[i];
     }
     const int64_t rows = cols == 0 ? 0 : input->numel() / cols;
     const void* gamma = affine_data_or_null(weight, *input, cols, name);
     const void* beta = affine_data_or_null(bias, *input, cols, name);
     TensorPtr out = make_cuda_result_like(*input);
     launch_layer_norm_rows(tensor_data(*input), tensor_data(*out), gamma, beta,
                            rows, cols, eps, dtype_code(input->dtype));
     return out;
   }

   TensorPtr group_norm(const TensorPtr& input, int64_t num_groups,
                        const TensorPtr& weight, const TensorPtr& bias, double eps) {
     const char* name = "group_norm";
     guard_grad(*input, name);
     guard_contiguous(*input, name);
     guard_float16_or_32(*input, name);
     if ((int64_t)input->sizes.size() < 2) raise_unsupported(name);  // (N, C, ...)
     const int64_t channels = input->sizes[1];
     if (num_groups <= 0 || channels % num_groups != 0) raise_unsupported(name);
     int64_t spatial = 1;
     for (size_t i = 2; i < input->sizes.size(); ++i) spatial *= input->sizes[i];
     const void* gamma = affine_data_or_null(weight, *input, channels, name);
     const void* beta = affine_data_or_null(bias, *input, channels, name);
     TensorPtr out = make_cuda_result_like(*input);
     launch_group_norm(tensor_data(*input), tensor_data(*out), gamma, beta,
                       input->sizes[0], num_groups, channels / num_groups, spatial,
                       eps, dtype_code(input->dtype));
     return out;
   }
   ```

5. Hook the three functions in `cpp/mtorch/core/normalization.cpp` (include block
   §0.5 item 4 first; placement per §0.5); pattern for `softmax` (repeat with the
   right arguments and op names `"layer_norm"` / `"group_norm"` for the others,
   matching each function's real parameter identifiers):

   ```cpp
     if (input->device.type == DeviceType::CUDA) {
   #if defined(MTORCH_WITH_CUDA)
       return ::mtorch::cuda::softmax(input, dim, dtype);
   #else
       @RAISE-with-op-"softmax"@
   #endif
     }
   ```

   (`log_softmax` is deliberately NOT hooked in v1.)

6. Build; oracle slice; parity rows; DEBT UPDATE; bench note:

   ```bash
   python3 setup.py build_ext --inplace
   python3 tests/compat/tools/cuda_oracle_check.py --only 'norm.*'
   python3 tests/compat/tools/cuda_oracle_check.py --bench --only 'norm.softmax_float32'
   ```

   Parity rows (§0.7):

   | case id | op | dtype | tolerance |
   |---|---|---|---|
   | `norm.softmax_f32` | `softmax(x, dim=-1)`, x (16, 33) | float32 | rtol=1e-4, atol=1e-5 |
   | `norm.softmax_f16` | same | float16 | rtol=1e-2, atol=1e-3 |
   | `norm.layer_norm_f32` | `layer_norm(x, [24], w, b)`, x (8, 24) | float32 | rtol=1e-4, atol=1e-5 |
   | `norm.layer_norm_nw_f32` | `layer_norm(x, [24])` (no affine) | float32 | rtol=1e-4, atol=1e-5 |
   | `norm.group_norm_f32` | `group_norm(x, 4, w, b)`, x (2, 8, 5, 5) | float32 | rtol=1e-4, atol=1e-5 |

   Notes line: `7-10 softmax 512x1024 f32: <ratio>x`.

**Verification**: STANDARD VERIFY (§0.8) + parity-debt rule + `ORACLE OK`
(`unsupported=0`) for `norm.*`. Cols of 33 (not a multiple of 32) in the softmax
cases exercise the partial-warp tail of the block reduce on purpose.

**On failure**: STANDARD FAIL. New path: `git clean -fd -- cpp/mtorch/core/cuda/rowwise.cu`.
Mark 7-10 **BLOCKED**.

**Commit**:

```bash
git add cpp/mtorch/core/cuda/rowwise.cu cpp/mtorch/core/cuda/kernels.h \
        cpp/mtorch/core/cuda/ops.h cpp/mtorch/core/cuda/ops.cpp \
        cpp/mtorch/core/normalization.cpp \
        tests/compat/test_device_parity.py docs/design/baseline/parity-debt.txt
git commit -m "refactor(phase7-10): cuda softmax/layer_norm/group_norm (row-per-block)"
```

Then update PROGRESS.md per §8 (separate commit).

---

## Step 7-11: Op family: conv2d (im2col + cuBLAS matmul)

**Goal**: `conv2d` for NCHW float32/float16, `groups == 1`, via the classic
*im2col + GEMM* lowering, reusing 7-8's `gemm_rm`.

**Background: im2col.** A convolution is a GEMM in disguise: every output pixel is
a dot product between the flattened filter (length `C*kh*kw`) and the flattened
input patch under it. *im2col* materializes those patches as a matrix `col` of
shape `(C*kh*kw) x (out_h*out_w)` — one column per output position, zero-filled
where the patch hangs over the padding — after which the whole convolution for one
image is a single GEMM: `out(O, out_h*out_w) = weight(O, C*kh*kw) x col`. It costs
extra memory (the col buffer) and bandwidth, but it is simple, deterministic
(rule N2), and inherits cuBLAS's correctness. This is exactly how the CPU core's
own fast path works (`conv2d_fill_im2col_float32` — see `cpp/mtorch/core/conv.cpp`),
so the structure will look familiar.

**Preconditions**: 7-10 committed; clean tree.

```bash
git status --short   # empty
grep -n '^TensorPtr conv2d(' cpp/mtorch/core/conv.cpp
grep -n 'TensorPtr conv2d' cpp/mtorch/core/tensor.h   # copy the real parameter list
```

**Actions**:

1. Append to `cpp/mtorch/core/cuda/kernels.h` (before `}  // namespace cuda`):

   ```cpp
   // 7-11 conv2d lowering (f16/f32).
   // im2col for ONE image: input (channels x height x width) contiguous ->
   // col ((channels*kernel_h*kernel_w) x (out_h*out_w)), zero-padded.
   void launch_im2col(const void* image_device, void* col_device, int64_t channels,
                      int64_t height, int64_t width, int64_t kernel_h, int64_t kernel_w,
                      int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w,
                      int64_t dilation_h, int64_t dilation_w, int64_t out_h,
                      int64_t out_w, int dtype_code);
   // out(channels x inner) += bias[channel], elementwise over a contiguous block.
   void launch_add_bias(void* out_device, const void* bias_device, int64_t channels,
                        int64_t inner, int dtype_code);
   ```

2. Create `cpp/mtorch/core/cuda/im2col.cu` with exactly this content:

   ```cpp
   // Phase 7 step 7-11: im2col + bias-add kernels (the GEMM half of conv2d is
   // cuBLAS via blas.cpp). Standard im2col indexing, one thread per
   // (channel, out_y, out_x) triple; each thread writes its kernel_h*kernel_w
   // column entries. Padding reads become zeros.

   #include "mtorch/core/cuda/common.h"
   #include "mtorch/core/cuda/dispatch.cuh"
   #include "mtorch/core/cuda/kernels.h"

   namespace mtorch {
   namespace cuda {
   namespace {

   template <typename scalar_t>
   __global__ void im2col_kernel(int64_t total, const scalar_t* image, int64_t height,
                                 int64_t width, int64_t kernel_h, int64_t kernel_w,
                                 int64_t pad_h, int64_t pad_w, int64_t stride_h,
                                 int64_t stride_w, int64_t dilation_h,
                                 int64_t dilation_w, int64_t out_h, int64_t out_w,
                                 scalar_t* col) {
     // total = channels * out_h * out_w. Decode i -> (channel, out_y, out_x).
     for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total;
          i += (int64_t)gridDim.x * blockDim.x) {
       const int64_t out_x = i % out_w;
       const int64_t out_y = (i / out_w) % out_h;
       const int64_t channel = i / (out_w * out_h);
       // This thread's column position, walking down the rows that belong to
       // `channel` (rows channel*kh*kw .. channel*kh*kw + kh*kw - 1):
       scalar_t* col_ptr =
           col + (channel * kernel_h * kernel_w) * (out_h * out_w) + out_y * out_w + out_x;
       const scalar_t* image_channel = image + channel * height * width;
       const int64_t in_y0 = out_y * stride_h - pad_h;
       const int64_t in_x0 = out_x * stride_w - pad_w;
       for (int64_t ky = 0; ky < kernel_h; ++ky) {
         const int64_t in_y = in_y0 + ky * dilation_h;
         for (int64_t kx = 0; kx < kernel_w; ++kx) {
           const int64_t in_x = in_x0 + kx * dilation_w;
           const bool inside =
               in_y >= 0 && in_y < height && in_x >= 0 && in_x < width;
           *col_ptr = inside ? image_channel[in_y * width + in_x] : (scalar_t)0.0f;
           col_ptr += out_h * out_w;  // next kernel element = next col row
         }
       }
     }
   }

   template <typename scalar_t>
   __global__ void add_bias_kernel(scalar_t* out, const scalar_t* bias,
                                   int64_t total, int64_t inner) {
     for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total;
          i += (int64_t)gridDim.x * blockDim.x) {
       // float accumulation even for __half storage (guide 0.9 N1).
       out[i] = (scalar_t)((float)out[i] + (float)bias[i / inner]);
     }
   }

   // f16/f32-only dispatch (same shape as 7-10's).
   #define MTORCH_CONV_DISPATCH(DTYPE_CODE, NAME, ...)                           \
     switch (DTYPE_CODE) {                                                       \
       case ::mtorch::cuda::kF32: { using scalar_t = float; __VA_ARGS__; break; }\
       case ::mtorch::cuda::kF16: { using scalar_t = __half; __VA_ARGS__; break; }\
       default: throw std::runtime_error(std::string(NAME) + ": dtype");         \
     }

   }  // namespace

   void launch_im2col(const void* image_device, void* col_device, int64_t channels,
                      int64_t height, int64_t width, int64_t kernel_h, int64_t kernel_w,
                      int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w,
                      int64_t dilation_h, int64_t dilation_w, int64_t out_h,
                      int64_t out_w, int dtype_code) {
     const int64_t total = channels * out_h * out_w;
     if (total == 0) return;
     const int blocks = grid_blocks_for(total);
     MTORCH_CONV_DISPATCH(dtype_code, "launch_im2col",
       im2col_kernel<scalar_t><<<blocks, 256>>>(
           total, reinterpret_cast<const scalar_t*>(image_device), height, width,
           kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
           dilation_w, out_h, out_w, reinterpret_cast<scalar_t*>(col_device));
       MTORCH_CUDA_LAUNCH_CHECK();
     );
   }

   void launch_add_bias(void* out_device, const void* bias_device, int64_t channels,
                        int64_t inner, int dtype_code) {
     const int64_t total = channels * inner;
     if (total == 0) return;
     const int blocks = grid_blocks_for(total);
     MTORCH_CONV_DISPATCH(dtype_code, "launch_add_bias",
       add_bias_kernel<scalar_t><<<blocks, 256>>>(
           reinterpret_cast<scalar_t*>(out_device),
           reinterpret_cast<const scalar_t*>(bias_device), total, inner);
       MTORCH_CUDA_LAUNCH_CHECK();
     );
   }

   }  // namespace cuda
   }  // namespace mtorch
   ```

3. Append to `cpp/mtorch/core/cuda/ops.h` (parameter list copied from
   `tensor.h`; verified 2026-07-09):

   ```cpp
   // 7-11 conv2d (NCHW, groups == 1).
   TensorPtr conv2d(const TensorPtr& input, const TensorPtr& weight,
                    const TensorPtr& bias, const std::vector<int64_t>& stride,
                    const std::vector<int64_t>& padding,
                    const std::vector<int64_t>& dilation, int64_t groups);
   ```

4. Append to `cpp/mtorch/core/cuda/ops.cpp` (before the final
   `}  // namespace cuda`):

   ```cpp
   // ---- 7-11 conv2d ------------------------------------------------------------

   TensorPtr conv2d(const TensorPtr& input, const TensorPtr& weight,
                    const TensorPtr& bias, const std::vector<int64_t>& stride,
                    const std::vector<int64_t>& padding,
                    const std::vector<int64_t>& dilation, int64_t groups) {
     const char* name = "conv2d";
     guard_grad(*input, name); guard_grad(*weight, name);
     if (!on_cuda(*input) || !on_cuda(*weight)) raise_unsupported(name);   // G4
     guard_contiguous(*input, name); guard_contiguous(*weight, name);      // G2
     guard_float16_or_32(*input, name);                                    // G5
     if (weight->dtype != input->dtype) raise_unsupported(name);
     if (groups != 1) raise_unsupported(name);                             // G6
     if (input->sizes.size() != 4 || weight->sizes.size() != 4) raise_unsupported(name);
     if (stride.size() != 2 || padding.size() != 2 || dilation.size() != 2) {
       raise_unsupported(name);
     }
     if (weight->sizes[1] != input->sizes[1]) raise_unsupported(name);
     const int64_t batch = input->sizes[0];
     const int64_t channels = input->sizes[1];
     const int64_t height = input->sizes[2];
     const int64_t width = input->sizes[3];
     const int64_t out_channels = weight->sizes[0];
     const int64_t kernel_h = weight->sizes[2];
     const int64_t kernel_w = weight->sizes[3];
     const int64_t out_h =
         (height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) / stride[0] + 1;
     const int64_t out_w =
         (width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) / stride[1] + 1;
     if (out_h <= 0 || out_w <= 0) raise_unsupported(name);
     const void* bias_data = affine_data_or_null(bias, *input, out_channels, name);

     TensorPtr out =
         zeros({batch, out_channels, out_h, out_w}, input->dtype, input->device);
     const int64_t k = channels * kernel_h * kernel_w;     // GEMM inner dim
     const int64_t inner = out_h * out_w;                  // GEMM n dim
     const int64_t element = element_size(input->dtype);
     // Scratch col buffer through the caching allocator (7-3): k x inner elems.
     void* col = cuda_allocator().allocate(k * inner * element);
     uint8_t* input_bytes = static_cast<uint8_t*>(tensor_data(*input));
     uint8_t* out_bytes = static_cast<uint8_t*>(tensor_data(*out));
     const int64_t image_bytes = channels * height * width * element;
     const int64_t out_image_bytes = out_channels * inner * element;
     for (int64_t n = 0; n < batch; ++n) {
       launch_im2col(input_bytes + n * image_bytes, col, channels, height, width,
                     kernel_h, kernel_w, padding[0], padding[1], stride[0], stride[1],
                     dilation[0], dilation[1], out_h, out_w, dtype_code(input->dtype));
       // out_n(out_channels x inner) = weight(out_channels x k) x col(k x inner)
       gemm_rm(dtype_code(input->dtype), /*trans_a=*/false, /*trans_b=*/false,
               out_channels, inner, k, /*alpha=*/1.0, tensor_data(*weight), col,
               /*beta=*/0.0, out_bytes + n * out_image_bytes);
       if (bias_data != nullptr) {
         launch_add_bias(out_bytes + n * out_image_bytes, bias_data, out_channels,
                         inner, dtype_code(input->dtype));
       }
     }
     cuda_allocator().deallocate(col);
     return out;
   }
   ```

   (`guard_float16_or_32` and `affine_data_or_null` exist since 7-10; they are in
   an anonymous namespace earlier in this same file, so they are visible here.)

5. Hook `conv2d` in `cpp/mtorch/core/conv.cpp` (include block §0.5 item 4 first;
   placement per §0.5; match the real parameter identifiers):

   ```cpp
     if (input->device.type == DeviceType::CUDA || weight->device.type == DeviceType::CUDA) {
   #if defined(MTORCH_WITH_CUDA)
       return ::mtorch::cuda::conv2d(input, weight, bias, stride, padding, dilation, groups);
   #else
       @RAISE-with-op-"conv2d"@
   #endif
     }
   ```

6. Build; oracle slice; parity rows; DEBT UPDATE; bench note:

   ```bash
   python3 setup.py build_ext --inplace
   python3 tests/compat/tools/cuda_oracle_check.py --only 'conv.*'
   python3 tests/compat/tools/cuda_oracle_check.py --bench --only 'conv.conv2d_float32'
   ```

   Parity rows (§0.7):

   | case id | op | dtype | tolerance |
   |---|---|---|---|
   | `conv.conv2d_f32` | `conv2d(x, w, b, stride=1, padding=1)`, x (2, 3, 12, 12), w (6, 3, 3, 3) | float32 | rtol=1e-4, atol=1e-5 |
   | `conv.conv2d_f16` | same | float16 | rtol=1e-2, atol=1e-3 |
   | `conv.conv2d_stride2_f32` | stride=(2, 2), padding=(0, 0) | float32 | rtol=1e-4, atol=1e-5 |
   | `conv.conv2d_nobias_f32` | bias omitted | float32 | rtol=1e-4, atol=1e-5 |

   Notes line: `7-11 conv2d 8x32x64x64 f32: <ratio>x` (torch uses cuDNN here; a
   large ratio is expected and fine — correctness is the gate).

**Verification**: STANDARD VERIFY (§0.8) + parity-debt rule + `ORACLE OK`
(`unsupported=0`) for `conv.*`.

**On failure**: STANDARD FAIL. New path: `git clean -fd -- cpp/mtorch/core/cuda/im2col.cu`.
Mark 7-11 **BLOCKED**.

**Commit**:

```bash
git add cpp/mtorch/core/cuda/im2col.cu cpp/mtorch/core/cuda/kernels.h \
        cpp/mtorch/core/cuda/ops.h cpp/mtorch/core/cuda/ops.cpp \
        cpp/mtorch/core/conv.cpp \
        tests/compat/test_device_parity.py docs/design/baseline/parity-debt.txt
git commit -m "refactor(phase7-11): cuda conv2d via im2col + cuBLAS"
```

Then update PROGRESS.md per §8 (separate commit).

---

## Step 7-12: Op family: scaled_dot_product_attention (composed)

**Goal**: SDPA for rank-4 `(B, H, L, D)` float32/float16, **composed entirely from
already-verified pieces**: two `gemm_rm` calls (7-8) and the softmax row kernel
(7-10). No new kernel is written in this step — that is the point: after the
building blocks are proven, attention is plumbing.

`out[b,h] = softmax(Q[b,h] x K[b,h]^T * scale) x V[b,h]`, scale defaulting to
`1/sqrt(D)`.

v1 guards (G6): `attn_mask == nullptr`, `dropout_p == 0.0`, `is_causal == false`,
`enable_gqa == false`, q/k/v same shape rank-4 — anything else raises. (Causal and
masked variants belong to the guard-lifting phase, same as broadcasting.)

**Preconditions**: 7-11 committed; clean tree.

```bash
git status --short   # empty
grep -n '^TensorPtr scaled_dot_product_attention(' cpp/mtorch/core/attention.cpp
grep -n 'TensorPtr scaled_dot_product_attention' cpp/mtorch/core/tensor.h   # real parameter list
```

**Actions**:

1. Append to `cpp/mtorch/core/cuda/ops.h` (parameter list copied from `tensor.h`,
   verified 2026-07-09 — note the defaults live in tensor.h only, not here):

   ```cpp
   // 7-12 scaled_dot_product_attention (rank-4, no mask/dropout/causal/gqa).
   TensorPtr scaled_dot_product_attention(const TensorPtr& query, const TensorPtr& key,
                                          const TensorPtr& value,
                                          const TensorPtr& attn_mask, double dropout_p,
                                          bool is_causal, std::optional<double> scale,
                                          bool enable_gqa);
   ```

   Add `#include <optional>` under the existing includes of ops.h if not present.

2. Append to `cpp/mtorch/core/cuda/ops.cpp` (before the final
   `}  // namespace cuda`); add `#include <cmath>` already present since 7-9:

   ```cpp
   // ---- 7-12 scaled_dot_product_attention ------------------------------------------

   TensorPtr scaled_dot_product_attention(const TensorPtr& query, const TensorPtr& key,
                                          const TensorPtr& value,
                                          const TensorPtr& attn_mask, double dropout_p,
                                          bool is_causal, std::optional<double> scale,
                                          bool enable_gqa) {
     const char* name = "scaled_dot_product_attention";
     // G6 first: reject every variant v1 does not do.
     if (attn_mask != nullptr || dropout_p != 0.0 || is_causal || enable_gqa) {
       raise_unsupported(name);
     }
     guard_grad(*query, name); guard_grad(*key, name); guard_grad(*value, name);
     if (!on_cuda(*query) || !on_cuda(*key) || !on_cuda(*value)) raise_unsupported(name);
     guard_contiguous(*query, name); guard_contiguous(*key, name);
     guard_contiguous(*value, name);
     guard_float16_or_32(*query, name);
     if (key->dtype != query->dtype || value->dtype != query->dtype) {
       raise_unsupported(name);
     }
     if (query->sizes.size() != 4 || key->sizes != query->sizes ||
         value->sizes != query->sizes) {
       raise_unsupported(name);  // G3: identical (B, H, L, D) for q/k/v in v1
     }
     const int64_t batch = query->sizes[0];
     const int64_t heads = query->sizes[1];
     const int64_t length = query->sizes[2];
     const int64_t depth = query->sizes[3];
     const double scale_value =
         scale.has_value() ? *scale : 1.0 / std::sqrt((double)depth);

     TensorPtr out = zeros(query->sizes, query->dtype, query->device);
     const int64_t element = element_size(query->dtype);
     const int64_t head_bytes = length * depth * element;
     const int64_t scores_elems = length * length;
     const int64_t plans = batch * heads;
     // One big scores buffer for all (b, h) pairs -> a single softmax launch
     // over plans*length rows afterwards (row kernel reused from 7-10).
     void* scores = cuda_allocator().allocate(plans * scores_elems * element);
     uint8_t* q_bytes = static_cast<uint8_t*>(tensor_data(*query));
     uint8_t* k_bytes = static_cast<uint8_t*>(tensor_data(*key));
     uint8_t* v_bytes = static_cast<uint8_t*>(tensor_data(*value));
     uint8_t* out_bytes = static_cast<uint8_t*>(tensor_data(*out));
     uint8_t* scores_bytes = static_cast<uint8_t*>(scores);

     // Pass 1 (7-8 gemm, trans_b): scores[p] = scale * Q[p] x K[p]^T.
     for (int64_t p = 0; p < plans; ++p) {
       gemm_rm(dtype_code(query->dtype), /*trans_a=*/false, /*trans_b=*/true,
               length, length, depth, scale_value, q_bytes + p * head_bytes,
               k_bytes + p * head_bytes, /*beta=*/0.0,
               scores_bytes + p * scores_elems * element);
     }
     // Pass 2 (7-10 kernel): softmax over the last dim of every score row.
     // In-place is safe: each element is read once more, then overwritten by
     // the same thread in the same loop iteration (see rowwise.cu).
     launch_softmax_rows(scores, scores, plans * length, length,
                         dtype_code(query->dtype));
     // Pass 3 (7-8 gemm): out[p] = probs[p] x V[p].
     for (int64_t p = 0; p < plans; ++p) {
       gemm_rm(dtype_code(query->dtype), /*trans_a=*/false, /*trans_b=*/false,
               length, depth, length, /*alpha=*/1.0,
               scores_bytes + p * scores_elems * element, v_bytes + p * head_bytes,
               /*beta=*/0.0, out_bytes + p * head_bytes);
     }
     cuda_allocator().deallocate(scores);
     return out;
   }
   ```

3. Hook it in `cpp/mtorch/core/attention.cpp` (include block §0.5 item 4 first;
   placement per §0.5; identifiers per the real signature):

   ```cpp
     if (query->device.type == DeviceType::CUDA || key->device.type == DeviceType::CUDA ||
         value->device.type == DeviceType::CUDA) {
   #if defined(MTORCH_WITH_CUDA)
       return ::mtorch::cuda::scaled_dot_product_attention(
           query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
   #else
       @RAISE-with-op-"scaled_dot_product_attention"@
   #endif
     }
   ```

4. Build; oracle slice; parity rows; DEBT UPDATE; bench note:

   ```bash
   python3 setup.py build_ext --inplace
   python3 tests/compat/tools/cuda_oracle_check.py --only 'sdpa.*'
   python3 tests/compat/tools/cuda_oracle_check.py --bench --only 'sdpa.basic_float32'
   ```

   Parity rows (§0.7):

   | case id | op | dtype | tolerance |
   |---|---|---|---|
   | `sdpa.basic_f32` | `scaled_dot_product_attention(q, k, v)`, all (2, 4, 16, 8) | float32 | rtol=1e-4, atol=1e-5 |
   | `sdpa.basic_f16` | same | float16 | rtol=1e-2, atol=1e-3 |
   | `sdpa.scale_f32` | same with `scale=0.5` | float32 | rtol=1e-4, atol=1e-5 |

   Notes line: `7-12 sdpa 4x8x256x64 f32: <ratio>x` (torch dispatches to
   flash-attention kernels; expect a big ratio, it is fine).

5. After this step the parity debt must be fully paid:

   ```bash
   wc -l docs/design/baseline/parity-debt.txt   # must print 0
   ```

   If any line remains, some cuda parity row exists for an op family the plan
   never scheduled (i.e. Phase 5 seeded a row outside the eight families). That
   is a plan inconsistency: mark 7-12 **BLOCKED** and list the leftover ids in
   the Notes — do NOT implement extra ops ad hoc to empty the file.

**Verification**: STANDARD VERIFY (§0.8) + `ORACLE OK` (`unsupported=0`) for
`sdpa.*` + Action 5 printing `0`.

**On failure**: STANDARD FAIL. Mark 7-12 **BLOCKED**.

**Commit**:

```bash
git add cpp/mtorch/core/cuda/ops.h cpp/mtorch/core/cuda/ops.cpp \
        cpp/mtorch/core/attention.cpp \
        tests/compat/test_device_parity.py docs/design/baseline/parity-debt.txt
git commit -m "refactor(phase7-12): cuda scaled_dot_product_attention (composed)"
```

Then update PROGRESS.md per §8 (separate commit).

---

## Step 7-13: CUDA ratio table + Linux phase gate

**Goal**: Record the performance ratio table and pass the whole-phase gate on the
Linux machine. Nothing is implemented here; this is the phase's exit exam.

**Preconditions**: 7-12 committed; clean tree.

**Actions**:

1. The full oracle must be perfectly green — zero unsupported, zero failed:

   ```bash
   python3 tests/compat/tools/cuda_oracle_check.py | tail -3
   ```

   Required final line: `ORACLE OK (<n> checks, 0 unsupported)`.

2. Record the ratio table (exact command; the script prints markdown):

   ```bash
   python3 tests/compat/tools/cuda_oracle_check.py --bench > docs/design/baseline/cuda-ratios.md
   cat docs/design/baseline/cuda-ratios.md
   ```

   Every row must show a numeric ratio (no `UNSUPPORTED` rows left). Ratios > 1
   (mtorch slower) are expected everywhere and gate nothing; they are the v1
   optimization worklist.

3. The full Linux gate, all four legs:

   ```bash
   pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
   tail -1 docs/design/baseline/tests-baseline-linux.txt
   pytest tests/compat/test_device_parity.py -q | tail -3
   test ! -s docs/design/baseline/parity-debt.txt && echo DEBT-EMPTY
   ```

   PASS iff the first two summaries have identical counts, the parity suite
   reports 0 failed / 0 error, and `DEBT-EMPTY` prints.

4. Determinism re-check (rule N2, whole-phase): run the 7-9 Action 7 snippet once
   more; `distinct results: 1`.

**Verification**: Actions 1–4 all pass.

**On failure**: STANDARD FAIL; `git clean -fd -- docs/design/baseline/cuda-ratios.md`.
Mark 7-13 **BLOCKED**.

**Commit**:

```bash
git add docs/design/baseline/cuda-ratios.md
git commit -m "refactor(phase7-13): record CUDA ratio table; Linux phase gate green"
```

Then update PROGRESS.md per §8 (separate commit). **After the PROGRESS commit, the
Linux machine's work is done.** The human maintainer moves the branch back to the
Mac (push/pull is the human's job — Iron Rule 6).

---

## Step 7-14: macOS regression check (back on the Mac; changes nothing)

**Goal**: Prove Phase 7 left the macOS tree exactly as healthy as Phase 6 left it:
same build behavior (no nvcc, no CUDA flags), `mtorch.cuda.is_available()` False,
full suite green. This step edits no source files.

**Preconditions**: On the Mac, repo root `/Users/hiramatsu/dev/mtorch`, with the
Phase 7 commits present (the human pulled them). Check:

```bash
pwd                                  # /Users/hiramatsu/dev/mtorch
git log --oneline -5                 # shows the phase7-13 commits at/near the top
git status --short                   # empty
```

**Actions**:

1. Fresh build, capturing the log; CUDA must be completely absent:

   ```bash
   rm -rf build mtorch/_C*.so
   python3 setup.py build_ext --inplace 2>&1 | tee /tmp/build-7-14.log
   grep -c 'nvcc\|MTORCH_WITH_CUDA\|cudart\|cublas' /tmp/build-7-14.log   # prints 0
   python3 -c "import mtorch; print(mtorch.__version__)"                  # 0.0.0
   python3 -c "import mtorch; print(mtorch.cuda.is_available())"          # False
   ```

2. CUDA ops must raise (not crash, not compute) on the Mac:

   ```bash
   python3 - <<'PY'
   import mtorch
   x = mtorch.tensor([1.0, 2.0])
   try:
       x.cuda()
       print("BUG: cuda() did not raise")
   except Exception as error:
       print("raised as expected:", type(error).__name__)
   PY
   ```

   Expected: `raised as expected: ...` (whatever exception type Phase 5 chose for
   unavailable devices).

3. Full suite against the **macOS** baseline. Determine which comparison applies
   (mechanical): `grep -c 'test_device_parity' docs/design/baseline/tests-baseline.txt`.

   - If it prints `0` (the committed macOS baseline predates the parity suite):

     ```bash
     pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3
     tail -1 docs/design/baseline/tests-baseline.txt
     pytest tests/compat/test_device_parity.py -q | tail -3
     ```

     PASS iff the first two summaries have identical counts and the parity run
     reports 0 failed / 0 error (cuda rows skip or don't parametrize — cuda is
     unavailable here; the rows added in 7-5…7-12 still exercise their cpu side).

   - If it prints nonzero (Phase 5/6 re-recorded the macOS baseline including
     parity tests): the full-run counts can legitimately exceed the baseline by
     exactly the rows this phase appended. Then the gate is: `pytest tests/compat
     -q | tail -3` shows **0 failed / 0 error**, and
     `pytest tests/compat -q --ignore=tests/compat/test_device_parity.py | tail -3`
     matches the baseline's non-parity counts if those are separable; if they are
     not separable, record both summary lines in the Phase 7 Notes and require
     only the 0-failed/0-error condition. Never edit the baseline file.

4. Benchmark regression guard (C++ shared with CPU paths was touched only inside
   `#if defined(MTORCH_WITH_CUDA)` and new files, so this should be flat) — run
   the §5.2 procedure of `01-rules-and-verification.md` against the macOS
   `benchmark-baseline.json`. PASS per §5.2 (`regressions=0 missing=0`).

**Verification**: Actions 1–4 all pass.

**On failure**: This step made no changes, so there is nothing to restore; a
failure here means a Phase 7 commit broke the Mac. Mark 7-14 **BLOCKED** in
`docs/design/PROGRESS.md` with the failing command's first error line, commit the
PROGRESS edit, and stop — reverting Phase 7 commits is a human decision.

**Commit**: None for the check itself (it changes nothing). Only the PROGRESS.md
completion edit per §8:

```bash
git add docs/design/PROGRESS.md && git commit -m "progress: complete 7-14"
```

---

## Appendix A: Debugging CUDA failures (read when — not before — something breaks)

**A.0 The iron debugging rule: shrink first.** Before touching any kernel code,
reproduce the failure with the smallest parity case you can construct — one op, one
dtype, the smallest failing shape (bisect the shape downward: 1024→256→64→8). A
4-element reproducer turns every tool below from a firehose into a tweezer, and
`print(x.cpu().tolist())` becomes readable. Most "kernel bugs" become obvious at
n=4 (off-by-one at the tail, wrong stride, swapped m/n).

**A.1 `CUDA_LAUNCH_BLOCKING=1`.** Environment variable that forces every kernel
launch to run synchronously. Under this phase's v1 policy launches are already
followed by `cudaDeviceSynchronize()`, so this normally changes nothing — which is
itself diagnostic: if setting it *changes* behavior, some code path skipped
`MTORCH_CUDA_LAUNCH_CHECK()` (grep the `.cu` files: every `<<<` line must be
followed by it).

```bash
CUDA_LAUNCH_BLOCKING=1 python3 -m pytest tests/compat/test_device_parity.py -q -k '<smallest case>'
```

**A.2 Reading "an illegal memory access was encountered".** This is CUDA's
segfault, and the message's location lies: because errors are sticky and surface at
the next checked call, the reported line is often an *innocent later* call — the
actual out-of-bounds write happened in an **earlier stale launch**. That is exactly
why every launch here carries `MTORCH_CUDA_LAUNCH_CHECK()` (which synchronizes):
with the macros in place, the reported `#expr` genuinely is the guilty kernel. If
you ever see the error attributed to a `cudaMemcpy` or an allocator call, suspect
the kernel launched just before it, not the reported line. Also rerun with
`MTORCH_CUDA_DISABLE_CACHING=1` — the caching allocator recycles buffers, which can
hide (or misattribute) use-after-free by making the "freed" memory still valid.

**A.3 compute-sanitizer.** CUDA's valgrind, ships with the toolkit. Checks every
device memory access against allocation bounds; slow (10-100x) — run it only on the
shrunk reproducer:

```bash
MTORCH_CUDA_DISABLE_CACHING=1 compute-sanitizer --tool memcheck \
    python3 -m pytest tests/compat/test_device_parity.py -q -k '<smallest case>'
```

It prints the exact kernel name, the thread/block coordinates, and the offending
address of the first bad access. `--tool racecheck` (same command shape) finds
shared-memory races — the missing-`__syncthreads()` class of bug from §0.2.
(Disable caching, or the sanitizer sees one big long-lived allocation and cannot
catch overruns between logical tensors inside it.)

**A.4 printf inside a kernel.** `printf` works in device code and is the fastest
probe for small cases. Gate it by coordinates or you get 10^6 lines:

```cpp
if (blockIdx.x == 0 && threadIdx.x == 0) {
  printf("row=%lld max=%f sum=%f\n", (long long)blockIdx.x, row_max, row_sum);
}
```

Output appears when the kernel synchronizes (our launch check guarantees that).
Remove every kernel printf before committing — they are never committed.

**A.5 cuda-gdb, one paragraph.** `cuda-gdb python3` then `run -m pytest ...` gives
you a real debugger inside kernels: `break mtorch::cuda::reduce_partial_kernel`,
`cuda thread (0,0,0) block (3,0,0)` to focus a specific thread, `print value`,
`info cuda kernels`. It is the tool of last resort for logic bugs that survive A.0
through A.4 — expect it to feel like gdb with two extra coordinates. Build with
`nvcc -G` (device debug info, disables optimization) for line-accurate stepping:
temporarily add `"-G"` to `MTORCH_NVCC_FLAGS`, rebuild, debug, then **revert
setup.py before committing** (`git restore setup.py`).

**A.6 Numeric (not crash) mismatches.** In order of prior probability: TF32 leaked
(0.9 N3 — check both switches); fp16 arithmetic done in `__half` somewhere (N1 —
grep the kernel for arithmetic on `scalar_t` instead of `acc_t`/float); summation
order (expected at ~1e-6 for fp32 reductions — that is why reduction-shaped ops get
1e-4 tolerance, do not chase it below that); `round` vs `rint` half-to-even (7-7);
an `int` where `int64_t` was needed in index math (shapes > 2^31 elements — v1
parity shapes never reach this, benchmarks might).

---

## Appendix B: What v1 explicitly leaves for later phases (do not start any of these)

- Asynchronous execution: per-op streams, events, removing the per-launch
  `cudaDeviceSynchronize()`.
- Non-contiguous and broadcasting kernels (shared strided-index helper, with the
  Metal backend).
- Autograd on CUDA (device-resident backward lambdas).
- Dim-wise reductions, `log_softmax`, `unary_predicate`, batched/vector matmul,
  grouped/strided conv variants, masked/causal SDPA.
- Multi-GPU (`cuda:1`, …): the allocator and all ops assume device 0.
- Performance work guided by `docs/design/baseline/cuda-ratios.md`.
