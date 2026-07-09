# 03. Phase 1: Splitting the Python package

Prerequisite: Phase 0 complete (git repo exists, `docs/design/baseline/` is committed).
`01-rules-and-verification.md` read.

Objective: Split `mtorch/__init__.py` (6,484 lines at the start of this phase) into a real
package structure that follows upstream torch. **This phase is purely mechanical moves.**
Changes to the class hierarchy (introducing `_ConvNd`, making Optimizer inheritance-based) are
done in Phases 3-5. The only rewrites allowed here are the dead-code deletions explicitly
stated in this guide, the name qualifications explicitly listed per step, and the
`globals()` replacements explicitly listed per step.

**Invariants (hold after every single step):**

1. The public API of `mtorch.*` is frozen: every name in `__all__` still resolves, and
   every dotted path that worked before (`mtorch.nn.Linear`, `mtorch.cuda.amp.autocast`,
   `import mtorch.nn.functional`, ...) still works.
2. `pytest tests/compat` produces the exact same summary breakdown as
   `docs/design/baseline/tests-baseline.txt`. Every step ends all-green (relative to that
   baseline).

No step in this phase touches C++. **`python3 setup.py build_ext --inplace` is NOT needed
anywhere in Phase 1.** The extension `mtorch/_C.cpython-*.so` must already exist from
Phase 0 (check once: `ls mtorch/_C.*.so`).

**About line numbers in this guide**: all line numbers are hints measured on the file as it
exists at the *start* of Phase 1. Earlier steps shift later line numbers. **Always re-run
the given `grep` command and use its output; never trust the hint.**

All commands are run from the repo root.

## 0. Design commitments (common to all steps)

### 0-a. Convention for avoiding circular imports

- The root `__init__.py` imports submodules **near the end of the file** (the trailing
  section created in Step 1-0), after Tensor, all operators, and Tensor patches have been
  defined (the same approach as torch).
- When a carved-out leaf module needs a function that stays in root, it does
  `import mtorch` at the module top and references it **inside function bodies** as
  `mtorch.zeros_like(...)` (call-time resolution). Do **not** write
  `from mtorch import zeros_like` at the top of a leaf module (risk of grabbing a name
  from a partially initialized root).
- `_C` is a real submodule imported at the very top of root, so `from mtorch import _C`
  is always safe in any file.
- **Tensor alias rule**: root defines `Tensor = _C.Tensor` (L175). Every new file that
  refers to `Tensor` (isinstance checks etc.) gets the line `Tensor = _C.Tensor` right
  after its imports. This is identical to root's own binding, needs no call-time
  indirection, and means `Tensor` references in moved code need **zero edits**.
- **Stdlib alias rule**: root uses the aliases `_builtins`, `_math`, `_functools`,
  `_pickle`. New files keep the exact same aliases
  (`import builtins as _builtins`, etc.) so that moved code needs zero edits for them.
- Leaf-to-leaf imports of already-created leaf modules (e.g.
  `from mtorch.nn.parameter import Parameter` inside `mtorch/nn/modules/module.py`) are
  allowed — only from-imports of the root module are banned.

### 0-b. The trailing section marker (created in Step 1-0)

All `from . import xxx` lines added by later steps go under this marker, which sits
immediately before `__all__ = [` (locate with `grep -n '^__all__' mtorch/__init__.py`,
hint L6154):

```python
# ============================================================
# Load the actual submodules.
# Must be placed after all definitions and all Tensor patches, before __all__.
# ============================================================
```

### 0-c. Standard verification block (referenced by every step as "STANDARD VERIFICATION")

Run all of this; every step's Verification section adds step-specific commands on top.

```bash
python3 -c "import mtorch; import mtorch.nn; import mtorch.nn.functional; import mtorch.optim; import mtorch.autograd; import mtorch.cuda.amp; print('import-ok')"
pytest tests/compat -q 2>&1 | tee /tmp/tests-current.txt | tail -3
tail -1 docs/design/baseline/tests-baseline.txt | tr -d '=' | sed 's/ in .*//' | xargs
tail -1 /tmp/tests-current.txt | tr -d '=' | sed 's/ in .*//' | xargs
```

Expected: first command prints `import-ok`; the last two commands print
**character-for-character identical** lines (e.g. `2200 passed, 72 xfailed`). Any
difference (one more `failed`, one fewer `passed`, anything) is a FAILURE.
The full suite includes the 415 benchmark tests and can take tens of minutes — do not
abort it; set your command timeout to at least 60 minutes.

### 0-d. Current pseudo-module machinery (deleted piecewise; each step says exactly which lines)

- Creation block: `nn = ModuleType("mtorch.nn")` ... `utils_checkpoint = ModuleType("mtorch.utils.checkpoint")`
  at L2049–L2063, plus `parameter = ModuleType("mtorch.nn.parameter")` (L5679),
  `autograd = ModuleType("mtorch.autograd")` (L5684), `optim = ModuleType("mtorch.optim")` (L5748).
  Enumerate at any time with `grep -n 'ModuleType(' mtorch/__init__.py` (currently 18 hits).
- Attribute wiring: `functional.xxx = ...` (L5515–5573), `linalg.xxx` (L5574–5579),
  `amp.autocast` (L5581), `init.xxx` (L5582–5592), `nn.xxx` (L5593–5677),
  `parameter.Parameter`/`nn.parameter` (L5680–5681), `autograd.grad` (L5745),
  `optim.xxx` (L6030–6033), `cuda./amp./cuda_amp./mps.` (L6035–6046),
  `backends*` (L6047–6070), `nn.attention`/`attention.xxx` (L6071–6077),
  `utils.checkpoint` (L6078–6079).
- Manual registration: `_sys.modules[...] = ...` block at L6081–6098
  (`grep -n '_sys.modules' mtorch/__init__.py`).

Delete in each step **only the lines that step names**. Never delete the whole block early.

### 0-e. Standard "On failure" procedure (referenced by every step as "STANDARD RECOVERY")

```bash
git restore --staged --worktree .
git clean -fd mtorch/ tools/
```

(`git clean -fd` without `-x` keeps the gitignored `_C.*.so`.) Then open
`docs/design/PROGRESS.md`, append ` **BLOCKED**` to the line of the current step, write
what happened in the phase Notes field (3 lines max), and commit only that:
`git add docs/design/PROGRESS.md && git commit -m "chore(phase1): mark <step> BLOCKED"`.
Stop; do not try the next step.

### 0-f. Standard "Commit" procedure

After Verification passes: mark the step `[x]` in `docs/design/PROGRESS.md`, fill in
`commit:` with the previous step's practice or leave until after committing, then:

```bash
git add -A
git commit -m "<message given in the step>"
git rev-parse --short HEAD   # write this hash into PROGRESS.md, then:
git add docs/design/PROGRESS.md
git commit --amend --no-edit
```

---

## Step 1-0: Insert the trailing-section marker (and the unbound-name checker)

**Goal**: Create the marker of §0-b, and a small static checker used by later steps to
catch names that were forgotten during a move.

**Preconditions**: Phase 0 complete. Check:

```bash
ls docs/design/baseline/tests-baseline.txt && git status --porcelain | head -1
```

Expected: the file path is printed; `git status` prints nothing (clean tree).

**Actions**:

1. Find the insertion point:

   ```bash
   grep -n '^__all__' mtorch/__init__.py
   ```

   Expected: exactly one hit (hint: L6154).

2. Insert the following 4 lines, plus one blank line after them, **immediately above**
   the `__all__ = [` line:

   ```python
   # ============================================================
   # Load the actual submodules.
   # Must be placed after all definitions and all Tensor patches, before __all__.
   # ============================================================
   ```

3. Create `tools/check_unbound_names.py` with exactly this content:

   ```python
   """Print names referenced in a Python file that are never bound in it.

   Usage: python3 tools/check_unbound_names.py <file.py>
   Coarse by design (all scopes merged); used in Phase 1 to catch root names
   that were forgotten when moving code out of mtorch/__init__.py.
   Names that shadow builtins in mtorch root (sum, all, any, bool, abs, round,
   min, max, compile, ...) are NOT flagged — those are enumerated per step.
   """
   import ast
   import builtins
   import sys


   def main(path):
       tree = ast.parse(open(path).read())
       bound = set()
       for node in ast.walk(tree):
           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
               bound.add(node.name)
           elif isinstance(node, ast.alias):
               bound.add((node.asname or node.name).split(".")[0])
           elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
               bound.add(node.id)
           elif isinstance(node, ast.arg):
               bound.add(node.arg)
           elif isinstance(node, ast.ExceptHandler) and node.name:
               bound.add(node.name)
       unresolved = sorted({
           n.id for n in ast.walk(tree)
           if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
           and n.id not in bound and not hasattr(builtins, n.id)
       })
       for name in unresolved:
           print(name)
       return 1 if unresolved else 0


   if __name__ == "__main__":
       sys.exit(main(sys.argv[1]))
   ```

**Verification**:

```bash
grep -n 'Load the actual submodules' mtorch/__init__.py
python3 tools/check_unbound_names.py tools/check_unbound_names.py; echo "exit=$?"
python3 -c "import mtorch; print('ok')"
```

Expected: one grep hit directly above `__all__`; the checker prints nothing and `exit=0`;
`ok`. (No need to run the full suite for a comment-only change.)

**On failure**: STANDARD RECOVERY (§0-e) for step 1-0.

**Commit** (§0-f): `refactor(phase1-0): add trailing-section marker and unbound-name checker`

---

## Step 1-1: `mtorch/serialization.py`

**Goal**: Move `_serialize_value` / `_deserialize_value` / `save` / `load` into
`mtorch/serialization.py`; delete the dead `_C.save`/`_C.load` bindings; replace the
`globals()` dtype lookup.

Background (verified, also noted in `00-overview.md` §2.2): root has
`save = _C.save` / `load = _C.load` at L1955–1956, which are unconditionally overwritten
by `def save` (L6135) and `def load` (L6145) later in the same file. The `_C` bindings are
therefore dead code and may be deleted here.

**Preconditions**: Step 1-0 done. Check:

```bash
grep -c 'Load the actual submodules' mtorch/__init__.py
```

Expected: `1`.

**Actions**:

1. Locate everything:

   ```bash
   grep -n 'save = _C.save\|load = _C.load' mtorch/__init__.py      # hint L1955-1956
   grep -n 'def _serialize_value\|def _deserialize_value\|^def save\|^def load' mtorch/__init__.py
   # hints: 6101, 6120, 6135, 6145
   grep -n 'globals().get' mtorch/__init__.py                        # hint L6121 (inside _deserialize_value)
   ```

2. Confirm the moved symbols have no other users in root (expect: hits only inside
   L6101–6151 and the two dead bindings):

   ```bash
   grep -n '_serialize_value\|_deserialize_value' mtorch/__init__.py
   ```

3. Create `mtorch/serialization.py` with exactly this content (this is the L6101–6151
   block with the three edits already applied: `Tensor` handled by the alias line;
   `globals().get(value["dtype"], float32)` replaced by `_dtype_by_name`; bare
   `tensor(...)` qualified as `mtorch.tensor(...)`):

   ```python
   """save/load (equivalent to torch/serialization.py)."""

   from __future__ import annotations

   import builtins as _builtins
   import pickle as _pickle
   from collections import OrderedDict

   import mtorch
   from mtorch import _C

   Tensor = _C.Tensor


   def _dtype_by_name(name):
       # Same behavior as the old `globals().get(name, float32)` in root:
       # unknown names fall back to float32 (do NOT raise; that would change behavior).
       value = getattr(mtorch, name, None)
       if isinstance(value, mtorch.dtype):
           return value
       return mtorch.float32


   def _serialize_value(value):
       if isinstance(value, Tensor):
           return {
               "__mtorch_tensor__": True,
               "data": value.tolist(),
               "dtype": str(value.dtype),
               "requires_grad": _builtins.bool(value.requires_grad),
           }
       if isinstance(value, OrderedDict):
           return {"__mtorch_ordered_dict__": True, "items": [(key, _serialize_value(item)) for key, item in value.items()]}
       if isinstance(value, dict):
           return {key: _serialize_value(item) for key, item in value.items()}
       if isinstance(value, tuple):
           return {"__mtorch_tuple__": True, "items": [_serialize_value(item) for item in value]}
       if isinstance(value, list):
           return [_serialize_value(item) for item in value]
       return value


   def _deserialize_value(value, map_location=None):
       if isinstance(value, dict) and value.get("__mtorch_tensor__"):
           dtype_object = _dtype_by_name(value["dtype"])
           return mtorch.tensor(value["data"], dtype=dtype_object, requires_grad=value["requires_grad"], device=map_location or "cpu")
       if isinstance(value, dict) and value.get("__mtorch_ordered_dict__"):
           return OrderedDict((key, _deserialize_value(item, map_location)) for key, item in value["items"])
       if isinstance(value, dict) and value.get("__mtorch_tuple__"):
           return tuple(_deserialize_value(item, map_location) for item in value["items"])
       if isinstance(value, dict):
           return {key: _deserialize_value(item, map_location) for key, item in value.items()}
       if isinstance(value, list):
           return [_deserialize_value(item, map_location) for item in value]
       return value


   def save(obj, f, pickle_module=_pickle, pickle_protocol=2, **pickle_save_args):
       payload = _serialize_value(obj)
       if hasattr(f, "write"):
           pickle_module.dump(payload, f, protocol=pickle_protocol, **pickle_save_args)
           return None
       with _builtins.open(f, "wb") as handle:
           pickle_module.dump(payload, handle, protocol=pickle_protocol, **pickle_save_args)
       return None


   def load(f, map_location=None, pickle_module=_pickle, **pickle_load_args):
       if hasattr(f, "read"):
           payload = pickle_module.load(f, **pickle_load_args)
       else:
           with _builtins.open(f, "rb") as handle:
               payload = pickle_module.load(handle, **pickle_load_args)
       return _deserialize_value(payload, map_location=map_location)
   ```

   Note: if the L6101–6151 block in the repo differs from the above (someone touched it),
   the repo wins — redo the move by hand from the current source, applying only the three
   listed edits.

4. Root edits, in this order:
   - Delete the block from `def _serialize_value(value):` through the last line of
     `def load(...)` (the line `return _deserialize_value(payload, map_location=map_location)`).
   - Delete the two dead lines `save = _C.save` and `load = _C.load` (hint L1955–1956).
   - Delete the now-unused import line `import pickle as _pickle` at the top of root
     (confirm first: `grep -n '_pickle' mtorch/__init__.py` must show only the import line).
   - Add under the §0-b marker:

     ```python
     from . import serialization as serialization
     from .serialization import save, load
     ```

**Verification**:

```bash
python3 tools/check_unbound_names.py mtorch/serialization.py
python3 -c "import mtorch; assert mtorch.save is mtorch.serialization.save; assert mtorch.load is mtorch.serialization.load; print('ok', mtorch.serialization)"
pytest tests/compat/test_serialization.py -q | tail -3
```

Expected: checker silent; `ok <module ...>`; serialization tests all pass. Then run
STANDARD VERIFICATION (§0-c) — summary lines identical to baseline.

**On failure**: STANDARD RECOVERY (§0-e) for step 1-1.

**Commit** (§0-f): `refactor(phase1-1): split out mtorch/serialization.py`

---

## Step 1-2: `mtorch/optim.py`

**Goal**: Move `_ensure_param_list`, `Optimizer`, `SGD`, `Adam`, `AdamW` verbatim into
`mtorch/optim.py`. Do not touch class bodies except the listed qualifications
(inheritance conversion is Phase 3-5).

**Preconditions**: Step 1-1 done (`test -f mtorch/serialization.py` prints nothing and exits 0).

**Actions**:

1. Locate the cut range:

   ```bash
   grep -n 'optim = ModuleType\|def _ensure_param_list\|^class Optimizer\|^class SGD\|^class Adam\|^class AdamW' mtorch/__init__.py
   # hints: 5748 (ModuleType), 5751, 5757, 5794, 5865, 5957
   grep -n '^optim\.' mtorch/__init__.py        # hints: 6030-6033
   ```

   The cut range is from `def _ensure_param_list(params):` through the last line of
   `class AdamW` (its final `return loss`; the next top-level statement is a comment or
   `optim.Optimizer = Optimizer` — check with `sed -n` around the grep hits).

2. Create `mtorch/optim.py`: the header below, then paste the cut range verbatim, then
   apply the edit list.

   Header:

   ```python
   """Optimizers (equivalent to torch/optim/). Composition-based; inheritance comes in Phase 3-5."""

   from __future__ import annotations

   import math as _math

   import mtorch
   from mtorch import _C

   Tensor = _C.Tensor
   ```

   Edit list (only these; count with grep before/after):
   - `zeros_like(` → `mtorch.zeros_like(` — exactly 6 sites
     (`grep -c 'zeros_like(' mtorch/optim.py` is 6 before and after; after the edit,
     `grep -c 'mtorch.zeros_like(' mtorch/optim.py` is 6).
   - `maximum(` → `mtorch.maximum(` — exactly 2 sites (in `Adam.step` and `AdamW.step`,
     the `max_exp_avg_sq.copy_(maximum(...))` lines).
   - `Tensor` — zero edits (covered by the alias line).

3. Root edits:
   - Delete the cut range (`def _ensure_param_list` ... end of `AdamW`).
   - Delete the line `optim = ModuleType("mtorch.optim")` (hint L5748).
   - Delete the 4 wiring lines `optim.Optimizer = Optimizer` ... `optim.AdamW = AdamW`
     (hint L6030–6033).
   - Delete the line `_sys.modules[__name__ + ".optim"] = optim` (hint L6098).
   - Add under the §0-b marker: `from . import optim as optim`

**Verification**:

```bash
python3 tools/check_unbound_names.py mtorch/optim.py
python3 -c "import mtorch, mtorch.optim; print(mtorch.optim.SGD, mtorch.optim.Adam, mtorch.optim.AdamW, mtorch.optim.Optimizer)"
pytest tests/compat/test_optim.py -q | tail -3
```

Expected: checker silent; four class reprs printed; optim tests pass. Then STANDARD
VERIFICATION (§0-c).

**On failure**: STANDARD RECOVERY (§0-e) for step 1-2.

**Commit** (§0-f): `refactor(phase1-2): split out mtorch/optim.py`

---

## Step 1-3: `mtorch/autograd.py`

**Goal**: Move the `grad` machinery (`_as_tensor_tuple`, `_as_optional_tensor_tuple`,
`_autograd_grad`) into `mtorch/autograd.py`. The pseudo-module exposes exactly one name:
`grad` (verified: `grep -n '^autograd\.' mtorch/__init__.py` → only `autograd.grad = _autograd_grad`, hint L5745).
Grad-mode contexts (`no_grad`, `enable_grad`, ...) stay in root — do not move them.

**Preconditions**: Step 1-2 done (`test -f mtorch/optim.py`).

**Actions**:

1. Locate:

   ```bash
   grep -n 'autograd = ModuleType\|def _as_tensor_tuple\|def _as_optional_tensor_tuple\|def _autograd_grad\|^autograd.grad' mtorch/__init__.py
   # hints: 5684, 5687, 5700, 5716, 5745
   ```

   Cut range: from `def _as_tensor_tuple(value, name):` through the closing of
   `_autograd_grad` (the line `)` after `materialize_grads=materialize_grads,`).

2. Create `mtorch/autograd.py`: header below + cut range verbatim (zero body edits —
   `Tensor` is covered by the alias, `_C._autograd_grad` already uses `_C`) + the final
   alias line.

   ```python
   """mtorch.autograd (equivalent to torch/autograd/): thin wrapper over _C._autograd_grad."""

   from __future__ import annotations

   import mtorch
   from mtorch import _C

   Tensor = _C.Tensor
   ```

   ...cut range pasted here verbatim...

   ```python
   grad = _autograd_grad
   ```

3. Root edits:
   - Delete the cut range.
   - Delete `autograd = ModuleType("mtorch.autograd")` (hint L5684).
   - Delete `autograd.grad = _autograd_grad` (hint L5745).
   - Delete `_sys.modules[__name__ + ".autograd"] = autograd` (hint L6097).
   - Add under the §0-b marker: `from . import autograd as autograd`

**Verification**:

```bash
python3 tools/check_unbound_names.py mtorch/autograd.py
python3 -c "import mtorch, mtorch.autograd; print(mtorch.autograd.grad)"
pytest tests/compat/test_autograd.py -q | tail -3
```

Then STANDARD VERIFICATION (§0-c).

**On failure**: STANDARD RECOVERY (§0-e) for step 1-3.

**Commit** (§0-f): `refactor(phase1-3): split out mtorch/autograd.py`

---

## Step 1-4: The stub group (`amp.py` / `cuda/` / `mps.py` / `backends/` / `utils/` / `linalg.py`) + setup.py

**Goal**: Carve out all small stub modules. Six sub-steps (a)–(g), **one commit each**, in
exactly this order (later sub-steps import files created by earlier ones). `linalg.py` is
included here (it is not listed in the old step title, but it must be carved out somewhere
before Step 1-8's zero-`ModuleType` check can pass; note this in PROGRESS.md).
`def compile(...)` (hint L2302) stays in root — do not move it.

**Preconditions**: Step 1-3 done (`test -f mtorch/autograd.py`).

**On failure** (any sub-step): STANDARD RECOVERY (§0-e) for step 1-4.

### 1-4-a: setup.py subpackage discovery

This step creates the first subpackages (`mtorch/cuda/`, `mtorch/backends/`,
`mtorch/utils/`). `setup.py` currently has `packages=["mtorch"]`, which would silently
omit subpackages from an installed distribution. It does **not** affect running the tests
from the repo root (pyproject sets `pythonpath = ["."]`), but fix it now anyway.

In `setup.py` apply exactly these two edits:

- `from setuptools import Extension, setup`
  → `from setuptools import Extension, find_packages, setup`
- `    packages=["mtorch"],`
  → `    packages=find_packages(include=["mtorch", "mtorch.*"]),`

Verify: `python3 -c "from setuptools import find_packages; print(find_packages(include=['mtorch','mtorch.*']))"`
prints at least `['mtorch']` (subpackages appear as the following sub-steps add them).
Commit: `refactor(phase1-4a): setup.py find_packages for upcoming subpackages`

### 1-4-b: `mtorch/amp.py`

1. Locate the cut range:

   ```bash
   grep -n '_autocast_stack = \|class autocast:\|def is_autocast_enabled\|class _CudaAutocast\|class GradScaler:\|class _BackendFlags:' mtorch/__init__.py
   # hints: 2066, 2070, 2098, 2104, 2109, 2142
   ```

   Cut range: from `_autocast_stack = []` through the last line of `class GradScaler`
   (`return None` of `load_state_dict`), i.e. everything before `class _BackendFlags:`.

2. Create `mtorch/amp.py`: header + cut range verbatim. **Zero body edits** (verified:
   the block uses only `_builtins` and `_functools`).

   ```python
   """mtorch.amp (equivalent to torch/amp/): autocast/GradScaler stubs."""

   from __future__ import annotations

   import builtins as _builtins
   import functools as _functools
   ```

3. Root edits:
   - Replace the deleted cut range (at the same location) with this single line — it is
     safe to import here because `mtorch/amp.py` only needs stdlib, and it keeps the root
     globals `autocast`/`is_autocast_enabled` (both in `__all__`) and the wiring lines
     `cuda_amp.autocast = _CudaAutocast` / `cuda_amp.GradScaler = GradScaler` working:

     ```python
     from .amp import GradScaler, _CudaAutocast, autocast, is_autocast_enabled
     ```

   - Delete `amp = ModuleType("mtorch.amp")` (hint L2054).
   - Delete `amp.autocast = autocast` (hint L5581) and `amp.GradScaler = GradScaler` (hint L6042).
   - Delete `_sys.modules[__name__ + ".amp"] = amp` (hint L6087).
   - Add under the §0-b marker: `from . import amp as amp`

4. Verify: `python3 tools/check_unbound_names.py mtorch/amp.py` (silent);
   `python3 -c "import mtorch; assert mtorch.amp.autocast is mtorch.autocast; assert mtorch.amp.GradScaler is mtorch.GradScaler; print('ok')"`;
   then STANDARD VERIFICATION (§0-c).
   Commit: `refactor(phase1-4b): split out mtorch/amp.py`

### 1-4-c: `mtorch/cuda/` package and `mtorch/mps.py`

`mtorch.cuda` must be a **package** (not `cuda.py`) because `import mtorch.cuda.amp`
works today via the manual `sys.modules` registration and must keep working
(tests use `torch_candidate.cuda.amp.autocast(...)` — see `tests/compat/test_tensor_ops.py`).

1. Locate the cut range:

   ```bash
   grep -n 'def _cuda_is_available\|def _mps_is_built\|def _checkpoint' mtorch/__init__.py
   # hints: 2320, 2344, 2348
   ```

   Cut range: from `def _cuda_is_available():` through the `return False` of
   `def _mps_is_built():` (do NOT include `def _checkpoint` — that is 1-4-e).

2. Create `mtorch/cuda/__init__.py` with exactly:

   ```python
   """mtorch.cuda stub (equivalent to torch/cuda/; CUDA is never available in this build)."""


   def is_available():
       return False


   def device_count():
       return 0


   def empty_cache():
       return None


   def manual_seed(seed):
       return None


   def is_bf16_supported():
       return False


   manual_seed_all = manual_seed

   from mtorch.cuda import amp as amp  # noqa: E402
   ```

   (These are the bodies of `_cuda_is_available` ... `_cuda_is_bf16_supported` under
   their wired public names; `manual_seed_all = manual_seed` preserves the old
   `cuda.manual_seed_all = _cuda_manual_seed` identity.)

3. Create `mtorch/cuda/amp.py` with exactly:

   ```python
   """mtorch.cuda.amp (equivalent to torch/cuda/amp/)."""

   from mtorch.amp import GradScaler as GradScaler
   from mtorch.amp import _CudaAutocast as autocast
   ```

4. Create `mtorch/mps.py` with exactly:

   ```python
   """mtorch.mps stub (equivalent to torch/mps/; MPS is never available in this build)."""


   def is_available():
       return False


   def is_built():
       return False
   ```

5. Root edits:
   - Delete the cut range from Action 1 (no replacement — verified: `_cuda_*`/`_mps_*`
     are referenced only by the wiring lines deleted next; re-check with
     `grep -n '_cuda_is_available\|_cuda_device_count\|_cuda_empty_cache\|_cuda_manual_seed\|_cuda_is_bf16_supported\|_mps_is_available\|_mps_is_built' mtorch/__init__.py`
     → after the deletions below, 0 hits).
   - Delete the three creation lines `cuda = ModuleType("mtorch.cuda")`,
     `cuda_amp = ModuleType("mtorch.cuda.amp")`, `mps = ModuleType("mtorch.mps")`
     (hints L2055–2057).
   - Delete the wiring lines (hints L6035–6041, L6043–6046):
     `cuda.is_available = ...` through `cuda.amp = cuda_amp`, and
     `cuda_amp.autocast = _CudaAutocast`, `cuda_amp.GradScaler = GradScaler`,
     `mps.is_available = ...`, `mps.is_built = ...`.
     The `backends_mps.*` wiring (hints L6064–6065) stays until 1-4-d, but it references
     the `_mps_*` functions deleted above, so rewrite those two lines now to use the new
     real module:

     ```python
     backends_mps.is_available = _mps_is_available
     backends_mps.is_built = _mps_is_built
     ```

     into:

     ```python
     from .mps import is_available as _mps_is_available_stub, is_built as _mps_is_built_stub
     backends_mps.is_available = _mps_is_available_stub
     backends_mps.is_built = _mps_is_built_stub
     ```

   - Delete the three `_sys.modules` lines for `".cuda"`, `".cuda.amp"`, `".mps"`
     (hints L6088–6090).
   - Add under the §0-b marker:

     ```python
     from . import cuda as cuda
     from . import mps as mps
     ```

6. Verify:

   ```bash
   python3 -c "import mtorch, mtorch.cuda.amp, mtorch.mps; print(mtorch.cuda.is_available(), mtorch.cuda.amp.autocast, mtorch.cuda.manual_seed is mtorch.cuda.manual_seed_all, mtorch.mps.is_available())"
   ```

   Expected: `False <class 'mtorch.amp._CudaAutocast'> True False`. Then STANDARD
   VERIFICATION (§0-c).
   Commit: `refactor(phase1-4c): split out mtorch/cuda/ and mtorch/mps.py`

### 1-4-d: `mtorch/backends/` package

1. Locate the cut range:

   ```bash
   grep -n 'class _BackendFlags:\|def _sdpa_kernel\|def compile' mtorch/__init__.py
   # hints: 2142, 2292, 2302
   ```

   Cut range: from `class _BackendFlags:` through the closing `)` of `_sdpa_kernel`
   (everything before `def compile(`). Contents (verify each is inside the range):
   `_BackendFlags`, `SDPBackend`, `SDPAParams`, the four flag globals
   `_sdp_flash_enabled` / `_sdp_math_enabled` / `_sdp_mem_efficient_enabled` /
   `_sdp_cudnn_enabled`, `_enable_flash_sdp`, `_flash_sdp_enabled`, `_enable_math_sdp`,
   `_math_sdp_enabled`, `_enable_mem_efficient_sdp`, `_mem_efficient_sdp_enabled`,
   `_enable_cudnn_sdp`, `_cudnn_sdp_enabled`, `_can_use_flash_attention`,
   `_can_use_efficient_attention`, `_can_use_cudnn_attention`, `_SdpKernelContext`,
   `_sdp_kernel`, `_normalize_sdpa_backends`, `_sdpa_kernel`.

2. Create `mtorch/backends/cuda.py`: header + cut range verbatim (**zero body edits**;
   verified externals are only `_builtins`, `_functools`, `_IntEnum`) + the public alias
   block, exactly:

   ```python
   """mtorch.backends.cuda (equivalent to torch/backends/cuda/): SDP kernel flags."""

   from __future__ import annotations

   import builtins as _builtins
   import functools as _functools
   from enum import IntEnum as _IntEnum
   ```

   ...cut range pasted here verbatim...

   ```python
   matmul = _BackendFlags(allow_tf32=False)
   enable_flash_sdp = _enable_flash_sdp
   flash_sdp_enabled = _flash_sdp_enabled
   enable_math_sdp = _enable_math_sdp
   math_sdp_enabled = _math_sdp_enabled
   enable_mem_efficient_sdp = _enable_mem_efficient_sdp
   mem_efficient_sdp_enabled = _mem_efficient_sdp_enabled
   enable_cudnn_sdp = _enable_cudnn_sdp
   cudnn_sdp_enabled = _cudnn_sdp_enabled
   can_use_flash_attention = _can_use_flash_attention
   can_use_efficient_attention = _can_use_efficient_attention
   can_use_cudnn_attention = _can_use_cudnn_attention
   sdp_kernel = _sdp_kernel
   ```

   (This alias block reproduces the old `backends_cuda.*` wiring at hints L6048–6062.
   `_sdpa_kernel` and `_normalize_sdpa_backends` stay private here; `mtorch/nn/attention.py`
   re-exports `_sdpa_kernel` in Step 1-6-e.)

3. Create `mtorch/backends/mps.py` with exactly:

   ```python
   """mtorch.backends.mps stub."""

   from mtorch.mps import is_available as is_available
   from mtorch.mps import is_built as is_built
   ```

4. Create `mtorch/backends/cudnn.py` with exactly:

   ```python
   """mtorch.backends.cudnn stub."""

   enabled = False
   benchmark = False
   deterministic = False
   allow_tf32 = False
   ```

5. Create `mtorch/backends/__init__.py` with exactly:

   ```python
   """mtorch.backends (equivalent to torch/backends/)."""

   from mtorch.backends import cuda as cuda
   from mtorch.backends import cudnn as cudnn
   from mtorch.backends import mps as mps
   ```

6. Root edits:
   - Replace the deleted cut range (same location) with — needed by the `nn.attention`
     pseudo-module wiring (hints L6071–6077), which stays until Step 1-6-e:

     ```python
     from .backends.cuda import (
         SDPAParams,
         SDPBackend,
         _can_use_efficient_attention,
         _can_use_flash_attention,
         _sdpa_kernel,
     )
     ```

   - Delete the four creation lines `backends = ModuleType(...)`,
     `backends_cuda = ModuleType(...)`, `backends_mps = ModuleType(...)`,
     `backends_cudnn = ModuleType(...)` (hints L2058–2061).
   - Delete the whole `backends*` wiring block: from `backends.cuda = backends_cuda`
     through `backends_cudnn.allow_tf32 = False` (hints L6047–6070, including the
     `_mps_is_available_stub` lines added in 1-4-c). Do NOT delete the `nn.attention` /
     `attention.*` lines that follow (hints L6071–6077).
   - Delete the four `_sys.modules` lines for `".backends"`, `".backends.cuda"`,
     `".backends.mps"`, `".backends.cudnn"` (hints L6091–6094).
   - Delete the now-unused import `from enum import IntEnum as _IntEnum` at the top of
     root (confirm first: `grep -n '_IntEnum' mtorch/__init__.py` → only the import line).
   - Add under the §0-b marker: `from . import backends as backends`

7. Verify:

   ```bash
   python3 tools/check_unbound_names.py mtorch/backends/cuda.py
   python3 -c "import mtorch; import mtorch.backends.cuda as bc; print(bc.SDPBackend.MATH, bc.flash_sdp_enabled(), mtorch.backends.cudnn.enabled, mtorch.backends.mps.is_available())"
   python3 -c "import mtorch; print(mtorch.nn.attention.SDPBackend, mtorch.nn.attention.sdpa_kernel)"
   ```

   Then STANDARD VERIFICATION (§0-c).
   Commit: `refactor(phase1-4d): split out mtorch/backends/`

### 1-4-e: `mtorch/utils/` package

1. Locate: `grep -n 'def _checkpoint' mtorch/__init__.py` (hint L2348). Cut range: the
   whole `_checkpoint` def (3 lines).
2. Create `mtorch/utils/__init__.py`:

   ```python
   """mtorch.utils (equivalent to torch/utils/)."""

   from mtorch.utils import checkpoint as checkpoint
   ```

3. Create `mtorch/utils/checkpoint.py` (the `_checkpoint` body under its wired public name):

   ```python
   """mtorch.utils.checkpoint stub."""


   def checkpoint(function, *args, use_reentrant=None, **kwargs):
       return function(*args, **kwargs)
   ```

4. Root edits: delete the `_checkpoint` def; delete the creation lines
   `utils = ModuleType("mtorch.utils")` and `utils_checkpoint = ModuleType("mtorch.utils.checkpoint")`
   (hints L2062–2063); delete the wiring `utils.checkpoint = utils_checkpoint` and
   `utils_checkpoint.checkpoint = _checkpoint` (hints L6078–6079); delete the two
   `_sys.modules` lines for `".utils"` and `".utils.checkpoint"` (hints L6095–6096);
   add under the §0-b marker: `from . import utils as utils`
5. Verify: `python3 -c "import mtorch, mtorch.utils.checkpoint; print(mtorch.utils.checkpoint.checkpoint(lambda: 42))"`
   → `42`. Then STANDARD VERIFICATION (§0-c).
   Commit: `refactor(phase1-4e): split out mtorch/utils/`

### 1-4-f: `mtorch/linalg.py`

1. Locate: `grep -n 'def _linalg_multi_dot\|def _linalg_vector_norm\|def _linalg_norm\|^linalg\.' mtorch/__init__.py`
   (hints: 1941, 1947, 1951; wiring 5574–5579). Confirm the three defs have no other
   users in root: `grep -n '_linalg_' mtorch/__init__.py` → only the defs + wiring lines.
2. Create `mtorch/linalg.py` with exactly (the three defs under their wired public names;
   the only edit is `norm(` → `mtorch.norm(`, 2 sites):

   ```python
   """mtorch.linalg (equivalent to torch/linalg/)."""

   from __future__ import annotations

   import mtorch
   from mtorch import _C

   matrix_power = _C.matrix_power
   matmul = _C.matmul
   diagonal = _C.diagonal


   def multi_dot(tensors, *, out=None):
       if out is not None:
           raise NotImplementedError("linalg.multi_dot out= is not implemented yet")
       return _C.chain_matmul(*tuple(tensors))


   def vector_norm(input, ord=2, dim=None, keepdim=False, *, dtype=None, out=None):
       return mtorch.norm(input, p=ord, dim=dim, keepdim=keepdim, dtype=dtype, out=out)


   def norm(input, ord=None, dim=None, keepdim=False, *, dtype=None, out=None):
       return mtorch.norm(input, p=2.0 if ord is None else ord, dim=dim, keepdim=keepdim, dtype=dtype, out=out)
   ```

3. Root edits: delete the three `_linalg_*` defs (1941–1952); delete
   `linalg = ModuleType("mtorch.linalg")` (hint L2053); delete the 6 wiring lines
   `linalg.matrix_power = ...` ... `linalg.vector_norm = ...` (hints L5574–5579); delete
   `_sys.modules[__name__ + ".linalg"] = linalg` (hint L6086); add under the §0-b marker:
   `from . import linalg as linalg`
4. Verify: `python3 -c "import mtorch; import mtorch.linalg; print(mtorch.linalg.matrix_power, mtorch.linalg.norm)"`.
   Then STANDARD VERIFICATION (§0-c).
   Commit: `refactor(phase1-4f): split out mtorch/linalg.py`
   Add a one-line note in PROGRESS.md Phase 1 Notes: "linalg.py was carved out in 1-4
   (required for the 1-8 zero-ModuleType check)."

### 1-4-g: sub-step wrap-up check

```bash
grep -n 'ModuleType(' mtorch/__init__.py
```

Expected: exactly 5 hits remaining: `nn`, `functional`, `attention`, `init` (block at
~L2049–2052) and `parameter` (~L5679). If anything else remains, a sub-step was missed —
STANDARD RECOVERY. (No commit for this check.)

---

## Step 1-5: `mtorch/_numeric.py`

**Goal**: Move the pure-Python numeric algorithms: `einsum` (+ its 12 helpers), `diff`,
`trapezoid`, `cumulative_trapezoid`, `gradient`, `bucketize`, `quantile` (+ their
helpers). `cumulative_trapezoid` is public and in `__all__` — the old step description
omitted it; it moves too.

**Important — these stay in root**: the interleaved `_C` binding lines
(`_sum = _C.sum`, `trace`, `_diff_float32`, `cumsum`, `cumprod`, `cummax`, `cummin`,
`_trapezoid_dx`, `_cumulative_trapezoid_dx`, `_gradient_uniform`, `argmin`, `sort`,
`argsort`, `topk`, `_quantile_flat`, `_quantile_dim_2d`, `searchsorted`, `unique`,
`unique_consecutive`) and the Tensor patch lines `Tensor.diff = lambda ...` (hint
L1205–1209) and `Tensor.quantile = lambda ...` (hint L1500–1502). The patch lambdas
resolve `diff`/`quantile` from root globals at call time; the trailing
`from ._numeric import ...` re-binds those names, so the patches keep working. The moved
code references the private `_C` helpers directly as `_C._diff_float32` etc.

**Preconditions**: Step 1-4 done (1-4-g check passed).

**Actions**:

1. Locate the four cut blocks (each block is contiguous; cut exactly between the anchors):

   ```bash
   grep -n '^_EINSUM_ELLIPSIS = \|^_sum = _C.sum' mtorch/__init__.py       # block A: from first hit to the line BEFORE the second (hints 892..1143)
   grep -n 'def _slice_for_diff\|^Tensor.diff = lambda' mtorch/__init__.py  # block B: from def to the line BEFORE the patch (hints 1155..1204)
   grep -n 'def _trapezoid_areas\|^argmin = _C.argmin' mtorch/__init__.py   # block C: from def to the line BEFORE argmin (hints 1210..1408)
   grep -n 'def bucketize\|^Tensor.quantile = lambda' mtorch/__init__.py    # block D: from def to the line BEFORE the patch (hints 1421..1499)
   ```

   Block A contains: `_EINSUM_ELLIPSIS`, `_einsum_prod`, `_einsum_unique`,
   `_einsum_parse_subscript`, `_einsum_expand_labels`, `_einsum_expand_output`,
   `_einsum_apply_repeated_labels`, `_einsum_update_label_size`, `_einsum_label_dim`,
   `_einsum_permute_if_needed`, `_einsum_align_operand`, `_einsum_try_binary_matmul`,
   `_einsum_generic_elementwise`, `einsum`.
   Block B: `_slice_for_diff`, `_normalize_dim_for_input`, `diff`.
   Block C: `_trapezoid_areas`, `trapezoid`, `cumulative_trapezoid`, `_gradient_dims`,
   `_gradient_scalar_spacing`, `_gradient_spacing`, `_gradient_shape_for_axis`,
   `_gradient_uniform_python`, `_gradient_coordinate`, `gradient`.
   Block D: `bucketize`, `_quantile_value`, `_quantile_index_pair`, `_quantile_select`,
   `quantile`.

2. Create `mtorch/_numeric.py`: header below, then blocks A, B, C, D in that order,
   verbatim, then apply the edit list.

   ```python
   """Pure-Python numeric algorithms (einsum fallback, diff/trapezoid/gradient/bucketize/quantile)."""

   from __future__ import annotations

   import builtins as _builtins
   import math as _math

   import mtorch
   from mtorch import _C

   Tensor = _C.Tensor
   ```

   Edit list (exact, with pre-Phase-1 line hints; find each with the grep shown):
   - `diagonal(` → `mtorch.diagonal(` — 1 site (hint 963, in `_einsum_apply_repeated_labels`).
   - `permute(` → `mtorch.permute(` — 2 sites (hints 988, 1048).
   - `reshape(` → `mtorch.reshape(` — 4 sites (hints 1004, 1040, 1041, 1046). Do NOT touch
     method calls like `.reshape(`.
   - `matmul(` → `mtorch.matmul(` — 1 site (hint 1042).
   - `result = sum(result, dim=axis)` → `result = mtorch.sum(result, dim=axis)` — 1 site
     (hint 1059). Do NOT touch `_builtins.sum(`.
   - `== bool` → `== mtorch.bool` — 2 sites (hints 1121, 1377).
   - `cat(` → `mtorch.cat(` — 3 sites (hints 1184, 1324, 1366).
   - `== float32` → `== mtorch.float32` — 1 site (hint 1191).
   - `Size` → `mtorch.Size` — 2 sites (hints 1247, 1267; the `isinstance(..., (tuple, list, Size))` lines).
   - `_diff_float32(` → `_C._diff_float32(` — 1 site (hint 1195).
   - `_trapezoid_dx(` → `_C._trapezoid_dx(` — 1 site (hint 1229).
   - `_cumulative_trapezoid_dx(` → `_C._cumulative_trapezoid_dx(` — 1 site (hint 1239).
   - `_gradient_uniform(` → `_C._gradient_uniform(` — 1 site (hint 1387). Do NOT touch
     `_gradient_uniform_python`.
   - `searchsorted(` → `_C.searchsorted(` — 1 site (hint 1422, inside `bucketize`).
   - `_quantile_flat(` → `_C._quantile_flat(` — 1 site (hint 1471).
   - `_quantile_dim_2d(` → `_C._quantile_dim_2d(` — 1 site (hint 1488).
   - `{float32, float64}` → `{mtorch.float32, mtorch.float64}` — 1 site (hint 1462).
   - `Tensor` — zero edits (alias line).
   - `_C.einsum` fast path (hint 1067) — zero edits.

3. Root edits:
   - Delete blocks A–D (nothing else; the `_C` binding lines and the two `Tensor.` patch
     lambdas listed above stay exactly where they are).
   - Add under the §0-b marker:

     ```python
     from ._numeric import (
         bucketize,
         cumulative_trapezoid,
         diff,
         einsum,
         gradient,
         quantile,
         trapezoid,
     )
     ```

4. Confirm nothing else in root still calls the moved private helpers:

   ```bash
   grep -n '_einsum_\|_slice_for_diff\|_normalize_dim_for_input\|_trapezoid_areas\|_quantile_value\|_quantile_index_pair\|_quantile_select\|_gradient_dims\|_gradient_spacing\|_gradient_coordinate\|_gradient_uniform_python\|_gradient_scalar_spacing\|_gradient_shape_for_axis' mtorch/__init__.py
   ```

   Expected: 0 hits.

**Verification**:

```bash
python3 tools/check_unbound_names.py mtorch/_numeric.py
python3 - <<'EOF'
import mtorch
a = mtorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(mtorch.einsum("ij->ji", a).tolist())
print(mtorch.diff(mtorch.tensor([1.0, 3.0, 6.0])).tolist())
print(float(mtorch.trapezoid(mtorch.tensor([1.0, 2.0, 3.0]))))
print(float(mtorch.quantile(mtorch.tensor([1.0, 2.0, 3.0, 4.0]), 0.5)))
t = mtorch.tensor([1.0, 3.0, 6.0]); print(t.diff().tolist())   # Tensor patch still works
EOF
```

Expected: checker silent; `[[1.0, 3.0], [2.0, 4.0]]`, `[2.0, 3.0]`, `4.0`, `2.5`,
`[2.0, 3.0]`. Then STANDARD VERIFICATION (§0-c). Note: the checker will not flag the
shadowed builtins `sum`/`bool` — that is exactly why the edit list above must be applied
by count (`grep -c 'mtorch.bool' mtorch/_numeric.py` → 2, `grep -c 'mtorch.sum(' ` → 1).

**On failure**: STANDARD RECOVERY (§0-e) for step 1-5.

**Commit** (§0-f): `refactor(phase1-5): split out mtorch/_numeric.py`

---

## Step 1-6: `mtorch/nn/` package (the biggest mountain; commit per sub-step)

**Preconditions for 1-6-a**: Step 1-5 done.
**On failure** (any sub-step): STANDARD RECOVERY (§0-e) for that sub-step.

### 1-6-a: Preparation — evacuate the Tensor patch inside the nn move range

Verified: within the nn-related range there is exactly **one** Tensor patch:
`Tensor.tensor_split = _tensor_tensor_split` (hint L3167). Confirm nothing new appeared:

```bash
grep -nE '^\s*Tensor\.[A-Za-z_]+ *=' mtorch/__init__.py
```

Every hit must be at a hint line < 1830 except the single `Tensor.tensor_split` hit
(hint 3167). If there are others in the 2350–5690 range, STOP and mark BLOCKED.

**Actions**:

1. Cut the contiguous block from `def tensor_split(input, indices_or_sections, dim=0):`
   (hint 3125) through `Tensor.tensor_split = _tensor_tensor_split` (hint 3167) — it
   contains `tensor_split` (public, stays in root!), `_tensor_tensor_split`, and the
   patch line. Anchors:

   ```bash
   grep -n '^def tensor_split(\|^Tensor.tensor_split' mtorch/__init__.py
   ```

2. Paste the block, prefixed by this comment line, **immediately above**
   `class _GradModeContext:` (`grep -n '^class _GradModeContext' mtorch/__init__.py`,
   hint 1959):

   ```python
   # --- Tensor patch moved out of the nn scope (goes to _tensor_patches.py in step 1-7) ---
   ```

   No edits to the block itself (it stays inside root; all names resolve as before).

**Verification**: `python3 -c "import mtorch; t = mtorch.tensor([1.0,2.0,3.0,4.0]); print([x.tolist() for x in t.tensor_split(2)])"`
→ `[[1.0, 2.0], [3.0, 4.0]]`. Then STANDARD VERIFICATION (§0-c).

**Commit** (§0-f): `refactor(phase1-6a): evacuate tensor_split block out of the nn range`

### 1-6-b: `mtorch/nn/parameter.py` and `mtorch/nn/modules/module.py`

**Goal**: Create the nn package skeleton and move `Parameter` and `Module` (and, contrary
to the terse old description, the `Module.to()` helper functions, which reference `Module`
and `Parameter` and therefore must live in module.py). The `nn` pseudo-module stays;
root keeps the names visible so the wiring lines keep working. Full pseudo-module removal
is 1-6-e.

**Actions**:

1. Locate the cut ranges (after 1-6-a the two ranges below are contiguous in the file):

   ```bash
   grep -n 'class _RemovableHandle:\|class _IncompatibleKeys(\|^class Module:\|class _ParameterMeta(\|^class Parameter(\|def _as_parameter\|^_DTYPE_NAMES = \|def _is_dtype_arg\|def _convert_module_state_value\|def _pair' mtorch/__init__.py
   # hints: 2352, 2374, 2383, 2961, 2966, 2979, 2985, 2989, 3056, and _pair marks the END boundary
   ```

   - module.py part 1: `class _RemovableHandle:` ... end of `class Module` (the line
     before `class _ParameterMeta(type):`).
   - parameter.py: `class _ParameterMeta(type):` ... end of `_as_parameter` (the line
     before `_DTYPE_NAMES = ...`).
   - module.py part 2 (Module.to helpers): `_DTYPE_NAMES = ...` ... end of
     `_convert_module_state_value` (the line before `def _pair(value):`).

2. Create `mtorch/nn/parameter.py`: header + parameter.py cut range verbatim + edits.

   ```python
   """Parameter (equivalent to torch/nn/parameter.py)."""

   from __future__ import annotations

   import mtorch
   from mtorch import _C

   Tensor = _C.Tensor
   ```

   Edits: `result = tensor(data, requires_grad=requires_grad)` →
   `result = mtorch.tensor(data, requires_grad=requires_grad)` — exactly 1 site.
   (`Tensor`, `_C._is_parameter`, `_C._mark_parameter`: zero edits.)

3. Create `mtorch/nn/modules/module.py`: header + module.py part 1 + part 2, verbatim + edits.

   ```python
   """Module base class (equivalent to torch/nn/modules/module.py)."""

   from __future__ import annotations

   import builtins as _builtins
   from collections import OrderedDict, namedtuple

   import mtorch
   from mtorch import _C
   from mtorch.nn.parameter import Parameter

   Tensor = _C.Tensor
   ```

   Edits (exact sites; `grep -n` inside the new file):
   - In `Module.float/double/half` (hints 2857/2860/2863): `dtype=float32` →
     `dtype=mtorch.float32`, `dtype=float64` → `dtype=mtorch.float64`,
     `dtype=float16` → `dtype=mtorch.float16` — 3 sites.
   - In `_is_dtype_arg`: `isinstance(value, dtype)` → `isinstance(value, mtorch.dtype)` — 1 site.
   - In `_is_device_arg`: `isinstance(value, device)` → `isinstance(value, mtorch.device)` — 1 site.
   - In `_parse_module_to_args`: `_normalize_memory_format(memory_format)` →
     `mtorch._normalize_memory_format(memory_format)` — 1 site.
   - In `_module_tensor_memory_format`: `preserve_format` → `mtorch.preserve_format`,
     `channels_last` → `mtorch.channels_last` (do not touch `channels_last_3d` on the
     same pass), `channels_last_3d` → `mtorch.channels_last_3d` — 3 sites total.
   - `Tensor`, `Parameter`, `OrderedDict`, `namedtuple`, `_builtins`, `_C._mark_parameter`:
     zero edits.

4. Create `mtorch/nn/modules/__init__.py` with exactly:

   ```python
   """nn.modules subpackage (layer files are added in step 1-6-d)."""
   ```

5. Create `mtorch/nn/__init__.py` with exactly:

   ```python
   """mtorch.nn (equivalent to torch/nn/). Grows step by step during Phase 1-6."""

   from mtorch.nn import parameter as parameter
   from mtorch.nn.parameter import Parameter
   from mtorch.nn.modules.module import Module
   ```

6. Root edits:
   - Replace the three deleted ranges (one combined location) with:

     ```python
     from .nn.modules.module import Module
     from .nn.parameter import Parameter, _as_parameter
     ```

     (Root still needs `Module`/`Parameter` for the remaining layer classes and wiring,
     and `_as_parameter` for `ParameterList`/`ParameterDict` until 1-6-d.)
   - Replace the two lines `parameter = ModuleType("mtorch.nn.parameter")` and
     `parameter.Parameter = Parameter` (hints L5679–5680) with:

     ```python
     from .nn import parameter as parameter
     ```

     Keep the next line `nn.parameter = parameter` (hint L5681) — it now attaches the
     real module to the pseudo `nn`. The `_sys.modules[... ".nn.parameter"]` line (hint
     L6085) also stays — it now registers the real module.

**Verification**:

```bash
python3 tools/check_unbound_names.py mtorch/nn/modules/module.py
python3 tools/check_unbound_names.py mtorch/nn/parameter.py
python3 - <<'EOF'
import mtorch
import mtorch.nn.parameter, mtorch.nn.modules.module
print(mtorch.nn.Module is mtorch.Module, mtorch.nn.Parameter is mtorch.Parameter)
lin = mtorch.nn.Linear(2, 2)          # still the root-defined class, wired onto pseudo nn
print(isinstance(lin.weight, mtorch.nn.Parameter))
EOF
```

Expected: checkers silent; `True True` and `True`. Then STANDARD VERIFICATION (§0-c),
plus `pytest tests/compat/test_nn_modules.py -q | tail -3`.

**Commit** (§0-f): `refactor(phase1-6b): nn/parameter.py and nn/modules/module.py`

### 1-6-c: `mtorch/nn/functional.py` and `mtorch/nn/init.py`

**Goal**: Make `functional` and `init` real modules; delete the dead `_dropout`; keep the
root-level public aliases (`conv1d`, `cosine_similarity`, ... — all in `__all__`) alive
via a trailing import.

**Actions**:

1. Dead-code deletion (explicitly permitted): the Python `_dropout` (hint L5310–5334) is
   wired nowhere (`functional.dropout = _C.dropout`). Confirm, then delete the whole def:

   ```bash
   grep -rn '_dropout' mtorch/__init__.py tests/compat/*.py
   ```

   The only permitted hits: the `def _dropout(` line itself and test names containing
   `dropout` as a substring of longer words (e.g. `test_dropout_...`,
   `scaled_dot_product_attention_dropout_one`). No call site `_dropout(` may exist.

2. Locate the cut blocks:

   ```bash
   grep -n 'def _pair(\|def _pixel_shuffle(\|^conv3d = _conv3d\|def _scaled_dot_product_attention(\|def _init_constant_(\|def _init_kaiming_normal_(\|^functional.relu = \|^functional.log_softmax = \|^init.calculate_gain = \|^init.kaiming_normal_ = ' mtorch/__init__.py
   ```

   - Block F1: `def _pair(value):` ... end of `_cosine_similarity` (the line before
     `def _pixel_shuffle(`). Contains `_pair`, `_triple`, `_single`, `_one_hot`,
     `_cosine_similarity`.
   - Block F2: `def _pixel_shuffle(` ... `conv3d = _conv3d` (inclusive). Contains
     `_pixel_shuffle`, `_pixel_unshuffle`, `_channel_shuffle`, `_adaptive_avg_pool1d`,
     `_adaptive_avg_pool2d`, the root alias lines `cosine_similarity = _cosine_similarity`,
     `pixel_shuffle = _pixel_shuffle`, `pixel_unshuffle = _pixel_unshuffle`,
     `channel_shuffle = _channel_shuffle`, `adaptive_avg_pool1d = _adaptive_avg_pool1d`,
     then `_validate_conv_string_padding`, `_reverse_repeat_padding`,
     `_conv_string_padding`, `_conv1d`, `conv1d = _conv1d`, `_conv2d`, `conv2d = _conv2d`,
     `_conv3d`, `conv3d = _conv3d`.
   - Block F3: `def _scaled_dot_product_attention(` ... end of `_softmin` (the line
     before `def _init_constant_(`). Contains `_scaled_dot_product_attention`,
     `_normalize`, `_glu`, `_logsigmoid`, `_softmin`.
   - Block I1: `def _init_constant_(` ... end of `_init_kaiming_normal_` (the line before
     `functional.relu = _C.relu`). Contains the 11 `_init_*` functions plus
     `_calculate_fan_in_and_fan_out` and `_calculate_gain`.
   - Wiring W-func: `functional.relu = _C.relu` ... `functional.log_softmax = _C.log_softmax`
     (hints 5515–5573; 59 lines).
   - Wiring W-init: `init.calculate_gain = _calculate_gain` ... `init.kaiming_normal_ = _init_kaiming_normal_`
     (hints 5582–5592; 11 lines).

3. Create `mtorch/nn/functional.py`: header, then F1, F2, F3 verbatim in that order,
   then the alias block, then apply the edit list.

   Header:

   ```python
   """nn.functional (equivalent to torch/nn/functional.py)."""

   from __future__ import annotations

   import builtins as _builtins
   import math as _math

   import mtorch
   from mtorch import _C
   ```

   Inside the pasted F2, **delete** the 8 root alias lines
   (`cosine_similarity = _cosine_similarity`, `pixel_shuffle = ...`,
   `pixel_unshuffle = ...`, `channel_shuffle = ...`, `adaptive_avg_pool1d = ...`,
   `conv1d = _conv1d`, `conv2d = _conv2d`, `conv3d = _conv3d`) — the alias block below
   re-creates all public names in wiring order.

   Alias block (this is wiring block W-func with the `functional.` prefix stripped —
   paste exactly; it must match the current W-func lines, so if W-func differs from the
   hints, regenerate it with `sed 's/^functional\.//'` applied to the W-func lines):

   ```python
   relu = _C.relu
   leaky_relu = _C.leaky_relu
   silu = _C.silu
   elu = _C.elu
   selu = _C.selu
   softplus = _C.softplus
   hardtanh = _C.hardtanh
   gelu = _C.gelu
   relu6 = _C.relu6
   hardsigmoid = _C.hardsigmoid
   hardswish = _C.hardswish
   softsign = _C.softsign
   logsigmoid = _logsigmoid
   glu = _glu
   mish = _C.mish
   dropout = _C.dropout
   linear = _C.linear
   conv1d = _conv1d
   conv_transpose1d = _C.conv_transpose1d
   conv2d = _conv2d
   conv3d = _conv3d
   conv_transpose2d = _C.conv_transpose2d
   conv_transpose3d = _C.conv_transpose3d
   max_pool1d = _C.max_pool1d
   avg_pool1d = _C.avg_pool1d
   max_pool2d = _C.max_pool2d
   avg_pool2d = _C.avg_pool2d
   adaptive_avg_pool1d = _adaptive_avg_pool1d
   adaptive_avg_pool2d = _adaptive_avg_pool2d
   unfold = _C.unfold
   fold = _C.fold
   pad = _C.pad
   interpolate = _C.interpolate
   grid_sample = _C.grid_sample
   affine_grid = _C.affine_grid
   upsample = _C.interpolate
   scaled_dot_product_attention = _scaled_dot_product_attention
   normalize = _normalize
   cosine_similarity = _cosine_similarity
   one_hot = _one_hot
   pixel_shuffle = _pixel_shuffle
   pixel_unshuffle = _pixel_unshuffle
   channel_shuffle = _channel_shuffle
   layer_norm = _C.layer_norm
   rms_norm = _C.rms_norm
   batch_norm = _C.batch_norm
   group_norm = _C.group_norm
   embedding = _C.embedding
   mse_loss = _C.mse_loss
   l1_loss = _C.l1_loss
   nll_loss = _C.nll_loss
   cross_entropy = _C.cross_entropy
   binary_cross_entropy = _C.binary_cross_entropy
   binary_cross_entropy_with_logits = _C.binary_cross_entropy_with_logits
   sigmoid = _C.sigmoid
   tanh = _C.tanh
   softmin = _softmin
   softmax = _C.softmax
   log_softmax = _C.log_softmax
   ```

   Edit list for the pasted bodies (exact sites):
   - `Size` → `mtorch.Size` — 6 sites: in `_pair`, `_triple`, `_single`,
     `_adaptive_avg_pool1d`, `_adaptive_avg_pool2d` (the `isinstance(..., (tuple, list, Size))`
     lines) and in `_normalize` (`isinstance(dim, (tuple, list, Size))`).
   - In `_cosine_similarity`: `sum(` → `mtorch.sum(` — 3 sites; `sqrt(` → `mtorch.sqrt(`
     — 2 sites; `clamp_min(` → `mtorch.clamp_min(` — 2 sites.
   - In `_scaled_dot_product_attention`: `repeat_interleave(` → `mtorch.repeat_interleave(`
     — 2 sites; `matmul(` → `mtorch.matmul(` — 2 sites; `transpose(` → `mtorch.transpose(`
     — 1 site; `tensor(` → `mtorch.tensor(` — 1 site; `dtype=bool` → `dtype=mtorch.bool`
     — 1 site; `attn_mask.dtype == bool` → `attn_mask.dtype == mtorch.bool` — 1 site;
     `where(` → `mtorch.where(` — 2 sites; `full(` → `mtorch.full(` — 2 sites;
     `softmax(scores, dim=-1)` → `mtorch.softmax(scores, dim=-1)` — 1 site.
   - In `_normalize`: `_normalize_l2(` → `_C._normalize_l2(` — 1 site; `norm(` →
     `mtorch.norm(` — 1 site; `clamp_min(` → `mtorch.clamp_min(` — 1 site;
     `{float16, float32, float64}` → `{mtorch.float16, mtorch.float32, mtorch.float64}` — 1 site.
   - In `_glu`: `split(` → `mtorch.split(` — 1 site; `sigmoid(second)` →
     `mtorch.sigmoid(second)` — 1 site.
   - In `_logsigmoid`: `functional.softplus(-input)` → `softplus(-input)` — 1 site.
   - In `_softmin`: `softmax(-input, ...)` → `mtorch.softmax(-input, ...)` — 1 site.
   - `_one_hot`, `_pixel_*`, `_channel_shuffle`, conv helpers: `_C.*` only — zero edits.

4. Create `mtorch/nn/init.py`: header + I1 verbatim (**zero body edits** — verified the
   block uses only `_builtins`, `_math`, `_C.trunc_normal_` and tensor methods) + alias
   block (= W-init with `init.` stripped):

   ```python
   """nn.init (equivalent to torch/nn/init.py)."""

   from __future__ import annotations

   import builtins as _builtins
   import math as _math

   from mtorch import _C
   ```

   ...I1 pasted here verbatim...

   ```python
   calculate_gain = _calculate_gain
   constant_ = _init_constant_
   zeros_ = _init_zeros_
   ones_ = _init_ones_
   normal_ = _init_normal_
   uniform_ = _init_uniform_
   trunc_normal_ = _init_trunc_normal_
   xavier_uniform_ = _init_xavier_uniform_
   xavier_normal_ = _init_xavier_normal_
   kaiming_uniform_ = _init_kaiming_uniform_
   kaiming_normal_ = _init_kaiming_normal_
   ```

5. Append to `mtorch/nn/__init__.py`, **immediately after** the
   `from mtorch.nn import parameter as parameter` line (submodule imports must precede
   the class imports so that layer files can do `from mtorch.nn import functional`):

   ```python
   from mtorch.nn import functional as functional
   from mtorch.nn import init as init
   ```

6. Root edits:
   - Delete blocks F1, F2, F3, I1 and the `_dropout` def.
   - Delete wiring blocks W-func and W-init.
   - Replace the two creation lines `functional = ModuleType("mtorch.nn.functional")`
     and `init = ModuleType("mtorch.nn.init")` (hints L2050, L2052) with:

     ```python
     from .nn import functional as functional
     from .nn import init as init
     ```

     (Safe at this position: functional.py/init.py only need `mtorch` (call-time) and
     `_C`. Root global `functional` must exist here because the layer classes still in
     root call `functional.relu(...)` etc. at forward time, and the wiring lines
     `nn.functional = functional` / `nn.init = init` (hints L5676–5677) still execute —
     keep those two lines; they now attach the real modules to the pseudo `nn`. The
     `_sys.modules` lines for `".nn.functional"` and `".nn.init"` (hints L6082, L6084)
     also stay until 1-6-e.)
   - Add under the §0-b marker (these keep the root-level public names from the deleted
     alias lines; all 8 are in `__all__`):

     ```python
     from .nn.functional import (
         adaptive_avg_pool1d,
         channel_shuffle,
         conv1d,
         conv2d,
         conv3d,
         cosine_similarity,
         pixel_shuffle,
         pixel_unshuffle,
     )
     ```

**Verification**:

```bash
python3 tools/check_unbound_names.py mtorch/nn/functional.py
python3 tools/check_unbound_names.py mtorch/nn/init.py
python3 - <<'EOF'
import mtorch
import mtorch.nn.functional as F
assert mtorch.nn.functional is F and mtorch.functional is F
assert mtorch.conv2d is F.conv2d and mtorch.cosine_similarity is F.cosine_similarity
x = mtorch.randn(2, 3)
print(F.relu(x).shape, F.softmin(x, dim=1).shape, F.normalize(x).shape)
print(mtorch.nn.init.calculate_gain("relu"))
EOF
```

Expected: checkers silent; three shapes printed; `1.414...`. Then STANDARD VERIFICATION
(§0-c) plus `pytest tests/compat/test_nn_modules.py -q | tail -3`. The checker cannot
flag shadowed builtins — additionally confirm by count:
`grep -c 'mtorch.bool' mtorch/nn/functional.py` → 2,
`grep -c 'mtorch.sum(' mtorch/nn/functional.py` → 3.

**Commit** (§0-f): `refactor(phase1-6c): nn/functional.py and nn/init.py (drop dead _dropout)`

### 1-6-d: The `mtorch/nn/modules/` layer group

**Goal**: Move every layer class out of root, one file = one commit, in the table order.
Do not rewrite class bodies except the per-file qualification lists. Classes reference
the name `functional` (e.g. `functional.conv1d(...)`); each layer file imports
`from mtorch.nn import functional`, so those references need **zero edits**.

The old table missed several classes that sit between its line ranges; they are assigned
here to their upstream-torch homes (leave a note in PROGRESS.md that the table was
extended): `Upsample`/`UpsamplingNearest2d`/`UpsamplingBilinear2d` → `upsampling.py`,
`PixelShuffle`/`PixelUnshuffle` → `pixelshuffle.py`, `ChannelShuffle` → `channelshuffle.py`,
`Identity` → `linear.py`, `Dropout` → `dropout.py`, `Flatten`/`Unflatten` → `flatten.py`,
`Sigmoid`/`Tanh`/`Softmax`/`Softmin`/`LogSoftmax`/`Softmax2d` → `activation.py`,
`CosineSimilarity` → `distance.py`.

Common procedure per file (referred to below as MOVE(file)):

1. Re-grep the class start lines: `grep -n '^class ' mtorch/__init__.py`. Cut each listed
   class from its `class X(...)` line through the line before the next top-level
   statement. All classes for one file are pasted into the new file in their original
   order.
2. New file = the file's header (given below) + pasted classes + the file's edit list.
3. Root: replace the (first) cut location with the file's root-import line (given below).
   If a file's classes came from two separate locations (only `linear.py` and
   `activation.py`), the second location is deleted with no replacement.
4. Append the file's line to `mtorch/nn/__init__.py` (at the end).
5. Run: `python3 tools/check_unbound_names.py mtorch/nn/modules/<file>` (must be silent),
   then STANDARD VERIFICATION (§0-c), then commit
   `refactor(phase1-6d): nn/modules/<file>`.

Base header (used by every file below unless noted):

```python
"""<description> (equivalent to torch/nn/modules/<file>)."""

from __future__ import annotations

import builtins as _builtins

import mtorch
from mtorch.nn import functional
from mtorch.nn.modules.module import Module
from mtorch.nn.parameter import Parameter
```

Process the table strictly top to bottom:

| # | File | Classes (start-line hints) | Header additions / removals | Edit list (exact) | Root-import replacement line | nn/__init__.py line |
|---|---|---|---|---|---|---|
| 1 | `linear.py` | `Linear` (3306); `Identity` (4764, second location) | — | `tensor(` → `mtorch.tensor(` (2), `float32` → `mtorch.float32` (2) | `from .nn.modules.linear import Identity, Linear` | `from mtorch.nn.modules.linear import Identity, Linear` |
| 2 | `conv.py` | `Conv1d` (3338), `Conv2d` (3433), `Conv3d` (3534), `ConvTranspose1d` (3642), `ConvTranspose2d` (3741), `ConvTranspose3d` (3853) | add `from mtorch.nn.functional import _conv_string_padding, _pair, _reverse_repeat_padding, _single, _triple, _validate_conv_string_padding` | `tensor(` → `mtorch.tensor(` and `float32` → `mtorch.float32` (24 sites combined; count with `grep -cE 'tensor\(|float32' `) | `from .nn.modules.conv import (Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d)` (parenthesized, one name per line) | `from mtorch.nn.modules.conv import (...)` same 6 names |
| 3 | `upsampling.py` | `Upsample` (3975), `UpsamplingNearest2d` (4001), `UpsamplingBilinear2d` (4006) | remove `_builtins`, `mtorch`, `Parameter` lines | none | `from .nn.modules.upsampling import Upsample, UpsamplingBilinear2d, UpsamplingNearest2d` | same names from `mtorch.nn.modules.upsampling` |
| 4 | `pooling.py` | `MaxPool1d` (4011), `AvgPool1d` (4040), `AdaptiveAvgPool1d` (4066), `MaxPool2d` (4074), `AvgPool2d` (4103), `AdaptiveAvgPool2d` (4132), `Unfold` (4140), `Fold` (4151) | remove `mtorch`, `Parameter`; add `from mtorch.nn.functional import _pair, _single` | none | `from .nn.modules.pooling import (AdaptiveAvgPool1d, AdaptiveAvgPool2d, AvgPool1d, AvgPool2d, Fold, MaxPool1d, MaxPool2d, Unfold)` | same names |
| 5 | `padding.py` | `_pad_tuple` (function, 4163), `_ConstantPadNd` (4173), `_NonConstantPadNd` (4184), `ConstantPad1d/2d/3d` (4195/4199/4203), `ZeroPad1d/2d/3d` (4207/4214/4221), `ReflectionPad1d/2d/3d` (4228/4233/4238), `ReplicationPad1d/2d/3d` (4243/4248/4253), `CircularPad1d/2d/3d` (4258/4263/4268) | remove `mtorch`, `Parameter` | none | `from .nn.modules.padding import (CircularPad1d, CircularPad2d, CircularPad3d, ConstantPad1d, ConstantPad2d, ConstantPad3d, ReflectionPad1d, ReflectionPad2d, ReflectionPad3d, ReplicationPad1d, ReplicationPad2d, ReplicationPad3d, ZeroPad1d, ZeroPad2d, ZeroPad3d)` | same names |
| 6 | `pixelshuffle.py` | `PixelShuffle` (4273), `PixelUnshuffle` (4281) | remove `mtorch`, `Parameter` | none | `from .nn.modules.pixelshuffle import PixelShuffle, PixelUnshuffle` | same names |
| 7 | `channelshuffle.py` | `ChannelShuffle` (4289) | remove `mtorch`, `Parameter` | none | `from .nn.modules.channelshuffle import ChannelShuffle` | same name |
| 8 | `attention.py` | `MultiheadAttention` (4297) | add `import math as _math` and `from mtorch.nn.modules.linear import Linear` | qualify with `mtorch.`: `cat(`, `full(`, `logical_and(`, `logical_not(`, `mean(`, `narrow(`, `reshape(`, `squeeze(`, `tensor(`, `transpose(`, `where(`, `zeros(`, `float32` (44 call sites combined; count with `grep -cE '(^|[^.[:alnum:]_])(cat|full|logical_and|logical_not|mean|narrow|reshape|squeeze|tensor|transpose|where|zeros|float32)\(' ` before, same count on `mtorch.<name>(` after; do not touch `.reshape(`/`.transpose(` method calls or `functional.*`) | `from .nn.modules.attention import MultiheadAttention` | same name |
| 9 | `sparse.py` | `Embedding` (4581) | — | `tensor(` → `mtorch.tensor(` (1), `float32` → `mtorch.float32` (1) | `from .nn.modules.sparse import Embedding` | same name |
| 10 | `activation.py` | `ReLU` (4635), `ReLU6` (4640), `LeakyReLU` (4648), `SiLU` (4657), `ELU` (4665), `SELU` (4674), `Softplus` (4682), `Hardtanh` (4691), `GELU` (4705), `Hardsigmoid` (4713), `Hardswish` (4721), `Softsign` (4729), `LogSigmoid` (4734), `GLU` (4739), `Mish` (4747); second location: `Sigmoid` (5009), `Tanh` (5014), `Softmax` (5019), `Softmin` (5027), `LogSoftmax` (5035), `Softmax2d` (5043) | remove `mtorch`, `Parameter` | none | `from .nn.modules.activation import (ELU, GELU, GLU, Hardsigmoid, Hardswish, Hardtanh, LeakyReLU, LogSigmoid, LogSoftmax, Mish, ReLU, ReLU6, SELU, SiLU, Sigmoid, Softmax, Softmax2d, Softmin, Softplus, Softsign, Tanh)` | same names |
| 11 | `dropout.py` | `Dropout` (4755) | remove `mtorch`, `Parameter` | none | `from .nn.modules.dropout import Dropout` | same name |
| 12 | `flatten.py` | `Flatten` (4769), `Unflatten` (4778) | remove `Parameter`, `functional` | `flatten(` → `mtorch.flatten(` (1), `unflatten(` → `mtorch.unflatten(` (1) | `from .nn.modules.flatten import Flatten, Unflatten` | same names |
| 13 | `container.py` | `Sequential` (4787), `ModuleList` (4834), `ModuleDict` (4877), `ParameterList` (4927), `ParameterDict` (4965) | remove `mtorch`, `Parameter`, `functional`; add `from collections import OrderedDict` and `from mtorch.nn.parameter import _as_parameter` | none | `from .nn.modules.container import ModuleDict, ModuleList, ParameterDict, ParameterList, Sequential` | same names |
| 14 | `normalization.py` | `LayerNorm` (5050), `RMSNorm` (5080), `BatchNorm1d` (5096), `BatchNorm2d` (5155), `BatchNorm3d` (5171), `GroupNorm` (5187) | — | qualify with `mtorch.`: `ones(` (5), `zeros(` (4), `tensor(` (1), `float32` (5), `int64` (1) — 16 sites total | `from .nn.modules.normalization import (BatchNorm1d, BatchNorm2d, BatchNorm3d, GroupNorm, LayerNorm, RMSNorm)` | same names |
| 15 | `distance.py` | `CosineSimilarity` (5206) | remove `mtorch`, `Parameter`, `_builtins` | none | `from .nn.modules.distance import CosineSimilarity` | same name |
| 16 | `loss.py` | `MSELoss` (5215), `L1Loss` (5225), `NLLLoss` (5235), `CrossEntropyLoss` (5253), `BCELoss` (5281), `BCEWithLogitsLoss` (5292) | remove `mtorch`, `Parameter` | none | `from .nn.modules.loss import (BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss, NLLLoss)` | same names |

Notes:

- After each move, root still contains the pseudo-`nn` wiring (`nn.Linear = Linear`, ...);
  the root-import replacement lines keep those names bound, so the wiring keeps working
  until 1-6-e. The root-import lines also intentionally preserve the historical root
  attributes (`mtorch.Linear` etc. have always existed); they are NOT removed in 1-6-e.
- After #13 (`container.py`), root no longer calls `_as_parameter`, but the 1-6-b import
  line `from .nn.parameter import Parameter, _as_parameter` stays exactly as it is (do
  not "clean it up"; the root attribute existed before this phase).
- If `grep -n '^class ' mtorch/__init__.py` shows a class between `_RemovableHandle` and
  `MSELoss` that is not in the table: STOP, mark the step BLOCKED in PROGRESS.md with the
  class name, and do not guess.

After all 16 commits, confirm no layer classes remain in root:

```bash
grep -n '^class ' mtorch/__init__.py
```

Expected hits only: `dtype`, `device`, `memory_format`, `layout`, `Size`, `finfo`,
`iinfo`, `_TensorConstructor`, `FloatTensor`, `DoubleTensor`, `HalfTensor`, `LongTensor`,
`IntTensor`, `BoolTensor`, `_GradModeContext`, `_InferenceModeContext`.

### 1-6-e: Completing `nn/__init__.py`, `nn/attention.py`, and removing the pseudo-module

**Goal**: Replace the pseudo `nn`/`attention` modules with the real package; give root the
tail import and the historical `functional` alias.

**Actions**:

1. Create `mtorch/nn/attention.py` with exactly (reproduces the old `attention.*` wiring,
   hints L6072–6077):

   ```python
   """nn.attention (equivalent to torch/nn/attention/)."""

   from mtorch.backends.cuda import SDPAParams as SDPAParams
   from mtorch.backends.cuda import SDPBackend as SDPBackend
   from mtorch.backends.cuda import _can_use_efficient_attention as can_use_efficient_attention
   from mtorch.backends.cuda import _can_use_flash_attention as can_use_flash_attention
   from mtorch.backends.cuda import _sdpa_kernel as sdpa_kernel

   WARN_FOR_UNFUSED_KERNELS = False
   ```

2. Overwrite `mtorch/nn/__init__.py` with exactly this final content (this is the old
   `nn.*` wiring block, hints L5593–5677, reproduced as imports; the submodule imports
   come first — do not reorder):

   ```python
   """mtorch.nn (equivalent to torch/nn/)."""

   from mtorch.nn import attention as attention
   from mtorch.nn import functional as functional
   from mtorch.nn import init as init
   from mtorch.nn import modules as modules
   from mtorch.nn import parameter as parameter
   from mtorch.nn.parameter import Parameter
   from mtorch.nn.modules.module import Module
   from mtorch.nn.modules.linear import Identity, Linear
   from mtorch.nn.modules.conv import (
       Conv1d,
       Conv2d,
       Conv3d,
       ConvTranspose1d,
       ConvTranspose2d,
       ConvTranspose3d,
   )
   from mtorch.nn.modules.upsampling import Upsample, UpsamplingBilinear2d, UpsamplingNearest2d
   from mtorch.nn.modules.pooling import (
       AdaptiveAvgPool1d,
       AdaptiveAvgPool2d,
       AvgPool1d,
       AvgPool2d,
       Fold,
       MaxPool1d,
       MaxPool2d,
       Unfold,
   )
   from mtorch.nn.modules.padding import (
       CircularPad1d,
       CircularPad2d,
       CircularPad3d,
       ConstantPad1d,
       ConstantPad2d,
       ConstantPad3d,
       ReflectionPad1d,
       ReflectionPad2d,
       ReflectionPad3d,
       ReplicationPad1d,
       ReplicationPad2d,
       ReplicationPad3d,
       ZeroPad1d,
       ZeroPad2d,
       ZeroPad3d,
   )
   from mtorch.nn.modules.pixelshuffle import PixelShuffle, PixelUnshuffle
   from mtorch.nn.modules.channelshuffle import ChannelShuffle
   from mtorch.nn.modules.attention import MultiheadAttention
   from mtorch.nn.modules.sparse import Embedding
   from mtorch.nn.modules.activation import (
       ELU,
       GELU,
       GLU,
       Hardsigmoid,
       Hardswish,
       Hardtanh,
       LeakyReLU,
       LogSigmoid,
       LogSoftmax,
       Mish,
       ReLU,
       ReLU6,
       SELU,
       SiLU,
       Sigmoid,
       Softmax,
       Softmax2d,
       Softmin,
       Softplus,
       Softsign,
       Tanh,
   )
   from mtorch.nn.modules.dropout import Dropout
   from mtorch.nn.modules.flatten import Flatten, Unflatten
   from mtorch.nn.modules.container import (
       ModuleDict,
       ModuleList,
       ParameterDict,
       ParameterList,
       Sequential,
   )
   from mtorch.nn.modules.normalization import (
       BatchNorm1d,
       BatchNorm2d,
       BatchNorm3d,
       GroupNorm,
       LayerNorm,
       RMSNorm,
   )
   from mtorch.nn.modules.distance import CosineSimilarity
   from mtorch.nn.modules.loss import (
       BCELoss,
       BCEWithLogitsLoss,
       CrossEntropyLoss,
       L1Loss,
       MSELoss,
       NLLLoss,
   )
   ```

   Cross-check it against the wiring before deleting the wiring: every name in the
   `nn.X = X` lines (hints L5593–5675) must appear in the file above
   (`grep -c '^nn\.' mtorch/__init__.py` → 87 including `nn.functional`/`nn.init`/
   `nn.parameter`/`nn.attention`).

3. Root edits:
   - Delete `nn = ModuleType("mtorch.nn")` (hint L2049) and
     `attention = ModuleType("mtorch.nn.attention")` (hint L2051).
   - Delete the entire remaining `nn.*` wiring block: `nn.Module = Module` ...
     `nn.init = init` (hints L5593–5677) and `nn.parameter = parameter` (hint L5681).
   - Delete the `nn.attention = attention` line and the `attention.*` wiring (hints
     L6071–6077).
   - Delete the five `_sys.modules` lines for `".nn"`, `".nn.functional"`,
     `".nn.attention"`, `".nn.init"`, `".nn.parameter"` (hints L6081–6085).
   - Delete the two mid-file lines added in 1-6-c: `from .nn import functional as functional`
     and `from .nn import init as init`, and the 1-6-b line
     `from .nn import parameter as parameter` (root no longer executes any statement that
     needs them before the tail).
   - The root import of the 1-4-d SDP names (`from .backends.cuda import (...)`) is now
     only needed for... nothing in root. Leave it as is — `mtorch.SDPBackend` etc. were
     root globals before this phase and removing them would shrink the surface.
   - Add under the §0-b marker, after the existing lines:

     ```python
     from . import nn as nn
     functional = nn.functional  # historical alias: mtorch.functional has always been nn.functional
     init = nn.init  # historical alias
     parameter = nn.parameter  # historical alias
     ```

**Verification**:

```bash
python3 - <<'EOF'
import mtorch
import mtorch.nn as nn
import mtorch.nn.functional as F
import mtorch.nn.attention
from mtorch.nn import Linear, Conv2d, Module, Parameter
assert nn.Linear is Linear and F.conv2d is not None
assert mtorch.functional is F and nn.functional is F
assert type(nn).__name__ == "module" and nn.__file__.endswith("nn/__init__.py")
assert mtorch.nn.attention.SDPBackend is mtorch.backends.cuda.SDPBackend
print("ok")
EOF
pytest tests/compat/test_nn_modules.py -q | tail -3
```

Expected: `ok`; nn tests pass. Then STANDARD VERIFICATION (§0-c). If `import mtorch.nn`
still returns a pseudo-module (`nn.__file__` assertion fails), a `_sys.modules` line
survived — re-check Action 3.

**Commit** (§0-f): `refactor(phase1-6e): real nn package, drop nn pseudo-module`

---

## Step 1-7: `mtorch/_tensor_patches.py`

**Goal**: Consolidate every remaining `Tensor.xxx = ...` patch in root, plus the helper
functions and raw-descriptor captures those patches use, into `mtorch/_tensor_patches.py`
— a module that applies the patches at import time. It is imported **first** in the
trailing section (patches must be applied before anything can call Tensor methods).

**Preconditions**: Step 1-6-e done. List the patches:

```bash
grep -nE '^\s*Tensor\.[A-Za-z_]+ *=' mtorch/__init__.py
```

Expected: the hits listed in the groups below (exactly 55 assignments), nothing in any
other region. An unexpected hit → STOP, BLOCKED.

**Actions**:

1. Create `mtorch/_tensor_patches.py` starting with exactly:

   ```python
   """Methods/properties monkey-patched onto the Tensor class (equivalent to torch/_tensor.py).

   Importing this module applies the patches. It must be imported first in the
   trailing section of mtorch/__init__.py, before any other submodule.
   """

   from __future__ import annotations

   import builtins as _builtins

   import mtorch
   from mtorch import _C

   Tensor = _C.Tensor
   ```

2. Move the following groups into the file **in this exact order** (original file order),
   each verbatim + its listed edits. "Capture lines" read the raw `_C` attribute before
   it is patched — they work unchanged here because this module runs before any patch is
   applied. Root loses each moved line/def; everything NOT listed stays in root.

   - **Group A — dtype/device/layout properties** (hints 176–199). Move: the capture
     lines `_tensor_dtype_descriptor = Tensor.dtype` and
     `_tensor_device_descriptor = Tensor.device`; the defs `_tensor_dtype_property`,
     `_tensor_device_property`; the 11 `Tensor.` assignments (`dtype`, `device`,
     `layout`, `is_cuda`, `is_mps`, `is_sparse`, `is_sparse_csr`, `is_mkldnn`,
     `is_quantized`, `is_meta`, `is_nested`).
     Edits: in `_tensor_dtype_property`, replace
     `return globals().get(str(name), dtype(str(name)))` with:

     ```python
     value = getattr(mtorch, str(name), None)
     if isinstance(value, mtorch.dtype):
         return value
     return mtorch.dtype(str(name))
     ```

     In `_tensor_device_property`: `device(` → `mtorch.device(` — 1 site.
     In the layout lambda: `strided` → `mtorch.strided` — 1 site.
   - **Group B — numpy/float/int/index** (hints 386–414). Move: defs `_tensor_numpy`,
     `_tensor_float`, `_tensor_int`, `_tensor_index`; assignments `Tensor.numpy`,
     `Tensor.__float__`, `Tensor.__int__`, `Tensor.__index__`.
     Edits: `_numpy_dtype_for_mtorch(` → `mtorch._numpy_dtype_for_mtorch(` — 1 site
     (the def stays in root; it is also used by `from_numpy`).
   - **Group C — in-place clamp** (hints 417–435). Move: defs `_tensor_clamp_`,
     `_tensor_clamp_min_`, `_tensor_clamp_max_`; assignments `Tensor.clamp_`,
     `Tensor.clip_`, `Tensor.clamp_min_`, `Tensor.clamp_max_`.
     Edits: `clamp(` → `mtorch.clamp(`, `clamp_min(` → `mtorch.clamp_min(`,
     `clamp_max(` → `mtorch.clamp_max(` — 3 sites.
   - **Group D — type()/new()** (hints 506–539). Move: defs `_dtype_from_type_request`,
     `_tensor_type`, `_tensor_new`; assignments `Tensor.type`, `Tensor.new`.
     (Verified: `_dtype_from_type_request` has no other users in root.)
     Edits: `isinstance(request, dtype)` → `isinstance(request, mtorch.dtype)` — 1 site;
     `_TensorConstructor` → `mtorch._TensorConstructor` — 1 site;
     `_TENSOR_DTYPE_BY_TYPE_NAME` → `mtorch._TENSOR_DTYPE_BY_TYPE_NAME` — 2 sites;
     `_TENSOR_TYPE_NAME_BY_DTYPE` → `mtorch._TENSOR_TYPE_NAME_BY_DTYPE` — 1 site;
     `empty(` → `mtorch.empty(` — 3 sites; `tensor(` → `mtorch.tensor(` — 1 site.
   - **Group E — add/sub with alpha** (hints 759–762, 791–821). Move: capture lines
     `_Tensor_add = Tensor.add`, `_Tensor_sub = Tensor.sub`, `_Tensor_add_ = Tensor.add_`,
     `_Tensor_sub_ = Tensor.sub_`; defs `_tensor_add`, `_tensor_sub`,
     `_tensor_add_alpha_`, `_tensor_sub_alpha_`; assignments `Tensor.add`, `Tensor.sub`,
     `Tensor.subtract`, `Tensor.add_`, `Tensor.sub_`.
     (Root keeps `_alpha_scaled_other`, `add`, `sub`, `subtract` — they are public.)
     Edits: `add(self, other, alpha=alpha)` → `mtorch.add(...)` — 2 sites;
     `sub(self, other, alpha=alpha)` → `mtorch.sub(...)` — 2 sites.
   - **Group F — clamp lambdas** (hints 846–849). Move: `Tensor.clamp = lambda ...`,
     `Tensor.clip = Tensor.clamp`, `Tensor.clamp_min = lambda ...`,
     `Tensor.clamp_max = lambda ...`.
     Edits: `clamp(` → `mtorch.clamp(`, `clamp_min(` → `mtorch.clamp_min(`,
     `clamp_max(` → `mtorch.clamp_max(` — 3 sites. `Tensor.clip = Tensor.clamp` unchanged.
   - **Group G — diff/quantile lambdas** (hints 1205–1209, 1500–1502; multi-line). Move
     both lambdas. Edits: `diff(` → `mtorch.diff(` — 1 site; `quantile(` →
     `mtorch.quantile(` — 1 site.
   - **Group H — broadcast_to** (hints 1519–1525). Move: def `_tensor_broadcast_to`;
     assignment `Tensor.broadcast_to`.
     Edits: `Size` → `mtorch.Size` — 1 site; `broadcast_to(self, tuple(shape))` →
     `mtorch.broadcast_to(self, tuple(shape))` — 1 site.
   - **Group I — tril_/triu_/aliases** (hints 1547–1560). Move: defs `_tensor_tril_`,
     `_tensor_triu_`; assignments `Tensor.tril_`, `Tensor.triu_`,
     `Tensor.ndimension = Tensor.dim`, `Tensor.nelement = Tensor.numel`.
     Edits: `tril(` → `mtorch.tril(`, `triu(` → `mtorch.triu(` — 2 sites.
   - **Group J — index_put/take_along_dim/unique/bernoulli** (hints 1594–1625). Move:
     defs `_tensor_index_put`, `_tensor_index_put_`; assignments/lambdas
     `Tensor.take_along_dim`, `Tensor.index_put`, `Tensor.index_put_`, `Tensor.unique`,
     `Tensor.unique_consecutive`, `Tensor.bernoulli`.
     (Root keeps `_normalize_index_put_indices`, `index_put`, `take_along_dim` — public
     / shared.)
     Edits: `index_put(self, ...)` → `mtorch.index_put(self, ...)` — 1 site;
     `_index_put_native(` → `_C.index_put(` — 1 site (in `_tensor_index_put_`);
     `_normalize_index_put_indices(` → `mtorch._normalize_index_put_indices(` — 1 site;
     `take_along_dim(` → `mtorch.take_along_dim(` — 1 site; `unique(` →
     `mtorch.unique(` — 1 site; `unique_consecutive(` → `mtorch.unique_consecutive(` — 1
     site; `bernoulli(` → `mtorch.bernoulli(` — 1 site.
   - **Group K — T/H/mT/mH** (hints 1634–1655). Move: defs `_tensor_T`, `_tensor_mT`,
     `_tensor_H`; assignments `Tensor.T`, `Tensor.H`, `Tensor.mT`, `Tensor.mH`.
     Edits: `permute(` → `mtorch.permute(` — 1 site; `transpose(` → `mtorch.transpose(`
     — 2 sites.
   - **Group L — reduction lambdas** (hints 1812–1824). Move: `Tensor.sum`,
     `Tensor.mean`, `Tensor.all`, `Tensor.any`, `Tensor.var`, `Tensor.std`,
     `Tensor.bincount` (all lambdas; var/std span 3 lines each).
     Edits: `sum(` → `mtorch.sum(`, `mean(` → `mtorch.mean(`, `all(` → `mtorch.all(`,
     `any(` → `mtorch.any(`, `var(` → `mtorch.var(`, `std(` → `mtorch.std(`,
     `bincount(` → `mtorch.bincount(` — 7 sites.
   - **Group M — tensor_split** (moved in 1-6-a to just above `class _GradModeContext`,
     marked by the 1-6-a comment). Move: def `_tensor_tensor_split`; assignment
     `Tensor.tensor_split`; delete the 1-6-a comment line. (`def tensor_split` itself
     STAYS in root — it is public.)
     Edits: `return tensor_split(self, ...)` → `return mtorch.tensor_split(self, ...)` — 1 site.

3. Root edit: insert `from . import _tensor_patches  # noqa: F401` as the **first** line
   under the §0-b marker (above every other trailing import).

4. Cross-check — no patches left in root, and the patches module has them all:

   ```bash
   grep -cE '^\s*Tensor\.[A-Za-z_]+ *=' mtorch/__init__.py          # must print 0
   grep -cE '^Tensor\.[A-Za-z_]+ *=' mtorch/_tensor_patches.py       # must print 55
   ```

**Verification**:

```bash
python3 tools/check_unbound_names.py mtorch/_tensor_patches.py
python3 - <<'EOF'
import mtorch
t = mtorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(t.dtype, t.device, t.T.tolist(), t.sum().item(), t.clamp(min=2.0).tolist())
print(t.type(), float(t.mean()), t.ndimension(), t.nelement())
import numpy  # noqa
print(t.numpy().shape)
EOF
```

Expected: checker silent; all lines print without AttributeError, `t.type()` prints
`torch.FloatTensor`. Then STANDARD VERIFICATION (§0-c). The checker will not flag the
shadowed builtins `sum/all/any/bool/min/max` — the Group L counts above are the guard
(`grep -c 'mtorch.sum(' mtorch/_tensor_patches.py` → 1, etc.).

**On failure**: STANDARD RECOVERY (§0-e) for step 1-7.

**Commit** (§0-f): `refactor(phase1-7): consolidate Tensor patches into mtorch/_tensor_patches.py`

---

## Step 1-8: Finishing touches on root

**Goal**: Prove the pseudo-module machinery is fully gone, drop its dead imports, and
verify the frozen API surface.

**Preconditions**: Step 1-7 done.

**Actions**:

1. Zero pseudo-modules and zero manual registrations:

   ```bash
   grep -n 'ModuleType(' mtorch/__init__.py    # must print nothing
   grep -n '_sys.modules' mtorch/__init__.py   # must print nothing besides (possibly) none
   ```

   If the second grep still shows the registration block (hints L6081–6098 originally),
   something was missed in 1-4/1-6 — go back to the step that owns those lines. When both
   greps are clean, also delete the now-dead imports at the top of root, each only after
   its own zero-usage check:

   ```bash
   grep -c 'ModuleType' mtorch/__init__.py     # 1 (the import) -> delete `from types import ModuleType`
   grep -c '_sys'       mtorch/__init__.py     # 1 (the import) -> delete `import sys as _sys`
   grep -c 'OrderedDict\|namedtuple' mtorch/__init__.py
   # if this prints 1 (only the import line), delete `from collections import OrderedDict, namedtuple`
   ```

   If any count is greater than 1, the name is still used — keep that import and list the
   remaining users in PROGRESS.md Notes instead.

2. The trailing section should now read (order matters only for `_tensor_patches` being
   first and `functional = nn.functional` coming after `from . import nn as nn`):

   ```python
   # ============================================================
   # Load the actual submodules.
   # Must be placed after all definitions and all Tensor patches, before __all__.
   # ============================================================
   from . import _tensor_patches  # noqa: F401
   from . import serialization as serialization
   from .serialization import save, load
   from . import optim as optim
   from . import autograd as autograd
   from . import amp as amp
   from . import cuda as cuda
   from . import mps as mps
   from . import backends as backends
   from . import utils as utils
   from . import linalg as linalg
   from ._numeric import (
       bucketize,
       cumulative_trapezoid,
       diff,
       einsum,
       gradient,
       quantile,
       trapezoid,
   )
   from .nn.functional import (
       adaptive_avg_pool1d,
       channel_shuffle,
       conv1d,
       conv2d,
       conv3d,
       cosine_similarity,
       pixel_shuffle,
       pixel_unshuffle,
   )
   from . import nn as nn
   functional = nn.functional  # historical alias: mtorch.functional has always been nn.functional
   init = nn.init  # historical alias
   parameter = nn.parameter  # historical alias
   ```

   If yours differs only in line order (with the two constraints kept), leave it.

3. Do not change `__all__`. Mechanically confirm every item resolves:

   ```bash
   python3 - <<'EOF'
   import mtorch
   missing = [name for name in mtorch.__all__ if not hasattr(mtorch, name)]
   print("missing:", missing)
   assert not missing
   EOF
   ```

4. Full verification:

   ```bash
   pytest tests/compat/test_api_surface.py -q | tail -3
   pytest tests/compat/test_serialization.py -q | tail -3
   ```

   Then STANDARD VERIFICATION (§0-c), and additionally compare the collection count:

   ```bash
   grep -oE '[0-9]+ tests collected' docs/design/baseline/collect-count.txt
   pytest tests/compat --collect-only -q | grep -oE '[0-9]+ tests collected'
   ```

   Both lines must be identical.

**On failure**: STANDARD RECOVERY (§0-e) for step 1-8.

**Commit** (§0-f): `refactor(phase1-8): root cleanup, pseudo-module machinery fully removed`

---

## Pitfall list (read this first when in trouble)

| Symptom | Cause and fix |
|---|---|
| `ImportError: cannot import name ...` while importing mtorch | Circular import: a leaf module does `from mtorch import <name>` at top level. Change to `import mtorch` + call-time `mtorch.<name>` (§0-a). `from mtorch import _C` is always fine. |
| `AttributeError: module 'mtorch' has no attribute ...` at call time | A re-export was forgotten. Compare the trailing section with the Step 1-8 listing; check the step that moved the name. |
| `NameError: name 'X' is not defined` inside a moved file | A root name was not qualified. Run `python3 tools/check_unbound_names.py <file>`; if it is silent, X shadows a builtin (`sum`, `bool`, `all`, `any`, `min`, `max`) — re-apply that step's edit-count checks. |
| `import mtorch.nn` returns the old pseudo-module | A `_sys.modules[...]` registration line survived. Step 1-6-e Action 3 / Step 1-8 Action 1. |
| `ModuleNotFoundError: mtorch.nn.functional` mid-phase (1-6-b..1-6-d) | The `_sys.modules` lines for `.nn.functional`/`.nn.init`/`.nn.parameter` were deleted too early — they must survive until 1-6-e (with pseudo `nn` still shadowing, the import machinery needs them). Restore them. |
| test_serialization fails | `_dtype_by_name` changed behavior (it must fall back to `mtorch.float32` for unknown names, never raise) or `__module__`-sensitive expectations; check `mtorch/serialization.py` against the Step 1-1 listing. |
| kind mismatch / missing entry in test_api_surface | Usually a missing re-export, not the pseudo→real module change (both have kind "module"). Read the failing path and add the corresponding import. |
| A Tensor method is missing (`'Tensor' object has no attribute 'clamp_'`) | `from . import _tensor_patches` is not the first line of the trailing section, or a patch group was skipped. Re-run the Step 1-7 cross-check counts. |
| `mtorch.cuda.amp` import fails after 1-4-c | `mtorch/cuda/` was created as a module (`cuda.py`) instead of a package. It must be `mtorch/cuda/__init__.py` + `mtorch/cuda/amp.py`. |
| Baseline summary differs by a benchmark-only count | Benchmarks are part of `pytest tests/compat`; a slower machine changes times, never counts. Any count difference is real — bisect with `git stash` / step-specific test files. |

## Phase completion criteria

1. `mtorch/__init__.py` is roughly 2,000 lines or fewer (metadata classes, factory
   wrappers, `_C` re-exports, grad-mode contexts, `compile`, public math wrappers,
   the trailing section, and `__all__` remain).
2. `pytest tests/compat` breakdown and collection count match
   `docs/design/baseline/` exactly (§0-c procedure).
3. `git log --oneline` shows one commit per step/sub-step
   (1-0, 1-1, 1-2, 1-3, 1-4a–1-4f, 1-5, 1-6-a, 1-6-b, 1-6-c, 16 × 1-6-d, 1-6-e, 1-7, 1-8).
4. PROGRESS.md: all Phase 1 boxes `[x]` with commit hashes and dates; Notes mention the
   two scope concretizations (linalg.py in 1-4; the extended 1-6-d file table).
