# 01. Common Work Rules and Verification Procedures

Common to all phases. **Always read this entire document before starting work.**
If a rule in this document appears to contradict an instruction in a phase guide
(02–07), stop working, mark the current step **BLOCKED** in
`docs/design/PROGRESS.md` (see §6, item 5), and wait for a human.

All commands in this document are written to be run from the repository root:

```bash
cd /Users/hiramatsu/dev/mtorch
```

Run that `cd` once at the start of every shell session, then copy-paste the
commands below unchanged.

Important timing note: git is initialized, but **the source tree is untracked
until Phase 0 step 0-2** (`02-phase0-baseline.md`) creates the full-tree snapshot
commit (before that, only `README.md`, `LICENSE`, `pyproject.toml`, and `docs/`
are tracked, from housekeeping commits). Every `git ...` recovery command in this
document therefore applies to source files **only after step 0-2 is complete**
(files under `docs/` are restorable at any time). Before that point there is no
git-based recovery for source files: if something breaks during step 0-1 or 0-2,
record the situation in `docs/design/PROGRESS.md` and stop.

## 1. The Basic Work Cycle

Every step follows this cycle. There are no exceptions.

```
1. Open docs/design/PROGRESS.md. The first line whose checkbox is `[ ]` and that
   is not marked **BLOCKED** is your task. Do exactly that step, nothing else.
2. Open the phase guide named in the PROGRESS.md section header (e.g.
   "procedure: 03-phase1-python-package.md") and read the section for that step.
3. Make the change exactly as the guide instructs.
4. Build:  python3 setup.py build_ext --inplace
   Run this in EVERY step, even for Python-only changes (it finishes in seconds
   when no C++ file changed; this removes the need to decide whether C++ was touched).
5. Run the tests and compare against the baseline using the procedure in §5.1.
   If the phase guide says the step also requires a benchmark comparison, run §5.2.
6. Commit (one commit per step; message format in §7).
7. Update PROGRESS.md for that step and commit the update (exact recipe in §8).
```

Never commit unless step 5 passed exactly as described in §5.1.
If step 5 does not pass and you cannot restore it, follow "§6 Protocol on Failure."

## 2. Iron Rules (Prohibitions)

1. **No rewriting tests to make them pass.** If a test fails, the cause is in
   your change. You must not change expected values, comparison conditions,
   tolerances, or add `skip`/`xfail` anywhere under `tests/` or `compat/`
   (the test-infrastructure work in Phase 4 is an exception, but even then you
   must not change the meaning of any comparison).
2. **No changes not in the procedure guide.** Even if you think "I could fix
   this while I'm here," do not fix it. Write at most 3 lines about it in the
   "Notes:" field of the current phase in `docs/design/PROGRESS.md` instead.
3. **No mixing logic changes into a mechanical-move commit.** Allowed in a move
   commit: cutting and pasting code verbatim, adding `#include`/`import` lines,
   renaming namespaces (only when the guide instructs it), and adding
   declarations. Nothing else. Move commits are checked with §5.3.
4. **No changes to public interfaces** unless the guide explicitly instructs it:
   the function signatures in `cpp/mtorch/core/tensor.h`, the symbols exported
   by `mtorch._C`, and the public names of `mtorch.*`.
5. **No skipping or reordering steps.**
6. **No `git push` or any other operation that leaves this machine.** A remote
   exists (`origin` → https://github.com/kazhiramatsu/mtorch), but pushing is
   the human maintainer's decision — never push, create PRs, or touch the
   remote in any way as part of this refactoring.
7. **No `git reset --hard`.** Undo only with the §6 procedure
   (`git restore` for uncommitted changes, `git revert` for committed ones).

## 3. Environment Check (once at the start of each work session)

**Goal**: Confirm the toolchain, the reference PyTorch, and the mtorch build all
work before touching anything.

**Preconditions**: None (this is the first thing you run). Check you are in the
repository root: `pwd` must print `/Users/hiramatsu/dev/mtorch`.

**Actions** (run in this order):

```bash
python3 -c "import torch; print(torch.__version__)"    # 1. reference PyTorch imports
python3 setup.py build_ext --inplace                   # 2. the C++ build passes
python3 -c "import mtorch; print(mtorch.__version__)"  # 3. mtorch imports (prints 0.0.0)
git status                                             # 4. only AFTER Phase 0 step 0-2
```

**Verification**: Commands 1–3 exit with status 0 and print a version string
(command 3 prints `0.0.0`). Command 4 (only applicable after step 0-2) prints
`nothing to commit, working tree clean`.

**On failure**: Do not start work. If command 4 shows uncommitted changes you did
not make in this session, do not touch them. Record which command failed and its
first error line in the "Notes:" field of the current phase in
`docs/design/PROGRESS.md`, append `**BLOCKED**` to the current step's line (§6,
item 5), and stop.

**Commit**: None (this procedure changes nothing).

## 4. Standard Command Reference

| Purpose | Command |
|---|---|
| Build | `python3 setup.py build_ext --inplace` |
| Full compatibility tests | `pytest tests/compat` |
| Tests (narrowed to an area) | `pytest tests/compat --compat-op 'binary.*'` |
| A specific file only | `pytest tests/compat/test_serialization.py` |
| Test collection count | `pytest tests/compat --collect-only -q \| tail -2` |
| Full benchmark (JSON output) | `pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 --compat-benchmark-json benchmark-results/current.json` |
| Benchmark (one case) | `pytest tests/compat/test_benchmarks.py --compat-benchmark 'bench.binary_add_64x64'` |
| Benchmark comparison | `python3 tools/compare_benchmarks.py docs/design/baseline/benchmark-baseline.json benchmark-results/current.json` |

Notes:

- `tools/compare_benchmarks.py` and everything under `docs/design/baseline/`
  **do not exist until Phase 0** (created in steps 0-3 and 0-4 of
  `02-phase0-baseline.md`). Do not use the last row of the table, or §5.1/§5.2,
  before those steps are complete.
- The pytest options `--compat-op`, `--compat-benchmark`,
  `--compat-benchmark-repeat`, `--compat-benchmark-json` are defined in
  `tests/conftest.py` (locate with `grep -n 'pytest_addoption' tests/conftest.py`,
  currently around L12). Pattern values are `fnmatch` patterns against case IDs.
- `pytest tests/compat` also runs the 415 benchmark tests (at the default
  `--compat-benchmark-repeat=5`), so the full run takes minutes to tens of
  minutes. Set a command timeout of at least 30 minutes and never abort partway.
- When running benchmarks, do not run any other build or test in parallel on
  this machine (results become noisy).

## 5. Details of the Verification Procedure

### 5.0 How the compat harness works (background; read once)

You never need to modify these files, but you must understand what "green" means:

- `tests/conftest.py` defines the harness options. The reference module is
  `torch` (`--compat-reference`), the candidate is `mtorch`
  (`--compat-candidate`). Both are imported into the same pytest process by the
  session fixtures `torch_reference` / `torch_candidate` in
  `tests/compat/conftest.py`.
- Test cases are data tables: `OP_CASES`, `INPLACE_CASES`, `VIEW_CASES`,
  `GRAD_CASES` in `tests/compat/cases.py`, and `BENCHMARK_CASES` (415 cases) in
  `tests/compat/benchmarking.py` (locate with
  `grep -n 'BENCHMARK_CASES' tests/compat/benchmarking.py`, currently around
  L8191). `pytest_generate_tests` in `tests/compat/conftest.py` parametrizes one
  pytest test per case; the full suite currently collects 2272 tests.
- Each functional case is executed **twice, once on torch and once on mtorch**,
  by `assert_same_result` in `tests/compat/harness.py`. It compares: raised
  exception type name, result shape, dtype name, device type, `requires_grad`,
  stride (when the case sets `check_stride`), and finally the values via
  `torch.testing.assert_close` after converting the mtorch result to a torch
  tensor. For in-place cases it also compares the argument tensors after the
  call. Gradient cases additionally compare `.grad` of every input.
- API-surface tests compare against the manifest `compat/api_surface_seed.json`.
  Never delete or edit that file.
- Each benchmark case (`run_benchmark_case` in `tests/compat/benchmarking.py`)
  first asserts value compatibility, then times torch and mtorch (warmup 2,
  repeat = `--compat-benchmark-repeat`) and records
  `ratio = mtorch median / torch median`. `--compat-benchmark-json` writes
  `{"benchmarks": [{"id": ..., "ratio": ...}, ...]}`.

Because every comparison is against live PyTorch in the same process, a red test
means mtorch's behavior changed. It never means "the test is wrong" (Iron Rule 1).

### 5.1 Confirming Test Result Consistency

**Goal**: Prove the current tree produces exactly the same test breakdown as the
recorded baseline.

**Preconditions**: Phase 0 step 0-4 is complete, i.e. this file exists:

```bash
ls docs/design/baseline/tests-baseline.txt docs/design/baseline/collect-count.txt
```

(If that `ls` fails and you are still inside Phase 0, follow the phase guide
instead of this section.)

**Actions**:

1. Run the full suite and capture the output:

   ```bash
   pytest tests/compat 2>&1 | tee /tmp/tests-current.txt
   ```

2. Print the normalized final summary line of the baseline and of the current
   run (the normalization strips the `=` bars and the elapsed time, which always
   differ):

   ```bash
   tail -1 docs/design/baseline/tests-baseline.txt | tr -d '=' | sed 's/ in .*//' | xargs
   tail -1 /tmp/tests-current.txt | tr -d '=' | sed 's/ in .*//' | xargs
   ```

   Worked example: if the last line of a run is
   `============ 2200 passed, 72 xfailed in 812.34s (0:13:32) ============`,
   the command prints exactly `2200 passed, 72 xfailed`.

3. Compare the collection counts:

   ```bash
   grep -oE '[0-9]+ tests collected' docs/design/baseline/collect-count.txt
   pytest tests/compat --collect-only -q | grep -oE '[0-9]+ tests collected'
   ```

**Verification**: PASS if and only if BOTH hold:

- The two lines printed in Action 2 are character-for-character identical
  (every count — passed, failed, xfailed, skipped, error — matches).
- The two lines printed in Action 3 are identical (currently
  `2272 tests collected`; the committed baseline file is authoritative, not this
  number).

Any difference is a FAILURE. In particular: `failed` increased by even 1 →
failure; `passed` decreased → failure (dropped tests); collection count changed
→ failure.

**On failure**: Go to §6. Do not re-run "to see if it goes away"; the suite is
deterministic.

**Commit**: None (this is a check, not a change). `/tmp/tests-current.txt` is
scratch output; never commit it.

### 5.2 Comparing Benchmark Results (mandatory for phases that change C++)

**Goal**: Prove the step did not make any benchmark case more than 5% slower
relative to PyTorch.

**Preconditions**: Phase 0 steps 0-3 and 0-4 are complete, i.e. both exist:

```bash
ls tools/compare_benchmarks.py docs/design/baseline/benchmark-baseline.json
```

**Actions**:

1. Run the full benchmark and write JSON:

   ```bash
   mkdir -p benchmark-results
   pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
       --compat-benchmark-json benchmark-results/current.json
   ```

2. Compare against the baseline:

   ```bash
   python3 tools/compare_benchmarks.py \
       docs/design/baseline/benchmark-baseline.json benchmark-results/current.json
   ```

   `tools/compare_benchmarks.py` (created verbatim in `02-phase0-baseline.md`
   step 0-3) matches cases by ID and compares the per-case ratio
   (mtorch median / torch median) old vs new. It prints one `REGRESSION ...`
   line per case degraded by more than 5%, one `MISSING ...` line per case that
   disappeared, and a final `compared=... regressions=... missing=... new=...`
   line. Exit code 0 = clean, 1 = regressions or missing cases.

**Verification**: PASS if and only if the final line reads `... regressions=0
missing=0 ...` (the `new=` count must also be 0 unless the phase guide for the
current step says new benchmark cases are expected).

**On failure**:

- If `missing=` is nonzero → dropped benchmarks. That is an immediate failure;
  go to §6.
- If `regressions=` is nonzero, the degradation may be noise. Recheck each
  reported case mechanically, 3 times. Worked example for a report line
  `REGRESSION bench.binary_add_64x64: ratio 0.980 -> 1.120 (x1.14)` — run the
  following block 3 times, changing only `recheck1` to `recheck2`, `recheck3`:

  ```bash
  pytest tests/compat/test_benchmarks.py --compat-benchmark 'bench.binary_add_64x64' \
      --compat-benchmark-repeat=10 --compat-benchmark-json benchmark-results/recheck1.json
  python3 tools/compare_benchmarks.py \
      docs/design/baseline/benchmark-baseline.json benchmark-results/recheck1.json \
      | grep '^REGRESSION bench.binary_add_64x64' || true
  ```

  (Ignore `MISSING` lines and the exit code here — the recheck JSON contains
  only one case, so every other baseline case is reported missing by design.
  Only the presence or absence of the `REGRESSION` line for the rechecked case
  matters.)

  Decision rule: the `grep` printed a line in **2 or more of the 3 reruns** →
  real performance regression; apply the performance-guard procedure of the
  current phase guide if it defines one, otherwise go to §6 and revert the
  step's commit. The `grep` printed a line in 0 or 1 of the 3 reruns → noise;
  the check PASSES; record one note line in PROGRESS.md
  (e.g. `bench.binary_add_64x64 flagged once, not reproducible (1/3)`).

**Commit**: None. `benchmark-results/` is listed in `.gitignore` (added in step
0-2); never commit anything from it.

### 5.3 Checking a Mechanical-Move Commit

**Goal**: Confirm a commit labeled as a mechanical move by the phase guide only
moved code, within a small tolerance for added `#include`s and declarations.

**Preconditions**: The move commit is `HEAD`. Check:
`git log --oneline -1` shows the step ID you just committed.

**Actions**: Run this exact command; it parses
`git show --stat HEAD` and applies the tolerance
`limit = max(10, min(insertions, deletions) / 20)` (i.e. 5% of the amount
moved, but never less than 10 lines):

```bash
git show --stat HEAD | tail -1 | python3 -c "import re,sys; s=sys.stdin.read(); i=int((re.search(r'(\d+) insertion',s) or [0,0])[1]); d=int((re.search(r'(\d+) deletion',s) or [0,0])[1]); moved=min(i,d); slack=abs(i-d); limit=max(10, moved//20); print(f'insertions={i} deletions={d} slack={slack} limit={limit}'); print('MOVE-CHECK OK' if slack<=limit else 'MOVE-CHECK EXCEEDED')"
```

Worked example: a commit whose stat line is
`3 files changed, 412 insertions(+), 405 deletions(-)` prints
`insertions=412 deletions=405 slack=7 limit=20` then `MOVE-CHECK OK`.

**Verification**: The last line printed is `MOVE-CHECK OK`.

**On failure** (`MOVE-CHECK EXCEEDED`): unless the phase guide for this exact
step states that extra lines are expected (and roughly how many), treat it as a
failed step: revert with `git revert --no-edit HEAD`, then follow §6 items 4–6
(record and mark **BLOCKED**). Do not attempt to "fix up" the diff.

**Commit**: None (the commit being checked already exists; a revert produced by
the failure path is its own commit).

## 6. Protocol on Failure

Follow this whenever the build fails, §5.1 or §5.2 fails, or a phase guide's
verification fails. Reminder: everything here that uses git applies **only after
Phase 0 step 0-2**; before that, skip items 3–4, do items 5–6 without the
`git add`/`git commit` (just edit the file), and stop.

1. Assume **your immediately preceding change is the cause** (the repository is
   green at baseline). Read only the FIRST error message; for C++ compile
   errors, everything after the first error is usually a cascade.
2. You may attempt to fix it, but stop attempting after **3 fix attempts or 30
   minutes, whichever comes first**. Each attempt must end with the full cycle
   `python3 setup.py build_ext --inplace` then the §5.1 check.
3. If still failing, undo the step:
   - Changes not yet committed (this form also undoes anything already staged
     with `git add` or `git mv`):

     ```bash
     git restore --staged --worktree .
     git status --short
     ```

     For every new (untracked) file the step created, `git status --short`
     shows a `??` line; delete each one explicitly by path, e.g.:

     ```bash
     git clean -fd -- mtorch/optim.py
     ```

     Never run `git clean -fd` without a path.
   - Changes already committed:

     ```bash
     git revert --no-edit HEAD
     ```

     (If the bad step produced more than one commit, revert each, newest
     first, one `git revert --no-edit <hash>` per commit; find the hashes with
     `git log --oneline -5`.)
4. Confirm you are back at baseline:

   ```bash
   python3 setup.py build_ext --inplace
   ```

   then run the full §5.1 procedure. It must PASS before you continue to item 5.
5. Mark the step **BLOCKED** in `docs/design/PROGRESS.md`:
   - On the step's own line, keep the checkbox `[ ]` and append ` **BLOCKED**`
     at the end. Worked example — change

     ```
     - [ ] 1-2 mtorch/optim.py — commit: / date:
     ```

     to

     ```
     - [ ] 1-2 mtorch/optim.py — commit: / date: **BLOCKED**
     ```

   - Under the same phase's `Notes:` line, add at most 3 lines: what you
     changed, the first error message, what you tried.
6. Commit the PROGRESS.md update:

   ```bash
   git add docs/design/PROGRESS.md && git commit -m "progress: mark 1-2 BLOCKED"
   ```

   (replace `1-2` with the actual step ID).
7. **Stop working.** Do not proceed to the next step; a BLOCKED step waits for
   human review. Never improvise a workaround.

## 7. Commit Discipline

- 1 step = 1 commit (plus one separate PROGRESS.md commit per §8; plus revert
  commits produced by §6). Include the step ID from the guide in the message,
  format `refactor(<step-id>): <what happened>`:

  ```
  refactor(phase1-2): split out mtorch/optim.py
  refactor(phase2b-4): carve views.cpp out of tensor.cpp
  ```

- Immediately before committing, run `git status --short` and confirm every
  listed file is one the phase guide told you to touch. Build artifacts
  (`build/`, `*.so`, `benchmark-results/`, `.pytest_cache/`) must never appear;
  they are covered by `.gitignore` from step 0-2. If an unintended file
  appears, restore it (`git restore <file>`) before committing.
- Stage explicitly by path — e.g.
  `git add mtorch/optim.py mtorch/__init__.py && git commit -m "refactor(phase1-2): split out mtorch/optim.py"`.
  Use `git add -A` only where a phase guide explicitly says so.
- After committing, run `git show --stat HEAD`, and for any commit the phase
  guide labels a mechanical move, run the §5.3 check.

## 8. Rules for Updating PROGRESS.md

- Read `docs/design/PROGRESS.md` at the start of every work session. Only ever
  edit (a) the line of the step you just finished or blocked, and (b) the
  `Notes:` field of the current phase. Touch nothing else in the file.
- On completing a step (i.e. after the step's own commit from §7):

  1. Get the hash of the work commit: `git rev-parse --short HEAD`
     (example output: `a1b2c3d`).
  2. Get today's date: `date +%F` (example output: `2026-07-09`).
  3. Edit the step's line: `[ ]` becomes `[x]`, and the hash and date are
     filled in. Worked example — change

     ```
     - [ ] 1-2 mtorch/optim.py — commit: / date:
     ```

     to

     ```
     - [x] 1-2 mtorch/optim.py — commit: a1b2c3d / date: 2026-07-09
     ```

  4. Commit the update as its own commit (do NOT amend the work commit — the
     recorded hash must stay valid):

     ```bash
     git add docs/design/PROGRESS.md && git commit -m "progress: complete 1-2"
     ```

     (replace `1-2` with the actual step ID).

- On failure or interruption: use §6 items 5–6 (checkbox stays `[ ]`,
  ` **BLOCKED**` appended, ≤3-line note, separate commit).

## 9. Re-locating Line Numbers and Symbols

The line numbers in the guides are as of the 2026-07-08 snapshot. They drift as
work progresses. Treat every line number as a hint, never as an address.

- **Locate positions by symbol, using the grep command given next to each line
  number in the guides.** Example: locate with
  `grep -n 'def _serialize_value' mtorch/__init__.py` (currently around L6101).
- Never open `cpp/mtorch/core/tensor.cpp` (~32k lines) or
  `cpp/mtorch/python/module.cpp` (~12k lines) in full. Always `grep -n` for the
  symbol first, then read a small window around the reported line.
- If a grep finds nothing, the code has probably already been moved by an
  earlier step. Check where it went before concluding anything:

  ```bash
  git log --oneline -20
  grep -rn 'def _serialize_value' mtorch/
  ```

  and cross-check completed steps in `docs/design/PROGRESS.md`. If it is still
  not found anywhere, treat the current step as failed and follow §6 (do not
  guess).
