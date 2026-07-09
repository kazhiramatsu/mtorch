# 02. Phase 0: Laying the Foundation (git setup, baseline recording, artifact cleanup)

Prerequisite: You must have finished reading `01-rules-and-verification.md`.
Do not start any other phase until this phase is complete.

Objectives:

1. Start git management and secure a rollback mechanism
2. Record the "green current state" as a baseline (the pass/fail criterion for all subsequent phases)
3. Prepare a benchmark comparison script
4. Delete unreferenced artifacts (the 403 `compat/benchmark_*.json` files)

## Conventions for this document

- **Run every command from the repository root**: `/Users/hiramatsu/dev/mtorch`.
  Every shell block below assumes this working directory. If unsure, run `cd /Users/hiramatsu/dev/mtorch` first.
- **Git state at the start of this phase**: git is already initialized (branch `main`,
  `user.name`/`user.email` configured, remote `origin` set) with housekeeping commits tracking
  only `README.md`, `LICENSE`, `pyproject.toml`, and `docs/`. **Everything else — the entire
  source tree — is untracked until the snapshot commit in step 0-2.** Therefore git can restore
  files under `docs/` at any time, but cannot restore any source file before 0-2 completes; the
  "On failure" sections of 0-1 and 0-2 use re-runs or file backups for source files instead.
- **Marking a step BLOCKED**: open `docs/design/PROGRESS.md`, find the line for the step under
  "## Phase 0" (for example `- [ ] 0-4 Record baseline (tests / collect-count / benchmark) — commit: / date:`),
  append ` **BLOCKED**` to the end of that line, and write what happened (max 3 lines) under the
  `Notes:` line of the Phase 0 section. Then stop working on this phase.
- **Marking a step done**: after the step's commit, run `git rev-parse --short HEAD` and
  `date +%Y-%m-%d`, then edit the step's line in `docs/design/PROGRESS.md` from
  `- [ ] 0-N <title> — commit: / date:` to
  `- [x] 0-N <title> — commit: <hash> / date: <YYYY-MM-DD>` (substitute the real hash and date),
  and commit that edit with the exact command given in each step's **Commit** section.
- Line numbers cited below are hints valid as of writing; always locate code with the given
  `grep` command, not the line number.

---

## Step 0-1: Environment check

**Goal**: Confirm that PyTorch is importable, the C extension builds, and `mtorch` is importable,
before anything else is changed.

**Preconditions**: You are in the repository root and the expected files exist. Check:

```bash
cd /Users/hiramatsu/dev/mtorch && test -f setup.py && test -d tests/compat && test -d compat && echo PRECONDITION-OK
```

Expected output: `PRECONDITION-OK`.

**Actions** (run each command separately, in order):

1. Check that the reference PyTorch is installed:

   ```bash
   python3 -c "import torch; print(torch.__version__)"
   ```

   Expected output: a version string (as of writing: `2.10.0`). Any version string is acceptable;
   an `ImportError` / `ModuleNotFoundError` is a failure.

2. Build the C extension in place (this may take several minutes; do not interrupt):

   ```bash
   python3 setup.py build_ext --inplace
   ```

   Expected: exit code 0. The last lines mention copying/creating a `.so` file
   (e.g. `mtorch/_C.cpython-314-darwin.so`). Compiler warnings are acceptable; errors are not.

3. Check that the candidate package imports:

   ```bash
   python3 -c "import mtorch; print(mtorch.__version__)"
   ```

   Expected output: `0.0.0`
   (defined in `mtorch/__init__.py`; locate with `grep -n "__version__" mtorch/__init__.py`, currently line 17).

**Verification**:

```bash
python3 -c "import torch, mtorch; print('ENV-OK', torch.__version__, mtorch.__version__)"
```

Expected output: `ENV-OK <torch version> 0.0.0`.

**On failure**: Nothing was modified by this step except build outputs, so recovery is: re-run the
failing command once. If it fails again, do NOT attempt to fix the environment or the code. Mark
step 0-1 as **BLOCKED** in `docs/design/PROGRESS.md` (see Conventions above) and record the exact
error message (first and last line) in the Phase 0 `Notes:` field. Stop.

**Commit**: There is no code change to commit. Edit `docs/design/PROGRESS.md`: change the line

```
- [ ] 0-1 Environment check — commit: / date:
```

to (substitute the real date from `date +%Y-%m-%d`):

```
- [x] 0-1 Environment check — commit: (pre-snapshot) / date: <YYYY-MM-DD>
```

Then commit just that file (docs/ is already tracked):

```bash
git add docs/design/PROGRESS.md && git commit -m "progress: mark 0-1 done"
```

---

## Step 0-2: .gitignore extension and full-tree snapshot commit

**Goal**: Extend `.gitignore` and record the exact current state of the whole source tree as a
snapshot commit (including the 403 `compat/benchmark_*.json` files — they are deleted in step 0-5,
so keeping them in the snapshot commit preserves the original state in history). Git itself is
already initialized (see Conventions), so this step does NOT run `git init`.

**Preconditions**: Step 0-1 is done; no snapshot commit exists yet, and the source tree is still
untracked (only the housekeeping files — `README.md`, `LICENSE`, `pyproject.toml`, `docs/` — are
tracked). Check:

```bash
echo "snapshot=$(git log --oneline | grep -c 'snapshot:') tracked-src=$(git ls-files -- mtorch cpp tests compat tools setup.py | wc -l | xargs)"
```

Expected output: `snapshot=0 tracked-src=0`. Anything else — see **On failure** below.

**Actions**:

1. Make a backup of the existing `.gitignore` (recovery mechanism for this step — `.gitignore`
   is still untracked, so git cannot restore it):

   ```bash
   cp .gitignore .gitignore.step02.bak
   ```

2. Overwrite `.gitignore` with the complete 12-line content below. The first 6 lines are the
   existing content (verify with `cat .gitignore.step02.bak`); the rest are new. Copy-paste this
   exact single command:

   ```bash
   printf '%s\n' '__pycache__/' '*.py[cod]' 'build/' 'dist/' '*.egg-info/' '*.so' '.pytest_cache/' '.codex-runs/' '.codex-stop' 'benchmark-results/' 'baseline-tests.txt' '.DS_Store' > .gitignore
   ```

   The resulting file must read exactly:

   ```
   __pycache__/
   *.py[cod]
   build/
   dist/
   *.egg-info/
   *.so
   .pytest_cache/
   .codex-runs/
   .codex-stop
   benchmark-results/
   baseline-tests.txt
   .DS_Store
   ```

3. Confirm the new `.gitignore` has exactly 12 lines:

   ```bash
   wc -l < .gitignore
   ```

   Expected output: `12`.

4. Remove the backup (so it does not enter the commit):

   ```bash
   rm .gitignore.step02.bak
   ```

5. Stage everything and confirm that ignored artifacts are excluded:

   ```bash
   git add -A
   git status --porcelain | grep -E '\.so$|^.. build/|\.pytest_cache|\.codex-runs|\.codex-stop' ; echo "grep-exit=$?"
   ```

   Expected output: no matching lines, then `grep-exit=1` (grep finding nothing is the success
   condition here).

6. Confirm the 403 benchmark JSON artifacts ARE staged (deliberate — see Goal):

   ```bash
   git status --porcelain | grep -c 'compat/benchmark_'
   ```

   Expected output: `403`.

7. Create the snapshot commit:

   ```bash
   git commit -m "snapshot: full source tree before starting refactoring"
   ```

**Verification**:

```bash
git log --oneline -1
git status --porcelain | wc -l
git ls-files -- mtorch cpp tests | wc -l
```

Expected output: the first command shows the snapshot commit
(`snapshot: full source tree before starting refactoring`); then `0` (clean working tree);
then a number well above 20 (the source tree is now tracked).

**On failure**:

- If the precondition check printed anything other than `snapshot=0 tracked-src=0`: do NOT
  delete `.git` or rewrite history. If `snapshot` is `1` or more, step 0-2 was already
  completed — mark it `[x]` in `docs/design/PROGRESS.md` and move on to step 0-3. If
  `tracked-src` is nonzero while `snapshot=0`, part of the source tree was tracked outside this
  procedure — mark step 0-2 **BLOCKED** in `docs/design/PROGRESS.md` and note "unexpected git
  state" plus the output of `git log --oneline` in the Phase 0 Notes. Stop.
- If Action 3 does not print `12`: restore with `cp .gitignore.step02.bak .gitignore` and redo
  Actions 2–3.
- If `git commit` fails with "Please tell me who you are" (should not happen — the identity is
  already configured): run these two commands, then re-run Action 7:

  ```bash
  git config user.email "kazutakehiramatsu@gmail.com"
  git config user.name "kazhiramatsu"
  ```

- If Action 5's grep prints matching lines (exit 0): `.gitignore` is wrong. Run `git rm -r --cached . -q`,
  redo Actions 2–4, then resume from Action 5. If it still fails, mark step 0-2 **BLOCKED** in
  `docs/design/PROGRESS.md` and stop.

**Commit**: The snapshot commit itself is Action 7. Afterwards, mark the step done in
`docs/design/PROGRESS.md` (change the 0-2 line to `[x]` with the hash from
`git rev-parse --short HEAD` and date from `date +%Y-%m-%d`, per the Conventions section), then:

```bash
git add docs/design/PROGRESS.md
git commit -m "chore(progress): mark step 0-2 done"
```

---

## Step 0-3: Creating the benchmark comparison script

**Goal**: Add `tools/compare_benchmarks.py`, which compares two benchmark JSON files and fails if
the candidate/PyTorch median ratio of any case degraded by more than 5% (the performance gate
defined in `00-overview.md` §"Performance preservation": median-ratio degradation relative to the
baseline must be within 5%).

Background on the JSON format (already verified — no need to re-derive it): benchmark JSON is
written by `write_benchmark_json` in `tests/compat/benchmarking.py`
(locate with `grep -n "def write_benchmark_json" tests/compat/benchmarking.py`, currently line 102).
The payload is `{"benchmarks": [ {...}, ... ]}` where each entry is a serialized `BenchmarkResult`
dataclass with keys `id`, `ratio`, `reference_median_ms`, `candidate_median_ms`, `status`,
`slow_ratio`, `slow_ratio_exceeded`, `max_ratio`, `max_ratio_exceeded`, `reference_min_ms`,
`candidate_min_ms`, `repeat`
(locate with `grep -n "class BenchmarkResult" tests/compat/benchmarking.py`, currently line 31).
The comparison script only uses `id` and `ratio`.

**Preconditions**: Step 0-2 is done; the `tools/` directory exists. Check:

```bash
test -d .git && test -d tools && test -f tools/generate_api_manifest.py && echo PRECONDITION-OK
```

Expected output: `PRECONDITION-OK`.

**Actions**:

Action 1: Create the file `tools/compare_benchmarks.py` with EXACTLY the following content (the
complete file, shown unindented — copy it verbatim, no modifications):

```python
#!/usr/bin/env python3
"""Compare two benchmark result JSON files and report performance regressions.

The JSON files are produced by running the benchmark suite with
--compat-benchmark-json, e.g.:

    pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
        --compat-benchmark-json benchmark-results/current.json

and their schema is {"benchmarks": [{"id": ..., "ratio": ..., ...}, ...]}
(see write_benchmark_json in tests/compat/benchmarking.py).

Usage:
    python3 tools/compare_benchmarks.py <baseline.json> <current.json> [--threshold 1.05]

Example:
    python3 tools/compare_benchmarks.py docs/design/baseline/benchmark-baseline.json \
        benchmark-results/current.json

"ratio" is the candidate/PyTorch median-time ratio for one benchmark case. A case is a
REGRESSION when its current ratio exceeds its baseline ratio by more than the threshold
factor (default 1.05 = the 5% gate from 00-overview.md). Because the ratio is relative
within the same machine and the same run, it is less affected by machine load than raw
times, but there is still noise. If a REGRESSION appears, rerun 3 times to confirm
reproducibility (01-rules-and-verification.md).

Exit code: 1 if there is at least one regression or a case missing from the current
results, otherwise 0. Newly added cases are reported but do not fail the comparison.
"""
from __future__ import annotations

import argparse
import json
import sys


def load(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return {b["id"]: b for b in payload["benchmarks"]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two benchmark result JSON files (median-ratio 5% gate)."
    )
    parser.add_argument("baseline", help="baseline benchmark JSON (the reference)")
    parser.add_argument("current", help="current benchmark JSON (the one being checked)")
    parser.add_argument("--threshold", type=float, default=1.05,
                        help="regression threshold factor (1.05 = allow up to 5%% degradation)")
    args = parser.parse_args()

    base = load(args.baseline)
    cur = load(args.current)

    missing = sorted(set(base) - set(cur))
    added = sorted(set(cur) - set(base))
    regressions = []
    for case_id in sorted(set(base) & set(cur)):
        old = base[case_id]["ratio"]
        new = cur[case_id]["ratio"]
        if old > 0 and new / old > args.threshold:
            regressions.append((case_id, old, new, new / old))

    for case_id, old, new, factor in regressions:
        print(f"REGRESSION {case_id}: ratio {old:.3f} -> {new:.3f} (x{factor:.2f})")
    for case_id in missing:
        print(f"MISSING {case_id}: exists in baseline but not in current results")
    for case_id in added:
        print(f"NEW {case_id}")
    print(
        f"compared={len(set(base) & set(cur))} regressions={len(regressions)} "
        f"missing={len(missing)} new={len(added)}"
    )
    if regressions or missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

Action 2: Confirm the file parses and prints its usage:

```bash
python3 tools/compare_benchmarks.py --help
```

Expected: exit code 0 and the usage text's first line reading exactly
`usage: compare_benchmarks.py [-h] [--threshold THRESHOLD] baseline current`.

**Verification** (functional self-test with fixture files; copy-paste each block exactly):

1. Create two fixture files (three commands; each is a single line):

   ```bash
   mkdir -p /tmp/compare_bench_test
   printf '%s\n' '{"benchmarks": [{"id": "bench.a", "ratio": 1.0}, {"id": "bench.b", "ratio": 2.0}]}' > /tmp/compare_bench_test/base.json
   printf '%s\n' '{"benchmarks": [{"id": "bench.a", "ratio": 1.2}, {"id": "bench.b", "ratio": 2.0}]}' > /tmp/compare_bench_test/cur.json
   ```

2. Identical files must pass with exit code 0:

   ```bash
   python3 tools/compare_benchmarks.py /tmp/compare_bench_test/base.json /tmp/compare_bench_test/base.json; echo "exit=$?"
   ```

   Expected output (exact):

   ```
   compared=2 regressions=0 missing=0 new=0
   exit=0
   ```

3. A 20% regression must fail with exit code 1:

   ```bash
   python3 tools/compare_benchmarks.py /tmp/compare_bench_test/base.json /tmp/compare_bench_test/cur.json; echo "exit=$?"
   ```

   Expected output (exact):

   ```
   REGRESSION bench.a: ratio 1.000 -> 1.200 (x1.20)
   compared=2 regressions=1 missing=0 new=0
   exit=1
   ```

4. Clean up the fixtures:

   ```bash
   rm -rf /tmp/compare_bench_test
   ```

**On failure**: If any verification output differs, delete the file (`rm tools/compare_benchmarks.py`),
re-create it from the listing above (the most likely cause is an incomplete or altered paste), and
re-run the verification. If it fails a second time, run `git status --porcelain` — if it shows
anything other than `?? tools/compare_benchmarks.py`, restore with `git checkout -- .`. Then mark
step 0-3 **BLOCKED** in `docs/design/PROGRESS.md` with the mismatching output in the Notes. Stop.

**Commit**: Commit the script first (the PROGRESS.md entry needs this commit's hash), then
update and commit PROGRESS.md:

```bash
git add tools/compare_benchmarks.py
git commit -m "refactor(phase0-3): add benchmark comparison script"
```

Then mark step 0-3 done in `docs/design/PROGRESS.md` (hash from `git rev-parse --short HEAD`,
date from `date +%Y-%m-%d`) and:

```bash
git add docs/design/PROGRESS.md
git commit -m "chore(progress): mark step 0-3 done"
```

---

## Step 0-4: Recording the baseline

**Goal**: Record the current test results, test collection count, and full benchmark results into
`docs/design/baseline/`. These three files are the pass/fail reference for every later phase.

**Preconditions**: Step 0-3 is done. Check:

```bash
test -f tools/compare_benchmarks.py && test ! -e docs/design/baseline && echo PRECONDITION-OK
```

Expected output: `PRECONDITION-OK`. (If `docs/design/baseline` already exists from an aborted
earlier attempt, delete it first: `rm -rf docs/design/baseline`, then re-check.)

**Actions**:

1. Create the output directories (`benchmark-results/` is gitignored and used by later phases;
   `docs/design/baseline/` is committed):

   ```bash
   mkdir -p docs/design/baseline benchmark-results
   ```

2. **Test results** (this takes a long time — possibly tens of minutes, because the full compat
   suite including benchmarks runs; do NOT abort, and use a command timeout of at least 60
   minutes if your tooling enforces one):

   ```bash
   pytest tests/compat 2>&1 | tee docs/design/baseline/tests-baseline.txt
   ```

   Then extract and inspect the summary line:

   ```bash
   grep -E "^=+ .*(passed|failed|xfailed|error).* =+$" docs/design/baseline/tests-baseline.txt | tail -1
   ```

   Expected: one line like `======== N passed, M xfailed in XXXs ========` (the exact counts are
   whatever they are — record them, do not judge them).
   **Note**: failures at this point are acceptable; this IS the current state. The pass criterion
   for all subsequent phases is "do not degrade from this breakdown".

3. **Test collection count**:

   ```bash
   pytest tests/compat --collect-only -q 2>&1 | tail -2 > docs/design/baseline/collect-count.txt
   cat docs/design/baseline/collect-count.txt
   ```

   Expected output: the last line reads `2272 tests collected in X.XXs` (2272 is the count as of
   writing; if it differs, that is still the baseline — record it as-is, do not judge it).

4. **Full benchmark** (this takes a long time; do NOT abort). The pytest options used here are
   defined in `tests/conftest.py`
   (locate with `grep -n "compat-benchmark-json" tests/conftest.py`, currently line 53):

   ```bash
   pytest tests/compat/test_benchmarks.py --compat-benchmark-repeat=10 \
       --compat-benchmark-json docs/design/baseline/benchmark-baseline.json
   ```

   Expected: near the end of the output, a line `benchmark JSON: docs/design/baseline/benchmark-baseline.json`.

**Verification**:

1. All three files exist:

   ```bash
   ls docs/design/baseline
   ```

   Expected output (exactly these three names, order may differ):

   ```
   benchmark-baseline.json
   collect-count.txt
   tests-baseline.txt
   ```

2. The benchmark JSON contains one entry per benchmark case:

   ```bash
   python3 -c "import json; print(len(json.load(open('docs/design/baseline/benchmark-baseline.json'))['benchmarks']))"
   ```

   Expected output: `415`. If it is not 415, cross-check against the case table with
   `grep -c "BenchmarkCase(" tests/compat/benchmarking.py` (the `BENCHMARK_CASES` tuple starts at
   the line found by `grep -n "BENCHMARK_CASES: tuple" tests/compat/benchmarking.py`, currently
   line 8191). If the two numbers match each other, the baseline is consistent and you may
   proceed; if they differ, treat it as a failure.

3. The comparison script accepts the baseline as its own comparison target (sanity round-trip):

   ```bash
   python3 tools/compare_benchmarks.py docs/design/baseline/benchmark-baseline.json docs/design/baseline/benchmark-baseline.json; echo "exit=$?"
   ```

   Expected output: last two lines are `compared=415 regressions=0 missing=0 new=0` and `exit=0`
   (415 or the consistent count established in check 2).

**On failure**: The only files this step creates are under `docs/design/baseline/` and the empty
`benchmark-results/`. Recovery: `rm -rf docs/design/baseline` and redo the step from Action 1.
If a pytest invocation itself crashes (not test failures — an abort/traceback of pytest), re-run
it once; if it crashes again, mark step 0-4 **BLOCKED** in `docs/design/PROGRESS.md` with the
crash's last output line in the Notes, and stop.

**Commit**:

```bash
git add docs/design/baseline
git commit -m "refactor(phase0-4): record test and benchmark baseline"
```

Then mark step 0-4 done in `docs/design/PROGRESS.md` (hash from `git rev-parse --short HEAD`,
date from `date +%Y-%m-%d`) and:

```bash
git add docs/design/PROGRESS.md
git commit -m "chore(progress): mark step 0-4 done"
```

---

## Step 0-5: Deleting the artifact JSON files

**Goal**: Delete the 403 `compat/benchmark_*.json` result snapshots (work history of the earlier
codex loop, referenced by no code) so that `compat/` contains only input data. The file
`compat/api_surface_seed.json` MUST survive — it is test input data (it is the default value of
`--compat-api-manifest`; verify with `grep -n "api_surface_seed" tests/conftest.py`, currently
line 9).

**Preconditions**: Step 0-4 is committed and `compat/` is in its original state. Check:

```bash
test -f docs/design/baseline/benchmark-baseline.json && test -f compat/api_surface_seed.json && echo PRECONDITION-OK
ls compat | wc -l
```

Expected output: `PRECONDITION-OK`, then `404` (403 benchmark JSONs + `api_surface_seed.json`).

**Actions**:

1. Pre-check: prove that no code references the files about to be deleted (grep over all Python
   code and build config):

   ```bash
   grep -rn "compat/benchmark_" --include="*.py" tests tools mtorch setup.py pyproject.toml ; echo "grep-exit=$?"
   ```

   Expected output: no matching lines, then `grep-exit=1` (no matches is the success condition).
   If any line is printed, STOP — do not delete anything; go to **On failure**.

2. Delete the artifacts:

   ```bash
   find compat -name 'benchmark_*.json' -delete
   ```

3. Confirm exactly one file remains:

   ```bash
   ls compat
   ```

   Expected output (exactly): `api_surface_seed.json`

**Verification**:

1. File count and survivor:

   ```bash
   ls compat | wc -l
   ls compat/api_surface_seed.json
   ```

   Expected output: `1`, then `compat/api_surface_seed.json`.

2. The API surface tests still collect and run exactly as before (they read
   `api_surface_seed.json`):

   ```bash
   pytest tests/compat/test_api_surface.py 2>&1 | tail -1
   ```

   Expected: a summary line with the same passed/failed breakdown as the `test_api_surface`
   portion of the baseline (in particular, NOT a collection error). Cross-check the baseline with:

   ```bash
   grep -c "test_api_surface" docs/design/baseline/tests-baseline.txt
   ```

   (any output is fine; the point of comparison is that the run above did not error out).

3. Git sees exactly 403 deletions and nothing else:

   ```bash
   git status --porcelain | grep -c '^ D compat/benchmark_'
   git status --porcelain | grep -v '^ D compat/benchmark_' | wc -l
   ```

   Expected output: `403`, then `0`.

**On failure**:

- If Action 1 printed a reference: do NOT delete anything. Mark step 0-5 **BLOCKED** in
  `docs/design/PROGRESS.md` and paste the referencing line into the Phase 0 Notes. Stop.
- If the wrong files were deleted (e.g. `ls compat` no longer shows `api_surface_seed.json`, or
  verification counts are wrong): restore everything with

  ```bash
  git checkout -- compat
  ```

  then redo the step from Action 1. If it fails again, mark step 0-5 **BLOCKED** in
  `docs/design/PROGRESS.md` and stop.

**Commit**:

```bash
git add -A
git commit -m "refactor(phase0-5): remove 403 unreferenced benchmark result JSON files"
```

Then mark step 0-5 done in `docs/design/PROGRESS.md` (hash from `git rev-parse --short HEAD`,
date from `date +%Y-%m-%d`) and:

```bash
git add docs/design/PROGRESS.md
git commit -m "chore(progress): mark step 0-5 done"
```

---

## Step 0-6: Phase completion check

**Goal**: Confirm the whole phase left the test suite identical to the baseline and the git
history complete, then close out Phase 0 in `docs/design/PROGRESS.md`.

**Preconditions**: Steps 0-1 through 0-5 are marked `[x]` in `docs/design/PROGRESS.md`. Check:

```bash
grep -c '^- \[x\] 0-[1-5]' docs/design/PROGRESS.md
```

Expected output: `5`.

**Actions**:

1. Re-run the full suite and capture the output to a throwaway file (long-running; do not abort):

   ```bash
   pytest tests/compat 2>&1 | tee /tmp/phase0-final-tests.txt | tail -5
   ```

2. Compare the summary line against the baseline:

   ```bash
   grep -E "^=+ .*(passed|failed|xfailed|error).* =+$" docs/design/baseline/tests-baseline.txt | tail -1
   grep -E "^=+ .*(passed|failed|xfailed|error).* =+$" /tmp/phase0-final-tests.txt | tail -1
   ```

   Expected: the passed/failed/xfailed counts in the two lines are identical (the elapsed time
   will differ — ignore it).

3. Confirm the test collection count is unchanged:

   ```bash
   pytest tests/compat --collect-only -q 2>&1 | tail -2
   cat docs/design/baseline/collect-count.txt
   ```

   Expected: the `N tests collected` count is identical in both outputs.

4. Confirm the commit history contains the phase's work:

   ```bash
   git log --oneline
   ```

   Expected: commits for steps 0-2 through 0-5 are all present, i.e. the list includes (newest
   first, with the `chore(progress)` commits interleaved):
   `refactor(phase0-5): remove 403 unreferenced benchmark result JSON files`,
   `refactor(phase0-4): record test and benchmark baseline`,
   `refactor(phase0-3): add benchmark comparison script`,
   `snapshot: full source tree before starting refactoring`.

5. Clean up the throwaway file:

   ```bash
   rm /tmp/phase0-final-tests.txt
   ```

**Verification**:

```bash
git status --porcelain | wc -l
```

Expected output: `0` (clean tree; the PROGRESS.md edit for 0-6 itself happens in **Commit** below).

**On failure**: If the test breakdown or collection count differs from the baseline, something in
steps 0-2..0-5 changed behavior — this phase must not change behavior at all. Run
`git status --porcelain` (expect clean) and re-run Action 1 once (flaky timing is possible for
benchmark-marked tests). If the difference persists, mark step 0-6 **BLOCKED** in
`docs/design/PROGRESS.md`, paste both summary lines into the Phase 0 Notes, and stop. Do NOT try
to fix individual tests.

**Commit**: Mark step 0-6 done in `docs/design/PROGRESS.md` (change the 0-6 line to `[x]`; for
its `commit:` field write the hash of the upcoming commit AFTER committing — i.e. first set
`commit: (this commit)` is NOT allowed; instead do it in two passes as follows). Concretely:

1. Edit the 0-6 line to `- [x] 0-6 Phase completion check — commit: pending / date: <YYYY-MM-DD>`
   (date from `date +%Y-%m-%d`), then:

   ```bash
   git add docs/design/PROGRESS.md
   git commit -m "refactor(phase0-6): phase 0 completion check"
   ```

2. Replace `pending` in that line with the hash from `git rev-parse --short HEAD`, then:

   ```bash
   git add docs/design/PROGRESS.md
   git commit -m "chore(progress): mark step 0-6 done"
   ```

Phase 0 is now complete. Proceed to `03-phase1-python-package.md`.
