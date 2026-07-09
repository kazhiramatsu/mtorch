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
