#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import importlib
import inspect
import json
from pathlib import Path
from typing import Any
import warnings


DEFAULT_SKIP_PREFIXES = (
    "_",
)

DEFAULT_SKIP_MODULES = (
    "torch.ops",
    "torch.classes",
    "torch._C",
    "torch._dynamo",
    "torch._inductor",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a public API manifest from a reference PyTorch module.")
    parser.add_argument("--module", default="torch", help="Reference module import path.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum attribute recursion depth.")
    parser.add_argument("--include-signatures", action="store_true", help="Record inspectable signatures.")
    args = parser.parse_args()

    module = importlib.import_module(args.module)
    entries = collect_entries(module, args.module, args.max_depth, args.include_signatures)
    payload = {
        "schema": 1,
        "reference": {
            "module": args.module,
            "version": getattr(module, "__version__", None),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "max_depth": args.max_depth,
        },
        "entries": entries,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_entries(root: Any, root_name: str, max_depth: int, include_signatures: bool) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_objects: set[int] = set()
    queue: deque[tuple[str, Any, int]] = deque([("", root, 0)])

    while queue:
        prefix, owner, depth = queue.popleft()
        if id(owner) in seen_objects:
            continue
        seen_objects.add(id(owner))

        for name in sorted(dir(owner)):
            if should_skip_name(name):
                continue

            path = f"{prefix}.{name}" if prefix else name
            try:
                value = getattr(owner, name)
            except Exception:
                continue

            qualified = f"{root_name}.{path}"
            if should_skip_module(qualified):
                continue

            entry = {"path": path, "kind": api_kind(value)}
            if include_signatures and callable(value):
                signature = maybe_signature(value)
                if signature is not None:
                    entry["signature"] = signature
            entries.append(entry)

            if depth < max_depth and inspect.ismodule(value) and getattr(value, "__name__", "").startswith(root_name):
                queue.append((path, value, depth + 1))

    return entries


def should_skip_name(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in DEFAULT_SKIP_PREFIXES)


def should_skip_module(qualified_path: str) -> bool:
    return any(qualified_path == module or qualified_path.startswith(module + ".") for module in DEFAULT_SKIP_MODULES)


def api_kind(value: Any) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if inspect.ismodule(value):
            return "module"
        if inspect.isclass(value):
            return "class"
    if callable(value):
        return "callable"
    return "value"


def maybe_signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
