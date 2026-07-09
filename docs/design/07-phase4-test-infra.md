# 07. Phase 4: Organizing the Test/Benchmark Infrastructure

Prerequisites: Phase 0 complete. Can proceed independently of (in parallel with) Phases 1–3.
`01-rules-and-verification.md` and `docs/test-harness-design.md` read.

Goals (unchanged):

1. Separate `tests/compat/benchmarking.py` (10,931 lines) into an "execution engine" and "scenario definitions"
2. Resolve the dependency inversion of private imports (120 names) from `tests/compat/test_tensor_ops.py`
3. Compress the name-differing copies of SD variants (139 `_case_stable_diffusion*_setup` functions) via parameterization
4. Incrementally migrate the hand-written individual tests (195 functions in `test_tensor_ops.py` + 89 in `test_nn_modules.py`) to the existing table-driven mechanism

**Because this phase touches the test infrastructure, a special rule applies**:
in every step, mechanically confirm that the "test collection count" and the "benchmark case ID list"
match before and after the step. Changes that alter the meaning of the comparison (what is compared and how) are prohibited.

## Fixed facts about this phase (do not re-derive them)

- All commands below are run from the repo root: `cd /Users/hiramatsu/dev/mtorch` first, always.
- Tests are run with `pytest tests/compat`. The pytest configuration lives in `pyproject.toml`
  (`[tool.pytest.ini_options]`: `pythonpath = ["."]`, `addopts = "-ra"`, markers include
  `compat`, `benchmark`, `slow`). Do not edit `pyproject.toml` in this phase.
- **No C++ build is needed anywhere in this phase.** Only Python files under `tests/`, `tools/`,
  `compat/` and docs change. Never run `python3 setup.py build_ext` in this phase.
- Key file sizes at the start of this phase (for orientation only):
  `benchmarking.py` 10,931 lines, `test_tensor_ops.py` 6,016 lines, `test_nn_modules.py` 2,423 lines,
  `cases.py` 4,107 lines, `conftest.py` 120 lines, `harness.py` 447 lines.
- `benchmarking.py` contains, top to bottom: 3 dataclasses + engine functions
  (`run_benchmark_case`, `write_benchmark_json`, `_measure`, `_touch`, ends near L132),
  data helpers (`_matrix` … `_tensor`, grep `def _matrix` → currently L134), 390 scenario/helper
  functions, and `BENCHMARK_CASES: tuple[BenchmarkCase, ...] = (` (grep `^BENCHMARK_CASES` → currently
  L8191) with 415 entries.
- `tests/compat/test_tensor_ops.py` starts with `from .benchmarking import (` at L8, closing `)` at
  L129; the block imports exactly 120 names (65 `_case_stable_diffusion*_setup`, 55
  `_stable_diffusion*_path`). Grep anchor: `grep -n 'from .benchmarking import' tests/compat/test_tensor_ops.py`.
- Every line reference in this document is a hint only. Always relocate with the given grep first;
  if the grep does not match exactly once where stated, STOP and follow "On failure".

## Recovery rule (referenced by every step as "On failure")

If any Action or Verification fails:

```bash
cd /Users/hiramatsu/dev/mtorch
git restore --staged --worktree .
git clean -fd tests tools
git status --porcelain          # must print nothing
```

Then open `docs/design/PROGRESS.md`, append ` **BLOCKED**` to the line of the current step
(Phase 4 section, lines "4-0 … 4-5"), write the failing command and its first error line into the
phase Notes field (3 lines max), and commit only that file:

```bash
git add docs/design/PROGRESS.md
git commit -m "docs: mark phase 4 step BLOCKED"
```

Do not attempt a different approach on your own.

## Standard check (referenced by every step; run from repo root)

```bash
cd /Users/hiramatsu/dev/mtorch
python3 tools/phase4_scenarios.py verify docs/design/baseline/bench-ids-phase4.txt
pytest tests/compat --collect-only -q | tail -2 | sed -E 's/ in [0-9.]+s//' > /tmp/phase4-collect-after.txt
diff docs/design/baseline/collect-count-phase4.txt /tmp/phase4-collect-after.txt && echo "COLLECT: MATCH"
pytest tests/compat -q 2>&1 | tail -1 | sed -E 's/ in [0-9.]+s.*$//' > /tmp/phase4-tests-after.txt
diff docs/design/baseline/tests-phase4.txt /tmp/phase4-tests-after.txt && echo "TESTS: MATCH"
```

All three of `verify OK`, `COLLECT: MATCH`, `TESTS: MATCH` must appear. (In step 4-0 the tool does
not exist yet; 4-0 defines its own verification.) The `sed` calls strip run times, which differ
between runs; the pass/fail/skip counts must be byte-identical.

---

## Step 4-0: Pre-capture the benchmark ID list and collection baseline

**Goal**: Record the exact benchmark case ID list, the test collection count, and the test result
summary as committed files. Every later step is verified against these files.

**Preconditions** (all must pass; otherwise On failure):

```bash
cd /Users/hiramatsu/dev/mtorch
git rev-parse --is-inside-work-tree        # prints: true
git status --porcelain                     # prints nothing (clean tree)
test -f docs/design/baseline/tests-baseline.txt && echo "phase0 OK"
python3 -c "import torch, mtorch; print('imports OK')"
```

**Actions**:

1. Capture the benchmark ID list (IDs are enumerable because `BENCHMARK_CASES` is a tuple of
   `BenchmarkCase` dataclass instances, each with an `id: str` field — see `class BenchmarkCase`,
   grep `class BenchmarkCase` in `tests/compat/benchmarking.py`, currently L14):

```bash
cd /Users/hiramatsu/dev/mtorch
python3 - <<'EOF'
from tests.compat.benchmarking import BENCHMARK_CASES
ids = sorted(case.id for case in BENCHMARK_CASES)
assert len(ids) == len(set(ids)), "duplicate benchmark ids"
with open("docs/design/baseline/bench-ids-phase4.txt", "w") as fh:
    fh.write("\n".join(ids) + "\n")
print(len(ids))
EOF
```

   Expected printed count: `415`.

2. Capture the collection count (time-normalized):

```bash
pytest tests/compat --collect-only -q | tail -2 | sed -E 's/ in [0-9.]+s//' > docs/design/baseline/collect-count-phase4.txt
cat docs/design/baseline/collect-count-phase4.txt
```

3. Capture the current test summary line (this runs the full suite; takes a while; do not abort):

```bash
pytest tests/compat -q 2>&1 | tail -1 | sed -E 's/ in [0-9.]+s.*$//' > docs/design/baseline/tests-phase4.txt
cat docs/design/baseline/tests-phase4.txt
```

**Verification**:

```bash
wc -l docs/design/baseline/bench-ids-phase4.txt     # 415
test -s docs/design/baseline/collect-count-phase4.txt && echo "collect file OK"
test -s docs/design/baseline/tests-phase4.txt && echo "tests file OK"
```

**On failure**: see Recovery rule (the three new files are untracked; `git clean -fd docs/design/baseline` removes them — do NOT let `git clean` touch other baseline files: run `git clean -f docs/design/baseline/bench-ids-phase4.txt docs/design/baseline/collect-count-phase4.txt docs/design/baseline/tests-phase4.txt`).

**Commit**:

```bash
git add docs/design/baseline/bench-ids-phase4.txt docs/design/baseline/collect-count-phase4.txt docs/design/baseline/tests-phase4.txt
git commit -m "refactor(phase4-0): capture benchmark id list and collection baseline"
```

Mark 4-0 done in `docs/design/PROGRESS.md` (fill commit hash via `git rev-parse --short HEAD`), include that edit in the next commit or commit it separately with `git add docs/design/PROGRESS.md && git commit -m "docs: progress 4-0"`. Do the same after every step below; it is not repeated again.

---

## Step 4-1: Introducing the `scenarios/` package and resolving the dependency inversion

Target layout (design decision, unchanged from the original plan):

```
tests/compat/scenarios/
  __init__.py       # public window (shim); tests import scenario functions ONLY from here
  _data.py          # synthetic data helpers: _matrix, _unique_matrix, _near_one_matrix,
                    #   _probability_matrix, _tensor3, _tensor4, _tensor5, _vector, _mask,
                    #   _bool_matrix, _tensor, _copy_parameter          (12 functions)
  matmul.py         # matmul/einsum/linear family                       (17 functions)
  elementwise.py    # unary/binary/reduction/loss family                (53 functions)
  indexing.py       # index/mask/gather/shape family                    (18 functions)
  conv.py           # conv/pool/pad/upsample family                     (14 functions)
  sd/
    __init__.py
    blocks.py       # SD attention / transformer / builder helpers      (186 functions)
    pipelines.py    # SD denoising loops / schedulers / pipelines       (90 functions)
```

Functions are moved **verbatim, never renamed** (keep the leading underscore; renaming would mean
rewriting all references, which this phase does not do). All moves are performed by a helper script
(`tools/phase4_scenarios.py`, created in 4-1-a) whose keyword rules ARE the authoritative domain
assignment — do not hand-pick functions. The required move order
`_data → matmul → elementwise → conv → indexing → sd_blocks → sd_pipelines` is dependency-safe
(verified: no function references a name from a later domain); the script aborts with `BLOCKED: ...`
if that invariant is ever violated.

The shim uses a lazy module `__getattr__` (PEP 562) instead of eager
`from tests.compat.benchmarking import ...` re-exports. This is still exactly "a shim re-exporting
from benchmarking"; the lazy form is required because from the first move onward `benchmarking.py`
imports scenario submodules, and an eager import in `scenarios/__init__.py` back into the
partially-initialized `benchmarking` module would be a circular import.

### Step 4-1-a: Create the scenarios/ shim and rewire the test_tensor_ops imports

**Goal**: `test_tensor_ops.py` imports its 120 scenario names from `tests.compat.scenarios` instead
of `tests.compat.benchmarking`. No function moves yet.

**Preconditions**:

```bash
cd /Users/hiramatsu/dev/mtorch
git status --porcelain                          # nothing
test -f docs/design/baseline/bench-ids-phase4.txt && echo "4-0 OK"
test ! -e tests/compat/scenarios && echo "no scenarios dir yet: OK"
grep -c 'from .benchmarking import (' tests/compat/test_tensor_ops.py   # 1
```

**Actions**:

1. Create `tools/phase4_scenarios.py` with **exactly** this content (the `SHIM_NAMES` tuple is the
   complete, verified list of the 120 names imported by `test_tensor_ops.py` L8–129):

```python
#!/usr/bin/env python3
"""Phase 4 helper for splitting tests/compat/benchmarking.py (see docs/design/07-phase4-test-infra.md).

Subcommands:
    list                 print the domain assignment of every top-level scenario function
    shim                 (re)generate tests/compat/scenarios/__init__.py
    move <domain>        move one domain's functions out of benchmarking.py
    cases                split BENCHMARK_CASES into per-domain CASES lists (step 4-1-c)
    verify <ids-file>    static check: scenario globals resolve + benchmark id list matches

Domains, in required move order:
    _data matmul elementwise conv indexing sd_blocks sd_pipelines
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "tests" / "compat" / "benchmarking.py"
SCEN = ROOT / "tests" / "compat" / "scenarios"

ORDER = ["_data", "matmul", "elementwise", "conv", "indexing", "sd_blocks", "sd_pipelines"]
RANK = {d: i for i, d in enumerate(ORDER)}
MODULES = {
    "_data": ("tests.compat.scenarios._data", SCEN / "_data.py"),
    "matmul": ("tests.compat.scenarios.matmul", SCEN / "matmul.py"),
    "elementwise": ("tests.compat.scenarios.elementwise", SCEN / "elementwise.py"),
    "conv": ("tests.compat.scenarios.conv", SCEN / "conv.py"),
    "indexing": ("tests.compat.scenarios.indexing", SCEN / "indexing.py"),
    "sd_blocks": ("tests.compat.scenarios.sd.blocks", SCEN / "sd" / "blocks.py"),
    "sd_pipelines": ("tests.compat.scenarios.sd.pipelines", SCEN / "sd" / "pipelines.py"),
}

ENGINE = {"run_benchmark_case", "write_benchmark_json", "_measure", "_touch"}
DATA = {
    "_matrix", "_unique_matrix", "_near_one_matrix", "_probability_matrix",
    "_tensor3", "_tensor4", "_tensor5", "_vector", "_mask", "_bool_matrix",
    "_tensor", "_copy_parameter",
}
EXPLICIT = {
    "_fill_stable_diffusion_parameter": "sd_blocks",
    "_case_stable_diffusion_mask_preprocess_setup": "sd_blocks",
    "_assign_nonleading_int_columns": "indexing",
    "_sgd_training_step": "elementwise",
    "_iadd_scalar": "elementwise",
}
PIPE = re.compile(r"denoising_loop|denoising_step|module_pipeline|img2img|scheduler|preprocess|add_noise|guidance|vae_posterior")
CONV = re.compile(r"conv|pool|pad|unfold|fold|interpolate|upsample|grid_sample|pixel_shuffle|col2im|im2col")
MATMUL = re.compile(r"matmul|einsum|bmm|addmm|baddbmm|dot|_mv_|_mm_|kron|tensordot|matrix|linear|inner|outer|chain")
INDEX = re.compile(
    r"index|gather|scatter|select|slice|take|put_|embedding|bincount|nonzero|mask|one_hot|where|topk|sort"
    r"|unique|searchsorted|repeat_interleave|roll|flip|diag|tril|triu|narrow|split|chunk|cat|stack|permute"
    r"|transpose|view|reshape|squeeze|expand|tile|unbind"
)

# Names imported by tests/compat/test_tensor_ops.py (its original L8-129 import block).
SHIM_NAMES = (
    "_case_stable_diffusion_half_channels_last_controlnet_merge_setup",
    "_case_stable_diffusion_half_channels_last_dpmpp_2m_scheduler_step_setup",
    "_case_stable_diffusion_half_channels_last_dpmpp_sde_scheduler_step_setup",
    "_case_stable_diffusion_half_channels_last_guidance_rescale_setup",
    "_case_stable_diffusion_half_channels_last_inpaint_module_pipeline_setup",
    "_case_stable_diffusion_half_channels_last_lora_conv_setup",
    "_case_stable_diffusion_half_channels_last_module_pipeline_setup",
    "_case_stable_diffusion_half_channels_last_resnet_scale_shift_setup",
    "_case_stable_diffusion_half_channels_last_vae_posterior_sample_setup",
    "_case_stable_diffusion_half_clip_large_text_encoder_stack_setup",
    "_case_stable_diffusion_half_lora_attention_projection_setup",
    "_case_stable_diffusion_inpaint_preprocess_bundle_setup",
    "_case_stable_diffusion_sd3_flowmatch_cfg_step_setup",
    "_case_stable_diffusion_sd3_joint_transformer_block_setup",
    "_case_stable_diffusion_sd3_large_controlnet_transformer_stack_setup",
    "_case_stable_diffusion_sd3_large_unpatchify_projection_setup",
    "_case_stable_diffusion_sd3_mini_transformer_stack_setup",
    "_case_stable_diffusion_sd3_multi_controlnet_transformer_stack_setup",
    "_case_stable_diffusion_sd3_patch_embed_setup",
    "_case_stable_diffusion_sd3_pooled_controlnet_denoising_step_setup",
    "_case_stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_setup",
    "_case_stable_diffusion_sd3_pooled_transformer_stack_setup",
    "_case_stable_diffusion_sd3_qk_norm_joint_attention_setup",
    "_case_stable_diffusion_sd3_rectangular_patch_embed_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_controlnet_long_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_disabled_windowed_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_disabled_windowed_img2img_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_zero_width_windowed_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_zero_width_windowed_img2img_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_disabled_windowed_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_disabled_windowed_img2img_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_zero_width_windowed_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_zero_width_windowed_img2img_denoising_loop_setup",
    "_case_stable_diffusion_sd3_rectangular_rotary_joint_attention_setup",
    "_case_stable_diffusion_sd3_rectangular_transformer_stack_setup",
    "_case_stable_diffusion_sd3_rectangular_unpatchify_projection_setup",
    "_case_stable_diffusion_sd3_rotary_joint_attention_setup",
    "_case_stable_diffusion_sd3_single_transformer_block_setup",
    "_case_stable_diffusion_sd3_time_text_conditioning_setup",
    "_case_stable_diffusion_sd3_unpatchify_projection_setup",
    "_case_stable_diffusion_sdxl_add_time_conditioning_setup",
    "_case_stable_diffusion_sdxl_dual_prompt_encode_setup",
    "_case_stable_diffusion_sdxl_half_channels_last_inpaint_controlnet_module_pipeline_setup",
    "_case_stable_diffusion_sdxl_ip_adapter_cross_attention_setup",
    "_case_stable_diffusion_sdxl_prompt_conditioning_bundle_setup",
    "_case_stable_diffusion_sdxl_text_encoder2_stack_setup",
    "_case_stable_diffusion_sdxl_unet_cross_attention_setup",
    "_stable_diffusion_half_channels_last_controlnet_merge_path",
    "_stable_diffusion_half_channels_last_dpmpp_2m_scheduler_step_path",
    "_stable_diffusion_half_channels_last_dpmpp_sde_scheduler_step_path",
    "_stable_diffusion_half_channels_last_guidance_rescale_path",
    "_stable_diffusion_half_channels_last_lora_conv_path",
    "_stable_diffusion_half_channels_last_resnet_scale_shift_path",
    "_stable_diffusion_half_channels_last_vae_posterior_sample_path",
    "_stable_diffusion_half_clip_large_text_encoder_stack_path",
    "_stable_diffusion_half_lora_attention_projection_path",
    "_stable_diffusion_inpaint_module_pipeline_path",
    "_stable_diffusion_inpaint_preprocess_bundle_path",
    "_stable_diffusion_module_pipeline_path",
    "_stable_diffusion_sd3_flowmatch_cfg_step_path",
    "_stable_diffusion_sd3_joint_transformer_block_path",
    "_stable_diffusion_sd3_large_controlnet_transformer_stack_path",
    "_stable_diffusion_sd3_large_unpatchify_projection_path",
    "_stable_diffusion_sd3_mini_transformer_stack_path",
    "_stable_diffusion_sd3_multi_controlnet_transformer_stack_path",
    "_stable_diffusion_sd3_patch_embed_path",
    "_stable_diffusion_sd3_pooled_controlnet_denoising_step_path",
    "_stable_diffusion_sd3_pooled_multi_controlnet_transformer_stack_path",
    "_stable_diffusion_sd3_pooled_transformer_stack_path",
    "_stable_diffusion_sd3_qk_norm_joint_attention_path",
    "_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path",
    "_stable_diffusion_sd3_rectangular_pooled_controlnet_img2img_denoising_loop_path",
    "_stable_diffusion_sd3_rectangular_pooled_controlnet_keep_decay_denoising_loop_path",
    "_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_denoising_loop_path",
    "_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_img2img_denoising_loop_path",
    "_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_keep_decay_denoising_loop_path",
    "_stable_diffusion_sd3_rectangular_pooled_inpaint_controlnet_transformer_stack_path",
    "_stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_path",
    "_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_denoising_loop_path",
    "_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_img2img_denoising_loop_path",
    "_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_transformer_stack_path",
    "_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_denoising_loop_path",
    "_stable_diffusion_sd3_rectangular_pooled_multi_image_inpaint_controlnet_windowed_img2img_denoising_loop_path",
    "_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_denoising_loop_path",
    "_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_img2img_denoising_loop_path",
    "_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_transformer_stack_path",
    "_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_denoising_loop_path",
    "_stable_diffusion_sd3_rectangular_pooled_triple_image_inpaint_controlnet_windowed_img2img_denoising_loop_path",
    "_stable_diffusion_sd3_rectangular_rotary_joint_attention_path",
    "_stable_diffusion_sd3_rectangular_transformer_stack_path",
    "_stable_diffusion_sd3_rectangular_unpatchify_projection_path",
    "_stable_diffusion_sd3_rotary_joint_attention_path",
    "_stable_diffusion_sd3_single_transformer_block_path",
    "_stable_diffusion_sd3_time_text_conditioning_path",
    "_stable_diffusion_sd3_unpatchify_projection_path",
    "_stable_diffusion_sdxl_add_time_conditioning_path",
    "_stable_diffusion_sdxl_dual_prompt_encode_path",
    "_stable_diffusion_sdxl_inpaint_controlnet_module_pipeline_path",
    "_stable_diffusion_sdxl_ip_adapter_cross_attention_path",
    "_stable_diffusion_sdxl_prompt_conditioning_bundle_path",
    "_stable_diffusion_sdxl_text_encoder2_stack_path",
    "_stable_diffusion_sdxl_unet_cross_attention_path",
)


def classify(name: str) -> str | None:
    if name in ENGINE:
        return None
    if name in DATA:
        return "_data"
    if name in EXPLICIT:
        return EXPLICIT[name]
    if "stable_diffusion" in name:
        return "sd_pipelines" if PIPE.search(name) else "sd_blocks"
    if CONV.search(name):
        return "conv"
    if MATMUL.search(name):
        return "matmul"
    if INDEX.search(name):
        return "indexing"
    return "elementwise"


def _parse():
    src = BENCH.read_text(encoding="utf-8")
    return src, src.splitlines(keepends=True), ast.parse(src)


def _import_block(module_name: str, names: set[str]) -> str:
    body = "".join(f"    {name},\n" for name in sorted(names))
    return f"from {module_name} import (\n{body})\n"


def _ensure_packages() -> None:
    SCEN.mkdir(parents=True, exist_ok=True)
    sd = SCEN / "sd"
    sd.mkdir(parents=True, exist_ok=True)
    init = sd / "__init__.py"
    if not init.exists():
        init.write_text("", encoding="utf-8")


def cmd_list() -> None:
    _, _, tree = _parse()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            domain = classify(node.name)
            if domain is not None:
                print(f"{domain}\t{node.name}")


def cmd_shim() -> None:
    _ensure_packages()
    src = BENCH.read_text(encoding="utf-8")
    moved: dict[str, list[str]] = {}
    remaining: list[str] = []
    for name in SHIM_NAMES:
        if re.search(rf"^def {re.escape(name)}\(", src, flags=re.MULTILINE):
            remaining.append(name)
        else:
            moved.setdefault(classify(name), []).append(name)
    parts = [
        '"""Public window for the scenario setup/path functions shared by benchmarks and correctness tests.\n',
        "\n",
        "Tests must import scenario functions from this module, never from\n",
        "tests.compat.benchmarking. Regenerate with: python3 tools/phase4_scenarios.py shim\n",
        '"""\n',
        "from __future__ import annotations\n",
        "\n",
        "from typing import Any\n",
        "\n",
    ]
    for domain in ORDER:
        if domain in moved:
            parts.append(_import_block(MODULES[domain][0], set(moved[domain])))
            parts.append("\n")
    parts.append("_BENCHMARKING_NAMES = frozenset({\n")
    for name in remaining:
        parts.append(f'    "{name}",\n')
    parts.append("})\n")
    parts.append(
        "\n\ndef __getattr__(name: str) -> Any:\n"
        "    if name in _BENCHMARKING_NAMES:\n"
        "        from tests.compat import benchmarking\n"
        "\n"
        "        return getattr(benchmarking, name)\n"
        "    raise AttributeError(f\"module {__name__!r} has no attribute {name!r}\")\n"
    )
    (SCEN / "__init__.py").write_text("".join(parts), encoding="utf-8")
    print(f"shim: {len(remaining)} names still re-exported from benchmarking, "
          f"{sum(len(v) for v in moved.values())} from scenario modules")


def cmd_move(domain: str) -> None:
    if domain not in MODULES:
        sys.exit(f"unknown domain {domain!r}; expected one of {ORDER}")
    _ensure_packages()
    src, lines, tree = _parse()
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    dom = {n.name: classify(n.name) for n in funcs}
    # Include functions that earlier move steps already placed in scenario modules,
    # so that cross-module references are turned into imports.
    for other_domain, (_, other_path) in MODULES.items():
        if other_domain != domain and other_path.exists():
            for node in ast.parse(other_path.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.FunctionDef):
                    dom.setdefault(node.name, other_domain)
    targets = [n for n in funcs if dom[n.name] == domain]
    if not targets:
        sys.exit(f"BLOCKED: no functions left in benchmarking.py for domain {domain}")
    for earlier in ORDER[: RANK[domain]]:
        if any(dom[n.name] == earlier for n in funcs):
            sys.exit(f"BLOCKED: domain {earlier} must be moved before {domain}")
    for node in targets:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and dom.get(sub.id) and RANK[dom[sub.id]] > RANK[domain]:
                sys.exit(f"BLOCKED: {node.name} references {sub.id} from later domain {dom[sub.id]}")

    moved = {n.name for n in targets}
    external: dict[str, set[str]] = {}
    for node in targets:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in dom and sub.id not in moved:
                external.setdefault(dom[sub.id], set()).add(sub.id)

    header = [
        '"""Scenario definitions moved verbatim from tests/compat/benchmarking.py (phase 4-1-b)."""\n',
        "from __future__ import annotations\n",
        "\n",
        "from typing import Any\n",
        "\n",
    ]
    for dep_domain in ORDER:
        if dep_domain in external and dep_domain != domain:
            header.append(_import_block(MODULES[dep_domain][0], external[dep_domain]))
    spans = sorted((n.lineno - 1, n.end_lineno) for n in targets)
    blocks = ["".join(lines[a:b]) for a, b in spans]
    dest = MODULES[domain][1]
    if dest.exists():
        sys.exit(f"BLOCKED: {dest} already exists")
    dest.write_text("".join(header) + "\n\n" + "\n\n".join(b.rstrip("\n") + "\n" for b in blocks),
                    encoding="utf-8")

    kept: list[str] = []
    prev = 0
    for a, b in spans:
        kept.append("".join(lines[prev:a]))
        prev = b
    kept.append("".join(lines[prev:]))
    out = "".join(kept)
    still_used = {n.id for n in ast.walk(ast.parse(out)) if isinstance(n, ast.Name) and n.id in moved}
    if still_used:
        anchor = "from .harness import assert_values_compatible\n"
        assert out.count(anchor) == 1
        out = out.replace(anchor, anchor + "\n" + _import_block(MODULES[domain][0], still_used), 1)
    BENCH.write_text(out, encoding="utf-8")
    cmd_shim()
    print(f"move: {len(targets)} functions -> {dest}")


def entry_domain(call: ast.Call, dom: dict[str, str]) -> str:
    case_id = call.args[0].value
    setup = call.args[1].id if isinstance(call.args[1], ast.Name) else None
    run = call.args[2].id if isinstance(call.args[2], ast.Name) else None
    if run and run in dom:
        return dom[run]
    if setup and setup in dom:
        return dom[setup]
    return classify(case_id[len("bench."):])


def cmd_cases() -> None:
    src, lines, tree = _parse()
    if not (ROOT / "tests" / "compat" / "bench_types.py").exists():
        sys.exit("BLOCKED: create tests/compat/bench_types.py first (step 4-1-c action 1)")
    dom: dict[str, str] = {}
    for domain, (module_name, path) in MODULES.items():
        if not path.exists():
            sys.exit(f"BLOCKED: {path} does not exist; finish step 4-1-b first")
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.FunctionDef):
                dom[node.name] = domain
    tuple_node = None
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign):
            target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "BENCHMARK_CASES":
            tuple_node = node
    if tuple_node is None or not isinstance(tuple_node.value, ast.Tuple):
        sys.exit("BLOCKED: BENCHMARK_CASES tuple not found in benchmarking.py")

    per_domain: dict[str, list[str]] = {d: [] for d in ORDER}
    per_domain_deps: dict[str, set[str]] = {d: set() for d in ORDER}
    for call in tuple_node.value.elts:
        home = entry_domain(call, dom)
        text = "".join(lines[call.lineno - 1:call.end_lineno])
        if not text.rstrip().endswith(","):
            text = text.rstrip("\n") + ",\n"
        per_domain[home].append(text)
        for sub in ast.walk(call):
            if isinstance(sub, ast.Name) and sub.id in dom and dom[sub.id] != home:
                per_domain_deps[home].add(sub.id)

    for domain in ORDER:
        if not per_domain[domain]:
            continue
        path = MODULES[domain][1]
        text = path.read_text(encoding="utf-8")
        extra = ["\n\n", "from tests.compat.bench_types import BenchmarkCase\n"]
        deps: dict[str, set[str]] = {}
        for name in per_domain_deps[domain]:
            deps.setdefault(dom[name], set()).add(name)
        for dep_domain in ORDER:
            if dep_domain in deps:
                extra.append(_import_block(MODULES[dep_domain][0], deps[dep_domain]))
        extra.append("\nCASES: tuple[BenchmarkCase, ...] = (\n")
        extra.extend(per_domain[domain])
        extra.append(")\n")
        path.write_text(text.rstrip("\n") + "\n" + "".join(extra), encoding="utf-8")

    concat = (
        "from tests.compat.scenarios import conv, elementwise, indexing, matmul\n"
        "from tests.compat.scenarios.sd import blocks, pipelines\n"
        "\n"
        "BENCHMARK_CASES: tuple[BenchmarkCase, ...] = tuple(\n"
        "    matmul.CASES\n"
        "    + elementwise.CASES\n"
        "    + conv.CASES\n"
        "    + indexing.CASES\n"
        "    + blocks.CASES\n"
        "    + pipelines.CASES\n"
        ")\n"
    )
    bench_import = "from .bench_types import BenchmarkCase  # noqa: F401  (re-exported for engine users)\n"
    # Keep the original import lines up to and including the harness import, then
    # keep every top-level statement except: the BenchmarkCase class, the old
    # BENCHMARK_CASES tuple, and any import statements below the prologue.
    src_lines = src.splitlines(keepends=True)
    harness_index = next(i for i, line in enumerate(src_lines) if line.startswith("from .harness import"))
    prologue = "".join(src_lines[: harness_index + 1])
    keep_nodes: list[str] = []
    for node in tree.body:
        if node.lineno - 1 <= harness_index:
            continue
        if isinstance(node, ast.ClassDef) and node.name == "BenchmarkCase":
            continue
        if node is tuple_node:
            continue
        if isinstance(node, ast.ImportFrom):
            continue
        start = node.lineno - 1
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start = min(start, decorators[0].lineno - 1)
        keep_nodes.append("".join(lines[start:node.end_lineno]).rstrip("\n"))
    new_src = (
        prologue
        + "\n" + bench_import
        + "\n\n" + "\n\n\n".join(keep_nodes)
        + "\n\n\n" + concat
    )
    BENCH.write_text(new_src, encoding="utf-8")
    counts = {d: len(per_domain[d]) for d in ORDER if per_domain[d]}
    print("cases split:", counts, "total", sum(counts.values()))


def _scan_code(code, module, errors, owner) -> None:
    import builtins
    import dis
    import types

    for instruction in dis.get_instructions(code):
        if instruction.opname == "LOAD_GLOBAL":
            name = instruction.argval
            if not hasattr(module, name) and not hasattr(builtins, name):
                errors.append(f"{module.__name__}.{owner}: unresolved global {name!r}")
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _scan_code(const, module, errors, owner)


def cmd_verify(ids_file: str) -> None:
    import importlib

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    errors: list[str] = []
    module_names = ["tests.compat.benchmarking", "tests.compat.scenarios"]
    module_names += [MODULES[d][0] for d in ORDER if MODULES[d][1].exists()]
    for module_name in module_names:
        module = importlib.import_module(module_name)
        for attr_name, attr in vars(module).items():
            if getattr(attr, "__module__", None) == module_name and hasattr(attr, "__code__"):
                _scan_code(attr.__code__, module, errors, attr_name)
    importlib.import_module("tests.compat.test_tensor_ops")
    from tests.compat.benchmarking import BENCHMARK_CASES

    ids = sorted(case.id for case in BENCHMARK_CASES)
    expected = Path(ids_file).read_text(encoding="utf-8").split()
    if ids != expected:
        errors.append(
            f"benchmark id list mismatch: {len(ids)} current vs {len(expected)} expected; "
            f"missing={sorted(set(expected) - set(ids))[:5]} added={sorted(set(ids) - set(expected))[:5]}"
        )
    for message in errors:
        print("FAIL:", message)
    if errors:
        sys.exit(1)
    print(f"verify OK: {len(ids)} benchmark cases, all scenario globals resolve")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    command = sys.argv[1]
    if command == "list":
        cmd_list()
    elif command == "shim":
        cmd_shim()
    elif command == "move":
        if len(sys.argv) != 3:
            sys.exit("usage: phase4_scenarios.py move <domain>")
        cmd_move(sys.argv[2])
    elif command == "cases":
        cmd_cases()
    elif command == "verify":
        if len(sys.argv) != 3:
            sys.exit("usage: phase4_scenarios.py verify <bench-ids-file>")
        cmd_verify(sys.argv[2])
    else:
        sys.exit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
```

2. Generate the shim package (this creates `tests/compat/scenarios/__init__.py` and
   `tests/compat/scenarios/sd/__init__.py`):

```bash
cd /Users/hiramatsu/dev/mtorch
python3 tools/phase4_scenarios.py shim
```

   Expected output, exactly: `shim: 120 names still re-exported from benchmarking, 0 from scenario modules`.
   The generated `__init__.py` contains only a docstring, `_BENCHMARKING_NAMES = frozenset({...120 names...})`,
   and a module-level `__getattr__` that forwards those names to `tests.compat.benchmarking`.

3. Rewire the import in `test_tensor_ops.py` (single mechanical replacement of the L8 header;
   the 120 imported names themselves stay untouched):

```bash
python3 - <<'EOF'
from pathlib import Path
p = Path("tests/compat/test_tensor_ops.py")
text = p.read_text(encoding="utf-8")
old = "from .benchmarking import ("
assert text.count(old) == 1, "unexpected import header"
p.write_text(text.replace(old, "from .scenarios import ("), encoding="utf-8")
print("rewired")
EOF
```

**Verification**:

```bash
cd /Users/hiramatsu/dev/mtorch
grep -c 'benchmarking' tests/compat/test_tensor_ops.py    # must print 0
python3 -c "import tests.compat.scenarios as s; print(len(s._BENCHMARKING_NAMES))"   # 120
python3 tools/phase4_scenarios.py verify docs/design/baseline/bench-ids-phase4.txt
pytest tests/compat --collect-only -q | tail -2 | sed -E 's/ in [0-9.]+s//' > /tmp/phase4-collect-after.txt
diff docs/design/baseline/collect-count-phase4.txt /tmp/phase4-collect-after.txt && echo "COLLECT: MATCH"
pytest tests/compat -q 2>&1 | tail -1 | sed -E 's/ in [0-9.]+s.*$//' > /tmp/phase4-tests-after.txt
diff docs/design/baseline/tests-phase4.txt /tmp/phase4-tests-after.txt && echo "TESTS: MATCH"
```

**On failure**: Recovery rule.

**Commit**:

```bash
git add tools/phase4_scenarios.py tests/compat/scenarios tests/compat/test_tensor_ops.py
git commit -m "refactor(phase4-1a): add scenarios shim and rewire test_tensor_ops imports"
```

### Step 4-1-b: Physically move the scenario definitions (1 domain = 1 commit)

**Goal**: All 390 scenario/helper functions leave `benchmarking.py` and land in the domain modules,
verbatim. `benchmarking.py` keeps the engine and `BENCHMARK_CASES` (entries now reference imported
names — the `move` subcommand inserts the needed explicit imports; it never uses `import *`).
The shim `__init__.py` is regenerated automatically after each move.

**Preconditions**:

```bash
cd /Users/hiramatsu/dev/mtorch
git status --porcelain                                    # nothing
test -f tests/compat/scenarios/__init__.py && echo "4-1-a OK"
test ! -f tests/compat/scenarios/_data.py && echo "no domain moved yet: OK"
```

**Actions** — run this identical 5-command sequence once per domain, in EXACTLY this order
(7 domains = 7 commits). Expected `move:` counts and shim counts are in the table below; if a
printed count differs, treat it as a failure.

```bash
cd /Users/hiramatsu/dev/mtorch
python3 tools/phase4_scenarios.py move _data
python3 tools/phase4_scenarios.py verify docs/design/baseline/bench-ids-phase4.txt
pytest tests/compat -q 2>&1 | tail -1 | sed -E 's/ in [0-9.]+s.*$//' > /tmp/phase4-tests-after.txt
diff docs/design/baseline/tests-phase4.txt /tmp/phase4-tests-after.txt && echo "TESTS: MATCH"
git add tests/compat/benchmarking.py tests/compat/scenarios && git commit -m "refactor(phase4-1b): move _data scenarios into tests/compat/scenarios"
```

Repeat, substituting the domain name in the `move` command and in the commit message, for:
`matmul`, `elementwise`, `conv`, `indexing`, `sd_blocks`, `sd_pipelines`.

| order | domain        | destination module                        | expected `move:` count | expected shim line after the move |
|-------|---------------|-------------------------------------------|-----|--------------------------------------------|
| 1     | `_data`       | `tests/compat/scenarios/_data.py`          | 12  | `120 ... from benchmarking, 0 from scenario modules` |
| 2     | `matmul`      | `tests/compat/scenarios/matmul.py`         | 17  | `120 ..., 0 ...`                            |
| 3     | `elementwise` | `tests/compat/scenarios/elementwise.py`    | 53  | `120 ..., 0 ...`                            |
| 4     | `conv`        | `tests/compat/scenarios/conv.py`           | 14  | `120 ..., 0 ...`                            |
| 5     | `indexing`    | `tests/compat/scenarios/indexing.py`       | 18  | `120 ..., 0 ...`                            |
| 6     | `sd_blocks`   | `tests/compat/scenarios/sd/blocks.py`      | 186 | `55 ..., 65 ...`                            |
| 7     | `sd_pipelines`| `tests/compat/scenarios/sd/pipelines.py`   | 90  | `0 ..., 120 ...`                            |

Notes:
- The full-suite pytest run per domain is intentional (the special rule). The `verify` subcommand
  additionally does a static scan proving every global name in every scenario module resolves.
- If the script prints anything starting with `BLOCKED:`, do not improvise — Recovery rule.

**Verification** (after the 7th commit):

```bash
cd /Users/hiramatsu/dev/mtorch
python3 tools/phase4_scenarios.py verify docs/design/baseline/bench-ids-phase4.txt
pytest tests/compat --collect-only -q | tail -2 | sed -E 's/ in [0-9.]+s//' > /tmp/phase4-collect-after.txt
diff docs/design/baseline/collect-count-phase4.txt /tmp/phase4-collect-after.txt && echo "COLLECT: MATCH"
grep -c '^def ' tests/compat/benchmarking.py     # 4  (engine functions only)
wc -l tests/compat/benchmarking.py               # roughly 4,000 (BENCHMARK_CASES + imports remain)
```

**On failure**: Recovery rule (this rolls back only the current, uncommitted domain move; already
committed domains stay).

**Commit**: included in the per-domain sequence above.

### Step 4-1-c: Split BENCHMARK_CASES (including extracting bench_types.py)

**Goal**: `BENCHMARK_CASES` (415 entries, ~2,740 lines) is split into a `CASES` tuple per domain
module; `benchmarking.py` becomes engine-only (< 200 lines) and just concatenates them. The
`BenchmarkCase` dataclass moves to `tests/compat/bench_types.py` so scenario modules never import
`benchmarking` (no cycle); `benchmarking` re-exports `BenchmarkCase` for its existing users.

**Preconditions**:

```bash
cd /Users/hiramatsu/dev/mtorch
git status --porcelain                                        # nothing
test -f tests/compat/scenarios/sd/pipelines.py && echo "4-1-b OK"
test ! -f tests/compat/bench_types.py && echo "no bench_types yet: OK"
```

**Actions**:

1. Create `tests/compat/bench_types.py` with exactly this content (field-for-field identical to the
   current `BenchmarkCase` at the top of `benchmarking.py`):

```python
"""Shared benchmark case type, importable by both the engine and the scenario modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    setup: Callable[[Any], tuple[Any, ...]]
    run: Callable[..., Any]
    rtol: float | None = None
    atol: float | None = None
```

2. Run the split (assigns each entry to the module that defines its `run` path function, falling
   back to its `setup` function, falling back to keyword classification of the id; appends a
   `CASES` tuple plus the imports it needs to each domain module; rewrites `benchmarking.py`
   to engine + concatenation):

```bash
python3 tools/phase4_scenarios.py cases
```

   Expected output, exactly:
   `cases split: {'matmul': 27, 'elementwise': 171, 'conv': 32, 'indexing': 48, 'sd_blocks': 89, 'sd_pipelines': 48} total 415`

3. Confirm the new tail of `benchmarking.py` is the concatenation block:

```bash
tail -14 tests/compat/benchmarking.py
```

   It must end with the `BENCHMARK_CASES: tuple[BenchmarkCase, ...] = tuple(...)` concatenation of
   `matmul.CASES + elementwise.CASES + conv.CASES + indexing.CASES + blocks.CASES + pipelines.CASES`.

**Verification**:

```bash
cd /Users/hiramatsu/dev/mtorch
wc -l tests/compat/benchmarking.py     # must be < 200 (expected: ~137)
python3 tools/phase4_scenarios.py verify docs/design/baseline/bench-ids-phase4.txt
pytest tests/compat --collect-only -q | tail -2 | sed -E 's/ in [0-9.]+s//' > /tmp/phase4-collect-after.txt
diff docs/design/baseline/collect-count-phase4.txt /tmp/phase4-collect-after.txt && echo "COLLECT: MATCH"
pytest tests/compat -q 2>&1 | tail -1 | sed -E 's/ in [0-9.]+s.*$//' > /tmp/phase4-tests-after.txt
diff docs/design/baseline/tests-phase4.txt /tmp/phase4-tests-after.txt && echo "TESTS: MATCH"
```

Note: the concatenation changes the *order* of `BENCHMARK_CASES` (domain by domain). The pass
criterion of this phase is the sorted ID list plus the collection count, both checked above; the
order is explicitly allowed to change here and nowhere else.

**On failure**: Recovery rule.

**Commit**:

```bash
git add tests/compat/bench_types.py tests/compat/benchmarking.py tests/compat/scenarios
git commit -m "refactor(phase4-1c): split BENCHMARK_CASES into scenario modules via bench_types"
```

---

## Step 4-2: Parameterize SD variants (1 family = 1 commit)

The SD-family scenarios (139 `_case_stable_diffusion*_setup` functions) are a "base form ×
combination of variant flags" expanded into name-differing functions. Each family is compressed to
one spec dataclass + one spec-driven implementation + thin wrappers **that keep every existing
function name**, so `test_tensor_ops.py`, `cases.py` and all `CASES` entries stay valid and the
benchmark ID list never changes.

**Preconditions**:

```bash
cd /Users/hiramatsu/dev/mtorch
git status --porcelain                                        # nothing
test -f tests/compat/bench_types.py && echo "4-1-c OK"
```

**Actions**:

1. Build the worklist once:

```bash
grep -hoE '^def _case_stable_diffusion[a-z0-9_]*_setup' tests/compat/scenarios/sd/pipelines.py tests/compat/scenarios/sd/blocks.py | sed 's/^def //' | sort > /tmp/phase4-2-setups.txt
wc -l /tmp/phase4-2-setups.txt      # 139
```

   Process base forms in this fixed order (count of setup functions containing the token):
   `denoising_loop` (23), `transformer_stack` (10), `module_pipeline` (5), `scheduler_step` (5),
   `joint_attention`/`cross_attention`/`attention` (19), `text_encoder` (4), `transformer_block` (3),
   `unpatchify_projection` (3), `patch_embed` (2). A *family* is the group of setups sharing one
   base form AND one shared `_..._path` function. **Skip rule (mandatory)**: if the variant bodies
   differ by anything other than literal values that can become dataclass fields (extra loop logic,
   different helper calls), leave that family untouched and note it in the PROGRESS Phase 4 Notes.
   Setups matching no token (64) are singletons — skip them.

2. For each family, apply exactly the recipe of the worked example below: diff the member bodies,
   define a frozen spec dataclass whose fields are the diffing literals, write ONE implementation
   function taking `(module, spec)`, and turn every existing `_case_*_setup` member into a wrapper.
   Then (same commit) replace the family's literal `CASES` entries with product generation.

**Worked example — family `sd3_rectangular_pooled_controlnet_denoising_loop`
(members: base + `long`; file: `tests/compat/scenarios/sd/pipelines.py`)**

Locate the two members (line hints are pre-4-1-b positions in benchmarking.py; in pipelines.py use
the greps):

```bash
grep -n 'def _case_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_setup' tests/compat/scenarios/sd/pipelines.py
grep -n 'def _case_stable_diffusion_sd3_rectangular_pooled_controlnet_long_denoising_loop_setup' tests/compat/scenarios/sd/pipelines.py
```

Diff result (verified): the `long` variant is the base setup with element 5 (`sigmas`), element 7
(`step_count`, 2→4) and element 8 (`guidance_scale`, 5.5→5.75) of the returned tuple replaced.
Those three literals are the spec.

(a) Add near the top of `tests/compat/scenarios/sd/pipelines.py`, directly below the existing
import block, the import `from dataclasses import dataclass` (keep `from __future__ import
annotations` first).

(b) Replace the **entire body** of the two member setup functions with this block (the impl body is
the original base body verbatim, with the three literals swapped for spec fields — everything else,
including every constant like `0.65`, `0.012`, `float16`, must be byte-identical to the original):

```python
@dataclass(frozen=True)
class Sd3RectPooledControlnetLoopSpec:
    sigmas: tuple[float, ...] = (1.0, 0.76, 0.42, 0.12, 0.0)
    step_count: int = 2
    guidance_scale: float = 5.5


_SD3_RECT_POOLED_CONTROLNET_LOOP_SPECS: dict[str, Sd3RectPooledControlnetLoopSpec] = {
    "2step": Sd3RectPooledControlnetLoopSpec(),
    "4step": Sd3RectPooledControlnetLoopSpec(
        sigmas=(1.0, 0.86, 0.64, 0.38, 0.14, 0.0),
        step_count=4,
        guidance_scale=5.75,
    ),
}


def _sd3_rectangular_pooled_controlnet_denoising_loop_setup_from_spec(
    module: Any, spec: Sd3RectPooledControlnetLoopSpec
) -> tuple[Any, ...]:
    stack_args = _case_stable_diffusion_sd3_rectangular_pooled_multi_controlnet_transformer_stack_setup(module)
    latent, patch_weight, patch_bias, pos_embed = stack_args[:4]
    context = stack_args[4]
    joint_layers, single_layers, unpatchify_args = stack_args[5:8]
    block_residuals, controlnet_keep = stack_args[8:10]
    pooled_prompt_embeds = stack_args[11]
    conditioning_weights = stack_args[12:20]
    joint_mod_layers, single_mod_layers, final_mod_weight, final_mod_bias = stack_args[20:24]

    latent_uncond, latent_cond = latent.chunk(2, dim=0)
    latents = (latent_uncond * 0.65 + module.flip(latent_cond, (-1,)) * 0.35).contiguous(
        memory_format=module.channels_last
    )
    negative_prompt_embeds, prompt_embeds = context.chunk(2, dim=0)
    negative_prompt_embeds = (negative_prompt_embeds - 0.012).to(dtype=module.float16)
    prompt_embeds = (prompt_embeds + 0.015).to(dtype=module.float16)
    negative_pooled_prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds.chunk(2, dim=0)
    negative_pooled_prompt_embeds = (negative_pooled_prompt_embeds - 0.01).to(dtype=module.float16)
    pooled_prompt_embeds = (pooled_prompt_embeds + 0.012).to(dtype=module.float16)
    sigmas = module.tensor(list(spec.sigmas), dtype=module.float32)
    index = module.tensor(0, dtype=module.long)
    model_args = (
        patch_weight,
        patch_bias,
        pos_embed,
        joint_layers,
        single_layers,
        unpatchify_args,
        block_residuals,
        controlnet_keep,
        *conditioning_weights,
        joint_mod_layers,
        single_mod_layers,
        final_mod_weight,
        final_mod_bias,
    )
    return (
        latents,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        pooled_prompt_embeds,
        sigmas,
        index,
        spec.step_count,
        spec.guidance_scale,
        model_args,
    )


def _case_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_setup(module: Any) -> tuple[Any, ...]:
    return _sd3_rectangular_pooled_controlnet_denoising_loop_setup_from_spec(
        module, _SD3_RECT_POOLED_CONTROLNET_LOOP_SPECS["2step"]
    )


def _case_stable_diffusion_sd3_rectangular_pooled_controlnet_long_denoising_loop_setup(
    module: Any,
) -> tuple[Any, ...]:
    return _sd3_rectangular_pooled_controlnet_denoising_loop_setup_from_spec(
        module, _SD3_RECT_POOLED_CONTROLNET_LOOP_SPECS["4step"]
    )
```

(Other functions such as `_case_..._img2img_denoising_loop_setup` call the `long` wrapper; wrappers
keep the names, so they need no edits.)

(c) Product generation for the family's benchmark entries. Add directly after the wrappers:

```python
def _sd3_rectangular_pooled_controlnet_denoising_loop_cases() -> tuple["BenchmarkCase", ...]:
    tolerances = {"2step": (2e-2, 2e-1), "4step": (3e-2, 3e-1)}
    setups = {
        "2step": _case_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_setup,
        "4step": _case_stable_diffusion_sd3_rectangular_pooled_controlnet_long_denoising_loop_setup,
    }
    cases = []
    for suffix in ("2step", "4step"):
        rtol, atol = tolerances[suffix]
        cases.append(
            BenchmarkCase(
                "bench.stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_half_1x16x16x24_"
                + suffix,
                setups[suffix],
                _stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path,
                rtol=rtol,
                atol=atol,
            )
        )
    return tuple(cases)
```

Then in the `CASES` tuple of the same file, find the two literal entries by their FULL id strings
(shorter substrings are ambiguous — `denoising_loop_half_1x16x16x24_4step` alone also appears in
12 other ids such as `..._keep_decay_...` and `..._inpaint_...`):

```bash
grep -n '"bench.stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_half_1x16x16x24_2step"' tests/compat/scenarios/sd/pipelines.py
grep -n '"bench.stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_half_1x16x16x24_4step"' tests/compat/scenarios/sd/pipelines.py
```

Each grep must match exactly once, and the two `BenchmarkCase(...)` entries are adjacent (the
`_2step` one uses setup `..._denoising_loop_setup`, the `_4step` one uses setup
`..._long_denoising_loop_setup`). Replace both complete entries (each spans from its
`    BenchmarkCase(` line through its closing `    ),` line) with the single line:

```python
    *_sd3_rectangular_pooled_controlnet_denoising_loop_cases(),
```

The generated ID strings are byte-identical to the previous literals — that is mandatory for every
family.

**Verification** (per family; correctness is guaranteed because `run_benchmark_case` calls
`assert_values_compatible` on every benchmark execution, so benchmarks passing = numeric behavior
preserved):

```bash
cd /Users/hiramatsu/dev/mtorch
# machine-check: generated benchmark ID set equals the 4-0 capture
python3 tools/phase4_scenarios.py verify docs/design/baseline/bench-ids-phase4.txt
# family-scoped machine-check of the generated ids
python3 - <<'EOF'
from tests.compat.benchmarking import BENCHMARK_CASES
prefix = "bench.stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_half_1x16x16x24_"
ids = sorted(c.id for c in BENCHMARK_CASES if c.id.startswith(prefix))
assert ids == [prefix + "2step", prefix + "4step"], ids
print("family ids OK:", ids)
EOF
# run the family's benchmarks (numeric assertion inside) and the SD correctness tests
pytest tests/compat/test_benchmarks.py --compat-benchmark 'bench.stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop*' -q
pytest tests/compat -q -k stable_diffusion 2>&1 | tail -1
pytest tests/compat --collect-only -q | tail -2 | sed -E 's/ in [0-9.]+s//' > /tmp/phase4-collect-after.txt
diff docs/design/baseline/collect-count-phase4.txt /tmp/phase4-collect-after.txt && echo "COLLECT: MATCH"
```

For other families, substitute the family's benchmark-id prefix in the `--compat-benchmark` pattern
and in the family-scoped check.

**On failure**: Recovery rule.

**Commit** (one per family):

```bash
git add tests/compat/scenarios
git commit -m "refactor(phase4-2): parameterize sd3_rectangular_pooled_controlnet_denoising_loop family"
```

Step 4-2 is complete when every family in the worklist is either parameterized or explicitly noted
as skipped in PROGRESS Notes.

---

## Step 4-3: Table-ify hand-written tests (`PATH_CASES`)

Most of the 195 hand-written functions in `test_tensor_ops.py` have the same shape:
"setup with reference → setup with candidate → run path on both → `assert_values_compatible`"
(there are 131 `test_*_path_matches_reference` functions:
`grep -c 'def test_.*_path_matches_reference' tests/compat/test_tensor_ops.py`). These migrate into
a `PATH_CASES` table driven by one parametrized test, using the same mechanism as the existing
`OP_CASES` (`pytest_generate_tests` in `tests/compat/conftest.py`).

**Collection-count invariant**: the driver plus its first table entry must land in the SAME commit
as the deletion of the first migrated hand-written test ("function −1, parametrize +1" — an empty
parametrize would add a phantom skipped item and break the count).

**Preconditions**:

```bash
cd /Users/hiramatsu/dev/mtorch
git status --porcelain                              # nothing
grep -c 'PATH_CASES' tests/compat/cases.py          # 0 (not started)
```

**Actions**:

1. In `tests/compat/cases.py`, replace the import header. Old (L1–3):

```python
from __future__ import annotations

from .harness import DType, FactorySpec, GradCase, OpCase, TensorSpec, ViewCase
```

   New:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .harness import DType, FactorySpec, GradCase, OpCase, TensorSpec, ViewCase
from .scenarios import (
    _case_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_setup,
    _stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path,
)
```

2. Append at the very end of `tests/compat/cases.py` (currently ends at L4107, after the
   `GRAD_CASES` tuple — anchor: `grep -n '^GRAD_CASES' tests/compat/cases.py` → L2906):

```python
@dataclass(frozen=True)
class PathCase:
    id: str
    setup: Callable[[Any], tuple[Any, ...]]
    run: Callable[..., Any]
    rtol: float | None = None
    atol: float | None = None
    check_stride: bool = True


PATH_CASES: tuple[PathCase, ...] = (
    PathCase(
        id="stable_diffusion.sd3_rectangular_pooled_controlnet_denoising_loop",
        setup=_case_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_setup,
        run=_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path,
        rtol=2e-2,
        atol=2e-1,
    ),
)
```

3. In `tests/compat/conftest.py`: change the import (anchor:
   `grep -n 'from .cases import' tests/compat/conftest.py` → L16) from
   `from .cases import GRAD_CASES, INPLACE_CASES, OP_CASES, VIEW_CASES` to
   `from .cases import GRAD_CASES, INPLACE_CASES, OP_CASES, PATH_CASES, VIEW_CASES`,
   and inside `pytest_generate_tests` (anchor: `grep -n 'if "grad_case"' tests/compat/conftest.py`
   → L69), insert immediately after the three-line `grad_case` block, with identical indentation:

```python
    if "path_case" in metafunc.fixturenames:
        cases = [case for case in PATH_CASES if _matches(case.id, patterns)]
        metafunc.parametrize("path_case", cases, ids=[case.id for case in cases])
```

4. In `tests/compat/test_tensor_ops.py`, insert directly after `test_operator_matches_reference`
   (anchor: `grep -n 'def test_operator_matches_reference' tests/compat/test_tensor_ops.py` → L136):

```python
def test_path_case_matches_reference(torch_reference, torch_candidate, path_case) -> None:
    expected_args = path_case.setup(torch_reference)
    actual_args = path_case.setup(torch_candidate)

    assert_values_compatible(
        torch_reference,
        path_case.run(torch_reference, *expected_args),
        path_case.run(torch_candidate, *actual_args),
        path=path_case.id,
        rtol=path_case.rtol,
        atol=path_case.atol,
        check_stride=path_case.check_stride,
    )
```

5. In the SAME commit, delete the hand-written function
   `test_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path_matches_reference`
   (anchor: `grep -n 'def test_stable_diffusion_sd3_rectangular_pooled_controlnet_denoising_loop_path_matches_reference' tests/compat/test_tensor_ops.py`
   → L3607; the function is 17 lines: two setup lines + one `assert_values_compatible(...)` call with
   `path="stable_diffusion.sd3_rectangular_pooled_controlnet_denoising_loop", rtol=2e-2, atol=2e-1,
   check_stride=True`). Delete from its `def` line through the line before the next `def`.
   Leave the import block of `test_tensor_ops.py` untouched even if a name becomes unused —
   import cleanup is NOT part of this step.

6. Create `tests/compat/README.md` with exactly this content:

```markdown
# tests/compat conventions

- New correctness tests MUST be table entries, not hand-written functions:
  operators -> `OP_CASES` / `INPLACE_CASES` / `VIEW_CASES` / `GRAD_CASES` (tests/compat/cases.py),
  multi-op numeric paths -> `PATH_CASES` (tests/compat/cases.py),
  nn modules -> `MODULE_CASES` (tests/compat/test_nn_modules.py),
  benchmarks -> `CASES` in the matching tests/compat/scenarios/ module.
  Adding new hand-written `def test_*` functions for these categories is prohibited.
- Scenario setup/path functions live in tests/compat/scenarios/ and are imported by tests
  ONLY via `tests.compat.scenarios` (the package __init__), never from
  `tests.compat.benchmarking`.
- The benchmark case ID list and the pytest collection count are contract surfaces: any change
  to them must be intentional and reviewed against docs/design/baseline/bench-ids-phase4.txt
  and collect-count-phase4.txt.
```

7. Migrate further tests gradually (about 10 in the next commit to establish the pattern, then
   mechanically): a hand-written test is eligible ONLY if its body is exactly the worked-example
   shape — two `*_setup(...)` lines plus one `assert_values_compatible(...)` call whose `path=`,
   `rtol=`, `atol=`, `check_stride=` are literals. For each: append one `PathCase` with `id` equal
   to the old `path=` string and the same tolerances, add the setup/run names to the
   `from .scenarios import (...)` block in `cases.py`, and delete the function. Candidate list:
   `grep -n 'def test_.*_path_matches_reference' tests/compat/test_tensor_ops.py`. Any test with
   extra statements: skip it, leave it hand-written.

**Verification** (after every 4-3 commit):

```bash
cd /Users/hiramatsu/dev/mtorch
python3 tools/phase4_scenarios.py verify docs/design/baseline/bench-ids-phase4.txt
pytest tests/compat --collect-only -q | tail -2 | sed -E 's/ in [0-9.]+s//' > /tmp/phase4-collect-after.txt
diff docs/design/baseline/collect-count-phase4.txt /tmp/phase4-collect-after.txt && echo "COLLECT: MATCH"
pytest tests/compat -q 2>&1 | tail -1 | sed -E 's/ in [0-9.]+s.*$//' > /tmp/phase4-tests-after.txt
diff docs/design/baseline/tests-phase4.txt /tmp/phase4-tests-after.txt && echo "TESTS: MATCH"
```

**On failure**: Recovery rule.

**Commit** (first commit shown; later migration batches use the same form):

```bash
git add tests/compat/cases.py tests/compat/conftest.py tests/compat/test_tensor_ops.py tests/compat/README.md
git commit -m "refactor(phase4-3): add PATH_CASES table, driver, and first migrated test"
```

---

## Step 4-4: Convert `test_nn_modules.py` to `ModuleCase`

Same approach for the 89 hand-written module tests: a `ModuleCase` table (module-construction spec +
input spec + tolerance) plus one parametrized driver, reusing the existing construction helpers
(`_linear` — grep `def _linear` → L21, `_conv2d` → L70, etc.; they already implement
"construct → copy fixed parameter values under `no_grad` via `_copy_parameter`"). The driver
faithfully transcribes the existing comparison procedure "construct on both sides → (parameters
already copied by the helper) → compare forward via `assert_values_compatible`"; do not change the
direction or dtype of any parameter copy. The table and driver live in `test_nn_modules.py` itself
(the helpers are private to that file) and use `@pytest.mark.parametrize`.

**Collection-count invariant**: same as 4-3 — driver + first entry + first deletion in one commit.

**Preconditions**:

```bash
cd /Users/hiramatsu/dev/mtorch
git status --porcelain                                   # nothing
grep -c 'MODULE_CASES' tests/compat/test_nn_modules.py   # 0 (not started)
```

**Actions**:

1. Replace the import header of `tests/compat/test_nn_modules.py`. Old (L1–6):

```python
from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .harness import assert_values_compatible
```

   New:

```python
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from .harness import assert_values_compatible
```

2. Insert the table + driver immediately BEFORE
   `def test_linear_module_forward_matches_reference` (anchor:
   `grep -n 'def test_linear_module_forward_matches_reference' tests/compat/test_nn_modules.py`
   → L232, i.e. after the last construction helper `_multihead_attention`):

```python
@dataclass(frozen=True)
class ModuleCase:
    id: str
    build: Callable[[Any], Any]
    inputs: Callable[[Any], tuple[Any, ...]]
    rtol: float | None = None
    atol: float | None = None


def _linear_forward_inputs(module: Any) -> tuple[Any, ...]:
    return (_tensor(module, [[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]]),)


MODULE_CASES: tuple[ModuleCase, ...] = (
    ModuleCase(
        id="nn.Linear.forward",
        build=_linear,
        inputs=_linear_forward_inputs,
        rtol=1e-6,
        atol=1e-6,
    ),
)


@pytest.mark.parametrize("module_case", MODULE_CASES, ids=[case.id for case in MODULE_CASES])
def test_module_case_forward_matches_reference(torch_reference, torch_candidate, module_case) -> None:
    expected_layer = module_case.build(torch_reference)
    actual_layer = module_case.build(torch_candidate)
    expected_inputs = module_case.inputs(torch_reference)
    actual_inputs = module_case.inputs(torch_candidate)

    assert_values_compatible(
        torch_reference,
        expected_layer(*expected_inputs),
        actual_layer(*actual_inputs),
        path=module_case.id,
        rtol=module_case.rtol,
        atol=module_case.atol,
    )
```

3. In the SAME commit, delete the hand-written `test_linear_module_forward_matches_reference`
   (L232, 15 lines: builds `_linear` on both sides, one input tensor
   `[[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]]`, asserts with `path="nn.Linear.forward"`, `rtol=1e-6`,
   `atol=1e-6` — exactly what the table entry reproduces).

4. Migrate further tests gradually (1–10 per commit). Eligible: tests whose body is exactly
   "build helper on both sides → build identical literal inputs on both sides → one
   `assert_values_compatible` on the forward call". For each, add one `ModuleCase` (id = the old
   `path=` string, an `*_inputs` function returning the literal input tuple, the old tolerances)
   and delete the function. Candidates:
   `grep -n 'def test_.*_forward_matches_reference' tests/compat/test_nn_modules.py`.
   Tests with extra logic (parameter mutation, backward, dtype conversion, output_size arguments,
   masks): skip, leave hand-written.

5. Append to `tests/compat/README.md` (after the first bullet list — already references
   `MODULE_CASES`; no change needed if step 4-3's README was created as specified; otherwise add
   the `MODULE_CASES` line to it).

**Verification** (after every 4-4 commit): same commands as step 4-3's Verification block.

**On failure**: Recovery rule.

**Commit** (first commit shown):

```bash
git add tests/compat/test_nn_modules.py tests/compat/README.md
git commit -m "refactor(phase4-4): add ModuleCase table, driver, and first migrated module test"
```

---

## Step 4-5: Operationalize the API manifest (optional, low priority)

**Goal**: A pinned-version full manifest exists in `compat/` for measuring the compatibility gap.
The default seed manifest (`compat/api_surface_seed.json`) remains the CI gate, unchanged.

**Preconditions**:

```bash
cd /Users/hiramatsu/dev/mtorch
git status --porcelain                       # nothing
python3 -c "import torch; print(torch.__version__)"
test -f tools/generate_api_manifest.py && echo "generator OK"
```

**Actions**:

1. Generate and commit the fixed-version full manifest (the generator's flags are `--module`,
   `--out`, `--max-depth` — see `grep -n add_argument tools/generate_api_manifest.py`):

```bash
cd /Users/hiramatsu/dev/mtorch
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
python3 tools/generate_api_manifest.py --module torch \
    --out "compat/api_surface_torch_${TORCH_VER}.json" --max-depth 3
ls -la "compat/api_surface_torch_${TORCH_VER}.json"
```

2. **Leave `compat/api_surface_seed.json` untouched.** Run the gap measurement once to record the
   current numbers (failures are EXPECTED — they are the unimplemented-API gap, not a regression;
   do not "fix" them and do not add this run to any gate):

```bash
pytest tests/compat/test_api_surface.py --compat-api-manifest "compat/api_surface_torch_${TORCH_VER}.json" 2>&1 | tail -1
```

3. Append to `tests/compat/README.md`:

```markdown

## API surface manifests

- `compat/api_surface_seed.json` is the default manifest used by `tests/compat/test_api_surface.py`
  and is the pass/fail gate. Do not regenerate it casually.
- `compat/api_surface_torch_<version>.json` is the full manifest of the pinned reference torch,
  generated with `python3 tools/generate_api_manifest.py --module torch --out <file> --max-depth 3`.
  Run it with `pytest tests/compat/test_api_surface.py --compat-api-manifest <file>` to measure the
  compatibility gap; many failures are expected and are NOT regressions. Making it a required CI
  gate is compatibility-expansion work, out of scope for refactoring.
```

**Verification**:

```bash
cd /Users/hiramatsu/dev/mtorch
python3 -c "import json,glob; p=glob.glob('compat/api_surface_torch_*.json')[0]; d=json.load(open(p)); print(p, len(d.get('entries', [])), 'entries')"
pytest tests/compat/test_api_surface.py -q 2>&1 | tail -1     # default seed manifest: same result as baseline
pytest tests/compat --collect-only -q | tail -2 | sed -E 's/ in [0-9.]+s//' > /tmp/phase4-collect-after.txt
diff docs/design/baseline/collect-count-phase4.txt /tmp/phase4-collect-after.txt && echo "COLLECT: MATCH"
```

**On failure**: Recovery rule.

**Commit**:

```bash
git add compat/api_surface_torch_*.json tests/compat/README.md
git commit -m "refactor(phase4-5): add pinned full API manifest and document its role"
```

---

## Phase completion criteria (all mechanically checkable)

```bash
cd /Users/hiramatsu/dev/mtorch
# 1. benchmarking.py is engine-only, under 200 lines (expected ~137)
wc -l tests/compat/benchmarking.py
# 2. no imports from benchmarking in test_tensor_ops.py
grep -c 'benchmarking' tests/compat/test_tensor_ops.py        # 0
# 3. the benchmark ID list exactly matches the 4-0 capture (415 ids)
python3 tools/phase4_scenarios.py verify docs/design/baseline/bench-ids-phase4.txt
# 4. the collection count matches the 4-0 capture
pytest tests/compat --collect-only -q | tail -2 | sed -E 's/ in [0-9.]+s//' | diff - docs/design/baseline/collect-count-phase4.txt && echo "COLLECT: MATCH"
# 5. the test-writing convention exists
grep -q 'OP_CASES' tests/compat/README.md && grep -q 'PATH_CASES' tests/compat/README.md && echo "README OK"
# 6. full suite summary matches the 4-0 capture
pytest tests/compat -q 2>&1 | tail -1 | sed -E 's/ in [0-9.]+s.*$//' | diff - docs/design/baseline/tests-phase4.txt && echo "TESTS: MATCH"
```

Check off all Phase 4 items in `docs/design/PROGRESS.md` and commit.
