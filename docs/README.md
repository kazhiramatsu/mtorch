# mtorch Documentation Index

This directory holds mtorch's design materials and the detailed procedure guides for carrying out the refactoring.

## Directory Layout

| Path | Contents |
|---|---|
| `native-core.md` | Notes on the initial implementation of the C++ core (layer structure, build instructions) |
| `test-harness-design.md` | Design of the PyTorch-compatibility test harness (**the safety net for the refactoring**. Required reading) |
| `design/` | The full set of refactoring designs and procedure guides (see table below) |

## Under `design/`

| File | Contents | When to read |
|---|---|---|
| [design/00-overview.md](design/00-overview.md) | Master design: current-state analysis, target architecture, phase plan, risks | Read through once, first |
| [design/01-rules-and-verification.md](design/01-rules-and-verification.md) | **Work rules and verification procedures common to all phases** (commit discipline, running and comparing tests/benchmarks, the protocol on failure) | Required reading before starting. Refer to it continuously while working |
| [design/02-phase0-baseline.md](design/02-phase0-baseline.md) | Phase 0: start git management, record the baseline, organize the generated JSON artifacts | While working on Phase 0 |
| [design/03-phase1-python-package.md](design/03-phase1-python-package.md) | Phase 1: split `mtorch/__init__.py` into a package | While working on Phase 1 |
| [design/04-phase2a-module-cpp-split.md](design/04-phase2a-module-cpp-split.md) | Phase 2a: split `module.cpp` (the bindings) | While working on Phase 2a |
| [design/05-phase2b-tensor-cpp-split.md](design/05-phase2b-tensor-cpp-split.md) | Phase 2b: split `tensor.cpp` (the C++ core) | While working on Phase 2b |
| [design/06-phase3-dedup.md](design/06-phase3-dedup.md) | Phase 3: deduplication and correcting layer responsibilities | While working on Phase 3 |
| [design/07-phase4-test-infra.md](design/07-phase4-test-infra.md) | Phase 4: organizing the test/benchmark infrastructure | While working on Phase 4 |
| [design/08-phase5-device-layer.md](design/08-phase5-device-layer.md) | Phase 5: device/backend infrastructure (Device/allocator registry, transfer API, Python device surface, parity test scaffolding) | While working on Phase 5 (only after the Phase 0–4 completion check) |
| [design/09-phase6-metal-backend.md](design/09-phase6-metal-backend.md) | Phase 6: Apple-Silicon Metal backend (device "mps") — kernels, allocator, dispatch guards, parity gates | While working on Phase 6 (after Phase 5) |
| [design/10-phase7-cuda-backend.md](design/10-phase7-cuda-backend.md) | Phase 7: CUDA backend (device "cuda") — allocator, nvcc build, per-family kernels; runs on a separate Linux/NVIDIA machine | While working on Phase 7 (Linux GPU box only) |
| [design/PROGRESS.md](design/PROGRESS.md) | **Progress checklist** (check and update your current position) | At the start and end of every work session |

Phases 5–7 (the GPU backends) are a follow-on project: they start only after the CPU
refactoring (phases 0–4) is fully complete, i.e. every box in the "Overall completion
check" section of `design/PROGRESS.md` is checked. Their checklists are registered into
`PROGRESS.md` by their own first steps (5-0 / 6-0 / 7-1).

## For the Agent Executing the Refactoring

Before starting work, always read these in this order.

1. **[design/PROGRESS.md](design/PROGRESS.md)** — check how far things have been completed. The first uncompleted step is your next task
2. **[design/01-rules-and-verification.md](design/01-rules-and-verification.md)** — the work rules (what you may and may not do) and the verification commands
3. **The procedure guide for your phase** (design/02–07) — execute the steps written in the guide, in the written order, one step at a time

The most important rules to follow (details in design/01):

- **1 step = 1 commit. Do not skip steps. Do not reorder them**
- **Do not proceed until each step's completion criteria (build succeeds + tests green) are met**
- **When a test fails, never rewrite the test itself to make it pass**
- **Do not perform refactoring not in the guide (no "while I'm at it" improvements)**
- When you finish working (including when you pause partway), update PROGRESS.md

The line numbers in the guides are based on a snapshot as of 2026-07-08. As work progresses the line numbers will drift, so **treat line numbers as approximate and always re-locate positions using the accompanying grep commands or function names**.
