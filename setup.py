from __future__ import annotations

import sys

from setuptools import Extension, setup


extra_compile_args = ["-std=c++20", "-O3"]
extra_link_args: list[str] = []
if sys.platform == "darwin":
    extra_compile_args.append("-DMTORCH_USE_ACCELERATE")
    extra_link_args.extend(["-framework", "Accelerate"])

setup(
    packages=["mtorch"],
    ext_modules=[
        Extension(
            "mtorch._C",
            sources=[
                "cpp/mtorch/core/tensor.cpp",
                "cpp/mtorch/python/module.cpp",
            ],
            include_dirs=["cpp"],
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
)
