# SPDX-License-Identifier: Apache-2.0
"""Build the SX_OPT_MOE_GROUPED32 sidecar (v2 grouped-decode ops only).

Compiles csrc/sm70_turbomind/ops/nvfp4_grouped_decode_sm70.cu alone for sm_70
with -DSX_NVFP4_GROUPED_SIDECAR=1. The result registers
torch.ops._C_qwen38_grouped.{nvfp4_grouped_w13_v2_sm70_out,
nvfp4_grouped_w2_v2_sm70_out, nvfp4_grouped_decode_v2_max_routes} and does not
touch the deployed vllm._C (which keeps the original grouped ops).

Run inside the image (torch 2.10 / cu128, nvcc from CUDA_HOME):
  /opt/venv/bin/python sx_tests/b2-moe-m32/build_sidecar.py [OUT_DIR]
then either install it as
  site-packages/vllm/_sx_nvfp4_grouped32_C.abi3.so   (auto-loaded)
or point SX_OPT_MOE_GROUPED32_LIBRARY at it.

Uses torch.utils.cpp_extension.load (ninja) when available, else a direct
nvcc command with the same flags torch uses for CUDA extensions.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "csrc" / "sm70_turbomind" / "ops" / (
    "nvfp4_grouped_decode_sm70.cu"
)
NAME = "_sx_nvfp4_grouped32_C"
DEFINE = "-DSX_NVFP4_GROUPED_SIDECAR=1"
NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-lineinfo",
    DEFINE,
    "-gencode=arch=compute_70,code=sm_70",
]


def _build_dir(out_dir: str | None) -> Path:
    path = Path(
        out_dir
        or os.environ.get("SX_B2_BUILD_DIR")
        or os.path.join(os.environ.get("TMPDIR", "/tmp"), "sx_b2_moe_m32_build")
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_with_ninja(build_dir: Path) -> Path:
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    load(
        name=NAME,
        sources=[str(SOURCE)],
        extra_cflags=["-O3", "-std=c++17", DEFINE],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", DEFINE],
        build_directory=str(build_dir),
        is_python_module=False,
        verbose=bool(os.environ.get("SX_B2_BUILD_VERBOSE")),
    )
    return build_dir / f"{NAME}.so"


def _paths(fn, cuda: bool) -> list[str]:
    try:
        return list(fn(device_type="cuda" if cuda else "cpu"))
    except TypeError:  # older torch
        return list(fn(cuda=cuda))


def _build_with_nvcc(build_dir: Path) -> Path:
    import torch
    from torch.utils import cpp_extension

    cuda_home = cpp_extension.CUDA_HOME or os.environ.get("CUDA_HOME", "")
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    includes = _paths(cpp_extension.include_paths, True)
    libdirs = _paths(cpp_extension.library_paths, True)
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    out = build_dir / f"{NAME}.so"
    cmd = [
        nvcc,
        *NVCC_FLAGS,
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        f"-DTORCH_EXTENSION_NAME={NAME}",
        "-Xcompiler",
        "-fPIC",
        "-shared",
        *[f"-I{p}" for p in includes],
        *[f"-L{p}" for p in libdirs],
        *[arg for p in libdirs for arg in ("-Xlinker", f"-rpath,{p}")],
        "-lc10",
        "-lc10_cuda",
        "-ltorch_cpu",
        "-ltorch_cuda",
        "-ltorch",
        "-lcudart",
        str(SOURCE),
        "-o",
        str(out),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out


def build(out_dir: str | None = None) -> str:
    """Build (or reuse) the sidecar and return its path."""
    build_dir = _build_dir(out_dir)
    if os.environ.get("SX_B2_BUILD_NVCC") != "1":
        try:
            return str(_build_with_ninja(build_dir))
        except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
            # e.g. "Ninja is required to load C++ extensions"
            print(f"cpp_extension.load failed ({exc}); using nvcc directly")
    return str(_build_with_nvcc(build_dir))


if __name__ == "__main__":
    library = build(sys.argv[1] if len(sys.argv) > 1 else None)
    import torch

    torch.ops.load_library(library)
    ns = torch.ops._C_qwen38_grouped
    print(library)
    print("max_routes =", ns.nvfp4_grouped_decode_v2_max_routes())
