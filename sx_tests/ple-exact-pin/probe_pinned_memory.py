#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU probe: what does pinning N GiB of host memory really cost?

Run on a V100 host, inside the serving image, with nothing else being loaded
(other tenants' page cache moves MemAvailable; the 2% budget is ~240 MiB):

    python3 sx_tests/ple-exact-pin/probe_pinned_memory.py              # 11.92 GiB
    python3 sx_tests/ple-exact-pin/probe_pinned_memory.py --gib 4      # quicker
    python3 sx_tests/ple-exact-pin/probe_pinned_memory.py --mode exact # one way

It allocates the table twice, each time in a fresh child process (the caching
host allocator never hands memory back, so two allocations in one process
would not be comparable):

    torch    torch.empty(rows, 64, float8_e4m3fn, pin_memory=True)
             -- what the PLE host table did before SX_OPT_PLE_EXACT_PIN
    exact    allocate_exact_pinned(...) from vllm/models/qwen4_exp/nvidia/
             exact_pin.py of THIS checkout (loaded by path, never the
             installed copy): page-aligned anonymous memory + cuMemHostRegister

and prints, per way: the time, the change of MemAvailable, MemFree, AnonPages,
Shmem, Unevictable and Mlocked (/proc/meminfo) and of VmPin/VmLck/VmRSS (the
process), after the allocation and after writing every page once. Then, on the
table itself, the things the PLE gather depends on:

    * Tensor.is_pinned(), page-aligned data pointer, exact byte count;
    * the UVA view (get_accelerator_view_from_cpu_tensor): CUDA tensor, no
      second copy of the table made (a copy means torch did not call the
      memory pinned), device pointer == host pointer;
    * CUDA kernels reading and writing the host memory through the view
      (clone, sum, in-place add) at page-unaligned offsets, a replayed CUDA
      graph that sees new host bytes, and non-blocking copies both ways;
    * how much MemAvailable comes back when the table is dropped (the exact
      path unregisters and unmaps; the caching allocator keeps its block).

Exit status: 0 all good; 1 the exact path took more than --max-overhead
(default 0.02) above the requested size; 2 a usability check failed; 3 the
environment cannot run it (no CUDA, no /proc/meminfo, child crashed).
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import os
import random
import subprocess
import sys
import time
import types

GIB = 1024**3
MIB = 1024**2
ROW_BYTES = 64
PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_HELPER = os.path.join(
    REPO, "vllm", "models", "qwen4_exp", "nvidia", "exact_pin.py"
)
MARKER = "PROBE_JSON "
MODES = ("torch", "exact")
MEMINFO_KEYS = (
    "MemAvailable",
    "MemFree",
    "AnonPages",
    "Shmem",
    "Unevictable",
    "Mlocked",
)
STATUS_KEYS = ("VmPin", "VmLck", "VmRSS")


# --- /proc readers (pure, unit-tested on any OS) ----------------------------


def parse_kib_fields(text: str, keys: tuple[str, ...]) -> dict[str, int]:
    """``Key:   123 kB`` lines of /proc/meminfo or /proc/self/status, in bytes."""
    found: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in keys and rest.split():
            found[key] = int(rest.split()[0]) * 1024
    return found


def read_proc(path: str, keys: tuple[str, ...]) -> dict[str, int]:
    try:
        with open(path) as handle:
            return parse_kib_fields(handle.read(), keys)
    except OSError:
        return {}


def snapshot() -> dict[str, dict[str, int]]:
    return {
        "meminfo": read_proc("/proc/meminfo", MEMINFO_KEYS),
        "status": read_proc("/proc/self/status", STATUS_KEYS),
    }


def consumed(before: dict, after: dict, key: str, *, source: str) -> int | None:
    """Bytes the host lost (MemAvailable, MemFree) or the field gained (the rest)."""
    if key not in before[source] or key not in after[source]:
        return None
    if key in ("MemAvailable", "MemFree"):
        return before[source][key] - after[source][key]
    return after[source][key] - before[source][key]


# --- verdict (pure, unit-tested on any OS) ----------------------------------


def evaluate(
    results: dict[str, dict], max_overhead: float
) -> tuple[int, list[str]]:
    """Exit status and report lines for the per-mode results."""
    lines: list[str] = []
    environment = usability = overhead = False
    for mode in MODES:
        result = results.get(mode)
        if result is None:
            continue
        if result.get("environment_error"):
            lines.append(f"FAIL [{mode}] environment: {result['environment_error']}")
            environment = True
            continue
        for name, outcome in result.get("checks", {}).items():
            if not outcome["ok"]:
                lines.append(f"FAIL [{mode}] {name}: {outcome['detail']}")
                usability = True
    exact = results.get("exact")
    if exact and "consumed_after_fill" in exact:
        requested = exact["requested"]
        used = exact["consumed_after_fill"].get("MemAvailable")
        if used is None:
            lines.append("FAIL [exact] MemAvailable could not be read")
            environment = True
        else:
            ratio = used / requested - 1.0
            overhead = ratio > max_overhead
            lines.append(
                f"{'FAIL' if overhead else 'PASS'} [exact] MemAvailable fell by "
                f"{used / GIB:.3f} GiB for {requested / GIB:.3f} GiB requested "
                f"({ratio:+.2%}; limit {max_overhead:+.2%})"
            )
            if used < 0.9 * requested:
                lines.append(
                    "WARN [exact] less than 90% of the request left MemAvailable: "
                    "another process freed memory meanwhile, or the pages are not "
                    "resident; rerun on a quiet host"
                )
    legacy = results.get("torch")
    if legacy and "consumed_after_fill" in legacy:
        used = legacy["consumed_after_fill"].get("MemAvailable")
        if used is not None:
            lines.append(
                f"INFO [torch] MemAvailable fell by {used / GIB:.3f} GiB for "
                f"{legacy['requested'] / GIB:.3f} GiB requested "
                f"({used / legacy['requested'] - 1.0:+.2%}; the power-of-two "
                "rounding predicts "
                f"{(1 << (legacy['requested'] - 1).bit_length()) / GIB:.3f} GiB)"
            )
    if exact and legacy and "consumed_after_fill" in exact and not environment:
        a = exact["consumed_after_fill"].get("MemAvailable")
        b = legacy["consumed_after_fill"].get("MemAvailable")
        if a is not None and b is not None:
            lines.append(f"INFO saving of the exact path: {(b - a) / GIB:.3f} GiB")
    code = 3 if environment else 2 if usability else 1 if overhead else 0
    return code, lines


# --- the child: one allocation in a fresh process ---------------------------


def _load_helper(path: str) -> types.ModuleType:
    try:
        import vllm.logger  # noqa: F401, PLC0415
    except Exception:  # noqa: BLE001 - not installed: a plain logger will do
        stub = types.ModuleType("vllm.logger")
        stub.init_logger = logging.getLogger  # type: ignore[attr-defined]
        sys.modules["vllm.logger"] = stub
    spec = importlib.util.spec_from_file_location("sx_exact_pin", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["sx_exact_pin"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _uva_view(table, device: int):
    import torch

    errors = []
    with torch.cuda.device(device):
        try:
            from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

            return get_accelerator_view_from_cpu_tensor(table), "vllm.utils.torch_utils"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
        try:
            import vllm._C  # noqa: F401, PLC0415

            return torch.ops._C.get_cuda_view_from_cpu_tensor(table), "torch.ops._C"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError("no UVA view available: " + "; ".join(errors))


def run_child(mode: str, gib: float, device: int, helper_path: str, gpu_checks: bool):
    import torch

    result: dict = {"mode": mode, "checks": {}}
    if not torch.cuda.is_available():
        result["environment_error"] = "torch.cuda.is_available() is False"
        return result
    if not snapshot()["meminfo"].get("MemAvailable"):
        result["environment_error"] = "/proc/meminfo has no MemAvailable"
        return result

    fp8 = torch.float8_e4m3fn
    rows = int(gib * GIB) // ROW_BYTES
    requested = rows * ROW_BYTES
    shape = (rows, ROW_BYTES)
    result["requested"] = requested
    result["page_size"] = PAGE

    # Everything one-off happens before the baseline: CUDA context, the caching
    # host allocator, the driver bindings, the helper module.
    torch.cuda.set_device(device)
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    torch.empty(1, pin_memory=True)
    helper = None
    if mode == "exact":
        try:
            import cuda.bindings.driver  # noqa: F401, PLC0415
        except Exception as exc:  # noqa: BLE001
            result["environment_error"] = f"cuda.bindings.driver: {exc}"
            return result
        helper = _load_helper(helper_path)
    gc.collect()

    before = snapshot()
    start = time.perf_counter()
    try:
        if mode == "torch":
            table = torch.empty(shape, dtype=fp8, device="cpu", pin_memory=True)
        else:
            table = helper.allocate_exact_pinned(shape, fp8)
    except Exception as exc:  # noqa: BLE001
        result["checks"]["allocate"] = {
            "ok": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }
        return result
    result["alloc_seconds"] = time.perf_counter() - start
    after_alloc = snapshot()

    raw = table.view(torch.uint8).reshape(-1)
    start = time.perf_counter()
    raw.zero_()  # every page written once, as loading the checkpoint does
    result["fill_seconds"] = time.perf_counter() - start
    after_fill = snapshot()

    for tag, after in (("alloc", after_alloc), ("fill", after_fill)):
        result[f"consumed_after_{tag}"] = {
            key: consumed(before, after, key, source="meminfo")
            for key in MEMINFO_KEYS
        }
        result[f"process_after_{tag}"] = {
            key: consumed(before, after, key, source="status") for key in STATUS_KEYS
        }
    result["meminfo_before"] = before["meminfo"]

    def check(name: str, ok: bool, detail: str = "") -> None:
        result["checks"][name] = {"ok": bool(ok), "detail": detail}

    check(
        "is_pinned",
        table.is_pinned(),
        "Tensor.is_pinned() is False: the UVA view would copy the table",
    )
    check(
        "page_aligned",
        table.data_ptr() % PAGE == 0,
        f"data_ptr {table.data_ptr():#x} is not a multiple of {PAGE}",
    )
    check(
        "exact_bytes",
        table.numel() * table.element_size() == requested
        and tuple(table.shape) == shape,
        f"{table.numel() * table.element_size()} bytes, shape {tuple(table.shape)}",
    )

    if gpu_checks:
        _gpu_checks(result, check, table, raw, requested, device, torch)

    del raw, table
    gc.collect()
    torch.cuda.synchronize()
    host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
    if host_empty_cache is not None:
        host_empty_cache()
    released = snapshot()
    gained = released["meminfo"]["MemAvailable"] - after_fill["meminfo"]["MemAvailable"]
    result["released_fraction"] = gained / requested
    return result


def _gpu_checks(result, check, table, raw, requested, device, torch) -> None:
    before_view = snapshot()
    try:
        view, source = _uva_view(table, device)
    except Exception as exc:  # noqa: BLE001
        check("uva_view", False, f"{type(exc).__name__}: {exc}")
        return
    after_view = snapshot()
    copied = consumed(before_view, after_view, "MemAvailable", source="meminfo")
    result["uva"] = {
        "source": source,
        "ptr_equal": view.data_ptr() == table.data_ptr(),
        "memavailable_drop_creating_view": copied,
    }
    check(
        "uva_view",
        view.is_cuda and tuple(view.shape) == tuple(table.shape),
        f"view is_cuda={view.is_cuda} shape={tuple(view.shape)}",
    )
    check(
        "uva_view_made_no_copy",
        copied is not None and copied < 0.05 * requested + 64 * MIB,
        f"creating the view cost {_gib(copied)} GiB of MemAvailable: the memory "
        "was copied, so torch did not see it as pinned",
    )

    window = min(MIB, requested // 4)
    rng = random.Random(12345)
    offsets = sorted(
        {0, 4096 + 13, requested // 2 + 7, requested - window}
        | {rng.randrange(0, requested - window) for _ in range(3)}
    )
    generator = torch.Generator().manual_seed(2024)
    device_name = f"cuda:{device}"
    view_u8 = view.view(torch.uint8).reshape(-1)
    for index, offset in enumerate(offsets):
        label = f"window@{offset}"
        pattern = torch.randint(
            0, 256, (window,), dtype=torch.uint8, generator=generator
        )
        raw[offset : offset + window].copy_(pattern)
        gpu_window = view_u8[offset : offset + window]
        try:
            check(
                f"{label} kernel clone reads host memory",
                torch.equal(gpu_window.clone().cpu(), pattern),
                "clone through the UVA view differs from the CPU bytes",
            )
            check(
                f"{label} kernel sum reads host memory",
                int(gpu_window.to(torch.int64).sum())
                == int(pattern.to(torch.int64).sum()),
                "sum through the UVA view differs",
            )
            gpu_window.add_(1)
            torch.cuda.synchronize()
            check(
                f"{label} kernel write reaches host memory",
                torch.equal(raw[offset : offset + window], pattern + 1),
                "in-place add through the UVA view did not reach the CPU bytes",
            )
            moved = raw[offset : offset + window].to(device_name, non_blocking=True)
            torch.cuda.synchronize()
            check(
                f"{label} copy host->device",
                torch.equal(moved.cpu(), pattern + 1),
                "non-blocking copy to the GPU differs",
            )
            back = (pattern + 2).to(device_name)
            raw[offset : offset + window].copy_(back, non_blocking=True)
            torch.cuda.synchronize()
            check(
                f"{label} copy device->host",
                torch.equal(raw[offset : offset + window], pattern + 2),
                "non-blocking copy from the GPU differs",
            )
        except Exception as exc:  # noqa: BLE001
            check(f"{label} cuda operations", False, f"{type(exc).__name__}: {exc}")
            return

    # The production gather reads this memory inside CUDA graphs, through a
    # pointer captured once: new host bytes must show up on replay.
    try:
        offset = offsets[-1]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                view_u8[offset : offset + window].clone()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = view_u8[offset : offset + window].clone()
        fresh = torch.randint(0, 256, (window,), dtype=torch.uint8, generator=generator)
        raw[offset : offset + window].copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        check(
            "cuda graph replay reads host memory",
            torch.equal(captured.cpu(), fresh),
            "the replayed graph did not see the new host bytes",
        )
        del graph, captured
    except Exception as exc:  # noqa: BLE001
        check(
            "cuda graph replay reads host memory",
            False,
            f"{type(exc).__name__}: {exc}",
        )
    del view_u8, view


# --- the parent -------------------------------------------------------------


def run_mode(args: argparse.Namespace, mode: str) -> dict:
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--child",
        mode,
        "--gib",
        repr(args.gib),
        "--device",
        str(args.device),
        "--helper",
        args.helper,
    ]
    if args.no_gpu_checks:
        command.append("--no-gpu-checks")
    print(f"== {mode}: {' '.join(command[2:])}", flush=True)
    done = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=False)
    for line in reversed(done.stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :])
    sys.stdout.write(done.stdout)
    return {
        "mode": mode,
        "environment_error": f"child exited with status {done.returncode} "
        "and no result",
    }


def _gib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / GIB:+.3f}"


def _mib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / MIB:+.1f}"


def print_table(results: dict[str, dict]) -> None:
    modes = [m for m in MODES if m in results and "consumed_after_fill" in results[m]]
    if not modes:
        return
    width = 24
    print()
    print(f"{'':36}" + "".join(f"{m:>{width}}" for m in modes))

    def row(label: str, getter) -> None:
        print(f"{label:36}" + "".join(f"{getter(results[m]):>{width}}" for m in modes))

    row("requested (GiB)", lambda r: f"{r['requested'] / GIB:.3f}")
    row("allocation time (s)", lambda r: f"{r['alloc_seconds']:.2f}")
    row("first write of every page (s)", lambda r: f"{r['fill_seconds']:.2f}")
    for tag, title in (("alloc", "after allocation"), ("fill", "after first write")):
        print(f"-- change of /proc/meminfo {title} (GiB; MemAvailable/MemFree: lost)")
        for key in MEMINFO_KEYS:
            row(f"  {key}", lambda r, k=key, t=tag: _gib(r[f"consumed_after_{t}"][k]))
        print(f"-- change of this process {title} (MiB)")
        for key in STATUS_KEYS:
            row(f"  {key}", lambda r, k=key, t=tag: _mib(r[f"process_after_{t}"][k]))
    row(
        "overhead vs requested",
        lambda r: (
            "n/a"
            if r["consumed_after_fill"]["MemAvailable"] is None
            else f"{r['consumed_after_fill']['MemAvailable'] / r['requested'] - 1:+.2%}"
        ),
    )
    row(
        "returned when dropped",
        lambda r: (
            "n/a"
            if r.get("released_fraction") is None
            else f"{r['released_fraction']:.0%} of request"
        ),
    )
    for mode in modes:
        uva = results[mode].get("uva")
        if uva:
            print(
                f"[{mode}] UVA view via {uva['source']}: device ptr == host ptr: "
                f"{uva['ptr_equal']}; MemAvailable lost creating it: "
                f"{_mib(uva['memavailable_drop_creating_view'])} MiB"
            )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gib", type=float, default=11.92, help="table size, GiB")
    parser.add_argument("--mode", choices=("both", *MODES), default="both")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--max-overhead",
        type=float,
        default=0.02,
        help="largest accepted MemAvailable loss above the request, exact path",
    )
    parser.add_argument("--helper", default=DEFAULT_HELPER, help="path of exact_pin.py")
    parser.add_argument("--no-gpu-checks", action="store_true")
    parser.add_argument("--child", choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child:
        try:
            result = run_child(
                args.child, args.gib, args.device, args.helper, not args.no_gpu_checks
            )
        except Exception as exc:  # noqa: BLE001 - reported through the parent
            result = {
                "mode": args.child,
                "environment_error": f"{type(exc).__name__}: {exc}",
            }
        print(MARKER + json.dumps(result), flush=True)
        return 0

    modes = MODES if args.mode == "both" else (args.mode,)
    results = {mode: run_mode(args, mode) for mode in modes}
    print_table(results)
    code, lines = evaluate(results, args.max_overhead)
    print("\n".join(lines))
    print(f"exit status {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
