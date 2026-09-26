# SPDX-License-Identifier: Apache-2.0
"""b2-allreduce [C4] SX_OPT_PUSH_AR_WIDE on FOUR GPUs (TP4 push all-reduce).

GPU: four idle, fully NVLink-connected SM70 GPUs (one V100 quad). Stop the
serving container first. Inside the image, with the rebuilt custom-AR
extension and the patched custom_all_reduce.py mounted:

  CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/venv/bin/python -m pytest -q -s \
      sx_tests/b2-allreduce/test_push_ar_wide_tp4.py

Each test launches ``torch.distributed.run --nproc-per-node=4`` on
_push_ar_wide_worker.py (see that file for exactly what each mode asserts).
Knobs: SX_B2_AR_EQ_CYCLES (default 8 per pattern), SX_B2_AR_GRAPH_CYCLES
(default 64 per pattern), SX_B2_AR_BENCH_STRICT=1 (fail if a newly admitted
size is not faster than the pull path it replaces), SX_B2_AR_OUT (directory
for the JSON results, default: pytest tmp dir).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WORKER = HERE / "_push_ar_wide_worker.py"
sys.path.insert(0, str(HERE))
import _push_ar_common as C  # noqa: E402


def _four_sm70() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        return False
    return all(torch.cuda.get_device_capability(i) == (7, 0) for i in range(4))


pytestmark = pytest.mark.skipif(
    not _four_sm70(), reason="needs four SM70 GPUs (TP4 push all-reduce)"
)


def _run(mode: str, tmp_path: Path, *extra: str, timeout: int = 3600) -> dict:
    out_dir = Path(os.environ.get("SX_B2_AR_OUT", str(tmp_path)))
    out = out_dir / f"b2_allreduce_{mode}.json"
    env = dict(os.environ)
    # CUDA IPC cannot export expandable-segment allocations.
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        if "expandable_segments:true" in env.get(name, "").replace(" ", "").lower():
            env.pop(name)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    env.setdefault("OMP_NUM_THREADS", "1")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        str(WORKER),
        "--mode",
        mode,
        "--out",
        str(out),
        *extra,
    ]
    proc = subprocess.run(
        cmd, cwd=HERE, env=env, capture_output=True, text=True, timeout=timeout
    )
    sys.stdout.write(proc.stdout[-30000:])
    sys.stderr.write(proc.stderr[-30000:])
    assert proc.returncode == 0, f"{mode} worker failed (rc={proc.returncode})"
    result = json.loads(out.read_text())
    assert result["failures"] == []
    return result


def test_admission_selects_push_for_every_row_multiple(tmp_path):
    result = _run("admission", tmp_path)
    table = result["table"]
    for op in ("plain", "sum2"):
        for nbytes in C.ROW_SIZES:
            new = table[f"new/{op}/{nbytes}"]["observed_ctas"]
            if nbytes == C.ROW_BYTES:
                # M1 keeps its established 3-CTA launch.
                assert new == 3, (op, nbytes, new)
            elif nbytes == C.M8_5120 and op == "sum2":
                assert new == 80  # established M16 sum2 launch kept
            else:
                assert new == C.covering_ctas(nbytes), (op, nbytes, new)
            assert table[f"old/{op}/{nbytes}"]["observed_ctas"] == C.native_push_ctas(
                op, nbytes, C.ARMS["old"]
            )
    # The deployment gaps this closes: M24 both collectives, M2/M32 sum2.
    assert table[f"old/plain/{24 * C.ROW_BYTES}"]["observed_ctas"] == 0
    assert table[f"old/sum2/{24 * C.ROW_BYTES}"]["observed_ctas"] == 0
    assert table[f"old/sum2/{2 * C.ROW_BYTES}"]["observed_ctas"] == 0
    assert table[f"old/sum2/{32 * C.ROW_BYTES}"]["observed_ctas"] == 0
    assert table[f"new/plain/{24 * C.ROW_BYTES}"]["observed_ctas"] == 60
    assert table[f"new/sum2/{32 * C.ROW_BYTES}"]["observed_ctas"] == 80
    # Rollback: SX_OPT_PUSH_AR_WIDE=0 restores the previous admission exactly.
    assert result["changed_vs_old"]["plain"] == [m * C.ROW_BYTES for m in range(17, 32)]
    assert result["changed_vs_old"]["sum2"] == [
        m * C.ROW_BYTES for m in range(2, 33) if m not in (4, 8, 16)
    ]


def test_push_equals_pull_bitwise_all_row_sizes(tmp_path):
    cycles = os.environ.get("SX_B2_AR_EQ_CYCLES", "8")
    result = _run("equality", tmp_path, "--cycles", cycles)
    totals = result["totals_[bits,nan_pos,nan_payload]"]
    for name, (bits, nan_pos, _) in totals.items():
        assert bits == 0 and nan_pos == 0, name


def test_mixed_size_graph_stress(tmp_path):
    cycles = os.environ.get("SX_B2_AR_GRAPH_CYCLES", "64")
    result = _run("graph", tmp_path, "--cycles", cycles)
    assert result["outputs_checked"] > 0


def test_push_microbenchmark(tmp_path):
    extra = ("--strict",) if os.environ.get("SX_B2_AR_BENCH_STRICT") == "1" else ()
    result = _run("bench", tmp_path, *extra)
    per_size = result["per_collective_us"]
    assert f"plain/{24 * C.ROW_BYTES}" in per_size
    assert "M24" in result["layer_step"] and "M32" in result["layer_step"]
