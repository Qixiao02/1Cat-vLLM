# SPDX-License-Identifier: Apache-2.0
"""Batch 3a "moe-verify": 25-KiB MTP5 push all-reduce gating on FOUR GPUs.

GPUs: four idle, fully NVLink-connected SM70 GPUs (one V100 quad). Stop the
serving container first. Inside image 1.8.0-dev2 with this group's
custom_all_reduce.py and fused_topk_router.py bind-mounted:

  CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/venv/bin/python -m pytest -q -s \
      sx_tests/b3-moe-verify/test_mtp5_push_ar_tp4.py

Launches ``torch.distributed.run --nproc-per-node=4`` on _mtp5_push_worker.py
(see that file for what it asserts). SX_B3_AR_CYCLES sets the stress cycles
(default 16).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WORKER = HERE / "_mtp5_push_worker.py"


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


def test_mtp5_push_gate_tp4(tmp_path):
    out = tmp_path / "b3_mtp5.json"
    env = dict(os.environ)
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        if "expandable_segments:true" in env.get(name, "").replace(" ", "").lower():
            env.pop(name)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.pop("VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5", None)
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node=4", str(WORKER), "--out", str(out),
        "--cycles", os.environ.get("SX_B3_AR_CYCLES", "16"),
    ]
    proc = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True,
                          timeout=1800)
    sys.stdout.write(proc.stdout[-20000:])
    sys.stderr.write(proc.stderr[-20000:])
    assert proc.returncode == 0, f"worker failed (rc={proc.returncode})"
    result = json.loads(out.read_text())
    assert result["failures"] == []
    table = result["admission"]
    assert table["lane_wide_off/sum2"] == 13 and table["dev2_wide_off/sum2"] == 0
    assert result["outputs_checked"] > 0
