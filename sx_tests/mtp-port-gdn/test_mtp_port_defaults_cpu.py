# SPDX-License-Identifier: Apache-2.0
"""MTP-lane defaults of the upstream d30469863 port (vllm/config/vllm.py).

Run (CPU, no vLLM install needed; see ``_port_boot.py``):

    python -m pytest -q sx_tests/mtp-port-gdn/test_mtp_port_defaults_cpu.py

``_apply_sm70_qwen38_nomtp_defaults``, ``_sx_mtp_lane_port_defaults`` and
``_sx_env_on`` are cut out of the real config module; the lane qualification
and the ``_C`` capability probes are stubs. The full config-level checks
(real contract, real VllmConfig stand-ins) are in
``sx_tests/b3-lane-core/test_mtp_lane_config_cpu.py`` (image only).

Asserted:
* without MTP the applied list is exactly the previous one (5 defaults plus
  the #704 exact gated norm when _C has it) and no port default appears;
* in the MTP lane every port default is applied when its SX_OPT_MTP_* switch
  is on and _C carries its op; "0" or a missing op drops just that default;
* explicit values of the upstream variables always win;
* an unqualified deployment gets nothing.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _port_boot as boot  # noqa: E402

BASE = (
    "VLLM_SM70_QWEN38_FP16_GEMV",
    "VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16",
    "VLLM_SM70_QWEN38_FUSED_HC_FP16",
    "VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP",
    "VLLM_SM70_MOE_ADD_ALLREDUCE",
)
SPLIT = "VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS"
NORM = "VLLM_SM70_RMSNORM_GATED_EXACT"
# Port defaults: (upstream variable, SX switch, capability probe).
PORT = (
    (NORM, "SX_OPT_MTP_RMSNORM_GATED_EXACT", "_sm70_rmsnorm_gated_exact_available"),
    ("VLLM_SM70_MTP_PLE_CONV", "SX_OPT_MTP_PLE_CONV", "_sm70_ple_spec_conv_available"),
)
PORT_KEYS = tuple(key for key, _, _ in PORT)
ALL_KEYS = (*BASE, SPLIT, *PORT_KEYS)
SWITCHES = tuple(switch for _, switch, _ in PORT)


@pytest.fixture
def cfg_ns(monkeypatch):
    for key in (*ALL_KEYS, *SWITCHES):
        monkeypatch.delenv(key, raising=False)
    state = {"qualified": True, **{probe: True for _, _, probe in PORT}}
    namespace = {
        "os": os,
        "_sm70_qwen38_lane_qualified": lambda cfg, *, is_sm70: (
            state["qualified"] and is_sm70
        ),
    }
    for _, _, probe in PORT:
        namespace[probe] = lambda probe=probe: state[probe]
    boot.cut(
        boot.CONFIG,
        ("_sx_env_on", "_sx_mtp_lane_port_defaults", "_apply_sm70_qwen38_nomtp_defaults"),
        namespace,
    )
    namespace["state"] = state
    yield namespace
    for key in ALL_KEYS:
        os.environ.pop(key, None)


def _cfg(mtp: bool):
    return SimpleNamespace(speculative_config=SimpleNamespace() if mtp else None)


def _apply(ns, mtp: bool, is_sm70: bool = True):
    applied = ns["_apply_sm70_qwen38_nomtp_defaults"](_cfg(mtp), is_sm70=is_sm70)
    snapshot = {key: os.environ.get(key) for key in ALL_KEYS}
    for key in applied:
        os.environ.pop(key, None)
    return applied, snapshot


def test_nomtp_unchanged(cfg_ns):
    applied, _ = _apply(cfg_ns, mtp=False)
    assert applied == (*BASE, NORM)
    cfg_ns["state"]["_sm70_rmsnorm_gated_exact_available"] = False
    applied, _ = _apply(cfg_ns, mtp=False)
    assert applied == BASE
    # The SX port switches never touch the no-MTP lane.
    cfg_ns["state"]["_sm70_rmsnorm_gated_exact_available"] = True
    for switch in SWITCHES:
        os.environ[switch] = "0"
    applied, _ = _apply(cfg_ns, mtp=False)
    assert applied == (*BASE, NORM)


def test_mtp_lane_gets_port_defaults(cfg_ns):
    applied, env = _apply(cfg_ns, mtp=True)
    assert applied == (*BASE, SPLIT, *PORT_KEYS)
    assert all(env[key] == "1" for key in PORT_KEYS)


@pytest.mark.parametrize("index", range(len(PORT)))
def test_each_switch_and_probe(cfg_ns, index):
    key, switch, probe = PORT[index]
    os.environ[switch] = "0"
    applied, _ = _apply(cfg_ns, mtp=True)
    assert key not in applied
    assert applied == (*BASE, SPLIT, *(k for k in PORT_KEYS if k != key))
    del os.environ[switch]
    cfg_ns["state"][probe] = False
    applied, _ = _apply(cfg_ns, mtp=True)
    assert key not in applied
    cfg_ns["state"][probe] = True
    os.environ[key] = "0"
    applied, env = _apply(cfg_ns, mtp=True)
    assert key not in applied and env[key] == "0"


def test_unqualified_gets_nothing(cfg_ns):
    cfg_ns["state"]["qualified"] = False
    assert _apply(cfg_ns, mtp=True)[0] == ()
    cfg_ns["state"]["qualified"] = True
    assert _apply(cfg_ns, mtp=True, is_sm70=False)[0] == ()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
